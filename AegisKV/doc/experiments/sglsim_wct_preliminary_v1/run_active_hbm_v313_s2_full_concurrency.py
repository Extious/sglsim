#!/usr/bin/env python3
"""Run the preregistered v3.1.3 S2 concurrency x physical-rate campaign.

The campaign is ledger-first and fail-closed.  Every event-bearing window
executes the persistent Poisson state machine; zero-event windows use an exact
memoized healthy DES only after a separate executable equivalence replay.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import active_hbm_v31 as v31
import active_hbm_v313 as v313
import poisson_fault_batch_v2 as batch
import run_preliminary as core
import test_active_hbm_v313 as test_inputs


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s2_full_concurrency_protocol.json"
GIB = 1024 ** 3
T95_9DF = 2.2621571627409915


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_workload_hash(tasks: Iterable[core.Task]) -> str:
    return canonical_sha256([
        (
            row.request_id,
            row.workflow_id,
            row.workflow_arrival_ms,
            row.arrival_ms,
            row.source_rank,
            row.service_ms,
            row.prompt_tokens,
            row.output_tokens,
            row.predicted_slack_ms,
        )
        for row in tasks
    ])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple)) else value
                )
                for key, value in row.items()
            })


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


class InstrumentedS2Replay(batch.BatchedPoissonFaultInjectedReplay):
    """Read-only resource integration layered on the frozen v3.1.3 DES."""

    RESOURCE_NAMES = ("serving_hbm", "peer_hbm_backup", "dram")

    def __init__(self, protocol: dict[str, Any], policy: str, capacity: dict[str, Any]) -> None:
        super().__init__(protocol, policy, capacity)
        self.s2_resource_last_ms = 0.0
        self.s2_resource_current_bytes = {name: 0 for name in self.RESOURCE_NAMES}
        self.s2_resource_area_byte_ms = {name: 0.0 for name in self.RESOURCE_NAMES}
        self.s2_resource_peak_total_bytes = {name: 0 for name in self.RESOURCE_NAMES}
        self.s2_resource_sample_count = 0

    def _resource_totals(self) -> dict[str, int]:
        if self.is_aegis:
            snapshots = [row.telemetry_snapshot() for row in self.aegis]
            dram = sum(int(row["pools"]["DRAM"]["used_bytes"]) for row in snapshots)
        else:
            dram = sum(
                int(runtime.used_bytes)
                for ranks in self.baselines
                for runtime in ranks
                if runtime.config.tier.value == "DRAM"
            )
        return {
            "serving_hbm": int(sum(self.active_serving_bytes)),
            "peer_hbm_backup": int(
                sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
            ),
            "dram": int(dram),
        }

    def _sample_resources(self, at_ms: float) -> None:
        horizon = self.occupancy_observation_horizon_ms
        if horizon is None:
            return
        capped = min(max(float(at_ms), 0.0), float(horizon))
        if capped + 1e-9 < self.s2_resource_last_ms:
            raise AssertionError("S2 resource time regressed")
        delta = max(capped - self.s2_resource_last_ms, 0.0)
        for name in self.RESOURCE_NAMES:
            self.s2_resource_area_byte_ms[name] += (
                self.s2_resource_current_bytes[name] * delta
            )
        self.s2_resource_last_ms = capped
        current = self._resource_totals()
        for name, value in current.items():
            if value < 0:
                raise AssertionError("negative resource occupancy")
            self.s2_resource_current_bytes[name] = int(value)
            self.s2_resource_peak_total_bytes[name] = max(
                self.s2_resource_peak_total_bytes[name], int(value)
            )
        self.s2_resource_sample_count += 1

    def record_capacity(self, rank: int, at: float, event: str) -> None:
        super().record_capacity(rank, at, event)
        if hasattr(self, "s2_resource_current_bytes"):
            self._sample_resources(at)

    def control(
        self,
        at: float,
        active_by_rank: dict[int, int],
        external_pressure_by_rank: dict[int, bool] | None = None,
    ) -> None:
        super().control(
            at,
            active_by_rank,
            external_pressure_by_rank=external_pressure_by_rank,
        )
        self._sample_resources(at)

    def finalize_resource_summary(self, horizon_ms: float) -> dict[str, float]:
        self._sample_resources(float(horizon_ms))
        denominator = max(float(horizon_ms) * self.dp, 1e-12)
        result: dict[str, float] = {
            "resource_sample_count": float(self.s2_resource_sample_count),
        }
        for name in self.RESOURCE_NAMES:
            result[f"{name}_time_mean_gib_per_gpu"] = (
                self.s2_resource_area_byte_ms[name] / denominator / GIB
            )
            result[f"{name}_peak_total_gib"] = (
                self.s2_resource_peak_total_bytes[name] / GIB
            )
            result[f"{name}_peak_mean_gib_per_gpu"] = (
                self.s2_resource_peak_total_bytes[name] / self.dp / GIB
            )
        result["peer_plus_serving_peak_mean_gib_per_gpu"] = max(
            (
                int(row["active_serving_kv_bytes"])
                + int(row["peer_hbm_backup_used_bytes"])
            ) / GIB
            for row in self.capacity_trace
        )
        return result


def configured_protocol(concurrency: int) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, capacity = test_inputs.inputs(context=16384, active=int(concurrency))
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["hardware"]["gpus_per_node"] = 8
    protocol["hardware"]["node_count"] = 2
    protocol["workload"]["center_context_tokens"] = 16384
    protocol["workload"]["center_active_requests_per_gpu"] = int(concurrency)
    protocol["workload"]["reuse_fraction"] = 0.0
    protocol.setdefault("runtime_overrides", {})["record_occupancy_trace"] = False
    return protocol, capacity


def mechanism_row(
    replay: InstrumentedS2Replay,
    mechanism: dict[str, Any],
    horizon_ms: float,
) -> dict[str, Any]:
    base = mechanism
    v3 = base["active_hbm_v3"]
    v313m = base["active_hbm_v313"]
    poisson = base["poisson_fault_batch_v2"]
    controls = base["control_step_trace"]
    alpha_values = [float(row["alpha_after_mean"]) for row in controls]
    pressure_values = [float(row["pressure_fraction"]) for row in controls]
    resource = replay.finalize_resource_summary(horizon_ms)
    churn = base.get("tier_churn", {})
    return {
        **resource,
        "alpha_final_mean": base["alpha_final_mean"],
        "alpha_final_min": base["alpha_final_min"],
        "alpha_control_mean": (
            statistics.fmean(alpha_values) if alpha_values else None
        ),
        "alpha_control_std": (
            statistics.pstdev(alpha_values) if len(alpha_values) > 1 else 0.0
        ),
        "alpha_control_range": (
            max(alpha_values) - min(alpha_values) if alpha_values else 0.0
        ),
        "pressure_fraction_mean": (
            statistics.fmean(pressure_values) if pressure_values else None
        ),
        "registered_control_opportunity_count": len(
            v313m["scheduled_control_opportunity_times_ms"]
        ),
        "executed_policy_control_count": int(base["control_window_count"]),
        "registered_control_times_ms": v313m[
            "scheduled_control_opportunity_times_ms"
        ],
        "memory_blocked_observation_count": int(
            v3["memory_blocked_observation_count"]
        ),
        "memory_blocked_unique_request_count": int(
            v3["memory_blocked_unique_request_count"]
        ),
        "queue_wait_mean_ms": float(v3["queue_wait_mean_ms"]),
        "queue_wait_p95_ms": float(v3["queue_wait_p95_ms"]),
        "capacity_violation_count": int(v3["capacity_violation_count"])
        + int(v313m["target_capacity_violation_count"]),
        "capacity_min_headroom_bytes": int(v3["capacity_min_headroom_bytes"]),
        "primary_priority_reclaimed_bytes": int(
            v313m["primary_priority_reclaimed_bytes"]
        ),
        "backup_bytes_released_by_control": int(
            v3["backup_bytes_released_by_control"]
        ),
        "scheduled_hbm_copy_bytes": int(v313m["scheduled_copy_bytes"]["HBM"]),
        "scheduled_dram_copy_bytes": int(v313m["scheduled_copy_bytes"]["DRAM"]),
        "hbm_copy_byte_conservation_error": int(
            v313m["copy_byte_conservation_error"]["HBM"]
        ),
        "dram_copy_byte_conservation_error": int(
            v313m["copy_byte_conservation_error"]["DRAM"]
        ),
        "copy_duration_formula_max_abs_error_ms": float(
            v313m["copy_duration_formula_max_abs_error_ms"]
        ),
        "copy_critical_path_sum_violation_count": int(
            v313m["copy_critical_path_sum_violation_count"]
        ),
        "health_burden_application_count": int(
            v313m["health_burden_application_count"]
        ),
        "future_backup_projection_dispatch_count": int(
            v313m["future_backup_projection_dispatch_count"]
        ),
        "seal_before_dispatch_count": int(v313m["seal_before_dispatch_count"]),
        "seal_uses_realized_output_or_service": bool(
            v313m["seal_uses_realized_output_or_service"]
        ),
        "hbm_admission_attempt_count": int(
            churn.get("HBM", {}).get("admission_attempt_count", 0)
        ),
        "hbm_admission_success_count": int(
            churn.get("HBM", {}).get("admission_success_count", 0)
        ),
        "hbm_admission_rejection_count": int(
            churn.get("HBM", {}).get("admission_rejection_count", 0)
        ),
        "hbm_eviction_count": int(churn.get("HBM", {}).get("eviction_count", 0)),
        "dram_admission_attempt_count": int(
            churn.get("DRAM", {}).get("admission_attempt_count", 0)
        ),
        "dram_admission_success_count": int(
            churn.get("DRAM", {}).get("admission_success_count", 0)
        ),
        "dram_admission_rejection_count": int(
            churn.get("DRAM", {}).get("admission_rejection_count", 0)
        ),
        "residency_event_counts": base["residency_event_counts"],
        "candidate_event_count": int(poisson["candidate_event_count"]),
        "accepted_event_count": int(poisson["accepted_event_count"]),
        "noop_event_count": int(poisson["noop_event_count"]),
        "replica_invalidated_bytes": int(poisson["replica_invalidated_bytes"]),
        "recovery_action_counts": poisson["recovery_action_counts"],
    }


def workflow_statistics(tasks: list[core.Task]) -> tuple[list[str], list[float], dict[str, float]]:
    ids, values = core.workflow_wct(tasks)
    return ids, values, {
        "wct_mean_ms": statistics.fmean(values),
        "wct_p95_ms": percentile(values, 0.95),
        "wct_p99_ms": percentile(values, 0.99),
        "wct_max_ms": max(values),
        "drain_horizon_ms": max(row.finish_ms for row in tasks),
    }


def slug_policy(policy: str) -> str:
    return policy.lower().replace(" ", "_")


def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    stage = Path(job["stage"])
    concurrency = int(job["concurrency"])
    seed = int(job["seed"])
    horizon = float(job["horizon_ms"])
    policies = list(job["policies"])
    windows_by_rate = job["windows_by_rate"]
    protocol, capacity = configured_protocol(concurrency)
    tasks, workload = v31.build_workload(
        protocol,
        seed=seed,
        active_per_gpu=concurrency,
        arrival_cutoff_ms=horizon,
    )
    workload_hash = canonical_workload_hash(tasks)
    if workload["slack_predictor"]["reads_realized_output_tokens"]:
        raise AssertionError("old oracle workload lineage reached S2")
    raw_dir = stage / "raw" / f"c{concurrency:02d}" / f"s{seed}"
    event_dir = stage / "raw_events" / f"c{concurrency:02d}" / f"s{seed}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    healthy: dict[str, dict[str, Any]] = {}
    zero_equivalence: dict[str, bool] = {}
    for policy in policies:
        replay = InstrumentedS2Replay(deepcopy(protocol), policy, capacity)
        scheduled, mechanism, capacity_trace = replay.run_rate_injected(
            tasks, [], control_horizon_ms=horizon
        )
        ids, values, stats = workflow_statistics(scheduled)
        resources = mechanism_row(replay, mechanism, horizon)
        npz_path = raw_dir / f"healthy__{slug_policy(policy)}.npz"
        np.savez_compressed(
            npz_path,
            workflow_ids=np.asarray(ids),
            wct_ms=np.asarray(values, dtype=float),
            finish_ms=np.asarray([row.finish_ms for row in scheduled], dtype=float),
            capacity_time_ms=np.asarray(
                [row["time_ms"] for row in capacity_trace], dtype=float
            ),
            capacity_rank=np.asarray(
                [row["rank"] for row in capacity_trace], dtype=np.int16
            ),
            capacity_serving_bytes=np.asarray(
                [row["active_serving_kv_bytes"] for row in capacity_trace],
                dtype=np.int64,
            ),
            capacity_peer_backup_bytes=np.asarray(
                [row["peer_hbm_backup_used_bytes"] for row in capacity_trace],
                dtype=np.int64,
            ),
        )
        healthy[policy] = {
            "workflow_ids": ids,
            "wct": values,
            "stats": stats,
            "resource": resources,
            "npz": str(npz_path.relative_to(stage)),
            "npz_sha256": core.sha256(npz_path),
        }

        zero = InstrumentedS2Replay(deepcopy(protocol), policy, capacity)
        zero_tasks, zero_mechanism, _zero_trace = zero.run_rate_injected(
            tasks, [], control_horizon_ms=horizon
        )
        zero_ids, zero_values, _zero_stats = workflow_statistics(zero_tasks)
        zero_resource = mechanism_row(zero, zero_mechanism, horizon)
        zero_equivalence[policy] = (
            zero_ids == ids
            and zero_values == values
            and zero_resource == resources
            and zero_mechanism["poisson_fault_batch_v2"]["accepted_event_count"] == 0
        )

    rows: list[dict[str, Any]] = []
    event_mechanisms: list[dict[str, Any]] = []
    minimum_fault_delta_ms = math.inf
    for rate_text, windows in windows_by_rate.items():
        rate = float(rate_text)
        for window in windows:
            for policy in policies:
                base = healthy[policy]
                memoized = int(window["candidate_event_count"]) == 0
                raw_sha = ""
                if memoized:
                    ids = base["workflow_ids"]
                    values = base["wct"]
                    stats = base["stats"]
                    resources = base["resource"]
                else:
                    replay = InstrumentedS2Replay(deepcopy(protocol), policy, capacity)
                    scheduled, mechanism, _trace = replay.run_rate_injected(
                        tasks,
                        window["event_ledger"],
                        control_horizon_ms=horizon,
                    )
                    ids, values, stats = workflow_statistics(scheduled)
                    resources = mechanism_row(replay, mechanism, horizon)
                    if ids != base["workflow_ids"]:
                        raise AssertionError("event window changed workflow denominator")
                    deltas = [
                        float(value) - float(reference)
                        for value, reference in zip(values, base["wct"])
                    ]
                    minimum_fault_delta_ms = min(minimum_fault_delta_ms, min(deltas))
                    if min(deltas) < -1e-7:
                        raise AssertionError("fault event reduced all-workflow WCT")
                    raw_path = event_dir / (
                        f"r{rate:.3f}__w{int(window['window_index']):04d}"
                        f"__{slug_policy(policy)}.npz"
                    )
                    np.savez_compressed(
                        raw_path,
                        workflow_ids=np.asarray(ids),
                        wct_ms=np.asarray(values, dtype=float),
                        healthy_wct_ms=np.asarray(base["wct"], dtype=float),
                    )
                    raw_sha = core.sha256(raw_path)
                    event_mechanisms.append({
                        "concurrent_workflows_per_worker": concurrency,
                        "workload_seed": seed,
                        "rate_per_1000_gpu_shard_hour": rate,
                        "window_id": window["window_id"],
                        "policy": policy,
                        **resources,
                        "raw_event_npz": str(raw_path.relative_to(stage)),
                        "raw_event_npz_sha256": raw_sha,
                    })
                rows.append({
                    "concurrent_workflows_per_worker": concurrency,
                    "workload_seed": seed,
                    "policy": policy,
                    "rate_per_1000_gpu_shard_hour": rate,
                    "window_id": window["window_id"],
                    "window_index": int(window["window_index"]),
                    "fault_seed": int(window["fault_seed"]),
                    "candidate_event_count": int(window["candidate_event_count"]),
                    "accepted_event_count": int(window["accepted_event_count"]),
                    "noop_event_count": int(window["noop_event_count"]),
                    "memoized_zero_event": memoized,
                    **stats,
                    "healthy_wct_mean_ms": base["stats"]["wct_mean_ms"],
                    "added_wct_mean_ms": stats["wct_mean_ms"]
                    - base["stats"]["wct_mean_ms"],
                    **resources,
                    "workload_universe_sha256": workload_hash,
                    "input_event_ledger_sha256": window["event_ledger_sha256"],
                    "raw_event_npz_sha256": raw_sha,
                })

    payload = {
        "schema_version": 1,
        "concurrent_workflows_per_worker": concurrency,
        "workload_seed": seed,
        "workload_metadata": workload,
        "workload_universe_sha256": workload_hash,
        "zero_event_equivalence": zero_equivalence,
        "healthy": healthy,
        "window_samples": rows,
        "event_mechanisms": event_mechanisms,
        "minimum_fault_minus_healthy_workflow_wct_ms": (
            None if math.isinf(minimum_fault_delta_ms) else minimum_fault_delta_ms
        ),
    }
    path = raw_dir / "cell.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    return {
        "concurrent_workflows_per_worker": concurrency,
        "workload_seed": seed,
        "cell_path": str(path.relative_to(stage)),
        "cell_sha256": core.sha256(path),
        "window_sample_count": len(rows),
        "event_mechanism_count": len(event_mechanisms),
        "status": "PASS",
    }


def paired_ci(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("empty paired sample")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean, 0.0
    se = statistics.stdev(values) / math.sqrt(len(values))
    critical = T95_9DF if len(values) == 10 else 1.96
    half = critical * se
    return mean, mean - half, mean + half, half


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def summarize_seed_rows(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        groups[(
            int(row["concurrent_workflows_per_worker"]),
            int(row["workload_seed"]),
            str(row["policy"]),
            float(row["rate_per_1000_gpu_shard_hour"]),
        )].append(row)
    output = []
    numeric_means = [
        "wct_mean_ms", "wct_p95_ms", "wct_p99_ms", "wct_max_ms",
        "drain_horizon_ms", "added_wct_mean_ms",
        "alpha_final_mean", "alpha_final_min", "alpha_control_mean",
        "alpha_control_std", "alpha_control_range", "pressure_fraction_mean",
        "queue_wait_mean_ms", "queue_wait_p95_ms",
        "serving_hbm_time_mean_gib_per_gpu",
        "peer_hbm_backup_time_mean_gib_per_gpu",
        "dram_time_mean_gib_per_gpu",
        "serving_hbm_peak_mean_gib_per_gpu",
        "peer_hbm_backup_peak_mean_gib_per_gpu",
        "dram_peak_mean_gib_per_gpu",
        "peer_plus_serving_peak_mean_gib_per_gpu",
        "memory_blocked_observation_count",
        "memory_blocked_unique_request_count",
        "primary_priority_reclaimed_bytes",
        "backup_bytes_released_by_control",
        "scheduled_hbm_copy_bytes", "scheduled_dram_copy_bytes",
        "hbm_admission_attempt_count", "hbm_admission_success_count",
        "hbm_admission_rejection_count", "hbm_eviction_count",
        "dram_admission_attempt_count", "dram_admission_success_count",
        "dram_admission_rejection_count", "replica_invalidated_bytes",
    ]
    for (concurrency, seed, policy, rate), rows in sorted(groups.items()):
        row: dict[str, Any] = {
            "concurrent_workflows_per_worker": concurrency,
            "workload_seed": seed,
            "policy": policy,
            "rate_per_1000_gpu_shard_hour": rate,
            "window_count": len(rows),
            "event_bearing_window_count": sum(
                int(item["candidate_event_count"]) > 0 for item in rows
            ),
            "candidate_event_count": sum(
                int(item["candidate_event_count"]) for item in rows
            ),
            "accepted_event_count": sum(
                int(item["accepted_event_count"]) for item in rows
            ),
            "noop_event_count": sum(int(item["noop_event_count"]) for item in rows),
            "memoized_zero_event_window_count": sum(
                bool(item["memoized_zero_event"]) for item in rows
            ),
            "workload_universe_sha256": rows[0]["workload_universe_sha256"],
            "registered_control_opportunity_count": int(
                rows[0]["registered_control_opportunity_count"]
            ),
            "executed_policy_control_count": int(
                rows[0]["executed_policy_control_count"]
            ),
            "capacity_violation_count_max": max(
                int(item["capacity_violation_count"]) for item in rows
            ),
            "copy_byte_conservation_error_max_abs": max(
                abs(int(item["hbm_copy_byte_conservation_error"]))
                + abs(int(item["dram_copy_byte_conservation_error"]))
                for item in rows
            ),
            "seal_before_dispatch_count_max": max(
                int(item["seal_before_dispatch_count"]) for item in rows
            ),
            "future_backup_projection_dispatch_count_max": max(
                int(item["future_backup_projection_dispatch_count"])
                for item in rows
            ),
            "health_burden_application_count_max": max(
                int(item["health_burden_application_count"]) for item in rows
            ),
            "reuse_fraction": 0.0,
        }
        for key in numeric_means:
            row[key] = mean_numeric(rows, key)
        output.append(row)
    return output


def summarize_paired(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            int(row["concurrent_workflows_per_worker"]),
            int(row["workload_seed"]),
            float(row["rate_per_1000_gpu_shard_hour"]),
            str(row["policy"]),
        ): row
        for row in seed_rows
    }
    keys = sorted({key[:3] for key in lookup})
    output = []
    for concurrency, seed, rate in keys:
        native = lookup[(concurrency, seed, rate, "Native Mooncake")]
        aegis = lookup[(concurrency, seed, rate, "AegisKV")]
        row: dict[str, Any] = {
            "concurrent_workflows_per_worker": concurrency,
            "workload_seed": seed,
            "rate_per_1000_gpu_shard_hour": rate,
            "workload_universe_sha256": native["workload_universe_sha256"],
            "paired_universe_hash_equal": (
                native["workload_universe_sha256"]
                == aegis["workload_universe_sha256"]
            ),
            "candidate_event_count": int(native["candidate_event_count"]),
            "accepted_event_count": int(native["accepted_event_count"]),
            "noop_event_count": int(native["noop_event_count"]),
        }
        for metric in ("wct_mean_ms", "wct_p95_ms", "wct_p99_ms"):
            nvalue = float(native[metric])
            avalue = float(aegis[metric])
            stem = metric[:-3] if metric.endswith("_ms") else metric
            row[f"native_{metric}"] = nvalue
            row[f"aegis_{metric}"] = avalue
            row[f"{stem}_saved_ms"] = nvalue - avalue
            row[f"{stem}_gain_pct"] = (nvalue - avalue) / nvalue * 100.0
        for key in (
            "alpha_final_mean", "alpha_final_min", "alpha_control_std",
            "alpha_control_range", "pressure_fraction_mean",
            "serving_hbm_time_mean_gib_per_gpu",
            "peer_hbm_backup_time_mean_gib_per_gpu",
            "dram_time_mean_gib_per_gpu",
            "serving_hbm_peak_mean_gib_per_gpu",
            "peer_hbm_backup_peak_mean_gib_per_gpu",
            "dram_peak_mean_gib_per_gpu",
            "memory_blocked_unique_request_count",
            "primary_priority_reclaimed_bytes",
            "backup_bytes_released_by_control",
            "hbm_admission_rejection_count", "dram_admission_rejection_count",
        ):
            row[f"aegis_{key}"] = aegis[key]
        output.append(row)
    return output


def summarize_ci(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        groups[(
            int(row["concurrent_workflows_per_worker"]),
            float(row["rate_per_1000_gpu_shard_hour"]),
        )].append(row)
    output = []
    metrics = [
        "wct_mean_gain_pct", "wct_p95_gain_pct", "wct_p99_gain_pct",
        "wct_mean_saved_ms", "wct_p95_saved_ms", "wct_p99_saved_ms",
        "native_wct_mean_ms", "aegis_wct_mean_ms",
        "native_wct_p95_ms", "aegis_wct_p95_ms",
        "native_wct_p99_ms", "aegis_wct_p99_ms",
        "aegis_alpha_final_mean", "aegis_alpha_final_min",
        "aegis_alpha_control_std", "aegis_alpha_control_range",
        "aegis_pressure_fraction_mean",
        "aegis_serving_hbm_time_mean_gib_per_gpu",
        "aegis_peer_hbm_backup_time_mean_gib_per_gpu",
        "aegis_dram_time_mean_gib_per_gpu",
        "aegis_memory_blocked_unique_request_count",
        "aegis_primary_priority_reclaimed_bytes",
        "aegis_backup_bytes_released_by_control",
        "aegis_hbm_admission_rejection_count",
        "aegis_dram_admission_rejection_count",
    ]
    for (concurrency, rate), rows in sorted(groups.items()):
        result: dict[str, Any] = {
            "concurrent_workflows_per_worker": concurrency,
            "rate_per_1000_gpu_shard_hour": rate,
            "paired_seed_count": len(rows),
            "workload_seeds": sorted(int(row["workload_seed"]) for row in rows),
            "candidate_event_count": sum(int(row["candidate_event_count"]) for row in rows),
            "accepted_event_count": sum(int(row["accepted_event_count"]) for row in rows),
            "noop_event_count": sum(int(row["noop_event_count"]) for row in rows),
        }
        for key in metrics:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if not values:
                result[f"{key}_mean"] = None
                result[f"{key}_ci95_low"] = None
                result[f"{key}_ci95_high"] = None
                result[f"{key}_ci95_half_width"] = None
                continue
            mean, low, high, half = paired_ci(values)
            result[f"{key}_mean"] = mean
            result[f"{key}_ci95_low"] = low
            result[f"{key}_ci95_high"] = high
            result[f"{key}_ci95_half_width"] = half
        output.append(result)
    return output


def render_figures(stage: Path, ci_rows: list[dict[str, Any]]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })
    figure_dir = stage / "figures"
    figure_dir.mkdir(exist_ok=True)
    rates = sorted({float(row["rate_per_1000_gpu_shard_hour"]) for row in ci_rows})
    concurrencies = sorted({int(row["concurrent_workflows_per_worker"]) for row in ci_rows})
    lookup = {
        (float(row["rate_per_1000_gpu_shard_hour"]), int(row["concurrent_workflows_per_worker"])): row
        for row in ci_rows
    }
    colors = {"mean": "#1769aa", "p95": "#d95f02", "p99": "#2a9d55"}
    markers = {"mean": "o", "p95": "s", "p99": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharey=True)
    for axis, rate in zip(axes, rates):
        for label in ("mean", "p95", "p99"):
            key = f"wct_{label}_gain_pct"
            values = [lookup[(rate, c)][f"{key}_mean"] for c in concurrencies]
            errors = [lookup[(rate, c)][f"{key}_ci95_half_width"] for c in concurrencies]
            axis.errorbar(
                concurrencies, values, yerr=errors, label=label,
                color=colors[label], marker=markers[label], linewidth=1.4,
                markersize=4.0, capsize=2.0,
            )
        axis.axhline(0.0, color="#555555", linewidth=0.75, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_xticks(concurrencies, [str(value) for value in concurrencies])
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.set_title(f"{rate:g} failures / 1K GPU-h")
        axis.set_xlabel("Concurrent workflows / worker")
    axes[0].set_ylabel("AegisKV WCT gain vs Native (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.23, top=0.78, wspace=0.12)
    main_pdf = figure_dir / "s2_v313_full_concurrency_observed.pdf"
    main_png = figure_dir / "s2_v313_full_concurrency_observed.png"
    fig.savefig(main_pdf)
    fig.savefig(main_png, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.8), sharex="col")
    resource_colors = {
        "Serving HBM": "#333333",
        "Peer-HBM backup": "#1769aa",
        "DRAM": "#d95f02",
    }
    for col, rate in enumerate(rates):
        rows = [lookup[(rate, c)] for c in concurrencies]
        axes[0, col].plot(
            concurrencies,
            [row["aegis_alpha_final_mean_mean"] for row in rows],
            color="#1769aa", marker="o", linewidth=1.4, markersize=4,
        )
        axes[0, col].fill_between(
            concurrencies,
            [row["aegis_alpha_final_mean_ci95_low"] for row in rows],
            [row["aegis_alpha_final_mean_ci95_high"] for row in rows],
            color="#1769aa", alpha=0.18, linewidth=0,
        )
        axes[0, col].set_ylim(-0.015, 0.315)
        axes[0, col].set_title(f"{rate:g} failures / 1K GPU-h")
        axes[1, col].plot(
            concurrencies,
            [100.0 * row["aegis_pressure_fraction_mean_mean"] for row in rows],
            color="#7b3294", marker="D", linewidth=1.4, markersize=3.8,
        )
        axes[1, col].set_ylim(-4, 104)
        resource_keys = {
            "Serving HBM": "aegis_serving_hbm_time_mean_gib_per_gpu_mean",
            "Peer-HBM backup": "aegis_peer_hbm_backup_time_mean_gib_per_gpu_mean",
            "DRAM": "aegis_dram_time_mean_gib_per_gpu_mean",
        }
        for name, key in resource_keys.items():
            axes[2, col].plot(
                concurrencies, [row[key] for row in rows], label=name,
                color=resource_colors[name], marker={
                    "Serving HBM": "o", "Peer-HBM backup": "s", "DRAM": "^"
                }[name], linewidth=1.35, markersize=3.8,
            )
        for row in range(3):
            axes[row, col].set_xscale("log", base=2)
            axes[row, col].set_xticks(concurrencies, [str(value) for value in concurrencies])
            axes[row, col].grid(axis="y", alpha=0.24, linewidth=0.55)
        axes[2, col].set_xlabel("Concurrent workflows / worker")
    axes[0, 0].set_ylabel("Final alpha")
    axes[1, 0].set_ylabel("Pressure targets (%)")
    axes[2, 0].set_ylabel("Time-mean occupancy\n(GiB / GPU)")
    handles, labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.91,
                        hspace=0.28, wspace=0.16)
    mech_pdf = figure_dir / "s2_v313_full_concurrency_mechanisms.pdf"
    mech_png = figure_dir / "s2_v313_full_concurrency_mechanisms.png"
    fig.savefig(mech_pdf)
    fig.savefig(mech_png, dpi=300)
    plt.close(fig)
    return {
        "main_pdf": str(main_pdf.relative_to(stage)),
        "main_png": str(main_png.relative_to(stage)),
        "mechanism_pdf": str(mech_pdf.relative_to(stage)),
        "mechanism_png": str(mech_png.relative_to(stage)),
    }


def pdf_min_font_pt(path: Path) -> float:
    import fitz
    document = fitz.open(path)
    sizes = []
    for page in document:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = str(span.get("text", "")).strip()
                    if text:
                        sizes.append(float(span["size"]))
    document.close()
    return min(sizes) if sizes else 0.0


def source_hash_checks(spec: dict[str, Any]) -> dict[str, bool]:
    checks = {}
    for name, source in spec["sources"].items():
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        checks[f"source_sha256_{name}"] = path.is_file() and core.sha256(path) == source["sha256"]
    return checks


def run_tests(stage: Path) -> list[dict[str, Any]]:
    tests = (
        ("test_active_hbm_v313_s2_full_concurrency_protocol.py", "RESULT 12/12 PASS"),
        ("test_active_hbm_v313_protocol.py", "RESULT 6/6 PASS"),
        ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
        ("test_poisson_fault_batch_v2.py", "RESULT 6/6 PASS"),
        ("test_poisson_fault_injection_v1.py", "RESULT 6/6 PASS"),
    )
    test_dir = stage / "tests"
    test_dir.mkdir(exist_ok=True)
    output = []
    environment = dict(os.environ)
    environment["TMPDIR"] = str(stage / "tmp")
    environment["MPLCONFIGDIR"] = str(stage / "mplconfig")
    (stage / "tmp").mkdir(exist_ok=True)
    (stage / "mplconfig").mkdir(exist_ok=True)
    for filename, marker in tests:
        result = subprocess.run(
            ["python", str(ROOT / filename)], cwd=ROOT, env=environment,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        log = test_dir / f"{Path(filename).stem}.log"
        log.write_text(result.stdout, encoding="utf-8")
        passed = result.returncode == 0 and marker in result.stdout
        output.append({
            "test": filename, "marker": marker, "returncode": result.returncode,
            "passed": passed, "log": str(log.relative_to(stage)),
        })
        if not passed:
            raise RuntimeError(f"test gate failed: {filename}\n{result.stdout}")
    return output


def make_windows(
    *, rates: list[float], seeds: list[int], windows_per_seed: int,
    spec: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    exposure = spec["failure_exposure"]
    return {
        f"{rate:.3f}": batch.generate_batch_windows(
            rate=rate,
            workload_seeds=seeds,
            windows_per_seed=windows_per_seed,
            horizon_ms=float(exposure["window_horizon_ms"]),
            dp=int(spec["hardware_topology"]["system_dp"]),
            gpus_per_node=int(spec["hardware_topology"]["gpus_per_node"]),
            repair_ms=float(exposure["domain_repair_ms"]),
            failure_type_weights=dict(exposure["failure_type_weights"]),
        )
        for rate in rates
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "full"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    if args.mode == "gate":
        concurrencies = [int(v) for v in spec["gate"]["concurrent_workflows_per_worker"]]
        seeds = [int(v) for v in spec["gate"]["workload_seeds"]]
        rates = [float(v) for v in spec["gate"]["rates"]]
        windows_per_seed = int(spec["gate"]["windows_per_rate"])
        result_name = "active_hbm_v313_s2_full_concurrency_gate"
        expected_seed_rows = int(spec["gate"]["expected_seed_policy_rate_concurrency_rows"])
    else:
        concurrencies = [int(v) for v in spec["workload"]["concurrent_workflows_per_worker"]]
        seeds = [int(v) for v in spec["workload"]["paired_workload_seeds"]]
        rates = [float(v) for v in spec["failure_exposure"]["failure_rates_per_1000_gpu_shard_hour"]]
        windows_per_seed = int(spec["failure_exposure"]["independent_windows_per_workload_seed"])
        result_name = "active_hbm_v313_s2_full_concurrency_physical_rate_v1"
        expected_seed_rows = int(spec["matrix"]["seed_policy_rate_concurrency_summary_rows"])
    policies = list(spec["policies"])
    target = ROOT / "results" / result_name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{result_name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()
    (stage / "raw_events").mkdir()

    tests = run_tests(stage)
    source_checks = source_hash_checks(spec)
    if not all(source_checks.values()):
        raise RuntimeError([key for key, value in source_checks.items() if not value])
    windows = make_windows(
        rates=rates, seeds=seeds, windows_per_seed=windows_per_seed, spec=spec
    )
    ledger_path = stage / "preregistered_window_ledgers.jsonl"
    with ledger_path.open("w", encoding="utf-8") as stream:
        for rate in rates:
            for row in windows[f"{rate:.3f}"]:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    execution_freeze = {
        "schema_version": 1,
        "generated_before_policy_replay": True,
        "mode": args.mode,
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner_sha256": core.sha256(Path(__file__)),
        "workload_builder": "active_hbm_v31.build_workload",
        "old_core_build_workload_lineage_rejected": True,
        "predicted_slack_in_workload_universe_hash": True,
        "ledger_sha256": core.sha256(ledger_path),
        "concurrent_workflows_per_worker": concurrencies,
        "workload_seeds": seeds,
        "rates": rates,
        "windows_per_seed": windows_per_seed,
        "window_counts_by_rate": {
            f"{rate:.3f}": len(windows[f"{rate:.3f}"]) for rate in rates
        },
        "candidate_event_counts_by_rate": {
            f"{rate:.3f}": sum(
                int(row["candidate_event_count"])
                for row in windows[f"{rate:.3f}"]
            ) for rate in rates
        },
        "source_hash_checks": source_checks,
        "selection": spec["selection"],
    }
    freeze_path = stage / "execution_freeze.json"
    freeze_path.write_text(json.dumps(execution_freeze, indent=2, sort_keys=True) + "\n")

    jobs = []
    for concurrency in concurrencies:
        for seed in seeds:
            jobs.append({
                "stage": str(stage),
                "concurrency": concurrency,
                "seed": seed,
                "horizon_ms": float(spec["failure_exposure"]["window_horizon_ms"]),
                "policies": policies,
                "windows_by_rate": {
                    f"{rate:.3f}": [
                        row for row in windows[f"{rate:.3f}"]
                        if int(row["workload_seed"]) == seed
                    ]
                    for rate in rates
                },
            })
    job_results = []
    failures = []
    max_workers = min(max(int(args.workers), 1), 50, len(jobs))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(execute_job, job): job for job in jobs}
        for index, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            try:
                result = future.result()
                job_results.append(result)
                print(
                    f"S2 {args.mode} {index}/{len(jobs)} "
                    f"C={result['concurrent_workflows_per_worker']} "
                    f"seed={result['workload_seed']} events={result['event_mechanism_count']}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "concurrent_workflows_per_worker": job["concurrency"],
                    "workload_seed": job["seed"],
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(f"S2 FAIL {failure}", flush=True)
    (stage / "job_results.json").write_text(
        json.dumps(sorted(job_results, key=lambda row: (
            row["concurrent_workflows_per_worker"], row["workload_seed"]
        )), indent=2, sort_keys=True) + "\n"
    )
    (stage / "job_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} S2 jobs failed; stage retained at {stage}")

    window_rows = []
    event_mechanisms = []
    cell_rows = []
    zero_equivalence = []
    workload_metadata = []
    minimum_fault_deltas = []
    for result in job_results:
        cell_path = stage / result["cell_path"]
        with gzip.open(cell_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        window_rows.extend(payload["window_samples"])
        event_mechanisms.extend(payload["event_mechanisms"])
        zero_equivalence.extend(payload["zero_event_equivalence"].values())
        workload_metadata.append({
            "concurrent_workflows_per_worker": payload["concurrent_workflows_per_worker"],
            "workload_seed": payload["workload_seed"],
            "workload_universe_sha256": payload["workload_universe_sha256"],
            **payload["workload_metadata"],
        })
        if payload["minimum_fault_minus_healthy_workflow_wct_ms"] is not None:
            minimum_fault_deltas.append(
                float(payload["minimum_fault_minus_healthy_workflow_wct_ms"])
            )
        cell_rows.append(result)

    seed_rows = summarize_seed_rows(window_rows)
    paired_rows = summarize_paired(seed_rows)
    ci_rows = summarize_ci(paired_rows)
    write_csv(stage / "window_samples.csv", window_rows)
    write_csv(stage / "seed_policy_metrics.csv", seed_rows)
    write_csv(stage / "paired_seed_metrics.csv", paired_rows)
    write_csv(stage / "paired_ci.csv", ci_rows)
    write_csv(stage / "workload_metadata.csv", workload_metadata)
    if event_mechanisms:
        write_csv(stage / "event_mechanisms.csv", event_mechanisms)
    else:
        (stage / "event_mechanisms.csv").write_text(
            "concurrent_workflows_per_worker,workload_seed,policy\n",
            encoding="utf-8",
        )

    figures: dict[str, Any] = {}
    figure_checks: dict[str, bool] = {}
    if args.mode == "full":
        figures = render_figures(stage, ci_rows)
        from PIL import Image
        for key in ("main_png", "mechanism_png"):
            width, height = Image.open(stage / figures[key]).size
            figures[f"{key}_width_px"] = width
            figures[f"{key}_height_px"] = height
        for key in ("main_pdf", "mechanism_pdf"):
            figures[f"{key}_min_font_pt"] = pdf_min_font_pt(stage / figures[key])
        figure_checks = {
            "png_width_at_least_1800": min(
                figures["main_png_width_px"], figures["mechanism_png_width_px"]
            ) >= 1800,
            "pdf_min_font_at_least_7pt": min(
                figures["main_pdf_min_font_pt"],
                figures["mechanism_pdf_min_font_pt"],
            ) >= 6.99,
            "vector_pdf_and_png_present": all(
                (stage / figures[key]).is_file()
                for key in ("main_pdf", "main_png", "mechanism_pdf", "mechanism_png")
            ),
        }

    expected_window_rows = (
        len(concurrencies) * len(seeds) * len(policies)
        * len(rates) * windows_per_seed
    )
    ledger_pairing = all(
        len({
            str(row["input_event_ledger_sha256"])
            for row in window_rows
            if int(row["workload_seed"]) == seed
            and int(row["window_index"]) == index
            and math.isclose(float(row["rate_per_1000_gpu_shard_hour"]), rate)
        }) == 1
        for seed in seeds for rate in rates for index in range(windows_per_seed)
    )
    expected_control_times = [30000.0, 60000.0, 90000.0]
    checks = {
        **source_checks,
        "all_tests_pass": all(row["passed"] for row in tests),
        "old_core_build_workload_oracle_lineage_fail_closed": (
            execution_freeze["workload_builder"] == "active_hbm_v31.build_workload"
            and execution_freeze["old_core_build_workload_lineage_rejected"]
            and execution_freeze["predicted_slack_in_workload_universe_hash"]
            and all(
                row["slack_predictor"]["reads_realized_output_tokens"] is False
                and row["slack_predictor"]["reads_realized_service_ms"] is False
                for row in workload_metadata
            )
        ),
        "ledger_generated_and_hashed_before_policy_replay": execution_freeze[
            "generated_before_policy_replay"
        ],
        "all_worker_jobs_completed": len(job_results) == len(jobs),
        "no_worker_failures": not failures,
        "exact_seed_policy_rate_concurrency_rows": len(seed_rows) == expected_seed_rows,
        "exact_window_sample_rows": len(window_rows) == expected_window_rows,
        "all_registered_cells_and_seeds_retained": len({
            (
                int(row["concurrent_workflows_per_worker"]),
                int(row["workload_seed"]),
                str(row["policy"]),
                float(row["rate_per_1000_gpu_shard_hour"]),
            ) for row in seed_rows
        }) == expected_seed_rows,
        "event_ledgers_paired_across_policies_and_concurrency": ledger_pairing,
        "workload_universe_paired_across_policies": all(
            bool(row["paired_universe_hash_equal"]) for row in paired_rows
        ),
        "zero_event_memoization_exact": all(zero_equivalence),
        "all_workflow_denominators_retained": all(
            int(row["window_count"]) == windows_per_seed for row in seed_rows
        ),
        "fault_replay_never_reduces_workflow_wct": (
            not minimum_fault_deltas or min(minimum_fault_deltas) >= -1e-7
        ),
        "no_capacity_violations": all(
            int(row["capacity_violation_count_max"]) == 0 for row in seed_rows
        ),
        "copy_byte_conservation": all(
            int(row["copy_byte_conservation_error_max_abs"]) == 0 for row in seed_rows
        ),
        "asynchronous_copy_not_directly_charged_to_wct": all(
            int(row["health_burden_application_count_max"]) == 0 for row in seed_rows
        ),
        "no_future_backup_or_seal_oracle": all(
            int(row["future_backup_projection_dispatch_count_max"]) == 0
            and int(row["seal_before_dispatch_count_max"]) == 0
            for row in seed_rows
        ),
        "exact_three_aegis_control_opportunities": all(
            int(row["registered_control_opportunity_count"]) == 3
            and (
                int(row["executed_policy_control_count"]) == 3
                if row["policy"] == "AegisKV" else True
            )
            for row in seed_rows
        ),
        "resource_integrals_nonnegative": all(
            float(row[key]) >= -1e-12
            for row in seed_rows
            for key in (
                "serving_hbm_time_mean_gib_per_gpu",
                "peer_hbm_backup_time_mean_gib_per_gpu",
                "dram_time_mean_gib_per_gpu",
            )
        ),
        "shared_hbm_peak_within_frozen_budget": all(
            float(row["peer_plus_serving_peak_mean_gib_per_gpu"])
            <= float(spec["capacity"]["serving_plus_peer_backup_budget_bytes_per_gpu"]) / GIB
            + 1e-9
            for row in seed_rows
        ),
        "reuse_fraction_exactly_zero": all(
            float(row["reuse_fraction"]) == 0.0 for row in seed_rows
        ),
        "no_performance_direction_gate": spec["selection"].startswith("NONE"),
        **figure_checks,
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": args.mode,
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner_sha256": core.sha256(Path(__file__)),
        "execution_freeze_sha256": core.sha256(freeze_path),
        "ledger_sha256": core.sha256(ledger_path),
        "old_fixed_c16_lineage": "INVALIDATED_ORACLE_SLACK_DIAGNOSTIC_ONLY",
        "minimum_fault_minus_healthy_workflow_wct_ms": (
            min(minimum_fault_deltas) if minimum_fault_deltas else None
        ),
        "expected_control_times_ms": expected_control_times,
        "actual_candidate_events_by_rate": execution_freeze[
            "candidate_event_counts_by_rate"
        ],
        "figures": figures,
        "evidence_boundary": spec["evidence_boundary"],
    }
    audit_path = stage / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(f"audit FAIL {audit['failed_checks']}; stage={stage}")

    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "mode": args.mode,
        "paper_ready": False,
        "protocol": SPEC_PATH.name,
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner": Path(__file__).name,
        "runner_sha256": core.sha256(Path(__file__)),
        "audit": "audit.json",
        "audit_sha256": core.sha256(audit_path),
        "execution_freeze": "execution_freeze.json",
        "ledger": "preregistered_window_ledgers.jsonl",
        "window_samples": "window_samples.csv",
        "seed_policy_metrics": "seed_policy_metrics.csv",
        "paired_seed_metrics": "paired_seed_metrics.csv",
        "paired_ci": "paired_ci.csv",
        "event_mechanisms": "event_mechanisms.csv",
        "workload_metadata": "workload_metadata.csv",
        "figures": figures,
        "seed_policy_rate_concurrency_rows": len(seed_rows),
        "paired_seed_rows": len(paired_rows),
        "paired_ci_rows": len(ci_rows),
        "window_sample_rows": len(window_rows),
        "execution_workers": max_workers,
        "old_fixed_c16_lineage": "INVALIDATED_ORACLE_SLACK_DIAGNOSTIC_ONLY",
        "workload_builder": "active_hbm_v31.build_workload",
        "predicted_slack_in_universe_hash": True,
        "no_result_or_seed_selection": True,
        "s9_status": "AWAITING_NEW_INDEPENDENT_PHYSICAL_HOLDOUT_NO_POSTHOC_141298_SPLIT",
        "evidence_boundary": spec["evidence_boundary"],
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (stage / SPEC_PATH.name).write_text(SPEC_PATH.read_text(), encoding="utf-8")
    caption = (
        "S2 uses actual 90 s Poisson exposure windows and ten paired workload seeds. "
        "All-workflow WCT is primary; error bars are paired Student-t 95% CIs. "
        "A100 path bandwidths, activation, replacement, and repair are modeled/unmeasured, "
        "and Native Mooncake is a modeled Router+Store approximation.\n"
    )
    (stage / "figure_caption.txt").write_text(caption, encoding="utf-8")
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())),
        encoding="utf-8",
    )
    os.replace(stage, target)
    print(target, flush=True)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}", flush=True)
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}", flush=True)
    print(f"sha256sums_sha256={core.sha256(target / 'SHA256SUMS')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
