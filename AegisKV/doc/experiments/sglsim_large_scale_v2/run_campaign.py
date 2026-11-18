#!/usr/bin/env python3
"""Parallel, resumable runner for the frozen large-scale campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matrix import load_protocol, matrix_manifest, validate
from workload import CompactWeka
from sglsim.srt.simulation.aegis_large_scale import (
    CELL_SCHEMA_VERSION,
    REUSE_BIN_CONTRACT,
    run_cell,
)


PROTOCOL = ROOT / "protocol.json"
MATRIX = ROOT / "matrix.json"
DEFAULT_RESULTS = ROOT / "results" / "campaign_v2"


def _cache_manifest_bundle(
    manifest_path: Path,
    *,
    auxiliary_records: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify a generated cache manifest and resolve every local payload."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_payload = dict(manifest)
    expected_canonical = str(canonical_payload.pop("canonical_sha256"))
    actual_canonical = hashlib.sha256(
        json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if actual_canonical != expected_canonical:
        raise ValueError(f"cache manifest canonical SHA mismatch: {manifest_path}")

    base = manifest_path.parent.resolve()
    records: dict[str, Any] = dict(manifest.get("arrays") or {})
    for name in auxiliary_records:
        record = manifest.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"cache manifest is missing record {name}: {manifest_path}")
        records[name] = record
    resolved: dict[str, Path] = {}
    for name, record in records.items():
        if not isinstance(record, dict) or not {
            "path",
            "sha256",
        }.issubset(record):
            raise ValueError(f"invalid cache file record {name}: {manifest_path}")
        candidate = (base / str(record["path"])).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError(
                f"cache file escapes bundle directory: {candidate}"
            ) from exc
        if not candidate.is_file() or _sha256(candidate) != record["sha256"]:
            raise ValueError(f"cache file SHA mismatch: {candidate}")
        resolved[name] = candidate
    return manifest, resolved


def _source_lineage(protocol_path: Path, matrix_path: Path) -> dict[str, Any]:
    compact_manifest_path = ROOT / "cache/weka_compact_v1/manifest.json"
    alg1_manifest_path = ROOT / "cache/modeled_algorithm1_v1/manifest.json"
    compact_manifest, compact_files = _cache_manifest_bundle(
        compact_manifest_path, auxiliary_records=("workflow_reuse",)
    )
    alg1_manifest, alg1_files = _cache_manifest_bundle(alg1_manifest_path)
    alg1_producer = (ROOT / "prepare_modeled_algorithm1.py").resolve()
    if alg1_manifest.get("producer", {}).get("sha256") != _sha256(
        alg1_producer
    ):
        raise ValueError("Algorithm-1 cache producer SHA mismatch")
    if alg1_manifest.get("protocol", {}).get("sha256") != _sha256(
        protocol_path
    ):
        raise ValueError("Algorithm-1 cache protocol SHA mismatch")
    if (
        alg1_manifest.get("compact_workload", {}).get("sha256")
        != _sha256(compact_manifest_path)
        or alg1_manifest.get("compact_workload", {}).get("canonical_sha256")
        != compact_manifest["canonical_sha256"]
    ):
        raise ValueError("Algorithm-1 cache compact-workload binding mismatch")

    paths = {
        "protocol": protocol_path.resolve(),
        "matrix": matrix_path.resolve(),
        "runner": Path(__file__).resolve(),
        "workload_projection": (ROOT / "workload.py").resolve(),
        "compact_trace_generator": (ROOT / "prepare_trace.py").resolve(),
        "modeled_algorithm1_generator": alg1_producer,
        "matrix_builder": (ROOT / "matrix.py").resolve(),
        "core": (
            REPO_ROOT
            / "python/sglsim/srt/simulation/aegis_large_scale.py"
        ).resolve(),
        "independent_tiers": (
            REPO_ROOT
            / "python/sglsim/srt/simulation/aegis_independent_tiers.py"
        ).resolve(),
        "full_design": (
            REPO_ROOT
            / "python/sglsim/srt/simulation/aegis_full_design.py"
        ).resolve(),
        "transfer_engine": (
            REPO_ROOT
            / "python/sglsim/srt/simulation/aegis_transfer_engine.py"
        ).resolve(),
        "residual_dag": (
            REPO_ROOT
            / "python/sglsim/srt/simulation/aegis_residual_dag.py"
        ).resolve(),
        "baselines": (
            REPO_ROOT / "python/sglsim/srt/simulation/aegis_baselines.py"
        ).resolve(),
        "compact_manifest": compact_manifest_path.resolve(),
        "modeled_algorithm1_manifest": alg1_manifest_path.resolve(),
        "authoritative_current_design": (
            REPO_ROOT / "AegisKV/design/current_design.md"
        ).resolve(),
    }
    paths.update(
        {
            f"compact_cache:{name}": path
            for name, path in compact_files.items()
        }
    )
    paths.update(
        {
            f"modeled_algorithm1_cache:{name}": path
            for name, path in alg1_files.items()
        }
    )
    records = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "simulator_cell_schema_version": CELL_SCHEMA_VERSION,
        "files": records,
        "matrix_canonical_sha256": matrix["canonical_sha256"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["lineage_sha256"] = hashlib.sha256(canonical).hexdigest()
    payload["lineage_id"] = payload["lineage_sha256"][:12]
    return payload


def _validate_lineage(lineage: dict[str, Any]) -> None:
    for name, record in lineage["files"].items():
        path = Path(record["path"])
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"lineage source changed during campaign: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_rows(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    if limit <= 1:
        return [rows[0]]
    indices = {
        round(index * (len(rows) - 1) / (limit - 1)) for index in range(limit)
    }
    return [rows[index] for index in sorted(indices)]


def _request_sample(
    rows: Sequence[dict[str, Any]], limit: int, *, seed: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['request_id']}".encode()
        ).digest(),
    )
    selected = ranked[:limit]
    failed = [row for row in rows if int(row.get("failure_count", 0)) > 0]
    by_id = {str(row["request_id"]): row for row in (*selected, *failed)}
    return [by_id[key] for key in sorted(by_id)]


def _window_diagnostics(rows: Sequence[dict[str, Any]], *, dp_size: int) -> dict[str, Any]:
    if not rows:
        return {
            "window_count": 0,
            "active_rank_load_mean": 0.0,
            "queued_per_rank_mean": 0.0,
            "pressure_fraction_mean": 0.0,
            "stable_alpha_increase_transitions": 0,
            "pressure_alpha_decrease_transitions": 0,
        }
    active = [float(row["active_requests"]) / max(dp_size, 1) for row in rows]
    queued = [float(row["queued_requests"]) / max(dp_size, 1) for row in rows]
    pressure = [float(row["pressure_fraction"]) for row in rows]
    alpha = [float(row["effective_alpha_mean"]) for row in rows]
    stable_increase = 0
    pressure_decrease = 0
    for prior, current in zip(rows, rows[1:]):
        delta = float(current["effective_alpha_mean"]) - float(
            prior["effective_alpha_mean"]
        )
        if float(prior["pressure_fraction"]) == 0.0 and delta > 1e-12:
            stable_increase += 1
        if float(prior["pressure_fraction"]) > 0.0 and delta < -1e-12:
            pressure_decrease += 1
    return {
        "window_count": len(rows),
        "active_rank_load_mean": sum(active) / len(active),
        "active_rank_load_peak": max(active),
        "queued_per_rank_mean": sum(queued) / len(queued),
        "queued_per_rank_peak": max(queued),
        "pressure_fraction_mean": sum(pressure) / len(pressure),
        "pressure_fraction_peak": max(pressure),
        "effective_alpha_final": alpha[-1],
        "stable_alpha_increase_transitions": stable_increase,
        "pressure_alpha_decrease_transitions": pressure_decrease,
    }


def _active_workflow_stats(
    rows: Sequence[dict[str, Any]],
    *,
    finish_key: str,
    clip_start_ms: float | None = None,
    clip_end_ms: float | None = None,
) -> dict[str, Any]:
    """Time-weighted active-workflow statistics from realized intervals."""

    if not rows:
        return {"time_mean": 0.0, "time_p50": 0.0, "time_p95": 0.0, "peak": 0}
    left = (
        min(float(row["arrival_ms"]) for row in rows)
        if clip_start_ms is None
        else float(clip_start_ms)
    )
    right = (
        max(float(row[finish_key]) for row in rows)
        if clip_end_ms is None
        else float(clip_end_ms)
    )
    if right <= left:
        return {"time_mean": 0.0, "time_p50": 0.0, "time_p95": 0.0, "peak": 0}
    events: dict[float, int] = {}
    for row in rows:
        start = max(float(row["arrival_ms"]), left)
        finish = min(float(row[finish_key]), right)
        if finish <= start:
            continue
        events[start] = events.get(start, 0) + 1
        events[finish] = events.get(finish, 0) - 1
    level = 0
    previous = left
    weighted: list[tuple[int, float]] = []
    for time_ms in sorted(events):
        if time_ms > previous:
            weighted.append((level, time_ms - previous))
        level += events[time_ms]
        previous = time_ms
    if previous < right:
        weighted.append((level, right - previous))
    duration = sum(span for _level, span in weighted)

    def quantile(q: float) -> float:
        threshold = duration * q
        cumulative = 0.0
        for active, span in sorted(weighted):
            cumulative += span
            if cumulative >= threshold:
                return float(active)
        return 0.0

    return {
        "time_mean": sum(active * span for active, span in weighted)
        / max(duration, 1e-12),
        "time_p50": quantile(0.50),
        "time_p95": quantile(0.95),
        "peak": max((active for active, _span in weighted), default=0),
        "interval_start_ms": left,
        "interval_end_ms": right,
        "interval_duration_ms": duration,
    }


def _realized_concurrency(
    workflow_rows: Sequence[dict[str, Any]], workload: dict[str, Any]
) -> dict[str, Any]:
    start = float(workload["steady_measurement_start_ms"])
    end = float(workload["steady_measurement_end_ms"])
    return {
        "semantics": (
            "recomputed from workflow arrival to base/simulated join completion; "
            "distinct from the fixed offered trace-lifespan schedule"
        ),
        "base_full_horizon": _active_workflow_stats(
            workflow_rows, finish_key="base_join_completion_ms"
        ),
        "simulated_full_horizon": _active_workflow_stats(
            workflow_rows, finish_key="simulated_join_completion_ms"
        ),
        "base_offered_steady_interval": _active_workflow_stats(
            workflow_rows,
            finish_key="base_join_completion_ms",
            clip_start_ms=start,
            clip_end_ms=end,
        ),
        "simulated_offered_steady_interval": _active_workflow_stats(
            workflow_rows,
            finish_key="simulated_join_completion_ms",
            clip_start_ms=start,
            clip_end_ms=end,
        ),
    }


def _scenario_config(
    scenario: dict[str, Any],
    protocol: dict[str, Any],
    *,
    request_count: int,
    reuse_fraction: float,
    measurement_start_ms: float,
    measurement_end_ms: float,
) -> dict[str, Any]:
    model = {"qwen3_8b": "qwen3-8b", "qwen3_32b": "qwen3-32b"}[
        str(scenario["model_id"])
    ]
    bandwidth = protocol["cluster"]["bandwidth"]
    algorithm3 = protocol["algorithm3"]
    return {
        "model": model,
        "dp_size": int(scenario["system_dp_size"]),
        "gpus_per_node": int(protocol["cluster"]["gpus_per_node"]),
        "concurrency": int(scenario["active_workflow_concurrency"]),
        "request_count": int(request_count),
        "workflow_count": int(protocol["source_trace"]["workflow_count"]),
        "prompt_scale": float(scenario["prompt_scale"]),
        "output_cap_tokens": int(protocol["workload"]["output_cap_tokens"]),
        "reuse_probability": float(reuse_fraction),
        "failure_rate_per_1000_worker_hours": float(
            scenario["failure_rate_per_1000_worker_hours"]
        ),
        "algorithm1_prediction_error_multiplier": float(
            scenario["algorithm1_prediction_error_multiplier"]
        ),
        "repair_time_ms": float(protocol["failure"]["worker_replacement_delay_s"])
        * 1000.0,
        "control_window_ms": float(algorithm3["window_s"]) * 1000.0,
        # This interval tags steady telemetry only.  The simulator still
        # injects its worker Poisson process across the complete run horizon
        # and includes all 393 workflows in full-workflow aggregate accounting.
        "measurement_start_ms": float(measurement_start_ms),
        "measurement_end_ms": float(measurement_end_ms),
        "hbm_capacity_gib_per_target": float(
            scenario["hbm_quota_gib_per_target"]
        ),
        "dram_capacity_gib_per_target": float(
            scenario["dram_quota_gib_per_target"]
        ),
        "hbm_copy_bandwidth_gbps": float(
            bandwidth["source_to_peer_hbm_same_node_gb_s"]
        ),
        "dram_copy_bandwidth_gbps": float(scenario["network_effective_gb_s"]),
        "remote_hbm_load_bandwidth_gbps": float(
            scenario["network_effective_gb_s"]
        ),
        "dram_load_bandwidth_gbps": float(scenario["network_effective_gb_s"]),
        "network_effective_bandwidth_gbps": float(
            scenario["network_effective_gb_s"]
        ),
        "hbm_copy_setup_ms": float(bandwidth["hbm_setup_ms"]),
        "dram_copy_setup_ms": float(bandwidth["dram_write_setup_ms"]),
        "hbm_load_setup_ms": float(bandwidth["dram_read_setup_ms"]),
        "dram_load_setup_ms": float(bandwidth["dram_read_setup_ms"]),
        "initial_hbm_alpha": float(algorithm3["initial_alpha"]),
        "minimum_hbm_alpha": float(algorithm3["minimum_alpha"]),
        "maximum_hbm_alpha": float(algorithm3["maximum_alpha"]),
        "aimd_decrease_factor": float(algorithm3["multiplicative_decrease"]),
        "aimd_increase_step": float(algorithm3["additive_increase"]),
        "aimd_stable_windows": int(algorithm3["stable_windows_before_increase"]),
        "pressure_queue_depth": int(
            algorithm3["queue_depth_pressure_threshold"]
        ),
        "hbm_quota_pressure_fraction": float(
            algorithm3["quota_occupancy_pressure_threshold"]
        ),
        "prefix_chunk_tokens": int(protocol["cluster"]["prefix_chunk_tokens"]),
        "decode_chunk_tokens": int(protocol["cluster"]["decode_chunk_tokens"]),
        "target_failure_probability": float(protocol["failure"]["target_first_fraction"]),
        "minimum_faults_per_cell": 0,
        "active_source_probe_count": int(
            protocol["fault_estimands"][
                "estimand_b_active_source_fault_conditioned_recovery"
            ]["active_source_probe_count"]
        ),
        "event_log_limit": 0,
    }


def _run_path(output_dir: Path, run: dict[str, Any]) -> Path:
    return (
        output_dir
        / "raw"
        / str(run["scenario_id"])
        / f"{run['policy']}__s{run['seed']}.json.gz"
    )


def _default_pilot_scenarios(
    scenarios: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    central_matches = [
        row
        for row in scenarios.values()
        if row["model_id"] == "qwen3_8b"
        and row["active_workflow_concurrency"] == 128
        and float(row["failure_rate_per_1000_worker_hours"]) == 5.952
        and row["total_gpu_count"] == 64
        and float(row["prompt_scale"]) == 0.016
        and float(row["network_effective_gb_s"]) == 10.0
        and float(row["hbm_quota_gib_per_target"]) == 4.0
        and float(row["dram_quota_gib_per_target"]) == 64.0
        and float(row["algorithm1_prediction_error_multiplier"]) == 1.0
        and row["tp_size"] == 1
        and row["pp_size"] == 1
        and row["system_dp_size"] == 64
    ]
    if len(central_matches) != 1:
        raise ValueError(
            f"default pilot central scenario is not unique: {central_matches}"
        )
    central = central_matches[0]
    comparison_fields = (
        "model_id",
        "prompt_scale",
        "active_workflow_concurrency",
        "total_gpu_count",
        "network_effective_gb_s",
        "hbm_quota_gib_per_target",
        "dram_quota_gib_per_target",
        "algorithm1_prediction_error_multiplier",
        "tp_size",
        "pp_size",
        "system_dp_size",
    )
    high_failure = max(
        (
            scenario
            for scenario in scenarios.values()
            if all(
                scenario[field] == central[field]
                for field in comparison_fields
            )
        ),
        key=lambda row: float(row["failure_rate_per_1000_worker_hours"]),
    )
    rate_zero_matches = [
        scenario
        for scenario in scenarios.values()
        if all(
            scenario[field] == central[field]
            for field in comparison_fields
        )
        and float(scenario["failure_rate_per_1000_worker_hours"]) == 0.0
    ]
    if len(rate_zero_matches) != 1:
        raise ValueError(
            "default pilot rate-zero reference is not unique: "
            f"{rate_zero_matches}"
        )
    return rate_zero_matches[0], central, high_failure


def _selection_provenance(
    manifest: dict[str, Any],
    runs: Sequence[dict[str, Any]],
    *,
    pilot: bool,
    scenario_ids: set[str] | None,
    policies: set[str] | None,
    max_runs: int | None,
) -> dict[str, Any]:
    scenarios = {row["scenario_id"]: row for row in manifest["scenarios"]}
    custom = bool(scenario_ids or policies or max_runs is not None)
    if pilot and not custom:
        mode = "DEFAULT_NINE_RUN_MECHANISM_PILOT"
    elif not pilot and not custom:
        mode = "FULL_FROZEN_MATRIX"
    else:
        mode = "CUSTOM_PARTIAL_DIAGNOSTIC"
    observed = sorted(
        (str(run["scenario_id"]), str(run["policy"]), int(run["seed"]))
        for run in runs
    )
    pilot_contract: dict[str, Any] = {"applies": bool(pilot)}
    if pilot and not custom:
        rate_zero, central, high_failure = _default_pilot_scenarios(scenarios)
        expected_policies = {
            "aegiskv_alg111",
            "aegiskv_alg011_no_alg1",
            "recompute_only",
        }
        expected_seed = min(int(run["seed"]) for run in manifest["runs"])
        expected = sorted(
            (scenario["scenario_id"], policy, expected_seed)
            for scenario in (rate_zero, central, high_failure)
            for policy in expected_policies
        )
        checks = {
            "exact_nine_tuples": observed == expected and len(observed) == 9,
            "rate_zero_reference_exact": float(
                rate_zero["failure_rate_per_1000_worker_hours"]
            )
            == 0.0,
            "central_rate_exact": float(
                central["failure_rate_per_1000_worker_hours"]
            )
            == 5.952,
            "highest_rate_exact": float(
                high_failure["failure_rate_per_1000_worker_hours"]
            )
            == 41.667,
            "same_single_seed": len({row[2] for row in observed}) == 1,
        }
        pilot_contract.update(
            {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
                "rate_zero_reference_scenario_id": rate_zero["scenario_id"],
                "central_scenario_id": central["scenario_id"],
                "high_failure_scenario_id": high_failure["scenario_id"],
                "expected_tuples": expected,
                "observed_tuples": observed,
            }
        )
        if pilot_contract["status"] != "PASS":
            raise ValueError(f"default pilot contract failed: {pilot_contract}")
    elif pilot:
        pilot_contract.update(
            {
                "status": "CUSTOM_PARTIAL_DIAGNOSTIC",
                "default_nine_run_contract_evaluated": False,
                "observed_tuples": observed,
            }
        )
    else:
        pilot_contract["status"] = "NOT_APPLICABLE"
    payload = {
        "mode": mode,
        "requested_filters": {
            "scenario_ids": sorted(scenario_ids or ()),
            "policies": sorted(policies or ()),
            "max_runs": max_runs,
        },
        "selected_run_count": len(runs),
        "selected_tuples": observed,
        "pilot_contract": pilot_contract,
        "allowed_analysis_status": (
            "PILOT_DIAGNOSTIC"
            if mode == "DEFAULT_NINE_RUN_MECHANISM_PILOT"
            else (
                "COMPLETE"
                if mode == "FULL_FROZEN_MATRIX"
                else "CUSTOM_PARTIAL_DIAGNOSTIC"
            )
        ),
    }
    payload["canonical_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return payload


def _selection_output_leaf(selection: dict[str, Any]) -> str:
    mode = str(selection["mode"])
    if mode == "DEFAULT_NINE_RUN_MECHANISM_PILOT":
        return "pilot"
    if mode == "FULL_FROZEN_MATRIX":
        return "full"
    if mode == "CUSTOM_PARTIAL_DIAGNOSTIC":
        return f"custom-{selection['canonical_sha256'][:12]}"
    raise ValueError(f"unsupported selection mode: {mode}")


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"refusing to overwrite immutable JSON: {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def _persist_campaign_outcome(
    output_dir: Path,
    index: dict[str, Any],
    failures: Sequence[dict[str, Any]],
    *,
    attempt_token: str | None = None,
) -> Path:
    """Commit only successful campaigns to the canonical index path."""

    payload = dict(index)
    if failures:
        payload["status"] = "FAILED_ATTEMPT"
        payload["canonical_campaign_index_written"] = False
        token = attempt_token or f"{time.time_ns()}-{os.getpid()}"
        path = output_dir / f"attempt-{token}-failed.json"
    else:
        payload["status"] = "COMPLETE"
        payload["canonical_campaign_index_written"] = True
        path = output_dir / "campaign-index.json"
    _write_immutable_json(path, payload)
    return path


def _validate_t0_aimd_result(
    result: dict[str, Any], protocol: dict[str, Any]
) -> bool:
    """Fail closed on t=0 mutation while respecting baseline alpha modes."""

    rows = list(result.get("window_rows") or [])
    initial_rows = [row for row in rows if bool(row.get("initial_sample"))]
    algorithm3_enabled = bool(
        initial_rows and initial_rows[0].get("algorithm3_enabled")
    )
    applied_rows = [row for row in rows if bool(row.get("aimd_update_applied"))]
    summary = result.get("summary") or {}
    gates = result.get("gates") or {}
    valid = bool(
        len(initial_rows) == 1
        and math.isclose(
            float(initial_rows[0]["time_ms"]), 0.0, abs_tol=0.0
        )
        and initial_rows[0].get("aimd_update_applied") is False
        and (
            not algorithm3_enabled
            or math.isclose(
                float(initial_rows[0]["controller_alpha_mean"]),
                float(protocol["algorithm3"]["initial_alpha"]),
                abs_tol=1e-12,
            )
        )
        and (
            not algorithm3_enabled
            or math.isclose(
                float(initial_rows[0]["controller_stable_windows_mean"]),
                0.0,
                abs_tol=0.0,
            )
        )
        and all(
            bool(row.get("aimd_update_applied"))
            == bool(algorithm3_enabled and float(row.get("time_ms", 0.0)) > 0.0)
            for row in rows
        )
        and int(summary.get("initial_snapshot_sample_count", -1)) == 1
        and int(summary.get("aimd_periodic_updates_applied", -1))
        == len(applied_rows)
        and summary.get("first_aimd_periodic_update_ms")
        == (float(applied_rows[0]["time_ms"]) if applied_rows else None)
        and gates.get("t0_control_window_is_snapshot_only") is True
        and gates.get("aimd_updates_only_at_positive_periodic_windows") is True
        and gates.get("initial_sample_does_not_advance_stable_counter") is True
    )
    if not valid:
        raise ValueError("t=0 AIMD snapshot-only contract failed")
    return algorithm3_enabled


def _validate_failure_state_snapshots(result: dict[str, Any]) -> None:
    """Keep base-horizon state distinct from post-repair event-drain state."""

    horizon = result.get("failure_state_at_horizon") or {}
    final = result.get("final_state", {}).get("failure_history") or {}
    summary = result.get("summary") or {}
    history_contract = result.get("formal_metric_contract", {}).get(
        "failure_history_contract", {}
    )
    stable_fields = (
        "physical_gpu_shards_ever_failed_count",
        "logical_replica_targets_ever_failed_count",
        "old_hbm_copy_invalidation_event_count_cumulative",
        "old_hbm_copy_invalidated_requests_cumulative",
        "old_hbm_copy_invalidated_bytes_cumulative",
    )
    repair_after_horizon = sum(
        bool(row.get("after_base_run_horizon"))
        for row in result.get("repair_rows", ())
    )
    gates = result.get("gates") or {}
    valid = bool(
        horizon.get("snapshot_scope")
        == "base_run_horizon_after_all_events_at_or_before_horizon"
        and math.isclose(
            float(horizon.get("snapshot_time_ms", math.nan)),
            float(summary.get("base_run_horizon_ms", math.nan)),
            abs_tol=1e-9,
        )
        and final.get("snapshot_scope")
        == "post_all_scheduled_events_including_repairs_after_horizon"
        and math.isclose(
            float(final.get("snapshot_time_ms", math.nan)),
            float(summary.get("post_all_events_last_event_time_ms", math.nan)),
            abs_tol=1e-9,
        )
        and int(horizon.get("current_physical_gpu_shard_outage_count", -1))
        == int(summary.get("current_physical_gpu_shard_outage_count_at_horizon", -2))
        and int(horizon.get("current_logical_replica_target_outage_count", -1))
        == int(summary.get("current_logical_replica_target_outage_count_at_horizon", -2))
        and int(final.get("current_physical_gpu_shard_outage_count", -1))
        == int(summary.get("current_physical_gpu_shard_outage_count_post_all_events", -2))
        and int(final.get("current_logical_replica_target_outage_count", -1))
        == int(summary.get("current_logical_replica_target_outage_count_post_all_events", -2))
        and all(horizon.get(field) == final.get(field) for field in stable_fields)
        and int(summary.get("repair_events_after_horizon", -1))
        == repair_after_horizon
        and int(summary.get("post_horizon_event_count", -1))
        >= repair_after_horizon
        and history_contract.get("horizon_state_source")
        == "failure_state_at_horizon"
        and history_contract.get("post_all_events_state_source")
        == "final_state.failure_history"
        and history_contract.get("events_after_horizon_are_cleanup_not_horizon_state")
        is True
        and gates.get("horizon_failure_state_frozen_before_post_horizon_events")
        is True
        and gates.get("summary_at_horizon_fields_derive_from_horizon_failure_state")
        is True
        and gates.get("final_state_is_post_all_events_not_horizon_state") is True
    )
    if not valid:
        raise ValueError("failure horizon/post-all-events snapshot contract failed")


def _execute_one(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    protocol = load_protocol(Path(payload["protocol_path"]))
    scenario = dict(payload["scenario"])
    run = dict(payload["run"])
    lineage = dict(payload["lineage"])
    _validate_lineage(lineage)
    output_path = Path(payload["output_path"])
    if output_path.exists():
        with gzip.open(output_path, "rt", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("lineage") != lineage:
            raise ValueError(f"immutable lineage payload mismatch: {output_path}")
        if existing.get("scenario") != scenario:
            raise ValueError(f"immutable scenario payload mismatch: {output_path}")
        expected = {
            "lineage_sha256": lineage["lineage_sha256"],
            "design_semantics": protocol["design_semantics"],
            "run_id": run["run_id"],
            "scenario_id": scenario["scenario_id"],
            "policy": run["policy"],
            "seed": int(run["seed"]),
        }
        observed = {
            "lineage_sha256": existing.get("lineage", {}).get(
                "lineage_sha256"
            ),
            "design_semantics": existing.get("design_semantics"),
            "run_id": existing.get("run_id"),
            "scenario_id": existing.get("scenario", {}).get("scenario_id"),
            "policy": existing.get("policy"),
            "seed": existing.get("seed"),
        }
        if observed != expected:
            raise ValueError(
                f"immutable output mismatch for {output_path}: {observed} != {expected}"
            )
        return {
            "run_id": run["run_id"],
            "status": "SKIP_VALIDATED",
            "output_path": str(output_path),
            "sha256": _sha256(output_path),
        }
    workload = CompactWeka().project(scenario, protocol, seed=int(run["seed"]))
    offered_error = float(
        workload.metadata[
            "achieved_offered_trace_steady_concurrency_relative_error"
        ]
    )
    offered_limit = float(
        protocol["workload"][
            "achieved_steady_concurrency_relative_error_gate"
        ]
    )
    if offered_error > offered_limit + 1e-12:
        raise ValueError(
            f"offered trace concurrency gate failed: {offered_error} > {offered_limit}"
        )
    if (
        int(protocol["workload"]["wct_measurement_workflows"]) != 393
        or workload.metadata.get("all_393_workflows_in_wct") is not True
        or workload.metadata.get("stabilization_prefix_excluded_from_wct")
        is not False
    ):
        raise ValueError(
            "all 393 workflows must remain in full-workflow aggregate accounting"
        )
    config = _scenario_config(
        scenario,
        protocol,
        request_count=len(workload.rows),
        reuse_fraction=float(workload.metadata["input_reuse_fraction"]),
        measurement_start_ms=float(
            workload.metadata["steady_measurement_start_ms"]
        ),
        measurement_end_ms=float(
            workload.metadata["steady_measurement_end_ms"]
        ),
    )
    started = time.time()
    result = run_cell(
        config,
        int(run["seed"]),
        policy=str(run["policy"]),
        workload=workload.rows,
    )
    algorithm3_enabled = _validate_t0_aimd_result(result, protocol)
    _validate_failure_state_snapshots(result)
    if int(scenario["active_workflow_concurrency"]) == max(
        int(value)
        for value in protocol["workload"]["active_workflow_concurrency_sweep"]
    ):
        last_launch_ms = float(workload.metadata["steady_measurement_end_ms"])
        minimum_last_launch_ms = float(
            protocol["workload"][
                "highest_concurrency_minimum_last_launch_ms"
            ]
        )
        if last_launch_ms < minimum_last_launch_ms:
            raise ValueError(
                "highest-concurrency update-horizon gate failed: "
                f"{last_launch_ms} < {minimum_last_launch_ms}"
            )
        steady_samples = sum(
            bool(row.get("in_measurement_window"))
            for row in result["window_rows"]
        )
        minimum_samples = int(
            protocol["workload"][
                "highest_concurrency_minimum_steady_window_samples"
            ]
        )
        if steady_samples < minimum_samples:
            raise ValueError(
                "highest-concurrency steady telemetry gate failed: "
                f"{steady_samples} < {minimum_samples}"
            )
        periodic_opportunities = sum(
            float(row.get("time_ms", 0.0)) > 0.0
            and float(row.get("time_ms", 0.0)) <= last_launch_ms + 1e-9
            for row in result["window_rows"]
        )
        periodic_updates = sum(
            bool(row.get("aimd_update_applied"))
            and float(row.get("time_ms", 0.0)) <= last_launch_ms + 1e-9
            for row in result["window_rows"]
        )
        minimum_updates = int(
            protocol["workload"][
                "highest_concurrency_minimum_periodic_aimd_updates"
            ]
        )
        if periodic_opportunities < minimum_updates or (
            algorithm3_enabled and periodic_updates < minimum_updates
        ) or (not algorithm3_enabled and periodic_updates != 0):
            raise ValueError(
                "highest-concurrency AIMD update-opportunity gate failed: "
                f"opportunities={periodic_opportunities}, "
                f"applied={periodic_updates}, enabled={algorithm3_enabled}, "
                f"minimum={minimum_updates}"
            )
    elapsed = time.time() - started
    retain_request_sample = int(
        protocol["output"]["retain_failure_sample_per_run"]
    )
    artifact = {
        "schema_version": 1,
        "simulator_cell_schema_version": result["schema_version"],
        "campaign_id": protocol["campaign_id"],
        "evidence_class": result["evidence_class"],
        "design_semantics": result["design_semantics"],
        "formal_metric_contract": result["formal_metric_contract"],
        "run_id": run["run_id"],
        "lineage": lineage,
        "scenario": scenario,
        "seed": int(run["seed"]),
        "policy": str(run["policy"]),
        "wall_time_s": elapsed,
        "process_maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "workload": workload.metadata,
        "resolved_config": result["config"],
        "algorithms": result["algorithms"],
        "ablation": result["ablation"],
        "summary": result["summary"],
        "window_diagnostics": _window_diagnostics(
            result["window_rows"], dp_size=int(config["dp_size"])
        ),
        "realized_concurrency": _realized_concurrency(
            result["workflow_rows"], workload.metadata
        ),
        "window_rows": result["window_rows"],
        "window_row_count": len(result["window_rows"]),
        "failure_rows": result["failure_rows"],
        "failure_row_count": len(result["failure_rows"]),
        "repair_rows": result["repair_rows"],
        "repair_row_count": len(result["repair_rows"]),
        "active_source_probe_metadata": result[
            "active_source_probe_metadata"
        ],
        "active_source_probe_rows": result["active_source_probe_rows"],
        "active_source_probe_row_count": len(
            result["active_source_probe_rows"]
        ),
        "active_source_fault_conditioned_recovery": result[
            "active_source_fault_conditioned_recovery"
        ],
        "request_rows_sample": _request_sample(
            result["request_rows"],
            retain_request_sample,
            seed=int(run["seed"]),
        ),
        "workflow_rows": result["workflow_rows"],
        "failure_state_at_horizon": result["failure_state_at_horizon"],
        "final_state": result["final_state"],
        "gates": result["gates"],
        "approximation_boundary": result["approximation_boundary"],
        "raw_retention": {
            "full_request_rows_retained": False,
            "full_workflow_rows_retained": True,
            "full_window_rows_retained": True,
            "full_failure_rows_retained": True,
            "full_repair_rows_retained": True,
            "full_active_source_probe_rows_retained": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(artifact, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    temporary.replace(output_path)
    return {
        "run_id": run["run_id"],
        "status": "PASS",
        "output_path": str(output_path),
        "sha256": _sha256(output_path),
        "wall_time_s": elapsed,
        "process_maxrss_mib": artifact["process_maxrss_mib"],
        "failure_events": result["summary"]["failure_events_injected"],
        "effective_failures": result["summary"]["failure_events_effective"],
    }


def _select_runs(
    manifest: dict[str, Any],
    *,
    pilot: bool,
    max_runs: int | None,
    scenario_ids: set[str] | None = None,
    policies: set[str] | None = None,
) -> list[dict[str, Any]]:
    runs = list(manifest["runs"])
    scenarios = {row["scenario_id"]: row for row in manifest["scenarios"]}
    runs.sort(
        key=lambda row: (
            -str(scenarios[row["scenario_id"]]["campaign_axis"]).count("+"),
            str(row["scenario_id"]),
            str(row["policy"]),
            int(row["seed"]),
        )
    )
    if scenario_ids:
        unknown = scenario_ids - set(scenarios)
        if unknown:
            raise ValueError(f"unknown scenario IDs: {sorted(unknown)}")
        runs = [row for row in runs if row["scenario_id"] in scenario_ids]
    if policies:
        runs = [row for row in runs if row["policy"] in policies]
    if pilot and not scenario_ids and not policies:
        rate_zero, central, high_failure = _default_pilot_scenarios(scenarios)
        pilot_scenarios = {
            rate_zero["scenario_id"],
            central["scenario_id"],
            high_failure["scenario_id"],
        }
        pilot_policies = {
            "aegiskv_alg111",
            "aegiskv_alg011_no_alg1",
            "recompute_only",
        }
        candidates = [
            row
            for row in runs
            if row["scenario_id"] in pilot_scenarios
            and row["policy"] in pilot_policies
        ]
        first_seed = min(row["seed"] for row in candidates)
        runs = [row for row in candidates if row["seed"] == first_seed]
    elif pilot:
        first_seed = min(row["seed"] for row in runs)
        runs = [row for row in runs if row["seed"] == first_seed]
    if max_runs is not None:
        runs = runs[: int(max_runs)]
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--scenario-id", action="append", default=[])
    parser.add_argument("--policy", action="append", default=[])
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if protocol["output"].get("retain_active_source_probe_rows_per_run") != "all_64":
        raise ValueError("protocol must retain all 64 active-source probe rows")
    protocol_bins = {
        name: protocol["fault_estimands"]["reuse_bin_contract"][name]
        for name in REUSE_BIN_CONTRACT
    }
    if protocol_bins != REUSE_BIN_CONTRACT:
        raise ValueError("protocol reuse bins differ from core contract")
    generated = matrix_manifest(protocol)
    validate(generated, protocol)
    frozen = json.loads(args.matrix.read_text(encoding="utf-8"))
    if frozen["canonical_sha256"] != generated["canonical_sha256"]:
        raise ValueError("matrix.json is stale relative to protocol.json")
    scenarios = {row["scenario_id"]: row for row in frozen["scenarios"]}
    lineage = _source_lineage(args.protocol, args.matrix)
    _validate_lineage(lineage)
    scenario_filters = set(args.scenario_id) or None
    policy_filters = set(args.policy) or None
    runs = _select_runs(
        frozen,
        pilot=args.pilot,
        max_runs=args.max_runs,
        scenario_ids=scenario_filters,
        policies=policy_filters,
    )
    selection = _selection_provenance(
        frozen,
        runs,
        pilot=args.pilot,
        scenario_ids=scenario_filters,
        policies=policy_filters,
        max_runs=args.max_runs,
    )
    if not runs:
        raise ValueError("run selection is empty")
    output_dir = (
        args.output_dir
        / lineage["lineage_id"]
        / _selection_output_leaf(selection)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "campaign-index.json"
    if index_path.exists():
        raise FileExistsError(
            f"immutable campaign index already exists; use a fresh selection or lineage: {index_path}"
        )
    payloads = [
        {
            "protocol_path": str(args.protocol.resolve()),
            "scenario": scenarios[run["scenario_id"]],
            "run": run,
            "lineage": lineage,
            "output_path": str(_run_path(output_dir, run)),
        }
        for run in runs
    ]
    started = time.time()
    load_start = os.getloadavg()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "event": "campaign_start",
                "pilot": args.pilot,
                "selection_mode": selection["mode"],
                "runs": len(payloads),
                "workers": args.workers,
                "system_load_1m_5m_15m": load_start,
                "output_dir": str(output_dir),
            }
        ),
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(int(args.workers), 1)
    ) as executor:
        future_by_run = {
            executor.submit(_execute_one, payload): payload["run"]
            for payload in payloads
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_by_run), start=1
        ):
            run = future_by_run[future]
            try:
                record = future.result()
                records.append(record)
            except Exception as exc:
                record = {
                    "run_id": run["run_id"],
                    "status": "FAIL",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                records.append(record)
                failures.append(record)
            if completed == 1 or completed % 10 == 0 or completed == len(payloads):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "completed": completed,
                            "total": len(payloads),
                            "failures": len(failures),
                            "elapsed_s": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    index = {
        "schema_version": 1,
        "campaign_id": protocol["campaign_id"],
        "pilot": args.pilot,
        "protocol": {"path": str(args.protocol.resolve()), "sha256": _sha256(args.protocol)},
        "matrix": {
            "path": str(args.matrix.resolve()),
            "sha256": _sha256(args.matrix),
            "canonical_sha256": frozen["canonical_sha256"],
        },
        "requested_run_count": len(payloads),
        "pass_count": sum(
            row["status"] in {"PASS", "SKIP_VALIDATED"} for row in records
        ),
        "failure_count": len(failures),
        "elapsed_s": time.time() - started,
        "host_resource_observation": {
            "cpu_count": os.cpu_count(),
            "load_1m_5m_15m_at_start": load_start,
            "load_1m_5m_15m_at_end": os.getloadavg(),
            "wall_time_is_not_a_simulated_performance_metric": True,
        },
        "selection": selection,
        "lineage": lineage,
        "records": sorted(records, key=lambda row: row["run_id"]),
    }
    persisted_index_path = _persist_campaign_outcome(
        output_dir, index, failures
    )
    print(
        json.dumps(
            {
                "event": "campaign_complete",
                "status": "PASS" if not failures else "FAIL",
                "pass_count": index["pass_count"],
                "failure_count": len(failures),
                "elapsed_s": index["elapsed_s"],
                "index": str(persisted_index_path),
                "canonical_campaign_index_written": not failures,
            },
            indent=2,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
