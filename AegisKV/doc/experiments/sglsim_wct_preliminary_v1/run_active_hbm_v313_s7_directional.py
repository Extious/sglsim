#!/usr/bin/env python3
"""Run the preregistered v3.1.3 S7 directional/repair sensitivity suite."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import active_hbm_v313 as v313
import run_preliminary as core
import test_active_hbm_v313 as v313_tests


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s7_directional_protocol.json"
RESULT_NAME = "active_hbm_v313_s7_directional_repair_pilot"
GATE_NAME = "active_hbm_v313_s7_directional_repair_gate"
T95_DF2 = 4.302652729911275
TESTS = (
    ("test_active_hbm_v3_capacity.py", "RESULT 5/5 PASS"),
    ("test_active_hbm_v313_protocol.py", "RESULT 6/6 PASS"),
    ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    ("test_active_hbm_v313_s7_directional.py", "RESULT 8/8 PASS"),
    ("test_active_hbm_v313_s7_protocol.py", "RESULT 10/10 PASS"),
)
METRICS = (
    "healthy_wct_mean_ms", "healthy_wct_p99_ms",
    "fault_wct_mean_ms", "fault_wct_p99_ms", "fault_added_wct_mean_ms",
    "critical_queue_mean_ms", "critical_copy_burden_mean_ms",
    "copy_transfer_service_mean_ms", "copy_queue_wait_mean_ms",
    "recovery_service_mean_ms", "recovery_hit_fraction",
    "recovery_token_fraction", "copy_amplification",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) < 2:
        return 0.0
    if len(rows) != 3:
        raise ValueError("S7 full aggregates require exactly three seeds")
    return T95_DF2 * statistics.stdev(rows) / math.sqrt(3)


def slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace(".", "p")


def configured_protocol(
    spec: dict[str, Any], *, axis: str, direction: str | None,
    bandwidth_gb_s: float | None, repair_ms: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workload_spec = spec["workload"]
    hardware = spec["hardware"]
    protocol, capacity = v313_tests.inputs(
        context=int(workload_spec["context_tokens"]),
        active=int(workload_spec["offered_active_requests_per_gpu"]),
    )
    protocol["models"]["qwen3_8b"]["center_system_dp"] = int(
        hardware["system_dp"]
    )
    protocol["workload"]["center_context_tokens"] = int(
        workload_spec["context_tokens"]
    )
    protocol["workload"]["center_active_requests_per_gpu"] = int(
        workload_spec["offered_active_requests_per_gpu"]
    )
    protocol["hardware"]["gpus_per_node"] = int(hardware["gpus_per_node"])
    protocol["hardware"]["node_count"] = int(hardware["node_count"])
    protocol["hardware"]["copy_bandwidth_gb_s"] = dict(
        hardware["baseline_copy_bandwidth_gb_s"]
    )
    protocol["hardware"]["load_bandwidth_gb_s"] = dict(
        hardware["baseline_recovery_bandwidth_gb_s"]
    )
    protocol.setdefault("runtime_overrides", {})["topology"] = {
        "peer_hbm": hardware["topology"]["peer_hbm"],
        "dram": hardware["topology"]["dram"],
    }
    if axis == "bandwidth":
        treatment = spec["sweep"]["bandwidth_directions"][str(direction)]
        protocol["hardware"][treatment["protocol_table"]][
            treatment["protocol_key"]
        ] = float(bandwidth_gb_s)
        replacement = 60000.0
    elif axis == "repair":
        replacement = float(repair_ms)
    else:
        raise ValueError(f"unknown S7 axis {axis!r}")
    protocol["failure_model"]["policy_failover"][
        "replacement_delay_ms"
    ] = replacement
    protocol["failure_model"]["policy_failover"][
        "fence_activation_delay_ms"
    ] = float(spec["sweep"]["fixed_modeled_activation_delay_ms"])
    return protocol, capacity


def build_universe(
    spec: dict[str, Any], protocol: dict[str, Any], *, seed: int,
    failure_type: str,
) -> tuple[list[core.Task], dict[str, Any], set[str], list[core.FaultProbe]]:
    workload_spec = spec["workload"]
    nominal, workload = core.build_workload(
        protocol, seed=int(seed), workflows=None,
        active_per_gpu=int(workload_spec["offered_active_requests_per_gpu"]),
        arrival_cutoff_ms=float(workload_spec["fixed_arrival_cutoff_ms"]),
    )
    cohort = {
        row.workflow_id for row in nominal
        if float(workload_spec["measurement_start_ms"])
        <= row.workflow_arrival_ms
        <= float(workload_spec["measurement_end_ms"])
    }
    cohort_tasks = [row for row in nominal if row.workflow_id in cohort]
    base_probes = core.make_fault_probes(
        cohort_tasks, seed=int(seed) ^ 0x57D1AEC7,
        count=int(workload_spec["fault_probes_per_cell"]),
        gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
        right_ms=float(workload_spec["measurement_end_ms"]),
        failure_types=("source_worker_gpu",),
    )
    probes = [replace(row, failure_type=failure_type) for row in base_probes]
    return nominal, workload, cohort, probes


def workflow_hash(tasks: list[core.Task]) -> str:
    return digest_json([
        (
            row.request_id, row.workflow_id, row.workflow_arrival_ms,
            row.arrival_ms, row.source_rank, row.service_ms,
            row.prompt_tokens, row.output_tokens, row.predicted_slack_ms,
        )
        for row in tasks
    ])


def probe_hash(probes: list[core.FaultProbe], *, include_domain: bool) -> str:
    return digest_json([
        (
            row.probe_id, row.time_ms, row.source_rank,
            row.failure_type if include_domain else None,
        )
        for row in probes
    ])


def cohort_wct(
    tasks: list[core.Task], cohort: set[str],
    completion: dict[str, float] | None = None,
) -> tuple[list[str], list[float]]:
    return core.workflow_wct(
        [row for row in tasks if row.workflow_id in cohort], completion
    )


def mean_p99(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), core.quantile(values, 0.99)


def run_cell(
    stage: Path, spec: dict[str, Any], job: dict[str, Any], policy: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    axis = str(job["axis"])
    direction = job.get("direction")
    bandwidth = job.get("bandwidth_gb_s")
    repair_ms = job.get("repair_ms")
    failure_type = str(job["failure_type"])
    protocol, capacity = configured_protocol(
        spec, axis=axis, direction=direction,
        bandwidth_gb_s=bandwidth, repair_ms=repair_ms,
    )
    nominal, workload, cohort, probes = build_universe(
        spec, protocol, seed=int(seed), failure_type=failure_type
    )
    replay = v313.CausalPhysicalTargetReplay(
        deepcopy(protocol), policy, capacity
    )
    tasks, delay_maps, mechanism, capacity_trace = replay.run_coupled(
        nominal, probes,
        control_horizon_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
    )
    base_ids, base_wct = cohort_wct(tasks, cohort)
    empty_ids, empty_wct = cohort_wct(
        tasks, cohort, core.replay_one_fault(tasks, {})
    )
    if base_ids != empty_ids:
        raise AssertionError("empty replay changed cohort workflow IDs")
    empty_error = max(abs(left - right) for left, right in zip(base_wct, empty_wct))

    fault_matrix: list[list[float]] = []
    probe_changed: list[int] = []
    monotonic_error = 0.0
    failed_rank_counts: list[int] = []
    for probe, delays in zip(probes, delay_maps):
        ids, values = cohort_wct(
            tasks, cohort, core.replay_one_fault(tasks, delays)
        )
        if ids != base_ids:
            raise AssertionError("fault replay changed cohort workflow IDs")
        monotonic_error = max(
            monotonic_error,
            max((base - value for base, value in zip(base_wct, values)), default=0.0),
        )
        fault_matrix.append(values)
        probe_changed.append(len(delays))
        failed_rank_counts.append(len(v313.v312.v31.failure_domain_ranks(
            probe, gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
            dp=int(spec["hardware"]["system_dp"]),
        )))

    healthy_mean, healthy_p99 = mean_p99(base_wct)
    fault_flat = [value for values in fault_matrix for value in values]
    fault_mean, fault_p99 = mean_p99(fault_flat)
    cohort_tasks = [row for row in tasks if row.workflow_id in cohort]
    decomp = v313.workflow_decomposition(cohort_tasks)
    v3m = mechanism["active_hbm_v3"]
    v313m = mechanism["active_hbm_v313"]
    v312m = mechanism["active_hbm_v312"]

    type_paths = mechanism["recovery_type_path_counts"]
    recovery_calls = sum(int(value) for value in type_paths.values())
    recovery_hits = sum(
        int(value) for key, value in type_paths.items()
        if not key.split(":", 1)[1].startswith("RECOMPUTE")
    )
    restored = int(mechanism["recovery_restored_tokens"])
    recomputed = int(mechanism["recovery_recomputed_tokens"])
    scheduled = v313m["scheduled_copy_bytes"]
    logical = sum(int(row.prompt_tokens) * int(replay.kv) for row in tasks)
    copy_amplification = (
        (int(scheduled["HBM"]) + int(scheduled["DRAM"])) / logical
        if logical else 0.0
    )
    budget = int(spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"])
    max_shared = max(
        (int(row["shared_used_bytes"]) for row in capacity_trace), default=0
    )
    min_headroom = min(
        (budget - int(row["shared_used_bytes"]) for row in capacity_trace),
        default=budget,
    )
    copy_bw = core.copy_bandwidth(protocol, 0)
    load_bw = core.load_bandwidth(protocol, 0)
    allowed_copy_paths = {"copy_peer_hbm_cross_node", "copy_dram_remote"}
    wrong_copy_paths = sorted(
        key for key, value in mechanism["path_event_counts"].items()
        if key.startswith("copy_") and int(value) > 0
        and key not in allowed_copy_paths
    )

    base_copy = spec["hardware"]["baseline_copy_bandwidth_gb_s"]
    base_load = spec["hardware"]["baseline_recovery_bandwidth_gb_s"]
    treatment_table = ""
    treatment_key = ""
    isolation_ok = True
    if axis == "bandwidth":
        treatment = spec["sweep"]["bandwidth_directions"][str(direction)]
        treatment_table = str(treatment["protocol_table"])
        treatment_key = str(treatment["protocol_key"])
        expected_copy = dict(base_copy)
        expected_load = dict(base_load)
        (expected_copy if treatment_table == "copy_bandwidth_gb_s" else expected_load)[
            treatment_key
        ] = float(bandwidth)
        isolation_ok = (
            protocol["hardware"]["copy_bandwidth_gb_s"] == expected_copy
            and protocol["hardware"]["load_bandwidth_gb_s"] == expected_load
        )
    else:
        isolation_ok = (
            protocol["hardware"]["copy_bandwidth_gb_s"] == base_copy
            and protocol["hardware"]["load_bandwidth_gb_s"] == base_load
        )

    raw_name = (
        f"{job['condition_id']}__{slug(policy)}__s{seed}.npz"
    )
    raw_path = stage / "raw" / raw_name
    np.savez_compressed(
        raw_path,
        workflow_ids=np.asarray(base_ids),
        healthy_wct_ms=np.asarray(base_wct, dtype=float),
        fault_wct_ms=np.asarray(fault_matrix, dtype=float),
        probe_id=np.asarray([row.probe_id for row in probes]),
        probe_time_ms=np.asarray([row.time_ms for row in probes], dtype=float),
        probe_source_rank=np.asarray([row.source_rank for row in probes], dtype=int),
        probe_failure_type=np.asarray([row.failure_type for row in probes]),
        probe_changed_request_count=np.asarray(probe_changed, dtype=int),
        capacity_time_ms=np.asarray(
            [row["time_ms"] for row in capacity_trace], dtype=float
        ),
        capacity_rank=np.asarray(
            [row["rank"] for row in capacity_trace], dtype=int
        ),
        capacity_serving_bytes=np.asarray(
            [row["active_serving_kv_bytes"] for row in capacity_trace],
            dtype=np.int64,
        ),
        capacity_peer_backup_bytes=np.asarray(
            [row["peer_hbm_backup_used_bytes"] for row in capacity_trace],
            dtype=np.int64,
        ),
        capacity_shared_used_bytes=np.asarray(
            [row["shared_used_bytes"] for row in capacity_trace], dtype=np.int64
        ),
    )

    cell: dict[str, Any] = {
        "condition_id": job["condition_id"],
        "axis": axis,
        "direction": direction or "",
        "failure_type": failure_type,
        "bandwidth_gb_s": "" if bandwidth is None else float(bandwidth),
        "repair_ms": "" if repair_ms is None else float(repair_ms),
        "treatment_table": treatment_table,
        "treatment_key": treatment_key,
        "policy": policy,
        "seed": int(seed),
        "system_dp": int(spec["hardware"]["system_dp"]),
        "topology": spec["hardware"]["registered_s7_topology"],
        "workflow_universe_sha256": workflow_hash(nominal),
        "measurement_cohort_sha256": digest_json(sorted(cohort)),
        "probe_base_tuple_sha256": probe_hash(probes, include_domain=False),
        "fault_probe_tuple_sha256": probe_hash(probes, include_domain=True),
        "scheduled_request_set_sha256": digest_json(
            sorted(row.request_id for row in tasks)
        ),
        "workflow_count": len(base_ids),
        "request_count": len(tasks),
        "actual_last_arrival_ms": float(workload["actual_last_arrival_ms"]),
        "arrival_horizon_ms": float(spec["workload"]["fixed_arrival_cutoff_ms"]),
        "measurement_start_ms": float(spec["workload"]["measurement_start_ms"]),
        "measurement_end_ms": float(spec["workload"]["measurement_end_ms"]),
        "drain_horizon_ms": max(row.finish_ms for row in tasks),
        "healthy_wct_mean_ms": healthy_mean,
        "healthy_wct_p99_ms": healthy_p99,
        "fault_wct_mean_ms": fault_mean,
        "fault_wct_p99_ms": fault_p99,
        "fault_added_wct_mean_ms": fault_mean - healthy_mean,
        "fault_probe_count": len(probes),
        "fault_observation_count": len(fault_flat),
        "critical_queue_mean_ms": float(decomp["critical_queue_mean_ms"]),
        "critical_copy_burden_mean_ms": float(
            decomp["critical_direct_copy_burden_mean_ms"]
        ),
        "copy_transfer_service_mean_ms": float(
            mechanism["copy_transfer_service_mean_ms"]
        ),
        "copy_queue_wait_mean_ms": float(mechanism["copy_queue_wait_mean_ms"]),
        "recovery_service_mean_ms": (
            statistics.fmean(replay.recovery_service_values_ms)
            if replay.recovery_service_values_ms else 0.0
        ),
        "failover_delay_mean_ms": (
            statistics.fmean(replay.failover_delay_values_ms)
            if replay.failover_delay_values_ms else 0.0
        ),
        "recovery_call_count": recovery_calls,
        "recovery_hit_count": recovery_hits,
        "recovery_hit_fraction": recovery_hits / recovery_calls if recovery_calls else 0.0,
        "recovery_restored_tokens": restored,
        "recovery_recomputed_tokens": recomputed,
        "recovery_token_fraction": restored / (restored + recomputed) if restored + recomputed else 0.0,
        "copy_amplification": copy_amplification,
        "hbm_scheduled_copy_bytes": int(scheduled["HBM"]),
        "dram_scheduled_copy_bytes": int(scheduled["DRAM"]),
        "hbm_copy_jobs": int(v313m["copy_job_counts"]["HBM"]),
        "dram_copy_jobs": int(v313m["copy_job_counts"]["DRAM"]),
        "primary_priority_reclaimed_bytes": int(v313m["primary_priority_reclaimed_bytes"]),
        "memory_blocked_unique_request_count": int(
            v3m["memory_blocked_unique_request_count"]
        ),
        "peer_copy_bandwidth_gb_s": float(copy_bw[0]),
        "dram_copy_bandwidth_gb_s": float(copy_bw[1]),
        "peer_recovery_bandwidth_gb_s": float(load_bw[0]),
        "dram_recovery_bandwidth_gb_s": float(load_bw[1]),
        "treatment_isolation_exact": isolation_ok,
        "capacity_max_shared_used_bytes": max_shared,
        "capacity_min_headroom_bytes": min_headroom,
        "capacity_violation_count": int(v313m["target_capacity_violation_count"]),
        "target_accounting_max_abs_error_bytes": int(
            v313m["target_accounting_max_abs_error_bytes"]
        ),
        "copy_byte_conservation_error": json.dumps(
            v313m["copy_byte_conservation_error"], sort_keys=True
        ),
        "future_backup_projection_dispatch_count": int(
            v313m["future_backup_projection_dispatch_count"]
        ),
        "seal_before_dispatch_count": int(v313m["seal_before_dispatch_count"]),
        "seal_uses_realized_output_or_service": bool(
            v313m["seal_uses_realized_output_or_service"]
        ),
        "scheduled_control_opportunity_times_ms": json.dumps(
            v313m["scheduled_control_opportunity_times_ms"]
        ),
        "failed_rank_count_min": min(failed_rank_counts),
        "failed_rank_count_max": max(failed_rank_counts),
        "topology_bijective": bool(v313m["topology_bijective"]),
        "wrong_copy_path_keys": json.dumps(wrong_copy_paths),
        "surviving_cross_domain_committed_count": int(
            v312m["surviving_cross_domain_committed_count"]
        ),
        "node_prebound_survivor_count": int(v312m["node_prebound_survivor_count"]),
        "node_replacement_no_survivor_count": int(
            v312m["node_replacement_no_survivor_count"]
        ),
        "path_event_counts": json.dumps(mechanism["path_event_counts"], sort_keys=True),
        "recovery_type_path_counts": json.dumps(type_paths, sort_keys=True),
        "failover_mode_counts": json.dumps(
            mechanism["failover_mode_counts"], sort_keys=True
        ),
        "residency_event_counts": json.dumps(
            mechanism["residency_event_counts"], sort_keys=True
        ),
        "empty_replay_max_abs_error_ms": empty_error,
        "fault_monotonic_max_negative_error_ms": monotonic_error,
        "raw_npz": f"raw/{raw_name}",
        "raw_npz_sha256": core.sha256(raw_path),
    }
    counts: list[dict[str, Any]] = []
    for category, rows in (
        ("path", mechanism["path_event_counts"]),
        ("recovery", type_paths),
        ("failover", mechanism["failover_mode_counts"]),
        ("residency", mechanism["residency_event_counts"]),
    ):
        for key, value in sorted(rows.items()):
            counts.append({
                "condition_id": job["condition_id"], "axis": axis,
                "direction": direction or "", "failure_type": failure_type,
                "policy": policy, "seed": int(seed),
                "category": category, "key": key, "value": value,
            })
    return cell, counts


def jobs(spec: dict[str, Any], *, gate: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bandwidths = (
        [10.0] if gate else [float(value) for value in spec["sweep"]["bandwidth_values_gb_s"]]
    )
    for direction in spec["sweep"]["bandwidth_directions"]:
        for bandwidth in bandwidths:
            rows.append({
                "condition_id": f"bw__{direction}__{slug(str(bandwidth))}",
                "axis": "bandwidth", "direction": direction,
                "bandwidth_gb_s": float(bandwidth), "repair_ms": None,
                "failure_type": spec["workload"]["bandwidth_failure_type"],
            })
    repair_values = (
        [30000.0] if gate else [float(value) for value in spec["sweep"]["repair_durations_ms"]]
    )
    for domain in spec["sweep"]["repair_failure_domains"]:
        for repair_ms in repair_values:
            rows.append({
                "condition_id": f"repair__{domain}__{int(repair_ms)}ms",
                "axis": "repair", "direction": None,
                "bandwidth_gb_s": None, "repair_ms": float(repair_ms),
                "failure_type": domain,
            })
    return rows


def aggregate(cells: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    aggregate_rows: list[dict[str, Any]] = []
    paired_seed_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    conditions = sorted({row["condition_id"] for row in cells})
    for condition in conditions:
        sample = next(row for row in cells if row["condition_id"] == condition)
        for policy in ("Native Mooncake", "AegisKV"):
            selected = [
                row for row in cells
                if row["condition_id"] == condition and row["policy"] == policy
            ]
            out: dict[str, Any] = {
                "condition_id": condition, "axis": sample["axis"],
                "direction": sample["direction"],
                "failure_type": sample["failure_type"],
                "bandwidth_gb_s": sample["bandwidth_gb_s"],
                "repair_ms": sample["repair_ms"], "policy": policy,
                "seed_count": len(selected),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                out[f"{metric}_mean"] = statistics.fmean(values)
                out[f"{metric}_ci95_half_width"] = ci95(values)
            aggregate_rows.append(out)

        for seed in sorted({int(row["seed"]) for row in cells if row["condition_id"] == condition}):
            by_policy = {
                row["policy"]: row for row in cells
                if row["condition_id"] == condition and int(row["seed"]) == seed
            }
            native = by_policy["Native Mooncake"]
            aegis = by_policy["AegisKV"]
            paired_seed_rows.append({
                "condition_id": condition, "axis": sample["axis"],
                "direction": sample["direction"],
                "failure_type": sample["failure_type"],
                "bandwidth_gb_s": sample["bandwidth_gb_s"],
                "repair_ms": sample["repair_ms"], "seed": seed,
                "fault_wct_saved_ms": float(native["fault_wct_mean_ms"]) - float(aegis["fault_wct_mean_ms"]),
                "fault_wct_relative_gain": 1.0 - float(aegis["fault_wct_mean_ms"]) / float(native["fault_wct_mean_ms"]),
                "healthy_wct_saved_ms": float(native["healthy_wct_mean_ms"]) - float(aegis["healthy_wct_mean_ms"]),
                "copy_burden_saved_ms": float(native["critical_copy_burden_mean_ms"]) - float(aegis["critical_copy_burden_mean_ms"]),
                "recovery_service_saved_ms": float(native["recovery_service_mean_ms"]) - float(aegis["recovery_service_mean_ms"]),
                "recovery_hit_fraction_delta": float(aegis["recovery_hit_fraction"]) - float(native["recovery_hit_fraction"]),
                "native_fault_wct_mean_ms": float(native["fault_wct_mean_ms"]),
                "aegis_fault_wct_mean_ms": float(aegis["fault_wct_mean_ms"]),
            })
    for condition in conditions:
        selected = [row for row in paired_seed_rows if row["condition_id"] == condition]
        sample = selected[0]
        out = {
            key: sample[key] for key in (
                "condition_id", "axis", "direction", "failure_type",
                "bandwidth_gb_s", "repair_ms",
            )
        }
        out["seed_count"] = len(selected)
        for metric in (
            "fault_wct_saved_ms", "fault_wct_relative_gain",
            "healthy_wct_saved_ms", "copy_burden_saved_ms",
            "recovery_service_saved_ms", "recovery_hit_fraction_delta",
            "native_fault_wct_mean_ms", "aegis_fault_wct_mean_ms",
        ):
            values = [float(row[metric]) for row in selected]
            out[f"{metric}_mean"] = statistics.fmean(values)
            out[f"{metric}_ci95_half_width"] = ci95(values)
        paired_rows.append(out)
    return aggregate_rows, paired_seed_rows, paired_rows


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.3,
        "axes.titlesize": 8.2, "axes.labelsize": 7.7,
        "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "lines.linewidth": 1.25,
    })


def plot_s7(
    path: Path, aggregates: list[dict[str, Any]], paired: list[dict[str, Any]],
) -> None:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.8))
    directions = {
        "source_to_peer_hbm_copy": ("Source→Peer copy", "#2166ac", "o"),
        "source_to_dram_copy": ("Source→DRAM copy", "#67a9cf", "s"),
        "peer_hbm_recovery": ("Peer recovery", "#b2182b", "^"),
        "dram_recovery": ("DRAM recovery", "#ef8a62", "D"),
    }
    for direction, (label, color, marker) in directions.items():
        rows = sorted(
            (row for row in paired if row["axis"] == "bandwidth" and row["direction"] == direction),
            key=lambda row: float(row["bandwidth_gb_s"]),
        )
        x = [float(row["bandwidth_gb_s"]) for row in rows]
        for ax, metric, scale in (
            (axes[0, 0], "fault_wct_saved_ms", 0.001),
            (axes[0, 1], "fault_wct_relative_gain", 100.0),
        ):
            ax.errorbar(
                x, [float(row[f"{metric}_mean"]) * scale for row in rows],
                yerr=[float(row[f"{metric}_ci95_half_width"]) * scale for row in rows],
                color=color, marker=marker, markersize=3.4, capsize=2.0,
                label=label,
            )
    axes[0, 0].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 0].set_title("Absolute fault-WCT saved")
    axes[0, 0].set_ylabel("Native − AegisKV (s)")
    axes[0, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[0, 1].set_title("Relative fault-WCT gain")
    axes[0, 1].set_ylabel("Gain (%)")
    axes[0, 0].legend(frameon=False, ncol=1, handlelength=1.6)

    policy_style = {
        "Native Mooncake": ("#4d4d4d", "s", "--"),
        "AegisKV": ("#00897b", "o", "-"),
    }
    for direction in ("source_to_peer_hbm_copy", "source_to_dram_copy"):
        for policy, (color, marker, line) in policy_style.items():
            rows = sorted(
                (row for row in aggregates if row["axis"] == "bandwidth" and row["direction"] == direction and row["policy"] == policy),
                key=lambda row: float(row["bandwidth_gb_s"]),
            )
            short = "Peer" if "peer" in direction else "DRAM"
            axes[0, 2].errorbar(
                [float(row["bandwidth_gb_s"]) for row in rows],
                [float(row["critical_copy_burden_mean_ms_mean"]) for row in rows],
                yerr=[float(row["critical_copy_burden_mean_ms_ci95_half_width"]) for row in rows],
                color=color, marker=marker, linestyle=line, markersize=3.2,
                capsize=2.0, label=f"{policy} / {short}",
            )
    axes[0, 2].set_title("Healthy copy burden")
    axes[0, 2].set_ylabel("Critical-path burden (ms)")
    axes[0, 2].legend(frameon=False, ncol=1, handlelength=1.6)

    for direction in ("peer_hbm_recovery", "dram_recovery"):
        for policy, (color, marker, line) in policy_style.items():
            rows = sorted(
                (row for row in aggregates if row["axis"] == "bandwidth" and row["direction"] == direction and row["policy"] == policy),
                key=lambda row: float(row["bandwidth_gb_s"]),
            )
            short = "Peer" if direction.startswith("peer") else "DRAM"
            x = [float(row["bandwidth_gb_s"]) for row in rows]
            axes[1, 0].errorbar(
                x, [float(row["recovery_service_mean_ms_mean"]) for row in rows],
                yerr=[float(row["recovery_service_mean_ms_ci95_half_width"]) for row in rows],
                color=color, marker=marker, linestyle=line, markersize=3.2,
                capsize=2.0, label=f"{policy} / {short}",
            )
            axes[1, 1].errorbar(
                x, [100.0 * float(row["recovery_hit_fraction_mean"]) for row in rows],
                yerr=[100.0 * float(row["recovery_hit_fraction_ci95_half_width"]) for row in rows],
                color=color, marker=marker, linestyle=line, markersize=3.2,
                capsize=2.0, label=f"{policy} / {short}",
            )
    axes[1, 0].set_title("Recovery service")
    axes[1, 0].set_ylabel("Service (ms)")
    axes[1, 1].set_title("Recovery coverage")
    axes[1, 1].set_ylabel("Replica-hit calls (%)")
    axes[1, 0].legend(frameon=False, ncol=1, handlelength=1.6)

    repair = axes[1, 2]
    repair_gain = repair.twinx()
    domain_style = {
        "source_worker_gpu": ("Worker/GPU", "#5e3c99", "o"),
        "source_node": ("Source node", "#e66101", "s"),
    }
    for domain, (label, color, marker) in domain_style.items():
        rows = sorted(
            (row for row in paired if row["axis"] == "repair" and row["failure_type"] == domain),
            key=lambda row: float(row["repair_ms"]),
        )
        x = [float(row["repair_ms"]) / 1000.0 for row in rows]
        repair.errorbar(
            x, [float(row["fault_wct_saved_ms_mean"]) / 1000.0 for row in rows],
            yerr=[float(row["fault_wct_saved_ms_ci95_half_width"]) / 1000.0 for row in rows],
            color=color, marker=marker, markersize=3.4, capsize=2.0,
            label=f"{label}: saved",
        )
        repair_gain.plot(
            x, [100.0 * float(row["fault_wct_relative_gain_mean"]) for row in rows],
            color=color, marker=marker, markersize=3.0, linestyle=":",
            label=f"{label}: gain",
        )
    repair.axhline(0, color="#777777", linewidth=0.8)
    repair.set_title("Modeled replacement sensitivity")
    repair.set_ylabel("WCT saved (s)")
    repair_gain.set_ylabel("Gain (%)")
    handles1, labels1 = repair.get_legend_handles_labels()
    handles2, labels2 = repair_gain.get_legend_handles_labels()
    repair.legend(
        handles1 + handles2, labels1 + labels2, frameon=False,
        ncol=1, handlelength=1.6, loc="best",
    )

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
        ax.set_xlabel("Bandwidth (GB/s)" if ax is not repair else "Replacement (s)")
    fig.suptitle("S7 directional bandwidth and repair sensitivity", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.45, w_pad=1.2)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def verify_figure(pdf: Path, png: Path, min_font: float, min_width: int) -> dict[str, Any]:
    with fitz.open(pdf) as document:
        sizes = [
            float(span["size"])
            for block in document[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", []) if span["text"].strip()
        ]
        page_count = len(document)
    with Image.open(png) as image:
        width, height = image.size
    return {
        "single_page_vector_pdf": page_count == 1 and bool(sizes),
        "minimum_extracted_font_pt": min(sizes),
        "minimum_font_at_least_registered": min(sizes) >= min_font - 1e-6,
        "png_width_px": width, "png_height_px": height,
        "png_width_at_least_registered": width >= min_width,
    }


def run_tests(stage: Path) -> list[dict[str, Any]]:
    rows = []
    for filename, marker in TESTS:
        result = subprocess.run(
            ["python", str(ROOT / filename)], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (stage / "test_logs" / f"{Path(filename).stem}.log").write_text(
            result.stdout, encoding="utf-8"
        )
        passed = result.returncode == 0 and marker in result.stdout
        rows.append({"test": filename, "marker": marker, "passed": passed})
        if not passed:
            raise RuntimeError(f"test gate failed: {filename}\n{result.stdout}")
    return rows


def policy_failover_contract_ok(cell: dict[str, Any]) -> bool:
    counts = json.loads(cell["failover_mode_counts"])
    domain = cell["failure_type"]
    if cell["policy"] == "Native Mooncake":
        return bool(counts) and set(counts) == {f"{domain}:replacement_wait"}
    allowed = (
        {"source_worker_gpu:prebound_fence_activation"}
        if domain == "source_worker_gpu" else {
            "source_node:cross_domain_committed_fence_activation",
            "source_node:replacement_no_survivor",
        }
    )
    return bool(counts) and set(counts).issubset(allowed)


def main() -> int:
    args = parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    result_name = GATE_NAME if args.gate else RESULT_NAME
    target = ROOT / "results" / result_name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{result_name}.stage-{os.getpid()}"
    (stage / "raw").mkdir(parents=True)
    (stage / "figures").mkdir()
    (stage / "test_logs").mkdir()
    shutil.copy2(SPEC_PATH, stage / "protocol.json")
    shutil.copy2(ROOT / "active_hbm_v3_protocol.json", stage / "capacity_protocol.json")
    tests = run_tests(stage)

    conditions = jobs(spec, gate=args.gate)
    seeds = (
        [int(spec["sweep"]["paired_seeds"][0])]
        if args.gate else [int(value) for value in spec["sweep"]["paired_seeds"]]
    )
    work = [
        (condition, policy, seed)
        for condition in conditions
        for policy in spec["sweep"]["policies"] for seed in seeds
    ]
    expected = 12 if args.gate else int(spec["sweep"]["registered_cell_count"])
    workers = 1 if args.gate else max(1, min(int(args.workers), len(work)))
    cells: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    completed = 0
    if workers == 1:
        for condition, policy, seed in work:
            cell, counts = run_cell(stage, spec, condition, policy, seed)
            cells.append(cell); count_rows.extend(counts); completed += 1
            print(f"S7 v313 {completed}/{expected} {condition['condition_id']} {policy} s{seed}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(run_cell, stage, spec, condition, policy, seed):
                (condition, policy, seed)
                for condition, policy, seed in work
            }
            for future in concurrent.futures.as_completed(future_to_job):
                condition, policy, seed = future_to_job[future]
                cell, counts = future.result()
                cells.append(cell); count_rows.extend(counts); completed += 1
                print(f"S7 v313 {completed}/{expected} {condition['condition_id']} {policy} s{seed}", flush=True)
    cells.sort(key=lambda row: (
        row["axis"], row["direction"], row["failure_type"],
        float(row["bandwidth_gb_s"] or -1), float(row["repair_ms"] or -1),
        row["policy"], int(row["seed"]),
    ))
    count_rows.sort(key=lambda row: (
        row["condition_id"], row["policy"], int(row["seed"]),
        row["category"], row["key"],
    ))
    write_csv(stage / "cell_metrics.csv", cells)
    write_csv(stage / "mechanism_counts.csv", count_rows)
    with (stage / "cell_summary.jsonl").open("w", encoding="utf-8") as stream:
        for row in cells:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    aggregate_rows: list[dict[str, Any]] = []
    paired_seed_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    figure_checks: dict[str, Any] = {}
    if not args.gate:
        aggregate_rows, paired_seed_rows, paired_rows = aggregate(cells)
        write_csv(stage / "aggregate_ci.csv", aggregate_rows)
        write_csv(stage / "paired_seed_metrics.csv", paired_seed_rows)
        write_csv(stage / "paired_ci.csv", paired_rows)
        stem = stage / "figures" / spec["figure_grammar"]["filename_stem"]
        plot_s7(stem, aggregate_rows, paired_rows)
        figure_checks = verify_figure(
            stem.with_suffix(".pdf"), stem.with_suffix(".png"),
            float(spec["figure_grammar"]["minimum_font_pt"]),
            int(spec["figure_grammar"]["png_minimum_width_px"]),
        )

    expected_controls = spec["workload"]["control_opportunity_times_ms"]
    same_seed_groups = sorted({int(row["seed"]) for row in cells})
    common_universe = all(
        len({
            (row["workflow_universe_sha256"], row["measurement_cohort_sha256"], row["probe_base_tuple_sha256"])
            for row in cells if int(row["seed"]) == seed
        }) == 1 for seed in same_seed_groups
    )
    condition_seed_groups = sorted({
        (row["condition_id"], int(row["seed"])) for row in cells
    })
    common_probe_policy = all(
        len({
            (row["fault_probe_tuple_sha256"], row["scheduled_request_set_sha256"])
            for row in cells if (row["condition_id"], int(row["seed"])) == key
        }) == 1 for key in condition_seed_groups
    )
    checks: dict[str, bool] = {
        "mechanism_and_protocol_tests_pass": all(row["passed"] for row in tests),
        "exact_registered_cell_count": len(cells) == expected,
        "all_treatments_policies_seeds_present": len(condition_seed_groups) * 2 == len(cells),
        "common_workflow_cohort_probe_base_universe": common_universe,
        "common_probe_and_request_hash_across_policies": common_probe_policy,
        "bandwidth_treatment_changes_only_registered_table_key": all(
            bool(row["treatment_isolation_exact"]) for row in cells
        ),
        "failure_domain_rank_count_exact": all(
            int(row["failed_rank_count_min"]) == int(row["failed_rank_count_max"])
            == (1 if row["failure_type"] == "source_worker_gpu" else 8)
            for row in cells
        ),
        "cross_node_remote_topology_and_copy_paths_exact": all(
            bool(row["topology_bijective"])
            and json.loads(row["wrong_copy_path_keys"]) == [] for row in cells
        ),
        "policy_specific_failover_contract_exact": all(
            policy_failover_contract_ok(row) for row in cells
        ),
        "physical_target_capacity_and_copy_conservation": all(
            int(row["capacity_violation_count"]) == 0
            and int(row["capacity_max_shared_used_bytes"])
            <= int(spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"])
            and int(row["capacity_min_headroom_bytes"]) >= 0
            and int(row["target_accounting_max_abs_error_bytes"]) == 0
            and json.loads(row["copy_byte_conservation_error"])
            == {"DRAM": 0, "HBM": 0}
            for row in cells
        ),
        "no_future_unsealed_or_oracle_backup_dependency": all(
            int(row["future_backup_projection_dispatch_count"]) == 0
            and int(row["seal_before_dispatch_count"]) == 0
            and row["seal_uses_realized_output_or_service"] is False
            for row in cells
        ),
        "empty_replay_exact_and_fault_monotone": all(
            float(row["empty_replay_max_abs_error_ms"]) <= 1e-7
            and float(row["fault_monotonic_max_negative_error_ms"]) <= 1e-7
            for row in cells
        ),
        "six_exact_control_opportunities": all(
            json.loads(row["scheduled_control_opportunity_times_ms"])
            == expected_controls for row in cells
        ),
        "raw_hashes_match": all(
            core.sha256(stage / row["raw_npz"]) == row["raw_npz_sha256"]
            for row in cells
        ),
        "preregistered_source_hashes_match": all(
            Path(row["path"]).exists()
            and core.sha256(Path(row["path"])) == row["sha256"]
            for row in spec["sources"].values()
        ),
        "no_performance_direction_gain_seed_cell_or_path_selection": True,
    }
    if not args.gate:
        checks.update({
            "exact_120_bandwidth_and_48_repair_cells": (
                sum(row["axis"] == "bandwidth" for row in cells) == 120
                and sum(row["axis"] == "repair" for row in cells) == 48
            ),
            "all_four_directions_and_five_bandwidths_retained": (
                {row["direction"] for row in cells if row["axis"] == "bandwidth"}
                == set(spec["sweep"]["bandwidth_directions"])
                and {float(row["bandwidth_gb_s"]) for row in cells if row["axis"] == "bandwidth"}
                == set(float(value) for value in spec["sweep"]["bandwidth_values_gb_s"])
            ),
            "exact_three_seeds_per_aggregate": all(
                int(row["seed_count"]) == 3
                for row in aggregate_rows + paired_rows
            ),
            "absolute_and_relative_paired_ci_complete": (
                len(paired_rows) == 28
                and all(
                    key in row for row in paired_rows
                    for key in (
                        "fault_wct_saved_ms_mean",
                        "fault_wct_saved_ms_ci95_half_width",
                        "fault_wct_relative_gain_mean",
                        "fault_wct_relative_gain_ci95_half_width",
                    )
                )
            ),
            "figure_single_page_vector_pdf": figure_checks["single_page_vector_pdf"],
            "figure_png_width_at_least_1800": figure_checks["png_width_at_least_registered"],
            "figure_minimum_font_at_least_7pt": figure_checks["minimum_font_at_least_registered"],
        })
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "schema_version": 1, "status": status,
        "mode": "gate" if args.gate else "full", "paper_ready": False,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "cell_count": len(cells), "expected_cell_count": expected,
        "protocol_sha256": core.sha256(stage / "protocol.json"),
        "capacity_protocol_sha256": core.sha256(stage / "capacity_protocol.json"),
        "cell_metrics_sha256": core.sha256(stage / "cell_metrics.csv"),
        "mechanism_counts_sha256": core.sha256(stage / "mechanism_counts.csv"),
        "figure_checks": figure_checks,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1, "status": status, "paper_ready": False,
        "mode": audit["mode"], "selection": "S7",
        "protocol": "protocol.json", "protocol_sha256": audit["protocol_sha256"],
        "runner_sha256": core.sha256(Path(__file__)),
        "execution_workers": workers,
        "audit": "audit.json", "audit_sha256": core.sha256(stage / "audit.json"),
        "cell_metrics": "cell_metrics.csv",
        "cell_summary": "cell_summary.jsonl",
        "mechanism_counts": "mechanism_counts.csv",
        "aggregate_ci": None if args.gate else "aggregate_ci.csv",
        "paired_seed_metrics": None if args.gate else "paired_seed_metrics.csv",
        "paired_ci": None if args.gate else "paired_ci.csv",
        "figure": None if args.gate else {
            "pdf": f"figures/{spec['figure_grammar']['filename_stem']}.pdf",
            "png": f"figures/{spec['figure_grammar']['filename_stem']}.png",
        },
        "no_direction_or_result_selection": True,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "PASS":
        raise RuntimeError(f"S7 audit failed: {audit['failed_checks']}")
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
    print(json.dumps({
        "output": str(target), "status": status, "cells": len(cells),
        "audit_sha256": core.sha256(target / "audit.json"),
        "manifest_sha256": core.sha256(target / "manifest.json"),
        "sha256sums_sha256": core.sha256(target / "SHA256SUMS"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
