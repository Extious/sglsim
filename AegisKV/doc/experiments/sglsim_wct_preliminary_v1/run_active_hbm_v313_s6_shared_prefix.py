#!/usr/bin/env python3
"""Run the preregistered v3.1.3 S6 explicit shared-prefix pilot."""

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
from pathlib import Path
from typing import Any, Iterable

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

import active_hbm_v313 as v313
import active_hbm_v313_s6 as s6
import run_preliminary as core
import test_active_hbm_v313 as v313_tests


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s6_shared_prefix_protocol.json"
RESULT_NAME = "active_hbm_v313_s6_shared_prefix_pilot"
GATE_NAME = "active_hbm_v313_s6_shared_prefix_gate"
T95_DF2 = 4.302652729911275
TESTS = (
    ("test_active_hbm_v3_capacity.py", "RESULT 5/5 PASS"),
    ("test_active_hbm_v313_protocol.py", "RESULT 6/6 PASS"),
    ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    ("test_active_hbm_v313_s6_protocol.py", "RESULT 10/10 PASS"),
    ("test_active_hbm_v313_s6_shared_prefix.py", "RESULT 10/10 PASS"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate", action="store_true",
        help="run reuse boundaries, same-node topology, two policies, first seed",
    )
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_csv(
    path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None,
) -> None:
    if not rows and fields is None:
        raise ValueError(f"cannot infer CSV fields for empty {path}")
    columns = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) < 2:
        return 0.0
    if len(rows) != 3:
        raise ValueError("S6 full pilot preregisters exactly three seeds")
    return T95_DF2 * statistics.stdev(rows) / math.sqrt(len(rows))


def slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def configured_protocol(
    spec: dict[str, Any], *, topology: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, capacity = v313_tests.inputs(
        context=int(spec["model"]["context_tokens"]),
        active=int(spec["workload"]["offered_active_requests_per_gpu"]),
    )
    protocol["models"]["qwen3_8b"]["center_system_dp"] = int(
        spec["hardware"]["system_dp"]
    )
    protocol["workload"]["center_context_tokens"] = int(
        spec["model"]["context_tokens"]
    )
    protocol["workload"]["center_active_requests_per_gpu"] = int(
        spec["workload"]["offered_active_requests_per_gpu"]
    )
    protocol["hardware"]["gpus_per_node"] = int(
        spec["hardware"]["gpus_per_node"]
    )
    protocol["hardware"]["node_count"] = int(spec["hardware"]["node_count"])
    protocol["hardware"]["copy_bandwidth_gb_s"] = dict(
        spec["hardware"]["copy_bandwidth_gb_s"]
    )
    protocol["hardware"]["load_bandwidth_gb_s"] = dict(
        spec["hardware"]["recovery_bandwidth_gb_s"]
    )
    route = spec["hardware"]["topologies"][topology]
    protocol.setdefault("runtime_overrides", {})["topology"] = {
        "peer_hbm": route["peer_hbm"], "dram": route["dram"]
    }
    return protocol, capacity


def workload_hash(tasks: list[core.Task]) -> str:
    return digest_json([
        (
            task.request_id, task.workflow_id, task.workflow_arrival_ms,
            task.arrival_ms, task.source_rank, task.service_ms,
            task.prompt_tokens, task.output_tokens, task.predicted_slack_ms,
        )
        for task in tasks
    ])


def probe_hash(probes: list[core.FaultProbe]) -> str:
    return digest_json([
        (row.probe_id, row.time_ms, row.source_rank, row.failure_type)
        for row in probes
    ])


def metric_mean_p99(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), core.quantile(values, 0.99)


def run_tests(stage: Path) -> list[dict[str, Any]]:
    rows = []
    for filename, marker in TESTS:
        result = subprocess.run(
            ["python", str(ROOT / filename)], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        log = stage / "test_logs" / f"{Path(filename).stem}.log"
        log.write_text(result.stdout, encoding="utf-8")
        passed = result.returncode == 0 and marker in result.stdout
        rows.append({"test": filename, "marker": marker, "passed": passed})
        if not passed:
            raise RuntimeError(f"test gate failed: {filename}\n{result.stdout}")
    return rows


def build_universe(
    spec: dict[str, Any], seed: int,
) -> tuple[list[core.Task], dict[str, Any], list[core.FaultProbe]]:
    protocol, _capacity = configured_protocol(spec, topology="same_node")
    nominal, workload = core.build_workload(
        protocol, seed=int(seed), workflows=None,
        active_per_gpu=int(spec["workload"]["offered_active_requests_per_gpu"]),
        arrival_cutoff_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
    )
    probes = core.make_fault_probes(
        nominal, seed=(int(seed) ^ 0x56360006),
        count=int(spec["workload"]["fault_probes_per_seed"]),
        gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
        right_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
        failure_types=("source_worker_gpu",),
    )
    return nominal, workload, probes


def run_cell(
    *, stage: Path, spec: dict[str, Any], reuse: float,
    topology: str, policy: str, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, capacity = configured_protocol(spec, topology=topology)
    nominal, workload, probes = build_universe(spec, seed)
    assignments = s6.build_prefix_assignments(
        nominal, reuse_fraction=float(reuse),
        group_size=int(spec["shared_prefix_semantics"]["group_size"]),
        chunk_tokens=int(spec["shared_prefix_semantics"]["prefix_chunk_tokens"]),
    )
    replay = s6.SharedPrefixReplay(
        deepcopy(protocol), policy, capacity,
        assignments=assignments, reuse_fraction=float(reuse),
        group_size=int(spec["shared_prefix_semantics"]["group_size"]),
    )
    tasks, delay_maps, mechanism, capacity_trace = replay.run_coupled(
        nominal, probes,
        control_horizon_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
    )
    base_ids, base_wct = core.workflow_wct(tasks)
    empty_completion = core.replay_one_fault(tasks, {})
    empty_ids, empty_wct = core.workflow_wct(tasks, empty_completion)
    if base_ids != empty_ids:
        raise AssertionError("empty replay changed workflow IDs")
    empty_error = max(abs(left - right) for left, right in zip(base_wct, empty_wct))
    fault_matrix: list[list[float]] = []
    probe_changed: list[int] = []
    monotonic_error = 0.0
    for delays in delay_maps:
        completion = core.replay_one_fault(tasks, delays)
        ids, values = core.workflow_wct(tasks, completion)
        if ids != base_ids:
            raise AssertionError("fault replay changed workflow IDs")
        monotonic_error = max(
            monotonic_error,
            max((base - value for base, value in zip(base_wct, values)), default=0.0),
        )
        fault_matrix.append(values)
        probe_changed.append(len(delays))
    healthy_mean, healthy_p99 = metric_mean_p99(base_wct)
    fault_flat = [value for row in fault_matrix for value in row]
    fault_mean, fault_p99 = metric_mean_p99(fault_flat)
    s6m = mechanism["active_hbm_v313_s6"]
    v3m = mechanism["active_hbm_v3"]
    v313m = mechanism["active_hbm_v313"]
    placement = spec["hardware"]["topologies"][topology]
    copy_bw = core.copy_bandwidth(protocol, 0)
    load_bw = core.load_bandwidth(protocol, 0)
    max_shared = max((int(row["shared_used_bytes"]) for row in capacity_trace), default=0)
    min_headroom = min(
        (
            int(spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"])
            - int(row["shared_used_bytes"])
            for row in capacity_trace
        ),
        default=int(spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"]),
    )
    occupancy = replay.physical_occupancy_trace
    raw_name = (
        f"reuse{str(reuse).replace('.', 'p')}__{topology}__"
        f"{slug(policy)}__s{seed}.npz"
    )
    raw_path = stage / "raw" / raw_name
    np.savez_compressed(
        raw_path,
        workflow_ids=np.asarray(base_ids),
        healthy_wct_ms=np.asarray(base_wct, dtype=float),
        worker_fault_wct_ms=np.asarray(fault_matrix, dtype=float),
        probe_id=np.asarray([row.probe_id for row in probes]),
        probe_time_ms=np.asarray([row.time_ms for row in probes], dtype=float),
        probe_source_rank=np.asarray([row.source_rank for row in probes], dtype=int),
        probe_changed_request_count=np.asarray(probe_changed, dtype=int),
        occupancy_time_ms=np.asarray([row[0] for row in occupancy], dtype=float),
        occupancy_hbm_bytes=np.asarray([row[1] for row in occupancy], dtype=np.int64),
        occupancy_dram_bytes=np.asarray([row[2] for row in occupancy], dtype=np.int64),
        capacity_time_ms=np.asarray(
            [row["time_ms"] for row in capacity_trace], dtype=float
        ),
        capacity_rank=np.asarray([row["rank"] for row in capacity_trace], dtype=int),
        capacity_serving_bytes=np.asarray(
            [row["active_serving_kv_bytes"] for row in capacity_trace], dtype=np.int64
        ),
        capacity_peer_backup_bytes=np.asarray(
            [row["peer_hbm_backup_used_bytes"] for row in capacity_trace], dtype=np.int64
        ),
        capacity_shared_used_bytes=np.asarray(
            [row["shared_used_bytes"] for row in capacity_trace], dtype=np.int64
        ),
    )
    logical = int(s6m["logical_useful_bytes"])
    theoretical_total_tiers = 2 if policy == "AegisKV" else 1
    evicted_total = sum(int(value) for value in s6m["physical_evicted_bytes"].values())
    reclaim = int(v313m["primary_priority_reclaimed_bytes"])
    restored = int(mechanism["recovery_restored_tokens"])
    recomputed = int(mechanism["recovery_recomputed_tokens"])
    occupancy_identity_error = max(
        abs(
            float(s6m["physical_occupancy_time_mean_gib_per_gpu"][tier])
            - float(s6m["physical_capacity_integral_byte_ms"][tier])
            / float(spec["workload"]["fixed_arrival_cutoff_ms"])
            / core.GIB / int(spec["hardware"]["system_dp"])
        )
        for tier in ("HBM", "DRAM")
    )
    cell = {
        "reuse_fraction": float(reuse),
        "topology": topology,
        "peer_hbm_path": placement["peer_hbm"],
        "dram_path": placement["dram"],
        "peer_copy_bandwidth_gb_s": copy_bw[0],
        "dram_copy_bandwidth_gb_s": copy_bw[1],
        "peer_recovery_bandwidth_gb_s": load_bw[0],
        "dram_recovery_bandwidth_gb_s": load_bw[1],
        "policy": policy,
        "seed": int(seed),
        "workflow_universe_sha256": workload_hash(nominal),
        "fault_probe_tuple_sha256": probe_hash(probes),
        "prefix_assignment_sha256": s6.assignment_hash(assignments),
        "scheduled_request_set_sha256": digest_json(sorted(row.request_id for row in tasks)),
        "workflow_count": len(base_ids),
        "request_count": len(tasks),
        "actual_last_arrival_ms": float(workload["actual_last_arrival_ms"]),
        "arrival_horizon_ms": float(spec["workload"]["fixed_arrival_cutoff_ms"]),
        "drain_horizon_ms": max(row.finish_ms for row in tasks),
        "healthy_wct_mean_ms": healthy_mean,
        "healthy_wct_p99_ms": healthy_p99,
        "fault_wct_mean_ms": fault_mean,
        "fault_wct_p99_ms": fault_p99,
        "fault_added_wct_mean_ms": fault_mean - healthy_mean,
        "fault_probe_count": len(probes),
        "fault_observation_count": len(fault_flat),
        "recovery_call_count": int(s6m["recovery_call_count"]),
        "recovery_request_hit_fraction": float(s6m["recovery_request_hit_fraction"]),
        "recovery_prefix_token_hit_fraction": float(
            s6m["recovery_prefix_token_hit_fraction"]
        ),
        "recoverable_byte_fraction": float(s6m["recoverable_byte_fraction"]),
        "recovery_restored_tokens": restored,
        "recovery_recomputed_tokens": recomputed,
        "shared_recovery_restored_tokens": int(
            s6m["shared_recovery_restored_tokens"]
        ),
        "unique_recovery_restored_tokens": int(
            s6m["unique_recovery_restored_tokens"]
        ),
        "logical_useful_bytes": logical,
        "logical_shared_reference_bytes": int(s6m["logical_shared_reference_bytes"]),
        "logical_unique_reference_bytes": int(s6m["logical_unique_reference_bytes"]),
        "theoretical_deduplicated_physical_bytes_per_tier": int(
            s6m["theoretical_deduplicated_physical_bytes_per_tier"]
        ),
        "theoretical_copy_ratio_per_tier": float(
            s6m["theoretical_copy_ratio_per_tier"]
        ),
        "theoretical_total_copy_ratio": float(
            s6m["theoretical_copy_ratio_per_tier"]
        ) * theoretical_total_tiers,
        "scheduled_hbm_copy_bytes": int(s6m["scheduled_physical_copy_bytes"]["HBM"]),
        "scheduled_dram_copy_bytes": int(s6m["scheduled_physical_copy_bytes"]["DRAM"]),
        "scheduled_shared_hbm_copy_bytes": int(
            s6m["scheduled_shared_physical_copy_bytes"]["HBM"]
        ),
        "scheduled_shared_dram_copy_bytes": int(
            s6m["scheduled_shared_physical_copy_bytes"]["DRAM"]
        ),
        "copy_amplification_physical_over_logical": float(
            s6m["copy_amplification_physical_over_logical"]
        ),
        "physical_hbm_evicted_bytes": int(s6m["physical_evicted_bytes"]["HBM"]),
        "physical_dram_evicted_bytes": int(s6m["physical_evicted_bytes"]["DRAM"]),
        "primary_priority_reclaimed_bytes": reclaim,
        "eviction_plus_reclaim_gib": (evicted_total + reclaim) / core.GIB,
        "hbm_occupancy_time_mean_gib_per_gpu": float(
            s6m["physical_occupancy_time_mean_gib_per_gpu"]["HBM"]
        ),
        "dram_occupancy_time_mean_gib_per_gpu": float(
            s6m["physical_occupancy_time_mean_gib_per_gpu"]["DRAM"]
        ),
        "hbm_occupancy_peak_gib_per_gpu": float(
            s6m["physical_occupancy_peak_gib_per_gpu"]["HBM"]
        ),
        "dram_occupancy_peak_gib_per_gpu": float(
            s6m["physical_occupancy_peak_gib_per_gpu"]["DRAM"]
        ),
        "hbm_capacity_integral_byte_ms": float(
            s6m["physical_capacity_integral_byte_ms"]["HBM"]
        ),
        "dram_capacity_integral_byte_ms": float(
            s6m["physical_capacity_integral_byte_ms"]["DRAM"]
        ),
        "occupancy_integral_identity_max_abs_error": occupancy_identity_error,
        "occupancy_trace_change_points": int(
            s6m["physical_occupancy_trace_change_points"]
        ),
        "shared_prefix_group_count": int(s6m["shared_prefix_group_count"]),
        "shared_physical_chunk_identity_count": int(
            s6m["shared_physical_chunk_identity_count"]
        ),
        "shared_reference_count": int(s6m["shared_reference_count"]),
        "shared_reference_alias_hit_count": int(
            s6m["shared_reference_alias_hit_count"]
        ),
        "shared_reference_alias_hit_fraction": float(
            s6m["shared_reference_alias_hit_fraction"]
        ),
        "refcount_peak_total": int(s6m["refcount_peak_total"]),
        "refcount_final_total": int(s6m["refcount_final_total"]),
        "refcount_max_abs_error": int(s6m["refcount_max_abs_error"]),
        "refcount_underflow_count": int(s6m["refcount_underflow_count"]),
        "future_member_count_used_for_admission": int(
            s6m["future_member_count_used_for_admission"]
        ),
        "capacity_max_shared_used_bytes": max_shared,
        "capacity_min_headroom_bytes": min_headroom,
        "capacity_violation_count": int(v3m["capacity_violation_count"])
        + int(v313m["target_capacity_violation_count"]),
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
        "empty_replay_max_abs_error_ms": empty_error,
        "fault_monotonic_max_negative_error_ms": monotonic_error,
        "topology_bijective": bool(v313m["topology_bijective"]),
        "raw_npz": f"raw/{raw_name}",
        "raw_npz_sha256": core.sha256(raw_path),
    }
    count_rows: list[dict[str, Any]] = []
    for category, values in (
        ("path_event", mechanism["path_event_counts"]),
        ("residency_event", mechanism["residency_event_counts"]),
        ("recovery_path", mechanism["recovery_type_path_counts"]),
        ("failover_mode", mechanism["failover_mode_counts"]),
    ):
        for key, value in sorted(values.items()):
            count_rows.append({
                "reuse_fraction": reuse, "topology": topology,
                "policy": policy, "seed": seed, "category": category,
                "key": key, "value": value,
            })
    return cell, count_rows


METRICS = (
    "healthy_wct_mean_ms", "healthy_wct_p99_ms",
    "fault_wct_mean_ms", "fault_wct_p99_ms", "fault_added_wct_mean_ms",
    "recovery_request_hit_fraction", "recovery_prefix_token_hit_fraction",
    "recoverable_byte_fraction", "copy_amplification_physical_over_logical",
    "theoretical_total_copy_ratio", "eviction_plus_reclaim_gib",
    "hbm_occupancy_time_mean_gib_per_gpu",
    "dram_occupancy_time_mean_gib_per_gpu",
    "hbm_occupancy_peak_gib_per_gpu", "dram_occupancy_peak_gib_per_gpu",
    "shared_reference_alias_hit_fraction", "shared_reference_count",
    "scheduled_hbm_copy_bytes", "scheduled_dram_copy_bytes",
    "physical_hbm_evicted_bytes", "physical_dram_evicted_bytes",
    "primary_priority_reclaimed_bytes",
)


def aggregate(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    keys = sorted({
        (float(row["reuse_fraction"]), row["topology"], row["policy"])
        for row in cells
    })
    for reuse, topology, policy in keys:
        selected = [
            row for row in cells
            if (float(row["reuse_fraction"]), row["topology"], row["policy"])
            == (reuse, topology, policy)
        ]
        out: dict[str, Any] = {
            "reuse_fraction": reuse, "topology": topology,
            "policy": policy, "seed_count": len(selected),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            out[f"{metric}_mean"] = statistics.fmean(values)
            out[f"{metric}_ci95_half_width"] = ci95(values)
        aggregate_rows.append(out)

    paired_rows: list[dict[str, Any]] = []
    for reuse, topology in sorted({
        (float(row["reuse_fraction"]), row["topology"]) for row in cells
    }):
        selected = [
            row for row in cells
            if float(row["reuse_fraction"]) == reuse
            and row["topology"] == topology
        ]
        by_policy_seed = {
            (row["policy"], int(row["seed"])): row for row in selected
        }
        seeds = sorted({int(row["seed"]) for row in selected})
        for metric in (
            "healthy_wct_mean_ms", "fault_wct_mean_ms",
            "copy_amplification_physical_over_logical",
            "recovery_prefix_token_hit_fraction",
        ):
            native = [
                float(by_policy_seed[("Native Mooncake", seed)][metric])
                for seed in seeds
            ]
            aegis = [
                float(by_policy_seed[("AegisKV", seed)][metric])
                for seed in seeds
            ]
            deltas = [right - left for left, right in zip(native, aegis)]
            paired_rows.append({
                "reuse_fraction": reuse, "topology": topology,
                "metric": metric, "seed_count": len(seeds),
                "native_mean": statistics.fmean(native),
                "aegis_mean": statistics.fmean(aegis),
                "aegis_minus_native_mean": statistics.fmean(deltas),
                "aegis_minus_native_ci95_half_width": ci95(deltas),
                "ci_method": "paired seed-level Student-t interval, df=2",
            })
    return aggregate_rows, paired_rows


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.3,
        "axes.titlesize": 8.3, "axes.labelsize": 7.8,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2,
    })


def plot_s6(path: Path, rows: list[dict[str, Any]]) -> None:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.1), sharex=True)
    specs = (
        (axes[0, 0], "recovery_prefix_token_hit_fraction", "Recovery token hit", "Fraction"),
        (axes[0, 1], "copy_amplification_physical_over_logical", "Physical / logical copy", "Ratio"),
        (axes[0, 2], "eviction_plus_reclaim_gib", "Eviction + reclaim", "GiB"),
        (axes[1, 0], "hbm_occupancy_time_mean_gib_per_gpu", "Peer-HBM occupancy", "GiB / GPU"),
        (axes[1, 1], "dram_occupancy_time_mean_gib_per_gpu", "DRAM occupancy", "GiB / GPU"),
        (axes[1, 2], "shared_reference_alias_hit_fraction", "Shared-reference hit", "Fraction"),
    )
    colors = {"Native Mooncake": "#4d4d4d", "AegisKV": "#008080"}
    markers = {"Native Mooncake": "s", "AegisKV": "o"}
    linestyles = {"same_node": "-", "cross_node": "--"}
    for ax, metric, title, ylabel in specs:
        for policy in ("Native Mooncake", "AegisKV"):
            for topology in ("same_node", "cross_node"):
                selected = sorted(
                    (
                        row for row in rows
                        if row["policy"] == policy and row["topology"] == topology
                    ),
                    key=lambda row: float(row["reuse_fraction"]),
                )
                x = np.asarray([float(row["reuse_fraction"]) for row in selected])
                mean = np.asarray([float(row[f"{metric}_mean"]) for row in selected])
                half = np.asarray([
                    float(row[f"{metric}_ci95_half_width"]) for row in selected
                ])
                ax.errorbar(
                    x, mean, yerr=half, color=colors[policy],
                    marker=markers[policy], linestyle=linestyles[topology],
                    markersize=3.0, capsize=2.0,
                )
        if metric == "copy_amplification_physical_over_logical":
            for policy in ("Native Mooncake", "AegisKV"):
                selected = sorted(
                    (
                        row for row in rows
                        if row["policy"] == policy and row["topology"] == "same_node"
                    ),
                    key=lambda row: float(row["reuse_fraction"]),
                )
                ax.plot(
                    [float(row["reuse_fraction"]) for row in selected],
                    [float(row["theoretical_total_copy_ratio_mean"]) for row in selected],
                    color=colors[policy], linestyle=":", linewidth=0.9,
                )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", alpha=0.24)
    for ax in axes[1, :]:
        ax.set_xlabel("Shared-prefix fraction")
    handles = [
        Line2D([0], [0], color=colors["Native Mooncake"], marker="s", label="Native Mooncake"),
        Line2D([0], [0], color=colors["AegisKV"], marker="o", label="AegisKV"),
        Line2D([0], [0], color="#333333", linestyle="-", label="Same-node / local DRAM"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Cross-node / remote DRAM"),
        Line2D([0], [0], color="#333333", linestyle=":", label="Dedup-only byte ratio"),
    ]
    fig.legend(
        handles=handles, frameon=False, ncol=5, loc="lower center",
        bbox_to_anchor=(0.5, 0.006), columnspacing=1.05, handlelength=1.8,
    )
    fig.suptitle("Shared-prefix reuse mechanisms", fontsize=9.5, y=0.995)
    fig.tight_layout(rect=(0.01, 0.065, 0.995, 0.965), h_pad=1.25, w_pad=1.1)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def verify_figure(
    pdf: Path, png: Path, minimum_font: float, minimum_width: int,
) -> dict[str, Any]:
    document = fitz.open(pdf)
    spans = [
        span
        for page in document
        for block in page.get_text("dict").get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if span["text"].strip()
    ]
    text = " ".join(str(span["text"]) for span in spans).lower()
    with Image.open(png) as image:
        width, height = image.size
    banned = ("pilot", "preliminary", "seed", "sha256", "evidence")
    return {
        "single_page_vector_pdf": len(document) == 1 and bool(spans),
        "minimum_extracted_font_pt": min(float(span["size"]) for span in spans),
        "minimum_font_at_least_registered": min(
            float(span["size"]) for span in spans
        ) >= float(minimum_font) - 1e-6,
        "png_width_px": width,
        "png_height_px": height,
        "png_width_at_least_registered": width >= int(minimum_width),
        "no_evidence_boundary_prose_inside_panel": not any(word in text for word in banned),
    }


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
    reuse_values = (
        [0.0, 1.0] if args.gate
        else [float(value) for value in spec["sweep"]["reuse_fraction"]]
    )
    topologies = ["same_node"] if args.gate else list(spec["sweep"]["topologies"])
    seeds = (
        [int(spec["sweep"]["paired_seeds"][0])]
        if args.gate else [int(seed) for seed in spec["sweep"]["paired_seeds"]]
    )
    jobs = [
        (reuse, topology, policy, seed)
        for reuse in reuse_values for topology in topologies
        for policy in spec["sweep"]["policies"] for seed in seeds
    ]
    expected_cells = len(jobs)
    cells: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    workers = 1 if args.gate else max(1, min(int(args.workers), len(jobs)))
    if workers == 1:
        for index, (reuse, topology, policy, seed) in enumerate(jobs, 1):
            cell, current_counts = run_cell(
                stage=stage, spec=spec, reuse=reuse, topology=topology,
                policy=policy, seed=seed,
            )
            cells.append(cell)
            counts.extend(current_counts)
            print(
                f"S6 v313 {index}/{expected_cells} reuse={reuse:g} "
                f"{topology} {policy} s{seed}", flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {
                executor.submit(
                    run_cell, stage=stage, spec=spec, reuse=reuse,
                    topology=topology, policy=policy, seed=seed,
                ): (reuse, topology, policy, seed)
                for reuse, topology, policy, seed in jobs
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(future_to_job), 1
            ):
                reuse, topology, policy, seed = future_to_job[future]
                cell, current_counts = future.result()
                cells.append(cell)
                counts.extend(current_counts)
                print(
                    f"S6 v313 {index}/{expected_cells} reuse={reuse:g} "
                    f"{topology} {policy} s{seed}", flush=True,
                )
    cells.sort(key=lambda row: (
        float(row["reuse_fraction"]), row["topology"], row["policy"], int(row["seed"])
    ))
    counts.sort(key=lambda row: (
        float(row["reuse_fraction"]), row["topology"], row["policy"],
        int(row["seed"]), row["category"], row["key"],
    ))
    write_csv(stage / "cell_metrics.csv", cells)
    write_csv(stage / "mechanism_counts.csv", counts, [
        "reuse_fraction", "topology", "policy", "seed",
        "category", "key", "value",
    ])
    with (stage / "cell_summary.jsonl").open("w", encoding="utf-8") as stream:
        for cell in cells:
            stream.write(json.dumps(cell, sort_keys=True) + "\n")

    aggregate_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    figure_checks: dict[str, Any] = {}
    if not args.gate:
        aggregate_rows, paired_rows = aggregate(cells)
        write_csv(stage / "aggregate_ci.csv", aggregate_rows)
        write_csv(stage / "paired_ci.csv", paired_rows)
        figure_path = stage / "figures" / spec["figure_grammar"]["filename_stem"]
        plot_s6(figure_path, aggregate_rows)
        figure_checks = verify_figure(
            figure_path.with_suffix(".pdf"), figure_path.with_suffix(".png"),
            float(spec["figure_grammar"]["minimum_font_pt"]),
            int(spec["figure_grammar"]["png_minimum_width_px"]),
        )

    seed_groups = sorted({int(cell["seed"]) for cell in cells})
    common_universe_probe = all(
        len({
            (cell["workflow_universe_sha256"], cell["fault_probe_tuple_sha256"])
            for cell in cells if int(cell["seed"]) == seed
        }) == 1
        for seed in seed_groups
    )
    assignment_groups = sorted({
        (float(cell["reuse_fraction"]), int(cell["seed"])) for cell in cells
    })
    common_assignment = all(
        len({
            cell["prefix_assignment_sha256"] for cell in cells
            if (float(cell["reuse_fraction"]), int(cell["seed"])) == key
        }) == 1
        for key in assignment_groups
    )
    budget = int(spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"])
    checks = {
        "mechanism_and_protocol_tests_pass": all(row["passed"] for row in tests),
        "exact_registered_cell_count": len(cells) == expected_cells,
        "common_workflow_and_probe_universe_across_all_treatments": common_universe_probe,
        "common_prefix_assignment_across_policy_and_topology": common_assignment,
        "logical_shared_unique_bytes_conserve": all(
            int(cell["logical_useful_bytes"])
            == int(cell["logical_shared_reference_bytes"])
            + int(cell["logical_unique_reference_bytes"])
            for cell in cells
        ),
        "reuse_zero_boundary_exact": all(
            int(cell["shared_prefix_group_count"]) == 0
            and int(cell["shared_reference_count"]) == 0
            and int(cell["logical_shared_reference_bytes"]) == 0
            and math.isclose(
                float(cell["theoretical_copy_ratio_per_tier"]), 1.0,
                abs_tol=1e-12,
            )
            for cell in cells if float(cell["reuse_fraction"]) == 0.0
        ),
        "reuse_one_boundary_exact": all(
            int(cell["logical_unique_reference_bytes"]) == 0
            and int(cell["theoretical_deduplicated_physical_bytes_per_tier"])
            == int(cell["shared_prefix_group_count"])
            * int(spec["model"]["context_tokens"])
            * int(spec["model"]["kv_bytes_per_token"])
            for cell in cells if float(cell["reuse_fraction"]) == 1.0
        ),
        "refcount_conservation_and_final_drain": all(
            int(cell["refcount_final_total"]) == 0
            and int(cell["refcount_max_abs_error"]) == 0
            and int(cell["refcount_underflow_count"]) == 0
            for cell in cells
        ),
        "physical_copy_bytes_conserve_by_independent_tier": all(
            json.loads(cell["copy_byte_conservation_error"])
            == {"DRAM": 0, "HBM": 0}
            for cell in cells
        ),
        "recovery_token_accounting_conserves": all(
            int(cell["recovery_restored_tokens"])
            + int(cell["recovery_recomputed_tokens"])
            == int(cell["recovery_call_count"])
            * int(spec["model"]["context_tokens"])
            and int(cell["shared_recovery_restored_tokens"])
            + int(cell["unique_recovery_restored_tokens"])
            == int(cell["recovery_restored_tokens"])
            for cell in cells
        ),
        "physical_occupancy_is_capacity_integral_not_event_fraction": all(
            float(cell["occupancy_integral_identity_max_abs_error"]) <= 1e-12
            and int(cell["occupancy_trace_change_points"]) >= 1
            and float(cell["hbm_capacity_integral_byte_ms"]) >= 0
            and float(cell["dram_capacity_integral_byte_ms"]) >= 0
            for cell in cells
        ),
        "physical_target_40gib_capacity_invariant": all(
            int(cell["capacity_violation_count"]) == 0
            and int(cell["capacity_max_shared_used_bytes"]) <= budget
            and int(cell["capacity_min_headroom_bytes"]) >= 0
            and int(cell["target_accounting_max_abs_error_bytes"]) == 0
            for cell in cells
        ),
        "no_future_or_unsealed_backup_dependency": all(
            int(cell["future_member_count_used_for_admission"]) == 0
            and int(cell["future_backup_projection_dispatch_count"]) == 0
            and int(cell["seal_before_dispatch_count"]) == 0
            and cell["seal_uses_realized_output_or_service"] is False
            for cell in cells
        ),
        "empty_replay_exact_and_fault_monotone": all(
            float(cell["empty_replay_max_abs_error_ms"]) <= 1e-7
            and float(cell["fault_monotonic_max_negative_error_ms"]) <= 1e-7
            for cell in cells
        ),
        "directional_topology_bandwidth_exact": all(
            bool(cell["topology_bijective"])
            and (
                (
                    cell["topology"] == "same_node"
                    and float(cell["peer_copy_bandwidth_gb_s"]) == 50.0
                    and float(cell["dram_copy_bandwidth_gb_s"]) == 16.0
                    and float(cell["peer_recovery_bandwidth_gb_s"]) == 50.0
                    and float(cell["dram_recovery_bandwidth_gb_s"]) == 16.0
                )
                or (
                    cell["topology"] == "cross_node"
                    and float(cell["peer_copy_bandwidth_gb_s"]) == 10.0
                    and float(cell["dram_copy_bandwidth_gb_s"]) == 10.0
                    and float(cell["peer_recovery_bandwidth_gb_s"]) == 10.0
                    and float(cell["dram_recovery_bandwidth_gb_s"]) == 10.0
                )
            )
            for cell in cells
        ),
        "three_scheduled_control_opportunities": all(
            json.loads(cell["scheduled_control_opportunity_times_ms"])
            == [30000.0, 60000.0, 90000.0]
            for cell in cells
        ),
        "raw_npz_hashes_match": all(
            core.sha256(stage / cell["raw_npz"]) == cell["raw_npz_sha256"]
            for cell in cells
        ),
        "preregistered_source_hashes_match": all(
            Path(row["path"]).is_file()
            and core.sha256(Path(row["path"])) == row["sha256"]
            for row in spec["sources"].values()
        ),
        "no_performance_direction_or_cell_selection_gate": True,
    }
    if not args.gate:
        checks.update({
            "exact_three_seeds_per_aggregate": all(
                int(row["seed_count"]) == 3 for row in aggregate_rows
            ),
            "exact_20_aggregate_rows": len(aggregate_rows) == 20,
            "exact_40_paired_rows": len(paired_rows) == 40,
            "figure_single_page_vector_pdf": figure_checks["single_page_vector_pdf"],
            "figure_png_width_at_least_1800": figure_checks[
                "png_width_at_least_registered"
            ],
            "figure_minimum_font_at_least_7pt": figure_checks[
                "minimum_font_at_least_registered"
            ],
            "figure_has_no_evidence_boundary_prose": figure_checks[
                "no_evidence_boundary_prose_inside_panel"
            ],
        })
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "schema_version": 1,
        "status": status,
        "mode": "gate" if args.gate else "full",
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "cell_count": len(cells),
        "expected_cell_count": expected_cells,
        "protocol_sha256": core.sha256(stage / "protocol.json"),
        "capacity_protocol_sha256": core.sha256(stage / "capacity_protocol.json"),
        "cell_metrics_sha256": core.sha256(stage / "cell_metrics.csv"),
        "mechanism_counts_sha256": core.sha256(stage / "mechanism_counts.csv"),
        "figure_checks": figure_checks,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "paper_ready": False,
        "selection": "S6",
        "mode": audit["mode"],
        "protocol": "protocol.json",
        "protocol_sha256": audit["protocol_sha256"],
        "capacity_protocol": "capacity_protocol.json",
        "capacity_protocol_sha256": audit["capacity_protocol_sha256"],
        "runner_sha256": core.sha256(Path(__file__)),
        "shared_prefix_implementation_sha256": core.sha256(ROOT / "active_hbm_v313_s6.py"),
        "execution_workers": workers,
        "audit": "audit.json",
        "audit_sha256": core.sha256(stage / "audit.json"),
        "cell_metrics": "cell_metrics.csv",
        "cell_summary": "cell_summary.jsonl",
        "mechanism_counts": "mechanism_counts.csv",
        "aggregate_ci": None if args.gate else "aggregate_ci.csv",
        "paired_ci": None if args.gate else "paired_ci.csv",
        "figure": None if args.gate else {
            "pdf": f"figures/{spec['figure_grammar']['filename_stem']}.pdf",
            "png": f"figures/{spec['figure_grammar']['filename_stem']}.png",
        },
        "invalidated_legacy_s6_preserved": [
            "results/s6_preliminary_v1",
            "results/s6_preliminary_v2",
            "results/s6_preliminary_v3",
        ],
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if status != "PASS":
        raise RuntimeError(f"S6 audit failed: {audit['failed_checks']}")
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text("".join(
        f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())
    ))
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

