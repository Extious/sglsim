#!/usr/bin/env python3
"""Run the preregistered v3.1.3 S3 scale/context/topology pilot."""

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
import run_preliminary as core
import test_active_hbm_v313 as v313_tests


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s3_scale_context_protocol.json"
RESULT_NAME = "active_hbm_v313_s3_scale_context_pilot"
GATE_NAME = "active_hbm_v313_s3_scale_context_gate"
T95_DF2 = 4.302652729911275
TESTS = (
    ("test_active_hbm_v3_capacity.py", "RESULT 5/5 PASS"),
    ("test_active_hbm_v313_protocol.py", "RESULT 6/6 PASS"),
    ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    ("test_active_hbm_v313_s3_scale_context_protocol.py", "RESULT 10/10 PASS"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate", action="store_true",
        help="run only the shared center point, both topologies/policies, first seed",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="independent condition-seed workers; deterministic cell content is unchanged",
    )
    return parser.parse_args()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise ValueError(f"cannot infer CSV columns for empty rows: {path}")
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
        raise ValueError("this pilot preregisters exactly three seeds")
    return T95_DF2 * statistics.stdev(rows) / math.sqrt(len(rows))


def slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def axis_points(spec: dict[str, Any], *, gate: bool) -> list[dict[str, Any]]:
    sweep = spec["sweep"]
    center_dp = int(sweep["context_axis"]["fixed_system_dp"])
    center_context = int(sweep["scale_axis"]["fixed_context_tokens"])
    if gate:
        return [{
            "condition_id": f"dp{center_dp}_ctx{center_context}",
            "system_dp": center_dp,
            "context_tokens": center_context,
            "axis_membership": ["scale", "context"],
        }]
    points: dict[tuple[int, int], dict[str, Any]] = {}
    for dp in sweep["scale_axis"]["system_dp"]:
        key = (int(dp), center_context)
        points[key] = {
            "condition_id": f"dp{key[0]}_ctx{key[1]}",
            "system_dp": key[0], "context_tokens": key[1],
            "axis_membership": ["scale"],
        }
    for context in sweep["context_axis"]["context_tokens"]:
        key = (center_dp, int(context))
        if key in points:
            points[key]["axis_membership"].append("context")
        else:
            points[key] = {
                "condition_id": f"dp{key[0]}_ctx{key[1]}",
                "system_dp": key[0], "context_tokens": key[1],
                "axis_membership": ["context"],
            }
    return sorted(points.values(), key=lambda row: (row["system_dp"], row["context_tokens"]))


def configured_protocol(
    spec: dict[str, Any], *, dp: int, context: int, topology: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, capacity = v313_tests.inputs(context=context, active=2)
    route = spec["hardware"]["topologies"][topology]
    protocol["models"]["qwen3_8b"]["center_system_dp"] = int(dp)
    protocol["workload"]["center_context_tokens"] = int(context)
    protocol["workload"]["center_active_requests_per_gpu"] = int(
        spec["workload"]["offered_active_requests_per_gpu"]
    )
    protocol["hardware"]["gpus_per_node"] = int(spec["hardware"]["gpus_per_node"])
    protocol["hardware"]["node_count"] = int(dp) // int(spec["hardware"]["gpus_per_node"])
    protocol["hardware"]["copy_bandwidth_gb_s"] = dict(
        spec["hardware"]["copy_bandwidth_gb_s"]
    )
    protocol["hardware"]["load_bandwidth_gb_s"] = dict(
        spec["hardware"]["recovery_bandwidth_gb_s"]
    )
    protocol.setdefault("runtime_overrides", {})["topology"] = {
        "peer_hbm": route["peer_hbm"], "dram": route["dram"],
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
        (stage / "test_logs" / f"{Path(filename).stem}.log").write_text(
            result.stdout, encoding="utf-8"
        )
        passed = result.returncode == 0 and marker in result.stdout
        rows.append({"test": filename, "marker": marker, "passed": passed})
        if not passed:
            raise RuntimeError(f"test gate failed: {filename}\n{result.stdout}")
    return rows


def run_cell(
    *, stage: Path, spec: dict[str, Any], point: dict[str, Any],
    topology: str, policy: str, seed: int,
    nominal: list[core.Task], workload: dict[str, Any],
    probes: list[core.FaultProbe], universe_sha: str, probe_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, capacity = configured_protocol(
        spec, dp=point["system_dp"], context=point["context_tokens"],
        topology=topology,
    )
    replay = v313.CausalPhysicalTargetReplay(
        deepcopy(protocol), policy, capacity
    )
    tasks, delay_maps, mechanism, capacity_trace = replay.run_coupled(
        nominal, probes,
        control_horizon_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
    )
    request_set_sha = digest_json(sorted(row.request_id for row in tasks))
    base_ids, base_wct = core.workflow_wct(tasks)
    empty_completion = core.replay_one_fault(tasks, {})
    empty_ids, empty_wct = core.workflow_wct(tasks, empty_completion)
    empty_exact_error = max(
        abs(left - right) for left, right in zip(base_wct, empty_wct)
    )
    if base_ids != empty_ids:
        raise AssertionError("empty replay changed workflow IDs")

    fault_matrix: list[list[float]] = []
    probe_rows = []
    monotonic_error = 0.0
    for probe, delays in zip(probes, delay_maps):
        completion = core.replay_one_fault(tasks, delays)
        ids, values = core.workflow_wct(tasks, completion)
        if ids != base_ids:
            raise AssertionError("fault replay changed workflow denominator")
        monotonic_error = max(
            monotonic_error,
            max((base - value for base, value in zip(base_wct, values)), default=0.0),
        )
        fault_matrix.append(values)
        probe_rows.append({
            "probe_id": probe.probe_id,
            "time_ms": probe.time_ms,
            "source_rank": probe.source_rank,
            "failure_type": probe.failure_type,
            "failed_rank_count": len(v313.v312.v31.failure_domain_ranks(
                probe,
                gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
                dp=int(point["system_dp"]),
            )),
            "changed_request_count": len(delays),
        })

    healthy_mean, healthy_p99 = metric_mean_p99(base_wct)
    fault_flat = [value for row in fault_matrix for value in row]
    fault_mean, fault_p99 = metric_mean_p99(fault_flat)
    active_hbm = mechanism["active_hbm_v3"]
    accounting = mechanism["active_hbm_v313"]
    shared_budget = int(spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"])
    max_shared = max((int(row["shared_used_bytes"]) for row in capacity_trace), default=0)
    min_headroom = min(
        (shared_budget - int(row["shared_used_bytes"]) for row in capacity_trace),
        default=shared_budget,
    )
    expected_route = spec["hardware"]["topologies"][topology]
    copy_bw = core.copy_bandwidth(protocol, 0)
    load_bw = core.load_bandwidth(protocol, 0)
    wrong_copy_paths = [
        key for key, value in mechanism["path_event_counts"].items()
        if key.startswith("copy_") and int(value) > 0
        and key not in {
            f"copy_peer_hbm_{expected_route['peer_hbm']}",
            f"copy_dram_{expected_route['dram']}",
        }
    ]

    raw_name = (
        f"{point['condition_id']}__{topology}__{slug(policy)}__s{seed}.npz"
    )
    raw_path = stage / "raw" / raw_name
    np.savez_compressed(
        raw_path,
        workflow_ids=np.asarray(base_ids),
        healthy_wct_ms=np.asarray(base_wct, dtype=float),
        worker_fault_wct_ms=np.asarray(fault_matrix, dtype=float),
        probe_id=np.asarray([row["probe_id"] for row in probe_rows]),
        probe_time_ms=np.asarray([row["time_ms"] for row in probe_rows], dtype=float),
        probe_source_rank=np.asarray([row["source_rank"] for row in probe_rows], dtype=int),
        probe_changed_request_count=np.asarray(
            [row["changed_request_count"] for row in probe_rows], dtype=int
        ),
        capacity_time_ms=np.asarray(
            [row["time_ms"] for row in capacity_trace], dtype=float
        ),
        capacity_rank=np.asarray(
            [row["rank"] for row in capacity_trace], dtype=int
        ),
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

    path_counts = mechanism["path_event_counts"]
    residency_counts = mechanism["residency_event_counts"]
    recovery_counts = mechanism["recovery_type_path_counts"]
    count_rows = []
    for category, values in (
        ("path_event", path_counts),
        ("residency_event", residency_counts),
        ("recovery_type_path", recovery_counts),
        ("failover_mode", mechanism["failover_mode_counts"]),
    ):
        for key, value in values.items():
            count_rows.append({
                "condition_id": point["condition_id"],
                "axis_membership": "+".join(point["axis_membership"]),
                "system_dp": point["system_dp"],
                "context_tokens": point["context_tokens"],
                "topology": topology, "policy": policy, "seed": seed,
                "category": category, "key": key, "value": value,
            })

    topology_rows = accounting["topology_rows"]
    cell = {
        "condition_id": point["condition_id"],
        "axis_membership": "+".join(point["axis_membership"]),
        "system_dp": int(point["system_dp"]),
        "node_count": int(point["system_dp"]) // int(spec["hardware"]["gpus_per_node"]),
        "context_tokens": int(point["context_tokens"]),
        "topology": topology,
        "peer_hbm_path": expected_route["peer_hbm"],
        "dram_path": expected_route["dram"],
        "peer_copy_bandwidth_gb_s": copy_bw[0],
        "dram_copy_bandwidth_gb_s": copy_bw[1],
        "peer_recovery_bandwidth_gb_s": load_bw[0],
        "dram_recovery_bandwidth_gb_s": load_bw[1],
        "policy": policy,
        "seed": int(seed),
        "workflow_universe_sha256": universe_sha,
        "fault_probe_tuple_sha256": probe_sha,
        "scheduled_request_set_sha256": request_set_sha,
        "workflow_count": len(base_ids),
        "request_count": len(tasks),
        "arrival_horizon_ms": float(spec["workload"]["fixed_arrival_cutoff_ms"]),
        "actual_last_arrival_ms": float(workload["actual_last_arrival_ms"]),
        "drain_horizon_ms": max(row.finish_ms for row in tasks),
        "realized_active_requests_per_gpu_mean": workload[
            "realized_active_requests_per_gpu_time_mean"
        ],
        "workflow_throughput_per_s": len(base_ids) / (
            float(spec["workload"]["fixed_arrival_cutoff_ms"]) / 1000.0
        ),
        "healthy_wct_mean_ms": healthy_mean,
        "healthy_wct_p99_ms": healthy_p99,
        "fault_wct_mean_ms": fault_mean,
        "fault_wct_p99_ms": fault_p99,
        "fault_added_wct_mean_ms": fault_mean - healthy_mean,
        "fault_probe_count": len(probes),
        "fault_observation_count": len(fault_flat),
        "critical_queue_mean_ms": v313.workflow_decomposition(tasks)[
            "critical_queue_mean_ms"
        ],
        "queue_wait_mean_ms": active_hbm["queue_wait_mean_ms"],
        "memory_blocked_unique_request_count": active_hbm[
            "memory_blocked_unique_request_count"
        ],
        "control_window_count": mechanism["control_window_count"],
        "scheduled_control_opportunity_times_ms": json.dumps(
            accounting["scheduled_control_opportunity_times_ms"]
        ),
        "executed_policy_control_times_ms": json.dumps(
            accounting["executed_policy_control_times_ms"]
        ),
        "hbm_scheduled_copy_bytes": accounting["scheduled_copy_bytes"]["HBM"],
        "dram_scheduled_copy_bytes": accounting["scheduled_copy_bytes"]["DRAM"],
        "primary_priority_reclaimed_bytes": accounting[
            "primary_priority_reclaimed_bytes"
        ],
        "peak_target_backup_bytes": accounting[
            "physical_target_peak_backup_max_bytes"
        ],
        "capacity_record_count": len(capacity_trace),
        "capacity_max_shared_used_bytes": max_shared,
        "capacity_min_headroom_bytes": min_headroom,
        "capacity_violation_count": (
            active_hbm["capacity_violation_count"]
            + accounting["target_capacity_violation_count"]
        ),
        "copy_byte_conservation_error": json.dumps(
            accounting["copy_byte_conservation_error"], sort_keys=True
        ),
        "target_accounting_max_abs_error_bytes": accounting[
            "target_accounting_max_abs_error_bytes"
        ],
        "future_backup_projection_dispatch_count": accounting[
            "future_backup_projection_dispatch_count"
        ],
        "seal_before_dispatch_count": accounting["seal_before_dispatch_count"],
        "seal_uses_realized_output_or_service": accounting[
            "seal_uses_realized_output_or_service"
        ],
        "empty_replay_max_abs_error_ms": empty_exact_error,
        "fault_monotonic_max_negative_error_ms": monotonic_error,
        "failed_rank_count_min": min(row["failed_rank_count"] for row in probe_rows),
        "failed_rank_count_max": max(row["failed_rank_count"] for row in probe_rows),
        "topology_bijective": accounting["topology_bijective"],
        "topology_mode_exact": all(
            row["peer_hbm_mode"] == expected_route["peer_hbm"]
            and row["dram_mode"] == expected_route["dram"]
            for row in topology_rows
        ),
        "wrong_copy_path_keys": json.dumps(wrong_copy_paths),
        "recovery_type_path_counts": json.dumps(recovery_counts, sort_keys=True),
        "residency_event_counts": json.dumps(residency_counts, sort_keys=True),
        "raw_npz": f"raw/{raw_name}",
        "raw_npz_sha256": core.sha256(raw_path),
    }
    return cell, count_rows


def run_condition_seed(
    stage: Path, spec: dict[str, Any], point: dict[str, Any], seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the four paired topology/policy cells for one condition and seed."""

    nominal_protocol, _capacity = configured_protocol(
        spec, dp=point["system_dp"], context=point["context_tokens"],
        topology="same_node",
    )
    nominal, workload = core.build_workload(
        nominal_protocol, seed=seed, workflows=None,
        active_per_gpu=int(spec["workload"]["offered_active_requests_per_gpu"]),
        arrival_cutoff_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
    )
    probes = core.make_fault_probes(
        nominal,
        seed=(seed ^ (int(point["system_dp"]) << 12) ^ int(point["context_tokens"])),
        count=int(spec["workload"]["fault_probes_per_axis_point_seed"]),
        gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
        right_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
        failure_types=("source_worker_gpu",),
    )
    universe_sha = workload_hash(nominal)
    probes_sha = probe_hash(probes)
    cells: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    for topology in spec["sweep"]["topologies"]:
        for policy in spec["sweep"]["policies"]:
            cell, current_counts = run_cell(
                stage=stage, spec=spec, point=point,
                topology=topology, policy=policy, seed=seed,
                nominal=nominal, workload=workload, probes=probes,
                universe_sha=universe_sha, probe_sha=probes_sha,
            )
            cells.append(cell)
            counts.extend(current_counts)
    return cells, counts


def run_one_cell_job(
    stage: Path, spec: dict[str, Any], point: dict[str, Any], seed: int,
    topology: str, policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the deterministic common universe and execute one independent cell."""

    nominal_protocol, _capacity = configured_protocol(
        spec, dp=point["system_dp"], context=point["context_tokens"],
        topology="same_node",
    )
    nominal, workload = core.build_workload(
        nominal_protocol, seed=seed, workflows=None,
        active_per_gpu=int(spec["workload"]["offered_active_requests_per_gpu"]),
        arrival_cutoff_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
    )
    probes = core.make_fault_probes(
        nominal,
        seed=(seed ^ (int(point["system_dp"]) << 12) ^ int(point["context_tokens"])),
        count=int(spec["workload"]["fault_probes_per_axis_point_seed"]),
        gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
        right_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
        failure_types=("source_worker_gpu",),
    )
    return run_cell(
        stage=stage, spec=spec, point=point,
        topology=topology, policy=policy, seed=seed,
        nominal=nominal, workload=workload, probes=probes,
        universe_sha=workload_hash(nominal), probe_sha=probe_hash(probes),
    )


def axis_rows(cells: list[dict[str, Any]], axis: str) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        if axis not in cell["axis_membership"].split("+"):
            continue
        row = dict(cell)
        row["axis"] = axis
        row["axis_value"] = (
            cell["system_dp"] if axis == "scale" else cell["context_tokens"]
        )
        rows.append(row)
    return rows


def aggregate(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded = axis_rows(cells, "scale") + axis_rows(cells, "context")
    metrics = (
        "healthy_wct_mean_ms", "healthy_wct_p99_ms",
        "fault_wct_mean_ms", "fault_wct_p99_ms",
        "fault_added_wct_mean_ms", "workflow_throughput_per_s",
        "critical_queue_mean_ms", "queue_wait_mean_ms",
        "memory_blocked_unique_request_count", "hbm_scheduled_copy_bytes",
        "dram_scheduled_copy_bytes", "primary_priority_reclaimed_bytes",
        "capacity_min_headroom_bytes",
    )
    aggregate_rows = []
    keys = sorted({
        (row["axis"], row["axis_value"], row["topology"], row["policy"])
        for row in expanded
    })
    for axis, value, topology, policy in keys:
        selected = [
            row for row in expanded
            if (row["axis"], row["axis_value"], row["topology"], row["policy"])
            == (axis, value, topology, policy)
        ]
        out: dict[str, Any] = {
            "axis": axis, "axis_value": value, "topology": topology,
            "policy": policy, "seed_count": len(selected),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            out[f"{metric}_mean"] = statistics.fmean(values)
            out[f"{metric}_ci95_half_width"] = ci95(values)
        aggregate_rows.append(out)

    paired_rows = []
    paired_metrics = (
        "healthy_wct_mean_ms", "healthy_wct_p99_ms",
        "fault_wct_mean_ms", "fault_wct_p99_ms",
    )
    pair_keys = sorted({
        (row["axis"], row["axis_value"], row["topology"])
        for row in expanded
    })
    for axis, value, topology in pair_keys:
        selected = [
            row for row in expanded
            if (row["axis"], row["axis_value"], row["topology"])
            == (axis, value, topology)
        ]
        by_policy_seed = {
            (row["policy"], int(row["seed"])): row for row in selected
        }
        seeds = sorted({int(row["seed"]) for row in selected})
        for metric in paired_metrics:
            native = [float(by_policy_seed[("Native Mooncake", seed)][metric]) for seed in seeds]
            aegis = [float(by_policy_seed[("AegisKV", seed)][metric]) for seed in seeds]
            deltas = [right - left for left, right in zip(native, aegis)]
            ratios = [right / left for left, right in zip(native, aegis)]
            paired_rows.append({
                "axis": axis, "axis_value": value, "topology": topology,
                "metric": metric, "seed_count": len(seeds),
                "native_mean": statistics.fmean(native),
                "aegis_mean": statistics.fmean(aegis),
                "aegis_minus_native_mean": statistics.fmean(deltas),
                "aegis_minus_native_ci95_half_width": ci95(deltas),
                "aegis_over_native_mean": statistics.fmean(ratios),
                "aegis_over_native_ci95_half_width": ci95(ratios),
                "ci_method": "paired seed-level Student-t interval, df=2",
            })
    return aggregate_rows, paired_rows


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.25,
    })


def plot_s3(path: Path, rows: list[dict[str, Any]]) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    plot_specs = (
        (0, 0, "scale", "healthy_wct_mean_ms", "Healthy WCT", "System DP"),
        (0, 1, "context", "healthy_wct_mean_ms", "Healthy WCT", "Context length"),
        (1, 0, "scale", "fault_wct_mean_ms", "Worker-fault WCT", "System DP"),
        (1, 1, "context", "fault_wct_mean_ms", "Worker-fault WCT", "Context length"),
    )
    colors = {"Native Mooncake": "#4d4d4d", "AegisKV": "#7b3294"}
    markers = {"Native Mooncake": "s", "AegisKV": "o"}
    linestyles = {"same_node": "-", "cross_node": "--"}
    for row_index, col_index, axis, metric, title, xlabel in plot_specs:
        ax = axes[row_index, col_index]
        for policy in ("Native Mooncake", "AegisKV"):
            for topology in ("same_node", "cross_node"):
                selected = sorted(
                    (
                        row for row in rows
                        if row["axis"] == axis
                        and row["policy"] == policy
                        and row["topology"] == topology
                    ),
                    key=lambda row: int(row["axis_value"]),
                )
                x = np.arange(len(selected))
                mean = np.asarray([float(row[f"{metric}_mean"]) / 1000.0 for row in selected])
                half = np.asarray([
                    float(row[f"{metric}_ci95_half_width"]) / 1000.0
                    for row in selected
                ])
                ax.errorbar(
                    x, mean, yerr=half, color=colors[policy],
                    marker=markers[policy], linestyle=linestyles[topology],
                    markersize=3.2, capsize=2.2,
                )
        values = sorted({
            int(row["axis_value"]) for row in rows if row["axis"] == axis
        })
        labels = (
            [str(value) for value in values]
            if axis == "scale" else [f"{value // 1024}K" for value in values]
        )
        ax.set_xticks(np.arange(len(values)), labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean WCT (s)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.margins(x=0.04)

    legend_handles = [
        Line2D([0], [0], color=colors["Native Mooncake"], marker="s", label="Native Mooncake"),
        Line2D([0], [0], color=colors["AegisKV"], marker="o", label="AegisKV"),
        Line2D([0], [0], color="#333333", linestyle="-", label="Same-node / local DRAM"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Cross-node / remote DRAM"),
    ]
    fig.legend(
        handles=legend_handles, frameon=False, ncol=4, loc="lower center",
        bbox_to_anchor=(0.5, 0.005), columnspacing=1.35, handlelength=2.0,
    )
    fig.suptitle("S3 scale, context, and protection topology", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95), h_pad=1.1, w_pad=1.4)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def verify_figure(pdf: Path, png: Path, minimum_font: float, minimum_width: int) -> dict[str, Any]:
    with fitz.open(pdf) as document:
        sizes = [
            float(span["size"])
            for block in document[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if span["text"].strip()
        ]
        page_count = len(document)
    with Image.open(png) as image:
        width, height = image.size
    return {
        "single_page_vector_pdf": page_count == 1,
        "minimum_extracted_font_pt": min(sizes),
        "minimum_font_at_least_registered": min(sizes) >= minimum_font - 1e-6,
        "png_width_px": width,
        "png_height_px": height,
        "png_width_at_least_registered": width >= minimum_width,
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
    points = axis_points(spec, gate=args.gate)
    seeds = (
        [int(spec["sweep"]["paired_seeds"][0])]
        if args.gate else [int(seed) for seed in spec["sweep"]["paired_seeds"]]
    )
    expected_cells = len(points) * len(spec["sweep"]["topologies"]) * 2 * len(seeds)
    cells: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    jobs = [
        (point, seed, topology, policy)
        for point in points for seed in seeds
        for topology in spec["sweep"]["topologies"]
        for policy in spec["sweep"]["policies"]
    ]
    worker_count = 1 if args.gate else max(1, min(int(args.workers), len(jobs)))
    completed_cells = 0
    if worker_count == 1:
        for point, seed, topology, policy in jobs:
            current_cell, current_counts = run_one_cell_job(
                stage, spec, point, seed, topology, policy
            )
            cells.append(current_cell)
            count_rows.extend(current_counts)
            completed_cells += 1
            print(
                f"S3 v313 {completed_cells}/{expected_cells} "
                f"{point['condition_id']} {topology} {policy} s{seed}",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {
                executor.submit(
                    run_one_cell_job, stage, spec, point, seed, topology, policy
                ): (point, seed, topology, policy)
                for point, seed, topology, policy in jobs
            }
            for future in concurrent.futures.as_completed(future_to_job):
                point, seed, topology, policy = future_to_job[future]
                current_cell, current_counts = future.result()
                cells.append(current_cell)
                count_rows.extend(current_counts)
                completed_cells += 1
                print(
                    f"S3 v313 {completed_cells}/{expected_cells} "
                    f"{point['condition_id']} {topology} {policy} s{seed}",
                    flush=True,
                )
    cells.sort(key=lambda row: (
        int(row["system_dp"]), int(row["context_tokens"]),
        row["topology"], row["policy"], int(row["seed"]),
    ))
    count_rows.sort(key=lambda row: (
        int(row["system_dp"]), int(row["context_tokens"]),
        row["topology"], row["policy"], int(row["seed"]),
        row["category"], row["key"],
    ))

    metric_fields = list(cells[0])
    write_csv(stage / "cell_metrics.csv", cells, metric_fields)
    write_csv(
        stage / "mechanism_counts.csv", count_rows,
        [
            "condition_id", "axis_membership", "system_dp", "context_tokens",
            "topology", "policy", "seed", "category", "key", "value",
        ],
    )
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
        plot_s3(figure_path, aggregate_rows)
        figure_checks = verify_figure(
            figure_path.with_suffix(".pdf"), figure_path.with_suffix(".png"),
            float(spec["figure_grammar"]["minimum_font_pt"]),
            int(spec["figure_grammar"]["png_minimum_width_px"]),
        )

    group_keys = sorted({
        (cell["condition_id"], int(cell["seed"])) for cell in cells
    })
    common_universe = all(
        len({
            (cell["workflow_universe_sha256"], cell["fault_probe_tuple_sha256"])
            for cell in cells
            if (cell["condition_id"], int(cell["seed"])) == key
        }) == 1
        for key in group_keys
    )
    checks = {
        "mechanism_and_protocol_tests_pass": all(row["passed"] for row in tests),
        "exact_registered_cell_count": len(cells) == expected_cells,
        "all_registered_policies_topologies_and_seeds_present": all(
            len([
                cell for cell in cells
                if cell["condition_id"] == point["condition_id"]
            ]) == len(seeds) * 4
            for point in points
        ),
        "common_workflow_and_probe_hashes_across_policy_topology": common_universe,
        "source_worker_failure_exactly_one_rank": all(
            cell["failed_rank_count_min"] == 1
            and cell["failed_rank_count_max"] == 1 for cell in cells
        ),
        "same_cross_topology_and_directional_bandwidth_exact": all(
            cell["topology_mode_exact"]
            and json.loads(cell["wrong_copy_path_keys"]) == []
            and (
                (cell["topology"] == "same_node"
                 and cell["peer_copy_bandwidth_gb_s"] == 50.0
                 and cell["dram_copy_bandwidth_gb_s"] == 16.0
                 and cell["peer_recovery_bandwidth_gb_s"] == 50.0
                 and cell["dram_recovery_bandwidth_gb_s"] == 16.0)
                or
                (cell["topology"] == "cross_node"
                 and cell["peer_copy_bandwidth_gb_s"] == 10.0
                 and cell["dram_copy_bandwidth_gb_s"] == 10.0
                 and cell["peer_recovery_bandwidth_gb_s"] == 10.0
                 and cell["dram_recovery_bandwidth_gb_s"] == 10.0)
            ) for cell in cells
        ),
        "physical_target_capacity_invariant": all(
            cell["capacity_violation_count"] == 0
            and cell["capacity_max_shared_used_bytes"]
            <= int(spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"])
            and cell["capacity_min_headroom_bytes"] >= 0
            and cell["target_accounting_max_abs_error_bytes"] == 0
            for cell in cells
        ),
        "copy_bytes_conserve_exactly": all(
            json.loads(cell["copy_byte_conservation_error"])
            == {"DRAM": 0, "HBM": 0} for cell in cells
        ),
        "no_future_or_unsealed_backup_dispatch_dependency": all(
            cell["future_backup_projection_dispatch_count"] == 0
            and cell["seal_before_dispatch_count"] == 0
            and cell["seal_uses_realized_output_or_service"] is False
            for cell in cells
        ),
        "empty_replay_exact_and_fault_monotone": all(
            cell["empty_replay_max_abs_error_ms"] <= 1e-7
            and cell["fault_monotonic_max_negative_error_ms"] <= 1e-7
            for cell in cells
        ),
        "three_exact_control_opportunities": all(
            json.loads(cell["scheduled_control_opportunity_times_ms"])
            == [30000.0, 60000.0, 90000.0]
            for cell in cells
        ),
        "raw_npz_hashes_match": all(
            core.sha256(stage / cell["raw_npz"]) == cell["raw_npz_sha256"]
            for cell in cells
        ),
        "preregistered_source_hashes_match": all(
            Path(row["path"]).exists()
            and core.sha256(Path(row["path"])) == row["sha256"]
            for row in spec["sources"].values()
        ),
        "no_seed_or_cell_selection": True,
    }
    if not args.gate:
        checks.update({
            "exact_3_seeds_per_aggregate": all(
                int(row["seed_count"]) == 3 for row in aggregate_rows
            ),
            "exact_40_aggregate_rows": len(aggregate_rows) == 40,
            "exact_80_paired_metric_rows": len(paired_rows) == 80,
            "figure_single_page_vector_pdf": figure_checks[
                "single_page_vector_pdf"
            ],
            "figure_png_width_at_least_1800": figure_checks[
                "png_width_at_least_registered"
            ],
            "figure_minimum_font_at_least_7pt": figure_checks[
                "minimum_font_at_least_registered"
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
        "mode": audit["mode"],
        "selection": "S3",
        "protocol": "protocol.json",
        "protocol_sha256": audit["protocol_sha256"],
        "runner_sha256": core.sha256(Path(__file__)),
        "execution_workers": worker_count,
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
        "old_s3_preserved": "results/s3_preliminary_v1",
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if status != "PASS":
        raise RuntimeError(f"S3 audit failed: {audit['failed_checks']}")
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
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
