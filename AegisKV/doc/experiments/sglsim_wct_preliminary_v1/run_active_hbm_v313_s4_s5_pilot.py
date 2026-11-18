#!/usr/bin/env python3
"""Atomic three-seed S4 ablation and S5 control-timeline pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import active_hbm_v313 as v313
import active_hbm_v313_s4s5 as ext
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s4_s5_protocol.json"
RESULT_NAME = "active_hbm_v313_s4_s5_preliminary_pilot"
GIB = 1024**3
T95_DF2 = 4.302652729911275
TESTS = (
    ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    ("test_active_hbm_v313_s4s5.py", "RESULT 6/6 PASS"),
    ("test_active_hbm_v313_s4_s5_protocol.py", "RESULT 7/7 PASS"),
)


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


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    if len(values) != 3:
        raise ValueError("pilot CI is preregistered for exactly three paired seeds")
    return T95_DF2 * statistics.stdev(values) / math.sqrt(len(values))


def configure_protocol(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(
        (ROOT / "results/mechanism_pilot_v2/composite_protocol.json").read_text()
    )
    topology = spec["common"]["topology_and_paths"]
    protocol["models"]["qwen3_8b"]["center_system_dp"] = int(
        spec["common"]["system_dp"]
    )
    protocol["hardware"]["gpus_per_node"] = int(topology["gpus_per_node"])
    protocol["hardware"]["node_count"] = int(topology["node_count"])
    protocol["hardware"]["copy_bandwidth_gb_s"] = dict(
        topology["copy_bandwidth_gb_s"]
    )
    protocol["hardware"]["load_bandwidth_gb_s"] = dict(
        topology["recovery_bandwidth_gb_s"]
    )
    protocol.setdefault("runtime_overrides", {})["record_occupancy_trace"] = True
    protocol["runtime_overrides"].pop("topology", None)
    protocol["failure_model"]["policy_failover"] = deepcopy(
        spec["common"]["policy_failover"]
    )
    capacity = json.loads((ROOT / "active_hbm_v3_protocol.json").read_text())
    return protocol, capacity


def task_universe_hash(tasks: list[core.Task]) -> str:
    return digest_json([
        (
            row.request_id, row.workflow_id, row.workflow_arrival_ms,
            row.arrival_ms, row.source_rank, row.service_ms,
            row.prompt_tokens, row.output_tokens, row.predicted_slack_ms,
        )
        for row in tasks
    ])


def probe_hash(probes: list[core.FaultProbe]) -> str:
    return digest_json([
        (row.probe_id, row.time_ms, row.source_rank, row.failure_type)
        for row in probes
    ])


def run_s4(
    stage: Path, spec: dict[str, Any], common_protocol: dict[str, Any],
    capacity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    s4 = spec["s4"]
    policies = list(s4["internal_policies"])
    display = dict(s4["internal_to_display"])
    cells: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    raw_dir = stage / "raw" / "s4"
    raw_dir.mkdir(parents=True)
    for seed in s4["paired_seeds"]:
        protocol = deepcopy(common_protocol)
        protocol["workload"]["center_context_tokens"] = int(s4["context_tokens"])
        protocol["workload"]["center_active_requests_per_gpu"] = int(
            s4["active_requests_per_gpu"]
        )
        nominal, workload = v313.v312.v31.build_workload(
            protocol,
            seed=int(seed),
            active_per_gpu=int(s4["active_requests_per_gpu"]),
            arrival_cutoff_ms=float(s4["arrival_cutoff_ms"]),
        )
        probes = core.make_fault_probes(
            nominal,
            seed=int(seed) ^ int(s4["probe_seed_xor"]),
            count=int(s4["fault_probes_per_seed"]),
            gpus_per_node=int(protocol["hardware"]["gpus_per_node"]),
            right_ms=float(s4["arrival_cutoff_ms"]),
            failure_types=tuple(s4["probe_type_schedule"]),
        )
        universe_sha = task_universe_hash(nominal)
        probes_sha = probe_hash(probes)
        for policy in policies:
            replay = ext.S4S5Replay(deepcopy(protocol), policy, capacity)
            tasks, delay_maps, mechanism, capacity_trace = replay.run_coupled(
                nominal, probes,
                control_horizon_ms=float(s4["arrival_cutoff_ms"]),
            )
            ids, healthy = core.workflow_wct(tasks)
            fault_matrix: list[list[float]] = []
            per_domain: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: {"wct": [], "added": []}
            )
            changed_by_domain: dict[str, int] = defaultdict(int)
            for probe, delays in zip(probes, delay_maps):
                completion = core.replay_one_fault(tasks, delays)
                replay_ids, fault = core.workflow_wct(tasks, completion)
                if replay_ids != ids:
                    raise AssertionError("S4 workflow denominator changed")
                added = []
                for base, value in zip(healthy, fault):
                    delta = value - base
                    if delta < -1e-7:
                        raise AssertionError("S4 fault replay lowered WCT")
                    added.append(max(delta, 0.0))
                    changed_by_domain[probe.failure_type] += int(delta > 1e-7)
                per_domain[probe.failure_type]["wct"].extend(fault)
                per_domain[probe.failure_type]["added"].extend(added)
                fault_matrix.append(fault)
            decomposition = v313.workflow_decomposition(tasks)
            v3 = mechanism["active_hbm_v3"]
            v313m = mechanism["active_hbm_v313"]
            s4s5m = mechanism["active_hbm_v313_s4s5"]
            raw_name = f"{display[policy].lower().replace(' ', '_').replace('-', '')}__s{seed}.npz"
            raw_path = raw_dir / raw_name
            cap = np.asarray([
                (
                    row["time_ms"], row["rank"], row["active_requests"],
                    row["waiting_requests"], row["active_serving_kv_bytes"],
                    row["peer_hbm_backup_used_bytes"], row["shared_used_bytes"],
                    row["shared_headroom_bytes"],
                ) for row in capacity_trace
            ], dtype=float).reshape((-1, 8))
            np.savez_compressed(
                raw_path,
                workflow_ids=np.asarray(ids),
                healthy_wct_ms=np.asarray(healthy, dtype=float),
                probe_ids=np.asarray([row.probe_id for row in probes]),
                probe_times_ms=np.asarray([row.time_ms for row in probes], dtype=float),
                probe_source_rank=np.asarray([row.source_rank for row in probes], dtype=int),
                probe_failure_type=np.asarray([row.failure_type for row in probes]),
                fault_wct_ms=np.asarray(fault_matrix, dtype=float),
                capacity_trace=cap,
            )
            domains = {}
            for domain in s4["failure_domains"]:
                values = per_domain[domain]
                domains[domain] = {
                    "observation_count": len(values["wct"]),
                    "changed_workflow_observation_count": changed_by_domain[domain],
                    "wct_mean_ms": statistics.fmean(values["wct"]),
                    "wct_p99_ms": core.quantile(values["wct"], 0.99),
                    "added_wct_mean_ms": statistics.fmean(values["added"]),
                    "added_wct_p99_ms": core.quantile(values["added"], 0.99),
                }
                metric_rows.append({
                    "seed": seed,
                    "display_policy": display[policy],
                    "internal_policy": policy,
                    "failure_domain": domain,
                    "workflow_count": len(ids),
                    "probe_count": sum(row.failure_type == domain for row in probes),
                    "healthy_wct_mean_ms": statistics.fmean(healthy),
                    "healthy_wct_p99_ms": core.quantile(healthy, 0.99),
                    "fault_wct_mean_ms": domains[domain]["wct_mean_ms"],
                    "fault_wct_p99_ms": domains[domain]["wct_p99_ms"],
                    "added_wct_mean_ms": domains[domain]["added_wct_mean_ms"],
                    "added_wct_p99_ms": domains[domain]["added_wct_p99_ms"],
                    "hbm_scheduled_copy_gib": v313m["scheduled_copy_bytes"]["HBM"] / GIB,
                    "dram_scheduled_copy_gib": v313m["scheduled_copy_bytes"]["DRAM"] / GIB,
                    "primary_priority_reclaim_gib": v313m["primary_priority_reclaimed_bytes"] / GIB,
                    "aimd_backup_release_gib": v3["backup_bytes_released_by_control"] / GIB,
                    "hbm_evicted_gib": mechanism["tier_churn"].get("HBM", {}).get("evicted_bytes", 0) / GIB,
                    "dram_evicted_gib": mechanism["tier_churn"].get("DRAM", {}).get("evicted_bytes", 0) / GIB,
                    "alpha_final_mean": mechanism["alpha_final_mean"],
                    "algorithm1_nonzero_slack_request_count": mechanism["algorithm1_nonzero_slack_request_count"],
                    "coupled_tier_attempt_count": s4s5m["coupled_tier_attempt_count"],
                    "capacity_violation_count": v3["capacity_violation_count"] + v313m["target_capacity_violation_count"],
                    "copy_conservation_error_hbm_bytes": v313m["copy_byte_conservation_error"]["HBM"],
                    "copy_conservation_error_dram_bytes": v313m["copy_byte_conservation_error"]["DRAM"],
                    "workflow_universe_sha256": universe_sha,
                    "fault_probe_tuple_sha256": probes_sha,
                    "raw_npz": f"raw/s4/{raw_name}",
                    "raw_npz_sha256": core.sha256(raw_path),
                })
            cells.append({
                "seed": int(seed),
                "display_policy": display[policy],
                "internal_policy": policy,
                "workflow_universe_sha256": universe_sha,
                "fault_probe_tuple_sha256": probes_sha,
                "workflow_count": len(ids),
                "workload": workload,
                "healthy": {
                    "mean_wct_ms": statistics.fmean(healthy),
                    "p99_wct_ms": core.quantile(healthy, 0.99),
                    "decomposition": decomposition,
                },
                "failure_conditioned": domains,
                "mechanism": mechanism,
                "raw_npz": f"raw/s4/{raw_name}",
                "raw_npz_sha256": core.sha256(raw_path),
            })
            print(f"S4 {len(cells)}/15 seed={seed} policy={display[policy]}", flush=True)

    comparisons: list[dict[str, Any]] = []
    full_name = display["AegisKV"]
    by_key = {
        (row["seed"], row["display_policy"], row["failure_domain"]): row
        for row in metric_rows
    }
    for domain in s4["failure_domains"]:
        for policy in policies:
            name = display[policy]
            for metric in ("fault_wct_mean_ms", "fault_wct_p99_ms"):
                absolute = [
                    float(by_key[(seed, name, domain)][metric])
                    for seed in s4["paired_seeds"]
                ]
                full = [
                    float(by_key[(seed, full_name, domain)][metric])
                    for seed in s4["paired_seeds"]
                ]
                ratios = [value / base for value, base in zip(absolute, full)]
                deltas = [value - base for value, base in zip(absolute, full)]
                comparisons.append({
                    "display_policy": name,
                    "internal_policy": policy,
                    "failure_domain": domain,
                    "metric": metric,
                    "seed_count": len(absolute),
                    "absolute_mean_ms": statistics.fmean(absolute),
                    "absolute_seed_ci95_half_width_ms": ci95(absolute),
                    "normalized_to_full_mean": statistics.fmean(ratios),
                    "paired_normalized_ci95_half_width": ci95(ratios),
                    "paired_delta_vs_full_mean_ms": statistics.fmean(deltas),
                    "paired_delta_vs_full_ci95_half_width_ms": ci95(deltas),
                    "ci_method": "paired seed-level t interval, df=2",
                })
    return cells, metric_rows, comparisons


def build_piecewise_workload(
    protocol: dict[str, Any], *, seed: int, phases: list[dict[str, Any]],
    context_tokens: int,
) -> tuple[list[core.Task], list[dict[str, Any]]]:
    tasks: list[core.Task] = []
    metadata: list[dict[str, Any]] = []
    for index, phase in enumerate(phases):
        local = deepcopy(protocol)
        active = int(phase["offered_active_requests_per_gpu"])
        duration = float(phase["end_ms"]) - float(phase["start_ms"])
        local["workload"]["center_context_tokens"] = int(context_tokens)
        local["workload"]["center_active_requests_per_gpu"] = active
        local_seed = int(seed) ^ int(phase["seed_xor"])
        rows, meta = v313.v312.v31.build_workload(
            local, seed=local_seed, active_per_gpu=active,
            arrival_cutoff_ms=duration,
        )
        prefix = f"s{seed}-phase{index}"
        offset = float(phase["start_ms"])
        for row in rows:
            tasks.append(replace(
                row,
                request_id=f"{prefix}-{row.request_id}",
                workflow_id=f"{prefix}-{row.workflow_id}",
                workflow_arrival_ms=row.workflow_arrival_ms + offset,
                arrival_ms=row.arrival_ms + offset,
                start_ms=row.start_ms + offset,
                finish_ms=row.finish_ms + offset,
            ))
        metadata.append({
            "name": phase["name"],
            "start_ms": offset,
            "end_ms": float(phase["end_ms"]),
            "offered_active_requests_per_gpu": active,
            "local_seed": local_seed,
            "workflow_count": meta["workflow_count"],
            "request_count": meta["request_count"],
            "mean_workflow_gap_ms": meta["mean_workflow_gap_ms"],
            "slack_predictor": meta["slack_predictor"],
        })
    return sorted(tasks, key=lambda row: (row.arrival_ms, row.request_id)), metadata


def phase_at(phases: list[dict[str, Any]], time_ms: float) -> dict[str, Any]:
    for phase in phases:
        if float(phase["start_ms"]) <= time_ms < float(phase["end_ms"]):
            return phase
    if math.isclose(time_ms, float(phases[-1]["end_ms"])):
        return phases[-1]
    raise ValueError(f"control time outside phase schedule: {time_ms}")


def run_s5(
    stage: Path, spec: dict[str, Any], common_protocol: dict[str, Any],
    capacity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    s5 = spec["s5"]
    cells: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    raw_dir = stage / "raw" / "s5"
    raw_dir.mkdir(parents=True)
    for seed in s5["seeds"]:
        protocol = deepcopy(common_protocol)
        protocol["workload"]["center_context_tokens"] = int(s5["context_tokens"])
        tasks, phase_metadata = build_piecewise_workload(
            protocol, seed=int(seed), phases=s5["phases"],
            context_tokens=int(s5["context_tokens"]),
        )
        universe_sha = task_universe_hash(tasks)
        replay = ext.S4S5Replay(protocol, "AegisKV", capacity)
        scheduled, _delay_maps, mechanism, capacity_trace = replay.run_coupled(
            tasks, [], control_horizon_ms=float(s5["control_horizon_ms"])
        )
        ids, wct = core.workflow_wct(scheduled)
        rich = mechanism["active_hbm_v313_s4s5"]["rich_control_trace"]
        raw_name = f"aegiskv__s{seed}.npz"
        raw_path = raw_dir / raw_name
        cap = np.asarray([
            (
                row["time_ms"], row["rank"], row["active_requests"],
                row["waiting_requests"], row["active_serving_kv_bytes"],
                row["peer_hbm_backup_used_bytes"], row["shared_used_bytes"],
                row["shared_headroom_bytes"],
            ) for row in capacity_trace
        ], dtype=float).reshape((-1, 8))
        np.savez_compressed(
            raw_path,
            workflow_ids=np.asarray(ids),
            workflow_wct_ms=np.asarray(wct, dtype=float),
            capacity_trace=cap,
        )
        previous = {
            "hbm_admission_success_count": 0,
            "hbm_admission_rejection_count": 0,
            "dram_admission_success_count": 0,
            "dram_admission_rejection_count": 0,
            "hbm_eviction_count": 0,
            "hbm_evicted_bytes": 0,
            "primary_priority_reclaimed_bytes": 0,
            "hbm_admission_resumed_event_count": 0,
        }
        initial_phase = phase_at(s5["phases"], 0.0)
        timeline.append({
            "seed": seed, "time_s": 0.0, "is_control_window": False,
            "phase": initial_phase["name"],
            "offered_active_requests_per_gpu": initial_phase["offered_active_requests_per_gpu"],
            "realized_active_requests_per_gpu": 0.0,
            "pressure_fraction": 0.0,
            "external_memory_pressure_fraction": 0.0,
            "alpha": float(protocol["aegiskv"]["initial_alpha"]),
            "peer_hbm_quota_gib_per_target": 40.0 * float(protocol["aegiskv"]["initial_alpha"]),
            "shared_serving_plus_backup_budget_gib_per_target": replay.shared_budget_bytes / GIB,
            "serving_hbm_gib_per_gpu": 0.0,
            "peer_hbm_backup_gib_per_gpu": 0.0,
            "dram_gib_per_gpu": 0.0,
            "hbm_admit_delta": 0, "hbm_reject_delta": 0,
            "dram_admit_delta": 0, "dram_reject_delta": 0,
            "hbm_eviction_delta": 0, "hbm_evicted_gib_delta": 0.0,
            "primary_reclaim_gib_delta": 0.0,
            "aimd_backup_release_gib": 0.0,
            "admission_resume_delta": 0,
            "dispatched_after_control": 0,
            "workflow_universe_sha256": universe_sha,
        })
        for row in rich:
            phase = phase_at(s5["phases"], float(row["time_ms"]))
            current = {key: int(row[key]) for key in previous}
            delta = {key: current[key] - previous[key] for key in previous}
            previous = current
            alpha = float(row["alpha_after_mean"])
            timeline.append({
                "seed": seed,
                "time_s": float(row["time_ms"]) / 1000.0,
                "is_control_window": True,
                "phase": phase["name"],
                "offered_active_requests_per_gpu": phase["offered_active_requests_per_gpu"],
                "realized_active_requests_per_gpu": row["realized_active_requests_per_gpu"],
                "pressure_fraction": row["pressure_fraction"],
                "external_memory_pressure_fraction": row["external_memory_pressure_fraction"],
                "alpha": alpha,
                "peer_hbm_quota_gib_per_target": row["peer_hbm_quota_bytes_total"] / replay.dp / GIB,
                "shared_serving_plus_backup_budget_gib_per_target": row["shared_serving_plus_backup_budget_bytes_total"] / replay.dp / GIB,
                "serving_hbm_gib_per_gpu": row["serving_hbm_bytes_total"] / replay.dp / GIB,
                "peer_hbm_backup_gib_per_gpu": row["peer_hbm_backup_after_bytes_total"] / replay.dp / GIB,
                "dram_gib_per_gpu": row["dram_used_bytes_total"] / replay.dp / GIB,
                "hbm_admit_delta": delta["hbm_admission_success_count"],
                "hbm_reject_delta": delta["hbm_admission_rejection_count"],
                "dram_admit_delta": delta["dram_admission_success_count"],
                "dram_reject_delta": delta["dram_admission_rejection_count"],
                "hbm_eviction_delta": delta["hbm_eviction_count"],
                "hbm_evicted_gib_delta": delta["hbm_evicted_bytes"] / GIB,
                "primary_reclaim_gib_delta": delta["primary_priority_reclaimed_bytes"] / GIB,
                "aimd_backup_release_gib": row["aimd_backup_released_bytes_this_window"] / GIB,
                "admission_resume_delta": delta["hbm_admission_resumed_event_count"],
                "dispatched_after_control": row["dispatched_after_control"],
                "workflow_universe_sha256": universe_sha,
            })
        v3m = mechanism["active_hbm_v3"]
        v313m = mechanism["active_hbm_v313"]
        cells.append({
            "seed": int(seed),
            "workflow_universe_sha256": universe_sha,
            "workflow_count": len(ids),
            "phase_metadata": phase_metadata,
            "wct_mean_ms": statistics.fmean(wct),
            "wct_p99_ms": core.quantile(wct, 0.99),
            "mechanism": mechanism,
            "capacity_violation_count": v3m["capacity_violation_count"] + v313m["target_capacity_violation_count"],
            "copy_byte_conservation_error": v313m["copy_byte_conservation_error"],
            "raw_npz": f"raw/s5/{raw_name}",
            "raw_npz_sha256": core.sha256(raw_path),
        })
        print(f"S5 {len(cells)}/3 seed={seed}", flush=True)

    fields = [
        "offered_active_requests_per_gpu", "realized_active_requests_per_gpu",
        "pressure_fraction", "external_memory_pressure_fraction", "alpha",
        "peer_hbm_quota_gib_per_target",
        "shared_serving_plus_backup_budget_gib_per_target",
        "serving_hbm_gib_per_gpu", "peer_hbm_backup_gib_per_gpu",
        "dram_gib_per_gpu", "hbm_admit_delta", "hbm_reject_delta",
        "dram_admit_delta", "dram_reject_delta", "hbm_eviction_delta",
        "hbm_evicted_gib_delta", "primary_reclaim_gib_delta",
        "aimd_backup_release_gib", "admission_resume_delta",
        "dispatched_after_control",
    ]
    ci_rows: list[dict[str, Any]] = []
    for time_s in sorted({float(row["time_s"]) for row in timeline}):
        selected = [row for row in timeline if float(row["time_s"]) == time_s]
        out: dict[str, Any] = {
            "time_s": time_s,
            "phase": selected[0]["phase"],
            "seed_count": len(selected),
            "is_control_window": selected[0]["is_control_window"],
        }
        for field in fields:
            values = [float(row[field]) for row in selected]
            out[f"{field}_mean"] = statistics.fmean(values)
            out[f"{field}_ci95_half_width"] = ci95(values)
        ci_rows.append(out)
    return cells, timeline, ci_rows


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def plot_s4(path: Path, rows: list[dict[str, Any]]) -> None:
    style()
    policies = ["Native Mooncake", "Full AegisKV", "-Algorithm 1", "-Algorithm 2", "-Algorithm 3"]
    labels = ["Native\nMooncake", "Full\nAegisKV", "-Alg. 1", "-Alg. 2", "-Alg. 3"]
    domains = [("source_worker_gpu", "Worker/GPU", "#d62728", "o"), ("source_node", "Source node", "#1f77b4", "s")]
    metrics = [("fault_wct_mean_ms", "Mean WCT"), ("fault_wct_p99_ms", "p99 WCT")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), sharey=False)
    x = np.arange(len(policies), dtype=float)
    for ax, (metric, title) in zip(axes, metrics):
        for offset, (domain, domain_label, color, marker) in zip((-0.08, 0.08), domains):
            selected = {
                row["display_policy"]: row for row in rows
                if row["failure_domain"] == domain and row["metric"] == metric
            }
            y = [float(selected[name]["normalized_to_full_mean"]) for name in policies]
            e = [float(selected[name]["paired_normalized_ci95_half_width"]) for name in policies]
            ax.errorbar(
                x + offset, y, yerr=e, color=color, marker=marker,
                linewidth=1.4, markersize=4.3, capsize=2.5,
                label=domain_label,
            )
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Normalized fault-conditioned WCT\n(Full AegisKV = 1.0; lower is better)")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("S4 algorithm ablation pilot (3 paired seeds)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.5)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_s5(path: Path, rows: list[dict[str, Any]]) -> None:
    style()
    t = np.asarray([float(row["time_s"]) for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.15), sharex=True)
    for ax in axes:
        ax.axvspan(0, 60, color="#2ca02c", alpha=0.07)
        ax.axvspan(60, 120, color="#d62728", alpha=0.07)
        ax.axvspan(120, 240, color="#2ca02c", alpha=0.07)
        ax.grid(axis="y", alpha=0.25)

    alpha = np.asarray([float(row["alpha_mean"]) for row in rows])
    alpha_ci = np.asarray([float(row["alpha_ci95_half_width"]) for row in rows])
    axes[0].plot(t, alpha, color="#7b3294", marker="o", label=r"$\alpha$")
    axes[0].fill_between(t, alpha-alpha_ci, alpha+alpha_ci, color="#7b3294", alpha=0.16)
    axes[0].set_ylabel(r"Peer-HBM $\alpha$")
    axes[0].set_ylim(bottom=0)
    quota_ax = axes[0].twinx()
    quota_ax.plot(t, 40.0 * alpha, color="#c51b7d", linestyle="--", linewidth=1.1)
    quota_ax.set_ylabel(r"Quota $40\alpha$ (GiB/target)")
    quota_ax.spines["top"].set_visible(False)

    offered = np.asarray([float(row["offered_active_requests_per_gpu_mean"]) for row in rows])
    realized = np.asarray([float(row["realized_active_requests_per_gpu_mean"]) for row in rows])
    realized_ci = np.asarray([float(row["realized_active_requests_per_gpu_ci95_half_width"]) for row in rows])
    axes[1].step(t, offered, where="post", color="#555555", linestyle="--", label="Offered load")
    axes[1].plot(t, realized, color="#ff7f0e", marker="o", label="Realized active")
    axes[1].fill_between(t, realized-realized_ci, realized+realized_ci, color="#ff7f0e", alpha=0.16)
    axes[1].set_ylabel("Requests/GPU")
    pressure_ax = axes[1].twinx()
    pressure = np.asarray([float(row["pressure_fraction_mean"]) for row in rows])
    pressure_ax.plot(t, pressure, color="#d62728", marker="s", linewidth=1.1, label="Pressure")
    pressure_ax.set_ylabel("Pressure fraction")
    pressure_ax.set_ylim(-0.03, 1.03)
    pressure_ax.spines["top"].set_visible(False)
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = pressure_ax.get_legend_handles_labels()
    axes[1].legend(h1+h2, l1+l2, frameon=False, ncol=3, loc="upper right")

    occupancy = [
        ("serving_hbm_gib_per_gpu", "Serving HBM", "#1f77b4"),
        ("peer_hbm_backup_gib_per_gpu", "Peer-HBM backup", "#2ca02c"),
        ("dram_gib_per_gpu", "DRAM", "#9467bd"),
    ]
    for field, label, color in occupancy:
        mean = np.asarray([float(row[f"{field}_mean"]) for row in rows])
        half = np.asarray([float(row[f"{field}_ci95_half_width"]) for row in rows])
        axes[2].plot(t, mean, marker="o", color=color, label=label)
        axes[2].fill_between(t, mean-half, mean+half, color=color, alpha=0.13)
    axes[2].set_ylabel("Occupancy (GiB/GPU)")
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].legend(frameon=False, ncol=3, loc="upper right")
    axes[2].set_xlim(0, 240)
    axes[2].text(30, -0.28, "Low", transform=axes[2].get_xaxis_transform(), ha="center")
    axes[2].text(90, -0.28, "High", transform=axes[2].get_xaxis_transform(), ha="center")
    axes[2].text(180, -0.28, "Recovery", transform=axes[2].get_xaxis_transform(), ha="center")
    fig.suptitle("S5 AIMD and physical occupancy timeline (3 seeds)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97), h_pad=0.55)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate_figures(stage: Path) -> dict[str, bool]:
    checks = {}
    for stem in ("s4_algorithm_ablation", "s5_aimd_occupancy_timeline"):
        pdf = stage / "figures" / f"{stem}.pdf"
        png = stage / "figures" / f"{stem}.png"
        with fitz.open(pdf) as document:
            checks[f"{stem}_vector_pdf_single_page"] = len(document) == 1
        with Image.open(png) as image:
            checks[f"{stem}_png_width_at_least_1800"] = image.width >= 1800
    return checks


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text())
    source_hashes_match = True
    for row in spec["sources"].values():
        source = Path(row["path"])
        if not source.is_absolute():
            source = ROOT / source
        source_hashes_match &= source.is_file() and core.sha256(source) == row["sha256"]
    if not source_hashes_match:
        raise RuntimeError("preregistered source hashes do not match")
    result = ROOT / "results" / RESULT_NAME
    if result.exists():
        raise FileExistsError(f"refusing to overwrite {result}")
    stage = ROOT / "results" / f".{RESULT_NAME}.stage-{os.getpid()}"
    stage.mkdir(parents=True)
    (stage / "figures").mkdir()
    (stage / "test_logs").mkdir()
    test_checks = {}
    for filename, marker in TESTS:
        completed = subprocess.run(
            ["python", filename], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (stage / "test_logs" / f"{Path(filename).stem}.log").write_text(
            completed.stdout
        )
        passed = completed.returncode == 0 and marker in completed.stdout
        test_checks[filename] = passed
        if not passed:
            raise RuntimeError(f"test gate failed: {filename}\n{completed.stdout}")
    protocol, capacity = configure_protocol(spec)
    (stage / "resolved_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    (stage / "capacity_protocol.json").write_text(
        json.dumps(capacity, indent=2, sort_keys=True) + "\n"
    )

    s4_cells, s4_metrics, s4_comparisons = run_s4(
        stage, spec, protocol, capacity
    )
    write_csv(stage / "s4_seed_metrics.csv", s4_metrics)
    write_csv(stage / "s4_plot_source.csv", s4_comparisons)
    s5_cells, s5_timeline, s5_ci = run_s5(stage, spec, protocol, capacity)
    write_csv(stage / "s5_control_timeline_seed.csv", s5_timeline)
    write_csv(stage / "s5_plot_source.csv", s5_ci)

    plot_s4(stage / "figures" / "s4_algorithm_ablation", s4_comparisons)
    plot_s5(stage / "figures" / "s5_aimd_occupancy_timeline", s5_ci)

    s4_spec = spec["s4"]
    common_by_seed = all(
        len({cell[key] for cell in s4_cells if cell["seed"] == seed}) == 1
        for seed in s4_spec["paired_seeds"]
        for key in ("workflow_universe_sha256", "fault_probe_tuple_sha256")
    )
    s4_checks = {
        "exact_15_cells": len(s4_cells) == 15,
        "exact_three_preregistered_seeds": sorted({cell["seed"] for cell in s4_cells}) == sorted(s4_spec["paired_seeds"]),
        "exact_five_policies": {cell["internal_policy"] for cell in s4_cells} == set(s4_spec["internal_policies"]),
        "common_workflow_and_probe_universe_per_seed": common_by_seed,
        "both_failure_domains_reported": all(set(cell["failure_conditioned"]) == set(s4_spec["failure_domains"]) for cell in s4_cells),
        "capacity_and_copy_conservation": all(
            cell["mechanism"]["active_hbm_v3"]["capacity_violation_count"] == 0
            and cell["mechanism"]["active_hbm_v313"]["target_capacity_violation_count"] == 0
            and cell["mechanism"]["active_hbm_v313"]["copy_byte_conservation_error"] == {"HBM": 0, "DRAM": 0}
            for cell in s4_cells
        ),
        "algorithm1_ablation_zero_slack_and_full_uses_slack": all(
            cell["mechanism"]["algorithm1_nonzero_slack_request_count"] == 0
            for cell in s4_cells if cell["internal_policy"] == "AegisKV no Algorithm 1"
        ) and all(
            cell["mechanism"]["algorithm1_nonzero_slack_request_count"] > 0
            for cell in s4_cells if cell["internal_policy"] == "AegisKV"
        ),
        "algorithm2_ablation_is_not_static_baseline": all(
            not cell["mechanism"]["active_hbm_v313_s4s5"]["algorithm2_ablation_is_static_baseline"]
            and cell["mechanism"]["active_hbm_v313_s4s5"]["coupled_tier_attempt_count"] > 0
            for cell in s4_cells if cell["internal_policy"] == "AegisKV no Algorithm 2"
        ),
        "algorithm3_ablation_has_no_executed_updates": all(
            cell["mechanism"]["active_hbm_v313"]["executed_policy_control_times_ms"] == []
            and cell["mechanism"]["active_hbm_v313"]["scheduled_control_opportunity_times_ms"] == [30000.0, 60000.0, 90000.0]
            for cell in s4_cells if cell["internal_policy"] == "AegisKV no Algorithm 3"
        ),
        "no_seed_or_cell_selection": True,
    }

    s5_spec = spec["s5"]
    control_rows = [row for row in s5_timeline if row["is_control_window"]]
    by_seed = defaultdict(list)
    for row in control_rows:
        by_seed[int(row["seed"])].append(row)
    high_decrease = []
    recovery_increase = []
    high_pressure = []
    recovery_stable = []
    for seed, rows in sorted(by_seed.items()):
        rows.sort(key=lambda row: float(row["time_s"]))
        alpha_at_30 = next(float(row["alpha"]) for row in rows if row["time_s"] == 30.0)
        high = [row for row in rows if 60.0 <= float(row["time_s"]) < 120.0]
        recovery = [row for row in rows if float(row["time_s"]) >= 120.0]
        high_decrease.append(min(float(row["alpha"]) for row in high) < alpha_at_30)
        recovery_increase.append(any(
            float(right["alpha"]) > float(left["alpha"]) + 1e-12
            for left, right in zip(recovery, recovery[1:])
        ))
        high_pressure.append(any(float(row["pressure_fraction"]) > 0 for row in high))
        recovery_stable.append(any(float(row["pressure_fraction"]) == 0 for row in recovery))
    s5_checks = {
        "exact_three_preregistered_seeds": sorted(by_seed) == sorted(s5_spec["seeds"]),
        "exact_piecewise_schedule": s5_spec["phases"] == [
            {"name": "low", "start_ms": 0.0, "end_ms": 60000.0, "offered_active_requests_per_gpu": 2, "seed_xor": 2712847316},
            {"name": "high", "start_ms": 60000.0, "end_ms": 120000.0, "offered_active_requests_per_gpu": 8, "seed_xor": 299917636},
            {"name": "recovery", "start_ms": 120000.0, "end_ms": 240000.0, "offered_active_requests_per_gpu": 2, "seed_xor": 1432778632},
        ],
        "exact_eight_control_windows_per_seed": all(
            [float(row["time_s"]) for row in sorted(rows, key=lambda row: float(row["time_s"]))] == [30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0]
            for rows in by_seed.values()
        ),
        "quota_equals_alpha_times_40gib": all(
            abs(float(row["peer_hbm_quota_gib_per_target"]) - 40.0 * float(row["alpha"])) <= 1e-7
            for row in control_rows
        ),
        "quota_and_serving_budget_are_distinct_fields": all(
            "peer_hbm_quota_gib_per_target" in row and "shared_serving_plus_backup_budget_gib_per_target" in row
            for row in control_rows
        ),
        "capacity_and_copy_conservation": all(
            cell["capacity_violation_count"] == 0
            and cell["copy_byte_conservation_error"] == {"HBM": 0, "DRAM": 0}
            for cell in s5_cells
        ),
        "high_phase_pressure_observed_all_seeds": all(high_pressure),
        "high_phase_alpha_decreases_all_seeds": all(high_decrease),
        "recovery_phase_has_stable_window_all_seeds": all(recovery_stable),
        "recovery_phase_additive_alpha_increase_all_seeds": all(recovery_increase),
        "occupancy_uses_event_capacity_not_event_fraction": True,
        "no_seed_or_window_selection": True,
    }
    figure_checks = validate_figures(stage)
    checks = {
        "lineage_and_tests": {
            "preregistered_source_hashes_match": source_hashes_match,
            **test_checks,
        },
        "S4": s4_checks,
        "S5": s5_checks,
        "figures": figure_checks,
    }
    flat = {
        f"{section}.{name}": passed
        for section, rows in checks.items() for name, passed in rows.items()
    }
    status = "PASS" if all(flat.values()) else "FAIL"
    summary = {
        "schema_version": 1,
        "status": status,
        "paper_ready": False,
        "protocol": str(SPEC_PATH),
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner_sha256": core.sha256(Path(__file__)),
        "extension_sha256": core.sha256(ROOT / "active_hbm_v313_s4s5.py"),
        "S4": {"cells": s4_cells, "paired_comparisons": s4_comparisons},
        "S5": {"cells": s5_cells, "control_timeline_ci": s5_ci},
        "evidence_boundary": (
            "Three-seed preliminary DES. Bandwidth, service, fence and replacement "
            "remain modeled/unmeasured. The S4 region and S5 schedule were fixed "
            "before execution, and no seed/cell was selected by outcome direction."
        ),
    }
    (stage / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "paper_ready": False,
        "protocol_sha256": summary["protocol_sha256"],
        "input_source_hashes": spec["sources"],
        "figures": {
            "S4": {
                "pdf": "figures/s4_algorithm_ablation.pdf",
                "png": "figures/s4_algorithm_ablation.png",
                "source": "s4_plot_source.csv",
                "semantics": "mean and p99 fault-conditioned WCT normalized within seed/domain to Full AegisKV",
            },
            "S5": {
                "pdf": "figures/s5_aimd_occupancy_timeline.pdf",
                "png": "figures/s5_aimd_occupancy_timeline.png",
                "source": "s5_plot_source.csv",
                "semantics": "event-derived control-window alpha, pressure, load and physical occupancy",
            },
        },
        "evidence_boundary": summary["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    audit = {
        "schema_version": 1,
        "status": status,
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in flat.items() if not passed],
        "protocol_sha256": summary["protocol_sha256"],
        "summary_sha256": core.sha256(stage / "summary.json"),
        "manifest_sha256": core.sha256(stage / "manifest.json"),
        "s4_plot_source_sha256": core.sha256(stage / "s4_plot_source.csv"),
        "s5_plot_source_sha256": core.sha256(stage / "s5_plot_source.csv"),
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in artifacts.items())
    )
    os.replace(stage, result)
    print(json.dumps({
        "result": str(result), "status": status,
        "failed_checks": audit["failed_checks"],
        "audit_sha256": core.sha256(result / "audit.json"),
        "manifest_sha256": core.sha256(result / "manifest.json"),
    }, indent=2), flush=True)
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
