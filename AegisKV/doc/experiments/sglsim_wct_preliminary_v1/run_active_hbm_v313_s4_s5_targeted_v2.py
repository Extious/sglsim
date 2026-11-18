#!/usr/bin/env python3
"""Atomic mechanism-targeted S4 and analytically safe-recovery S5 v2 pilot."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
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
import active_hbm_v313_s4s5_v2 as ext
import run_active_hbm_v313_s4_s5_pilot as v1run
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s4_s5_targeted_v2_protocol.json"
RESULT_NAME = "active_hbm_v313_s4_s5_targeted_v2"
GIB = 1024**3
TESTS = (
    ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    ("test_active_hbm_v313_s4s5.py", "RESULT 6/6 PASS"),
    ("test_active_hbm_v313_s4s5_v2.py", "RESULT 5/5 PASS"),
    ("test_active_hbm_v313_s4_s5_targeted_v2_protocol.py", "RESULT 8/8 PASS"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return v1run.configure_protocol(spec,)[0:2]


def branch_index(task: core.Task) -> int:
    return int(task.request_id.rsplit("-b", 1)[1])


def build_alg1_workload(
    protocol: dict[str, Any], *, seed: int, spec: dict[str, Any],
) -> tuple[list[core.Task], dict[str, str], dict[str, Any]]:
    rows, metadata = v313.v312.v31.build_workload(
        protocol, seed=seed,
        active_per_gpu=int(spec["active_requests_per_gpu"]),
        arrival_cutoff_ms=float(spec["arrival_cutoff_ms"]),
    )
    model = protocol["models"]["qwen3_8b"]
    active = int(spec["active_requests_per_gpu"])
    slowdown = 1.0 + 0.035 * max(active - 1, 0)
    predicted = []
    for output in spec["branch_role_predicted_output_tokens"]:
        predicted.append(slowdown * (
            float(model["prefill_setup_ms"])
            + int(spec["context_tokens"]) * float(model["prefill_ms_per_token"])
            + int(output) * float(model["decode_ms_per_token"])
        ))
    critical = max(predicted)
    classes = {0: "critical", 1: "medium_slack", 2: "high_slack", 3: "high_slack"}
    class_by_request = {}
    rewritten = []
    for row in rows:
        branch = branch_index(row)
        class_by_request[row.request_id] = classes[branch]
        rewritten.append(replace(
            row, predicted_slack_ms=max(critical - predicted[branch], 0.0)
        ))
    metadata["residual_dag_slack"] = {
        "kind": "fixed_branch_role_causal_prediction",
        "predicted_output_tokens_by_branch": spec["branch_role_predicted_output_tokens"],
        "predicted_service_ms_by_branch": predicted,
        "reads_realized_output_or_service": False,
    }
    return rewritten, class_by_request, metadata


def build_alg2_workload(
    protocol: dict[str, Any], *, seed: int, spec: dict[str, Any],
) -> tuple[list[core.Task], dict[str, str], dict[str, tuple[core.ProtectionTier, ...]], dict[str, Any]]:
    rows, metadata = v313.v312.v31.build_workload(
        protocol, seed=seed,
        active_per_gpu=int(spec["active_requests_per_gpu"]),
        arrival_cutoff_ms=float(spec["arrival_cutoff_ms"]),
    )
    classes = ["both_benefit", "hbm_benefit", "dram_benefit", "neither_benefit"]
    tiers = {
        "both_benefit": (core.ProtectionTier.HBM, core.ProtectionTier.DRAM),
        "hbm_benefit": (core.ProtectionTier.HBM,),
        "dram_benefit": (core.ProtectionTier.DRAM,),
        "neither_benefit": (),
    }
    class_by_request = {}
    masks = {}
    rewritten = []
    for row in rows:
        name = classes[branch_index(row)]
        class_by_request[row.request_id] = name
        masks[row.request_id] = tiers[name]
        rewritten.append(replace(row, predicted_slack_ms=0.0))
    metadata["tier_benefit_classes"] = {
        "class_by_branch": dict(enumerate(classes)),
        "physical_target_capacity_changed": False,
        "policy_quota_changed": False,
    }
    return rewritten, class_by_request, masks, metadata


def workflow_fault_stats(
    tasks: list[core.Task], probes: list[core.FaultProbe],
    delay_maps: list[dict[str, float]],
) -> tuple[list[str], list[float], list[list[float]], dict[str, float]]:
    ids, healthy = core.workflow_wct(tasks)
    matrix = []
    added = []
    for delays in delay_maps:
        completion = core.replay_one_fault(tasks, delays)
        replay_ids, fault = core.workflow_wct(tasks, completion)
        if replay_ids != ids:
            raise AssertionError("targeted S4 workflow denominator changed")
        for base, value in zip(healthy, fault):
            delta = value - base
            if delta < -1e-7:
                raise AssertionError("targeted S4 fault lowered WCT")
            added.append(max(delta, 0.0))
        matrix.append(fault)
    flattened = [value for values in matrix for value in values]
    return ids, healthy, matrix, {
        "fault_observation_count": len(flattened),
        "fault_wct_mean_ms": statistics.fmean(flattened),
        "fault_wct_p99_ms": core.quantile(flattened, 0.99),
        "fault_added_wct_mean_ms": statistics.fmean(added),
        "fault_added_wct_p99_ms": core.quantile(added, 0.99),
    }


def save_raw(
    path: Path, ids: list[str], healthy: list[float], fault: list[list[float]],
    probes: list[core.FaultProbe], capacity_trace: list[dict[str, Any]],
) -> None:
    cap = np.asarray([
        (
            row["time_ms"], row["rank"], row["active_requests"],
            row["waiting_requests"], row["active_serving_kv_bytes"],
            row["peer_hbm_backup_used_bytes"], row["shared_used_bytes"],
            row["shared_headroom_bytes"],
        ) for row in capacity_trace
    ], dtype=float).reshape((-1, 8))
    np.savez_compressed(
        path,
        workflow_ids=np.asarray(ids),
        healthy_wct_ms=np.asarray(healthy, dtype=float),
        fault_wct_ms=np.asarray(fault, dtype=float),
        probe_id=np.asarray([row.probe_id for row in probes]),
        probe_time_ms=np.asarray([row.time_ms for row in probes], dtype=float),
        probe_source_rank=np.asarray([row.source_rank for row in probes], dtype=int),
        capacity_trace=cap,
    )


def copy_outcome(replay: ext.TargetedReplay, request_id: str) -> str:
    row = replay.copy_accounting_by_request.get(request_id, {})
    hbm = bool(row.get("HBM", {}).get("bytes", 0))
    dram = bool(row.get("DRAM", {}).get("bytes", 0))
    if hbm and dram:
        return "HBM_AND_DRAM"
    if hbm:
        return "HBM_ONLY"
    if dram:
        return "DRAM_ONLY"
    return "BYPASS"


def cell_metrics(
    *, algorithm: str, policy: str, seed: int, tasks: list[core.Task],
    probes: list[core.FaultProbe], replay: ext.TargetedReplay,
    scheduled: list[core.Task], delays: list[dict[str, float]],
    mechanism: dict[str, Any], capacity_trace: list[dict[str, Any]],
    raw_path: Path, metadata: dict[str, Any], extra: dict[str, Any],
) -> dict[str, Any]:
    ids, healthy, fault, failure = workflow_fault_stats(scheduled, probes, delays)
    save_raw(raw_path, ids, healthy, fault, probes, capacity_trace)
    decomposition = v313.workflow_decomposition(scheduled)
    v3m = mechanism["active_hbm_v3"]
    v313m = mechanism["active_hbm_v313"]
    return {
        "algorithm": algorithm,
        "policy": policy,
        "seed": seed,
        "workflow_universe_sha256": v1run.task_universe_hash(tasks),
        "fault_probe_tuple_sha256": v1run.probe_hash(probes),
        "workflow_count": len(ids),
        "healthy_wct_mean_ms": statistics.fmean(healthy),
        "healthy_wct_p99_ms": core.quantile(healthy, 0.99),
        "healthy_decomposition": decomposition,
        **failure,
        "hbm_scheduled_copy_bytes": v313m["scheduled_copy_bytes"]["HBM"],
        "dram_scheduled_copy_bytes": v313m["scheduled_copy_bytes"]["DRAM"],
        "primary_priority_reclaimed_bytes": v313m["primary_priority_reclaimed_bytes"],
        "aimd_backup_released_bytes": v3m["backup_bytes_released_by_control"],
        "capacity_violation_count": v3m["capacity_violation_count"] + v313m["target_capacity_violation_count"],
        "copy_byte_conservation_error": v313m["copy_byte_conservation_error"],
        "alpha_final_mean": mechanism["alpha_final_mean"],
        "mechanism": mechanism,
        "workload": metadata,
        "extra": extra,
        "raw_npz": str(raw_path),
        "raw_npz_sha256": core.sha256(raw_path),
    }


def probes_for(tasks: list[core.Task], *, seed: int, count: int, horizon: float) -> list[core.FaultProbe]:
    return core.make_fault_probes(
        tasks, seed=seed ^ 0x53475032, count=count, gpus_per_node=8,
        right_ms=horizon, failure_types=("source_worker_gpu",),
    )


def run_s4_alg1(
    stage: Path, spec: dict[str, Any], protocol: dict[str, Any], capacity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = spec["s4"]["algorithm1"]
    cells = []
    metric_rows = []
    for seed in spec["common"]["seeds"]:
        local = deepcopy(protocol)
        local["workload"]["center_context_tokens"] = cfg["context_tokens"]
        local["workload"]["center_active_requests_per_gpu"] = cfg["active_requests_per_gpu"]
        tasks, classes, metadata = build_alg1_workload(local, seed=seed, spec=cfg)
        probes = probes_for(tasks, seed=seed, count=cfg["fault_probes_per_seed"], horizon=cfg["arrival_cutoff_ms"])
        for policy in ("AegisKV", "AegisKV no Algorithm 1"):
            replay = ext.TargetedReplay(deepcopy(local), policy, capacity)
            scheduled, delays, mechanism, cap_trace = replay.run_coupled(
                tasks, probes, control_horizon_ms=cfg["arrival_cutoff_ms"]
            )
            copy_by_class = defaultdict(lambda: {"HBM": 0, "DRAM": 0})
            for request_id, class_name in classes.items():
                row = replay.copy_accounting_by_request.get(request_id, {})
                for tier in ("HBM", "DRAM"):
                    copy_by_class[class_name][tier] += int(row.get(tier, {}).get("bytes", 0))
            total_hbm = sum(row["HBM"] for row in copy_by_class.values())
            critical_share = copy_by_class["critical"]["HBM"] / max(total_hbm, 1)
            raw = stage / "raw" / f"alg1__{policy.lower().replace(' ', '_')}__s{seed}.npz"
            cell = cell_metrics(
                algorithm="Algorithm 1", policy=policy, seed=seed, tasks=tasks,
                probes=probes, replay=replay, scheduled=scheduled, delays=delays,
                mechanism=mechanism, capacity_trace=cap_trace, raw_path=raw,
                metadata=metadata,
                extra={"copy_bytes_by_slack_class": copy_by_class, "critical_hbm_copy_share": critical_share},
            )
            cells.append(cell)
            metric_rows.append({
                "algorithm": "Algorithm 1", "seed": seed, "policy": policy,
                "critical_hbm_copy_share": critical_share,
                "healthy_wct_mean_ms": cell["healthy_wct_mean_ms"],
                "fault_wct_mean_ms": cell["fault_wct_mean_ms"],
                "fault_wct_p99_ms": cell["fault_wct_p99_ms"],
                "hbm_copy_gib": cell["hbm_scheduled_copy_bytes"] / GIB,
                "dram_copy_gib": cell["dram_scheduled_copy_bytes"] / GIB,
                "workflow_universe_sha256": cell["workflow_universe_sha256"],
                "fault_probe_tuple_sha256": cell["fault_probe_tuple_sha256"],
            })
            print(f"S4 Alg1 {len(cells)}/6 seed={seed} policy={policy}", flush=True)
    return cells, metric_rows


def run_s4_alg2(
    stage: Path, spec: dict[str, Any], protocol: dict[str, Any], capacity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = spec["s4"]["algorithm2"]
    cells = []
    outcome_rows = []
    for seed in spec["common"]["seeds"]:
        local = deepcopy(protocol)
        local["workload"]["center_context_tokens"] = cfg["context_tokens"]
        local["workload"]["center_active_requests_per_gpu"] = cfg["active_requests_per_gpu"]
        tasks, classes, masks, metadata = build_alg2_workload(local, seed=seed, spec=cfg)
        probes = probes_for(tasks, seed=seed, count=cfg["fault_probes_per_seed"], horizon=cfg["arrival_cutoff_ms"])
        for policy in ("AegisKV", "AegisKV no Algorithm 2"):
            replay = ext.BenefitMaskedReplay(
                deepcopy(local), policy, capacity,
                allowed_tiers_by_request=masks,
            )
            scheduled, delays, mechanism, cap_trace = replay.run_coupled(
                tasks, probes, control_horizon_ms=cfg["arrival_cutoff_ms"]
            )
            counts = defaultdict(Counter)
            for request_id, class_name in classes.items():
                counts[class_name][copy_outcome(replay, request_id)] += 1
            raw = stage / "raw" / f"alg2__{policy.lower().replace(' ', '_')}__s{seed}.npz"
            cell = cell_metrics(
                algorithm="Algorithm 2", policy=policy, seed=seed, tasks=tasks,
                probes=probes, replay=replay, scheduled=scheduled, delays=delays,
                mechanism=mechanism, capacity_trace=cap_trace, raw_path=raw,
                metadata=metadata,
                extra={"outcome_counts_by_benefit_class": counts},
            )
            cells.append(cell)
            for class_name in cfg["benefit_classes"]:
                total = sum(counts[class_name].values())
                for outcome in ("HBM_AND_DRAM", "HBM_ONLY", "DRAM_ONLY", "BYPASS"):
                    outcome_rows.append({
                        "algorithm": "Algorithm 2", "seed": seed,
                        "policy": policy, "benefit_class": class_name,
                        "outcome": outcome, "count": counts[class_name][outcome],
                        "fraction": counts[class_name][outcome] / max(total, 1),
                        "healthy_wct_mean_ms": cell["healthy_wct_mean_ms"],
                        "fault_wct_mean_ms": cell["fault_wct_mean_ms"],
                        "workflow_universe_sha256": cell["workflow_universe_sha256"],
                        "fault_probe_tuple_sha256": cell["fault_probe_tuple_sha256"],
                    })
            print(f"S4 Alg2 {len(cells)}/6 seed={seed} policy={policy}", flush=True)
    return cells, outcome_rows


def extract_s5_seed_rows(
    *, seed: int, replay: ext.TargetedReplay, mechanism: dict[str, Any],
    phases: list[dict[str, Any]], capacity: dict[str, Any], universe_sha: str,
) -> list[dict[str, Any]]:
    rich = mechanism["active_hbm_v313_s4s5"]["rich_control_trace"]
    targets = mechanism["active_hbm_v313_s4s5_v2"]["target_control_trace"]
    targets_by_time = defaultdict(list)
    for row in targets:
        targets_by_time[float(row["time_ms"])].append(row)
    rows = [{
        "seed": seed, "time_s": 0.0, "phase": phases[0]["name"],
        "offered_active_requests_per_gpu": phases[0]["offered_active_requests_per_gpu"],
        "realized_active_requests_per_gpu": 0.0, "pressure_fraction": 0.0,
        "alpha": 0.25, "quota_gib_per_target": 10.0,
        "serving_hbm_gib_per_gpu": 0.0, "peer_hbm_backup_gib_per_gpu": 0.0,
        "dram_gib_per_gpu": 0.0, "stable_target_count": 16,
        "additive_increase_target_count": 0, "decrease_target_count": 0,
        "shared_budget_gib_per_target": int(capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]) / GIB,
        "workflow_universe_sha256": universe_sha,
    }]
    for row in rich:
        at = float(row["time_ms"])
        phase = v1run.phase_at(phases, at)
        target_rows = targets_by_time[at]
        if len(target_rows) != 16:
            raise AssertionError("missing per-target S5 control rows")
        rows.append({
            "seed": seed, "time_s": at / 1000.0, "phase": phase["name"],
            "offered_active_requests_per_gpu": phase["offered_active_requests_per_gpu"],
            "realized_active_requests_per_gpu": row["realized_active_requests_per_gpu"],
            "pressure_fraction": row["pressure_fraction"],
            "alpha": row["alpha_after_mean"],
            "quota_gib_per_target": row["peer_hbm_quota_bytes_total"] / 16 / GIB,
            "serving_hbm_gib_per_gpu": row["serving_hbm_bytes_total"] / 16 / GIB,
            "peer_hbm_backup_gib_per_gpu": row["peer_hbm_backup_after_bytes_total"] / 16 / GIB,
            "dram_gib_per_gpu": row["dram_used_bytes_total"] / 16 / GIB,
            "stable_target_count": sum(not item["pressure"] for item in target_rows),
            "additive_increase_target_count": sum(item["alpha_after"] > item["alpha_before"] + 1e-12 for item in target_rows),
            "decrease_target_count": sum(item["alpha_after"] < item["alpha_before"] - 1e-12 for item in target_rows),
            "shared_budget_gib_per_target": int(capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]) / GIB,
            "workflow_universe_sha256": universe_sha,
        })
    return rows


def run_s4_alg3_and_s5(
    stage: Path, spec: dict[str, Any], protocol: dict[str, Any], capacity: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = spec["s4"]["algorithm3"]
    cells = []
    timeline = []
    s5_seed_rows = []
    for seed in spec["common"]["seeds"]:
        local = deepcopy(protocol)
        local["workload"]["center_context_tokens"] = cfg["context_tokens"]
        tasks, metadata = v1run.build_piecewise_workload(
            local, seed=seed, phases=cfg["phases"], context_tokens=cfg["context_tokens"]
        )
        probes = probes_for(tasks, seed=seed, count=cfg["fault_probes_per_seed"], horizon=cfg["control_horizon_ms"])
        universe_sha = v1run.task_universe_hash(tasks)
        for policy in ("AegisKV", "AegisKV no Algorithm 3"):
            replay = ext.TargetedReplay(deepcopy(local), policy, capacity)
            scheduled, delays, mechanism, cap_trace = replay.run_coupled(
                tasks, probes, control_horizon_ms=cfg["control_horizon_ms"]
            )
            raw = stage / "raw" / f"alg3__{policy.lower().replace(' ', '_')}__s{seed}.npz"
            cell = cell_metrics(
                algorithm="Algorithm 3", policy=policy, seed=seed, tasks=tasks,
                probes=probes, replay=replay, scheduled=scheduled, delays=delays,
                mechanism=mechanism, capacity_trace=cap_trace, raw_path=raw,
                metadata={"phases": metadata}, extra={},
            )
            cells.append(cell)
            if policy == "AegisKV":
                seed_rows = extract_s5_seed_rows(
                    seed=seed, replay=replay, mechanism=mechanism,
                    phases=cfg["phases"], capacity=capacity,
                    universe_sha=universe_sha,
                )
                s5_seed_rows.extend(seed_rows)
                for row in seed_rows:
                    timeline.append({**row, "policy": "Full AegisKV"})
            else:
                for at in [0.0, *[value / 1000.0 for value in cfg["control_opportunities_ms"]]]:
                    phase = v1run.phase_at(cfg["phases"], at * 1000.0)
                    timeline.append({
                        "seed": seed, "time_s": at, "phase": phase["name"],
                        "offered_active_requests_per_gpu": phase["offered_active_requests_per_gpu"],
                        "realized_active_requests_per_gpu": "",
                        "pressure_fraction": "", "alpha": 0.25,
                        "quota_gib_per_target": 10.0,
                        "serving_hbm_gib_per_gpu": "", "peer_hbm_backup_gib_per_gpu": "",
                        "dram_gib_per_gpu": "", "stable_target_count": "",
                        "additive_increase_target_count": 0, "decrease_target_count": 0,
                        "shared_budget_gib_per_target": int(capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]) / GIB,
                        "workflow_universe_sha256": universe_sha,
                        "policy": "-Algorithm 3",
                    })
            print(f"S4 Alg3 {len(cells)}/6 seed={seed} policy={policy}", flush=True)
    return cells, timeline, s5_seed_rows


def paired_summary(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by = {(row["algorithm"], row["seed"], row["policy"]): row for row in cells}
    ablation = {
        "Algorithm 1": "AegisKV no Algorithm 1",
        "Algorithm 2": "AegisKV no Algorithm 2",
        "Algorithm 3": "AegisKV no Algorithm 3",
    }
    for algorithm, no_policy in ablation.items():
        for metric in (
            "healthy_wct_mean_ms", "healthy_wct_p99_ms",
            "fault_wct_mean_ms", "fault_wct_p99_ms",
        ):
            full = [by[(algorithm, seed, "AegisKV")][metric] for seed in sorted({row["seed"] for row in cells})]
            no = [by[(algorithm, seed, no_policy)][metric] for seed in sorted({row["seed"] for row in cells})]
            ratios = [value / base for value, base in zip(no, full)]
            deltas = [value - base for value, base in zip(no, full)]
            rows.append({
                "algorithm": algorithm, "ablation_policy": no_policy,
                "metric": metric, "seed_count": len(ratios),
                "full_mean_ms": statistics.fmean(full),
                "ablation_mean_ms": statistics.fmean(no),
                "ablation_over_full_mean": statistics.fmean(ratios),
                "paired_ratio_ci95_half_width": v1run.ci95(ratios),
                "paired_delta_mean_ms": statistics.fmean(deltas),
                "paired_delta_ci95_half_width_ms": v1run.ci95(deltas),
                "ci_method": "paired seed-level t interval, df=2",
            })
    return rows


def aggregate_alg1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for policy in ("AegisKV", "AegisKV no Algorithm 1"):
        selected = [row for row in rows if row["policy"] == policy]
        values = [float(row["critical_hbm_copy_share"]) for row in selected]
        output.append({
            "policy": "Full AegisKV" if policy == "AegisKV" else "-Algorithm 1",
            "seed_count": len(values), "critical_hbm_copy_share_mean": statistics.fmean(values),
            "critical_hbm_copy_share_ci95_half_width": v1run.ci95(values),
        })
    return output


def aggregate_alg2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["policy"], row["benefit_class"], row["outcome"]) for row in rows})
    for policy, benefit, outcome in keys:
        values = [float(row["fraction"]) for row in rows if row["policy"] == policy and row["benefit_class"] == benefit and row["outcome"] == outcome]
        output.append({
            "policy": "Full AegisKV" if policy == "AegisKV" else "-Algorithm 2",
            "benefit_class": benefit, "outcome": outcome,
            "seed_count": len(values), "fraction_mean": statistics.fmean(values),
            "fraction_ci95_half_width": v1run.ci95(values),
        })
    return output


def aggregate_timeline(rows: list[dict[str, Any]], *, policies: tuple[str, ...]) -> list[dict[str, Any]]:
    output = []
    for policy in policies:
        for at in sorted({float(row["time_s"]) for row in rows if row["policy"] == policy}):
            selected = [row for row in rows if row["policy"] == policy and float(row["time_s"]) == at]
            values = [float(row["alpha"]) for row in selected]
            output.append({
                "policy": policy, "time_s": at, "phase": selected[0]["phase"],
                "seed_count": len(values), "alpha_mean": statistics.fmean(values),
                "alpha_ci95_half_width": v1run.ci95(values),
            })
    return output


def aggregate_s5(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "offered_active_requests_per_gpu", "realized_active_requests_per_gpu",
        "pressure_fraction", "alpha", "quota_gib_per_target",
        "serving_hbm_gib_per_gpu", "peer_hbm_backup_gib_per_gpu",
        "dram_gib_per_gpu", "stable_target_count",
        "additive_increase_target_count", "decrease_target_count",
    ]
    output = []
    for at in sorted({float(row["time_s"]) for row in rows}):
        selected = [row for row in rows if float(row["time_s"]) == at]
        out = {"time_s": at, "phase": selected[0]["phase"], "seed_count": len(selected)}
        for field in fields:
            values = [float(row[field]) for row in selected]
            out[f"{field}_mean"] = statistics.fmean(values)
            out[f"{field}_ci95_half_width"] = v1run.ci95(values)
        output.append(out)
    return output


def plot_s4(path: Path, alg1: list[dict[str, Any]], alg2: list[dict[str, Any]], alg3: list[dict[str, Any]]) -> None:
    v1run.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75))
    # Algorithm 1 target: critical-path protection share.
    x = np.arange(2)
    y = [row["critical_hbm_copy_share_mean"] for row in alg1]
    e = [row["critical_hbm_copy_share_ci95_half_width"] for row in alg1]
    axes[0].errorbar(x, y, yerr=e, fmt="o", color="#d62728", capsize=3, markersize=5)
    axes[0].set_xticks(x, ["Full", "-Alg. 1"])
    axes[0].set_ylabel("Critical-branch share of\nPeer-HBM copy bytes")
    axes[0].set_ylim(0, max(0.5, max(y) * 1.25))
    axes[0].set_title("Algorithm 1: slack priority")
    axes[0].grid(axis="y", alpha=0.25)

    # Algorithm 2 target: independent residency outcomes.
    benefits = ["both_benefit", "hbm_benefit", "dram_benefit", "neither_benefit"]
    short = ["Both", "HBM", "DRAM", "Neither"]
    outcomes = ["HBM_AND_DRAM", "HBM_ONLY", "DRAM_ONLY", "BYPASS"]
    colors = ["#1b9e77", "#66a61e", "#7570b3", "#bdbdbd"]
    positions = []
    labels = []
    for i, benefit in enumerate(benefits):
        for j, policy in enumerate(("Full AegisKV", "-Algorithm 2")):
            pos = i * 2.4 + j
            positions.append(pos)
            labels.append(f"{short[i]}\n{'Full' if j == 0 else '-A2'}")
            bottom = 0.0
            for outcome, color in zip(outcomes, colors):
                row = next(item for item in alg2 if item["policy"] == policy and item["benefit_class"] == benefit and item["outcome"] == outcome)
                axes[1].bar(pos, row["fraction_mean"], bottom=bottom, color=color, width=0.82, label=outcome if i == 0 and j == 0 else None)
                bottom += row["fraction_mean"]
    axes[1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Residency fraction")
    axes[1].set_title("Algorithm 2: tier independence")
    axes[1].legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)

    # Algorithm 3 target: alpha response.
    for policy, color, marker in (("Full AegisKV", "#7b3294", "o"), ("-Algorithm 3", "#555555", "s")):
        selected = [row for row in alg3 if row["policy"] == policy]
        t = np.asarray([row["time_s"] for row in selected])
        mean = np.asarray([row["alpha_mean"] for row in selected])
        half = np.asarray([row["alpha_ci95_half_width"] for row in selected])
        axes[2].plot(t, mean, color=color, marker=marker, markersize=3, label=policy)
        axes[2].fill_between(t, mean-half, mean+half, color=color, alpha=0.13)
    axes[2].axvspan(90, 180, color="#d62728", alpha=0.06)
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_ylabel(r"Peer-HBM $\alpha$")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("Algorithm 3: AIMD response")
    axes[2].legend(frameon=False, loc="best")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("S4 mechanism-targeted ablations (3 paired seeds)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95), w_pad=1.2)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_s5(path: Path, rows: list[dict[str, Any]]) -> None:
    v1run.style()
    t = np.asarray([row["time_s"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.2), sharex=True)
    spans = [(0, 90, "#2ca02c"), (90, 180, "#d62728"), (180, 420, "#2ca02c")]
    for ax in axes:
        for left, right, color in spans:
            ax.axvspan(left, right, color=color, alpha=0.07)
        ax.grid(axis="y", alpha=0.25)
    alpha = np.asarray([row["alpha_mean"] for row in rows])
    half = np.asarray([row["alpha_ci95_half_width"] for row in rows])
    axes[0].plot(t, alpha, color="#7b3294", marker="o", markersize=3)
    axes[0].fill_between(t, alpha-half, alpha+half, color="#7b3294", alpha=0.15)
    axes[0].set_ylabel(r"Peer-HBM $\alpha$")
    axes[0].set_ylim(bottom=0)
    for x, label in ((45, "Low"), (135, "High"), (300, "Recovery")):
        axes[0].text(x, 0.96, label, transform=axes[0].get_xaxis_transform(), ha="center", va="top", color="#444444", fontsize=7.5)
    quota = axes[0].twinx()
    quota.plot(t, 40 * alpha, color="#c51b7d", linestyle="--", linewidth=1.0)
    quota.set_ylabel(r"Quota $40\alpha$ (GiB/target)")
    quota.spines["top"].set_visible(False)

    offered = np.asarray([row["offered_active_requests_per_gpu_mean"] for row in rows])
    realized = np.asarray([row["realized_active_requests_per_gpu_mean"] for row in rows])
    pressure = np.asarray([row["pressure_fraction_mean"] for row in rows])
    axes[1].step(t, offered, where="post", color="#555555", linestyle="--", label="Offered")
    axes[1].plot(t, realized, color="#ff7f0e", marker="o", markersize=3, label="Realized")
    axes[1].set_ylabel("Requests/GPU")
    paxis = axes[1].twinx()
    paxis.plot(t, pressure, color="#d62728", marker="s", markersize=3, label="Pressure")
    paxis.set_ylabel("Pressure fraction")
    paxis.set_ylim(-0.03, 1.03)
    paxis.spines["top"].set_visible(False)
    h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = paxis.get_legend_handles_labels()
    axes[1].legend(h1+h2, l1+l2, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.04))

    for field, label, color in (
        ("serving_hbm_gib_per_gpu", "Serving HBM", "#1f77b4"),
        ("peer_hbm_backup_gib_per_gpu", "Peer-HBM backup", "#2ca02c"),
        ("dram_gib_per_gpu", "DRAM", "#9467bd"),
    ):
        mean = np.asarray([row[f"{field}_mean"] for row in rows])
        ci = np.asarray([row[f"{field}_ci95_half_width"] for row in rows])
        axes[2].plot(t, mean, color=color, marker="o", markersize=3, label=label)
        axes[2].fill_between(t, mean-ci, mean+ci, color=color, alpha=0.13)
    axes[2].set_ylabel("Occupancy (GiB/GPU)")
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_xlim(0, 420)
    axes[2].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("S5 analytically bounded recovery-load pilot (3 seeds)", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97), h_pad=1.3)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_checks(stage: Path) -> dict[str, bool]:
    checks = {}
    for stem in ("s4_targeted_ablation", "s5_safe_recovery_timeline"):
        with fitz.open(stage / "figures" / f"{stem}.pdf") as doc:
            checks[f"{stem}_vector_pdf_single_page"] = len(doc) == 1
        with Image.open(stage / "figures" / f"{stem}.png") as image:
            checks[f"{stem}_png_width_at_least_1800"] = image.width >= 1800
    return checks


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text())
    source_ok = True
    for row in spec["sources"].values():
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        source_ok &= path.is_file() and core.sha256(path) == row["sha256"]
    if not source_ok:
        raise RuntimeError("v2 preregistered source hash mismatch")
    result = ROOT / "results" / RESULT_NAME
    if result.exists():
        raise FileExistsError(f"refusing to overwrite {result}")
    stage = ROOT / "results" / f".{RESULT_NAME}.stage-{os.getpid()}"
    (stage / "raw").mkdir(parents=True)
    (stage / "figures").mkdir()
    (stage / "test_logs").mkdir()
    test_checks = {}
    for filename, marker in TESTS:
        completed = subprocess.run(
            ["python", filename], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (stage / "test_logs" / f"{Path(filename).stem}.log").write_text(completed.stdout)
        passed = completed.returncode == 0 and marker in completed.stdout
        test_checks[filename] = passed
        if not passed:
            raise RuntimeError(f"test failed: {filename}\n{completed.stdout}")
    protocol, capacity = configure(spec)
    (stage / "resolved_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (stage / "capacity_protocol.json").write_text(json.dumps(capacity, indent=2, sort_keys=True) + "\n")

    alg1_cells, alg1_seed = run_s4_alg1(stage, spec, protocol, capacity)
    alg2_cells, alg2_seed = run_s4_alg2(stage, spec, protocol, capacity)
    alg3_cells, alg3_timeline_seed, s5_seed = run_s4_alg3_and_s5(stage, spec, protocol, capacity)
    cells = alg1_cells + alg2_cells + alg3_cells
    paired = paired_summary(cells)
    alg1_plot = aggregate_alg1(alg1_seed)
    alg2_plot = aggregate_alg2(alg2_seed)
    alg3_plot = aggregate_timeline(alg3_timeline_seed, policies=("Full AegisKV", "-Algorithm 3"))
    s5_plot = aggregate_s5(s5_seed)
    write_csv(stage / "s4_alg1_seed.csv", alg1_seed)
    write_csv(stage / "s4_alg1_plot_source.csv", alg1_plot)
    write_csv(stage / "s4_alg2_seed.csv", alg2_seed)
    write_csv(stage / "s4_alg2_plot_source.csv", alg2_plot)
    write_csv(stage / "s4_alg3_timeline_seed.csv", alg3_timeline_seed)
    write_csv(stage / "s4_alg3_plot_source.csv", alg3_plot)
    write_csv(stage / "s4_paired_wct_ci.csv", paired)
    write_csv(stage / "s5_timeline_seed.csv", s5_seed)
    write_csv(stage / "s5_plot_source.csv", s5_plot)
    plot_s4(stage / "figures/s4_targeted_ablation", alg1_plot, alg2_plot, alg3_plot)
    plot_s5(stage / "figures/s5_safe_recovery_timeline", s5_plot)

    seeds = spec["common"]["seeds"]
    common_pairing = all(
        len({cell[key] for cell in cells if cell["algorithm"] == algorithm and cell["seed"] == seed}) == 1
        for algorithm in ("Algorithm 1", "Algorithm 2", "Algorithm 3")
        for seed in seeds
        for key in ("workflow_universe_sha256", "fault_probe_tuple_sha256")
    )
    alg2_expected = spec["s4"]["algorithm2"]["expected_full_outcome_by_class"]
    alg2_coverage = all(
        all(
            cell["extra"]["outcome_counts_by_benefit_class"][class_name][outcome] > 0
            and sum(cell["extra"]["outcome_counts_by_benefit_class"][class_name].values())
            == cell["extra"]["outcome_counts_by_benefit_class"][class_name][outcome]
            for class_name, outcome in alg2_expected.items()
        )
        for cell in alg2_cells if cell["policy"] == "AegisKV"
    )
    alg2_ablation_coupled = all(
        set(
            outcome for counts in cell["extra"]["outcome_counts_by_benefit_class"].values()
            for outcome, count in counts.items() if count
        ).issubset({"HBM_AND_DRAM", "BYPASS"})
        for cell in alg2_cells if cell["policy"] == "AegisKV no Algorithm 2"
    )
    control_times = spec["s4"]["algorithm3"]["control_opportunities_ms"]
    s4_checks = {
        "exact_18_cells": len(cells) == 18,
        "exact_three_seeds": sorted({cell["seed"] for cell in cells}) == seeds,
        "paired_universe_and_probe_hashes": common_pairing,
        "capacity_and_copy_conservation": all(
            cell["capacity_violation_count"] == 0
            and cell["copy_byte_conservation_error"] == {"HBM": 0, "DRAM": 0}
            for cell in cells
        ),
        "alg1_has_causal_slack_heterogeneity": all(
            len(set(cell["workload"]["residual_dag_slack"]["predicted_service_ms_by_branch"])) >= 3
            for cell in alg1_cells
        ),
        "alg1_ablation_only_zeroes_slack": all(
            cell["mechanism"]["algorithm1_nonzero_slack_request_count"] == 0
            for cell in alg1_cells if cell["policy"] == "AegisKV no Algorithm 1"
        ) and all(
            cell["mechanism"]["algorithm1_nonzero_slack_request_count"] > 0
            for cell in alg1_cells if cell["policy"] == "AegisKV"
        ),
        "alg2_full_covers_all_four_outcomes_by_preregistered_class": alg2_coverage,
        "alg2_ablation_remains_coupled": alg2_ablation_coupled,
        "alg3_full_executes_all_windows": all(
            cell["mechanism"]["active_hbm_v313"]["executed_policy_control_times_ms"] == control_times
            for cell in alg3_cells if cell["policy"] == "AegisKV"
        ),
        "alg3_ablation_executes_zero_updates": all(
            cell["mechanism"]["active_hbm_v313"]["scheduled_control_opportunity_times_ms"] == control_times
            and cell["mechanism"]["active_hbm_v313"]["executed_policy_control_times_ms"] == []
            for cell in alg3_cells if cell["policy"] == "AegisKV no Algorithm 3"
        ),
        "paired_ci_present_without_win_threshold": len(paired) == 12,
        "no_seed_cell_or_direction_selection": True,
    }

    s5_cfg = spec["s5"]
    by_seed = defaultdict(list)
    for row in s5_seed:
        if float(row["time_s"]) > 0:
            by_seed[int(row["seed"])].append(row)
    additive = []
    high_decrease = []
    observed_stable = []
    for seed, rows in sorted(by_seed.items()):
        rows.sort(key=lambda row: float(row["time_s"]))
        high = [row for row in rows if row["phase"] == "high"]
        recovery = [row for row in rows if row["phase"] == "recovery"]
        low_alpha = max(float(row["alpha"]) for row in rows if row["phase"] == "low")
        high_decrease.append(min(float(row["alpha"]) for row in high) < low_alpha)
        additive.append(any(
            float(right["alpha"]) > float(left["alpha"]) + 1e-12
            for left, right in zip(recovery, recovery[1:])
        ))
        observed_stable.append(sum(int(row["stable_target_count"]) for row in recovery))
    analytic = s5_cfg["analytic_bound"]
    s5_checks = {
        "p99_request_reservation_is_2_8125_gib": analytic["request_p99_serving_kv_bytes"] == int(2.8125 * GIB),
        "capacity_safe_ceiling_is_two": analytic["capacity_safe_active_requests"] == 2,
        "recovery_offered_load_is_one_with_guard": s5_cfg["recovery_offered_active_requests_per_gpu"] == 1,
        "recovery_schedule_has_eight_control_windows": len([value for value in control_times if value >= 210000.0]) == 8,
        "exact_three_unfiltered_seeds": sorted(by_seed) == seeds,
        "high_phase_alpha_decreases_all_seeds": all(high_decrease),
        "recovery_mean_alpha_additive_increase_all_seeds": all(additive),
        "recovery_observes_target_stability": all(value > 0 for value in observed_stable),
        "event_state_occupancy_not_fraction_inference": True,
        "quota_equals_alpha_times_40gib": all(
            abs(float(row["quota_gib_per_target"]) - 40.0 * float(row["alpha"])) <= 1e-7
            for row in s5_seed
        ),
        "no_schedule_seed_or_window_selection": True,
    }
    figures = figure_checks(stage)
    checks = {
        "lineage_and_tests": {"source_hashes_match": source_ok, **test_checks},
        "S4": s4_checks, "S5": s5_checks, "figures": figures,
    }
    flat = {f"{section}.{name}": passed for section, rows in checks.items() for name, passed in rows.items()}
    s4_status = "PASS" if all(s4_checks.values()) else "FAIL"
    s5_status = "PASS" if all(s5_checks.values()) else "FAIL_DIAGNOSTIC"
    overall = "PASS" if all(flat.values()) else "PARTIAL_OR_DIAGNOSTIC"
    summary = {
        "schema_version": 1, "status": overall, "paper_ready": False,
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner_sha256": core.sha256(Path(__file__)),
        "extension_sha256": core.sha256(ROOT / "active_hbm_v313_s4s5_v2.py"),
        "S4": {"status": s4_status, "cells": cells, "paired_wct_ci": paired},
        "S5": {"status": s5_status, "timeline_ci": s5_plot, "observed_recovery_stable_target_windows": observed_stable},
        "evidence_boundary": (
            "Mechanism-targeted three-seed preliminary DES. Workloads, benefit classes, "
            "capacity equation and schedule were preregistered without performance win thresholds. "
            "Bandwidth/service/failover remain modeled and are not physical A100 measurements."
        ),
    }
    (stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    s4_audit = {
        "schema_version": 1, "status": s4_status, "paper_ready": False,
        "checks": s4_checks, "failed_checks": [key for key, passed in s4_checks.items() if not passed],
        "protocol_sha256": summary["protocol_sha256"],
        "figure_pdf_sha256": core.sha256(stage / "figures/s4_targeted_ablation.pdf"),
    }
    s5_audit = {
        "schema_version": 1, "status": s5_status, "paper_ready": False,
        "checks": s5_checks, "failed_checks": [key for key, passed in s5_checks.items() if not passed],
        "protocol_sha256": summary["protocol_sha256"],
        "figure_pdf_sha256": core.sha256(stage / "figures/s5_safe_recovery_timeline.pdf"),
        "diagnostic_if_failed": "An analytically guarded C=1 recovery still failed to produce the registered mean-alpha rise; inspect per-target pressure/stable-window trace rather than changing the load.",
    }
    (stage / "s4_audit.json").write_text(json.dumps(s4_audit, indent=2, sort_keys=True) + "\n")
    (stage / "s5_audit.json").write_text(json.dumps(s5_audit, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "status": overall, "paper_ready": False,
        "S4": {"status": s4_status, "pdf": "figures/s4_targeted_ablation.pdf", "png": "figures/s4_targeted_ablation.png", "audits": "s4_audit.json"},
        "S5": {"status": s5_status, "pdf": "figures/s5_safe_recovery_timeline.pdf", "png": "figures/s5_safe_recovery_timeline.png", "audit": "s5_audit.json"},
        "protocol": str(SPEC_PATH), "protocol_sha256": summary["protocol_sha256"],
        "old_results_preserved": [
            "results/active_hbm_v313_s4_s5_preliminary_pilot",
            "results/active_hbm_v313_s4_s5_audit_bound_figures_v2"
        ],
        "evidence_boundary": summary["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema_version": 1, "status": overall, "paper_ready": False,
        "checks": checks, "failed_checks": [key for key, passed in flat.items() if not passed],
        "summary_sha256": core.sha256(stage / "summary.json"),
        "manifest_sha256": core.sha256(stage / "manifest.json"),
        "s4_audit_sha256": core.sha256(stage / "s4_audit.json"),
        "s5_audit_sha256": core.sha256(stage / "s5_audit.json"),
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    artifacts = {str(path.relative_to(stage)): core.sha256(path) for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"}
    (stage / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in artifacts.items()))
    os.replace(stage, result)
    print(json.dumps({
        "result": str(result), "status": overall,
        "s4_status": s4_status, "s5_status": s5_status,
        "audit_sha256": core.sha256(result / "audit.json"),
        "manifest_sha256": core.sha256(result / "manifest.json"),
        "failed_checks": audit["failed_checks"],
    }, indent=2), flush=True)
    if overall != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
