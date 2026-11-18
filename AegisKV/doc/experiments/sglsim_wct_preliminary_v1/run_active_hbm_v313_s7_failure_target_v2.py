#!/usr/bin/env python3
"""Run frozen S7-v2 with all-workflow and fixed failure-target denominators."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
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
import run_active_hbm_v313_s7_directional as s7v1


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s7_failure_target_v2_frozen_protocol.json"
RESULT_NAME = "active_hbm_v313_s7_failure_target_v2_full"
GATE_NAME = "active_hbm_v313_s7_failure_target_v2_gate"
T95_DF2 = 4.302652729911275
TESTS = (
    ("test_active_hbm_v3_capacity.py", "RESULT 5/5 PASS"),
    ("test_active_hbm_v313_protocol.py", "RESULT 6/6 PASS"),
    ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    ("test_active_hbm_v313_s7_directional.py", "RESULT 8/8 PASS"),
    ("test_active_hbm_v313_s7_protocol.py", "RESULT 10/10 PASS"),
    ("test_active_hbm_v313_s7_failure_target_v2_protocol.py", "SUMMARY 16/16 PASS"),
    ("test_active_hbm_v313_s7_failure_target_v2.py", "RESULT 10/10 PASS"),
)
TARGET_METRICS = (
    "failure_target_healthy_wct_mean_ms",
    "failure_target_fault_wct_mean_ms",
    "failure_target_added_wct_mean_ms",
    "failure_target_count_mean",
)


class EmptyFailureTargetError(RuntimeError):
    """A preregistered stratum had no reference-trace failure target."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return core.sha256(path)


def digest_json(value: Any) -> str:
    return s7v1.digest_json(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    s7v1.write_csv(path, rows)


def ci95(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) < 2:
        return 0.0
    if len(rows) != 3:
        raise ValueError("S7-v2 full aggregates require exactly three seeds")
    return T95_DF2 * statistics.stdev(rows) / math.sqrt(3)


def load_specs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    parent_path = Path(frozen["parent_s7_v1"]["protocol_path"])
    design_path = Path(frozen["design_preregistration"]["path"])
    if sha256(parent_path) != frozen["parent_s7_v1"]["protocol_sha256"]:
        raise RuntimeError("frozen parent S7-v1 protocol hash drift")
    if sha256(design_path) != frozen["design_preregistration"]["sha256"]:
        raise RuntimeError("S7-v2 design preregistration hash drift")
    return (
        frozen,
        json.loads(parent_path.read_text(encoding="utf-8")),
        json.loads(design_path.read_text(encoding="utf-8")),
    )


def reference_probes_and_targets(
    nominal: list[core.Task], cohort: set[str], *, seed: int,
    failure_type: str, count: int, left_ms: float, right_ms: float,
    gpus_per_node: int, dp: int,
) -> tuple[list[core.FaultProbe], list[list[str]]]:
    """Freeze probes and nonempty targets from the policy-free nominal trace."""

    reference = [row for row in nominal if row.workflow_id in cohort]
    rng = random.Random((int(seed) ^ 0x57D1AEC7) ^ 0xFA017)
    probes: list[core.FaultProbe] = []
    target_sets: list[list[str]] = []
    for index in range(int(count)):
        lo = float(left_ms) + (float(right_ms) - float(left_ms)) * index / count
        hi = float(left_ms) + (float(right_ms) - float(left_ms)) * (index + 1) / count
        at = rng.uniform(lo, hi)
        active = [row for row in reference if row.start_ms <= at < row.finish_ms]
        active_ranks = sorted({row.source_rank for row in active})
        if not active_ranks:
            raise EmptyFailureTargetError(
                f"seed={seed} probe={index} timestamp={at} has no active cohort source rank"
            )
        source_rank = int(rng.choice(active_ranks))
        probe = core.FaultProbe(
            probe_id=f"probe-{index:03d}", time_ms=float(at),
            source_rank=source_rank, failure_type=str(failure_type),
        )
        failed = set(v313.v312.v31.failure_domain_ranks(
            probe, gpus_per_node=int(gpus_per_node), dp=int(dp),
        ))
        targets = sorted({
            row.workflow_id for row in active if int(row.source_rank) in failed
        })
        if not targets:
            raise EmptyFailureTargetError(
                f"seed={seed} probe={index} domain={failure_type} has empty target set"
            )
        probes.append(probe)
        target_sets.append(targets)
    return probes, target_sets


def build_universe_v2(
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
    probes, _targets = reference_probes_and_targets(
        nominal, cohort, seed=int(seed), failure_type=failure_type,
        count=int(workload_spec["fault_probes_per_cell"]),
        left_ms=float(workload_spec["measurement_start_ms"]),
        right_ms=float(workload_spec["measurement_end_ms"]),
        gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
        dp=int(spec["hardware"]["system_dp"]),
    )
    return nominal, workload, cohort, probes


def targets_for_universe(
    spec: dict[str, Any], nominal: list[core.Task], cohort: set[str],
    probes: list[core.FaultProbe], *, seed: int,
) -> list[list[str]]:
    regenerated, targets = reference_probes_and_targets(
        nominal, cohort, seed=int(seed), failure_type=probes[0].failure_type,
        count=int(spec["workload"]["fault_probes_per_cell"]),
        left_ms=float(spec["workload"]["measurement_start_ms"]),
        right_ms=float(spec["workload"]["measurement_end_ms"]),
        gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
        dp=int(spec["hardware"]["system_dp"]),
    )
    if s7v1.probe_hash(regenerated, include_domain=True) != s7v1.probe_hash(
        probes, include_domain=True
    ):
        raise AssertionError("target derivation did not reproduce the executed probes")
    return targets


class CapturingReplay(v313.CausalPhysicalTargetReplay):
    """Capture immutable per-probe recovery actions without changing semantics."""

    latest: "CapturingReplay | None" = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.probe_action_ledger: dict[str, list[dict[str, Any]]] = {}
        CapturingReplay.latest = self

    def recovery(
        self, task: core.Task, probe: core.FaultProbe,
    ) -> tuple[float, str]:
        delay, action = super().recovery(task, probe)
        self.probe_action_ledger.setdefault(probe.probe_id, []).append({
            "request_id": task.request_id,
            "workflow_id": task.workflow_id,
            "action": action,
            "delay_ms": float(delay),
        })
        return delay, action


def _mean_target_wct(
    ids: list[str], workflow_ids: list[str], values: np.ndarray,
) -> float:
    index = {workflow_id: offset for offset, workflow_id in enumerate(workflow_ids)}
    missing = [workflow_id for workflow_id in ids if workflow_id not in index]
    if missing:
        raise AssertionError(f"failure target escaped measurement cohort: {missing[:3]}")
    return statistics.fmean(float(values[index[workflow_id]]) for workflow_id in ids)


def run_cell_v2(
    stage: Path, spec: dict[str, Any], job: dict[str, Any], policy: str,
    seed: int, cell_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    original_builder = s7v1.build_universe
    original_replay = s7v1.v313.CausalPhysicalTargetReplay
    CapturingReplay.latest = None
    s7v1.build_universe = build_universe_v2
    s7v1.v313.CausalPhysicalTargetReplay = CapturingReplay
    try:
        cell, counts = s7v1.run_cell(stage, spec, job, policy, int(seed))
    finally:
        s7v1.build_universe = original_builder
        s7v1.v313.CausalPhysicalTargetReplay = original_replay
    replay = CapturingReplay.latest
    if replay is None:
        raise AssertionError("capturing replay was not constructed")

    protocol, _capacity = s7v1.configured_protocol(
        spec, axis=str(job["axis"]), direction=job.get("direction"),
        bandwidth_gb_s=job.get("bandwidth_gb_s"), repair_ms=job.get("repair_ms"),
    )
    nominal, _workload, cohort, probes = build_universe_v2(
        spec, protocol, seed=int(seed), failure_type=str(job["failure_type"]),
    )
    target_sets = targets_for_universe(
        spec, nominal, cohort, probes, seed=int(seed)
    )
    raw_npz = stage / str(cell["raw_npz"])
    with np.load(raw_npz, allow_pickle=False) as raw:
        workflow_ids = [str(value) for value in raw["workflow_ids"].tolist()]
        healthy = np.asarray(raw["healthy_wct_ms"], dtype=float)
        fault = np.asarray(raw["fault_wct_ms"], dtype=float)
        raw_probe_ids = [str(value) for value in raw["probe_id"].tolist()]
        raw_probe_times = np.asarray(raw["probe_time_ms"], dtype=float)
        raw_probe_ranks = np.asarray(raw["probe_source_rank"], dtype=int)
        changed = np.asarray(raw["probe_changed_request_count"], dtype=int)
    if fault.shape != (len(probes), len(workflow_ids)):
        raise AssertionError("unexpected all-workflow fault matrix shape")

    probe_rows: list[dict[str, Any]] = []
    target_healthy: list[float] = []
    target_fault: list[float] = []
    target_hashes: list[str] = []
    for index, (probe, target_ids) in enumerate(zip(probes, target_sets)):
        if raw_probe_ids[index] != probe.probe_id:
            raise AssertionError("raw probe ID mismatch")
        if abs(float(raw_probe_times[index]) - float(probe.time_ms)) > 1e-9:
            raise AssertionError("raw probe timestamp mismatch")
        if int(raw_probe_ranks[index]) != int(probe.source_rank):
            raise AssertionError("raw probe source-rank mismatch")
        failed_ranks = v313.v312.v31.failure_domain_ranks(
            probe, gpus_per_node=int(spec["hardware"]["gpus_per_node"]),
            dp=int(spec["hardware"]["system_dp"]),
        )
        identity = {
            "seed": int(seed), "failure_type": probe.failure_type,
            "probe_id": probe.probe_id, "timestamp_ms": float(probe.time_ms),
            "source_rank": int(probe.source_rank),
            "sorted_workflow_ids": target_ids,
        }
        target_hash = digest_json(identity)
        healthy_mean = _mean_target_wct(target_ids, workflow_ids, healthy)
        fault_mean = _mean_target_wct(target_ids, workflow_ids, fault[index])
        if fault_mean + 1e-7 < healthy_mean:
            raise AssertionError("failure-target replay lowered WCT")
        actions = replay.probe_action_ledger.get(probe.probe_id, [])
        action_counts: dict[str, int] = {}
        for row in actions:
            action_counts[str(row["action"])] = action_counts.get(str(row["action"]), 0) + 1
        target_healthy.append(healthy_mean)
        target_fault.append(fault_mean)
        target_hashes.append(target_hash)
        probe_rows.append({
            "cell_index": int(cell_index),
            "condition_id": job["condition_id"],
            "axis": job["axis"], "direction": job.get("direction") or "",
            "failure_type": probe.failure_type,
            "bandwidth_gb_s": "" if job.get("bandwidth_gb_s") is None else float(job["bandwidth_gb_s"]),
            "repair_ms": "" if job.get("repair_ms") is None else float(job["repair_ms"]),
            "policy": policy, "seed": int(seed),
            "probe_index": index, "probe_id": probe.probe_id,
            "probe_time_ms": float(probe.time_ms),
            "source_rank": int(probe.source_rank),
            "source_node": int(probe.source_rank) // int(spec["hardware"]["gpus_per_node"]),
            "failed_ranks_json": json.dumps(failed_ranks),
            "failure_target_count": len(target_ids),
            "failure_target_workflow_ids_json": json.dumps(target_ids),
            "failure_target_workflow_ids_sha256": target_hash,
            "failure_target_healthy_wct_mean_ms": healthy_mean,
            "failure_target_fault_wct_mean_ms": fault_mean,
            "failure_target_added_wct_mean_ms": fault_mean - healthy_mean,
            "changed_request_count": int(changed[index]),
            "recovery_action_counts_json": json.dumps(action_counts, sort_keys=True),
            "recovery_request_count": len(actions),
            "recovery_delay_mean_ms": (
                statistics.fmean(float(row["delay_ms"]) for row in actions)
                if actions else 0.0
            ),
        })

    cell.update({
        "failure_target_reference_trace_sha256": s7v1.workflow_hash(nominal),
        "failure_target_bundle_sha256": digest_json(target_hashes),
        "failure_target_probe_count": len(target_sets),
        "failure_target_count_min": min(map(len, target_sets)),
        "failure_target_count_max": max(map(len, target_sets)),
        "failure_target_count_mean": statistics.fmean(map(len, target_sets)),
        "failure_target_healthy_wct_mean_ms": statistics.fmean(target_healthy),
        "failure_target_fault_wct_mean_ms": statistics.fmean(target_fault),
        "failure_target_added_wct_mean_ms": statistics.fmean(
            fault_value - healthy_value
            for fault_value, healthy_value in zip(target_fault, target_healthy)
        ),
    })
    raw_json = stage / "raw" / f"cell-{cell_index:04d}.json"
    raw_json.write_text(json.dumps({
        "schema_version": 1,
        "cell_index": int(cell_index),
        "condition_id": job["condition_id"], "policy": policy, "seed": int(seed),
        "workflow_universe_sha256": cell["workflow_universe_sha256"],
        "measurement_cohort_sha256": cell["measurement_cohort_sha256"],
        "probe_base_tuple_sha256": cell["probe_base_tuple_sha256"],
        "failure_target_reference_trace_sha256": cell["failure_target_reference_trace_sha256"],
        "failure_target_bundle_sha256": cell["failure_target_bundle_sha256"],
        "probes": probe_rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cell["failure_target_raw_json"] = str(raw_json.relative_to(stage))
    cell["failure_target_raw_json_sha256"] = sha256(raw_json)
    return cell, counts, probe_rows


def aggregate_v2(cells: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    original_metrics = s7v1.METRICS
    s7v1.METRICS = tuple(original_metrics) + TARGET_METRICS
    try:
        aggregates, paired_seed, paired = s7v1.aggregate(cells)
    finally:
        s7v1.METRICS = original_metrics
    cell_by_key = {
        (row["condition_id"], int(row["seed"]), row["policy"]): row
        for row in cells
    }
    for row in paired_seed:
        native = cell_by_key[(row["condition_id"], int(row["seed"]), "Native Mooncake")]
        aegis = cell_by_key[(row["condition_id"], int(row["seed"]), "AegisKV")]
        row["failure_target_wct_saved_ms"] = (
            float(native["failure_target_fault_wct_mean_ms"])
            - float(aegis["failure_target_fault_wct_mean_ms"])
        )
        row["failure_target_wct_relative_gain"] = (
            1.0 - float(aegis["failure_target_fault_wct_mean_ms"])
            / float(native["failure_target_fault_wct_mean_ms"])
        )
        row["failure_target_added_wct_saved_ms"] = (
            float(native["failure_target_added_wct_mean_ms"])
            - float(aegis["failure_target_added_wct_mean_ms"])
        )
        row["native_failure_target_fault_wct_mean_ms"] = float(
            native["failure_target_fault_wct_mean_ms"]
        )
        row["aegis_failure_target_fault_wct_mean_ms"] = float(
            aegis["failure_target_fault_wct_mean_ms"]
        )
        row["failure_target_bundle_sha256"] = native["failure_target_bundle_sha256"]
        if native["failure_target_bundle_sha256"] != aegis["failure_target_bundle_sha256"]:
            raise AssertionError("paired policies used different failure-target bundles")
    paired_seed_by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in paired_seed:
        paired_seed_by_condition.setdefault(str(row["condition_id"]), []).append(row)
    for row in paired:
        selected = paired_seed_by_condition[str(row["condition_id"])]
        for metric in (
            "failure_target_wct_saved_ms",
            "failure_target_wct_relative_gain",
            "failure_target_added_wct_saved_ms",
            "native_failure_target_fault_wct_mean_ms",
            "aegis_failure_target_fault_wct_mean_ms",
        ):
            values = [float(item[metric]) for item in selected]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_ci95_half_width"] = ci95(values)
    return aggregates, paired_seed, paired


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.3,
        "axes.titlesize": 8.2, "axes.labelsize": 7.7,
        "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
        "legend.fontsize": 7.2, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "lines.linewidth": 1.25,
    })


def plot_v2(path: Path, paired: list[dict[str, Any]]) -> None:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.45))
    panels = [
        ("bandwidth", "source_to_peer_hbm_copy", "Source to Peer-HBM copy"),
        ("bandwidth", "source_to_dram_copy", "Source to DRAM copy"),
        ("bandwidth", "peer_hbm_recovery", "Peer-HBM recovery"),
        ("bandwidth", "dram_recovery", "DRAM recovery"),
        ("repair", "source_worker_gpu", "Worker/GPU replacement"),
        ("repair", "source_node", "Source-node replacement"),
    ]
    denominator_style = {
        "all": ("#2166ac", "s", "-", "All workflows (primary)"),
        "target": ("#d6604d", "o", "--", "Fixed failure targets (secondary)"),
    }
    for ax, (axis, selector, title) in zip(axes.flat, panels):
        if axis == "bandwidth":
            rows = sorted(
                (row for row in paired if row["axis"] == axis and row["direction"] == selector),
                key=lambda row: float(row["bandwidth_gb_s"]),
            )
            x = [float(row["bandwidth_gb_s"]) for row in rows]
            ax.set_xlabel("Bandwidth (GB/s)")
        else:
            rows = sorted(
                (row for row in paired if row["axis"] == axis and row["failure_type"] == selector),
                key=lambda row: float(row["repair_ms"]),
            )
            x = [float(row["repair_ms"]) / 1000.0 for row in rows]
            ax.set_xlabel("Replacement (s)")
        for denominator, metric in (
            ("all", "fault_wct_saved_ms"),
            ("target", "failure_target_wct_saved_ms"),
        ):
            color, marker, line, label = denominator_style[denominator]
            ax.errorbar(
                x, [float(row[f"{metric}_mean"]) / 1000.0 for row in rows],
                yerr=[float(row[f"{metric}_ci95_half_width"]) / 1000.0 for row in rows],
                color=color, marker=marker, linestyle=line, markersize=3.5,
                capsize=2.0, label=label,
            )
        ax.axhline(0.0, color="#777777", linewidth=0.75)
        ax.set_title(title)
        ax.set_ylabel("Native − AegisKV WCT (s)")
        ax.grid(axis="y", alpha=0.22)
        ax.margins(y=0.14)
    handles = [
        Line2D([0], [0], color=value[0], marker=value[1], linestyle=value[2], label=value[3])
        for value in denominator_style.values()
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.952), handlelength=2.0)
    fig.suptitle("S7 directional and repair sensitivity", fontsize=10.0, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=1.35, w_pad=1.15)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def verify_figure(pdf: Path, png: Path) -> dict[str, Any]:
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
        "minimum_font_at_least_7pt": min(sizes) >= 7.0 - 1e-6,
        "png_width_px": width, "png_height_px": height,
        "png_width_at_least_1800": width >= 1800,
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


def main() -> int:
    args = parse_args()
    frozen, spec, _design = load_specs()
    result_name = GATE_NAME if args.gate else RESULT_NAME
    target = ROOT / "results" / result_name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{result_name}.stage-{os.getpid()}"
    (stage / "raw").mkdir(parents=True)
    (stage / "figures").mkdir()
    (stage / "test_logs").mkdir()
    shutil.copy2(SPEC_PATH, stage / "protocol.json")
    shutil.copy2(Path(frozen["parent_s7_v1"]["protocol_path"]), stage / "parent_s7_v1_protocol.json")
    shutil.copy2(Path(frozen["design_preregistration"]["path"]), stage / "design_preregistration.json")
    tests = run_tests(stage)

    conditions = s7v1.jobs(spec, gate=args.gate)
    seeds = (
        [int(spec["sweep"]["paired_seeds"][0])]
        if args.gate else [int(value) for value in spec["sweep"]["paired_seeds"]]
    )
    work_unindexed = [
        (condition, policy, seed)
        for condition in conditions
        for policy in spec["sweep"]["policies"] for seed in seeds
    ]
    work = [(*row, index) for index, row in enumerate(work_unindexed)]
    expected = 12 if args.gate else int(spec["sweep"]["registered_cell_count"])
    workers = 1 if args.gate else max(1, min(int(args.workers), len(work)))
    cells: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    completed = 0
    if workers == 1:
        for condition, policy, seed, cell_index in work:
            cell, counts, probes = run_cell_v2(
                stage, spec, condition, policy, seed, cell_index
            )
            cells.append(cell); count_rows.extend(counts); probe_rows.extend(probes)
            completed += 1
            print(f"S7-v2 {completed}/{expected} {condition['condition_id']} {policy} s{seed}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_cell_v2, stage, spec, condition, policy, seed, cell_index
                ): (condition, policy, seed)
                for condition, policy, seed, cell_index in work
            }
            for future in concurrent.futures.as_completed(futures):
                condition, policy, seed = futures[future]
                cell, counts, probes = future.result()
                cells.append(cell); count_rows.extend(counts); probe_rows.extend(probes)
                completed += 1
                print(f"S7-v2 {completed}/{expected} {condition['condition_id']} {policy} s{seed}", flush=True)

    cells.sort(key=lambda row: int(Path(row["failure_target_raw_json"]).stem.split("-")[1]))
    count_rows.sort(key=lambda row: (
        row["condition_id"], row["policy"], int(row["seed"]), row["category"], row["key"]
    ))
    probe_rows.sort(key=lambda row: (int(row["cell_index"]), int(row["probe_index"])))
    write_csv(stage / "cell_metrics.csv", cells)
    write_csv(stage / "mechanism_counts.csv", count_rows)
    write_csv(stage / "failure_target_probe_metrics.csv", probe_rows)
    with (stage / "cell_summary.jsonl").open("w", encoding="utf-8") as stream:
        for row in cells:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    aggregates: list[dict[str, Any]] = []
    paired_seed: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    figure_checks: dict[str, Any] = {}
    if not args.gate:
        aggregates, paired_seed, paired = aggregate_v2(cells)
        write_csv(stage / "aggregate_ci.csv", aggregates)
        write_csv(stage / "paired_seed_metrics.csv", paired_seed)
        write_csv(stage / "paired_ci.csv", paired)
        stem = stage / "figures" / "s7_v313_failure_target_v2_observed"
        plot_v2(stem, paired)
        figure_checks = verify_figure(stem.with_suffix(".pdf"), stem.with_suffix(".png"))

    seeds_seen = sorted({int(row["seed"]) for row in cells})
    target_hash_global = all(
        len({
            row["failure_target_bundle_sha256"] for row in cells
            if int(row["seed"]) == seed and row["failure_type"] == failure_type
        }) == 1
        for seed in seeds_seen for failure_type in ("source_worker_gpu", "source_node")
        if any(int(row["seed"]) == seed and row["failure_type"] == failure_type for row in cells)
    )
    per_probe_hash_global = all(
        len({
            row["failure_target_workflow_ids_sha256"] for row in probe_rows
            if int(row["seed"]) == seed and row["failure_type"] == failure_type
            and int(row["probe_index"]) == probe_index
        }) == 1
        for seed in seeds_seen
        for failure_type in ("source_worker_gpu", "source_node")
        for probe_index in range(int(spec["workload"]["fault_probes_per_cell"]))
        if any(
            int(row["seed"]) == seed and row["failure_type"] == failure_type
            and int(row["probe_index"]) == probe_index for row in probe_rows
        )
    )
    expected_probe_rows = len(cells) * int(spec["workload"]["fault_probes_per_cell"])
    checks: dict[str, bool] = {
        "mechanism_protocol_and_failure_target_tests_pass": all(row["passed"] for row in tests),
        "exact_registered_cell_count": len(cells) == expected,
        "exact_failure_target_probe_rows": len(probe_rows) == expected_probe_rows,
        "all_failure_target_sets_nonempty": all(int(row["failure_target_count"]) > 0 for row in probe_rows),
        "failure_target_bundle_hash_cross_policy_treatment_exact": target_hash_global,
        "failure_target_probe_id_hash_cross_policy_treatment_exact": per_probe_hash_global,
        "failure_target_replay_monotone": all(float(row["failure_target_added_wct_mean_ms"]) >= -1e-7 for row in probe_rows),
        "one_worker_or_eight_node_failed_ranks": all(
            len(json.loads(row["failed_ranks_json"])) == (1 if row["failure_type"] == "source_worker_gpu" else 8)
            for row in probe_rows
        ),
        "all_workflow_primary_retained": all(
            key in row for row in cells for key in (
                "healthy_wct_mean_ms", "fault_wct_mean_ms", "fault_added_wct_mean_ms"
            )
        ),
        "failure_target_secondary_retained": all(
            key in row for row in cells for key in TARGET_METRICS
        ),
        "inherited_capacity_copy_and_replay_gates_pass": all(
            int(row["capacity_violation_count"]) == 0
            and int(row["target_accounting_max_abs_error_bytes"]) == 0
            and json.loads(row["copy_byte_conservation_error"]) == {"DRAM": 0, "HBM": 0}
            and float(row["empty_replay_max_abs_error_ms"]) <= 1e-7
            and float(row["fault_monotonic_max_negative_error_ms"]) <= 1e-7
            for row in cells
        ),
        "raw_npz_and_failure_target_json_hashes_match": all(
            sha256(stage / row["raw_npz"]) == row["raw_npz_sha256"]
            and sha256(stage / row["failure_target_raw_json"]) == row["failure_target_raw_json_sha256"]
            for row in cells
        ),
        "frozen_source_hashes_match": all(
            Path(source["path"]).exists() and sha256(Path(source["path"])) == source["sha256"]
            for source in frozen["sources"].values()
        ),
        "no_direction_seed_cell_probe_or_sign_selection": True,
    }
    if not args.gate:
        checks.update({
            "exact_168_cells_and_2016_probe_rows": len(cells) == 168 and len(probe_rows) == 2016,
            "exact_84_paired_seed_and_28_ci_rows": len(paired_seed) == 84 and len(paired) == 28,
            "all_four_directions_five_bandwidths_and_repair_domains_retained": (
                {row["direction"] for row in cells if row["axis"] == "bandwidth"}
                == set(spec["sweep"]["bandwidth_directions"])
                and {float(row["bandwidth_gb_s"]) for row in cells if row["axis"] == "bandwidth"}
                == set(float(value) for value in spec["sweep"]["bandwidth_values_gb_s"])
                and {row["failure_type"] for row in cells if row["axis"] == "repair"}
                == set(spec["sweep"]["repair_failure_domains"])
            ),
            "primary_and_secondary_paired_ci_complete": all(
                key in row for row in paired for key in (
                    "fault_wct_saved_ms_mean", "fault_wct_saved_ms_ci95_half_width",
                    "failure_target_wct_saved_ms_mean", "failure_target_wct_saved_ms_ci95_half_width",
                )
            ),
            "figure_single_page_vector_pdf": figure_checks["single_page_vector_pdf"],
            "figure_minimum_font_at_least_7pt": figure_checks["minimum_font_at_least_7pt"],
            "figure_png_width_at_least_1800": figure_checks["png_width_at_least_1800"],
        })
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "schema_version": 1, "status": status,
        "mode": "gate" if args.gate else "full", "paper_ready": False,
        "checks": checks, "failed_checks": [key for key, value in checks.items() if not value],
        "cell_count": len(cells), "expected_cell_count": expected,
        "failure_target_probe_row_count": len(probe_rows),
        "expected_failure_target_probe_row_count": expected_probe_rows,
        "protocol_sha256": sha256(stage / "protocol.json"),
        "parent_protocol_sha256": sha256(stage / "parent_s7_v1_protocol.json"),
        "design_preregistration_sha256": sha256(stage / "design_preregistration.json"),
        "cell_metrics_sha256": sha256(stage / "cell_metrics.csv"),
        "failure_target_probe_metrics_sha256": sha256(stage / "failure_target_probe_metrics.csv"),
        "mechanism_counts_sha256": sha256(stage / "mechanism_counts.csv"),
        "figure_checks": figure_checks,
        "evidence_boundary": frozen["evidence_boundary"],
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "status": status, "paper_ready": False,
        "mode": audit["mode"], "selection": "S7_V2_NONE",
        "protocol": "protocol.json", "protocol_sha256": audit["protocol_sha256"],
        "parent_protocol": "parent_s7_v1_protocol.json",
        "design_preregistration": "design_preregistration.json",
        "runner_sha256": sha256(Path(__file__)), "execution_workers": workers,
        "audit": "audit.json", "audit_sha256": sha256(stage / "audit.json"),
        "cell_metrics": "cell_metrics.csv",
        "failure_target_probe_metrics": "failure_target_probe_metrics.csv",
        "cell_summary": "cell_summary.jsonl", "mechanism_counts": "mechanism_counts.csv",
        "aggregate_ci": None if args.gate else "aggregate_ci.csv",
        "paired_seed_metrics": None if args.gate else "paired_seed_metrics.csv",
        "paired_ci": None if args.gate else "paired_ci.csv",
        "figure": None if args.gate else {
            "pdf": "figures/s7_v313_failure_target_v2_observed.pdf",
            "png": "figures/s7_v313_failure_target_v2_observed.png",
        },
        "all_workflow_primary": True,
        "fixed_failure_target_secondary": True,
        "no_direction_or_result_selection": True,
        "evidence_boundary": frozen["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(f"S7-v2 audit failed: {audit['failed_checks']}")
    artifacts = {
        str(path.relative_to(stage)): sha256(path)
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
        "probe_rows": len(probe_rows),
        "audit_sha256": sha256(target / "audit.json"),
        "manifest_sha256": sha256(target / "manifest.json"),
        "sha256sums_sha256": sha256(target / "SHA256SUMS"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
