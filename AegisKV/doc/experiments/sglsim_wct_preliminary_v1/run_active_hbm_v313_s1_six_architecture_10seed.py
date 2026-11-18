#!/usr/bin/env python3
"""Preregistered ten-seed, six-architecture S1 failure-conditioned WCT suite."""

from __future__ import annotations

import argparse
import csv
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
import run_preliminary as core
import test_active_hbm_v313 as test_inputs


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s1_six_architecture_10seed_protocol.json"
T95_9DF = 2.2621571627409915


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def workload_hash(tasks: Iterable[core.Task]) -> str:
    return digest([
        (
            row.request_id, row.workflow_id, row.workflow_arrival_ms,
            row.arrival_ms, row.source_rank, row.service_ms,
            row.prompt_tokens, row.output_tokens, row.predicted_slack_ms,
        )
        for row in tasks
    ])


def probe_hash(probes: Iterable[core.FaultProbe]) -> str:
    return digest([
        (row.probe_id, row.time_ms, row.source_rank, row.failure_type)
        for row in probes
    ])


def domain_set_hash(probes: Iterable[core.FaultProbe]) -> str:
    return digest([
        (
            row.probe_id,
            row.failure_type,
            tuple(v31.failure_domain_ranks(row, gpus_per_node=8, dp=16)),
        )
        for row in probes
    ])


def percentile(values: Iterable[float], q: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=float), q))


def stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("empty statistic")
    return {
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "max_ms": float(np.max(array)),
    }


def configured_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, capacity = test_inputs.inputs(context=40960, active=16)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["hardware"]["gpus_per_node"] = 8
    protocol["hardware"]["node_count"] = 2
    protocol["workload"]["center_context_tokens"] = 40960
    protocol["workload"]["center_active_requests_per_gpu"] = 16
    protocol["workload"]["reuse_fraction"] = 0.0
    protocol.setdefault("runtime_overrides", {})["record_occupancy_trace"] = False
    return protocol, capacity


def make_probes(
    tasks: list[core.Task], *, seed: int, per_domain: int, horizon_ms: float,
) -> list[core.FaultProbe]:
    probes = core.make_fault_probes(
        tasks,
        seed=int(seed) ^ 0x5A1F311,
        count=int(per_domain) * 2,
        gpus_per_node=8,
        right_ms=float(horizon_ms),
        failure_types=("source_worker_gpu", "source_node"),
    )
    for domain in ("source_worker_gpu", "source_node"):
        if sum(row.failure_type == domain for row in probes) != int(per_domain):
            raise AssertionError(f"{domain} probe count mismatch")
    return probes


def slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_")


def mechanism_summary(mechanism: dict[str, Any]) -> dict[str, Any]:
    v3 = mechanism["active_hbm_v3"]
    v313m = mechanism["active_hbm_v313"]
    v312m = mechanism["active_hbm_v312"]
    return {
        "scheduled_control_opportunity_times_ms": v313m[
            "scheduled_control_opportunity_times_ms"
        ],
        "executed_policy_control_times_ms": v313m[
            "executed_policy_control_times_ms"
        ],
        "control_window_count": int(mechanism["control_window_count"]),
        "control_target_evaluations": int(mechanism["control_target_evaluations"]),
        "pressure_target_evaluation_fraction": mechanism[
            "pressure_target_evaluation_fraction"
        ],
        "alpha_final_mean": mechanism["alpha_final_mean"],
        "capacity_violation_count": int(v3["capacity_violation_count"])
        + int(v313m["target_capacity_violation_count"]),
        "capacity_min_headroom_bytes": int(v3["capacity_min_headroom_bytes"]),
        "memory_blocked_unique_request_count": int(
            v3["memory_blocked_unique_request_count"]
        ),
        "queue_wait_mean_ms": float(v3["queue_wait_mean_ms"]),
        "queue_wait_p95_ms": float(v3["queue_wait_p95_ms"]),
        "copy_byte_conservation_error": v313m["copy_byte_conservation_error"],
        "copy_duration_formula_max_abs_error_ms": float(
            v313m["copy_duration_formula_max_abs_error_ms"]
        ),
        "copy_critical_path_sum_violation_count": int(
            v313m["copy_critical_path_sum_violation_count"]
        ),
        "health_burden_application_count": int(
            v313m["health_burden_application_count"]
        ),
        "target_accounting_max_abs_error_bytes": int(
            v313m["target_accounting_max_abs_error_bytes"]
        ),
        "future_backup_projection_dispatch_count": int(
            v313m["future_backup_projection_dispatch_count"]
        ),
        "seal_before_dispatch_count": int(v313m["seal_before_dispatch_count"]),
        "seal_uses_realized_output_or_service": bool(
            v313m["seal_uses_realized_output_or_service"]
        ),
        "primary_priority_reclaimed_bytes": int(
            v313m["primary_priority_reclaimed_bytes"]
        ),
        "path_event_counts": mechanism["path_event_counts"],
        "residency_event_counts": mechanism["residency_event_counts"],
        "recovery_path_counts": mechanism["recovery_path_counts"],
        "recovery_type_path_counts": mechanism["recovery_type_path_counts"],
        "failover_mode_counts": mechanism["failover_mode_counts"],
        "surviving_cross_domain_committed_count": int(
            v312m["surviving_cross_domain_committed_count"]
        ),
        "node_prebound_survivor_count": int(
            v312m["node_prebound_survivor_count"]
        ),
        "node_replacement_no_survivor_count": int(
            v312m["node_replacement_no_survivor_count"]
        ),
    }


def execute_cell(job: dict[str, Any]) -> dict[str, Any]:
    stage = Path(job["stage"])
    display = str(job["display_policy"])
    internal = str(job["internal_policy"])
    seed = int(job["seed"])
    horizon = float(job["horizon_ms"])
    per_domain = int(job["probes_per_domain"])
    protocol, capacity = configured_protocol()
    tasks, workload = v31.build_workload(
        protocol,
        seed=seed,
        active_per_gpu=16,
        arrival_cutoff_ms=horizon,
    )
    if workload["slack_predictor"]["reads_realized_output_tokens"]:
        raise AssertionError("old oracle workload lineage reached S1")
    probes = make_probes(tasks, seed=seed, per_domain=per_domain, horizon_ms=horizon)
    universe_sha = workload_hash(tasks)
    probes_sha = probe_hash(probes)
    domains_sha = domain_set_hash(probes)

    replay = v313.CausalPhysicalTargetReplay(deepcopy(protocol), internal, capacity)
    scheduled, delay_maps, mechanism, capacity_trace = replay.run_coupled(
        tasks, probes, control_horizon_ms=horizon
    )
    workflow_ids, healthy_wct = core.workflow_wct(scheduled)
    empty = core.replay_one_fault(scheduled, {})
    empty_delta = max(
        abs(float(empty[row.request_id]) - float(row.finish_ms))
        for row in scheduled
    )
    matrices: dict[str, list[list[float]]] = {
        "source_worker_gpu": [], "source_node": []
    }
    probe_rows = []
    minimum_fault_delta = math.inf
    for probe, delays in zip(probes, delay_maps):
        completion = core.replay_one_fault(scheduled, delays)
        ids, values = core.workflow_wct(scheduled, completion)
        if ids != workflow_ids:
            raise AssertionError("fault changed workflow denominator")
        deltas = [
            float(value) - float(reference)
            for value, reference in zip(values, healthy_wct)
        ]
        minimum_fault_delta = min(minimum_fault_delta, min(deltas))
        if min(deltas) < -1e-7:
            raise AssertionError("fault lowered WCT")
        matrices[probe.failure_type].append(values)
        row_stats = stats(values)
        probe_rows.append({
            "display_policy": display,
            "internal_policy": internal,
            "workload_seed": seed,
            "probe_id": probe.probe_id,
            "probe_time_ms": probe.time_ms,
            "source_rank": probe.source_rank,
            "failure_domain": probe.failure_type,
            "failure_domain_rank_ids": list(v31.failure_domain_ranks(
                probe, gpus_per_node=8, dp=16
            )),
            "affected_request_count": len(delays),
            "changed_workflow_count": sum(delta > 1e-7 for delta in deltas),
            **row_stats,
        })

    raw_dir = stage / "raw" / f"s{seed}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{slug(display)}.npz"
    np.savez_compressed(
        raw_path,
        workflow_ids=np.asarray(workflow_ids),
        healthy_wct_ms=np.asarray(healthy_wct, dtype=float),
        worker_fault_wct_ms=np.asarray(matrices["source_worker_gpu"], dtype=float),
        node_fault_wct_ms=np.asarray(matrices["source_node"], dtype=float),
        probe_ids=np.asarray([row.probe_id for row in probes]),
        probe_time_ms=np.asarray([row.time_ms for row in probes], dtype=float),
        probe_source_rank=np.asarray([row.source_rank for row in probes], dtype=np.int16),
        probe_failure_domain=np.asarray([row.failure_type for row in probes]),
        capacity_time_ms=np.asarray([row["time_ms"] for row in capacity_trace], dtype=float),
        capacity_rank=np.asarray([row["rank"] for row in capacity_trace], dtype=np.int16),
        capacity_serving_bytes=np.asarray(
            [row["active_serving_kv_bytes"] for row in capacity_trace], dtype=np.int64
        ),
        capacity_peer_backup_bytes=np.asarray(
            [row["peer_hbm_backup_used_bytes"] for row in capacity_trace], dtype=np.int64
        ),
    )
    cell = {
        "schema_version": 1,
        "display_policy": display,
        "internal_policy": internal,
        "workload_seed": seed,
        "workload_metadata": workload,
        "workload_universe_sha256": universe_sha,
        "fault_probe_tuple_sha256": probes_sha,
        "failure_domain_rank_sets_sha256": domains_sha,
        "workflow_count": len(workflow_ids),
        "request_count": len(scheduled),
        "healthy_wct": stats(healthy_wct),
        "empty_fault_replay_max_completion_delta_ms": empty_delta,
        "minimum_fault_minus_healthy_workflow_wct_ms": minimum_fault_delta,
        "probe_metrics": probe_rows,
        "domain_metrics": {
            domain: {
                **stats(np.asarray(matrix, dtype=float).reshape(-1)),
                "probe_count": len(matrix),
                "probe_workflow_observation_count": int(
                    np.asarray(matrix, dtype=float).size
                ),
                "changed_workflow_observation_count": sum(
                    row["changed_workflow_count"]
                    for row in probe_rows if row["failure_domain"] == domain
                ),
            }
            for domain, matrix in matrices.items()
        },
        "mechanism": mechanism_summary(mechanism),
        "raw_npz": str(raw_path.relative_to(stage)),
        "raw_npz_sha256": core.sha256(raw_path),
    }
    cell_path = raw_dir / f"{slug(display)}.json"
    cell_path.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    return {
        "display_policy": display,
        "internal_policy": internal,
        "workload_seed": seed,
        "cell_json": str(cell_path.relative_to(stage)),
        "cell_json_sha256": core.sha256(cell_path),
        "raw_npz": cell["raw_npz"],
        "raw_npz_sha256": cell["raw_npz_sha256"],
        "status": "PASS",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV {path}")
    fields = []
    seen = set()
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
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def ci95(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean, 0.0
    se = statistics.stdev(values) / math.sqrt(len(values))
    critical = T95_9DF if len(values) == 10 else 1.96
    half = critical * se
    return mean, mean - half, mean + half, half


def summarize(cells: list[dict[str, Any]], names: list[str]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    seed_rows = []
    probe_rows = []
    for cell in cells:
        for row in cell["probe_metrics"]:
            probe_rows.append(row)
        for domain, metrics in cell["domain_metrics"].items():
            seed_rows.append({
                "display_policy": cell["display_policy"],
                "internal_policy": cell["internal_policy"],
                "workload_seed": cell["workload_seed"],
                "failure_domain": domain,
                "workflow_count": cell["workflow_count"],
                "probe_count": metrics["probe_count"],
                "probe_workflow_observation_count": metrics[
                    "probe_workflow_observation_count"
                ],
                "changed_workflow_observation_count": metrics[
                    "changed_workflow_observation_count"
                ],
                "wct_mean_ms": metrics["mean_ms"],
                "wct_p50_ms": metrics["p50_ms"],
                "wct_p95_ms": metrics["p95_ms"],
                "wct_p99_ms": metrics["p99_ms"],
                "healthy_wct_mean_ms": cell["healthy_wct"]["mean_ms"],
                "workload_universe_sha256": cell["workload_universe_sha256"],
                "fault_probe_tuple_sha256": cell["fault_probe_tuple_sha256"],
                "failure_domain_rank_sets_sha256": cell[
                    "failure_domain_rank_sets_sha256"
                ],
                "raw_npz_sha256": cell["raw_npz_sha256"],
            })
    lookup = {
        (row["workload_seed"], row["failure_domain"], row["display_policy"]): row
        for row in seed_rows
    }
    paired_seed = []
    seeds = sorted({int(row["workload_seed"]) for row in seed_rows})
    domains = ["source_worker_gpu", "source_node"]
    for seed in seeds:
        for domain in domains:
            native = lookup[(seed, domain, "Native Mooncake")]
            for name in names[1:]:
                current = lookup[(seed, domain, name)]
                row = {
                    "display_policy": name,
                    "workload_seed": seed,
                    "failure_domain": domain,
                    "paired_universe_hash_equal": (
                        native["workload_universe_sha256"]
                        == current["workload_universe_sha256"]
                    ),
                    "paired_probe_hash_equal": (
                        native["fault_probe_tuple_sha256"]
                        == current["fault_probe_tuple_sha256"]
                    ),
                    "paired_domain_set_hash_equal": (
                        native["failure_domain_rank_sets_sha256"]
                        == current["failure_domain_rank_sets_sha256"]
                    ),
                }
                for metric in ("mean", "p50", "p95", "p99"):
                    key = f"wct_{metric}_ms"
                    nvalue = float(native[key])
                    value = float(current[key])
                    row[f"native_{key}"] = nvalue
                    row[f"architecture_{key}"] = value
                    row[f"paired_{metric}_delta_ms"] = value - nvalue
                    row[f"paired_{metric}_gain_pct"] = (
                        (nvalue - value) / nvalue * 100.0
                    )
                paired_seed.append(row)
    paired_ci = []
    for domain in domains:
        for name in names[1:]:
            rows = [
                row for row in paired_seed
                if row["failure_domain"] == domain
                and row["display_policy"] == name
            ]
            result = {
                "display_policy": name,
                "failure_domain": domain,
                "paired_seed_count": len(rows),
            }
            for metric in ("mean", "p50", "p95", "p99"):
                for kind in ("delta_ms", "gain_pct"):
                    key = f"paired_{metric}_{kind}"
                    mean, low, high, half = ci95([float(row[key]) for row in rows])
                    result[f"{key}_mean"] = mean
                    result[f"{key}_ci95_low"] = low
                    result[f"{key}_ci95_high"] = high
                    result[f"{key}_ci95_half_width"] = half
            paired_ci.append(result)
    architecture_ci = []
    for domain in domains:
        for name in names:
            rows = [
                row for row in seed_rows
                if row["failure_domain"] == domain
                and row["display_policy"] == name
            ]
            result = {
                "display_policy": name,
                "failure_domain": domain,
                "seed_count": len(rows),
            }
            for metric in ("mean", "p50", "p95", "p99"):
                key = f"wct_{metric}_ms"
                mean, low, high, half = ci95([float(row[key]) for row in rows])
                result[f"{key}_mean"] = mean
                result[f"{key}_ci95_low"] = low
                result[f"{key}_ci95_high"] = high
                result[f"{key}_ci95_half_width"] = half
            architecture_ci.append(result)
    return seed_rows, probe_rows, paired_seed, paired_ci, architecture_ci


def build_cdf_source(
    stage: Path, results: list[dict[str, Any]], names: list[str],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    arrays: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for result in results:
        with np.load(stage / result["raw_npz"], allow_pickle=False) as raw:
            arrays[(result["display_policy"], "source_worker_gpu")].append(
                np.asarray(raw["worker_fault_wct_ms"], dtype=float).reshape(-1)
            )
            arrays[(result["display_policy"], "source_node")].append(
                np.asarray(raw["node_fault_wct_ms"], dtype=float).reshape(-1)
            )
    pooled = {key: np.concatenate(value) for key, value in arrays.items()}
    source = []
    for domain in ("source_worker_gpu", "source_node"):
        for name in names:
            values = np.sort(pooled[(name, domain)])
            count = len(values)
            if count <= 4000:
                indices = np.arange(count, dtype=int)
            else:
                indices = np.unique(np.linspace(0, count - 1, 4000).astype(int))
            for index in indices:
                source.append({
                    "display_policy": name,
                    "failure_domain": domain,
                    "pooled_observation_count": count,
                    "render_index": int(index),
                    "wct_ms": float(values[index]),
                    "ecdf": float(index + 1) / count,
                    "render_decimation_only": count > len(indices),
                })
    return source, pooled


def render_figure(
    stage: Path, cdf_source: list[dict[str, Any]], names: list[str],
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import fitz

    colors = {
        "Native Mooncake": "#222222", "Static": "#8c564b",
        "DejaVu": "#1f77b4", "KevlarFlow": "#9467bd",
        "GhostServe": "#2ca02c", "AegisKV": "#d62728",
    }
    styles = {
        "Native Mooncake": "-", "Static": "--", "DejaVu": "-.",
        "KevlarFlow": ":", "GhostServe": (0, (5, 2)),
        "AegisKV": "-",
    }
    plt.rcParams.update({
        "font.size": 8.0, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for axis, domain, title in zip(
        axes,
        ("source_worker_gpu", "source_node"),
        ("Worker / GPU failure", "Source-node failure"),
    ):
        for name in names:
            rows = [
                row for row in cdf_source
                if row["display_policy"] == name
                and row["failure_domain"] == domain
            ]
            axis.plot(
                np.asarray([row["wct_ms"] for row in rows]) / 1000.0,
                [row["ecdf"] for row in rows],
                label=name, color=colors[name], linestyle=styles[name],
                linewidth=1.4 if name != "AegisKV" else 1.8,
            )
        axis.set_title(title)
        axis.set_xlabel("Workflow completion time (s)")
        axis.set_ylim(0.0, 1.01)
        axis.grid(True, alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel("Empirical CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, 1.01), columnspacing=0.9, handlelength=2.0,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.82, wspace=0.18)
    figure_dir = stage / "figures"
    figure_dir.mkdir(exist_ok=True)
    pdf = figure_dir / "s1_v313_six_architecture_10seed_observed.pdf"
    png = figure_dir / "s1_v313_six_architecture_10seed_observed.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    width, height = Image.open(png).size
    document = fitz.open(pdf)
    sizes = []
    for page in document:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        sizes.append(float(span["size"]))
    document.close()
    return {
        "pdf": str(pdf.relative_to(stage)),
        "png": str(png.relative_to(stage)),
        "png_width_px": width,
        "png_height_px": height,
        "pdf_min_font_pt": min(sizes),
    }


def source_hash_checks(spec: dict[str, Any]) -> dict[str, bool]:
    output = {}
    for name, source in spec["sources"].items():
        if "sha256" not in source:
            continue
        path = Path(source["path"])
        if not path.is_absolute():
            path = ROOT / path
        output[f"source_sha256_{name}"] = (
            path.is_file() and core.sha256(path) == source["sha256"]
        )
    return output


def run_tests(stage: Path) -> list[dict[str, Any]]:
    tests = (
        ("test_active_hbm_v313_s1_six_architecture_protocol.py", "RESULT 13/13 PASS"),
        ("test_active_hbm_v313_protocol.py", "RESULT 6/6 PASS"),
        ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    )
    test_dir = stage / "tests"
    test_dir.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment["TMPDIR"] = str(stage / "tmp")
    environment["MPLCONFIGDIR"] = str(stage / "mplconfig")
    (stage / "tmp").mkdir(exist_ok=True)
    (stage / "mplconfig").mkdir(exist_ok=True)
    output = []
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
            "test": filename, "marker": marker,
            "returncode": result.returncode, "passed": passed,
            "log": str(log.relative_to(stage)),
        })
        if not passed:
            raise RuntimeError(f"test gate failed: {filename}\n{result.stdout}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "full"), required=True)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    names = list(spec["display_policy_order"])
    mapping = dict(spec["display_to_internal_policy"])
    if args.mode == "gate":
        seeds = [int(value) for value in spec["gate"]["workload_seeds"]]
        per_domain = int(spec["gate"]["probes_per_domain"])
        target_name = "active_hbm_v313_s1_six_architecture_10seed_gate"
    else:
        seeds = [int(value) for value in spec["workload"]["paired_workload_seeds"]]
        per_domain = int(spec["failure_conditioning"]["probes_per_domain_per_seed"])
        target_name = "active_hbm_v313_s1_six_architecture_10seed_v1"
    horizon = float(spec["workload"]["arrival_horizon_ms"])
    target = ROOT / "results" / target_name
    if target.exists():
        raise FileExistsError(target)
    stage = target.parent / f".{target_name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()
    tests = run_tests(stage)
    source_checks = source_hash_checks(spec)
    if not all(source_checks.values()):
        raise RuntimeError([key for key, value in source_checks.items() if not value])
    freeze = {
        "schema_version": 1,
        "generated_before_policy_replay": True,
        "mode": args.mode,
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner_sha256": core.sha256(Path(__file__)),
        "workload_builder": "active_hbm_v31.build_workload",
        "old_core_build_workload_lineage_rejected": True,
        "predicted_slack_in_workload_universe_hash": True,
        "display_policy_order": names,
        "display_to_internal_policy": mapping,
        "workload_seeds": seeds,
        "probes_per_domain": per_domain,
        "arrival_horizon_ms": horizon,
        "control_opportunities_ms": [30000.0, 60000.0, 90000.0],
        "selection": spec["selection"],
        "source_hash_checks": source_checks,
    }
    (stage / "execution_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    jobs = [
        {
            "stage": str(stage),
            "display_policy": display,
            "internal_policy": mapping[display],
            "seed": seed,
            "horizon_ms": horizon,
            "probes_per_domain": per_domain,
        }
        for seed in seeds for display in names
    ]
    max_workers = min(max(int(args.workers), 1), 50, len(jobs))
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(execute_cell, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"S1 {args.mode} {index}/{len(jobs)} seed={result['workload_seed']} "
                    f"policy={result['display_policy']}", flush=True,
                )
            except Exception as error:
                failures.append({
                    "display_policy": job["display_policy"],
                    "workload_seed": job["seed"],
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                })
    (stage / "job_results.json").write_text(
        json.dumps(sorted(results, key=lambda row: (
            row["workload_seed"], names.index(row["display_policy"])
        )), indent=2, sort_keys=True) + "\n"
    )
    (stage / "job_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; stage retained at {stage}")

    cells = []
    for result in results:
        path = stage / result["cell_json"]
        if core.sha256(path) != result["cell_json_sha256"]:
            raise AssertionError("cell JSON changed after worker return")
        if core.sha256(stage / result["raw_npz"]) != result["raw_npz_sha256"]:
            raise AssertionError("raw NPZ changed after worker return")
        cells.append(json.loads(path.read_text()))
    seed_rows, probe_rows, paired_seed, paired_ci, architecture_ci = summarize(
        cells, names
    )
    write_csv(stage / "seed_policy_domain_metrics.csv", seed_rows)
    write_csv(stage / "probe_metrics.csv", probe_rows)
    write_csv(stage / "paired_vs_native_seed_metrics.csv", paired_seed)
    write_csv(stage / "paired_vs_native_ci.csv", paired_ci)
    write_csv(stage / "architecture_ci.csv", architecture_ci)

    figure = {}
    cdf_source = []
    pooled = {}
    if args.mode == "full":
        cdf_source, pooled = build_cdf_source(stage, results, names)
        write_csv(stage / "cdf_render_source.csv", cdf_source)
        figure = render_figure(stage, cdf_source, names)

    by_seed = defaultdict(list)
    for cell in cells:
        by_seed[int(cell["workload_seed"])].append(cell)
    common_hashes = all(
        len({cell[key] for cell in seed_cells}) == 1
        for seed_cells in by_seed.values()
        for key in (
            "workload_universe_sha256", "fault_probe_tuple_sha256",
            "failure_domain_rank_sets_sha256",
        )
    )
    probes_exact = all(
        all(
            sum(
                row["failure_domain"] == domain
                for row in cell["probe_metrics"]
            ) == per_domain
            for domain in ("source_worker_gpu", "source_node")
        )
        for cell in cells
    )
    domain_sizes = all(
        len(row["failure_domain_rank_ids"])
        == (1 if row["failure_domain"] == "source_worker_gpu" else 8)
        for cell in cells for row in cell["probe_metrics"]
    )
    exposure_early_bound_ms = horizon / per_domain
    exposure_late_bound_ms = horizon * (per_domain - 1) / per_domain
    exposure_span = all(
        min(
            row["probe_time_ms"] for row in cell["probe_metrics"]
            if row["failure_domain"] == domain
        ) < exposure_early_bound_ms
        and max(
            row["probe_time_ms"] for row in cell["probe_metrics"]
            if row["failure_domain"] == domain
        ) > exposure_late_bound_ms
        for cell in cells for domain in ("source_worker_gpu", "source_node")
    )
    expected_cells = len(seeds) * 6
    expected_seed_rows = expected_cells * 2
    checks = {
        **source_checks,
        "all_tests_pass": all(row["passed"] for row in tests),
        "exact_six_display_names_and_mapping": names == [
            "Native Mooncake", "Static", "DejaVu", "KevlarFlow",
            "GhostServe", "AegisKV",
        ] and mapping["Static"] == "Static FCFS",
        "old_single_seed_core_builder_lineage_fail_closed": (
            freeze["workload_builder"] == "active_hbm_v31.build_workload"
            and freeze["old_core_build_workload_lineage_rejected"]
            and freeze["predicted_slack_in_workload_universe_hash"]
            and all(
                cell["workload_metadata"]["slack_predictor"][
                    "reads_realized_output_tokens"
                ] is False
                and cell["workload_metadata"]["slack_predictor"][
                    "reads_realized_service_ms"
                ] is False
                for cell in cells
            )
        ),
        "all_registered_cells_complete": len(cells) == expected_cells,
        "all_raw_npz_complete_and_hashed": len(results) == expected_cells,
        "exact_seed_policy_domain_metric_rows": len(seed_rows) == expected_seed_rows,
        "common_workload_probe_domain_hashes_per_seed": common_hashes,
        "exact_worker_and_node_probe_counts": probes_exact,
        "worker_one_rank_node_eight_ranks": domain_sizes,
        "probe_exposure_spans_90s_horizon": exposure_span,
        "exact_registered_control_windows": all(
            cell["mechanism"]["scheduled_control_opportunity_times_ms"]
            == [30000.0, 60000.0, 90000.0]
            for cell in cells
        ),
        "aegis_executes_exactly_three_control_windows": all(
            cell["mechanism"]["control_window_count"] == 3
            and cell["mechanism"]["executed_policy_control_times_ms"]
            == [30000.0, 60000.0, 90000.0]
            for cell in cells if cell["display_policy"] == "AegisKV"
        ),
        "empty_fault_replay_exact": all(
            cell["empty_fault_replay_max_completion_delta_ms"] <= 1e-7
            for cell in cells
        ),
        "fault_conditioning_never_lowers_wct": all(
            cell["minimum_fault_minus_healthy_workflow_wct_ms"] >= -1e-7
            for cell in cells
        ),
        "no_capacity_violation": all(
            cell["mechanism"]["capacity_violation_count"] == 0 for cell in cells
        ),
        "copy_and_physical_target_conservation": all(
            cell["mechanism"]["copy_byte_conservation_error"]
            == {"HBM": 0, "DRAM": 0}
            and cell["mechanism"]["target_accounting_max_abs_error_bytes"] == 0
            for cell in cells
        ),
        "asynchronous_copy_not_direct_wct_burden": all(
            cell["mechanism"]["health_burden_application_count"] == 0
            for cell in cells
        ),
        "no_future_backup_seal_or_realized_oracle": all(
            cell["mechanism"]["future_backup_projection_dispatch_count"] == 0
            and cell["mechanism"]["seal_before_dispatch_count"] == 0
            and cell["mechanism"]["seal_uses_realized_output_or_service"] is False
            for cell in cells
        ),
        "paired_ci_uses_seed_as_unit": (
            len(paired_seed) == (len(seeds) * 5 * 2)
            and len(paired_ci) == 10
            and all(row["paired_seed_count"] == len(seeds) for row in paired_ci)
        ),
        "all_pairing_hashes_equal": all(
            row["paired_universe_hash_equal"]
            and row["paired_probe_hash_equal"]
            and row["paired_domain_set_hash_equal"]
            for row in paired_seed
        ),
        "all_node_results_retained": all(
            (name, seed, "source_node") in {
                (row["display_policy"], row["workload_seed"], row["failure_domain"])
                for row in seed_rows
            }
            for name in names for seed in seeds
        ),
        "no_performance_direction_gate": spec["selection"].startswith("NONE"),
    }
    if args.mode == "full":
        exact_pooled = {
            (row["display_policy"], row["failure_domain"]): sum(
                cell["domain_metrics"][row["failure_domain"]][
                    "probe_workflow_observation_count"
                ]
                for cell in cells
                if cell["display_policy"] == row["display_policy"]
            )
            for row in seed_rows
        }
        checks.update({
            "cdf_source_covers_every_pooled_observation": all(
                len(pooled[key]) == value for key, value in exact_pooled.items()
            ),
            "png_width_at_least_1800": figure["png_width_px"] >= 1800,
            "pdf_min_font_at_least_7pt": figure["pdf_min_font_pt"] >= 6.99,
            "vector_pdf_and_png_present": all(
                (stage / figure[key]).is_file() for key in ("pdf", "png")
            ),
        })
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": args.mode,
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner_sha256": core.sha256(Path(__file__)),
        "execution_freeze_sha256": core.sha256(stage / "execution_freeze.json"),
        "old_single_seed_lineage": "INVALIDATED_ORACLE_SLACK_DIAGNOSTIC_ONLY",
        "cell_count": len(cells),
        "seed_count": len(seeds),
        "probes_per_domain_per_seed": per_domain,
        "figure": figure,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
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
        "audit_sha256": core.sha256(stage / "audit.json"),
        "execution_freeze": "execution_freeze.json",
        "job_results": "job_results.json",
        "seed_policy_domain_metrics": "seed_policy_domain_metrics.csv",
        "probe_metrics": "probe_metrics.csv",
        "paired_vs_native_seed_metrics": "paired_vs_native_seed_metrics.csv",
        "paired_vs_native_ci": "paired_vs_native_ci.csv",
        "architecture_ci": "architecture_ci.csv",
        "cdf_render_source": "cdf_render_source.csv" if args.mode == "full" else None,
        "figure": figure,
        "cell_count": len(cells),
        "seed_count": len(seeds),
        "display_policy_order": names,
        "probes_per_domain_per_seed": per_domain,
        "old_single_seed_lineage": "INVALIDATED_ORACLE_SLACK_DIAGNOSTIC_ONLY",
        "workload_builder": "active_hbm_v31.build_workload",
        "pooled_cdf_is_descriptive": True,
        "paired_ci_inference_unit": "workload_seed",
        "no_result_seed_architecture_or_domain_selection": True,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (stage / SPEC_PATH.name).write_text(SPEC_PATH.read_text(), encoding="utf-8")
    (stage / "figure_caption.txt").write_text(
        "Failure-conditioned all-workflow WCT over ten paired workload seeds and policy-independent probes. Pooled ECDFs are descriptive; paired Student-t confidence intervals use workload seed as the inference unit. A100 bandwidth and failover timing are modeled/unmeasured, and all six systems are component-model approximations.\n",
        encoding="utf-8",
    )
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted(artifacts.items())),
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
