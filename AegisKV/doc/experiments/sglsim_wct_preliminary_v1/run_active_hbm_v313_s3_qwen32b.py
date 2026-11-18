#!/usr/bin/env python3
"""Preregistered Qwen3-32B TP1/PP4 S3 v3.1.3 preliminary campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import active_hbm_v31 as v31
import active_hbm_v313 as v313
import run_active_hbm_v313_s3_scale_context as base
import run_preliminary as core
import test_active_hbm_v313 as v313_tests


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s3_qwen32b_protocol.json"
RESULT_NAME = "active_hbm_v313_s3_qwen32b_full_v1"
GATE_NAME = "active_hbm_v313_s3_qwen32b_gate_v1"
T95_DF2 = 4.302652729911275
ORIGINAL_REPLAY = v313.CausalPhysicalTargetReplay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def ci95(values: list[float]) -> float:
    if len(values) != 3:
        raise ValueError("full inference requires exactly three paired seeds")
    return T95_DF2 * statistics.stdev(values) / math.sqrt(3)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolved_protocol(
    spec: dict[str, Any], *, dp: int, context: int, topology: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, capacity = v313_tests.inputs(context=context, active=2)
    model = spec["model"]
    service = spec["service_model"]
    compat = protocol["models"]["qwen3_8b"]
    compat.update({
        "display": model["display"],
        "tp": int(model["tp"]),
        "pp": int(model["pp"]),
        "center_system_dp": int(dp),
        "kv_bytes_per_token": int(model["kv_bytes_per_token_per_stage"]),
        "prefill_setup_ms": float(service["prefill_setup_ms"]),
        "prefill_ms_per_token": float(service["prefill_ms_per_token"]),
        "decode_ms_per_token": float(service["decode_ms_per_token"]),
        "logical_model_id": model["model_id"],
        "logical_rank_semantics": spec["hardware"]["logical_rank_semantics"],
    })
    protocol["workload"]["center_context_tokens"] = int(context)
    protocol["workload"]["center_active_requests_per_gpu"] = int(
        spec["workload"]["offered_active_requests_per_pipeline"]
    )
    protocol["hardware"]["gpus_per_node"] = int(
        spec["hardware"]["logical_ranks_per_node_in_DES"]
    )
    protocol["hardware"]["node_count"] = int(dp) // int(
        spec["hardware"]["logical_ranks_per_node_in_DES"]
    )
    protocol["hardware"]["copy_bandwidth_gb_s"] = dict(
        spec["hardware"]["copy_bandwidth_gb_s"]
    )
    protocol["hardware"]["load_bandwidth_gb_s"] = dict(
        spec["hardware"]["recovery_bandwidth_gb_s"]
    )
    route = spec["hardware"]["topologies"][topology]
    protocol.setdefault("runtime_overrides", {})["topology"] = {
        "peer_hbm": route["peer_hbm"], "dram": route["dram"],
    }
    protocol["runtime_overrides"]["model_abstraction"] = {
        "model_id": model["model_id"],
        "parallel_layout": "TP1/PP4",
        "logical_rank_is_pipeline_replica": True,
        "physical_gpus_per_pipeline": 4,
    }

    capacity = deepcopy(capacity)
    frozen = capacity["frozen_model"]
    frozen.update({
        "model_id": model["model_id"],
        "weight_bytes_per_tp1_pp1_rank": int(model["capacity_stage_weight_bytes"]),
        "weight_bytes_per_tp1_pp4_stage_rank": int(model["capacity_stage_weight_bytes"]),
        "layers": int(model["layers_per_pp_stage"]),
        "kv_heads": int(model["kv_heads"]),
        "head_dim": int(model["head_dim"]),
        "kv_dtype_bytes": int(model["kv_dtype_bytes"]),
        "kv_bytes_per_token": int(model["kv_bytes_per_token_per_stage"]),
        "kv_bytes_per_token_formula": model["kv_bytes_per_token_per_stage_formula"],
        "parallel_layout": "TP1/PP4 representative stage",
    })
    shared = capacity["shared_capacity"]
    shared["serving_plus_peer_backup_budget_bytes"] = int(
        spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_stage_gpu"]
    )
    shared["serving_plus_peer_backup_budget_gib"] = (
        shared["serving_plus_peer_backup_budget_bytes"] / 2**30
    )
    shared["request_max_output_tokens"] = int(model["request_max_output_tokens"])
    shared["request_serving_reservation_formula"] = (
        "(prompt_tokens + request_max_output_tokens) * "
        "kv_bytes_per_token_per_stage"
    )
    shared["rank_semantics"] = "one conservative representative PP-stage GPU"
    capacity["evidence_boundary"] = spec["evidence_boundary"]
    return protocol, capacity


class Qwen32BReplay(ORIGINAL_REPLAY):
    """v3.1.3 replay with Qwen3-32B PP4 provenance metadata."""

    def _profile(self, task: core.Task) -> core.RequestRecoveryProfile:
        placement = v313.topology_placement(self.protocol, task.source_rank)
        return core.RequestRecoveryProfile(
            request_id=task.request_id,
            source_worker=f"pipeline-{task.source_rank}",
            source_failure_domain=f"node-{placement['source_node']}",
            hbm_target=(
                f"peer-node{placement['peer_hbm_target_node']}"
                f"-pipeline{placement['local_gpu_index']}"
            ),
            hbm_failure_domain=f"node-{placement['peer_hbm_target_node']}",
            route_version=0,
            source_token_end=task.prompt_tokens,
            slack_ms=task.predicted_slack_ms,
            model_revision="Qwen3-32B-official-metadata",
            kv_abi="qwen3-32b-bf16-stage-kv-v1",
            parallel_layout="TP1-PP4-stage-aligned",
        )


# The old runner's audited cell serializer is reused, but its two hard-coded
# factories are replaced in this process by the preregistered 32B variants.
base.configured_protocol = resolved_protocol
base.v313.CausalPhysicalTargetReplay = Qwen32BReplay


def points(spec: dict[str, Any], *, gate: bool) -> list[dict[str, Any]]:
    if gate:
        return [{
            "condition_id": "dp4_ctx4096",
            "system_dp": 4,
            "context_tokens": 4096,
            "axis_membership": ["full_cross"],
        }]
    return [
        {
            "condition_id": f"dp{dp}_ctx{context}",
            "system_dp": int(dp),
            "context_tokens": int(context),
            "axis_membership": ["full_cross"],
        }
        for dp in spec["sweep"]["system_dp"]
        for context in spec["sweep"]["context_tokens"]
    ]


def run_job(
    stage: Path, spec: dict[str, Any], point: dict[str, Any], seed: int,
    topology: str, policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, _capacity = resolved_protocol(
        spec, dp=point["system_dp"], context=point["context_tokens"],
        topology="same_node",
    )
    nominal, workload = v31.build_workload(
        protocol, seed=seed,
        active_per_gpu=int(spec["workload"]["offered_active_requests_per_pipeline"]),
        arrival_cutoff_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
    )
    probes = core.make_fault_probes(
        nominal,
        seed=(seed ^ (int(point["system_dp"]) << 12) ^ int(point["context_tokens"])),
        count=int(spec["workload"]["fault_probes_per_condition_seed"]),
        gpus_per_node=int(spec["hardware"]["logical_ranks_per_node_in_DES"]),
        right_ms=float(spec["workload"]["fixed_arrival_cutoff_ms"]),
        failure_types=("source_worker_gpu",),
    )
    runtime_spec = deepcopy(spec)
    runtime_spec["hardware"]["gpus_per_node"] = int(
        spec["hardware"]["logical_ranks_per_node_in_DES"]
    )
    runtime_spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"] = int(
        spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_stage_gpu"]
    )
    cell, counts = base.run_cell(
        stage=stage, spec=runtime_spec, point=point, topology=topology,
        policy=policy, seed=seed, nominal=nominal, workload=workload,
        probes=probes, universe_sha=base.workload_hash(nominal),
        probe_sha=base.probe_hash(probes),
    )
    cell.update({
        "model_id": spec["model"]["model_id"],
        "model_display": spec["model"]["display"],
        "parallel_layout": "TP1/PP4",
        "logical_system_dp": int(point["system_dp"]),
        "physical_total_gpu_count": int(point["system_dp"]) * 4,
        "physical_node_count": int(point["system_dp"]) // 2,
        "physical_gpus_per_node": 8,
        "logical_pipeline_replicas_per_node": 2,
        "failed_physical_gpu_count": 1,
        "capacity_stage_weight_bytes": int(spec["model"]["capacity_stage_weight_bytes"]),
        "kv_bytes_per_token_per_stage": int(spec["model"]["kv_bytes_per_token_per_stage"]),
        "service_profile_status": spec["service_model"]["status"],
        "slack_predictor_kind": workload["slack_predictor"]["kind"],
        "slack_reads_realized_output_tokens": workload["slack_predictor"][
            "reads_realized_output_tokens"
        ],
        "slack_reads_realized_service_ms": workload["slack_predictor"][
            "reads_realized_service_ms"
        ],
        "realized_active_requests_per_pipeline_mean": cell.pop(
            "realized_active_requests_per_gpu_mean"
        ),
    })
    return cell, counts


METRICS = (
    "healthy_wct_mean_ms", "healthy_wct_p99_ms",
    "fault_wct_mean_ms", "fault_wct_p99_ms",
    "fault_added_wct_mean_ms", "workflow_throughput_per_s",
    "critical_queue_mean_ms", "queue_wait_mean_ms",
    "memory_blocked_unique_request_count", "hbm_scheduled_copy_bytes",
    "dram_scheduled_copy_bytes", "primary_priority_reclaimed_bytes",
    "capacity_min_headroom_bytes",
)


def aggregate(
    cells: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    group_keys = sorted({
        (int(row["system_dp"]), int(row["context_tokens"]),
         row["topology"], row["policy"])
        for row in cells
    })
    for dp, context, topology, policy in group_keys:
        selected = [
            row for row in cells
            if (int(row["system_dp"]), int(row["context_tokens"]),
                row["topology"], row["policy"])
            == (dp, context, topology, policy)
        ]
        out: dict[str, Any] = {
            "model": "Qwen3-32B TP1/PP4", "system_dp": dp,
            "physical_gpu_count": dp * 4, "context_tokens": context,
            "topology": topology, "policy": policy,
            "seed_count": len(selected),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            out[f"{metric}_mean"] = statistics.fmean(values)
            out[f"{metric}_ci95_half_width"] = ci95(values)
        aggregate_rows.append(out)

    paired_rows: list[dict[str, Any]] = []
    pair_metrics = (
        "healthy_wct_mean_ms", "healthy_wct_p99_ms",
        "fault_wct_mean_ms", "fault_wct_p99_ms",
    )
    pair_keys = sorted({
        (int(row["system_dp"]), int(row["context_tokens"]), row["topology"])
        for row in cells
    })
    for dp, context, topology in pair_keys:
        selected = [
            row for row in cells
            if (int(row["system_dp"]), int(row["context_tokens"]), row["topology"])
            == (dp, context, topology)
        ]
        by_key = {(row["policy"], int(row["seed"])): row for row in selected}
        seeds = sorted({int(row["seed"]) for row in selected})
        for metric in pair_metrics:
            native = [float(by_key[("Native Mooncake", seed)][metric]) for seed in seeds]
            aegis = [float(by_key[("AegisKV", seed)][metric]) for seed in seeds]
            delta = [right - left for left, right in zip(native, aegis)]
            reduction = [
                100.0 * (left - right) / left for left, right in zip(native, aegis)
            ]
            paired_rows.append({
                "model": "Qwen3-32B TP1/PP4", "system_dp": dp,
                "physical_gpu_count": dp * 4, "context_tokens": context,
                "topology": topology, "metric": metric,
                "paired_seed_count": len(seeds),
                "native_mean": statistics.fmean(native),
                "aegis_mean": statistics.fmean(aegis),
                "aegis_minus_native_mean": statistics.fmean(delta),
                "aegis_minus_native_ci95_half_width": ci95(delta),
                "aegis_wct_reduction_pct_mean": statistics.fmean(reduction),
                "aegis_wct_reduction_pct_ci95_half_width": ci95(reduction),
                "ci_method": "paired seed-level Student-t interval, df=2",
            })
    return aggregate_rows, paired_rows


def render_figure(
    stage: Path, aggregate_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]], spec: dict[str, Any],
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    import fitz

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.8,
        "axes.titlesize": 8.8, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2,
    })
    contexts = [int(value) for value in spec["sweep"]["context_tokens"]]
    dps = [int(value) for value in spec["sweep"]["system_dp"]]
    context_colors = {
        4096: "#1b9e77", 8192: "#66a61e", 16384: "#7570b3",
        32768: "#e6ab02", 40960: "#d95f02",
    }
    policy_colors = {"Native Mooncake": "#4d4d4d", "AegisKV": "#7b3294"}
    policy_markers = {"Native Mooncake": "s", "AegisKV": "o"}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    x = np.arange(len(dps))
    for column, topology in enumerate(("same_node", "cross_node")):
        ax = axes[0, column]
        for context in contexts:
            selected = sorted(
                [row for row in paired_rows
                 if row["topology"] == topology
                 and int(row["context_tokens"]) == context
                 and row["metric"] == "fault_wct_mean_ms"],
                key=lambda row: int(row["system_dp"]),
            )
            mean = np.asarray([
                float(row["aegis_wct_reduction_pct_mean"]) for row in selected
            ])
            half = np.asarray([
                float(row["aegis_wct_reduction_pct_ci95_half_width"])
                for row in selected
            ])
            ax.errorbar(
                x, mean, yerr=half, color=context_colors[context], marker="o",
                markersize=3.0, capsize=2.0, label=f"{context // 1024}K",
            )
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xticks(x, [str(value) for value in dps])
        ax.set_xlabel("System DP (PP4 pipelines)")
        ax.set_ylabel("Mean WCT reduction vs Native (%)")
        ax.set_title(
            "Same-node / local DRAM" if topology == "same_node"
            else "Cross-node / remote DRAM"
        )
        ax.grid(axis="y", alpha=0.23)

        ax = axes[1, column]
        for policy in ("Native Mooncake", "AegisKV"):
            selected = sorted(
                [row for row in aggregate_rows
                 if row["topology"] == topology and row["policy"] == policy
                 and int(row["context_tokens"]) == 16384],
                key=lambda row: int(row["system_dp"]),
            )
            mean = np.asarray([
                float(row["fault_wct_mean_ms_mean"]) / 1000.0 for row in selected
            ])
            half = np.asarray([
                float(row["fault_wct_mean_ms_ci95_half_width"]) / 1000.0
                for row in selected
            ])
            ax.errorbar(
                x, mean, yerr=half, color=policy_colors[policy],
                marker=policy_markers[policy], markersize=3.2,
                capsize=2.0, label=policy,
            )
        ax.set_xticks(x, [str(value) for value in dps])
        ax.set_xlabel("System DP (PP4 pipelines)")
        ax.set_ylabel("Worker-fault mean WCT (s)")
        ax.set_title("16K context")
        ax.grid(axis="y", alpha=0.23)

    context_handles = [
        Line2D([0], [0], color=context_colors[value], marker="o",
               label=f"{value // 1024}K")
        for value in contexts
    ]
    policy_handles = [
        Line2D([0], [0], color=policy_colors[name], marker=policy_markers[name],
               label=name)
        for name in ("Native Mooncake", "AegisKV")
    ]
    first = fig.legend(
        handles=context_handles, title="Context", frameon=False, ncol=5,
        loc="lower left", bbox_to_anchor=(0.05, 0.005),
        columnspacing=0.8, handlelength=1.6, title_fontsize=7.0,
    )
    fig.add_artist(first)
    fig.legend(
        handles=policy_handles, frameon=False, ncol=2,
        loc="lower right", bbox_to_anchor=(0.97, 0.005),
        columnspacing=1.0, handlelength=1.6,
    )
    fig.suptitle("Qwen3-32B TP1/PP4 scale and context", fontsize=10.0, y=0.992)
    fig.tight_layout(rect=(0, 0.105, 1, 0.95), h_pad=1.1, w_pad=1.4)
    figure_dir = stage / "figures"
    figure_dir.mkdir(exist_ok=True)
    stem = figure_dir / spec["figure_grammar"]["filename_stem"]
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    with Image.open(png) as image:
        width, height = image.size
    document = fitz.open(pdf)
    font_sizes: list[float] = []
    text_inside = True
    image_count = 0
    for page in document:
        image_count += len(page.get_images(full=True))
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        font_sizes.append(float(span["size"]))
                        text_inside = text_inside and page.rect.contains(
                            fitz.Rect(span["bbox"])
                        )
    page_count = document.page_count
    document.close()
    return {
        "pdf": str(pdf.relative_to(stage)),
        "png": str(png.relative_to(stage)),
        "pdf_page_count": page_count,
        "pdf_raster_image_count": image_count,
        "pdf_min_font_pt": min(font_sizes),
        "all_text_inside_pdf_page": text_inside,
        "png_width_px": width,
        "png_height_px": height,
    }


def run_tests(stage: Path) -> list[dict[str, Any]]:
    tests = (
        ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
        ("test_active_hbm_v313_s3_qwen32b.py", "RESULT 12/12 PASS"),
    )
    rows = []
    for filename, marker in tests:
        result = subprocess.run(
            [sys.executable, str(ROOT / filename)], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, check=False, env={**os.environ, "TMPDIR": str(stage)},
        )
        (stage / "test_logs" / f"{Path(filename).stem}.log").write_text(
            result.stdout, encoding="utf-8"
        )
        passed = result.returncode == 0 and marker in result.stdout
        rows.append({"test": filename, "marker": marker, "passed": passed})
        if not passed:
            raise RuntimeError(f"test failed: {filename}\n{result.stdout}")
    return rows


def main() -> int:
    args = parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    name = GATE_NAME if args.gate else RESULT_NAME
    target = ROOT / "results" / name
    if target.exists():
        raise FileExistsError(f"immutable target exists: {target}")
    stage = target.with_name(f".{name}.stage-{os.getpid()}")
    (stage / "raw").mkdir(parents=True)
    (stage / "figures").mkdir()
    (stage / "test_logs").mkdir()
    shutil.copy2(SPEC_PATH, stage / "protocol.json")
    _protocol, resolved_capacity = resolved_protocol(
        spec, dp=4, context=4096, topology="same_node"
    )
    (stage / "resolved_capacity_protocol.json").write_text(
        json.dumps(resolved_capacity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tests = run_tests(stage)
    registered_points = points(spec, gate=args.gate)
    seeds = (
        [int(spec["sweep"]["paired_seeds"][0])]
        if args.gate else [int(seed) for seed in spec["sweep"]["paired_seeds"]]
    )
    jobs = [
        (point, seed, topology, policy)
        for point in registered_points for seed in seeds
        for topology in spec["sweep"]["topologies"]
        for policy in spec["sweep"]["policies"]
    ]
    worker_count = 1 if args.gate else max(1, min(int(args.workers), len(jobs)))
    freeze = {
        "schema_version": 1,
        "mode": "gate" if args.gate else "full",
        "registered_before_raw": True,
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner_sha256": core.sha256(Path(__file__)),
        "job_count": len(jobs),
        "worker_count": worker_count,
        "seeds": seeds,
        "condition_count": len(registered_points),
        "policies": spec["sweep"]["policies"],
        "topologies": spec["sweep"]["topologies"],
        "selection_policy": spec["selection_policy"],
    }
    (stage / "execution_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cells: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    completed = 0
    if worker_count == 1:
        for job in jobs:
            cell, counts = run_job(stage, spec, *job)
            cells.append(cell)
            count_rows.extend(counts)
            completed += 1
            print(f"S3-32B {completed}/{len(jobs)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:
            future_map = {
                executor.submit(run_job, stage, spec, *job): job for job in jobs
            }
            for future in concurrent.futures.as_completed(future_map):
                cell, counts = future.result()
                cells.append(cell)
                count_rows.extend(counts)
                completed += 1
                if completed % 10 == 0 or completed == len(jobs):
                    print(f"S3-32B {completed}/{len(jobs)}", flush=True)

    cells.sort(key=lambda row: (
        int(row["system_dp"]), int(row["context_tokens"]), row["topology"],
        row["policy"], int(row["seed"]),
    ))
    count_rows.sort(key=lambda row: (
        int(row["system_dp"]), int(row["context_tokens"]), row["topology"],
        row["policy"], int(row["seed"]), row["category"], row["key"],
    ))
    write_csv(stage / "cell_metrics.csv", cells)
    write_csv(stage / "mechanism_counts.csv", count_rows)
    with (stage / "cell_summary.jsonl").open("w", encoding="utf-8") as handle:
        for row in cells:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    aggregate_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    figure: dict[str, Any] = {}
    if not args.gate:
        aggregate_rows, paired_rows = aggregate(cells)
        write_csv(stage / "aggregate_ci.csv", aggregate_rows)
        write_csv(stage / "paired_ci.csv", paired_rows)
        figure = render_figure(stage, aggregate_rows, paired_rows, spec)

    condition_seed_keys = sorted({
        (row["condition_id"], int(row["seed"])) for row in cells
    })
    common_hashes = all(
        len({
            (row["workflow_universe_sha256"], row["fault_probe_tuple_sha256"])
            for row in cells
            if (row["condition_id"], int(row["seed"])) == key
        }) == 1
        for key in condition_seed_keys
    )
    expected_cells = len(jobs)
    compatibility_budget = int(
        spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_stage_gpu"]
    )
    checks = {
        "mechanism_and_qwen32b_protocol_tests_pass": all(
            row["passed"] for row in tests
        ),
        "exact_registered_cell_count": len(cells) == expected_cells,
        "all_policy_topology_pairs_per_condition_seed": all(
            len([row for row in cells
                 if (row["condition_id"], int(row["seed"])) == key]) == 4
            for key in condition_seed_keys
        ),
        "common_workflow_and_probe_hashes_across_policy_topology": common_hashes,
        "no_oracle_prediction_lineage": all(
            row["slack_predictor_kind"]
            == "independent_registered_output_distribution_v1"
            and row["slack_reads_realized_output_tokens"] is False
            and row["slack_reads_realized_service_ms"] is False
            for row in cells
        ),
        "source_worker_fault_one_logical_pipeline_one_physical_gpu": all(
            row["failed_rank_count_min"] == 1
            and row["failed_rank_count_max"] == 1
            and row["failed_physical_gpu_count"] == 1
            for row in cells
        ),
        "pp4_physical_layout_exact": all(
            row["parallel_layout"] == "TP1/PP4"
            and row["physical_total_gpu_count"] == row["system_dp"] * 4
            and row["physical_node_count"] == row["system_dp"] // 2
            and row["node_count"] == row["system_dp"] // 2
            and row["physical_gpus_per_node"] == 8
            for row in cells
        ),
        "same_cross_topology_and_directional_bandwidth_exact": all(
            row["topology_mode_exact"]
            and json.loads(row["wrong_copy_path_keys"]) == []
            and (
                (row["topology"] == "same_node"
                 and row["peer_copy_bandwidth_gb_s"] == 50.0
                 and row["dram_copy_bandwidth_gb_s"] == 16.0
                 and row["peer_recovery_bandwidth_gb_s"] == 50.0
                 and row["dram_recovery_bandwidth_gb_s"] == 16.0)
                or
                (row["topology"] == "cross_node"
                 and row["peer_copy_bandwidth_gb_s"] == 10.0
                 and row["dram_copy_bandwidth_gb_s"] == 10.0
                 and row["peer_recovery_bandwidth_gb_s"] == 10.0
                 and row["dram_recovery_bandwidth_gb_s"] == 10.0)
            ) for row in cells
        ),
        "physical_stage_gpu_capacity_invariant": all(
            row["capacity_violation_count"] == 0
            and row["capacity_max_shared_used_bytes"] <= compatibility_budget
            and row["capacity_min_headroom_bytes"] >= 0
            and row["target_accounting_max_abs_error_bytes"] == 0
            for row in cells
        ),
        "copy_bytes_conserve_exactly": all(
            json.loads(row["copy_byte_conservation_error"])
            == {"DRAM": 0, "HBM": 0} for row in cells
        ),
        "no_future_or_unsealed_backup_dispatch_dependency": all(
            row["future_backup_projection_dispatch_count"] == 0
            and row["seal_before_dispatch_count"] == 0
            and row["seal_uses_realized_output_or_service"] is False
            for row in cells
        ),
        "empty_replay_exact_and_fault_monotone": all(
            row["empty_replay_max_abs_error_ms"] <= 1e-7
            and row["fault_monotonic_max_negative_error_ms"] <= 1e-7
            for row in cells
        ),
        "three_scheduled_control_opportunities": all(
            json.loads(row["scheduled_control_opportunity_times_ms"])
            == [30000.0, 60000.0, 90000.0] for row in cells
        ),
        "raw_npz_hashes_match": all(
            core.sha256(stage / row["raw_npz"]) == row["raw_npz_sha256"]
            for row in cells
        ),
        "preregistered_local_source_hashes_match": all(
            Path(row["path"]).exists()
            and core.sha256(Path(row["path"])) == row["sha256"]
            for row in spec["sources"].values()
        ),
        "analytic_profile_explicitly_not_matched_or_physical": all(
            row["service_profile_status"]
            == "MODELED_ANALYTIC_WEIGHT_RATIO_NOT_PHYSICAL" for row in cells
        ),
        "no_performance_direction_or_cell_selection": True,
    }
    if not args.gate:
        checks.update({
            "exact_full_cross_300_cells": (
                len(cells) == spec["sweep"]["registered_cell_count"] == 300
            ),
            "exact_100_aggregate_rows_three_seeds": (
                len(aggregate_rows) == 100
                and all(row["seed_count"] == 3 for row in aggregate_rows)
            ),
            "exact_200_paired_metric_rows_three_seeds": (
                len(paired_rows) == 200
                and all(row["paired_seed_count"] == 3 for row in paired_rows)
            ),
            "single_page_vector_pdf": (
                figure["pdf_page_count"] == 1
                and figure["pdf_raster_image_count"] == 0
            ),
            "png_width_at_least_1800": figure["png_width_px"] >= 1800,
            "pdf_min_font_at_least_7pt": figure["pdf_min_font_pt"] >= 6.99,
            "pdf_text_not_cropped": figure["all_text_inside_pdf_page"],
        })
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "schema_version": 1, "status": status,
        "mode": "gate" if args.gate else "full", "paper_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "cell_count": len(cells), "expected_cell_count": expected_cells,
        "protocol_sha256": core.sha256(stage / "protocol.json"),
        "runner_sha256": core.sha256(Path(__file__)),
        "resolved_capacity_protocol_sha256": core.sha256(
            stage / "resolved_capacity_protocol.json"
        ),
        "cell_metrics_sha256": core.sha256(stage / "cell_metrics.csv"),
        "mechanism_counts_sha256": core.sha256(stage / "mechanism_counts.csv"),
        "figure": figure,
        "profile_status": spec["service_model"]["status"],
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "PASS":
        raise RuntimeError(f"audit FAIL {audit['failed_checks']}; stage={stage}")

    manifest = {
        "schema_version": 1, "status": "PASS", "paper_ready": False,
        "mode": audit["mode"], "selection": "S3-Qwen3-32B",
        "profile_status": spec["service_model"]["status"],
        "protocol": "protocol.json", "protocol_sha256": audit["protocol_sha256"],
        "runner": Path(__file__).name, "runner_sha256": audit["runner_sha256"],
        "execution_freeze": "execution_freeze.json",
        "execution_workers": worker_count,
        "audit": "audit.json", "audit_sha256": core.sha256(stage / "audit.json"),
        "resolved_capacity_protocol": "resolved_capacity_protocol.json",
        "cell_metrics": "cell_metrics.csv", "cell_summary": "cell_summary.jsonl",
        "mechanism_counts": "mechanism_counts.csv",
        "aggregate_ci": None if args.gate else "aggregate_ci.csv",
        "paired_ci": None if args.gate else "paired_ci.csv",
        "figure": None if args.gate else figure,
        "cell_count": len(cells), "seed_count": len(seeds),
        "condition_count": len(registered_points),
        "same_sample_as_qwen3_8b_s3": False,
        "no_result_selection": True,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage / "figure_caption.txt").write_text(
        "Qwen3-32B TP1/PP4 modeled S3 scale and context results. Top panels "
        "show paired all-workflow mean WCT reduction for every registered "
        "context; bottom panels show absolute failure-conditioned mean WCT "
        "at the preregistered 16K context. Error bars are Student-t 95% "
        "confidence intervals across three workload seeds. The service profile "
        "is an analytic total-weight-ratio model, not a matched or measured "
        "A100 profile.\n",
        encoding="utf-8",
    )
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())
        ), encoding="utf-8",
    )
    os.replace(stage, target)
    print(json.dumps({
        "output": str(target), "status": "PASS", "cells": len(cells),
        "audit_sha256": core.sha256(target / "audit.json"),
        "manifest_sha256": core.sha256(target / "manifest.json"),
        "sha256sums_sha256": core.sha256(target / "SHA256SUMS"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
