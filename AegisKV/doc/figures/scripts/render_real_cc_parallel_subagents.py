#!/usr/bin/env python3
"""Render compact Real CC parallel-runtime motivation panels.

Panel 1 is a workflow-level CDF.  A parser fan-out is a maximal consecutive
run of top-level ``type=subagent`` records, matching ``weka_trace._segments``.
It is retained only when every recorded branch interval shares a non-empty
common overlap.  Each workflow contributes one sample: the largest
slowest-minus-fastest duration gap among its retained fan-outs.

Panel 2 uses the same five-way fan-out shown in the workflow schematic.  At a
common trace time, it takes the input-prefix length of each branch's latest
started LLM call, predicts full-prefix recomputation with AIConfigurator, and
reports the component exposed at fan-in as ``max(0, recompute - slack)``.
This is a trace-plus-configuration counterfactual, not a measured failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.spec.accelerator.base import AcceleratorInfo
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.time_predictor.aiconfigurator import AIConfiguratorTimePredictor
from sglsim.srt.time_predictor.base import ScheduleBatch, ScheduleRequest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "Archived"
    / "2026-08-06-pre-kvcache-centric"
    / "presentations"
    / "presentation"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "cc-traces-weka-062126"
    / "traces.materialized.jsonl"
)
DEFAULT_RAW_INPUT = DEFAULT_INPUT.with_name("traces.jsonl")
DEFAULT_DATA_DIR = ROOT / "figures" / "data"
DEFAULT_GAP_OUTPUT = ROOT / "figures" / "real-cc-parallel-subagent-duration.png"
DEFAULT_FAILURE_OUTPUT = ROOT / "figures" / "real-cc-failure-position-counterfactual.png"
POSITIONS = (0.20, 0.40, 0.60, 0.80, 1.00)
EXAMPLE_WORKFLOW_ID = "8ab0a0892b0035b681a85c71d603caeb9e0f"
EXAMPLE_FANOUT_ID = f"{EXAMPLE_WORKFLOW_ID}-fanout-1"
EXAMPLE_FAILURE_TIME_S = 349.5
CHUNK_TOKENS = 8192
PREDICTOR_MODEL = "Qwen/Qwen3-235B-A22B"
PREDICTOR_TP = 8
PREDICTOR_PP = 4
PREDICTOR_EP = 8
PREDICTOR_CONTEXT_LIMIT = 131_072


def quantile(values: Iterable[float], p: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=float), p))


def quantile_summary(values: Iterable[float]) -> dict[str, float]:
    array = list(values)
    return {
        "p50": round(quantile(array, 0.50), 6),
        "p75": round(quantile(array, 0.75), 6),
        "p90": round(quantile(array, 0.90), 6),
        "p95": round(quantile(array, 0.95), 6),
        "max": round(max(array), 6),
    }


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(list(values), dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float) -> float:
    return round(float(value), 6)


def _trace_sha256(input_path: Path) -> str:
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _make_recompute_predictor() -> AIConfiguratorTimePredictor:
    return AIConfiguratorTimePredictor(
        ModelInfo.from_model_path(
            PREDICTOR_MODEL,
            context_len=PREDICTOR_CONTEXT_LIMIT,
        ),
        AcceleratorInfo(name="a100_sxm", hbm_capacity_gb=80.0),
        SchedulerSimConfig(
            tp_size=PREDICTOR_TP,
            pp_size=PREDICTOR_PP,
            ep_size=PREDICTOR_EP,
            dp_size=1,
            backend_name="sglang",
            backend_version="0.5.10",
        ),
        {
            "database_mode": "SILICON",
            "prefill_scale_factor": 1.05,
            "decode_scale_factor": 0.9,
            "forward_scale": 1.0,
        },
    )


def _predict_full_prefix_recompute_s(
    predictor: AIConfiguratorTimePredictor,
    context_tokens: int,
) -> float:
    if not 0 < context_tokens <= PREDICTOR_CONTEXT_LIMIT:
        raise ValueError(f"prefix outside predictor context limit: {context_tokens}")
    prefix = 0
    latency_s = 0.0
    while prefix < context_tokens:
        extend = min(CHUNK_TOKENS, context_tokens - prefix)
        latency_s += float(
            predictor.predict_infer_time(
                ScheduleBatch(
                    reqs=[
                        ScheduleRequest(
                            extend_length=extend,
                            past_kv_length=prefix,
                        )
                    ]
                )
            )
        )
        prefix += extend
    return latency_s


def _load_example_prefixes(
    raw_input_path: Path,
) -> tuple[dict[str, dict[str, object]], str]:
    """Read per-subagent input prefixes at the common counterfactual time."""
    raw_trace_sha256 = _trace_sha256(raw_input_path)
    with raw_input_path.open("rb") as stream:
        for raw_line in stream:
            if not raw_line.strip():
                continue
            workflow = json.loads(raw_line)
            if str(workflow.get("id")) != EXAMPLE_WORKFLOW_ID:
                continue

            prefixes: dict[str, dict[str, object]] = {}
            for request in workflow.get("requests", []):
                if request.get("type") != "subagent":
                    continue
                agent_id = str(request.get("agent_id"))
                eligible = [
                    nested
                    for nested in request.get("requests", [])
                    if _float(nested.get("t"), float("inf"))
                    <= EXAMPLE_FAILURE_TIME_S
                    and int(nested.get("in", 0)) > 0
                ]
                if not eligible:
                    raise RuntimeError(
                        f"no nested request at t<={EXAMPLE_FAILURE_TIME_S} for {agent_id}"
                    )
                selected = max(eligible, key=lambda nested: _float(nested.get("t")))
                prefixes[agent_id] = {
                    "prefix_request_start_s": _float(selected.get("t")),
                    "prefix_tokens": int(selected["in"]),
                }
            if not prefixes:
                raise RuntimeError(f"no subagents in raw workflow {EXAMPLE_WORKFLOW_ID}")
            return prefixes, raw_trace_sha256
    raise RuntimeError(f"workflow {EXAMPLE_WORKFLOW_ID} not found in {raw_input_path}")


def collect(
    input_path: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    int,
    str,
]:
    trace_sha256 = _trace_sha256(input_path)
    workflow_count = 0
    branch_rows: list[dict[str, object]] = []
    fanout_rows: list[dict[str, object]] = []
    workflow_fanouts: dict[str, list[dict[str, object]]] = {}
    workflow_branches: dict[str, list[dict[str, object]]] = {}

    with input_path.open("rb") as stream:
        for raw_line in stream:
            if not raw_line.strip():
                continue
            workflow = json.loads(raw_line)
            workflow_count += 1
            workflow_id = str(workflow["id"])
            requests = list(workflow.get("requests", []))
            request_index = 0
            fanout_segment_index = 0

            while request_index < len(requests):
                if requests[request_index].get("type") != "subagent":
                    request_index += 1
                    continue

                fanout_segment_index += 1
                group: list[dict[str, object]] = []
                while (
                    request_index < len(requests)
                    and requests[request_index].get("type") == "subagent"
                ):
                    group.append(requests[request_index])
                    request_index += 1

                if len(group) < 2:
                    continue

                intervals: list[dict[str, object]] = []
                valid = True
                for branch_index, branch in enumerate(group):
                    start_s = _float(branch.get("t"))
                    duration_s = _float(branch.get("duration_ms")) / 1000.0
                    if duration_s <= 0.0:
                        valid = False
                        break
                    intervals.append(
                        {
                            "branch_index": branch_index,
                            "agent_id": str(
                                branch.get("agent_id") or f"branch-{branch_index}"
                            ),
                            "start_s": start_s,
                            "end_s": start_s + duration_s,
                            "duration_s": duration_s,
                        }
                    )
                if not valid:
                    continue

                overlap_start_s = max(float(row["start_s"]) for row in intervals)
                overlap_end_s = min(float(row["end_s"]) for row in intervals)
                if not overlap_start_s < overlap_end_s:
                    continue

                fanout_id = f"{workflow_id}-fanout-{fanout_segment_index}"
                durations = [float(row["duration_s"]) for row in intervals]
                fastest_s = min(durations)
                slowest_s = max(durations)
                duration_gap_s = slowest_s - fastest_s
                fanout_completion_s = max(float(row["end_s"]) for row in intervals)
                fanout_row: dict[str, object] = {
                    "trace_sha256": trace_sha256,
                    "workflow_id": workflow_id,
                    "fanout_id": fanout_id,
                    "fanout_segment_index": fanout_segment_index,
                    "fanout_width": len(intervals),
                    "overlap_start_s": _round(overlap_start_s),
                    "overlap_end_s": _round(overlap_end_s),
                    "overlap_duration_s": _round(overlap_end_s - overlap_start_s),
                    "fanout_completion_s": _round(fanout_completion_s),
                    "fastest_duration_s": _round(fastest_s),
                    "slowest_duration_s": _round(slowest_s),
                    "duration_gap_s": _round(duration_gap_s),
                    "parallel_definition": "max_start_lt_min_end",
                }
                fanout_rows.append(fanout_row)
                workflow_fanouts.setdefault(workflow_id, []).append(fanout_row)

                for interval in intervals:
                    branch_end_s = float(interval["end_s"])
                    row = {
                        "trace_sha256": trace_sha256,
                        "workflow_id": workflow_id,
                        "fanout_id": fanout_id,
                        "fanout_segment_index": fanout_segment_index,
                        "fanout_width": len(intervals),
                        "branch_index": int(interval["branch_index"]),
                        "agent_id": str(interval["agent_id"]),
                        "branch_start_s": _round(float(interval["start_s"])),
                        "branch_end_s": _round(branch_end_s),
                        "duration_s": _round(float(interval["duration_s"])),
                        "slack_s": _round(fanout_completion_s - branch_end_s),
                        "overlap_start_s": _round(overlap_start_s),
                        "overlap_end_s": _round(overlap_end_s),
                        "fanout_completion_s": _round(fanout_completion_s),
                        "fanout_duration_gap_s": _round(duration_gap_s),
                        "parallel_definition": "max_start_lt_min_end",
                    }
                    branch_rows.append(row)
                    workflow_branches.setdefault(workflow_id, []).append(row)

    workflow_gap_rows: list[dict[str, object]] = []
    for workflow_id, rows in sorted(workflow_fanouts.items()):
        selected = max(
            rows,
            key=lambda row: (
                float(row["duration_gap_s"]),
                -int(row["fanout_segment_index"]),
            ),
        )
        workflow_gap_rows.append(
            {
                "trace_sha256": trace_sha256,
                "workflow_id": workflow_id,
                "selected_fanout_id": selected["fanout_id"],
                "strict_parallel_fanout_count": len(rows),
                "fanout_width": selected["fanout_width"],
                "overlap_start_s": selected["overlap_start_s"],
                "overlap_end_s": selected["overlap_end_s"],
                "fastest_duration_s": selected["fastest_duration_s"],
                "slowest_duration_s": selected["slowest_duration_s"],
                "max_duration_gap_s": selected["duration_gap_s"],
                "selection_rule": "max_within_fanout_duration_gap_per_workflow",
                "parallel_definition": "max_start_lt_min_end",
            }
        )

    failure_branch_rows: list[dict[str, object]] = []
    by_workflow_position: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in branch_rows:
        for position in POSITIONS:
            restart_delay_s = position * float(row["duration_s"])
            delta_wct_s = max(0.0, restart_delay_s - float(row["slack_s"]))
            position_pct = int(round(position * 100))
            counterfactual = {
                **row,
                "failure_position_fraction": position,
                "failure_position_pct": position_pct,
                "restart_delay_s": _round(restart_delay_s),
                "delta_wct_s": _round(delta_wct_s),
                "counterfactual": "restart_from_branch_start",
                "wct_contribution": "max_0_restart_delay_minus_fanin_slack",
            }
            failure_branch_rows.append(counterfactual)
            by_workflow_position.setdefault((str(row["workflow_id"]), position_pct), []).append(
                counterfactual
            )

    failure_workflow_rows: list[dict[str, object]] = []
    for (workflow_id, position_pct), rows in sorted(by_workflow_position.items()):
        selected = max(
            rows,
            key=lambda row: (
                float(row["delta_wct_s"]),
                float(row["duration_s"]),
                str(row["fanout_id"]),
                int(row["branch_index"]),
            ),
        )
        failure_workflow_rows.append(
            {
                "trace_sha256": trace_sha256,
                "workflow_id": workflow_id,
                "failure_position_fraction": selected["failure_position_fraction"],
                "failure_position_pct": position_pct,
                "selected_fanout_id": selected["fanout_id"],
                "selected_agent_id": selected["agent_id"],
                "branch_index": selected["branch_index"],
                "branch_start_s": selected["branch_start_s"],
                "branch_end_s": selected["branch_end_s"],
                "duration_s": selected["duration_s"],
                "slack_s": selected["slack_s"],
                "restart_delay_s": selected["restart_delay_s"],
                "max_delta_wct_s": selected["delta_wct_s"],
                "selection_rule": "max_counterfactual_delta_per_workflow_position",
                "counterfactual": "restart_from_branch_start",
                "wct_contribution": "max_0_restart_delay_minus_fanin_slack",
                "parallel_definition": "max_start_lt_min_end",
            }
        )

    return (
        branch_rows,
        fanout_rows,
        workflow_gap_rows,
        failure_branch_rows,
        failure_workflow_rows,
        workflow_count,
        trace_sha256,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.2,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render_gap_cdf(rows: list[dict[str, object]], output: Path) -> None:
    values = [float(row["max_duration_gap_s"]) for row in rows]
    x, y = ecdf(values)
    p50 = quantile(values, 0.50)
    p90 = quantile(values, 0.90)

    _plot_style()
    fig, ax = plt.subplots(figsize=(2.05, 1.72))
    ax.step(x, y, where="post", color="#0072B2", linewidth=1.65)
    ax.set_xscale("log")
    ax.set_xlim(max(min(x) * 0.82, 1e-3), max(x) * 1.18)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Max branch-duration gap (s)", labelpad=2)
    ax.set_ylabel("CDF of workflows", labelpad=2)
    ax.axhline(0.5, color="#CDD3DC", linewidth=0.65, linestyle=":")
    ax.axhline(0.9, color="#CDD3DC", linewidth=0.65, linestyle=":")
    ax.scatter([p50, p90], [0.5, 0.9], s=10, color="#D55E00", zorder=3)
    ax.annotate(
        "P50",
        xy=(p50, 0.5),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=5.8,
        color="#8B3C00",
    )
    ax.annotate(
        "P90",
        xy=(p90, 0.9),
        xytext=(2, -8),
        textcoords="offset points",
        fontsize=5.8,
        color="#8B3C00",
    )
    example = next(
        row for row in rows if row["workflow_id"] == EXAMPLE_WORKFLOW_ID
    )
    example_gap = float(example["max_duration_gap_s"])
    example_percentile = float(np.mean(np.asarray(values) <= example_gap))
    ax.scatter(
        [example_gap],
        [example_percentile],
        s=20,
        marker="D",
        facecolor="#009E73",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.annotate(
        "example: 39 s",
        xy=(example_gap, example_percentile),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=5.7,
        color="#08765B",
        ha="left",
        va="bottom",
    )
    ax.text(
        0.04,
        0.96,
        f"n={len(rows)} workflows",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.1,
        color="#263746",
    )
    ax.grid(True, which="major", color="#DDE2E8", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.97, bottom=0.27)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def render_failure_position(
    rows: list[dict[str, object]], raw_input_path: Path, output: Path
) -> list[dict[str, object]]:
    selected = sorted(
        (row for row in rows if row["fanout_id"] == EXAMPLE_FANOUT_ID),
        key=lambda row: int(row["branch_index"]),
    )
    if len(selected) != 5:
        raise RuntimeError(
            f"expected five branches for {EXAMPLE_FANOUT_ID}, got {len(selected)}"
        )

    prefixes, raw_trace_sha256 = _load_example_prefixes(raw_input_path)
    predictor = _make_recompute_predictor()
    example_rows: list[dict[str, object]] = []
    for row in selected:
        agent_id = str(row["agent_id"])
        prefix = prefixes[agent_id]
        prefix_tokens = int(prefix["prefix_tokens"])
        slack_s = float(row["slack_s"])
        recompute_s = _predict_full_prefix_recompute_s(predictor, prefix_tokens)
        delta_wct_s = max(0.0, recompute_s - slack_s)
        example_rows.append(
            {
                **row,
                "raw_trace_sha256": raw_trace_sha256,
                "failure_time_s": EXAMPLE_FAILURE_TIME_S,
                "prefix_request_start_s": _round(
                    float(prefix["prefix_request_start_s"])
                ),
                "prefix_tokens": prefix_tokens,
                "recompute_s": _round(recompute_s),
                "delta_wct_s": _round(delta_wct_s),
                "counterfactual": "aiconfigurator_full_prefix_recompute_at_tf",
                "wct_contribution": "max_0_recompute_minus_fanin_slack",
                "predictor_model": PREDICTOR_MODEL,
                "predictor_topology": (
                    f"TP{PREDICTOR_TP}xPP{PREDICTOR_PP}xEP{PREDICTOR_EP}"
                ),
                "predictor_accelerator": "A100 80GB",
                "predictor_database_mode": "SILICON",
                "predictor_chunk_tokens": CHUNK_TOKENS,
                "predictor_evidence": "configuration_prediction_not_physical_pp_run",
            }
        )

    _plot_style()
    fig, ax = plt.subplots(figsize=(2.05, 1.72))
    labels = [f"A{i + 1}" for i in range(len(example_rows))]
    values_s = [float(row["delta_wct_s"]) for row in example_rows]
    colors = ["#D55E00" if value > 0.0 else "#94A3B8" for value in values_s]
    y = np.arange(len(example_rows))
    ax.barh(
        y,
        values_s,
        height=0.52,
        color=colors,
        edgecolor="#334155",
        linewidth=0.65,
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Exposed fan-in delay (s)", labelpad=2)
    max_value = max(values_s)
    ax.set_xlim(0.0, max_value * 1.23)
    for yi, value in zip(y, values_s):
        x = value + max_value * 0.025 if value > 0 else max_value * 0.018
        ax.text(
            x,
            yi,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=6.2,
            color="#334155",
        )
    ax.text(
        0.98,
        0.04,
        rf"$t_f={EXAMPLE_FAILURE_TIME_S:g}$ s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color="#475569",
    )
    ax.grid(True, axis="x", color="#DDE2E8", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.96, bottom=0.27)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return example_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_GAP_OUTPUT)
    parser.add_argument(
        "--failure-output", type=Path, default=DEFAULT_FAILURE_OUTPUT
    )
    args = parser.parse_args()

    (
        branch_rows,
        fanout_rows,
        workflow_gap_rows,
        failure_branch_rows,
        failure_workflow_rows,
        workflow_count,
        trace_sha256,
    ) = collect(args.input)
    if not branch_rows or not fanout_rows or not workflow_gap_rows:
        raise RuntimeError("No strictly overlapping parallel fan-outs found")

    csv_paths = {
        "branches": args.data_dir / "real-cc-parallel-subagent-duration.csv",
        "fanouts": args.data_dir / "real-cc-parallel-fanout-summary.csv",
        "workflow_gaps": args.data_dir / "real-cc-workflow-parallel-gap.csv",
        "failure_example": args.data_dir / "real-cc-example-failure-position.csv",
    }
    write_csv(csv_paths["branches"], branch_rows)
    write_csv(csv_paths["fanouts"], fanout_rows)
    write_csv(csv_paths["workflow_gaps"], workflow_gap_rows)
    render_gap_cdf(workflow_gap_rows, args.output)
    example_failure_rows = render_failure_position(
        branch_rows,
        args.raw_input,
        args.failure_output,
    )
    write_csv(csv_paths["failure_example"], example_failure_rows)
    gap_values = [float(row["max_duration_gap_s"]) for row in workflow_gap_rows]
    print(
        json.dumps(
            {
                "input": str(args.input),
                "trace_sha256": trace_sha256,
                "workflows_total": workflow_count,
                "strict_parallel_workflows": len(workflow_gap_rows),
                "strict_parallel_fanouts": len(fanout_rows),
                "strict_parallel_branches": len(branch_rows),
                "workflow_gap_quantiles_s": quantile_summary(gap_values),
                "failure_example_workflow_id": EXAMPLE_WORKFLOW_ID,
                "failure_example_fanout_id": EXAMPLE_FANOUT_ID,
                "failure_example_time_s": EXAMPLE_FAILURE_TIME_S,
                "failure_example_prefix_tokens": [
                    row["prefix_tokens"] for row in example_failure_rows
                ],
                "failure_example_recompute_s": [
                    row["recompute_s"] for row in example_failure_rows
                ],
                "failure_example_delta_wct_s": [
                    row["delta_wct_s"] for row in example_failure_rows
                ],
                "counterfactual": "aiconfigurator_full_prefix_recompute_at_tf",
                "predictor_topology": (
                    f"TP{PREDICTOR_TP}xPP{PREDICTOR_PP}xEP{PREDICTOR_EP}"
                ),
                "parallel_definition": "max_start_lt_min_end",
                "csv": {key: str(path) for key, path in csv_paths.items()},
                "figures": {
                    "workflow_gap_png": str(args.output),
                    "workflow_gap_pdf": str(args.output.with_suffix(".pdf")),
                    "failure_position_png": str(args.failure_output),
                    "failure_position_pdf": str(args.failure_output.with_suffix(".pdf")),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
