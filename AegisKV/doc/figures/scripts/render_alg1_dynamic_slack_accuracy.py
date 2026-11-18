#!/usr/bin/env python3
"""Replay and render request-level dynamic Slack prediction accuracy.

This is a causal fixed-DAG replay over persisted SGLSim online completion
events.  At stage start, each branch uses the workflow-disjoint fitted request
cost model.  After every completed request, only the completion prefix visible
at that time updates the triggering branch's projected duration; all active
agents then receive a new stage-local ``slack_epoch``.

Future completion timestamps are used only after replay to backfill the actual
residual Slack target.  The live client knew stage-start timestamps, but the
archived result did not persist them, so this script reconstructs each start as
``branch_finish - measured_branch_duration`` and audits cross-branch agreement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "Archived" / "2026-08-06-pre-kvcache-centric"
DEFAULT_RESULT = (
    ARCHIVE
    / "experiments"
    / "demo"
    / "results"
    / "qwen3_32b_dp4_pp4_alg1_formal_calibration_v2"
    / "20260804-021032"
    / "no_backup.json"
)
DEFAULT_TRACE = (
    ARCHIVE
    / "presentations"
    / "presentation"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "weka_subagent_cdf_24w_workload.jsonl"
)
DEFAULT_CALIBRATION = (
    ARCHIVE
    / "presentations"
    / "presentation"
    / "experiments"
    / "alg1_slack_calibration.json"
)
DEFAULT_OUTPUT_PREFIX = ROOT / "figures" / "alg1-dynamic-slack-accuracy"
DEFAULT_SNAPSHOTS = ROOT / "figures" / "data" / "alg1-dynamic-slack-snapshots.csv"
DEFAULT_SUMMARY = (
    ROOT / "figures" / "data" / "alg1-dynamic-slack-accuracy-summary.json"
)
SNAPSHOT_FIELDS = (
    "workflow_id",
    "workflow_short",
    "stage_id",
    "stage_width",
    "agent_id",
    "agent_short",
    "stage_slack_epoch",
    "event_elapsed_ms",
    "trigger_agent_id",
    "trigger_request_id",
    "trigger_request_seq",
    "agent_next_request_seq",
    "agent_progress_fraction",
    "predicted_remaining_lcb_ms",
    "predicted_remaining_q50_ms",
    "predicted_remaining_q90_ms",
    "actual_remaining_ms",
    "predicted_slack_lcb_ms",
    "predicted_slack_q50_ms",
    "predicted_slack_q90_ms",
    "actual_slack_ms",
    "absolute_error_ms",
    "lower_covered",
    "interval_covered",
    "information_scope",
)
PROGRESS_LABELS = (
    "0%",
    "(0,10%]",
    "(10,20%]",
    "(20,30%]",
    "(30,40%]",
    "(40,50%]",
    "(50,60%]",
    "(60,70%]",
    "(70,80%]",
    "(80,90%]",
    "(90,100%)",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split(workflow_id: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{workflow_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket < 6:
        return "fit"
    return "calibration" if bucket < 8 else "evaluation"


def _quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return 0.0
    return float(np.quantile(array, q))


def _scale_tokens(
    row: dict[str, Any], replay: dict[str, Any]
) -> tuple[int, int]:
    """Mirror the archived workload client's proportional context scaling."""

    prompt_len = max(int(row.get("prompt_len", 1)), 1)
    output_len = max(int(row.get("output_len", 1)), 1)
    input_tokens = max(
        int(math.ceil(prompt_len * max(float(replay.get("input_scale", 1.0)), 0.0001))),
        128,
    )
    if replay.get("max_input_tokens") is not None:
        input_tokens = min(input_tokens, int(replay["max_input_tokens"]))
    output_tokens = output_len
    if replay.get("max_output_tokens") is not None:
        output_tokens = min(output_tokens, int(replay["max_output_tokens"]))
    output_tokens = max(output_tokens, 1)

    token_budget = max(int(replay.get("context_length", 131072)) - 1, 2)
    context_scale = min(1.0, token_budget / (input_tokens + output_tokens))
    if context_scale < 1.0:
        output_tokens = min(
            max(round(output_tokens * context_scale), 1),
            token_budget - 1,
        )
        input_tokens = token_budget - output_tokens
    return input_tokens, output_tokens


def _load_trace(path: Path) -> dict[str, list[dict[str, Any]]]:
    workflows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                workflows[str(row["workflow_id"])].append(row)
    for rows in workflows.values():
        rows.sort(
            key=lambda row: (
                int(row.get("stage_id", 0)),
                float(row.get("arrival_t", 0.0)),
                int(row.get("group_id", 0)),
                str(row.get("rid") or ""),
            )
        )
    return dict(workflows)


def _progress_bucket(progress: float) -> int:
    progress = max(min(float(progress), 0.999999), 0.0)
    if progress <= 1e-12:
        return 0
    return min(int(math.ceil(progress * 10.0)), 10)


def _symmetric_relative_error_pct(predicted: float, actual: float) -> float:
    """Return pointwise sMAPE in percent, with the all-zero case defined as 0."""

    predicted = abs(float(predicted))
    actual = abs(float(actual))
    denominator = predicted + actual
    if denominator <= 1e-12:
        return 0.0
    return 200.0 * abs(predicted - actual) / denominator


def _stage_snapshots(
    *,
    workflow_id: str,
    stage_id: int,
    stage_rows: dict[str, list[dict[str, Any]]],
    evaluation_rows: dict[tuple[int, str], dict[str, Any]],
    completion_ms: dict[str, float],
    replay: dict[str, Any],
    model: dict[str, float],
    slack_lower_margin_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    agents = sorted(stage_rows)
    request_ids: dict[str, list[str]] = {}
    request_costs_ms: dict[str, list[float]] = {}
    finish_ms: dict[str, float] = {}
    actual_duration_ms: dict[str, float] = {}

    for agent_id in agents:
        rows = sorted(
            stage_rows[agent_id],
            key=lambda row: (int(row.get("group_id", 0)), str(row.get("rid") or "")),
        )
        ids: list[str] = []
        costs: list[float] = []
        max_materialized_prompt = 0
        prior_completion = -math.inf
        for row in rows:
            request_id = str(row.get("rid") or "")
            if request_id not in completion_ms:
                raise ValueError(f"missing completion time for {request_id}")
            if completion_ms[request_id] < prior_completion:
                raise ValueError(f"non-monotonic branch completion order for {agent_id}")
            prior_completion = completion_ms[request_id]
            input_tokens, output_tokens = _scale_tokens(row, replay)
            new_prefill_tokens = max(input_tokens - max_materialized_prompt, 0)
            max_materialized_prompt = max(max_materialized_prompt, input_tokens)
            request_cost = (
                float(model["request_setup_ms"])
                + float(model["prefill_ms_per_token"]) * new_prefill_tokens
                + float(model["decode_ms_per_token"]) * output_tokens
            )
            ids.append(request_id)
            costs.append(max(request_cost, 0.0))
        request_ids[agent_id] = ids
        request_costs_ms[agent_id] = costs
        finish_ms[agent_id] = max(completion_ms[request_id] for request_id in ids)
        actual_duration_ms[agent_id] = float(
            evaluation_rows[(stage_id, agent_id)]["actual_duration_ms"]
        )

    start_candidates = [
        finish_ms[agent_id] - actual_duration_ms[agent_id]
        for agent_id in agents
    ]
    stage_start_ms = statistics.mean(start_candidates)
    start_spread_ms = max(start_candidates) - min(start_candidates)
    if start_spread_ms > 5.0:
        raise ValueError(
            f"stage-start reconstruction disagrees by {start_spread_ms:.3f} ms "
            f"for {workflow_id} stage {stage_id}"
        )

    total_cost_ms = {
        agent_id: sum(request_costs_ms[agent_id]) for agent_id in agents
    }
    completed_cost_ms = dict.fromkeys(agents, 0.0)
    completed_requests = dict.fromkeys(agents, 0)
    latest_completion_elapsed_ms: dict[str, float] = {}
    events: list[tuple[float, str, str, int]] = []
    for agent_id in agents:
        for index, request_id in enumerate(request_ids[agent_id]):
            events.append(
                (completion_ms[request_id], agent_id, request_id, index)
            )
    events.sort(key=lambda value: (value[0], value[1], value[2]))

    snapshots: list[dict[str, Any]] = []

    def emit(
        *,
        epoch: int,
        now_ms: float,
        trigger_agent_id: str,
        trigger_request_id: str,
        trigger_request_seq: int,
    ) -> None:
        elapsed_ms = max(now_ms - stage_start_ms, 0.0)
        remaining_lower_ms: dict[str, float] = {}
        remaining_q50_ms: dict[str, float] = {}
        remaining_upper_ms: dict[str, float] = {}
        actual_remaining_ms: dict[str, float] = {}

        for agent_id in agents:
            done = completed_requests[agent_id] >= len(request_ids[agent_id])
            if done:
                lower = median = upper = 0.0
            else:
                if completed_cost_ms[agent_id] > 0.0:
                    projected_total_ms = (
                        latest_completion_elapsed_ms[agent_id]
                        * total_cost_ms[agent_id]
                        / completed_cost_ms[agent_id]
                    )
                else:
                    projected_total_ms = total_cost_ms[agent_id]
                lower = max(
                    projected_total_ms * float(model["lower_scale"])
                    - elapsed_ms,
                    0.0,
                )
                median = max(projected_total_ms - elapsed_ms, 0.0)
                upper = max(
                    projected_total_ms * float(model["upper_scale"])
                    - elapsed_ms,
                    median,
                )
            remaining_lower_ms[agent_id] = lower
            remaining_q50_ms[agent_id] = median
            remaining_upper_ms[agent_id] = upper
            actual_remaining_ms[agent_id] = max(finish_ms[agent_id] - now_ms, 0.0)

        for agent_id in agents:
            if completed_requests[agent_id] >= len(request_ids[agent_id]):
                continue
            other_agents = [candidate for candidate in agents if candidate != agent_id]
            raw_lower = max(
                max(remaining_lower_ms[candidate] for candidate in other_agents)
                - remaining_upper_ms[agent_id],
                0.0,
            )
            predicted_lower = max(raw_lower - slack_lower_margin_ms, 0.0)
            predicted_q50 = max(
                max(remaining_q50_ms[candidate] for candidate in other_agents)
                - remaining_q50_ms[agent_id],
                predicted_lower,
            )
            predicted_upper = max(
                max(remaining_upper_ms[candidate] for candidate in other_agents)
                - remaining_lower_ms[agent_id],
                predicted_q50,
            )
            actual_slack = max(
                max(actual_remaining_ms[candidate] for candidate in other_agents)
                - actual_remaining_ms[agent_id],
                0.0,
            )
            progress = completed_requests[agent_id] / len(request_ids[agent_id])
            snapshots.append(
                {
                    "workflow_id": workflow_id,
                    "workflow_short": workflow_id[:8],
                    "stage_id": stage_id,
                    "stage_width": len(agents),
                    "agent_id": agent_id,
                    "agent_short": agent_id.split("-")[-1],
                    "stage_slack_epoch": epoch,
                    "event_elapsed_ms": elapsed_ms,
                    "trigger_agent_id": trigger_agent_id,
                    "trigger_request_id": trigger_request_id,
                    "trigger_request_seq": trigger_request_seq,
                    "agent_next_request_seq": completed_requests[agent_id] + 1,
                    "agent_progress_fraction": progress,
                    "predicted_remaining_lcb_ms": remaining_lower_ms[agent_id],
                    "predicted_remaining_q50_ms": remaining_q50_ms[agent_id],
                    "predicted_remaining_q90_ms": remaining_upper_ms[agent_id],
                    "actual_remaining_ms": actual_remaining_ms[agent_id],
                    "predicted_slack_lcb_ms": predicted_lower,
                    "predicted_slack_q50_ms": predicted_q50,
                    "predicted_slack_q90_ms": predicted_upper,
                    "actual_slack_ms": actual_slack,
                    "absolute_error_ms": abs(predicted_q50 - actual_slack),
                    "lower_covered": actual_slack >= predicted_lower,
                    "interval_covered": predicted_lower <= actual_slack <= predicted_upper,
                    "information_scope": "fixed_dag_completion_prefix_only",
                }
            )

    emit(
        epoch=0,
        now_ms=stage_start_ms,
        trigger_agent_id="stage_start",
        trigger_request_id="",
        trigger_request_seq=0,
    )
    for epoch, (finish_time, agent_id, request_id, request_index) in enumerate(
        events, start=1
    ):
        completed_cost_ms[agent_id] += request_costs_ms[agent_id][request_index]
        completed_requests[agent_id] += 1
        latest_completion_elapsed_ms[agent_id] = max(
            finish_time - stage_start_ms, 0.0
        )
        emit(
            epoch=epoch,
            now_ms=finish_time,
            trigger_agent_id=agent_id,
            trigger_request_id=request_id,
            trigger_request_seq=request_index + 1,
        )

    audit = {
        "workflow_id": workflow_id,
        "stage_id": stage_id,
        "stage_width": len(agents),
        "request_completion_events": len(events),
        "actionable_snapshots": len(snapshots),
        "stage_start_elapsed_ms": stage_start_ms,
        "stage_start_reconstruction_spread_ms": start_spread_ms,
    }
    return snapshots, audit


def build_dynamic_snapshots(
    *,
    result_path: Path,
    trace_path: Path,
    calibration_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("semantics") != "fixed_dag_cpm_slack_v2":
        raise ValueError("dynamic replay requires fixed_dag_cpm_slack_v2 calibration")
    if not bool((calibration.get("evaluation_gate") or {}).get("passed", False)):
        raise ValueError("Algorithm-1 calibration evaluation gate did not pass")
    split_salt = str(calibration["split_salt"])
    model = dict(calibration["request_cost_model"])
    slack_lower_margin_ms = float(calibration.get("slack_lower_margin_ms", 0.0))
    trace = _load_trace(trace_path)
    replay = dict(result.get("replay") or {})

    snapshots: list[dict[str, Any]] = []
    stage_audits: list[dict[str, Any]] = []
    evaluation_workflows: set[str] = set()
    for workflow in result.get("workflows", []):
        workflow_id = str(workflow["workflow_id"])
        if _split(workflow_id, split_salt) != "evaluation":
            continue
        evaluation_workflows.add(workflow_id)
        if workflow_id not in trace:
            raise ValueError(f"workflow missing from trace: {workflow_id}")
        completion_ms = {
            str(row["rid"]): float(row["completion_elapsed_s"]) * 1000.0
            for row in workflow.get("request_routes", [])
        }
        evaluation_rows = {
            (int(row["stage_id"]), str(row["agent_id"])): row
            for row in workflow.get("algorithm1_slack_evaluation", [])
        }
        by_stage: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in trace[workflow_id]:
            by_stage[int(row.get("stage_id", 0))][str(row.get("agent_id") or "")].append(row)
        for stage_id, stage_rows in sorted(by_stage.items()):
            if len(stage_rows) <= 1:
                continue
            replay_rows, audit = _stage_snapshots(
                workflow_id=workflow_id,
                stage_id=stage_id,
                stage_rows=dict(stage_rows),
                evaluation_rows=evaluation_rows,
                completion_ms=completion_ms,
                replay=replay,
                model=model,
                slack_lower_margin_ms=slack_lower_margin_ms,
            )
            snapshots.extend(replay_rows)
            stage_audits.append(audit)

    if not snapshots:
        raise ValueError("no evaluation fan-out snapshots were generated")
    for row in snapshots:
        lower = float(row["predicted_slack_lcb_ms"])
        median = float(row["predicted_slack_q50_ms"])
        upper = float(row["predicted_slack_q90_ms"])
        if not 0.0 <= lower <= median <= upper:
            raise ValueError(f"unordered Slack envelope: {row}")

    source = {
        "result": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": _sha256(calibration_path),
        "calibration_id": calibration.get("calibration_id"),
        "split_salt": split_salt,
        "evidence_tier": result.get("evidence_tier"),
        "model": result.get("model"),
        "dp_size": result.get("dp_size"),
        "pp_size": result.get("pp_size"),
        "context_length": replay.get("context_length"),
        "evaluation_workflows": len(evaluation_workflows),
        "stage_audits": stage_audits,
    }
    return snapshots, source


def summarize(
    snapshots: list[dict[str, Any]], source: dict[str, Any]
) -> dict[str, Any]:
    errors = np.asarray([float(row["absolute_error_ms"]) for row in snapshots])
    actual = np.asarray([float(row["actual_slack_ms"]) for row in snapshots])
    lower = np.asarray(
        [float(row["predicted_slack_lcb_ms"]) for row in snapshots]
    )
    upper = np.asarray(
        [float(row["predicted_slack_q90_ms"]) for row in snapshots]
    )
    by_progress: list[dict[str, Any]] = []
    for bucket, label in enumerate(PROGRESS_LABELS):
        bucket_rows = [
            row
            for row in snapshots
            if _progress_bucket(float(row["agent_progress_fraction"])) == bucket
        ]
        if not bucket_rows:
            continue
        bucket_errors = [float(row["absolute_error_ms"]) for row in bucket_rows]
        bucket_relative_errors = [
            _symmetric_relative_error_pct(
                float(row["predicted_slack_q50_ms"]),
                float(row["actual_slack_ms"]),
            )
            for row in bucket_rows
        ]
        by_progress.append(
            {
                "bucket": bucket,
                "label": label,
                "progress_midpoint_pct": 0.0 if bucket == 0 else bucket * 10.0 - 5.0,
                "snapshots": len(bucket_rows),
                "mean_absolute_error_ms": float(np.mean(bucket_errors)),
                "mean_symmetric_relative_error_pct": float(
                    np.mean(bucket_relative_errors)
                ),
            }
        )

    distinct_trajectories: dict[tuple[str, int, str], set[float]] = defaultdict(set)
    for row in snapshots:
        distinct_trajectories[
            (str(row["workflow_id"]), int(row["stage_id"]), str(row["agent_id"]))
        ].add(round(float(row["predicted_slack_q50_ms"]), 6))
    changing_trajectories = sum(
        len(values) > 1 for values in distinct_trajectories.values()
    )

    return {
        "schema_version": 2,
        "figure": "alg1-dynamic-slack-accuracy",
        "semantics": "fixed_dag_request_completion_causal_replay_v1",
        "source_status": "archived_sglsim_online_measured_dynamic_replay",
        "prediction_information": (
            "known fixed DAG, fitted request-cost model, and request completion "
            "prefixes available through the current stage_slack_epoch"
        ),
        "ground_truth_information": (
            "future branch completion timestamps used only after replay to "
            "backfill actual residual Slack"
        ),
        "relative_error_definition": (
            "pointwise symmetric absolute percentage error: "
            "200 * |predicted_q50 - actual| / "
            "(|predicted_q50| + |actual|); defined as 0 when both are 0"
        ),
        "source": source,
        "selection": {
            "split": "evaluation",
            "fanout_only": True,
            "workflows": len({str(row["workflow_id"]) for row in snapshots}),
            "fanout_stages": len(
                {
                    (str(row["workflow_id"]), int(row["stage_id"]))
                    for row in snapshots
                }
            ),
            "agent_trajectories": len(distinct_trajectories),
            "changing_agent_trajectories": changing_trajectories,
            "actionable_snapshots": len(snapshots),
            "slack_epochs": len(
                {
                    (
                        str(row["workflow_id"]),
                        int(row["stage_id"]),
                        int(row["stage_slack_epoch"]),
                    )
                    for row in snapshots
                }
            ),
        },
        "point_accuracy_ms": {
            "mae": float(np.mean(errors)),
            "rmse": float(math.sqrt(float(np.mean(errors**2)))),
            "median_absolute_error": _quantile(errors, 0.50),
            "p90_absolute_error": _quantile(errors, 0.90),
        },
        "uncertainty": {
            "target_lower_coverage": 0.90,
            "observed_lower_coverage": float(np.mean(actual >= lower)),
            "lcb_to_q90_interval_coverage": float(
                np.mean((actual >= lower) & (actual <= upper))
            ),
            "nonzero_lower_bound_fraction": float(np.mean(lower > 0.0)),
        },
        "accuracy_by_agent_progress": by_progress,
        "interpretation": (
            "Slack now changes at request-completion epochs, but this simple "
            "progress extrapolator is inaccurate and its conservative lower "
            "bound remains mostly zero. It is an instrumentation and baseline "
            "result, not evidence for the proposed online quantile GBDT."
        ),
    }


def _write_snapshots(path: Path, snapshots: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SNAPSHOT_FIELDS)
        writer.writeheader()
        writer.writerows(snapshots)


def render(
    *,
    summary: dict[str, Any],
    output_prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, absolute_ax = plt.subplots(figsize=(8.8, 4.8))
    relative_ax = absolute_ax.twinx()

    progress_rows = summary["accuracy_by_agent_progress"]
    positions = np.arange(len(progress_rows), dtype=float)
    mean_absolute_error = np.asarray(
        [float(row["mean_absolute_error_ms"]) / 1000.0 for row in progress_rows]
    )
    mean_relative_error = np.asarray(
        [float(row["mean_symmetric_relative_error_pct"]) for row in progress_rows]
    )
    absolute_line = absolute_ax.plot(
        positions,
        mean_absolute_error,
        color="#2878B5",
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        label="Mean absolute error",
    )
    relative_line = relative_ax.plot(
        positions,
        mean_relative_error,
        color="#D95F45",
        marker="s",
        linewidth=2.0,
        markersize=5.2,
        label="Mean symmetric relative error",
    )
    tick_labels = [
        (
            "0"
            if int(row["bucket"]) == 0
            else f"{(int(row['bucket']) - 1) * 10}–{int(row['bucket']) * 10}"
        )
        for row in progress_rows
    ]
    absolute_ax.set_xticks(positions, tick_labels, rotation=32, ha="right")
    absolute_ax.set_xlabel("Observed agent progress (%)")
    absolute_ax.set_ylabel("Mean absolute q50 Slack error (s)", color="#2878B5")
    relative_ax.set_ylabel("Mean symmetric relative error (%)", color="#D95F45")
    absolute_ax.tick_params(axis="y", colors="#2878B5")
    relative_ax.tick_params(axis="y", colors="#D95F45")
    absolute_ax.spines["left"].set_color("#2878B5")
    relative_ax.spines["right"].set_color("#D95F45")
    absolute_ax.set_ylim(bottom=0.0)
    relative_ax.set_ylim(0.0, 200.0)
    absolute_ax.grid(True, axis="both", linewidth=0.5, alpha=0.24)
    absolute_ax.legend(
        handles=absolute_line + relative_line,
        loc="upper right",
        frameon=False,
    )

    selection = summary["selection"]
    fig.suptitle(
        "Dynamic Slack prediction error vs. observed agent progress",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.925,
        (
            f"Evaluation-only: {selection['workflows']} workflows · "
            f"{selection['fanout_stages']} fan-out stages · "
            f"{selection['actionable_snapshots']} actionable snapshots · "
            "10%-progress bins · Qwen3-32B · DP4 x PP4"
        ),
        ha="center",
        va="top",
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Causal fixed-DAG baseline. Relative error uses pointwise sMAPE "
            "(0% when prediction and actual are both zero; range 0–200%)."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.89))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    snapshots, source = build_dynamic_snapshots(
        result_path=args.result,
        trace_path=args.trace,
        calibration_path=args.calibration,
    )
    summary = summarize(snapshots, source)
    _write_snapshots(args.snapshots, snapshots)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    render(
        summary=summary,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
