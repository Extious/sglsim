#!/usr/bin/env python3
"""Build a leakage-free, workflow-disjoint Algorithm-1 baseline artifact.

The model intentionally uses only information available at a stage/request
observation: elapsed time, completed-request count, role/stage metadata, and
prompt lengths for requests revealed so far.  Trace ``output_len`` is never a
feature.  Runtime completion timestamps are labels and are read only while
building fit/calibration/evaluation samples.

This is an evidence generator, not a claim that a linear model is the final
predictor.  A split-conformal margin is fitted on calibration workflows and
the evaluation partition is touched only once for final metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "Archived" / "2026-08-06-pre-kvcache-centric"
DEFAULT_TRACE = (
    ARCHIVE
    / "presentations"
    / "presentation"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "weka_subagent_cdf_24w_workload.jsonl"
)
DEFAULT_RESULT = (
    ARCHIVE
    / "experiments"
    / "demo"
    / "results"
    / "qwen3_32b_dp4_pp4_alg1_formal_calibration_v2"
    / "20260804-021032"
    / "no_backup.json"
)
DEFAULT_CALIBRATION = (
    ARCHIVE
    / "presentations"
    / "presentation"
    / "experiments"
    / "alg1_slack_calibration.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
FORBIDDEN_FEATURE_FRAGMENTS = (
    "output_len",
    "final_output",
    "ground_truth",
    "actual",
    "finish",
    "future",
    "oracle",
    "posthoc",
)
FEATURE_NAMES = (
    "log_elapsed_ms",
    "log_time_since_own_completion_ms",
    "completed_request_count",
    "completed_request_count_squared",
    "revealed_request_count",
    "log_current_prompt_tokens",
    "log_max_revealed_prompt_tokens",
    "log_sum_revealed_prompt_tokens",
    "stage_width",
    "stage_id",
    "role_is_subagent",
    "has_no_completion",
    "role_x_log_current_prompt",
    "completed_x_log_elapsed",
)


@dataclass(frozen=True)
class Sample:
    workflow_id: str
    split: str
    stage_id: int
    stage_width: int
    epoch: int
    event_elapsed_ms: float
    agent_id: str
    role: str
    actual_remaining_ms: float
    features: tuple[float, ...]

    @property
    def snapshot_key(self) -> tuple[str, int, int]:
        return (self.workflow_id, self.stage_id, self.epoch)


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


def _finite_sample_upper_quantile(
    values: Iterable[float], coverage: float
) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(value))
    if not ordered:
        raise ValueError("calibration requires at least one finite score")
    rank = math.ceil((len(ordered) + 1) * float(coverage))
    return ordered[min(max(rank - 1, 0), len(ordered) - 1)]


def _load_trace(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    workflows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            workflow_id = str(row["workflow_id"])
            rid = str(row["rid"])
            workflows[workflow_id][rid] = row
    return dict(workflows)


def _role(row: dict[str, Any]) -> str:
    raw = str(row.get("source_model_role") or "").strip().lower()
    if raw:
        return raw
    return "subagent" if "sub" in str(row.get("agent_id", "")).lower() else "main"


def _prompt_tokens(row: dict[str, Any], context_length: int) -> int:
    # Prompt length is admission-time information.  Do not reproduce the
    # archived proportional prompt/output scaling because it consumes the
    # post-hoc output length.  The cap is deterministic and output-independent.
    return min(max(int(row.get("prompt_len", 1)), 1), max(context_length - 1, 1))


def _features(
    *,
    elapsed_ms: float,
    last_completion_elapsed_ms: float | None,
    completed_count: int,
    revealed_rows: list[dict[str, Any]],
    stage_width: int,
    stage_id: int,
    role: str,
    context_length: int,
) -> tuple[float, ...]:
    prompts = [_prompt_tokens(row, context_length) for row in revealed_rows]
    current_prompt = prompts[-1]
    log_elapsed = math.log1p(max(elapsed_ms, 0.0))
    since_own = (
        elapsed_ms
        if last_completion_elapsed_ms is None
        else max(elapsed_ms - last_completion_elapsed_ms, 0.0)
    )
    role_subagent = 1.0 if role == "subagent" else 0.0
    return (
        log_elapsed,
        math.log1p(since_own),
        float(completed_count),
        float(completed_count * completed_count),
        float(len(revealed_rows)),
        math.log1p(current_prompt),
        math.log1p(max(prompts)),
        math.log1p(sum(prompts)),
        float(stage_width),
        float(stage_id),
        role_subagent,
        1.0 if last_completion_elapsed_ms is None else 0.0,
        role_subagent * math.log1p(current_prompt),
        float(completed_count) * log_elapsed,
    )


def _build_samples(
    *,
    trace: dict[str, dict[str, dict[str, Any]]],
    result: dict[str, Any],
    split_salt: str,
) -> tuple[list[Sample], list[dict[str, Any]]]:
    context_length = int(result.get("replay", {}).get("context_length", 131072))
    samples: list[Sample] = []
    stage_audits: list[dict[str, Any]] = []
    for workflow in result.get("workflows", ()):
        workflow_id = str(workflow.get("workflow_id") or "")
        trace_rows = trace.get(workflow_id)
        if not workflow_id or not trace_rows:
            continue
        completion_ms = {
            str(row["rid"]): float(row["completion_elapsed_s"]) * 1000.0
            for row in workflow.get("request_routes", ())
        }
        evaluation = {
            (int(row["stage_id"]), str(row["agent_id"])): row
            for row in workflow.get("algorithm1_slack_evaluation", ())
        }
        by_stage: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for rid, row in trace_rows.items():
            if rid not in completion_ms:
                continue
            by_stage[int(row.get("stage_id", 0))][str(row["agent_id"])].append(row)

        for stage_id, raw_agents in sorted(by_stage.items()):
            if len(raw_agents) < 2:
                continue
            agents = {
                agent_id: sorted(
                    rows,
                    key=lambda row: (
                        int(row.get("group_id", 0)),
                        str(row.get("rid") or ""),
                    ),
                )
                for agent_id, rows in raw_agents.items()
            }
            if any((stage_id, agent_id) not in evaluation for agent_id in agents):
                continue
            finish_ms = {
                agent_id: max(completion_ms[str(row["rid"])] for row in rows)
                for agent_id, rows in agents.items()
            }
            start_candidates = [
                finish_ms[agent_id]
                - float(evaluation[(stage_id, agent_id)]["actual_duration_ms"])
                for agent_id in agents
            ]
            stage_start_ms = float(np.mean(start_candidates))
            start_spread_ms = max(start_candidates) - min(start_candidates)
            if start_spread_ms > 5.0:
                raise ValueError(
                    f"stage-start spread {start_spread_ms:.3f} ms exceeds audit "
                    f"limit for {workflow_id} stage {stage_id}"
                )
            event_times = sorted(
                {
                    stage_start_ms,
                    *(
                        completion_ms[str(row["rid"])]
                        for rows in agents.values()
                        for row in rows
                    ),
                }
            )
            actionable_epochs = 0
            for epoch, event_ms in enumerate(event_times):
                active = [
                    agent_id
                    for agent_id in sorted(agents)
                    if finish_ms[agent_id] > event_ms + 1e-9
                ]
                if len(active) < 2:
                    continue
                actionable_epochs += 1
                elapsed_ms = max(event_ms - stage_start_ms, 0.0)
                for agent_id in active:
                    rows = agents[agent_id]
                    completed_rows = [
                        row
                        for row in rows
                        if completion_ms[str(row["rid"])] <= event_ms + 1e-9
                    ]
                    completed_count = len(completed_rows)
                    next_index = min(completed_count, len(rows) - 1)
                    revealed_rows = rows[: next_index + 1]
                    last_completion = (
                        completion_ms[str(completed_rows[-1]["rid"])]
                        - stage_start_ms
                        if completed_rows
                        else None
                    )
                    role = _role(rows[0])
                    samples.append(
                        Sample(
                            workflow_id=workflow_id,
                            split=_split(workflow_id, split_salt),
                            stage_id=stage_id,
                            stage_width=len(agents),
                            epoch=epoch,
                            event_elapsed_ms=elapsed_ms,
                            agent_id=agent_id,
                            role=role,
                            actual_remaining_ms=max(
                                finish_ms[agent_id] - event_ms, 0.0
                            ),
                            features=_features(
                                elapsed_ms=elapsed_ms,
                                last_completion_elapsed_ms=last_completion,
                                completed_count=completed_count,
                                revealed_rows=revealed_rows,
                                stage_width=len(agents),
                                stage_id=stage_id,
                                role=role,
                                context_length=context_length,
                            ),
                        )
                    )
            stage_audits.append(
                {
                    "workflow_id": workflow_id,
                    "stage_id": stage_id,
                    "stage_width": len(agents),
                    "stage_start_reconstruction_spread_ms": start_spread_ms,
                    "actionable_epochs": actionable_epochs,
                }
            )
    return samples, stage_audits


def _fit_ridge(
    samples: list[Sample], *, ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("fit partition is empty")
    features = np.asarray([sample.features for sample in samples], dtype=float)
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-9] = 1.0
    normalized = (features - means) / scales
    design = np.column_stack([np.ones(len(samples)), normalized])
    targets = np.log1p(
        np.asarray([sample.actual_remaining_ms for sample in samples], dtype=float)
    )
    counts = Counter(sample.workflow_id for sample in samples)
    weights = np.asarray(
        [1.0 / counts[sample.workflow_id] for sample in samples], dtype=float
    )
    weights *= len(samples) / weights.sum()
    root_weights = np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    weighted_targets = targets * root_weights
    penalty = np.eye(design.shape[1]) * max(float(ridge), 0.0)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_targets,
    )
    return coefficients, means, scales


def _predict_log(
    samples: list[Sample],
    coefficients: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    features = np.asarray([sample.features for sample in samples], dtype=float)
    normalized = (features - means) / scales
    design = np.column_stack([np.ones(len(samples)), normalized])
    return design @ coefficients


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_rank = _rank(left)
    right_rank = _rank(right)
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _slack_rows(
    samples: list[Sample],
    lower_ms: np.ndarray,
    median_ms: np.ndarray,
    upper_ms: np.ndarray,
    *,
    slack_margin_ms: float,
) -> list[dict[str, Any]]:
    indexed = list(zip(samples, lower_ms, median_ms, upper_ms))
    by_snapshot: dict[
        tuple[str, int, int], list[tuple[Sample, float, float, float]]
    ] = defaultdict(list)
    for row in indexed:
        by_snapshot[row[0].snapshot_key].append(row)
    output: list[dict[str, Any]] = []
    for snapshot_key, rows in sorted(by_snapshot.items()):
        actual_values = [row[0].actual_remaining_ms for row in rows]
        predicted_values = [float(row[2]) for row in rows]
        rank_correlation = _spearman(predicted_values, actual_values)
        for sample, lower, median, upper in rows:
            others = [row for row in rows if row[0].agent_id != sample.agent_id]
            raw_lower = max(
                max((float(row[1]) for row in others), default=0.0) - float(upper),
                0.0,
            )
            predicted_median = max(
                max((float(row[2]) for row in others), default=0.0) - float(median),
                0.0,
            )
            predicted_upper = max(
                max((float(row[3]) for row in others), default=0.0) - float(lower),
                predicted_median,
            )
            actual_slack = max(
                max(
                    (row[0].actual_remaining_ms for row in others),
                    default=0.0,
                )
                - sample.actual_remaining_ms,
                0.0,
            )
            calibrated_lower = max(raw_lower - slack_margin_ms, 0.0)
            output.append(
                {
                    "workflow_id": sample.workflow_id,
                    "split": sample.split,
                    "stage_id": sample.stage_id,
                    "stage_width": sample.stage_width,
                    "epoch": sample.epoch,
                    "event_elapsed_ms": round(sample.event_elapsed_ms, 6),
                    "agent_id": sample.agent_id,
                    "role": sample.role,
                    "predicted_remaining_lower_ms": round(float(lower), 6),
                    "predicted_remaining_q50_ms": round(float(median), 6),
                    "predicted_remaining_upper_ms": round(float(upper), 6),
                    "actual_remaining_ms": round(sample.actual_remaining_ms, 6),
                    "raw_slack_lower_ms": round(raw_lower, 6),
                    "predicted_slack_lcb_ms": round(calibrated_lower, 6),
                    "predicted_slack_q50_ms": round(predicted_median, 6),
                    "predicted_slack_upper_ms": round(predicted_upper, 6),
                    "actual_slack_ms": round(actual_slack, 6),
                    "snapshot_spearman": rank_correlation,
                    "information_scope": (
                        "causal_elapsed_completion_prefix_prompt_only"
                    ),
                }
            )
    return output


def _metrics(rows: list[dict[str, Any]], *, target_coverage: float) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    errors = np.asarray(
        [
            abs(row["predicted_slack_q50_ms"] - row["actual_slack_ms"])
            for row in rows
        ],
        dtype=float,
    )
    remaining_errors = np.asarray(
        [
            abs(
                row["predicted_remaining_q50_ms"]
                - row["actual_remaining_ms"]
            )
            for row in rows
        ],
        dtype=float,
    )
    correlations = [
        float(row["snapshot_spearman"])
        for row in rows
        if row["snapshot_spearman"] is not None
    ]
    covered = [
        row["actual_slack_ms"] + 1e-9 >= row["predicted_slack_lcb_ms"]
        for row in rows
    ]
    actual_critical = [row["actual_slack_ms"] <= 1.0 for row in rows]
    predicted_critical = [row["predicted_slack_q50_ms"] <= 1.0 for row in rows]
    true_positive = sum(a and p for a, p in zip(actual_critical, predicted_critical))
    predicted_positive = sum(predicted_critical)
    actual_positive = sum(actual_critical)
    return {
        "rows": len(rows),
        "workflows": len({row["workflow_id"] for row in rows}),
        "snapshots": len(
            {(row["workflow_id"], row["stage_id"], row["epoch"]) for row in rows}
        ),
        "target_lower_coverage": target_coverage,
        "observed_lower_coverage": sum(covered) / len(covered),
        "nonzero_lower_bound_fraction": sum(
            row["predicted_slack_lcb_ms"] > 1e-9 for row in rows
        )
        / len(rows),
        "slack_mae_ms": float(np.mean(errors)),
        "slack_median_absolute_error_ms": float(np.median(errors)),
        "slack_p90_absolute_error_ms": float(np.quantile(errors, 0.9)),
        "remaining_mae_ms": float(np.mean(remaining_errors)),
        "median_snapshot_spearman": (
            float(np.median(correlations)) if correlations else None
        ),
        "critical_recall": true_positive / actual_positive if actual_positive else None,
        "critical_precision": (
            true_positive / predicted_positive if predicted_positive else None
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty snapshot CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()

    forbidden_features = [
        name
        for name in FEATURE_NAMES
        if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
    ]
    if forbidden_features:
        raise ValueError(f"noncausal feature names: {forbidden_features}")

    trace = _load_trace(args.trace)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    old_calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    split_salt = str(old_calibration.get("split_salt") or "")
    if not split_salt:
        raise ValueError("split_salt is required")
    target_coverage = float(old_calibration.get("target_lower_coverage", 0.9))
    samples, stage_audits = _build_samples(
        trace=trace,
        result=result,
        split_salt=split_salt,
    )
    partitions = {
        split: [sample for sample in samples if sample.split == split]
        for split in ("fit", "calibration", "evaluation")
    }
    coefficients, means, scales = _fit_ridge(
        partitions["fit"], ridge=args.ridge
    )
    fit_log = _predict_log(partitions["fit"], coefficients, means, scales)
    fit_actual_log = np.log1p(
        np.asarray(
            [sample.actual_remaining_ms for sample in partitions["fit"]],
            dtype=float,
        )
    )
    residuals = fit_actual_log - fit_log
    residual_quantiles = {
        "lower": float(np.quantile(residuals, 0.10)),
        "median": float(np.quantile(residuals, 0.50)),
        "upper": float(np.quantile(residuals, 0.90)),
    }

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    raw_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, split_samples in partitions.items():
        predicted_log = _predict_log(split_samples, coefficients, means, scales)
        lower = np.maximum(
            np.expm1(predicted_log + residual_quantiles["lower"]), 0.0
        )
        median = np.maximum(
            np.expm1(predicted_log + residual_quantiles["median"]), 0.0
        )
        upper = np.maximum(
            np.expm1(predicted_log + residual_quantiles["upper"]), median
        )
        raw_rows_by_split[split] = _slack_rows(
            split_samples,
            lower,
            median,
            upper,
            slack_margin_ms=0.0,
        )

    calibration_scores = [
        row["raw_slack_lower_ms"] - row["actual_slack_ms"]
        for row in raw_rows_by_split["calibration"]
    ]
    slack_margin_ms = max(
        _finite_sample_upper_quantile(calibration_scores, target_coverage),
        0.0,
    )
    for split, split_samples in partitions.items():
        predicted_log = _predict_log(split_samples, coefficients, means, scales)
        lower = np.maximum(
            np.expm1(predicted_log + residual_quantiles["lower"]), 0.0
        )
        median = np.maximum(
            np.expm1(predicted_log + residual_quantiles["median"]), 0.0
        )
        upper = np.maximum(
            np.expm1(predicted_log + residual_quantiles["upper"]), median
        )
        rows_by_split[split] = _slack_rows(
            split_samples,
            lower,
            median,
            upper,
            slack_margin_ms=slack_margin_ms,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [
        row
        for split in ("fit", "calibration", "evaluation")
        for row in rows_by_split[split]
    ]
    _write_csv(output_dir / "causal_slack_snapshots.csv", all_rows)
    artifact = {
        "schema_version": "aegiskv-causal-slack-baseline-v1",
        "evidence_class": "workflow_disjoint_causal_baseline",
        "paper_ready": False,
        "semantics": "revealed_dag_causal_slack_v1",
        "information_scope": "elapsed_completion_prefix_prompt_only",
        "uses_future_information": False,
        "output_len_used_as_feature": False,
        "prediction_inputs_use_future_information": False,
        "label_caveat": (
            "Actual completion labels come from an archived SGLSim replay whose "
            "proportional context-overflow policy used trace output length. This "
            "does not enter model features, but prevents treating the artifact "
            "as a clean paper-ready online evaluation."
        ),
        "stage_start_reconstruction": (
            "posthoc reconstruction for evaluation coordinates only; online "
            "runtime observes stage start directly"
        ),
        "feature_names": list(FEATURE_NAMES),
        "split_salt": split_salt,
        "workflow_disjoint": True,
        "ridge": args.ridge,
        "target_lower_coverage": target_coverage,
        "slack_lower_margin_ms": slack_margin_ms,
        "duration_model": {
            "target": "log1p(actual_remaining_ms)",
            "coefficients": coefficients.tolist(),
            "feature_means": means.tolist(),
            "feature_scales": scales.tolist(),
            "fit_residual_log_quantiles": residual_quantiles,
        },
        "partitions": {
            split: {
                "workflows": len(
                    {sample.workflow_id for sample in partitions[split]}
                ),
                "rows": len(partitions[split]),
                "metrics": _metrics(
                    rows_by_split[split], target_coverage=target_coverage
                ),
            }
            for split in ("fit", "calibration", "evaluation")
        },
        "evaluation_gate": {
            "passed": False,
            "status": "NOT_PRE_REGISTERED",
            "reason": (
                "This first leakage-free baseline has no pre-registered error, "
                "coverage, rank, or nonzero-LCB thresholds."
            ),
        },
        "runtime_ticket_status": {
            "slack_kind": "UNKNOWN",
            "usable": False,
            "effective_slack_ms": 0.0,
            "reason": "evaluation gate is not passed",
        },
        "stage_audits": stage_audits,
        "source": {
            "trace": str(args.trace.resolve()),
            "trace_sha256": _sha256(args.trace),
            "runtime_result": str(args.result.resolve()),
            "runtime_result_sha256": _sha256(args.result),
            "legacy_split_artifact": str(args.calibration.resolve()),
            "legacy_split_artifact_sha256": _sha256(args.calibration),
        },
        "limitations": [
            "small 19-workflow archived SGLSim sample",
            "linear ridge duration baseline, not the proposed final predictor",
            "stage start reconstructed because the archived result omitted it",
            "completion-event snapshots only; no token-level in-flight progress",
            (
                "archived proportional context scaling used post-hoc output length "
                "while generating timing labels"
            ),
            "SGLSim timing labels, not physical serving measurements",
            "evaluation gate intentionally remains unpassed",
        ],
    }
    artifact_path = output_dir / "causal_slack_baseline.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    evaluation_metrics = artifact["partitions"]["evaluation"]["metrics"]
    print("evidence_class=workflow_disjoint_causal_baseline")
    print(f"evaluation_workflows={evaluation_metrics['workflows']}")
    print(f"evaluation_rows={evaluation_metrics['rows']}")
    print(
        "evaluation_slack_mae_ms="
        f"{evaluation_metrics['slack_mae_ms']:.6f}"
    )
    print(
        "evaluation_lower_coverage="
        f"{evaluation_metrics['observed_lower_coverage']:.6f}"
    )
    print(
        "evaluation_nonzero_lcb_fraction="
        f"{evaluation_metrics['nonzero_lower_bound_fraction']:.6f}"
    )
    print("evaluation_gate=NOT_PRE_REGISTERED")
    print(f"output={artifact_path}")


if __name__ == "__main__":
    main()
