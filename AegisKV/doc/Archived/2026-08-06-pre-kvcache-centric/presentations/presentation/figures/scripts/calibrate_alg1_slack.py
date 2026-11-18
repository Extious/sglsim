#!/usr/bin/env python3
"""Fit and validate Algorithm 1 from DP4/PP4 runtime observations.

This calibrator intentionally consumes the fixed-DAG ``algorithm1_slack_evaluation``
rows emitted by the online SGLSim workflow client.  Legacy
``actual_remaining_s`` duration-prediction CSVs are not accepted: duration and
CPM slack are different targets.

Workflow IDs are deterministically split into cost-model fit, conformal
calibration, and final evaluation partitions.  Repeated policies/runs for one
workflow always remain in the same partition, so runtime noise cannot leak a
workflow across the gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AegisKV.demo.wct_cdf_client import load_workflows
from AegisKV.demo.workflow_client import _scaled_tokens
from sglsim.srt.simulation.aegis_slack import (
    ONLINE_PROGRESS_MIN_FRACTION,
    AegisOnlineProgressTracker,
    calibrated_lower_margin_s,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    ROOT / "presentation" / "experiments" / "alg1_slack_calibration_candidate.json"
)
DEFAULT_PROMOTED_OUTPUT = (
    ROOT / "presentation" / "experiments" / "alg1_slack_calibration.json"
)
DEFAULT_AUDIT = (
    ROOT
    / "presentation"
    / "figures"
    / "data"
    / "measured"
    / "fig_alg1_runtime_slack_calibration.csv"
)
# Chosen once before collecting runtime rows.  On the 19 workflows that pass
# the fixed fan-out/fan-in and <=128-request filter it yields 11/3/5
# workflow-disjoint fit/calibration/evaluation partitions; never tune this salt
# after observing accuracy.
SPLIT_SALT = "aegiskv-fixed-dag-runtime-v2-2"
SEMANTICS = "fixed_dag_cpm_slack_v2"
FEATURES = (
    "request_count",
    "new_prefill_tokens",
    "output_tokens",
)
MODEL_KEYS = (
    "request_setup_ms",
    "prefill_ms_per_token",
    "decode_ms_per_token",
)


@dataclass(frozen=True)
class RuntimeProgressObservation:
    request_id: str
    completion_elapsed_ms: float
    new_prefill_tokens: int
    output_tokens: int

    @property
    def features(self) -> tuple[float, float, float]:
        return (1.0, float(self.new_prefill_tokens), float(self.output_tokens))


@dataclass(frozen=True)
class RuntimeRow:
    source: str
    policy: str
    workflow_id: str
    stage_id: int
    agent_id: str
    stage_width: int
    request_count: int
    new_prefill_tokens: int
    output_tokens: int
    actual_duration_ms: float
    actual_slack_ms: float
    evidence_tier: str
    model: str
    dp_size: int
    pp_size: int
    context_length: int
    progress_observations: tuple[RuntimeProgressObservation, ...] = ()

    @property
    def sample_key(self) -> tuple[str, str, int]:
        return (self.source, self.workflow_id, self.stage_id)

    @property
    def features(self) -> tuple[float, float, float]:
        return (
            float(self.request_count),
            float(self.new_prefill_tokens),
            float(self.output_tokens),
        )


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(max(float(quantile), 0.0), 1.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _split(workflow_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{workflow_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10
    if bucket < 6:
        return "fit"
    if bucket < 8:
        return "calibration"
    return "evaluation"


def _result_paths(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw in inputs:
        path = raw.resolve()
        if path.is_dir():
            candidates = path.rglob("*.json")
        elif path.is_file():
            candidates = [path]
        else:
            raise FileNotFoundError(path)
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload.get("workflows"), list) and payload.get("policy"):
                paths.add(candidate.resolve())
    if not paths:
        raise RuntimeError("no complete WCT result JSON files were found")
    return sorted(paths)


def _load_rows(paths: Iterable[Path]) -> list[RuntimeRow]:
    rows: list[RuntimeRow] = []
    trace_cache: dict[Path, dict[str, list[dict[str, Any]]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        replay = dict(payload.get("replay") or {})
        common = {
            "source": str(path),
            "policy": str(payload.get("policy") or "unknown"),
            "evidence_tier": str(payload.get("evidence_tier") or "unknown"),
            "model": str(payload.get("model") or ""),
            "dp_size": int(payload.get("dp_size", 0)),
            "pp_size": int(payload.get("pp_size", 0)),
            "context_length": int(replay.get("context_length", 0)),
        }
        for workflow in payload.get("workflows", []):
            workflow_id = str(workflow.get("workflow_id") or "")
            progress_by_branch: dict[
                tuple[int, str], tuple[RuntimeProgressObservation, ...]
            ] = {}
            trace_raw = workflow.get("trace_path") or payload.get("trace")
            request_routes = {
                str(route.get("rid") or ""): float(
                    route.get("completion_elapsed_s", 0.0)
                )
                * 1000.0
                for route in workflow.get("request_routes", [])
                if route.get("rid")
            }
            if trace_raw and request_routes:
                trace_path = Path(str(trace_raw)).resolve()
                if trace_path not in trace_cache:
                    trace_cache[trace_path] = load_workflows(trace_path)
                requests_by_branch: dict[
                    tuple[int, str], list[dict[str, Any]]
                ] = defaultdict(list)
                for trace_row in trace_cache[trace_path].get(workflow_id, []):
                    requests_by_branch[
                        (
                            int(trace_row.get("stage_id", 0)),
                            str(trace_row.get("agent_id") or ""),
                        )
                    ].append(trace_row)
                for branch_key, branch_requests in requests_by_branch.items():
                    max_materialized_prompt = 0
                    observations: list[RuntimeProgressObservation] = []
                    for trace_row in sorted(
                        branch_requests,
                        key=lambda value: (
                            int(value.get("group_id", 0)),
                            str(value.get("rid") or ""),
                        ),
                    ):
                        request_id = str(trace_row.get("rid") or "")
                        if request_id not in request_routes:
                            observations = []
                            break
                        input_tokens, output_tokens, *_ = _scaled_tokens(
                            trace_row,
                            input_scale=float(replay.get("input_scale", 1.0)),
                            max_input_tokens=replay.get("max_input_tokens"),
                            max_output_tokens=replay.get("max_output_tokens"),
                            context_length=int(replay.get("context_length", 131072)),
                        )
                        new_prefill = max(
                            input_tokens - max_materialized_prompt, 0
                        )
                        max_materialized_prompt = max(
                            max_materialized_prompt, input_tokens
                        )
                        observations.append(
                            RuntimeProgressObservation(
                                request_id=request_id,
                                completion_elapsed_ms=request_routes[request_id],
                                new_prefill_tokens=new_prefill,
                                output_tokens=output_tokens,
                            )
                        )
                    if observations:
                        progress_by_branch[branch_key] = tuple(observations)
            for raw in workflow.get("algorithm1_slack_evaluation", []):
                required = {
                    "actual_duration_ms",
                    "actual_slack_ms",
                    "request_count",
                    "new_prefill_tokens",
                    "output_tokens",
                }
                missing = sorted(required - raw.keys())
                if missing:
                    raise RuntimeError(
                        f"{path} has legacy/incomplete Algorithm-1 rows; "
                        f"missing {missing}. Re-run with runtime schema v2."
                    )
                rows.append(
                    RuntimeRow(
                        **common,
                        workflow_id=workflow_id,
                        stage_id=int(raw["stage_id"]),
                        agent_id=str(raw["agent_id"]),
                        stage_width=int(raw["stage_width"]),
                        request_count=int(raw["request_count"]),
                        new_prefill_tokens=int(raw["new_prefill_tokens"]),
                        output_tokens=int(raw["output_tokens"]),
                        actual_duration_ms=float(raw["actual_duration_ms"]),
                        actual_slack_ms=float(raw["actual_slack_ms"]),
                        progress_observations=progress_by_branch.get(
                            (int(raw["stage_id"]), str(raw["agent_id"])), ()
                        ),
                    )
                )
    if not rows:
        raise RuntimeError("no Algorithm-1 runtime evaluation rows were found")
    return rows


def _fit_cost_model(rows: list[RuntimeRow]) -> dict[str, float]:
    matrix = np.asarray([row.features for row in rows], dtype=float)
    target = np.asarray([row.actual_duration_ms for row in rows], dtype=float)
    coefficients, _residual = nnls(matrix, target)
    predictions = matrix @ coefficients
    ratios = [
        actual / predicted
        for actual, predicted in zip(target, predictions)
        if predicted > 1e-9
    ]
    if not ratios:
        raise RuntimeError("cost model produced no positive duration predictions")
    lower_scale = min(_percentile(ratios, 0.05), 1.0)
    upper_scale = max(_percentile(ratios, 0.95), 1.0)
    return {
        "request_setup_ms": float(coefficients[0]),
        "prefill_ms_per_token": float(coefficients[1]),
        "decode_ms_per_token": float(coefficients[2]),
        "lower_scale": max(lower_scale, 0.0),
        "upper_scale": upper_scale,
    }


def _duration_ms(row: RuntimeRow, model: dict[str, float]) -> float:
    return sum(
        feature * model[name]
        for feature, name in zip(row.features, MODEL_KEYS)
    )


def _online_progress_critical_agents(
    stage_rows: list[RuntimeRow],
    *,
    model: dict[str, float],
) -> tuple[set[str], dict[str, float], dict[str, float]]:
    """Replay only completion-prefix observations available to the client.

    ``actual_duration_ms`` only recovers the stage-start timestamp that the
    live client knows directly. Decisions are replayed in completion order,
    before an upgraded branch dispatches its next request.
    """

    if len(stage_rows) <= 1 or any(
        not row.progress_observations for row in stage_rows
    ):
        return set(), {}, {}
    request_costs_by_agent: dict[str, list[tuple[str, float]]] = {}
    stage_start_candidates: list[float] = []
    events: list[tuple[float, str]] = []
    for row in stage_rows:
        request_costs: list[tuple[str, float]] = []
        for observation in row.progress_observations:
            cost_ms = sum(
                feature * model[name]
                for feature, name in zip(observation.features, MODEL_KEYS)
            )
            request_costs.append((observation.request_id, cost_ms))
            events.append(
                (observation.completion_elapsed_ms, observation.request_id)
            )
        request_costs_by_agent[row.agent_id] = request_costs
        stage_start_candidates.append(
            max(
                observation.completion_elapsed_ms
                for observation in row.progress_observations
            )
            - row.actual_duration_ms
        )
    stage_start_ms = sum(stage_start_candidates) / len(stage_start_candidates)
    tracker = AegisOnlineProgressTracker(
        request_costs_by_agent=request_costs_by_agent,
        minimum_progress_fraction=ONLINE_PROGRESS_MIN_FRACTION,
    )
    for completion_ms, request_id in sorted(events):
        tracker.observe_completion(
            request_id=request_id,
            elapsed_ms=max(completion_ms - stage_start_ms, 0.0),
        )
    return (
        set(tracker.final_critical_agents)
        if tracker.ever_dynamically_critical_agents
        else set(),
        tracker.trigger_elapsed_ms,
        tracker.final_projected_slack_ms,
    )


def _predict_rows(
    rows: list[RuntimeRow],
    *,
    model: dict[str, float],
    slack_margin_ms: float,
    critical_threshold_ms: float,
) -> list[dict[str, Any]]:
    by_stage: dict[tuple[str, str, int], list[RuntimeRow]] = defaultdict(list)
    for row in rows:
        by_stage[row.sample_key].append(row)
    predictions: list[dict[str, Any]] = []
    for stage_rows in by_stage.values():
        durations = {row.agent_id: _duration_ms(row, model) for row in stage_rows}
        online_critical, online_trigger_elapsed_ms, online_slack_ms = (
            _online_progress_critical_agents(stage_rows, model=model)
        )
        for row in stage_rows:
            others = [value for agent, value in durations.items() if agent != row.agent_id]
            duration_q50 = durations[row.agent_id]
            duration_lower = duration_q50 * model["lower_scale"]
            duration_upper = duration_q50 * model["upper_scale"]
            if not others:
                raw_lcb = q50 = upper = 0.0
            else:
                raw_lcb = max(
                    max(value * model["lower_scale"] for value in others)
                    - duration_upper,
                    0.0,
                )
                q50 = max(max(others) - duration_q50, 0.0)
                upper = max(
                    max(value * model["upper_scale"] for value in others)
                    - duration_lower,
                    q50,
                )
            lcb = max(raw_lcb - slack_margin_ms, 0.0)
            online_q50 = online_slack_ms.get(row.agent_id)
            if online_q50 is not None:
                q50 = max(float(online_q50), 0.0)
                upper = max(upper, q50)
            static_predicted_critical = q50 <= critical_threshold_ms
            online_progress_critical = row.agent_id in online_critical
            predictions.append(
                {
                    **{
                        key: value
                        for key, value in row.__dict__.items()
                        if key != "progress_observations"
                    },
                    "split": _split(row.workflow_id),
                    "predicted_duration_lcb_ms": duration_lower,
                    "predicted_duration_q50_ms": duration_q50,
                    "predicted_duration_q90_ms": duration_upper,
                    "raw_slack_lcb_ms": raw_lcb,
                    "predicted_slack_lcb_ms": lcb,
                    "predicted_slack_q50_ms": max(q50, lcb),
                    "predicted_slack_q90_ms": max(upper, q50, lcb),
                    "actual_critical": row.actual_slack_ms <= critical_threshold_ms,
                    "static_predicted_critical": static_predicted_critical,
                    "online_progress_critical": online_progress_critical,
                    "online_progress_trigger_elapsed_ms": (
                        online_trigger_elapsed_ms.get(row.agent_id, 0.0)
                    ),
                    "predicted_critical": bool(
                        online_progress_critical
                        if online_critical
                        else static_predicted_critical
                    ),
                }
            )
    return predictions


def _metrics(rows: list[dict[str, Any]], *, target_coverage: float) -> dict[str, Any]:
    fanout = [row for row in rows if int(row["stage_width"]) > 1]
    scored = fanout or rows
    lower_coverage = sum(
        float(row["actual_slack_ms"]) + 5.0
        >= float(row["predicted_slack_lcb_ms"])
        for row in scored
    ) / max(len(scored), 1)
    interval_coverage = sum(
        float(row["actual_slack_ms"]) + 5.0
        >= float(row["predicted_slack_lcb_ms"])
        and float(row["actual_slack_ms"])
        <= float(row["predicted_slack_q90_ms"]) + 5.0
        for row in scored
    ) / max(len(scored), 1)
    actual_critical = [row for row in scored if row["actual_critical"]]
    predicted_critical = [row for row in scored if row["predicted_critical"]]
    true_critical = [
        row for row in scored if row["actual_critical"] and row["predicted_critical"]
    ]
    errors = [
        abs(float(row["actual_slack_ms"]) - float(row["predicted_slack_q50_ms"]))
        for row in scored
    ]
    stage_spans = [
        float(row["actual_duration_ms"]) + float(row["actual_slack_ms"])
        for row in scored
    ]
    error_budget_ms = max(250.0, 0.20 * _percentile(stage_spans, 0.90))
    p90_error = _percentile(errors, 0.90)
    critical_recall = len(true_critical) / max(len(actual_critical), 1)
    critical_precision = len(true_critical) / max(len(predicted_critical), 1)
    return {
        "examples": len(rows),
        "fanout_examples": len(fanout),
        "workflows": len({row["workflow_id"] for row in rows}),
        "observed_lower_coverage": lower_coverage,
        "observed_interval_coverage": interval_coverage,
        "critical_recall": critical_recall,
        "critical_precision": critical_precision,
        "median_absolute_error_ms": _percentile(errors, 0.50),
        "p90_absolute_error_ms": p90_error,
        "p90_error_budget_ms": error_budget_ms,
        "nonzero_lcb_fraction": sum(
            float(row["predicted_slack_lcb_ms"]) > 0.0 for row in scored
        )
        / max(len(scored), 1),
        "accuracy_passed": (
            lower_coverage >= target_coverage
            and critical_recall >= 0.90
            and critical_precision >= 0.75
            and p90_error <= error_budget_ms
        ),
    }


def build_artifact(
    rows: list[RuntimeRow],
    *,
    target_coverage: float,
    critical_threshold_ms: float,
    minimum_split_workflows: tuple[int, int, int],
    expected_model: str,
    expected_dp_size: int,
    expected_pp_size: int,
    expected_context_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_rows = {
        split: [row for row in rows if _split(row.workflow_id) == split]
        for split in ("fit", "calibration", "evaluation")
    }
    split_workflows = {
        split: len({row.workflow_id for row in values})
        for split, values in split_rows.items()
    }
    required = dict(zip(("fit", "calibration", "evaluation"), minimum_split_workflows))
    if any(split_workflows[key] < required[key] for key in required):
        raise RuntimeError(
            f"workflow-disjoint split is too small: observed={split_workflows}, "
            f"required={required}"
        )

    model = _fit_cost_model(split_rows["fit"])
    raw_calibration = _predict_rows(
        split_rows["calibration"],
        model=model,
        slack_margin_ms=0.0,
        critical_threshold_ms=critical_threshold_ms,
    )
    fanout_calibration = [row for row in raw_calibration if row["stage_width"] > 1]
    if not fanout_calibration:
        raise RuntimeError("calibration split has no fan-out stage rows")
    slack_margin_ms = 1000.0 * calibrated_lower_margin_s(
        actual_s=(row["actual_slack_ms"] / 1000.0 for row in fanout_calibration),
        raw_lower_s=(row["raw_slack_lcb_ms"] / 1000.0 for row in fanout_calibration),
        target_coverage=target_coverage,
    )
    audit = _predict_rows(
        rows,
        model=model,
        slack_margin_ms=slack_margin_ms,
        critical_threshold_ms=critical_threshold_ms,
    )
    evaluation = [row for row in audit if row["split"] == "evaluation"]
    metrics = _metrics(evaluation, target_coverage=target_coverage)

    formal_rows = [
        row
        for row in rows
        if row.evidence_tier == "sglsim-online-measured"
        and row.model == expected_model
        and row.dp_size == expected_dp_size
        and row.pp_size == expected_pp_size
        and row.context_length == expected_context_length
    ]
    formal_workflows = len({row.workflow_id for row in formal_rows})
    formal_scope_passed = len(formal_rows) == len(rows)
    gate_passed = bool(metrics["accuracy_passed"] and formal_scope_passed)
    source_digest = hashlib.sha256(
        "\n".join(sorted({row.source for row in rows})).encode()
    ).hexdigest()
    artifact = {
        "schema_version": 2,
        "semantics": SEMANTICS,
        "calibration_id": f"alg1-fixed-dag-runtime-{source_digest[:12]}",
        "algorithm": "AegisKV Algorithm 1 fixed-DAG CPM runtime envelope",
        "source_results": sorted({row.source for row in rows}),
        "source_results_sha256": source_digest,
        "split_salt": SPLIT_SALT,
        "target_lower_coverage": target_coverage,
        "global_lower_margin_s": 0.0,
        "role_lower_margins_s": {},
        "fallback_lower_bounds_s": {},
        "slack_lower_margin_ms": slack_margin_ms,
        "request_cost_model": model,
        "online_progress_correction": {
            "enabled": True,
            "minimum_progress_fraction": ONLINE_PROGRESS_MIN_FRACTION,
            "requires_future_request": True,
            "replaces_static_ranking_after_trigger": True,
            "replay_inputs": (
                "measured request completion prefixes plus modeled remaining work"
            ),
        },
        "partitions": {
            split: {
                "workflows": split_workflows[split],
                "examples": len(split_rows[split]),
            }
            for split in split_rows
        },
        "evaluation": metrics,
        "evaluation_gate": {
            "passed": gate_passed,
            "formal_runtime_gate_passed": gate_passed,
            "accuracy_passed": bool(metrics["accuracy_passed"]),
            "formal_scope_passed": formal_scope_passed,
            "formal_workflows": formal_workflows,
            "required_scope": {
                "evidence_tier": "sglsim-online-measured",
                "model": expected_model,
                "dp_size": expected_dp_size,
                "pp_size": expected_pp_size,
                "context_length": expected_context_length,
            },
            "minimum_runtime_lower_coverage": target_coverage,
            "minimum_runtime_critical_recall": 0.90,
            "minimum_runtime_critical_precision": 0.75,
            "p90_absolute_error_budget_ms": metrics["p90_error_budget_ms"],
        },
        "safety_contract": (
            "Durations are fitted only on workflow-disjoint runtime rows and "
            "transformed through the fixed stage DAG. Singleton stages have zero "
            "Slack. A causal last-quarter progress correction may only upgrade a "
            "branch while another request remains to protect. Algorithm 2 consumes "
            "only the conformal lower bound; payload and manifest ACK state remain "
            "the recoverability authority."
        ),
    }
    return artifact, audit


def _write_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--critical-threshold-ms", type=float, default=5.0)
    parser.add_argument("--minimum-fit-workflows", type=int, default=6)
    parser.add_argument("--minimum-calibration-workflows", type=int, default=3)
    parser.add_argument("--minimum-evaluation-workflows", type=int, default=3)
    parser.add_argument("--expected-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--expected-dp-size", type=int, default=4)
    parser.add_argument("--expected-pp-size", type=int, default=4)
    parser.add_argument("--expected-context-length", type=int, default=32768)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Replace the live artifact only when the formal runtime gate passes.",
    )
    args = parser.parse_args()
    paths = _result_paths(args.results)
    rows = _load_rows(paths)
    artifact, audit = build_artifact(
        rows,
        target_coverage=args.target_coverage,
        critical_threshold_ms=args.critical_threshold_ms,
        minimum_split_workflows=(
            args.minimum_fit_workflows,
            args.minimum_calibration_workflows,
            args.minimum_evaluation_workflows,
        ),
        expected_model=args.expected_model,
        expected_dp_size=args.expected_dp_size,
        expected_pp_size=args.expected_pp_size,
        expected_context_length=args.expected_context_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    _write_audit(args.audit_csv, audit)
    if args.promote:
        if not artifact["evaluation_gate"]["passed"]:
            raise SystemExit("refusing to promote: formal Algorithm-1 gate failed")
        DEFAULT_PROMOTED_OUTPUT.write_text(
            json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(artifact, indent=2))
    return 0 if artifact["evaluation_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
