#!/usr/bin/env python3
"""Render a paper-facing case study for Algorithm 1's causal GBDT predictor.

This script implements the ``Predict`` stage of the current Algorithm 1 with
two causal GBDT heads:

    (G_t, H_t) <- Observe(...)
    prior_a(t) <- max(GBDT_total(phi_static) - elapsed_a(t), 0)
    dynamic_a(t) <- GBDT_remaining(phi_a(G_t, H_t))
    lambda_t <- 1 - exp(-rounds_completed / tau)
    remaining_a(t) <- (1-lambda_t) * prior_a(t) + lambda_t * dynamic_a(t)

For the selected multi-round agent, one update is recorded whenever a request
returns, matching ``current_design.md``.  At that instant the completed request
and prefix are observable; future API latency, output length, arrival time, and
completion time are not features.  The complete target workflow is held out
from cohort selection, cross-validation, and model fitting.  A final structural
zero is emitted when the completion event removes the agent from the residual
DAG; the GBDT is not invoked for that terminal node.

The offline trace oracle is used only to construct training labels and score the
held-out predictions.  It is never exposed to the predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

try:
    import sklearn
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.model_selection import GroupKFold
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "scikit-learn is required to render the Algorithm 1 GBDT figure"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE = (
    ROOT
    / "Archived"
    / "2026-08-06-pre-kvcache-centric"
    / "presentations"
    / "presentation"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "weka_subagent_cdf_24w_trace_oracle_workload.jsonl"
)
DEFAULT_OUTPUT_PREFIX = ROOT / "figures" / "alg1-agent-time-accuracy"
DEFAULT_SNAPSHOTS = (
    ROOT / "figures" / "data" / "alg1-agent-time-prediction-snapshots.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "figures" / "data" / "alg1-agent-time-accuracy-summary.json"
)

DEFAULT_TRACE_PREFIX = "0bc9c4f7"
DEFAULT_STAGE_ID = 2
DEFAULT_AGENT_SUFFIX = "sub2"
RANDOM_STATE = 20260814

SNAPSHOT_FIELDS = (
    "agent_trace_id",
    "prediction_epoch",
    "rounds_completed",
    "current_round",
    "round_count_evaluation_only",
    "progress_fraction_evaluation_only",
    "current_request_id",
    "elapsed_time_s",
    "total_prior_time_s",
    "prior_remaining_time_s",
    "dynamic_remaining_time_s",
    "dynamic_weight",
    "predicted_remaining_time_s",
    "predicted_total_time_s",
    "actual_remaining_time_s",
    "actual_total_time_s",
    "absolute_error_s",
    "relative_error_pct",
    "prediction_source",
    "cohort_source",
    "information_scope",
)


@dataclass(frozen=True)
class GBDTConfig:
    loss: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int
    subsample: float


@dataclass
class SnapshotSample:
    key: tuple[str, int, str]
    workflow_id: str
    rounds_completed: int
    round_count: int
    current_request_id: str
    elapsed_time_s: float
    actual_remaining_time_s: float
    actual_total_time_s: float
    features: dict[str, float]

    @property
    def agent_weight(self) -> float:
        return 1.0 / self.round_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_id(key: tuple[str, int, str]) -> str:
    material = f"{key[0]}:{key[1]}:{key[2]}".encode("utf-8")
    return f"agent-{hashlib.sha256(material).hexdigest()[:8]}"


def _load_trace(path: Path) -> list[dict[str, Any]]:
    required = {
        "workflow_id",
        "stage_id",
        "stage_width",
        "stage_local_agent_index",
        "agent_id",
        "rid",
        "model",
        "source_model_role",
        "prompt_len",
        "output_len",
        "arrival_t",
        "trace_api_time_s",
        "trace_completion_t",
        "trace_oracle_branch_started_at_s",
        "trace_oracle_branch_completed_at_s",
        "trace_oracle_branch_duration_ms",
    }
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - row.keys()
            if missing:
                raise ValueError(
                    f"trace line {line_number} misses required fields: {sorted(missing)}"
                )
            if float(row["trace_api_time_s"]) <= 0.0:
                raise ValueError(
                    f"request {row.get('rid')} has non-positive trace_api_time_s"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"empty trace: {path}")
    return rows


def _agent_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["workflow_id"]),
        int(row["stage_id"]),
        str(row["agent_id"]),
    )


def _ordered(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("group_id", 0)),
            float(row["arrival_t"]),
            str(row["rid"]),
        ),
    )


def _group_agents(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_agent_key(row)].append(row)
    return {key: _ordered(value) for key, value in grouped.items()}


def _select_agent(
    agents: dict[tuple[str, int, str], list[dict[str, Any]]],
    *,
    trace_prefix: str,
    stage_id: int,
    agent_suffix: str,
) -> tuple[tuple[str, int, str], list[dict[str, Any]]]:
    matches = [
        (key, rows)
        for key, rows in agents.items()
        if key[0].startswith(trace_prefix)
        and key[1] == stage_id
        and key[2].endswith(agent_suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one target agent for "
            f"prefix={trace_prefix!r}, stage={stage_id}, suffix={agent_suffix!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _ewma(values: list[float], alpha: float = 0.5) -> float:
    if not values:
        return 0.0
    result = float(values[0])
    for value in values[1:]:
        result = alpha * float(value) + (1.0 - alpha) * result
    return result


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value))


def _progress_bucket(rounds_completed: int) -> str:
    if rounds_completed == 0:
        return "0"
    if rounds_completed == 1:
        return "1"
    if rounds_completed <= 3:
        return "2-3"
    if rounds_completed <= 7:
        return "4-7"
    if rounds_completed <= 15:
        return "8-15"
    return "16+"


def _causal_features(
    agent_rows: list[dict[str, Any]], rounds_completed: int
) -> tuple[dict[str, float], float]:
    """Return phi_a(G_t, H_t) immediately after a request return.

    Exactly ``rounds_completed`` rows are complete and causally observable.
    No field from a later row is read.  This matches Algorithm 1's event trigger:
    every returned LLM request or tool call refreshes the execution summary.
    """

    if not 1 <= rounds_completed <= len(agent_rows):
        raise ValueError("request-return epoch needs a non-empty completed prefix")
    first = agent_rows[0]
    prefix = agent_rows[:rounds_completed]
    current = prefix[-1]
    branch_start_s = float(first["trace_oracle_branch_started_at_s"])
    elapsed_time_s = max(
        float(current["trace_completion_t"]) - branch_start_s,
        0.0,
    )

    api_times = [float(row["trace_api_time_s"]) for row in prefix]
    ttfts = [float(row.get("trace_ttft_s", 0.0)) for row in prefix]
    prompts = [int(row["prompt_len"]) for row in prefix]
    # The just-returned request belongs to observed history.  Later requests
    # remain inaccessible to both feature construction and prediction.
    observed_outputs = [int(row["output_len"]) for row in prefix]
    observed_gaps = [
        max(
            float(agent_rows[index]["arrival_t"])
            - float(agent_rows[index - 1]["trace_completion_t"]),
            0.0,
        )
        for index in range(1, rounds_completed)
    ]
    api_total_s = float(sum(api_times))
    gap_total_s = max(elapsed_time_s - api_total_s, 0.0)
    initial_prompt = int(first["prompt_len"])
    current_prompt = int(current["prompt_len"])
    prompt_growth = prompts[-1] - prompts[0] if len(prompts) > 1 else 0

    model = str(current["model"])
    role = str(current["source_model_role"])
    stage_id = int(current["stage_id"])
    stage_width = int(current["stage_width"])
    agent_index = int(current["stage_local_agent_index"])
    bucket = _progress_bucket(rounds_completed)
    profile = f"{model}|{role}|{stage_id}|{stage_width}|{agent_index}"

    features = {
        "elapsed_log_s": math.log1p(elapsed_time_s),
        "rounds_completed": float(rounds_completed),
        "has_progress": 1.0,
        "stage_id": float(stage_id),
        "stage_width": float(stage_width),
        "agent_index": float(agent_index),
        "initial_prompt_log_tokens": math.log1p(max(initial_prompt, 0)),
        "current_prompt_log_tokens": math.log1p(max(current_prompt, 0)),
        "current_prompt_growth_signed_log": _signed_log1p(
            current_prompt - initial_prompt
        ),
        "observed_api_total_log_s": math.log1p(api_total_s),
        "observed_gap_total_log_s": math.log1p(gap_total_s),
        "observed_prompt_total_log_tokens": math.log1p(sum(prompts)),
        "observed_output_total_log_tokens": math.log1p(sum(observed_outputs)),
        "last_api_log_s": math.log1p(api_times[-1] if api_times else 0.0),
        "last_gap_log_s": math.log1p(
            observed_gaps[-1] if observed_gaps else 0.0
        ),
        "last_ttft_log_s": math.log1p(ttfts[-1] if ttfts else 0.0),
        "last_prompt_log_tokens": math.log1p(
            prompts[-1] if prompts else initial_prompt
        ),
        "last_observed_output_log_tokens": math.log1p(
            observed_outputs[-1] if observed_outputs else 0
        ),
        "mean_api_log_s": math.log1p(_mean(api_times)),
        "mean_gap_log_s": math.log1p(_mean(observed_gaps)),
        "mean_wall_round_log_s": math.log1p(
            elapsed_time_s / rounds_completed if rounds_completed else 0.0
        ),
        "std_api_log_s": math.log1p(_std(api_times)),
        "std_gap_log_s": math.log1p(_std(observed_gaps)),
        "ewma_api_log_s": math.log1p(_ewma(api_times)),
        "ewma_gap_log_s": math.log1p(_ewma(observed_gaps)),
        "observed_prompt_growth_signed_log": _signed_log1p(prompt_growth),
        "observed_output_per_round_log_tokens": math.log1p(
            sum(observed_outputs) / rounds_completed if rounds_completed else 0.0
        ),
        f"model={model}": 1.0,
        f"role={role}": 1.0,
        f"stage_role={stage_id}|{role}": 1.0,
        f"profile={profile}": 1.0,
        f"progress_bucket={bucket}": 1.0,
        f"profile_progress={profile}|{bucket}": 1.0,
    }
    return features, elapsed_time_s


def _build_samples(
    agents: dict[tuple[str, int, str], list[dict[str, Any]]]
) -> list[SnapshotSample]:
    samples: list[SnapshotSample] = []
    for key, agent_rows in agents.items():
        first = agent_rows[0]
        branch_start_s = float(first["trace_oracle_branch_started_at_s"])
        branch_end_s = float(first["trace_oracle_branch_completed_at_s"])
        actual_total_s = float(first["trace_oracle_branch_duration_ms"]) / 1000.0
        if not math.isclose(
            branch_end_s - branch_start_s,
            actual_total_s,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"branch duration disagrees with start/end: {key}")
        for completed in range(1, len(agent_rows) + 1):
            features, elapsed_time_s = _causal_features(agent_rows, completed)
            actual_remaining_s = max(actual_total_s - elapsed_time_s, 0.0)
            samples.append(
                SnapshotSample(
                    key=key,
                    workflow_id=key[0],
                    rounds_completed=completed,
                    round_count=len(agent_rows),
                    current_request_id=str(agent_rows[completed - 1]["rid"]),
                    elapsed_time_s=elapsed_time_s,
                    actual_remaining_time_s=actual_remaining_s,
                    actual_total_time_s=actual_total_s,
                    features=features,
                )
            )
    return samples


def _agent_metadata(
    agents: dict[tuple[str, int, str], list[dict[str, Any]]],
    key: tuple[str, int, str],
) -> tuple[str, str, int, int, int]:
    first = agents[key][0]
    return (
        str(first["model"]),
        str(first["source_model_role"]),
        int(first["stage_id"]),
        int(first["stage_width"]),
        int(first["stage_local_agent_index"]),
    )


def _select_training_cohort(
    *,
    agents: dict[tuple[str, int, str], list[dict[str, Any]]],
    samples: list[SnapshotSample],
    target_key: tuple[str, int, str],
) -> tuple[str, list[SnapshotSample], list[dict[str, Any]]]:
    """Select the narrowest workflow-disjoint historical peer cohort."""

    target = _agent_metadata(agents, target_key)
    target_workflow = target_key[0]
    eligible_keys = {
        sample.key
        for sample in samples
        if sample.workflow_id != target_workflow
    }

    Predicate = Callable[[tuple], bool]
    specifications: list[tuple[str, int, Predicate]] = [
        (
            "model_role_stage_agent_index",
            4,
            lambda value: (
                value[0] == target[0]
                and value[1] == target[1]
                and value[2] == target[2]
                and value[4] == target[4]
            ),
        ),
        (
            "model_role_stage",
            8,
            lambda value: (
                value[0] == target[0]
                and value[1] == target[1]
                and value[2] == target[2]
            ),
        ),
        (
            "model_role",
            12,
            lambda value: value[0] == target[0] and value[1] == target[1],
        ),
        (
            "role",
            16,
            lambda value: value[1] == target[1],
        ),
        ("global", 4, lambda value: True),
    ]

    audits: list[dict[str, Any]] = []
    for name, minimum_agents, predicate in specifications:
        cohort_keys = {
            key
            for key in eligible_keys
            if predicate(_agent_metadata(agents, key))
        }
        audits.append(
            {
                "name": name,
                "minimum_agents": minimum_agents,
                "available_agents": len(cohort_keys),
                "selected": len(cohort_keys) >= minimum_agents,
            }
        )
        if len(cohort_keys) >= minimum_agents:
            cohort_samples = [sample for sample in samples if sample.key in cohort_keys]
            if len({sample.workflow_id for sample in cohort_samples}) < 3:
                raise ValueError("GBDT cohort needs at least three workflows")
            return name, cohort_samples, audits
    raise AssertionError("global cohort selection must succeed")


def _config_grid() -> list[GBDTConfig]:
    return [
        GBDTConfig(
            loss=loss,
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=0.9,
        )
        for loss in ("squared_error", "huber")
        for n_estimators in (120, 160)
        for max_depth in (2, 3)
        for min_samples_leaf in (1, 2, 4)
    ]


def _make_model(config: GBDTConfig) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        **asdict(config),
        random_state=RANDOM_STATE,
    )


def _static_features(sample: SnapshotSample) -> dict[str, float]:
    """Return the per-agent features available to the initial prior head."""

    fixed_names = {
        "stage_id",
        "stage_width",
        "agent_index",
        "initial_prompt_log_tokens",
    }
    return {
        name: value
        for name, value in sample.features.items()
        if name in fixed_names
        or name.startswith("model=")
        or name.startswith("role=")
        or (name.startswith("profile=") and "progress" not in name)
    }


def _first_sample_per_agent(
    samples: list[SnapshotSample],
) -> list[SnapshotSample]:
    return [
        min(
            (sample for sample in samples if sample.key == key),
            key=lambda sample: sample.rounds_completed,
        )
        for key in sorted({sample.key for sample in samples})
    ]


def _fit_total_prior(
    samples: list[SnapshotSample], config: GBDTConfig
) -> tuple[DictVectorizer, GradientBoostingRegressor]:
    first_samples = _first_sample_per_agent(samples)
    vectorizer = DictVectorizer(sparse=False)
    design = vectorizer.fit_transform(
        [_static_features(sample) for sample in first_samples]
    )
    target = np.log1p(
        np.asarray([sample.actual_total_time_s for sample in first_samples])
    )
    model = _make_model(config)
    model.fit(design, target)
    return vectorizer, model


def _predict_total_prior(
    vectorizer: DictVectorizer,
    model: GradientBoostingRegressor,
    samples: list[SnapshotSample],
) -> np.ndarray:
    design = vectorizer.transform(
        [_static_features(sample) for sample in samples]
    )
    return np.maximum(np.expm1(model.predict(design)), 0.0)


def _fit_dynamic_remaining(
    samples: list[SnapshotSample], config: GBDTConfig
) -> tuple[DictVectorizer, GradientBoostingRegressor]:
    nonterminal = [
        sample
        for sample in samples
        if sample.rounds_completed < sample.round_count
    ]
    vectorizer = DictVectorizer(sparse=False)
    design = vectorizer.fit_transform(
        [sample.features for sample in nonterminal]
    )
    target = np.log1p(
        np.asarray([sample.actual_remaining_time_s for sample in nonterminal])
    )
    weights = np.asarray(
        [1.0 / max(sample.round_count - 1, 1) for sample in nonterminal]
    )
    model = _make_model(config)
    model.fit(design, target, sample_weight=weights)
    return vectorizer, model


def _predict_dynamic_remaining(
    vectorizer: DictVectorizer,
    model: GradientBoostingRegressor,
    samples: list[SnapshotSample],
) -> np.ndarray:
    design = vectorizer.transform([sample.features for sample in samples])
    return np.maximum(np.expm1(model.predict(design)), 0.0)


def _workflow_folds(
    samples: list[SnapshotSample],
) -> Iterable[tuple[list[SnapshotSample], list[SnapshotSample]]]:
    workflows = np.asarray([sample.workflow_id for sample in samples])
    splitter = GroupKFold(n_splits=len(set(workflows)))
    dummy = np.zeros((len(samples), 1), dtype=float)
    for fit_indices, validation_indices in splitter.split(dummy, groups=workflows):
        fit_samples = [samples[index] for index in fit_indices]
        validation_samples = [
            samples[index]
            for index in validation_indices
            if samples[index].rounds_completed < samples[index].round_count
        ]
        yield fit_samples, validation_samples


def _select_head_config(
    samples: list[SnapshotSample], *, head: str
) -> tuple[GBDTConfig, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []

    for config in _config_grid():
        fold_errors: list[float] = []
        for fit_samples, validation_samples in _workflow_folds(samples):
            if head == "total_prior":
                vectorizer, model = _fit_total_prior(fit_samples, config)
                predicted_total = np.maximum(
                    _predict_total_prior(
                        vectorizer, model, validation_samples
                    ),
                    np.asarray(
                        [sample.elapsed_time_s for sample in validation_samples]
                    ),
                )
            elif head == "dynamic_remaining":
                vectorizer, model = _fit_dynamic_remaining(fit_samples, config)
                predicted_total = _predict_dynamic_remaining(
                    vectorizer, model, validation_samples
                ) + np.asarray(
                    [sample.elapsed_time_s for sample in validation_samples]
                )
            else:
                raise ValueError(f"unknown GBDT head: {head}")
            actual_total = np.asarray(
                [sample.actual_total_time_s for sample in validation_samples]
            )
            errors = (
                100.0
                * np.abs(predicted_total - actual_total)
                / actual_total
            )
            fold_errors.append(float(np.mean(errors)))
        results.append(
            {
                "config": asdict(config),
                "selection_score": float(np.mean(fold_errors)),
                "fold_agent_mean_relative_error_pct": fold_errors,
            }
        )

    results.sort(
        key=lambda row: (
            float(row["selection_score"]),
            json.dumps(row["config"], sort_keys=True),
        )
    )
    return GBDTConfig(**results[0]["config"]), results


BLEND_TAU_GRID: tuple[float | None, ...] = (
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    192.0,
    256.0,
    384.0,
    512.0,
    1024.0,
    None,
)
BLEND_SELECTION_TOLERANCE_PCT = 0.05


def _dynamic_weight(rounds_completed: int, tau_rounds: float | None) -> float:
    if tau_rounds is None:
        return 0.0
    return 1.0 - math.exp(-max(rounds_completed, 0) / tau_rounds)


def _blend_components(
    *,
    samples: list[SnapshotSample],
    total_prior_s: np.ndarray,
    dynamic_remaining_s: np.ndarray,
    tau_rounds: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elapsed = np.asarray([sample.elapsed_time_s for sample in samples])
    prior_remaining = np.maximum(total_prior_s - elapsed, 0.0)
    weights = np.asarray(
        [_dynamic_weight(sample.rounds_completed, tau_rounds) for sample in samples]
    )
    blended_remaining = (
        (1.0 - weights) * prior_remaining
        + weights * dynamic_remaining_s
    )
    return np.maximum(blended_remaining, 0.0), prior_remaining, weights


def _select_blend_tau(
    samples: list[SnapshotSample],
    *,
    total_config: GBDTConfig,
    dynamic_config: GBDTConfig,
) -> tuple[float | None, list[dict[str, Any]]]:
    fold_components: list[
        tuple[list[SnapshotSample], np.ndarray, np.ndarray]
    ] = []
    for fit_samples, validation_samples in _workflow_folds(samples):
        total_vectorizer, total_model = _fit_total_prior(
            fit_samples, total_config
        )
        dynamic_vectorizer, dynamic_model = _fit_dynamic_remaining(
            fit_samples, dynamic_config
        )
        fold_components.append(
            (
                validation_samples,
                _predict_total_prior(
                    total_vectorizer, total_model, validation_samples
                ),
                _predict_dynamic_remaining(
                    dynamic_vectorizer, dynamic_model, validation_samples
                ),
            )
        )

    results: list[dict[str, Any]] = []
    for tau_rounds in BLEND_TAU_GRID:
        fold_errors: list[float] = []
        for validation_samples, total_prior, dynamic_remaining in fold_components:
            blended_remaining, _, _ = _blend_components(
                samples=validation_samples,
                total_prior_s=total_prior,
                dynamic_remaining_s=dynamic_remaining,
                tau_rounds=tau_rounds,
            )
            predicted_total = blended_remaining + np.asarray(
                [sample.elapsed_time_s for sample in validation_samples]
            )
            actual_total = np.asarray(
                [sample.actual_total_time_s for sample in validation_samples]
            )
            fold_errors.append(
                float(
                    np.mean(
                        100.0
                        * np.abs(predicted_total - actual_total)
                        / actual_total
                    )
                )
            )
        results.append(
            {
                "tau_rounds": tau_rounds if tau_rounds is not None else "static",
                "mean_agent_relative_error_pct": float(np.mean(fold_errors)),
                "fold_agent_mean_relative_error_pct": fold_errors,
            }
        )

    best_error = min(
        float(row["mean_agent_relative_error_pct"]) for row in results
    )
    eligible = [
        row
        for row in results
        if float(row["mean_agent_relative_error_pct"])
        <= best_error + BLEND_SELECTION_TOLERANCE_PCT
        and row["tau_rounds"] != "static"
    ]
    if eligible:
        # Prefer the strongest finite shrinkage among blends that are
        # indistinguishable at the selected LOOWF resolution.  This prevents
        # a tiny cohort from overreacting to coarse dynamic-head leaf values
        # while still producing a genuine update at every return epoch.
        selected = max(eligible, key=lambda row: float(row["tau_rounds"]))
        selected_tau: float | None = float(selected["tau_rounds"])
    else:
        selected_tau = None
    for row in results:
        row["selected"] = (
            row["tau_rounds"] == selected_tau
            if selected_tau is not None
            else row["tau_rounds"] == "static"
        )
    return selected_tau, results


def _workflow_disjoint_accuracy(
    samples: list[SnapshotSample],
    *,
    total_config: GBDTConfig,
    dynamic_config: GBDTConfig,
    tau_rounds: float | None,
) -> dict[str, Any]:
    """Audit the selected two-head model on workflow-disjoint peers."""

    per_agent: list[dict[str, Any]] = []
    for fit_samples, validation_samples in _workflow_folds(samples):
        total_vectorizer, total_model = _fit_total_prior(
            fit_samples, total_config
        )
        dynamic_vectorizer, dynamic_model = _fit_dynamic_remaining(
            fit_samples, dynamic_config
        )
        total_prior = _predict_total_prior(
            total_vectorizer, total_model, validation_samples
        )
        dynamic_remaining = _predict_dynamic_remaining(
            dynamic_vectorizer, dynamic_model, validation_samples
        )
        blended_remaining, _, _ = _blend_components(
            samples=validation_samples,
            total_prior_s=total_prior,
            dynamic_remaining_s=dynamic_remaining,
            tau_rounds=tau_rounds,
        )
        predicted_total = blended_remaining + np.asarray(
            [sample.elapsed_time_s for sample in validation_samples]
        )
        actual_total = np.asarray(
            [sample.actual_total_time_s for sample in validation_samples]
        )
        errors = 100.0 * np.abs(predicted_total - actual_total) / actual_total
        for key in sorted({sample.key for sample in validation_samples}):
            mask = np.asarray(
                [sample.key == key for sample in validation_samples]
            )
            per_agent.append(
                {
                    "agent_trace_id": _trace_id(key),
                    "workflow_id_sha256_prefix": hashlib.sha256(
                        key[0].encode("utf-8")
                    ).hexdigest()[:8],
                    "prediction_points": int(np.sum(mask)),
                    "initial_relative_error_pct": float(errors[mask][0]),
                    "mean_relative_error_pct": float(np.mean(errors[mask])),
                    "max_relative_error_pct": float(np.max(errors[mask])),
                }
            )

    means = np.asarray(
        [row["mean_relative_error_pct"] for row in per_agent]
    )
    return {
        "protocol": "leave_one_workflow_out_on_matched_historical_peers",
        "agents": len(per_agent),
        "median_agent_mean_relative_error_pct": float(np.median(means)),
        "mean_agent_mean_relative_error_pct": float(np.mean(means)),
        "p90_agent_mean_relative_error_pct": float(np.percentile(means, 90)),
        "per_agent": per_agent,
    }


def build_snapshots(
    *,
    trace_path: Path,
    trace_prefix: str,
    stage_id: int,
    agent_suffix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_rows = _load_trace(trace_path)
    agents = _group_agents(trace_rows)
    target_key, target_rows = _select_agent(
        agents,
        trace_prefix=trace_prefix,
        stage_id=stage_id,
        agent_suffix=agent_suffix,
    )
    all_samples = _build_samples(agents)
    target_samples = sorted(
        [sample for sample in all_samples if sample.key == target_key],
        key=lambda sample: sample.rounds_completed,
    )
    cohort_source, cohort_samples, cohort_audit = _select_training_cohort(
        agents=agents,
        samples=all_samples,
        target_key=target_key,
    )
    total_config, total_cv_results = _select_head_config(
        cohort_samples, head="total_prior"
    )
    dynamic_config, dynamic_cv_results = _select_head_config(
        cohort_samples, head="dynamic_remaining"
    )
    blend_tau_rounds, blend_cv_results = _select_blend_tau(
        cohort_samples,
        total_config=total_config,
        dynamic_config=dynamic_config,
    )
    cohort_validation = _workflow_disjoint_accuracy(
        cohort_samples,
        total_config=total_config,
        dynamic_config=dynamic_config,
        tau_rounds=blend_tau_rounds,
    )
    total_vectorizer, total_model = _fit_total_prior(
        cohort_samples, total_config
    )
    dynamic_vectorizer, dynamic_model = _fit_dynamic_remaining(
        cohort_samples, dynamic_config
    )
    total_prior_s = _predict_total_prior(
        total_vectorizer, total_model, target_samples
    )
    dynamic_remaining_s = _predict_dynamic_remaining(
        dynamic_vectorizer, dynamic_model, target_samples
    )
    predicted_remaining_s, prior_remaining_s, dynamic_weights = (
        _blend_components(
            samples=target_samples,
            total_prior_s=total_prior_s,
            dynamic_remaining_s=dynamic_remaining_s,
            tau_rounds=blend_tau_rounds,
        )
    )

    agent_trace_id = _trace_id(target_key)
    snapshots: list[dict[str, Any]] = []
    for (
        sample,
        total_prior,
        prior_remaining,
        dynamic_remaining,
        dynamic_weight,
        predicted_remaining_raw,
    ) in zip(
        target_samples,
        total_prior_s,
        prior_remaining_s,
        dynamic_remaining_s,
        dynamic_weights,
        predicted_remaining_s,
    ):
        terminal = sample.rounds_completed == sample.round_count
        predicted_remaining = 0.0 if terminal else float(predicted_remaining_raw)
        predicted_total = (
            sample.actual_total_time_s
            if terminal
            else sample.elapsed_time_s + predicted_remaining
        )
        absolute_error = abs(predicted_total - sample.actual_total_time_s)
        snapshots.append(
            {
                "agent_trace_id": agent_trace_id,
                "prediction_epoch": "request_or_tool_return",
                "rounds_completed": sample.rounds_completed,
                "current_round": sample.rounds_completed,
                "round_count_evaluation_only": sample.round_count,
                "progress_fraction_evaluation_only": (
                    sample.rounds_completed / sample.round_count
                ),
                "current_request_id": sample.current_request_id,
                "elapsed_time_s": sample.elapsed_time_s,
                "total_prior_time_s": float(total_prior),
                "prior_remaining_time_s": float(prior_remaining),
                "dynamic_remaining_time_s": float(dynamic_remaining),
                "dynamic_weight": float(dynamic_weight),
                "predicted_remaining_time_s": float(predicted_remaining),
                "predicted_total_time_s": predicted_total,
                "actual_remaining_time_s": sample.actual_remaining_time_s,
                "actual_total_time_s": sample.actual_total_time_s,
                "absolute_error_s": absolute_error,
                "relative_error_pct": (
                    100.0 * absolute_error / sample.actual_total_time_s
                ),
                "prediction_source": (
                    "terminal_observation"
                    if terminal
                    else "two_head_gbdt_blend"
                ),
                "cohort_source": cohort_source,
                "information_scope": (
                    "revealed_dag_static_metadata_plus_returned_request_prefix"
                ),
            }
        )

    first = target_rows[0]
    source = {
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "source_status": "measured_weka_agent_branch_wall_clock",
        "paper_ready": False,
        "case_study_figure_ready": True,
        "paper_ready_scope": "preselected_held_out_case_study_only",
        "generalization_ready": False,
        "paper_ready_reason": (
            "the case-study curve is reproducible and causal, but the matched "
            "peer LOOWF error is too large for a population-level accuracy claim"
        ),
        "generalization_limitation": (
            "the matched historical cohort contains four agents from four "
            "workflows; a paper-wide aggregate claim needs a preregistered "
            "multi-workflow test set"
        ),
        "agent_trace_id": agent_trace_id,
        "target_workflow_id": target_key[0],
        "holdout_scope": "entire_target_workflow",
        "round_count": len(target_rows),
        "actual_total_agent_wall_clock_s": (
            float(first["trace_oracle_branch_duration_ms"]) / 1000.0
        ),
        "ground_truth_field": "trace_oracle_branch_duration_ms",
        "label_uses_trace_oracle": True,
        "predictor_uses_trace_oracle": False,
        "future_output_length_used": False,
        "current_output_length_used": True,
        "future_api_time_used": False,
        "current_api_time_used": True,
        "future_arrival_or_completion_time_used": False,
        "current_request_prompt_length_used": True,
        "completed_prefix_api_time_used": True,
        "completed_prefix_output_length_used": True,
        "workflow_or_dag_static_features_used": [
            "stage_id",
            "stage_width",
            "stage_local_agent_index",
            "model",
            "source_model_role",
        ],
        "cohort_source": cohort_source,
        "cohort_audit": cohort_audit,
        "training_workflows": len(
            {sample.workflow_id for sample in cohort_samples}
        ),
        "training_agents": len({sample.key for sample in cohort_samples}),
        "training_snapshots": len(cohort_samples),
        "total_prior_training_samples": len(
            _first_sample_per_agent(cohort_samples)
        ),
        "dynamic_training_snapshots": len(
            [
                sample
                for sample in cohort_samples
                if sample.rounds_completed < sample.round_count
            ]
        ),
        "model_family": "two sklearn.ensemble.GradientBoostingRegressor heads",
        "sklearn_version": sklearn.__version__,
        "target_transform": {
            "total_prior": "log1p(actual_total_agent_wall_clock_s)",
            "dynamic_remaining": "log1p(actual_remaining_wall_clock_s)",
        },
        "sample_weight": (
            "one static sample per historical agent; each agent has total "
            "weight one across nonterminal dynamic snapshots"
        ),
        "hyperparameter_selection": (
            "leave-one-workflow-out CV on the historical cohort; target "
            "workflow excluded"
        ),
        "selection_objective": (
            "minimum LOOWF agent-mean relative total-time error per head; "
            "then the strongest finite prior shrinkage within 0.05 percentage "
            "points of the best LOOWF blend"
        ),
        "selected_config": {
            "total_prior": asdict(total_config),
            "dynamic_remaining": asdict(dynamic_config),
            "blend_tau_rounds": blend_tau_rounds,
        },
        "cross_validation_results": {
            "total_prior": total_cv_results,
            "dynamic_remaining": dynamic_cv_results,
            "blend": blend_cv_results,
        },
        "cohort_validation": cohort_validation,
        "feature_names": {
            "total_prior": list(total_vectorizer.get_feature_names_out()),
            "dynamic_remaining": list(
                dynamic_vectorizer.get_feature_names_out()
            ),
        },
        "remaining_time_prediction": (
            "(1-lambda_k)*max(GBDT_total(static)-elapsed_k,0) + "
            "lambda_k*GBDT_remaining(history_k), where "
            "lambda_k=1-exp(-rounds_completed/tau)"
        ),
        "known_limitation": (
            "the terminal point is a structural zero from the observed agent "
            "completion event; it is not counted as a GBDT invocation"
        ),
    }
    return snapshots, source


def summarize(
    snapshots: list[dict[str, Any]], source: dict[str, Any]
) -> dict[str, Any]:
    errors = np.asarray(
        [float(row["relative_error_pct"]) for row in snapshots]
    )
    gbdt_errors = np.asarray(
        [
            float(row["relative_error_pct"])
            for row in snapshots
            if row["prediction_source"] == "two_head_gbdt_blend"
        ]
    )
    round_count = int(source["round_count"])
    checkpoints: list[dict[str, Any]] = []
    for progress_pct in range(0, 100, 10):
        index = min(round(round_count * progress_pct / 100.0), round_count - 1)
        row = snapshots[index]
        checkpoints.append(
            {
                "progress_pct_evaluation_only": progress_pct,
                "rounds_completed": int(row["rounds_completed"]),
                "elapsed_time_s": float(row["elapsed_time_s"]),
                "predicted_remaining_time_s": float(
                    row["predicted_remaining_time_s"]
                ),
                "predicted_total_time_s": float(row["predicted_total_time_s"]),
                "actual_total_time_s": float(row["actual_total_time_s"]),
                "relative_error_pct": float(row["relative_error_pct"]),
            }
        )

    return {
        "schema_version": 8,
        "figure": "alg1-agent-time-accuracy",
        "semantics": "request_return_two_head_gbdt_remaining_v3",
        "algorithm1_stage": "Predict",
        "observation_epoch": "request_or_tool_return",
        "prediction_formula": (
            "prior_remaining(k)=max(GBDT_total(static)-elapsed(k),0); "
            "lambda(k)=1-exp(-rounds_completed/tau); "
            "predicted_remaining(k)=(1-lambda(k))*prior_remaining(k)+"
            "lambda(k)*GBDT_remaining(history(k))"
        ),
        "relative_error_formula": (
            "100 * abs(predicted_total(k) - actual_branch_duration) / "
            "actual_branch_duration"
        ),
        "source": source,
        "selection": {
            "agents": 1,
            "agent_trace_id": source["agent_trace_id"],
            "round_count": round_count,
            "prediction_points": len(snapshots),
            "gbdt_prediction_points": sum(
                row["prediction_source"] == "two_head_gbdt_blend"
                for row in snapshots
            ),
            "terminal_observation_included": True,
            "terminal_observation_excluded_from_gbdt_metrics": True,
            "workflow_structure_used": True,
        },
        "accuracy": {
            "metric_scope": "GBDT points only; terminal structural zero excluded",
            "initial_relative_error_pct": float(gbdt_errors[0]),
            "mean_relative_error_pct": float(np.mean(gbdt_errors)),
            "late_half_mean_relative_error_pct": float(
                np.mean(gbdt_errors[len(gbdt_errors) // 2 :])
            ),
            "last_pre_completion_relative_error_pct": float(gbdt_errors[-1]),
            "terminal_relative_error_pct": float(errors[-1]),
            "max_relative_error_pct": float(np.max(gbdt_errors)),
            "all_point_mean_including_terminal_pct": float(np.mean(errors)),
        },
        "checkpoints": checkpoints,
    }


def _write_snapshots(path: Path, snapshots: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SNAPSHOT_FIELDS)
        writer.writeheader()
        writer.writerows(snapshots)


def render(
    *,
    snapshots: list[dict[str, Any]],
    summary: dict[str, Any],
    output_prefix: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, error_ax = plt.subplots(figsize=(8.8, 4.8))
    elapsed = np.asarray([float(row["elapsed_time_s"]) for row in snapshots])
    errors = np.asarray(
        [float(row["relative_error_pct"]) for row in snapshots]
    )

    error_ax.plot(
        elapsed,
        errors,
        color="#2878B5",
        linewidth=2.2,
        marker="o",
        markersize=4.2,
    )
    error_ax.set_xlim(0.0, float(elapsed[-1]) * 1.035)
    error_max = float(np.max(errors))
    y_limit = max(error_max * 1.35, 0.25)
    if y_limit <= 2.0:
        y_step = 0.2
    elif y_limit <= 5.0:
        y_step = 1.0
    elif y_limit <= 10.0:
        y_step = 2.0
    else:
        y_step = 5.0
    error_ax.set_ylim(0.0, y_limit)
    error_ax.xaxis.set_major_locator(MultipleLocator(20.0))
    error_ax.yaxis.set_major_locator(MultipleLocator(y_step))
    error_ax.set_xlabel("Elapsed agent wall-clock time at request/tool return (s)")
    error_ax.set_ylabel("Relative execution-time prediction error (%)")
    error_ax.grid(True, linewidth=0.5, alpha=0.24)

    source = summary["source"]
    accuracy = summary["accuracy"]
    peer_cv = source["cohort_validation"]
    first = snapshots[0]
    last = snapshots[-1]
    error_ax.annotate(
        (
            f"round 1 total {float(first['predicted_total_time_s']):.1f} s\n"
            f"error {float(first['relative_error_pct']):.2f}%"
        ),
        xy=(float(first["elapsed_time_s"]), float(first["relative_error_pct"])),
        xytext=(10, 13),
        textcoords="offset points",
        fontsize=8.5,
        color="#555555",
    )
    error_ax.annotate(
        (
            f"completion observed\n"
            f"remaining 0 s · error {float(last['relative_error_pct']):.0f}%"
        ),
        xy=(float(last["elapsed_time_s"]), float(last["relative_error_pct"])),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color="#555555",
    )

    fig.suptitle("Algorithm 1: Causal GBDT Agent-Time Accuracy", y=0.98)
    error_ax.set_title(
        (
            f"held-out 40-round case · true wall-clock "
            f"{float(source['actual_total_agent_wall_clock_s']):.1f} s · "
            f"target mean {float(accuracy['mean_relative_error_pct']):.2f}% · "
            f"matched-peer LOOWF median "
            f"{float(peer_cv['median_agent_mean_relative_error_pct']):.1f}%"
        ),
        fontsize=9.0,
        color="#555555",
        pad=8,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=360, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--trace-prefix", default=DEFAULT_TRACE_PREFIX)
    parser.add_argument("--stage-id", type=int, default=DEFAULT_STAGE_ID)
    parser.add_argument("--agent-suffix", default=DEFAULT_AGENT_SUFFIX)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    snapshots, source = build_snapshots(
        trace_path=args.trace,
        trace_prefix=args.trace_prefix,
        stage_id=args.stage_id,
        agent_suffix=args.agent_suffix,
    )
    summary = summarize(snapshots, source)
    _write_snapshots(args.snapshots, snapshots)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    render(snapshots=snapshots, summary=summary, output_prefix=args.output_prefix)
    print(
        json.dumps(
            {
                "png": str(args.output_prefix.with_suffix('.png')),
                "pdf": str(args.output_prefix.with_suffix('.pdf')),
                "snapshots": str(args.snapshots),
                "summary": str(args.summary),
                "accuracy": summary["accuracy"],
                "selected_config": source["selected_config"],
                "cohort": {
                    "source": source["cohort_source"],
                    "agents": source["training_agents"],
                    "snapshots": source["training_snapshots"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
