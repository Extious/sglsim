"""Streaming loader and causal Algorithm-1 replay for the full Weka corpus."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from sglsim.srt.simulation.aegis_gbdt import (
    AegisGBDTConfig,
    AegisGBDTRegressor,
)

try:
    import orjson
except ImportError:  # pragma: no cover - server/test extras normally provide it.
    orjson = None


WEKA_DATASET = "semianalysisai/cc-traces-weka-062126"
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
FORBIDDEN_FEATURE_FRAGMENTS = (
    "output",
    "finish",
    "future",
    "oracle",
    "actual",
    "posthoc",
    "failure",
)


def default_full_trace_path() -> Path:
    root = Path(__file__).resolve().parents[4]
    return (
        root
        / "AegisKV"
        / "doc"
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


@dataclass(frozen=True)
class WekaRequest:
    rid: str
    workflow_id: str
    stage_id: int
    agent_id: str
    role: str
    model: str
    prompt_tokens: int
    output_tokens: int
    arrival_ms: float
    api_time_ms: float
    ttft_ms: float | None
    source_agent_id: str = ""
    hash_ids: tuple[int, ...] = ()

    @property
    def completion_ms(self) -> float:
        return self.arrival_ms + self.api_time_ms


@dataclass(frozen=True)
class WekaAgentBranch:
    agent_id: str
    role: str
    requests: tuple[WekaRequest, ...]

    @property
    def start_ms(self) -> float:
        return min(row.arrival_ms for row in self.requests)

    @property
    def finish_ms(self) -> float:
        return max(row.completion_ms for row in self.requests)


@dataclass(frozen=True)
class WekaStage:
    stage_id: int
    kind: str
    branches: tuple[WekaAgentBranch, ...]

    @property
    def start_ms(self) -> float:
        return min(branch.start_ms for branch in self.branches)

    @property
    def finish_ms(self) -> float:
        return max(branch.finish_ms for branch in self.branches)


@dataclass(frozen=True)
class WekaWorkflow:
    workflow_id: str
    models: tuple[str, ...]
    stages: tuple[WekaStage, ...]
    block_size: int = 64

    @property
    def requests(self) -> tuple[WekaRequest, ...]:
        return tuple(
            request
            for stage in self.stages
            for branch in stage.branches
            for request in branch.requests
        )

    @property
    def start_ms(self) -> float:
        return min(stage.start_ms for stage in self.stages)

    @property
    def finish_ms(self) -> float:
        return max(stage.finish_ms for stage in self.stages)


@dataclass(frozen=True)
class CausalRemainingSample:
    workflow_id: str
    stage_id: int
    epoch: int
    event_ms: float
    agent_id: str
    role: str
    actual_remaining_ms: float
    features: tuple[float, ...]
    publish_for_request_id: str = ""

    @property
    def snapshot_key(self) -> tuple[str, int, int]:
        return (self.workflow_id, self.stage_id, self.epoch)


@dataclass(frozen=True)
class CausalSlackTicket:
    request_id: str
    workflow_id: str
    stage_id: int
    agent_id: str
    published_at_ms: float
    slack_lcb_ms: float
    slack_q50_ms: float
    actual_slack_ms: float
    remaining_q50_ms: float
    remaining_interval_lower_ms: float
    remaining_interval_upper_ms: float
    fold: int
    predictor_sha256: str


def _json_loads(line: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def _segments(requests: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    output: list[tuple[str, list[dict[str, Any]]]] = []
    current_kind = ""
    current: list[dict[str, Any]] = []
    for request in requests:
        kind = "subagent" if request.get("type") == "subagent" else "main"
        if current and kind != current_kind:
            output.append((current_kind, current))
            current = []
        current_kind = kind
        current.append(request)
    if current:
        output.append((current_kind, current))
    return output


def _subagent_interval(group: dict[str, Any]) -> tuple[float, float]:
    inner = tuple(group.get("requests") or ())
    starts = [float(group.get("t", 0.0))]
    finishes = [
        float(group.get("t", 0.0))
        + max(float(group.get("duration_ms", 0.0)), 0.0) / 1000.0
    ]
    for request in inner:
        started = float(request.get("t", group.get("t", 0.0)))
        starts.append(started)
        finishes.append(started + max(float(request.get("api_time", 0.0)), 0.0))
    return min(starts), max(finishes)


def _split_parallel_subagent_groups(
    groups: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Split consecutive subagents into actual overlap-connected fan-outs."""

    ordered = sorted(
        groups,
        key=lambda group: (*_subagent_interval(group), str(group.get("agent_id") or "")),
    )
    stages: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_finish = -math.inf
    for group in ordered:
        started, finished = _subagent_interval(group)
        if current and started > current_finish + 1e-9:
            stages.append(current)
            current = []
            current_finish = -math.inf
        current.append(group)
        current_finish = max(current_finish, finished)
    if current:
        stages.append(current)
    return stages


def _causal_stages(
    requests: Sequence[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    stages: list[tuple[str, list[dict[str, Any]]]] = []
    for kind, items in _segments(requests):
        if kind == "main":
            stages.append((kind, items))
        else:
            stages.extend(
                (kind, group)
                for group in _split_parallel_subagent_groups(items)
            )
    return stages


def _request(
    raw: dict[str, Any],
    *,
    workflow_id: str,
    stage_id: int,
    agent_id: str,
    role: str,
    rid: str,
    source_agent_id: str = "",
) -> WekaRequest:
    ttft = raw.get("ttft")
    return WekaRequest(
        rid=rid,
        workflow_id=workflow_id,
        stage_id=stage_id,
        agent_id=agent_id,
        role=role,
        model=str(raw.get("model") or "unknown"),
        # Preserve the dataset's exact token accounting.  Feature construction
        # applies its own log-safe floor without mutating workload scale.
        prompt_tokens=max(int(raw.get("in", 0)), 0),
        output_tokens=max(int(raw.get("out", 0)), 0),
        arrival_ms=max(float(raw.get("t", 0.0)), 0.0) * 1000.0,
        api_time_ms=max(float(raw.get("api_time", 0.0)), 0.0) * 1000.0,
        ttft_ms=(
            max(float(ttft), 0.0) * 1000.0 if ttft is not None else None
        ),
        source_agent_id=str(source_agent_id),
        hash_ids=tuple(int(value) for value in raw.get("hash_ids") or ()),
    )


def parse_weka_workflow(raw: dict[str, Any]) -> WekaWorkflow:
    workflow_id = str(raw["id"])
    short = workflow_id[:8]
    stages: list[WekaStage] = []
    for stage_id, (kind, items) in enumerate(
        _causal_stages(raw["requests"]), start=1
    ):
        branches: list[WekaAgentBranch] = []
        if kind == "main":
            agent_id = f"{short}-p{stage_id}-main"
            requests = tuple(
                _request(
                    row,
                    workflow_id=workflow_id,
                    stage_id=stage_id,
                    agent_id=agent_id,
                    role="main",
                    rid=f"{short}-p{stage_id}-m{index}",
                )
                for index, row in enumerate(items)
            )
            if requests:
                branches.append(WekaAgentBranch(agent_id, "main", requests))
        else:
            for branch_index, group in enumerate(items):
                agent_id = f"{short}-p{stage_id}-sub{branch_index + 1}"
                requests = tuple(
                    _request(
                        row,
                        workflow_id=workflow_id,
                        stage_id=stage_id,
                        agent_id=agent_id,
                        role="subagent",
                        rid=(
                            f"{short}-p{stage_id}-s{branch_index}-t{turn_index}"
                        ),
                        source_agent_id=str(group.get("agent_id") or ""),
                    )
                    for turn_index, row in enumerate(group.get("requests") or ())
                )
                if requests:
                    branches.append(
                        WekaAgentBranch(agent_id, "subagent", requests)
                    )
        if branches:
            stages.append(WekaStage(stage_id, kind, tuple(branches)))
    if not stages:
        raise ValueError(f"Weka workflow {workflow_id} contains no model requests")
    return WekaWorkflow(
        workflow_id=workflow_id,
        models=tuple(str(model) for model in raw.get("models") or ()),
        stages=tuple(stages),
        block_size=max(int(raw.get("block_size", 64)), 1),
    )


def iter_weka_workflows(
    path: Path,
    *,
    limit: int | None = None,
) -> Iterator[WekaWorkflow]:
    if not path.is_file():
        raise FileNotFoundError(path)
    emitted = 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            yield parse_weka_workflow(_json_loads(line))
            emitted += 1
            if limit is not None and emitted >= int(limit):
                break


def load_weka_workflows(path: Path, *, limit: int | None = None) -> list[WekaWorkflow]:
    return list(iter_weka_workflows(path, limit=limit))


def dataset_manifest(
    workflows: Sequence[WekaWorkflow],
    *,
    source_path: Path,
    source_sha256: str = "",
) -> dict[str, Any]:
    requests = [request for workflow in workflows for request in workflow.requests]
    return {
        "dataset": WEKA_DATASET,
        "source_path": str(source_path.resolve()),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": str(source_sha256),
        "workflow_count": len(workflows),
        "request_count": len(requests),
        "input_tokens": sum(row.prompt_tokens for row in requests),
        "output_tokens": sum(row.output_tokens for row in requests),
        "subagent_request_count": sum(row.role == "subagent" for row in requests),
        "parallel_stage_count": sum(
            _peak_concurrent_branches(stage) > 1
            for workflow in workflows
            for stage in workflow.stages
        ),
        "max_stage_width": max(
            _peak_concurrent_branches(stage)
            for workflow in workflows
            for stage in workflow.stages
        ),
        "max_prompt_tokens": max(row.prompt_tokens for row in requests),
        "max_output_tokens": max(row.output_tokens for row in requests),
        "block_sizes": sorted({workflow.block_size for workflow in workflows}),
        "request_with_prefix_hash_count": sum(bool(row.hash_ids) for row in requests),
        "source_semantics": "real_weka_request_scale_and_timing",
    }


def _peak_concurrent_branches(stage: WekaStage) -> int:
    events: list[tuple[float, int]] = []
    for branch in stage.branches:
        # End before start at an identical timestamp: a completed branch and a
        # newly revealed branch are not simultaneously in the active frontier.
        events.append((branch.start_ms, 1))
        events.append((branch.finish_ms, -1))
    active = 0
    peak = 0
    for _time_ms, delta in sorted(events, key=lambda row: (row[0], row[1])):
        active += delta
        peak = max(peak, active)
    return peak


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _features(
    *,
    event_ms: float,
    stage_start_ms: float,
    completed: Sequence[WekaRequest],
    revealed: Sequence[WekaRequest],
    stage_width: int,
    stage_id: int,
    role: str,
) -> tuple[float, ...]:
    elapsed = max(float(event_ms) - float(stage_start_ms), 0.0)
    last_completion = completed[-1].completion_ms if completed else None
    since_own = elapsed if last_completion is None else max(event_ms - last_completion, 0.0)
    prompts = [max(row.prompt_tokens, 1) for row in revealed]
    current_prompt = prompts[-1]
    completed_count = len(completed)
    role_subagent = 1.0 if role == "subagent" else 0.0
    log_elapsed = math.log1p(elapsed)
    return (
        log_elapsed,
        math.log1p(since_own),
        float(completed_count),
        float(completed_count * completed_count),
        float(len(revealed)),
        math.log1p(current_prompt),
        math.log1p(max(prompts)),
        math.log1p(sum(prompts)),
        float(stage_width),
        float(stage_id),
        role_subagent,
        1.0 if last_completion is None else 0.0,
        role_subagent * math.log1p(current_prompt),
        float(completed_count) * log_elapsed,
    )


def build_causal_samples(
    workflows: Sequence[WekaWorkflow],
) -> list[CausalRemainingSample]:
    """Build only request-return snapshots; labels never enter features."""

    for name in FEATURE_NAMES:
        if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            raise AssertionError(f"Algorithm-1 feature {name!r} is not causal")
    samples: list[CausalRemainingSample] = []
    for workflow in workflows:
        for stage in workflow.stages:
            if len(stage.branches) < 2:
                continue
            event_agents: dict[float, dict[str, set[str]]] = defaultdict(
                lambda: {"started": set(), "completed": set()}
            )
            completion_agents: dict[float, set[str]] = defaultdict(set)
            for branch in stage.branches:
                event_agents[branch.start_ms]["started"].add(branch.agent_id)
                for request in branch.requests:
                    completion_agents[request.completion_ms].add(branch.agent_id)
            for event_ms, agents in completion_agents.items():
                event_agents[event_ms]["completed"].update(agents)
            event_rows = sorted(event_agents.items())
            for epoch, (event_ms, event) in enumerate(event_rows):
                newly_started = event["started"]
                newly_completed = event["completed"]
                active = [
                    branch
                    for branch in stage.branches
                    if branch.start_ms <= event_ms + 1e-9
                    and branch.finish_ms > event_ms + 1e-9
                ]
                if not active:
                    continue
                for branch in active:
                    ordered = tuple(
                        sorted(
                            branch.requests,
                            key=lambda row: (row.arrival_ms, row.rid),
                        )
                    )
                    completed = tuple(
                        row for row in ordered if row.completion_ms <= event_ms + 1e-9
                    )
                    next_index = min(len(completed), len(ordered) - 1)
                    revealed = ordered[: next_index + 1]
                    publish = ""
                    if (
                        branch.agent_id in newly_started
                        or branch.agent_id in newly_completed
                    ):
                        if len(completed) < len(ordered):
                            candidate = ordered[len(completed)]
                            # A static request descriptor may consume only a
                            # ticket that existed when that request arrived.
                            # Some Weka branch requests overlap, so a prior
                            # completion can occur after the next request has
                            # already started.  Such mid-flight knowledge must
                            # not be backdated into the request; fail closed and
                            # leave it for an explicit future update path.
                            if event_ms <= candidate.arrival_ms + 1e-9:
                                publish = candidate.rid
                    samples.append(
                        CausalRemainingSample(
                            workflow_id=workflow.workflow_id,
                            stage_id=stage.stage_id,
                            epoch=epoch,
                            event_ms=event_ms,
                            agent_id=branch.agent_id,
                            role=branch.role,
                            actual_remaining_ms=max(branch.finish_ms - event_ms, 0.0),
                            features=_features(
                                event_ms=event_ms,
                                stage_start_ms=stage.start_ms,
                                completed=completed,
                                revealed=revealed,
                                stage_width=len(active),
                                stage_id=stage.stage_id,
                                role=branch.role,
                            ),
                            publish_for_request_id=publish,
                        )
                    )
    return samples


def workflow_fold(workflow_id: str, folds: int, *, salt: str) -> int:
    if int(folds) < 3:
        raise ValueError("causal cross-fitting requires at least three folds")
    digest = hashlib.sha256(f"{salt}:{workflow_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % int(folds)


def _upper_quantile(values: np.ndarray, coverage: float) -> float:
    if not len(values):
        raise ValueError("calibration partition is empty")
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = math.ceil((len(ordered) + 1) * float(coverage))
    return float(ordered[min(max(rank - 1, 0), len(ordered) - 1)])


def crossfit_causal_slack(
    samples: Sequence[CausalRemainingSample],
    *,
    folds: int = 5,
    split_salt: str = "aegiskv-weka-crossfit-v1",
    interval_coverage: float = 0.90,
    gbdt_config: AegisGBDTConfig | None = None,
    blend_tau_rounds: float = 2.0,
) -> tuple[dict[str, CausalSlackTicket], dict[str, Any]]:
    if not samples:
        raise ValueError("Algorithm-1 cross-fitting requires causal samples")
    if not 0.5 < float(interval_coverage) < 1.0:
        raise ValueError("interval_coverage must be in (0.5, 1)")
    if not math.isfinite(float(blend_tau_rounds)) or float(blend_tau_rounds) <= 0.0:
        raise ValueError("blend_tau_rounds must be finite and positive")
    fold_ids = np.asarray(
        [workflow_fold(row.workflow_id, folds, salt=split_salt) for row in samples],
        dtype=np.int16,
    )
    feature_matrix = np.asarray([row.features for row in samples], dtype=float)
    target = np.asarray([row.actual_remaining_ms for row in samples], dtype=float)
    predictions = np.empty(len(samples), dtype=float)
    margins = np.empty(len(samples), dtype=float)
    predictor_by_fold: dict[int, dict[str, Any]] = {}
    config = gbdt_config or AegisGBDTConfig()

    # Algorithm 1 uses two heads.  The static head sees the first causally
    # revealed state of each agent; the dynamic head sees the current request-
    # return prefix.  A request-count-dependent blend shifts weight to the
    # dynamic head as real progress accumulates.
    agent_keys = [
        (row.workflow_id, row.stage_id, row.agent_id) for row in samples
    ]
    first_index_by_agent: dict[tuple[str, int, str], int] = {}
    for index, key in enumerate(agent_keys):
        prior = first_index_by_agent.get(key)
        if prior is None or samples[index].event_ms < samples[prior].event_ms:
            first_index_by_agent[key] = index
    first_index = np.asarray(
        [first_index_by_agent[key] for key in agent_keys], dtype=np.int64
    )
    initial_feature_matrix = feature_matrix[first_index]

    def blended_prediction(
        total_model: AegisGBDTRegressor,
        dynamic_model: AegisGBDTRegressor,
        indices: np.ndarray,
    ) -> np.ndarray:
        total = total_model.predict(initial_feature_matrix[indices])
        elapsed = np.expm1(feature_matrix[indices, 0])
        static_remaining = np.maximum(total - elapsed, 0.0)
        dynamic_remaining = dynamic_model.predict(feature_matrix[indices])
        completed = feature_matrix[indices, 2]
        dynamic_weight = 1.0 - np.exp(
            -completed / float(blend_tau_rounds)
        )
        return (
            (1.0 - dynamic_weight) * static_remaining
            + dynamic_weight * dynamic_remaining
        )

    for target_fold in range(int(folds)):
        calibration_fold = (target_fold + 1) % int(folds)
        fit_mask = (fold_ids != target_fold) & (fold_ids != calibration_fold)
        calibration_mask = fold_ids == calibration_fold
        target_mask = fold_ids == target_fold
        if not np.any(fit_mask) or not np.any(calibration_mask) or not np.any(target_mask):
            raise ValueError(f"cross-fit fold {target_fold} is empty")
        dynamic_model = AegisGBDTRegressor(config).fit(
            feature_matrix[fit_mask], target[fit_mask]
        )
        fit_workflow_ids = {
            samples[index].workflow_id for index in np.flatnonzero(fit_mask)
        }
        total_fit_indices = np.asarray(
            [
                index
                for key, index in first_index_by_agent.items()
                if key[0] in fit_workflow_ids
            ],
            dtype=np.int64,
        )
        total_model = AegisGBDTRegressor(config).fit(
            feature_matrix[total_fit_indices], target[total_fit_indices]
        )
        calibration_indices = np.flatnonzero(calibration_mask)
        target_indices = np.flatnonzero(target_mask)
        calibration_prediction = blended_prediction(
            total_model, dynamic_model, calibration_indices
        )
        # Requests within one agent workflow are highly correlated.  Calibrating
        # one residual per request therefore overstates the effective sample size
        # and under-covered the complete Weka corpus.  Use the maximum absolute
        # residual in each calibration workflow as the conformal score.  The
        # resulting interval is simultaneous over every causal snapshot in one
        # unseen workflow (subject to the usual workflow-exchangeability
        # assumption), which is the unit sampled by the paper experiment.
        calibration_error = np.abs(
            calibration_prediction - target[calibration_mask]
        )
        calibration_workflow_scores: dict[str, float] = {}
        for index, error in zip(calibration_indices, calibration_error):
            workflow_id = samples[int(index)].workflow_id
            calibration_workflow_scores[workflow_id] = max(
                calibration_workflow_scores.get(workflow_id, 0.0),
                float(error),
            )
        margin = _upper_quantile(
            np.asarray(list(calibration_workflow_scores.values()), dtype=float),
            interval_coverage,
        )
        predictions[target_mask] = blended_prediction(
            total_model, dynamic_model, target_indices
        )
        margins[target_mask] = margin
        dynamic_manifest = dynamic_model.manifest()
        total_manifest = total_model.manifest()
        combined_sha256 = hashlib.sha256(
            (
                f"{total_manifest['sha256']}:{dynamic_manifest['sha256']}:"
                f"{float(blend_tau_rounds):.12g}:{float(interval_coverage):.12g}:"
                f"{float(margin):.12g}:workflow_max_absolute_residual"
            ).encode()
        ).hexdigest()
        predictor_by_fold[target_fold] = {
            "schema": "aegis_dual_head_histogram_gbdt_v1",
            "sha256": combined_sha256,
            "total_head": total_manifest,
            "dynamic_remaining_head": dynamic_manifest,
            "blend_tau_rounds": float(blend_tau_rounds),
            "fit_workflows": len(
                {samples[index].workflow_id for index in np.flatnonzero(fit_mask)}
            ),
            "calibration_workflows": len(
                {
                    samples[index].workflow_id
                    for index in np.flatnonzero(calibration_mask)
                }
            ),
            "evaluation_workflows": len(
                {
                    samples[index].workflow_id
                    for index in np.flatnonzero(target_mask)
                }
            ),
            "calibrated_absolute_error_ms": margin,
            "calibration_unit": "workflow_max_absolute_residual",
            "calibration_score_count": len(calibration_workflow_scores),
        }

    by_snapshot: dict[
        tuple[str, int, int], list[tuple[int, CausalRemainingSample]]
    ] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_snapshot[sample.snapshot_key].append((index, sample))

    tickets: dict[str, CausalSlackTicket] = {}
    metric_rows: list[dict[str, float | bool]] = []
    for rows in by_snapshot.values():
        lower = {
            index: max(float(predictions[index] - margins[index]), 0.0)
            for index, _sample in rows
        }
        upper = {
            index: float(predictions[index] + margins[index])
            for index, _sample in rows
        }
        for index, sample in rows:
            other_indices = [other for other, _ in rows if other != index]
            slack_lcb = max(
                max((lower[other] for other in other_indices), default=0.0)
                - upper[index],
                0.0,
            )
            slack_q50 = max(
                max(
                    (float(predictions[other]) for other in other_indices),
                    default=0.0,
                )
                - float(predictions[index]),
                0.0,
            )
            actual_slack = max(
                max(
                    (
                        samples[other].actual_remaining_ms
                        for other in other_indices
                    ),
                    default=0.0,
                )
                - sample.actual_remaining_ms,
                0.0,
            )
            metric_rows.append(
                {
                    "remaining_error_ms": abs(
                        float(predictions[index]) - sample.actual_remaining_ms
                    ),
                    "remaining_covered": (
                        lower[index] - 1e-9
                        <= sample.actual_remaining_ms
                        <= upper[index] + 1e-9
                    ),
                    "slack_error_ms": abs(slack_q50 - actual_slack),
                    "slack_lcb_covered": actual_slack + 1e-9 >= slack_lcb,
                }
            )
            if not sample.publish_for_request_id:
                continue
            fold = int(fold_ids[index])
            ticket = CausalSlackTicket(
                request_id=sample.publish_for_request_id,
                workflow_id=sample.workflow_id,
                stage_id=sample.stage_id,
                agent_id=sample.agent_id,
                published_at_ms=sample.event_ms,
                slack_lcb_ms=slack_lcb,
                slack_q50_ms=slack_q50,
                actual_slack_ms=actual_slack,
                remaining_q50_ms=float(predictions[index]),
                remaining_interval_lower_ms=lower[index],
                remaining_interval_upper_ms=upper[index],
                fold=fold,
                predictor_sha256=str(predictor_by_fold[fold]["sha256"]),
            )
            prior = tickets.get(ticket.request_id)
            if prior is not None and prior != ticket:
                raise AssertionError(
                    f"request {ticket.request_id} received conflicting Slack tickets"
                )
            tickets[ticket.request_id] = ticket

    remaining_errors = np.asarray(
        [float(row["remaining_error_ms"]) for row in metric_rows], dtype=float
    )
    slack_errors = np.asarray(
        [float(row["slack_error_ms"]) for row in metric_rows], dtype=float
    )
    coverage_by_workflow: dict[str, bool] = defaultdict(lambda: True)
    for index, sample in enumerate(samples):
        coverage_by_workflow[sample.workflow_id] = bool(
            coverage_by_workflow[sample.workflow_id]
            and predictions[index] - margins[index] - 1e-9
            <= sample.actual_remaining_ms
            <= predictions[index] + margins[index] + 1e-9
        )
    metrics = {
        "schema": "causal_gbdt_cpm_crossfit_v1",
        "information_scope": "causal_request_return_prefix_only",
        "feature_names": list(FEATURE_NAMES),
        "forbidden_feature_audit_passed": True,
        "workflow_disjoint_crossfit_passed": True,
        "sample_count": len(samples),
        "ticket_count": len(tickets),
        "workflow_count": len({row.workflow_id for row in samples}),
        "folds": int(folds),
        "interval_target_coverage": float(interval_coverage),
        "interval_calibration_unit": "workflow_max_absolute_residual",
        "predictor_kind": "dual_head_gbdt_total_prior_plus_dynamic_remaining",
        "blend_tau_rounds": float(blend_tau_rounds),
        "remaining_mae_ms": float(remaining_errors.mean()),
        "remaining_p50_absolute_error_ms": float(
            np.quantile(remaining_errors, 0.50)
        ),
        "remaining_p90_absolute_error_ms": float(
            np.quantile(remaining_errors, 0.90)
        ),
        "remaining_interval_coverage": float(
            np.mean([bool(row["remaining_covered"]) for row in metric_rows])
        ),
        "workflow_simultaneous_interval_coverage": float(
            np.mean(list(coverage_by_workflow.values()))
        ),
        "slack_mae_ms": float(slack_errors.mean()),
        "slack_p50_absolute_error_ms": float(np.quantile(slack_errors, 0.50)),
        "slack_p90_absolute_error_ms": float(np.quantile(slack_errors, 0.90)),
        "slack_lcb_coverage": float(
            np.mean([bool(row["slack_lcb_covered"]) for row in metric_rows])
        ),
        "predictors": {
            str(key): value for key, value in sorted(predictor_by_fold.items())
        },
    }
    return tickets, metrics


def zero_slack_tickets_for_unmodeled_requests(
    workflows: Iterable[WekaWorkflow],
    tickets: dict[str, CausalSlackTicket],
) -> dict[str, CausalSlackTicket]:
    """Fail close for singleton stages or missing causal predictions."""

    output = dict(tickets)
    for workflow in workflows:
        for request in workflow.requests:
            if request.rid in output:
                continue
            output[request.rid] = CausalSlackTicket(
                request_id=request.rid,
                workflow_id=request.workflow_id,
                stage_id=request.stage_id,
                agent_id=request.agent_id,
                published_at_ms=request.arrival_ms,
                slack_lcb_ms=0.0,
                slack_q50_ms=0.0,
                actual_slack_ms=0.0,
                remaining_q50_ms=0.0,
                remaining_interval_lower_ms=0.0,
                remaining_interval_upper_ms=0.0,
                fold=-1,
                predictor_sha256="fail-close-zero",
            )
    return output


__all__ = [
    "CausalRemainingSample",
    "CausalSlackTicket",
    "FEATURE_NAMES",
    "WEKA_DATASET",
    "WekaAgentBranch",
    "WekaRequest",
    "WekaStage",
    "WekaWorkflow",
    "build_causal_samples",
    "crossfit_causal_slack",
    "dataset_manifest",
    "default_full_trace_path",
    "iter_weka_workflows",
    "load_weka_workflows",
    "parse_weka_workflow",
    "sha256_file",
    "workflow_fold",
    "zero_slack_tickets_for_unmodeled_requests",
]
