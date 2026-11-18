"""Legacy fixed-DAG Algorithm-1 replay helpers for AegisKV.

``AegisDagRequest.output_tokens`` has no provenance and archived callers pass
post-hoc trace output lengths.  This module remains for artifact replay and
must not be used as production causal evidence.  New online code must use
``aegis_causal_slack``.  Legacy tickets explicitly publish
``aegis_slack_uses_future_information=true`` so a causal snapshot fails closed.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


ONLINE_PROGRESS_MIN_FRACTION = 0.75


def _finite_sample_upper_quantile(values: Iterable[float], coverage: float) -> float:
    """Return the split-conformal upper quantile with finite-sample correction."""

    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        raise ValueError("at least one finite calibration score is required")
    coverage = min(max(float(coverage), 0.0), 1.0)
    rank = math.ceil((len(ordered) + 1) * coverage)
    return ordered[min(max(rank - 1, 0), len(ordered) - 1)]


def calibrated_lower_margin_s(
    *,
    actual_s: Iterable[float],
    raw_lower_s: Iterable[float],
    target_coverage: float,
) -> float:
    """Calibrate ``P(actual >= raw_lower - margin)`` to ``target_coverage``.

    A negative empirical margin would make the raw lower bound more
    aggressive.  AegisKV deliberately clamps it to zero: calibration may make
    a deadline more conservative, but never less conservative.
    """

    actual = list(actual_s)
    raw = list(raw_lower_s)
    if len(actual) != len(raw):
        raise ValueError("actual and raw lower-bound arrays must have equal length")
    scores = [float(lower) - float(observed) for observed, lower in zip(actual, raw)]
    return max(_finite_sample_upper_quantile(scores, target_coverage), 0.0)


def empirical_lower_bound_s(
    values: Iterable[float], *, target_coverage: float
) -> float:
    """Return a conservative historical lower bound for model-missing fallback.

    For a desired ``P(Y >= L) >= coverage``, ``L`` is the lower-tail order
    statistic at ``1 - coverage`` with a finite-sample conservative rank.
    """

    ordered = sorted(max(float(value), 0.0) for value in values)
    if not ordered:
        raise ValueError("at least one fallback observation is required")
    miss_probability = 1.0 - min(max(float(target_coverage), 0.0), 1.0)
    rank = math.floor((len(ordered) + 1) * miss_probability)
    return ordered[min(max(rank - 1, 0), len(ordered) - 1)]


def turn_bucket(turns_completed: int) -> str:
    turn = max(int(turns_completed), 1)
    if turn <= 5:
        return str(turn)
    return "6-9" if turn <= 9 else "10+"


@dataclass(frozen=True)
class AegisSlackEnvelope:
    lower_ms: float
    median_ms: float
    upper_ms: float
    target_coverage: float
    calibration_id: str
    calibration_group: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AegisSlackCalibration:
    calibration_id: str
    target_lower_coverage: float
    global_lower_margin_s: float
    role_lower_margins_s: dict[str, float]
    fallback_lower_bounds_s: dict[str, float]
    source: str
    evaluation_gate_passed: bool
    schema_version: int = 1
    semantics: str = "legacy_remaining_duration"
    slack_lower_margin_ms: float = 0.0
    request_cost_model: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AegisSlackCalibration":
        target = float(raw.get("target_lower_coverage", 0.0))
        if not 0.0 < target < 1.0:
            raise ValueError("target_lower_coverage must be strictly between 0 and 1")
        calibration_id = str(raw.get("calibration_id") or "").strip()
        if not calibration_id:
            raise ValueError("calibration_id is required")
        role_margins = {
            str(key): max(float(value), 0.0)
            for key, value in dict(raw.get("role_lower_margins_s") or {}).items()
        }
        fallback_bounds = {
            str(key): max(float(value), 0.0)
            for key, value in dict(raw.get("fallback_lower_bounds_s") or {}).items()
        }
        gate = raw.get("evaluation_gate") or {}
        return cls(
            calibration_id=calibration_id,
            target_lower_coverage=target,
            global_lower_margin_s=max(float(raw.get("global_lower_margin_s", 0.0)), 0.0),
            role_lower_margins_s=role_margins,
            fallback_lower_bounds_s=fallback_bounds,
            source=str(raw.get("source_predictions") or "unknown"),
            evaluation_gate_passed=bool(gate.get("passed", False)),
            schema_version=int(raw.get("schema_version", 1)),
            semantics=str(raw.get("semantics") or "legacy_remaining_duration"),
            slack_lower_margin_ms=max(
                float(raw.get("slack_lower_margin_ms", 0.0)), 0.0
            ),
            request_cost_model={
                str(key): float(value)
                for key, value in dict(raw.get("request_cost_model") or {}).items()
            },
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_passed_gate: bool = True,
        require_cpm_semantics: bool = True,
    ) -> "AegisSlackCalibration":
        artifact = Path(path)
        raw = json.loads(artifact.read_text(encoding="utf-8"))
        calibration = cls.from_dict(raw)
        if require_passed_gate and not calibration.evaluation_gate_passed:
            raise ValueError(f"Algorithm-1 calibration gate did not pass: {artifact}")
        if require_cpm_semantics and calibration.semantics != "fixed_dag_cpm_slack_v2":
            raise ValueError(
                "Algorithm-1 artifact predicts duration rather than fixed-DAG "
                f"CPM slack: {artifact} semantics={calibration.semantics!r}"
            )
        return calibration

    def _role_key(self, role: str) -> str:
        normalized = str(role or "unknown").strip().lower()
        return f"role={normalized}"

    def _fallback_keys(self, role: str, turns_completed: int) -> tuple[str, ...]:
        role_key = self._role_key(role)
        return (
            f"{role_key}|turn={turn_bucket(turns_completed)}",
            role_key,
            "global",
        )

    def calibrated_prediction(
        self,
        *,
        raw_lower_s: float,
        raw_median_s: float,
        raw_upper_s: float,
        role: str,
    ) -> AegisSlackEnvelope:
        role_key = self._role_key(role)
        margin = self.role_lower_margins_s.get(
            role_key, self.global_lower_margin_s
        )
        lower_s = max(float(raw_lower_s) - margin, 0.0)
        median_s = max(float(raw_median_s), lower_s)
        upper_s = max(float(raw_upper_s), median_s)
        return AegisSlackEnvelope(
            lower_ms=round(lower_s * 1000.0, 3),
            median_ms=round(median_s * 1000.0, 3),
            upper_ms=round(upper_s * 1000.0, 3),
            target_coverage=self.target_lower_coverage,
            calibration_id=self.calibration_id,
            calibration_group=role_key if role_key in self.role_lower_margins_s else "global",
            source="quantile_model_plus_split_conformal",
        )

    def fallback_prediction(
        self, *, role: str, turns_completed: int
    ) -> AegisSlackEnvelope:
        for key in self._fallback_keys(role, turns_completed):
            if key in self.fallback_lower_bounds_s:
                lower_s = self.fallback_lower_bounds_s[key]
                return AegisSlackEnvelope(
                    lower_ms=round(lower_s * 1000.0, 3),
                    median_ms=round(lower_s * 1000.0, 3),
                    upper_ms=round(lower_s * 1000.0, 3),
                    target_coverage=self.target_lower_coverage,
                    calibration_id=self.calibration_id,
                    calibration_group=key,
                    source="workflow_disjoint_historical_lower_bound",
                )
        # Fail closed.  Missing calibration context removes opportunistic
        # slack, but it never disables the correctness fallback to recompute.
        return AegisSlackEnvelope(
            lower_ms=0.0,
            median_ms=0.0,
            upper_ms=0.0,
            target_coverage=self.target_lower_coverage,
            calibration_id=self.calibration_id,
            calibration_group="missing",
            source="fail_closed_zero_slack",
        )


@dataclass(frozen=True)
class AegisDagRequest:
    """Legacy/offline request whose output token source is not auditable."""

    request_id: str
    stage_id: int
    agent_id: str
    order: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class AegisRequestCostModel:
    request_setup_ms: float = 50.0
    prefill_ms_per_token: float = 0.333
    decode_ms_per_token: float = 28.6
    lower_scale: float = 0.90
    upper_scale: float = 1.10

    @classmethod
    def from_calibration(
        cls, calibration: AegisSlackCalibration
    ) -> "AegisRequestCostModel":
        raw = calibration.request_cost_model
        model = cls(
            request_setup_ms=max(float(raw.get("request_setup_ms", 50.0)), 0.0),
            prefill_ms_per_token=max(
                float(raw.get("prefill_ms_per_token", 0.333)), 0.0
            ),
            decode_ms_per_token=max(
                float(raw.get("decode_ms_per_token", 28.6)), 0.0
            ),
            lower_scale=max(float(raw.get("lower_scale", 0.90)), 0.0),
            upper_scale=max(float(raw.get("upper_scale", 1.10)), 0.0),
        )
        if model.lower_scale > 1.0 or model.upper_scale < 1.0:
            raise ValueError(
                "request-cost interval must satisfy lower_scale <= 1 <= upper_scale"
            )
        return model

    def request_median_ms(
        self, *, new_prefill_tokens: int, output_tokens: int
    ) -> float:
        return max(
            self.request_setup_ms
            + self.prefill_ms_per_token * max(int(new_prefill_tokens), 0)
            + self.decode_ms_per_token * max(int(output_tokens), 0),
            0.0,
        )


def fixed_dag_request_costs_ms(
    requests: Iterable[AegisDagRequest],
    *,
    calibration: AegisSlackCalibration,
) -> dict[str, float]:
    """Return legacy per-request costs used by archived fixed-DAG replay.

    Prefix materialization is tracked independently for every stage/agent
    branch.  This helper is shared by the initial CPM pass and the online
    progress correction so both paths charge identical work.
    """

    cost_model = AegisRequestCostModel.from_calibration(calibration)
    by_agent: dict[tuple[int, str], list[AegisDagRequest]] = defaultdict(list)
    for request in requests:
        if not request.request_id:
            raise ValueError("DAG request_id must be non-empty")
        if not request.agent_id:
            raise ValueError("DAG agent_id must be non-empty")
        by_agent[(int(request.stage_id), request.agent_id)].append(request)

    costs: dict[str, float] = {}
    for agent_requests in by_agent.values():
        ordered = sorted(agent_requests, key=lambda item: (item.order, item.request_id))
        max_materialized_prompt = 0
        for request in ordered:
            if request.request_id in costs:
                raise ValueError(f"duplicate DAG request_id: {request.request_id}")
            prompt_tokens = max(int(request.input_tokens), 0)
            new_prefill_tokens = max(prompt_tokens - max_materialized_prompt, 0)
            max_materialized_prompt = max(max_materialized_prompt, prompt_tokens)
            costs[request.request_id] = cost_model.request_median_ms(
                new_prefill_tokens=new_prefill_tokens,
                output_tokens=max(int(request.output_tokens), 0),
            )
    return costs


@dataclass(frozen=True)
class AegisOnlineProgressDecision:
    active_critical_agents: frozenset[str]
    newly_critical_agents: frozenset[str]
    projected_finish_ms: dict[str, float]
    trigger_agent_id: str | None
    trigger_progress_fraction: float


class AegisOnlineProgressTracker:
    """Causal late-stage correction for routing and batching interference.

    The fixed-DAG CPM estimate is available before a stage starts.  Once each
    branch has made observable progress, this tracker extrapolates finish time
    from completed modeled work and measured elapsed time.  A dynamic upgrade
    is permitted only in the last quarter of a branch and only while at least
    one future request remains, so the decision can still change protection.
    No future completion or barrier time is consumed.
    """

    def __init__(
        self,
        *,
        request_costs_by_agent: dict[str, list[tuple[str, float]]],
        static_critical_agents: Iterable[str] = (),
        minimum_progress_fraction: float = ONLINE_PROGRESS_MIN_FRACTION,
    ) -> None:
        self._request_agent: dict[str, str] = {}
        self._request_cost_ms: dict[str, float] = {}
        self._request_count: dict[str, int] = {}
        self._total_cost_ms: dict[str, float] = {}
        for agent_id, requests in request_costs_by_agent.items():
            normalized_agent = str(agent_id)
            self._request_count[normalized_agent] = len(requests)
            self._total_cost_ms[normalized_agent] = sum(
                max(float(cost_ms), 0.0) for _request_id, cost_ms in requests
            )
            for request_id, cost_ms in requests:
                normalized_request = str(request_id)
                if normalized_request in self._request_agent:
                    raise ValueError(
                        f"duplicate online-progress request_id: {normalized_request}"
                    )
                self._request_agent[normalized_request] = normalized_agent
                self._request_cost_ms[normalized_request] = max(float(cost_ms), 0.0)
        self._static_critical = frozenset(str(value) for value in static_critical_agents)
        self._minimum_progress_fraction = min(
            max(float(minimum_progress_fraction), 0.0), 1.0
        )
        self._completed_requests: dict[str, int] = defaultdict(int)
        self._completed_cost_ms: dict[str, float] = defaultdict(float)
        self._latest_elapsed_ms: dict[str, float] = {}
        self._observed_requests: set[str] = set()
        self._dynamic_active: frozenset[str] = frozenset()
        self._dynamic_ever: set[str] = set()
        self._last_dynamic_candidate: str | None = None
        self._last_dynamic_projection_ms: dict[str, float] = {}
        self._trigger_elapsed_ms: dict[str, float] = {}

    @property
    def active_critical_agents(self) -> frozenset[str]:
        if self._dynamic_active:
            return self._dynamic_active
        return self._static_critical

    @property
    def final_critical_agents(self) -> frozenset[str]:
        if self._last_dynamic_candidate is not None:
            return frozenset({self._last_dynamic_candidate})
        return self._static_critical

    @property
    def final_projected_finish_ms(self) -> dict[str, float]:
        return dict(self._last_dynamic_projection_ms)

    @property
    def final_projected_slack_ms(self) -> dict[str, float]:
        if not self._last_dynamic_projection_ms:
            return {}
        barrier_ms = max(self._last_dynamic_projection_ms.values())
        return {
            agent_id: max(barrier_ms - finish_ms, 0.0)
            for agent_id, finish_ms in self._last_dynamic_projection_ms.items()
        }

    @property
    def ever_critical_agents(self) -> frozenset[str]:
        return self._static_critical | frozenset(self._dynamic_ever)

    @property
    def dynamically_critical_agents(self) -> frozenset[str]:
        return self._dynamic_active

    @property
    def ever_dynamically_critical_agents(self) -> frozenset[str]:
        return frozenset(self._dynamic_ever)

    @property
    def trigger_elapsed_ms(self) -> dict[str, float]:
        return dict(self._trigger_elapsed_ms)

    def is_critical(self, agent_id: str) -> bool:
        return str(agent_id) in self.active_critical_agents

    def observe_completion(
        self, *, request_id: str, elapsed_ms: float
    ) -> AegisOnlineProgressDecision:
        request_id = str(request_id)
        if request_id not in self._request_agent:
            raise KeyError(f"unknown online-progress request_id: {request_id}")
        if request_id in self._observed_requests:
            raise ValueError(
                f"duplicate online-progress completion: {request_id}"
            )
        self._observed_requests.add(request_id)
        agent_id = self._request_agent[request_id]
        self._completed_requests[agent_id] += 1
        self._completed_cost_ms[agent_id] += self._request_cost_ms[request_id]
        self._latest_elapsed_ms[agent_id] = max(float(elapsed_ms), 0.0)

        projected: dict[str, float] = {}
        if all(
            self._completed_requests[candidate] > 0
            for candidate in self._request_count
        ):
            for candidate in self._request_count:
                completed_cost = self._completed_cost_ms[candidate]
                if completed_cost <= 0.0:
                    continue
                projected[candidate] = (
                    self._latest_elapsed_ms[candidate]
                    * self._total_cost_ms[candidate]
                    / completed_cost
                )

        trigger_agent: str | None = None
        trigger_fraction = 0.0
        if len(projected) == len(self._request_count) and projected:
            eligible = []
            for candidate in projected:
                completed = self._completed_requests[candidate]
                total = self._request_count[candidate]
                progress = completed / max(total, 1)
                # A literal 75% threshold is unreachable while work remains
                # for two- and three-request branches.  Use the configured
                # last-quarter point when possible, otherwise the final causal
                # observation before completion.  No future barrier or finish
                # time is consumed.
                trigger_completed = min(
                    max(math.ceil(self._minimum_progress_fraction * total), 1),
                    max(total - 1, 1),
                )
                if completed < total and completed >= trigger_completed:
                    eligible.append((candidate, progress))
            if eligible:
                candidate, progress = max(
                    eligible,
                    key=lambda value: (projected[value[0]], value[0]),
                )
                trigger_agent = candidate
                trigger_fraction = progress
                self._dynamic_active = frozenset({candidate})
                self._last_dynamic_candidate = candidate
                self._last_dynamic_projection_ms = dict(projected)
                if candidate not in self._dynamic_ever:
                    self._trigger_elapsed_ms[candidate] = max(float(elapsed_ms), 0.0)
                self._dynamic_ever.add(candidate)
            elif self._last_dynamic_candidate is not None:
                self._dynamic_active = frozenset()

        newly_critical = self._dynamic_active - self._static_critical
        return AegisOnlineProgressDecision(
            active_critical_agents=self.active_critical_agents,
            newly_critical_agents=frozenset(newly_critical),
            projected_finish_ms=projected,
            trigger_agent_id=trigger_agent,
            trigger_progress_fraction=trigger_fraction,
        )


@dataclass(frozen=True)
class AegisDagSlackTicket:
    envelope: AegisSlackEnvelope
    stage_id: int
    agent_id: str
    stage_width: int
    predicted_critical: bool
    critical_probability: float
    raw_lower_ms: float
    duration_lower_ms: float
    duration_median_ms: float
    duration_upper_ms: float
    request_count: int
    new_prefill_tokens: int
    output_tokens: int

    def to_simulation_fields(self) -> dict[str, Any]:
        return {
            "workflow_slack_ms": self.envelope.lower_ms,
            "aegis_slack_lcb_ms": self.envelope.lower_ms,
            "aegis_slack_q50_ms": self.envelope.median_ms,
            "aegis_slack_q90_ms": self.envelope.upper_ms,
            "aegis_slack_raw_lcb_ms": round(self.raw_lower_ms, 3),
            "aegis_slack_target_coverage": self.envelope.target_coverage,
            "aegis_slack_calibration_id": self.envelope.calibration_id,
            "aegis_slack_calibration_group": self.envelope.calibration_group,
            "aegis_slack_source": self.envelope.source,
            "aegis_slack_stage_id": self.stage_id,
            "aegis_slack_stage_width": self.stage_width,
            "aegis_slack_agent_id": self.agent_id,
            "aegis_slack_predicted_critical": self.predicted_critical,
            "aegis_critical_probability": round(self.critical_probability, 6),
            "aegis_duration_lcb_ms": round(self.duration_lower_ms, 3),
            "aegis_duration_q50_ms": round(self.duration_median_ms, 3),
            "aegis_duration_q90_ms": round(self.duration_upper_ms, 3),
            "aegis_branch_request_count": self.request_count,
            "aegis_branch_new_prefill_tokens": self.new_prefill_tokens,
            "aegis_branch_output_tokens": self.output_tokens,
            "aegis_slack_semantics": "legacy_fixed_dag_trace_token_cost_v2",
            "aegis_slack_uses_future_information": True,
        }


def build_fixed_dag_slack_tickets(
    requests: Iterable[AegisDagRequest],
    *,
    calibration: AegisSlackCalibration,
    critical_threshold_ms: float = 1.0,
) -> dict[str, AegisDagSlackTicket]:
    """Run the legacy/offline fixed-DAG CPM transformation.

    The scalar ``output_tokens`` input is not provenance-safe.  Even when the
    CPM arithmetic is correct, its result must not enter a causal recovery
    snapshot.  Use :func:`aegis_causal_slack.build_causal_slack_tickets` for
    production-facing work.
    """

    if calibration.semantics != "fixed_dag_cpm_slack_v2":
        raise ValueError(
            "fixed-DAG ticket construction requires fixed_dag_cpm_slack_v2"
        )
    request_list = list(requests)
    cost_model = AegisRequestCostModel.from_calibration(calibration)
    request_costs_ms = fixed_dag_request_costs_ms(
        request_list,
        calibration=calibration,
    )
    by_agent: dict[tuple[int, str], list[AegisDagRequest]] = defaultdict(list)
    for request in request_list:
        if not request.request_id:
            raise ValueError("DAG request_id must be non-empty")
        if not request.agent_id:
            raise ValueError("DAG agent_id must be non-empty")
        by_agent[(int(request.stage_id), request.agent_id)].append(request)
    if not by_agent:
        return {}

    duration_by_agent: dict[tuple[int, str], tuple[float, float, float]] = {}
    work_by_agent: dict[tuple[int, str], tuple[int, int, int]] = {}
    request_ids_by_agent: dict[tuple[int, str], list[str]] = {}
    for key, agent_requests in by_agent.items():
        ordered = sorted(agent_requests, key=lambda item: (item.order, item.request_id))
        max_materialized_prompt = 0
        median_ms = 0.0
        new_prefill_tokens_total = 0
        output_tokens_total = 0
        request_ids: list[str] = []
        for request in ordered:
            prompt_tokens = max(int(request.input_tokens), 0)
            new_prefill_tokens = max(prompt_tokens - max_materialized_prompt, 0)
            max_materialized_prompt = max(max_materialized_prompt, prompt_tokens)
            new_prefill_tokens_total += new_prefill_tokens
            output_tokens_total += max(int(request.output_tokens), 0)
            median_ms += request_costs_ms[request.request_id]
            request_ids.append(request.request_id)
        duration_by_agent[key] = (
            median_ms * cost_model.lower_scale,
            median_ms,
            median_ms * cost_model.upper_scale,
        )
        work_by_agent[key] = (
            len(ordered),
            new_prefill_tokens_total,
            output_tokens_total,
        )
        request_ids_by_agent[key] = request_ids

    agents_by_stage: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for key in duration_by_agent:
        agents_by_stage[key[0]].append(key)

    tickets: dict[str, AegisDagSlackTicket] = {}
    threshold = max(float(critical_threshold_ms), 0.0)
    for stage_id, stage_agents in agents_by_stage.items():
        stage_width = len(stage_agents)
        for key in stage_agents:
            duration_lower, duration_median, duration_upper = duration_by_agent[key]
            other_agents = [candidate for candidate in stage_agents if candidate != key]
            if not other_agents:
                raw_lower_ms = median_slack_ms = upper_slack_ms = 0.0
                source = "fixed_dag_singleton_stage_exact_zero"
            else:
                raw_lower_ms = max(
                    max(duration_by_agent[candidate][0] for candidate in other_agents)
                    - duration_upper,
                    0.0,
                )
                median_slack_ms = max(
                    max(duration_by_agent[candidate][1] for candidate in other_agents)
                    - duration_median,
                    0.0,
                )
                upper_slack_ms = max(
                    max(duration_by_agent[candidate][2] for candidate in other_agents)
                    - duration_lower,
                    median_slack_ms,
                )
                source = "fixed_dag_cpm_token_cost_plus_split_conformal"
            calibrated_lower_ms = max(
                raw_lower_ms - calibration.slack_lower_margin_ms, 0.0
            )
            median_slack_ms = max(median_slack_ms, calibrated_lower_ms)
            upper_slack_ms = max(upper_slack_ms, median_slack_ms)
            predicted_critical = median_slack_ms <= threshold
            if not other_agents:
                critical_probability = 0.99
            elif predicted_critical:
                critical_probability = 0.90
            elif calibrated_lower_ms > threshold:
                critical_probability = 0.10
            else:
                critical_probability = 0.50
            envelope = AegisSlackEnvelope(
                lower_ms=round(calibrated_lower_ms, 3),
                median_ms=round(median_slack_ms, 3),
                upper_ms=round(upper_slack_ms, 3),
                target_coverage=calibration.target_lower_coverage,
                calibration_id=calibration.calibration_id,
                calibration_group=f"stage_width={stage_width}",
                source=source,
            )
            ticket = AegisDagSlackTicket(
                envelope=envelope,
                stage_id=stage_id,
                agent_id=key[1],
                stage_width=stage_width,
                predicted_critical=predicted_critical,
                critical_probability=critical_probability,
                raw_lower_ms=raw_lower_ms,
                duration_lower_ms=duration_lower,
                duration_median_ms=duration_median,
                duration_upper_ms=duration_upper,
                request_count=work_by_agent[key][0],
                new_prefill_tokens=work_by_agent[key][1],
                output_tokens=work_by_agent[key][2],
            )
            for request_id in request_ids_by_agent[key]:
                if request_id in tickets:
                    raise ValueError(f"duplicate DAG request_id: {request_id}")
                tickets[request_id] = ticket
    return tickets
