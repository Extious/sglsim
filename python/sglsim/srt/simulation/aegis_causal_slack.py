"""Strictly causal Algorithm-1 contract for AegisKV.

The legacy fixed-DAG helper accepts a scalar ``output_tokens`` value and cannot
prove whether it came from an admission-time limit or a post-hoc trace.  This
module is the production-facing replacement.  It represents only remaining
work at the observation time, requires explicit quantile and calibration
provenance, rejects post-hoc sources, and fails closed when unrevealed agent
work has no calibrated envelope.

The CPM transformation itself is simple.  The important contract is that a
positive Slack lower bound can be published only when every input interval and
the final Slack margin were calibrated on workflow-disjoint data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable


CAUSAL_SLACK_SEMANTICS = "revealed_dag_causal_slack_v1"
FORBIDDEN_PROVENANCE_FRAGMENTS = (
    "actual_output",
    "final_output",
    "ground_truth",
    "posthoc",
    "post_hoc",
    "trace_oracle",
)


class AegisNeedState(str, Enum):
    """Whether a context has a released LLM call at the observation time."""

    ACTIVE = "ACTIVE"
    DEFERRED = "DEFERRED"


class AegisSlackKind(str, Enum):
    """Evidence class for a published Slack value."""

    BOUNDED = "BOUNDED"
    CALIBRATED = "CALIBRATED"
    UNKNOWN = "UNKNOWN"


def _finite_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def _validate_provenance(
    *, source: str, calibration_id: str, uses_future_information: bool
) -> None:
    normalized_source = str(source).strip()
    normalized_calibration = str(calibration_id).strip()
    if not normalized_source or not normalized_calibration:
        raise ValueError("causal estimate requires source and calibration_id")
    if uses_future_information:
        raise ValueError("causal estimate cannot use future information")
    lowered = normalized_source.lower()
    if any(fragment in lowered for fragment in FORBIDDEN_PROVENANCE_FRAGMENTS):
        raise ValueError(f"post-hoc prediction source is not causal: {source!r}")


@dataclass(frozen=True)
class AegisCausalTokenEnvelope:
    """Remaining output-token interval at one causal observation time."""

    requested_max_new_tokens: int
    observed_generated_tokens: int
    remaining_lower_tokens: float
    remaining_median_tokens: float
    remaining_upper_tokens: float
    target_coverage: float
    calibration_id: str
    source: str
    uses_future_information: bool = False

    def __post_init__(self) -> None:
        requested = int(self.requested_max_new_tokens)
        observed = int(self.observed_generated_tokens)
        if requested < 0 or not 0 <= observed <= requested:
            raise ValueError("observed output must lie within the requested limit")
        lower = _finite_nonnegative(
            "remaining_lower_tokens", self.remaining_lower_tokens
        )
        median = _finite_nonnegative(
            "remaining_median_tokens", self.remaining_median_tokens
        )
        upper = _finite_nonnegative(
            "remaining_upper_tokens", self.remaining_upper_tokens
        )
        if not lower <= median <= upper:
            raise ValueError("remaining token quantiles must be ordered")
        remaining_budget = requested - observed
        if upper > remaining_budget + 1e-9:
            raise ValueError("remaining token upper bound exceeds request budget")
        coverage = float(self.target_coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("target_coverage must be strictly between 0 and 1")
        _validate_provenance(
            source=self.source,
            calibration_id=self.calibration_id,
            uses_future_information=self.uses_future_information,
        )


@dataclass(frozen=True)
class AegisCausalDurationEnvelope:
    """Calibrated duration interval for unrevealed turns or tool gaps."""

    lower_ms: float
    median_ms: float
    upper_ms: float
    target_coverage: float
    calibration_id: str
    source: str
    uses_future_information: bool = False

    def __post_init__(self) -> None:
        lower = _finite_nonnegative("lower_ms", self.lower_ms)
        median = _finite_nonnegative("median_ms", self.median_ms)
        upper = _finite_nonnegative("upper_ms", self.upper_ms)
        if not lower <= median <= upper:
            raise ValueError("duration quantiles must be ordered")
        coverage = float(self.target_coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("target_coverage must be strictly between 0 and 1")
        _validate_provenance(
            source=self.source,
            calibration_id=self.calibration_id,
            uses_future_information=self.uses_future_information,
        )

    @classmethod
    def exact_zero(
        cls, *, calibration_id: str, source: str
    ) -> "AegisCausalDurationEnvelope":
        return cls(
            lower_ms=0.0,
            median_ms=0.0,
            upper_ms=0.0,
            target_coverage=0.999999,
            calibration_id=calibration_id,
            source=source,
        )


@dataclass(frozen=True)
class AegisCausalRequestWork:
    request_id: str
    stage_id: int
    agent_id: str
    order: int
    setup_pending: bool
    remaining_prefill_tokens: int
    output: AegisCausalTokenEnvelope

    def __post_init__(self) -> None:
        if not str(self.request_id).strip() or not str(self.agent_id).strip():
            raise ValueError("request_id and agent_id must be nonempty")
        if int(self.stage_id) < 0 or int(self.order) < 0:
            raise ValueError("stage_id and order must be nonnegative")
        if int(self.remaining_prefill_tokens) < 0:
            raise ValueError("remaining_prefill_tokens must be nonnegative")


@dataclass(frozen=True)
class AegisCausalAgentWork:
    stage_id: int
    agent_id: str
    requests: tuple[AegisCausalRequestWork, ...]
    unrevealed_tail: AegisCausalDurationEnvelope | None
    need_state: AegisNeedState = AegisNeedState.ACTIVE

    def __post_init__(self) -> None:
        if int(self.stage_id) < 0 or not str(self.agent_id).strip():
            raise ValueError("agent stage_id and agent_id must be valid")
        try:
            normalized_need_state = AegisNeedState(self.need_state)
        except ValueError as exc:
            raise ValueError("agent need_state must be ACTIVE or DEFERRED") from exc
        object.__setattr__(self, "need_state", normalized_need_state)
        request_ids: set[str] = set()
        orders: set[int] = set()
        for request in self.requests:
            if int(request.stage_id) != int(self.stage_id):
                raise ValueError("request stage does not match agent stage")
            if str(request.agent_id) != str(self.agent_id):
                raise ValueError("request agent does not match owner")
            if request.request_id in request_ids or request.order in orders:
                raise ValueError("agent requests require unique IDs and orders")
            request_ids.add(request.request_id)
            orders.add(request.order)


@dataclass(frozen=True)
class AegisCausalSlackCalibration:
    calibration_id: str
    target_lower_coverage: float
    slack_lower_margin_ms: float
    request_setup_ms: float
    prefill_ms_per_token: float
    decode_ms_per_token: float
    source: str
    workflow_disjoint: bool
    evaluation_gate_passed: bool
    semantics: str = CAUSAL_SLACK_SEMANTICS

    def __post_init__(self) -> None:
        if self.semantics != CAUSAL_SLACK_SEMANTICS:
            raise ValueError("causal Slack calibration has incompatible semantics")
        if not self.workflow_disjoint:
            raise ValueError("causal Slack calibration must be workflow-disjoint")
        if not self.evaluation_gate_passed:
            raise ValueError("causal Slack calibration gate has not passed")
        coverage = float(self.target_lower_coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("target_lower_coverage must be between 0 and 1")
        for name in (
            "slack_lower_margin_ms",
            "request_setup_ms",
            "prefill_ms_per_token",
            "decode_ms_per_token",
        ):
            _finite_nonnegative(name, getattr(self, name))
        _validate_provenance(
            source=self.source,
            calibration_id=self.calibration_id,
            uses_future_information=False,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AegisCausalSlackCalibration":
        cost = dict(raw.get("request_cost_model") or {})
        gate = dict(raw.get("evaluation_gate") or {})
        return cls(
            calibration_id=str(raw.get("calibration_id") or ""),
            target_lower_coverage=float(raw.get("target_lower_coverage", 0.0)),
            slack_lower_margin_ms=float(raw.get("slack_lower_margin_ms", 0.0)),
            request_setup_ms=float(cost.get("request_setup_ms", 0.0)),
            prefill_ms_per_token=float(cost.get("prefill_ms_per_token", 0.0)),
            decode_ms_per_token=float(cost.get("decode_ms_per_token", 0.0)),
            source=str(raw.get("source") or ""),
            workflow_disjoint=bool(raw.get("workflow_disjoint", False)),
            evaluation_gate_passed=bool(gate.get("passed", False)),
            semantics=str(raw.get("semantics") or ""),
        )


@dataclass(frozen=True)
class AegisCausalSlackTicket:
    stage_id: int
    agent_id: str
    stage_width: int
    slack_lower_ms: float
    slack_median_ms: float
    slack_upper_ms: float
    raw_lower_ms: float
    duration_lower_ms: float
    duration_median_ms: float
    duration_upper_ms: float
    target_coverage: float
    calibration_id: str
    source: str
    usable_slack: bool
    input_provenance_sha256: str
    need_state: AegisNeedState
    slack_kind: AegisSlackKind
    uses_future_information: bool = False

    def to_simulation_fields(self) -> dict[str, Any]:
        return {
            "workflow_slack_ms": self.slack_lower_ms,
            "aegis_slack_lcb_ms": self.slack_lower_ms,
            "aegis_slack_q50_ms": self.slack_median_ms,
            "aegis_slack_q90_ms": self.slack_upper_ms,
            "aegis_slack_raw_lcb_ms": self.raw_lower_ms,
            "aegis_slack_target_coverage": self.target_coverage,
            "aegis_slack_calibration_id": self.calibration_id,
            "aegis_slack_source": self.source,
            "aegis_need_state": self.need_state.value,
            "aegis_slack_kind": self.slack_kind.value,
            "aegis_slack_semantics": CAUSAL_SLACK_SEMANTICS,
            "aegis_slack_uses_future_information": False,
            "aegis_slack_usable": self.usable_slack,
            "aegis_slack_input_provenance_sha256": (
                self.input_provenance_sha256
            ),
            "aegis_slack_stage_id": self.stage_id,
            "aegis_slack_stage_width": self.stage_width,
            "aegis_slack_agent_id": self.agent_id,
            "aegis_duration_lcb_ms": self.duration_lower_ms,
            "aegis_duration_q50_ms": self.duration_median_ms,
            "aegis_duration_q90_ms": self.duration_upper_ms,
        }


def _request_interval_ms(
    request: AegisCausalRequestWork,
    calibration: AegisCausalSlackCalibration,
) -> tuple[float, float, float]:
    base = calibration.request_setup_ms if request.setup_pending else 0.0
    base += (
        calibration.prefill_ms_per_token
        * int(request.remaining_prefill_tokens)
    )
    output = request.output
    return (
        base + calibration.decode_ms_per_token * output.remaining_lower_tokens,
        base + calibration.decode_ms_per_token * output.remaining_median_tokens,
        base + calibration.decode_ms_per_token * output.remaining_upper_tokens,
    )


def _agent_interval_ms(
    agent: AegisCausalAgentWork,
    calibration: AegisCausalSlackCalibration,
) -> tuple[float, float, float] | None:
    if agent.unrevealed_tail is None:
        return None
    lower = agent.unrevealed_tail.lower_ms
    median = agent.unrevealed_tail.median_ms
    upper = agent.unrevealed_tail.upper_ms
    for request in sorted(
        agent.requests, key=lambda item: (item.order, item.request_id)
    ):
        request_lower, request_median, request_upper = _request_interval_ms(
            request, calibration
        )
        lower += request_lower
        median += request_median
        upper += request_upper
    return (lower, median, upper)


def _provenance_sha256(
    agent: AegisCausalAgentWork,
    calibration: AegisCausalSlackCalibration,
) -> str:
    payload = json.dumps(
        {"agent": asdict(agent), "calibration": asdict(calibration)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_causal_slack_tickets(
    agents: Iterable[AegisCausalAgentWork],
    *,
    calibration: AegisCausalSlackCalibration,
) -> dict[tuple[int, str], AegisCausalSlackTicket]:
    """Transform causal agent remaining-work intervals into Slack tickets.

    A stage containing any agent without a calibrated unrevealed-tail envelope
    fails closed to zero Slack for every sibling.  This prevents a missing
    future-turn model on one branch from making another branch look safely
    non-critical.
    """

    by_stage: dict[int, list[AegisCausalAgentWork]] = {}
    seen: set[tuple[int, str]] = set()
    for agent in agents:
        key = (int(agent.stage_id), str(agent.agent_id))
        if key in seen:
            raise ValueError(f"duplicate causal agent: {key}")
        seen.add(key)
        by_stage.setdefault(key[0], []).append(agent)

    tickets: dict[tuple[int, str], AegisCausalSlackTicket] = {}
    for stage_id, stage_agents in sorted(by_stage.items()):
        stage_agents.sort(key=lambda item: item.agent_id)
        stage_width = len(stage_agents)
        intervals = {
            agent.agent_id: _agent_interval_ms(agent, calibration)
            for agent in stage_agents
        }
        stage_usable = all(value is not None for value in intervals.values())
        for agent in stage_agents:
            interval = intervals[agent.agent_id]
            known_interval = interval or (0.0, 0.0, 0.0)
            lower, median, upper = known_interval
            other_intervals = [
                value
                for other_id, value in intervals.items()
                if other_id != agent.agent_id and value is not None
            ]
            if not stage_usable:
                raw_lower = slack_lower = slack_median = slack_upper = 0.0
                source = "fail_closed_unrevealed_agent_tail"
                slack_kind = AegisSlackKind.UNKNOWN
            elif not other_intervals:
                raw_lower = slack_lower = slack_median = slack_upper = 0.0
                source = "causal_singleton_stage_exact_zero"
                slack_kind = AegisSlackKind.BOUNDED
            else:
                raw_lower = max(
                    max(value[0] for value in other_intervals) - upper,
                    0.0,
                )
                slack_lower = max(
                    raw_lower - calibration.slack_lower_margin_ms,
                    0.0,
                )
                slack_median = max(
                    max(value[1] for value in other_intervals) - median,
                    slack_lower,
                )
                slack_upper = max(
                    max(value[2] for value in other_intervals) - lower,
                    slack_median,
                )
                source = "causal_remaining_quantiles_plus_split_conformal"
                slack_kind = AegisSlackKind.CALIBRATED
            tickets[(stage_id, agent.agent_id)] = AegisCausalSlackTicket(
                stage_id=stage_id,
                agent_id=agent.agent_id,
                stage_width=stage_width,
                slack_lower_ms=round(slack_lower, 6),
                slack_median_ms=round(slack_median, 6),
                slack_upper_ms=round(slack_upper, 6),
                raw_lower_ms=round(raw_lower, 6),
                duration_lower_ms=round(lower, 6),
                duration_median_ms=round(median, 6),
                duration_upper_ms=round(upper, 6),
                target_coverage=calibration.target_lower_coverage,
                calibration_id=calibration.calibration_id,
                source=source,
                usable_slack=slack_kind is not AegisSlackKind.UNKNOWN,
                input_provenance_sha256=_provenance_sha256(
                    agent, calibration
                ),
                need_state=agent.need_state,
                slack_kind=slack_kind,
            )
    return tickets
