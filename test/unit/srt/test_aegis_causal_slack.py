from __future__ import annotations

import pytest

from sglsim.srt.simulation.aegis_causal_slack import (
    CAUSAL_SLACK_SEMANTICS,
    AegisCausalAgentWork,
    AegisCausalDurationEnvelope,
    AegisCausalRequestWork,
    AegisCausalSlackCalibration,
    AegisCausalTokenEnvelope,
    AegisNeedState,
    AegisSlackKind,
    build_causal_slack_tickets,
)
from sglsim.srt.simulation.aegis_slack import (
    AegisDagRequest,
    AegisSlackCalibration,
    build_fixed_dag_slack_tickets,
)


def _calibration(**overrides: object) -> AegisCausalSlackCalibration:
    values: dict[str, object] = {
        "calibration_id": "causal-unit-v1",
        "target_lower_coverage": 0.9,
        "slack_lower_margin_ms": 20.0,
        "request_setup_ms": 0.0,
        "prefill_ms_per_token": 1.0,
        "decode_ms_per_token": 1.0,
        "source": "workflow_disjoint_unit_calibration",
        "workflow_disjoint": True,
        "evaluation_gate_passed": True,
    }
    values.update(overrides)
    return AegisCausalSlackCalibration(**values)


def _tokens(
    *,
    requested: int = 100,
    observed: int = 0,
    lower: float = 0.0,
    median: float = 0.0,
    upper: float = 0.0,
    source: str = "causal_token_quantile_model",
    uses_future: bool = False,
) -> AegisCausalTokenEnvelope:
    return AegisCausalTokenEnvelope(
        requested_max_new_tokens=requested,
        observed_generated_tokens=observed,
        remaining_lower_tokens=lower,
        remaining_median_tokens=median,
        remaining_upper_tokens=upper,
        target_coverage=0.9,
        calibration_id="token-unit-v1",
        source=source,
        uses_future_information=uses_future,
    )


def _zero_tail() -> AegisCausalDurationEnvelope:
    return AegisCausalDurationEnvelope.exact_zero(
        calibration_id="tail-unit-v1",
        source="runtime_declared_no_unrevealed_tail",
    )


def _agent(
    agent_id: str,
    *,
    prefill: int,
    output: AegisCausalTokenEnvelope | None = None,
    tail: AegisCausalDurationEnvelope | None = None,
    need_state: AegisNeedState = AegisNeedState.ACTIVE,
) -> AegisCausalAgentWork:
    return AegisCausalAgentWork(
        stage_id=2,
        agent_id=agent_id,
        requests=(
            AegisCausalRequestWork(
                request_id=f"request-{agent_id}",
                stage_id=2,
                agent_id=agent_id,
                order=0,
                setup_pending=False,
                remaining_prefill_tokens=prefill,
                output=output or _tokens(),
            ),
        ),
        unrevealed_tail=_zero_tail() if tail is None else tail,
        need_state=need_state,
    )


@pytest.mark.parametrize(
    "source,uses_future,match",
    [
        ("trace_oracle_output", False, "post-hoc"),
        ("actual_output_length", False, "post-hoc"),
        ("causal_token_quantile_model", True, "future information"),
    ],
)
def test_token_envelope_rejects_noncausal_provenance(
    source: str, uses_future: bool, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _tokens(source=source, uses_future=uses_future)


def test_token_envelope_enforces_budget_and_quantile_order() -> None:
    with pytest.raises(ValueError, match="ordered"):
        _tokens(lower=3.0, median=2.0, upper=4.0)
    with pytest.raises(ValueError, match="exceeds request budget"):
        _tokens(requested=100, observed=90, upper=11.0)


def test_causal_cpm_uses_interval_endpoints_and_direct_slack_margin() -> None:
    short = _agent(
        "short",
        prefill=100,
        output=_tokens(lower=10.0, median=20.0, upper=30.0),
    )
    long = _agent("long", prefill=300)

    tickets = build_causal_slack_tickets(
        [short, long], calibration=_calibration()
    )

    short_ticket = tickets[(2, "short")]
    assert short_ticket.duration_lower_ms == 110.0
    assert short_ticket.duration_median_ms == 120.0
    assert short_ticket.duration_upper_ms == 130.0
    assert short_ticket.raw_lower_ms == 170.0
    assert short_ticket.slack_lower_ms == 150.0
    assert short_ticket.slack_median_ms == 180.0
    assert short_ticket.slack_upper_ms == 190.0
    assert tickets[(2, "long")].slack_lower_ms == 0.0


def test_remaining_cost_does_not_recharge_observed_output_tokens() -> None:
    active = _agent(
        "active",
        prefill=0,
        output=_tokens(
            requested=100,
            observed=80,
            lower=5.0,
            median=10.0,
            upper=15.0,
        ),
    )

    ticket = build_causal_slack_tickets(
        [active], calibration=_calibration(slack_lower_margin_ms=0.0)
    )[(2, "active")]

    assert ticket.duration_lower_ms == 5.0
    assert ticket.duration_median_ms == 10.0
    assert ticket.duration_upper_ms == 15.0
    assert ticket.slack_lower_ms == 0.0


def test_missing_unrevealed_tail_fails_closed_for_all_stage_siblings() -> None:
    unknown = AegisCausalAgentWork(
        stage_id=2,
        agent_id="unknown",
        requests=_agent("unknown", prefill=10).requests,
        unrevealed_tail=None,
    )
    known = _agent("known", prefill=500)

    tickets = build_causal_slack_tickets(
        [unknown, known], calibration=_calibration()
    )

    assert all(ticket.slack_lower_ms == 0.0 for ticket in tickets.values())
    assert all(ticket.usable_slack is False for ticket in tickets.values())
    assert {
        ticket.source for ticket in tickets.values()
    } == {"fail_closed_unrevealed_agent_tail"}
    assert {
        ticket.slack_kind for ticket in tickets.values()
    } == {AegisSlackKind.UNKNOWN}


def test_causal_ticket_publishes_auditable_provenance() -> None:
    tickets = build_causal_slack_tickets(
        [_agent("a", prefill=100), _agent("b", prefill=300)],
        calibration=_calibration(),
    )

    fields = tickets[(2, "a")].to_simulation_fields()
    assert fields["aegis_slack_semantics"] == CAUSAL_SLACK_SEMANTICS
    assert fields["aegis_slack_uses_future_information"] is False
    assert fields["aegis_slack_usable"] is True
    assert fields["aegis_need_state"] == "ACTIVE"
    assert fields["aegis_slack_kind"] == "CALIBRATED"
    assert len(fields["aegis_slack_input_provenance_sha256"]) == 64


def test_deferred_need_state_and_exact_zero_kind_are_not_conflated() -> None:
    ticket = build_causal_slack_tickets(
        [
            _agent(
                "waiting",
                prefill=0,
                need_state=AegisNeedState.DEFERRED,
            )
        ],
        calibration=_calibration(),
    )[(2, "waiting")]

    fields = ticket.to_simulation_fields()
    assert ticket.slack_lower_ms == 0.0
    assert ticket.usable_slack is True
    assert fields["aegis_need_state"] == "DEFERRED"
    assert fields["aegis_slack_kind"] == "BOUNDED"


def test_agent_rejects_unknown_need_state() -> None:
    with pytest.raises(ValueError, match="need_state"):
        AegisCausalAgentWork(
            stage_id=2,
            agent_id="bad",
            requests=(),
            unrevealed_tail=_zero_tail(),
            need_state="MAYBE",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"workflow_disjoint": False}, "workflow-disjoint"),
        ({"evaluation_gate_passed": False}, "gate has not passed"),
        ({"source": "posthoc_duration_fit"}, "post-hoc"),
        ({"semantics": "fixed_dag_cpm_slack_v2"}, "incompatible semantics"),
    ],
)
def test_causal_calibration_rejects_unsafe_artifacts(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _calibration(**overrides)


def test_legacy_ticket_is_explicitly_noncausal() -> None:
    legacy_calibration = AegisSlackCalibration.from_dict(
        {
            "schema_version": 2,
            "semantics": "fixed_dag_cpm_slack_v2",
            "calibration_id": "legacy-unit-v2",
            "target_lower_coverage": 0.9,
            "request_cost_model": {
                "request_setup_ms": 0.0,
                "prefill_ms_per_token": 1.0,
                "decode_ms_per_token": 1.0,
                "lower_scale": 0.9,
                "upper_scale": 1.1,
            },
            "evaluation_gate": {"passed": True},
        }
    )
    ticket = build_fixed_dag_slack_tickets(
        [AegisDagRequest("legacy", 1, "agent", 0, 10, 20)],
        calibration=legacy_calibration,
    )["legacy"]

    fields = ticket.to_simulation_fields()
    assert fields["aegis_slack_uses_future_information"] is True
    assert fields["aegis_slack_semantics"].startswith("legacy_")
