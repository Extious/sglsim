import json

import pytest

from sglsim.srt.simulation.aegis_slack import (
    AegisDagRequest,
    AegisOnlineProgressTracker,
    AegisSlackCalibration,
    build_fixed_dag_slack_tickets,
    calibrated_lower_margin_s,
    empirical_lower_bound_s,
    turn_bucket,
)
from sglsim.srt.simulation.types import RequestStats


def test_conformal_margin_is_finite_sample_and_never_aggressive():
    margin = calibrated_lower_margin_s(
        actual_s=[10.0, 20.0, 30.0, 40.0],
        raw_lower_s=[12.0, 18.0, 31.0, 35.0],
        target_coverage=0.75,
    )
    assert margin == 2.0
    assert calibrated_lower_margin_s(
        actual_s=[10.0, 20.0],
        raw_lower_s=[1.0, 2.0],
        target_coverage=0.5,
    ) == 0.0


def test_empirical_fallback_and_turn_buckets_are_conservative():
    assert empirical_lower_bound_s(range(1, 101), target_coverage=0.90) == 10.0
    assert [turn_bucket(value) for value in (1, 5, 6, 9, 10, 40)] == [
        "1",
        "5",
        "6-9",
        "6-9",
        "10+",
        "10+",
    ]


def test_calibration_applies_role_margin_and_orders_interval():
    calibration = AegisSlackCalibration.from_dict(
        {
            "calibration_id": "unit-v1",
            "target_lower_coverage": 0.9,
            "global_lower_margin_s": 3.0,
            "role_lower_margins_s": {"role=subagent": 1.5},
            "fallback_lower_bounds_s": {"global": 2.0},
            "evaluation_gate": {"passed": True},
        }
    )
    envelope = calibration.calibrated_prediction(
        raw_lower_s=10.0,
        raw_median_s=7.0,
        raw_upper_s=8.0,
        role="subagent",
    )
    assert envelope.lower_ms == 8500.0
    assert envelope.median_ms == 8500.0
    assert envelope.upper_ms == 8500.0
    assert envelope.calibration_group == "role=subagent"


def test_fallback_is_specific_then_fail_closed():
    calibration = AegisSlackCalibration.from_dict(
        {
            "calibration_id": "unit-v1",
            "target_lower_coverage": 0.9,
            "global_lower_margin_s": 1.0,
            "role_lower_margins_s": {},
            "fallback_lower_bounds_s": {
                "role=subagent|turn=2": 4.0,
                "role=subagent": 2.0,
            },
            "evaluation_gate": {"passed": True},
        }
    )
    assert calibration.fallback_prediction(
        role="subagent", turns_completed=2
    ).lower_ms == 4000.0
    assert calibration.fallback_prediction(
        role="subagent", turns_completed=8
    ).lower_ms == 2000.0
    missing = calibration.fallback_prediction(role="main_phase", turns_completed=1)
    assert missing.lower_ms == 0.0
    assert missing.source == "fail_closed_zero_slack"


def test_loader_rejects_unpassed_artifact(tmp_path):
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "calibration_id": "bad-v1",
                "target_lower_coverage": 0.9,
                "evaluation_gate": {"passed": False},
            }
        )
    )
    with pytest.raises(ValueError, match="gate did not pass"):
        AegisSlackCalibration.load(artifact)


def _cpm_calibration(**overrides):
    raw = {
        "schema_version": 2,
        "semantics": "fixed_dag_cpm_slack_v2",
        "calibration_id": "unit-cpm-v2",
        "target_lower_coverage": 0.9,
        "slack_lower_margin_ms": 0.0,
        "request_cost_model": {
            "request_setup_ms": 0.0,
            "prefill_ms_per_token": 1.0,
            "decode_ms_per_token": 0.0,
            "lower_scale": 0.9,
            "upper_scale": 1.1,
        },
        "evaluation_gate": {"passed": True},
    }
    raw.update(overrides)
    return AegisSlackCalibration.from_dict(raw)


def test_fixed_dag_singleton_stage_has_exact_zero_slack():
    tickets = build_fixed_dag_slack_tickets(
        [AegisDagRequest("r0", 1, "main", 0, 100, 0)],
        calibration=_cpm_calibration(),
    )
    ticket = tickets["r0"]
    assert ticket.envelope.lower_ms == 0.0
    assert ticket.envelope.median_ms == 0.0
    assert ticket.envelope.upper_ms == 0.0
    assert ticket.predicted_critical
    assert ticket.envelope.source == "fixed_dag_singleton_stage_exact_zero"


def test_fixed_dag_cpm_marks_only_short_parallel_branch_with_slack():
    tickets = build_fixed_dag_slack_tickets(
        [
            AegisDagRequest("short", 2, "a", 0, 100, 0),
            AegisDagRequest("long", 2, "b", 0, 300, 0),
        ],
        calibration=_cpm_calibration(),
    )
    assert tickets["long"].envelope.median_ms == 0.0
    assert tickets["long"].predicted_critical
    assert tickets["short"].envelope.lower_ms == 160.0
    assert tickets["short"].envelope.median_ms == 200.0
    assert tickets["short"].envelope.upper_ms == 240.0
    assert not tickets["short"].predicted_critical


def test_fixed_dag_costs_only_stable_prefix_growth_and_applies_margin():
    calibration = _cpm_calibration(slack_lower_margin_ms=20.0)
    tickets = build_fixed_dag_slack_tickets(
        [
            AegisDagRequest("a0", 2, "a", 0, 100, 0),
            AegisDagRequest("a1", 2, "a", 1, 150, 0),
            AegisDagRequest("b0", 2, "b", 0, 400, 0),
        ],
        calibration=calibration,
    )
    # Agent a costs 100 + only 50 new prefix tokens, not 100 + 150.
    assert tickets["a0"].duration_median_ms == 150.0
    assert tickets["a0"].request_count == 2
    assert tickets["a0"].new_prefill_tokens == 150
    assert tickets["a0"].output_tokens == 0
    assert tickets["a0"].raw_lower_ms == 195.0
    assert tickets["a0"].envelope.lower_ms == 175.0
    assert tickets["a0"] == tickets["a1"]


def test_online_progress_upgrade_is_causal_and_replaces_static_window():
    tracker = AegisOnlineProgressTracker(
        request_costs_by_agent={
            "static": [(f"a{index}", 1.0) for index in range(4)],
            "lagging": [(f"b{index}", 1.0) for index in range(4)],
        },
        static_critical_agents={"static"},
    )
    for index, elapsed_ms in enumerate((2.0, 4.0, 6.0, 8.0)):
        tracker.observe_completion(
            request_id=f"a{index}", elapsed_ms=elapsed_ms
        )
    for index, elapsed_ms in enumerate((10.0, 20.0)):
        decision = tracker.observe_completion(
            request_id=f"b{index}", elapsed_ms=elapsed_ms
        )
        assert "lagging" not in decision.active_critical_agents

    decision = tracker.observe_completion(request_id="b2", elapsed_ms=30.0)
    assert decision.trigger_agent_id == "lagging"
    assert decision.trigger_progress_fraction == 0.75
    assert decision.active_critical_agents == {"lagging"}
    assert tracker.trigger_elapsed_ms == {"lagging": 30.0}

    tracker.observe_completion(request_id="b3", elapsed_ms=40.0)
    assert tracker.ever_dynamically_critical_agents == {"lagging"}
    assert tracker.final_critical_agents == {"lagging"}


def test_online_progress_uses_last_causal_sample_for_short_branch() -> None:
    tracker = AegisOnlineProgressTracker(
        request_costs_by_agent={
            "static": [(f"a{index}", 1.0) for index in range(3)],
            "lagging": [(f"b{index}", 1.0) for index in range(3)],
        },
        static_critical_agents={"static"},
        minimum_progress_fraction=0.75,
    )
    for index, elapsed_ms in enumerate((2.0, 4.0, 6.0)):
        tracker.observe_completion(request_id=f"a{index}", elapsed_ms=elapsed_ms)

    first = tracker.observe_completion(request_id="b0", elapsed_ms=10.0)
    assert first.trigger_agent_id is None
    second = tracker.observe_completion(request_id="b1", elapsed_ms=20.0)

    assert second.trigger_agent_id == "lagging"
    assert second.trigger_progress_fraction == pytest.approx(2 / 3)
    assert second.active_critical_agents == {"lagging"}


def test_request_stats_persist_algorithm1_provenance():
    stats = RequestStats(
        aegis_slack_lcb_ms=1234.0,
        aegis_slack_target_coverage=0.9,
        aegis_slack_calibration_id="alg1-v1",
        aegis_slack_source="split_conformal",
        aegis_slack_turns_completed=3,
    )
    payload = stats.to_sim_stats_dict()
    assert payload["aegis_slack_lcb_ms"] == 1234.0
    assert payload["aegis_slack_target_coverage"] == 0.9
    assert payload["aegis_slack_calibration_id"] == "alg1-v1"
    assert payload["aegis_slack_source"] == "split_conformal"
    assert payload["aegis_slack_turns_completed"] == 3
