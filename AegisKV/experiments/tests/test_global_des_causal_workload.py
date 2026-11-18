from __future__ import annotations

import copy

import pytest

from AegisKV.experiments.src.global_des_causal_workload import (
    CONTROLLED_OVERLAY_MODE,
    CausalWorkloadError,
    SEQUENTIAL_ZERO_MODE,
    enrich_causal_workload,
)
from AegisKV.experiments.src.global_des_config import load_suite
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.aegis_residual_dag import resolve_slack_for_design


def _record(index: int = 0) -> dict:
    return {
        "rid": f"request-{index}",
        "workflow": "observed-sequential-session",
        "session": "session",
        "provider": "tracelab",
        "round_index": index,
        "arrival_ns": index * 1_000_000,
        "input_tokens": 128,
        "output_tokens": 16,
        "cached_prefix_tokens": 64,
        "prefix_key": "session-prefix",
        "source_round_pk": index,
        "source_round_id": str(index),
    }


def _contract(group: str) -> dict:
    return copy.deepcopy(
        load_suite().group_map[group].request.fixed["workload"][
            "causal_slack_contract"
        ]
    )


def test_tracelab_session_gets_valid_exact_zero_ticket_without_future_data() -> None:
    records, audit = enrich_causal_workload(
        [_record()],
        group="P1",
        seed=2026082900,
        context_tokens=16_384,
        contract=_contract("P1"),
    )
    assert audit["mode"] == SEQUENTIAL_ZERO_MODE
    assert audit["background"]["nonzero_ticket_count"] == 0
    simulation = records[0]["simulation"]
    slack, reason = resolve_slack_for_design(
        simulation, design_semantics=DESIGN_SEMANTICS
    )
    assert slack == 0.0
    assert reason == "visible_residual_dag_cpm"
    assert simulation["aegis_slack_uses_future_information"] is False
    assert simulation["aegis_slack_positive_value_source"] == "none"


def test_p4_appends_modeled_visible_fork_join_with_positive_cpm_slack() -> None:
    records, audit = enrich_causal_workload(
        [_record()],
        group="P4",
        seed=2026082900,
        context_tokens=16_384,
        contract=_contract("P4"),
    )
    assert audit["mode"] == CONTROLLED_OVERLAY_MODE
    assert audit["overlay"]["workflow_count"] == 8
    assert audit["overlay"]["branch_count"] == 32
    assert audit["overlay"]["nonzero_ticket_count"] > 0
    assert audit["overlay"]["evidence_class"] == (
        "modeled_mechanism_workload_not_observed_trace"
    )
    assert len(records) == 33
    foreground = records[:32]
    assert all(row["provider"] == "controlled_visible_fork_join" for row in foreground)
    first_workflow = [
        row for row in foreground if row["workflow"] == foreground[0]["workflow"]
    ]
    assert len({row["agent_id"] for row in first_workflow}) == 4
    assert {row["worker_seq"] for row in first_workflow} == {1}
    values = []
    for row in foreground:
        slack, reason = resolve_slack_for_design(
            row["simulation"], design_semantics=DESIGN_SEMANTICS
        )
        assert reason == "visible_residual_dag_cpm"
        assert row["simulation"]["aegis_slack_uses_future_information"] is False
        values.append(slack)
    assert any(value > 0.0 for value in values)
    assert records[-1]["simulation"]["workflow_slack_ms"] == 0.0


def test_p4_predictor_cannot_claim_hardware_calibration_or_use_completion() -> None:
    contract = _contract("P4")
    contract["admission_predictor"]["hardware_calibrated"] = True
    with pytest.raises(CausalWorkloadError, match="modeled and uncalibrated"):
        enrich_causal_workload(
            [_record()],
            group="P4",
            seed=2026082900,
            context_tokens=16_384,
            contract=contract,
        )
    contract = _contract("P4")
    contract["admission_predictor"]["uses_simulated_completion"] = True
    with pytest.raises(CausalWorkloadError, match="simulated completion"):
        enrich_causal_workload(
            [_record()],
            group="P4",
            seed=2026082900,
            context_tokens=16_384,
            contract=contract,
        )


def test_sequential_contract_cannot_enable_positive_slack() -> None:
    contract = _contract("P1")
    contract["background"]["positive_slack_forbidden"] = False
    with pytest.raises(CausalWorkloadError, match="forbid positive Slack"):
        enrich_causal_workload(
            [_record()],
            group="P1",
            seed=2026082900,
            context_tokens=16_384,
            contract=contract,
        )
