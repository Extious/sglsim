from __future__ import annotations

import pytest

from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.aegis_residual_dag import (
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
    resolve_slack_for_design,
)


def _graph(**overrides: object) -> VisibleResidualDag:
    values: dict[str, object] = {
        "workflow_id": "workflow-0",
        "graph_version": "visible-v3",
        "observed_at_ms": 17.0,
        "nodes": (
            ResidualDagNode("fork", "control", 0.0),
            ResidualDagNode("short", "agent", 2.0, active=True, agent_id="a"),
            ResidualDagNode("long", "agent", 1.0, active=True, agent_id="b"),
            ResidualDagNode("tail", "agent", 4.0, agent_id="b"),
            ResidualDagNode("join", "control", 0.0),
        ),
        "edges": (
            ("fork", "short"),
            ("fork", "long"),
            ("short", "join"),
            ("long", "tail"),
            ("tail", "join"),
        ),
        "next_join_subgraph_complete": True,
    }
    values.update(overrides)
    return VisibleResidualDag(**values)


def test_visible_residual_dag_runs_standard_forward_backward_cpm() -> None:
    tickets = build_residual_dag_slack_tickets(_graph())

    assert tickets["short"].earliest_start_ms == 0.0
    assert tickets["short"].latest_start_ms == 3.0
    assert tickets["short"].slack_ms == 3.0
    assert tickets["long"].slack_ms == 0.0
    assert tickets["long"].critical_path_ms == 5.0
    fields = tickets["short"].to_simulation_fields()
    assert fields["aegis_slack_residual_dag_visible"] is True
    assert fields["aegis_slack_metadata_only"] is True
    assert fields["aegis_slack_uses_future_information"] is False


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"next_join_subgraph_complete": False}, "next_join_subgraph_incomplete"),
        ({"metadata_complete": False}, "residual_dag_metadata_incomplete"),
        ({"uses_future_information": True}, "future_information_forbidden"),
    ],
)
def test_incomplete_or_noncausal_graph_fails_all_active_slack_to_zero(
    overrides: dict[str, object], reason: str
) -> None:
    tickets = build_residual_dag_slack_tickets(_graph(**overrides))

    assert {ticket.slack_ms for ticket in tickets.values()} == {0.0}
    assert {ticket.usable_slack for ticket in tickets.values()} == {False}
    assert {ticket.reason for ticket in tickets.values()} == {reason}


def test_missing_visible_agent_prediction_fails_closed() -> None:
    graph = _graph(
        nodes=(
            ResidualDagNode("a", "agent", None, active=True, agent_id="a"),
            ResidualDagNode("b", "agent", 5.0, active=True, agent_id="b"),
        ),
        edges=(),
    )
    tickets = build_residual_dag_slack_tickets(graph)

    assert all(ticket.slack_ms == 0.0 for ticket in tickets.values())
    assert all(
        ticket.reason == "visible_agent_prediction_missing"
        for ticket in tickets.values()
    )


def test_cycle_is_rejected_instead_of_publishing_slack() -> None:
    graph = _graph(
        nodes=(
            ResidualDagNode("a", "agent", 1.0, active=True, agent_id="a"),
            ResidualDagNode("b", "agent", 1.0, agent_id="b"),
        ),
        edges=(("a", "b"), ("b", "a")),
    )
    with pytest.raises(ValueError, match="must be a DAG"):
        build_residual_dag_slack_tickets(graph)


def test_current_design_slack_resolver_rejects_unproven_positive_scalar() -> None:
    value, reason = resolve_slack_for_design(
        {"workflow_slack_ms": 999.0}, design_semantics=DESIGN_SEMANTICS
    )

    assert value == 0.0
    assert reason.startswith("missing_or_invalid_")


def test_current_design_slack_resolver_accepts_only_complete_ticket_fields() -> None:
    fields = build_residual_dag_slack_tickets(_graph())["short"].to_simulation_fields()
    value, reason = resolve_slack_for_design(fields, design_semantics=DESIGN_SEMANTICS)

    assert value == 3.0
    assert reason == "visible_residual_dag_cpm"

    tampered = {**fields, "aegis_slack_uses_future_information": True}
    assert (
        resolve_slack_for_design(tampered, design_semantics=DESIGN_SEMANTICS)[0]
        == 0.0
    )


def test_legacy_semantics_preserve_the_existing_scalar_contract() -> None:
    value, reason = resolve_slack_for_design(
        {"workflow_slack_ms": 7.5}, design_semantics="legacy_l1_l2_v1"
    )
    assert (value, reason) == (7.5, "legacy_scalar")
