import itertools

import pytest

from sglsim.srt.simulation.aegis_bounded_slack import (
    BOUNDED_SLACK_SEMANTICS,
    AegisHardDurationBound,
    AegisProofScopeKind,
    AegisResidualDag,
    AegisResidualEdge,
    AegisResidualNode,
    build_bounded_slack_tickets,
)
from sglsim.srt.simulation.aegis_causal_slack import (
    AegisNeedState,
    AegisSlackKind,
)


def _bound(
    lower_ms: float,
    upper_ms: float,
    *,
    graph_version: str = "graph-v1",
    source: str = "application_contract_plus_reserved_service",
) -> AegisHardDurationBound:
    return AegisHardDurationBound(
        lower_ms=lower_ms,
        upper_ms=upper_ms,
        graph_version=graph_version,
        service_envelope_version="service-v3",
        bound_id=f"bound-{lower_ms}-{upper_ms}",
        source=source,
    )


def _node(
    node_id: str,
    lower_ms: float,
    upper_ms: float,
    *,
    need_state: AegisNeedState = AegisNeedState.ACTIVE,
) -> AegisResidualNode:
    return AegisResidualNode(
        node_id=node_id,
        agent_id=f"agent-{node_id}",
        need_state=need_state,
        duration=_bound(lower_ms, upper_ms),
    )


def _dag(
    nodes: tuple[AegisResidualNode, ...],
    edges: tuple[tuple[str, str], ...] = (),
    *,
    complete: bool = True,
    observed_at_ms: float = 100.0,
    completion_barrier_id: str | None = None,
    proof_scope_kind: AegisProofScopeKind | None = None,
    witness_contract_valid_until_ms: float = 10_000.0,
) -> AegisResidualDag:
    if proof_scope_kind is None:
        proof_scope_kind = (
            AegisProofScopeKind.COMPLETE_RESIDUAL_DAG
            if completion_barrier_id is None
            else AegisProofScopeKind.CLOSED_MANDATORY_AND_JOIN
        )
    return AegisResidualDag(
        workflow_id="workflow",
        graph_version="graph-v1",
        observed_at_ms=observed_at_ms,
        nodes=nodes,
        edges=tuple(AegisResidualEdge(*edge) for edge in edges),
        proof_scope_kind=proof_scope_kind,
        proof_scope_complete=complete,
        witness_contract_id="scope-and-service-contract-v1",
        witness_contract_valid_until_ms=witness_contract_valid_until_ms,
        completion_barrier_id=completion_barrier_id,
    )


def test_parallel_hard_bounds_produce_positive_bounded_slack() -> None:
    dag = _dag((_node("short", 1.0, 2.0), _node("long", 10.0, 12.0)))

    tickets = build_bounded_slack_tickets(dag)
    short = tickets["short"]

    assert short.slack_kind is AegisSlackKind.BOUNDED
    assert short.lower_avoiding_ms == 10.0
    assert short.upper_through_ms == 2.0
    assert short.slack_ms == 8.0
    assert short.observed_at_ms == 100.0
    assert short.deadline_at_ms == 108.0
    assert short.usable_slack is True
    assert tickets["long"].slack_ms == 0.0

    fields = short.to_simulation_fields()
    assert fields["aegis_slack_semantics"] == BOUNDED_SLACK_SEMANTICS
    assert fields["aegis_slack_kind"] == "BOUNDED"
    assert fields["aegis_deadline_at_ms"] == 108.0
    assert fields["aegis_slack_service_envelope_versions"] == ["service-v3"]
    assert fields["aegis_slack_proof_scope_kind"] == "COMPLETE_RESIDUAL_DAG"
    assert fields["aegis_slack_proof_scope_complete"] is True
    assert (
        fields["aegis_slack_witness_contract_id"]
        == "scope-and-service-contract-v1"
    )
    assert fields["aegis_slack_witness_contract_valid_until_ms"] == 10_000.0
    assert len(fields["aegis_slack_witness_sha256"]) == 64


def test_duration_bound_is_derived_from_typed_work_and_service_envelope() -> None:
    bound = AegisHardDurationBound.from_work_service_envelope(
        work_lower_units=100.0,
        work_upper_units=200.0,
        service_rate_lower_units_per_ms=2.0,
        service_rate_upper_units_per_ms=4.0,
        fixed_lower_ms=5.0,
        fixed_upper_ms=10.0,
        queue_wait_upper_ms=20.0,
        work_unit="tokens",
        service_rate_work_unit="tokens",
        graph_version="graph-v1",
        service_envelope_version="service-v3",
        bound_id="derived-bound",
        source="declared_work_plus_reserved_service",
    )

    assert bound.lower_ms == 30.0
    assert bound.upper_ms == 130.0

    with pytest.raises(ValueError, match="finite, positive, and ordered"):
        AegisHardDurationBound.from_work_service_envelope(
            work_lower_units=1.0,
            work_upper_units=2.0,
            service_rate_lower_units_per_ms=0.0,
            service_rate_upper_units_per_ms=1.0,
            fixed_lower_ms=0.0,
            fixed_upper_ms=0.0,
            queue_wait_upper_ms=0.0,
            work_unit="tokens",
            service_rate_work_unit="tokens",
            graph_version="graph-v1",
            service_envelope_version="unreserved",
            bound_id="bad",
            source="unit-test",
        )

    with pytest.raises(ValueError, match="same unit"):
        AegisHardDurationBound.from_work_service_envelope(
            work_lower_units=1.0,
            work_upper_units=2.0,
            service_rate_lower_units_per_ms=1.0,
            service_rate_upper_units_per_ms=1.0,
            fixed_lower_ms=0.0,
            fixed_upper_ms=0.0,
            queue_wait_upper_ms=0.0,
            work_unit="tokens",
            service_rate_work_unit="bytes",
            graph_version="graph-v1",
            service_envelope_version="wrong-unit",
            bound_id="bad-unit",
            source="unit-test",
        )


def test_shared_join_tail_is_included_on_both_paths() -> None:
    dag = _dag(
        (
            _node("short", 1.0, 1.0),
            _node("long", 10.0, 10.0),
            _node("join-tail", 2.0, 2.0),
        ),
        (("short", "join-tail"), ("long", "join-tail")),
        completion_barrier_id="join-tail",
    )

    tickets = build_bounded_slack_tickets(dag)

    assert tickets["short"].lower_avoiding_ms == 12.0
    assert tickets["short"].upper_through_ms == 3.0
    assert tickets["short"].slack_ms == 9.0
    assert tickets["short"].completion_barrier_id == "join-tail"
    assert tickets["join-tail"].slack_ms == 0.0


def test_removing_a_mandatory_node_does_not_create_a_truncated_path() -> None:
    dag = _dag(
        (_node("source", 4.0, 4.0), _node("sink", 7.0, 7.0)),
        (("source", "sink"),),
    )

    tickets = build_bounded_slack_tickets(dag)

    assert tickets["source"].lower_avoiding_ms == 0.0
    assert tickets["source"].slack_ms == 0.0
    assert tickets["sink"].lower_avoiding_ms == 0.0
    assert tickets["sink"].slack_ms == 0.0


def _all_original_paths(dag: AegisResidualDag) -> tuple[tuple[str, ...], ...]:
    successors = {node.node_id: [] for node in dag.nodes}
    predecessors = {node.node_id: [] for node in dag.nodes}
    for edge in dag.edges:
        successors[edge.source_id].append(edge.target_id)
        predecessors[edge.target_id].append(edge.source_id)
    sources = sorted(node_id for node_id, rows in predecessors.items() if not rows)
    sinks = {node_id for node_id, rows in successors.items() if not rows}
    paths: list[tuple[str, ...]] = []

    def visit(node_id: str, prefix: tuple[str, ...]) -> None:
        current = prefix + (node_id,)
        if node_id in sinks:
            paths.append(current)
            return
        for successor in sorted(successors[node_id]):
            visit(successor, current)

    for source in sources:
        visit(source, ())
    return tuple(paths)


def test_bounded_slack_never_exceeds_true_slack_over_endpoint_durations() -> None:
    dag = _dag(
        (
            _node("a", 1.0, 3.0),
            _node("b", 4.0, 7.0),
            _node("join", 1.0, 2.0),
            _node("independent", 5.0, 9.0),
        ),
        (("a", "join"), ("b", "join")),
    )
    tickets = build_bounded_slack_tickets(dag)
    paths = _all_original_paths(dag)
    endpoints = {
        node.node_id: (node.duration.lower_ms, node.duration.upper_ms)
        for node in dag.nodes
        if node.duration is not None
    }

    for choices in itertools.product((0, 1), repeat=len(dag.nodes)):
        actual = {
            node.node_id: endpoints[node.node_id][choice]
            for node, choice in zip(dag.nodes, choices)
        }
        for node in dag.nodes:
            through = max(
                sum(actual[item] for item in path)
                for path in paths
                if node.node_id in path
            )
            avoiding_paths = [path for path in paths if node.node_id not in path]
            avoiding = max(
                (sum(actual[item] for item in path) for path in avoiding_paths),
                default=0.0,
            )
            true_slack = max(avoiding - through, 0.0)
            assert tickets[node.node_id].slack_ms <= true_slack + 1e-9


@pytest.mark.parametrize(
    "dag,expected_source",
    [
        (
            _dag((_node("a", 1.0, 2.0),), complete=False),
            "fail_closed_incomplete_proof_scope",
        ),
        (
            _dag(
                (
                    AegisResidualNode(
                        "a", "agent-a", AegisNeedState.ACTIVE, None
                    ),
                )
            ),
            "fail_closed_missing_hard_duration_bound",
        ),
        (
            _dag(
                (
                    AegisResidualNode(
                        "a",
                        "agent-a",
                        AegisNeedState.ACTIVE,
                        _bound(1.0, 2.0, graph_version="old-graph"),
                    ),
                )
            ),
            "fail_closed_stale_hard_duration_bound",
        ),
    ],
)
def test_incomplete_missing_or_stale_witness_fails_closed(
    dag: AegisResidualDag, expected_source: str
) -> None:
    ticket = build_bounded_slack_tickets(dag)["a"]

    assert ticket.slack_kind is AegisSlackKind.UNKNOWN
    assert ticket.slack_ms == 0.0
    assert ticket.deadline_at_ms == dag.observed_at_ms
    assert ticket.usable_slack is False
    assert ticket.source == expected_source


def test_witness_contract_must_cover_declared_upper_completion_horizon() -> None:
    dag = _dag(
        (_node("short", 1.0, 2.0), _node("long", 10.0, 12.0)),
        observed_at_ms=100.0,
        witness_contract_valid_until_ms=111.999,
    )

    tickets = build_bounded_slack_tickets(dag)

    assert all(
        ticket.slack_kind is AegisSlackKind.UNKNOWN
        for ticket in tickets.values()
    )
    assert all(
        ticket.source == "fail_closed_short_witness_contract"
        for ticket in tickets.values()
    )


def test_witness_contract_expiring_at_upper_horizon_is_accepted() -> None:
    dag = _dag(
        (_node("short", 1.0, 2.0), _node("long", 10.0, 12.0)),
        observed_at_ms=100.0,
        witness_contract_valid_until_ms=112.0,
    )

    assert (
        build_bounded_slack_tickets(dag)["short"].slack_kind
        is AegisSlackKind.BOUNDED
    )


def test_deferred_need_state_is_preserved_independently_of_bound_kind() -> None:
    dag = _dag(
        (
            _node(
                "waiting",
                1.0,
                2.0,
                need_state=AegisNeedState.DEFERRED,
            ),
            _node("long", 10.0, 10.0),
        )
    )

    ticket = build_bounded_slack_tickets(dag)["waiting"]

    assert ticket.need_state is AegisNeedState.DEFERRED
    assert ticket.slack_kind is AegisSlackKind.BOUNDED
    assert ticket.slack_ms == 8.0


def test_posthoc_bound_source_and_cyclic_graph_are_rejected() -> None:
    with pytest.raises(ValueError, match="post-hoc"):
        _bound(1.0, 2.0, source="trace_oracle_final_output")

    with pytest.raises(ValueError, match="acyclic"):
        _dag(
            (_node("a", 1.0, 1.0), _node("b", 1.0, 1.0)),
            (("a", "b"), ("b", "a")),
        )


def test_closed_region_requires_every_node_to_reach_its_barrier() -> None:
    with pytest.raises(ValueError, match="reach the completion barrier"):
        _dag(
            (
                _node("branch", 1.0, 1.0),
                _node("barrier", 0.0, 0.0),
                _node("outside", 2.0, 2.0),
            ),
            (("branch", "barrier"),),
            completion_barrier_id="barrier",
        )


def test_core_type_rejects_scope_kind_and_barrier_mismatch() -> None:
    with pytest.raises(ValueError, match="must not declare"):
        _dag(
            (_node("barrier", 0.0, 0.0),),
            completion_barrier_id="barrier",
            proof_scope_kind=AegisProofScopeKind.COMPLETE_RESIDUAL_DAG,
        )

    with pytest.raises(ValueError, match="requires a completion barrier"):
        _dag(
            (_node("node", 1.0, 1.0),),
            proof_scope_kind=AegisProofScopeKind.CLOSED_MANDATORY_AND_JOIN,
        )

    with pytest.raises(ValueError, match="must be a boolean"):
        _dag((_node("node", 1.0, 1.0),), complete="true")  # type: ignore[arg-type]
