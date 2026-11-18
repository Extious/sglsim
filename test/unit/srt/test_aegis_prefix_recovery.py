import itertools

import pytest

from sglsim.srt.simulation.aegis_prefix_recovery import (
    PrefixRecoveryEdge,
    PrefixRecoveryKind,
    PrefixRecoveryProblem,
)


def _edge(
    edge_id: str,
    start: int,
    end: int,
    cost: float,
    kind: PrefixRecoveryKind = PrefixRecoveryKind.RECOMPUTE,
    *,
    action_id: str | None = None,
    access_valid_for: float | None = None,
) -> PrefixRecoveryEdge:
    return PrefixRecoveryEdge(
        edge_id=edge_id,
        start_group=start,
        end_group=end,
        service_upper_s=cost,
        kind=kind,
        action_id=action_id,
        access_valid_for_s=access_valid_for,
    )


def test_shortest_path_composes_dram_and_ssd_segments() -> None:
    problem = PrefixRecoveryProblem(
        terminal_group=4,
        edges=(
            _edge("recompute-all", 0, 4, 10.0),
            _edge(
                "dram-prefix",
                0,
                2,
                1.0,
                PrefixRecoveryKind.STORE_DRAM,
                action_id="retain-dram",
            ),
            _edge(
                "ssd-suffix",
                2,
                4,
                3.0,
                PrefixRecoveryKind.STORE_SSD,
                action_id="retain-ssd",
            ),
        ),
        resume_barrier_upper_s=0.5,
    )

    plan = problem.solve(
        enabled_action_ids=frozenset({"retain-dram", "retain-ssd"})
    )

    assert [step.edge_id for step in plan.steps] == [
        "dram-prefix",
        "ssd-suffix",
    ]
    assert plan.total_service_upper_s == 4.5


def test_action_edge_is_invisible_before_commit() -> None:
    problem = PrefixRecoveryProblem(
        terminal_group=2,
        edges=(
            _edge("recompute", 0, 2, 5.0),
            _edge(
                "peer",
                0,
                2,
                0.1,
                PrefixRecoveryKind.PEER_ADOPT,
                action_id="peer-copy",
            ),
        ),
    )

    assert problem.solve().total_service_upper_s == 5.0
    assert (
        problem.solve(enabled_action_ids=frozenset({"peer-copy"}))
        .total_service_upper_s
        == 0.1
    )


def test_suffix_edge_requires_a_continuous_path_to_its_start() -> None:
    problem = PrefixRecoveryProblem(
        terminal_group=4,
        edges=(
            _edge("recompute-prefix", 0, 2, 2.0),
            _edge(
                "stored-suffix",
                2,
                4,
                1.0,
                PrefixRecoveryKind.STORE_DRAM,
                action_id="suffix",
            ),
        ),
    )

    with pytest.raises(ValueError, match="no valid continuous"):
        problem.solve()
    plan = problem.solve(enabled_action_ids=frozenset({"suffix"}))
    assert [step.edge_id for step in plan.steps] == [
        "recompute-prefix",
        "stored-suffix",
    ]


def test_access_lease_must_cover_one_edge_service_bound() -> None:
    with pytest.raises(ValueError, match="access lease must cover"):
        _edge(
            "short-lease",
            0,
            1,
            2.0,
            PrefixRecoveryKind.STORE_DRAM,
            action_id="store",
            access_valid_for=1.999,
        )

    edge = _edge(
        "exact-lease",
        0,
        1,
        2.0,
        PrefixRecoveryKind.STORE_DRAM,
        action_id="store",
        access_valid_for=2.0,
    )
    assert edge.access_valid_for_s == 2.0


def test_step_offsets_are_relative_to_job_service_not_failure_arrival() -> None:
    problem = PrefixRecoveryProblem(
        terminal_group=2,
        edges=(
            _edge("first", 0, 1, 3.0),
            _edge("second", 1, 2, 2.0),
        ),
    )

    plan = problem.solve()

    assert plan.steps[0].starts_after_service_s == 0.0
    assert plan.steps[0].completes_after_service_s == 3.0
    assert plan.steps[1].starts_after_service_s == 3.0
    assert plan.steps[1].completes_after_service_s == 5.0


def test_enabling_valid_edges_never_increases_modeled_service() -> None:
    problem = PrefixRecoveryProblem(
        terminal_group=3,
        edges=(
            _edge("r0", 0, 1, 2.0),
            _edge("r1", 1, 2, 2.0),
            _edge("r2", 2, 3, 2.0),
            _edge(
                "a",
                0,
                2,
                1.0,
                PrefixRecoveryKind.PEER_ADOPT,
                action_id="a",
            ),
            _edge(
                "b",
                1,
                3,
                1.0,
                PrefixRecoveryKind.STORE_DRAM,
                action_id="b",
            ),
        ),
    )
    costs = {}
    for count in range(3):
        for subset in itertools.combinations(("a", "b"), count):
            selected = frozenset(subset)
            costs[selected] = problem.solve(
                enabled_action_ids=selected
            ).total_service_upper_s

    assert costs[frozenset()] == 6.0
    assert costs[frozenset({"a"})] <= costs[frozenset()]
    assert costs[frozenset({"b"})] <= costs[frozenset()]
    assert costs[frozenset({"a", "b"})] <= min(
        costs[frozenset({"a"})], costs[frozenset({"b"})]
    )


def test_equal_cost_plan_has_deterministic_fewer_step_tie_break() -> None:
    problem = PrefixRecoveryProblem(
        terminal_group=2,
        edges=(
            _edge("one-step", 0, 2, 2.0),
            _edge("two-step-a", 0, 1, 1.0),
            _edge("two-step-b", 1, 2, 1.0),
        ),
    )

    assert [step.edge_id for step in problem.solve().steps] == ["one-step"]


def test_invalid_graph_and_unknown_action_fail_closed() -> None:
    with pytest.raises(ValueError, match="advance the prefix"):
        _edge("backward", 2, 1, 1.0)
    with pytest.raises(ValueError, match="exceeds terminal_group"):
        PrefixRecoveryProblem(
            terminal_group=1,
            edges=(_edge("too-far", 0, 2, 1.0),),
        )

    problem = PrefixRecoveryProblem(
        terminal_group=1,
        edges=(_edge("recompute", 0, 1, 1.0),),
    )
    with pytest.raises(ValueError, match="unknown enabled actions"):
        problem.solve(enabled_action_ids=frozenset({"not-committed"}))
