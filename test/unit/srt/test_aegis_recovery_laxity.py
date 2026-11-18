import itertools

import pytest

from sglsim.srt.simulation.aegis_recovery_laxity import (
    ProtectionAction,
    RecoveryEffect,
    RecoveryExecutionSlice,
    RecoveryJob,
    RecoveryLaxityInstance,
    RecoveryTarget,
    schedule_versioned_suffix,
)


def _instance(*, targets, jobs, actions=()):
    return RecoveryLaxityInstance(
        failure_domain="dp:0",
        targets=tuple(targets),
        jobs=tuple(jobs),
        actions=tuple(actions),
    )


def test_static_edd_is_simultaneously_released_edf_special_case() -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0")],
        jobs=[
            RecoveryJob("long", "w-long", "t0", 10.0, 9.0),
            RecoveryJob("urgent", "w-urgent", "t0", 2.0, 1.0),
        ],
    )

    schedule = instance.schedule(frozenset())

    assert schedule.target_order == (("t0", ("urgent", "long")),)
    assert dict(schedule.completion_s) == {"long": 10.0, "urgent": 1.0}
    assert schedule.risk.maximum_tardiness_s == 0.0


def test_edf_preempts_at_a_later_release_event() -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0")],
        jobs=[
            RecoveryJob(
                "slow", "w-slow", "t0", 10.0, 8.0, release_offset_s=0.0
            ),
            RecoveryJob(
                "urgent", "w-urgent", "t0", 4.0, 1.0, release_offset_s=2.0
            ),
        ],
    )

    schedule = instance.schedule(frozenset())

    assert schedule.target_order == (
        ("t0", ("slow", "urgent", "slow")),
    )
    assert schedule.target_timeline == (
        (
            "t0",
            (
                ("slow", 0.0, 2.0),
                ("urgent", 2.0, 3.0),
                ("slow", 3.0, 9.0),
            ),
        ),
    )
    assert dict(schedule.completion_s) == {"slow": 9.0, "urgent": 3.0}
    assert schedule.risk.maximum_tardiness_s == 0.0


def test_release_time_does_not_recharge_an_expired_deadline() -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0")],
        jobs=[
            RecoveryJob(
                "late-on-release",
                "workflow",
                "t0",
                2.0,
                1.0,
                release_offset_s=5.0,
            )
        ],
    )

    schedule = instance.schedule(frozenset())

    assert schedule.target_timeline == (
        ("t0", (("late-on-release", 5.0, 6.0),)),
    )
    assert dict(schedule.lateness_s) == {"late-on-release": 4.0}


def test_target_availability_delays_all_jobs_without_changing_deadlines() -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0", availability_offset_s=3.0)],
        jobs=[
            RecoveryJob("less-urgent", "w0", "t0", 10.0, 2.0),
            RecoveryJob("urgent", "w1", "t0", 4.0, 1.0),
        ],
    )

    schedule = instance.schedule(frozenset())

    assert schedule.target_order == (("t0", ("urgent", "less-urgent")),)
    assert dict(schedule.completion_s) == {"urgent": 4.0, "less-urgent": 6.0}


def test_target_availability_does_not_recharge_zero_slack() -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0", availability_offset_s=3.0)],
        jobs=[RecoveryJob("unknown", "workflow", "t0", 0.0, 1.0)],
    )

    schedule = instance.schedule(frozenset())

    assert schedule.target_timeline == (
        ("t0", (("unknown", 3.0, 4.0),)),
    )
    assert dict(schedule.lateness_s) == {"unknown": 4.0}


def test_chunked_edf_preempts_only_at_the_quantum_boundary() -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0")],
        jobs=[
            RecoveryJob("slow", "w-slow", "t0", 10.0, 3.0),
            RecoveryJob(
                "urgent",
                "w-urgent",
                "t0",
                2.0,
                0.5,
                release_offset_s=0.1,
            ),
        ],
    )

    schedule = instance.schedule_chunked(
        frozenset(), max_quantum_s=1.0
    )

    assert schedule.target_timeline == (
        (
            "t0",
            (
                ("slow", 0.0, 1.0),
                ("urgent", 1.0, 1.5),
                ("slow", 1.5, 2.5),
                ("slow", 2.5, 3.5),
            ),
        ),
    )


@pytest.mark.parametrize("quantum", [0.0, -1.0, float("inf")])
def test_chunk_quantum_must_be_finite_and_positive(quantum: float) -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0")],
        jobs=[RecoveryJob("r0", "w0", "t0", 0.0, 1.0)],
    )

    with pytest.raises(ValueError, match="max_quantum_s"):
        instance.schedule_chunked(frozenset(), max_quantum_s=quantum)


def test_small_exhaustive_chunked_lmax_is_within_one_quantum() -> None:
    max_quantum_s = 1.0
    releases = (0.0, 0.5, 1.0)
    deadlines = (0.0, 1.0, 3.0)
    services = (0.5, 1.5)

    for release_tuple in itertools.product(releases, repeat=3):
        for deadline_tuple in itertools.product(deadlines, repeat=3):
            for service_tuple in itertools.product(services, repeat=3):
                jobs = [
                    RecoveryJob(
                        f"j{index}",
                        f"w{index}",
                        "t0",
                        deadline_tuple[index],
                        service_tuple[index],
                        release_offset_s=release_tuple[index],
                    )
                    for index in range(3)
                ]
                instance = _instance(
                    targets=[RecoveryTarget("t0")], jobs=jobs
                )
                ideal = instance.schedule(frozenset())
                chunked = instance.schedule_chunked(
                    frozenset(), max_quantum_s=max_quantum_s
                )
                ideal_lmax = max(dict(ideal.lateness_s).values())
                chunked_lmax = max(dict(chunked.lateness_s).values())

                assert chunked_lmax <= ideal_lmax + max_quantum_s + 1e-9


def test_zero_slack_fail_closed_tie_uses_shortest_service_first() -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0")],
        jobs=[
            RecoveryJob("slow", "w-slow", "t0", 0.0, 4.0),
            RecoveryJob("fast", "w-fast", "t0", 0.0, 1.0),
        ],
    )

    schedule = instance.schedule(frozenset())

    assert schedule.target_order == (("t0", ("fast", "slow")),)
    assert dict(schedule.completion_s) == {"fast": 1.0, "slow": 5.0}
    assert schedule.risk.maximum_tardiness_s == 5.0
    assert schedule.risk.total_workflow_overrun_s == 6.0


@pytest.mark.parametrize(
    "slack,p0,p1,expected_gain",
    [
        (2.0, 1.5, 0.2, 0.0),
        (0.5, 2.0, 0.2, 1.5),
        (0.0, 0.10, 0.05, 0.05),
    ],
)
def test_single_job_action_value_matches_the_three_analytic_regions(
    slack: float, p0: float, p1: float, expected_gain: float
) -> None:
    action = ProtectionAction(
        "protect", 1.0, (RecoveryEffect("job", p1),)
    )
    instance = _instance(
        targets=[RecoveryTarget("t0")],
        jobs=[RecoveryJob("job", "workflow", "t0", slack, p0)],
        actions=[action],
    )

    before = instance.schedule(frozenset()).risk
    after = instance.schedule(frozenset({"protect"})).risk

    assert before.maximum_tardiness_s - after.maximum_tardiness_s == (
        pytest.approx(expected_gain)
    )


def test_equal_demand_size_score_cannot_express_slack_crossing_value() -> None:
    actions = [
        ProtectionAction(
            "hidden-by-slack", 1.0, (RecoveryEffect("a", 0.2),)
        ),
        ProtectionAction(
            "crosses-deadline", 1.0, (RecoveryEffect("b", 0.2),)
        ),
    ]
    instance = _instance(
        targets=[RecoveryTarget("t0"), RecoveryTarget("t1")],
        jobs=[
            RecoveryJob("a", "wa", "t0", 2.0, 1.5),
            RecoveryJob("b", "wb", "t1", 0.5, 2.0),
        ],
        actions=actions,
    )

    result = instance.greedy(budget=1.0)

    assert actions[0].demand == actions[1].demand
    assert result.selected_actions == ("crosses-deadline",)


def test_correlated_branches_use_workflow_max_instead_of_request_sum() -> None:
    instance = _instance(
        targets=[RecoveryTarget("t0"), RecoveryTarget("t1")],
        jobs=[
            RecoveryJob("critical", "workflow", "t0", 0.0, 8.0),
            RecoveryJob("hidden", "workflow", "t1", 10.0, 4.0),
        ],
    )

    schedule = instance.schedule(frozenset())

    assert dict(schedule.workflow_overrun_s) == {"workflow": 8.0}
    assert schedule.risk.total_workflow_overrun_s == 8.0


def test_action_on_hidden_correlated_branch_has_zero_risk_gain() -> None:
    action = ProtectionAction(
        "protect-hidden",
        demand=1.0,
        effects=(RecoveryEffect("hidden", 1.0),),
    )
    instance = _instance(
        targets=[RecoveryTarget("t0"), RecoveryTarget("t1")],
        jobs=[
            RecoveryJob("critical", "workflow", "t0", 0.0, 8.0),
            RecoveryJob("hidden", "workflow", "t1", 10.0, 4.0),
        ],
        actions=[action],
    )

    before = instance.schedule(frozenset()).risk
    after = instance.schedule(frozenset({"protect-hidden"})).risk
    greedy = instance.greedy(budget=1.0)

    assert before == after
    assert greedy.selected_actions == ()


def test_shared_action_charges_demand_once_and_reduces_two_workflows() -> None:
    action = ProtectionAction(
        "shared-store-object",
        demand=1.0,
        effects=(
            RecoveryEffect("r0", 1.0),
            RecoveryEffect("r1", 1.0),
        ),
    )
    instance = _instance(
        targets=[RecoveryTarget("t0"), RecoveryTarget("t1")],
        jobs=[
            RecoveryJob("r0", "w0", "t0", 0.0, 5.0),
            RecoveryJob("r1", "w1", "t1", 0.0, 5.0),
        ],
        actions=[action],
    )

    result = instance.greedy(budget=1.0)

    assert result.selected_actions == ("shared-store-object",)
    assert result.used_budget == 1.0
    assert result.schedule.risk.maximum_tardiness_s == 1.0
    assert result.schedule.risk.total_workflow_overrun_s == 2.0


def test_exact_certificate_and_greedy_match_on_one_action_window() -> None:
    actions = [
        ProtectionAction(
            "large-saving",
            demand=1.0,
            effects=(RecoveryEffect("critical", 1.0),),
        ),
        ProtectionAction(
            "hidden-saving",
            demand=1.0,
            effects=(RecoveryEffect("hidden", 0.0),),
        ),
    ]
    instance = _instance(
        targets=[RecoveryTarget("t0"), RecoveryTarget("t1")],
        jobs=[
            RecoveryJob("critical", "w0", "t0", 0.0, 6.0),
            RecoveryJob("hidden", "w1", "t1", 20.0, 10.0),
        ],
        actions=actions,
    )

    greedy = instance.greedy(budget=1.0)
    exact = instance.solve_exact(budget=1.0)

    assert greedy.selected_actions == ("large-saving",)
    assert exact.selected_actions == greedy.selected_actions
    assert exact.certificate.status == "OPTIMAL"
    assert exact.certificate.objective == exact.certificate.best_bound
    assert exact.certificate.gap == 0.0
    assert exact.certificate.enumerated_subsets == 3
    assert len(exact.certificate.instance_sha256) == 64


def test_protection_effect_cannot_increase_service_time() -> None:
    with pytest.raises(ValueError, match="cannot increase"):
        _instance(
            targets=[RecoveryTarget("t0")],
            jobs=[RecoveryJob("r0", "w0", "t0", 0.0, 1.0)],
            actions=[
                ProtectionAction(
                    "bad", 1.0, (RecoveryEffect("r0", 2.0),)
                )
            ],
        )


def test_release_offset_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="release_offset_s"):
        RecoveryJob(
            "r0", "w0", "t0", 1.0, 1.0, release_offset_s=-1.0
        )


def test_live_deadline_update_preserves_prefix_and_only_schedules_suffix() -> None:
    executed = (RecoveryExecutionSlice("b", 0.0, 1.0),)
    updated = schedule_versioned_suffix(
        failure_domain="dp:0",
        target=RecoveryTarget("t0"),
        remaining_jobs=(RecoveryJob("a", "wa", "t0", 1.0, 1.0),),
        executed_prefix=executed,
        linearization_s=1.0,
        route_version=1,
        due_version=2,
        service_version=1,
    )

    # The online history B@[0,1] is immutable.  A's tightened deadline is
    # applied only to the remaining-work suffix, so A completes at 2 and is
    # late by 1.  A clairvoyant schedule using the final deadlines from time 0
    # could instead run A then B and achieve zero maximum tardiness.
    clairvoyant = _instance(
        targets=[RecoveryTarget("t0")],
        jobs=[
            RecoveryJob("a", "wa", "t0", 1.0, 1.0),
            RecoveryJob("b", "wb", "t0", 2.0, 1.0),
        ],
    ).schedule(frozenset())

    assert updated.executed_prefix == executed
    assert updated.suffix.target_timeline == (
        ("t0", (("a", 1.0, 2.0),)),
    )
    assert updated.suffix.risk.maximum_tardiness_s == 1.0
    assert clairvoyant.risk.maximum_tardiness_s == 0.0


def test_due_version_atomically_rekeys_all_remaining_jobs() -> None:
    common = dict(
        failure_domain="dp:0",
        target=RecoveryTarget("t0"),
        executed_prefix=(RecoveryExecutionSlice("done", 0.0, 1.0),),
        linearization_s=1.0,
        route_version=4,
        service_version=7,
    )
    old = schedule_versioned_suffix(
        **common,
        remaining_jobs=(
            RecoveryJob("a", "wa", "t0", 100.0, 1.0),
            RecoveryJob("b", "wb", "t0", 3.0, 1.0),
        ),
        due_version=8,
    )
    new = schedule_versioned_suffix(
        **common,
        remaining_jobs=(
            RecoveryJob("a", "wa", "t0", 2.0, 1.0),
            RecoveryJob("b", "wb", "t0", 3.0, 1.0),
        ),
        due_version=9,
    )

    assert old.suffix.target_order == (("t0", ("b", "a")),)
    assert new.suffix.target_order == (("t0", ("a", "b")),)
    assert new.executed_prefix == old.executed_prefix


def test_inflight_block_requires_old_contract_terminal_fence() -> None:
    kwargs = dict(
        failure_domain="dp:0",
        target=RecoveryTarget("t0"),
        remaining_jobs=(RecoveryJob("a", "wa", "t0", 5.0, 1.0),),
        executed_prefix=(RecoveryExecutionSlice("done", 0.0, 1.0),),
        linearization_s=1.0,
        route_version=2,
        due_version=2,
        service_version=2,
        initial_blocking_upper_s=0.5,
    )

    with pytest.raises(ValueError, match="terminal fence"):
        schedule_versioned_suffix(
            **kwargs,
            old_contract_fenced_to_terminal=False,
        )

    fenced = schedule_versioned_suffix(
        **kwargs,
        old_contract_fenced_to_terminal=True,
    )
    assert fenced.suffix.target_timeline == (
        ("t0", (("a", 1.5, 2.5),)),
    )


def test_versioned_suffix_rejects_prefix_crossing_linearization() -> None:
    with pytest.raises(ValueError, match="crosses the update"):
        schedule_versioned_suffix(
            failure_domain="dp:0",
            target=RecoveryTarget("t0"),
            remaining_jobs=(RecoveryJob("a", "wa", "t0", 5.0, 1.0),),
            executed_prefix=(RecoveryExecutionSlice("old", 0.0, 1.5),),
            linearization_s=1.0,
            route_version=1,
            due_version=1,
            service_version=1,
        )
