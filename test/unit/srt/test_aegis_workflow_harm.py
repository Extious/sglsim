from __future__ import annotations

import pytest

from sglsim.srt.simulation.aegis_workflow_harm import (
    ProtectionAction,
    RecoveryEffect,
    ShadowJob,
    ShadowScenario,
    TargetOption,
    TargetState,
    WeightedScenario,
    WorkflowHarmPortfolio,
    WorkflowState,
)


def _action(
    action_id: str,
    *effects: tuple[str, str, float],
    demand: float = 1.0,
) -> ProtectionAction:
    return ProtectionAction(
        action_id=action_id,
        demand=demand,
        effects=tuple(
            RecoveryEffect(
                failure_domain="node0",
                job_id=job_id,
                target_id=target_id,
                protected_recovery_s=protected_s,
            )
            for job_id, target_id, protected_s in effects
        ),
    )


def _portfolio(
    *,
    scenario_id: str,
    workflows: tuple[WorkflowState, ...],
    targets: tuple[TargetState, ...],
    jobs: tuple[ShadowJob, ...],
    actions: tuple[ProtectionAction, ...],
) -> WorkflowHarmPortfolio:
    return WorkflowHarmPortfolio(
        scenarios=(
            WeightedScenario(
                probability=1.0,
                scenario=ShadowScenario(
                    scenario_id=scenario_id,
                    failure_domain="node0",
                    workflows=workflows,
                    targets=targets,
                    jobs=jobs,
                ),
            ),
        ),
        actions=actions,
    )


def _affected(
    job_id: str,
    workflow_id: str,
    target_id: str,
    recovery_s: float,
    *,
    slack_s: float = 0.0,
    criticality: float = 1.0,
) -> ShadowJob:
    return ShadowJob(
        job_id=job_id,
        workflow_id=workflow_id,
        source_domain="node0",
        target_options=(
            TargetOption(
                target_id=target_id,
                resume_service_s=1.0,
                unprotected_recovery_s=recovery_s,
            ),
        ),
        slack_lcb_s=slack_s,
        criticality=criticality,
    )


def test_correlated_domain_rejects_request_local_double_counting() -> None:
    portfolio = _portfolio(
        scenario_id="correlated-domain",
        workflows=(
            WorkflowState("workflow-a", healthy_finish_s=1.0),
            WorkflowState("workflow-b", healthy_finish_s=1.0),
        ),
        targets=(TargetState("t0"), TargetState("t1"), TargetState("t2")),
        jobs=(
            _affected(
                "a-fast",
                "workflow-a",
                "t0",
                10.0,
                slack_s=2.0,
                criticality=1.0,
            ),
            _affected("a-slow", "workflow-a", "t1", 12.0),
            _affected(
                "b-only",
                "workflow-b",
                "t2",
                6.0,
                criticality=0.5,
            ),
        ),
        actions=(
            _action("protect-a-fast", ("a-fast", "t0", 0.0)),
            _action("protect-b", ("b-only", "t2", 0.0)),
        ),
    )

    load = portfolio.greedy("load_size", budget=1.0)
    criticality = portfolio.greedy("criticality", budget=1.0)
    slack = portfolio.greedy("request_slack", budget=1.0)
    aegis = portfolio.greedy("aegis_joint", budget=1.0)
    exact = portfolio.solve_exact(budget=1.0)

    assert load.selected_actions == ("protect-a-fast",)
    assert criticality.selected_actions == ("protect-a-fast",)
    assert slack.selected_actions == ("protect-a-fast",)
    assert aegis.selected_actions == ("protect-b",)
    assert exact.selected_actions == aegis.selected_actions
    assert load.true_outcome.expected_harm.wct_s == 18.0
    assert aegis.true_outcome.expected_harm.wct_s == 12.0


def test_joint_objective_counts_bystander_queue_delay() -> None:
    bystanders = tuple(
        ShadowJob(
            job_id=f"bystander-{index}",
            workflow_id=f"bystander-workflow-{index}",
            source_domain=None,
            target_options=(TargetOption("busy", 100.0),),
            queue_rank=index,
        )
        for index in range(3)
    )
    portfolio = _portfolio(
        scenario_id="bystander-queue",
        workflows=(
            WorkflowState("affected-busy", healthy_finish_s=1.0),
            WorkflowState("affected-isolated", healthy_finish_s=1.0),
            WorkflowState("bystander-workflow-0", healthy_finish_s=100.0),
            WorkflowState("bystander-workflow-1", healthy_finish_s=200.0),
            WorkflowState("bystander-workflow-2", healthy_finish_s=300.0),
        ),
        targets=(TargetState("busy"), TargetState("isolated")),
        jobs=(
            _affected(
                "affected-on-busy",
                "affected-busy",
                "busy",
                4.0,
                criticality=0.5,
            ),
            _affected(
                "affected-on-isolated",
                "affected-isolated",
                "isolated",
                5.0,
            ),
            *bystanders,
        ),
        actions=(
            _action("protect-busy", ("affected-on-busy", "busy", 0.0)),
            _action(
                "protect-isolated",
                ("affected-on-isolated", "isolated", 0.0),
            ),
        ),
    )

    for policy in ("load_size", "criticality", "request_slack"):
        result = portfolio.greedy(policy, budget=1.0)
        assert result.selected_actions == ("protect-isolated",)
        assert result.true_outcome.expected_harm.wct_s == 19.0

    aegis = portfolio.greedy("aegis_joint", budget=1.0)
    exact = portfolio.solve_exact(budget=1.0)
    assert aegis.selected_actions == ("protect-busy",)
    assert exact.selected_actions == aegis.selected_actions
    assert aegis.true_outcome.expected_harm.wct_s == 8.0

    busy_order = dict(aegis.true_outcome.scenarios[0].target_order)["busy"]
    assert tuple(job for job in busy_order if job.startswith("bystander")) == (
        "bystander-0",
        "bystander-1",
        "bystander-2",
    )


def test_shared_copy_is_charged_once_and_can_help_multiple_workflows() -> None:
    portfolio = _portfolio(
        scenario_id="shared-copy",
        workflows=tuple(
            WorkflowState(f"workflow-{index}", healthy_finish_s=1.0)
            for index in range(3)
        ),
        targets=tuple(TargetState(f"t{index}") for index in range(3)),
        jobs=(
            _affected("shared-0", "workflow-0", "t0", 4.0, criticality=0.25),
            _affected("shared-1", "workflow-1", "t1", 4.0, criticality=0.25),
            _affected("single", "workflow-2", "t2", 6.0),
        ),
        actions=(
            _action(
                "one-shared-copy",
                ("shared-0", "t0", 0.0),
                ("shared-1", "t1", 0.0),
            ),
            _action("one-critical-copy", ("single", "t2", 0.0)),
        ),
    )

    assert portfolio.greedy("criticality", budget=1.0).selected_actions == (
        "one-critical-copy",
    )
    assert portfolio.greedy("aegis_joint", budget=1.0).selected_actions == (
        "one-shared-copy",
    )
    assert portfolio.solve_exact(budget=1.0).selected_actions == (
        "one-shared-copy",
    )


def test_negative_control_does_not_manufacture_policy_disagreement() -> None:
    portfolio = _portfolio(
        scenario_id="negative-control",
        workflows=(
            WorkflowState("workflow-a", healthy_finish_s=1.0),
            WorkflowState("workflow-b", healthy_finish_s=1.0),
        ),
        targets=(TargetState("t0"), TargetState("t1")),
        jobs=(
            _affected("a", "workflow-a", "t0", 10.0),
            _affected("b", "workflow-b", "t1", 5.0),
        ),
        actions=(
            _action("protect-a", ("a", "t0", 0.0)),
            _action("protect-b", ("b", "t1", 0.0)),
        ),
    )

    selected = {
        portfolio.greedy(policy, budget=1.0).selected_actions
        for policy in ("load_size", "criticality", "request_slack", "aegis_joint")
    }
    assert selected == {("protect-a",)}
    assert portfolio.solve_exact(budget=1.0).selected_actions == ("protect-a",)


def test_slack_clipping_makes_protection_value_zero() -> None:
    portfolio = _portfolio(
        scenario_id="slack-clipping",
        workflows=(
            WorkflowState(
                "workflow-a",
                healthy_finish_s=10.0,
                unaffected_finish_s=10.0,
            ),
        ),
        targets=(TargetState("t0"),),
        jobs=(
            _affected("a", "workflow-a", "t0", 5.0, slack_s=10.0),
        ),
        actions=(_action("protect-a", ("a", "t0", 0.0)),),
    )

    assert portfolio.evaluate(frozenset()).expected_harm.wct_s == 0.0
    assert portfolio.greedy("request_slack", budget=1.0).selected_actions == ()
    assert portfolio.greedy("aegis_joint", budget=1.0).selected_actions == ()
    assert portfolio.solve_exact(budget=1.0).selected_actions == ()


def test_lexicographic_slo_reduction_precedes_larger_wct_reduction() -> None:
    portfolio = _portfolio(
        scenario_id="lexicographic-slo",
        workflows=(
            WorkflowState("workflow-slo", healthy_finish_s=1.0, deadline_s=2.0),
            WorkflowState("workflow-wct", healthy_finish_s=1.0),
        ),
        targets=(TargetState("t0"), TargetState("t1")),
        jobs=(
            _affected("slo", "workflow-slo", "t0", 2.0),
            _affected("wct", "workflow-wct", "t1", 10.0),
        ),
        actions=(
            _action("protect-slo", ("slo", "t0", 0.0)),
            _action("protect-wct", ("wct", "t1", 0.0)),
        ),
    )

    assert portfolio.greedy("load_size", budget=1.0).selected_actions == (
        "protect-wct",
    )
    assert portfolio.greedy("aegis_joint", budget=1.0).selected_actions == (
        "protect-slo",
    )
    assert portfolio.solve_exact(budget=1.0).selected_actions == ("protect-slo",)


def test_faster_copy_saturates_slower_copy_for_the_same_request() -> None:
    portfolio = _portfolio(
        scenario_id="action-saturation",
        workflows=(WorkflowState("workflow-a", healthy_finish_s=1.0),),
        targets=(TargetState("t0"),),
        jobs=(_affected("a", "workflow-a", "t0", 10.0),),
        actions=(
            _action("fast-copy", ("a", "t0", 0.0)),
            _action("slow-copy", ("a", "t0", 5.0)),
        ),
    )

    result = portfolio.greedy("aegis_joint", budget=2.0)
    assert result.selected_actions == ("fast-copy",)
    assert len(result.steps) == 1
    assert portfolio.solve_exact(budget=2.0).selected_actions == ("fast-copy",)


def test_action_cannot_claim_an_unknown_failure_domain() -> None:
    scenario = ShadowScenario(
        scenario_id="domain-correctness",
        failure_domain="node0",
        workflows=(WorkflowState("workflow-a", healthy_finish_s=1.0),),
        targets=(TargetState("t0"),),
        jobs=(_affected("a", "workflow-a", "t0", 5.0),),
    )
    invalid = ProtectionAction(
        action_id="wrong-domain-copy",
        demand=1.0,
        effects=(
            RecoveryEffect(
                failure_domain="node1",
                job_id="a",
                target_id="t0",
                protected_recovery_s=0.0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown domain"):
        WorkflowHarmPortfolio(
            scenarios=(WeightedScenario(1.0, scenario),),
            actions=(invalid,),
        )


def test_exact_certificate_is_deterministic_and_complete() -> None:
    portfolio = _portfolio(
        scenario_id="certificate",
        workflows=(WorkflowState("workflow-a", healthy_finish_s=1.0),),
        targets=(TargetState("t0"),),
        jobs=(_affected("a", "workflow-a", "t0", 5.0),),
        actions=(_action("protect-a", ("a", "t0", 0.0)),),
    )

    first = portfolio.solve_exact(budget=1.0)
    second = portfolio.solve_exact(budget=1.0)

    assert first.certificate.status == "OPTIMAL"
    assert first.certificate.objective == first.certificate.best_bound
    assert first.certificate.gap == 0.0
    assert first.certificate.enumerated_subsets == 2
    assert first.certificate.enumerated_schedules > 0
    assert first.certificate.instance_sha256 == portfolio.sha256
    assert first.certificate.instance_sha256 == second.certificate.instance_sha256
    assert first.selected_actions == second.selected_actions
