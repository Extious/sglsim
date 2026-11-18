import pytest

from sglsim.srt.simulation.aegis_prefix_recovery import (
    PrefixRecoveryEdge,
    PrefixRecoveryKind,
    PrefixRecoveryProblem,
)
from sglsim.srt.simulation.aegis_protection_portfolio import (
    FailureScenario,
    PortfolioAction,
    ProtectionPortfolio,
    ResourceBudget,
    ResourceDemand,
    ScenarioRecoveryProblem,
    ScenarioEffect,
)
from sglsim.srt.simulation.aegis_recovery_laxity import (
    RecoveryJob,
    RecoveryTarget,
)


def _scenario(
    domain: str,
    *,
    job_id: str,
    service_s: float,
    deadline_s: float = 0.0,
) -> FailureScenario:
    return FailureScenario(
        failure_domain=domain,
        targets=(RecoveryTarget(f"target-{domain}"),),
        jobs=(
            RecoveryJob(
                job_id,
                f"workflow-{domain}",
                f"target-{domain}",
                deadline_s,
                service_s,
            ),
        ),
    )


def _demand(resource_id: str, amount: float) -> tuple[ResourceDemand, ...]:
    return (ResourceDemand(resource_id, amount),)


def _effect(
    domain: str, job_id: str, protected_service_s: float
) -> tuple[ScenarioEffect, ...]:
    return (ScenarioEffect(domain, job_id, protected_service_s),)


def test_robust_pair_prefers_reducing_the_worst_domain() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(
            _scenario("d0", job_id="j0", service_s=10.0),
            _scenario("d1", job_id="j1", service_s=6.0),
        ),
        budgets=(ResourceBudget("peer_hbm_bytes", 1.0),),
        actions=(
            PortfolioAction(
                "protect-worst",
                _demand("peer_hbm_bytes", 1.0),
                _effect("d0", "j0", 5.0),
            ),
            PortfolioAction(
                "protect-more-total",
                _demand("peer_hbm_bytes", 1.0),
                _effect("d1", "j1", 0.0),
            ),
        ),
    )

    greedy = portfolio.greedy()
    exact = portfolio.solve_exact()

    assert greedy.selected_actions == ("protect-worst",)
    assert exact.selected_actions == greedy.selected_actions
    assert greedy.outcome.risk.key == (6.0, 11.0)


def test_shared_cross_domain_action_is_charged_once() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(
            _scenario("d0", job_id="j0", service_s=5.0),
            _scenario("d1", job_id="j1", service_s=5.0),
        ),
        budgets=(ResourceBudget("store_bytes", 1.0),),
        actions=(
            PortfolioAction(
                "shared-prefix",
                _demand("store_bytes", 1.0),
                (
                    ScenarioEffect("d0", "j0", 1.0),
                    ScenarioEffect("d1", "j1", 1.0),
                ),
            ),
        ),
    )

    result = portfolio.greedy()

    assert result.selected_actions == ("shared-prefix",)
    assert dict(result.used_resources) == {"store_bytes": 1.0}
    assert result.outcome.risk.key == (1.0, 2.0)


def test_portfolio_recomputes_complementary_multi_tier_prefix_plan() -> None:
    problem = PrefixRecoveryProblem(
        terminal_group=3,
        edges=(
            PrefixRecoveryEdge(
                "recompute-0-1", 0, 1, 10.0, PrefixRecoveryKind.RECOMPUTE
            ),
            PrefixRecoveryEdge(
                "recompute-1-2", 1, 2, 10.0, PrefixRecoveryKind.RECOMPUTE
            ),
            PrefixRecoveryEdge(
                "recompute-2-3", 2, 3, 10.0, PrefixRecoveryKind.RECOMPUTE
            ),
            PrefixRecoveryEdge(
                "peer-0-1",
                0,
                1,
                1.0,
                PrefixRecoveryKind.PEER_ADOPT,
                action_id="peer-prefix",
            ),
            PrefixRecoveryEdge(
                "dram-1-3",
                1,
                3,
                2.0,
                PrefixRecoveryKind.STORE_DRAM,
                action_id="dram-suffix",
            ),
        ),
    )
    portfolio = ProtectionPortfolio(
        scenarios=(
            FailureScenario(
                failure_domain="d0",
                targets=(RecoveryTarget("target-d0"),),
                jobs=(
                    RecoveryJob(
                        "j0", "workflow-d0", "target-d0", 0.0, 30.0
                    ),
                ),
                recovery_problems=(ScenarioRecoveryProblem("j0", problem),),
            ),
        ),
        budgets=(ResourceBudget("protection_slots", 2.0),),
        actions=(
            PortfolioAction(
                "peer-prefix",
                _demand("protection_slots", 1.0),
                (),
            ),
            PortfolioAction(
                "dram-suffix",
                _demand("protection_slots", 1.0),
                (),
            ),
        ),
    )

    peer_only = portfolio.evaluate(frozenset({"peer-prefix"}))
    dram_only = portfolio.evaluate(frozenset({"dram-suffix"}))
    combined = portfolio.evaluate(
        frozenset({"peer-prefix", "dram-suffix"})
    )

    assert peer_only.risk.key == (21.0, 21.0)
    assert dram_only.risk.key == (12.0, 12.0)
    assert combined.risk.key == (3.0, 3.0)
    assert portfolio.greedy().selected_actions == (
        "dram-suffix",
        "peer-prefix",
    )


def test_dominant_resource_share_normalizes_action_gain() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(_scenario("d0", job_id="j0", service_s=10.0),),
        budgets=(
            ResourceBudget("peer_hbm_bytes", 10.0),
            ResourceBudget("copy_bytes", 10.0),
        ),
        actions=(
            PortfolioAction(
                "large-share",
                (
                    ResourceDemand("peer_hbm_bytes", 8.0),
                    ResourceDemand("copy_bytes", 1.0),
                ),
                _effect("d0", "j0", 1.0),
            ),
            PortfolioAction(
                "small-share",
                (
                    ResourceDemand("peer_hbm_bytes", 2.0),
                    ResourceDemand("copy_bytes", 1.0),
                ),
                _effect("d0", "j0", 1.0),
            ),
        ),
    )

    result = portfolio.greedy()

    assert portfolio.dominant_share("large-share") == 0.8
    assert portfolio.dominant_share("small-share") == 0.2
    assert result.selected_actions == ("small-share",)
    assert result.steps[0].added_action_id == "small-share"


def test_prefix_action_requires_a_dependency_closed_selection() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(_scenario("d0", job_id="j0", service_s=10.0),),
        budgets=(ResourceBudget("store_bytes", 2.0),),
        actions=(
            PortfolioAction(
                "retain-prefix",
                _demand("store_bytes", 1.0),
                _effect("d0", "j0", 10.0),
            ),
            PortfolioAction(
                "extend-prefix",
                _demand("store_bytes", 1.0),
                _effect("d0", "j0", 0.0),
                predecessors=("retain-prefix",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="missing predecessors"):
        portfolio.evaluate(frozenset({"extend-prefix"}))

    exact = portfolio.solve_exact()
    greedy = portfolio.greedy()

    assert exact.selected_actions == ("extend-prefix", "retain-prefix")
    assert exact.outcome.risk.key == (0.0, 0.0)
    assert exact.certificate.enumerated_subsets == 4
    assert exact.certificate.feasible_subsets == 3
    assert greedy.selected_actions == exact.selected_actions
    assert greedy.steps[0].added_action_id == "extend-prefix"
    assert greedy.steps[0].added_action_ids == (
        "extend-prefix",
        "retain-prefix",
    )
    assert greedy.steps[0].dominant_share == 1.0


def test_bundle_horizon_must_fit_every_new_prefix_member_lifetime() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(_scenario("d0", job_id="j0", service_s=10.0),),
        budgets=(ResourceBudget("store_bytes", 2.0),),
        actions=(
            PortfolioAction(
                "chunk-1",
                _demand("store_bytes", 1.0),
                _effect("d0", "j0", 10.0),
                completion_offset_s=1.0,
                valid_for_s=1.5,
            ),
            PortfolioAction(
                "chunk-2",
                _demand("store_bytes", 1.0),
                _effect("d0", "j0", 0.0),
                predecessors=("chunk-1",),
                completion_offset_s=2.0,
                valid_for_s=3.0,
            ),
        ),
    )

    result = portfolio.greedy()

    assert result.selected_actions == ()
    assert result.steps == ()
    with pytest.raises(ValueError, match="static exact oracle"):
        portfolio.solve_exact()


def test_break_before_make_rejects_a_risk_bearing_victim() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(_scenario("d0", job_id="j0", service_s=10.0),),
        budgets=(ResourceBudget("peer_hbm_bytes", 1.0),),
        actions=(
            PortfolioAction(
                "old-shadow",
                _demand("peer_hbm_bytes", 1.0),
                _effect("d0", "j0", 8.0),
            ),
            PortfolioAction(
                "better-shadow",
                _demand("peer_hbm_bytes", 1.0),
                _effect("d0", "j0", 1.0),
            ),
        ),
    )

    result = portfolio.greedy(initial_selected=frozenset({"old-shadow"}))

    assert result.selected_actions == ("old-shadow",)
    assert result.steps == ()


def test_break_before_make_can_replace_a_zero_loss_victim() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(_scenario("d0", job_id="j0", service_s=10.0),),
        budgets=(ResourceBudget("peer_hbm_bytes", 1.0),),
        actions=(
            PortfolioAction(
                "redundant-shadow",
                _demand("peer_hbm_bytes", 1.0),
                _effect("d0", "j0", 10.0),
            ),
            PortfolioAction(
                "better-shadow",
                _demand("peer_hbm_bytes", 1.0),
                _effect("d0", "j0", 1.0),
            ),
        ),
    )

    result = portfolio.greedy(
        initial_selected=frozenset({"redundant-shadow"})
    )

    assert result.selected_actions == ("better-shadow",)
    assert result.steps[0].removed_action_ids == ("redundant-shadow",)
    assert result.steps[0].risk_before == (10.0, 10.0)
    assert result.steps[0].risk_after == (1.0, 1.0)


def test_nonreleasable_action_cannot_be_used_as_a_swap_victim() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(_scenario("d0", job_id="j0", service_s=10.0),),
        budgets=(ResourceBudget("peer_hbm_bytes", 1.0),),
        actions=(
            PortfolioAction(
                "pinned-shadow",
                _demand("peer_hbm_bytes", 1.0),
                _effect("d0", "j0", 8.0),
                releasable=False,
            ),
            PortfolioAction(
                "better-shadow",
                _demand("peer_hbm_bytes", 1.0),
                _effect("d0", "j0", 1.0),
            ),
        ),
    )

    result = portfolio.greedy(
        initial_selected=frozenset({"pinned-shadow"})
    )

    assert result.selected_actions == ("pinned-shadow",)
    assert result.steps == ()


def test_action_gain_at_commit_consumes_only_positive_remaining_slack() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(
            _scenario(
                "d0", job_id="j0", service_s=5.0, deadline_s=3.0
            ),
        ),
        budgets=(ResourceBudget("copy_bytes", 1.0),),
        actions=(
            PortfolioAction(
                "copy",
                _demand("copy_bytes", 1.0),
                _effect("d0", "j0", 1.0),
                completion_offset_s=2.0,
                valid_for_s=3.0,
            ),
        ),
    )

    immediate_before = portfolio.evaluate(frozenset()).risk.key
    result = portfolio.greedy()

    assert immediate_before == (2.0, 2.0)
    assert result.selected_actions == ("copy",)
    assert result.steps[0].completion_offset_s == 2.0
    assert result.steps[0].risk_before == (4.0, 4.0)
    assert result.steps[0].risk_after == (0.0, 0.0)


def test_action_that_cannot_commit_before_expiry_is_not_admitted() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(_scenario("d0", job_id="j0", service_s=5.0),),
        budgets=(ResourceBudget("copy_bytes", 1.0),),
        actions=(
            PortfolioAction(
                "too-late",
                _demand("copy_bytes", 1.0),
                _effect("d0", "j0", 0.0),
                completion_offset_s=5.0,
                valid_for_s=4.0,
            ),
        ),
    )

    result = portfolio.greedy()

    assert result.selected_actions == ()
    assert result.steps == ()


def test_expired_effect_is_absent_but_capacity_stays_charged() -> None:
    portfolio = ProtectionPortfolio(
        scenarios=(_scenario("d0", job_id="j0", service_s=10.0),),
        budgets=(ResourceBudget("store_bytes", 1.0),),
        actions=(
            PortfolioAction(
                "short-lease",
                _demand("store_bytes", 1.0),
                _effect("d0", "j0", 0.0),
                valid_for_s=1.0,
            ),
        ),
    )
    selected = frozenset({"short-lease"})

    before_expiry = portfolio.evaluate(selected, fail_offset_s=1.0)
    after_expiry = portfolio.evaluate(selected, fail_offset_s=2.0)

    assert before_expiry.risk.key == (0.0, 0.0)
    assert after_expiry.risk.key == (10.0, 10.0)
    assert dict(portfolio.used_resources(selected)) == {"store_bytes": 1.0}


def test_exact_oracle_exposes_a_multi_item_greedy_gap() -> None:
    scenarios = [
        _scenario("sentinel", job_id="sentinel", service_s=100.0),
        _scenario("a", job_id="a", service_s=30.0),
        _scenario("b", job_id="b", service_s=24.0),
        _scenario("c", job_id="c", service_s=24.0),
    ]
    actions = (
        PortfolioAction(
            "item-a",
            _demand("capacity", 6.0),
            _effect("a", "a", 0.0),
        ),
        PortfolioAction(
            "item-b",
            _demand("capacity", 5.0),
            _effect("b", "b", 0.0),
        ),
        PortfolioAction(
            "item-c",
            _demand("capacity", 5.0),
            _effect("c", "c", 0.0),
        ),
    )
    portfolio = ProtectionPortfolio(
        scenarios=tuple(scenarios),
        budgets=(ResourceBudget("capacity", 10.0),),
        actions=actions,
    )

    greedy = portfolio.greedy()
    exact = portfolio.solve_exact()

    assert greedy.selected_actions == ("item-a",)
    assert greedy.outcome.risk.key == (100.0, 148.0)
    assert exact.selected_actions == ("item-b", "item-c")
    assert exact.outcome.risk.key == (100.0, 130.0)
    assert exact.certificate.status == "OPTIMAL"
    assert exact.certificate.objective == exact.certificate.best_bound
    assert exact.certificate.gap == 0.0
    assert len(exact.certificate.instance_sha256) == 64


def test_dependency_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="acyclic"):
        ProtectionPortfolio(
            scenarios=(_scenario("d0", job_id="j0", service_s=10.0),),
            budgets=(ResourceBudget("store_bytes", 2.0),),
            actions=(
                PortfolioAction(
                    "a",
                    _demand("store_bytes", 1.0),
                    _effect("d0", "j0", 5.0),
                    predecessors=("b",),
                ),
                PortfolioAction(
                    "b",
                    _demand("store_bytes", 1.0),
                    _effect("d0", "j0", 4.0),
                    predecessors=("a",),
                ),
            ),
        )
