from __future__ import annotations

from dataclasses import replace

import pytest

from sglsim.srt.managers.aegis_l1_l2 import (
    AegisBackupPlan,
    AegisL1L2Config,
    AegisRankBackupManager,
)
from sglsim.srt.managers.pipeline_parallel import DPBudgetSnapshot


def _plan(
    *,
    rid: str,
    tokens: int = 512,
    allowed_tiers: tuple[str, ...] = ("l1", "l2"),
    critical_probability: float = 0.9,
) -> AegisBackupPlan:
    return AegisBackupPlan(
        rid=rid,
        backup_key=rid,
        source_dp_rank=0,
        target_dp_rank=1,
        token_count=tokens,
        critical_probability=critical_probability,
        slack_ms=0.0,
        normal_l1_pressure=0.2,
        normal_l2_pressure=0.1,
        allowed_tiers=allowed_tiers,
    )


def test_algorithm2_prefers_l1_and_returns_authoritative_ack() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=4096,
            l2_total_tokens=8192,
            initial_alpha=0.5,
            initial_beta=0.5,
            chunk_tokens=256,
        ),
    )

    ack = manager.place(_plan(rid="req-0"))

    assert ack.ack is True
    assert ack.target_dp_rank == 1
    assert ack.l1_tokens == 512
    assert ack.l2_tokens == 0
    assert ack.recovery_action == "l1_backup"
    assert ack.safe_frontier_tokens == 512
    assert ack.expected_recovery_ms < 4.0 + 512 * 0.0108


def test_group_commit_materializes_only_source_committed_chunks() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=1.0,
            initial_beta=1.0,
            chunk_tokens=64,
        ),
    )
    reservation = manager.place(_plan(rid="committed", tokens=224))
    assert reservation.safe_frontier_tokens == 224

    committed = manager.materialize_committed_prefix(
        "committed",
        expected_version=1,
        source_committed_token_end=64,
        execution_id="exec-1",
        block_generation=0,
    )

    assert committed["materialization_scope"] == "committed_prefix"
    assert committed["group_commit_verified"] is True
    assert committed["source_committed_token_end"] == 64
    assert committed["safe_frontier_tokens"] == 64
    assert committed["l1_tokens"] + committed["l2_tokens"] == 64
    assert committed["reserved_token_count"] == 224
    assert committed["safe_frontier_tokens"] <= committed[
        "source_committed_token_end"
    ]


def test_group_commit_stops_at_first_unprotected_chunk() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=256,
            l2_total_tokens=256,
            initial_alpha=0.25,
            initial_beta=0.0,
            chunk_tokens=64,
        ),
    )
    reservation = manager.place(
        _plan(rid="hole", tokens=192, allowed_tiers=("l1",))
    )
    assert (reservation.l1_tokens, reservation.recompute_tokens) == (64, 128)

    committed = manager.materialize_committed_prefix(
        "hole",
        expected_version=1,
        source_committed_token_end=192,
        execution_id="exec-hole",
        block_generation=3,
    )

    assert committed["safe_frontier_tokens"] == 64
    assert committed["recompute_tokens"] == 128


def test_group_commit_rejects_stale_identity_and_uncommitted_future() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=1.0,
            initial_beta=1.0,
            chunk_tokens=64,
        ),
    )
    manager.place(_plan(rid="identity", tokens=128))
    manager.materialize_committed_prefix(
        "identity",
        expected_version=1,
        source_committed_token_end=64,
        execution_id="exec-current",
        block_generation=7,
    )

    with pytest.raises(ValueError, match="execution_id"):
        manager.materialize_committed_prefix(
            "identity",
            expected_version=1,
            source_committed_token_end=64,
            execution_id="exec-stale",
            block_generation=7,
        )
    with pytest.raises(ValueError, match="block_generation"):
        manager.materialize_committed_prefix(
            "identity",
            expected_version=1,
            source_committed_token_end=64,
            execution_id="exec-current",
            block_generation=6,
        )
    with pytest.raises(ValueError, match="outside the reserved"):
        manager.materialize_committed_prefix(
            "identity",
            expected_version=1,
            source_committed_token_end=129,
            execution_id="exec-current",
            block_generation=7,
        )
    with pytest.raises(ValueError, match="regressed"):
        manager.materialize_committed_prefix(
            "identity",
            expected_version=1,
            source_committed_token_end=32,
            execution_id="exec-current",
            block_generation=7,
        )


def test_dp_owned_budget_caps_stage_and_disables_rank_local_publication() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=2048,
            initial_alpha=1.0,
            initial_beta=1.0,
            chunk_tokens=64,
            window_requests=1,
        ),
    )
    budget = DPBudgetSnapshot(
        dp_rank=1,
        budget_version=4,
        window_id=7,
        alpha=0.25,
        beta=0.125,
        hbm_limit_tokens_per_stage=(64, 128, 192, 256),
        host_policy_limit_tokens=256,
        valid_from_logical_time=3.0,
        objective_id="exact-window-7",
        solver_certificate_id="optimal-certificate-7",
    )

    applied = manager.apply_dp_budget(budget, pp_rank=1)
    ack = manager.place(
        _plan(rid="dp-budgeted", tokens=256, allowed_tiers=("l1",))
    )
    state = manager.snapshot()

    assert applied["hbm_limit_tokens"] == 128
    assert (ack.l1_tokens, ack.recompute_tokens) == (128, 128)
    assert ack.alpha == 0.25
    assert ack.beta == 0.125
    assert ack.window_id == 7
    assert ack.window_update is not None
    assert ack.window_update["new_alpha"] == 0.25
    assert "D" not in ack.window_update["tiers"]["l1"]
    assert ack.window_update["tiers"]["l1"]["attempted_tokens"] == 256
    assert state["budget_version"] == 4
    assert state["budget_owner"] == "dp_controller"
    assert state["window_id"] == 7
    assert state["hbm_policy_limit_tokens"] == 128

    assert manager.apply_dp_budget(budget, pp_rank=1)["applied"] is False
    with pytest.raises(ValueError, match="stale"):
        manager.apply_dp_budget(replace(budget, budget_version=3), pp_rank=1)


def test_certified_exact_plan_executes_frozen_chunk_tiers_without_greedy() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=100,
            l2_total_tokens=100,
            chunk_tokens=2,
        ),
    )
    budget = DPBudgetSnapshot(
        dp_rank=1,
        budget_version=1,
        window_id=1,
        alpha=0.06,
        beta=0.06,
        hbm_limit_tokens_per_stage=(6, 6, 6, 6),
        host_policy_limit_tokens=6,
        valid_from_logical_time=0.0,
        objective_id="exact-objective",
        solver_certificate_id="highs:decision",
    )
    manager.apply_dp_budget(budget, pp_rank=0)
    plan = replace(
        _plan(rid="exact", tokens=10),
        dp_host_beta=0.06,
        placement_mode="certified_exact",
        exact_chunk_placements=(
            (0, 4, "HBM"),
            (1, 4, "HOST"),
            (2, 2, "RECOMPUTE"),
        ),
        solver_instance_sha256="instance",
        solver_decision_sha256="decision",
        solver_certificate_id="highs:decision",
    )

    ack = manager.place(plan)

    assert (ack.l1_tokens, ack.l2_tokens, ack.recompute_tokens) == (4, 4, 2)
    assert ack.placement_mode == "certified_exact"
    assert ack.exact_chunk_placements == plan.exact_chunk_placements
    assert ack.solver_decision_sha256 == "decision"
    snapshot = manager.snapshot()
    assert snapshot["attempted_l1_tokens"] == 4
    assert snapshot["attempted_l2_tokens"] == 4
    assert snapshot["resident_end_l1_tokens"] == 4
    assert snapshot["resident_end_l2_tokens"] == 4
    assert "D_l1_tokens" not in snapshot
    assert "O_l2_tokens" not in snapshot

    over_budget = replace(
        plan,
        rid="exact-over-budget",
        backup_key="exact-over-budget",
        token_count=3,
        exact_chunk_placements=((0, 3, "HBM"),),
    )
    with pytest.raises(RuntimeError, match="exceeds deployed DP budget"):
        manager.place(over_budget)


def test_algorithm2_never_overcommits_a_tier_within_one_multichunk_plan() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=0.25,
            initial_beta=0.25,
            chunk_tokens=256,
        ),
    )

    ack = manager.place(_plan(rid="req-large", tokens=1024))
    state = manager.snapshot()

    assert ack.l1_tokens <= state["l1_budget_tokens"]
    assert ack.l2_tokens <= state["l2_budget_tokens"]
    assert ack.l1_tokens + ack.l2_tokens + ack.recompute_tokens == 1024
    assert ack.recompute_tokens == 512


def test_l1_only_and_l2_only_ablation_never_use_the_other_tier() -> None:
    config = AegisL1L2Config(
        l1_total_tokens=4096,
        l2_total_tokens=8192,
        initial_alpha=0.5,
        initial_beta=0.5,
        chunk_tokens=256,
    )
    l1_ack = AegisRankBackupManager(dp_rank=1, config=config).place(
        _plan(rid="l1", allowed_tiers=("l1",))
    )
    l2_ack = AegisRankBackupManager(dp_rank=1, config=config).place(
        _plan(rid="l2", allowed_tiers=("l2",))
    )

    assert (l1_ack.l1_tokens, l1_ack.l2_tokens) == (512, 0)
    assert (l2_ack.l1_tokens, l2_ack.l2_tokens) == (0, 512)


def test_no_backup_ablation_returns_recompute_only() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(l1_total_tokens=4096, l2_total_tokens=8192),
    )

    ack = manager.place(_plan(rid="none", allowed_tiers=()))

    assert ack.recovery_action == "recompute"
    assert ack.safe_frontier_tokens == 0
    assert ack.recompute_tokens == 512


def test_historical_failure_hazard_rejects_negative_value_fallback_copy() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=4096,
            l2_total_tokens=8192,
            initial_alpha=0.5,
            initial_beta=0.5,
            chunk_tokens=256,
            failure_hazard_probability=0.0,
            cost_weight=0.05,
        ),
    )

    ack = manager.place(_plan(rid="hazard-zero"))

    assert ack.recovery_action == "recompute"
    assert ack.l1_tokens == 0
    assert ack.l2_tokens == 0
    assert ack.recompute_tokens == 512


def test_algorithm2_uses_topology_candidate_utility_for_tier_choice() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=4096,
            l2_total_tokens=4096,
            initial_alpha=0.5,
            initial_beta=0.5,
            chunk_tokens=256,
        ),
    )
    plan = replace(
        _plan(rid="topology-utility", tokens=512),
        gpu_candidate={
            "valid": True,
            "token_count": 512,
            "deadline_benefit_ms": 10.0,
            "write_ms": 40.0,
            "utility": -10.0,
        },
        host_candidate={
            "valid": True,
            "token_count": 512,
            "deadline_benefit_ms": 60.0,
            "write_ms": 10.0,
            "utility": 40.0,
        },
    )

    ack = manager.place(plan)

    assert ack.l1_tokens == 0
    assert ack.l2_tokens == 512
    assert ack.recovery_action == "l2_backup"


def test_algorithm3_consumes_window_duo_and_publishes_next_alpha_beta() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=4096,
            l2_total_tokens=8192,
            initial_alpha=0.5,
            initial_beta=0.5,
            chunk_tokens=256,
            window_requests=2,
            demand_ewma=1.0,
            publish_smoothing=1.0,
            hysteresis=0.0,
            burst_reserve_tokens=0,
        ),
    )

    manager.place(_plan(rid="req-0", tokens=512))
    ack = manager.place(_plan(rid="req-1", tokens=512))

    assert ack.window_update is not None
    assert ack.window_update["window_id"] == 1
    assert ack.window_update["tiers"]["l1"]["D"] == 1024
    assert ack.window_update["tiers"]["l1"]["U"] == 1024
    assert ack.window_update["tiers"]["l1"]["O"] == 0
    assert ack.d_l1_tokens == 1024
    assert ack.u_l1_tokens == 1024
    assert ack.alpha == 0.25
    assert manager.snapshot()["window_requests"] == 0


def test_algorithm3_uses_direct_capacity_miss_as_overflow_signal() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=0.25,
            initial_beta=0.25,
            chunk_tokens=256,
            window_requests=1,
            demand_ewma=1.0,
            publish_smoothing=1.0,
            hysteresis=0.0,
            burst_reserve_tokens=0,
        ),
    )

    first = manager.place(
        replace(
            _plan(
                rid="critical-resident",
                tokens=256,
                allowed_tiers=("l1",),
                critical_probability=0.9,
            ),
            normal_l1_pressure=0.0,
            normal_l2_pressure=0.0,
        )
    )
    assert first.l1_tokens == 256
    assert first.window_update["tiers"]["l1"]["O"] == 0

    rejected = manager.place(
        replace(
            _plan(
                rid="lower-density-candidate",
                tokens=256,
                allowed_tiers=("l1",),
                critical_probability=0.35,
            ),
            normal_l1_pressure=0.0,
            normal_l2_pressure=0.0,
        )
    )

    assert rejected.l1_tokens == 0
    assert rejected.recompute_tokens == 256
    assert rejected.o_l1_tokens == 256
    assert rejected.window_update["tiers"]["l1"]["D"] == 256
    assert rejected.window_update["tiers"]["l1"]["U"] == 256
    assert rejected.window_update["tiers"]["l1"]["O"] == 256
    assert rejected.window_update["tiers"]["l1"]["Z"] == 512
    assert rejected.alpha == 0.5
    assert rejected.beta >= rejected.alpha


def test_algorithm3_adaptive_window_waits_for_minimum_then_uses_time_trigger() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=4096,
            l2_total_tokens=4096,
            initial_alpha=0.5,
            initial_beta=0.25,
            chunk_tokens=256,
            window_requests=32,
            window_min_requests=2,
            window_max_requests=10,
            window_target_ms=100.0,
            window_overflow_trigger_tokens=256,
        ),
    )
    manager._window_started_monotonic -= 1.0

    first = manager.place(_plan(rid="time-0", tokens=256))
    second = manager.place(_plan(rid="time-1", tokens=256))

    assert first.window_update is None
    assert second.window_update is not None
    assert second.window_update["trigger"] == "target_time"
    assert second.window_update["alpha_scope"] == "dp_replica_pp_bottleneck"
    assert second.window_update["beta_scope"] == "dp_replica_policy"
    assert second.window_update["new_beta"] >= second.window_update["new_alpha"]


def test_algorithm3_alpha_shrink_revokes_low_value_gpu_hint() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=0.5,
            initial_beta=0.5,
            chunk_tokens=256,
            window_requests=1,
            demand_ewma=1.0,
            publish_smoothing=1.0,
            hysteresis=0.0,
            burst_reserve_tokens=0,
        ),
    )
    protected = manager.place(
        replace(
            _plan(rid="old", tokens=256, allowed_tiers=("l1",)),
            normal_l1_pressure=0.0,
            normal_l2_pressure=0.0,
        )
    )
    assert protected.l1_tokens == 256
    manager.pop_pending_hint_updates()

    idle = manager.place(
        replace(
            _plan(rid="idle", tokens=256, allowed_tiers=()),
            normal_l1_pressure=0.0,
            normal_l2_pressure=0.0,
        )
    )
    updates = manager.pop_pending_hint_updates()

    assert idle.window_update["new_alpha"] == 0.0
    assert idle.window_update["l1_evicted_backup_keys"] == ["old"]
    assert manager.l1_used_tokens == 0
    assert any(
        hint["backup_key"] == "old"
        and hint["l1_tokens"] == 0
        and hint["recompute_tokens"] == 256
        for hint in updates
    )


def test_algorithm3_max_alpha_preserves_execution_capacity_under_overflow() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=0.10,
            max_alpha=0.25,
            initial_beta=0.25,
            chunk_tokens=256,
            window_requests=1,
            demand_ewma=1.0,
            publish_smoothing=1.0,
            hysteresis=0.0,
            safety_margin=0.0,
            burst_reserve_tokens=0,
        ),
    )

    overflow = manager.place(
        replace(
            _plan(rid="overflow", tokens=1024, allowed_tiers=("l1",)),
            normal_l1_pressure=0.0,
            normal_l2_pressure=0.0,
        )
    )

    assert overflow.window_update is not None
    assert overflow.window_update["new_alpha"] == 0.25
    assert overflow.window_update["new_beta"] >= overflow.window_update["new_alpha"]
    assert overflow.window_update["tiers"]["l1"]["configured_cap"] == 0.25
    assert manager.snapshot()["l1_budget_tokens"] == 256

    admitted = manager.place(
        replace(
            _plan(rid="bounded", tokens=256, allowed_tiers=("l1",)),
            normal_l1_pressure=0.0,
            normal_l2_pressure=0.0,
        )
    )
    assert admitted.l1_tokens == 256
    assert admitted.alpha == 0.25


def test_gpu_failure_invalidates_l1_hints_but_preserves_l2() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=0.25,
            initial_beta=0.5,
            chunk_tokens=256,
        ),
    )
    ack = manager.place(_plan(rid="mixed", tokens=512))
    assert (ack.l1_tokens, ack.l2_tokens) == (256, 256)

    result = manager.clear_l1_for_gpu_failure()
    hint = result["updated_hints"][0]

    assert result["cleared_l1_backup_tokens"] == 256
    assert result["preserved_l2_backup_tokens"] == 256
    assert hint["l1_tokens"] == 0
    assert hint["l2_tokens"] == 256
    assert hint["recompute_tokens"] == 256
    assert hint["recovery_action"] == "l2_backup"
    assert manager.snapshot()["l1_used_tokens"] == 0


def test_controller_host_eviction_invalidates_matching_l2_version_only() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=0.0,
            initial_beta=1.0,
            chunk_tokens=256,
        ),
    )
    manager.place(_plan(rid="host", tokens=512, allowed_tiers=("l2",)))

    stale = manager.invalidate_l2_for_host_lease("host", expected_version=2)
    assert stale is None
    assert manager.l2_used_tokens == 512

    hint = manager.invalidate_l2_for_host_lease("host", expected_version=1)
    assert hint is not None
    assert hint["l2_tokens"] == 0
    assert hint["recompute_tokens"] == 512
    assert manager.l2_used_tokens == 0


def test_rank_manager_ignores_stale_backup_plan_version() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=1.0,
            initial_beta=0.0,
        ),
    )
    current = manager.place(replace(_plan(rid="same", tokens=256), version=2))
    stale = manager.place(replace(_plan(rid="same", tokens=512), version=1))

    assert current.version == 2
    assert stale.version == 2
    assert stale.l1_tokens == 256
    assert manager.snapshot()["entries"] == 1


def test_algorithm2_replaces_lower_density_entry_and_updates_old_hint() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=0.25,
            initial_beta=0.0,
            chunk_tokens=256,
        ),
    )
    low = manager.place(
        _plan(
            rid="low",
            tokens=256,
            allowed_tiers=("l1",),
            critical_probability=0.35,
        )
    )
    assert low.l1_tokens == 256

    high = manager.place(
        _plan(
            rid="high",
            tokens=256,
            allowed_tiers=("l1",),
            critical_probability=0.9,
        )
    )
    updates = manager.pop_pending_hint_updates()

    assert high.l1_tokens == 256
    assert len(updates) == 1
    assert updates[0]["backup_key"] == "low"
    assert updates[0]["l1_tokens"] == 0
    assert updates[0]["recompute_tokens"] == 256
    assert updates[0]["recovery_action"] == "recompute"


def test_cached_usage_matches_entries_after_replace_and_gpu_clear() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=2048,
            l2_total_tokens=4096,
            initial_alpha=0.5,
            initial_beta=0.5,
            chunk_tokens=256,
        ),
    )
    manager.place(_plan(rid="a", tokens=768))
    manager.place(_plan(rid="b", tokens=512, allowed_tiers=("l2",)))
    manager.place(_plan(rid="a", tokens=256))

    assert manager.l1_used_tokens == sum(
        entry.l1_tokens for entry in manager._entries.values()
    )
    assert manager.l2_used_tokens == sum(
        entry.l2_tokens for entry in manager._entries.values()
    )


def test_lineage_supersede_releases_rank_local_capacity_version_safely() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=0.5,
            initial_beta=0.5,
            chunk_tokens=256,
        ),
    )
    plan = replace(
        _plan(rid="old-frontier", tokens=512),
        workflow_id="workflow-a",
        agent_id="main",
        worker_seq=7,
        version=2,
    )
    ack = manager.place(plan)
    assert ack.workflow_id == "workflow-a"
    assert ack.agent_id == "main"
    assert ack.worker_seq == 7
    before = manager.snapshot()
    assert before["entries"] == 1
    assert before["l1_used_tokens"] + before["l2_used_tokens"] == 512

    stale = manager.supersede(
        "old-frontier",
        expected_version=1,
        superseded_by="new-frontier",
    )
    assert stale["released"] is False
    assert manager.snapshot()["entries"] == 1

    released = manager.supersede(
        "old-frontier",
        expected_version=2,
        superseded_by="new-frontier",
    )
    assert released["released"] is True
    assert released["released_l1_tokens"] + released["released_l2_tokens"] == 512
    assert released["worker_seq"] == 7
    assert manager.snapshot()["entries"] == 0
    assert manager.l1_used_tokens == 0
    assert manager.l2_used_tokens == 0


def test_algorithm2_critical_host_floor_diversifies_direct_route_l1() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=4096,
            l2_total_tokens=4096,
            initial_alpha=1.0,
            initial_beta=1.0,
            chunk_tokens=256,
            critical_host_floor_fraction=0.5,
            critical_host_floor_probability=0.75,
        ),
    )
    plan = replace(
        _plan(rid="critical-diversified", tokens=512),
        gpu_candidate={
            "valid": True,
            "token_count": 512,
            "deadline_benefit_ms": 100.0,
            "write_ms": 1.0,
            "utility": 100.0,
            "direct_retry_route_required": True,
        },
        host_candidate={
            "valid": True,
            "token_count": 512,
            "deadline_benefit_ms": 80.0,
            "write_ms": 2.0,
            # The explicit diversification floor may spend a bounded amount
            # on a route-independent copy even when standalone utility is
            # slightly negative.
            "utility": -5.0,
        },
    )

    ack = manager.place(plan)

    assert ack.l1_tokens == 256
    assert ack.l2_tokens == 256
    assert ack.recovery_action == "l1_l2_backup"
    assert ack.critical_host_floor_tokens == 256
    assert ack.critical_host_floor_fraction == 0.5

    manager.clear_l1_for_gpu_failure()
    assert manager.l1_used_tokens == 0
    assert manager.l2_used_tokens == sum(
        entry.l2_tokens for entry in manager._entries.values()
    )
