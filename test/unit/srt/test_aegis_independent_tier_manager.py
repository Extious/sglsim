from __future__ import annotations

import pytest

from sglsim.srt.managers.aegis_independent_tier_manager import (
    AegisIndependentTierRankManager,
)
from sglsim.srt.managers.aegis_l1_l2 import AegisBackupPlan
from sglsim.srt.managers.pipeline_parallel import DPBudgetSnapshot
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS


def _raw() -> dict:
    return {
        "design_semantics": DESIGN_SEMANTICS,
        "committed_copy_protocol": True,
        "chunk_tokens": 1024,
        "initial_alpha": 1.0,
        "min_alpha": 0.1,
        "max_alpha": 1.0,
        "window_requests": 100,
        "recompute_setup_ms": 0.0,
        "recompute_ms_per_token": 0.010,
        "l1_restore_setup_ms": 0.0,
        "l1_restore_ms_per_token": 0.001,
        "l2_restore_setup_ms": 0.0,
        "l2_restore_ms_per_token": 0.003,
    }


def _plan(request_id: str = "latest") -> AegisBackupPlan:
    return AegisBackupPlan(
        rid=request_id,
        backup_key=request_id,
        source_dp_rank=0,
        target_dp_rank=1,
        token_count=1024,
        critical_probability=1.0,
        slack_ms=0.0,
        route_version=3,
        required_pp_size=1,
        model_revision="model-rev",
        kv_abi="kv-abi",
        parallel_layout="dp2-tp1-pp1",
    )


def test_latest_rank_manager_reservation_has_no_recovery_value_until_commit() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=_raw(),
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
        now_fn=lambda: 10.0,
    )

    ack = manager.place(_plan())

    assert ack.design_semantics == DESIGN_SEMANTICS
    assert ack.independent_tier_results is True
    assert ack.l1_tokens == 1024
    assert ack.l2_tokens == 1024
    assert ack.hbm_pending_tokens == 1024
    assert ack.dram_pending_tokens == 1024
    assert ack.safe_frontier_tokens == 0
    assert ack.recovery_action == "recompute"
    assert ack.hbm_copy_path == "source_d2d"
    assert ack.dram_copy_path == "source_d2h"


def test_latest_rank_manager_group_commit_counts_overlapping_tiers_once() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=_raw(),
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
        now_fn=lambda: 11.0,
    )
    manager.place(_plan())

    payload = manager.materialize_committed_prefix(
        "latest",
        expected_version=1,
        source_committed_token_end=1024,
        execution_id="copy-exec-1",
        block_generation=0,
    )

    assert payload["materialization_scope"] == "committed_prefix"
    assert payload["group_commit_verified"] is True
    assert payload["l1_tokens"] == 1024
    assert payload["l2_tokens"] == 1024
    assert payload["safe_frontier_tokens"] == 1024
    assert payload["continuous_recoverable_prefix_tokens"] == 1024
    assert payload["recompute_tokens"] == 0
    assert payload["residency"] == "HBM_AND_DRAM"
    assert payload["recovery_action"] == "l1_backup"
    assert payload["cross_tier_transfer"] is False
    manager.controller.validate_invariants()


def test_latest_rank_manager_commits_physical_tiers_independently() -> None:
    raw = _raw()
    raw["prefix_chunk_tokens"] = 64
    raw["decode_chunk_tokens"] = 64
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=raw,
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )
    base = _plan("tier-specific")
    manager.place(AegisBackupPlan(**{**base.__dict__, "token_count": 128}))

    hbm = manager.materialize_committed_prefix(
        "tier-specific",
        expected_version=1,
        source_committed_token_end=64,
        execution_id="copy-exec-tier-specific",
        block_generation=0,
        completed_tiers=["hbm"],
    )

    assert hbm["completed_tiers"] == ["hbm"]
    assert hbm["tier_commit_scope"] == "physical_completed_tiers"
    assert hbm["copy_completion_evidence"] == (
        "sglsim_physical_independent_tier_group_commit"
    )
    assert hbm["l1_tokens"] == 64
    assert hbm["l2_tokens"] == 0
    assert hbm["hbm_pending_tokens"] == 64
    assert hbm["dram_pending_tokens"] == 128
    assert hbm["tier_committed_source_frontiers"] == {
        "hbm": 64,
        "dram": 0,
    }

    # The two physical queues may finish different frontiers out of order.
    # A slower DRAM group at 64 must not be rejected merely because HBM has
    # already advanced farther.
    hbm_advanced = manager.materialize_committed_prefix(
        "tier-specific",
        expected_version=1,
        source_committed_token_end=128,
        execution_id="copy-exec-tier-specific",
        block_generation=0,
        completed_tiers=["hbm"],
    )
    dram_late = manager.materialize_committed_prefix(
        "tier-specific",
        expected_version=1,
        source_committed_token_end=64,
        execution_id="copy-exec-tier-specific",
        block_generation=0,
        completed_tiers=["dram"],
    )

    assert hbm_advanced["l1_tokens"] == 128
    assert hbm_advanced["l2_tokens"] == 0
    assert dram_late["source_committed_token_end"] == 128
    assert dram_late["tier_source_committed_token_end"] == 64
    assert dram_late["l1_tokens"] == 128
    assert dram_late["l2_tokens"] == 64
    assert dram_late["dram_pending_tokens"] == 64
    assert dram_late["tier_committed_source_frontiers"] == {
        "hbm": 128,
        "dram": 64,
    }
    manager.controller.validate_invariants()


def test_latest_rank_manager_tier_commit_is_idempotent_and_fail_closed() -> None:
    raw = _raw()
    raw["prefix_chunk_tokens"] = 64
    raw["decode_chunk_tokens"] = 64
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=raw,
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )
    base = _plan("tier-validation")
    manager.place(AegisBackupPlan(**{**base.__dict__, "token_count": 128}))
    kwargs = {
        "expected_version": 1,
        "source_committed_token_end": 64,
        "execution_id": "copy-exec-tier-validation",
        "block_generation": 0,
        "completed_tiers": ["dram"],
    }

    first = manager.materialize_committed_prefix("tier-validation", **kwargs)
    duplicate = manager.materialize_committed_prefix("tier-validation", **kwargs)

    assert duplicate["l1_tokens"] == first["l1_tokens"] == 0
    assert duplicate["l2_tokens"] == first["l2_tokens"] == 64
    assert duplicate["hbm_pending_tokens"] == 128
    assert duplicate["dram_pending_tokens"] == 64
    with pytest.raises(ValueError, match="dram committed prefix regressed"):
        manager.materialize_committed_prefix(
            "tier-validation",
            **{
                **kwargs,
                "source_committed_token_end": 0,
            },
        )
    with pytest.raises(ValueError, match="non-empty tier list"):
        manager.materialize_committed_prefix(
            "tier-validation",
            **{**kwargs, "completed_tiers": []},
        )
    with pytest.raises(ValueError, match="expected hbm or dram"):
        manager.materialize_committed_prefix(
            "tier-validation",
            **{**kwargs, "completed_tiers": ["hbm_to_dram"]},
        )
    manager.controller.validate_invariants()


def test_latest_rank_manager_host_invalidation_preserves_hbm_only() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=_raw(),
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )
    manager.place(_plan("host-loss"))
    manager.materialize_committed_prefix(
        "host-loss",
        expected_version=1,
        source_committed_token_end=1024,
        execution_id="copy-exec-host",
        block_generation=0,
    )

    hint = manager.invalidate_l2_for_host_lease(
        "host-loss", expected_version=1
    )

    assert hint is not None
    assert hint["l1_tokens"] == 1024
    assert hint["l2_tokens"] == 0
    assert hint["safe_frontier_tokens"] == 1024
    assert hint["residency"] == "HBM_ONLY"
    manager.controller.validate_invariants()


def test_latest_rank_manager_applies_hbm_and_dram_limits_independently() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=_raw(),
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )
    manager.place(_plan("budgeted"))
    manager.materialize_committed_prefix(
        "budgeted",
        expected_version=1,
        source_committed_token_end=1024,
        execution_id="copy-exec-budget",
        block_generation=0,
    )
    budget = DPBudgetSnapshot(
        dp_rank=1,
        budget_version=2,
        window_id=4,
        alpha=0.125,
        beta=0.25,
        hbm_limit_tokens_per_stage=(512,),
        host_policy_limit_tokens=1024,
        valid_from_logical_time=0.0,
        objective_id="latest-budget",
        solver_certificate_id="latest-budget-cert",
    )

    result = manager.apply_dp_budget(budget, pp_rank=0)
    state = manager.snapshot()

    assert result["hbm_limit_tokens"] == 512
    assert result["host_policy_limit_tokens"] == 1024
    assert state["l1_used_tokens"] <= 512
    assert state["l2_used_tokens"] == 1024
    assert state["design_semantics"] == DESIGN_SEMANTICS
    releases = manager.pop_pending_physical_releases()
    assert releases == [
        {
            "request_backup_key": "budgeted",
            "stage_backup_key": "budgeted",
            "version": 1,
            "route_version": 3,
            "target_dp_rank": 1,
            "pp_rank": 0,
            "tier": "HBM",
            "ordinal": 0,
            "token_start": 0,
            "token_end": 1024,
            "reason": "dp_budget_hbm_reclaim",
        }
    ]
    manager.controller.validate_invariants()


def test_latest_rank_manager_materializes_all_reserved_chunks() -> None:
    raw = _raw()
    raw["chunk_tokens"] = 256
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=raw,
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )

    reservation = manager.place(_plan("multi-chunk"))
    committed = manager.materialize_committed_prefix(
        "multi-chunk",
        expected_version=1,
        source_committed_token_end=1024,
        execution_id="copy-exec-multi",
        block_generation=0,
    )

    assert reservation.reserved_l1_tokens == 1024
    assert reservation.reserved_l2_tokens == 1024
    assert committed["l1_tokens"] == 1024
    assert committed["l2_tokens"] == 1024
    assert committed["safe_frontier_tokens"] == 1024
    assert committed["hbm_pending_tokens"] == 0
    assert committed["dram_pending_tokens"] == 0
    manager.controller.validate_invariants()


def test_latest_rank_manager_honors_independent_tier_ablation() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=_raw(),
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )
    plan = AegisBackupPlan(
        **{
            **_plan("dram-only").__dict__,
            "allowed_tiers": ("l2",),
        }
    )

    reservation = manager.place(plan)
    committed = manager.materialize_committed_prefix(
        "dram-only",
        expected_version=1,
        source_committed_token_end=1024,
        execution_id="copy-exec-dram-only",
        block_generation=0,
    )

    assert reservation.reserved_l1_tokens == 0
    assert reservation.reserved_l2_tokens == 1024
    assert committed["residency"] == "DRAM_ONLY"
    assert committed["safe_frontier_tokens"] == 1024


def test_latest_rank_manager_target_failure_aborts_pending_hbm_only() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=_raw(),
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )
    manager.place(_plan("pending-target-loss"))

    result = manager.clear_l1_for_gpu_failure()
    state = manager.snapshot()

    assert result["cleared_l1_backup_tokens"] == 1024
    assert result["affected_backup_entries"] == 1
    assert state["l1_used_tokens"] == 0
    assert state["l2_used_tokens"] == 1024
    hint = result["updated_hints"][0]
    assert hint["hbm_pending_tokens"] == 0
    assert hint["dram_pending_tokens"] == 1024
    releases = manager.pop_pending_physical_releases()
    assert len(releases) == 1
    assert releases[0]["tier"] == "HBM"
    assert releases[0]["token_start"] is None
    assert releases[0]["token_end"] is None
    manager.controller.validate_invariants()


def test_latest_rank_manager_complete_request_emits_versioned_terminal_release(
) -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=_raw(),
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )
    manager.place(_plan("completed"))
    manager.materialize_committed_prefix(
        "completed",
        expected_version=1,
        source_committed_token_end=1024,
        execution_id="copy-exec-completed",
        block_generation=0,
    )

    stale = manager.complete_request("completed", expected_version=2)
    assert stale["completed"] is False
    assert manager.pop_pending_physical_releases() == []

    completed = manager.complete_request("completed", expected_version=1)
    releases = manager.pop_pending_physical_releases()

    assert completed["completed"] is True
    assert completed["released_l1_tokens"] == 1024
    assert completed["released_l2_tokens"] == 1024
    assert releases == [
        {
            "request_backup_key": "completed",
            "stage_backup_key": "completed",
            "version": 1,
            "route_version": 3,
            "target_dp_rank": 1,
            "pp_rank": 0,
            "tier": None,
            "ordinal": None,
            "token_start": None,
            "token_end": None,
            "reason": "request_completed",
        }
    ]
    assert manager.snapshot()["entries"] == 0
    manager.controller.validate_invariants()


def test_latest_rank_manager_supersede_is_version_safe_and_releases_both_tiers(
) -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=_raw(),
        hbm_total_tokens=4096,
        dram_total_tokens=4096,
    )
    manager.place(_plan("superseded"))

    stale = manager.supersede(
        "superseded", expected_version=2, superseded_by="new-lineage"
    )
    assert stale["released"] is False
    assert manager.pop_pending_physical_releases() == []

    current = manager.supersede(
        "superseded", expected_version=1, superseded_by="new-lineage"
    )
    releases = manager.pop_pending_physical_releases()

    assert current["released"] is True
    assert len(releases) == 1
    assert releases[0]["tier"] is None
    assert releases[0]["version"] == 1
    assert releases[0]["reason"] == "backup_frontier_superseded"
    assert manager.snapshot()["entries"] == 0
    manager.controller.validate_invariants()


def test_latest_rank_manager_defaults_to_prefix_then_decode_granularity() -> None:
    raw = _raw()
    raw.pop("chunk_tokens")
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config=raw,
        hbm_total_tokens=16384,
        dram_total_tokens=16384,
    )
    base = _plan("granularity")
    plan = AegisBackupPlan(**{**base.__dict__, "token_count": 4608})

    manager.place(plan)
    specs = manager._entries["granularity"].specs

    assert [(spec.token_start, spec.token_end) for spec in specs] == [
        (0, 4096),
        (4096, 4352),
        (4352, 4608),
    ]
    assert manager.sealed_source_frontier(
        4095, request_finished=False, reserved_tokens=4608
    ) == 0
    assert manager.sealed_source_frontier(
        4351, request_finished=False, reserved_tokens=4608
    ) == 4096
    assert manager.sealed_source_frontier(
        4352, request_finished=False, reserved_tokens=4608
    ) == 4352
    assert manager.sealed_source_frontier(
        4477, request_finished=True, reserved_tokens=4608
    ) == 4477
