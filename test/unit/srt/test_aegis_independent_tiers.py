from __future__ import annotations

from dataclasses import replace

import pytest

from sglsim.srt.simulation.aegis_independent_tiers import (
    DESIGN_SEMANTICS,
    AegisIndependentTierController,
    ChunkSpec,
    CopyState,
    IndependentTierConfig,
    ProtectionTier,
    RecoveryTiming,
    RequestRecoveryProfile,
    Residency,
)


def _config(
    *,
    hbm_bytes: int = 16_384,
    dram_bytes: int = 16_384,
    alpha: float = 1.0,
    decrease: float = 0.5,
) -> IndependentTierConfig:
    return IndependentTierConfig(
        hbm_capacity_bytes=hbm_bytes,
        dram_capacity_bytes=dram_bytes,
        initial_hbm_alpha=alpha,
        minimum_hbm_alpha=0.1,
        maximum_hbm_alpha=1.0,
        aimd_decrease_factor=decrease,
        aimd_increase_step=0.1,
        aimd_stable_windows=2,
        recompute=RecoveryTiming(0.0, 0.010),
        hbm_load=RecoveryTiming(0.0, 0.001),
        dram_load=RecoveryTiming(0.0, 0.003),
        hbm_copy=RecoveryTiming(0.1, 0.0004),
        dram_copy=RecoveryTiming(0.2, 0.0012),
    )


def _profile(
    request_id: str,
    *,
    tokens: int = 1024,
    slack_ms: float = 0.0,
    target: str = "worker-1",
    route_version: int = 7,
) -> RequestRecoveryProfile:
    return RequestRecoveryProfile(
        request_id=request_id,
        source_worker="worker-0",
        source_failure_domain="node-0",
        hbm_target=target,
        hbm_failure_domain=f"{target}-domain",
        route_version=route_version,
        source_token_end=tokens,
        slack_ms=slack_ms,
        model_revision="qwen-test-revision",
        kv_abi="kv-abi-v1",
        parallel_layout="dp2-tp1-pp1",
    )


def _chunk(
    request_id: str,
    *,
    ordinal: int = 0,
    token_start: int = 0,
    token_end: int = 1024,
    byte_size: int = 1024,
    target: str = "worker-1",
    route_version: int = 7,
    shard_count: int = 2,
) -> ChunkSpec:
    return ChunkSpec(
        request_id=request_id,
        ordinal=ordinal,
        token_start=token_start,
        token_end=token_end,
        byte_size=byte_size,
        route_version=route_version,
        source_worker="worker-0",
        hbm_target=target,
        model_revision="qwen-test-revision",
        kv_abi="kv-abi-v1",
        parallel_layout="dp2-tp1-pp1",
        shard_count=shard_count,
    )


def _commit(
    controller: AegisIndependentTierController,
    spec: ChunkSpec,
    *tiers: ProtectionTier,
) -> None:
    for tier in tiers:
        controller.complete_copy(
            tier,
            spec.request_id,
            spec.ordinal,
            controller.completion_proof(spec, tier),
        )


def test_independent_hbm_and_dram_paths_commit_to_combined_residency() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("both"))
    spec = _chunk("both")

    decision = controller.admit_chunk(spec)
    assert decision.hbm.admitted is True
    assert decision.dram.admitted is True
    assert decision.hbm.copy_kind == "source_d2d"
    assert decision.dram.copy_kind == "source_d2h"
    assert decision.intended_residency is Residency.HBM_AND_DRAM
    assert controller.residency("both", 0) is Residency.BYPASS

    _commit(controller, spec, ProtectionTier.HBM, ProtectionTier.DRAM)

    assert controller.residency("both", 0) is Residency.HBM_AND_DRAM
    snapshot = controller.snapshot()
    assert snapshot["design_semantics"] == DESIGN_SEMANTICS
    pending = [
        event
        for event in snapshot["events"]
        if event["kind"] == "tier_copy_pending"
    ]
    assert {event["copy_kind"] for event in pending} == {
        "source_d2d",
        "source_d2h",
    }
    assert all(event["source_owned"] is True for event in pending)
    assert all(event["base_hbm_chunk_id"] is None for event in pending)
    assert all(event["writeback_cost_ms"] == 0.0 for event in pending)
    controller.validate_invariants()


@pytest.mark.parametrize(
    "allowed,expected",
    [
        ((ProtectionTier.HBM, ProtectionTier.DRAM), Residency.HBM_AND_DRAM),
        ((ProtectionTier.HBM,), Residency.HBM_ONLY),
        ((ProtectionTier.DRAM,), Residency.DRAM_ONLY),
        ((), Residency.BYPASS),
    ],
)
def test_independent_admission_exercises_all_four_residency_outcomes(
    allowed: tuple[ProtectionTier, ...], expected: Residency
) -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("four"))
    spec = _chunk("four")

    decision = controller.admit_chunk(spec, allowed_tiers=allowed)
    for tier, admission in (
        (ProtectionTier.HBM, decision.hbm),
        (ProtectionTier.DRAM, decision.dram),
    ):
        if admission.admitted:
            _commit(controller, spec, tier)

    assert decision.intended_residency is expected
    assert controller.residency("four", 0) is expected
    controller.validate_invariants()


def test_each_tier_gain_is_unchanged_by_the_other_tier_future_path() -> None:
    both = AegisIndependentTierController(_config())
    both.register_request(_profile("same"))
    both_decision = both.admit_chunk(_chunk("same"))

    hbm_only = AegisIndependentTierController(_config())
    hbm_only.register_request(_profile("same"))
    hbm_decision = hbm_only.admit_chunk(
        _chunk("same"), allowed_tiers=(ProtectionTier.HBM,)
    )

    dram_only = AegisIndependentTierController(_config())
    dram_only.register_request(_profile("same"))
    dram_decision = dram_only.admit_chunk(
        _chunk("same"), allowed_tiers=(ProtectionTier.DRAM,)
    )

    assert hbm_decision.hbm.gain_ms == both_decision.hbm.gain_ms
    assert dram_decision.dram.gain_ms == both_decision.dram.gain_ms
    assert all(
        record["base_hbm_chunk_id"] is None
        and record["writeback_cost_ms"] == 0.0
        for pool in both.snapshot()["pools"].values()
        for record in pool["records"]
    )


def test_one_copy_failure_does_not_abort_the_other_tier() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("partial"))
    spec = _chunk("partial")
    controller.admit_chunk(spec)

    bad_hbm = replace(
        controller.completion_proof(spec, ProtectionTier.HBM),
        transferred_bytes=0,
        acked_shards=(),
        transfer_succeeded=False,
    )
    failed = controller.complete_copy(
        ProtectionTier.HBM, "partial", 0, bad_hbm
    )
    assert failed.state is CopyState.ABORTED

    committed = controller.complete_copy(
        ProtectionTier.DRAM,
        "partial",
        0,
        controller.completion_proof(spec, ProtectionTier.DRAM),
    )
    assert committed.state is CopyState.COMMITTED
    assert controller.residency("partial", 0) is Residency.DRAM_ONLY
    controller.validate_invariants()


def test_completion_ack_is_bound_to_exact_tier_path_and_pending_record() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("path-bound"))
    spec = _chunk("path-bound")
    controller.admit_chunk(spec)

    cross_tier = controller.completion_proof(spec, ProtectionTier.DRAM)
    rejected = controller.complete_copy(
        ProtectionTier.HBM, spec.request_id, spec.ordinal, cross_tier
    )

    assert rejected.state is CopyState.ABORTED
    assert controller.copy_state(
        spec.request_id, spec.ordinal, ProtectionTier.DRAM
    ) is CopyState.PENDING
    committed = controller.complete_copy(
        ProtectionTier.DRAM,
        spec.request_id,
        spec.ordinal,
        controller.completion_proof(spec, ProtectionTier.DRAM),
    )
    assert committed.state is CopyState.COMMITTED
    assert controller.residency(spec.request_id, spec.ordinal) is Residency.DRAM_ONLY
    commit_event = next(
        event
        for event in reversed(controller.snapshot()["events"])
        if event["kind"] == "tier_copy_committed"
    )
    assert commit_event["copy_kind"] == "source_d2h"
    assert commit_event["source_worker"] == spec.source_worker
    assert commit_event["destination_id"] == controller.config.dram_pool_id
    assert commit_event["independent_ack"] is True


@pytest.mark.parametrize(
    "changes",
    [
        {"tier": ProtectionTier.DRAM},
        {"pending_id": "store-dram:other:0"},
        {"copy_kind": "source_d2h"},
        {"source_worker": "target-owned-worker"},
        {"destination_id": "store-dram"},
        {"source_owned": False},
    ],
)
def test_hbm_ack_fails_closed_on_each_path_identity_tamper(
    changes: dict[str, object]
) -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("tamper"))
    spec = _chunk("tamper")
    controller.admit_chunk(spec)
    proof = replace(
        controller.completion_proof(spec, ProtectionTier.HBM), **changes
    )

    result = controller.complete_copy(
        ProtectionTier.HBM, spec.request_id, spec.ordinal, proof
    )

    assert result.state is CopyState.ABORTED
    assert controller.copy_state(
        spec.request_id, spec.ordinal, ProtectionTier.DRAM
    ) is CopyState.PENDING


def test_hbm_only_is_valid_without_a_dram_prerequisite() -> None:
    controller = AegisIndependentTierController(
        _config(hbm_bytes=2048, dram_bytes=1)
    )
    controller.register_request(_profile("hbm-only"))
    spec = _chunk("hbm-only")
    decision = controller.admit_chunk(spec)

    assert decision.hbm.admitted is True
    assert decision.dram.admitted is False
    _commit(controller, spec, ProtectionTier.HBM)
    assert controller.residency("hbm-only", 0) is Residency.HBM_ONLY
    controller.validate_invariants()


def test_slack_update_is_metadata_only_and_lazy() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("lazy", slack_ms=0.0))
    spec = _chunk("lazy")
    controller.admit_chunk(spec)
    _commit(controller, spec, ProtectionTier.HBM, ProtectionTier.DRAM)
    before = controller.snapshot()

    controller.update_slack("lazy", 3.5)
    after = controller.snapshot()

    assert after["pools"]["HBM"]["used_bytes"] == before["pools"]["HBM"][
        "used_bytes"
    ]
    assert after["pools"]["DRAM"]["used_bytes"] == before["pools"]["DRAM"][
        "used_bytes"
    ]
    event = after["events"][-1]
    assert event["kind"] == "slack_metadata_updated"
    assert event["physical_movement"] is False
    controller.validate_invariants()


def test_slack_landlord_replaces_lower_value_pool_local_tail() -> None:
    controller = AegisIndependentTierController(
        _config(hbm_bytes=1024, dram_bytes=1024)
    )
    controller.register_request(_profile("low", slack_ms=8.0))
    low_spec = _chunk("low")
    low = controller.admit_chunk(low_spec)
    assert low.hbm.admitted and low.dram.admitted
    _commit(controller, low_spec, ProtectionTier.HBM, ProtectionTier.DRAM)

    controller.register_request(_profile("high", slack_ms=0.0))
    high_spec = _chunk("high")
    high = controller.admit_chunk(high_spec)

    assert high.hbm.admitted and high.dram.admitted
    assert high.hbm.victim_ids == ("low:0",)
    assert high.dram.victim_ids == ("low:0",)
    assert controller.residency("low", 0) is Residency.BYPASS
    assert controller.residency("high", 0) is Residency.BYPASS
    _commit(controller, high_spec, ProtectionTier.HBM, ProtectionTier.DRAM)
    assert controller.residency("high", 0) is Residency.HBM_AND_DRAM
    controller.validate_invariants()


def test_hbm_quota_aimd_can_directly_reclaim_hbm_only_tail() -> None:
    controller = AegisIndependentTierController(
        _config(hbm_bytes=200, dram_bytes=1, alpha=1.0, decrease=0.5)
    )
    for request_id in ("h0", "h1"):
        controller.register_request(_profile(request_id, tokens=100))
        spec = _chunk(request_id, token_end=100, byte_size=100)
        decision = controller.admit_chunk(spec)
        assert decision.hbm.admitted and not decision.dram.admitted
        _commit(controller, spec, ProtectionTier.HBM)
        assert controller.residency(request_id, 0) is Residency.HBM_ONLY

    dram_before = controller.snapshot()["pools"]["DRAM"]
    update = controller.control_hbm_quota(pressure=True)
    dram_after = controller.snapshot()["pools"]["DRAM"]

    assert update["new_alpha"] == pytest.approx(0.5)
    assert update["quota_bytes"] == 100
    assert update["used_bytes"] <= 100
    assert len(update["evicted_hbm_tails"]) == 1
    assert update["admission_pause_observed"] is True
    assert update["admission_resumed"] is True
    assert update["reclaim_satisfied"] is True
    assert update["dram_quota_changed"] is False
    assert {
        key: dram_after[key]
        for key in ("capacity_bytes", "limit_bytes", "used_bytes")
    } == {
        key: dram_before[key]
        for key in ("capacity_bytes", "limit_bytes", "used_bytes")
    }
    remaining = [
        request_id
        for request_id in ("h0", "h1")
        if controller.residency(request_id, 0) is Residency.HBM_ONLY
    ]
    assert len(remaining) == 1
    events = controller.snapshot()["events"]
    pause_sequence = next(
        event["sequence"] for event in events if event["kind"] == "hbm_admission_paused"
    )
    reclaim_sequence = next(
        event["sequence"]
        for event in events
        if event["kind"] == "tier_copy_removed"
        and event["reason"] == "hbm_quota_contraction"
    )
    resume_sequence = next(
        event["sequence"]
        for event in events
        if event["kind"] == "hbm_admission_resumed"
    )
    assert pause_sequence < reclaim_sequence < resume_sequence
    controller.validate_invariants()


def test_each_pool_exposes_only_one_safe_tail_per_request() -> None:
    controller = AegisIndependentTierController(
        _config(hbm_bytes=512, dram_bytes=1)
    )
    controller.register_request(_profile("suffix", tokens=512))
    first = _chunk("suffix", token_end=256, byte_size=256, shard_count=1)
    second = _chunk(
        "suffix",
        ordinal=1,
        token_start=256,
        token_end=512,
        byte_size=256,
        shard_count=1,
    )
    for spec in (first, second):
        decision = controller.admit_chunk(
            spec, allowed_tiers=(ProtectionTier.HBM,)
        )
        assert decision.hbm.admitted
        _commit(controller, spec, ProtectionTier.HBM)

    frontiers = controller.victim_frontiers(ProtectionTier.HBM)
    assert [(row.request_id, row.ordinal) for row in frontiers] == [("suffix", 1)]
    assert controller.snapshot()["pools"]["HBM"]["victim_frontiers"][0][
        "logical_id"
    ] == "suffix:1"

    contraction = controller.set_pool_limit(ProtectionTier.HBM, 256)
    assert contraction["evicted_tails"] == ["suffix:1"]
    assert [
        row.ordinal for row in controller.victim_frontiers(ProtectionTier.HBM)
    ] == [0]
    controller.validate_invariants()


def test_target_rotation_invalidates_only_hbm_and_keeps_route_version() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("rotate"))
    spec = _chunk("rotate")
    controller.admit_chunk(spec)
    _commit(controller, spec, ProtectionTier.HBM, ProtectionTier.DRAM)

    result = controller.rotate_hbm_target(
        "rotate",
        failed_target="worker-1",
        replacement_target="worker-2",
        replacement_failure_domain="node-2",
    )

    assert result["route_version"] == 7
    assert result["invalidated_hbm_records"] == 1
    assert result["dram_records_preserved"] == 1
    assert controller.residency("rotate", 0) is Residency.DRAM_ONLY
    event = controller.snapshot()["events"][-1]
    assert event["route_version_changed"] is False
    controller.validate_invariants()


def test_recovery_ignores_pending_and_uses_fastest_committed_prefix() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("recover"))
    spec = _chunk("recover")
    controller.admit_chunk(spec)
    controller.complete_copy(
        ProtectionTier.DRAM,
        "recover",
        0,
        controller.completion_proof(spec, ProtectionTier.DRAM),
    )

    decision = controller.recover_after_source_failure(
        "recover", activated_target="worker-1", next_route_version=8
    )

    assert decision.action == "DRAM"
    assert decision.restored_tokens == 1024
    assert decision.recompute_tokens == 0
    assert controller.residency("recover", 0) is Residency.DRAM_ONLY
    assert not any(
        record["state"] == CopyState.PENDING.value
        for record in controller.snapshot()["pools"]["HBM"]["records"]
    )
    controller.validate_invariants()


def test_recovery_prefers_peer_hbm_when_both_tiers_are_committed() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("fast"))
    spec = _chunk("fast")
    controller.admit_chunk(spec)
    _commit(controller, spec, ProtectionTier.HBM, ProtectionTier.DRAM)

    decision = controller.recover_after_source_failure(
        "fast", activated_target="worker-1", next_route_version=8
    )

    assert decision.action == "PEER_HBM"
    assert decision.recovery_ms == pytest.approx(1.024)
    assert decision.visible_harm_ms == pytest.approx(1.024)


def test_completion_proof_is_route_layout_and_all_shard_strict() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("strict"))
    spec = _chunk("strict", shard_count=3)
    controller.admit_chunk(spec)
    bad = replace(
        controller.completion_proof(spec, ProtectionTier.HBM), acked_shards=(0, 2)
    )

    result = controller.complete_copy(ProtectionTier.HBM, "strict", 0, bad)

    assert result.state is CopyState.ABORTED
    assert controller.residency("strict", 0) is Residency.BYPASS
    # The independent DRAM path remains live and can still commit.
    controller.complete_copy(
        ProtectionTier.DRAM,
        "strict",
        0,
        controller.completion_proof(spec, ProtectionTier.DRAM),
    )
    assert controller.residency("strict", 0) is Residency.DRAM_ONLY
    controller.validate_invariants()


def test_request_completion_waits_for_both_paths_to_be_terminal() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("finish"))
    spec = _chunk("finish")
    controller.admit_chunk(spec)
    controller.complete_copy(
        ProtectionTier.HBM,
        "finish",
        0,
        controller.completion_proof(spec, ProtectionTier.HBM),
    )

    blocked = controller.complete_request("finish")
    assert blocked["completed"] is False
    assert blocked["reason"] == "pending_copy_not_terminal"

    controller.complete_copy(
        ProtectionTier.DRAM,
        "finish",
        0,
        controller.completion_proof(spec, ProtectionTier.DRAM),
    )
    completed = controller.complete_request("finish")
    assert completed["completed"] is True
    assert completed["released_bytes"] == {"HBM": 1024, "DRAM": 1024}
    controller.validate_invariants()


def test_multiple_pending_chunks_reserve_a_contiguous_tier_prefix() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("queued", tokens=1024))
    chunks = [
        _chunk(
            "queued",
            ordinal=ordinal,
            token_start=ordinal * 256,
            token_end=(ordinal + 1) * 256,
            byte_size=256,
            shard_count=1,
        )
        for ordinal in range(4)
    ]

    decisions = [controller.admit_chunk(spec) for spec in chunks]

    assert all(decision.hbm.admitted for decision in decisions)
    assert all(decision.dram.admitted for decision in decisions)
    assert controller.committed_prefix_tokens(
        "queued", ProtectionTier.HBM
    ) == 0
    assert controller.committed_prefix_tokens(
        "queued", ProtectionTier.DRAM
    ) == 0

    for spec in chunks:
        _commit(controller, spec, ProtectionTier.HBM, ProtectionTier.DRAM)

    assert controller.committed_prefix_tokens(
        "queued", ProtectionTier.HBM
    ) == 1024
    assert controller.committed_prefix_tokens(
        "queued", ProtectionTier.DRAM
    ) == 1024
    controller.validate_invariants()


def test_failed_pending_predecessor_aborts_only_its_tier_suffix() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("queued-failure", tokens=512))
    first = _chunk(
        "queued-failure", token_end=256, byte_size=256, shard_count=1
    )
    second = _chunk(
        "queued-failure",
        ordinal=1,
        token_start=256,
        token_end=512,
        byte_size=256,
        shard_count=1,
    )
    controller.admit_chunk(first)
    controller.admit_chunk(second)

    controller.complete_copy(
        ProtectionTier.HBM,
        first.request_id,
        first.ordinal,
        controller.completion_proof(
            first, ProtectionTier.HBM, succeeded=False
        ),
    )
    _commit(controller, first, ProtectionTier.DRAM)
    _commit(controller, second, ProtectionTier.DRAM)

    assert controller.copy_state(
        "queued-failure", 1, ProtectionTier.HBM
    ) is None
    assert controller.committed_prefix_tokens(
        "queued-failure", ProtectionTier.HBM
    ) == 0
    assert controller.committed_prefix_tokens(
        "queued-failure", ProtectionTier.DRAM
    ) == 512
    assert controller.residency("queued-failure", 1) is Residency.DRAM_ONLY
    controller.validate_invariants()


def test_tier_ablation_does_not_create_cross_tier_dependency() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("dram-ablation"))
    spec = _chunk("dram-ablation")

    decision = controller.admit_chunk(
        spec, allowed_tiers=(ProtectionTier.DRAM,)
    )

    assert decision.hbm.reason == "tier_not_allowed"
    assert decision.dram.admitted is True
    _commit(controller, spec, ProtectionTier.DRAM)
    assert controller.residency("dram-ablation", 0) is Residency.DRAM_ONLY
    controller.validate_invariants()


def test_out_of_order_copy_completion_waits_then_commits_contiguous_prefix() -> None:
    controller = AegisIndependentTierController(_config())
    controller.register_request(_profile("out-of-order", tokens=512))
    first = _chunk(
        "out-of-order", token_end=256, byte_size=256, shard_count=1
    )
    second = _chunk(
        "out-of-order",
        ordinal=1,
        token_start=256,
        token_end=512,
        byte_size=256,
        shard_count=1,
    )
    controller.admit_chunk(first)
    controller.admit_chunk(second)

    waiting = controller.complete_copy(
        ProtectionTier.HBM,
        second.request_id,
        second.ordinal,
        controller.completion_proof(second, ProtectionTier.HBM),
    )

    assert waiting.state is CopyState.PENDING
    assert waiting.reason == "waiting_for_prefix_predecessor"
    assert controller.committed_prefix_tokens(
        "out-of-order", ProtectionTier.HBM
    ) == 0

    controller.complete_copy(
        ProtectionTier.HBM,
        first.request_id,
        first.ordinal,
        controller.completion_proof(first, ProtectionTier.HBM),
    )

    assert controller.copy_state(
        "out-of-order", 1, ProtectionTier.HBM
    ) is CopyState.COMMITTED
    assert controller.committed_prefix_tokens(
        "out-of-order", ProtectionTier.HBM
    ) == 512
    controller.validate_invariants()
