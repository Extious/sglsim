from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from AegisKV.experiments.real_weka_baselines import run as baseline_replay
from sglsim.srt.simulation.aegis_baselines import (
    BaselinePolicy,
    BaselineRequestProfile,
    LumenBaselineRuntime,
    LumenConfig,
    ProtectionBaselineConfig,
    ProtectionBaselineRuntime,
    bounded_state_restore,
    cacheflow_3d_restore,
    restore_or_recompute_split,
)
from sglsim.srt.simulation.aegis_full_design import (
    CalibratedTransferPath,
    SealedChunk,
)
from sglsim.srt.simulation.aegis_independent_tiers import RecoveryTiming


def _config(
    policy: BaselinePolicy,
    *,
    hbm_capacity: int = 10_000,
    dram_capacity: int = 10_000,
    bandwidth: float = 100.0,
) -> ProtectionBaselineConfig:
    return ProtectionBaselineConfig(
        policy=policy,
        hbm_capacity_bytes=hbm_capacity,
        dram_capacity_bytes=dram_capacity,
        kv_bytes_per_token=10,
        hbm_copy=CalibratedTransferPath(0.0, bandwidth, "unit"),
        dram_copy=CalibratedTransferPath(0.0, bandwidth, "unit"),
        recompute=RecoveryTiming(0.0, 1.0),
        hbm_load=RecoveryTiming(0.0, 0.1),
        dram_load=RecoveryTiming(0.0, 0.5),
        parity_data_shards=8,
        parity_shards=2,
        parity_encode_ms_per_token=0.01,
        parity_decode_ms_per_token=0.02,
    )


def _profile(request_id: str, *, target: str = "peer-1", total: int = 100):
    return BaselineRequestProfile(
        request_id=request_id,
        source_worker="source-0",
        source_failure_domain="source-domain-0",
        target_worker=target,
        target_failure_domain=f"{target}-domain",
        route_version=1,
        total_tokens=total,
        slack_ms=0.0,
    )


def test_pending_copy_has_no_recovery_value_until_commit() -> None:
    runtime = ProtectionBaselineRuntime(
        _config(BaselinePolicy.DEJAVU_NEIGHBOR, bandwidth=10.0),
        target_id="peer-1",
    )
    runtime.register_request(_profile("r"), time_ms=0.0)
    admission = runtime.seal_chunk(
        "r", SealedChunk(0, 0, 100, "prefix"), time_ms=0.0
    )
    assert admission.admitted

    pending = runtime.estimate_source_failure(
        "r",
        produced_tokens=100,
        remaining_slack_ms=0.0,
        time_ms=99.0,
    )
    committed = runtime.estimate_source_failure(
        "r",
        produced_tokens=100,
        remaining_slack_ms=0.0,
        time_ms=100.0,
    )
    assert pending.action == "RECOMPUTE"
    assert pending.restored_tokens == 0
    assert committed.action == "PEER_HBM"
    assert committed.restored_tokens == 100


def test_capacity_miss_blocks_later_suffix_to_preserve_prefix() -> None:
    runtime = ProtectionBaselineRuntime(
        _config(BaselinePolicy.RAIDSERVE_HOST, dram_capacity=1_500),
        target_id="host-1",
    )
    runtime.register_request(_profile("r", total=300), time_ms=0.0)
    first = runtime.seal_chunk(
        "r", SealedChunk(0, 0, 100, "prefix"), time_ms=0.0
    )
    miss = runtime.seal_chunk(
        "r", SealedChunk(1, 100, 200, "prefix"), time_ms=1.0
    )
    suffix = runtime.seal_chunk(
        "r", SealedChunk(2, 200, 300, "prefix"), time_ms=2.0
    )
    assert first.admitted
    assert not miss.admitted and miss.reason == "capacity_miss"
    assert not suffix.admitted and suffix.reason == "prefix_already_unprotected"
    runtime.validate_invariants()


def test_kevlarflow_pressure_drop_invalidates_whole_old_replica() -> None:
    runtime = ProtectionBaselineRuntime(
        _config(BaselinePolicy.KEVLARFLOW_RING, hbm_capacity=1_000),
        target_id="ring-peer",
    )
    runtime.register_request(_profile("old"), time_ms=0.0)
    runtime.seal_chunk("old", SealedChunk(0, 0, 100, "prefix"), time_ms=0.0)
    runtime.advance_to(20.0)
    runtime.register_request(_profile("new"), time_ms=20.0)
    admission = runtime.seal_chunk(
        "new", SealedChunk(0, 0, 100, "prefix"), time_ms=20.0
    )
    old = runtime.estimate_source_failure(
        "old", produced_tokens=100, remaining_slack_ms=0.0, time_ms=40.0
    )
    assert admission.admitted
    assert admission.evicted_requests == ("old",)
    assert old.action == "RECOMPUTE"
    assert runtime.snapshot()["metrics"]["evicted_requests"] == 1


def test_ghostserve_parity_charges_parity_footprint_and_reconstructs() -> None:
    with pytest.raises(ValueError, match="more than one data shard"):
        ProtectionBaselineConfig(
            **{
                **_config(BaselinePolicy.GHOSTSERVE_PARITY).__dict__,
                "parity_data_shards": 1,
                "parity_shards": 1,
            }
        )

    runtime = ProtectionBaselineRuntime(
        _config(BaselinePolicy.GHOSTSERVE_PARITY), target_id="host-parity"
    )
    runtime.register_request(_profile("r"), time_ms=0.0)
    admission = runtime.seal_chunk(
        "r", SealedChunk(0, 0, 100, "prefix"), time_ms=0.0
    )
    assert admission.logical_bytes == 1_000
    assert admission.stored_bytes == 250
    runtime.flush()
    decision = runtime.estimate_source_failure(
        "r",
        produced_tokens=100,
        remaining_slack_ms=0.0,
        time_ms=runtime.now_ms,
    )
    assert decision.action == "GHOST_PARITY_RECONSTRUCT"
    assert decision.restored_tokens == 100


def test_ghostserve_parity_decode_uses_hbm_recovery_not_protection_rate() -> None:
    base = _config(BaselinePolicy.GHOSTSERVE_PARITY, bandwidth=100.0)
    runtime = SimpleNamespace(
        **{
            **base.__dict__,
            "hbm_load": RecoveryTiming(setup_ms=0.0, ms_per_token=0.8),
        }
    )
    configs = baseline_replay._baseline_configs(
        runtime,
        {"ghostserve": {"data_shards": 8, "parity_shards": 2}},
    )
    ghost = configs[BaselinePolicy.GHOSTSERVE_PARITY]

    assert ghost.parity_encode_ms_per_token == pytest.approx(10.0 / 100.0 * 0.25)
    assert ghost.parity_decode_ms_per_token == pytest.approx(0.8 / 8.0)
    assert ghost.parity_decode_ms_per_token != pytest.approx(
        10.0 / 100.0 / 8.0
    )


def test_hbm_target_failure_invalidates_replica_but_host_survives() -> None:
    runtimes = {
        policy: ProtectionBaselineRuntime(_config(policy), target_id=policy.value)
        for policy in (
            BaselinePolicy.DEJAVU_NEIGHBOR,
            BaselinePolicy.RAIDSERVE_HOST,
        )
    }
    for runtime in runtimes.values():
        runtime.register_request(_profile("r"), time_ms=0.0)
        runtime.seal_chunk("r", SealedChunk(0, 0, 100, "prefix"), time_ms=0.0)
        runtime.flush()
        runtime.rotate_target(
            "r",
            replacement_target="peer-2",
            replacement_failure_domain="peer-2-domain",
            time_ms=runtime.now_ms,
        )
    hbm = runtimes[BaselinePolicy.DEJAVU_NEIGHBOR].estimate_source_failure(
        "r", produced_tokens=100, remaining_slack_ms=0.0, time_ms=20.0
    )
    host = runtimes[BaselinePolicy.RAIDSERVE_HOST].estimate_source_failure(
        "r", produced_tokens=100, remaining_slack_ms=0.0, time_ms=20.0
    )
    assert hbm.action == "RECOMPUTE"
    assert host.action == "HOST_DRAM"


@pytest.mark.parametrize(
    "policy", [BaselinePolicy.DEJAVU_NEIGHBOR, BaselinePolicy.KEVLARFLOW_RING]
)
def test_hbm_target_invalidation_is_generation_idempotent(
    policy: BaselinePolicy,
) -> None:
    runtime = ProtectionBaselineRuntime(_config(policy), target_id="peer-1")
    runtime.register_request(_profile("r", total=300), time_ms=0.0)
    runtime.seal_chunk("r", SealedChunk(0, 0, 100, "prefix"), time_ms=0.0)

    released = runtime.invalidate_target_failure("r", time_ms=20.0)
    state = runtime._requests["r"]
    assert released == 1_000
    assert state.generation == 1
    assert state.blocked is True

    assert runtime.invalidate_target_failure("r", time_ms=21.0) == 0
    assert state.generation == 1
    assert sum(
        row["kind"] == "request_state_released"
        for row in runtime.snapshot()["events"]
    ) == 1

    rejected = runtime.seal_chunk(
        "r", SealedChunk(1, 100, 200, "prefix"), time_ms=22.0
    )
    assert not rejected.admitted
    assert rejected.reason == "prefix_already_unprotected"
    assert state.next_ordinal == 2
    with pytest.raises(ValueError, match="chunk ordinal is not progressive"):
        runtime.seal_chunk(
            "r", SealedChunk(1, 100, 200, "prefix"), time_ms=23.0
        )
    with pytest.raises(ValueError, match="chunk ordinal is not progressive"):
        runtime.seal_chunk(
            "r", SealedChunk(3, 300, 400, "prefix"), time_ms=24.0
        )


def test_lumen_holder_reservation_and_dispatch_are_load_aware() -> None:
    config = LumenConfig(
        holder_count=3,
        bytes_per_holder=2_000,
        kv_bytes_per_token=10,
        copy_path=CalibratedTransferPath(0.0, 100.0, "unit"),
        recompute=RecoveryTiming(0.0, 1.0),
        host_load=RecoveryTiming(0.0, 0.5),
    )
    runtime = LumenBaselineRuntime(config)
    holder = runtime.register_request(
        _profile("r", total=100), source_rank=0, time_ms=0.0
    )
    assert holder in (1, 2)
    assert holder != 0
    runtime.seal_chunk("r", SealedChunk(0, 0, 100, "prefix"), time_ms=0.0)
    runtime.flush()
    decision = runtime.estimate_source_failure(
        "r", produced_tokens=100, remaining_slack_ms=0.0, time_ms=10.0
    )
    moved = runtime.dispatch([decision, decision, decision], target_count=2)
    assert all(row.activated_target.startswith("target-") for row in moved)
    assert runtime.snapshot()["metrics"]["dispatch_moves"] == 2

    no_space = runtime.register_request(
        _profile("huge", total=10_000), source_rank=0, time_ms=10.0
    )
    assert no_space is None


def test_restore_planners_preserve_declared_resource_semantics() -> None:
    split = restore_or_recompute_split(
        total_tokens=1_000,
        mandatory_recompute_tokens=100,
        io_ms_per_token=0.5,
        recompute_ms_per_token=1.0,
        kv_bytes_per_token=10,
    )
    assert split.restored_tokens + split.recomputed_tokens == 1_000
    assert math.isclose(split.completion_ms, max(split.io_service_ms, split.compute_service_ms))

    cacheflow = cacheflow_3d_restore(
        total_tokens=1_000,
        externally_available_tokens=800,
        io_ms_per_token=0.2,
        recompute_ms_per_token=1.0,
        kv_bytes_per_token=10,
        layers=32,
        gpu_lanes=4,
        token_tiles=8,
    )
    assert cacheflow.restored_tokens == 800
    assert cacheflow.recomputed_tokens == 200
    assert cacheflow.completion_ms < 1_000

    bsr = bounded_state_restore(
        total_tokens=1_000,
        externally_available_tokens=900,
        chunk_tokens=100,
        window_chunks=2,
        kv_bytes_per_token=10,
        io_timing=RecoveryTiming(0.0, 0.2),
        recompute_timing=RecoveryTiming(0.0, 1.0),
    )
    assert bsr.peak_staging_bytes == 2_000
    assert bsr.restored_tokens == 900
    assert bsr.request_commit

    failed = bounded_state_restore(
        total_tokens=1_000,
        externally_available_tokens=900,
        chunk_tokens=100,
        window_chunks=2,
        kv_bytes_per_token=10,
        io_timing=RecoveryTiming(0.0, 0.2),
        recompute_timing=RecoveryTiming(0.0, 1.0),
        failed_install_chunk=3,
    )
    assert failed.restored_tokens == 0
    assert failed.recomputed_tokens == 1_000
    assert not failed.request_commit
    assert failed.fallback_reason == "partial_install_invalidated"
