from __future__ import annotations

import pytest

from sglsim.srt.simulation.global_des import (
    EventPriority,
    GlobalDES,
    KvCopyState,
    KvLocation,
    PhysicalGpuId,
    TransferResourceId,
    TransferState,
)
from sglsim.srt.simulation.offline_coordinator import (
    OfflineCausalBarrierError,
    OfflineEventCoordinator,
    OfflineEventPriority,
)


def test_independent_rank_completion_does_not_max_sync_other_rank() -> None:
    coordinator = OfflineEventCoordinator(dp_size=2)
    coordinator.schedule_rank_completion(0, duration_s=10.0, payload="dp0")
    coordinator.schedule_rank_completion(1, duration_s=3.0, payload="dp1")
    coordinator.schedule_at(
        4.0,
        kind="request_arrival",
        priority=OfflineEventPriority.REQUEST_ARRIVAL,
        dp_rank=1,
    )

    first = coordinator.pop_next()
    assert (first.dp_rank, first.timestamp_s) == (1, 3.0)
    assert coordinator.rank_drivers[1].now() == 3.0
    assert coordinator.rank_drivers[0].now() == 0.0

    second = coordinator.pop_next()
    assert second.kind == "request_arrival"
    assert coordinator.now_s == 4.0
    # An external parent event also must not push an unrelated/busy rank.
    assert coordinator.rank_drivers[0].now() == 0.0

    third = coordinator.pop_next()
    assert (third.dp_rank, third.timestamp_s) == (0, 10.0)
    assert coordinator.rank_drivers[0].now() == 10.0
    assert coordinator.rank_drivers[1].now() == 3.0
    # Two ranks ran concurrently: the makespan is max(10, 3), not their sum.
    assert coordinator.makespan_s() == 10.0


def test_same_timestamp_uses_semantic_priority_then_sequence() -> None:
    coordinator = OfflineEventCoordinator(dp_size=1)
    coordinator.schedule_at(
        5.0,
        kind="arrival-a",
        priority=OfflineEventPriority.REQUEST_ARRIVAL,
    )
    coordinator.schedule_at(
        5.0,
        kind="failure",
        priority=OfflineEventPriority.FAILURE_CONTROL,
    )
    coordinator.schedule_at(
        5.0,
        kind="arrival-b",
        priority=OfflineEventPriority.REQUEST_ARRIVAL,
    )

    assert [coordinator.pop_next().kind for _ in range(3)] == [
        "failure",
        "arrival-a",
        "arrival-b",
    ]


def test_rank_allows_only_one_inflight_scheduler_step() -> None:
    coordinator = OfflineEventCoordinator(dp_size=1)
    coordinator.schedule_rank_completion(0, duration_s=1.0)
    with pytest.raises(RuntimeError, match="in-flight"):
        coordinator.schedule_rank_completion(0, duration_s=1.0)

    coordinator.pop_next()
    coordinator.schedule_rank_completion(0, duration_s=2.0)
    assert coordinator.pop_next().timestamp_s == 3.0


def test_cancel_stage_completion_clears_only_target_at_cancellation_time() -> None:
    coordinator = OfflineEventCoordinator(dp_size=2, pp_size=2)
    cancelled = coordinator.schedule_stage_completion(
        0,
        1,
        duration_s=10.0,
        payload="interrupted",
    )
    coordinator.schedule_stage_completion(
        1,
        0,
        duration_s=8.0,
        payload="unrelated",
    )
    coordinator.schedule_at(
        3.0,
        kind="gpu_failure",
        priority=OfflineEventPriority.FAILURE_CONTROL,
        dp_rank=0,
        pp_rank=1,
    )

    assert coordinator.pop_next().kind == "gpu_failure"
    assert coordinator.cancel_stage_completion(
        0,
        1,
        reason="gpu_fail_stop",
        metadata={"failure_id": "failure-0", "dp_rank": 99},
    ) == cancelled
    assert coordinator.cancel_stage_completion(0, 1, reason="replay") is None
    assert not coordinator.stage_is_busy(0, 1)
    assert coordinator.stage_is_busy(1, 0)
    # Only the interrupted stage advances to the failure instant.  It does
    # not jump to the cancelled t=10 completion, and the other stage is not
    # max-synchronized.
    assert coordinator.stage_driver(0, 1).now() == 3.0
    assert coordinator.stage_driver(1, 0).now() == 0.0

    replacement = coordinator.schedule_stage_completion(
        0,
        1,
        duration_s=2.0,
        payload="replacement",
    )
    assert coordinator.pop_next() == replacement
    assert replacement.timestamp_s == 5.0
    unrelated = coordinator.pop_next()
    assert (unrelated.dp_rank, unrelated.pp_rank, unrelated.timestamp_s) == (
        1,
        0,
        8.0,
    )
    assert not coordinator.has_events()
    assert coordinator.now_s == 8.0

    cancellation_rows = [
        row.as_dict()
        for row in coordinator.global_des.journal
        if row.kind == "event_cancelled"
    ]
    assert len(cancellation_rows) == 1
    assert cancellation_rows[0]["event_id"] == cancelled.event_id
    assert cancellation_rows[0]["metadata"]["reason"] == "gpu_fail_stop"
    assert cancellation_rows[0]["metadata"]["cancellation_metadata"] == {
        "failure_id": "failure-0",
        "dp_rank": 0,
        "pp_rank": 1,
    }
    assert coordinator.global_des.audit_invariants() == []


def test_idle_activation_advances_only_target_rank() -> None:
    coordinator = OfflineEventCoordinator(dp_size=2)
    coordinator.schedule_at(
        7.0,
        kind="arrival",
        priority=OfflineEventPriority.REQUEST_ARRIVAL,
        dp_rank=1,
    )
    coordinator.pop_next()

    coordinator.activate_idle_rank(1)
    assert coordinator.rank_drivers[1].now() == 7.0
    assert coordinator.rank_drivers[0].now() == 0.0


def test_physical_pp_stages_have_independent_clocks_without_cross_dp_sync() -> None:
    coordinator = OfflineEventCoordinator(dp_size=2, pp_size=2)
    coordinator.schedule_stage_completion(0, 0, duration_s=10.0)
    coordinator.schedule_stage_completion(0, 1, duration_s=3.0)
    coordinator.schedule_stage_completion(1, 0, duration_s=4.0)

    first = coordinator.pop_next()
    assert (first.dp_rank, first.pp_rank, first.timestamp_s) == (0, 1, 3.0)
    assert coordinator.stage_driver(0, 1).now() == 3.0
    assert coordinator.stage_driver(0, 0).now() == 0.0
    assert coordinator.stage_driver(1, 0).now() == 0.0

    second = coordinator.pop_next()
    assert (second.dp_rank, second.pp_rank, second.timestamp_s) == (1, 0, 4.0)
    assert coordinator.stage_driver(1, 0).now() == 4.0
    assert coordinator.stage_driver(0, 0).now() == 0.0

    third = coordinator.pop_next()
    assert (third.dp_rank, third.pp_rank, third.timestamp_s) == (0, 0, 10.0)
    assert coordinator.dp_makespan_s(0) == 10.0


def _two_gpu_transfer_des() -> tuple[
    GlobalDES,
    PhysicalGpuId,
    PhysicalGpuId,
]:
    des = GlobalDES()
    source_gpu = PhysicalGpuId("node0", 0, 0, 0, 0)
    destination_gpu = PhysicalGpuId("node1", 0, 1, 0, 0)
    des.register_gpu(source_gpu, capacity_bytes=4096)
    des.register_gpu(destination_gpu, capacity_bytes=4096)
    des.register_extent(extent_id="e0", request_id="r0", size_bytes=1024)
    des.place_extent("e0", KvLocation.gpu_primary(source_gpu))
    return des, source_gpu, destination_gpu


def test_resource_transfer_and_rank_completion_share_authoritative_heap() -> None:
    des, source_gpu, destination_gpu = _two_gpu_transfer_des()
    coordinator = OfflineEventCoordinator(dp_size=2, global_des=des)
    transfer = des.reserve_transfer(
        extent_id="e0",
        source=KvLocation.gpu_primary(source_gpu),
        destination=KvLocation.gpu_aegis(destination_gpu),
        resources=(TransferResourceId("nic", "node0"),),
        duration_ns=1_000_000_000,
    )
    rank_completion = coordinator.schedule_rank_completion(0, duration_s=1.0)

    transfer_start = coordinator.pop_next()
    assert (transfer_start.kind, transfer_start.payload) == (
        "transfer_start",
        {"transfer_id": transfer.transfer_id},
    )
    assert des.transfers[transfer.transfer_id].state is TransferState.IN_FLIGHT

    # TRANSFER_COMPLETION precedes COMPUTE_COMPLETION at the same nanosecond,
    # and coordinator.pop_next executes the GlobalDES transfer callback.
    transfer_completion = coordinator.pop_next()
    assert transfer_completion.kind == "transfer_completion"
    assert transfer_completion.timestamp_s == 1.0
    assert des.transfers[transfer.transfer_id].state is TransferState.COMMITTED
    assert (
        des.extents["e0"].locations[KvLocation.gpu_aegis(destination_gpu)]
        is KvCopyState.COMMITTED
    )

    assert coordinator.pop_next() == rank_completion
    assert coordinator.now_ns == 1_000_000_000
    assert not coordinator.has_events()


def test_global_failure_wins_tie_and_its_callback_aborts_transfer() -> None:
    des, source_gpu, destination_gpu = _two_gpu_transfer_des()
    coordinator = OfflineEventCoordinator(dp_size=2, global_des=des)
    transfer = des.reserve_transfer(
        extent_id="e0",
        source=KvLocation.gpu_primary(source_gpu),
        destination=KvLocation.gpu_aegis(destination_gpu),
        resources=(TransferResourceId("copy_engine", destination_gpu.key),),
        duration_ns=2_000_000_000,
    )
    rank_completion = coordinator.schedule_rank_completion(1, duration_s=2.0)
    des.schedule_at(
        2_000_000_000,
        kind="gpu_failure",
        priority=EventPriority.FAILURE,
        payload={"gpu": destination_gpu.key},
        callback=lambda simulator, _event: simulator.fail_gpu(destination_gpu),
    )

    assert coordinator.pop_next().kind == "transfer_start"
    failure = coordinator.pop_next()
    assert failure.kind == "gpu_failure"
    assert failure.payload == {"gpu": destination_gpu.key}
    assert not des.gpu_hbm[destination_gpu].healthy
    assert des.transfers[transfer.transfer_id].state is TransferState.ABORTED

    # The already queued completion callback is still consumed safely, then
    # the rank completion follows according to the shared priority/sequence.
    assert coordinator.pop_next().kind == "transfer_completion"
    assert coordinator.pop_next() == rank_completion
    assert not coordinator.has_events()


def test_seconds_are_quantized_causally_to_nanoseconds() -> None:
    coordinator = OfflineEventCoordinator(dp_size=1)
    event = coordinator.schedule_at(
        0.0000000001,
        kind="sub-nanosecond-arrival",
        priority=OfflineEventPriority.REQUEST_ARRIVAL,
    )

    # The compatibility event retains the exact seconds supplied to the
    # offline Scheduler, while the authoritative heap cannot dispatch before
    # it and therefore advances to the next integer nanosecond.
    assert event.timestamp_s == 0.0000000001
    assert coordinator.pop_next() == event
    assert coordinator.now_ns == 1


def test_unpublished_rank_work_blocks_future_global_event_without_wall_sleep() -> None:
    coordinator = OfflineEventCoordinator(dp_size=2)
    coordinator.schedule_at(
        2.0,
        kind="request_arrival",
        priority=OfflineEventPriority.REQUEST_ARRIVAL,
        dp_rank=0,
    )
    assert coordinator.pop_next().kind == "request_arrival"
    barrier = coordinator.begin_stage_work(
        0,
        0,
        lower_bound_s=coordinator.now_s,
        operation="prepare",
        metadata={"request_ids": ["active-r0"]},
    )
    coordinator.schedule_at(
        2.0,
        kind="same-time-arrival",
        priority=OfflineEventPriority.REQUEST_ARRIVAL,
        dp_rank=1,
    )
    coordinator.schedule_at(
        10.0,
        kind="failure",
        priority=OfflineEventPriority.FAILURE_CONTROL,
        dp_rank=0,
    )

    with pytest.raises(OfflineCausalBarrierError, match="cannot dispatch"):
        coordinator.pop_next()
    assert coordinator.now_s == 2.0
    assert coordinator.stage_has_unpublished_work(0, 0)
    assert coordinator.rank_work_barriers() == (barrier,)

    coordinator.finish_stage_work(
        0,
        0,
        barrier_id=barrier.barrier_id,
    )
    assert coordinator.pop_next().kind == "same-time-arrival"
    assert coordinator.pop_next().kind == "failure"
    assert coordinator.now_s == 10.0
