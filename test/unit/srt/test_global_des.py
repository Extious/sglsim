from __future__ import annotations

import pytest

from sglsim.srt.simulation.global_des import (
    CommitGroupState,
    EventPriority,
    ExtentAccountingAuditMode,
    GlobalDES,
    GlobalEvent,
    HbmAccount,
    KvCopyState,
    KvLocation,
    LocationAccounting,
    PhysicalGpuId,
    TransferDestinationMode,
    TransferResourceId,
    TransferState,
)


def _gpu(node: str, local: int, dp: int, tp: int = 0) -> PhysicalGpuId:
    return PhysicalGpuId(
        node_id=node,
        local_gpu_id=local,
        dp_rank=dp,
        pp_rank=0,
        tp_rank=tp,
    )


def test_global_heap_has_stable_timestamp_priority_sequence_order() -> None:
    simulator = GlobalDES()
    order: list[str] = []

    simulator.schedule_at(
        10,
        kind="same-priority-first",
        priority=EventPriority.CONTROL,
        callback=lambda _sim, event: order.append(event.kind),
    )
    simulator.schedule_at(
        10,
        kind="failure-wins-priority",
        priority=EventPriority.FAILURE,
        callback=lambda _sim, event: order.append(event.kind),
    )
    simulator.schedule_at(
        10,
        kind="same-priority-second",
        priority=EventPriority.CONTROL,
        callback=lambda _sim, event: order.append(event.kind),
    )

    simulator.run()

    assert order == [
        "failure-wins-priority",
        "same-priority-first",
        "same-priority-second",
    ]
    assert simulator.now_ns == 10


def test_concurrent_dp_ranks_have_max_not_sum_makespan() -> None:
    simulator = GlobalDES()
    completions: list[tuple[int, int]] = []

    # Both ranks start at global virtual time zero.  Their events share the
    # global heap but no max-sync or rank-to-rank serialization is introduced.
    simulator.schedule_after(
        13,
        kind="dp0-complete",
        priority=EventPriority.COMPUTE_COMPLETION,
        payload={"dp_rank": 0},
        callback=lambda sim, event: completions.append(
            (event.payload["dp_rank"], sim.now_ns)
        ),
    )
    simulator.schedule_after(
        29,
        kind="dp1-complete",
        priority=EventPriority.COMPUTE_COMPLETION,
        payload={"dp_rank": 1},
        callback=lambda sim, event: completions.append(
            (event.payload["dp_rank"], sim.now_ns)
        ),
    )

    simulator.run()

    assert completions == [(0, 13), (1, 29)]
    assert simulator.now_ns == max(13, 29)
    assert simulator.now_ns != 13 + 29


def test_cancel_future_event_is_idempotent_audited_and_does_not_advance() -> None:
    simulator = GlobalDES()
    callbacks: list[str] = []
    event = simulator.schedule_at(
        50,
        kind="cancel-me",
        priority=EventPriority.COMPUTE_COMPLETION,
        callback=lambda _sim, row: callbacks.append(row.kind),
    )

    assert simulator.cancel_event(
        event.event_id,
        reason="gpu_fail_stop",
        metadata={"failure_id": "failure-0"},
    )
    assert not simulator.cancel_event(event.event_id, reason="replay")
    assert not simulator.has_events()
    assert simulator.run() == []
    assert simulator.now_ns == 0
    assert callbacks == []

    cancellation_rows = [
        row.as_dict()
        for row in simulator.journal
        if row.kind == "event_cancelled"
    ]
    assert len(cancellation_rows) == 1
    assert cancellation_rows[0]["event_id"] == event.event_id
    assert cancellation_rows[0]["before"] == {
        "queued": True,
        "status": "queued",
    }
    assert cancellation_rows[0]["after"] == {
        "queued": False,
        "status": "cancelled",
    }
    assert cancellation_rows[0]["metadata"]["reason"] == "gpu_fail_stop"
    assert cancellation_rows[0]["metadata"]["scheduled_timestamp_ns"] == 50
    assert cancellation_rows[0]["metadata"]["cancellation_metadata"] == {
        "failure_id": "failure-0"
    }
    assert simulator.audit_invariants() == []


def test_cancel_event_rejects_unknown_or_dispatched_and_preserves_heap_order() -> None:
    simulator = GlobalDES()
    order: list[str] = []

    def reject_self_cancellation(sim: GlobalDES, row: GlobalEvent) -> None:
        with pytest.raises(RuntimeError, match="state 'dispatching'"):
            sim.cancel_event(row.event_id)
        order.append("self")

    first = simulator.schedule_at(
        10,
        kind="first",
        callback=lambda _sim, row: order.append(row.kind),
    )
    middle = simulator.schedule_at(
        10,
        kind="middle",
        callback=lambda _sim, row: order.append(row.kind),
    )
    last = simulator.schedule_at(
        10,
        kind="last",
        callback=lambda _sim, row: order.append(row.kind),
    )
    self_cancelling = simulator.schedule_at(
        10,
        kind="self",
        callback=reject_self_cancellation,
    )

    with pytest.raises(KeyError, match="unknown event_id"):
        simulator.cancel_event("evt-does-not-exist")
    assert simulator.cancel_event(middle.event_id)
    assert simulator.pop_next() == first
    with pytest.raises(RuntimeError, match="state 'completed'"):
        simulator.cancel_event(first.event_id)
    assert simulator.pop_next() == last
    assert simulator.pop_next() == self_cancelling
    assert order == ["first", "last", "self"]
    assert simulator.now_ns == 10
    assert simulator.audit_invariants() == []


def test_hbm_capacity_and_negative_accounting_are_rejected_atomically() -> None:
    simulator = GlobalDES()
    gpu = _gpu("node0", 0, 0)
    simulator.register_gpu(
        gpu,
        capacity_bytes=100,
        weights_bytes=50,
        runtime_bytes=10,
    )
    simulator.adjust_gpu_hbm(gpu, HbmAccount.PRIMARY_KV, 40)
    before = simulator.gpu_hbm[gpu].snapshot()

    with pytest.raises(ValueError, match="exceeds capacity"):
        simulator.adjust_gpu_hbm(gpu, HbmAccount.PREFIX_KV, 1)
    assert simulator.gpu_hbm[gpu].snapshot() == before

    with pytest.raises(ValueError, match="would become negative"):
        simulator.adjust_gpu_hbm(gpu, HbmAccount.PRIMARY_KV, -41)
    assert simulator.gpu_hbm[gpu].snapshot() == before


def test_transfer_reserves_multiple_resources_and_commits_pending_hbm() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("node0", 0, 0)
    target_gpu = _gpu("node1", 0, 1)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(target_gpu, capacity_bytes=1_000)
    simulator.register_extent(
        extent_id="req0/chunk0/tp0",
        request_id="req0",
        size_bytes=128,
        token_start=0,
        token_end=16,
    )
    source = KvLocation.gpu_primary(source_gpu)
    target = KvLocation.gpu_aegis(target_gpu)
    simulator.place_extent("req0/chunk0/tp0", source)
    resources = (
        TransferResourceId("copy_engine", source_gpu.key),
        TransferResourceId("nic", "node0"),
        TransferResourceId("link", "node0->node1"),
        TransferResourceId("nic", "node1"),
        TransferResourceId("copy_engine", target_gpu.key),
    )

    transfer = simulator.reserve_transfer(
        extent_id="req0/chunk0/tp0",
        source=source,
        destination=target,
        resources=resources,
        duration_ns=25,
        ready_ns=5,
    )

    assert (transfer.ready_ns, transfer.start_ns, transfer.end_ns) == (5, 5, 30)
    assert simulator.gpu_hbm[target_gpu].aegis_pending == 128
    assert simulator.gpu_hbm[target_gpu].aegis_committed == 0
    assert simulator.extents[transfer.extent_id].locations[target] is KvCopyState.PENDING
    assert all(simulator.resource_busy_until_ns[row] == 30 for row in resources)
    assert transfer.state is TransferState.QUEUED

    simulator.run(until_ns=5)

    assert simulator.now_ns == 5
    assert transfer.state is TransferState.IN_FLIGHT
    assert simulator.export_timeseries()["transfers"][-1]["reason"] == (
        "transfer_started"
    )

    simulator.run()

    assert simulator.now_ns == 30
    assert transfer.state is TransferState.COMMITTED
    assert simulator.gpu_hbm[target_gpu].aegis_pending == 0
    assert simulator.gpu_hbm[target_gpu].aegis_committed == 128
    assert (
        simulator.extents[transfer.extent_id].locations[target]
        is KvCopyState.COMMITTED
    )
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_gpu_failure_only_clears_target_volatile_hbm_not_dram() -> None:
    simulator = GlobalDES()
    failed_gpu = _gpu("node0", 0, 0)
    other_gpu = _gpu("node1", 0, 1)
    simulator.register_gpu(
        failed_gpu,
        capacity_bytes=1_000,
        weights_bytes=100,
        runtime_bytes=50,
    )
    simulator.register_gpu(other_gpu, capacity_bytes=1_000, weights_bytes=75)
    simulator.register_node_dram("node0", capacity_bytes=2_000, runtime_bytes=200)

    simulator.register_extent(extent_id="failed-copy", request_id="r0", size_bytes=80)
    simulator.place_extent("failed-copy", KvLocation.gpu_aegis(failed_gpu))
    simulator.register_extent(extent_id="other-copy", request_id="r0", size_bytes=90)
    simulator.place_extent("other-copy", KvLocation.gpu_aegis(other_gpu))
    simulator.register_extent(extent_id="dram-copy", request_id="r0", size_bytes=120)
    dram_location = KvLocation.node_dram("node0")
    simulator.place_extent("dram-copy", dram_location)

    other_before = simulator.gpu_hbm[other_gpu].snapshot()
    dram_before = simulator.node_dram["node0"].snapshot()
    affected = simulator.fail_gpu(failed_gpu)

    failed = simulator.gpu_hbm[failed_gpu]
    assert affected == ("failed-copy",)
    assert not failed.healthy
    assert failed.used_bytes == 0
    assert all(getattr(failed, account.value) == 0 for account in HbmAccount)
    assert simulator.gpu_hbm[other_gpu].snapshot() == other_before
    assert simulator.node_dram["node0"].snapshot() == dram_before
    assert (
        simulator.extents["failed-copy"].locations[
            KvLocation.gpu_aegis(failed_gpu)
        ]
        is KvCopyState.LOST
    )
    assert (
        simulator.extents["dram-copy"].locations[dram_location]
        is KvCopyState.COMMITTED
    )


def test_snapshot_timeseries_and_journal_are_defensive_and_auditable() -> None:
    simulator = GlobalDES()
    gpu = _gpu("node0", 3, 2, tp=1)
    simulator.register_gpu(gpu, capacity_bytes=100)
    simulator.adjust_gpu_hbm(gpu, HbmAccount.PRIMARY_KV, 10, reason="decode")

    snapshot = simulator.snapshot()
    tables = simulator.export_timeseries()

    assert snapshot["gpus"][gpu.key]["primary_kv"] == 10
    assert tables["gpu_hbm"][-1]["dp_rank"] == 2
    assert tables["gpu_hbm"][-1]["tp_rank"] == 1
    assert all(row["event_id"] for row in tables["journal"])
    assert all("before" in row and "after" in row for row in tables["journal"])
    tables["gpu_hbm"][-1]["primary_kv"] = 99
    assert simulator.export_timeseries()["gpu_hbm"][-1]["primary_kv"] == 10
    assert simulator.audit_invariants() == []


def test_source_failure_at_completion_aborts_and_releases_destination() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("node0", 0, 0)
    target_gpu = _gpu("node1", 0, 1)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(target_gpu, capacity_bytes=1_000)
    simulator.register_extent(extent_id="e0", request_id="r0", size_bytes=128)
    source = KvLocation.gpu_primary(source_gpu)
    target = KvLocation.gpu_aegis(target_gpu)
    simulator.place_extent("e0", source)
    transfer = simulator.reserve_transfer(
        extent_id="e0",
        source=source,
        destination=target,
        resources=(TransferResourceId("link", "node0->node1"),),
        duration_ns=20,
    )
    simulator.run(until_ns=0)
    assert transfer.state is TransferState.IN_FLIGHT

    # Inserted after the completion, but FAILURE priority wins at timestamp 20.
    simulator.schedule_at(
        20,
        kind="source_failure",
        priority=EventPriority.FAILURE,
        callback=lambda sim, _event: sim.fail_gpu(source_gpu),
    )
    simulator.run()

    assert transfer.state is TransferState.ABORTED
    assert simulator.gpu_hbm[target_gpu].healthy
    assert simulator.gpu_hbm[target_gpu].aegis_pending == 0
    assert simulator.gpu_hbm[target_gpu].aegis_committed == 0
    assert simulator.extents["e0"].locations[target] is KvCopyState.ABORTED
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_destination_failure_aborts_without_double_releasing_pending_hbm() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("node0", 0, 0)
    target_gpu = _gpu("node1", 0, 1)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(target_gpu, capacity_bytes=1_000)
    simulator.register_extent(extent_id="e0", request_id="r0", size_bytes=64)
    source = KvLocation.gpu_primary(source_gpu)
    target = KvLocation.gpu_aegis(target_gpu)
    simulator.place_extent("e0", source)
    transfer = simulator.reserve_transfer(
        extent_id="e0",
        source=source,
        destination=target,
        resources=(TransferResourceId("link", "node0->node1"),),
        duration_ns=10,
    )
    assert simulator.gpu_hbm[target_gpu].aegis_pending == 64

    simulator.fail_gpu(target_gpu)
    # Public abort is idempotent and must not subtract the reservation twice.
    simulator.abort_transfer(transfer.transfer_id, reason="repeat_abort")
    simulator.run()

    assert transfer.state is TransferState.ABORTED
    assert simulator.gpu_hbm[target_gpu].used_bytes == 0
    assert simulator.extents["e0"].locations[target] is KvCopyState.ABORTED
    reasons = [row["reason"] for row in simulator.export_timeseries()["transfers"]]
    assert "transfer_start_skipped" in reasons
    assert "transfer_completion_skipped" in reasons
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_node_failure_is_local_and_repair_restores_admission() -> None:
    simulator = GlobalDES()
    local_gpu = _gpu("node0", 0, 0)
    remote_gpu = _gpu("node1", 0, 1)
    simulator.register_gpu(local_gpu, capacity_bytes=1_000, weights_bytes=100)
    simulator.register_gpu(remote_gpu, capacity_bytes=1_000, weights_bytes=100)
    simulator.register_node_dram(
        "node0",
        capacity_bytes=2_000,
        runtime_bytes=100,
        hicache_primary_bytes=50,
    )
    simulator.register_node_dram("node1", capacity_bytes=2_000, runtime_bytes=100)
    simulator.register_extent(extent_id="local-hbm", request_id="r0", size_bytes=40)
    simulator.place_extent("local-hbm", KvLocation.gpu_primary(local_gpu))
    simulator.register_extent(extent_id="local-dram", request_id="r0", size_bytes=60)
    simulator.place_extent("local-dram", KvLocation.node_dram("node0"))
    simulator.register_extent(extent_id="remote-dram", request_id="r0", size_bytes=70)
    remote_location = KvLocation.node_dram("node1")
    simulator.place_extent("remote-dram", remote_location)
    remote_before = simulator.node_dram["node1"].snapshot()

    affected = simulator.fail_node("node0")

    assert affected == ("local-dram", "local-hbm")
    assert not simulator.gpu_hbm[local_gpu].healthy
    assert simulator.gpu_hbm[local_gpu].used_bytes == 0
    assert not simulator.node_dram["node0"].healthy
    assert simulator.node_dram["node0"].used_bytes == 0
    assert simulator.node_dram["node1"].snapshot() == remote_before
    assert simulator.extents["remote-dram"].locations[remote_location] is (
        KvCopyState.COMMITTED
    )

    simulator.register_extent(extent_id="blocked", request_id="r1", size_bytes=10)
    with pytest.raises(ValueError, match="unhealthy"):
        simulator.place_extent("blocked", KvLocation.gpu_primary(local_gpu))
    with pytest.raises(ValueError, match="unhealthy"):
        simulator.reserve_transfer(
            extent_id="remote-dram",
            source=remote_location,
            destination=KvLocation.gpu_primary(local_gpu),
            resources=(TransferResourceId("link", "node1->node0"),),
            duration_ns=1,
        )

    simulator.repair_gpu(local_gpu, weights_bytes=100, runtime_bytes=20)
    simulator.repair_node_dram("node0", runtime_bytes=100)
    assert simulator.gpu_hbm[local_gpu].healthy
    assert simulator.node_dram["node0"].healthy
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_dram_recovery_load_reserves_primary_once_and_preserves_metadata() -> None:
    simulator = GlobalDES()
    recovered_gpu = _gpu("node1", 0, 1, tp=2)
    simulator.register_gpu(recovered_gpu, capacity_bytes=1_000)
    simulator.register_node_dram("store0", capacity_bytes=2_000)
    extent = simulator.register_extent(
        extent_id="workflow0/r0/ordinal3/tp2",
        request_id="r0",
        size_bytes=96,
        route_version=4,
        ordinal=3,
        model_revision="qwen3-rev",
        kv_abi="kv-v2",
        parallel_layout="tp4-pp1",
        pp_rank=0,
        tp_rank=2,
        shard_count=4,
        source_failure_domain="node0",
        workflow_id="workflow0",
    )
    dram = KvLocation.node_dram("store0")
    primary = KvLocation.gpu_primary(recovered_gpu)
    simulator.place_extent(extent.extent_id, dram)

    transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=dram,
        destination=primary,
        resources=(TransferResourceId("pcie", "store0->node1"),),
        duration_ns=9,
    )

    assert simulator.gpu_hbm[recovered_gpu].primary_kv == 96
    assert extent.locations[primary] is KvCopyState.PENDING
    simulator.run()
    assert transfer.state is TransferState.COMMITTED
    assert simulator.gpu_hbm[recovered_gpu].primary_kv == 96
    assert extent.locations[primary] is KvCopyState.COMMITTED
    assert extent.snapshot()["route_version"] == 4
    assert extent.snapshot()["workflow_id"] == "workflow0"
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_peer_hbm_to_dram_destage_is_rejected_by_current_design() -> None:
    simulator = GlobalDES()
    peer_gpu = _gpu("node1", 0, 1)
    simulator.register_gpu(peer_gpu, capacity_bytes=1_000)
    simulator.register_node_dram("node0", capacity_bytes=1_000)
    simulator.register_extent(extent_id="e0", request_id="r0", size_bytes=32)
    peer = KvLocation.gpu_aegis(peer_gpu)
    simulator.place_extent("e0", peer)

    with pytest.raises(ValueError, match="destage is forbidden"):
        simulator.reserve_transfer(
            extent_id="e0",
            source=peer,
            destination=KvLocation.node_dram("node0"),
            resources=(TransferResourceId("link", "node1->node0"),),
            duration_ns=3,
        )


def test_release_location_returns_capacity_and_conserves_extents() -> None:
    simulator = GlobalDES()
    gpu = _gpu("node0", 0, 0)
    simulator.register_gpu(gpu, capacity_bytes=100)
    simulator.register_extent(extent_id="e0", request_id="r0", size_bytes=35)
    location = KvLocation.gpu_prefix(gpu)
    simulator.place_extent("e0", location)

    old_state = simulator.release_extent_location("e0", location)

    assert old_state is KvCopyState.COMMITTED
    assert simulator.gpu_hbm[gpu].prefix_kv == 0
    assert location not in simulator.extents["e0"].locations
    assert simulator.audit_invariants(strict_extent_accounting=True) == []
    assert any(
        row.kind == "extent_location_released" for row in simulator.journal
    )


def test_tp4_group_commits_all_destinations_only_on_fourth_ack() -> None:
    simulator = GlobalDES()
    shard_keys = tuple(f"tp{rank}" for rank in range(4))
    group = simulator.register_commit_group(
        group_id="chunk0/hbm",
        tier="gpu_aegis",
        route_version=7,
        model_revision="model-r7",
        kv_abi="kv-v3",
        parallel_layout="tp4-pp1",
        expected_shard_keys=shard_keys,
    )
    transfers = []
    destinations = []
    for rank, shard_key in enumerate(shard_keys):
        source_gpu = _gpu("source", rank, 0, tp=rank)
        target_gpu = _gpu("peer", rank, 1, tp=rank)
        simulator.register_gpu(source_gpu, capacity_bytes=1_000)
        simulator.register_gpu(target_gpu, capacity_bytes=1_000)
        extent = simulator.register_extent(
            extent_id=f"chunk0/{shard_key}",
            request_id="r0",
            size_bytes=40 + rank,
            route_version=7,
            model_revision="model-r7",
            kv_abi="kv-v3",
            parallel_layout="tp4-pp1",
            tp_rank=rank,
            shard_count=4,
        )
        source = KvLocation.gpu_primary(source_gpu)
        destination = KvLocation.gpu_aegis(target_gpu)
        simulator.place_extent(extent.extent_id, source)
        transfers.append(
            simulator.reserve_transfer(
                extent_id=extent.extent_id,
                source=source,
                destination=destination,
                resources=(TransferResourceId("link", f"shard-{rank}"),),
                duration_ns=rank + 1,
                group_id=group.group_id,
                shard_key=shard_key,
            )
        )
        destinations.append((extent, destination, target_gpu))

    simulator.run(until_ns=3)

    assert group.state is CommitGroupState.PENDING
    assert group.acked_shards == {"tp0", "tp1", "tp2"}
    assert [transfer.state for transfer in transfers[:3]] == [
        TransferState.ACKED,
        TransferState.ACKED,
        TransferState.ACKED,
    ]
    assert transfers[3].state is TransferState.IN_FLIGHT
    assert all(
        extent.locations[destination] is KvCopyState.PENDING
        for extent, destination, _gpu_id in destinations
    )
    assert simulator.audit_invariants(strict_extent_accounting=True) == []

    simulator.run()

    assert simulator.now_ns == 4
    assert group.state is CommitGroupState.COMMITTED
    assert all(transfer.state is TransferState.COMMITTED for transfer in transfers)
    assert all(
        extent.locations[destination] is KvCopyState.COMMITTED
        for extent, destination, _gpu_id in destinations
    )
    assert all(
        simulator.gpu_hbm[gpu_id].aegis_pending == 0
        and simulator.gpu_hbm[gpu_id].aegis_committed == extent.size_bytes
        for extent, _destination, gpu_id in destinations
    )
    final_event_ids = {
        row.event_id
        for row in simulator.journal
        if row.kind in {"transfer_acked", "commit_group_committed"}
        and row.timestamp_ns == 4
    }
    assert len(final_event_ids) == 1
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_third_shard_destination_failure_aborts_entire_hbm_group() -> None:
    simulator = GlobalDES()
    shard_keys = tuple(f"tp{rank}" for rank in range(4))
    group = simulator.register_commit_group(
        group_id="chunk1/hbm",
        tier="gpu_aegis",
        route_version=1,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp4-pp1",
        expected_shard_keys=shard_keys,
    )
    transfers = []
    target_gpus = []
    for rank, shard_key in enumerate(shard_keys):
        source_gpu = _gpu("source", rank, 0, tp=rank)
        target_gpu = _gpu("peer", rank, 1, tp=rank)
        target_gpus.append(target_gpu)
        simulator.register_gpu(source_gpu, capacity_bytes=1_000)
        simulator.register_gpu(target_gpu, capacity_bytes=1_000)
        extent = simulator.register_extent(
            extent_id=f"chunk1/{shard_key}",
            request_id="r1",
            size_bytes=32,
            model_revision="model-r1",
            kv_abi="kv-v1",
            parallel_layout="tp4-pp1",
            tp_rank=rank,
            shard_count=4,
        )
        source = KvLocation.gpu_primary(source_gpu)
        simulator.place_extent(extent.extent_id, source)
        transfers.append(
            simulator.reserve_transfer(
                extent_id=extent.extent_id,
                source=source,
                destination=KvLocation.gpu_aegis(target_gpu),
                resources=(TransferResourceId("link", f"abort-{rank}"),),
                duration_ns=10,
                group_id=group.group_id,
                shard_key=shard_key,
            )
        )
    simulator.run(until_ns=0)
    assert all(transfer.state is TransferState.IN_FLIGHT for transfer in transfers)

    simulator.fail_gpu(target_gpus[2])
    simulator.run()

    assert group.state is CommitGroupState.ABORTED
    assert all(transfer.state is TransferState.ABORTED for transfer in transfers)
    assert all(
        simulator.gpu_hbm[gpu].aegis_pending == 0
        and simulator.gpu_hbm[gpu].aegis_committed == 0
        for gpu in target_gpus
    )
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_failed_peer_hbm_group_does_not_abort_independent_dram_group() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("source", 0, 0)
    peer_gpu = _gpu("peer", 0, 1)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(peer_gpu, capacity_bytes=1_000)
    simulator.register_node_dram("store", capacity_bytes=1_000)
    extent = simulator.register_extent(
        extent_id="chunk2/tp0",
        request_id="r2",
        size_bytes=80,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp1-pp1",
    )
    source = KvLocation.gpu_primary(source_gpu)
    simulator.place_extent(extent.extent_id, source)
    hbm_group = simulator.register_commit_group(
        group_id="chunk2/hbm",
        tier="gpu_aegis",
        route_version=1,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp1-pp1",
        expected_shard_keys=("tp0",),
    )
    dram_group = simulator.register_commit_group(
        group_id="chunk2/dram",
        tier="node_dram",
        route_version=1,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp1-pp1",
        expected_shard_keys=("tp0",),
    )
    hbm_transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=KvLocation.gpu_aegis(peer_gpu),
        resources=(TransferResourceId("link", "hbm"),),
        duration_ns=5,
        group_id=hbm_group.group_id,
        shard_key="tp0",
    )
    dram_location = KvLocation.node_dram("store")
    dram_transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=dram_location,
        resources=(TransferResourceId("link", "dram"),),
        duration_ns=5,
        group_id=dram_group.group_id,
        shard_key="tp0",
    )

    simulator.fail_gpu(peer_gpu)
    simulator.run()

    assert hbm_group.state is CommitGroupState.ABORTED
    assert hbm_transfer.state is TransferState.ABORTED
    assert dram_group.state is CommitGroupState.COMMITTED
    assert dram_transfer.state is TransferState.COMMITTED
    assert extent.locations[dram_location] is KvCopyState.COMMITTED
    assert simulator.node_dram["store"].aegis_committed == extent.size_bytes
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_group_identity_mismatch_and_missing_shard_fail_closed() -> None:
    simulator = GlobalDES()
    with pytest.raises(ValueError, match="must not contain duplicates"):
        simulator.register_commit_group(
            group_id="duplicate-expected",
            tier="gpu_aegis",
            route_version=1,
            model_revision="model-r1",
            kv_abi="kv-v1",
            parallel_layout="tp2-pp1",
            expected_shard_keys=("tp0", "tp0"),
        )
    source_gpu = _gpu("source", 0, 0)
    peer_gpu = _gpu("peer", 0, 1)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(peer_gpu, capacity_bytes=1_000)
    extent = simulator.register_extent(
        extent_id="chunk3/tp0",
        request_id="r3",
        size_bytes=50,
        route_version=1,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp2-pp1",
        shard_count=2,
    )
    source = KvLocation.gpu_primary(source_gpu)
    destination = KvLocation.gpu_aegis(peer_gpu)
    simulator.place_extent(extent.extent_id, source)
    mismatch = simulator.register_commit_group(
        group_id="chunk3/mismatch",
        tier="gpu_aegis",
        route_version=2,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp2-pp1",
        expected_shard_keys=("tp0", "tp1"),
    )

    with pytest.raises(ValueError, match="identity/layout/version/tier mismatch"):
        simulator.reserve_transfer(
            extent_id=extent.extent_id,
            source=source,
            destination=destination,
            resources=(TransferResourceId("link", "mismatch"),),
            duration_ns=1,
            group_id=mismatch.group_id,
            shard_key="tp0",
        )
    assert mismatch.shard_transfers == {}
    assert destination not in extent.locations
    assert simulator.gpu_hbm[peer_gpu].aegis_pending == 0

    partial = simulator.register_commit_group(
        group_id="chunk3/partial",
        tier="gpu_aegis",
        route_version=1,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp2-pp1",
        expected_shard_keys=("tp0", "tp1"),
    )
    only_transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=destination,
        resources=(TransferResourceId("link", "partial"),),
        duration_ns=1,
        group_id=partial.group_id,
        shard_key="tp0",
    )
    simulator.run()

    assert only_transfer.state is TransferState.ACKED
    assert partial.state is CommitGroupState.PENDING
    assert partial.acked_shards == {"tp0"}
    assert extent.locations[destination] is KvCopyState.PENDING
    assert simulator.gpu_hbm[peer_gpu].aegis_committed == 0
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_external_source_alias_is_not_double_counted_across_hbm_dram_groups() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("source", 0, 0)
    peer_gpu = _gpu("peer", 0, 1)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(peer_gpu, capacity_bytes=1_000)
    simulator.register_node_dram("store", capacity_bytes=1_000)
    simulator.adjust_gpu_hbm(
        source_gpu,
        HbmAccount.PRIMARY_KV,
        100,
        reason="scheduler_aggregate",
    )
    common = {
        "request_id": "r-external",
        "size_bytes": 64,
        "token_start": 0,
        "token_end": 16,
        "model_revision": "model-r1",
        "kv_abi": "kv-v1",
        "parallel_layout": "tp1-pp1",
    }
    hbm_extent = simulator.register_extent(extent_id="external/hbm", **common)
    dram_extent = simulator.register_extent(extent_id="external/dram", **common)
    source = KvLocation.gpu_primary(source_gpu)
    source_range_key = "r-external:tokens0-16:pp0:tp0"
    simulator.alias_external_source_location(
        hbm_extent.extent_id,
        source,
        external_account_key=source_range_key,
    )
    simulator.alias_external_source_location(
        dram_extent.extent_id,
        source,
        external_account_key=source_range_key,
    )

    assert simulator.gpu_hbm[source_gpu].primary_kv == 100
    source_rows = {
        extent.extent_id: next(
            row
            for row in extent.snapshot()["locations"]
            if row["location"]["tier"] == "gpu_primary"
        )
        for extent in (hbm_extent, dram_extent)
    }
    assert all(
        row["accounting"] == LocationAccounting.EXTERNAL.value
        and row["external_account_key"] == source_range_key
        for row in source_rows.values()
    )

    hbm_group = simulator.register_commit_group(
        group_id="external/hbm-group",
        tier="gpu_aegis",
        route_version=1,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp1-pp1",
        expected_shard_keys=("tp0",),
    )
    dram_group = simulator.register_commit_group(
        group_id="external/dram-group",
        tier="node_dram",
        route_version=1,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="tp1-pp1",
        expected_shard_keys=("tp0",),
    )
    hbm_transfer = simulator.reserve_transfer(
        extent_id=hbm_extent.extent_id,
        source=source,
        destination=KvLocation.gpu_aegis(peer_gpu),
        resources=(TransferResourceId("link", "external-hbm"),),
        duration_ns=2,
        group_id=hbm_group.group_id,
        shard_key="tp0",
    )
    dram_transfer = simulator.reserve_transfer(
        extent_id=dram_extent.extent_id,
        source=source,
        destination=KvLocation.node_dram("store"),
        resources=(TransferResourceId("link", "external-dram"),),
        duration_ns=3,
        group_id=dram_group.group_id,
        shard_key="tp0",
    )
    simulator.run()

    assert hbm_transfer.state is TransferState.COMMITTED
    assert dram_transfer.state is TransferState.COMMITTED
    assert hbm_group.state is CommitGroupState.COMMITTED
    assert dram_group.state is CommitGroupState.COMMITTED
    assert simulator.gpu_hbm[source_gpu].primary_kv == 100
    assert simulator.gpu_hbm[peer_gpu].aegis_committed == 64
    assert simulator.node_dram["store"].aegis_committed == 64
    assert simulator.audit_invariants(
        extent_accounting_mode=ExtentAccountingAuditMode.EXTERNAL_AGGREGATE
    ) == []
    assert simulator.audit_invariants(strict_extent_accounting=True)


def test_releasing_external_alias_does_not_decrement_scheduler_aggregate() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("source", 0, 0)
    simulator.register_gpu(source_gpu, capacity_bytes=500)
    simulator.adjust_gpu_hbm(source_gpu, HbmAccount.PRIMARY_KV, 100)
    extent = simulator.register_extent(
        extent_id="release-external", request_id="r0", size_bytes=40
    )
    source = KvLocation.gpu_primary(source_gpu)
    simulator.alias_external_source_location(
        extent.extent_id,
        source,
        external_account_key="r0:production",
    )

    state = simulator.release_extent_location(extent.extent_id, source)

    assert state is KvCopyState.COMMITTED
    assert simulator.gpu_hbm[source_gpu].primary_kv == 100
    assert source not in extent.locations
    assert simulator.audit_invariants(
        extent_accounting_mode="external_aggregate"
    ) == []


def test_gpu_failure_marks_external_alias_lost_without_negative_accounting() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("source", 0, 0)
    simulator.register_gpu(source_gpu, capacity_bytes=500)
    simulator.adjust_gpu_hbm(source_gpu, HbmAccount.PRIMARY_KV, 100)
    extent = simulator.register_extent(
        extent_id="failed-external", request_id="r0", size_bytes=40
    )
    source = KvLocation.gpu_primary(source_gpu)
    simulator.alias_external_source_location(
        extent.extent_id,
        source,
        external_account_key="r0:production",
    )

    simulator.fail_gpu(source_gpu)

    assert simulator.gpu_hbm[source_gpu].primary_kv == 0
    assert simulator.gpu_hbm[source_gpu].used_bytes == 0
    assert extent.locations[source] is KvCopyState.LOST
    assert extent.location_accounting[source] is LocationAccounting.EXTERNAL
    assert simulator.audit_invariants(
        extent_accounting_mode="external_aggregate"
    ) == []


def test_external_alias_capacity_and_shared_key_identity_fail_closed() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("source", 0, 0)
    simulator.register_gpu(source_gpu, capacity_bytes=500)
    simulator.adjust_gpu_hbm(source_gpu, HbmAccount.PRIMARY_KV, 50)
    source = KvLocation.gpu_primary(source_gpu)
    first = simulator.register_extent(
        extent_id="first", request_id="r0", size_bytes=30
    )
    overflow = simulator.register_extent(
        extent_id="overflow", request_id="r0", size_bytes=30
    )
    conflict = simulator.register_extent(
        extent_id="conflict", request_id="r0", size_bytes=31
    )
    simulator.alias_external_source_location(
        first.extent_id,
        source,
        external_account_key="range-a",
    )

    with pytest.raises(ValueError, match="external live bytes 60 exceed"):
        simulator.alias_external_source_location(
            overflow.extent_id,
            source,
            external_account_key="range-b",
        )
    with pytest.raises(ValueError, match="size/identity conflict"):
        simulator.alias_external_source_location(
            conflict.extent_id,
            source,
            external_account_key="range-a",
        )

    assert source not in overflow.locations
    assert source not in conflict.locations
    assert simulator.gpu_hbm[source_gpu].primary_kv == 50
    assert simulator.audit_invariants(
        extent_accounting_mode="external_aggregate"
    ) == []


def test_external_recovery_partial_span_holds_capacity_without_destination() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("peer", 0, 1)
    target_gpu = _gpu("retry", 0, 2)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(target_gpu, capacity_bytes=1_000)
    extent = simulator.register_extent(
        extent_id="recovery-partial",
        request_id="r-partial",
        size_bytes=160,
        token_start=0,
        token_end=16,
        route_version=3,
    )
    source = KvLocation.gpu_aegis(source_gpu)
    destination = KvLocation.gpu_primary(target_gpu)
    simulator.place_extent(extent.extent_id, source)
    group = simulator.register_recovery_load_group(
        group_id="recovery-partial-group",
        retry_request_id="retry-partial",
        request_backup_key="r-partial",
        route_version=3,
        expected_shard_keys=("pp0/tp0",),
    )
    transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=destination,
        resources=(TransferResourceId("h2d", target_gpu.key),),
        duration_ns=10,
        destination_mode=TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK,
        transfer_size_bytes=80,
        token_start=4,
        token_end=12,
        recovery_group_id=group.group_id,
        recovery_shard_key="pp0/tp0",
    )
    simulator.seal_recovery_load_group(group.group_id)

    assert destination not in extent.locations
    assert simulator.gpu_hbm[target_gpu].recovery_pending == 80
    assert transfer.size_bytes == 80
    assert (transfer.token_start, transfer.token_end) == (4, 12)

    simulator.run()

    assert group.state is CommitGroupState.COMMITTED
    assert transfer.state is TransferState.COMMITTED
    # The scheduler has not adopted the receive pages yet, so their capacity
    # remains protected in RECOVERY_PENDING after physical completion.
    assert simulator.gpu_hbm[target_gpu].recovery_pending == 80
    assert destination not in extent.locations
    assert simulator.audit_invariants(strict_extent_accounting=True) == []

    simulator.fail_gpu(target_gpu)
    assert simulator.gpu_hbm[target_gpu].recovery_pending == 0
    assert not transfer.recovery_capacity_hold_active
    assert transfer.recovery_capacity_hold_lost
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


@pytest.mark.parametrize(
    "failure_ns",
    (0, 5, 10),
    ids=("at-start", "mid-transfer", "at-completion"),
)
def test_external_recovery_source_failure_aborts_group_and_releases_hold(
    failure_ns: int,
) -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("peer", 0, 1)
    target_gpu = _gpu("retry", 0, 2)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(target_gpu, capacity_bytes=1_000)
    extent = simulator.register_extent(
        extent_id=f"failure-{failure_ns}",
        request_id="r-failure",
        size_bytes=100,
        token_start=0,
        token_end=10,
        route_version=1,
    )
    source = KvLocation.gpu_aegis(source_gpu)
    destination = KvLocation.gpu_primary(target_gpu)
    simulator.place_extent(extent.extent_id, source)
    group = simulator.register_recovery_load_group(
        group_id=f"failure-group-{failure_ns}",
        retry_request_id=f"retry-{failure_ns}",
        request_backup_key="r-failure",
        route_version=1,
        expected_shard_keys=("pp0/tp0",),
    )
    transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=destination,
        resources=(TransferResourceId("link", f"failure-{failure_ns}"),),
        duration_ns=10,
        destination_mode="EXTERNAL_PREALLOCATED_SINK",
        recovery_group_id=group.group_id,
        recovery_shard_key="pp0/tp0",
    )
    simulator.seal_recovery_load_group(group.group_id)
    simulator.schedule_at(
        failure_ns,
        kind="recovery-source-failure",
        priority=EventPriority.FAILURE,
        callback=lambda sim, _event: sim.fail_gpu(source_gpu),
    )

    simulator.run()

    assert group.state is CommitGroupState.ABORTED
    assert transfer.state is TransferState.ABORTED
    assert simulator.gpu_hbm[target_gpu].recovery_pending == 0
    assert not transfer.recovery_capacity_hold_active
    assert destination not in extent.locations
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_external_recovery_capacity_overflow_is_atomic_and_abort_releases() -> None:
    simulator = GlobalDES()
    source_gpu = _gpu("peer", 0, 1)
    target_gpu = _gpu("retry", 0, 2)
    simulator.register_gpu(source_gpu, capacity_bytes=1_000)
    simulator.register_gpu(target_gpu, capacity_bytes=100, weights_bytes=30)
    extent = simulator.register_extent(
        extent_id="capacity-recovery",
        request_id="r-capacity",
        size_bytes=80,
        token_start=0,
        token_end=8,
    )
    source = KvLocation.gpu_aegis(source_gpu)
    destination = KvLocation.gpu_primary(target_gpu)
    simulator.place_extent(extent.extent_id, source)
    resource = TransferResourceId("link", "capacity")

    with pytest.raises(ValueError, match="exceeds capacity"):
        simulator.reserve_transfer(
            extent_id=extent.extent_id,
            source=source,
            destination=destination,
            resources=(resource,),
            duration_ns=10,
            destination_mode="EXTERNAL_PREALLOCATED_SINK",
        )
    assert simulator.gpu_hbm[target_gpu].recovery_pending == 0
    assert resource not in simulator.resource_busy_until_ns
    assert destination not in extent.locations

    transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=destination,
        resources=(resource,),
        duration_ns=10,
        destination_mode="EXTERNAL_PREALLOCATED_SINK",
        transfer_size_bytes=60,
        token_start=0,
        token_end=6,
    )
    assert simulator.gpu_hbm[target_gpu].recovery_pending == 60
    simulator.abort_transfer(transfer.transfer_id, reason="test_release")
    assert simulator.gpu_hbm[target_gpu].recovery_pending == 0
    assert transfer.state is TransferState.ABORTED
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_same_gpu_ownership_activation_has_no_hbm_peak_increase() -> None:
    simulator = GlobalDES()
    gpu = _gpu("retry", 0, 1)
    simulator.register_gpu(gpu, capacity_bytes=1_000, weights_bytes=100)
    extent = simulator.register_extent(
        extent_id="activation",
        request_id="r-activation",
        size_bytes=128,
        token_start=0,
        token_end=16,
    )
    source = KvLocation.gpu_aegis(gpu)
    destination = KvLocation.gpu_primary(gpu)
    simulator.place_extent(extent.extent_id, source)
    used_before = simulator.gpu_hbm[gpu].used_bytes
    group = simulator.register_recovery_load_group(
        group_id="activation-group",
        retry_request_id="retry-activation",
        request_backup_key="r-activation",
        route_version=1,
        expected_shard_keys=("pp0/tp0",),
    )
    transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=destination,
        resources=(),
        duration_ns=0,
        destination_mode="OWNERSHIP_ACTIVATION",
        recovery_group_id=group.group_id,
        recovery_shard_key="pp0/tp0",
    )
    simulator.seal_recovery_load_group(group.group_id)

    assert simulator.gpu_hbm[gpu].used_bytes == used_before
    assert simulator.gpu_hbm[gpu].recovery_pending == 0
    assert destination not in extent.locations
    simulator.run()
    assert transfer.state is TransferState.COMMITTED
    assert group.state is CommitGroupState.COMMITTED
    assert simulator.gpu_hbm[gpu].used_bytes == used_before
    assert destination not in extent.locations
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def _build_pp2_tp4_recovery_group(
    simulator: GlobalDES,
    *,
    group_id: str,
    omit_shard: str | None = None,
):
    shard_keys = tuple(
        f"pp{pp_rank}/tp{tp_rank}"
        for pp_rank in range(2)
        for tp_rank in range(4)
    )
    group = simulator.register_recovery_load_group(
        group_id=group_id,
        retry_request_id=f"retry-{group_id}",
        request_backup_key="r-grid",
        route_version=1,
        expected_shard_keys=shard_keys,
    )
    transfers = []
    targets = []
    for pp_rank in range(2):
        for tp_rank in range(4):
            shard_key = f"pp{pp_rank}/tp{tp_rank}"
            source_gpu = PhysicalGpuId(
                node_id="peer",
                local_gpu_id=pp_rank * 4 + tp_rank,
                dp_rank=0,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
            )
            target_gpu = PhysicalGpuId(
                node_id="retry",
                local_gpu_id=pp_rank * 4 + tp_rank,
                dp_rank=1,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
            )
            simulator.register_gpu(source_gpu, capacity_bytes=100)
            simulator.register_gpu(target_gpu, capacity_bytes=100)
            targets.append(target_gpu)
            extent = simulator.register_extent(
                extent_id=f"{group_id}/{shard_key}",
                request_id="r-grid",
                size_bytes=10,
                token_start=0,
                token_end=1,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                shard_count=4,
            )
            source = KvLocation.gpu_aegis(source_gpu)
            simulator.place_extent(extent.extent_id, source)
            if shard_key == omit_shard:
                continue
            transfers.append(
                simulator.reserve_transfer(
                    extent_id=extent.extent_id,
                    source=source,
                    destination=KvLocation.gpu_primary(target_gpu),
                    resources=(TransferResourceId("link", shard_key),),
                    duration_ns=10,
                    destination_mode="EXTERNAL_PREALLOCATED_SINK",
                    recovery_group_id=group.group_id,
                    recovery_shard_key=shard_key,
                )
            )
    return group, transfers, targets


def test_pp2_tp4_missing_recovery_shard_aborts_and_releases_all_holds() -> None:
    simulator = GlobalDES()
    group, transfers, targets = _build_pp2_tp4_recovery_group(
        simulator,
        group_id="missing-grid",
        omit_shard="pp1/tp3",
    )

    with pytest.raises(ValueError, match="missing shards: pp1/tp3"):
        simulator.seal_recovery_load_group(group.group_id)

    assert group.state is CommitGroupState.ABORTED
    assert all(transfer.state is TransferState.ABORTED for transfer in transfers)
    assert all(simulator.gpu_hbm[gpu].recovery_pending == 0 for gpu in targets)
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_pp2_tp4_one_transfer_abort_fails_entire_recovery_group() -> None:
    simulator = GlobalDES()
    group, transfers, targets = _build_pp2_tp4_recovery_group(
        simulator,
        group_id="abort-grid",
    )
    simulator.seal_recovery_load_group(group.group_id)
    simulator.run(until_ns=0)
    assert all(transfer.state is TransferState.IN_FLIGHT for transfer in transfers)

    simulator.abort_transfer(transfers[5].transfer_id, reason="one_shard_failed")

    assert group.state is CommitGroupState.ABORTED
    assert all(transfer.state is TransferState.ABORTED for transfer in transfers)
    assert all(simulator.gpu_hbm[gpu].recovery_pending == 0 for gpu in targets)
    simulator.run()
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_pp2_tp4_recovery_adopts_only_the_prepared_pp_shards() -> None:
    simulator = GlobalDES()
    group, transfers, targets = _build_pp2_tp4_recovery_group(
        simulator,
        group_id="adopt-grid",
    )
    simulator.seal_recovery_load_group(group.group_id)
    simulator.run()

    assert group.state is CommitGroupState.COMMITTED
    assert all(transfer.state is TransferState.COMMITTED for transfer in transfers)
    assert all(simulator.gpu_hbm[gpu].recovery_pending == 10 for gpu in targets)
    assert simulator.recovery_load_shards_adoption_error(group.group_id) is None

    pp0 = tuple(f"pp0/tp{tp_rank}" for tp_rank in range(4))
    simulator.adopt_recovery_load_shards(group.group_id, pp0)
    assert group.adopted_shards == set(pp0)
    assert group.adopted_at_ns is None
    assert all(
        simulator.gpu_hbm[gpu].recovery_pending
        == (0 if gpu.pp_rank == 0 else 10)
        for gpu in targets
    )

    pp1 = tuple(f"pp1/tp{tp_rank}" for tp_rank in range(4))
    simulator.adopt_recovery_load_shards(group.group_id, pp1)
    # Idempotence is required because a buffered Scheduler stage may publish a
    # second prepared boundary for the same logical batch.
    simulator.adopt_recovery_load_shard(group.group_id, "pp1/tp0")
    assert group.adopted_shards == set(group.expected_shard_keys)
    assert group.adopted_at_ns == simulator.now_ns
    assert all(not transfer.recovery_capacity_hold_active for transfer in transfers)
    assert all(transfer.recovery_adopted for transfer in transfers)
    assert all(simulator.gpu_hbm[gpu].recovery_pending == 0 for gpu in targets)
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_pp2_tp4_target_failure_abandons_only_unadopted_shards() -> None:
    simulator = GlobalDES()
    group, transfers, targets = _build_pp2_tp4_recovery_group(
        simulator,
        group_id="partial-failure-grid",
    )
    simulator.seal_recovery_load_group(group.group_id)
    simulator.run()
    pp0 = tuple(f"pp0/tp{tp_rank}" for tp_rank in range(4))
    simulator.adopt_recovery_load_shards(group.group_id, pp0)

    simulator.abandon_partially_adopted_recovery_group(
        group.group_id,
        reason="target_hard_failure:test",
    )

    assert group.state is CommitGroupState.COMMITTED
    assert group.discarded is True
    assert group.adopted_shards == set(pp0)
    assert group.adopted_at_ns is None
    snapshot = group.snapshot()
    assert snapshot["discard_scope"] == "unadopted_remainder"
    assert snapshot["abandoned_shards"] == [
        f"pp1/tp{tp_rank}" for tp_rank in range(4)
    ]
    assert snapshot["terminal_reason"] == "target_hard_failure:test"
    for transfer in transfers:
        if transfer.recovery_shard_key in pp0:
            assert transfer.recovery_adopted is True
            assert transfer.recovery_discarded is False
        else:
            assert transfer.recovery_adopted is False
            assert transfer.recovery_discarded is True
    assert all(simulator.gpu_hbm[gpu].recovery_pending == 0 for gpu in targets)
    assert simulator.audit_invariants(strict_extent_accounting=True) == []


def test_same_gpu_activation_adoption_hands_off_without_peak() -> None:
    simulator = GlobalDES()
    gpu = _gpu("activation", 0, 1)
    simulator.register_gpu(gpu, capacity_bytes=1_000, weights_bytes=100)
    extent = simulator.register_extent(
        extent_id="activation-adopt",
        request_id="r-activation-adopt",
        size_bytes=128,
        token_start=0,
        token_end=16,
    )
    source = KvLocation.gpu_aegis(gpu)
    simulator.place_extent(extent.extent_id, source)
    group = simulator.register_recovery_load_group(
        group_id="activation-adopt-group",
        retry_request_id="retry-activation-adopt",
        request_backup_key="r-activation-adopt",
        route_version=1,
        expected_shard_keys=("pp0/tp0",),
    )
    transfer = simulator.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=KvLocation.gpu_primary(gpu),
        resources=(),
        duration_ns=0,
        destination_mode="OWNERSHIP_ACTIVATION",
        recovery_group_id=group.group_id,
        recovery_shard_key="pp0/tp0",
    )
    simulator.seal_recovery_load_group(group.group_id)
    used_before = simulator.gpu_hbm[gpu].used_bytes
    simulator.run()

    simulator.adopt_recovery_load_shard(group.group_id, "pp0/tp0")

    assert transfer.recovery_adopted
    assert source not in extent.locations
    assert simulator.gpu_hbm[gpu].used_bytes == used_before - extent.size_bytes
    assert simulator.gpu_hbm[gpu].recovery_pending == 0
    assert simulator.audit_invariants(strict_extent_accounting=True) == []
