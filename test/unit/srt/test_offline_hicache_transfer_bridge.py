from __future__ import annotations

import json

import pytest

from sglsim.srt.mem_cache.hiradix_cache import HiRadixCache
from sglsim.srt.simulation.global_des import (
    DramAccount,
    EventPriority,
    GlobalDES,
    HbmAccount,
    KvLocation,
    PhysicalGpuId,
    TransferResourceId,
)
from sglsim.srt.simulation.global_des_artifacts import (
    ArtifactInvariantError,
    write_global_des_artifacts,
)
from sglsim.srt.simulation.offline_hicache_transfer_bridge import (
    OfflineHiCacheTransferBridge,
    OrdinaryHiCacheTransferState,
)
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    PhysicalGpuSpec,
)


PAGE_SIZE = 4
PAGES = 2
TOKENS = PAGE_SIZE * PAGES
KV_BYTES = 10
SIZE_BYTES = TOKENS * KV_BYTES


def _topology(*, pp_size: int = 1, tp_size: int = 2) -> OfflinePhysicalTopology:
    gpus = []
    for pp_rank in range(pp_size):
        node_id = f"node{pp_rank}"
        for tp_rank in range(tp_size):
            gpus.append(
                PhysicalGpuSpec(
                    gpu=PhysicalGpuId(
                        node_id=node_id,
                        local_gpu_id=tp_rank,
                        dp_rank=0,
                        pp_rank=pp_rank,
                        tp_rank=tp_rank,
                    ),
                    hbm_capacity_bytes=100_000,
                )
            )
    return OfflinePhysicalTopology(
        gpus=tuple(gpus),
        nodes=tuple(
            NodeDramSpec(f"node{pp_rank}", capacity_bytes=1_000_000)
            for pp_rank in range(pp_size)
        ),
    )


def _des(topology: OfflinePhysicalTopology) -> GlobalDES:
    des = GlobalDES()
    for node in topology.nodes:
        des.register_node_dram(node.node_id, capacity_bytes=node.capacity_bytes)
    for spec in topology.gpus:
        des.register_gpu(spec.gpu, capacity_bytes=spec.hbm_capacity_bytes)
        des.adjust_gpu_hbm(
            spec.gpu,
            HbmAccount.PREFIX_KV,
            SIZE_BYTES,
            reason="scheduler_provisional_l1",
        )
        des.adjust_node_dram(
            spec.gpu.node_id,
            DramAccount.HICACHE_PRIMARY,
            SIZE_BYTES,
            reason="scheduler_provisional_l2",
        )
    return des


def _intent(
    *,
    direction: str = "d2h",
    pp_rank: int = 0,
    sequence: int = 0,
    blocks_compute: bool | None = None,
) -> dict:
    return {
        "schema": "sglsim.offline.hicache-intent.v1",
        "intent_id": f"hicache-dp0-pp{pp_rank}-tp0-i0-{sequence:012d}",
        "direction": direction,
        "owner": "request-0",
        "radix_node_id": 7,
        "host_page_ids": [10, 11],
        "device_page_ids": [20, 21],
        "page_size_tokens": PAGE_SIZE,
        "page_count": PAGES,
        "token_count": TOKENS,
        "kv_bytes_per_token_per_tp_rank": KV_BYTES,
        "size_bytes_per_tp_rank": SIZE_BYTES,
        "dp_rank": 0,
        "pp_rank": pp_rank,
        "scheduler_tp_rank": 0,
        "worker_incarnation": 0,
        "logical_tp_scope": "one-symmetric-tp-rank-view",
        "requires_parent_tp_expansion": True,
        "operation_sequence": sequence,
        "predicted_duration_s": 0.01,
        "blocks_compute": (
            direction == "h2d" if blocks_compute is None else blocks_compute
        ),
        "state": "PENDING",
    }


def test_d2h_tp_expansion_commits_without_aegis_account_pollution() -> None:
    topology = _topology(tp_size=2)
    des = _des(topology)
    terminal = []
    bridge = OfflineHiCacheTransferBridge(
        des, topology, on_group_terminal=terminal.append
    )
    before_hbm = {
        gpu: (ledger.aegis_pending, ledger.aegis_committed)
        for gpu, ledger in des.gpu_hbm.items()
    }
    before_dram = {
        node: (ledger.aegis_pending, ledger.aegis_committed)
        for node, ledger in des.node_dram.items()
    }

    group = bridge.submit_intent(_intent())

    assert len(group.transfer_ids) == 2
    assert {bridge.transfers[tid].tp_rank for tid in group.transfer_ids} == {0, 1}
    assert all(
        "pcie-d2h:node0:lane0"
        in {resource.key for resource in bridge.transfers[tid].resources}
        for tid in group.transfer_ids
    )
    des.run()

    assert group.state is OrdinaryHiCacheTransferState.COMMITTED
    assert len(terminal) == 1
    assert terminal[0]["state"] == "COMMITTED"
    assert bridge.audit() == []
    assert before_hbm == {
        gpu: (ledger.aegis_pending, ledger.aegis_committed)
        for gpu, ledger in des.gpu_hbm.items()
    }
    assert before_dram == {
        node: (ledger.aegis_pending, ledger.aegis_committed)
        for node, ledger in des.node_dram.items()
    }


def test_h2d_uses_shared_resource_calendar_and_blocks_compute() -> None:
    topology = _topology(tp_size=1)
    des = _des(topology)
    gpu = topology.gpus[0].gpu
    busy_resource = TransferResourceId("pcie-h2d", gpu.node_id)
    des.reserve_resources((busy_resource,), duration_ns=1_000, ready_ns=0)
    bridge = OfflineHiCacheTransferBridge(des, topology)

    group = bridge.submit_intent(_intent(direction="h2d"))
    transfer = bridge.transfers[group.transfer_ids[0]]

    assert group.intent.blocks_compute is True
    assert transfer.start_ns >= 1_000
    assert "dram-read:node0:lane0" in {
        resource.key for resource in transfer.resources
    }
    des.run()
    assert group.state is OrdinaryHiCacheTransferState.COMMITTED


def test_ordinary_d2h_contends_with_global_des_aegis_copy() -> None:
    topology = _topology(tp_size=1)
    des = _des(topology)
    gpu = topology.gpus[0].gpu
    source = KvLocation.gpu_primary(gpu)
    destination = KvLocation.node_dram(gpu.node_id)
    extent = des.register_extent(
        extent_id="aegis-contention-extent",
        request_id="aegis-request",
        size_bytes=16,
    )
    des.place_extent(extent.extent_id, source)
    aegis = des.reserve_transfer(
        extent_id=extent.extent_id,
        source=source,
        destination=destination,
        resources=(
            TransferResourceId("gpu-d2h-read", gpu.key),
            TransferResourceId("pcie-d2h", gpu.node_id),
            TransferResourceId("dram-write", gpu.node_id),
        ),
        duration_ns=10_000,
    )
    bridge = OfflineHiCacheTransferBridge(des, topology)

    group = bridge.submit_intent(_intent(direction="d2h"))
    ordinary = bridge.transfers[group.transfer_ids[0]]

    assert ordinary.start_ns >= aegis.end_ns
    des.run()
    assert group.state is OrdinaryHiCacheTransferState.COMMITTED


@pytest.mark.parametrize("boundary", ["start", "mid", "completion"])
def test_gpu_failure_start_mid_or_exact_completion_aborts(boundary: str) -> None:
    topology = _topology(tp_size=1)
    des = _des(topology)
    bridge = OfflineHiCacheTransferBridge(des, topology)
    group = bridge.submit_intent(_intent(direction="h2d"))
    transfer = bridge.transfers[group.transfer_ids[0]]
    when = {
        "start": transfer.start_ns,
        "mid": (transfer.start_ns + transfer.end_ns) // 2,
        "completion": transfer.end_ns,
    }[boundary]

    def fail(simulator, _event):
        bridge.fail_gpu(transfer.gpu, reason="unit_gpu_failure")
        simulator.fail_gpu(transfer.gpu)

    des.schedule_at(
        when,
        kind="unit_gpu_failure",
        priority=EventPriority.FAILURE,
        callback=fail,
    )
    des.run()

    assert group.state is OrdinaryHiCacheTransferState.ABORTED
    assert bridge.transfers[transfer.transfer_id].state is (
        OrdinaryHiCacheTransferState.ABORTED
    )
    assert bridge.audit() == []


def test_node_failure_aborts_every_tp_shard_atomically() -> None:
    topology = _topology(tp_size=4)
    des = _des(topology)
    bridge = OfflineHiCacheTransferBridge(des, topology)
    group = bridge.submit_intent(_intent())
    bridge.fail_node("node0", reason="unit_node_failure")
    des.fail_node("node0")
    des.run()

    assert group.state is OrdinaryHiCacheTransferState.ABORTED
    assert {
        bridge.transfers[tid].state for tid in group.transfer_ids
    } == {OrdinaryHiCacheTransferState.ABORTED}


def test_first_tp_ack_then_second_exact_failure_aborts_entire_group() -> None:
    topology = _topology(tp_size=2)
    des = _des(topology)
    bridge = OfflineHiCacheTransferBridge(des, topology)
    group = bridge.submit_intent(_intent(direction="h2d"))
    ordered = sorted(
        (bridge.transfers[tid] for tid in group.transfer_ids),
        key=lambda transfer: transfer.end_ns,
    )
    first, second = ordered
    assert first.end_ns < second.end_ns  # shared node PCIe serializes TP shards

    def fail_second(simulator, _event):
        bridge.fail_gpu(second.gpu, reason="unit_exact_second_shard_failure")
        simulator.fail_gpu(second.gpu)

    des.schedule_at(
        second.end_ns,
        kind="unit_exact_second_shard_failure",
        priority=EventPriority.FAILURE,
        callback=fail_second,
    )
    des.run()

    assert group.state is OrdinaryHiCacheTransferState.ABORTED
    assert first.state is OrdinaryHiCacheTransferState.ABORTED
    assert second.state is OrdinaryHiCacheTransferState.ABORTED
    assert bridge.audit() == []


def test_pp2_tp4_expands_only_selected_stage() -> None:
    topology = _topology(pp_size=2, tp_size=4)
    des = _des(topology)
    bridge = OfflineHiCacheTransferBridge(des, topology)
    group = bridge.submit_intent(_intent(pp_rank=1))

    assert len(group.transfer_ids) == 4
    assert {
        bridge.transfers[tid].gpu.pp_rank for tid in group.transfer_ids
    } == {1}
    assert {
        bridge.transfers[tid].gpu.tp_rank for tid in group.transfer_ids
    } == {0, 1, 2, 3}


def test_missing_scheduler_capacity_fails_closed_without_orphan() -> None:
    topology = _topology(tp_size=1)
    des = GlobalDES()
    des.register_node_dram("node0", capacity_bytes=1_000_000)
    des.register_gpu(topology.gpus[0].gpu, capacity_bytes=100_000)
    bridge = OfflineHiCacheTransferBridge(des, topology)

    with pytest.raises(ValueError, match="absent from DRAM snapshot"):
        bridge.submit_intent(_intent())

    group = bridge.groups[_intent()["intent_id"]]
    assert group.state is OrdinaryHiCacheTransferState.ABORTED
    assert bridge.audit() == []


def test_artifact_unifies_ordinary_and_global_des_transfers(tmp_path) -> None:
    topology = _topology(tp_size=1)
    des = _des(topology)
    bridge = OfflineHiCacheTransferBridge(des, topology)
    bridge.submit_intent(_intent(direction="h2d"))
    des.run()

    cell = write_global_des_artifacts(
        tmp_path / "cell",
        des=des,
        rank_steps=[],
        requests=[],
        failures=[],
        resolved_config={"request": {}, "failure": {}, "deployment": {}},
        lineage_metadata={"code_hash": "unit-code"},
        extent_accounting_mode="scheduler_aggregate",
        ordinary_hicache_transfers=bridge.export(),
    )

    transfers = [
        json.loads(line)
        for line in (cell / "transfers.jsonl").read_text().splitlines()
    ]
    groups = [
        json.loads(line)
        for line in (cell / "transfer_groups.jsonl").read_text().splitlines()
    ]
    audit = json.loads((cell / "audit.json").read_text())
    assert [row["transfer_class"] for row in transfers] == ["ordinary_hicache"]
    assert [row["group_class"] for row in groups] == ["ordinary_hicache"]
    assert audit["details"]["ordinary_hicache"]["accounting"][
        "aegis_accounts_used"
    ] is False


def test_final_artifact_refuses_nonterminal_ordinary_transfer(tmp_path) -> None:
    topology = _topology(tp_size=2)
    des = _des(topology)
    bridge = OfflineHiCacheTransferBridge(des, topology)
    bridge.submit_intent(_intent(direction="h2d"))

    with pytest.raises(ArtifactInvariantError, match="remains pending"):
        write_global_des_artifacts(
            tmp_path / "cell",
            des=des,
            rank_steps=[],
            requests=[],
            failures=[],
            resolved_config={"request": {}, "failure": {}, "deployment": {}},
            lineage_metadata={"code_hash": "unit-code"},
            extent_accounting_mode="scheduler_aggregate",
            ordinary_hicache_transfers=bridge.export(),
        )


def _offline_cache(*, write_policy: str = "write_through") -> HiRadixCache:
    cache = HiRadixCache(
        page_size=1,
        l1_capacity_pages=16,
        l2_capacity_pages=16,
        write_policy=write_policy,
        write_through_threshold=1,
        load_back_threshold=1,
    )
    cache.cache_controller.enable_offline_des(
        dp_rank=0,
        pp_rank=0,
        scheduler_tp_rank=0,
        kv_bytes_per_token_per_tp_rank=10,
        worker_incarnation=3,
    )
    return cache


def _insert_device_prefix(cache: HiRadixCache):
    pages = cache.mem_pool_device.alloc(1, owner="request-0")
    assert pages is not None
    result = cache.tree.insert([1], [pages[0].page_id])
    return result.last_node


def _force_tree_evict(cache: HiRadixCache, *, owner: str) -> int:
    pool_pages = cache.mem_pool_device._allocator._pages_by_id
    freed = cache._evict_device_pages(
        1,
        pool_pages,
        cache.cache_controller.write_policy,
        owner,
    )
    cache.mem_pool_device.free(freed)
    return len(freed)


def test_d2h_host_copy_is_not_recoverable_until_ack() -> None:
    cache = _offline_cache()
    node = _insert_device_prefix(cache)

    cache.write_backup(node, "request-0")
    [intent] = cache.cache_controller.drain_offline_intents()

    assert intent["intent_id"].startswith("hicache-dp0-pp0-tp0-i3-")
    assert node.host_value is None
    assert cache.mem_pool_host is not None
    assert cache.mem_pool_host.used_pages == 1
    assert cache.match_prefix([1]).host_length == 0

    cache.apply_offline_hicache_acks(
        [{"intent_id": intent["intent_id"], "state": "COMMITTED"}]
    )

    assert node.host_value is not None
    assert cache.offline_hicache_intent_audit() == []


def test_d2h_abort_releases_provisional_host_capacity() -> None:
    cache = _offline_cache()
    node = _insert_device_prefix(cache)
    cache.write_backup(node, "request-0")
    [intent] = cache.cache_controller.drain_offline_intents()

    cache.apply_offline_hicache_acks(
        [{"intent_id": intent["intent_id"], "state": "ABORTED"}]
    )

    assert node.host_value is None
    assert cache.mem_pool_host is not None
    assert cache.mem_pool_host.used_pages == 0
    assert cache.offline_hicache_intent_audit() == []


def test_h2d_abort_rolls_back_provisional_device_residency() -> None:
    cache = _offline_cache()
    node = _insert_device_prefix(cache)
    cache.write_backup(node, "request-0")
    [d2h] = cache.cache_controller.drain_offline_intents()
    cache.apply_offline_hicache_acks(
        [{"intent_id": d2h["intent_id"], "state": "COMMITTED"}]
    )
    assert _force_tree_evict(cache, owner="evict") == 1
    assert cache.match_prefix([1]).device_length == 0

    loaded = cache.load_back("request-1", node)
    [h2d] = cache.cache_controller.drain_offline_intents()
    assert loaded == 1
    assert cache.match_prefix([1]).device_length == 1

    cache.apply_offline_hicache_acks(
        [{"intent_id": h2d["intent_id"], "state": "ABORTED"}]
    )

    assert cache.match_prefix([1]).device_length == 0
    assert cache.mem_pool_device.used_pages == 0
    assert cache.offline_hicache_intent_audit() == []


def test_h2d_commit_keeps_device_prefix_only_after_parent_ack_gate() -> None:
    cache = _offline_cache()
    node = _insert_device_prefix(cache)
    cache.write_backup(node, "request-0")
    [d2h] = cache.cache_controller.drain_offline_intents()
    cache.apply_offline_hicache_acks(
        [{"intent_id": d2h["intent_id"], "state": "COMMITTED"}]
    )
    assert _force_tree_evict(cache, owner="evict") == 1

    loaded = cache.load_back("request-1", node)
    [h2d] = cache.cache_controller.drain_offline_intents()
    assert loaded == 1
    assert node.lock_ref > 0
    assert cache.cache_controller.offline_intent(h2d["intent_id"]).state.value == (
        "PENDING"
    )

    cache.apply_offline_hicache_acks(
        [{"intent_id": h2d["intent_id"], "state": "COMMITTED"}]
    )

    assert cache.match_prefix([1]).device_length == 1
    assert cache.mem_pool_device.used_pages == 1
    assert node.lock_ref == 0
    assert cache.offline_hicache_intent_audit() == []


def test_gpu_clear_aborts_unacked_d2h_and_releases_host_hold() -> None:
    cache = _offline_cache()
    node = _insert_device_prefix(cache)
    cache.write_backup(node, "request-0")
    [d2h] = cache.cache_controller.drain_offline_intents()

    result = cache.clear_l1_for_gpu_failure()

    assert result["cleared_device_pages"] == 1
    assert node.host_value is None
    assert cache.mem_pool_host is not None
    assert cache.mem_pool_host.used_pages == 0
    assert cache.mem_pool_device.used_pages == 0
    assert cache.cache_controller.offline_intent(d2h["intent_id"]).state.value == (
        "ABORTED"
    )
    assert cache.offline_hicache_intent_audit() == []


def test_offline_write_back_retains_source_until_d2h_ack() -> None:
    cache = _offline_cache(write_policy="write_back")
    node = _insert_device_prefix(cache)

    assert _force_tree_evict(cache, owner="writeback") == 0
    [intent] = cache.cache_controller.drain_offline_intents()

    assert node.value is not None
    assert node.host_value is None
    assert intent["blocks_compute"] is True

    cache.apply_offline_hicache_acks(
        [{"intent_id": intent["intent_id"], "state": "COMMITTED"}]
    )
    assert _force_tree_evict(cache, owner="writeback-retry") == 1

    assert node.value is None
    assert node.host_value is not None
