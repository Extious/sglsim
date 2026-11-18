from __future__ import annotations

import copy

import pytest

from sglsim.srt.simulation.global_des import (
    GlobalDES,
    HbmAccount,
    KvCopyState,
    KvLocation,
    PhysicalGpuId,
)
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    OfflineResourceBridge,
    PhysicalGpuSpec,
)


def _topology() -> OfflinePhysicalTopology:
    # Both DP replicas have TP=4, and each TP group spans two nodes.  Local GPU
    # slots remain physically unique even though TP coordinates repeat by DP.
    gpus: list[PhysicalGpuSpec] = []
    for dp_rank in range(2):
        for tp_rank in range(4):
            node_id = "node0" if tp_rank < 2 else "node1"
            local_gpu_id = dp_rank * 2 + tp_rank % 2
            gpus.append(
                PhysicalGpuSpec(
                    gpu=PhysicalGpuId(
                        node_id=node_id,
                        local_gpu_id=local_gpu_id,
                        dp_rank=dp_rank,
                        pp_rank=0,
                        tp_rank=tp_rank,
                    ),
                    hbm_capacity_bytes=20_000,
                    weights_bytes=1_000,
                    runtime_bytes=500,
                )
            )
    return OfflinePhysicalTopology(
        gpus=tuple(gpus),
        nodes=(
            NodeDramSpec("node0", capacity_bytes=100_000, runtime_bytes=2_000),
            NodeDramSpec("node1", capacity_bytes=100_000, runtime_bytes=2_000),
        ),
    )


def _dense_topology() -> OfflinePhysicalTopology:
    """Two dense TP=4 replicas, with one complete replica per node."""

    gpus: list[PhysicalGpuSpec] = []
    for dp_rank, node_id in enumerate(("node0", "node1")):
        for tp_rank in range(4):
            gpus.append(
                PhysicalGpuSpec(
                    gpu=PhysicalGpuId(
                        node_id=node_id,
                        local_gpu_id=tp_rank,
                        dp_rank=dp_rank,
                        pp_rank=0,
                        tp_rank=tp_rank,
                    ),
                    hbm_capacity_bytes=20_000,
                    weights_bytes=1_000,
                    runtime_bytes=500,
                )
            )
    return OfflinePhysicalTopology(
        gpus=tuple(gpus),
        nodes=(
            NodeDramSpec("node0", capacity_bytes=100_000, runtime_bytes=2_000),
            NodeDramSpec("node1", capacity_bytes=100_000, runtime_bytes=2_000),
        ),
    )


def _pool(
    owner_pages: dict[str, int],
    *,
    capacity_pages: int = 16,
    page_size_tokens: int = 4,
    kv_bytes_per_token: int = 10,
) -> dict:
    used_pages = sum(owner_pages.values())
    free_pages = capacity_pages - used_pages
    bytes_per_page = page_size_tokens * kv_bytes_per_token
    return {
        "capacity_pages": capacity_pages,
        "used_pages": used_pages,
        "free_pages": free_pages,
        "capacity_tokens": capacity_pages * page_size_tokens,
        "used_tokens": used_pages * page_size_tokens,
        "free_tokens": free_pages * page_size_tokens,
        "capacity_bytes": capacity_pages * bytes_per_page,
        "used_bytes": used_pages * bytes_per_page,
        "free_bytes": free_pages * bytes_per_page,
        "owner_pages": dict(owner_pages),
    }


def _snapshot(
    dp_rank: int,
    *,
    l1_owners: dict[str, int],
    l2_owners: dict[str, int],
    running: tuple[str, ...] = (),
    waiting: tuple[str, ...] = (),
    timestamp_s: float = 0.0,
) -> dict:
    return {
        "schema": "sglsim.offline.rank-kv-resource.v1",
        "timestamp_s": timestamp_s,
        "dp_rank": dp_rank,
        "pp_rank": 0,
        "scheduler_tp_rank": 0,
        "logical_tp_group_size": 4,
        "scope": "one-symmetric-tp-rank-view",
        "page_size_tokens": 4,
        "kv_bytes_per_token_per_tp_rank": 10,
        "l1_device": _pool(l1_owners),
        "l2_host": _pool(l2_owners),
        "running_request_ids": list(running),
        "waiting_request_ids": list(waiting),
    }


def _gpu(topology: OfflinePhysicalTopology, dp: int, tp: int) -> PhysicalGpuId:
    return next(
        spec.gpu
        for spec in topology.gpus
        if spec.gpu.dp_rank == dp and spec.gpu.tp_rank == tp
    )


def test_dp2_tp4_snapshot_expansion_is_per_gpu_and_node_local() -> None:
    des = GlobalDES()
    topology = _topology()
    bridge = OfflineResourceBridge(des, topology)

    protected_gpu = _gpu(topology, 1, 3)
    des.adjust_gpu_hbm(
        protected_gpu,
        HbmAccount.AEGIS_COMMITTED,
        77,
        reason="preexisting-aegis-copy",
    )

    dp0 = _snapshot(
        0,
        l1_owners={"req0": 2, "prefill-floor-req0": 1, "cached": 1},
        l2_owners={"req0": 2},
        running=("req0",),
    )
    step0 = bridge.consume_snapshot(
        dp0,
        phase="complete",
        event_metadata={"event_id": "rank0-step0"},
    )

    # The snapshot reports 160 L1 bytes for one TP rank: 120 primary and 40
    # prefix.  Every physical shard receives that per-rank value once, never
    # the 640-byte TP-group total.
    for tp_rank in range(4):
        ledger = des.gpu_hbm[_gpu(topology, 0, tp_rank)]
        assert ledger.primary_kv == 120
        assert ledger.prefix_kv == 40
    assert step0["physical_primary_bytes_total"] == 4 * 120
    assert step0["physical_prefix_bytes_total"] == 4 * 40

    # Two TP shards reside on each node.  The per-rank L2 usage is 80 bytes,
    # so there is no process-global host pool and each node receives 160.
    assert des.node_dram["node0"].hicache_primary == 160
    assert des.node_dram["node1"].hicache_primary == 160

    dp1 = _snapshot(
        1,
        l1_owners={"req1": 1},
        l2_owners={"req1": 1},
        waiting=("req1",),
    )
    bridge.consume_snapshot(dp1, phase="prepare")
    assert des.node_dram["node0"].hicache_primary == 240
    assert des.node_dram["node1"].hicache_primary == 240
    assert des.gpu_hbm[protected_gpu].aegis_committed == 77
    assert bridge.audit() == []


def test_replacement_is_idempotent_then_primary_becomes_prefix_on_finish() -> None:
    des = GlobalDES()
    topology = _topology()
    bridge = OfflineResourceBridge(des, topology)
    active = _snapshot(
        0,
        l1_owners={"request-a": 3, "retract-last-request-a": 1},
        l2_owners={"request-a": 2},
        running=("request-a",),
    )

    bridge.consume_snapshot(active, phase="complete")
    state_once = des.snapshot()
    bridge.consume_snapshot(copy.deepcopy(active), phase="poll")
    state_twice = des.snapshot()
    assert state_twice == state_once
    assert des.node_dram["node0"].hicache_primary == 160
    assert des.node_dram["node1"].hicache_primary == 160

    finished = _snapshot(
        0,
        l1_owners={"request-a": 3, "retract-last-request-a": 1},
        l2_owners={"request-a": 2},
    )
    bridge.consume_snapshot(finished, phase="request_finished")
    for tp_rank in range(4):
        ledger = des.gpu_hbm[_gpu(topology, 0, tp_rank)]
        assert ledger.primary_kv == 0
        assert ledger.prefix_kv == 160
    assert bridge.audit() == []


def test_hard_worker_abort_releases_only_provisional_hicache_d2h_bytes() -> None:
    des = GlobalDES()
    topology = _topology()
    bridge = OfflineResourceBridge(des, topology)
    bridge.consume_snapshot(
        _snapshot(
            0,
            l1_owners={"committed-prefix": 3},
            l2_owners={"committed-prefix": 1, "pending-d2h": 2},
        ),
        phase="d2h_provisional_hold",
    )
    assert des.node_dram["node0"].hicache_primary == 240
    assert des.node_dram["node1"].hicache_primary == 240

    rollback = bridge.release_provisional_hicache_d2h(
        0,
        0,
        bytes_per_tp_rank=80,
        reason="unit_pending_d2h_abort",
        intent_ids=("pending-d2h",),
    )

    assert rollback["physical_released_bytes"] == 320
    # Two TP shards on each node retain only the one committed 40-byte page.
    assert des.node_dram["node0"].hicache_primary == 80
    assert des.node_dram["node1"].hicache_primary == 80
    assert bridge.export()["rank_steps"][-1]["phase"] == (
        "ordinary_hicache_d2h_aborted"
    )
    assert bridge.audit() == []


def test_rank_failure_zero_snapshot_only_replaces_that_rank() -> None:
    des = GlobalDES()
    topology = _topology()
    bridge = OfflineResourceBridge(des, topology)
    bridge.consume_snapshot(
        _snapshot(
            0,
            l1_owners={"req0": 2},
            l2_owners={"req0": 2},
            running=("req0",),
        ),
        phase="complete",
    )
    bridge.consume_snapshot(
        _snapshot(
            1,
            l1_owners={"req1": 3},
            l2_owners={"req1": 1},
            running=("req1",),
        ),
        phase="complete",
    )
    dp1_before = {
        tp: des.gpu_hbm[_gpu(topology, 1, tp)].snapshot() for tp in range(4)
    }

    bridge.consume_snapshot(
        _snapshot(0, l1_owners={}, l2_owners={}),
        phase="rank_failure_clear",
        event_metadata={"failure_id": "fail-dp0"},
    )

    for tp_rank in range(4):
        dp0_ledger = des.gpu_hbm[_gpu(topology, 0, tp_rank)]
        assert dp0_ledger.primary_kv == 0
        assert dp0_ledger.prefix_kv == 0
        assert des.gpu_hbm[_gpu(topology, 1, tp_rank)].snapshot() == dp1_before[tp_rank]
    # Only the two dp1 shards on each node remain: 2 * 40 bytes.
    assert des.node_dram["node0"].hicache_primary == 80
    assert des.node_dram["node1"].hicache_primary == 80
    assert bridge.audit() == []


def test_dp_rank_failure_and_repair_preserve_l2_and_lose_volatile_aegis() -> None:
    des = GlobalDES()
    topology = _topology()
    bridge = OfflineResourceBridge(des, topology)
    dp0_snapshot = _snapshot(
        0,
        l1_owners={"req0": 2},
        l2_owners={"req0": 2},
        running=("req0",),
    )
    dp1_snapshot = _snapshot(
        1,
        l1_owners={"req1": 3},
        l2_owners={"req1": 1},
        running=("req1",),
    )
    bridge.consume_snapshot(dp0_snapshot, phase="complete")
    bridge.consume_snapshot(dp1_snapshot, phase="complete")

    pending_gpu = _gpu(topology, 0, 0)
    committed_gpu = _gpu(topology, 0, 1)
    pending_location = KvLocation.gpu_aegis(pending_gpu)
    committed_location = KvLocation.gpu_aegis(committed_gpu)
    des.register_extent(extent_id="pending", request_id="req0", size_bytes=25)
    des.place_extent(
        "pending",
        pending_location,
        state=KvCopyState.PENDING,
    )
    des.register_extent(extent_id="committed", request_id="req0", size_bytes=35)
    des.place_extent("committed", committed_location)

    dram_before = {
        node_id: ledger.snapshot() for node_id, ledger in des.node_dram.items()
    }
    dp1_before = {
        tp: des.gpu_hbm[_gpu(topology, 1, tp)].snapshot() for tp in range(4)
    }
    failure = bridge.fail_dp_rank(
        0,
        reason="unit-gpu-group-failure",
        event_metadata={"failure_id": "f0"},
    )

    assert failure["status"] == "failed"
    assert failure["affected_extents"] == ["committed", "pending"]
    assert bridge.failed_dp_ranks == (0,)
    for tp_rank in range(4):
        ledger = des.gpu_hbm[_gpu(topology, 0, tp_rank)]
        assert not ledger.healthy
        assert ledger.used_bytes == 0
        assert ledger.aegis_pending == 0
        assert ledger.aegis_committed == 0
        assert des.gpu_hbm[_gpu(topology, 1, tp_rank)].snapshot() == dp1_before[tp_rank]
    assert {
        node_id: ledger.snapshot() for node_id, ledger in des.node_dram.items()
    } == dram_before
    assert des.extents["pending"].locations[pending_location] is KvCopyState.ABORTED
    assert (
        des.extents["committed"].locations[committed_location]
        is KvCopyState.LOST
    )
    assert bridge.audit() == []

    rank_steps_after_first_failure = len(bridge.rank_steps)
    repeated_failure = bridge.fail_dp_rank(0, reason="duplicate")
    assert repeated_failure["status"] == "already_failed"
    assert len(bridge.rank_steps) == rank_steps_after_first_failure

    with pytest.raises(ValueError, match="nonzero L1 snapshot"):
        bridge.consume_snapshot(dp0_snapshot, phase="poll_while_failed")
    with pytest.raises(ValueError, match="cannot change durable L2"):
        bridge.consume_snapshot(
            _snapshot(0, l1_owners={}, l2_owners={}),
            phase="bad_clear_l2_while_failed",
        )
    bridge.consume_snapshot(
        _snapshot(0, l1_owners={}, l2_owners={"req0": 2}),
        phase="clear_l1_preserve_l2",
    )
    assert {
        node_id: ledger.snapshot() for node_id, ledger in des.node_dram.items()
    } == dram_before

    with pytest.raises(ValueError, match="never-failed"):
        bridge.repair_dp_rank(1)
    repaired = bridge.repair_dp_rank(
        0,
        reason="unit-repair",
        event_metadata={"repair_id": "r0"},
    )
    assert repaired["status"] == "repaired"
    assert bridge.failed_dp_ranks == ()
    for spec in topology.gpus:
        if spec.gpu.dp_rank != 0:
            continue
        ledger = des.gpu_hbm[spec.gpu]
        assert ledger.healthy
        assert ledger.weights == spec.weights_bytes
        assert ledger.runtime == spec.runtime_bytes
        assert ledger.primary_kv == 0
        assert ledger.prefix_kv == 0
        assert ledger.aegis_pending == 0
        assert ledger.aegis_committed == 0
    assert des.extents["pending"].locations[pending_location] is KvCopyState.ABORTED
    assert (
        des.extents["committed"].locations[committed_location]
        is KvCopyState.LOST
    )

    repeated_repair = bridge.repair_dp_rank(0, reason="duplicate")
    assert repeated_repair["status"] == "already_repaired"
    bridge.consume_snapshot(dp0_snapshot, phase="post_repair_new_allocation")
    assert des.gpu_hbm[_gpu(topology, 0, 0)].primary_kv == 80
    exported = bridge.export()
    assert [event["status"] for event in exported["lifecycle_events"]] == [
        "failed",
        "already_failed",
        "repaired",
        "already_repaired",
    ]
    assert bridge.audit() == []


def test_dense_node_fail_stop_and_repair_lose_only_local_volatile_state() -> None:
    des = GlobalDES()
    topology = _dense_topology()
    bridge = OfflineResourceBridge(des, topology)
    dp0_snapshot = _snapshot(
        0,
        l1_owners={"req0": 2, "cached0": 1},
        l2_owners={"req0": 2},
        running=("req0",),
    )
    dp1_snapshot = _snapshot(
        1,
        l1_owners={"req1": 3},
        l2_owners={"req1": 1},
        running=("req1",),
    )
    bridge.consume_snapshot(dp0_snapshot, phase="complete")
    bridge.consume_snapshot(dp1_snapshot, phase="complete")

    local_gpu = _gpu(topology, 0, 0)
    local_pending = KvLocation.gpu_aegis(local_gpu)
    local_dram_committed = KvLocation.node_dram("node0")
    remote_gpu = _gpu(topology, 1, 0)
    remote_committed = KvLocation.gpu_aegis(remote_gpu)
    for extent_id, request_id, size in (
        ("local-gpu-pending", "req0", 25),
        ("local-dram-committed", "req0", 35),
        ("remote-gpu-committed", "req1", 45),
    ):
        des.register_extent(
            extent_id=extent_id,
            request_id=request_id,
            size_bytes=size,
        )
    des.place_extent(
        "local-gpu-pending",
        local_pending,
        state=KvCopyState.PENDING,
    )
    des.place_extent("local-dram-committed", local_dram_committed)
    des.place_extent("remote-gpu-committed", remote_committed)

    remote_gpu_before = {
        spec.gpu: des.gpu_hbm[spec.gpu].snapshot()
        for spec in topology.gpus
        if spec.gpu.node_id == "node1"
    }
    remote_dram_before = des.node_dram["node1"].snapshot()
    failure = bridge.fail_node(
        "node0",
        reason="unit-node-failure",
        event_metadata={"failure_id": "nf0"},
    )

    assert failure["status"] == "failed"
    assert failure["node_id"] == "node0"
    assert failure["affected_dp_ranks"] == [0]
    assert failure["affected_extents"] == [
        "local-dram-committed",
        "local-gpu-pending",
    ]
    assert bridge.failed_nodes == ("node0",)
    assert bridge.node_affected_dp_ranks == {"node0": (0,)}
    for spec in topology.gpus:
        ledger = des.gpu_hbm[spec.gpu]
        if spec.gpu.node_id == "node0":
            assert not ledger.healthy
            assert ledger.used_bytes == 0
        else:
            assert ledger.snapshot() == remote_gpu_before[spec.gpu]
    assert not des.node_dram["node0"].healthy
    assert des.node_dram["node0"].used_bytes == 0
    assert des.node_dram["node1"].snapshot() == remote_dram_before
    assert (
        des.extents["local-gpu-pending"].locations[local_pending]
        is KvCopyState.ABORTED
    )
    assert (
        des.extents["local-dram-committed"].locations[local_dram_committed]
        is KvCopyState.LOST
    )
    assert (
        des.extents["remote-gpu-committed"].locations[remote_committed]
        is KvCopyState.COMMITTED
    )
    latest = bridge.export()["latest_rank_views"]
    assert latest["dp0/pp0"]["primary_bytes_per_tp_rank"] == 0
    assert latest["dp0/pp0"]["prefix_bytes_per_tp_rank"] == 0
    assert latest["dp0/pp0"]["l2_used_by_node"] == {"node0": 0}
    assert latest["dp1/pp0"]["primary_bytes_per_tp_rank"] == 120
    assert latest["dp1/pp0"]["l2_used_by_node"] == {"node1": 160}
    assert bridge.audit() == []

    first_failure_steps = len(bridge.rank_steps)
    assert bridge.fail_node("node0", reason="duplicate")["status"] == "already_failed"
    assert len(bridge.rank_steps) == first_failure_steps
    with pytest.raises(ValueError, match="failed node"):
        bridge.consume_snapshot(dp0_snapshot, phase="allocation_while_failed")
    with pytest.raises(ValueError, match="unhealthy GPU"):
        des.adjust_gpu_hbm(
            local_gpu,
            HbmAccount.PRIMARY_KV,
            1,
            reason="forbidden-while-failed",
        )
    with pytest.raises(ValueError, match="unhealthy node"):
        des.adjust_node_dram(
            "node0",
            "hicache_primary",
            1,
            reason="forbidden-while-failed",
        )
    with pytest.raises(ValueError, match="never-failed"):
        bridge.repair_node("node1")
    with pytest.raises(ValueError, match="unknown physical node"):
        bridge.repair_node("ghost")

    repair = bridge.repair_node(
        "node0",
        reason="unit-node-repair",
        event_metadata={"repair_id": "nr0"},
    )
    assert repair["status"] == "repaired"
    assert bridge.failed_nodes == ()
    assert bridge.node_affected_dp_ranks == {}
    for spec in topology.gpus:
        if spec.gpu.node_id != "node0":
            continue
        ledger = des.gpu_hbm[spec.gpu]
        assert ledger.healthy
        assert ledger.weights == spec.weights_bytes
        assert ledger.runtime == spec.runtime_bytes
        assert ledger.primary_kv == 0
        assert ledger.prefix_kv == 0
        assert ledger.aegis_pending == 0
        assert ledger.aegis_committed == 0
    repaired_dram = des.node_dram["node0"]
    assert repaired_dram.healthy
    assert repaired_dram.runtime == 2_000
    assert repaired_dram.hicache_primary == 0
    assert repaired_dram.aegis_pending == 0
    assert repaired_dram.aegis_committed == 0
    assert (
        des.extents["local-gpu-pending"].locations[local_pending]
        is KvCopyState.ABORTED
    )
    assert (
        des.extents["local-dram-committed"].locations[local_dram_committed]
        is KvCopyState.LOST
    )

    repeated_repair = bridge.repair_node("node0", reason="duplicate")
    assert repeated_repair["status"] == "already_repaired"
    exported = bridge.export()
    assert exported["failed_nodes"] == []
    assert exported["affected_dp_ranks_by_failed_node"] == {}
    assert [
        event["status"] for event in exported["node_lifecycle_events"]
    ] == ["failed", "already_failed", "repaired", "already_repaired"]
    assert bridge.audit() == []

    # A later Scheduler snapshot is a fresh allocation, not KV resurrection.
    bridge.consume_snapshot(dp0_snapshot, phase="post_repair_new_allocation")
    assert des.gpu_hbm[local_gpu].primary_kv == 80
    assert des.node_dram["node0"].hicache_primary == 320
    assert bridge.audit() == []


def test_node_failure_rejects_cross_node_tp_group_before_mutation() -> None:
    des = GlobalDES()
    topology = _topology()
    bridge = OfflineResourceBridge(des, topology)
    bridge.consume_snapshot(
        _snapshot(
            0,
            l1_owners={"req0": 1},
            l2_owners={"req0": 1},
            running=("req0",),
        ),
        phase="complete",
    )
    state_before = des.snapshot()
    steps_before = bridge.rank_steps

    with pytest.raises(ValueError, match="cross-node TP group"):
        bridge.fail_node("node0")

    assert des.snapshot() == state_before
    assert bridge.rank_steps == steps_before
    assert bridge.failed_nodes == ()
    assert bridge.export()["node_lifecycle_events"] == []
    assert bridge.audit() == []


def test_process_failure_targets_one_pp_process_and_complete_tp_group() -> None:
    des = GlobalDES()
    topology = OfflinePhysicalTopology(
        gpus=tuple(
            PhysicalGpuSpec(
                PhysicalGpuId("node0", pp * 2 + tp, 0, pp, tp),
                hbm_capacity_bytes=20_000,
                weights_bytes=1_000,
                runtime_bytes=500,
            )
            for pp in range(2)
            for tp in range(2)
        ),
        nodes=(NodeDramSpec("node0", 100_000, runtime_bytes=2_000),),
    )
    bridge = OfflineResourceBridge(des, topology)

    def snapshot(pp_rank: int) -> dict:
        row = _snapshot(
            0,
            l1_owners={f"req{pp_rank}": 2},
            l2_owners={f"req{pp_rank}": 1},
            running=(f"req{pp_rank}",),
        )
        row["pp_rank"] = pp_rank
        row["logical_tp_group_size"] = 2
        return row

    bridge.consume_snapshot(snapshot(0), phase="complete")
    bridge.consume_snapshot(snapshot(1), phase="complete")
    pp1_before = {
        spec.gpu: des.gpu_hbm[spec.gpu].snapshot()
        for spec in topology.group(0, 1)
    }
    dram_before = des.node_dram["node0"].snapshot()

    failure = bridge.fail_process_group(
        0,
        0,
        event_metadata={"failure_id": "process-0"},
    )
    assert failure["failure_domain"] == "process"
    assert failure["scheduler_process_id"] == "dp0/pp0"
    assert bridge.failed_process_groups == ((0, 0),)
    for spec in topology.group(0, 0):
        assert not des.gpu_hbm[spec.gpu].healthy
        assert des.gpu_hbm[spec.gpu].used_bytes == 0
    for spec in topology.group(0, 1):
        assert des.gpu_hbm[spec.gpu].snapshot() == pp1_before[spec.gpu]
    assert des.node_dram["node0"].snapshot() == dram_before
    assert bridge.audit() == []

    with pytest.raises(ValueError, match="unavailable TP group"):
        bridge.consume_snapshot(snapshot(0), phase="failed-process-poll")
    repair = bridge.repair_process_group(0, 0)
    assert repair["status"] == "repaired"
    assert bridge.failed_process_groups == ()
    for spec in topology.group(0, 0):
        ledger = des.gpu_hbm[spec.gpu]
        assert ledger.healthy
        assert ledger.weights == spec.weights_bytes
        assert ledger.runtime == spec.runtime_bytes
        assert ledger.primary_kv == 0
        assert ledger.prefix_kv == 0
        assert ledger.aegis_pending == 0
        assert ledger.aegis_committed == 0
    assert bridge.audit() == []


def test_single_gpu_failure_clears_only_that_gpu_and_invalidates_group_l1() -> None:
    des = GlobalDES()
    topology = _dense_topology()
    bridge = OfflineResourceBridge(des, topology)
    bridge.consume_snapshot(
        _snapshot(
            0,
            l1_owners={"active": 2, "cached": 1},
            l2_owners={"active": 2},
            running=("active",),
        ),
        phase="complete",
    )
    failed_gpu = _gpu(topology, 0, 1)
    survivor_gpu = _gpu(topology, 0, 2)
    other_dp_gpu = _gpu(topology, 1, 0)
    failed_location = KvLocation.gpu_aegis(failed_gpu)
    survivor_location = KvLocation.gpu_aegis(survivor_gpu)
    des.register_extent(extent_id="failed-shard", request_id="r", size_bytes=25)
    des.place_extent("failed-shard", failed_location)
    des.register_extent(extent_id="survivor-shard", request_id="r", size_bytes=35)
    des.place_extent("survivor-shard", survivor_location)
    other_before = des.gpu_hbm[other_dp_gpu].snapshot()
    dram_before = {
        node_id: ledger.snapshot() for node_id, ledger in des.node_dram.items()
    }

    failure = bridge.fail_gpu(
        failed_gpu,
        event_metadata={"failure_id": "gpu-exact"},
    )

    assert failure["physical_gpu_id"] == failed_gpu.key
    assert failure["physical_gpus"] == [failed_gpu.as_dict()]
    assert failure["recovery_source_forbidden"] is True
    assert bridge.failed_gpus == (failed_gpu,)
    failed_ledger = des.gpu_hbm[failed_gpu]
    assert not failed_ledger.healthy
    assert failed_ledger.used_bytes == 0
    survivor_ledger = des.gpu_hbm[survivor_gpu]
    assert survivor_ledger.healthy
    assert survivor_ledger.weights == 1_000
    assert survivor_ledger.runtime == 500
    assert survivor_ledger.primary_kv == 0
    assert survivor_ledger.prefix_kv == 0
    assert survivor_ledger.aegis_committed == 35
    assert des.gpu_hbm[other_dp_gpu].snapshot() == other_before
    assert {
        node_id: ledger.snapshot() for node_id, ledger in des.node_dram.items()
    } == dram_before
    assert des.extents["failed-shard"].locations[failed_location] is KvCopyState.LOST
    assert (
        des.extents["survivor-shard"].locations[survivor_location]
        is KvCopyState.COMMITTED
    )
    assert bridge.export()["failed_physical_gpu_ids"] == [failed_gpu.key]
    assert bridge.audit() == []

    repair = bridge.repair_gpu(failed_gpu)
    assert repair["repair_restored_empty_volatile_state"] is True
    assert bridge.failed_gpus == ()
    repaired = des.gpu_hbm[failed_gpu]
    assert repaired.healthy
    assert repaired.weights == 1_000
    assert repaired.runtime == 500
    assert repaired.primary_kv == 0
    assert repaired.prefix_kv == 0
    assert repaired.aegis_committed == 0
    assert des.gpu_hbm[survivor_gpu].aegis_committed == 35
    assert des.extents["failed-shard"].locations[failed_location] is KvCopyState.LOST
    assert bridge.audit() == []


def test_narrow_failure_overlap_is_rejected_before_mutation() -> None:
    des = GlobalDES()
    topology = _dense_topology()
    bridge = OfflineResourceBridge(des, topology)
    target = _gpu(topology, 0, 0)
    bridge.fail_gpu(target)
    state = des.snapshot()

    with pytest.raises(ValueError, match="narrower failure"):
        bridge.fail_dp_rank(0)
    with pytest.raises(ValueError, match="failed GPUs"):
        bridge.fail_process_group(0, 0)
    with pytest.raises(ValueError, match="narrower failure"):
        bridge.fail_node("node0")
    assert des.snapshot() == state


def test_validation_is_fail_closed_and_export_is_defensive() -> None:
    des = GlobalDES()
    topology = _topology()
    bridge = OfflineResourceBridge(des, topology)
    valid = _snapshot(
        0,
        l1_owners={"r": 1},
        l2_owners={},
        running=("r",),
    )

    wrong_tp = copy.deepcopy(valid)
    wrong_tp["logical_tp_group_size"] = 8
    with pytest.raises(ValueError, match="TP size"):
        bridge.consume_snapshot(wrong_tp, phase="bad")
    wrong_scope = copy.deepcopy(valid)
    wrong_scope["scope"] = "whole-tp-group"
    with pytest.raises(ValueError, match="one-symmetric"):
        bridge.consume_snapshot(wrong_scope, phase="bad")
    broken_bytes = copy.deepcopy(valid)
    broken_bytes["l1_device"]["used_bytes"] += 1
    with pytest.raises(ValueError, match="used_bytes is inconsistent"):
        bridge.consume_snapshot(broken_bytes, phase="bad")

    bridge.consume_snapshot(valid, phase="complete")
    exported = bridge.export()
    exported["rank_steps"][0]["primary_bytes_per_tp_rank"] = 999_999
    exported["physical_resources"]["gpus"].clear()
    assert bridge.export()["rank_steps"][0]["primary_bytes_per_tp_rank"] == 40
    assert bridge.export()["physical_resources"]["gpus"]
    assert bridge.audit() == []


def test_topology_rejects_duplicate_physical_slot() -> None:
    first = PhysicalGpuSpec(
        PhysicalGpuId("n", 0, 0, 0, 0),
        hbm_capacity_bytes=100,
    )
    second = PhysicalGpuSpec(
        PhysicalGpuId("n", 0, 1, 0, 0),
        hbm_capacity_bytes=100,
    )
    with pytest.raises(ValueError, match="same physical GPU slot"):
        OfflinePhysicalTopology(
            gpus=(first, second),
            nodes=(NodeDramSpec("n", capacity_bytes=100),),
        )
