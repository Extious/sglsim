from __future__ import annotations

import pytest

from sglsim.srt.simulation.global_des import (
    CommitGroupState,
    EventPriority,
    GlobalDES,
    HbmAccount,
    KvCopyState,
    KvLocation,
    PhysicalGpuId,
    StorageTier,
    TransferDestinationMode,
    TransferState,
)
from sglsim.srt.simulation.offline_aegis_transfer_bridge import (
    CopyFabricBandwidths,
    CopyFabricConfig,
    CopyFabricSetupLatencies,
    CopyPathKind,
    OfflineAegisTransferBridge,
)
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    PhysicalGpuSpec,
)


TOKENS = 8
BYTES_PER_TOKEN = 100


def _topology(
    *,
    target_hbm_capacity: int = 100_000,
    include_replacement: bool = False,
) -> OfflinePhysicalTopology:
    gpus = []
    replicas = [(0, "source"), (1, "peer")]
    if include_replacement:
        replicas.append((2, "replacement"))
    for dp_rank, node_id in replicas:
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
                    hbm_capacity_bytes=(
                        100_000 if dp_rank == 0 else target_hbm_capacity
                    ),
                )
            )
    return OfflinePhysicalTopology(
        gpus=tuple(gpus),
        nodes=(
            NodeDramSpec("source", 1_000_000),
            NodeDramSpec("peer", 1_000_000),
            *(
                (NodeDramSpec("replacement", 1_000_000),)
                if include_replacement
                else ()
            ),
            NodeDramSpec("store", 1_000_000),
        ),
    )


def _des(topology: OfflinePhysicalTopology) -> GlobalDES:
    des = GlobalDES()
    for node in topology.nodes:
        des.register_node_dram(
            node.node_id,
            capacity_bytes=node.capacity_bytes,
            runtime_bytes=node.runtime_bytes,
        )
    for spec in topology.gpus:
        des.register_gpu(
            spec.gpu,
            capacity_bytes=spec.hbm_capacity_bytes,
            weights_bytes=spec.weights_bytes,
            runtime_bytes=spec.runtime_bytes,
        )
        if spec.gpu.dp_rank == 0:
            des.adjust_gpu_hbm(
                spec.gpu,
                HbmAccount.PRIMARY_KV,
                TOKENS * BYTES_PER_TOKEN,
                reason="test_scheduler_primary",
            )
    return des


def _fabric(*, fabric_bw: int = 1_000_000_000) -> CopyFabricConfig:
    return CopyFabricConfig(
        bandwidths=CopyFabricBandwidths(
            gpu_copy_bytes_per_s=100_000_000_000,
            nvlink_bytes_per_s=100_000_000_000,
            pcie_bytes_per_s=100_000_000_000,
            nic_bytes_per_s=fabric_bw,
            fabric_bytes_per_s=fabric_bw,
            dram_write_bytes_per_s=100_000_000_000,
        ),
        setup=CopyFabricSetupLatencies(
            same_node_d2d_ns=10,
            cross_node_d2d_ns=1_000,
            same_node_d2h_ns=10,
            cross_node_d2h_ns=1_000,
        ),
    )


def _event(*, end: int = TOKENS, version: int = 1) -> dict[str, object]:
    return {
        "type": "aegis_source_chunk_committed",
        "rid": "request-1",
        "request_backup_key": "backup-1",
        "source_dp_rank": 0,
        "target_dp_rank": 1,
        "source_pp_rank": 0,
        "target_pp_rank": 0,
        "required_pp_size": 1,
        "version": version,
        "route_version": version,
        "source_committed_token_end": end,
        "reserved_token_count": TOKENS,
        "execution_id": f"execution-v{version}",
        "block_generation": version - 1,
    }


def _hint(
    *,
    hbm: int = TOKENS,
    dram: int = TOKENS,
    host: str = "source",
    version: int = 1,
) -> dict[str, object]:
    return {
        "backup_key": "backup-1",
        "target_dp_rank": 1,
        "version": version,
        "route_version": version,
        "reserved_l1_tokens": hbm,
        "reserved_l2_tokens": dram,
        "host_pool_node_id": host,
    }


def _consume(
    bridge: OfflineAegisTransferBridge,
    *,
    event: dict[str, object] | None = None,
    hint: dict[str, object] | None = None,
    layout: str = "dp2-pp1-tp4",
    callback=None,
):
    return bridge.consume_source_chunk_committed(
        event or _event(),
        hint or _hint(),
        kv_bytes_per_token_by_tp_rank={rank: BYTES_PER_TOKEN for rank in range(4)},
        model_revision="model-revision-1",
        kv_abi="bf16-kv-v1",
        parallel_layout=layout,
        on_tier_notification=callback,
    )


def _tier_plan(plan, tier: StorageTier):
    return next(group for group in plan.tier_groups if group.tier is tier)


def test_tp4_group_commits_only_on_fourth_ack() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    plan = _consume(bridge, hint=_hint(dram=0, host=""))
    hbm = _tier_plan(plan, StorageTier.GPU_AEGIS)
    jobs = sorted(
        (job for job in bridge.jobs if job.group_id == hbm.group_id),
        key=lambda job: job.end_ns,
    )

    assert len(jobs) == 4
    des.run(until_ns=jobs[-2].end_ns)
    group = des.commit_groups[hbm.group_id]
    assert group.state is CommitGroupState.PENDING
    assert len(group.acked_shards) == 3
    assert all(
        des.extents[job.extent_id].locations[job.destination]
        is KvCopyState.PENDING
        for job in jobs
    )

    des.run(until_ns=jobs[-1].end_ns)
    assert group.state is CommitGroupState.COMMITTED
    assert len(group.acked_shards) == 4
    assert all(
        des.extents[job.extent_id].locations[job.destination]
        is KvCopyState.COMMITTED
        for job in jobs
    )
    assert bridge.tier_notifications[-1].success


def test_hbm_and_dram_finish_and_notify_independently() -> None:
    topology = _topology()
    des = _des(topology)
    observed = []
    bridge = OfflineAegisTransferBridge(
        des, topology, fabric=_fabric(fabric_bw=100_000_000)
    )
    plan = _consume(bridge, callback=observed.append)
    hbm = _tier_plan(plan, StorageTier.GPU_AEGIS)
    dram = _tier_plan(plan, StorageTier.NODE_DRAM)

    assert dram.notify_at_ns < hbm.notify_at_ns
    des.run(until_ns=dram.notify_at_ns)
    assert [item.tier for item in observed] == [StorageTier.NODE_DRAM]
    assert observed[0].success
    assert des.commit_groups[dram.group_id].state is CommitGroupState.COMMITTED
    assert des.commit_groups[hbm.group_id].state is CommitGroupState.PENDING

    des.run()
    assert [item.tier for item in observed] == [
        StorageTier.NODE_DRAM,
        StorageTier.GPU_AEGIS,
    ]
    assert all(item.success for item in observed)


def test_one_source_alias_serves_both_tiers_without_double_accounting() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    primary_before = {
        spec.gpu: des.gpu_hbm[spec.gpu].primary_kv
        for spec in topology.gpus
        if spec.gpu.dp_rank == 0
    }
    plan = _consume(bridge)

    assert len(plan.extent_ids) == 4
    for extent_id in plan.extent_ids:
        extent = des.extents[extent_id]
        sources = [
            location
            for location in extent.locations
            if location.tier is StorageTier.GPU_PRIMARY
        ]
        assert len(sources) == 1
        assert extent.external_account_keys[sources[0]].startswith(
            "scheduler-primary-range-"
        )
    assert {
        gpu: des.gpu_hbm[gpu].primary_kv for gpu in primary_before
    } == primary_before

    duplicate = _consume(bridge)
    assert duplicate.duplicate
    assert len(des.extents) == 4
    assert len(bridge.jobs) == 8
    des.run()
    assert bridge.audit_invariants() == []


def test_cross_node_fabric_calendar_queues_tp_shards() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    plan = _consume(bridge, hint=_hint(dram=0, host=""))
    hbm = _tier_plan(plan, StorageTier.GPU_AEGIS)
    jobs = [job for job in bridge.jobs if job.group_id == hbm.group_id]

    assert all(job.path_kind is CopyPathKind.CROSS_NODE_D2D_FABRIC for job in jobs)
    assert all(
        any(resource.kind == "fabric" for resource in job.resources) for job in jobs
    )
    for previous, current in zip(jobs, jobs[1:]):
        assert current.start_ns >= previous.end_ns
    assert all(job.queue_delay_ns > 0 for job in jobs[1:])


def test_hbm_capacity_rejection_does_not_block_dram_commit() -> None:
    topology = _topology(target_hbm_capacity=TOKENS * BYTES_PER_TOKEN - 1)
    des = _des(topology)
    observed = []
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    plan = _consume(bridge, callback=observed.append)
    hbm = _tier_plan(plan, StorageTier.GPU_AEGIS)
    dram = _tier_plan(plan, StorageTier.NODE_DRAM)

    assert not hbm.admitted
    assert "hbm_capacity" in hbm.admission_error
    assert dram.admitted
    assert des.commit_groups[hbm.group_id].state is CommitGroupState.ABORTED
    des.run()
    by_tier = {item.tier: item for item in observed}
    assert not by_tier[StorageTier.GPU_AEGIS].success
    assert by_tier[StorageTier.NODE_DRAM].success
    assert des.commit_groups[dram.group_id].state is CommitGroupState.COMMITTED
    assert des.node_dram["source"].aegis_committed == (
        4 * TOKENS * BYTES_PER_TOKEN
    )


def test_pending_source_failure_aborts_whole_hbm_group_but_keeps_dram() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(
        des, topology, fabric=_fabric(fabric_bw=10_000_000)
    )
    plan = _consume(bridge)
    hbm = _tier_plan(plan, StorageTier.GPU_AEGIS)
    dram = _tier_plan(plan, StorageTier.NODE_DRAM)

    des.run(until_ns=dram.notify_at_ns)
    assert des.commit_groups[dram.group_id].state is CommitGroupState.COMMITTED
    assert des.commit_groups[hbm.group_id].state is CommitGroupState.PENDING
    last_source_gpu = topology.group(0, 0)[-1].gpu
    des.schedule_at(
        des.now_ns + 1,
        kind="test_source_gpu_failure",
        priority=EventPriority.FAILURE,
        callback=lambda simulator, _event: simulator.fail_gpu(last_source_gpu),
    )
    des.run()

    assert des.commit_groups[hbm.group_id].state is CommitGroupState.ABORTED
    assert des.commit_groups[dram.group_id].state is CommitGroupState.COMMITTED
    assert des.node_dram["source"].aegis_committed == (
        4 * TOKENS * BYTES_PER_TOKEN
    )
    hbm_jobs = [job for job in bridge.jobs if job.group_id == hbm.group_id]
    assert all(
        des.transfers[job.transfer_id].state.value == "ABORTED"
        for job in hbm_jobs
    )
    assert not next(
        item
        for item in bridge.tier_notifications
        if item.group_id == hbm.group_id
    ).success


def test_no_backup_hbm_to_dram_edge_and_exports_are_defensive() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    _consume(bridge)
    assert all(job.source.tier is StorageTier.GPU_PRIMARY for job in bridge.jobs)
    assert all(
        not (
            transfer.source.tier is StorageTier.GPU_AEGIS
            and transfer.destination.tier is StorageTier.NODE_DRAM
        )
        for transfer in des.transfers.values()
    )
    first = bridge.audit
    first[0]["kind"] = "mutated"
    assert bridge.audit[0]["kind"] != "mutated"


def test_duplicate_regression_and_layout_mismatch_are_fail_closed() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    _consume(bridge, event=_event(end=4))
    assert _consume(bridge, event=_event(end=4)).duplicate

    with pytest.raises(ValueError, match="frontier regressed"):
        _consume(bridge, event=_event(end=3))
    with pytest.raises(ValueError, match="parallel_layout changed"):
        _consume(bridge, event=_event(end=4), layout="wrong-layout")

    topology2 = _topology()
    des2 = _des(topology2)
    bridge2 = OfflineAegisTransferBridge(des2, topology2, fabric=_fabric())
    _consume(
        bridge2,
        event=_event(end=4, version=2),
        hint=_hint(version=2),
    )
    with pytest.raises(ValueError, match="version regressed"):
        _consume(bridge2, event=_event(end=4), hint=_hint())


def test_physical_groups_follow_prefix_and_decode_chunk_boundaries() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    event = _event()
    event.update({"prefix_chunk_tokens": 4, "decode_chunk_tokens": 2})

    plan = _consume(bridge, event=event)

    expected_ranges = {(0, 4), (4, 6), (6, 8)}
    assert {
        (group.token_start, group.token_end) for group in plan.tier_groups
    } == expected_ranges
    assert len(plan.tier_groups) == len(expected_ranges) * 2
    assert {
        (job.token_start, job.token_end) for job in bridge.jobs
    } == expected_ranges
    assert len(bridge.jobs) == len(expected_ranges) * 2 * 4
    assert bridge.audit_invariants() == []


def test_tier_local_release_preserves_independent_tier_and_newer_version() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    event = _event()
    event.update({"prefix_chunk_tokens": 4, "decode_chunk_tokens": 2})
    _consume(bridge, event=event)
    des.run()
    target_gpus = [spec.gpu for spec in topology.gpus if spec.gpu.dp_rank == 1]
    hbm_before = sum(des.gpu_hbm[gpu].aegis_committed for gpu in target_gpus)
    dram_before = des.node_dram["source"].aegis_committed
    assert hbm_before == dram_before == 4 * TOKENS * BYTES_PER_TOKEN

    released_hbm = bridge.release_owned_copies(
        request_backup_key="backup-1",
        version=1,
        tier=StorageTier.GPU_AEGIS,
        reason="unit_hbm_eviction",
    )

    assert len(released_hbm) == 3
    assert sum(des.gpu_hbm[gpu].aegis_committed for gpu in target_gpus) == 0
    assert des.node_dram["source"].aegis_committed == dram_before
    assert all(row["tier"] == StorageTier.GPU_AEGIS.value for row in released_hbm)
    assert any(
        location.tier is StorageTier.NODE_DRAM
        and state is KvCopyState.COMMITTED
        for extent in des.extents.values()
        for location, state in extent.locations.items()
    )
    released_v1_dram = bridge.release_owned_copies(
        request_backup_key="backup-1",
        version=1,
        tier=StorageTier.NODE_DRAM,
        reason="version1_completed",
    )
    assert len(released_v1_dram) == 3
    assert des.node_dram["source"].aegis_committed == 0

    # A stale terminal callback must not touch a newer physical lineage.  Use
    # a fresh scheduler-owned source ledger because old source aliases are
    # released by OfflineCluster (the bridge owns destinations only).
    topology2 = _topology()
    des2 = _des(topology2)
    bridge2 = OfflineAegisTransferBridge(des2, topology2, fabric=_fabric())
    version2_event = _event(version=2)
    version2_event.update({"prefix_chunk_tokens": 4, "decode_chunk_tokens": 2})
    _consume(bridge2, event=version2_event, hint=_hint(version=2))
    des2.run()
    target_gpus2 = [
        spec.gpu for spec in topology2.gpus if spec.gpu.dp_rank == 1
    ]
    version2_hbm = sum(
        des2.gpu_hbm[gpu].aegis_committed for gpu in target_gpus2
    )
    assert version2_hbm == 4 * TOKENS * BYTES_PER_TOKEN
    assert bridge2.release_owned_copies(
        request_backup_key="backup-1",
        version=1,
        reason="stale_completion",
    ) == ()
    assert sum(
        des2.gpu_hbm[gpu].aegis_committed for gpu in target_gpus2
    ) == version2_hbm

    released_v2 = bridge2.release_owned_copies(
        request_backup_key="backup-1",
        version=2,
        reason="request_completed",
    )
    assert len(released_v2) == 6
    assert sum(des2.gpu_hbm[gpu].aegis_committed for gpu in target_gpus2) == 0
    assert des2.node_dram["source"].aegis_committed == 0
    assert bridge.audit_invariants() == []
    assert bridge2.audit_invariants() == []


def test_pending_tier_release_aborts_only_its_own_commit_groups() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(
        des, topology, fabric=_fabric(fabric_bw=10_000_000)
    )
    plan = _consume(bridge)
    hbm = _tier_plan(plan, StorageTier.GPU_AEGIS)
    dram = _tier_plan(plan, StorageTier.NODE_DRAM)

    released = bridge.release_owned_copies(
        request_backup_key="backup-1",
        version=1,
        tier=StorageTier.GPU_AEGIS,
        reason="pending_hbm_eviction",
    )
    des.run()

    assert len(released) == 1
    assert des.commit_groups[hbm.group_id].state is CommitGroupState.ABORTED
    assert des.commit_groups[dram.group_id].state is CommitGroupState.COMMITTED
    assert des.node_dram["source"].aegis_committed == (
        4 * TOKENS * BYTES_PER_TOKEN
    )
    assert all(
        des.gpu_hbm[spec.gpu].aegis_pending == 0
        and des.gpu_hbm[spec.gpu].aegis_committed == 0
        for spec in topology.gpus
        if spec.gpu.dp_rank == 1
    )
    assert bridge.audit_invariants() == []


def test_target_first_rotation_reprotects_hbm_without_replaying_dram() -> None:
    topology = _topology(include_replacement=True)
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    first_event = _event(end=4)
    first = _consume(bridge, event=first_event)
    des.run()
    old_target = [
        spec.gpu for spec in topology.gpus if spec.gpu.dp_rank == 1
    ]
    replacement = [
        spec.gpu for spec in topology.gpus if spec.gpu.dp_rank == 2
    ]
    dram_before = des.node_dram["source"].aegis_committed
    assert dram_before == 4 * 4 * BYTES_PER_TOKEN
    assert sum(des.gpu_hbm[gpu].aegis_committed for gpu in old_target) == (
        4 * 4 * BYTES_PER_TOKEN
    )

    rotated_hint = {
        **_hint(),
        "target_dp_rank": 2,
        "hbm_target_rotation": {
            "previous_target_dp_rank": 1,
            "replacement_target_dp_rank": 2,
            "failed_target_dp_ranks": [1],
            "generation": 1,
            "route_version": 1,
            "version": 1,
        },
    }
    fenced_hint = {**rotated_hint, "reserved_l1_tokens": 0}
    fence = _consume(
        bridge,
        event=first_event,
        hint=fenced_hint,
    )
    assert fence.duplicate
    assert fence.target_dp_rank == 2
    assert fence.tier_groups == ()
    assert sum(des.gpu_hbm[gpu].aegis_committed for gpu in old_target) == 0
    assert des.node_dram["source"].aegis_committed == dram_before

    # The replacement manager may ACK its HBM-only reservation after the
    # Router fence but before another source token is produced.  Activating
    # that same-generation reservation replays only HBM from source buffers.
    rotation = _consume(
        bridge,
        event=first_event,
        hint=rotated_hint,
    )

    assert not rotation.duplicate
    assert rotation.target_dp_rank == 2
    assert {group.tier for group in rotation.tier_groups} == {
        StorageTier.GPU_AEGIS
    }
    assert all(
        owner.target_dp_rank == 2
        for owner in bridge._group_owners.values()
        if owner.group_id in {group.group_id for group in rotation.tier_groups}
    )
    des.run()
    assert sum(des.gpu_hbm[gpu].aegis_committed for gpu in old_target) == 0
    assert sum(des.gpu_hbm[gpu].aegis_committed for gpu in replacement) == (
        4 * 4 * BYTES_PER_TOKEN
    )
    assert des.node_dram["source"].aegis_committed == dram_before

    # The cumulative source event still names its admission-time target.  The
    # fenced rotation rewrites only the HBM destination; DRAM continues from
    # its own prior frontier and is copied exactly once.
    second = _consume(
        bridge,
        event=_event(end=TOKENS),
        hint=rotated_hint,
    )
    assert {
        (group.tier, group.token_start, group.token_end)
        for group in second.tier_groups
    } == {
        (StorageTier.GPU_AEGIS, 4, TOKENS),
        (StorageTier.NODE_DRAM, 4, TOKENS),
    }
    des.run()
    replacement_bytes = sum(
        des.gpu_hbm[gpu].aegis_committed for gpu in replacement
    )
    assert replacement_bytes == 4 * TOKENS * BYTES_PER_TOKEN
    assert des.node_dram["source"].aegis_committed == (
        4 * TOKENS * BYTES_PER_TOKEN
    )

    # A delayed release from the failed manager is target-scoped and cannot
    # reclaim the replacement's same-version HBM groups.
    assert bridge.apply_release_intents(
        (
            {
                "request_backup_key": "backup-1",
                "version": 1,
                "route_version": 1,
                "target_dp_rank": 1,
                "tier": "hbm",
                "reason": "late_failed_target_release",
            },
        )
    ) == ()
    assert sum(des.gpu_hbm[gpu].aegis_committed for gpu in replacement) == (
        replacement_bytes
    )
    assert _consume(
        bridge,
        event=_event(end=TOKENS),
        hint=rotated_hint,
    ).duplicate
    assert first.route_version == rotation.route_version == second.route_version == 1
    assert bridge.audit_invariants() == []


def test_target_rotation_rejects_source_or_unfenced_replacement() -> None:
    topology = _topology(include_replacement=True)
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    _consume(bridge, event=_event(end=4))
    des.run()

    source_replacement = {
        **_hint(),
        "target_dp_rank": 0,
        "hbm_target_rotation": {
            "previous_target_dp_rank": 1,
            "replacement_target_dp_rank": 0,
            "failed_target_dp_ranks": [1],
            "generation": 1,
        },
    }
    with pytest.raises(ValueError, match="must differ from source"):
        _consume(bridge, event=_event(end=4), hint=source_replacement)

    failed_replacement = {
        **_hint(),
        "target_dp_rank": 2,
        "hbm_target_rotation": {
            "previous_target_dp_rank": 1,
            "replacement_target_dp_rank": 2,
            "failed_target_dp_ranks": [1, 2],
            "generation": 1,
        },
    }
    with pytest.raises(ValueError, match="replacement target is already fenced"):
        _consume(bridge, event=_event(end=4), hint=failed_replacement)


def test_global_des_still_rejects_any_direct_hbm_to_dram_destage() -> None:
    topology = _topology()
    des = _des(topology)
    source_gpu = topology.group(0, 0)[0].gpu
    target_gpu = topology.group(1, 0)[0].gpu
    des.register_extent(
        extent_id="manual-hbm-copy",
        request_id="manual",
        size_bytes=100,
        model_revision="m",
        kv_abi="a",
        parallel_layout="tp4",
        shard_count=1,
    )
    des.place_extent(
        "manual-hbm-copy", KvLocation.gpu_aegis(target_gpu)
    )
    with pytest.raises(ValueError, match="destage is forbidden"):
        des.reserve_transfer(
            extent_id="manual-hbm-copy",
            source=KvLocation.gpu_aegis(target_gpu),
            destination=KvLocation.node_dram("store"),
            resources=(),
            duration_ns=1,
        )
    assert des.gpu_hbm[source_gpu].healthy


def test_failover_peer_hbm_on_retry_gpu_is_zero_copy_ownership_activation() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    _consume(bridge, hint=_hint(dram=0, host=""))
    des.run()
    retry_gpus = tuple(spec.gpu for spec in topology.group(1, 0))
    used_before = {
        gpu: des.gpu_hbm[gpu].used_bytes for gpu in retry_gpus
    }

    plan = bridge.plan_failover_recovery(
        retry_request_id="retry-hbm",
        request_backup_key="backup-1",
        retry_dp_rank=1,
        tier_tokens={"hbm": TOKENS},
    )

    assert plan["selected_tier"] == StorageTier.GPU_AEGIS.value
    assert len(plan["jobs"]) == 4
    assert plan["max_end_ns"] == des.now_ns
    assert all(
        job.path_kind is CopyPathKind.PEER_HBM_OWNERSHIP_ACTIVATION
        and not job.resources
        and job.start_ns == job.end_ns
        and job.source.gpu == job.destination.gpu
        for job in bridge.recovery_jobs
    )
    assert all(
        des.transfers[job.transfer_id].destination_mode
        is TransferDestinationMode.OWNERSHIP_ACTIVATION
        and des.transfers[job.transfer_id].recovery_group_id == job.group_id
        and job.destination not in des.extents[job.extent_id].locations
        for job in bridge.recovery_jobs
    )
    assert all(
        des.gpu_hbm[gpu].used_bytes == used_before[gpu]
        and des.gpu_hbm[gpu].recovery_pending == 0
        for gpu in retry_gpus
    )
    recovery_group = des.recovery_load_groups[plan["group_id"]]
    assert recovery_group.expected_shard_keys == tuple(
        f"pp0/tp{rank}" for rank in range(4)
    )
    des.run()
    assert set(bridge.export()["recovery_outcomes"].values()) == {"COMPLETED"}
    assert recovery_group.state is CommitGroupState.COMMITTED
    assert all(
        des.transfers[job.transfer_id].state is TransferState.COMMITTED
        for job in bridge.recovery_jobs
    )
    assert all(des.gpu_hbm[gpu].used_bytes == used_before[gpu] for gpu in retry_gpus)
    assert bridge.audit_invariants() == []


def test_failover_dram_load_reserves_read_pcie_fabric_and_gpu_write() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(
        des, topology, fabric=_fabric(fabric_bw=10_000_000)
    )
    _consume(bridge, hint=_hint(hbm=0, dram=TOKENS, host="source"))
    des.run()

    plan = bridge.plan_failover_recovery(
        retry_request_id="retry-dram",
        request_backup_key="backup-1",
        retry_dp_rank=1,
        tier_tokens={"dram": TOKENS},
    )
    jobs = list(bridge.recovery_jobs)

    assert plan["selected_tier"] == StorageTier.NODE_DRAM.value
    assert len(jobs) == 4
    assert all(
        job.path_kind is CopyPathKind.CROSS_NODE_DRAM_H2D_FABRIC
        for job in jobs
    )
    assert all(
        {"dram-read", "fabric", "gpu-h2d-write"}
        <= {resource.kind for resource in job.resources}
        for job in jobs
    )
    for previous, current in zip(jobs, jobs[1:]):
        assert current.start_ns >= previous.end_ns
    assert plan["max_end_ns"] == jobs[-1].end_ns
    assert all(
        des.transfers[job.transfer_id].destination_mode
        is TransferDestinationMode.EXTERNAL_PREALLOCATED_SINK
        and job.destination not in des.extents[job.extent_id].locations
        for job in jobs
    )
    assert sum(
        des.gpu_hbm[spec.gpu].recovery_pending
        for spec in topology.group(1, 0)
    ) == TOKENS * BYTES_PER_TOKEN * 4

    des.run()
    assert set(bridge.export()["recovery_outcomes"].values()) == {"COMPLETED"}
    assert des.recovery_load_groups[plan["group_id"]].state is (
        CommitGroupState.COMMITTED
    )
    assert sum(
        des.gpu_hbm[spec.gpu].recovery_pending
        for spec in topology.group(1, 0)
    ) == TOKENS * BYTES_PER_TOKEN * 4
    assert bridge.audit_invariants() == []


def test_failover_exact_completion_source_failure_aborts_canonical_group() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(
        des, topology, fabric=_fabric(fabric_bw=10_000_000)
    )
    _consume(bridge, hint=_hint(hbm=0, dram=TOKENS, host="source"))
    des.run()
    plan = bridge.plan_failover_recovery(
        retry_request_id="retry-exact-failure",
        request_backup_key="backup-1",
        retry_dp_rank=1,
        tier_tokens={"dram": TOKENS},
    )
    group = des.recovery_load_groups[plan["group_id"]]
    des.schedule_at(
        plan["max_end_ns"],
        kind="recovery-source-node-failure",
        priority=EventPriority.FAILURE,
        callback=lambda simulator, _event: simulator.fail_node("source"),
    )

    des.run()

    assert group.state is CommitGroupState.ABORTED
    assert all(
        des.transfers[job.transfer_id].state is TransferState.ABORTED
        for job in bridge.recovery_jobs
    )
    assert set(bridge.export()["recovery_outcomes"].values()) == {"ABORTED"}
    assert all(
        des.gpu_hbm[spec.gpu].recovery_pending == 0
        for spec in topology.group(1, 0)
    )
    assert bridge.audit_invariants() == []


def test_failover_recovery_capacity_preflight_leaves_no_partial_group() -> None:
    topology = _topology(target_hbm_capacity=500)
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    _consume(bridge, hint=_hint(hbm=0, dram=TOKENS, host="source"))
    des.run()
    group_ids_before = set(des.recovery_load_groups)
    busy_before = dict(des.resource_busy_until_ns)

    with pytest.raises(ValueError, match="RECOVERY_PENDING bytes"):
        bridge.plan_failover_recovery(
            retry_request_id="retry-no-capacity",
            request_backup_key="backup-1",
            retry_dp_rank=1,
            tier_tokens={"dram": TOKENS},
        )

    assert bridge.recovery_jobs == ()
    assert set(des.recovery_load_groups) == group_ids_before
    assert des.resource_busy_until_ns == busy_before
    assert all(
        des.gpu_hbm[spec.gpu].recovery_pending == 0
        for spec in topology.group(1, 0)
    )


def test_recovery_coverage_failure_has_no_partial_jobs_or_resource_reservations() -> None:
    topology = _topology()
    des = _des(topology)
    bridge = OfflineAegisTransferBridge(des, topology, fabric=_fabric())
    _consume(bridge, event=_event(end=4), hint=_hint(dram=0, host=""))
    des.run()
    busy_before = dict(des.resource_busy_until_ns)

    with pytest.raises(ValueError, match="does not cover retry shard"):
        bridge.plan_failover_recovery(
            retry_request_id="retry-too-long",
            request_backup_key="backup-1",
            retry_dp_rank=1,
            tier_tokens={"hbm": TOKENS},
        )

    assert bridge.recovery_jobs == ()
    assert des.resource_busy_until_ns == busy_before
