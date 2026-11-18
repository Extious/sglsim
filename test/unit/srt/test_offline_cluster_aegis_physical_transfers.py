from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sglsim.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.aegis_residual_dag import (
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
)
from sglsim.srt.simulation.global_des import (
    CommitGroupState,
    EventPriority,
    GlobalDES,
    PhysicalGpuId,
    StorageTier,
    TransferState,
)
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    PhysicalGpuSpec,
)


TOKENS = 64
KV_BYTES_PER_TOKEN = 128


def _config(tmp_path) -> str:
    path = tmp_path / "offline-aegis-physical.json"
    path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "tp_size": 2,
                    "pp_size": 1,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 256,
                    "chunked_prefill_size": 256,
                    "max_running_requests": 8,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 4096,
                    "aegis_total_l1_capacity_pages": 4096,
                },
                "communication": {
                    "kv_bytes_per_token": KV_BYTES_PER_TOKEN
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "committed_copy_protocol": True,
                    "design_semantics": DESIGN_SEMANTICS,
                    "chunk_tokens": 16,
                    "prefix_chunk_tokens": 16,
                    "decode_chunk_tokens": 1,
                    "host_alignment_tokens": 1,
                    "initial_alpha": 1.0,
                    "initial_beta": 1.0,
                    "recompute_setup_ms": 10.0,
                    "recompute_ms_per_token": 0.1,
                    "l1_restore_setup_ms": 0.0,
                    "l1_restore_ms_per_token": 0.001,
                    "l2_restore_setup_ms": 0.0,
                    "l2_restore_ms_per_token": 0.001,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _topology() -> OfflinePhysicalTopology:
    return OfflinePhysicalTopology(
        gpus=tuple(
            PhysicalGpuSpec(
                PhysicalGpuId(
                    node_id=f"node{dp_rank}",
                    local_gpu_id=tp_rank,
                    dp_rank=dp_rank,
                    pp_rank=0,
                    tp_rank=tp_rank,
                ),
                hbm_capacity_bytes=1 << 30,
            )
            for dp_rank in range(2)
            for tp_rank in range(2)
        ),
        nodes=(
            NodeDramSpec("node0", 1 << 30),
            NodeDramSpec("node1", 1 << 30),
        ),
    )


def _request(
    *,
    rid: str = "physical-request",
    token_count: int = TOKENS,
    max_new_tokens: int = 0,
) -> TokenizedGenerateReqInput:
    graph = VisibleResidualDag(
        workflow_id="physical-copy-workflow",
        graph_version="visible-v1",
        observed_at_ms=0.0,
        nodes=(
            ResidualDagNode("fork", "control", 0.0),
            ResidualDagNode(
                rid, "agent", 0.0, active=True, agent_id="a"
            ),
            ResidualDagNode("peer", "agent", 1.0, active=True, agent_id="b"),
            ResidualDagNode("join", "control", 0.0),
        ),
        edges=(
            ("fork", rid),
            ("fork", "peer"),
            (rid, "join"),
            ("peer", "join"),
        ),
        next_join_subgraph_complete=True,
    )
    simulation = build_residual_dag_slack_tickets(graph)[
        rid
    ].to_simulation_fields()
    simulation.update(
        {
            "server_created_time": 0.0,
            "force_dp_rank": 0,
            "aegis_policy": "aegiskv_l1_l2",
            "backup_kv": True,
            "critical_agent": True,
            "aegis_critical_probability": 1.0,
            "aegis_backup_dp_rank": 1,
            "aegis_failure_scope": "gpu",
        }
    )
    return TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid=rid,
            input_ids=list(range(token_count)),
            max_new_tokens=max_new_tokens,
            simulation=simulation,
        )
    )


def _cluster(tmp_path) -> OfflineSchedulerCluster:
    return OfflineSchedulerCluster(
        ServerArgs(
            model_path="mock-model",
            dp_size=2,
            tp_size=2,
            streaming_arrival=True,
            sim_config_path=_config(tmp_path),
        ),
        physical_topology=_topology(),
    )


def _drive_until_jobs(cluster: OfflineSchedulerCluster) -> None:
    cluster.dispatch(_request())
    cluster.pump_ready_stages()
    initial = cluster.dp_controller.recovery_hint("physical-request")
    assert initial is not None
    assert initial["reserved_l1_tokens"] == TOKENS
    assert initial["reserved_l2_tokens"] == TOKENS
    assert initial["safe_frontier_tokens"] == 0
    assert not initial.get("group_commit_verified", False)
    for _ in range(20):
        if cluster.aegis_transfer_bridge.jobs:
            return
        cluster.process_next_event()
    raise AssertionError("physical transfer jobs were not materialized")


def test_cluster_commits_independent_physical_tp2_tiers(tmp_path) -> None:
    with _cluster(tmp_path) as cluster:
        assert cluster.aegis_transfer_bridge is not None
        _drive_until_jobs(cluster)
        bridge = cluster.aegis_transfer_bridge
        chunk_count = 1 + (TOKENS - 16)
        assert len(bridge.jobs) == chunk_count * 2 * 2
        assert {
            (job.token_start, job.token_end) for job in bridge.jobs
        } == {(0, 16), *((token, token + 1) for token in range(16, TOKENS))}
        assert not bridge.tier_notifications
        assert all(job.source.tier is StorageTier.GPU_PRIMARY for job in bridge.jobs)
        assert {job.tp_rank for job in bridge.jobs} == {0, 1}
        assert {job.tier for job in bridge.jobs} == {
            StorageTier.GPU_AEGIS,
            StorageTier.NODE_DRAM,
        }
        assert all(
            cluster.global_des.commit_groups[job.group_id].state
            is CommitGroupState.PENDING
            for job in bridge.jobs
        )

        notify_times = sorted(
            {
                max(job.end_ns for job in bridge.jobs if job.group_id == group_id)
                for group_id in {job.group_id for job in bridge.jobs}
            }
        )
        assert len(notify_times) == chunk_count * 2
        assert notify_times[0] < notify_times[-1]
        while not bridge.tier_notifications:
            cluster.process_next_event()
        assert len(bridge.tier_notifications) == 1
        first = bridge.tier_notifications[0]
        assert first.success
        other_group = next(
            group
            for group in cluster.global_des.commit_groups.values()
            if group.group_id != first.group_id
        )
        assert other_group.state is CommitGroupState.PENDING

        result = cluster.run_until_complete(
            expected_request_ids={"physical-request"}, max_events=1000
        )
        final_hint = cluster.dp_controller.recovery_hint("physical-request")

    assert result.resource_audit_errors == []
    assert len(result.physical_aegis_transfers["notifications"]) == chunk_count * 2
    assert all(
        row["success"]
        for row in result.physical_aegis_transfers["notifications"]
    )
    assert all(
        row["controller_dispatched"]
        for row in result.physical_aegis_transfers["controller_dispatches"]
    )
    assert final_hint is None
    assert result.physical_aegis_transfers["completion_events"][-1][
        "dispatched"
    ]
    assert sum(
        row["kind"] == "physical_copy_released"
        for row in result.physical_aegis_transfers["audit"]
    ) == chunk_count * 2
    assert all(
        row["aegis_pending"] == 0 and row["aegis_committed"] == 0
        for row in result.global_des_snapshot["gpus"].values()
    )
    assert all(
        row["aegis_pending"] == 0 and row["aegis_committed"] == 0
        for row in result.global_des_snapshot["node_dram"].values()
    )
    assert all(
        not (
            row["source"]["tier"] == StorageTier.GPU_AEGIS.value
            and row["destination"]["tier"] == StorageTier.NODE_DRAM.value
        )
        for row in result.physical_aegis_transfers["jobs"]
    )


def test_physical_source_frontier_excludes_unowned_partial_page_tail(
    tmp_path,
) -> None:
    """A token frontier cannot alias bytes absent from the same response."""

    rid = "physical-partial-page"
    with _cluster(tmp_path) as cluster:
        cluster.dispatch(_request(rid=rid, token_count=17))
        result = cluster.run_until_complete(
            expected_request_ids={rid}, max_events=300
        )

    adjustments = result.physical_aegis_transfers[
        "source_frontier_adjustments"
    ]
    assert len(adjustments) == 1
    assert adjustments[0] == {
        "timestamp_ns": adjustments[0]["timestamp_ns"],
        "rid": rid,
        "request_backup_key": rid,
        "source_dp_rank": 0,
        "source_pp_rank": 0,
        "version": 1,
        "route_version": 1,
        "page_size_tokens": 16,
        "source_observed_token_end": 17,
        "retained_page_frontier": 16,
        "previous_materialized_frontier": 0,
        "physical_source_token_end": 16,
        "deferred_recompute_tokens": 1,
        "reason": "logical_frontier_exceeds_allocator_owned_pages",
    }
    assert max(
        row["token_end"]
        for row in result.physical_aegis_transfers["jobs"]
    ) == 16
    assert all(
        extent["token_end"] <= 16
        for extent in result.global_des_snapshot["extents"].values()
        if extent["request_id"] == rid
    )
    assert result.resource_audit_errors == []


def test_p2_1807_source_frontier_is_bounded_to_1792_owned_tokens() -> None:
    """Regression for P2:21's exact 15-token / 2,211,840-byte shortfall."""

    cluster = object.__new__(OfflineSchedulerCluster)
    cluster._latest_kv_resource_snapshots = {
        (9, 0): {
            "page_size_tokens": 16,
            "l1_device": {
                "owner_pages": {"tl-98658b6b1aca296f7f025d7b": 112}
            },
        }
    }
    cluster.aegis_transfer_bridge = SimpleNamespace(
        materialized_source_frontier_token_end=lambda **_kwargs: 0
    )
    cluster.global_des = SimpleNamespace(now_ns=123)
    cluster._physical_aegis_source_frontier_adjustments = []

    bounded = cluster._bound_physical_source_frontier_to_allocator(
        {
            "type": "aegis_source_chunk_committed",
            "rid": "tl-98658b6b1aca296f7f025d7b",
            "request_backup_key": "tl-98658b6b1aca296f7f025d7b",
            "source_dp_rank": 9,
            "source_pp_rank": 0,
            "version": 1,
            "route_version": 1,
            "execution_id": "p2-21:v1:r1:dp9",
            "source_committed_token_end": 1807,
            "source_observed_token_end": 1807,
            "request_finished": False,
        }
    )

    assert bounded["source_committed_token_end"] == 1792
    assert bounded["physical_source_observed_token_end"] == 1807
    assert bounded["physical_source_deferred_recompute_tokens"] == 15
    assert cluster._physical_aegis_source_frontier_adjustments == [
        {
            "timestamp_ns": 123,
            "rid": "tl-98658b6b1aca296f7f025d7b",
            "request_backup_key": "tl-98658b6b1aca296f7f025d7b",
            "source_dp_rank": 9,
            "source_pp_rank": 0,
            "version": 1,
            "route_version": 1,
            "page_size_tokens": 16,
            "source_observed_token_end": 1807,
            "retained_page_frontier": 1792,
            "previous_materialized_frontier": 0,
            "physical_source_token_end": 1792,
            "deferred_recompute_tokens": 15,
            "reason": "logical_frontier_exceeds_allocator_owned_pages",
        }
    ]


def test_target_physical_gpu_failure_aborts_hbm_but_dram_commits(tmp_path) -> None:
    """A physical-ledger failure only; control-plane fail-stop is separate."""

    with _cluster(tmp_path) as cluster:
        _drive_until_jobs(cluster)
        bridge = cluster.aegis_transfer_bridge
        target_gpu = cluster.resource_bridge.topology.group(1, 0)[0].gpu
        cluster.global_des.schedule_at(
            cluster.global_des.now_ns,
            kind="unit_target_physical_gpu_failure",
            priority=EventPriority.FAILURE,
            callback=lambda simulator, _event: simulator.fail_gpu(target_gpu),
        )
        result = cluster.run_until_complete(
            expected_request_ids={"physical-request"}, max_events=1000
        )
        final_hint = cluster.dp_controller.recovery_hint("physical-request")

    notifications = result.physical_aegis_transfers["notifications"]
    assert all(
        not row["success"]
        for row in notifications
        if row["tier"] == StorageTier.GPU_AEGIS.value
    )
    assert all(
        row["success"]
        for row in notifications
        if row["tier"] == StorageTier.NODE_DRAM.value
    )
    dispatches = result.physical_aegis_transfers["controller_dispatches"]
    assert all(
        not row["controller_dispatched"]
        for row in dispatches
        if row["tier"] == StorageTier.GPU_AEGIS.value
    )
    assert all(
        row["controller_dispatched"]
        for row in dispatches
        if row["tier"] == StorageTier.NODE_DRAM.value
    )
    hbm_jobs = [
        job for job in bridge.jobs if job.tier is StorageTier.GPU_AEGIS
    ]
    assert all(
        cluster.global_des.transfers[job.transfer_id].state
        is TransferState.ABORTED
        for job in hbm_jobs
    )
    assert final_hint is None
    assert result.resource_audit_errors == []


def test_hard_target_failstop_loses_hbm_but_preserves_source_dram(
    tmp_path,
) -> None:
    """The control plane must mirror the independent physical tier loss."""

    request = _request()
    request.sampling_params.max_new_tokens = 8
    with _cluster(tmp_path) as cluster:
        cluster.dispatch(request)
        cluster.pump_ready_stages()
        for _ in range(1000):
            committed = cluster.dp_controller.recovery_hint("physical-request")
            if (
                isinstance(committed, dict)
                and committed["l1_tokens"] >= TOKENS
                and committed["l2_tokens"] >= TOKENS
                and committed["safe_frontier_tokens"] >= TOKENS
                and committed.get("group_commit_verified", False)
                and cluster.coordinator.stage_is_busy(0, 0)
            ):
                break
            cluster.process_next_event()
        else:
            raise AssertionError("physical prefix did not commit while source was live")

        cluster.schedule_fail_stop(
            event_id="hard-target-rank1",
            dp_rank=1,
            offset_s=cluster.coordinator.now_s,
            duration_s=0.001,
            failure_scope="gpu",
        )
        result = cluster.run_until_complete(
            expected_request_ids={"physical-request"}, max_events=1000
        )
        after_failure = cluster.dp_controller.recovery_hint(
            "physical-request"
        )

    assert result.resource_audit_errors == []
    assert after_failure is None
    invalidated = result.failure_events[0]["controller_invalidation"]
    assert invalidated["invalidated_hints"] == [
        {
            "backup_key": "physical-request",
            "version": 1,
            "target_dp_rank": 1,
            "invalidated_tiers": ("hbm",),
            "invalidated_hbm_reserved_tokens": TOKENS + 8,
            "safe_frontier_tokens": TOKENS,
        }
    ]


def test_hard_target_failstop_aborts_pending_hbm_reservation_only(
    tmp_path,
) -> None:
    """A failure before ACK must clear the HBM reservation immediately."""

    with _cluster(tmp_path) as cluster:
        _drive_until_jobs(cluster)
        initial = cluster.dp_controller.recovery_hint("physical-request")
        assert initial is not None
        assert initial["hbm_pending_tokens"] == TOKENS
        assert initial["dram_pending_tokens"] == TOKENS

        cluster.schedule_fail_stop(
            event_id="hard-pending-target-rank1",
            dp_rank=1,
            offset_s=cluster.coordinator.now_s,
            duration_s=0.001,
            failure_scope="gpu",
        )
        injected = cluster.process_next_event()
        assert injected.kind == "hard_failure_inject"
        during_failure = cluster.dp_controller.recovery_hint(
            "physical-request"
        )
        assert during_failure is not None
        assert during_failure["hbm_pending_tokens"] == 0
        assert during_failure["reserved_l1_tokens"] == 0
        assert during_failure["dram_pending_tokens"] == TOKENS
        assert during_failure["reserved_l2_tokens"] == TOKENS

        result = cluster.run_until_complete(
            expected_request_ids={"physical-request"}, max_events=1000
        )
        final_hint = cluster.dp_controller.recovery_hint("physical-request")

    assert result.resource_audit_errors == []
    assert final_hint is None


def test_hard_source_failstop_reroutes_real_scheduler_replay_from_peer_hbm(
    tmp_path,
) -> None:
    """A killed active request must recover through the real retry path."""

    original = _request()
    original.sampling_params.max_new_tokens = 8
    with _cluster(tmp_path) as cluster:
        cluster.dispatch(original)
        cluster.pump_ready_stages()
        for _ in range(1000):
            hint = cluster.dp_controller.recovery_hint("physical-request")
            original_finished = any(
                row.get("rid") == "physical-request"
                and bool(row.get("finished", False))
                for row in cluster.detokenizer_messages
            )
            if (
                isinstance(hint, dict)
                and int(hint.get("l1_tokens", 0)) >= TOKENS
                and int(hint.get("l2_tokens", 0)) >= TOKENS
                and int(hint.get("safe_frontier_tokens", 0)) >= TOKENS
                and bool(hint.get("group_commit_verified", False))
                and cluster.coordinator.stage_is_busy(0, 0)
                and not original_finished
            ):
                break
            cluster.process_next_event()
        else:
            raise AssertionError(
                "physical copies did not commit while source remained active"
            )

        cluster.schedule_fail_stop(
            event_id="hard-active-source-rank0",
            dp_rank=0,
            offset_s=cluster.coordinator.now_s,
            duration_s=0.001,
            failure_scope="gpu",
        )
        injected = cluster.process_next_event()
        assert injected.kind == "hard_failure_inject"
        failure = cluster._active_failures["hard-active-source-rank0"]
        assert failure["affected_request_ids"] == ["physical-request"]

        retry = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid="physical-request-retry",
                input_ids=list(range(TOKENS)),
                max_new_tokens=8,
                simulation={
                    "server_created_time": cluster.coordinator.now_s,
                    "aegis_policy": "aegiskv_l1_l2",
                    "backup_kv": False,
                    "aegis_backup_key": "physical-request",
                    "failover_replay": True,
                    "aegis_hard_failure_injected": True,
                    "failed_dp_rank": 0,
                    "pre_failure_total_tokens": TOKENS + 8,
                    "aegis_route_plan_retry_dp_rank": 1,
                },
            )
        )
        cluster.schedule_request_arrival(
            retry, at_s=cluster.coordinator.now_s
        )
        result = cluster.run_until_complete(
            expected_request_ids={"physical-request-retry"}, max_events=300
        )
        retry_sim = retry.sampling_params.custom_params["simulation"]

    final = next(
        row
        for row in result.detokenizer_messages
        if row.get("rid") == "physical-request-retry"
        and bool(row.get("finished", False))
    )
    assert final["aegis_assigned_dp_rank"] == 1
    assert final["aegis_cross_rank_retry"] is True
    assert retry_sim["restore_action"] == "l1_backup"
    assert retry_sim["restore_tokens"] >= TOKENS
    assert (
        retry_sim["restore_tokens"]
        + retry_sim["restore_recompute_tokens"]
        == TOKENS + 8
    )
    # The controller selects the committed token frontier.  The Scheduler's
    # page allocator may materialize a smaller page-aligned prefix and moves
    # the shortfall back into recomputation without exceeding that frontier.
    assert 0 < final["aegis_replay_restored_tokens"] <= retry_sim[
        "restore_tokens"
    ]
    recovery_jobs = result.physical_aegis_transfers["recovery_jobs"]
    assert recovery_jobs
    assert all(
        row["source_tier"] == StorageTier.GPU_AEGIS.value
        and row["path_kind"] == "peer_hbm_ownership_activation"
        and row["resources"] == []
        and row["start_ns"] == row["end_ns"]
        and row["source"]["gpu"] == row["destination"]["gpu"]
        for row in recovery_jobs
    )
    assert set(
        result.physical_aegis_transfers["recovery_outcomes"].values()
    ) == {"COMPLETED"}
    assert retry_sim["restore_timing_source"] == "global_des_physical_recovery"
    assert retry_sim["restore_overhead_s"] == retry_sim[
        "restore_recompute_delay_s"
    ]
    assert retry_sim["physical_recovery_plan"]["max_end_ns"] >= retry_sim[
        "physical_recovery_plan"
    ]["ready_ns"]
    assert result.resource_audit_errors == []
    admission_gate = result.physical_aegis_transfers[
        "recovery_admission_gate"
    ]
    admission = next(
        row
        for row in admission_gate["admissions"]
        if row["request_id"] == "physical-request-retry"
    )
    assert admission["state"] == "ADOPTED"
    assert admission["adopted_shards"] == ["pp0/tp0", "pp0/tp1"]
    assert admission_gate["pending_holds"] == []
    canonical_group = admission_gate["canonical_groups"][0]
    assert canonical_group["state"] == CommitGroupState.COMMITTED.value
    assert canonical_group["fully_adopted"] is True
    assert result.router_output_ledger[
        "physical_recovery_job_activation_gate_in_scope"
    ] is True
    assert all(
        row["aegis_pending"] == 0 and row["aegis_committed"] == 0
        for row in result.global_des_snapshot["gpus"].values()
    )
    assert all(
        row["aegis_pending"] == 0 and row["aegis_committed"] == 0
        for row in result.global_des_snapshot["node_dram"].values()
    )


@pytest.mark.parametrize("partial_adoption", (False, True))
def test_pending_recovery_target_failure_fences_route_before_scheduler_admission(
    tmp_path,
    partial_adoption,
) -> None:
    """A second fail-stop must fence v2; it may not silently reroute v2."""

    original = _request()
    original.sampling_params.max_new_tokens = 8
    with _cluster(tmp_path) as cluster:
        cluster.dispatch(original)
        cluster.pump_ready_stages()
        for _ in range(1000):
            hint = cluster.dp_controller.recovery_hint("physical-request")
            if (
                isinstance(hint, dict)
                and int(hint.get("l1_tokens", 0)) >= TOKENS
                and bool(hint.get("group_commit_verified", False))
                and cluster.coordinator.stage_is_busy(0, 0)
            ):
                break
            cluster.process_next_event()
        else:
            raise AssertionError("peer-HBM protection did not commit")

        cluster.schedule_fail_stop(
            event_id="first-source-failure",
            dp_rank=0,
            offset_s=cluster.coordinator.now_s,
            duration_s=0.001,
            failure_scope="gpu",
        )
        assert cluster.process_next_event().kind == "hard_failure_inject"
        for _ in range(100):
            repaired = cluster.process_next_event()
            if repaired.kind == "hard_failure_repair":
                break
        else:
            raise AssertionError("first failed rank did not repair")

        retry = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid="physical-request-retry-v2",
                input_ids=list(range(TOKENS)),
                max_new_tokens=8,
                simulation={
                    "server_created_time": cluster.coordinator.now_s,
                    "aegis_policy": "aegiskv_l1_l2",
                    "backup_kv": False,
                    "aegis_backup_key": "physical-request",
                    "failover_replay": True,
                    "aegis_hard_failure_injected": True,
                    "failed_dp_rank": 0,
                    "pre_failure_total_tokens": TOKENS + 8,
                    "aegis_route_plan_retry_dp_rank": 1,
                },
            )
        )
        assigned = cluster.dispatch(retry)
        assert assigned == 1
        admission = cluster._physical_recovery_admissions[retry.rid]
        assert admission["state"] == "PENDING_TRANSFER"
        assert retry not in cluster._request_inboxes[1]
        group = cluster.global_des.recovery_load_groups[admission["group_id"]]
        if partial_adoption:
            group.acked_transfers.update(group.transfer_ids)
            group.state = CommitGroupState.COMMITTED
            for transfer_id in group.transfer_ids:
                cluster.global_des.transfers[
                    transfer_id
                ].state = TransferState.COMMITTED
            adopted_key = group.expected_shard_keys[0]
            cluster.global_des.adopt_recovery_load_shard(
                group.group_id, adopted_key
            )
            admission["state"] = "STAGE_ADOPTION_IN_PROGRESS"
            admission["adopted_shards"] = [adopted_key]

        cluster.schedule_fail_stop(
            event_id="second-pending-target-failure",
            dp_rank=1,
            offset_s=cluster.coordinator.now_s,
            duration_s=0.001,
            failure_scope="gpu",
        )
        assert cluster.process_next_event().kind == "hard_failure_inject"

        failure = cluster._active_failures["second-pending-target-failure"]
        assert retry.rid in failure["affected_request_ids"]
        assert failure["fenced_pending_recovery_admissions"] == [retry.rid]
        assert admission["state"] == "FENCED_BY_HARD_FAILURE"
        assert admission["fallback_reason"] == (
            "target_hard_failure:second-pending-target-failure"
        )
        assert admission["partial_adoption_at_failure"] is partial_adoption
        assert admission["lost_adopted_shards"] == (
            [group.expected_shard_keys[0]] if partial_adoption else []
        )
        assert group.discarded is True
        assert group.state is (
            CommitGroupState.COMMITTED
            if partial_adoption
            else CommitGroupState.ABORTED
        )
        if partial_adoption:
            assert group.adopted_shards == {group.expected_shard_keys[0]}
            for transfer_id in group.transfer_ids:
                transfer = cluster.global_des.transfers[transfer_id]
                if transfer.recovery_shard_key in group.adopted_shards:
                    assert transfer.recovery_adopted is True
                    assert transfer.recovery_discarded is False
                else:
                    assert transfer.recovery_discarded is True
        assert all(
            not cluster.global_des.transfers[transfer_id]
            .recovery_capacity_hold_active
            for transfer_id in group.transfer_ids
        )
        assert retry not in cluster._request_inboxes[0]
        assert retry not in cluster._request_inboxes[1]
        assert cluster.router_output_ledger.execution_record(retry.rid)[
            "state"
        ] == "fenced"


def test_fallback_output_gate_rejects_incomplete_terminal_evidence() -> None:
    """Fallback output requires a reason, terminal group, and zero holds."""

    des = GlobalDES()
    cluster = object.__new__(OfflineSchedulerCluster)
    cluster.global_des = des
    admission = {
        "request_id": "retry-fallback-gate",
        "request_backup_key": "backup-fallback-gate",
        "current_dp_rank": 1,
        "route_version": 2,
        "group_id": "",
        "state": "FALLBACK_RECOMPUTE",
        "fallback_reason": "",
    }
    cluster._physical_recovery_admissions = {
        admission["request_id"]: admission
    }

    with pytest.raises(RuntimeError, match="fallback fence has no reason"):
        cluster._physical_recovery_gate_evidence(admission["request_id"])

    admission["fallback_reason"] = "capacity_shortage"
    evidence = cluster._physical_recovery_gate_evidence(
        admission["request_id"]
    )
    assert evidence is not None
    assert evidence["canonical_group"] is None
    assert evidence["active_recovery_capacity_hold_transfer_ids"] == []

    admission["group_id"] = "missing-group"
    des.transfers["orphan-transfer"] = SimpleNamespace(
        transfer_id="orphan-transfer",
        recovery_group_id="missing-group",
        recovery_capacity_hold_active=True,
    )
    with pytest.raises(RuntimeError, match="missing recovery group with orphan"):
        cluster._physical_recovery_gate_evidence(admission["request_id"])
    del des.transfers["orphan-transfer"]

    group = des.register_recovery_load_group(
        group_id="fallback-gate-group",
        retry_request_id=admission["request_id"],
        request_backup_key=admission["request_backup_key"],
        route_version=2,
        expected_shard_keys=("pp0/tp0",),
    )
    admission["group_id"] = group.group_id
    with pytest.raises(RuntimeError, match="retained recovery group"):
        cluster._physical_recovery_gate_evidence(admission["request_id"])

    group.discarded = True
    with pytest.raises(RuntimeError, match="has no terminal reason"):
        cluster._physical_recovery_gate_evidence(admission["request_id"])

    group.discard_reason = "capacity_shortage"
    group.adopted_shards.add("pp0/tp0")
    with pytest.raises(RuntimeError, match="has adopted shards"):
        cluster._physical_recovery_gate_evidence(admission["request_id"])

    group.adopted_shards.clear()
    group.shard_transfers["pp0/tp0"] = ["active-hold"]
    des.transfers["active-hold"] = SimpleNamespace(
        recovery_capacity_hold_active=True
    )
    with pytest.raises(RuntimeError, match="retained RECOVERY_PENDING holds"):
        cluster._physical_recovery_gate_evidence(admission["request_id"])


def test_unavailable_fallback_target_requires_new_router_execution(tmp_path) -> None:
    retry = _request()
    retry.rid = "unavailable-fallback-route"
    with _cluster(tmp_path) as cluster:
        cluster.dp_controller.mark_rank_hard_failed(0)
        record = {
            "request": retry,
            "request_id": retry.rid,
            "request_backup_key": "unavailable-fallback-backup",
            "current_dp_rank": 0,
            "group_id": None,
            "state": "PLANNING_FAILED",
        }

        with pytest.raises(RuntimeError, match="new execution"):
            cluster._activate_physical_recovery_fallback(
                record, reason="target_rank_unavailable"
            )

        assert record["current_dp_rank"] == 0
        assert record["state"] == "PLANNING_FAILED"


def test_recovery_capacity_shortage_falls_back_without_protected_prefix(
    tmp_path,
    monkeypatch,
) -> None:
    """Planner admission failure must become an explicit recompute execution."""

    original = _request()
    original.sampling_params.max_new_tokens = 8
    with _cluster(tmp_path) as cluster:
        cluster.dispatch(original)
        cluster.pump_ready_stages()
        for _ in range(1000):
            hint = cluster.dp_controller.recovery_hint("physical-request")
            if (
                isinstance(hint, dict)
                and int(hint.get("l1_tokens", 0)) >= TOKENS
                and bool(hint.get("group_commit_verified", False))
                and cluster.coordinator.stage_is_busy(0, 0)
            ):
                break
            cluster.process_next_event()
        else:
            raise AssertionError("peer-HBM protection did not commit")

        cluster.schedule_fail_stop(
            event_id="capacity-source-failure",
            dp_rank=0,
            offset_s=cluster.coordinator.now_s,
            duration_s=0.001,
            failure_scope="gpu",
        )
        assert cluster.process_next_event().kind == "hard_failure_inject"

        assert cluster.aegis_transfer_bridge is not None
        monkeypatch.setattr(
            cluster.aegis_transfer_bridge,
            "plan_failover_recovery",
            lambda **_kwargs: (_ for _ in ()).throw(
                ValueError("RECOVERY_PENDING capacity shortage")
            ),
        )
        retry = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid="physical-request-capacity-fallback",
                input_ids=list(range(TOKENS)),
                max_new_tokens=8,
                simulation={
                    "server_created_time": cluster.coordinator.now_s,
                    "aegis_policy": "aegiskv_l1_l2",
                    "backup_kv": False,
                    "aegis_backup_key": "physical-request",
                    "failover_replay": True,
                    "aegis_hard_failure_injected": True,
                    "failed_dp_rank": 0,
                    "pre_failure_total_tokens": TOKENS + 8,
                    "aegis_route_plan_retry_dp_rank": 1,
                },
            )
        )
        cluster.schedule_request_arrival(retry, at_s=cluster.coordinator.now_s)
        result = cluster.run_until_complete(
            expected_request_ids={retry.rid}, max_events=500
        )

    sim = retry.sampling_params.custom_params["simulation"]
    admission = result.physical_aegis_transfers[
        "recovery_admission_gate"
    ]["admissions"][0]
    assert admission["state"] == "FALLBACK_RECOMPUTE"
    assert "capacity shortage" in admission["fallback_reason"]
    assert sim["restore_tokens"] == 0
    assert sim["restore_action"] == "recompute"
    assert sim["restore_timing_source"] == "global_des_recompute_fallback"
    final = next(
        row
        for row in result.detokenizer_messages
        if row.get("rid") == retry.rid and bool(row.get("finished", False))
    )
    assert final["aegis_replay_restored_tokens"] == 0
    assert result.physical_aegis_transfers[
        "recovery_admission_gate"
    ]["pending_holds"] == []
    assert result.resource_audit_errors == []


def test_recovery_gate_rejects_stale_protection_route(monkeypatch) -> None:
    """A committed byte group cannot authorize a newer/stale control route."""

    des = GlobalDES()
    group = des.register_recovery_load_group(
        group_id="stale-route-group",
        retry_request_id="retry-stale",
        request_backup_key="backup-stale",
        route_version=7,
        expected_shard_keys=("pp0/tp0",),
    )
    group.state = CommitGroupState.COMMITTED
    cluster = object.__new__(OfflineSchedulerCluster)
    cluster.global_des = des
    cluster._physical_recovery_admissions = {
        "retry-stale": {
            "request_id": "retry-stale",
            "request_backup_key": "backup-stale",
            "current_dp_rank": 1,
            "route_version": 7,
            "group_id": group.group_id,
            "state": "PENDING_TRANSFER",
        }
    }

    class _StaleController:
        @staticmethod
        def recovery_hint(_backup_key):
            return {"route_version": 8}

    cluster.dp_controller = _StaleController()
    observed = {}

    def fallback(record, *, reason):
        observed.update({"record": record, "reason": reason})
        return 1

    monkeypatch.setattr(
        cluster,
        "_activate_physical_recovery_fallback",
        fallback,
    )
    resolved = cluster._resolve_physical_recovery_admission(
        SimpleNamespace(
            payload={
                "request_id": "retry-stale",
                "group_id": group.group_id,
            }
        )
    )

    assert resolved == 1
    assert observed["reason"] == "stale_route:route_version_changed"
