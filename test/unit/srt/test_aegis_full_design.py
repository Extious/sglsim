from __future__ import annotations

from sglsim.srt.simulation.aegis_full_design import (
    AegisFullDesignConfig,
    AegisFullDesignRuntime,
    CalibratedTransferPath,
    sealed_chunks,
)
from sglsim.srt.simulation.aegis_independent_tiers import (
    CopyState,
    ProtectionTier,
    RecoveryTiming,
)
from sglsim.srt.simulation.aegis_residual_dag import (
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
)


def _config() -> AegisFullDesignConfig:
    return AegisFullDesignConfig(
        hbm_capacity_bytes=1 << 30,
        dram_capacity_bytes=1 << 30,
        kv_bytes_per_token=1024,
        hbm_copy=CalibratedTransferPath(1.0, 1024.0, "unit-hbm"),
        dram_copy=CalibratedTransferPath(2.0, 512.0, "unit-dram"),
        recompute=RecoveryTiming(5.0, 1.0),
        hbm_load=RecoveryTiming(1.0, 0.1),
        dram_load=RecoveryTiming(2.0, 0.2),
        prefix_chunk_tokens=4,
        decode_chunk_tokens=2,
        pp_size=2,
        tp_size=2,
        event_log_limit=32,
    )


def _register(runtime: AegisFullDesignRuntime, request_id: str = "r") -> None:
    runtime.register_request(
        request_id=request_id,
        source_worker="source-0",
        source_failure_domain="source-domain-0",
        hbm_target="target-0",
        hbm_failure_domain="target-domain-0",
        route_version=1,
        prompt_tokens=6,
        output_tokens=5,
        slack_ms=0.0,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="pp2-tp2",
        time_ms=0.0,
    )


def _ticket(slack_ms: float, *, version: str = "v1", observed_at_ms: float = 0.0):
    graph = VisibleResidualDag(
        workflow_id="workflow",
        graph_version=version,
        observed_at_ms=observed_at_ms,
        nodes=(
            ResidualDagNode("request", "agent", 0.0, active=True, agent_id="a"),
            ResidualDagNode("critical", "agent", slack_ms, active=True, agent_id="b"),
        ),
        edges=(),
        next_join_subgraph_complete=True,
    )
    return build_residual_dag_slack_tickets(graph)["request"]


def test_chunking_separates_prefix_and_only_full_decode_chunks() -> None:
    chunks = sealed_chunks(
        6,
        5,
        prefix_chunk_tokens=4,
        decode_chunk_tokens=2,
    )
    assert [(row.token_start, row.token_end, row.kind) for row in chunks] == [
        (0, 4, "prefix"),
        (4, 6, "prefix"),
        (6, 8, "decode"),
        (8, 10, "decode"),
    ]


def test_independent_transfer_completion_and_request_reclaim_wait() -> None:
    runtime = AegisFullDesignRuntime(_config())
    _register(runtime)
    chunk = sealed_chunks(6, 5, prefix_chunk_tokens=4, decode_chunk_tokens=2)[0]
    decision = runtime.seal_chunk("r", chunk, time_ms=1.0)
    assert decision.hbm.admitted
    assert decision.dram.admitted
    assert runtime.controller.copy_state("r", 0, ProtectionTier.HBM) is CopyState.PENDING
    assert runtime.controller.copy_state("r", 0, ProtectionTier.DRAM) is CopyState.PENDING

    # HBM: 1 ms setup + 4096/1024 = 5 ms, so it commits at t=6.
    runtime.advance_to(6.0)
    assert runtime.controller.copy_state("r", 0, ProtectionTier.HBM) is CopyState.COMMITTED
    assert runtime.controller.copy_state("r", 0, ProtectionTier.DRAM) is CopyState.PENDING

    blocked = runtime.finish_request("r", time_ms=7.0)
    assert blocked["completed"] is False
    runtime.flush()
    snapshot = runtime.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["metrics"]["copy_shard_acks"] == 8
    assert snapshot["metrics"]["request_reclaim_waits"] == 1
    assert snapshot["metrics"]["request_reclaims"] == 1
    assert snapshot["transfer_engine"]["control_boundary"] == (
        "queue_and_rate_limit_only"
    )
    assert snapshot["transfer_engine"]["storage_control_selects_victims"] is True
    assert snapshot["transfer_engine"]["transfer_engine_selects_victims"] is False


def test_hbm_pressure_sample_drains_only_local_allocation_and_reclaim_ops() -> None:
    runtime = AegisFullDesignRuntime(_config())
    _register(runtime)
    chunk = sealed_chunks(6, 5, prefix_chunk_tokens=4, decode_chunk_tokens=2)[0]
    runtime.seal_chunk("r", chunk, time_ms=1.0)

    allocation = runtime.sample_hbm_local_resource_operations(time_ms=1.0)
    assert allocation == {
        "operation_count": 1,
        "allocation_count": 1,
        "reclaim_count": 0,
        "allocation_bytes": 4096,
        "reclaim_bytes": 0,
    }
    # Completing the source D2D transfer is not a target-local allocator stall.
    assert runtime.sample_hbm_local_resource_operations(time_ms=6.0)[
        "operation_count"
    ] == 0

    runtime.finish_request("r", time_ms=6.0)
    runtime.flush()
    reclaim = runtime.sample_hbm_local_resource_operations(time_ms=runtime.now_ms)
    assert reclaim["operation_count"] == 1
    assert reclaim["allocation_count"] == 0
    assert reclaim["reclaim_count"] == 1
    assert reclaim["reclaim_bytes"] == 4096


def test_target_rotation_preserves_committed_dram_and_source_failure_uses_it() -> None:
    runtime = AegisFullDesignRuntime(_config())
    _register(runtime)
    chunk = sealed_chunks(6, 5, prefix_chunk_tokens=4, decode_chunk_tokens=2)[0]
    runtime.seal_chunk("r", chunk, time_ms=1.0)
    runtime.flush()
    runtime.rotate_target(
        "r",
        failed_target="target-0",
        replacement_target="target-1",
        replacement_failure_domain="target-domain-1",
        time_ms=runtime.now_ms,
    )
    assert runtime.controller.committed_prefix_tokens("r", ProtectionTier.HBM) == 0
    assert runtime.controller.committed_prefix_tokens("r", ProtectionTier.DRAM) == 4
    recovery = runtime.recover_source_failure(
        "r",
        activated_target="target-1",
        next_route_version=2,
        remaining_slack_ms=0.0,
        time_ms=runtime.now_ms,
    )
    assert recovery.action == "DRAM"
    assert recovery.restored_tokens == 4
    assert recovery.route_version == 2


def test_slack_decays_at_chunk_seal_without_triggering_copy_movement() -> None:
    runtime = AegisFullDesignRuntime(_config())
    runtime.register_request(
        request_id="decay",
        source_worker="source-0",
        source_failure_domain="source-domain-0",
        hbm_target="target-0",
        hbm_failure_domain="target-domain-0",
        route_version=1,
        prompt_tokens=6,
        output_tokens=0,
        slack_ms=10.0,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="pp2-tp2",
        time_ms=2.0,
        slack_evidence=_ticket(
            10.0, version="decay-v1", observed_at_ms=2.0
        ).to_simulation_fields(),
    )
    chunk = sealed_chunks(6, 0, prefix_chunk_tokens=4, decode_chunk_tokens=2)[0]
    runtime.seal_chunk("decay", chunk, time_ms=7.0)
    assert runtime.controller.request_profile("decay").slack_ms == 5.0
    snapshot = runtime.snapshot(include_controller=True)
    slack_events = [
        event
        for event in snapshot["controller"]["events"]
        if event["kind"] == "slack_metadata_updated"
    ]
    assert len(slack_events) == 1
    assert slack_events[0]["physical_movement"] is False


def test_full_runtime_fails_unproven_positive_slack_closed_to_zero() -> None:
    runtime = AegisFullDesignRuntime(_config())
    runtime.register_request(
        request_id="unsafe",
        source_worker="source-0",
        source_failure_domain="source-domain-0",
        hbm_target="target-0",
        hbm_failure_domain="target-domain-0",
        route_version=1,
        prompt_tokens=4,
        output_tokens=0,
        slack_ms=999.0,
        model_revision="model-r1",
        kv_abi="kv-v1",
        parallel_layout="pp2-tp2",
        time_ms=0.0,
    )

    assert runtime.controller.request_profile("unsafe").slack_ms == 0.0
    priority = runtime.prioritize_recovery(
        ["unsafe"],
        activated_targets={"unsafe": "target-0"},
        time_ms=0.0,
    )
    assert priority[0]["remaining_slack_ms"] == 0.0
    assert priority[0]["slack_evidence_reason"].startswith("missing_or_invalid_")


def test_recovery_priority_uses_decayed_visible_residual_dag_slack() -> None:
    runtime = AegisFullDesignRuntime(_config())
    for request_id, slack, evidence in (
        ("unknown", 100.0, None),
        ("causal", 5.0, _ticket(5.0).to_simulation_fields()),
    ):
        runtime.register_request(
            request_id=request_id,
            source_worker=f"source-{request_id}",
            source_failure_domain=f"domain-{request_id}",
            hbm_target="target-0",
            hbm_failure_domain="target-domain-0",
            route_version=1,
            prompt_tokens=4,
            output_tokens=0,
            slack_ms=slack,
            model_revision="model-r1",
            kv_abi="kv-v1",
            parallel_layout="pp2-tp2",
            time_ms=0.0,
            slack_evidence=evidence,
        )

    priority = runtime.prioritize_recovery(
        ["causal", "unknown"],
        activated_targets={"causal": "target-0", "unknown": "target-0"},
        time_ms=2.0,
    )

    assert [row["request_id"] for row in priority] == ["unknown", "causal"]
    assert [row["remaining_slack_ms"] for row in priority] == [0.0, 3.0]
    assert all(row["metadata_only"] is True for row in priority)


def test_slack_ticket_update_never_queues_or_moves_kv() -> None:
    runtime = AegisFullDesignRuntime(_config())
    _register(runtime, "metadata")
    before = runtime.snapshot(include_controller=True)

    runtime.update_slack_from_ticket(
        "metadata", _ticket(8.0, version="metadata-v2"), time_ms=1.0
    )
    after = runtime.snapshot(include_controller=True)

    assert after["pending_transfer_jobs"] == before["pending_transfer_jobs"] == 0
    assert after["controller"]["pools"]["HBM"]["used_bytes"] == 0
    assert after["controller"]["pools"]["DRAM"]["used_bytes"] == 0
    assert after["controller"]["events"][-1]["physical_movement"] is False
