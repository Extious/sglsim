from __future__ import annotations

import json
import queue
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from sglsim.srt.managers.data_parallel_controller import (
    launch_pipeline_scheduler_processes,
)
from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.managers.pipeline_parallel import (
    DPBudgetSnapshot,
    PPOutputEnvelope,
    PPProxyEnvelope,
    PPRequestEnvelope,
    PipelineProcessGroup,
    PipelineBatchIdentityError,
    PipelineWorkerHandle,
    PPStageBatchEnvelope,
    RecoveryHint,
    RoutingDecision,
    StageCommitAck,
    StageId,
    StagePlacementDecision,
    assert_batch_identity,
    enumerate_pipeline_workers,
)
from sglsim.srt.managers.scheduler import Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.network import build_dp_pp_ipc_registry


def _route() -> RoutingDecision:
    return RoutingDecision(
        request_id="r0",
        workflow_id="w0",
        execution_dp_rank=0,
        backup_dp_rank=2,
        route_version=3,
        decided_logical_time=1.25,
        failure_domain_constraints=("node0", "node2"),
    )


def _budget() -> DPBudgetSnapshot:
    return DPBudgetSnapshot(
        dp_rank=2,
        budget_version=7,
        window_id=4,
        alpha=0.25,
        beta=0.5,
        hbm_limit_tokens_per_stage=(100, 110, 120, 130),
        host_policy_limit_tokens=500,
        valid_from_logical_time=1.0,
        objective_id="obj-1",
        solver_certificate_id="cert-1",
    )


def _ack(pp_rank: int, frontier: int, *, domain: str | None = None) -> StageCommitAck:
    total = 128
    protected = frontier
    return StageCommitAck(
        request_id="r0",
        source_stage=StageId(0, pp_rank),
        backup_stage=StageId(2, pp_rank),
        route_version=3,
        budget_version=7,
        stage_total_tokens=total,
        committed_frontier_tokens=frontier,
        hbm_tokens=protected,
        host_tokens=0,
        recompute_tokens=total - protected,
        commit_logical_time=2.0 + pp_rank,
        storage_failure_domain=domain or f"node2/pp{pp_rank}",
        recovery_path=("direct_hbm_reuse",),
        copy_latency_s=0.001,
        expected_restore_latency_s=0.002 + pp_rank * 0.001,
        expected_recompute_latency_s=0.1,
        copy_bytes=frontier * 16,
        link_id=f"dp0.pp{pp_rank}->dp2.pp{pp_rank}",
        ack_sequence=pp_rank + 1,
    )


def test_dp4_pp4_launch_contract_enumerates_sixteen_independent_workers() -> None:
    workers = enumerate_pipeline_workers(dp_size=4, pp_size=4, tp_size=1)

    assert len(workers) == 16
    assert len({worker.stage for worker in workers}) == 16
    assert [worker.stage for worker in workers[:5]] == [
        StageId(0, 0),
        StageId(0, 1),
        StageId(0, 2),
        StageId(0, 3),
        StageId(1, 0),
    ]
    assert workers[-1].worker_index == 15
    assert workers[-1].stage == StageId(3, 3)


class _FakeProcess:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[object] = []

    def put(self, item: object) -> None:
        self.items.append(item)


def test_pipeline_process_group_broadcasts_by_dp_and_tracks_stage_liveness() -> None:
    specs = enumerate_pipeline_workers(dp_size=2, pp_size=2)
    queues = [_FakeQueue() for _ in specs]
    group = PipelineProcessGroup(
        dp_size=2,
        pp_size=2,
        tp_size=1,
        workers=tuple(
            PipelineWorkerHandle(
                spec=spec,
                process=_FakeProcess(alive=spec.worker_index != 2),
                command_queue=queue,
                ready_event=object(),
            )
            for spec, queue in zip(specs, queues)
        ),
    )

    group.broadcast_dp_command(1, ("budget", 7))

    assert group.process_count == 4
    assert queues[0].items == []
    assert queues[1].items == []
    assert queues[2].items == [("budget", 7)]
    assert queues[3].items == [("budget", 7)]
    assert group.leader_for_dp(1).spec.stage == StageId(1, 0)
    assert group.alive_stages() == (StageId(0, 0), StageId(0, 1), StageId(1, 1))


def test_dp2_pp2_real_process_group_launches_ready_and_exits_cleanly() -> None:
    server_args = ServerArgs(dp_size=2, pp_size=2, tp_size=1, log_level="error")
    port_args = PortArgs.init_new(server_args)
    registry = build_dp_pp_ipc_registry(port_args, dp_size=2, pp_size=2)
    group = None
    try:
        group, _stats_queue = launch_pipeline_scheduler_processes(
            server_args,
            port_args,
            registry,
            dp_size=2,
            pp_size=2,
            tp_size=1,
            ready_timeout_s=15.0,
        )
        assert group.process_count == 4
        assert group.alive_stages() == (
            StageId(0, 0),
            StageId(0, 1),
            StageId(1, 0),
            StageId(1, 1),
        )

        group.broadcast_command(("stop",))
        for process in group.processes:
            process.join(timeout=5.0)
        assert [process.exitcode for process in group.processes] == [0, 0, 0, 0]
    finally:
        if group is not None:
            for process in group.processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)


def test_pp4_request_crosses_every_stage_and_only_pp0_emits_client_output() -> None:
    server_args = ServerArgs(
        dp_size=1,
        pp_size=4,
        tp_size=1,
        log_level="error",
        scheduler_telemetry=True,
    )
    port_args = PortArgs.init_new(server_args)
    registry = build_dp_pp_ipc_registry(port_args, dp_size=1, pp_size=4)
    group = None
    try:
        group, stats_queue = launch_pipeline_scheduler_processes(
            server_args,
            port_args,
            registry,
            dp_size=1,
            pp_size=4,
            tp_size=1,
            ready_timeout_s=15.0,
        )
        request = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid="pp4-e2e",
                input_ids=list(range(8)),
                max_new_tokens=2,
            )
        )
        registry.get_queue(
            port_args.scheduler_input_ipc_name_for_dp(0)
        ).put(request)

        outputs: list[dict] = []
        detokenizer_queue = registry.get_queue(port_args.detokenizer_ipc_name)
        while True:
            output = detokenizer_queue.get(timeout=15.0)
            outputs.append(output)
            if output.get("op") == "finalize":
                break

        observed_stages: set[int] = set()
        observed_rows: dict[int, dict] = {}
        deadline_misses = 0
        while observed_stages != {0, 1, 2, 3} and deadline_misses < 2:
            try:
                event = stats_queue.get(timeout=2.0)
            except queue.Empty:
                deadline_misses += 1
                continue
            if event.get("type") != "scheduler_telemetry":
                continue
            row = event["row"]
            if "pp4-e2e" in row["request_ids"]:
                pp_rank = int(row["pp_rank"])
                observed_stages.add(pp_rank)
                observed_rows.setdefault(pp_rank, row)

        assert observed_stages == {0, 1, 2, 3}
        assert all(
            row["forward_s"] * 4
            == pytest.approx(row["pipeline_inference_s"])
            for row in observed_rows.values()
        )
        assert [output["rid"] for output in outputs] == ["pp4-e2e"] * len(outputs)
        assert sum(output.get("op") == "finalize" for output in outputs) == 1
        assert outputs[-1]["generated_tokens"] == 2
        assert all(process.is_alive() for process in group.processes)

        group.broadcast_command(("stop",))
        for process in group.processes:
            process.join(timeout=5.0)
        assert [process.exitcode for process in group.processes] == [0, 0, 0, 0]
    finally:
        if group is not None:
            for process in group.processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)


def test_engine_dp2_pp4_routes_to_both_replicas_and_stops_all_workers() -> None:
    engine = Engine(
        ServerArgs(dp_size=2, pp_size=4, tp_size=1, log_level="error")
    )
    group = engine._pipeline_process_group
    assert group is not None
    try:
        engine.start_background_scheduler()
        for index in range(4):
            engine.submit(
                GenerateReqInput(
                    rid=f"engine-dp2-pp4-{index}",
                    input_ids=list(range(8)),
                    max_new_tokens=1,
                )
            )
        result = engine.run_until_idle()

        assert group.process_count == 8
        assert len(engine._dp_controller.workers) == 2
        assert {
            stats.aegis_assigned_dp_rank
            for stats in result.request_stats.values()
        } == {0, 1}
        assert all(
            stats.status == "completed"
            for stats in result.request_stats.values()
        )
    finally:
        engine.shutdown()

    assert [process.exitcode for process in group.processes] == [0] * 8


def test_engine_pipeline_supersedes_frontier_on_every_stage(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "aegis-dp2-pp4.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "cache": {
                    "page_size": 1,
                    "l1_capacity_pages": 256,
                    "l2_capacity_pages": 512,
                    "enable_hierarchical_cache": True,
                    "hicache_write_policy": "write_through_selective",
                },
                "scheduler": {
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 4,
                    "max_running_requests": 4,
                    "max_prefill_tokens": 128,
                    "chunked_prefill_size": 128,
                    "load_balance_method": "cache_aware",
                    "disable_overlap_schedule": True,
                },
                "predictor": {
                    "name": "heuristic",
                    "forward_scale": 0.001,
                },
                "platform": {
                    "accelerator": {
                        "name": "a100_sxm",
                        "hbm_capacity_gb": 80,
                    }
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "initial_alpha": 0.20,
                    "initial_beta": 0.50,
                    "host_alignment_tokens": 1,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 4,
                        "gpu_backup_requires_direct_retry_route": True,
                        "host_pool_scope": "node_shared_process_external",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    # Engine publishes the config path for forked workers.  Track that process
    # environment mutation with pytest so later topology tests cannot inherit
    # this PP=4 fixture.
    monkeypatch.setenv("SGLSIM_CONFIG_PATH", str(config_path))
    engine = Engine(
        ServerArgs(
            model_path="mock-model",
            dp_size=2,
            pp_size=4,
            tp_size=1,
            sim_config_path=str(config_path),
            disable_overlap_schedule=True,
            log_level="error",
        )
    )
    group = engine._pipeline_process_group
    assert group is not None
    try:
        engine.start_background_scheduler()

        def request(rid: str, worker_seq: int, input_len: int):
            return GenerateReqInput(
                rid=rid,
                input_ids=list(range(input_len)),
                max_new_tokens=2,
                routing_key="workflow-stage-supersede/main",
                simulation={
                    "aegis_policy": "aegiskv_l1_l2",
                    "backup_kv": True,
                    "critical_agent": True,
                    "aegis_backup_key": rid,
                    "workflow_id": "workflow-stage-supersede",
                    "agent_id": "main",
                    "worker_seq": worker_seq,
                    "workflow_slack_ms": 1000.0,
                    "aegis_failure_scope": "node",
                },
            )

        old = request("pipeline-old-frontier", 1, 32)
        engine.submit(old)
        assert engine.wait_for_rid(old.rid, timeout_s=30).status == "completed"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            old_hint = next(
                (
                    hint
                    for hint in engine.aegis_status()["recovery_hints"]
                    if hint.get("backup_key") == old.rid
                ),
                None,
            )
            if old_hint and old_hint.get("all_stages_committed"):
                break
            time.sleep(0.01)
        assert old_hint is not None
        assert old_hint["all_stages_committed"] is True

        new = request("pipeline-new-frontier", 2, 48)
        engine.submit(new)
        assert engine.wait_for_rid(new.rid, timeout_s=30).status == "completed"
        deadline = time.monotonic() + 10.0
        aggregate = None
        status = None
        while time.monotonic() < deadline:
            status = engine.aegis_status()
            aggregate = next(
                (
                    event
                    for event in reversed(status["events"])
                    if event.get("kind")
                    == "backup_frontier_supersede_applied"
                    and event.get("backup_key") == old.rid
                ),
                None,
            )
            if aggregate is not None:
                break
            time.sleep(0.01)

        assert status is not None
        assert aggregate is not None
        assert aggregate["all_stages_applied"] is True
        assert aggregate["applied_stage_count"] == 4
        stage_events = [
            event
            for event in status["events"]
            if event.get("kind")
            == "backup_frontier_supersede_stage_applied"
            and event.get("backup_key") == old.rid
        ]
        assert sorted(event["pp_rank"] for event in stage_events) == [
            0,
            1,
            2,
            3,
        ]
        assert all(
            hint.get("backup_key") != old.rid
            for hint in status["recovery_hints"]
        )
        assert all(process.is_alive() for process in group.processes)
    finally:
        engine.shutdown()

    assert [process.exitcode for process in group.processes] == [0] * 8


def test_engine_hard_restart_replaces_entire_dp_pipeline_group() -> None:
    engine = Engine(
        ServerArgs(dp_size=2, pp_size=2, tp_size=1, log_level="error")
    )
    original_group = engine._pipeline_process_group
    assert original_group is not None
    original_dp0_pids = [
        worker.process.pid for worker in original_group.for_dp(0)
    ]
    original_dp1_pids = [
        worker.process.pid for worker in original_group.for_dp(1)
    ]
    try:
        failed = engine.hard_fail_dp_rank(1)
        assert failed["accepted"] is True
        assert failed["terminated_process_count"] == 2
        assert [row["pp_rank"] for row in failed["stage_exitcodes"]] == [0, 1]
        assert all(
            not worker.process.is_alive()
            for worker in original_group.for_dp(1)
        )
        assert all(
            worker.process.is_alive()
            for worker in original_group.for_dp(0)
        )

        restarted = engine.restart_hard_failed_dp_rank(1)
        replacement_group = engine._pipeline_process_group
        assert replacement_group is not None
        assert replacement_group is not original_group
        assert restarted["restarted"] is True
        assert restarted["restarted_process_count"] == 2
        assert [row["pp_rank"] for row in restarted["stage_pids"]] == [0, 1]
        assert [
            worker.process.pid for worker in replacement_group.for_dp(0)
        ] == original_dp0_pids
        replacement_dp1_pids = [
            worker.process.pid for worker in replacement_group.for_dp(1)
        ]
        assert replacement_dp1_pids != original_dp1_pids
        assert all(
            worker.process.is_alive()
            for worker in replacement_group.workers
        )
        assert engine.soft_failure_status()["hard_failed_ranks"] == []

        engine.start_background_scheduler()
        for index in range(2):
            engine.submit(
                GenerateReqInput(
                    rid=f"post-hard-restart-{index}",
                    input_ids=list(range(8)),
                    max_new_tokens=1,
                )
            )
        result = engine.run_until_idle()
        assert {
            stats.aegis_assigned_dp_rank
            for stats in result.request_stats.values()
        } == {0, 1}
        assert all(
            stats.status == "completed"
            for stats in result.request_stats.values()
        )
    finally:
        engine.shutdown()


def test_engine_soft_recovery_waits_for_entire_dp_pipeline_group() -> None:
    engine = Engine(
        ServerArgs(dp_size=2, pp_size=2, tp_size=1, log_level="error")
    )
    try:
        engine.start_background_scheduler()
        assert engine.soft_fail_dp_rank(1)["accepted"] is True

        recovered = engine.recover_soft_failed_dp_rank(1)

        clear = recovered["l1_clear"]
        assert recovered["released"] is True
        assert clear["all_stages_cleared"] is True
        assert clear["cleared_stage_count"] == 2
        assert clear["required_stage_count"] == 2
        assert [row["pp_rank"] for row in clear["stage_results"]] == [0, 1]
        assert engine.soft_failure_status()["soft_failed_ranks"] == []
    finally:
        engine.shutdown()


def test_dp_pp_registry_keeps_public_ingress_at_pp0_and_isolates_stage_channels() -> None:
    port_args = PortArgs.init_new(ServerArgs(dp_size=2, pp_size=4))
    registry = build_dp_pp_ipc_registry(port_args, dp_size=2, pp_size=4)

    assert port_args.scheduler_input_ipc_name_for_stage(1, 0) == (
        port_args.scheduler_input_ipc_name_for_dp(1)
    )
    assert port_args.scheduler_input_ipc_name_for_stage(1, 1) != (
        port_args.scheduler_input_ipc_name_for_stage(1, 2)
    )
    expected_per_dp = 4 * 3 + 1
    assert len(registry.endpoints()) == 2 * expected_per_dp + 2
    assert len(set(registry.endpoints())) == len(registry.endpoints())


def test_dp_pp_registry_reset_starts_an_empty_pipeline_generation() -> None:
    port_args = PortArgs.init_new(ServerArgs(dp_size=2, pp_size=2))
    registry = build_dp_pp_ipc_registry(port_args, dp_size=2, pp_size=2)
    endpoint = port_args.pp_output_ipc_name_for_stage(1, 0)
    old_queue = registry.get_queue(endpoint)
    assert old_queue is not None
    old_queue.put({"stale_generation": 1})

    new_queue = registry.reset(endpoint)

    assert new_queue is registry.get_queue(endpoint)
    assert new_queue is not old_queue
    with pytest.raises(queue.Empty):
        new_queue.get(timeout=0.05)


def test_scheduler_has_explicit_stage_identity_and_only_pp0_emits_outputs() -> None:
    ConfigManager._raw = {
        "scheduler": {"dp_size": 2, "pp_size": 4},
        "cache": {"l1_capacity_pages": 128, "page_size": 1},
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    ConfigManager._hicache = None
    server_args = ServerArgs(dp_size=2, pp_size=4)
    port_args = PortArgs.init_new(server_args)

    first = Scheduler(server_args, port_args, dp_rank=1, pp_rank=0)
    later = Scheduler(server_args, port_args, dp_rank=1, pp_rank=3)

    assert first.stage_id == StageId(1, 0)
    assert later.stage_id == StageId(1, 3)
    assert first._emit_client_outputs is True
    assert later._emit_client_outputs is False
    assert first.recv_from_tokenizer.endpoint == (
        port_args.scheduler_input_ipc_name_for_dp(1)
    )
    assert later.recv_from_tokenizer.endpoint == (
        port_args.scheduler_input_ipc_name_for_stage(1, 3)
    )


def test_scheduler_rejects_stage_identity_outside_declared_topology() -> None:
    ConfigManager._raw = {"scheduler": {"dp_size": 1, "pp_size": 4}}
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    ConfigManager._hicache = None
    server_args = ServerArgs(dp_size=1, pp_size=4)

    with pytest.raises(ValueError, match="smaller than pp_size"):
        Scheduler(
            server_args,
            PortArgs.init_new(server_args),
            dp_rank=0,
            pp_rank=4,
        )


def test_route_and_budget_are_immutable_and_owned_by_one_dp() -> None:
    route = _route()
    budget = _budget()

    with pytest.raises(FrozenInstanceError):
        route.backup_dp_rank = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        budget.alpha = 0.9  # type: ignore[misc]
    assert [budget.hbm_limit_for_stage(pp) for pp in range(4)] == [
        100,
        110,
        120,
        130,
    ]
    assert budget.dp_rank == route.backup_dp_rank


def test_stage_placement_rejects_cross_stage_kv_and_wrong_immutable_route() -> None:
    with pytest.raises(ValueError, match="same PP stage"):
        StagePlacementDecision(
            request_id="r0",
            source_stage=StageId(0, 1),
            backup_stage=StageId(2, 2),
            route_version=3,
            budget_version=7,
            chunk_id=0,
            token_start=0,
            token_end=64,
            tier="HBM",
            failure_domain="node2/pp2",
            utility=2.0,
            copy_link="invalid",
            planned_commit_deadline=5.0,
        )

    placement = StagePlacementDecision(
        request_id="r0",
        source_stage=StageId(0, 1),
        backup_stage=StageId(2, 1),
        route_version=3,
        budget_version=7,
        chunk_id=0,
        token_start=0,
        token_end=64,
        tier="HOST",
        failure_domain="node2",
        utility=2.0,
        copy_link="dp0-node0->dp2-node2",
        planned_commit_deadline=5.0,
    )
    placement.validate_against(_route(), _budget())
    with pytest.raises(ValueError, match="immutable route"):
        placement.validate_against(
            RoutingDecision(
                request_id="r0",
                workflow_id="w0",
                execution_dp_rank=0,
                backup_dp_rank=3,
                route_version=3,
                decided_logical_time=1.25,
            ),
            _budget(),
        )


def test_pp_batch_identity_excludes_stage_rank_but_covers_lifecycle_state() -> None:
    common = dict(
        microbatch_id=1,
        batch_id=8,
        request_ids=("r0", "r1"),
        forward_mode="DECODE",
        token_frontiers=(33, 65),
        route_versions=(3, 1),
        budget_version=7,
        causal_sequence=12,
    )
    stage0 = PPStageBatchEnvelope(stage=StageId(0, 0), **common)
    stage3 = PPStageBatchEnvelope(stage=StageId(0, 3), **common)

    assert stage0.identity_tuple() == stage3.identity_tuple()
    with pytest.raises(ValueError, match="must align"):
        PPStageBatchEnvelope(
            stage=StageId(0, 1),
            token_frontiers=(33,),
            **{key: value for key, value in common.items() if key != "token_frontiers"},
        )


def test_pp_batch_identity_preserves_running_decode_batch_sentinel() -> None:
    envelope = PPStageBatchEnvelope(
        stage=StageId(0, 0),
        microbatch_id=0,
        batch_id=-1,
        request_ids=("r0",),
        forward_mode="DECODE",
        token_frontiers=(32,),
        route_versions=(1,),
        budget_version=1,
        causal_sequence=0,
    )

    assert envelope.batch_id == -1
    with pytest.raises(ValueError, match="at least -1"):
        PPStageBatchEnvelope(
            stage=StageId(0, 0),
            microbatch_id=0,
            batch_id=-2,
            request_ids=(),
            forward_mode="DECODE",
            token_frontiers=(),
            route_versions=(),
            budget_version=1,
            causal_sequence=0,
        )


def test_batch_identity_detects_cross_stage_lifecycle_divergence() -> None:
    common = dict(
        microbatch_id=0,
        batch_id=3,
        request_ids=("r0",),
        forward_mode="PREFILL",
        token_frontiers=(64,),
        route_versions=(2,),
        budget_version=4,
        causal_sequence=9,
    )
    upstream = PPStageBatchEnvelope(stage=StageId(0, 0), **common)
    matching = PPStageBatchEnvelope(stage=StageId(0, 1), **common)
    divergent = PPStageBatchEnvelope(
        stage=StageId(0, 1),
        token_frontiers=(63,),
        **{key: value for key, value in common.items() if key != "token_frontiers"},
    )

    assert_batch_identity(upstream, matching)
    with pytest.raises(PipelineBatchIdentityError, match="lifecycle mismatch"):
        assert_batch_identity(upstream, divergent)


def test_request_and_proxy_envelopes_follow_adjacent_pp_stages() -> None:
    request = PPRequestEnvelope(
        source_stage=StageId(1, 0),
        target_stage=StageId(1, 1),
        step_sequence=5,
        request_ids=("r0",),
        request_payloads=({"rid": "r0"},),
    )
    at_pp2 = request.forward_from(StageId(1, 1))
    at_pp3 = at_pp2.forward_from(StageId(1, 2))
    batch = PPStageBatchEnvelope(
        stage=StageId(1, 2),
        microbatch_id=0,
        batch_id=4,
        request_ids=("r0",),
        forward_mode="DECODE",
        token_frontiers=(65,),
        route_versions=(2,),
        budget_version=4,
        causal_sequence=5,
    )
    proxy = PPProxyEnvelope(
        source_stage=StageId(1, 2),
        target_stage=StageId(1, 3),
        step_sequence=5,
        batch=batch,
    )

    assert at_pp3.target_stage == StageId(1, 3)
    assert at_pp3.request_ids == ("r0",)
    assert proxy.batch is batch
    with pytest.raises(ValueError, match="adjacent"):
        PPRequestEnvelope(
            source_stage=StageId(1, 0),
            target_stage=StageId(1, 2),
            step_sequence=5,
            request_ids=(),
            request_payloads=(),
        )


def test_output_envelope_visits_last_pp0_pp1_pp2_last_exactly_once() -> None:
    last_batch = PPStageBatchEnvelope(
        stage=StageId(0, 3),
        microbatch_id=0,
        batch_id=7,
        request_ids=("r0",),
        forward_mode="DECODE",
        token_frontiers=(70,),
        route_versions=(1,),
        budget_version=2,
        causal_sequence=11,
    )
    envelope = PPOutputEnvelope.from_last_stage(
        last_stage=StageId(0, 3),
        pp_size=4,
        step_sequence=11,
        batch=last_batch,
        result_logical_time=4.5,
    )
    targets = []
    while envelope is not None:
        target = envelope.target_stage
        targets.append(target.pp_rank)
        envelope = envelope.consume_and_forward(
            current_stage=target,
            pp_size=4,
        )

    assert targets == [0, 1, 2, 3]


def test_pp_output_is_applied_only_at_pinned_sglang_microbatch_frontier() -> None:
    """A later PP stage must not expose a returned radix prefix early."""

    local_batch = PPStageBatchEnvelope(
        stage=StageId(0, 0),
        microbatch_id=0,
        batch_id=7,
        request_ids=("restore-r0",),
        forward_mode="EXTEND",
        token_frontiers=(63,),
        route_versions=(2,),
        budget_version=3,
        causal_sequence=0,
    )
    last_batch = PPStageBatchEnvelope(
        stage=StageId(0, 3),
        microbatch_id=0,
        batch_id=7,
        request_ids=("restore-r0",),
        forward_mode="EXTEND",
        token_frontiers=(63,),
        route_versions=(2,),
        budget_version=3,
        causal_sequence=0,
    )
    output = PPOutputEnvelope.from_last_stage(
        last_stage=StageId(0, 3),
        pp_size=4,
        step_sequence=0,
        batch=last_batch,
        result_logical_time=1.0,
    )

    class _Recv:
        def __init__(self, values):
            self.values = list(values)

        def recv_pyobj(self, *, flags=0):
            del flags
            if not self.values:
                raise queue.Empty
            return self.values.pop(0)

    class _Send:
        def __init__(self):
            self.values = []

        def send_pyobj(self, value):
            self.values.append(value)

    class _Stage:
        pass

    stage = _Stage()
    stage.stage_id = StageId(0, 0)
    stage.server_args = SimpleNamespace(pp_size=4)
    stage._pp_loop_size = 4
    stage._pp_step_sequence = 1
    stage._pp_last_output_sequence = -1
    stage._pp_last_received_output_sequence = -1
    stage._pp_received_outputs = {}
    stage._pp_pending_batches = {
        0: SimpleNamespace(envelope=local_batch),
    }
    stage._pp_recv_output = _Recv([output])
    stage._pp_send_output = _Send()
    applied = []
    stage._pp_apply_batch_result = applied.append

    # Sequence 0 has returned, but q=0 may commit no older slot for PP=4.
    Scheduler._pp_pump_outputs(stage, None, None)
    assert applied == []
    assert set(stage._pp_received_outputs) == {0}
    assert stage._pp_last_output_sequence == -1

    # After launching q=3, pinned SGLang commits sequence q-(PP-1)=0.
    stage._pp_step_sequence = 4
    Scheduler._pp_pump_outputs(stage, None, None)
    assert len(applied) == 1
    assert stage._pp_received_outputs == {}
    assert stage._pp_last_output_sequence == 0
    assert len(stage._pp_send_output.values) == 1


def test_recovery_hint_uses_minimum_of_all_four_committed_frontiers() -> None:
    hint = RecoveryHint.from_stage_acks(
        request_id="r0",
        route_version=3,
        budget_version=7,
        pp_size=4,
        requested_tokens=128,
        stage_acks=[
            _ack(0, 128),
            _ack(1, 96),
            _ack(2, 112),
            _ack(3, 64),
        ],
    )

    assert hint.safe_frontier_tokens == 64
    assert hint.bottleneck_pp_rank == 3
    assert hint.restored_tokens_by_stage == (128, 96, 112, 64)
    assert hint.recomputed_tokens_by_stage == (0, 32, 16, 64)
    assert hint.total_restore_latency_s == pytest.approx(0.005)
    serialized = hint.to_dict()
    assert StageCommitAck.from_dict(serialized["stage_acks"][3]) == _ack(3, 64)


def test_missing_or_failed_stage_forces_zero_recoverable_prefix() -> None:
    hint = RecoveryHint.from_stage_acks(
        request_id="r0",
        route_version=3,
        budget_version=7,
        pp_size=4,
        requested_tokens=128,
        stage_acks=[
            _ack(0, 128),
            _ack(1, 128, domain="failed-stage"),
            _ack(2, 128),
        ],
        failed_domains=("failed-stage",),
    )

    assert hint.safe_frontier_tokens == 0
    assert hint.bottleneck_pp_rank == 1
    assert hint.stage_acks[1] is None
    assert hint.stage_acks[3] is None
    assert hint.recomputed_tokens_by_stage == (0, 128, 0, 128)
    assert hint.recovery_path_by_stage[1] == ("recompute",)


def test_ack_rejects_frontier_larger_than_protected_tokens() -> None:
    with pytest.raises(ValueError, match="frontier"):
        StageCommitAck(
            request_id="r0",
            source_stage=StageId(0, 0),
            backup_stage=StageId(2, 0),
            route_version=3,
            budget_version=7,
            stage_total_tokens=128,
            committed_frontier_tokens=65,
            hbm_tokens=32,
            host_tokens=32,
            recompute_tokens=64,
            commit_logical_time=1.0,
            storage_failure_domain="node2/pp0",
            recovery_path=("direct_hbm_reuse",),
            copy_latency_s=0.001,
            expected_restore_latency_s=0.001,
            expected_recompute_latency_s=0.1,
            copy_bytes=1024,
            link_id="link",
            ack_sequence=1,
        )


def test_recovery_hint_rejects_stale_stage_ack_version() -> None:
    stale = _ack(0, 128)
    object.__setattr__(stale, "budget_version", 6)

    with pytest.raises(ValueError, match="budget version"):
        RecoveryHint.from_stage_acks(
            request_id="r0",
            route_version=3,
            budget_version=7,
            pp_size=4,
            requested_tokens=128,
            stage_acks=[stale],
        )
