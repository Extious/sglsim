from __future__ import annotations

import json

from sglsim.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster
from sglsim.srt.simulation.calibration_metrics import build_offline_cluster_artifact
from sglsim.srt.simulation.aegis_exact_contract import (
    CertifiedDPBudgetPlan,
    CertifiedExactWindowPlan,
    CertifiedRoutePlan,
    CertifiedStageChunkPlacement,
)


def _request(
    rid: str,
    *,
    dp_rank: int,
    input_tokens: int,
    output_tokens: int,
    backup_dp_rank: int | None = None,
):
    simulation = {
        "server_created_time": 0.0,
        "force_dp_rank": dp_rank,
    }
    if backup_dp_rank is not None:
        simulation.update(
            {
                "aegis_policy": "aegiskv_l1_l2",
                "backup_kv": True,
                "critical_agent": True,
                "aegis_critical_probability": 0.9,
                "workflow_slack_ms": 0.0,
                "aegis_backup_dp_rank": backup_dp_rank,
                "aegis_failure_scope": "gpu",
            }
        )
    return TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid=rid,
            input_ids=list(range(input_tokens)),
            max_new_tokens=output_tokens,
            simulation=simulation,
        )
    )


def test_offline_cluster_runs_two_real_scheduler_processes(tmp_path) -> None:
    config_path = tmp_path / "offline-smoke.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "pp_size": 1,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 1024,
                    "chunked_prefill_size": 1024,
                    "max_running_requests": 8,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 0,
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "initial_alpha": 0.5,
                    "initial_beta": 0.5,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        assert cluster.dispatch(
            _request(
                "rank0-r0",
                dp_rank=0,
                input_tokens=16,
                output_tokens=2,
                backup_dp_rank=1,
            )
        ) == 0
        assert cluster.dispatch(
            _request("rank1-r0", dp_rank=1, input_tokens=32, output_tokens=3)
        ) == 1
        cluster.schedule_soft_failure(
            event_id="smoke-failure",
            dp_rank=0,
            offset_s=0.0,
            duration_s=0.001,
        )
        result = cluster.run_until_complete(
            expected_request_ids={"rank0-r0", "rank1-r0"},
            max_events=100,
        )
        status = cluster.dp_controller.aegis_status()
        failure_events = list(cluster.failure_events)
        artifact = build_offline_cluster_artifact(
            result,
            identity={"seed": 7, "trace": "unit-smoke"},
        )

    finished = {
        str(message["rid"])
        for message in result.detokenizer_messages
        if bool(message.get("finished", False))
    }
    assert finished == {"rank0-r0", "rank1-r0"}
    final_messages = {
        str(message["rid"]): message
        for message in result.detokenizer_messages
        if bool(message.get("finished", False))
    }
    assert final_messages["rank0-r0"]["router_committed_output_position"] == 2
    assert final_messages["rank0-r0"]["router_published_output_position"] == 2
    assert final_messages["rank1-r0"]["router_committed_output_position"] == 3
    assert final_messages["rank1-r0"]["router_published_output_position"] == 3
    assert result.router_output_ledger["audit"]["status"] == "PASS"
    assert all(
        row["state"] == "completed"
        for row in result.router_output_ledger["logical_requests"]
    )
    assert {row["dp_rank"] for row in result.telemetry} == {0, 1}
    assert result.rank_snapshots[0]["request_stats"]["rank0-r0"]["status"] == (
        "completed"
    )
    assert result.rank_snapshots[1]["request_stats"]["rank1-r0"]["status"] == (
        "completed"
    )
    assert all(
        row.get("cpu_overflow_s", 0.0) == 0.0 for row in result.telemetry
    )
    hint = next(
        row for row in status["recovery_hints"] if row["backup_key"] == "rank0-r0"
    )
    assert hint["ack"] is True
    assert hint["target_dp_rank"] == 1
    assert len(failure_events) == 1
    assert failure_events[0]["event_id"] == "smoke-failure"
    assert failure_events[0]["actual_outage_s"] >= 0.001
    metrics = artifact["calibration_metrics"]
    assert metrics["request_count"] == 2
    assert metrics["status_counts"] == {"completed": 2}
    assert metrics["routes"] == {"rank0-r0": 0, "rank1-r0": 1}
    assert metrics["failure_event_order"] == ["smoke-failure"]
    assert set(metrics["pipeline_accounting_by_dp"]) == {"0", "1"}


def test_offline_kv_resource_telemetry_tracks_alloc_free_and_dp_isolation(
    tmp_path,
) -> None:
    config_path = tmp_path / "offline-kv-resource-snapshot.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "tp_size": 4,
                    "pp_size": 1,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 256,
                    "chunked_prefill_size": 256,
                    "max_running_requests": 8,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 128,
                    "l2_capacity_pages": 64,
                },
                "communication": {"kv_bytes_per_token": 128},
                "aegis_l1_l2": {
                    "enabled": True,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        tp_size=4,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        poll = cluster._call(0, 0, "poll", {"at_s": 0.0})
        assert poll["kv_resource_snapshot"]["l1_device"]["used_pages"] == 0

        cluster.dispatch(
            _request("kv-rank0", dp_rank=0, input_tokens=32, output_tokens=2)
        )
        cluster.dispatch(
            _request("kv-rank1", dp_rank=1, input_tokens=32, output_tokens=2)
        )
        # The rank-0 recovery path clears only that scheduler's volatile L1.
        # Rank 1 is the control proving that rank-local pools stay isolated.
        cluster.schedule_soft_failure(
            event_id="kv-rank0-clear",
            dp_rank=0,
            offset_s=0.0,
            duration_s=0.001,
        )
        result = cluster.run_until_complete(
            expected_request_ids={"kv-rank0", "kv-rank1"},
            max_events=200,
        )

    assert result.telemetry
    for row in result.telemetry:
        before = row["kv_resource_before_prepare"]
        after = row["kv_resource_after_prepare"]
        completed = row["kv_resource_at_completion"]
        for snapshot in (before, after, completed):
            assert snapshot["schema"] == (
                "sglsim.offline.rank-kv-resource.v1"
            )
            assert snapshot["dp_rank"] == row["dp_rank"]
            assert snapshot["pp_rank"] == 0
            assert snapshot["scheduler_tp_rank"] == 0
            assert snapshot["logical_tp_group_size"] == 4
            assert snapshot["scope"] == "one-symmetric-tp-rank-view"
            assert snapshot["page_size_tokens"] == 16
            assert snapshot["kv_bytes_per_token_per_tp_rank"] == 128
            for pool_name in ("l1_device", "l2_host"):
                pool = snapshot[pool_name]
                assert pool["capacity_pages"] == (
                    pool["used_pages"] + pool["free_pages"]
                )
                assert pool["capacity_tokens"] == (
                    pool["capacity_pages"] * 16
                )
                assert pool["used_bytes"] == pool["used_tokens"] * 128
                assert list(pool["owner_pages"]) == sorted(
                    pool["owner_pages"]
                )

    for rank in (0, 1):
        rank_rows = [row for row in result.telemetry if row["dp_rank"] == rank]
        assert any(
            row["kv_resource_after_prepare"]["l1_device"]["used_pages"]
            > row["kv_resource_before_prepare"]["l1_device"]["used_pages"]
            for row in rank_rows
        )

    rank0_final = result.rank_snapshots[0]["kv_resource_snapshot"]
    rank1_final = result.rank_snapshots[1]["kv_resource_snapshot"]
    assert rank0_final["l1_device"]["used_pages"] == 0
    assert rank0_final["l1_device"]["owner_pages"] == {}
    assert rank0_final["l2_host"]["used_pages"] > 0
    assert rank1_final["l1_device"]["used_pages"] > 0
    assert rank1_final["l1_device"]["owner_pages"] == {"kv-rank1": 2}
    # Per-rank bytes are not multiplied by the logical TP group size.
    assert rank1_final["l1_device"]["used_bytes"] == (
        rank1_final["l1_device"]["used_tokens"] * 128
    )


def test_offline_cluster_reuses_online_retry_route_and_restore_hint(tmp_path) -> None:
    config_path = tmp_path / "offline-retry-smoke.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "pp_size": 1,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 1024,
                    "chunked_prefill_size": 1024,
                    "max_running_requests": 8,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 0,
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "initial_alpha": 0.5,
                    "initial_beta": 0.5,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        original = _request(
            "retry-source",
            dp_rank=0,
            input_tokens=32,
            output_tokens=2,
            backup_dp_rank=1,
        )
        cluster.dispatch(original)
        cluster.run_until_complete(expected_request_ids={"retry-source"})
        hint = cluster.dp_controller.recovery_hint("retry-source")
        assert hint is not None and int(hint["l1_tokens"]) > 0

        retry = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid="retry-source-retry",
                    input_ids=list(range(32)),
                max_new_tokens=2,
                simulation={
                    "server_created_time": cluster.coordinator.now_s,
                    "aegis_policy": "aegiskv_l1_l2",
                    "backup_kv": False,
                    "aegis_backup_key": "retry-source",
                    "failover_replay": True,
                    "soft_failure_injected": True,
                    "failed_dp_rank": 0,
                        "pre_failure_total_tokens": 34,
                    "aegis_route_plan_retry_dp_rank": 1,
                },
            )
        )
        failure_at = cluster.coordinator.now_s
        cluster.schedule_soft_failure(
            event_id="retry-failure",
            dp_rank=0,
            offset_s=failure_at,
            duration_s=0.001,
        )
        cluster.schedule_request_arrival(retry, at_s=failure_at)
        result = cluster.run_until_complete(
            expected_request_ids={"retry-source-retry"},
            max_events=100,
        )
        retry_simulation = retry.sampling_params.custom_params["simulation"]

    final = next(
        message
        for message in result.detokenizer_messages
        if message.get("rid") == "retry-source-retry"
        and bool(message.get("finished", False))
    )
    assert final["aegis_assigned_dp_rank"] == 1
    assert final["aegis_cross_rank_retry"] is True
    assert retry_simulation["restore_action"] == "l1_backup"
    assert retry_simulation["restore_tokens"] == 34
    assert final["aegis_replay_restore_action"] == "l1_backup"
    assert final["aegis_replay_restored_tokens"] > 0


def test_offline_request_handoff_precedes_unrelated_future_failure(tmp_path) -> None:
    """A feeder-thread race must not turn one-rank failure into global idle."""

    config_path = tmp_path / "offline-independent-clock.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "pp_size": 1,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 1024,
                    "chunked_prefill_size": 1024,
                    "max_running_requests": 8,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 0,
                },
                "aegis_l1_l2": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        cluster.schedule_request_arrival(
            _request(
                "rank0-before-future-rank1-failure",
                dp_rank=0,
                input_tokens=16,
                output_tokens=2,
            ),
            at_s=0.0,
        )
        cluster.schedule_soft_failure(
            event_id="future-rank1-failure",
            dp_rank=1,
            offset_s=10.0,
            duration_s=1.0,
        )
        result = cluster.run_until_complete(
            expected_request_ids={"rank0-before-future-rank1-failure"},
            max_events=200,
        )

    stats = result.rank_snapshots[0]["request_stats"][
        "rank0-before-future-rank1-failure"
    ]
    assert stats["queue_start"] == 0.0
    assert stats["last_event_time"] < 10.0


def test_offline_cluster_runs_physical_dp2_pp2_pipeline(tmp_path) -> None:
    config_path = tmp_path / "offline-dp2-pp2.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 2,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 1024,
                    "chunked_prefill_size": 1024,
                    "max_running_requests": 8,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 4096,
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "initial_alpha": 0.5,
                    "initial_beta": 0.5,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        assert cluster.pp_size == 2
        assert len(cluster.processes) == 4
        assert cluster.pipeline_process_group.process_count == 4
        assert cluster.dispatch(
            _request(
                "physical-pp-r0",
                dp_rank=0,
                input_tokens=32,
                output_tokens=2,
                backup_dp_rank=1,
            )
        ) == 0
        cluster.schedule_soft_failure(
            event_id="physical-pp-failure",
            dp_rank=0,
            offset_s=0.0,
            duration_s=0.001,
        )
        result = cluster.run_until_complete(
            expected_request_ids={"physical-pp-r0"},
            max_events=1000,
        )

    finished = [
        message
        for message in result.detokenizer_messages
        if message.get("rid") == "physical-pp-r0"
        and bool(message.get("finished", False))
    ]
    assert len(finished) == 1
    assert {
        (int(row["dp_rank"]), int(row["pp_rank"]))
        for row in result.telemetry
        if "physical-pp-r0" in row.get("request_ids", ())
    } == {(0, 0), (0, 1)}

    stages = result.rank_snapshots[0]["stages"]
    assert set(stages) == {"0", "1"}
    assert result.rank_snapshots[0]["process_count"] == 2
    for pp_rank, snapshot in stages.items():
        assert snapshot["pp_rank"] == int(pp_rank)
        assert snapshot["request_stats"]["physical-pp-r0"]["status"] == (
            "completed"
        )
        assert snapshot["aegis_rank_state"]["budget_version"] == 1
        assert snapshot["pp_state"]["pending_sequences"] == []

    hint = next(
        row
        for row in result.aegis_status["recovery_hints"]
        if row["backup_key"] == "physical-pp-r0"
    )
    assert hint["ack"] is True
    assert hint["all_stages_committed"] is True
    assert {
        int(row["backup_stage"]["pp_rank"])
        for row in hint["stage_acks"]
    } == {0, 1}
    assert len(result.failure_events) == 1
    clear = result.failure_events[0]["l1_clear"]
    assert clear["all_stages_cleared"] is True
    assert clear["cleared_stage_count"] == 2
    assert {
        int(row["pp_rank"]) for row in clear["stage_results"]
    } == {0, 1}


def test_offline_pp4_restore_and_shared_prefix_successor_remain_stage_aligned(
    tmp_path,
) -> None:
    """A returned retry prefix may not become visible at only later PP stages."""

    config_path = tmp_path / "offline-restore-successor-dp2-pp4.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 4,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 1024,
                    "chunked_prefill_size": 1024,
                    "max_running_requests": 8,
                },
                "cache": {
                    "page_size": 1,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 4096,
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "initial_alpha": 0.5,
                    "initial_beta": 0.5,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 4,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        source = _request(
            "restore-lineage-r0",
            dp_rank=0,
            input_tokens=32,
            output_tokens=2,
            backup_dp_rank=1,
        )
        cluster.dispatch(source)
        cluster.run_until_complete(expected_request_ids={source.rid}, max_events=2000)
        hint = cluster.dp_controller.recovery_hint(source.rid)
        assert hint is not None and hint["all_stages_committed"] is True

        retry = TokenizedGenerateReqInput.from_generate(
            GenerateReqInput(
                rid="restore-lineage-r0-retry",
                input_ids=list(range(32)),
                max_new_tokens=2,
                simulation={
                    "server_created_time": cluster.coordinator.now_s,
                    "aegis_policy": "aegiskv_l1_l2",
                    "backup_kv": False,
                    "aegis_backup_key": source.rid,
                    "failover_replay": True,
                    "soft_failure_injected": True,
                    "failed_dp_rank": 0,
                    "pre_failure_total_tokens": 34,
                    "aegis_route_plan_retry_dp_rank": 1,
                },
            )
        )
        # The successor has the same first 32 token ids and reaches the retry
        # DP at the same logical timestamp.  Before output-frontier ordering,
        # PP3 observed the restored prefix before PP2 and the strict lifecycle
        # identity check failed.
        successor = _request(
            "restore-lineage-r1",
            dp_rank=1,
            input_tokens=40,
            output_tokens=2,
        )
        failure_at = cluster.coordinator.now_s
        cluster.schedule_soft_failure(
            event_id="restore-lineage-failure",
            dp_rank=0,
            offset_s=failure_at,
            duration_s=0.001,
        )
        cluster.schedule_request_arrival(retry, at_s=failure_at)
        cluster.schedule_request_arrival(successor, at_s=failure_at)
        result = cluster.run_until_complete(
            expected_request_ids={retry.rid, successor.rid},
            max_events=4000,
        )

    stages = result.rank_snapshots[1]["stages"]
    for snapshot in stages.values():
        assert snapshot["request_stats"][retry.rid]["status"] == "completed"
        assert snapshot["request_stats"][successor.rid]["status"] == "completed"
        assert snapshot["pp_state"]["pending_sequences"] == []
    assert len(
        {
            int(snapshot["request_stats"][successor.rid]["cache_hit_tokens"])
            for snapshot in stages.values()
        }
    ) == 1


def test_offline_physical_pp_executes_certified_exact_stage_plan(tmp_path) -> None:
    config_path = tmp_path / "offline-exact-dp2-pp2.json"
    config_path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 2,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 1024,
                    "chunked_prefill_size": 1024,
                    "max_running_requests": 8,
                },
                "cache": {
                    "page_size": 16,
                    "l1_capacity_pages": 4096,
                    "l2_capacity_pages": 4096,
                },
                "aegis_l1_l2": {
                    "enabled": True,
                    "initial_alpha": 0.5,
                    "initial_beta": 0.5,
                    "topology": {
                        "dp_rank_to_node": [0, 1],
                        "pp_stages_per_node": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    plan = CertifiedExactWindowPlan(
        instance_id="offline-unit-instance",
        instance_sha256="instance-sha",
        decision_sha256="decision-sha",
        solver="unit-exact-enumerator",
        solver_version="1",
        solver_status="OPTIMAL",
        objective=1.0,
        dual_bound=1.0,
        mip_gap=0.0,
        solve_time_s=0.001,
        numerical_tolerance=1e-9,
        validation_errors=(),
        dp_size=2,
        pp_size=2,
        window_id=1,
        route_version=2,
        valid_from_logical_time=0.0,
        routes=(
            CertifiedRoutePlan(
                request_id="exact-physical-pp-r0",
                workflow_id="wf-exact",
                execution_dp_rank=0,
                backup_dp_rank=1,
            ),
        ),
        budgets=tuple(
            CertifiedDPBudgetPlan(
                dp_rank=dp_rank,
                hbm_budget_tokens_per_stage=32768,
                host_budget_tokens=32768,
                alpha=0.5,
                beta=0.5,
            )
            for dp_rank in range(2)
        ),
        placements=tuple(
            CertifiedStageChunkPlacement(
                request_id="exact-physical-pp-r0",
                chunk_id=0,
                pp_rank=pp_rank,
                backup_dp_rank=1,
                tier="HBM",
                token_count=34,
            )
            for pp_rank in range(2)
        ),
    )
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        streaming_arrival=True,
        sim_config_path=str(config_path),
    )

    with OfflineSchedulerCluster(server_args) as cluster:
        # Apply and ACK bootstrap v1 before atomically publishing exact v2.
        cluster.pump_ready_stages()
        installed = cluster.dp_controller.install_certified_exact_window(plan)
        assert installed["budget_versions"] == {0: 2, 1: 2}
        cluster.pump_ready_stages()
        cluster.dispatch(
            _request(
                "exact-physical-pp-r0",
                dp_rank=0,
                input_tokens=32,
                output_tokens=2,
                backup_dp_rank=1,
            )
        )
        result = cluster.run_until_complete(
            expected_request_ids={"exact-physical-pp-r0"},
            max_events=1000,
        )

    exact = result.aegis_status["certified_exact_window"]
    assert exact["certified_optimal"] is True
    assert exact["decision_sha256"] == "decision-sha"
    assert result.aegis_status["exact_deployment_violations"] == []
    hint = next(
        row
        for row in result.aegis_status["recovery_hints"]
        if row["backup_key"] == "exact-physical-pp-r0"
    )
    assert hint["all_stages_committed"] is True
    assert hint["budget_version"] == 2
    assert all(
        payload["placement_mode"] == "certified_exact"
        and payload["l1_tokens"] == 34
        and payload["l2_tokens"] == 0
        and payload["recompute_tokens"] == 0
        and payload["solver_decision_sha256"] == "decision-sha"
        for payload in hint["stage_payloads"]
    )
