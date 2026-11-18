from __future__ import annotations

import json

from sglsim.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.global_des import PhysicalGpuId
from sglsim.srt.simulation.offline_cluster import OfflineSchedulerCluster
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    PhysicalGpuSpec,
)


def _config(tmp_path) -> str:
    path = tmp_path / "offline-hard-failstop.json"
    path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": 2,
                    "tp_size": 1,
                    "pp_size": 1,
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
                    "enabled": False,
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
                gpu=PhysicalGpuId(
                    node_id=f"node{rank}",
                    local_gpu_id=0,
                    dp_rank=rank,
                    pp_rank=0,
                    tp_rank=0,
                ),
                hbm_capacity_bytes=1 << 30,
                weights_bytes=1 << 20,
                runtime_bytes=1 << 20,
            )
            for rank in range(2)
        ),
        nodes=(
            NodeDramSpec("node0", 1 << 30, runtime_bytes=1 << 20),
            NodeDramSpec("node1", 1 << 30, runtime_bytes=1 << 20),
        ),
    )


def _request(rid: str, dp_rank: int) -> TokenizedGenerateReqInput:
    return TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid=rid,
            input_ids=list(range(64)),
            max_new_tokens=4,
            simulation={
                "server_created_time": 0.0,
                "force_dp_rank": int(dp_rank),
            },
        )
    )


def _cluster(tmp_path) -> OfflineSchedulerCluster:
    return OfflineSchedulerCluster(
        ServerArgs(
            model_path="mock-model",
            dp_size=2,
            streaming_arrival=True,
            sim_config_path=_config(tmp_path),
        ),
        physical_topology=_topology(),
    )


def test_hard_failstop_cancels_compute_restarts_empty_and_preserves_other_dp(
    tmp_path,
) -> None:
    with _cluster(tmp_path) as cluster:
        rank0_process = cluster._stage_process((0, 0, 0))
        rank0_pid = rank0_process.pid
        cluster.dispatch(_request("killed-rank0", 0))
        cluster.dispatch(_request("survives-rank1", 1))
        cluster.pump_ready_stages()
        assert cluster.coordinator.stage_is_busy(0, 0)
        assert cluster.coordinator.stage_is_busy(1, 0)

        cluster.schedule_fail_stop(
            event_id="hard-worker-0",
            dp_rank=0,
            offset_s=cluster.coordinator.now_s,
            duration_s=0.01,
            failure_scope="worker",
        )
        injected = cluster.process_next_event()
        assert injected.kind == "hard_failure_inject"
        assert cluster.coordinator.now_s == 0.0
        assert not cluster.coordinator.stage_is_busy(0, 0)
        assert cluster.coordinator.stage_is_busy(1, 0)
        assert not rank0_process.is_alive()
        assert cluster.dp_controller.hard_failed() == [0]
        failed_gpu = cluster.resource_bridge.topology.group(0, 0)[0].gpu
        assert not cluster.global_des.gpu_hbm[failed_gpu].healthy

        while "hard-worker-0" not in cluster._completed_failure_ids:
            cluster.process_next_event()
        restarted_process = cluster._stage_process((0, 0, 0))
        assert restarted_process.pid != rank0_pid
        assert restarted_process.is_alive()
        assert cluster.dp_controller.hard_failed() == []
        repaired = cluster.global_des.gpu_hbm[failed_gpu]
        assert repaired.healthy
        assert repaired.primary_kv == 0
        assert repaired.prefix_kv == 0
        assert repaired.aegis_pending == 0
        assert repaired.aegis_committed == 0

        cluster.dispatch(_request("after-repair-rank0", 0))
        result = cluster.run_until_complete(
            expected_request_ids={"survives-rank1", "after-repair-rank0"},
            max_events=500,
        )

    finished = {
        str(message.get("rid"))
        for message in result.detokenizer_messages
        if bool(message.get("finished", False))
    }
    assert finished == {"survives-rank1", "after-repair-rank0"}
    assert "killed-rank0" not in finished
    assert result.resource_audit_errors == []
    failure = result.failure_events[0]
    assert failure["failure_mode"] == "hard_fail_stop"
    assert failure["affected_request_ids"] == ["killed-rank0"]
    assert len(failure["cancelled_completions"]) == 1
    assert failure["repair_restored_empty_volatile_state"] is True
    assert any(
        row["kind"] == "event_cancelled"
        for row in result.global_des_timeseries["journal"]
    )


def test_node_failstop_restarts_all_local_workers_and_empty_node_dram(
    tmp_path,
) -> None:
    with _cluster(tmp_path) as cluster:
        cluster.schedule_fail_stop(
            event_id="hard-node-0",
            node_id="node0",
            offset_s=0.0,
            duration_s=0.005,
            failure_scope="node",
        )
        cluster.schedule_request_arrival(_request("node1-control", 1), at_s=0.0)
        result = cluster.run_until_complete(
            expected_request_ids={"node1-control"}, max_events=200
        )

    assert result.resource_audit_errors == []
    failure = result.failure_events[0]
    assert failure["node_id"] == "node0"
    assert failure["affected_dp_ranks"] == [0]
    assert failure["repair_restored_empty_volatile_state"] is True
    node0 = result.global_des_snapshot["node_dram"]["node0"]
    assert node0["healthy"] is True
    assert node0["hicache_primary"] == 0
    assert node0["aegis_pending"] == 0
    assert node0["aegis_committed"] == 0


def test_completed_request_stats_survive_later_empty_worker_restart(
    tmp_path,
) -> None:
    """Parent terminal evidence must outlive process-local REQUEST_STATS."""

    rid = "completed-before-future-failure"
    with _cluster(tmp_path) as cluster:
        cluster.dispatch(_request(rid, 0))
        before = cluster.run_until_complete(
            expected_request_ids={rid}, max_events=200
        )
        assert before.rank_snapshots[0]["request_stats"][rid]["status"] == (
            "completed"
        )

        failure_at = cluster.coordinator.now_s
        cluster.schedule_fail_stop(
            event_id="post-completion-restart",
            dp_rank=0,
            offset_s=failure_at,
            duration_s=0.005,
            failure_scope="worker",
        )
        after = cluster.run_until_complete(
            expected_request_ids={rid}, max_events=200
        )

    assert after.failure_events[0]["affected_request_ids"] == []
    assert after.rank_snapshots[0]["request_stats"][rid]["status"] == (
        "completed"
    )
    assert after.rank_snapshots[0]["stages"]["0"]["request_stats"][rid][
        "status"
    ] == "completed"
    assert after.rank_snapshots[0]["request_stats"][rid][
        "aegis_assigned_dp_rank"
    ] == 0


def test_hard_failure_accounts_for_parent_owned_work_without_completion_event(
    tmp_path,
) -> None:
    """A boundary-owned request is active even before a next event exists."""

    rid = "owned-before-next-event"
    with _cluster(tmp_path) as cluster:
        request = _request(rid, 0)
        cluster._request_catalog[rid] = request
        cluster._stage_owned_request_ids[(0, 0, 0)].add(rid)
        cluster.schedule_fail_stop(
            event_id="owned-work-failure",
            dp_rank=0,
            offset_s=0.0,
            duration_s=0.005,
            failure_scope="worker",
        )
        result = cluster.run_until_complete(
            expected_request_ids=set(), max_events=50
        )

    assert result.failure_events[0]["affected_request_ids"] == [rid]
    assert result.failure_events[0]["cancelled_completions"] == []
