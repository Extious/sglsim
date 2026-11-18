from __future__ import annotations

import json

import pytest

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


def _config(tmp_path, *, dp_size: int = 1, tp_size: int = 2) -> str:
    path = tmp_path / f"offline-hicache-dp{dp_size}-tp{tp_size}.json"
    path.write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {
                    "dp_size": dp_size,
                    "tp_size": tp_size,
                    "pp_size": 1,
                    "load_balance_method": "round_robin",
                    "max_prefill_tokens": 256,
                    "chunked_prefill_size": 256,
                    "max_running_requests": 4,
                },
                "cache": {
                    "page_size": 8,
                    "l1_capacity_pages": 4,
                    "l2_capacity_pages": 512,
                    "hicache_write_policy": "write_through",
                    "write_through_threshold": 1,
                    "load_back_threshold": 1,
                },
                "communication": {"kv_bytes_per_token": 64},
                "aegis_l1_l2": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _topology(*, dp_size: int = 1, tp_size: int = 2) -> OfflinePhysicalTopology:
    return OfflinePhysicalTopology(
        gpus=tuple(
            PhysicalGpuSpec(
                gpu=PhysicalGpuId(
                    node_id=f"node{dp_rank}",
                    local_gpu_id=tp_rank,
                    dp_rank=dp_rank,
                    pp_rank=0,
                    tp_rank=tp_rank,
                ),
                hbm_capacity_bytes=1 << 30,
                weights_bytes=1 << 20,
                runtime_bytes=1 << 20,
            )
            for dp_rank in range(dp_size)
            for tp_rank in range(tp_size)
        ),
        nodes=tuple(
            NodeDramSpec(
                f"node{dp_rank}",
                capacity_bytes=1 << 30,
                runtime_bytes=1 << 20,
            )
            for dp_rank in range(dp_size)
        ),
    )


def _request(rid: str, token_ids: list[int], *, dp_rank: int = 0):
    return TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid=rid,
            input_ids=token_ids,
            max_new_tokens=1,
            simulation={
                "server_created_time": 0.0,
                "force_dp_rank": dp_rank,
            },
        )
    )


def _pool(owner_pages: dict[str, int]) -> dict:
    capacity_pages = 64
    page_size_tokens = 8
    kv_bytes_per_token = 64
    used_pages = sum(owner_pages.values())
    bytes_per_page = page_size_tokens * kv_bytes_per_token
    return {
        "capacity_pages": capacity_pages,
        "used_pages": used_pages,
        "free_pages": capacity_pages - used_pages,
        "capacity_tokens": capacity_pages * page_size_tokens,
        "used_tokens": used_pages * page_size_tokens,
        "free_tokens": (capacity_pages - used_pages) * page_size_tokens,
        "capacity_bytes": capacity_pages * bytes_per_page,
        "used_bytes": used_pages * bytes_per_page,
        "free_bytes": (capacity_pages - used_pages) * bytes_per_page,
        "owner_pages": dict(owner_pages),
    }


def _snapshot(*, l1: dict[str, int], l2: dict[str, int]) -> dict:
    return {
        "schema": "sglsim.offline.rank-kv-resource.v1",
        "timestamp_s": 0.0,
        "dp_rank": 0,
        "pp_rank": 0,
        "scheduler_tp_rank": 0,
        "logical_tp_group_size": 1,
        "scope": "one-symmetric-tp-rank-view",
        "page_size_tokens": 8,
        "kv_bytes_per_token_per_tp_rank": 64,
        "l1_device": _pool(l1),
        "l2_host": _pool(l2),
        "running_request_ids": [],
        "waiting_request_ids": [],
    }


def _d2h_intent(intent_id: str) -> dict:
    return {
        "schema": "sglsim.offline.hicache-intent.v1",
        "intent_id": intent_id,
        "direction": "d2h",
        "owner": "request-d2h",
        "radix_node_id": 7,
        "host_page_ids": [10],
        "device_page_ids": [20],
        "page_size_tokens": 8,
        "page_count": 1,
        "token_count": 8,
        "kv_bytes_per_token_per_tp_rank": 64,
        "size_bytes_per_tp_rank": 512,
        "dp_rank": 0,
        "pp_rank": 0,
        "scheduler_tp_rank": 0,
        "worker_incarnation": 0,
        "logical_tp_scope": "one-symmetric-tp-rank-view",
        "requires_parent_tp_expansion": True,
        "operation_sequence": 0,
        "predicted_duration_s": 0.001,
        "blocks_compute": False,
        "state": "PENDING",
    }


def test_cluster_expands_real_d2h_and_h2d_and_gates_replay_compute(
    tmp_path,
) -> None:
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=1,
        tp_size=2,
        streaming_arrival=True,
        sim_config_path=_config(tmp_path),
    )
    prefix = list(range(128))
    with OfflineSchedulerCluster(
        server_args,
        physical_topology=_topology(),
    ) as cluster:
        cluster.dispatch(_request("seed-prefix", prefix))
        cluster.run_until_complete(
            expected_request_ids={"seed-prefix"}, max_events=300
        )
        for index in range(5):
            rid = f"evict-prefix-{index}"
            filler = list(range(1000 + index * 256, 1128 + index * 256))
            cluster.dispatch(_request(rid, filler))
            cluster.run_until_complete(
                expected_request_ids={rid}, max_events=300
            )
        cluster.dispatch(_request("host-replay", prefix))
        result = cluster.run_until_complete(
            expected_request_ids={"host-replay"}, max_events=500
        )

    artifact = result.physical_hicache_transfers
    assert artifact["audit_errors"] == []
    groups = artifact["groups"]
    transfers = artifact["transfers"]
    assert {group["state"] for group in groups} == {"COMMITTED"}
    assert all(len(group["transfer_ids"]) == 2 for group in groups)
    assert {row["direction"] for row in transfers} == {"d2h", "h2d"}
    h2d = [row for row in transfers if row["direction"] == "h2d"]
    assert h2d
    assert all(row["state"] == "COMMITTED" for row in h2d)
    assert all("pcie-h2d:node0:lane0" in row["resources"] for row in h2d)
    replay_rows = [
        row
        for row in result.telemetry
        if "host-replay" in row.get("request_ids", ())
    ]
    assert replay_rows
    replay_h2d_end_s = max(
        row["end_ns"] / 1_000_000_000
        for row in h2d
        if row["owner"] == "host-replay"
    )
    assert min(float(row["started_at"]) for row in replay_rows) >= (
        replay_h2d_end_s
    )
    # The aggregate Pipeline v2 H2D duration is suppressed; parent physical
    # transfer completion is the sole admission dependency.
    assert all(float(row.get("h2d_s", 0.0)) == 0.0 for row in replay_rows)
    for row in replay_rows:
        assert float(row["completed_at"]) - float(row["started_at"]) == (
            pytest.approx(float(row["pipeline_base_s"]), abs=1e-9)
        )
    assert result.resource_audit_errors == []


def test_hard_failure_mid_d2h_drops_unacked_parent_dram_projection(
    tmp_path,
) -> None:
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        tp_size=1,
        streaming_arrival=True,
        sim_config_path=_config(tmp_path, dp_size=2, tp_size=1),
    )
    with OfflineSchedulerCluster(
        server_args,
        physical_topology=_topology(dp_size=2, tp_size=1),
    ) as cluster:
        assert cluster.resource_bridge is not None
        cluster.resource_bridge.consume_snapshot(
            _snapshot(l1={"source": 1}, l2={"pending-d2h": 1}),
            phase="unit_pending_d2h",
        )
        intent = _d2h_intent("hicache-dp0-pp0-tp0-i0-000000000000")
        cluster._hicache_intent_stage[intent["intent_id"]] = (0, 0, 0)
        group = cluster.hicache_transfer_bridge.submit_intent(intent)
        transfer = cluster.hicache_transfer_bridge.transfers[
            group.transfer_ids[0]
        ]
        failure_ns = (transfer.start_ns + transfer.end_ns) // 2
        cluster.schedule_fail_stop(
            event_id="mid-d2h-worker-loss",
            dp_rank=0,
            offset_s=failure_ns / 1_000_000_000,
            duration_s=0.01,
            failure_scope="worker",
        )
        while True:
            event = cluster.process_next_event()
            if event.kind == "hard_failure_inject":
                break

        assert group.state.value == "ABORTED"
        assert cluster.global_des.node_dram["node0"].hicache_primary == 0
        failure = cluster._active_failures["mid-d2h-worker-loss"]
        rollbacks = failure["ordinary_hicache_failure"][
            "projection_rollbacks"
        ]
        assert rollbacks[0]["released_bytes_per_tp_rank"] == 512


def test_hard_failure_preserves_committed_parent_dram_backup(tmp_path) -> None:
    server_args = ServerArgs(
        model_path="mock-model",
        dp_size=2,
        tp_size=1,
        streaming_arrival=True,
        sim_config_path=_config(tmp_path, dp_size=2, tp_size=1),
    )
    with OfflineSchedulerCluster(
        server_args,
        physical_topology=_topology(dp_size=2, tp_size=1),
    ) as cluster:
        assert cluster.resource_bridge is not None
        cluster.resource_bridge.consume_snapshot(
            _snapshot(l1={"source": 1}, l2={"committed-d2h": 1}),
            phase="unit_committed_d2h",
        )
        bridge = cluster.hicache_transfer_bridge
        callback = bridge.on_group_terminal
        bridge.on_group_terminal = None
        group = bridge.submit_intent(
            _d2h_intent("hicache-dp0-pp0-tp0-i0-000000000001")
        )
        while group.state.value != "COMMITTED":
            cluster.coordinator.pop_next()
        bridge.on_group_terminal = callback

        cluster.schedule_fail_stop(
            event_id="post-commit-worker-loss",
            dp_rank=0,
            offset_s=cluster.coordinator.now_s,
            duration_s=0.01,
            failure_scope="worker",
        )
        injected = cluster.process_next_event()
        assert injected.kind == "hard_failure_inject"

        assert group.state.value == "COMMITTED"
        assert cluster.global_des.node_dram["node0"].hicache_primary == 512
        failure = cluster._active_failures["post-commit-worker-loss"]
        assert failure["ordinary_hicache_failure"][
            "projection_rollbacks"
        ] == []
