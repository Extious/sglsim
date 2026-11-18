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


def _request(rid: str, dp_rank: int) -> TokenizedGenerateReqInput:
    return TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid=rid,
            input_ids=list(range(32)),
            max_new_tokens=2,
            simulation={
                "server_created_time": 0.0,
                "force_dp_rank": dp_rank,
            },
        )
    )


def _dp2_tp4_topology() -> OfflinePhysicalTopology:
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
            for dp_rank in range(2)
            for tp_rank in range(4)
        ),
        nodes=(
            NodeDramSpec("node0", capacity_bytes=1 << 32, runtime_bytes=1 << 20),
            NodeDramSpec("node1", capacity_bytes=1 << 32, runtime_bytes=1 << 20),
        ),
    )


def test_offline_cluster_exports_dp2_tp4_physical_resource_timeline(
    tmp_path,
) -> None:
    config_path = tmp_path / "offline-physical-resource.json"
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
                "aegis_l1_l2": {"enabled": False},
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

    with OfflineSchedulerCluster(
        server_args,
        physical_topology=_dp2_tp4_topology(),
    ) as cluster:
        cluster.dispatch(_request("physical-dp0", 0))
        cluster.dispatch(_request("physical-dp1", 1))
        result = cluster.run_until_complete(
            expected_request_ids={"physical-dp0", "physical-dp1"},
            max_events=200,
        )

    assert result.resource_audit_errors == []
    assert result.physical_resource_bridge["rank_steps"]
    assert set(result.global_des_snapshot["gpus"]) == {
        f"node{dp}/gpu{tp}/dp{dp}/pp0/tp{tp}"
        for dp in range(2)
        for tp in range(4)
    }
    assert set(result.global_des_snapshot["node_dram"]) == {"node0", "node1"}
    assert {
        (step["dp_rank"], step["pp_rank"])
        for step in result.physical_resource_bridge["rank_steps"]
    } == {(0, 0), (1, 0)}
    assert any(
        step["phase"] == "prepare_after"
        and step["physical_primary_bytes_total"] > 0
        for step in result.physical_resource_bridge["rank_steps"]
    )
    assert result.global_des_timeseries["gpu_hbm"]
    assert result.global_des_timeseries["node_dram"]
