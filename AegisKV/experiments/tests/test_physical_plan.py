from __future__ import annotations

from pathlib import Path

from AegisKV.experiments.physical.contract import (
    build_matched_replay_plan,
    load_physical_configs,
    preflight_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_inventory() -> dict:
    nodes = []
    for node_index in range(4):
        nodes.append(
            {
                "hostname": f"a100-{node_index}",
                "host_memory_gib": 1024,
                "aegiskv_reserved_host_memory_gib": 512,
                "network_line_rate_gbit_per_s": 200,
                "all_pairs_intra_node_p2p": True,
                "zero_foreign_gpu_processes": True,
                "gpus": [
                    {
                        "index": gpu_index,
                        "uuid": f"GPU-{node_index}-{gpu_index}",
                        "name": "NVIDIA A100-SXM4-40GB",
                        "memory_mib": 40536,
                        "compute_capability": "8.0",
                    }
                    for gpu_index in range(8)
                ],
            }
        )
    return {
        "schema_version": "aegiskv.physical-inventory.v1",
        "allocation": {"exclusive": True, "operator_authorized": True},
        "nodes": nodes,
        "fabric": {"all_pairs_inter_node_rdma": True},
        "runtime_identity": {
            "sglang_git_sha": "1" * 64,
            "mooncake_git_sha": "2" * 64,
            "container_digest": "sha256:" + "3" * 64,
            "current_design_sha256": "0fdfd097a536ab73cd9f7eb12099eb6f5b9c068a061da30af57a8e6798caacac",
            "model_snapshot_revisions": {
                "qwen3-8b": "4" * 64,
                "qwen3-32b": "5" * 64,
            },
        },
    }


def test_matched_plan_binds_final_lineage_without_outcome_fields() -> None:
    plan = build_matched_replay_plan()
    assert plan["selection"]["cell_count"] == 201
    assert plan["selection"]["cells_by_group"] == {
        "S1": 12,
        "S2": 54,
        "S3": 24,
        "S4": 12,
        "S5": 9,
        "S6": 6,
        "S7": 36,
        "S8": 24,
        "S9": 18,
        "E1": 6,
    }
    assert all(len(seeds) == 3 for seeds in plan["selection"]["seeds_by_group"].values())
    forbidden = {"workflow_wct_mean_ms", "failure_added_wct_mean_ms", "signed_wct_delta_mean_ms"}
    assert all(not (forbidden & set(cell["inputs"])) for cell in plan["cells"])
    assert plan["authorization"]["matched_replay_authorized"] is False
    assert plan["authorization"]["job_submission_performed"] is False


def test_exact_inventory_authorizes_calibration_but_not_replay() -> None:
    configs = load_physical_configs()
    plan = build_matched_replay_plan()
    result = preflight_inventory(inventory=_valid_inventory(), plan=plan, configs=configs)
    assert result["status"] == "READY_FOR_CALIBRATION"
    assert result["failures"] == []
    assert result["authorization"]["calibration_collection_authorized"] is True
    assert result["authorization"]["matched_replay_authorized"] is False


def test_two_a100_80gb_inventory_fails_closed() -> None:
    inventory = _valid_inventory()
    inventory["nodes"] = inventory["nodes"][:1]
    inventory["nodes"][0]["gpus"] = inventory["nodes"][0]["gpus"][:2]
    for gpu in inventory["nodes"][0]["gpus"]:
        gpu["name"] = "NVIDIA A100 80GB PCIe"
        gpu["memory_mib"] = 81920
    result = preflight_inventory(inventory=inventory, plan=build_matched_replay_plan())
    assert result["status"] == "BLOCKED_TOPOLOGY_OR_IDENTITY"
    assert "node_count_mismatch" in result["failures"]
    assert any(value.endswith("gpu_count_mismatch") for value in result["failures"])
    assert any(value.endswith("gpu_memory_mismatch") for value in result["failures"])
    assert result["authorization"]["calibration_collection_authorized"] is False


def test_recorded_local_inventory_is_explicitly_blocked() -> None:
    inventory_path = ROOT / "physical" / "status" / "current_dp2_inventory.json"
    import json

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    result = preflight_inventory(inventory=inventory, plan=build_matched_replay_plan())
    assert result["status"] == "BLOCKED_TOPOLOGY_OR_IDENTITY"
    assert result["authorization"]["matched_replay_authorized"] is False
