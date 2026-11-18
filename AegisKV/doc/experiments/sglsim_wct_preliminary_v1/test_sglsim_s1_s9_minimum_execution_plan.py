#!/usr/bin/env python3
"""Executable structural gates for the preregistered S1--S9 plan."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "sglsim_s1_s9_minimum_execution_plan.json"
ARCHITECTURES = [
    "Native Mooncake",
    "Static",
    "DejaVu",
    "KevlarFlow",
    "GhostServe",
    "AegisKV",
]
S9_PATHS = [
    "source_to_peer_hbm",
    "source_to_dram",
    "peer_hbm_recovery",
    "dram_recovery",
]


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    figures = plan["figures"]
    checks: list[tuple[str, bool]] = []

    registry = plan["architecture_registry"]
    excluded = registry["excluded_from_primary_comparison"]
    checks.append(
        (
            "six_named_architectures_no_pseudo_baseline",
            registry["primary_architectures"] == ARCHITECTURES
            and registry["count"] == 6
            and registry["internal_policy_map"] == {
                "Native Mooncake": "Native Mooncake",
                "Static": "Static FCFS",
                "DejaVu": "DejaVu",
                "KevlarFlow": "KevlarFlow",
                "GhostServe": "GhostServe",
                "AegisKV": "AegisKV",
            }
            and "RAIDServe" in excluded
            and "Recompute Only" in excluded
            and not ({"RAIDServe", "Recompute Only"} & set(ARCHITECTURES)),
        )
    )

    topology = plan["common"]["hardware_topology"]
    bandwidth = topology["modeled_bandwidth_profile"]
    checks.append(
        (
            "a100_40gib_node_and_directional_topology_frozen",
            topology["hbm_bytes_per_gpu"] == 40 * 1024**3
            and topology["gpus_per_node"] == 8
            and set(topology["fault_domains"])
            == {"source_worker_gpu", "source_node"}
            and "cross-node paths must never inherit same-node values"
            in topology["path_bandwidth_rule"]
            and bandwidth["profile_id"]
            == "sglsim_a100_40gib_modeled_directional_v1"
            and bandwidth["source_sha256"]
            == "7535a7bf7162f036d0154a67f25608bcb43c0f59526593f4f06e37515fcca9c9"
            and bandwidth["copy_bandwidth_gb_s"]
            == {
                "peer_hbm_same_node": 50.0,
                "peer_hbm_cross_node": 10.0,
                "dram_local": 16.0,
                "dram_remote": 10.0,
            }
            and bandwidth["recovery_bandwidth_gb_s"]
            == bandwidth["copy_bandwidth_gb_s"]
            and bandwidth["calibration_status"]
            == "MODELED_NOT_PHYSICALLY_CALIBRATED_A100"
            and figures["S1"]["execution_topology"]["node_count"] == 2
            and figures["S1"]["execution_topology"]["total_gpu_count"] == 16
            and figures["S2"]["execution_topology"]
            == {
                "model": "Qwen3-8B TP1/PP1",
                "system_dp": 16,
                "total_gpu_count": 16,
                "node_count": 2,
                "replicas_per_node": 8,
                "placement": "identical to S1; rate cells share paired workflow/fault-probe universes before rate aggregation",
            },
        )
    )

    checks.append(
        (
            "s1_matrix_is_6x2x3",
            figures["S1"]["architectures"] == ARCHITECTURES
            and figures["S1"]["failure_domains"]
            == ["source_worker_gpu", "source_node"]
            and figures["S1"]["cells"] == 6 * 2 * 3 == 36,
        )
    )
    checks.append(
        (
            "s8_reuses_exact_s1_architecture_domain_universe",
            figures["S8"]["architectures"] == figures["S1"]["architectures"]
            and figures["S8"]["failure_domains"]
            == figures["S1"]["failure_domains"]
            and figures["S8"]["cells"] == 36,
        )
    )

    s3 = figures["S3"]
    models = {entry["model_id"]: entry for entry in s3["models"]}
    q8 = models["Qwen/Qwen3-8B"]
    q32 = models["Qwen/Qwen3-32B"]
    checks.append(
        (
            "s3_has_explicit_8b_and_32b_layout_matrices",
            set(models) == {"Qwen/Qwen3-8B", "Qwen/Qwen3-32B"}
            and q8["layout"] == {"tp": 1, "pp": 1}
            and q32["layout"]["tp"] == 1
            and q32["layout"]["pp"] == 4
            and q8["system_dp"] == [16, 32, 64, 128, 256]
            and q32["system_dp"] == [4, 8, 16, 32, 64]
            and q8["topology_by_system_dp"]["node_count"]
            == [2, 4, 8, 16, 32]
            and q8["topology_by_system_dp"]["replicas_per_node"] == 8
            and q32["topology_by_system_dp"]["gpus_per_pipeline_replica"] == 4
            and q32["topology_by_system_dp"]["total_gpu_count"]
            == [16, 32, 64, 128, 256]
            and q32["topology_by_system_dp"]["node_count"]
            == [2, 4, 8, 16, 32]
            and q32["topology_by_system_dp"]["pipeline_replicas_per_node"] == 2
            and s3["contexts_tokens"] == [4096, 8192, 16384, 32768, 40960]
            and q8["cells"] == q32["cells"] == 2 * 5 * 5 * 3 == 150
            and s3["cells"] == 300,
        )
    )

    q8_total = q8["capacity"]["total_hbm_bytes_per_rank"]
    q8_usable = q8["capacity"]["usable_hbm_bytes_per_rank"]
    q8_reserve = q8["capacity"]["runtime_reserve_bytes_per_rank"]
    checks.append(
        (
            "qwen3_8b_weight_kv_and_capacity_arithmetic",
            q8["kv"]["bytes_per_token_per_rank"] == 2 * 36 * 8 * 128 * 2
            and q8_usable + q8_reserve == q8_total
            and q8["capacity"]["serving_plus_backup_budget_bytes_per_rank"]
            == [q8_usable - q8["weight_bytes_per_rank"][0]]
            and q8["service_and_recompute"]["status"]
            == "READY_PRELIMINARY_METADATA_DES",
        )
    )

    q32_weights = q32["weight_bytes_per_pp_rank"]
    q32_usable = q32["capacity"]["usable_hbm_bytes_per_rank"]
    q32_budgets = q32["capacity"]["serving_plus_backup_budget_bytes_per_pp_rank"]
    checks.append(
        (
            "qwen3_32b_weight_kv_capacity_and_profile_fail_close",
            sum(q32_weights) == q32["weight_bytes_total"] == 65524246528
            and q32["kv"]["bytes_per_token_per_rank"] == 2 * 16 * 8 * 128 * 2
            and q32["kv"]["bytes_per_token_whole_pipeline"]
            == 4 * q32["kv"]["bytes_per_token_per_rank"]
            and q32_budgets == [q32_usable - weight for weight in q32_weights]
            and min(range(4), key=q32_budgets.__getitem__)
            == q32["capacity"]["bottleneck_pp_rank"]
            and min(q32_budgets) > 0
            and q32["service_and_recompute"]["status"]
            == "BLOCKED_REQUIRES_MATCHED_A100_40GB_PROFILE",
        )
    )

    checks.append(
        (
            "s9_is_four_direction_by_four_payload_join",
            figures["S9"]["paths"] == S9_PATHS
            and figures["S9"]["payload_mib"] == [64, 256, 512, 1024]
            and figures["S9"]["join_key"] == ["path", "payload_mib"]
            and figures["S9"]["cells"] == 4 * 4 == 16,
        )
    )

    expected_total = sum(figures[name]["cells"] for name in figures)
    checks.append(
        (
            "nominal_and_unique_cell_totals",
            expected_total == plan["nominal_cells_before_reuse"] == 715
            and plan["minimum_unique_cells"] == 715 - 36 == 679
            and plan["status"] == "PLANNED_NOT_RUN",
        )
    )

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS")
    if failed:
        raise SystemExit(f"failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
