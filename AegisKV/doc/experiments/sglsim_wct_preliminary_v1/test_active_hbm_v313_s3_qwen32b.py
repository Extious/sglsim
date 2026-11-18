#!/usr/bin/env python3
"""Fail-closed structure/mechanism gate for Qwen3-32B S3."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import active_hbm_v31 as v31
import active_hbm_v313 as v313
import run_active_hbm_v313_s3_qwen32b as runner
import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def main() -> None:
    spec = json.loads((ROOT / "active_hbm_v313_s3_qwen32b_protocol.json").read_text())
    checks: list[tuple[str, bool]] = []
    model = spec["model"]
    service = spec["service_model"]
    hardware = spec["hardware"]

    checks.append((
        "explicit_analytic_not_matched_profile_boundary",
        service["status"] == "MODELED_ANALYTIC_WEIGHT_RATIO_NOT_PHYSICAL"
        and service["not_a100_calibration"] is True
        and spec["paper_ready"] is False
        and "matched A100 profile" in spec["profile_gate_resolution"]["forbidden_claims"],
    ))
    weights = model["weight_bytes_per_pp_stage"]
    checks.append((
        "official_weight_partition_sum_and_conservative_stage_exact",
        sum(weights) == model["weight_bytes_total"] == 65524246528
        and max(weights) == model["capacity_stage_weight_bytes"] == 17158981632,
    ))
    expected_kv = 2 * 16 * 8 * 128 * 2
    checks.append((
        "pp_stage_kv_formula_exact",
        expected_kv == model["kv_bytes_per_token_per_stage"] == 65536
        and expected_kv * 4 == model["kv_bytes_per_token_whole_pipeline"],
    ))
    ratio = model["weight_bytes_total"] / service["reference_weight_bytes"]
    checks.append((
        "analytic_service_coefficients_derive_only_from_frozen_ratio",
        abs(ratio - service["weight_ratio"]) <= 1e-15
        and abs(service["prefill_setup_ms"] - 4.0 * ratio) <= 1e-12
        and abs(service["prefill_ms_per_token"] - 0.06 * ratio) <= 1e-12
        and abs(service["decode_ms_per_token"] - 0.34 * ratio) <= 1e-12,
    ))
    budget = (
        hardware["physical_total_hbm_bytes_per_gpu"]
        - hardware["runtime_reserve_bytes_per_gpu"]
        - model["capacity_stage_weight_bytes"]
    )
    reservation = (40960 + model["request_max_output_tokens"]) * expected_kv
    equation = spec["capacity_equation"]
    checks.append((
        "physical_stage_40gib_capacity_equation_and_two_request_gate",
        budget == hardware["serving_plus_peer_backup_budget_bytes_per_stage_gpu"]
        == 20636730572
        and reservation == equation["max_context_reservation_bytes_per_request"]
        and 2 * reservation == equation["max_context_two_request_serving_bytes"]
        and budget - 2 * reservation == equation["max_context_two_request_headroom_bytes"]
        and equation["max_context_two_request_headroom_bytes"] > 0,
    ))
    sweep = spec["sweep"]
    checks.append((
        "full_cross_matrix_exactly_300_no_axis_dedup",
        sweep["system_dp"] == [4, 8, 16, 32, 64]
        and sweep["context_tokens"] == [4096, 8192, 16384, 32768, 40960]
        and len(runner.points(spec, gate=False)) == 25
        and sweep["registered_cell_count"] == 25 * 2 * 2 * 3 == 300,
    ))

    topology_ok = True
    for topology in sweep["topologies"]:
        for dp in sweep["system_dp"]:
            protocol, capacity = runner.resolved_protocol(
                spec, dp=dp, context=16384, topology=topology
            )
            placements = [v313.topology_placement(protocol, rank) for rank in range(dp)]
            expected_peer = hardware["topologies"][topology]["peer_hbm"]
            expected_dram = hardware["topologies"][topology]["dram"]
            topology_ok &= (
                sorted(row["peer_hbm_target_rank"] for row in placements)
                == list(range(dp))
                and all(row["peer_hbm_target_rank"] != row["rank"] for row in placements)
                and all(row["peer_hbm_mode"] == expected_peer for row in placements)
                and all(row["dram_mode"] == expected_dram for row in placements)
                and protocol["hardware"]["gpus_per_node"] == 2
                and protocol["hardware"]["node_count"] == dp // 2
                and capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]
                == budget
            )
    checks.append((
        "pp4_two_pipeline_per_node_bijective_topology",
        topology_ok,
    ))

    protocol, capacity = runner.resolved_protocol(
        spec, dp=4, context=4096, topology="cross_node"
    )
    probe = core.FaultProbe("gate", 1.0, 1, "source_worker_gpu")
    checks.append((
        "worker_gpu_fault_invalidates_one_pipeline_via_one_stage",
        v31.failure_domain_ranks(probe, gpus_per_node=2, dp=4) == (1,)
        and hardware["fault_abstraction"].startswith(
            "a source_worker_gpu event invalidates one logical pipeline"
        ),
    ))

    nominal, _meta = core.build_workload(
        protocol, seed=2026082600, workflows=None, active_per_gpu=2,
        arrival_cutoff_ms=500.0,
    )
    altered = [
        replace(row, output_tokens=row.output_tokens + 9999,
                service_ms=row.service_ms + 999999.0)
        for row in nominal
    ]
    predicted_a = v31.apply_nonoracle_modeled_slack(
        nominal, protocol, seed=2026082600
    )
    predicted_b = v31.apply_nonoracle_modeled_slack(
        altered, protocol, seed=2026082600
    )
    checks.append((
        "dispatch_prediction_uses_no_realized_output_or_service_oracle",
        [(row.request_id, row.predicted_slack_ms) for row in predicted_a]
        == [(row.request_id, row.predicted_slack_ms) for row in predicted_b],
    ))
    compat = protocol["models"]["qwen3_8b"]
    checks.append((
        "compatibility_key_resolves_to_explicit_qwen32b_layout",
        compat["logical_model_id"] == "Qwen/Qwen3-32B"
        and compat["pp"] == 4 and compat["tp"] == 1
        and compat["kv_bytes_per_token"] == 65536
        and protocol["runtime_overrides"]["model_abstraction"]
        ["logical_rank_is_pipeline_replica"] is True,
    ))
    replay = runner.Qwen32BReplay(protocol, "AegisKV", capacity)
    task = predicted_a[0]
    profile = replay._profile(task)
    checks.append((
        "controller_profile_has_qwen32b_pp4_metadata",
        profile.model_revision == "Qwen3-32B-official-metadata"
        and profile.kv_abi == "qwen3-32b-bf16-stage-kv-v1"
        and profile.parallel_layout == "TP1-PP4-stage-aligned",
    ))
    checks.append((
        "fixed_exposure_controls_sources_and_no_result_gate",
        spec["workload"]["fixed_arrival_cutoff_ms"] == 90000.0
        and spec["workload"]["control_opportunity_times_ms"]
        == [30000.0, 60000.0, 90000.0]
        and spec["workload"]["fault_probes_per_condition_seed"] == 12
        and all(Path(row["path"]).exists() and core.sha256(Path(row["path"])) == row["sha256"]
                for row in spec["sources"].values())
        and spec["selection_policy"].startswith("All 300 cells"),
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
