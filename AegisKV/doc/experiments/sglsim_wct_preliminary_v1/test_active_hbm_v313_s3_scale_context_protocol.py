#!/usr/bin/env python3
"""Fail-closed structure gates for the preregistered v3.1.3 S3 pilot."""

from __future__ import annotations

import json
from pathlib import Path

import active_hbm_v313 as v313
import run_preliminary as core
import test_active_hbm_v313 as v313_tests


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s3_scale_context_protocol.json"


def configured_protocol(*, dp: int, context: int, topology: str) -> tuple[dict, dict]:
    protocol, capacity = v313_tests.inputs(context=context, active=2)
    spec = json.loads(SPEC_PATH.read_text())
    route = spec["hardware"]["topologies"][topology]
    protocol["models"]["qwen3_8b"]["center_system_dp"] = int(dp)
    protocol["workload"]["center_context_tokens"] = int(context)
    protocol["workload"]["center_active_requests_per_gpu"] = 2
    protocol["hardware"]["node_count"] = int(dp) // 8
    protocol.setdefault("runtime_overrides", {})["topology"] = {
        "peer_hbm": route["peer_hbm"],
        "dram": route["dram"],
    }
    return protocol, capacity


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text())
    checks: list[tuple[str, bool]] = []
    sweep = spec["sweep"]
    checks.append((
        "exact_axis_values_and_shared_center",
        sweep["scale_axis"]["system_dp"] == [16, 32, 64, 128, 256]
        and sweep["scale_axis"]["fixed_context_tokens"] == 16384
        and sweep["context_axis"]["context_tokens"]
        == [4096, 8192, 16384, 32768, 40960]
        and sweep["context_axis"]["fixed_system_dp"] == 64
        and sweep["unique_axis_points"] == 9,
    ))
    checks.append((
        "exact_policies_topologies_seeds_and_108_cells",
        sweep["policies"] == ["Native Mooncake", "AegisKV"]
        and sweep["topologies"] == ["same_node", "cross_node"]
        and sweep["paired_seeds"] == [2026082600, 2026082601, 2026082602]
        and sweep["registered_cell_count"]
        == sweep["unique_axis_points"] * 2 * 2 * 3 == 108,
    ))
    hardware = spec["hardware"]
    checks.append((
        "a100_40gib_capacity_constants_exact",
        hardware["total_hbm_bytes_per_gpu"] == 40 * 2**30
        and hardware["usable_hbm_bytes_per_gpu"] == 37795712204
        and hardware["runtime_reserve_bytes_per_gpu"] == 5153960756
        and hardware["serving_plus_peer_backup_budget_bytes_per_gpu"]
        == 21414241484,
    ))
    equation = spec["capacity_equation"]
    expected_reservation = (40960 + 4096) * 147456
    checks.append((
        "max_context_two_request_analytic_capacity_gate",
        equation["max_context_reservation_bytes_per_request"] == expected_reservation
        and equation["max_context_two_request_serving_bytes"] == 2 * expected_reservation
        and equation["max_context_two_request_headroom_bytes"]
        == hardware["serving_plus_peer_backup_budget_bytes_per_gpu"]
        - 2 * expected_reservation
        and equation["max_context_two_request_headroom_bytes"] > 0,
    ))
    expected_bandwidth = {
        "same_node": (50.0, 16.0),
        "cross_node": (10.0, 10.0),
    }
    route_ok = True
    for topology in sweep["topologies"]:
        for dp in (16, 32, 64, 128, 256):
            protocol, _capacity = configured_protocol(
                dp=dp, context=16384, topology=topology
            )
            rows = [v313.topology_placement(protocol, rank) for rank in range(dp)]
            peer_targets = [int(row["peer_hbm_target_rank"]) for row in rows]
            expected_peer = hardware["topologies"][topology]["peer_hbm"]
            expected_dram = hardware["topologies"][topology]["dram"]
            route_ok &= (
                sorted(peer_targets) == list(range(dp))
                and all(target != rank for rank, target in enumerate(peer_targets))
                and all(row["peer_hbm_mode"] == expected_peer for row in rows)
                and all(row["dram_mode"] == expected_dram for row in rows)
                and core.copy_bandwidth(protocol, 0) == expected_bandwidth[topology]
                and core.load_bandwidth(protocol, 0) == expected_bandwidth[topology]
                and int(protocol["hardware"]["node_count"]) == dp // 8
            )
    checks.append((
        "same_cross_topology_bijective_and_directional_bandwidth_exact",
        route_ok,
    ))
    protocol, _capacity = configured_protocol(dp=16, context=16384, topology="cross_node")
    probe = core.FaultProbe("gate", 1.0, 7, "source_worker_gpu")
    failed_ranks = v313.v312.v31.failure_domain_ranks(
        probe, gpus_per_node=8, dp=16
    )
    checks.append((
        "source_worker_gpu_invalidates_exactly_one_rank",
        failed_ranks == (7,),
    ))
    workload = spec["workload"]
    checks.append((
        "fixed_90s_arrivals_three_controls_and_12_worker_probes",
        workload["fixed_arrival_cutoff_ms"] == 90000.0
        and workload["control_opportunity_times_ms"]
        == [30000.0, 60000.0, 90000.0]
        and workload["fault_probes_per_axis_point_seed"] == 12
        and workload["failure_type"] == "source_worker_gpu"
        and workload["offered_active_requests_per_gpu"] == 2,
    ))
    checks.append((
        "source_hashes_match",
        all(
            Path(row["path"]).exists()
            and core.sha256(Path(row["path"])) == row["sha256"]
            for row in spec["sources"].values()
        ),
    ))
    grammar = spec["figure_grammar"]
    checks.append((
        "paper_figure_grammar_frozen",
        grammar["layout"].startswith("2x2")
        and grammar["minimum_font_pt"] >= 7.0
        and grammar["png_minimum_width_px"] >= 1800
        and grammar["topology_encoding"]
        == {"same_node": "solid line", "cross_node": "dashed line"},
    ))
    gates = set(spec["hard_gates"])
    checks.append((
        "no_result_threshold_or_changed_only_primary_metric",
        "no seed or cell is selected by result shape" in gates
        and spec["statistics"]["forbidden_primary_metric"]
        == "changed-only workflow WCT across policies"
        and spec["paper_ready"] is False,
    ))
    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
