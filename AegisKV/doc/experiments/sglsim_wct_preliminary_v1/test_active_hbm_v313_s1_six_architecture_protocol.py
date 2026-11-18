#!/usr/bin/env python3
"""Explicit structure gates for the preregistered ten-seed S1 campaign."""

import json
from pathlib import Path

import active_hbm_v31 as v31
import run_preliminary as core
import test_active_hbm_v313 as inputs


ROOT = Path(__file__).resolve().parent


def main() -> None:
    spec = json.loads((ROOT / "active_hbm_v313_s1_six_architecture_10seed_protocol.json").read_text())
    names = ["Native Mooncake", "Static", "DejaVu", "KevlarFlow", "GhostServe", "AegisKV"]
    checks = []
    checks.append(("exact_display_names_and_order", spec["display_policy_order"] == names and list(spec["display_to_internal_policy"]) == names))
    checks.append(("static_display_mapping", spec["display_to_internal_policy"]["Static"] == "Static FCFS"))
    checks.append(("ten_unique_paired_seeds", len(spec["workload"]["paired_workload_seeds"]) == 10 and len(set(spec["workload"]["paired_workload_seeds"])) == 10))
    checks.append(("exact_60_cells", spec["matrix"]["des_cells"] == 6 * 10))
    checks.append(("exact_probe_exposure", spec["failure_conditioning"]["probes_per_domain_per_seed"] == 24 and spec["failure_conditioning"]["total_probes_per_seed"] == 48))
    checks.append(("two_failure_domains", spec["failure_conditioning"]["failure_domains"] == ["source_worker_gpu", "source_node"]))
    checks.append(("a100_two_node_topology", spec["hardware_topology"]["hbm_bytes_per_gpu"] == 40 * 1024**3 and spec["hardware_topology"]["system_dp"] == 16 and spec["hardware_topology"]["gpus_per_node"] == 8))
    checks.append(("capacity_formula", spec["capacity"]["serving_reservation_bytes_per_request"] == (40960 + 4096) * 147456 and spec["capacity"]["analytical_max_serving_requests_without_peer_backup"] == 3))
    checks.append(("exact_control_windows", spec["workload"]["control_opportunities_ms"] == [30000.0, 60000.0, 90000.0]))
    checks.append(("invalidated_old_oracle_lineage", spec["sources"]["invalidated_single_seed_runner"]["status"] == "INVALIDATED_ORACLE_SLACK_DIAGNOSTIC_ONLY" and spec["workload"]["workload_builder"] == "active_hbm_v31.build_workload"))
    protocol, _capacity = inputs.inputs(context=40960, active=16)
    tasks, metadata = v31.build_workload(protocol, seed=2026082600, active_per_gpu=16, arrival_cutoff_ms=1000.0)
    probes = core.make_fault_probes(tasks, seed=2026082600 ^ 0x5A1F311, count=8, gpus_per_node=8, right_ms=1000.0, failure_types=("source_worker_gpu", "source_node"))
    checks.append(("no_oracle_workload", metadata["slack_predictor"]["reads_realized_output_tokens"] is False and metadata["slack_predictor"]["reads_realized_service_ms"] is False))
    checks.append(("worker_one_node_eight_ranks", all(len(v31.failure_domain_ranks(probe, gpus_per_node=8, dp=16)) == (1 if probe.failure_type == "source_worker_gpu" else 8) for probe in probes)))
    checks.append(("no_direction_selection_and_50_workers", spec["selection"].startswith("NONE") and spec["matrix"]["maximum_parallel_workers"] == 50 and spec["paper_ready"] is False))
    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS")
    if failed:
        raise SystemExit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
