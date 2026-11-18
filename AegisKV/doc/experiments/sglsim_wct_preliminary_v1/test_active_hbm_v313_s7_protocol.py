#!/usr/bin/env python3
"""Executable structural gates for the preregistered v3.1.3 S7 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_s7_directional_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text())
    sweep = spec["sweep"]
    hardware = spec["hardware"]
    workload = spec["workload"]
    directions = sweep["bandwidth_directions"]
    checks: list[tuple[str, bool]] = []

    checks.append((
        "exact_minimum_plan_168_cells",
        len(sweep["policies"]) * len(directions)
        * len(sweep["bandwidth_values_gb_s"]) * len(sweep["paired_seeds"])
        == 120
        and len(sweep["policies"]) * len(sweep["repair_failure_domains"])
        * len(sweep["repair_durations_ms"]) * len(sweep["paired_seeds"])
        == 48
        and sweep["registered_cell_count"] == 168,
    ))
    checks.append((
        "exact_four_direction_names_and_phase_tables",
        directions == {
            "source_to_peer_hbm_copy": directions["source_to_peer_hbm_copy"],
            "source_to_dram_copy": directions["source_to_dram_copy"],
            "peer_hbm_recovery": directions["peer_hbm_recovery"],
            "dram_recovery": directions["dram_recovery"],
        }
        and directions["source_to_peer_hbm_copy"]["protocol_table"] == "copy_bandwidth_gb_s"
        and directions["source_to_dram_copy"]["protocol_table"] == "copy_bandwidth_gb_s"
        and directions["peer_hbm_recovery"]["protocol_table"] == "load_bandwidth_gb_s"
        and directions["dram_recovery"]["protocol_table"] == "load_bandwidth_gb_s",
    ))
    checks.append((
        "exact_values_policies_domains_and_seeds",
        sweep["bandwidth_values_gb_s"] == [1.0, 5.0, 10.0, 25.0, 50.0]
        and sweep["repair_durations_ms"] == [10000.0, 30000.0, 60000.0, 120000.0]
        and sweep["policies"] == ["Native Mooncake", "AegisKV"]
        and sweep["repair_failure_domains"] == ["source_worker_gpu", "source_node"]
        and sweep["paired_seeds"] == [2026082600, 2026082601, 2026082602],
    ))
    checks.append((
        "cross_node_remote_topology_frozen",
        hardware["system_dp"] == 64 and hardware["gpus_per_node"] == 8
        and hardware["node_count"] == 8
        and hardware["topology"] == {
            "peer_hbm": "cross_node", "dram": "remote",
            "peer_rule": "same local GPU index on the next node in a node ring",
            "dram_rule": "CPU DRAM attached to the next node in the same node ring",
        },
    ))
    checks.append((
        "copy_and_recovery_profiles_are_separate_and_complete",
        set(hardware["baseline_copy_bandwidth_gb_s"])
        == {"peer_hbm_same_node", "peer_hbm_cross_node", "dram_local", "dram_remote"}
        and set(hardware["baseline_recovery_bandwidth_gb_s"])
        == {"peer_hbm_same_node", "peer_hbm_cross_node", "dram_local", "dram_remote"}
        and hardware["bandwidth_calibration_status"]
        == "MODELED_NOT_PHYSICALLY_CALIBRATED_A100",
    ))
    checks.append((
        "common_warmup_measurement_and_six_control_windows",
        workload["fixed_arrival_cutoff_ms"] == 180000.0
        and workload["common_warmup_ms"] == 90000.0
        and workload["measurement_start_ms"] == 90000.0
        and workload["measurement_end_ms"] == 180000.0
        and workload["control_opportunity_times_ms"]
        == [30000.0, 60000.0, 90000.0, 120000.0, 150000.0, 180000.0],
    ))
    checks.append((
        "capacity_and_no_oracle_formula_frozen",
        hardware["total_hbm_bytes_per_gpu"] == 42949672960
        and spec["model"]["weight_bytes_per_rank"] == 16381470720
        and spec["model"]["kv_bytes_per_token"] == 147456
        and spec["model"]["request_max_output_tokens"] == 4096
        and "request_max_output_tokens" in spec["capacity_equation"]["dispatch_reservation_formula"]
        and "realized" not in spec["capacity_equation"]["dispatch_reservation_formula"],
    ))
    checks.append((
        "policy_specific_replacement_activation_contract_frozen",
        sweep["fixed_modeled_activation_delay_ms"] == 500.0
        and sweep["policy_contract"]["Native Mooncake"] == {
            "source_worker_gpu": "replacement_wait",
            "source_node": "replacement_wait",
        }
        and "fence activation" in sweep["policy_contract"]["AegisKV"]["source_node"],
    ))
    checks.append((
        "no_direction_or_performance_selection_gate",
        "no direction" in spec["selection_policy"].lower()
        and spec["statistics"]["performance_direction_gates"].startswith("none")
        and not any(
            "improve" in gate.lower() or "minimum gain" in gate.lower()
            for gate in spec["hard_gates"]
        ),
    ))
    checks.append((
        "all_preregistered_source_hashes_match",
        bool(spec["sources"])
        and all(
            Path(row["path"]).exists()
            and sha256(Path(row["path"])) == row["sha256"]
            for row in spec["sources"].values()
        ),
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
