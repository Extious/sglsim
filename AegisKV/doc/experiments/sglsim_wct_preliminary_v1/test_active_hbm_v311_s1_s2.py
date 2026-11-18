#!/usr/bin/env python3
"""Executable structural gates for the v3.1.1 S1/S2 campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v311_s1_s2_protocol.json"
DISPLAY = [
    "Native Mooncake",
    "Static",
    "DejaVu",
    "KevlarFlow",
    "GhostServe",
    "AegisKV",
]
MAPPING = {
    "Native Mooncake": "Native Mooncake",
    "Static": "Static FCFS",
    "DejaVu": "DejaVu",
    "KevlarFlow": "KevlarFlow",
    "GhostServe": "GhostServe",
    "AegisKV": "AegisKV",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text())
    checks: list[tuple[str, bool]] = []

    sources_ok = True
    for path_key, sha_key in (
        ("current_design_path", "current_design_sha256"),
        ("v31_des_path", "v31_des_sha256"),
        ("core_runner_path", "core_runner_sha256"),
        ("capacity_protocol_path", "capacity_protocol_sha256"),
        ("base_protocol_path", "base_protocol_sha256"),
        ("execution_plan_path", "execution_plan_sha256"),
        ("topology_recovery_path", "topology_recovery_sha256"),
        ("poisson_injection_path", "poisson_injection_sha256"),
    ):
        path = Path(spec["sources"][path_key])
        if not path.is_absolute():
            path = ROOT / path
        sources_ok &= path.is_file() and sha256(path) == spec["sources"][sha_key]
    checks.append(("all_frozen_source_hashes_match", sources_ok))

    checks.append((
        "display_names_and_internal_mapping_exact",
        spec["display_to_internal_policy"] == MAPPING
        and spec["S1"]["display_policies"] == DISPLAY
        and spec["S2"]["display_policies"] == ["Native Mooncake", "AegisKV"]
        and "Static FCFS" not in spec["S1"]["display_policies"],
    ))

    topology = spec["hardware_topology"]
    placement_ok = True
    same_count = cross_count = 0
    for rank in range(topology["system_dp"]):
        source_node = rank // topology["gpus_per_node"]
        same = rank % 2 == 0
        target_node = source_node if same else (source_node + 1) % topology["node_count"]
        placement_ok &= 0 <= source_node < topology["node_count"]
        placement_ok &= 0 <= target_node < topology["node_count"]
        placement_ok &= (target_node == source_node) if same else (target_node != source_node)
        same_count += int(same)
        cross_count += int(not same)
    checks.append((
        "dp16_resolves_to_two_8gpu_nodes_with_both_topologies",
        topology["hbm_bytes_per_gpu"] == 40 * 1024**3
        and topology["gpus_per_node"] == 8
        and topology["node_count"] == 2
        and topology["system_dp"] == 16
        and same_count == cross_count == 8
        and placement_ok,
    ))

    copy_bw = topology["copy_bandwidth_gb_s"]
    recovery_bw = topology["recovery_bandwidth_gb_s"]
    checks.append((
        "four_directional_copy_and_recovery_values_frozen",
        copy_bw == {
            "peer_hbm_same_node": 50.0,
            "peer_hbm_cross_node": 10.0,
            "dram_local": 16.0,
            "dram_remote": 10.0,
        }
        and recovery_bw == copy_bw
        and copy_bw["peer_hbm_same_node"] != copy_bw["peer_hbm_cross_node"]
        and copy_bw["dram_local"] != copy_bw["dram_remote"]
        and topology["bandwidth_calibration_status"]
        == "MODELED_NOT_PHYSICALLY_CALIBRATED_A100",
    ))

    modes = spec["policy_failover_contract"]["policy_modes"]
    checks.append((
        "six_policy_failover_contract_complete_and_not_tuned",
        set(modes) == set(MAPPING.values())
        and modes["Native Mooncake"] == {
            "source_worker_gpu": "replacement_wait",
            "source_node": "replacement_wait",
        }
        and all(
            modes[policy] == {
                "source_worker_gpu": "prebound_fence_activation",
                "source_node": "topology_conditioned_committed_replica",
            }
            for policy in set(MAPPING.values()) - {"Native Mooncake"}
        )
        and spec["policy_failover_contract"]["timing_status"]
        == "MODELED_UNMEASURED_SENSITIVITY_INPUTS",
    ))

    checks.append((
        "failure_domain_sizes_exact",
        topology["source_worker_gpu_domain_size"] == 1
        and topology["source_node_domain_size"] == topology["gpus_per_node"] == 8,
    ))

    checks.append((
        "s1_and_pilot_cell_counts_exact",
        spec["pilot"]["expected_cells"] == 6 * 2 * 1 == 12
        and spec["S1"]["expected_cells"] == 6 * 2 * 3 == 36
        and spec["S1"]["fault_probes_per_domain"] == 24,
    ))

    checks.append((
        "s2_factor_counts_and_sparse_rate_boundary_explicit",
        spec["S2"]["reported_factor_cells"] == 2 * 5 * 3 * 3 == 90
        and spec["S2"]["unique_des_cells"] == 2 * 5 * 3 * 3 == 90
        and spec["S2"]["failure_rates_per_1000_gpu_shard_hour"]
        == [0.0, 5.952, 20.833]
        and "rate-specific Poisson fail-stop events are injected"
        in spec["S2"]["rate_estimator"]
        and "zero events" in spec["S2"]["rate_estimator_boundary"],
    ))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS")
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
