#!/usr/bin/env python3
"""Executable mechanism tests for topology-aware v3.1.2 recovery."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import active_hbm_v312 as v312
import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def protocol() -> tuple[dict, dict, dict]:
    base = json.loads(
        (ROOT / "results/mechanism_pilot_v2/composite_protocol.json").read_text()
    )
    capacity = json.loads((ROOT / "active_hbm_v3_protocol.json").read_text())
    spec = json.loads((ROOT / "active_hbm_v311_s1_s2_protocol.json").read_text())
    result = deepcopy(base)
    result["models"]["qwen3_8b"]["center_system_dp"] = 16
    result["workload"]["center_context_tokens"] = 4096
    result["workload"]["center_active_requests_per_gpu"] = 1
    result["hardware"]["gpus_per_node"] = 8
    result["hardware"]["node_count"] = 2
    result["hardware"]["copy_bandwidth_gb_s"] = dict(
        spec["hardware_topology"]["copy_bandwidth_gb_s"]
    )
    result["hardware"]["load_bandwidth_gb_s"] = dict(
        spec["hardware_topology"]["recovery_bandwidth_gb_s"]
    )
    result.setdefault("runtime_overrides", {}).pop("topology", None)
    result["failure_model"]["policy_failover"] = {
        "replacement_delay_ms": spec["policy_failover_contract"]["replacement_delay_ms"],
        "fence_activation_delay_ms": spec["policy_failover_contract"]["fence_activation_delay_ms"],
        "policy_modes": deepcopy(spec["policy_failover_contract"]["policy_modes"]),
    }
    return result, capacity, spec


def node_fault_mechanism(policy: str) -> dict:
    base, capacity, _spec = protocol()
    tasks = [
        core.Task(
            request_id=f"r{rank}", workflow_id=f"w{rank}",
            workflow_arrival_ms=0.0, arrival_ms=0.0,
            source_rank=rank, lane=0, start_ms=0.0, finish_ms=10000.0,
            service_ms=10000.0, prompt_tokens=4096, output_tokens=256,
            predicted_slack_ms=0.0,
        )
        for rank in range(8)
    ]
    probe = core.FaultProbe(
        probe_id="node0", time_ms=500.0, source_rank=0,
        failure_type="source_node",
    )
    replay = v312.TopologyAwareExactAdmissionProjectionReplay(
        base, policy, capacity
    )
    _scheduled, delays, mechanism, _trace = replay.run_coupled(
        tasks, [probe], control_horizon_ms=1000.0
    )
    assert set(delays[0]) == {f"r{rank}" for rank in range(8)}
    return mechanism


def main() -> None:
    base, capacity, spec = protocol()
    checks: list[tuple[str, bool]] = []

    placements = [v312.topology_placement(base, rank) for rank in range(16)]
    checks.append((
        "topology_targets_exist_and_same_cross_are_distinct",
        all(0 <= row["source_node"] < 2 for row in placements)
        and all(0 <= row["peer_hbm_target_node"] < 2 for row in placements)
        and all(0 <= row["dram_target_node"] < 2 for row in placements)
        and all(
            (row["peer_hbm_target_node"] == row["source_node"])
            == (row["peer_hbm_mode"] == "same_node")
            for row in placements
        )
        and all(row["peer_hbm_target_rank"] != row["rank"] for row in placements)
        and all(
            (row["dram_target_node"] == row["source_node"])
            == (row["dram_mode"] == "local")
            for row in placements
        ),
    ))

    worker = core.FaultProbe("worker", 1.0, 9, "source_worker_gpu")
    node = core.FaultProbe("node", 1.0, 9, "source_node")
    checks.append((
        "fault_domains_are_exact_one_and_eight",
        v312.v31.failure_domain_ranks(worker, gpus_per_node=8, dp=16) == (9,)
        and v312.v31.failure_domain_ranks(node, gpus_per_node=8, dp=16)
        == tuple(range(8, 16)),
    ))

    native = node_fault_mechanism("Native Mooncake")
    checks.append((
        "native_node_fault_retains_modeled_replacement_boundary",
        native["failover_mode_counts"] == {"source_node:replacement_wait": 8}
        and native["active_hbm_v312"]["node_prebound_survivor_count"] == 0,
    ))

    topology_aware_ok = True
    for policy in ["Static FCFS", "DejaVu", "KevlarFlow", "GhostServe", "AegisKV"]:
        mechanism = node_fault_mechanism(policy)
        modes = mechanism["failover_mode_counts"]
        topology_aware_ok &= (
            modes.get("source_node:cross_domain_committed_fence_activation", 0) == 4
            and modes.get("source_node:replacement_no_survivor", 0) == 4
            and mechanism["active_hbm_v312"]["node_prebound_survivor_count"] == 4
            and mechanism["active_hbm_v312"]["node_replacement_no_survivor_count"] == 4
            and mechanism["active_hbm_v312"]["surviving_cross_domain_committed_count"] == 4
        )
    checks.append((
        "five_protection_architectures_use_only_committed_cross_domain_survivors",
        topology_aware_ok,
    ))

    checks.append((
        "directional_bandwidth_and_capacity_are_unchanged",
        base["hardware"]["copy_bandwidth_gb_s"]
        == spec["hardware_topology"]["copy_bandwidth_gb_s"]
        and base["hardware"]["load_bandwidth_gb_s"]
        == spec["hardware_topology"]["recovery_bandwidth_gb_s"]
        and capacity["hardware"]["total_hbm_bytes"] == 40 * 1024**3,
    ))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS")
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
