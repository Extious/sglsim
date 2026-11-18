#!/usr/bin/env python3
"""Executable zero/single/multi/duplicate/survivor gates for rate injection."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import active_hbm_v312 as v312
import poisson_fault_injection_v1 as injected
import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def protocol() -> tuple[dict, dict]:
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
        "replacement_delay_ms": 60000.0,
        "fence_activation_delay_ms": 500.0,
        "policy_modes": deepcopy(spec["policy_failover_contract"]["policy_modes"]),
    }
    return result, capacity


def tasks() -> list[core.Task]:
    return [
        core.Task(
            request_id=f"r{rank}", workflow_id=f"w{rank}",
            workflow_arrival_ms=0.0, arrival_ms=0.0,
            source_rank=rank, lane=0, start_ms=0.0, finish_ms=10000.0,
            service_ms=10000.0, prompt_tokens=4096, output_tokens=256,
            predicted_slack_ms=0.0,
        )
        for rank in range(16)
    ]


def run(events: list[dict], policy: str = "AegisKV") -> tuple[list[core.Task], dict]:
    base, capacity = protocol()
    replay = injected.PoissonFaultInjectedReplay(base, policy, capacity)
    rows, mechanism, _capacity = replay.run_rate_injected(
        tasks(), events, control_horizon_ms=1000.0
    )
    return rows, mechanism


def main() -> None:
    checks: list[tuple[str, bool]] = []
    zero = injected.generate_poisson_fault_ledger(
        rate_per_1000_gpu_shard_hour=0.0,
        seed=1,
        horizon_ms=90000.0,
        dp=16,
        gpus_per_node=8,
        repair_ms=60000.0,
        failure_type_weights={"source_worker_gpu": 0.5, "source_node": 0.5},
    )
    healthy, healthy_mechanism = run(zero)
    healthy_replay = core.replay_one_fault(healthy, {})
    checks.append((
        "zero_event_trace_is_exact_healthy_des",
        zero == []
        and healthy_mechanism["poisson_fault_injection_v1"]["accepted_event_count"] == 0
        and all(
            abs(healthy_replay[row.request_id] - row.finish_ms) <= 1e-8
            for row in healthy
        ),
    ))

    one = injected.resolve_fault_candidates(
        [injected.FaultCandidate(500.0, "source_worker_gpu", 0)],
        dp=16, gpus_per_node=8, repair_ms=60000.0,
    )
    single_rows, single_mechanism = run(one)
    checks.append((
        "single_event_is_applied_once_and_increases_wct",
        len(one) == 1 and one[0]["accepted"]
        and single_mechanism["poisson_fault_injection_v1"]["accepted_event_count"] == 1
        and next(row for row in single_rows if row.request_id == "r0").finish_ms > 10000.0
        and all(row.finish_ms >= 10000.0 for row in single_rows),
    ))

    multi = injected.resolve_fault_candidates(
        [
            injected.FaultCandidate(500.0, "source_worker_gpu", 0),
            injected.FaultCandidate(700.0, "source_worker_gpu", 9),
            injected.FaultCandidate(900.0, "source_node", 0),
        ],
        dp=16, gpus_per_node=8, repair_ms=60000.0,
    )
    multi_rows, multi_mechanism = run(multi)
    checks.append((
        "multiple_distinct_events_update_routes_queue_recovery_and_capacity",
        sum(bool(row["accepted"]) for row in multi) == 3
        and multi_mechanism["poisson_fault_injection_v1"]["accepted_event_count"] == 3
        and len(multi_mechanism["poisson_fault_injection_v1"]["event_ledger"]) == 3
        and all(
            "capacity_shared_used_bytes_after" in row
            and "queue_count_after" in row
            and "recovery_actions" in row
            for row in multi_mechanism["poisson_fault_injection_v1"]["event_ledger"]
        )
        and all(row.finish_ms >= 10000.0 for row in multi_rows),
    ))

    duplicate = injected.resolve_fault_candidates(
        [
            injected.FaultCandidate(500.0, "source_worker_gpu", 0),
            injected.FaultCandidate(600.0, "source_worker_gpu", 0),
        ],
        dp=16, gpus_per_node=8, repair_ms=60000.0,
    )
    duplicate_rows, duplicate_mechanism = run(duplicate)
    single_finish = {row.request_id: row.finish_ms for row in single_rows}
    duplicate_finish = {row.request_id: row.finish_ms for row in duplicate_rows}
    checks.append((
        "same_domain_repeat_is_ledgered_noop_without_double_damage",
        [row["accepted"] for row in duplicate] == [True, False]
        and duplicate[1]["outcome"] == "NOOP_WORKER_ALREADY_DOWN"
        and duplicate_mechanism["poisson_fault_injection_v1"]["noop_event_count"] == 1
        and duplicate_finish == single_finish,
    ))

    node = injected.resolve_fault_candidates(
        [injected.FaultCandidate(500.0, "source_node", 0)],
        dp=16, gpus_per_node=8, repair_ms=60000.0,
    )
    _node_rows, node_mechanism = run(node)
    ledger = node_mechanism["poisson_fault_injection_v1"]["event_ledger"][0]
    actions = ledger["recovery_actions"]
    checks.append((
        "node_fault_loses_local_and_preserves_cross_domain_committed_replicas",
        ledger["affected_request_count"] == 8
        and ledger["surviving_replica_count"] > 0
        and any("cross_node" in key or "remote" in key for key in actions)
        and any("RECOMPUTE" == key for key in actions)
        and node_mechanism["active_hbm_v3"]["capacity_violation_count"] == 0,
    ))

    placements = [v312.topology_placement(protocol()[0], rank) for rank in range(16)]
    checks.append((
        "same_node_peer_targets_another_gpu_and_cross_targets_other_node",
        all(row["peer_hbm_target_rank"] != row["rank"] for row in placements)
        and all(
            row["peer_hbm_target_node"] != row["source_node"]
            for row in placements if row["peer_hbm_mode"] == "cross_node"
        ),
    ))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}")
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS")
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
