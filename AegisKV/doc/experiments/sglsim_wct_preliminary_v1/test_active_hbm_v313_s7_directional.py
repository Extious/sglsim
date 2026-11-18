#!/usr/bin/env python3
"""Executable mechanism gates for the v3.1.3 directional S7 sweep."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import active_hbm_v313 as v313
import run_preliminary as core
import test_active_hbm_v313 as v313_tests


DIRECTIONS = {
    "peer_hbm_same_node": ("same_node", "local"),
    "dram_local": ("same_node", "local"),
    "peer_hbm_cross_node": ("cross_node", "remote"),
    "dram_remote": ("cross_node", "remote"),
}


def protocol_for(
    *, direction: str, phase: str, bandwidth: float,
    replacement_ms: float = 60000.0,
) -> tuple[dict, dict]:
    protocol, capacity = v313_tests.inputs(context=16384, active=4)
    peer, dram = DIRECTIONS[direction]
    protocol.setdefault("runtime_overrides", {})["topology"] = {
        "peer_hbm": peer, "dram": dram,
    }
    key = "copy_bandwidth_gb_s" if phase == "copy" else "load_bandwidth_gb_s"
    protocol["hardware"][key][direction] = float(bandwidth)
    protocol["failure_model"]["policy_failover"]["replacement_delay_ms"] = float(
        replacement_ms
    )
    return protocol, capacity


def one_task() -> core.Task:
    return core.Task(
        request_id="one", workflow_id="one", workflow_arrival_ms=0.0,
        arrival_ms=0.0, source_rank=0, lane=0, start_ms=0.0,
        finish_ms=10000.0, service_ms=10000.0, prompt_tokens=4096,
        output_tokens=256, predicted_slack_ms=0.0,
    )


def committed_replay(
    *, topology: str, policy: str = "AegisKV", replacement_ms: float = 60000.0,
) -> tuple[v313.CausalPhysicalTargetReplay, core.Task, float]:
    direction = (
        "peer_hbm_same_node" if topology == "same_node"
        else "peer_hbm_cross_node"
    )
    protocol, capacity = protocol_for(
        direction=direction, phase="recovery", bandwidth=25.0,
        replacement_ms=replacement_ms,
    )
    task = one_task()
    replay = v313.CausalPhysicalTargetReplay(deepcopy(protocol), policy, capacity)
    replay.task_by_id = {task.request_id: task}
    replay.active.add(task.request_id)
    replay.start(task, 0.0)
    seal_at = replay._seal_time(task, task.prompt_tokens)
    replay.handle_protection_seal((task.request_id, 0, 0, task.prompt_tokens), seal_at)
    if policy == "AegisKV":
        completions = sorted(
            (replay.aegis_lane[(0, tier)], tier)
            for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM)
            if (task.request_id, 0, tier) in replay.aegis_specs
        )
        for done, tier in completions:
            replay.complete_copy((0, task.request_id, 0, tier), done)
        ready = max(replay.aegis_lane[(0, core.ProtectionTier.HBM)], replay.aegis_lane[(0, core.ProtectionTier.DRAM)])
    else:
        # Baseline runtime seal_chunk commits synchronously at its copy lane.
        ready = max(runtime.lane_available_ms for runtime in replay.baselines[0])
    return replay, task, ready


def main() -> None:
    checks: list[tuple[str, bool]] = []

    # A phase sweep must change only the registered table and direction.
    p0, _ = protocol_for(
        direction="peer_hbm_cross_node", phase="copy", bandwidth=1.0
    )
    p1, _ = protocol_for(
        direction="peer_hbm_cross_node", phase="copy", bandwidth=50.0
    )
    checks.append((
        "copy_sweep_changes_only_copy_direction",
        p0["hardware"]["load_bandwidth_gb_s"] == p1["hardware"]["load_bandwidth_gb_s"]
        and p0["hardware"]["copy_bandwidth_gb_s"]["peer_hbm_cross_node"] == 1.0
        and p1["hardware"]["copy_bandwidth_gb_s"]["peer_hbm_cross_node"] == 50.0
        and all(
            p0["hardware"]["copy_bandwidth_gb_s"][key]
            == p1["hardware"]["copy_bandwidth_gb_s"][key]
            for key in p0["hardware"]["copy_bandwidth_gb_s"]
            if key != "peer_hbm_cross_node"
        ),
    ))
    p0, _ = protocol_for(
        direction="dram_local", phase="recovery", bandwidth=1.0
    )
    p1, _ = protocol_for(
        direction="dram_local", phase="recovery", bandwidth=50.0
    )
    checks.append((
        "recovery_sweep_changes_only_recovery_direction",
        p0["hardware"]["copy_bandwidth_gb_s"] == p1["hardware"]["copy_bandwidth_gb_s"]
        and p0["hardware"]["load_bandwidth_gb_s"]["dram_local"] == 1.0
        and p1["hardware"]["load_bandwidth_gb_s"]["dram_local"] == 50.0
        and all(
            p0["hardware"]["load_bandwidth_gb_s"][key]
            == p1["hardware"]["load_bandwidth_gb_s"][key]
            for key in p0["hardware"]["load_bandwidth_gb_s"]
            if key != "dram_local"
        ),
    ))

    # Exact duration uses copied bytes and the copy table, while recovery uses
    # the independently configured load table.
    copy_service = []
    for bandwidth in (1.0, 50.0):
        protocol, capacity = protocol_for(
            direction="peer_hbm_same_node", phase="copy", bandwidth=bandwidth
        )
        task = one_task()
        replay = v313.CausalPhysicalTargetReplay(protocol, "AegisKV", capacity)
        replay.task_by_id = {task.request_id: task}
        replay.active.add(task.request_id)
        replay.start(task, 0.0)
        at = replay._seal_time(task, task.prompt_tokens)
        replay.handle_protection_seal((task.request_id, 0, 0, task.prompt_tokens), at)
        copy_service.append(replay.copy_accounting_by_request["one"]["HBM"]["service_ms"])
    expected_ratio = (
        0.05 + 4096 * 147456 / (1.0 * core.GB_PER_S_TO_BYTES_PER_MS)
    ) / (
        0.05 + 4096 * 147456 / (50.0 * core.GB_PER_S_TO_BYTES_PER_MS)
    )
    checks.append((
        "copy_duration_uses_exact_directional_bytes",
        abs(copy_service[0] / copy_service[1] - expected_ratio) <= 1e-12,
    ))

    recovery_values = []
    # At 1 GB/s the preregistered controller may correctly reject Peer-HBM as
    # having no positive recovery benefit.  Use two admitted points here so
    # this unit gate isolates load-service accounting rather than admission.
    for bandwidth in (5.0, 50.0):
        protocol, capacity = protocol_for(
            direction="peer_hbm_same_node", phase="recovery", bandwidth=bandwidth
        )
        # Make DRAM recovery slower so the Peer-HBM path remains selected.
        protocol["hardware"]["load_bandwidth_gb_s"]["dram_local"] = 0.5
        task = one_task()
        replay = v313.CausalPhysicalTargetReplay(protocol, "AegisKV", capacity)
        replay.task_by_id = {task.request_id: task}
        replay.active.add(task.request_id)
        replay.start(task, 0.0)
        seal_at = replay._seal_time(task, task.prompt_tokens)
        replay.handle_protection_seal((task.request_id, 0, 0, task.prompt_tokens), seal_at)
        completions = sorted(
            (replay.aegis_lane[(0, tier)], tier)
            for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM)
            if (task.request_id, 0, tier) in replay.aegis_specs
        )
        for done, tier in completions:
            replay.complete_copy((0, task.request_id, 0, tier), done)
        probe = core.FaultProbe("worker", 5000.0, 0, "source_worker_gpu")
        delay, action = replay.recovery(task, probe)
        recovery_values.append((delay, action))
    checks.append((
        "recovery_duration_uses_load_not_copy_table",
        recovery_values[0][0] > recovery_values[1][0]
        and all(row[1] == "peer_hbm_same_node" for row in recovery_values),
    ))

    same, task, ready = committed_replay(topology="same_node", replacement_ms=30000.0)
    same_delay, same_action = same.recovery(
        task, core.FaultProbe("node", ready + 1.0, 0, "source_node")
    )
    checks.append((
        "source_node_invalidates_exact_same_node_replicas",
        same.node_replacement_no_survivor_count == 1
        and same.node_prebound_survivor_count == 0
        and same.failover_delay_values_ms[-1] == 30000.0
        and same_action == "RECOMPUTE"
        and same_delay > 30000.0,
    ))

    cross, task, ready = committed_replay(topology="cross_node", replacement_ms=30000.0)
    cross_delay, cross_action = cross.recovery(
        task, core.FaultProbe("node", ready + 1.0, 0, "source_node")
    )
    checks.append((
        "source_node_preserves_cross_node_committed_replicas",
        cross.node_prebound_survivor_count == 1
        and cross.node_replacement_no_survivor_count == 0
        and cross.surviving_cross_domain_committed_count == 1
        and cross.failover_delay_values_ms[-1] == 500.0
        and (cross_action.startswith("peer_hbm_cross_node") or cross_action.startswith("dram_remote"))
        and cross_delay < 30000.0,
    ))

    native, task, ready = committed_replay(
        topology="cross_node", policy="Native Mooncake", replacement_ms=10000.0
    )
    native_delay, _native_action = native.recovery(
        task, core.FaultProbe("node", ready + 1.0, 0, "source_node")
    )
    checks.append((
        "modeled_native_waits_for_registered_replacement",
        native.failover_delay_values_ms[-1] == 10000.0
        and native_delay >= 10000.0,
    ))

    base, capacity = v313_tests.inputs(context=16384, active=4)
    task = one_task()
    replay = v313.CausalPhysicalTargetReplay(base, "AegisKV", capacity)
    rows, delays, mechanism, trace = replay.run_coupled(
        [replace(task, finish_ms=task.service_ms)], [], control_horizon_ms=1000.0
    )
    checks.append((
        "v313_capacity_copy_and_no_oracle_gates_remain_active",
        len(rows) == 1 and delays == []
        and mechanism["active_hbm_v313"]["copy_byte_conservation_error"] == {"HBM": 0, "DRAM": 0}
        and mechanism["active_hbm_v313"]["target_capacity_violation_count"] == 0
        and mechanism["active_hbm_v313"]["future_backup_projection_dispatch_count"] == 0
        and mechanism["active_hbm_v313"]["seal_uses_realized_output_or_service"] is False
        and all(int(row["shared_used_bytes"]) <= replay.shared_budget_bytes for row in trace),
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
