#!/usr/bin/env python3
"""Executable mechanism and conservation gates for v3.1.3."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import active_hbm_v313 as v313
import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def inputs(*, context: int = 40960, active: int = 16) -> tuple[dict, dict]:
    protocol = json.loads(
        (ROOT / "results/mechanism_pilot_v2/composite_protocol.json").read_text()
    )
    spec = json.loads((ROOT / "active_hbm_v311_s1_s2_protocol.json").read_text())
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["workload"]["center_context_tokens"] = int(context)
    protocol["workload"]["center_active_requests_per_gpu"] = int(active)
    protocol["hardware"]["gpus_per_node"] = 8
    protocol["hardware"]["node_count"] = 2
    protocol["hardware"]["copy_bandwidth_gb_s"] = dict(
        spec["hardware_topology"]["copy_bandwidth_gb_s"]
    )
    protocol["hardware"]["load_bandwidth_gb_s"] = dict(
        spec["hardware_topology"]["recovery_bandwidth_gb_s"]
    )
    protocol.setdefault("runtime_overrides", {}).pop("topology", None)
    protocol["failure_model"]["policy_failover"] = {
        "replacement_delay_ms": 60000.0,
        "fence_activation_delay_ms": 500.0,
        "policy_modes": deepcopy(spec["policy_failover_contract"]["policy_modes"]),
    }
    capacity = json.loads((ROOT / "active_hbm_v3_protocol.json").read_text())
    return protocol, capacity


def main() -> None:
    checks: list[tuple[str, bool]] = []
    protocol, capacity = inputs()
    placements = [v313.topology_placement(protocol, rank) for rank in range(16)]
    targets = [row["peer_hbm_target_rank"] for row in placements]
    checks.append((
        "peer_hbm_placement_is_bijective_and_never_self_targets",
        sorted(targets) == list(range(16))
        and all(row["peer_hbm_target_rank"] != row["rank"] for row in placements)
        and all(
            (row["peer_hbm_target_node"] == row["source_node"])
            == (row["peer_hbm_mode"] == "same_node")
            for row in placements
        ),
    ))

    task = core.Task(
        request_id="one", workflow_id="one", workflow_arrival_ms=0.0,
        arrival_ms=0.0, source_rank=0, lane=0, start_ms=0.0,
        finish_ms=10000.0, service_ms=10000.0, prompt_tokens=4096,
        output_tokens=256, predicted_slack_ms=0.0,
    )
    replay = v313.CausalPhysicalTargetReplay(protocol, "AegisKV", capacity)
    altered = replace(task, output_tokens=9999, service_ms=999999.0)
    checks.append((
        "seal_schedule_is_after_dispatch_and_ignores_realized_output_and_service",
        replay._seal_time(task, 4096) > task.start_ms
        and replay._seal_time(task, 4096) == replay._seal_time(altered, 4096),
    ))

    rows, _delays, mechanism, trace = replay.run_coupled(
        [task], [], control_horizon_ms=1000.0
    )
    accounting = mechanism["active_hbm_v313"]
    expected_bytes = 4096 * int(protocol["models"]["qwen3_8b"]["kv_bytes_per_token"])
    hbm_ms = replay.copy_accounting_by_request["one"]["HBM"]["service_ms"]
    dram_ms = replay.copy_accounting_by_request["one"]["DRAM"]["service_ms"]
    checks.append((
        "independent_paths_use_exact_bytes_and_parallel_max_not_serial_sum",
        accounting["scheduled_copy_bytes"] == {"HBM": expected_bytes, "DRAM": expected_bytes}
        and accounting["copy_byte_conservation_error"] == {"HBM": 0, "DRAM": 0}
        and replay.copy_service_ms["one"] == max(hbm_ms, dram_ms)
        and replay.copy_service_ms["one"] < hbm_ms + dram_ms
        and accounting["copy_duration_formula_max_abs_error_ms"] <= 1e-12,
    ))
    checks.append((
        "asynchronous_copy_not_directly_added_to_serving_finish",
        rows[0].finish_ms == task.service_ms
        and replay.health_burden_ms["one"] == 0.0
        and accounting["health_burden_application_count"] == 0,
    ))
    checks.append((
        "physical_target_capacity_and_logical_byte_conservation",
        accounting["topology_bijective"]
        and accounting["target_accounting_max_abs_error_bytes"] == 0
        and accounting["target_capacity_violation_count"] == 0
        and all(
            int(row["shared_used_bytes"]) <= int(replay.shared_budget_bytes)
            for row in trace
        )
        and accounting["physical_target_peak_backup_max_bytes"]
        <= accounting["physical_target_hbm_capacity_bytes"],
    ))

    burst_tasks, _meta = core.build_workload(
        protocol, seed=2026082600, workflows=None, active_per_gpu=16,
        arrival_cutoff_ms=1000.0,
    )
    healthy = {}
    mechanisms = {}
    for policy in ("Native Mooncake", "AegisKV"):
        current = v313.CausalPhysicalTargetReplay(deepcopy(protocol), policy, capacity)
        scheduled, _delays, current_mechanism, _trace = current.run_coupled(
            burst_tasks, [], control_horizon_ms=1000.0
        )
        healthy[policy] = {row.request_id: (row.start_ms, row.finish_ms) for row in scheduled}
        mechanisms[policy] = current_mechanism
    checks.append((
        "future_unsealed_backup_never_blocks_its_source_request",
        healthy["Native Mooncake"] == healthy["AegisKV"]
        and mechanisms["AegisKV"]["active_hbm_v313"]["primary_priority_reclaimed_bytes"] > 0
        and mechanisms["AegisKV"]["active_hbm_v313"]["future_backup_projection_dispatch_count"] == 0,
    ))

    reclaim = v313.CausalPhysicalTargetReplay(protocol, "AegisKV", capacity)
    reclaim.task_by_id = {task.request_id: task}
    reclaim.active.add(task.request_id)
    reclaim.start(task, 0.0)
    seal_at = reclaim._seal_time(task, task.prompt_tokens)
    reclaim.handle_protection_seal((task.request_id, 0, 0, task.prompt_tokens), seal_at)
    hbm_done = reclaim.aegis_lane[(0, core.ProtectionTier.HBM)]
    reclaim.complete_copy(
        (0, task.request_id, 0, core.ProtectionTier.HBM), hbm_done
    )
    before_prefix = reclaim.aegis[0].committed_prefix_tokens(
        task.request_id, core.ProtectionTier.HBM
    )
    target = v313.topology_placement(protocol, 0)["peer_hbm_target_rank"]
    reclaim._enforce_target_headroom(
        target, hbm_done, serving_bytes=reclaim.shared_budget_bytes
    )
    after_prefix = reclaim.aegis[0].committed_prefix_tokens(
        task.request_id, core.ProtectionTier.HBM
    )
    _delay, recovery_action = reclaim.recovery(
        task, core.FaultProbe("after-reclaim", hbm_done, 0, "source_worker_gpu")
    )
    checks.append((
        "reclaimed_replica_is_not_recoverable_as_committed_prefix",
        before_prefix == task.prompt_tokens
        and after_prefix == 0
        and not recovery_action.startswith("peer_hbm"),
    ))

    decomposition = v313.workflow_decomposition(rows)
    checks.append((
        "healthy_wct_accounting_identity",
        decomposition["identity_max_abs_error_ms"] <= 1e-7
        and abs(decomposition["critical_direct_copy_burden_mean_ms"]) <= 1e-7,
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
