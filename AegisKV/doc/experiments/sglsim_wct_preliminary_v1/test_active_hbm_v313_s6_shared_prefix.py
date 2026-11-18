#!/usr/bin/env python3
"""Executable mechanism gates for explicit S6 shared-prefix semantics."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import active_hbm_v313 as v313
import active_hbm_v313_s6 as s6
import run_preliminary as core
import test_active_hbm_v313 as v313_tests


ROOT = Path(__file__).resolve().parent


def inputs() -> tuple[dict, dict]:
    protocol, capacity = v313_tests.inputs(context=4096, active=4)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["hardware"]["gpus_per_node"] = 8
    protocol["hardware"]["node_count"] = 2
    protocol.setdefault("runtime_overrides", {})["topology"] = {
        "peer_hbm": "same_node", "dram": "local"
    }
    return protocol, capacity


def tasks(count: int = 4) -> list[core.Task]:
    return [
        core.Task(
            request_id=f"request-{index}", workflow_id=f"workflow-{index}",
            workflow_arrival_ms=0.0, arrival_ms=0.0, source_rank=0,
            lane=0, start_ms=0.0, finish_ms=20000.0,
            service_ms=20000.0, prompt_tokens=4096, output_tokens=256,
            predicted_slack_ms=0.0,
        )
        for index in range(count)
    ]


def run_reuse(reuse: float, *, with_probe: bool = True):
    protocol, capacity = inputs()
    nominal = tasks()
    assignments = s6.build_prefix_assignments(
        nominal, reuse_fraction=reuse, group_size=4, chunk_tokens=4096
    )
    replay = s6.SharedPrefixReplay(
        deepcopy(protocol), "AegisKV", capacity,
        assignments=assignments, reuse_fraction=reuse, group_size=4,
    )
    probes = (
        [core.FaultProbe("probe", 5000.0, 0, "source_worker_gpu")]
        if with_probe else []
    )
    scheduled, delay_maps, mechanism, trace = replay.run_coupled(
        nominal, probes, control_horizon_ms=10000.0
    )
    return protocol, capacity, assignments, replay, scheduled, delay_maps, mechanism, trace


def main() -> None:
    checks: list[tuple[str, bool]] = []
    nominal = [replace(row, prompt_tokens=8192) for row in tasks(8)]
    half = s6.build_prefix_assignments(
        nominal, reuse_fraction=0.5, group_size=4, chunk_tokens=4096
    )
    half_again = s6.build_prefix_assignments(
        list(reversed(nominal)), reuse_fraction=0.5,
        group_size=4, chunk_tokens=4096,
    )
    checks.append((
        "identity_is_deterministic_same_source_and_chunk_aligned",
        s6.assignment_hash(half) == s6.assignment_hash(half_again)
        and len({row.group_id for row in half.values()}) == 2
        and all(row.source_rank == 0 for row in half.values())
        and all(row.shared_tokens == 4096 for row in half.values()),
    ))
    try:
        s6.build_prefix_assignments(
            tasks(4), reuse_fraction=0.5, group_size=4, chunk_tokens=4096
        )
        half_chunk_gate = False
    except ValueError:
        half_chunk_gate = True
    checks[-1] = (
        "identity_is_deterministic_same_source_and_chunk_aligned",
        s6.assignment_hash(half) == s6.assignment_hash(half_again)
        and len({row.group_id for row in half.values()}) == 2
        and all(row.shared_tokens == 4096 for row in half.values())
        and half_chunk_gate,
    )

    protocol, _capacity, assignments, _replay, _scheduled, _delay_maps, mechanism, _trace = run_reuse(0.0)
    logical = 4 * 4096 * int(protocol["models"]["qwen3_8b"]["kv_bytes_per_token"])
    zero = mechanism["active_hbm_v313_s6"]
    checks.append((
        "reuse_zero_has_no_shared_identity_and_one_copy_per_tier_per_request",
        zero["shared_prefix_group_count"] == 0
        and zero["shared_reference_count"] == 0
        and zero["logical_useful_bytes"] == logical
        and zero["scheduled_physical_copy_bytes"] == {
            "HBM": logical, "DRAM": logical
        }
        and zero["theoretical_copy_ratio_per_tier"] == 1.0,
    ))

    protocol, capacity, assignments, replay, scheduled, delay_maps, mechanism, trace = run_reuse(1.0)
    full = mechanism["active_hbm_v313_s6"]
    one_prompt = 4096 * int(protocol["models"]["qwen3_8b"]["kv_bytes_per_token"])
    checks.append((
        "reuse_one_copies_one_physical_prefix_for_four_logical_requests",
        full["shared_prefix_group_count"] == 1
        and full["shared_physical_chunk_identity_count"] == 1
        and full["shared_reference_count"] == 4
        and full["shared_reference_alias_hit_count"] == 3
        and full["scheduled_physical_copy_bytes"] == {
            "HBM": one_prompt, "DRAM": one_prompt
        }
        and full["theoretical_copy_ratio_per_tier"] == 0.25,
    ))
    checks.append((
        "logical_and_physical_copy_bytes_conserve_independently",
        full["logical_useful_bytes"]
        == full["logical_shared_reference_bytes"]
        + full["logical_unique_reference_bytes"]
        and full["scheduled_shared_physical_copy_bytes"]
        == full["scheduled_physical_copy_bytes"]
        and mechanism["active_hbm_v313"]["copy_byte_conservation_error"]
        == {"HBM": 0, "DRAM": 0},
    ))
    checks.append((
        "refcounts_increment_and_drain_exactly",
        full["refcount_peak_total"] == 4
        and full["refcount_final_total"] == 0
        and full["refcount_max_abs_error"] == 0
        and full["refcount_underflow_count"] == 0,
    ))
    checks.append((
        "same_probe_recovery_uses_shared_committed_prefix",
        len(delay_maps) == 1
        and len(delay_maps[0]) == 4
        and full["recovery_call_count"] == 4
        and full["recovery_request_hit_count"] == 4
        and full["shared_recovery_restored_tokens"] == 4 * 4096
        and full["unique_recovery_restored_tokens"] == 0
        and full["recovery_prefix_token_hit_fraction"] == 1.0,
    ))
    checks.append((
        "physical_hbm_dram_occupancy_uses_byte_time_integrals",
        full["physical_capacity_integral_byte_ms"]["HBM"] > 0
        and full["physical_capacity_integral_byte_ms"]["DRAM"] > 0
        and full["physical_occupancy_time_mean_gib_per_gpu"]["HBM"] > 0
        and full["physical_occupancy_time_mean_gib_per_gpu"]["DRAM"] > 0
        and full["physical_occupancy_trace_change_points"] >= 2,
    ))
    checks.append((
        "shared_copy_never_changes_serving_wct_directly",
        all(row.start_ms == 0.0 and row.finish_ms == row.service_ms for row in scheduled)
        and mechanism["active_hbm_v313"]["health_burden_application_count"] == 0,
    ))
    checks.append((
        "physical_target_capacity_and_no_future_member_oracle",
        full["future_member_count_used_for_admission"] == 0
        and mechanism["active_hbm_v313"]["target_capacity_violation_count"] == 0
        and mechanism["active_hbm_v313"]["future_backup_projection_dispatch_count"] == 0
        and all(
            int(row["shared_used_bytes"]) <= int(replay.shared_budget_bytes)
            for row in trace
        ),
    ))

    protocol, capacity = inputs()
    nominal = tasks(1)
    assignments = s6.build_prefix_assignments(
        nominal, reuse_fraction=1.0, group_size=4, chunk_tokens=4096
    )
    reclaim = s6.SharedPrefixReplay(
        protocol, "AegisKV", capacity, assignments=assignments,
        reuse_fraction=1.0, group_size=4,
    )
    task = nominal[0]
    reclaim.task_by_id = {task.request_id: task}
    reclaim.occupancy_observation_horizon_ms = 10000.0
    reclaim.start(task, 0.0)
    seal_at = reclaim._seal_time(task, task.prompt_tokens)
    reclaim.handle_protection_seal(
        ("shared", task.request_id, 0, 0, task.prompt_tokens), seal_at
    )
    assignment = assignments[task.request_id]
    owner = reclaim.shared_owner_by_group[assignment.group_id]
    hbm_done = reclaim.aegis_lane[(0, core.ProtectionTier.HBM)]
    reclaim.complete_copy((0, owner, 0, core.ProtectionTier.HBM), hbm_done)
    before = reclaim.aegis[0].committed_prefix_tokens(
        owner, core.ProtectionTier.HBM
    )
    target = v313.topology_placement(protocol, 0)["peer_hbm_target_rank"]
    reclaim._enforce_target_headroom(
        target, hbm_done, serving_bytes=reclaim.shared_budget_bytes
    )
    after = reclaim.aegis[0].committed_prefix_tokens(
        owner, core.ProtectionTier.HBM
    )
    _delay, action = reclaim.recovery(
        task, core.FaultProbe("reclaim", hbm_done, 0, "source_worker_gpu")
    )
    checks.append((
        "evicted_shared_replica_is_not_recoverable",
        before == task.prompt_tokens and after == 0
        and not action.startswith("peer_hbm"),
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
