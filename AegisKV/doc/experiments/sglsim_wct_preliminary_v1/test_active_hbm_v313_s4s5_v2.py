#!/usr/bin/env python3
"""Executable gates for mechanism-targeted S4/S5 v2 extensions."""

from __future__ import annotations

from copy import deepcopy

import active_hbm_v313_s4s5_v2 as ext
import run_preliminary as core
import test_active_hbm_v313 as base_test


def task(name: str, rank: int) -> core.Task:
    return core.Task(
        request_id=name, workflow_id=name, workflow_arrival_ms=0.0,
        arrival_ms=0.0, source_rank=rank, lane=0, start_ms=0.0,
        finish_ms=10000.0, service_ms=10000.0, prompt_tokens=4096,
        output_tokens=256, predicted_slack_ms=0.0,
    )


def observed_outcome(replay: ext.TargetedReplay, request_id: str) -> str:
    row = replay.copy_accounting_by_request.get(request_id, {})
    hbm = bool(row.get("HBM", {}).get("bytes", 0))
    dram = bool(row.get("DRAM", {}).get("bytes", 0))
    if hbm and dram:
        return "HBM_AND_DRAM"
    if hbm:
        return "HBM_ONLY"
    if dram:
        return "DRAM_ONLY"
    return "BYPASS"


def main() -> None:
    checks: list[tuple[str, bool]] = []
    protocol, capacity = base_test.inputs(context=16384, active=1)
    per_request = (16384 + 4096) * 147456
    shared = int(
        capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]
    )
    max_quota = int(0.30 * 40 * 1024**3)
    checks.append((
        "analytic_recovery_capacity_ceiling_is_two_requests",
        per_request == int(2.8125 * 1024**3)
        and (shared - max_quota) // per_request == 2,
    ))

    rows = [
        task("both", 0), task("hbm", 1), task("dram", 2), task("neither", 3)
    ]
    masks = {
        "both": (core.ProtectionTier.HBM, core.ProtectionTier.DRAM),
        "hbm": (core.ProtectionTier.HBM,),
        "dram": (core.ProtectionTier.DRAM,),
        "neither": (),
    }
    full = ext.BenefitMaskedReplay(
        deepcopy(protocol), "AegisKV", capacity,
        allowed_tiers_by_request=masks,
    )
    full.run_coupled(rows, [], control_horizon_ms=1000.0)
    outcomes = {name: observed_outcome(full, name) for name in masks}
    checks.append((
        "full_independent_tiers_cover_four_preregistered_outcomes",
        outcomes == {
            "both": "HBM_AND_DRAM", "hbm": "HBM_ONLY",
            "dram": "DRAM_ONLY", "neither": "BYPASS",
        },
    ))

    no2 = ext.BenefitMaskedReplay(
        deepcopy(protocol), "AegisKV no Algorithm 2", capacity,
        allowed_tiers_by_request=masks,
    )
    no2.run_coupled(rows, [], control_horizon_ms=1000.0)
    no2_outcomes = {name: observed_outcome(no2, name) for name in masks}
    checks.append((
        "coupled_ablation_keeps_only_both_or_neither",
        no2_outcomes == {
            "both": "HBM_AND_DRAM", "hbm": "BYPASS",
            "dram": "BYPASS", "neither": "BYPASS",
        },
    ))

    traced = ext.TargetedReplay(deepcopy(protocol), "AegisKV", capacity)
    _tasks, _delays, mechanism, _capacity = traced.run_coupled(
        [task("trace", 0)], [], control_horizon_ms=30000.0
    )
    target_trace = mechanism["active_hbm_v313_s4s5_v2"]["target_control_trace"]
    checks.append((
        "control_window_persists_all_sixteen_physical_targets",
        len(target_trace) == 16
        and {row["physical_target_rank"] for row in target_trace} == set(range(16))
        and all("stable_windows_after" in row for row in target_trace),
    ))

    no3 = ext.TargetedReplay(
        deepcopy(protocol), "AegisKV no Algorithm 3", capacity
    )
    _tasks, _delays, no3_mechanism, _capacity = no3.run_coupled(
        [task("no3", 0)], [], control_horizon_ms=30000.0
    )
    checks.append((
        "algorithm3_ablation_still_executes_zero_target_updates",
        no3_mechanism["active_hbm_v313"]["scheduled_control_opportunity_times_ms"] == [30000.0]
        and no3_mechanism["active_hbm_v313"]["executed_policy_control_times_ms"] == []
        and no3_mechanism["active_hbm_v313_s4s5_v2"]["target_control_trace"] == [],
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
