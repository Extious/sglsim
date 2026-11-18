#!/usr/bin/env python3
"""Executable hard gates for the v3.1.3 S4/S5 extension."""

from __future__ import annotations

from copy import deepcopy

import active_hbm_v313_s4s5 as ext
import run_preliminary as core
import test_active_hbm_v313 as base_test


def one_task(*, slack_ms: float = 123.0) -> core.Task:
    return core.Task(
        request_id="one", workflow_id="one", workflow_arrival_ms=0.0,
        arrival_ms=0.0, source_rank=0, lane=0, start_ms=0.0,
        finish_ms=10000.0, service_ms=10000.0, prompt_tokens=4096,
        output_tokens=256, predicted_slack_ms=slack_ms,
    )


def main() -> None:
    checks: list[tuple[str, bool]] = []
    protocol, capacity = base_test.inputs(context=4096, active=2)
    expected = {
        "Native Mooncake", "AegisKV", "AegisKV no Algorithm 1",
        "AegisKV no Algorithm 2", "AegisKV no Algorithm 3",
    }
    checks.append((
        "exact_five_s4_policies_are_supported",
        ext.S4S5Replay.SUPPORTED_POLICIES == expected,
    ))

    full = ext.S4S5Replay(deepcopy(protocol), "AegisKV", capacity)
    no1 = ext.S4S5Replay(
        deepcopy(protocol), "AegisKV no Algorithm 1", capacity
    )
    checks.append((
        "algorithm1_ablation_only_zeroes_slack_metadata",
        full._profile(one_task()).slack_ms == 123.0
        and no1._profile(one_task()).slack_ms == 0.0
        and type(full.aegis[0]) is type(no1.aegis[0]),
    ))

    no2 = ext.S4S5Replay(
        deepcopy(protocol), "AegisKV no Algorithm 2", capacity
    )
    rows, _delays, mechanism, _trace = no2.run_coupled(
        [one_task()], [], control_horizon_ms=1000.0
    )
    ablation = mechanism["active_hbm_v313_s4s5"]
    checks.append((
        "algorithm2_ablation_keeps_aegis_and_admits_both_or_neither",
        no2.is_aegis
        and isinstance(no2.aegis[0], ext.CoupledFCFSTierAblationController)
        and mechanism["residency_event_counts"] == {"HBM_AND_DRAM": 1}
        and ablation["coupled_tier_attempt_count"] == 1
        and ablation["coupled_tier_success_count"] == 1
        and rows[0].finish_ms == one_task().service_ms,
    ))

    no2_reject = ext.S4S5Replay(
        deepcopy(protocol), "AegisKV no Algorithm 2", capacity
    )
    no2_reject.aegis[0].set_pool_limit(core.ProtectionTier.DRAM, 0)
    _rows, _delays, rejected_mechanism, _trace = no2_reject.run_coupled(
        [one_task()], [], control_horizon_ms=1000.0
    )
    rejected = rejected_mechanism["active_hbm_v313_s4s5"]
    checks.append((
        "algorithm2_ablation_does_not_leave_single_tier_partial_admission",
        rejected_mechanism["residency_event_counts"] == {"BYPASS": 1}
        and rejected["coupled_tier_rejection_count"] == 1
        and rejected_mechanism["active_hbm_v313"]["scheduled_copy_bytes"]
        == {"HBM": 0, "DRAM": 0},
    ))

    no3 = ext.S4S5Replay(
        deepcopy(protocol), "AegisKV no Algorithm 3", capacity
    )
    _rows, _delays, no3_mechanism, _trace = no3.run_coupled(
        [one_task()], [], control_horizon_ms=30000.0
    )
    checks.append((
        "algorithm3_ablation_keeps_scheduled_window_but_executes_no_update",
        no3_mechanism["active_hbm_v313"][
            "scheduled_control_opportunity_times_ms"
        ] == [30000.0]
        and no3_mechanism["active_hbm_v313"][
            "executed_policy_control_times_ms"
        ] == []
        and no3_mechanism["active_hbm_v313_s4s5"]["rich_control_trace"] == [],
    ))

    full_trace = ext.S4S5Replay(deepcopy(protocol), "AegisKV", capacity)
    _rows, _delays, traced_mechanism, _trace = full_trace.run_coupled(
        [one_task()], [], control_horizon_ms=30000.0
    )
    trace = traced_mechanism["active_hbm_v313_s4s5"]["rich_control_trace"]
    required = {
        "time_ms", "alpha_after_mean", "pressure_fraction",
        "realized_active_requests_per_gpu", "serving_hbm_bytes_total",
        "peer_hbm_backup_after_bytes_total", "dram_used_bytes_total",
        "peer_hbm_quota_bytes_total",
        "shared_serving_plus_backup_budget_bytes_total",
        "hbm_admission_success_count", "hbm_admission_rejection_count",
        "hbm_evicted_bytes", "primary_priority_reclaimed_bytes",
        "hbm_admission_resumed_event_count", "dispatched_after_control",
    }
    checks.append((
        "s5_control_snapshot_has_capacity_churn_and_quota_fields",
        len(trace) == 1 and required.issubset(trace[0]),
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
