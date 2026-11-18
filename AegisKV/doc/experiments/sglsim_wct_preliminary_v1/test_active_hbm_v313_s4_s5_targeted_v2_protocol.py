#!/usr/bin/env python3
"""Fail-closed preregistration gates for targeted S4/S5 v2."""

from __future__ import annotations

import json
from pathlib import Path

import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def main() -> None:
    spec = json.loads((ROOT / "active_hbm_v313_s4_s5_targeted_v2_protocol.json").read_text())
    checks: list[tuple[str, bool]] = []
    source_ok = True
    for row in spec["sources"].values():
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        source_ok &= path.is_file() and core.sha256(path) == row["sha256"]
    checks.append(("all_source_hashes_match", source_ok))

    s4 = spec["s4"]
    checks.append((
        "algorithm1_causal_residual_dag_roles",
        s4["algorithm1"]["branch_role_predicted_output_tokens"] == [4096, 1024, 256, 256]
        and "no realized output/service" in s4["algorithm1"]["slack_formula"],
    ))
    checks.append((
        "algorithm2_four_exact_benefit_outcomes",
        s4["algorithm2"]["expected_full_outcome_by_class"] == {
            "both_benefit": "HBM_AND_DRAM", "hbm_benefit": "HBM_ONLY",
            "dram_benefit": "DRAM_ONLY", "neither_benefit": "BYPASS",
        }
        and "not a capacity change" in s4["algorithm2"]["capacity_rule"],
    ))
    checks.append((
        "algorithm3_exact_repeated_window_schedule",
        s4["algorithm3"]["control_opportunities_ms"]
        == [float(value) for value in range(30000, 420001, 30000)]
        and [(row["name"], row["offered_active_requests_per_gpu"]) for row in s4["algorithm3"]["phases"]]
        == [("low", 1), ("high", 8), ("recovery", 1)],
    ))
    checks.append((
        "three_seeds_and_no_total_wct_win_gate",
        spec["common"]["seeds"] == [2026082600, 2026082601, 2026082602]
        and s4["no_requirement_that_every_algorithm_improve_total_wct"] is True,
    ))

    s5 = spec["s5"]
    bound = s5["analytic_bound"]
    expected_request = (16384 + 4096) * 147456
    expected_safe = (
        bound["shared_serving_plus_backup_budget_bytes"]
        - bound["maximum_peer_hbm_quota_bytes"]
    ) // expected_request
    checks.append((
        "s5_capacity_equation_exact",
        expected_request == bound["request_p99_serving_kv_bytes"]
        and expected_safe == bound["capacity_safe_active_requests"] == 2,
    ))
    checks.append((
        "s5_guarded_c1_without_load_search",
        s5["recovery_offered_active_requests_per_gpu"] == 1
        and bound["load_search_performed"] is False
        and "one offered active request" in bound["guard_rule"],
    ))
    checks.append((
        "s5_eight_recovery_windows_and_fail_close",
        s5["recovery_control_window_count"] == 8
        and s5["stable_windows_before_increase"] == 3
        and "FAIL_DIAGNOSTIC" in s5["behavior_gate"]
        and spec["paper_ready"] is False,
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
