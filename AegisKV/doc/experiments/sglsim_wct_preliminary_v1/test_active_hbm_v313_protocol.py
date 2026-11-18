#!/usr/bin/env python3
"""Fail-closed preregistration structure and source-hash gate."""

from __future__ import annotations

import json
from pathlib import Path

import poisson_fault_batch_v2 as batch
import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def main() -> None:
    spec = json.loads(
        (ROOT / "active_hbm_v313_accounting_batched_s2_protocol.json").read_text()
    )
    checks: list[tuple[str, bool]] = []
    expected_names = [
        "Native Mooncake", "Static", "DejaVu", "KevlarFlow",
        "GhostServe", "AegisKV",
    ]
    checks.append((
        "six_exact_display_names_and_internal_mapping",
        spec["healthy_accounting"]["policies"] == expected_names
        and spec["healthy_accounting"]["display_to_internal_policy"]["Static"]
        == "Static FCFS",
    ))
    cold = spec["healthy_accounting"]["cold_start_inclusive"]
    steady = spec["healthy_accounting"]["steady_state"]
    checks.append((
        "cold_and_steady_cohorts_are_fixed_before_run",
        cold["arrival_cutoff_ms"] == 5000.0
        and cold["control_opportunities_ms"] == []
        and steady["common_warmup_arrival_ms"] == [0.0, 90000.0]
        and steady["measurement_cohort_arrival_ms"] == [90000.0, 180000.0]
        and steady["control_opportunities_ms"]
        == [30000.0, 60000.0, 90000.0, 120000.0, 150000.0, 180000.0],
    ))
    batch_spec = spec["batched_s2"]
    checks.append((
        "batched_exposure_is_8400_windows_per_rate",
        batch_spec["independent_exposure_windows_per_workload_seed"] == 2800
        and batch_spec["total_windows_per_rate"] == 8400
        and abs(
            batch.expected_events(5.952, dp=16, horizon_ms=90000, window_count=8400)
            - batch_spec["expected_candidate_events"]["5.952"]
        ) <= 1e-12
        and abs(
            batch.expected_events(20.833, dp=16, horizon_ms=90000, window_count=8400)
            - batch_spec["expected_candidate_events"]["20.833"]
        ) <= 1e-12,
    ))
    checks.append((
        "source_hashes_match_preregistered_files",
        all(
            Path(row["path"]).exists()
            and core.sha256(Path(row["path"])) == row["sha256"]
            for row in spec["sources"].values()
        ),
    ))
    gates = set(spec["hard_gates"])
    checks.append((
        "new_accounting_and_recovery_hard_gates_present",
        {
            "a request's unsealed or future backup is never part of its own dispatch gate",
            "physical target aggregation across source routes never exceeds 40 GiB",
            "evicted or reclaimed replica bytes are absent from the committed recovery prefix",
        }.issubset(gates),
    ))
    checks.append((
        "no_result_shape_threshold_or_seed_selection",
        "no seed or window is selected by result shape" in gates
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
