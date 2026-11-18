#!/usr/bin/env python3
"""Negative regression for the S3 paired-row audit-count mismatch."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import run_preliminary as core


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "active_hbm_v313_s3_scale_context_protocol.json"
FAILED_STAGE = ROOT / "results/active_hbm_v313_s3_scale_context_pilot.invalidated_audit_count_v1"
FROZEN_RUNNER = ROOT / "run_active_hbm_v313_s3_scale_context_simulation_v1_frozen.py"


def main() -> None:
    spec = json.loads(SPEC.read_text())
    audit = json.loads((FAILED_STAGE / "audit.json").read_text())
    manifest = json.loads((FAILED_STAGE / "manifest.json").read_text())
    rows = list(csv.DictReader((FAILED_STAGE / "paired_ci.csv").open()))
    scale_points = len(spec["sweep"]["scale_axis"]["system_dp"])
    context_points = len(spec["sweep"]["context_axis"]["context_tokens"])
    display_axis_points = scale_points + context_points
    topology_count = len(spec["sweep"]["topologies"])
    paired_metrics = {
        "healthy_wct_mean_ms", "healthy_wct_p99_ms",
        "fault_wct_mean_ms", "fault_wct_p99_ms",
    }
    derived = display_axis_points * topology_count * len(paired_metrics)
    keys = {
        (row["axis"], row["axis_value"], row["topology"], row["metric"])
        for row in rows
    }
    checks = [
        (
            "shared_center_is_one_DES_point_but_two_registered_display_axis_points",
            spec["sweep"]["unique_axis_points"] == 9
            and display_axis_points == 10
            and spec["sweep"]["shared_center_cell"][
                "executed_once_per_topology_policy_seed"
            ] is True,
        ),
        (
            "paired_schema_has_exact_four_registered_WCT_metrics",
            {row["metric"] for row in rows} == paired_metrics,
        ),
        (
            "paired_schema_has_exact_two_registered_topologies",
            {row["topology"] for row in rows} == {"same_node", "cross_node"},
        ),
        (
            "derived_paired_row_count_is_10_times_2_times_4_equals_80",
            derived == 80,
        ),
        (
            "actual_paired_rows_and_unique_keys_equal_80",
            len(rows) == 80 and len(keys) == 80,
        ),
        (
            "old_expected_40_is_rejected_by_registered_dimensions",
            derived != 40 and len(rows) != 40,
        ),
        (
            "failed_stage_was_fail_closed_only_by_old_count_assertion",
            audit["status"] == "FAIL"
            and audit["failed_checks"] == ["exact_40_paired_metric_rows"]
            and all(
                passed for name, passed in audit["checks"].items()
                if name != "exact_40_paired_metric_rows"
            ),
        ),
        (
            "simulation_runner_source_is_frozen_to_failed_stage_manifest",
            core.sha256(FROZEN_RUNNER) == manifest["runner_sha256"]
            == "b50c98506ddbeea042309781ef8c4b29149923504f72073488b03b8c283c265d",
        ),
        (
            "every_paired_row_has_three_seeds",
            all(int(row["seed_count"]) == 3 for row in rows),
        ),
    ]
    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
