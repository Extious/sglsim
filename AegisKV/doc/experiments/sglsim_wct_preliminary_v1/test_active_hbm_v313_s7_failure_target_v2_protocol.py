#!/usr/bin/env python3
"""Fail-closed structural gate for the S7 failure-target v2 preregistration."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "active_hbm_v313_s7_failure_target_v2_protocol.json"


def load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def checks(protocol: dict) -> list[tuple[str, bool]]:
    common = protocol["frozen_common_configuration"]
    sweep = protocol["sweep"]
    target = protocol["failure_target_denominator"]
    outputs = protocol["planned_outputs"]
    state = protocol["implementation_state"]
    expected_bandwidth = (
        len(common["policies"])
        * len(sweep["bandwidth_directions"])
        * len(sweep["bandwidth_values_gb_s"])
        * len(common["paired_seeds"])
    )
    expected_repair = (
        len(common["policies"])
        * len(sweep["repair_failure_domains"])
        * len(sweep["repair_durations_ms"])
        * len(common["paired_seeds"])
    )
    return [
        ("design_only_and_no_raw_started", protocol["status"] == "PREREGISTERED_DESIGN_ONLY_NO_V2_RAW_STARTED" and state["v2_raw_started"] is False),
        ("all_workflow_primary_retained", protocol["statistics"]["primary_denominator"].startswith("unchanged all workflows")),
        ("failure_target_is_secondary", target["role"].startswith("SECONDARY_DIAGNOSTIC")),
        ("reference_trace_is_policy_independent", "no policy-specific" in target["reference_trace"]),
        ("target_ids_frozen_before_policy", "before any policy" in target["probe_construction_order"][2]),
        ("empty_target_fails_closed_without_resampling", "FAIL_CLOSE" in target["empty_target_rule"] and "never resample" in target["empty_target_rule"]),
        ("worker_target_uses_selected_rank", "source_rank equals probe source_rank" in target["source_worker_gpu_target_formula"]),
        ("node_target_uses_complete_eight_rank_node", "/ 8" in target["source_node_target_formula"] and common["gpus_per_node"] == 8),
        ("cross_policy_target_hash_required", "byte-identical" in target["cross_policy_invariant"] and "identity hash" in target["cross_policy_invariant"]),
        ("equal_probe_weighting_frozen", target["aggregation"].startswith("Average the 12 probe-level") and "equal probe weight" in target["aggregation"]),
        ("copy_remains_async_no_direction_gate", "copy is asynchronous" in protocol["statistics"]["performance_direction_gates"] and "none" in protocol["statistics"]["performance_direction_gates"]),
        ("exact_168_cell_schema", expected_bandwidth == sweep["bandwidth_cell_count"] == 120 and expected_repair == sweep["repair_cell_count"] == 48 and expected_bandwidth + expected_repair == sweep["registered_cell_count"] == outputs["cell_metrics_rows"] == 168),
        ("exact_2016_probe_rows_derived", outputs["probe_metric_rows"] == outputs["cell_metrics_rows"] * common["fault_probes_per_cell"] == 2016),
        ("exact_paired_row_counts_derived", outputs["paired_seed_rows"] == sweep["registered_cell_count"] // len(common["policies"]) == 84 and outputs["paired_ci_rows"] == 28),
        ("execution_blocked_until_sources_frozen", state["v2_runner"] == "NOT_IMPLEMENTED_OR_FROZEN" and state["v2_mechanism_tests"] == "NOT_IMPLEMENTED_OR_FROZEN" and state["execution_gate"].startswith("BLOCKED")),
        ("no_changed_only_cross_policy_denominator", "never used as a cross-policy denominator" in target["replay_metric_per_probe"]["changed_only_wct"]),
    ]


def main() -> None:
    protocol = load()
    outcomes = checks(protocol)
    for name, passed in outcomes:
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    passed = sum(value for _name, value in outcomes)
    print(f"SUMMARY {passed}/{len(outcomes)} PASS")
    if passed != len(outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
