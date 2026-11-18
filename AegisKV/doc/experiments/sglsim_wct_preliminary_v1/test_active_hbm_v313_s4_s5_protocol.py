#!/usr/bin/env python3
"""Fail-closed structure and lineage gates for the S4/S5 preregistration."""

from __future__ import annotations

import json
from pathlib import Path

import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def main() -> None:
    spec = json.loads((ROOT / "active_hbm_v313_s4_s5_protocol.json").read_text())
    checks: list[tuple[str, bool]] = []
    sources_ok = True
    for row in spec["sources"].values():
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        sources_ok &= path.is_file() and core.sha256(path) == row["sha256"]
    checks.append(("all_preregistered_source_hashes_match", sources_ok))

    s4 = spec["s4"]
    checks.append((
        "s4_exact_five_policies_and_three_seeds",
        s4["internal_policies"] == [
            "Native Mooncake", "AegisKV", "AegisKV no Algorithm 1",
            "AegisKV no Algorithm 2", "AegisKV no Algorithm 3",
        ]
        and s4["paired_seeds"] == [2026082600, 2026082601, 2026082602],
    ))
    checks.append((
        "s4_fixed_region_and_paired_probe_domains",
        s4["context_tokens"] == 16384
        and s4["active_requests_per_gpu"] == 8
        and s4["arrival_cutoff_ms"] == 90000.0
        and s4["fault_probes_per_seed"] == 24
        and s4["probe_type_schedule"] == ["source_worker_gpu", "source_node"],
    ))
    checks.append((
        "algorithm2_ablation_keeps_controller_and_aimd",
        "keep Aegis controller" in s4["ablation_contract"]["Algorithm 2"]
        and "AIMD" in s4["ablation_contract"]["Algorithm 2"],
    ))

    s5 = spec["s5"]
    checks.append((
        "s5_exact_low_high_recovery_schedule",
        [(row["name"], row["start_ms"], row["end_ms"], row["offered_active_requests_per_gpu"]) for row in s5["phases"]]
        == [("low", 0.0, 60000.0, 2), ("high", 60000.0, 120000.0, 8), ("recovery", 120000.0, 240000.0, 2)]
        and s5["control_opportunities_ms"] == [30000.0, 60000.0, 90000.0, 120000.0, 150000.0, 180000.0, 210000.0, 240000.0],
    ))
    checks.append((
        "s5_event_occupancy_and_behavior_fail_close_registered",
        any("event-integrated" in row for row in s5["recorded_fields"])
        and "publish a FAIL diagnostic" in s5["behavior_gates"]["failure_rule"],
    ))
    checks.append((
        "preliminary_and_no_shape_selection",
        spec["paper_ready"] is False
        and s4["no_result_shape_threshold"] is True
        and s5["no_result_shape_threshold"] is True,
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
