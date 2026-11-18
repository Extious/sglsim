#!/usr/bin/env python3
"""Run and audit the common 90-second arrival-horizon S2 sweep."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "s2_fixed_arrival_v2")
    args = parser.parse_args()
    contract_path = ROOT / "s2_fixed_arrival_protocol.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    protocol = core.load_protocol()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for concurrency_value in contract["active_requests_per_gpu"]:
        concurrency = int(concurrency_value)
        for policy in contract["policies"]:
            for seed in contract["paired_seeds"]:
                label = policy.lower().replace(" ", "_")
                raw = output / "raw" / f"{label}__c{concurrency}__s{seed}.npz"
                print(
                    f"RUN-S2-V2 policy={policy!r} c={concurrency} cutoff_ms={contract['arrival_cutoff_ms']:g} seed={seed}",
                    flush=True,
                )
                cells.append(
                    core.run_cell(
                        protocol,
                        policy=policy,
                        seed=int(seed),
                        workflows=None,
                        active_per_gpu=concurrency,
                        probe_count=int(contract["fault_probes_per_seed"]),
                        output=raw,
                        arrival_cutoff_ms=float(contract["arrival_cutoff_ms"]),
                    )
                )

    payload = {
        "schema_version": 2,
        "status": "PRELIMINARY_COMPLETE",
        "selection": "S2_FIXED_ARRIVAL_V2",
        "workflow_completion_time_result": True,
        "contract": {"path": str(contract_path), "sha256": core.sha256(contract_path)},
        "parent_protocol": {"path": str(ROOT / "protocol.json"), "sha256": core.sha256(ROOT / "protocol.json")},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "core_runner": {"path": str(ROOT / "run_preliminary.py"), "sha256": core.sha256(ROOT / "run_preliminary.py")},
        "cells": cells,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    seed_rows: list[dict[str, object]] = []
    for cell in cells:
        for rate in cell["rate_rows"]:
            seed_rows.append({
                "policy": cell["policy"],
                "seed": cell["seed"],
                "active_requests_per_gpu": cell["active_requests_per_gpu"],
                "failure_rate_per_1000_gpu_shard_hour": rate["failure_rate_per_1000_gpu_shard_hour"],
                "hazard_integrated_mean_wct_ms": rate["hazard_integrated_mean_wct_ms"],
                "expected_faults_per_trace": rate["expected_faults_per_trace"],
                "arrival_horizon_ms": cell["workload"]["arrival_horizon_ms"],
                "actual_last_arrival_ms": cell["workload"]["actual_last_arrival_ms"],
                "drain_horizon_ms": cell["workload"]["drain_horizon_ms"],
                "workflow_count": cell["workflow_count"],
                "control_window_count": cell["mechanism"]["control_window_count"],
                "control_opportunity_times_ms": ";".join(
                    f"{value:g}" for value in cell["mechanism"]["control_opportunity_times_ms"]
                ),
                "pressure_target_evaluation_fraction": cell["mechanism"]["pressure_target_evaluation_fraction"],
                "alpha_final_mean": cell["mechanism"]["alpha_final_mean"],
                "realized_active_requests_per_gpu_time_mean": cell["workload"]["realized_active_requests_per_gpu_time_mean"],
                "raw_npz": cell["raw_npz"],
            })
    seed_path = output / "summary.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    grouped: dict[tuple[str, int, float], list[tuple[int, float]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["policy"]), int(row["active_requests_per_gpu"]), float(row["failure_rate_per_1000_gpu_shard_hour"]))].append((int(row["seed"]), float(row["hazard_integrated_mean_wct_ms"])))
    aggregate = []
    for (policy, concurrency, rate), values in sorted(grouped.items()):
        values.sort()
        native = dict(grouped[("Native Mooncake", concurrency, rate)])
        wct = [value for _seed, value in values]
        ratio = [value / native[seed] for seed, value in values]
        mean, low, high = core.bootstrap_ci(wct, seed=20260826 ^ concurrency ^ int(rate * 1000))
        rmean, rlow, rhigh = core.bootstrap_ci(ratio, seed=20260827 ^ concurrency ^ int(rate * 1000))
        aggregate.append({
            "policy": policy,
            "active_requests_per_gpu": concurrency,
            "failure_rate_per_1000_gpu_shard_hour": rate,
            "seed_count": len(values),
            "mean_wct_ms": mean,
            "ci95_low_ms": low,
            "ci95_high_ms": high,
            "normalized_to_native": rmean,
            "normalized_ci95_low": rlow,
            "normalized_ci95_high": rhigh,
        })
    aggregate_path = output / "aggregate_ci.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    checks: dict[str, bool] = {}
    expected = {
        (policy, int(concurrency), int(seed))
        for policy in contract["policies"]
        for concurrency in contract["active_requests_per_gpu"]
        for seed in contract["paired_seeds"]
    }
    checks["exact_policy_concurrency_seed_matrix"] = {
        (cell["policy"], int(cell["active_requests_per_gpu"]), int(cell["seed"]))
        for cell in cells
    } == expected
    cutoff = float(contract["arrival_cutoff_ms"])
    checks["fixed_arrival_horizon_exact"] = all(
        abs(float(cell["workload"]["arrival_horizon_ms"]) - cutoff) <= 1e-9
        for cell in cells
    )
    checks["last_arrival_within_common_cutoff"] = all(
        0.0 <= float(cell["workload"]["actual_last_arrival_ms"]) <= cutoff
        and cutoff - float(cell["workload"]["actual_last_arrival_ms"])
        < 10.0 * float(cell["workload"]["mean_workflow_gap_ms"])
        for cell in cells
    )
    checks["drain_horizon_reported_after_cutoff"] = all(
        float(cell["workload"]["drain_horizon_ms"]) >= cutoff
        for cell in cells
    )
    checks["raw_hashes_and_wct"] = True
    checks["paired_fault_tuples"] = True
    signatures: dict[tuple[int, int], tuple[bytes, bytes, tuple[str, ...]]] = {}
    for cell in cells:
        if core.sha256(Path(cell["raw_npz"])) != cell["raw_npz_sha256"]:
            checks["raw_hashes_and_wct"] = False
        with np.load(cell["raw_npz"], allow_pickle=False) as raw:
            if np.any(raw["base_wct_ms"] <= 0) or np.any(raw["fault_wct_ms"] <= 0):
                checks["raw_hashes_and_wct"] = False
            signature = (
                raw["probe_time_ms"].tobytes(),
                raw["probe_source_rank"].tobytes(),
                tuple(raw["probe_failure_type"].tolist()),
            )
            key = (int(cell["seed"]), int(cell["active_requests_per_gpu"]))
            if key in signatures and signatures[key] != signature:
                checks["paired_fault_tuples"] = False
            signatures[key] = signature
    aegis = [cell for cell in cells if cell["policy"] == "AegisKV"]
    expected_control_times = [float(value) for value in contract["control_opportunity_times_ms"]]
    checks["exact_common_aimd_control_opportunities"] = all(
        int(cell["mechanism"]["control_window_count"]) == len(expected_control_times)
        and [float(value) for value in cell["mechanism"]["control_opportunity_times_ms"]]
        == expected_control_times
        for cell in aegis
    )
    checks["pressure_exposure_reported"] = all(
        cell["mechanism"]["pressure_target_evaluation_fraction"] is not None
        for cell in aegis
    )
    checks["no_target_owned_or_hbm_to_dram"] = all(
        cell["mechanism"]["target_owned_copy_events"] == 0
        and cell["mechanism"]["hbm_to_dram_copy_events"] == 0
        for cell in cells
    )
    audit = {
        "schema_version": 2,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selection": "S2_FIXED_ARRIVAL_V2",
        "checks": checks,
        "arrival_horizon_ms": cutoff,
        "control_opportunity_times_ms": expected_control_times,
        "drain_horizon_range_ms": [
            min(float(cell["workload"]["drain_horizon_ms"]) for cell in cells),
            max(float(cell["workload"]["drain_horizon_ms"]) for cell in cells),
        ],
        "pressure_exposure_contract": contract["pressure_exposure_contract"],
        "workflow_completion_time_result": True,
        "paper_ready": False,
        "evidence_boundary": contract["evidence_boundary"],
        "summary_sha256": core.sha256(summary_path),
        "seed_summary_sha256": core.sha256(seed_path),
        "aggregate_ci_sha256": core.sha256(aggregate_path),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError([name for name, passed in checks.items() if not passed])
    print(output / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
