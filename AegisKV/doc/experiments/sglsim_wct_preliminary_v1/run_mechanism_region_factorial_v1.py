#!/usr/bin/env python3
"""Run the preregistered SGLSim mechanism-region factorial."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_mechanism_pilot_v2 as pilot  # noqa: E402
import run_preliminary as core  # noqa: E402


SPEC_PATH = ROOT / "mechanism_region_factorial_v1_protocol.json"
TEST_PATH = ROOT / "test_mechanism_pilot_v2.py"


def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if core.sha256(ROOT / spec["parent_mechanism_protocol"]) != spec["parent_mechanism_protocol_sha256"]:
        raise RuntimeError("parent mechanism protocol changed after preregistration")
    if core.sha256(ROOT / spec["core_runner"]) != spec["core_runner_sha256"]:
        raise RuntimeError("core runner changed after factorial preregistration")
    protocol, _parent_spec = pilot.load_composite_protocol()
    return protocol, spec


def run_one(
    args: tuple[dict[str, Any], str, int, int, int, str, int, float, Path],
) -> dict[str, Any]:
    base, policy, seed, concurrency, context, failure_type, probes, cutoff, raw = args
    protocol = deepcopy(base)
    protocol["workload"]["center_context_tokens"] = int(context)
    protocol["workload"]["center_active_requests_per_gpu"] = int(concurrency)
    protocol["failure_model"]["probe_type_schedule"] = [failure_type]
    cell = core.run_cell(
        protocol,
        policy=policy,
        seed=seed,
        workflows=None,
        active_per_gpu=concurrency,
        probe_count=probes,
        output=raw,
        arrival_cutoff_ms=cutoff,
    )
    cell["context_tokens"] = int(context)
    cell["failure_type"] = failure_type
    return cell


def cell_metric_row(cell: dict[str, Any]) -> dict[str, Any]:
    mechanism = cell["mechanism"]
    failure_type = cell["failure_type"]
    failure = cell["failure_conditioned_changed_by_type"][failure_type]
    churn = mechanism.get("tier_churn", {})
    residency = mechanism.get("residency_event_counts", {})
    residency_total = sum(int(value) for value in residency.values())
    return {
        "policy": cell["policy"],
        "seed": cell["seed"],
        "active_requests_per_gpu": cell["active_requests_per_gpu"],
        "context_tokens": cell["context_tokens"],
        "failure_type": failure_type,
        "workflow_count": cell["workflow_count"],
        "healthy_wct_mean_ms": cell["base_wct_mean_ms"],
        "healthy_wct_p95_ms": cell["base_wct_p95_ms"],
        "changed_workflow_observations": failure["changed_workflow_observations"],
        "changed_wct_mean_ms": failure["changed_wct_mean_ms"],
        "changed_wct_p95_ms": failure["changed_wct_p95_ms"],
        "added_wct_mean_ms": failure["added_wct_mean_ms"],
        "added_wct_p95_ms": failure["added_wct_p95_ms"],
        "normal_copy_burden_mean_ms": mechanism["normal_path_copy_burden_mean_ms"],
        "copy_service_mean_ms": mechanism["copy_transfer_service_mean_ms"],
        "copy_queue_wait_mean_ms": mechanism["copy_queue_wait_mean_ms"],
        "copy_amplification": mechanism["copy_amplification"],
        "recovery_hit_fraction": mechanism["recovery_hit_fraction"],
        "recovery_restored_tokens": mechanism["recovery_restored_tokens"],
        "recovery_recomputed_tokens": mechanism["recovery_recomputed_tokens"],
        "alpha_final_mean": mechanism["alpha_final_mean"],
        "control_window_count": mechanism["control_window_count"],
        "pressure_evaluation_fraction": mechanism["pressure_target_evaluation_fraction"],
        "alg1_nonzero_slack_requests": mechanism["algorithm1_nonzero_slack_request_count"],
        "hbm_admission_rejections": churn.get("HBM", {}).get("admission_rejection_count", 0),
        "dram_admission_rejections": churn.get("DRAM", {}).get("admission_rejection_count", 0),
        "hbm_evictions": churn.get("HBM", {}).get("eviction_count", 0),
        "dram_evictions": churn.get("DRAM", {}).get("eviction_count", 0),
        "hbm_occupancy_mean_gib_per_gpu": mechanism["occupancy_time_mean_gib_per_gpu"]["HBM"],
        "dram_occupancy_mean_gib_per_gpu": mechanism["occupancy_time_mean_gib_per_gpu"]["DRAM"],
        "residency_bypass_fraction": (residency.get("BYPASS", 0) / residency_total if residency_total else None),
        "residency_hbm_only_fraction": (residency.get("HBM_ONLY", 0) / residency_total if residency_total else None),
        "residency_dram_only_fraction": (residency.get("DRAM_ONLY", 0) / residency_total if residency_total else None),
        "residency_both_fraction": (residency.get("HBM_AND_DRAM", 0) / residency_total if residency_total else None),
        "empty_replay_delta_ms": mechanism["empty_fault_replay_max_completion_delta_ms"],
        "fixed_point_iterations": mechanism["normal_path_schedule_solver"]["iterations"],
        "fixed_point_residual_ms": mechanism["normal_path_schedule_solver"]["final_burden_residual_finish_ms"],
        "raw_npz": cell["raw_npz"],
        "raw_npz_sha256": cell["raw_npz_sha256"],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ("seed", "active_requests_per_gpu", "context_tokens", "failure_type")
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, {})[str(row["policy"])] = row
    result = []
    for key, policies in sorted(groups.items()):
        full = policies["AegisKV"]
        for policy in (
            "Native Mooncake",
            "AegisKV no Algorithm 1",
            "AegisKV no Algorithm 2",
            "AegisKV no Algorithm 3",
        ):
            other = policies[policy]
            for metric in (
                "healthy_wct_mean_ms", "changed_wct_mean_ms", "added_wct_mean_ms"
            ):
                reference = float(full[metric])
                value = float(other[metric])
                result.append({
                    **dict(zip(key_fields, key)),
                    "comparison_policy": policy,
                    "reference_policy": "AegisKV",
                    "metric": metric,
                    "aegiskv_value": reference,
                    "comparison_value": value,
                    "absolute_delta_comparison_minus_full": value - reference,
                    "relative_delta_comparison_minus_full": value / reference - 1.0,
                })
    return result


def ci_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "active_requests_per_gpu", "context_tokens", "failure_type",
        "comparison_policy", "reference_policy", "metric",
    )
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in paired:
        key = tuple(row[field] for field in fields)
        grouped.setdefault(key, []).append(
            float(row["relative_delta_comparison_minus_full"])
        )
    result = []
    for key, values in sorted(grouped.items()):
        mean = statistics.fmean(values)
        if len(values) < 2:
            low = high = mean
        else:
            half = 4.302652729911275 * statistics.stdev(values) / math.sqrt(len(values))
            low, high = mean - half, mean + half
        result.append({
            **dict(zip(fields, key)),
            "seed_count": len(values),
            "relative_delta_mean": mean,
            "relative_delta_ci95_low": low,
            "relative_delta_ci95_high": high,
        })
    return result


def coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    by_key = {
        (
            row["seed"], row["active_requests_per_gpu"], row["context_tokens"],
            row["failure_type"], row["policy"],
        ): row
        for row in rows
    }
    for seed in sorted({int(row["seed"]) for row in rows}):
        for concurrency in sorted({int(row["active_requests_per_gpu"]) for row in rows}):
            for context in sorted({int(row["context_tokens"]) for row in rows}):
                for failure_type in sorted({str(row["failure_type"]) for row in rows}):
                    full = by_key[(seed, concurrency, context, failure_type, "AegisKV")]
                    no1 = by_key[(seed, concurrency, context, failure_type, "AegisKV no Algorithm 1")]
                    no2 = by_key[(seed, concurrency, context, failure_type, "AegisKV no Algorithm 2")]
                    no3 = by_key[(seed, concurrency, context, failure_type, "AegisKV no Algorithm 3")]
                    result.append({
                        "seed": seed,
                        "active_requests_per_gpu": concurrency,
                        "context_tokens": context,
                        "failure_type": failure_type,
                        "full_alg1_slack_triggered": int(full["alg1_nonzero_slack_requests"]) > 0,
                        "noalg1_slack_disabled": int(no1["alg1_nonzero_slack_requests"]) == 0,
                        "full_alg2_admission_triggered": (
                            int(full["hbm_admission_rejections"]) + int(full["dram_admission_rejections"])
                            + int(full["hbm_evictions"]) + int(full["dram_evictions"])
                        ) > 0,
                        "noalg2_component_adaptation_active": no2["alpha_final_mean"] in {None, ""},
                        "full_alg3_three_windows": int(full["control_window_count"]) == 3,
                        "full_alg3_pressure_triggered": float(full["pressure_evaluation_fraction"] or 0.0) > 0.0,
                        "noalg3_control_disabled": int(no3["control_window_count"]) == 0,
                        "changed_workflow_observations": int(full["changed_workflow_observations"]),
                        "alpha_final_full": full["alpha_final_mean"],
                        "alpha_final_noalg3": no3["alpha_final_mean"],
                        "hbm_rejections_full": full["hbm_admission_rejections"],
                        "dram_rejections_full": full["dram_admission_rejections"],
                        "hbm_evictions_full": full["hbm_evictions"],
                        "dram_evictions_full": full["dram_evictions"],
                        "recovery_hit_fraction_full": full["recovery_hit_fraction"],
                    })
    return result


def audit(
    spec: dict[str, Any], cells: list[dict[str, Any]], metrics: list[dict[str, Any]],
    coverage: list[dict[str, Any]], *, phase: str, stage: Path, target: Path,
) -> dict[str, Any]:
    import numpy as np

    matrix = spec["matrix"]
    seeds = [matrix["phase1_seed"]] if phase == "phase1" else matrix["paired_seeds"]
    expected = (
        len(matrix["active_requests_per_gpu"]) * len(matrix["context_tokens"])
        * len(matrix["failure_types"]) * len(matrix["policies"]) * len(seeds)
    )
    checks: dict[str, bool] = {
        "cell_count_exact": len(cells) == expected,
        "no_performance_selection_gate": spec["phase2_expansion_gate"]["uses_performance_direction_or_win_threshold"] is False,
        "fixed_arrival_90s": all(cell["workload"]["arrival_horizon_ms"] == 90000.0 for cell in cells),
        "empty_replay_exact": all(cell["mechanism"]["empty_fault_replay_max_completion_delta_ms"] <= 1e-7 for cell in cells),
        "fixed_point_converged": all(
            cell["mechanism"]["normal_path_schedule_solver"]["converged"] is True
            and cell["mechanism"]["normal_path_schedule_solver"]["final_burden_residual_finish_ms"] <= 1e-7
            for cell in cells
        ),
        "probe_failure_type_exact": True,
        "fault_wct_monotone": True,
        "raw_hashes_match": True,
        "paired_workloads_and_probes": True,
        "full_alg1_covered": any(row["full_alg1_slack_triggered"] for row in coverage),
        "noalg1_disabled": all(row["noalg1_slack_disabled"] for row in coverage),
        "full_alg2_pressure_covered": any(row["full_alg2_admission_triggered"] for row in coverage),
        "noalg2_component_adaptation_active": all(row["noalg2_component_adaptation_active"] for row in coverage),
        "full_alg3_windows_covered": all(row["full_alg3_three_windows"] for row in coverage),
        "full_alg3_pressure_covered": any(row["full_alg3_pressure_triggered"] for row in coverage),
        "noalg3_disabled": all(row["noalg3_control_disabled"] for row in coverage),
        "both_failure_types_changed_workflows": all(
            any(row["failure_type"] == failure_type and int(row["changed_workflow_observations"]) > 0 for row in coverage)
            for failure_type in matrix["failure_types"]
        ),
        "no_source_node_peer_hbm_recovery": all(
            not any(key.startswith("source_node:peer_hbm") for key in cell["mechanism"]["recovery_type_path_counts"])
            for cell in cells if cell["failure_type"] == "source_node"
        ),
    }
    signatures: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for cell in cells:
        raw = stage / Path(cell["raw_npz"]).relative_to(target)
        if core.sha256(raw) != cell["raw_npz_sha256"]:
            checks["raw_hashes_match"] = False
        with np.load(raw, allow_pickle=False) as data:
            if set(data["probe_failure_type"].tolist()) != {cell["failure_type"]}:
                checks["probe_failure_type_exact"] = False
            if np.any(data["fault_wct_ms"] < data["base_wct_ms"][None, :] - 1e-7):
                checks["fault_wct_monotone"] = False
            signature = (
                tuple(data["workflow_id"].tolist()), data["probe_time_ms"].tobytes(),
                data["probe_source_rank"].tobytes(), tuple(data["probe_failure_type"].tolist()),
            )
            key = (cell["seed"], cell["active_requests_per_gpu"], cell["context_tokens"], cell["failure_type"])
            if key in signatures and signatures[key] != signature:
                checks["paired_workloads_and_probes"] = False
            signatures[key] = signature
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "phase": phase,
        "paper_ready": False,
        "physical_result": False,
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "cell_count": len(cells),
        "phase2_expansion_gate_status": "PASS" if phase == "phase1" and all(checks.values()) else ("NOT_APPLICABLE" if phase != "phase1" else "FAIL"),
        "evidence_boundary": spec["evidence_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "all"), default="phase1")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    protocol, spec = load_inputs()
    matrix = spec["matrix"]
    seeds = [matrix["phase1_seed"]] if args.phase == "phase1" else matrix["paired_seeds"]
    target = (args.output_dir or ROOT / "results" / f"mechanism_region_factorial_v1_{args.phase}").resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()
    test = subprocess.run(
        [sys.executable, str(TEST_PATH)], cwd=str(ROOT), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (stage / "regression_test.log").write_text(test.stdout, encoding="utf-8")
    if test.returncode or "RESULT 6/6 PASS" not in test.stdout:
        raise RuntimeError("regression test failed before factorial")
    jobs = []
    for seed in seeds:
        for concurrency in matrix["active_requests_per_gpu"]:
            for context in matrix["context_tokens"]:
                for failure_type in matrix["failure_types"]:
                    for policy in matrix["policies"]:
                        label = policy.lower().replace(" ", "_")
                        raw = stage / "raw" / (
                            f"{label}__c{concurrency}__ctx{context}__{failure_type}__s{seed}.npz"
                        )
                        jobs.append((
                            protocol, policy, int(seed), int(concurrency), int(context),
                            failure_type, int(matrix["fault_probes_per_cell"]),
                            float(matrix["arrival_cutoff_ms"]), raw,
                        ))
    cells = []
    with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
        futures = {
            pool.submit(run_one, job): (job[1], job[2], job[3], job[4], job[5])
            for job in jobs
        }
        done = 0
        for future in as_completed(futures):
            cells.append(future.result())
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"DONE {done}/{len(jobs)}", flush=True)
    cells.sort(key=lambda row: (
        row["seed"], row["active_requests_per_gpu"], row["context_tokens"],
        row["failure_type"], row["policy"],
    ))
    for cell in cells:
        raw = Path(cell["raw_npz"])
        cell["raw_npz"] = str(target / raw.relative_to(stage))
    metrics = [cell_metric_row(cell) for cell in cells]
    paired = paired_differences(metrics)
    coverage = coverage_rows(metrics)
    write_csv(metrics, stage / "cell_metrics.csv")
    write_csv(paired, stage / "paired_differences.csv")
    write_csv(ci_rows(paired), stage / "paired_differences_ci.csv")
    write_csv(coverage, stage / "mechanism_coverage.csv")
    result_audit = audit(
        spec, cells, metrics, coverage, phase=args.phase, stage=stage, target=target
    )
    result_audit["checks"]["regression_harness_6_of_6"] = True
    result_audit["status"] = "PASS" if all(result_audit["checks"].values()) else "FAIL"
    result_audit["failed_checks"] = [key for key, passed in result_audit["checks"].items() if not passed]
    (stage / "audit.json").write_text(json.dumps(result_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "status": "PRELIMINARY_COMPLETE",
        "phase": args.phase,
        "paper_ready": False,
        "protocol": {"path": str(SPEC_PATH), "sha256": core.sha256(SPEC_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "core_runner": {"path": str(Path(core.__file__).resolve()), "sha256": core.sha256(Path(core.__file__).resolve())},
        "cells": cells,
    }
    (stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}:
            artifacts[str(path.relative_to(stage))] = core.sha256(path)
    manifest = {
        "schema_version": 1,
        "status": result_audit["status"],
        "phase": args.phase,
        "paper_ready": False,
        "target": str(target),
        "artifacts": artifacts,
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        + f"{core.sha256(manifest_path)}  manifest.json\n",
        encoding="utf-8",
    )
    if result_audit["status"] != "PASS":
        raise RuntimeError(f"factorial audit failed in {stage}: {result_audit['failed_checks']}")
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
