#!/usr/bin/env python3
"""Run the preregistered three-seed active-HBM v3 paired central pilot."""

from __future__ import annotations

import csv
import hashlib
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

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import active_hbm_v3 as v3  # noqa: E402
import run_mechanism_pilot_v2 as pilot  # noqa: E402
import run_preliminary as core  # noqa: E402


SPEC_PATH = ROOT / "active_hbm_v3_paired_pilot_protocol.json"
CAPACITY_PROTOCOL_PATH = ROOT / "active_hbm_v3_protocol.json"
TESTS = (ROOT / "test_active_hbm_v3_capacity.py", ROOT / "test_active_hbm_v3_des.py")
T_CRITICAL_95_N3 = 4.302652729911275


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    capacity = json.loads(CAPACITY_PROTOCOL_PATH.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("parent_capacity_protocol", "parent_capacity_protocol_sha256"),
        ("v3_des", "v3_des_sha256"),
        ("core_runner", "core_runner_sha256"),
    ):
        path = ROOT / spec[path_key]
        if core.sha256(path) != spec[hash_key]:
            raise RuntimeError(f"preregistered source changed: {path}")
    base, _parent_spec = pilot.load_composite_protocol()
    return base, capacity, spec


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def run_one(
    args: tuple[dict[str, Any], dict[str, Any], str, int, str, Path],
) -> dict[str, Any]:
    base, capacity, policy, seed, failure_type, raw = args
    protocol = deepcopy(base)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["workload"]["center_context_tokens"] = 40960
    protocol["failure_model"]["probe_type_schedule"] = [failure_type]
    return v3.run_v3_cell(
        protocol,
        capacity,
        policy=policy,
        seed=seed,
        active_per_gpu=16,
        probe_count=24,
        output=raw,
        arrival_cutoff_ms=90000.0,
    )


def metric_row(cell: dict[str, Any]) -> dict[str, Any]:
    failure_type = cell["failure_type"]
    failure = cell["failure_conditioned_changed_by_type"][failure_type]
    mechanism = cell["mechanism"]
    capacity = mechanism["active_hbm_v3"]
    churn = mechanism["tier_churn"]
    recovery_total = sum(int(value) for value in mechanism["recovery_path_counts"].values())
    recompute_count = int(mechanism["recovery_path_counts"].get("RECOMPUTE", 0))
    return {
        "policy": cell["policy"],
        "seed": cell["seed"],
        "failure_type": failure_type,
        "active_requests_per_gpu": cell["active_requests_per_gpu"],
        "context_tokens": cell["context_tokens"],
        "workflow_count": cell["workflow_count"],
        "fault_probe_count": cell["fault_probe_count"],
        "all_workflow_observations": failure["all_workflow_observations"],
        "workflow_universe_sha256": cell["workflow_universe_sha256"],
        "fault_probe_tuple_sha256": cell["fault_probe_tuple_sha256"],
        "healthy_wct_mean_ms": cell["base_wct_mean_ms"],
        "healthy_wct_p95_ms": cell["base_wct_p95_ms"],
        "all_workflow_wct_mean_ms": failure["all_workflow_wct_mean_ms"],
        "all_workflow_wct_p95_ms": failure["all_workflow_wct_p95_ms"],
        "all_workflow_added_wct_mean_ms": failure["all_workflow_added_wct_mean_ms"],
        "all_workflow_added_wct_p95_ms": failure["all_workflow_added_wct_p95_ms"],
        "changed_workflow_fraction": failure["changed_workflow_fraction"],
        "diagnostic_changed_only_observations": failure["changed_workflow_observations"],
        "diagnostic_changed_only_wct_mean_ms": failure["changed_wct_mean_ms"],
        "diagnostic_changed_only_wct_p95_ms": failure["changed_wct_p95_ms"],
        "diagnostic_changed_only_added_wct_mean_ms": failure["added_wct_mean_ms"],
        "diagnostic_changed_only_added_wct_p95_ms": failure["added_wct_p95_ms"],
        "healthy_copy_burden_mean_ms": mechanism["normal_path_copy_burden_mean_ms"],
        "healthy_copy_burden_p95_ms": mechanism["normal_path_copy_burden_p95_ms"],
        "coupled_active_per_gpu_mean": cell["workload"]["coupled_realized_active_requests_per_gpu_time_mean"],
        "coupled_active_per_gpu_peak": cell["workload"]["coupled_realized_active_requests_per_gpu_peak"],
        "queue_wait_mean_ms": capacity["queue_wait_mean_ms"],
        "queue_wait_p95_ms": capacity["queue_wait_p95_ms"],
        "drain_horizon_ms": cell["workload"]["drain_horizon_ms"],
        "capacity_peak_serving_gib": capacity["capacity_peak_serving_bytes"] / 1024**3,
        "capacity_peak_backup_gib": capacity["capacity_peak_backup_bytes"] / 1024**3,
        "capacity_peak_shared_gib": capacity["capacity_peak_shared_used_bytes"] / 1024**3,
        "capacity_min_headroom_gib": capacity["capacity_min_headroom_bytes"] / 1024**3,
        "capacity_violation_count": capacity["capacity_violation_count"],
        "memory_blocked_unique_requests": capacity["memory_blocked_unique_request_count"],
        "dispatch_at_control_count": capacity["dispatch_at_control_count"],
        "backup_released_at_control_gib": capacity["backup_bytes_released_by_control"] / 1024**3,
        "control_window_count": mechanism["control_window_count"],
        "pressure_evaluation_fraction": mechanism["pressure_target_evaluation_fraction"],
        "alpha_final_mean": mechanism["alpha_final_mean"],
        "alg1_nonzero_slack_requests": mechanism["algorithm1_nonzero_slack_request_count"],
        "hbm_admission_rejections": churn.get("HBM", {}).get("admission_rejection_count", 0),
        "hbm_evictions": churn.get("HBM", {}).get("eviction_count", 0),
        "hbm_evicted_gib": churn.get("HBM", {}).get("evicted_bytes", 0) / 1024**3,
        "dram_admission_rejections": churn.get("DRAM", {}).get("admission_rejection_count", 0),
        "dram_evictions": churn.get("DRAM", {}).get("eviction_count", 0),
        "recovery_observations": recovery_total,
        "recovery_hit_fraction": (
            (recovery_total - recompute_count) / recovery_total if recovery_total else None
        ),
        "recovery_restored_tokens": mechanism["recovery_restored_tokens"],
        "recovery_recomputed_tokens": mechanism["recovery_recomputed_tokens"],
        "raw_npz": cell["raw_npz"],
        "raw_npz_sha256": cell["raw_npz_sha256"],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    half = T_CRITICAL_95_N3 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def aggregate_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reported = (
        "healthy_wct_mean_ms",
        "all_workflow_wct_mean_ms",
        "all_workflow_added_wct_mean_ms",
        "changed_workflow_fraction",
        "diagnostic_changed_only_wct_mean_ms",
        "diagnostic_changed_only_added_wct_mean_ms",
        "healthy_copy_burden_mean_ms",
        "coupled_active_per_gpu_mean",
        "queue_wait_mean_ms",
        "drain_horizon_ms",
        "capacity_peak_serving_gib",
        "capacity_peak_backup_gib",
        "capacity_peak_shared_gib",
        "capacity_min_headroom_gib",
        "memory_blocked_unique_requests",
        "dispatch_at_control_count",
        "backup_released_at_control_gib",
        "alpha_final_mean",
        "hbm_admission_rejections",
        "hbm_evictions",
        "recovery_hit_fraction",
    )
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in metrics:
        for metric in reported:
            value = row[metric]
            if value is None or value == "":
                continue
            grouped.setdefault((row["policy"], row["failure_type"], metric), []).append(float(value))
    result = []
    for (policy, failure_type, metric), values in sorted(grouped.items()):
        mean, low, high = ci95(values)
        result.append({
            "policy": policy,
            "failure_type": failure_type,
            "metric": metric,
            "seed_count": len(values),
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
        })
    return result


def paired_contrast_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = (
        "healthy_wct_mean_ms",
        "all_workflow_wct_mean_ms",
        "all_workflow_added_wct_mean_ms",
    )
    lookup = {
        (int(row["seed"]), str(row["failure_type"]), str(row["policy"])): row
        for row in metrics
    }
    observations: dict[tuple[str, str, str], list[float]] = {}
    for seed in sorted({int(row["seed"]) for row in metrics}):
        for failure_type in sorted({str(row["failure_type"]) for row in metrics}):
            full = lookup[(seed, failure_type, "AegisKV")]
            for policy in sorted({str(row["policy"]) for row in metrics} - {"AegisKV"}):
                other = lookup[(seed, failure_type, policy)]
                for metric in primary:
                    observations.setdefault((failure_type, policy, metric), []).append(
                        float(other[metric]) - float(full[metric])
                    )
    result = []
    for (failure_type, policy, metric), values in sorted(observations.items()):
        mean, low, high = ci95(values)
        result.append({
            "failure_type": failure_type,
            "comparison_policy": policy,
            "reference_policy": "AegisKV",
            "metric": metric,
            "seed_count": len(values),
            "paired_difference_comparison_minus_full_mean": mean,
            "paired_difference_ci95_low": low,
            "paired_difference_ci95_high": high,
        })
    return result


def workflow_digest(raw: np.lib.npyio.NpzFile) -> str:
    return hashlib.sha256(json.dumps(
        raw["workflow_id"].tolist(), separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def probe_digest(raw: np.lib.npyio.NpzFile) -> str:
    payload = [
        {
            "probe_id": str(probe_id),
            "time_ms": float(time_ms),
            "source_rank": int(source_rank),
            "failure_type": str(failure_type),
        }
        for probe_id, time_ms, source_rank, failure_type in zip(
            raw["probe_id"], raw["probe_time_ms"],
            raw["probe_source_rank"], raw["probe_failure_type"],
        )
    ]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def main() -> int:
    base, capacity_protocol, spec = load_inputs()
    target = ROOT / "results" / "active_hbm_v3_paired_pilot"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()

    test_records = []
    for test_path in TESTS:
        completed = subprocess.run(
            [sys.executable, str(test_path)], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        log = stage / f"{test_path.stem}.log"
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode or "RESULT 5/5 PASS" not in completed.stdout:
            raise RuntimeError(f"v3 regression failed: {log}")
        test_records.append({
            "test": str(test_path),
            "test_sha256": core.sha256(test_path),
            "returncode": completed.returncode,
            "log": str(target / log.name),
            "log_sha256": core.sha256(log),
        })

    matrix = spec["matrix"]
    jobs = []
    for seed in matrix["paired_seeds"]:
        for failure_type in matrix["failure_types"]:
            for policy in matrix["policies"]:
                raw = stage / "raw" / f"{slug(policy)}__{failure_type}__s{seed}.npz"
                jobs.append((base, capacity_protocol, policy, int(seed), failure_type, raw))
    cells = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = {
            pool.submit(run_one, job): (job[2], job[3], job[4]) for job in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            policy, seed, failure_type = futures[future]
            cells.append(future.result())
            print(f"DONE {index}/{len(jobs)} {policy} seed={seed} {failure_type}", flush=True)
    cells.sort(key=lambda row: (row["seed"], row["failure_type"], row["policy"]))
    for cell in cells:
        raw = Path(cell["raw_npz"])
        cell["raw_npz"] = str(target / raw.relative_to(stage))

    metrics = [metric_row(cell) for cell in cells]
    aggregate = aggregate_rows(metrics)
    contrasts = paired_contrast_rows(metrics)
    write_csv(metrics, stage / "paired_metrics.csv")
    write_csv(aggregate, stage / "aggregate_ci95.csv")
    write_csv(contrasts, stage / "paired_primary_contrasts_ci95.csv")

    checks: dict[str, bool] = {
        "cell_count_exact_30": len(cells) == int(matrix["expected_cells"]),
        "matrix_exact": {
            (cell["policy"], cell["seed"], cell["failure_type"]) for cell in cells
        } == {
            (policy, int(seed), failure_type)
            for policy in matrix["policies"]
            for seed in matrix["paired_seeds"]
            for failure_type in matrix["failure_types"]
        },
        "fixed_arrival_90s": all(cell["workload"]["arrival_horizon_ms"] == 90000.0 for cell in cells),
        "fixed_probe_count_24": all(cell["fault_probe_count"] == 24 for cell in cells),
        "paired_seed_count_3": len({int(cell["seed"]) for cell in cells}) == 3,
        "same_workflow_universe_across_policies": True,
        "same_fault_probe_tuples_across_policies": True,
        "raw_id_and_probe_hashes_match_summary": True,
        "all_workflow_denominator_exact": True,
        "all_workflow_statistics_match_raw": True,
        "changed_only_statistics_match_raw": True,
        "fault_wct_monotone": True,
        "raw_hashes_match": True,
        "capacity_never_exceeded": True,
        "empty_fault_replay_exact": all(
            cell["mechanism"]["empty_fault_replay_max_completion_delta_ms"] <= 1e-7
            for cell in cells
        ),
        "dispatch_uses_no_realized_output_oracle": all(
            cell["mechanism"]["active_hbm_v3"]["dispatch_uses_no_realized_output_oracle"]
            and cell["mechanism"]["active_hbm_v3"]["request_max_output_tokens"] == 4096
            for cell in cells
        ),
        "algorithm1_switch_exercised": all(
            cell["mechanism"]["algorithm1_nonzero_slack_request_count"] > 0
            for cell in cells if cell["policy"] == "AegisKV"
        ) and all(
            cell["mechanism"]["algorithm1_nonzero_slack_request_count"] == 0
            for cell in cells if cell["policy"] == "AegisKV no Algorithm 1"
        ),
        "algorithm2_static_component_active": all(
            cell["mechanism"]["alpha_final_mean"] is None
            for cell in cells if cell["policy"] == "AegisKV no Algorithm 2"
        ),
        "algorithm3_shared_capacity_coupling_exercised": all(
            cell["mechanism"]["control_window_count"] == 3
            and cell["mechanism"]["active_hbm_v3"]["backup_bytes_released_by_control"] > 0
            and cell["mechanism"]["active_hbm_v3"]["dispatch_at_control_count"] > 0
            for cell in cells if cell["policy"] == "AegisKV"
        ),
        "noalgorithm3_control_disabled": all(
            cell["mechanism"]["control_window_count"] == 0
            and cell["mechanism"]["active_hbm_v3"]["backup_bytes_released_by_control"] == 0
            and cell["mechanism"]["active_hbm_v3"]["dispatch_at_control_count"] == 0
            for cell in cells if cell["policy"] == "AegisKV no Algorithm 3"
        ),
        "regression_logs_nonempty_10_of_10_pass": len(test_records) == 2 and all(
            Path(record["log"].replace(str(target), str(stage), 1)).stat().st_size > 0
            for record in test_records
        ),
        "no_performance_win_threshold": spec["hard_gates"]["no_performance_win_threshold"] is True,
        "changed_only_marked_secondary": "changed-only WCT and added-WCT within each policy" in spec["secondary_diagnostic_estimands"],
    }

    workflow_signatures: dict[int, str] = {}
    probe_signatures: dict[tuple[int, str], str] = {}
    for cell in cells:
        raw_path = stage / Path(cell["raw_npz"]).relative_to(target)
        if core.sha256(raw_path) != cell["raw_npz_sha256"]:
            checks["raw_hashes_match"] = False
        with np.load(raw_path, allow_pickle=False) as raw:
            if workflow_digest(raw) != cell["workflow_universe_sha256"]:
                checks["raw_id_and_probe_hashes_match_summary"] = False
            if probe_digest(raw) != cell["fault_probe_tuple_sha256"]:
                checks["raw_id_and_probe_hashes_match_summary"] = False
            workflow_key = int(cell["seed"])
            if workflow_key in workflow_signatures and workflow_signatures[workflow_key] != cell["workflow_universe_sha256"]:
                checks["same_workflow_universe_across_policies"] = False
            workflow_signatures[workflow_key] = cell["workflow_universe_sha256"]
            probe_key = (int(cell["seed"]), str(cell["failure_type"]))
            if probe_key in probe_signatures and probe_signatures[probe_key] != cell["fault_probe_tuple_sha256"]:
                checks["same_fault_probe_tuples_across_policies"] = False
            probe_signatures[probe_key] = cell["fault_probe_tuple_sha256"]

            base = raw["base_wct_ms"].astype(np.float64)
            fault = raw["fault_wct_ms"].astype(np.float64)
            if fault.shape != (int(cell["fault_probe_count"]), int(cell["workflow_count"])):
                checks["all_workflow_denominator_exact"] = False
            if np.any(fault < base[None, :] - 1e-7):
                checks["fault_wct_monotone"] = False
            delta = np.maximum(fault - base[None, :], 0.0)
            failure = cell["failure_conditioned_changed_by_type"][cell["failure_type"]]
            expected_denominator = int(cell["workflow_count"]) * int(cell["fault_probe_count"])
            if int(failure["all_workflow_observations"]) != expected_denominator:
                checks["all_workflow_denominator_exact"] = False
            if not (
                math.isclose(float(failure["all_workflow_wct_mean_ms"]), float(np.mean(fault)), abs_tol=1e-7)
                and math.isclose(float(failure["all_workflow_added_wct_mean_ms"]), float(np.mean(delta)), abs_tol=1e-7)
            ):
                checks["all_workflow_statistics_match_raw"] = False
            changed = delta > 1e-7
            changed_count = int(np.count_nonzero(changed))
            if changed_count != int(failure["changed_workflow_observations"]):
                checks["changed_only_statistics_match_raw"] = False
            if not math.isclose(
                float(failure["changed_workflow_fraction"]), changed_count / expected_denominator,
                abs_tol=1e-12,
            ):
                checks["changed_only_statistics_match_raw"] = False
            if changed_count and not (
                math.isclose(float(failure["changed_wct_mean_ms"]), float(np.mean(fault[changed])), abs_tol=1e-7)
                and math.isclose(float(failure["added_wct_mean_ms"]), float(np.mean(delta[changed])), abs_tol=1e-7)
            ):
                checks["changed_only_statistics_match_raw"] = False
            if np.any(raw["capacity_shared_used_bytes"] > capacity_protocol["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]):
                checks["capacity_never_exceeded"] = False
            if cell["mechanism"]["active_hbm_v3"]["capacity_violation_count"] != 0:
                checks["capacity_never_exceeded"] = False

    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "cell_count": len(cells),
        "workflow_universe_count": len(workflow_signatures),
        "fault_probe_tuple_set_count": len(probe_signatures),
        "primary_comparison_denominator": "all workflows times all probes in the paired cell",
        "changed_only_cross_policy_primary_comparison_allowed": False,
        "paper_ready": False,
        "physical_result": False,
        "tests": test_records,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "status": "PRELIMINARY_COMPLETE" if audit["status"] == "PASS" else "FAILED_AUDIT",
        "paper_ready": False,
        "preregistered_protocol": {"path": str(SPEC_PATH), "sha256": core.sha256(SPEC_PATH)},
        "capacity_protocol": {"path": str(CAPACITY_PROTOCOL_PATH), "sha256": core.sha256(CAPACITY_PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "v3_des": {"path": str(Path(v3.__file__).resolve()), "sha256": core.sha256(Path(v3.__file__).resolve())},
        "core_runner": {"path": str(Path(core.__file__).resolve()), "sha256": core.sha256(Path(core.__file__).resolve())},
        "primary_estimands": spec["primary_estimands"],
        "secondary_diagnostic_estimands": spec["secondary_diagnostic_estimands"],
        "cells": cells,
    }
    (stage / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}:
            artifacts[str(path.relative_to(stage))] = core.sha256(path)
    manifest = {
        "schema_version": 1,
        "status": audit["status"],
        "paper_ready": False,
        "physical_result": False,
        "target": str(target),
        "artifacts": artifacts,
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        + f"{core.sha256(stage / 'manifest.json')}  manifest.json\n",
        encoding="utf-8",
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"v3 paired pilot failed in {stage}: {audit['failed_checks']}")
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
