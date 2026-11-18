#!/usr/bin/env python3
"""Run the preregistered 10-cell active-serving-HBM v3 mechanism gate."""

from __future__ import annotations

import csv
import json
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


CAPACITY_PROTOCOL_PATH = ROOT / "active_hbm_v3_protocol.json"
TESTS = (ROOT / "test_active_hbm_v3_capacity.py", ROOT / "test_active_hbm_v3_des.py")


def run_one(
    args: tuple[dict[str, Any], dict[str, Any], str, str, Path],
) -> dict[str, Any]:
    base, capacity_protocol, policy, failure_type, raw = args
    protocol = deepcopy(base)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["workload"]["center_context_tokens"] = 40960
    protocol["failure_model"]["probe_type_schedule"] = [failure_type]
    return v3.run_v3_cell(
        protocol,
        capacity_protocol,
        policy=policy,
        seed=2026082600,
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
        "normal_copy_burden_mean_ms": mechanism["normal_path_copy_burden_mean_ms"],
        "coupled_realized_active_per_gpu_mean": cell["workload"]["coupled_realized_active_requests_per_gpu_time_mean"],
        "coupled_realized_active_per_gpu_peak": cell["workload"]["coupled_realized_active_requests_per_gpu_peak"],
        "drain_horizon_ms": cell["workload"]["drain_horizon_ms"],
        "queue_wait_mean_ms": capacity["queue_wait_mean_ms"],
        "queue_wait_p95_ms": capacity["queue_wait_p95_ms"],
        "memory_blocked_unique_requests": capacity["memory_blocked_unique_request_count"],
        "dispatch_at_control_count": capacity["dispatch_at_control_count"],
        "backup_released_at_control_bytes": capacity["backup_bytes_released_by_control"],
        "capacity_peak_serving_gib": capacity["capacity_peak_serving_bytes"] / 1024**3,
        "capacity_peak_backup_gib": capacity["capacity_peak_backup_bytes"] / 1024**3,
        "capacity_peak_shared_gib": capacity["capacity_peak_shared_used_bytes"] / 1024**3,
        "capacity_min_headroom_gib": capacity["capacity_min_headroom_bytes"] / 1024**3,
        "capacity_violation_count": capacity["capacity_violation_count"],
        "alpha_final_mean": mechanism["alpha_final_mean"],
        "control_window_count": mechanism["control_window_count"],
        "pressure_evaluation_fraction": mechanism["pressure_target_evaluation_fraction"],
        "alg1_nonzero_slack_requests": mechanism["algorithm1_nonzero_slack_request_count"],
        "hbm_rejections": churn.get("HBM", {}).get("admission_rejection_count", 0),
        "hbm_evictions": churn.get("HBM", {}).get("eviction_count", 0),
        "recovery_restored_tokens": mechanism["recovery_restored_tokens"],
        "recovery_recomputed_tokens": mechanism["recovery_recomputed_tokens"],
        "raw_npz": cell["raw_npz"],
        "raw_npz_sha256": cell["raw_npz_sha256"],
    }


def main() -> int:
    target = ROOT / "results" / "active_hbm_v3_gate"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()
    capacity_protocol = json.loads(CAPACITY_PROTOCOL_PATH.read_text(encoding="utf-8"))
    base, _spec = pilot.load_composite_protocol()
    test_records = []
    for test_path in TESTS:
        completed = subprocess.run(
            [sys.executable, str(test_path)], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        log = stage / f"{test_path.stem}.log"
        log.write_text(completed.stdout, encoding="utf-8")
        expected = "RESULT 5/5 PASS"
        if completed.returncode or expected not in completed.stdout:
            raise RuntimeError(f"v3 regression failed: {log}")
        test_records.append({
            "path": str(test_path), "sha256": core.sha256(test_path),
            "returncode": completed.returncode,
            "log": str(target / log.name), "log_sha256": core.sha256(log),
        })
    policies = capacity_protocol["pilot_matrix"]["policies"]
    jobs = []
    for failure_type in capacity_protocol["pilot_matrix"]["failure_types"]:
        for policy in policies:
            label = policy.lower().replace(" ", "_")
            raw = stage / "raw" / f"{label}__{failure_type}__s2026082600.npz"
            jobs.append((base, capacity_protocol, policy, failure_type, raw))
    cells = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(run_one, job): (job[2], job[3]) for job in jobs}
        for future in as_completed(futures):
            policy, failure_type = futures[future]
            cells.append(future.result())
            print(f"DONE {policy} {failure_type}", flush=True)
    cells.sort(key=lambda row: (row["failure_type"], row["policy"]))
    for cell in cells:
        raw = Path(cell["raw_npz"])
        cell["raw_npz"] = str(target / raw.relative_to(stage))
    metrics = [metric_row(cell) for cell in cells]
    metrics_path = stage / "capacity_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    checks: dict[str, bool] = {
        "ten_cell_matrix_exact": len(cells) == 10,
        "no_performance_win_threshold": capacity_protocol["registered_gates"]["no_performance_win_threshold"] is True,
        "dispatch_uses_no_realized_output_oracle": all(
            cell["mechanism"]["active_hbm_v3"]["dispatch_uses_no_realized_output_oracle"]
            and cell["mechanism"]["active_hbm_v3"]["request_max_output_tokens"] == 4096
            for cell in cells
        ),
        "capacity_never_exceeded": all(
            cell["mechanism"]["active_hbm_v3"]["capacity_violation_count"] == 0
            and cell["mechanism"]["active_hbm_v3"]["capacity_peak_total_hbm_bytes"]
            <= cell["mechanism"]["active_hbm_v3"]["total_hbm_bytes"]
            for cell in cells
        ),
        "memory_queue_exercised": all(
            cell["mechanism"]["active_hbm_v3"]["memory_blocked_unique_request_count"] > 0
            for cell in cells
        ),
        "empty_replay_preserves_coupled_completion": all(
            cell["mechanism"]["empty_fault_replay_max_completion_delta_ms"] <= 1e-7
            for cell in cells
        ),
        "fault_wct_monotone": True,
        "raw_hashes_match": True,
        "paired_workload_and_probes": True,
        "algorithm1_switch_exercised": all(
            cell["mechanism"]["algorithm1_nonzero_slack_request_count"] > 0
            for cell in cells if cell["policy"] == "AegisKV"
        ) and all(
            cell["mechanism"]["algorithm1_nonzero_slack_request_count"] == 0
            for cell in cells if cell["policy"] == "AegisKV no Algorithm 1"
        ),
        "algorithm3_releases_backup_and_dispatches": all(
            cell["mechanism"]["control_window_count"] == 3
            and cell["mechanism"]["active_hbm_v3"]["backup_bytes_released_by_control"] > 0
            and cell["mechanism"]["active_hbm_v3"]["dispatch_at_control_count"] > 0
            for cell in cells if cell["policy"] == "AegisKV"
        ),
        "noalgorithm3_keeps_control_disabled": all(
            cell["mechanism"]["control_window_count"] == 0
            and cell["mechanism"]["active_hbm_v3"]["backup_bytes_released_by_control"] == 0
            and cell["mechanism"]["active_hbm_v3"]["dispatch_at_control_count"] == 0
            for cell in cells if cell["policy"] == "AegisKV no Algorithm 3"
        ),
        "failure_types_have_changed_workflows": all(
            cell["failure_conditioned_changed_by_type"][cell["failure_type"]]["changed_workflow_observations"] > 0
            for cell in cells
        ),
        "node_failure_never_uses_same_node_peer_hbm": all(
            not any(key.startswith("source_node:peer_hbm") for key in cell["mechanism"]["recovery_type_path_counts"])
            for cell in cells if cell["failure_type"] == "source_node"
        ),
        "regressions_10_of_10_pass": True,
    }
    signatures = {}
    for cell in cells:
        raw = stage / Path(cell["raw_npz"]).relative_to(target)
        if core.sha256(raw) != cell["raw_npz_sha256"]:
            checks["raw_hashes_match"] = False
        with np.load(raw, allow_pickle=False) as data:
            if np.any(data["fault_wct_ms"] < data["base_wct_ms"][None, :] - 1e-7):
                checks["fault_wct_monotone"] = False
            if np.any(data["capacity_shared_used_bytes"] > capacity_protocol["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]):
                checks["capacity_never_exceeded"] = False
            signature = (
                tuple(data["workflow_id"].tolist()), data["probe_time_ms"].tobytes(),
                data["probe_source_rank"].tobytes(), tuple(data["probe_failure_type"].tolist()),
            )
            key = cell["failure_type"]
            if key in signatures and signatures[key] != signature:
                checks["paired_workload_and_probes"] = False
            signatures[key] = signature
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "cell_count": len(cells),
        "paper_ready": False,
        "physical_result": False,
        "tests": test_records,
        "evidence_boundary": capacity_protocol["evidence_boundary"],
    }
    audit_path = stage / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "status": "PRELIMINARY_COMPLETE",
        "paper_ready": False,
        "capacity_protocol": {"path": str(CAPACITY_PROTOCOL_PATH), "sha256": core.sha256(CAPACITY_PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "v3_des": {"path": str(Path(v3.__file__).resolve()), "sha256": core.sha256(Path(v3.__file__).resolve())},
        "core_runner": {"path": str(Path(core.__file__).resolve()), "sha256": core.sha256(Path(core.__file__).resolve())},
        "cells": cells,
    }
    summary_path = stage / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}:
            artifacts[str(path.relative_to(stage))] = core.sha256(path)
    manifest = {
        "schema_version": 1, "status": audit["status"], "paper_ready": False,
        "target": str(target), "artifacts": artifacts,
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        + f"{core.sha256(manifest_path)}  manifest.json\n", encoding="utf-8",
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"v3 gate failed in {stage}: {audit['failed_checks']}")
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
