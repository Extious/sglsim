#!/usr/bin/env python3
"""Run the preregistered v3.1.1 cold-start and fixed steady-cohort audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import active_hbm_v31 as v31  # noqa: E402
import run_mechanism_pilot_v2 as pilot  # noqa: E402
import run_preliminary as core  # noqa: E402


SPEC_PATH = ROOT / "active_hbm_v311_cold_start_diagnostic_protocol.json"
CAPACITY_PATH = ROOT / "active_hbm_v3_protocol.json"
TEST_PATH = ROOT / "test_active_hbm_v31.py"
T95_N3 = 4.302652729911275
GIB = 1024**3


def sha_payload(values: Any) -> str:
    return hashlib.sha256(json.dumps(
        values, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def workflow_components(tasks: list[core.Task]) -> list[dict[str, Any]]:
    grouped: dict[str, list[core.Task]] = defaultdict(list)
    for task in tasks:
        grouped[task.workflow_id].append(task)
    result = []
    for workflow_id in sorted(grouped):
        critical = max(grouped[workflow_id], key=lambda row: (row.finish_ms, row.request_id))
        queue_ms = critical.start_ms - critical.workflow_arrival_ms
        burden_ms = critical.finish_ms - critical.start_ms - critical.service_ms
        wct_ms = critical.finish_ms - critical.workflow_arrival_ms
        if not math.isclose(
            wct_ms, queue_ms + critical.service_ms + burden_ms, abs_tol=1e-7
        ):
            raise AssertionError("cold-start workflow decomposition failed")
        result.append({
            "workflow_id": workflow_id,
            "arrival_ms": critical.workflow_arrival_ms,
            "wct_ms": wct_ms,
            "queue_ms": queue_ms,
            "service_ms": critical.service_ms,
            "copy_burden_ms": burden_ms,
        })
    return result


def cohort_metrics(rows: list[dict[str, Any]], left: float, right: float) -> dict[str, Any]:
    cohort = [row for row in rows if left <= float(row["arrival_ms"]) < right]
    return {
        "workflow_count": len(cohort),
        "workflow_id_sha256": sha_payload([row["workflow_id"] for row in cohort]),
        "healthy_wct_mean_ms": statistics.fmean(row["wct_ms"] for row in cohort),
        "critical_queue_mean_ms": statistics.fmean(row["queue_ms"] for row in cohort),
        "critical_service_mean_ms": statistics.fmean(row["service_ms"] for row in cohort),
        "critical_copy_burden_mean_ms": statistics.fmean(row["copy_burden_ms"] for row in cohort),
    }


def integrate_capacity_windows(
    trace: list[dict[str, Any]], *, dp: int, horizon_ms: float, window_ms: float,
) -> list[dict[str, Any]]:
    fields = (
        "active_requests", "waiting_requests", "active_serving_kv_bytes",
        "peer_hbm_backup_used_bytes", "shared_headroom_bytes",
    )
    states = [{field: 0.0 for field in fields} for _ in range(dp)]
    window_count = int(math.ceil(horizon_ms / window_ms))
    area = [{field: 0.0 for field in fields} for _ in range(window_count)]

    def integrate(left: float, right: float) -> None:
        cursor = max(left, 0.0)
        right = min(right, horizon_ms)
        totals = {field: sum(state[field] for state in states) for field in fields}
        while cursor < right - 1e-12:
            index = min(int(cursor // window_ms), window_count - 1)
            boundary = min((index + 1) * window_ms, right)
            duration = boundary - cursor
            for field in fields:
                area[index][field] += totals[field] * duration
            cursor = boundary

    prior = 0.0
    for row in trace:
        at = float(row["time_ms"])
        if at > horizon_ms:
            break
        integrate(prior, at)
        rank = int(row["rank"])
        for field in fields:
            states[rank][field] = float(row[field])
        prior = at
    integrate(prior, horizon_ms)
    result = []
    for index, values in enumerate(area):
        left = index * window_ms
        right = min((index + 1) * window_ms, horizon_ms)
        duration = right - left
        result.append({
            "window_start_ms": left,
            "window_end_ms": right,
            "active_requests_per_gpu_time_mean": values["active_requests"] / duration / dp,
            "waiting_requests_per_gpu_time_mean": values["waiting_requests"] / duration / dp,
            "serving_gib_per_gpu_time_mean": values["active_serving_kv_bytes"] / duration / dp / GIB,
            "peer_backup_gib_per_gpu_time_mean": values["peer_hbm_backup_used_bytes"] / duration / dp / GIB,
            "shared_headroom_gib_per_gpu_time_mean": values["shared_headroom_bytes"] / duration / dp / GIB,
        })
    return result


def run_one(args: tuple[dict, dict, str, int, float | None, float | None, Path]) -> dict[str, Any]:
    base, capacity, policy, seed, initial_alpha, control_window_ms, raw_path = args
    protocol = deepcopy(base)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["workload"]["center_context_tokens"] = 40960
    protocol["workload"]["center_active_requests_per_gpu"] = 16
    if policy == "AegisKV":
        protocol["aegiskv"]["initial_alpha"] = float(initial_alpha)
        protocol["aegiskv"]["window_ms"] = float(control_window_ms)
    nominal, workload = v31.build_workload(
        protocol, seed=seed, active_per_gpu=16, arrival_cutoff_ms=180000.0
    )
    replay = v31.ExactAdmissionProjectionReplay(protocol, policy, capacity)
    tasks, delay_maps, mechanism, trace = replay.run_coupled(
        nominal, [], control_horizon_ms=180000.0
    )
    if delay_maps:
        raise AssertionError("healthy cold-start diagnostic generated fault delays")
    workflow_rows = workflow_components(tasks)
    inclusive = cohort_metrics(workflow_rows, 0.0, 180000.0)
    steady = cohort_metrics(workflow_rows, 90000.0, 180000.0)
    windows = integrate_capacity_windows(
        trace, dp=16, horizon_ms=180000.0, window_ms=30000.0
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_path,
        workflow_id=np.asarray([row["workflow_id"] for row in workflow_rows]),
        workflow_arrival_ms=np.asarray([row["arrival_ms"] for row in workflow_rows]),
        healthy_wct_ms=np.asarray([row["wct_ms"] for row in workflow_rows]),
        critical_queue_ms=np.asarray([row["queue_ms"] for row in workflow_rows]),
        critical_service_ms=np.asarray([row["service_ms"] for row in workflow_rows]),
        critical_copy_burden_ms=np.asarray([row["copy_burden_ms"] for row in workflow_rows]),
        capacity_time_ms=np.asarray([row["time_ms"] for row in trace]),
        capacity_rank=np.asarray([row["rank"] for row in trace], dtype=np.int16),
        capacity_active_requests=np.asarray([row["active_requests"] for row in trace]),
        capacity_waiting_requests=np.asarray([row["waiting_requests"] for row in trace]),
        capacity_active_serving_kv_bytes=np.asarray([row["active_serving_kv_bytes"] for row in trace]),
        capacity_peer_hbm_backup_used_bytes=np.asarray([row["peer_hbm_backup_used_bytes"] for row in trace]),
        capacity_shared_used_bytes=np.asarray([row["shared_used_bytes"] for row in trace]),
        capacity_shared_headroom_bytes=np.asarray([row["shared_headroom_bytes"] for row in trace]),
    )
    return {
        "policy": policy,
        "seed": seed,
        "initial_alpha": initial_alpha,
        "control_window_ms": control_window_ms,
        "configuration_id": (
            "native_reference" if policy == "Native Mooncake"
            else f"aegis_alpha_{initial_alpha}_window_{int(control_window_ms)}"
        ),
        "workflow_universe_sha256": inclusive["workflow_id_sha256"],
        "inclusive": inclusive,
        "steady": steady,
        "windows": windows,
        "control_steps": mechanism["control_step_trace"],
        "control_capacity_steps": mechanism["active_hbm_v3"]["control_capacity_trace"],
        "mechanism": {
            "alpha_final_mean": mechanism["alpha_final_mean"],
            "control_window_count": mechanism["control_window_count"],
            "copy_burden_mean_ms": mechanism["normal_path_copy_burden_mean_ms"],
            "queue_wait_mean_ms": mechanism["active_hbm_v3"]["queue_wait_mean_ms"],
            "capacity_peak_shared_gib": mechanism["active_hbm_v3"]["capacity_peak_shared_used_bytes"] / GIB,
            "capacity_min_headroom_gib": mechanism["active_hbm_v3"]["capacity_min_headroom_bytes"] / GIB,
            "capacity_violation_count": mechanism["active_hbm_v3"]["capacity_violation_count"],
            "old_projection_false_block_unique_requests": mechanism["active_hbm_v31"]["old_projection_false_block_unique_request_count"],
            "projection_prediction_mismatch_count": mechanism["active_hbm_v31"]["projection_prediction_mismatch_count"],
        },
        "workload": workload,
        "raw_npz": str(raw_path),
        "raw_npz_sha256": core.sha256(raw_path),
    }


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text())
    if core.sha256(ROOT / spec["v311_protocol"]) != spec["v311_protocol_sha256"]:
        raise RuntimeError("v3.1.1 protocol changed")
    if core.sha256(ROOT / spec["v31_des"]) != spec["v31_des_sha256"]:
        raise RuntimeError("v3.1 DES changed")
    base, _parent = pilot.load_composite_protocol()
    capacity = json.loads(CAPACITY_PATH.read_text())
    target = ROOT / "results" / "active_hbm_v311_cold_start_diagnostic"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()
    test = subprocess.run(
        [sys.executable, str(TEST_PATH)], cwd=str(ROOT), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (stage / "test_active_hbm_v31.log").write_text(test.stdout)
    if test.returncode or "RESULT 7/7 PASS" not in test.stdout:
        raise RuntimeError("v3.1 regression failed")
    jobs = []
    for seed in spec["matrix"]["paired_seeds"]:
        raw = stage / "raw" / f"native_reference__s{seed}.npz"
        jobs.append((base, capacity, "Native Mooncake", int(seed), None, None, raw))
        for alpha in spec["matrix"]["aegiskv_initial_alpha"]:
            for window in spec["matrix"]["aegiskv_control_window_ms"]:
                raw = stage / "raw" / f"aegis_a{alpha}_w{int(window)}__s{seed}.npz"
                jobs.append((base, capacity, "AegisKV", int(seed), float(alpha), float(window), raw))
    cells = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(run_one, job): (job[2], job[3], job[4], job[5]) for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            cells.append(future.result())
            print(f"DONE {index}/{len(jobs)} {key}", flush=True)
    cells.sort(key=lambda row: (row["seed"], row["configuration_id"]))
    for cell in cells:
        raw = Path(cell["raw_npz"])
        cell["raw_npz"] = str(target / raw.relative_to(stage))

    metrics = []
    window_rows = []
    control_rows = []
    for cell in cells:
        row = {
            "configuration_id": cell["configuration_id"],
            "policy": cell["policy"], "seed": cell["seed"],
            "initial_alpha": cell["initial_alpha"],
            "control_window_ms": cell["control_window_ms"],
            "workflow_universe_sha256": cell["workflow_universe_sha256"],
            "inclusive_workflow_count": cell["inclusive"]["workflow_count"],
            "inclusive_healthy_wct_mean_ms": cell["inclusive"]["healthy_wct_mean_ms"],
            "inclusive_critical_queue_mean_ms": cell["inclusive"]["critical_queue_mean_ms"],
            "steady_workflow_count": cell["steady"]["workflow_count"],
            "steady_workflow_id_sha256": cell["steady"]["workflow_id_sha256"],
            "steady_healthy_wct_mean_ms": cell["steady"]["healthy_wct_mean_ms"],
            "steady_critical_queue_mean_ms": cell["steady"]["critical_queue_mean_ms"],
            **cell["mechanism"],
            "raw_npz": cell["raw_npz"], "raw_npz_sha256": cell["raw_npz_sha256"],
        }
        metrics.append(row)
        for window in cell["windows"]:
            window_rows.append({
                "configuration_id": cell["configuration_id"],
                "policy": cell["policy"], "seed": cell["seed"],
                "initial_alpha": cell["initial_alpha"],
                "control_window_ms": cell["control_window_ms"],
                **window,
            })
        capacity_by_time = {
            float(step["time_ms"]): step for step in cell["control_capacity_steps"]
        }
        for step in cell["control_steps"]:
            capacity_step = capacity_by_time[float(step["time_ms"])]
            control_rows.append({
                "configuration_id": cell["configuration_id"],
                "seed": cell["seed"], "initial_alpha": cell["initial_alpha"],
                "control_window_ms": cell["control_window_ms"],
                **step,
                "backup_released_gib": capacity_step["backup_released_bytes"] / GIB,
                "dispatched_after_control": capacity_step["dispatched_after_control"],
                "waiting_before": capacity_step["waiting_before"],
                "waiting_after": capacity_step["waiting_after"],
            })
    write_csv(metrics, stage / "cohort_metrics.csv")
    write_csv(window_rows, stage / "fixed_30s_window_integrals.csv")
    write_csv(control_rows, stage / "control_steps.csv")

    aggregate = []
    for configuration in sorted({row["configuration_id"] for row in metrics}):
        selected = [row for row in metrics if row["configuration_id"] == configuration]
        for metric in (
            "inclusive_healthy_wct_mean_ms", "inclusive_critical_queue_mean_ms",
            "steady_healthy_wct_mean_ms", "steady_critical_queue_mean_ms",
            "queue_wait_mean_ms", "alpha_final_mean",
        ):
            values = [float(row[metric]) for row in selected if row[metric] not in (None, "")]
            if not values:
                continue
            mean = statistics.fmean(values)
            half = T95_N3 * statistics.stdev(values) / math.sqrt(len(values))
            aggregate.append({
                "configuration_id": configuration, "metric": metric,
                "seed_count": len(values), "mean": mean,
                "ci95_low": mean - half, "ci95_high": mean + half,
            })
    write_csv(aggregate, stage / "aggregate_ci95.csv")

    lookup = {(row["seed"], row["configuration_id"]): row for row in metrics}
    paired = []
    for row in metrics:
        if row["policy"] != "AegisKV":
            continue
        native = lookup[(row["seed"], "native_reference")]
        for cohort in ("inclusive", "steady"):
            metric = f"{cohort}_healthy_wct_mean_ms"
            paired.append({
                "configuration_id": row["configuration_id"], "seed": row["seed"],
                "cohort": cohort,
                "aegis_wct_mean_ms": row[metric], "native_wct_mean_ms": native[metric],
                "aegis_minus_native_ms": float(row[metric]) - float(native[metric]),
            })
    write_csv(paired, stage / "paired_aegis_minus_native.csv")

    checks = {
        "cell_count_exact_21": len(cells) == int(spec["matrix"]["expected_cells"]),
        "raw_count_exact_21": len(list((stage / "raw").glob("*.npz"))) == 21,
        "workflow_universe_common_by_seed": True,
        "steady_cohort_common_by_seed": True,
        "raw_hashes_match": True,
        "capacity_never_exceeded": True,
        "projection_matches_actual": all(row["projection_prediction_mismatch_count"] == 0 for row in metrics),
        "inclusive_contains_first_30s": all(
            any(
                window["configuration_id"] == row["configuration_id"]
                and int(window["seed"]) == int(row["seed"])
                and float(window["window_start_ms"]) == 0.0
                for window in window_rows
            ) for row in metrics
        ),
        "six_fixed_reporting_windows_per_cell": len(window_rows) == 21 * 6,
        "control_step_counts_exact": all(
            row["control_window_count"] == 180000.0 / float(row["control_window_ms"])
            for row in metrics if row["policy"] == "AegisKV"
        ),
        "no_performance_selection": spec["hard_gates"]["no_performance_direction_or_win_threshold"] is True,
        "regression_7_of_7_pass": True,
    }
    for seed in spec["matrix"]["paired_seeds"]:
        selected = [row for row in metrics if int(row["seed"]) == int(seed)]
        if len({row["workflow_universe_sha256"] for row in selected}) != 1:
            checks["workflow_universe_common_by_seed"] = False
        if len({row["steady_workflow_id_sha256"] for row in selected}) != 1:
            checks["steady_cohort_common_by_seed"] = False
    for cell in cells:
        raw = stage / Path(cell["raw_npz"]).relative_to(target)
        if core.sha256(raw) != cell["raw_npz_sha256"]:
            checks["raw_hashes_match"] = False
        with np.load(raw, allow_pickle=False) as data:
            if np.any(data["capacity_shared_used_bytes"] > capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]):
                checks["capacity_never_exceeded"] = False
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "paper_ready": False,
        "fault_result": False,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (stage / "summary.json").write_text(json.dumps({
        "schema_version": 1, "status": audit["status"], "paper_ready": False,
        "protocol": {"path": str(SPEC_PATH), "sha256": core.sha256(SPEC_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "cells": cells,
    }, indent=2, sort_keys=True) + "\n")
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    }
    (stage / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "status": audit["status"], "paper_ready": False,
        "target": str(target), "artifacts": artifacts,
    }, indent=2, sort_keys=True) + "\n")
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        + f"{core.sha256(stage / 'manifest.json')}  manifest.json\n"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"cold-start audit failed: {audit['failed_checks']}")
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
