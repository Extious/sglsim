#!/usr/bin/env python3
"""Run the preregistered mechanism-first SGLSim high-pressure pilot."""

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

import run_preliminary as core  # noqa: E402


SPEC_PATH = ROOT / "mechanism_pilot_v2_protocol.json"
BASE_PROTOCOL_PATH = ROOT / "protocol.json"
PHYSICAL_ROOT = Path(
    "/home/comp/zhanzhao/Mooncake/aegiskv/results/physical_profiles/"
    "rtx4090-pcie-host-staged-141298-v1"
)
PHYSICAL_CSV = PHYSICAL_ROOT / "rtx4090-pcie-host-staged-four-path-profile.csv"
PHYSICAL_MANIFEST = PHYSICAL_ROOT / "physical-profile-manifest.json"
TEST_PATH = ROOT / "test_mechanism_pilot_v2.py"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_composite_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if core.sha256(BASE_PROTOCOL_PATH) != spec["parent_protocol_sha256"]:
        raise RuntimeError("parent protocol changed after pilot preregistration")
    base = json.loads(BASE_PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = deep_merge(base, spec["overrides"])
    if protocol["design_semantics"] != spec["design_semantics"]:
        raise RuntimeError("pilot design semantics mismatch")
    return protocol, spec


def run_one(args: tuple[dict[str, Any], str, int, int, float, Path]) -> dict[str, Any]:
    protocol, policy, seed, probes, cutoff, raw_path = args
    return core.run_cell(
        protocol,
        policy=policy,
        seed=seed,
        workflows=None,
        active_per_gpu=int(protocol["workload"]["center_active_requests_per_gpu"]),
        probe_count=probes,
        output=raw_path,
        arrival_cutoff_ms=cutoff,
    )


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    half = 4.302652729911275 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def write_aggregate(cells: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    metrics = (
        "base_wct_mean_ms",
        "failure_changed_wct_mean_ms",
        "failure_changed_added_wct_mean_ms",
    )
    rows: list[dict[str, Any]] = []
    policies = sorted({str(cell["policy"]) for cell in cells})
    for policy in policies:
        selected = [cell for cell in cells if cell["policy"] == policy]
        for metric in metrics:
            values = [float(cell[metric]) for cell in selected if cell[metric] is not None]
            mean, low, high = mean_ci(values)
            rows.append({
                "policy": policy,
                "failure_type": "all",
                "metric": metric,
                "seed_count": len(values),
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
            })
        for failure_type in ("source_worker_gpu", "source_node"):
            for metric in ("changed_wct_mean_ms", "added_wct_mean_ms"):
                values = [
                    float(cell["failure_conditioned_changed_by_type"][failure_type][metric])
                    for cell in selected
                    if failure_type in cell["failure_conditioned_changed_by_type"]
                    and cell["failure_conditioned_changed_by_type"][failure_type][metric]
                    is not None
                ]
                mean, low, high = mean_ci(values)
                rows.append({
                    "policy": policy,
                    "failure_type": failure_type,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_trigger_table(cells: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    rows = []
    for cell in sorted(cells, key=lambda row: (row["policy"], row["seed"])):
        mechanism = cell["mechanism"]
        churn = mechanism.get("tier_churn", {})
        rows.append({
            "policy": cell["policy"],
            "seed": cell["seed"],
            "workflows": cell["workflow_count"],
            "healthy_wct_mean_ms": cell["base_wct_mean_ms"],
            "healthy_copy_burden_mean_ms": mechanism["normal_path_copy_burden_mean_ms"],
            "copy_service_mean_ms": mechanism["copy_transfer_service_mean_ms"],
            "copy_queue_wait_mean_ms": mechanism["copy_queue_wait_mean_ms"],
            "copy_queue_wait_positive_count": mechanism["copy_queue_wait_positive_count"],
            "hbm_admission_rejections": churn.get("HBM", {}).get("admission_rejection_count", 0),
            "dram_admission_rejections": churn.get("DRAM", {}).get("admission_rejection_count", 0),
            "hbm_evictions": churn.get("HBM", {}).get("eviction_count", 0),
            "dram_evictions": churn.get("DRAM", {}).get("eviction_count", 0),
            "alg1_nonzero_slack_requests": mechanism["algorithm1_nonzero_slack_request_count"],
            "alg3_control_windows": mechanism["control_window_count"],
            "alg3_pressure_evaluations": mechanism["pressure_target_evaluations"],
            "recovery_restored_tokens": mechanism["recovery_restored_tokens"],
            "recovery_recomputed_tokens": mechanism["recovery_recomputed_tokens"],
            "recovery_hit_fraction": mechanism["recovery_hit_fraction"],
            "failover_mode_counts": json.dumps(mechanism["failover_mode_counts"], sort_keys=True),
            "recovery_type_path_counts": json.dumps(mechanism["recovery_type_path_counts"], sort_keys=True),
            "residency_event_counts": json.dumps(mechanism["residency_event_counts"], sort_keys=True),
            "fixed_point_iterations": mechanism["normal_path_schedule_solver"]["iterations"],
            "fixed_point_residual_ms": mechanism["normal_path_schedule_solver"]["final_burden_residual_finish_ms"],
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_replacement_sensitivity(
    cells: list[dict[str, Any]], path: Path,
) -> list[dict[str, Any]]:
    rows = []
    for delay in sorted({float(cell["replacement_delay_ms"]) for cell in cells}):
        selected = [cell for cell in cells if float(cell["replacement_delay_ms"]) == delay]
        for failure_type in ("source_worker_gpu", "source_node"):
            values = [
                float(cell["failure_conditioned_changed_by_type"][failure_type]["changed_wct_mean_ms"])
                for cell in selected
            ]
            added = [
                float(cell["failure_conditioned_changed_by_type"][failure_type]["added_wct_mean_ms"])
                for cell in selected
            ]
            mean, low, high = mean_ci(values)
            added_mean, added_low, added_high = mean_ci(added)
            rows.append({
                "policy": "Native Mooncake",
                "replacement_delay_ms": delay,
                "replacement_delay_provenance": "MODELED_UNMEASURED",
                "failure_type": failure_type,
                "seed_count": len(values),
                "changed_wct_mean_ms": mean,
                "changed_wct_ci95_low_ms": low,
                "changed_wct_ci95_high_ms": high,
                "added_wct_mean_ms": added_mean,
                "added_wct_ci95_low_ms": added_low,
                "added_wct_ci95_high_ms": added_high,
            })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_calibration_diagnostic(protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    path_map = {
        "source_to_peer_hbm": ("copy", "peer_hbm_same_node"),
        "source_to_dram": ("copy", "dram_local"),
        "peer_hbm_recovery": ("load", "peer_hbm_same_node"),
        "dram_recovery": ("load", "dram_local"),
    }
    rows = []
    with PHYSICAL_CSV.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            phase, model_path = path_map[source["path"]]
            model = float(protocol["hardware"][f"{phase}_bandwidth_gb_s"][model_path])
            physical = float(source["physical_gib_s_p50"])
            rows.append({
                "payload_mib": int(source["payload_mib"]),
                "physical_direction": source["path"],
                "simulator_phase": phase,
                "simulator_path": model_path,
                "physical_rtx4090_gib_s_p50": physical,
                "main_a100_model_gb_s_decimal": model,
                "model_to_physical_numeric_ratio": model / physical,
                "comparison_status": "DIAGNOSTIC_ONLY_HARDWARE_AND_UNIT_UNMATCHED",
            })
    csv_path = output_dir / "calibration_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_payload = {}
    for payload in sorted({row["payload_mib"] for row in rows}):
        selected = {row["physical_direction"]: row for row in rows if row["payload_mib"] == payload}
        by_payload[str(payload)] = {
            "peer_recovery_to_copy_physical_ratio": (
                selected["peer_hbm_recovery"]["physical_rtx4090_gib_s_p50"]
                / selected["source_to_peer_hbm"]["physical_rtx4090_gib_s_p50"]
            ),
            "dram_recovery_to_copy_physical_ratio": (
                selected["dram_recovery"]["physical_rtx4090_gib_s_p50"]
                / selected["source_to_dram"]["physical_rtx4090_gib_s_p50"]
            ),
        }
    payload = {
        "status": "PASS_DIAGNOSTIC",
        "paper_ready_calibration": False,
        "physical_hardware": "RTX4090 PCIe host-staged",
        "simulator_hardware": "modeled A100 40 GiB",
        "physical_manifest": {"path": str(PHYSICAL_MANIFEST), "sha256": core.sha256(PHYSICAL_MANIFEST)},
        "physical_csv": {"path": str(PHYSICAL_CSV), "sha256": core.sha256(PHYSICAL_CSV)},
        "row_count": len(rows),
        "copy_and_recovery_directions_are_distinct": True,
        "direction_ratios_by_payload": by_payload,
        "numeric_ratio_range": [
            min(row["model_to_physical_numeric_ratio"] for row in rows),
            max(row["model_to_physical_numeric_ratio"] for row in rows),
        ],
        "unit_note": "Simulator inputs are decimal GB/s while physical results are GiB/s; the at-most 7.4 percent unit conversion cannot explain the order-of-magnitude mismatch.",
        "conclusion": "The old shared bandwidth scalar was semantically invalid for the measured staged path. Split copy/load parameters are now wired, but no A100-to-RTX4090 calibration claim is made.",
    }
    json_path = output_dir / "calibration_diagnostic.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_difference_audit(output_dir: Path) -> dict[str, Any]:
    rows = {
        "S1": {"old_difference": "architecture CDFs overlap under a common 60 s delay", "falsifiable_hypothesis": "policy-specific failover plus preserved health burden separates worker-failure changed WCT", "pilot_test": "Native versus Full, source_worker_gpu changed-workflow WCT"},
        "S2": {"old_difference": "three sparse rates visually overlap", "falsifiable_hypothesis": "90 s x DP64 exposure is too small for rate separation", "pilot_test": "direct failure-conditioned changed WCT is primary; official-rate mixture remains secondary"},
        "S3": {"old_difference": "scale/context effects near zero", "falsifiable_hypothesis": "fixed 60 s replacement masks context-sensitive transfer/recompute", "pilot_test": "40K high-pressure mechanism cell only; scale sweep not rerun"},
        "S4": {"old_difference": "Algorithm ablations are ppm-level", "falsifiable_hypothesis": "center cell did not trigger slack, capacity admission, or AIMD", "pilot_test": "all three switch trigger counts and WCT at c16/context40K"},
        "S5": {"old_difference": "alpha load evidence was endpoint-heavy", "falsifiable_hypothesis": "fixed windows and pressure counters expose Algorithm 3", "pilot_test": "exact 30/60/90 s control trace; no new figure"},
        "S6": {"old_difference": "reuse_fraction only shortens fresh context", "falsifiable_hypothesis": "there is no shared-prefix object identity in this runner", "pilot_test": "NOT FIXED; old reuse hit result remains invalid for semantic reuse"},
        "S7": {"old_difference": "lower bandwidth appeared to improve Aegis relative WCT", "falsifiable_hypothesis": "queued copy wait was repeatedly charged as health burden and recovery used the same scalar", "pilot_test": "own-transfer health burden plus split copy/load plumbing; no bandwidth sweep rerun"},
        "S8": {"old_difference": "failure types were dominated by common replacement", "falsifiable_hypothesis": "same-node worker and node failures differ because node loss invalidates peer HBM", "pilot_test": "paired alternating probes, type-specific routes and changed WCT"},
        "S9": {"old_difference": "A100 scalar differs 11.5x-32x from RTX4090 staged physical", "falsifiable_hypothesis": "copy and recovery directions are not represented by one bandwidth", "pilot_test": "16-row direction-split calibration diagnostic; hardware remains unmatched"},
    }
    payload = {
        "schema_version": 1,
        "status": "MECHANISM_AUDIT",
        "no_target_trend_or_seed_selection": True,
        "figures": rows,
    }
    path = output_dir / "s1_s9_difference_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def audit_run(
    protocol: dict[str, Any], spec: dict[str, Any], cells: list[dict[str, Any]],
    sensitivity_cells: list[dict[str, Any]], *, stage: Path, target: Path,
) -> dict[str, Any]:
    import numpy as np

    matrix = spec["locked_matrix"]
    policies = set(matrix["policies"])
    seeds = set(matrix["paired_seeds"])
    checks: dict[str, bool] = {}
    checks["cell_count_exact"] = len(cells) == len(policies) * len(seeds)
    checks["policy_and_paired_seed_matrix_exact"] = all(
        {cell["seed"] for cell in cells if cell["policy"] == policy} == seeds
        for policy in policies
    ) and {cell["policy"] for cell in cells} == policies
    checks["fixed_arrival_90s_exact"] = all(
        cell["workload"]["fixed_arrival_cutoff"]
        and cell["workload"]["arrival_horizon_ms"] == matrix["arrival_cutoff_ms"]
        for cell in cells
    )
    checks["empty_fault_replay_preserves_completion"] = all(
        cell["mechanism"]["empty_fault_replay_max_completion_delta_ms"] <= 1e-7
        for cell in cells
    )
    checks["fault_wct_never_below_healthy"] = True
    checks["paired_workload_and_probe_identity"] = True
    signatures: dict[int, tuple[Any, ...]] = {}
    def staged_raw(cell: dict[str, Any]) -> Path:
        return stage / Path(cell["raw_npz"]).relative_to(target)

    for cell in cells:
        raw_path = staged_raw(cell)
        if core.sha256(raw_path) != cell["raw_npz_sha256"]:
            checks["paired_workload_and_probe_identity"] = False
        with np.load(raw_path, allow_pickle=False) as raw:
            if np.any(raw["fault_wct_ms"] < raw["base_wct_ms"][None, :] - 1e-7):
                checks["fault_wct_never_below_healthy"] = False
            signature = (
                tuple(raw["workflow_id"].tolist()),
                raw["probe_time_ms"].tobytes(),
                raw["probe_source_rank"].tobytes(),
                tuple(raw["probe_failure_type"].tolist()),
            )
            if cell["seed"] in signatures and signatures[cell["seed"]] != signature:
                checks["paired_workload_and_probe_identity"] = False
            signatures[cell["seed"]] = signature
    checks["split_copy_and_load_tables_present"] = all(
        key in protocol["hardware"]
        for key in ("copy_bandwidth_gb_s", "load_bandwidth_gb_s")
    )
    checks["fixed_point_converged"] = all(
        cell["mechanism"]["normal_path_schedule_solver"]["converged"] is True
        and cell["mechanism"]["normal_path_schedule_solver"]["final_burden_residual_finish_ms"] <= 1e-7
        for cell in cells
    )
    full = [cell for cell in cells if cell["policy"] == "AegisKV"]
    no1 = [cell for cell in cells if cell["policy"] == "AegisKV no Algorithm 1"]
    no2 = [cell for cell in cells if cell["policy"] == "AegisKV no Algorithm 2"]
    no3 = [cell for cell in cells if cell["policy"] == "AegisKV no Algorithm 3"]
    native = [cell for cell in cells if cell["policy"] == "Native Mooncake"]
    checks["three_control_windows_full"] = all(
        cell["mechanism"]["control_opportunity_times_ms"] == [30000.0, 60000.0, 90000.0]
        for cell in full
    )
    checks["alg1_switch_exercised"] = all(
        cell["mechanism"]["algorithm1_nonzero_slack_request_count"] > 0 for cell in full
    ) and all(
        cell["mechanism"]["algorithm1_nonzero_slack_request_count"] == 0 for cell in no1
    )
    checks["alg2_switch_exercised"] = all(
        cell["mechanism"]["residency_event_counts"] for cell in full
    ) and all(
        not cell["mechanism"]["residency_event_counts"] for cell in no2
    )
    checks["alg3_switch_exercised"] = all(
        cell["mechanism"]["control_window_count"] == 3 for cell in full
    ) and all(
        cell["mechanism"]["control_window_count"] == 0 for cell in no3
    )
    checks["capacity_pressure_exercised"] = all(
        sum(
            cell["mechanism"]["tier_churn"].get(tier, {}).get("admission_rejection_count", 0)
            for tier in ("HBM", "DRAM")
        ) > 0
        for cell in full
    )
    checks["copy_queue_pressure_exercised"] = all(
        cell["mechanism"]["copy_queue_wait_positive_count"] > 0 for cell in full
    )
    checks["native_replacement_contract_exercised"] = all(
        any(key.startswith("source_worker_gpu:replacement_wait") for key in cell["mechanism"]["failover_mode_counts"])
        and any(key.startswith("source_node:replacement_wait") for key in cell["mechanism"]["failover_mode_counts"])
        for cell in native
    )
    checks["aegis_worker_prebound_node_replacement_exercised"] = all(
        any(key.startswith("source_worker_gpu:prebound_fence_activation") for key in cell["mechanism"]["failover_mode_counts"])
        and any(key.startswith("source_node:replacement_wait") for key in cell["mechanism"]["failover_mode_counts"])
        for cell in (*full, *no1, *no2, *no3)
    )
    checks["same_node_source_node_never_restores_peer_hbm"] = all(
        not any(
            key.startswith("source_node:peer_hbm")
            for key in cell["mechanism"]["recovery_type_path_counts"]
        )
        for cell in (*full, *no1, *no2, *no3)
    )
    checks["500ms_is_modeled_unmeasured"] = (
        protocol["failure_model"]["policy_failover"]["fence_activation_delay_ms"] == 500.0
        and "unmeasured" in protocol["failure_model"]["policy_failover"]["fence_activation_provenance"]
        and "not" in spec["evidence_boundary"].lower()
    )
    sensitivity = spec["postregistration_sensitivity_diagnostic"]
    expected_delays = set(float(value) for value in sensitivity["replacement_delay_ms"])
    checks["replacement_delays_are_modeled_unmeasured"] = (
        "unmeasured" in protocol["failure_model"]["policy_failover"]["replacement_delay_provenance"]
        and "unmeasured" in sensitivity["evidence_boundary"]
    )
    checks["native_replacement_sensitivity_matrix_exact"] = (
        {float(cell["replacement_delay_ms"]) for cell in sensitivity_cells}
        == expected_delays
        and all(
            {cell["seed"] for cell in sensitivity_cells if float(cell["replacement_delay_ms"]) == delay}
            == seeds
            for delay in expected_delays
        )
    )
    sensitivity_signatures: dict[int, tuple[Any, ...]] = {}
    checks["native_sensitivity_paired_workload_and_probes"] = True
    for cell in sensitivity_cells:
        with np.load(staged_raw(cell), allow_pickle=False) as raw:
            signature = (
                tuple(raw["workflow_id"].tolist()),
                raw["probe_time_ms"].tobytes(),
                raw["probe_source_rank"].tobytes(),
                tuple(raw["probe_failure_type"].tolist()),
            )
            if cell["seed"] in sensitivity_signatures and sensitivity_signatures[cell["seed"]] != signature:
                checks["native_sensitivity_paired_workload_and_probes"] = False
            sensitivity_signatures[cell["seed"]] = signature
            if np.any(raw["fault_wct_ms"] < raw["base_wct_ms"][None, :] - 1e-7):
                checks["native_sensitivity_paired_workload_and_probes"] = False
    checks["no_performance_victory_gate"] = spec["registered_checks"]["no_performance_win_gate"] is True
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "physical_result": False,
        "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "cell_count": len(cells),
        "evidence_boundary": spec["evidence_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "mechanism_pilot_v2")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()
    protocol, spec = load_composite_protocol()
    target = args.output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()
    composite_path = stage / "composite_protocol.json"
    composite_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test = subprocess.run(
        [sys.executable, str(TEST_PATH)],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    test_log_path = stage / "regression_test.log"
    test_log_path.write_text(test.stdout, encoding="utf-8")
    if test.returncode != 0 or "RESULT 6/6 PASS" not in test.stdout:
        raise RuntimeError(f"mechanism regression failed; see {test_log_path}")
    matrix = spec["locked_matrix"]
    jobs = []
    for policy in matrix["policies"]:
        for seed in matrix["paired_seeds"]:
            label = policy.lower().replace(" ", "_")
            raw = stage / "raw" / f"{label}__s{seed}.npz"
            jobs.append((protocol, policy, int(seed), int(matrix["fault_probes_per_seed"]), float(matrix["arrival_cutoff_ms"]), raw))
    if args.max_cells is not None:
        jobs = jobs[: int(args.max_cells)]
    cells = []
    with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
        futures = {pool.submit(run_one, job): (job[1], job[2]) for job in jobs}
        for future in as_completed(futures):
            policy, seed = futures[future]
            cells.append(future.result())
            print(f"DONE policy={policy!r} seed={seed}", flush=True)
    cells.sort(key=lambda row: (row["policy"], row["seed"]))
    sensitivity_cells = []
    primary_native = {
        int(cell["seed"]): cell for cell in cells if cell["policy"] == "Native Mooncake"
    }
    for seed, cell in primary_native.items():
        row = dict(cell)
        row["replacement_delay_ms"] = 60000.0
        row["sensitivity_reuses_primary_cell"] = True
        sensitivity_cells.append(row)
    sensitivity_jobs = []
    for delay in (10000.0, 30000.0):
        sensitivity_protocol = deepcopy(protocol)
        sensitivity_protocol["failure_model"]["policy_failover"]["replacement_delay_ms"] = delay
        for seed in matrix["paired_seeds"]:
            raw = stage / "raw" / f"native_mooncake__replacement_{int(delay)}ms__s{seed}.npz"
            sensitivity_jobs.append((sensitivity_protocol, "Native Mooncake", int(seed), int(matrix["fault_probes_per_seed"]), float(matrix["arrival_cutoff_ms"]), raw, delay))
    with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
        futures = {
            pool.submit(run_one, job[:6]): (job[6], job[2])
            for job in sensitivity_jobs
        }
        for future in as_completed(futures):
            delay, seed = futures[future]
            row = future.result()
            row["replacement_delay_ms"] = delay
            row["sensitivity_reuses_primary_cell"] = False
            sensitivity_cells.append(row)
            print(f"DONE Native sensitivity replacement_ms={delay} seed={seed}", flush=True)
    sensitivity_cells.sort(key=lambda row: (row["replacement_delay_ms"], row["seed"]))
    for cell in (*cells, *sensitivity_cells):
        raw_path = Path(cell["raw_npz"])
        cell["raw_npz"] = str(target / raw_path.relative_to(stage))
    summary = {
        "schema_version": 1,
        "status": "PRELIMINARY_COMPLETE",
        "experiment_id": spec["experiment_id"],
        "paper_ready": False,
        "physical_result": False,
        "preregistered_spec": {"path": str(SPEC_PATH), "sha256": core.sha256(SPEC_PATH)},
        "parent_protocol": {"path": str(BASE_PROTOCOL_PATH), "sha256": core.sha256(BASE_PROTOCOL_PATH)},
        "composite_protocol_sha256": core.sha256(composite_path),
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "core_runner": {"path": str(Path(core.__file__).resolve()), "sha256": core.sha256(Path(core.__file__).resolve())},
        "cells": cells,
        "native_replacement_sensitivity_cells": sensitivity_cells,
    }
    summary_path = stage / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_aggregate(cells, stage / "aggregate_ci.csv")
    write_trigger_table(cells, stage / "mechanism_triggers.csv")
    write_replacement_sensitivity(
        sensitivity_cells, stage / "native_replacement_sensitivity.csv"
    )
    build_difference_audit(stage)
    write_calibration_diagnostic(protocol, stage)
    audit = audit_run(
        protocol, spec, cells, sensitivity_cells, stage=stage, target=target
    )
    audit["checks"]["regression_harness_6_of_6"] = True
    audit["regression_test"] = {
        "command": [sys.executable, str(TEST_PATH)],
        "returncode": test.returncode,
        "log": str(target / "regression_test.log"),
        "log_sha256": core.sha256(test_log_path),
        "runner_sha256": core.sha256(TEST_PATH),
    }
    audit["status"] = "PASS" if all(audit["checks"].values()) else "FAIL"
    audit["failed_checks"] = [
        key for key, passed in audit["checks"].items() if not passed
    ]
    audit_path = stage / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lineage = {
        "experiment_id": spec["experiment_id"],
        "preregistered_before_results": True,
        "no_seed_selection": True,
        "paths": {
            "spec": str(SPEC_PATH),
            "base_protocol": str(BASE_PROTOCOL_PATH),
            "composite_protocol": str(target / "composite_protocol.json"),
            "runner": str(Path(__file__).resolve()),
            "core_runner": str(Path(core.__file__).resolve()),
            "physical_profile_manifest": str(PHYSICAL_MANIFEST),
            "physical_profile_csv": str(PHYSICAL_CSV),
            "regression_test": str(TEST_PATH),
        },
        "sha256": {
            "spec": core.sha256(SPEC_PATH),
            "base_protocol": core.sha256(BASE_PROTOCOL_PATH),
            "composite_protocol": core.sha256(composite_path),
            "runner": core.sha256(Path(__file__).resolve()),
            "core_runner": core.sha256(Path(core.__file__).resolve()),
            "physical_profile_manifest": core.sha256(PHYSICAL_MANIFEST),
            "physical_profile_csv": core.sha256(PHYSICAL_CSV),
            "regression_test": core.sha256(TEST_PATH),
        },
    }
    lineage_path = stage / "lineage.json"
    lineage_path.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}:
            artifacts[str(path.relative_to(stage))] = core.sha256(path)
    manifest = {
        "schema_version": 1,
        "status": audit["status"],
        "experiment_id": spec["experiment_id"],
        "paper_ready": False,
        "target": str(target),
        "artifacts": artifacts,
    }
    manifest_path = stage / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = stage / "SHA256SUMS"
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        + f"{core.sha256(manifest_path)}  manifest.json\n",
        encoding="utf-8",
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"pilot audit failed in staging directory {stage}: {audit['failed_checks']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
