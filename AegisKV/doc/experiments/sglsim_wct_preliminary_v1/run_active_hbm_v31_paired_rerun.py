#!/usr/bin/env python3
"""Run and audit the preregistered v3.1 paired correction rerun."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
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

import active_hbm_v31 as v31  # noqa: E402
import run_active_hbm_v3_paired_pilot as helpers  # noqa: E402
import run_mechanism_pilot_v2 as pilot  # noqa: E402
import run_preliminary as core  # noqa: E402


SPEC_PATH = ROOT / "active_hbm_v31_paired_rerun_protocol.json"
CAPACITY_PATH = ROOT / "active_hbm_v3_protocol.json"
OLD_RESULT = ROOT / "results" / "active_hbm_v3_paired_pilot"
TESTS = (
    (ROOT / "test_active_hbm_v3_capacity.py", "RESULT 5/5 PASS"),
    (ROOT / "test_active_hbm_v3_des.py", "RESULT 5/5 PASS"),
    (ROOT / "test_active_hbm_v31.py", "RESULT 7/7 PASS"),
)
RESULT_NAME = "active_hbm_v31_paired_rerun"


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    capacity = json.loads(CAPACITY_PATH.read_text(encoding="utf-8"))
    for path_key, sha_key in (
        ("current_design_path", "current_design_sha256"),
        ("parent_capacity_protocol", "parent_capacity_protocol_sha256"),
        ("v31_des", "v31_des_sha256"),
        ("core_runner", "core_runner_sha256"),
    ):
        path = Path(spec[path_key]) if Path(spec[path_key]).is_absolute() else ROOT / spec[path_key]
        if core.sha256(path) != spec[sha_key]:
            raise RuntimeError(f"preregistered source changed: {path}")
    base, _parent = pilot.load_composite_protocol()
    return base, capacity, spec


def run_one(args: tuple[dict, dict, str, int, str, Path]) -> dict[str, Any]:
    base, capacity, policy, seed, failure_type, raw = args
    protocol = deepcopy(base)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["workload"]["center_context_tokens"] = 40960
    protocol["workload"]["center_active_requests_per_gpu"] = 16
    protocol["failure_model"]["probe_type_schedule"] = [failure_type]
    return v31.run_v31_cell(
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
    row = helpers.metric_row(cell)
    failure = cell["failure_conditioned_changed_by_type"][cell["failure_type"]]
    correction = cell["mechanism"]["active_hbm_v31"]
    row["same_probe_failure_conditioned_wct_mean_ms"] = failure[
        "same_probe_failure_conditioned_wct_mean_ms"
    ]
    row["all_workflow_total_wct_mean_ms"] = (
        cell["base_wct_mean_ms"] + failure["all_workflow_added_wct_mean_ms"]
    )
    row["fault_domain_tuple_sha256"] = cell["fault_domain_tuple_sha256"]
    row["exact_projection_evaluations"] = correction["exact_projection_evaluation_count"]
    row["old_projection_false_block_unique_requests"] = correction[
        "old_projection_false_block_unique_request_count"
    ]
    row["old_projection_excess_gib_max"] = correction["old_projection_excess_bytes_max"] / 1024**3
    row["projection_prediction_mismatches"] = correction[
        "projection_prediction_mismatch_count"
    ]
    return row


def per_probe_rows(cells: list[dict[str, Any]], stage: Path, target: Path) -> list[dict[str, Any]]:
    result = []
    for cell in cells:
        raw_path = stage / Path(cell["raw_npz"]).relative_to(target)
        with np.load(raw_path, allow_pickle=False) as raw:
            base = raw["base_wct_ms"].astype(np.float64)
            fault = raw["fault_wct_ms"].astype(np.float64)
            for index in range(fault.shape[0]):
                added = np.maximum(fault[index] - base, 0.0)
                result.append({
                    "policy": cell["policy"],
                    "seed": cell["seed"],
                    "failure_type": cell["failure_type"],
                    "probe_id": str(raw["probe_id"][index]),
                    "probe_time_ms": float(raw["probe_time_ms"][index]),
                    "probe_source_rank": int(raw["probe_source_rank"][index]),
                    "probe_failure_domain_rank_ids": ";".join(
                        str(int(rank))
                        for rank in raw["probe_failure_domain_rank_ids"][index]
                        if int(rank) >= 0
                    ),
                    "workflow_count": len(base),
                    "healthy_wct_mean_ms": float(np.mean(base)),
                    "same_probe_failure_conditioned_wct_mean_ms": float(np.mean(fault[index])),
                    "all_workflow_added_wct_mean_ms": float(np.mean(added)),
                    "all_workflow_total_wct_mean_ms": float(np.mean(base) + np.mean(added)),
                    "diagnostic_changed_only_workflow_count": int(np.count_nonzero(added > 1e-7)),
                })
    return result


def comparison_rows(new_aggregate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old_rows = list(csv.DictReader((OLD_RESULT / "aggregate_ci95.csv").open()))
    old = {
        (row["policy"], row["failure_type"], row["metric"]): row
        for row in old_rows
    }
    result = []
    for row in new_aggregate:
        key = (row["policy"], row["failure_type"], row["metric"])
        if key not in old:
            continue
        before = float(old[key]["mean"])
        after = float(row["mean"])
        result.append({
            "policy": row["policy"],
            "failure_type": row["failure_type"],
            "metric": row["metric"],
            "v3_mean": before,
            "v31_mean": after,
            "v31_minus_v3": after - before,
            "v31_over_v3_minus_one": after / before - 1.0 if before else None,
        })
    return result


def main() -> int:
    base, capacity, spec = load_inputs()
    target = ROOT / "results" / RESULT_NAME
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()

    old_checksum = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=str(OLD_RESULT),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (stage / "old_v3_checksum.log").write_text(old_checksum.stdout, encoding="utf-8")
    if old_checksum.returncode:
        raise RuntimeError("old v3 lineage checksum failed")
    test_records = []
    for path, expected in TESTS:
        completed = subprocess.run(
            [sys.executable, str(path)], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        log = stage / f"{path.stem}.log"
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode or expected not in completed.stdout:
            raise RuntimeError(f"regression failed: {log}")
        test_records.append({
            "test": str(path), "test_sha256": core.sha256(path),
            "expected": expected, "returncode": completed.returncode,
            "log": str(target / log.name), "log_sha256": core.sha256(log),
        })

    matrix = spec["matrix"]
    jobs = []
    for seed in matrix["paired_seeds"]:
        for failure_type in matrix["failure_types"]:
            for policy in matrix["policies"]:
                raw = stage / "raw" / (
                    f"{helpers.slug(policy)}__{failure_type}__s{seed}.npz"
                )
                jobs.append((base, capacity, policy, int(seed), failure_type, raw))
    cells = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(run_one, job): (job[2], job[3], job[4]) for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            policy, seed, failure_type = futures[future]
            cells.append(future.result())
            print(f"DONE {index}/{len(jobs)} {policy} seed={seed} {failure_type}", flush=True)
    cells.sort(key=lambda row: (row["seed"], row["failure_type"], row["policy"]))
    for cell in cells:
        raw = Path(cell["raw_npz"])
        cell["raw_npz"] = str(target / raw.relative_to(stage))

    metrics = [metric_row(cell) for cell in cells]
    aggregate = helpers.aggregate_rows(metrics)
    contrasts = helpers.paired_contrast_rows(metrics)
    probe_metrics = per_probe_rows(cells, stage, target)
    comparison = comparison_rows(aggregate)
    helpers.write_csv(metrics, stage / "paired_metrics.csv")
    helpers.write_csv(aggregate, stage / "aggregate_ci95.csv")
    helpers.write_csv(contrasts, stage / "paired_primary_contrasts_ci95.csv")
    helpers.write_csv(probe_metrics, stage / "same_probe_metrics.csv")
    helpers.write_csv(comparison, stage / "v3_vs_v31.csv")

    checks: dict[str, bool] = {
        "old_v3_checksum_pass": old_checksum.returncode == 0,
        "cell_count_exact_30": len(cells) == int(matrix["expected_cells"]),
        "matrix_exact": {
            (cell["policy"], cell["seed"], cell["failure_type"]) for cell in cells
        } == {
            (policy, int(seed), failure_type)
            for policy in matrix["policies"]
            for seed in matrix["paired_seeds"]
            for failure_type in matrix["failure_types"]
        },
        "same_workflow_universe_across_policies": True,
        "same_probe_tuples_across_policies": True,
        "same_failure_domain_sets_across_policies": True,
        "raw_hashes_and_identity_hashes_match": True,
        "all_workflow_denominator_exact": True,
        "same_probe_statistics_match_raw": True,
        "healthy_plus_added_equals_total": True,
        "fault_wct_monotone": True,
        "capacity_never_exceeded": True,
        "projection_exactly_matches_admission": all(
            cell["mechanism"]["active_hbm_v31"]["projection_prediction_mismatch_count"] == 0
            for cell in cells
        ),
        "old_false_projection_block_exercised": any(
            cell["mechanism"]["active_hbm_v31"]["old_projection_false_block_unique_request_count"] > 0
            for cell in cells
        ),
        "slack_predictor_no_realized_output_or_service": all(
            cell["workload"]["slack_predictor"]["reads_realized_output_tokens"] is False
            and cell["workload"]["slack_predictor"]["reads_realized_service_ms"] is False
            for cell in cells
        ),
        "dispatch_no_realized_output_oracle": all(
            cell["mechanism"]["active_hbm_v3"]["dispatch_uses_no_realized_output_oracle"]
            for cell in cells
        ),
        "empty_replay_exact": all(
            cell["mechanism"]["empty_fault_replay_max_completion_delta_ms"] <= 1e-7
            for cell in cells
        ),
        "three_control_opportunities": all(
            cell["mechanism"]["control_window_count"] == 3
            for cell in cells if cell["policy"] == "AegisKV"
        ),
        "changed_only_forbidden_as_primary": "forbidden" in spec["secondary_diagnostic"],
        "no_performance_selection": spec["hard_gates"]["no_performance_direction_or_win_threshold"] is True,
        "regressions_all_pass": len(test_records) == len(TESTS),
    }
    workflow_signatures: dict[int, str] = {}
    probe_signatures: dict[tuple[int, str], str] = {}
    domain_signatures: dict[tuple[int, str], str] = {}
    for cell in cells:
        raw_path = stage / Path(cell["raw_npz"]).relative_to(target)
        if core.sha256(raw_path) != cell["raw_npz_sha256"]:
            checks["raw_hashes_and_identity_hashes_match"] = False
        with np.load(raw_path, allow_pickle=False) as raw:
            if helpers.workflow_digest(raw) != cell["workflow_universe_sha256"]:
                checks["raw_hashes_and_identity_hashes_match"] = False
            if helpers.probe_digest(raw) != cell["fault_probe_tuple_sha256"]:
                checks["raw_hashes_and_identity_hashes_match"] = False
            domain_payload = [
                {
                    "probe_id": str(probe_id),
                    "rank_ids": [int(rank) for rank in ranks if int(rank) >= 0],
                }
                for probe_id, ranks in zip(
                    raw["probe_id"], raw["probe_failure_domain_rank_ids"]
                )
            ]
            domain_digest = hashlib.sha256(json.dumps(
                domain_payload, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            if domain_digest != cell["fault_domain_tuple_sha256"]:
                checks["raw_hashes_and_identity_hashes_match"] = False
            workflow_key = int(cell["seed"])
            signature = cell["workflow_universe_sha256"]
            if workflow_key in workflow_signatures and workflow_signatures[workflow_key] != signature:
                checks["same_workflow_universe_across_policies"] = False
            workflow_signatures[workflow_key] = signature
            probe_key = (int(cell["seed"]), str(cell["failure_type"]))
            probe_signature = cell["fault_probe_tuple_sha256"]
            if probe_key in probe_signatures and probe_signatures[probe_key] != probe_signature:
                checks["same_probe_tuples_across_policies"] = False
            probe_signatures[probe_key] = probe_signature
            domain_signature = cell["fault_domain_tuple_sha256"]
            if probe_key in domain_signatures and domain_signatures[probe_key] != domain_signature:
                checks["same_failure_domain_sets_across_policies"] = False
            domain_signatures[probe_key] = domain_signature
            expected_domain_size = (
                int(capacity["hardware"].get("gpus_per_node", 8))
                if cell["failure_type"] == "source_node" else 1
            )
            actual_domain_sizes = np.sum(
                raw["probe_failure_domain_rank_ids"] >= 0, axis=1
            )
            if not np.all(actual_domain_sizes == expected_domain_size):
                checks["same_failure_domain_sets_across_policies"] = False
            healthy = raw["base_wct_ms"].astype(np.float64)
            fault = raw["fault_wct_ms"].astype(np.float64)
            added = np.maximum(fault - healthy[None, :], 0.0)
            failure = cell["failure_conditioned_changed_by_type"][cell["failure_type"]]
            denominator = len(healthy) * fault.shape[0]
            if int(failure["all_workflow_observations"]) != denominator:
                checks["all_workflow_denominator_exact"] = False
            if not math.isclose(
                float(failure["same_probe_failure_conditioned_wct_mean_ms"]),
                float(np.mean(fault)), abs_tol=1e-7,
            ):
                checks["same_probe_statistics_match_raw"] = False
            if not math.isclose(
                float(np.mean(healthy) + np.mean(added)), float(np.mean(fault)),
                abs_tol=1e-7,
            ):
                checks["healthy_plus_added_equals_total"] = False
            if np.any(fault < healthy[None, :] - 1e-7):
                checks["fault_wct_monotone"] = False
            if np.any(raw["capacity_shared_used_bytes"] > capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]):
                checks["capacity_never_exceeded"] = False
            if cell["mechanism"]["active_hbm_v3"]["capacity_violation_count"]:
                checks["capacity_never_exceeded"] = False

    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "cell_count": len(cells),
        "same_probe_row_count": len(probe_metrics),
        "primary_denominator": "common workflow universe times common probes",
        "changed_only_cross_policy_primary_allowed": False,
        "paper_ready": False,
        "physical_result": False,
        "tests": test_records,
        "old_v3_lineage": {
            "path": str(OLD_RESULT),
            "audit_sha256": core.sha256(OLD_RESULT / "audit.json"),
            "manifest_sha256": core.sha256(OLD_RESULT / "manifest.json"),
        },
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "status": "PRELIMINARY_COMPLETE" if audit["status"] == "PASS" else "FAILED_AUDIT",
        "paper_ready": False,
        "preregistered_protocol": {"path": str(SPEC_PATH), "sha256": core.sha256(SPEC_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "v31_des": {"path": str(Path(v31.__file__).resolve()), "sha256": core.sha256(Path(v31.__file__).resolve())},
        "cells": cells,
    }
    (stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    artifacts = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}:
            artifacts[str(path.relative_to(stage))] = core.sha256(path)
    manifest = {
        "schema_version": 1, "status": audit["status"], "paper_ready": False,
        "target": str(target), "artifacts": artifacts,
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        + f"{core.sha256(stage / 'manifest.json')}  manifest.json\n"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"v3.1 audit failed in {stage}: {audit['failed_checks']}")
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
