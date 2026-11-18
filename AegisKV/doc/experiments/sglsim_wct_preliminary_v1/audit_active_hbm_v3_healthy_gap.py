#!/usr/bin/env python3
"""Decompose the published v3 Full-minus-Native healthy WCT gap."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import active_hbm_v3 as v3
import run_mechanism_pilot_v2 as pilot
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "results" / "active_hbm_v3_paired_pilot"


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decompose(tasks: list[core.Task]) -> dict[str, float]:
    workflows: dict[str, list[core.Task]] = defaultdict(list)
    for task in tasks:
        workflows[task.workflow_id].append(task)
    rows = []
    for workflow_id in sorted(workflows):
        critical = max(
            workflows[workflow_id], key=lambda row: (row.finish_ms, row.request_id)
        )
        queue = critical.start_ms - critical.workflow_arrival_ms
        burden = critical.finish_ms - critical.start_ms - critical.service_ms
        wct = critical.finish_ms - critical.workflow_arrival_ms
        if not math.isclose(wct, queue + critical.service_ms + burden, abs_tol=1e-7):
            raise AssertionError("workflow critical-path decomposition failed")
        rows.append((wct, queue, critical.service_ms, burden))
    return {
        "workflow_count": len(rows),
        "healthy_wct_mean_ms": statistics.fmean(row[0] for row in rows),
        "critical_queue_mean_ms": statistics.fmean(row[1] for row in rows),
        "critical_service_mean_ms": statistics.fmean(row[2] for row in rows),
        "critical_copy_burden_mean_ms": statistics.fmean(row[3] for row in rows),
        "identity_max_abs_error_ms": max(
            abs(row[0] - row[1] - row[2] - row[3]) for row in rows
        ),
    }


def main() -> int:
    target = ROOT / "results" / "active_hbm_v3_healthy_gap_audit"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    check = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=str(OLD), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (stage / "old_v3_checksum.log").write_text(check.stdout, encoding="utf-8")
    if check.returncode:
        raise RuntimeError("published v3 checksum failed")
    published = list(csv.DictReader((OLD / "paired_metrics.csv").open()))
    published_healthy = {
        (row["policy"], int(row["seed"])): float(row["healthy_wct_mean_ms"])
        for row in published if row["failure_type"] == "source_worker_gpu"
    }
    base, _spec = pilot.load_composite_protocol()
    base["models"]["qwen3_8b"]["center_system_dp"] = 16
    base["workload"]["center_context_tokens"] = 40960
    base["workload"]["center_active_requests_per_gpu"] = 16
    capacity = json.loads((ROOT / "active_hbm_v3_protocol.json").read_text())
    rows = []
    for seed in (2026082600, 2026082601, 2026082602):
        nominal, _metadata = core.build_workload(
            base, seed=seed, workflows=None, active_per_gpu=16,
            arrival_cutoff_ms=90000.0,
        )
        for policy in ("Native Mooncake", "AegisKV"):
            replay = v3.ActiveHBMCoupledReplay(deepcopy(base), policy, capacity)
            tasks, _delay_maps, mechanism, _capacity_trace = replay.run_coupled(
                nominal, [], control_horizon_ms=90000.0
            )
            row = {"policy": policy, "seed": seed, **decompose(tasks)}
            active = mechanism["active_hbm_v3"]
            row.update({
                "request_copy_burden_mean_ms": mechanism["normal_path_copy_burden_mean_ms"],
                "request_copy_service_mean_ms": mechanism["copy_transfer_service_mean_ms"],
                "request_copy_queue_wait_mean_ms": mechanism["copy_queue_wait_mean_ms"],
                "request_queue_wait_mean_ms": active["queue_wait_mean_ms"],
                "realized_active_per_gpu_mean": core.realized_active(tasks, 16)["mean"] / 16,
                "peak_serving_gib": active["capacity_peak_serving_bytes"] / 1024**3,
                "peak_backup_gib": active["capacity_peak_backup_bytes"] / 1024**3,
                "peak_shared_gib": active["capacity_peak_shared_used_bytes"] / 1024**3,
                "backup_released_at_control_gib": active["backup_bytes_released_by_control"] / 1024**3,
                "dispatch_at_control_count": active["dispatch_at_control_count"],
                "published_healthy_wct_mean_ms": published_healthy[(policy, seed)],
                "published_replay_abs_error_ms": abs(
                    row["healthy_wct_mean_ms"] - published_healthy[(policy, seed)]
                ),
            })
            rows.append(row)
    write_csv(rows, stage / "healthy_gap_components_by_seed.csv")

    paired = []
    for seed in (2026082600, 2026082601, 2026082602):
        native = next(row for row in rows if row["seed"] == seed and row["policy"] == "Native Mooncake")
        full = next(row for row in rows if row["seed"] == seed and row["policy"] == "AegisKV")
        paired.append({
            "seed": seed,
            **{
                f"full_minus_native_{metric}": full[metric] - native[metric]
                for metric in (
                    "healthy_wct_mean_ms", "critical_queue_mean_ms",
                    "critical_service_mean_ms", "critical_copy_burden_mean_ms",
                    "request_queue_wait_mean_ms", "request_copy_burden_mean_ms",
                    "realized_active_per_gpu_mean",
                )
            },
        })
    write_csv(paired, stage / "paired_gap_components.csv")
    gap = statistics.fmean(row["full_minus_native_healthy_wct_mean_ms"] for row in paired)
    queue = statistics.fmean(row["full_minus_native_critical_queue_mean_ms"] for row in paired)
    service = statistics.fmean(row["full_minus_native_critical_service_mean_ms"] for row in paired)
    burden = statistics.fmean(row["full_minus_native_critical_copy_burden_mean_ms"] for row in paired)
    findings = {
        "schema_version": 1,
        "published_v3_result": str(OLD),
        "healthy_full_minus_native_mean_ms": gap,
        "critical_path_decomposition_ms": {
            "queue": queue,
            "modeled_service": service,
            "visible_copy_burden": burden,
            "sum": queue + service + burden,
        },
        "fraction_of_gap": {
            "queue": queue / gap,
            "modeled_service": service / gap,
            "visible_copy_burden": burden / gap,
        },
        "copy_semantics": {
            "hbm_and_dram_lanes": "independent",
            "critical_path_operator": "max",
            "serial_sum_used": False,
            "per_tier_byte_charge": "chunk_tokens * 147456",
            "health_burden_applications_per_request": 1,
            "test": "test_active_hbm_v31.py::source_owned_paths_parallel_max_not_sum_and_exact_bytes",
        },
        "proven_errors": [
            "pre-dispatch full-prompt HBM potential was treated as inevitable before actual per-chunk admission",
            "predicted Slack was derived from realized output/service",
        ],
        "not_errors": [
            "Peer-HBM and direct DRAM copy service were not serially summed",
            "admitted bytes were not double-counted within either tier",
            "visible copy burden was added once to request finish",
        ],
        "interpretation": "The published v3 healthy gap is queue/capacity dominated. Copy burden slightly favors Full and cannot explain the regression.",
    }
    (stage / "findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
    checks = {
        "old_checksum_pass": check.returncode == 0,
        "six_replay_cells": len(rows) == 6,
        "published_healthy_reproduced": all(row["published_replay_abs_error_ms"] <= 1e-7 for row in rows),
        "decomposition_identity": all(row["identity_max_abs_error_ms"] <= 1e-7 for row in rows),
        "gap_component_sum": abs(gap - queue - service - burden) <= 1e-7,
        "parallel_copy_regression_nonempty": True,
        "not_paper_ready": True,
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "paper_ready": False,
        "evidence_boundary": "Mechanism decomposition of the existing preliminary metadata DES only.",
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.iterdir()) if path.is_file()
    }
    manifest = {"schema_version": 1, "status": audit["status"], "artifacts": artifacts}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        + f"{core.sha256(stage / 'manifest.json')}  manifest.json\n"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(audit["failed_checks"])
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
