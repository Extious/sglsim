#!/usr/bin/env python3
"""Atomic six-architecture cold-start and steady-state v3.1.3 pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import active_hbm_v313 as v313
import run_preliminary as core
import test_active_hbm_v313 as test_inputs


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_accounting_batched_s2_protocol.json"
RESULT_NAME = "active_hbm_v313_s1_cold_steady_pilot"
TESTS = (
    ("test_active_hbm_v313_protocol.py", "RESULT 6/6 PASS"),
    ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    ("test_poisson_fault_batch_v2.py", "RESULT 6/6 PASS"),
)


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cohort_wct(
    tasks: list[core.Task], cohort_ids: set[str],
    completion: dict[str, float] | None = None,
) -> tuple[list[str], list[float]]:
    selected = [task for task in tasks if task.workflow_id in cohort_ids]
    return core.workflow_wct(selected, completion)


def probes_for_cohort(
    nominal: list[core.Task], *, seed: int, left_ms: float, right_ms: float,
    per_domain: int,
) -> list[core.FaultProbe]:
    cohort = [
        task for task in nominal
        if left_ms <= task.workflow_arrival_ms <= right_ms
    ]
    probes = core.make_fault_probes(
        cohort,
        seed=seed,
        count=per_domain * 2,
        gpus_per_node=8,
        right_ms=right_ms,
        failure_types=("source_worker_gpu", "source_node"),
    )
    if sum(row.failure_type == "source_worker_gpu" for row in probes) != per_domain:
        raise AssertionError("worker probe count mismatch")
    if sum(row.failure_type == "source_node" for row in probes) != per_domain:
        raise AssertionError("node probe count mismatch")
    return probes


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text())
    protocol, capacity = test_inputs.inputs(context=40960, active=16)
    target = ROOT / "results" / RESULT_NAME
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{RESULT_NAME}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw").mkdir()
    (stage / "tests").mkdir()
    try:
        test_rows = []
        for filename, marker in TESTS:
            result = subprocess.run(
                ["python", str(ROOT / filename)], cwd=str(ROOT), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            (stage / "tests" / f"{Path(filename).stem}.log").write_text(
                result.stdout, encoding="utf-8"
            )
            passed = result.returncode == 0 and marker in result.stdout
            test_rows.append({"test": filename, "passed": passed})
            if not passed:
                raise RuntimeError(f"test gate failed: {filename}\n{result.stdout}")

        regimes = {
            "cold_start_inclusive": {
                "arrival_cutoff_ms": 5000.0,
                "cohort_left_ms": 0.0,
                "cohort_right_ms": 5000.0,
                "control_horizon_ms": 5000.0,
            },
            "steady_state_after_common_90s_warmup": {
                "arrival_cutoff_ms": 180000.0,
                "cohort_left_ms": 90000.0,
                "cohort_right_ms": 180000.0,
                "control_horizon_ms": 180000.0,
            },
        }
        policies = spec["healthy_accounting"]["display_to_internal_policy"]
        seed = int(spec["healthy_accounting"]["pilot_seeds"][0])
        per_domain = int(
            spec["healthy_accounting"]["pilot_fault_probes_per_domain_per_regime"]
        )
        cells = []
        metric_rows = []
        for regime_name, regime in regimes.items():
            nominal, workload = core.build_workload(
                protocol, seed=seed, workflows=None, active_per_gpu=16,
                arrival_cutoff_ms=regime["arrival_cutoff_ms"],
            )
            cohort_ids = {
                task.workflow_id for task in nominal
                if regime["cohort_left_ms"]
                <= task.workflow_arrival_ms <= regime["cohort_right_ms"]
            }
            probes = probes_for_cohort(
                nominal, seed=seed ^ (0xC01D if regime_name.startswith("cold") else 0x57EAD),
                left_ms=regime["cohort_left_ms"],
                right_ms=regime["cohort_right_ms"], per_domain=per_domain,
            )
            workflow_hash = digest_json([
                (task.request_id, task.workflow_id, task.workflow_arrival_ms,
                 task.arrival_ms, task.source_rank, task.service_ms,
                 task.prompt_tokens, task.output_tokens)
                for task in nominal
            ])
            cohort_hash = digest_json(sorted(cohort_ids))
            probe_hash = digest_json([
                (probe.probe_id, probe.time_ms, probe.source_rank, probe.failure_type)
                for probe in probes
            ])
            for display, internal in policies.items():
                replay = v313.CausalPhysicalTargetReplay(
                    deepcopy(protocol), internal, capacity
                )
                tasks, delay_maps, mechanism, capacity_trace = replay.run_coupled(
                    nominal, probes,
                    control_horizon_ms=regime["control_horizon_ms"],
                )
                base_ids, base_wct = cohort_wct(tasks, cohort_ids)
                decomposition = v313.workflow_decomposition(
                    [task for task in tasks if task.workflow_id in cohort_ids]
                )
                fault_by_domain: dict[str, list[list[float]]] = {
                    "source_worker_gpu": [], "source_node": []
                }
                fault_probe_rows = []
                for probe, delays in zip(probes, delay_maps):
                    completion = core.replay_one_fault(tasks, delays)
                    ids, values = cohort_wct(tasks, cohort_ids, completion)
                    if ids != base_ids:
                        raise AssertionError("cohort workflow denominator changed")
                    if any(value < base - 1e-7 for value, base in zip(values, base_wct)):
                        raise AssertionError("fault lowered cohort WCT")
                    fault_by_domain[probe.failure_type].append(values)
                    fault_probe_rows.append((
                        probe.probe_id, probe.failure_type, probe.time_ms,
                        probe.source_rank, len(delays),
                    ))
                raw_name = f"{regime_name}__{display.lower().replace(' ', '_')}__s{seed}.npz"
                raw_path = stage / "raw" / raw_name
                np.savez_compressed(
                    raw_path,
                    workflow_ids=np.asarray(base_ids),
                    base_wct_ms=np.asarray(base_wct, dtype=float),
                    worker_fault_wct_ms=np.asarray(
                        fault_by_domain["source_worker_gpu"], dtype=float
                    ),
                    node_fault_wct_ms=np.asarray(
                        fault_by_domain["source_node"], dtype=float
                    ),
                    probe_rows=np.asarray(fault_probe_rows, dtype=str),
                    capacity_time_ms=np.asarray(
                        [row["time_ms"] for row in capacity_trace], dtype=float
                    ),
                    capacity_rank=np.asarray(
                        [row["rank"] for row in capacity_trace], dtype=int
                    ),
                    capacity_serving_bytes=np.asarray(
                        [row["active_serving_kv_bytes"] for row in capacity_trace], dtype=np.int64
                    ),
                    capacity_target_backup_bytes=np.asarray(
                        [row["peer_hbm_backup_used_bytes"] for row in capacity_trace], dtype=np.int64
                    ),
                )
                failures = {}
                for domain, matrix in fault_by_domain.items():
                    flat = [value for row in matrix for value in row]
                    failures[domain] = {
                        "probe_count": len(matrix),
                        "same_probe_observations": len(flat),
                        "wct_mean_ms": statistics.fmean(flat),
                        "added_wct_mean_ms": statistics.fmean(flat) - statistics.fmean(base_wct),
                    }
                cell = {
                    "regime": regime_name,
                    "display_policy": display,
                    "internal_policy": internal,
                    "seed": seed,
                    "workflow_universe_sha256": workflow_hash,
                    "measurement_cohort_sha256": cohort_hash,
                    "fault_probe_tuple_sha256": probe_hash,
                    "workflow_count_total": workload["workflow_count"],
                    "measurement_workflow_count": len(base_ids),
                    "base_wct_mean_ms": statistics.fmean(base_wct),
                    "healthy_decomposition": decomposition,
                    "failure_conditioned": failures,
                    "mechanism": mechanism,
                    "raw_npz": f"raw/{raw_name}",
                    "raw_npz_sha256": core.sha256(raw_path),
                }
                cells.append(cell)
                a = mechanism["active_hbm_v3"]
                v = mechanism["active_hbm_v313"]
                metric_rows.append({
                    "regime": regime_name,
                    "display_policy": display,
                    "internal_policy": internal,
                    "seed": seed,
                    "measurement_workflow_count": len(base_ids),
                    "healthy_wct_mean_ms": statistics.fmean(base_wct),
                    "critical_queue_mean_ms": decomposition["critical_queue_mean_ms"],
                    "critical_service_mean_ms": decomposition["critical_service_mean_ms"],
                    "direct_copy_burden_mean_ms": decomposition["critical_direct_copy_burden_mean_ms"],
                    "worker_fault_wct_mean_ms": failures["source_worker_gpu"]["wct_mean_ms"],
                    "node_fault_wct_mean_ms": failures["source_node"]["wct_mean_ms"],
                    "queue_wait_mean_ms": a["queue_wait_mean_ms"],
                    "memory_blocked_unique_request_count": a["memory_blocked_unique_request_count"],
                    "control_window_count": mechanism["control_window_count"],
                    "scheduled_control_opportunity_times_ms": json.dumps(
                        v["scheduled_control_opportunity_times_ms"]
                    ),
                    "executed_policy_control_times_ms": json.dumps(
                        v["executed_policy_control_times_ms"]
                    ),
                    "primary_priority_reclaimed_gib": v["primary_priority_reclaimed_bytes"] / 2**30,
                    "peak_target_backup_gib": v["physical_target_peak_backup_max_bytes"] / 2**30,
                    "capacity_violation_count": a["capacity_violation_count"] + v["target_capacity_violation_count"],
                    "copy_byte_conservation_error": json.dumps(v["copy_byte_conservation_error"], sort_keys=True),
                    "workflow_universe_sha256": workflow_hash,
                    "measurement_cohort_sha256": cohort_hash,
                    "fault_probe_tuple_sha256": probe_hash,
                    "raw_npz_sha256": core.sha256(raw_path),
                })
                print(
                    f"S1 v313 {len(cells)}/12 {regime_name} {display}", flush=True
                )

        write_csv(stage / "metrics.csv", metric_rows)
        common = all(
            len({cell[key] for cell in cells if cell["regime"] == regime}) == 1
            for regime in regimes
            for key in (
                "workflow_universe_sha256", "measurement_cohort_sha256",
                "fault_probe_tuple_sha256",
            )
        )
        checks = {
            "mechanism_tests_all_pass": all(row["passed"] for row in test_rows),
            "exact_12_cells": len(cells) == 12,
            "six_display_names_each_regime": all(
                {cell["display_policy"] for cell in cells if cell["regime"] == regime}
                == set(policies) for regime in regimes
            ),
            "common_workflow_cohort_probe_universe": common,
            "no_capacity_or_copy_conservation_error": all(
                cell["mechanism"]["active_hbm_v3"]["capacity_violation_count"] == 0
                and cell["mechanism"]["active_hbm_v313"]["target_capacity_violation_count"] == 0
                and cell["mechanism"]["active_hbm_v313"]["copy_byte_conservation_error"]
                == {"HBM": 0, "DRAM": 0}
                for cell in cells
            ),
            "future_backup_never_dispatch_projected": all(
                cell["mechanism"]["active_hbm_v313"]["future_backup_projection_dispatch_count"] == 0
                for cell in cells
            ),
            "cold_has_no_control_opportunity": all(
                cell["mechanism"]["active_hbm_v313"]["scheduled_control_opportunity_times_ms"] == []
                for cell in cells if cell["regime"] == "cold_start_inclusive"
            ),
            "steady_has_exact_six_control_opportunities": all(
                cell["mechanism"]["active_hbm_v313"]["scheduled_control_opportunity_times_ms"]
                == [30000.0, 60000.0, 90000.0, 120000.0, 150000.0, 180000.0]
                for cell in cells if cell["regime"].startswith("steady")
            ),
            "no_seed_or_cell_selection": True,
        }
        summary = {
            "schema_version": 1,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "paper_ready": False,
            "protocol": str(SPEC_PATH),
            "protocol_sha256": core.sha256(SPEC_PATH),
            "runner_sha256": core.sha256(Path(__file__)),
            "cells": cells,
            "evidence_boundary": (
                "One-seed mechanism pilot. Copy/service/bandwidth and failover delays are modeled; "
                "cold and steady cohorts are complementary and neither may be discarded by shape."
            ),
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        audit = {
            "schema_version": 1,
            "status": summary["status"],
            "checks": checks,
            "failed_checks": [key for key, passed in checks.items() if not passed],
            "paper_ready": False,
            "protocol_sha256": summary["protocol_sha256"],
            "summary_sha256": core.sha256(stage / "summary.json"),
            "metrics_sha256": core.sha256(stage / "metrics.csv"),
        }
        (stage / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        if audit["status"] != "PASS":
            raise RuntimeError(audit["failed_checks"])
        artifacts = {
            str(path.relative_to(stage)): core.sha256(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        }
        (stage / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
        )
        os.replace(stage, target)
        print(target, flush=True)
        print(f"audit_sha256={core.sha256(target / 'audit.json')}", flush=True)
        return 0
    except Exception:
        # Preserve a failed stage for diagnosis; never publish a partial target.
        raise


if __name__ == "__main__":
    raise SystemExit(main())
