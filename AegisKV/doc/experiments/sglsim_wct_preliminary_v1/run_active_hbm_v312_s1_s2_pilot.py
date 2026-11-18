#!/usr/bin/env python3
"""Run the atomic v3.1.2 six-architecture and rate-injection pilots."""

from __future__ import annotations

import csv
import hashlib
import json
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
import active_hbm_v312 as v312  # noqa: E402
import poisson_fault_injection_v1 as injected  # noqa: E402
import run_preliminary as core  # noqa: E402


SPEC_PATH = ROOT / "active_hbm_v311_s1_s2_protocol.json"
CAPACITY_PATH = ROOT / "active_hbm_v3_protocol.json"
BASE_PATH = ROOT / "results/mechanism_pilot_v2/composite_protocol.json"
RESULT_NAME = "active_hbm_v312_s1_s2_pilot"
TESTS = (
    (ROOT / "test_active_hbm_v311_s1_s2.py", "RESULT 8/8 PASS"),
    (ROOT / "test_active_hbm_v312.py", "RESULT 5/5 PASS"),
    (ROOT / "test_poisson_fault_injection_v1.py", "RESULT 6/6 PASS"),
    (ROOT / "test_active_hbm_v31.py", "RESULT 7/7 PASS"),
    (ROOT / "test_active_hbm_v3_capacity.py", "RESULT 5/5 PASS"),
)


def sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def load_protocols() -> tuple[dict, dict, dict]:
    spec = json.loads(SPEC_PATH.read_text())
    capacity = json.loads(CAPACITY_PATH.read_text())
    base = json.loads(BASE_PATH.read_text())
    for path_key, sha_key in (
        ("current_design_path", "current_design_sha256"),
        ("v31_des_path", "v31_des_sha256"),
        ("core_runner_path", "core_runner_sha256"),
        ("capacity_protocol_path", "capacity_protocol_sha256"),
        ("base_protocol_path", "base_protocol_sha256"),
        ("execution_plan_path", "execution_plan_sha256"),
        ("topology_recovery_path", "topology_recovery_sha256"),
        ("poisson_injection_path", "poisson_injection_sha256"),
    ):
        path = Path(spec["sources"][path_key])
        if not path.is_absolute():
            path = ROOT / path
        if core.sha256(path) != spec["sources"][sha_key]:
            raise RuntimeError(f"frozen source changed: {path}")
    return base, capacity, spec


def configured(base: dict, spec: dict, *, active: int) -> dict:
    protocol = deepcopy(base)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["workload"]["center_context_tokens"] = 40960
    protocol["workload"]["center_active_requests_per_gpu"] = int(active)
    protocol["hardware"]["gpus_per_node"] = 8
    protocol["hardware"]["node_count"] = 2
    protocol["hardware"]["copy_bandwidth_gb_s"] = dict(
        spec["hardware_topology"]["copy_bandwidth_gb_s"]
    )
    protocol["hardware"]["load_bandwidth_gb_s"] = dict(
        spec["hardware_topology"]["recovery_bandwidth_gb_s"]
    )
    protocol.setdefault("runtime_overrides", {}).pop("topology", None)
    contract = spec["policy_failover_contract"]
    protocol["failure_model"]["policy_failover"] = {
        "replacement_delay_ms": contract["replacement_delay_ms"],
        "fence_activation_delay_ms": contract["fence_activation_delay_ms"],
        "policy_modes": deepcopy(contract["policy_modes"]),
    }
    return protocol


def run_s1_job(args: tuple[dict, dict, dict, str, str, int, Path]) -> dict:
    base, capacity, spec, display, failure_type, seed, output = args
    protocol = configured(base, spec, active=int(spec["pilot"]["active_requests_per_gpu"]))
    protocol["failure_model"]["probe_type_schedule"] = [failure_type]
    internal = spec["display_to_internal_policy"][display]
    cell = v312.run_v312_cell(
        protocol,
        capacity,
        policy=internal,
        seed=int(seed),
        active_per_gpu=int(spec["pilot"]["active_requests_per_gpu"]),
        probe_count=int(spec["pilot"]["fault_probes_per_domain"]),
        output=output,
        arrival_cutoff_ms=float(spec["pilot"]["arrival_cutoff_ms"]),
    )
    cell["display_policy"] = display
    return cell


def run_s2_job(args: tuple[dict, dict, dict, str, float, int, Path]) -> dict:
    base, capacity, spec, display, rate, seed, output = args
    active = 16
    protocol = configured(base, spec, active=active)
    internal = spec["display_to_internal_policy"][display]
    cutoff = float(spec["common"]["arrival_cutoff_ms"])
    nominal, workload = v31.build_workload(
        protocol, seed=int(seed), active_per_gpu=active, arrival_cutoff_ms=cutoff
    )
    ledger = injected.generate_poisson_fault_ledger(
        rate_per_1000_gpu_shard_hour=float(rate),
        seed=int(seed),
        horizon_ms=cutoff,
        dp=16,
        gpus_per_node=8,
        repair_ms=float(spec["S2"]["domain_repair_ms"]),
        failure_type_weights=spec["S2"]["failure_type_weights"],
    )
    healthy_replay = injected.PoissonFaultInjectedReplay(protocol, internal, capacity)
    healthy_tasks, healthy_mechanism, _healthy_capacity = healthy_replay.run_rate_injected(
        nominal, [], control_horizon_ms=cutoff
    )
    fault_replay = injected.PoissonFaultInjectedReplay(protocol, internal, capacity)
    fault_tasks, mechanism, capacity_trace = fault_replay.run_rate_injected(
        nominal, ledger, control_horizon_ms=cutoff
    )
    ids, healthy_wct = core.workflow_wct(healthy_tasks)
    fault_ids, fault_wct = core.workflow_wct(fault_tasks)
    if ids != fault_ids:
        raise AssertionError("rate injection changed workflow universe")
    if any(fault < healthy - 1e-7 for healthy, fault in zip(healthy_wct, fault_wct)):
        raise AssertionError("rate injection reduced workflow WCT")
    topology_rows = mechanism["active_hbm_v312"]["topology_rows"]
    capacity_array = np.asarray([
        (
            row["time_ms"], row["rank"], row["active_requests"],
            row["waiting_requests"], row["active_serving_kv_bytes"],
            row["peer_hbm_backup_used_bytes"], row["shared_used_bytes"],
            row["shared_headroom_bytes"], row["total_hbm_used_bytes"],
        )
        for row in capacity_trace
    ], dtype=np.float64).reshape((-1, 9))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        workflow_id=np.asarray(ids),
        healthy_wct_ms=np.asarray(healthy_wct, dtype=np.float64),
        fault_wct_ms=np.asarray(fault_wct, dtype=np.float64),
        event_id=np.asarray([row["event_id"] for row in ledger]),
        event_time_ms=np.asarray([row["time_ms"] for row in ledger], dtype=np.float64),
        event_failure_type=np.asarray([row["failure_type"] for row in ledger]),
        event_source_rank=np.asarray([row["source_rank"] for row in ledger], dtype=np.int16),
        event_accepted=np.asarray([row["accepted"] for row in ledger], dtype=np.bool_),
        capacity_time_ms=capacity_array[:, 0],
        capacity_rank=capacity_array[:, 1],
        capacity_active_requests=capacity_array[:, 2],
        capacity_waiting_requests=capacity_array[:, 3],
        capacity_active_serving_kv_bytes=capacity_array[:, 4],
        capacity_peer_hbm_backup_used_bytes=capacity_array[:, 5],
        capacity_shared_used_bytes=capacity_array[:, 6],
        capacity_shared_headroom_bytes=capacity_array[:, 7],
        capacity_total_hbm_used_bytes=capacity_array[:, 8],
    )
    workflow_hash = sha_json(ids)
    event_hash = sha_json(ledger)
    return {
        "display_policy": display,
        "policy": internal,
        "seed": int(seed),
        "active_requests_per_gpu": active,
        "failure_rate_per_1000_gpu_shard_hour": float(rate),
        "arrival_cutoff_ms": cutoff,
        "workflow_count": len(ids),
        "workflow_universe_sha256": workflow_hash,
        "event_ledger_sha256": event_hash,
        "candidate_event_count": len(ledger),
        "accepted_event_count": sum(bool(row["accepted"]) for row in ledger),
        "noop_event_count": sum(not bool(row["accepted"]) for row in ledger),
        "healthy_wct_mean_ms": float(np.mean(healthy_wct)),
        "fault_injected_wct_mean_ms": float(np.mean(fault_wct)),
        "added_wct_mean_ms": float(np.mean(np.asarray(fault_wct) - np.asarray(healthy_wct))),
        "capacity_violation_count": mechanism["active_hbm_v3"]["capacity_violation_count"],
        "projection_prediction_mismatch_count": mechanism["active_hbm_v31"]["projection_prediction_mismatch_count"],
        "topology_rows_sha256": sha_json(topology_rows),
        "healthy_mechanism": healthy_mechanism,
        "mechanism": mechanism,
        "event_ledger": mechanism["poisson_fault_injection_v1"]["event_ledger"],
        "workload": workload,
        "raw_npz": str(output),
        "raw_npz_sha256": core.sha256(output),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def main() -> int:
    base, capacity, spec = load_protocols()
    target = ROOT / "results" / RESULT_NAME
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{target.name}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "raw/s1").mkdir(parents=True)
    (stage / "raw/s2").mkdir(parents=True)

    tests = []
    for path, expected in TESTS:
        completed = subprocess.run(
            [sys.executable, str(path)], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        log = stage / f"{path.stem}.log"
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode or expected not in completed.stdout:
            raise RuntimeError(f"test failed: {log}")
        tests.append({
            "path": str(path), "sha256": core.sha256(path),
            "expected": expected, "log": str(target / log.name),
            "log_sha256": core.sha256(log),
        })

    mapping = spec["display_to_internal_policy"]
    s1_jobs = []
    for seed in spec["pilot"]["paired_seeds"]:
        for failure_type in spec["S1"]["failure_types"]:
            for display in spec["S1"]["display_policies"]:
                raw = stage / "raw/s1" / (
                    f"{slug(display)}__{failure_type}__s{seed}.npz"
                )
                s1_jobs.append((base, capacity, spec, display, failure_type, int(seed), raw))
    s1_cells = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(run_s1_job, job): job[3:6] for job in s1_jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            s1_cells.append(future.result())
            print(f"S1 PILOT {index}/{len(s1_jobs)} {futures[future]}", flush=True)

    s2_jobs = []
    seed = int(spec["pilot"]["paired_seeds"][0])
    for rate in spec["S2"]["failure_rates_per_1000_gpu_shard_hour"]:
        for display in spec["S2"]["display_policies"]:
            raw = stage / "raw/s2" / f"{slug(display)}__r{rate:g}__s{seed}.npz"
            s2_jobs.append((base, capacity, spec, display, float(rate), seed, raw))
    s2_cells = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = {pool.submit(run_s2_job, job): job[3:6] for job in s2_jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            s2_cells.append(future.result())
            print(f"S2 PILOT {index}/{len(s2_jobs)} {futures[future]}", flush=True)

    s1_cells.sort(key=lambda row: (row["seed"], row["failure_type"], row["display_policy"]))
    s2_cells.sort(key=lambda row: (row["seed"], row["failure_rate_per_1000_gpu_shard_hour"], row["display_policy"]))
    for cell in s1_cells + s2_cells:
        raw = Path(cell["raw_npz"])
        cell["raw_npz"] = str(target / raw.relative_to(stage))

    s1_rows = []
    for cell in s1_cells:
        failure = cell["failure_conditioned_changed_by_type"][cell["failure_type"]]
        mechanism = cell["mechanism"]
        s1_rows.append({
            "display_policy": cell["display_policy"],
            "internal_policy": cell["policy"],
            "seed": cell["seed"],
            "failure_type": cell["failure_type"],
            "workflow_count": cell["workflow_count"],
            "probe_count": cell["fault_probe_count"],
            "healthy_wct_mean_ms": cell["base_wct_mean_ms"],
            "same_probe_all_workflow_wct_mean_ms": failure["same_probe_failure_conditioned_wct_mean_ms"],
            "all_workflow_added_wct_mean_ms": failure["all_workflow_added_wct_mean_ms"],
            "capacity_violation_count": mechanism["active_hbm_v3"]["capacity_violation_count"],
            "projection_mismatch_count": mechanism["active_hbm_v31"]["projection_prediction_mismatch_count"],
            "surviving_cross_domain_committed_count": mechanism["active_hbm_v312"]["surviving_cross_domain_committed_count"],
            "node_prebound_survivor_count": mechanism["active_hbm_v312"]["node_prebound_survivor_count"],
            "node_replacement_no_survivor_count": mechanism["active_hbm_v312"]["node_replacement_no_survivor_count"],
            "workflow_universe_sha256": cell["workflow_universe_sha256"],
            "fault_probe_tuple_sha256": cell["fault_probe_tuple_sha256"],
            "fault_domain_tuple_sha256": cell["fault_domain_tuple_sha256"],
            "raw_npz": cell["raw_npz"],
            "raw_npz_sha256": cell["raw_npz_sha256"],
        })
    s2_rows = [{
        key: cell[key] for key in (
            "display_policy", "policy", "seed", "active_requests_per_gpu",
            "failure_rate_per_1000_gpu_shard_hour", "arrival_cutoff_ms",
            "workflow_count", "candidate_event_count", "accepted_event_count",
            "noop_event_count", "healthy_wct_mean_ms", "fault_injected_wct_mean_ms",
            "added_wct_mean_ms", "capacity_violation_count",
            "projection_prediction_mismatch_count", "workflow_universe_sha256",
            "event_ledger_sha256", "topology_rows_sha256", "raw_npz", "raw_npz_sha256",
        )
    } for cell in s2_cells]
    write_csv(stage / "s1_pilot_metrics.csv", s1_rows, list(s1_rows[0]))
    write_csv(stage / "s2_pilot_metrics.csv", s2_rows, list(s2_rows[0]))
    with (stage / "s2_event_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for cell in s2_cells:
            for event in cell["event_ledger"]:
                stream.write(json.dumps({
                    "display_policy": cell["display_policy"],
                    "seed": cell["seed"],
                    "failure_rate_per_1000_gpu_shard_hour": cell["failure_rate_per_1000_gpu_shard_hour"],
                    **event,
                }, sort_keys=True) + "\n")

    checks = {
        "regressions_all_pass": len(tests) == len(TESTS),
        "s1_cell_count_exact_12": len(s1_cells) == int(spec["pilot"]["expected_cells"]),
        "s2_cell_count_exact_6": len(s2_cells) == 6,
        "display_names_exact": {row["display_policy"] for row in s1_cells} == set(mapping),
        "capacity_never_exceeded": all(
            row["mechanism"]["active_hbm_v3"]["capacity_violation_count"] == 0
            for row in s1_cells
        ) and all(row["capacity_violation_count"] == 0 for row in s2_cells),
        "projection_matches_admission": all(
            row["mechanism"]["active_hbm_v31"]["projection_prediction_mismatch_count"] == 0
            for row in s1_cells
        ) and all(row["projection_prediction_mismatch_count"] == 0 for row in s2_cells),
        "s1_common_workflow_probe_domain_universe": True,
        "s1_source_node_exact_eight_worker_exact_one": True,
        "s1_cross_domain_survivor_exercised": any(
            row["mechanism"]["active_hbm_v312"]["surviving_cross_domain_committed_count"] > 0
            for row in s1_cells if row["failure_type"] == "source_node" and row["policy"] != "Native Mooncake"
        ),
        "s2_rate_event_trace_paired_across_policies": True,
        "s2_fixed_90s_arrivals": all(row["arrival_cutoff_ms"] == 90000.0 for row in s2_cells),
        "s2_zero_rate_has_zero_events": all(
            row["candidate_event_count"] == 0
            for row in s2_cells if row["failure_rate_per_1000_gpu_shard_hour"] == 0.0
        ),
        "s2_fault_wct_monotone": True,
        "no_seed_or_performance_selection": spec["common"]["performance_direction_or_win_threshold"] is False,
    }
    s1_signatures: dict[tuple[int, str], tuple[str, str, str]] = {}
    for cell in s1_cells:
        key = (cell["seed"], cell["failure_type"])
        signature = (
            cell["workflow_universe_sha256"], cell["fault_probe_tuple_sha256"],
            cell["fault_domain_tuple_sha256"],
        )
        if key in s1_signatures and s1_signatures[key] != signature:
            checks["s1_common_workflow_probe_domain_universe"] = False
        s1_signatures[key] = signature
        raw = stage / Path(cell["raw_npz"]).relative_to(target)
        with np.load(raw, allow_pickle=False) as data:
            widths = np.sum(data["probe_failure_domain_rank_ids"] >= 0, axis=1)
            expected = 8 if cell["failure_type"] == "source_node" else 1
            if not np.all(widths == expected):
                checks["s1_source_node_exact_eight_worker_exact_one"] = False
    event_signatures: dict[tuple[int, float], str] = {}
    for cell in s2_cells:
        key = (cell["seed"], cell["failure_rate_per_1000_gpu_shard_hour"])
        signature = cell["event_ledger_sha256"]
        if key in event_signatures and event_signatures[key] != signature:
            checks["s2_rate_event_trace_paired_across_policies"] = False
        event_signatures[key] = signature
        raw = stage / Path(cell["raw_npz"]).relative_to(target)
        with np.load(raw, allow_pickle=False) as data:
            if np.any(data["fault_wct_ms"] < data["healthy_wct_ms"] - 1e-7):
                checks["s2_fault_wct_monotone"] = False

    summary = {
        "schema_version": 1,
        "status": "PILOT_COMPLETE" if all(checks.values()) else "PILOT_FAILED",
        "paper_ready": False,
        "protocol": {"path": str(SPEC_PATH), "sha256": core.sha256(SPEC_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "tests": tests,
        "s1_cells": s1_cells,
        "s2_cells": s2_cells,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "protocol_sha256": core.sha256(SPEC_PATH),
        "summary_sha256": core.sha256(stage / "summary.json"),
        "s1_metrics_sha256": core.sha256(stage / "s1_pilot_metrics.csv"),
        "s2_metrics_sha256": core.sha256(stage / "s2_pilot_metrics.csv"),
        "s2_event_ledger_sha256": core.sha256(stage / "s2_event_ledger.jsonl"),
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(
        path for path in stage.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (stage / "SHA256SUMS").write_text(
        "".join(f"{core.sha256(path)}  {path.relative_to(stage)}\n" for path in files),
        encoding="utf-8",
    )
    if audit["status"] != "PASS":
        raise RuntimeError(audit["failed_checks"])
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"summary_sha256={core.sha256(target / 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
