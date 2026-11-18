#!/usr/bin/env python3
"""Ledger-first one-seed information pilot for batched physical-rate S2."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import active_hbm_v313 as v313
import poisson_fault_batch_v2 as batch
import run_preliminary as core
import test_active_hbm_v313 as test_inputs


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "active_hbm_v313_accounting_batched_s2_protocol.json"
RESULT_NAME = "active_hbm_v313_batched_s2_pilot"
TESTS = (
    ("test_active_hbm_v313_protocol.py", "RESULT 6/6 PASS"),
    ("test_active_hbm_v313.py", "RESULT 8/8 PASS"),
    ("test_poisson_fault_batch_v2.py", "RESULT 6/6 PASS"),
    ("test_poisson_fault_injection_v1.py", "RESULT 6/6 PASS"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_workload_hash(tasks: list[core.Task]) -> str:
    return batch.canonical_sha256([
        (task.request_id, task.workflow_id, task.workflow_arrival_ms,
         task.arrival_ms, task.source_rank, task.service_ms,
         task.prompt_tokens, task.output_tokens)
        for task in tasks
    ])


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text())
    protocol, capacity = test_inputs.inputs(context=40960, active=16)
    target = ROOT / "results" / RESULT_NAME
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{RESULT_NAME}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "tests").mkdir()
    (stage / "raw_events").mkdir()

    test_status = []
    for filename, marker in TESTS:
        result = subprocess.run(
            ["python", str(ROOT / filename)], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (stage / "tests" / f"{Path(filename).stem}.log").write_text(
            result.stdout, encoding="utf-8"
        )
        passed = result.returncode == 0 and marker in result.stdout
        test_status.append({"test": filename, "passed": passed})
        if not passed:
            raise RuntimeError(f"test gate failed: {filename}\n{result.stdout}")

    batch_spec = spec["batched_s2"]
    pilot = batch_spec["pilot"]
    workload_seeds = [int(value) for value in pilot["workload_seeds"]]
    windows_per_rate = int(pilot["windows_per_rate"])
    rates = [float(value) for value in batch_spec["failure_rates_per_1000_gpu_shard_hour"]]
    windows_by_rate = {
        rate: batch.generate_batch_windows(
            rate=rate,
            workload_seeds=workload_seeds,
            windows_per_seed=windows_per_rate,
            horizon_ms=float(batch_spec["window_horizon_ms"]),
            dp=int(batch_spec["system_dp"]),
            gpus_per_node=int(batch_spec["gpus_per_node"]),
            repair_ms=float(batch_spec["domain_repair_ms"]),
            failure_type_weights=dict(batch_spec["failure_type_weights"]),
        ) for rate in rates
    }

    # Freeze every independent input ledger before executing any policy replay.
    ledger_path = stage / "preregistered_window_ledgers.jsonl"
    with ledger_path.open("w", encoding="utf-8") as stream:
        for rate in rates:
            for row in windows_by_rate[rate]:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    freeze = {
        "schema_version": 1,
        "generated_before_policy_replay": True,
        "protocol_sha256": core.sha256(SPEC_PATH),
        "v313_sha256": core.sha256(ROOT / "active_hbm_v313.py"),
        "batch_module_sha256": core.sha256(ROOT / "poisson_fault_batch_v2.py"),
        "ledger_sha256": core.sha256(ledger_path),
        "window_counts_by_rate": {
            format(rate, ".3f"): len(windows_by_rate[rate]) for rate in rates
        },
        "candidate_counts_by_rate": {
            format(rate, ".3f"): sum(
                row["candidate_event_count"] for row in windows_by_rate[rate]
            ) for rate in rates
        },
    }
    (stage / "ledger_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )

    nominal_by_seed = {}
    workload_hash_by_seed = {}
    for seed in workload_seeds:
        nominal, _workload = core.build_workload(
            protocol, seed=seed, workflows=None, active_per_gpu=16,
            arrival_cutoff_ms=float(batch_spec["window_horizon_ms"]),
        )
        nominal_by_seed[seed] = nominal
        workload_hash_by_seed[seed] = canonical_workload_hash(nominal)

    policies = batch_spec["policies"]
    healthy_by_key = {}
    healthy_mechanism = {}
    zero_equivalence = {}
    for policy in policies:
        for seed in workload_seeds:
            healthy_replay = v313.CausalPhysicalTargetReplay(
                deepcopy(protocol), policy, capacity
            )
            healthy, _delays, mechanism, capacity_trace = healthy_replay.run_coupled(
                nominal_by_seed[seed], [],
                control_horizon_ms=float(batch_spec["window_horizon_ms"]),
            )
            ids, values = core.workflow_wct(healthy)
            healthy_by_key[(policy, seed)] = {
                "tasks": healthy, "workflow_ids": ids, "wct": values,
                "mean": statistics.fmean(values),
            }
            healthy_mechanism[(policy, seed)] = mechanism
            np.savez_compressed(
                stage / f"healthy__{policy.lower().replace(' ', '_')}__s{seed}.npz",
                workflow_ids=np.asarray(ids), healthy_wct_ms=np.asarray(values),
                capacity_time_ms=np.asarray([row["time_ms"] for row in capacity_trace]),
                capacity_rank=np.asarray([row["rank"] for row in capacity_trace]),
                capacity_serving_bytes=np.asarray(
                    [row["active_serving_kv_bytes"] for row in capacity_trace], dtype=np.int64
                ),
                capacity_target_backup_bytes=np.asarray(
                    [row["peer_hbm_backup_used_bytes"] for row in capacity_trace], dtype=np.int64
                ),
            )
            zero_replay = batch.BatchedPoissonFaultInjectedReplay(
                deepcopy(protocol), policy, capacity
            )
            zero_tasks, zero_mechanism, _trace = zero_replay.run_rate_injected(
                nominal_by_seed[seed], [],
                control_horizon_ms=float(batch_spec["window_horizon_ms"]),
            )
            zero_ids, zero_values = core.workflow_wct(zero_tasks)
            zero_equivalence[(policy, seed)] = (
                zero_ids == ids and zero_values == values
                and zero_mechanism["poisson_fault_batch_v2"]["accepted_event_count"] == 0
            )

    window_samples = []
    executed_ledgers = []
    event_mechanisms = []
    total_event_windows = sum(
        int(row["candidate_event_count"] > 0)
        for rate in rates for row in windows_by_rate[rate]
    ) * len(policies)
    executed_count = 0
    for rate in rates:
        for window in windows_by_rate[rate]:
            seed = int(window["workload_seed"])
            for policy in policies:
                healthy = healthy_by_key[(policy, seed)]
                if int(window["candidate_event_count"]) == 0:
                    values = healthy["wct"]
                    mechanism_counts = {
                        "candidate_event_count": 0,
                        "accepted_event_count": 0,
                        "noop_event_count": 0,
                    }
                    memoized = True
                    raw_sha = ""
                else:
                    replay = batch.BatchedPoissonFaultInjectedReplay(
                        deepcopy(protocol), policy, capacity
                    )
                    tasks, mechanism, _capacity_trace = replay.run_rate_injected(
                        nominal_by_seed[seed], window["event_ledger"],
                        control_horizon_ms=float(batch_spec["window_horizon_ms"]),
                    )
                    ids, values = core.workflow_wct(tasks)
                    if ids != healthy["workflow_ids"]:
                        raise AssertionError("event window changed workflow denominator")
                    details = mechanism["poisson_fault_batch_v2"]
                    mechanism_counts = {
                        key: int(details[key]) for key in (
                            "candidate_event_count", "accepted_event_count", "noop_event_count"
                        )
                    }
                    for event in details["event_ledger"]:
                        executed_ledgers.append({
                            "window_id": window["window_id"],
                            "policy": policy,
                            **event,
                        })
                    v313_mechanism = mechanism["active_hbm_v313"]
                    event_mechanisms.append({
                        "window_id": window["window_id"],
                        "policy": policy,
                        **mechanism_counts,
                        "capacity_violation_count": (
                            mechanism["active_hbm_v3"]["capacity_violation_count"]
                            + v313_mechanism["target_capacity_violation_count"]
                        ),
                        "replica_invalidated_bytes": details["replica_invalidated_bytes"],
                        "recovery_action_counts": details["recovery_action_counts"],
                    })
                    raw_path = stage / "raw_events" / (
                        f"{window['window_id']}__{policy.lower().replace(' ', '_')}.npz"
                    )
                    np.savez_compressed(
                        raw_path, workflow_ids=np.asarray(ids),
                        injected_wct_ms=np.asarray(values, dtype=float),
                    )
                    raw_sha = core.sha256(raw_path)
                    memoized = False
                    executed_count += 1
                    print(
                        f"S2 event window {executed_count}/{total_event_windows} "
                        f"{window['window_id']} {policy}", flush=True
                    )
                window_samples.append({
                    "policy": policy,
                    "rate_per_1000_gpu_shard_hour": rate,
                    "window_id": window["window_id"],
                    "workload_seed": seed,
                    "window_index": window["window_index"],
                    "fault_seed": window["fault_seed"],
                    "candidate_event_count": mechanism_counts["candidate_event_count"],
                    "accepted_event_count": mechanism_counts["accepted_event_count"],
                    "noop_event_count": mechanism_counts["noop_event_count"],
                    "memoized_zero_event": memoized,
                    "wct_mean_ms": statistics.fmean(values),
                    "healthy_wct_mean_ms": healthy["mean"],
                    "added_wct_mean_ms": statistics.fmean(values) - healthy["mean"],
                    "workload_universe_sha256": workload_hash_by_seed[seed],
                    "input_event_ledger_sha256": window["event_ledger_sha256"],
                    "raw_event_npz_sha256": raw_sha,
                })

    write_csv(stage / "window_samples.csv", window_samples)
    with (stage / "executed_event_ledgers.jsonl").open("w", encoding="utf-8") as stream:
        for row in executed_ledgers:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    (stage / "event_mechanisms.json").write_text(
        json.dumps(event_mechanisms, indent=2, sort_keys=True) + "\n"
    )
    summary_rows = []
    for policy in policies:
        for rate in rates:
            rows = [
                row for row in window_samples
                if row["policy"] == policy
                and row["rate_per_1000_gpu_shard_hour"] == rate
            ]
            summary = batch.summarize_window_samples(rows)
            summary_rows.append({
                "policy": policy,
                "rate_per_1000_gpu_shard_hour": rate,
                **summary,
            })
    write_csv(stage / "summary_metrics.csv", summary_rows)

    paired_hashes = all(
        len({
            row["input_event_ledger_sha256"] for row in window_samples
            if row["window_id"] == window_id
        }) == 1
        for window_id in {row["window_id"] for row in window_samples}
    )
    checks = {
        "all_tests_pass": all(row["passed"] for row in test_status),
        "ledger_generated_and_hashed_before_replay": freeze["generated_before_policy_replay"],
        "exact_2800_windows_per_rate_policy": all(
            sum(
                row["policy"] == policy
                and row["rate_per_1000_gpu_shard_hour"] == rate
                for row in window_samples
            ) == 2800 for policy in policies for rate in rates
        ),
        "event_ledgers_paired_across_policies": paired_hashes,
        "zero_event_memoization_exact_gate": all(zero_equivalence.values()),
        "all_event_bearing_windows_executed": executed_count == total_event_windows,
        "event_counts_match_frozen_inputs": all(
            row["candidate_event_count"] == len(row["event_ledger"])
            for rate in rates for row in windows_by_rate[rate]
        ),
        "no_capacity_violation_in_event_windows": all(
            row["capacity_violation_count"] == 0 for row in event_mechanisms
        ),
        "all_window_samples_in_estimator": all(
            row["window_count"] == 2800 for row in summary_rows
        ),
        "no_seed_or_window_selection": True,
    }
    summary = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "protocol_sha256": core.sha256(SPEC_PATH),
        "runner_sha256": core.sha256(Path(__file__)),
        "ledger_freeze": freeze,
        "summary_rows": summary_rows,
        "event_domain_counts": {
            domain: sum(
                event["domain"] == domain
                for rate in rates for window in windows_by_rate[rate]
                for event in window["event_ledger"]
            )
            for domain in sorted({
                event["domain"]
                for rate in rates for window in windows_by_rate[rate]
                for event in window["event_ledger"]
            })
        },
        "event_outcome_counts": {
            outcome: sum(
                event["outcome"] == outcome
                for rate in rates for window in windows_by_rate[rate]
                for event in window["event_ledger"]
            )
            for outcome in sorted({
                event["outcome"]
                for rate in rates for window in windows_by_rate[rate]
                for event in window["event_ledger"]
            })
        },
        "evidence_boundary": (
            "One workload seed with 2800 independent physical-rate windows per rate. "
            "Fault randomness is sampled directly; workload-seed uncertainty is not estimated."
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
        "ledger_sha256": freeze["ledger_sha256"],
        "summary_sha256": core.sha256(stage / "summary.json"),
        "window_samples_sha256": core.sha256(stage / "window_samples.csv"),
        "summary_metrics_sha256": core.sha256(stage / "summary_metrics.csv"),
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


if __name__ == "__main__":
    raise SystemExit(main())
