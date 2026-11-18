#!/usr/bin/env python3
"""Executable ledger-first and memoization gates for batched S2."""

from __future__ import annotations

from copy import deepcopy

import active_hbm_v313 as v313
import poisson_fault_batch_v2 as batch
import poisson_fault_injection_v1 as v1
import run_preliminary as core
import test_active_hbm_v313 as inputs


def main() -> None:
    checks: list[tuple[str, bool]] = []
    protocol, capacity = inputs.inputs(context=4096, active=1)
    rates = (0.0, 5.952, 20.833)
    full_expected = {
        rate: batch.expected_events(
            rate, dp=16, horizon_ms=90000.0, window_count=8400
        ) for rate in rates
    }
    checks.append((
        "preregistered_full_exposure_has_expected_20_and_70_events",
        full_expected[0.0] == 0.0
        and abs(full_expected[5.952] - 19.99872) <= 1e-12
        and abs(full_expected[20.833] - 69.99888) <= 1e-12,
    ))

    windows = {
        rate: batch.generate_batch_windows(
            rate=rate, workload_seeds=[2026082600], windows_per_seed=2800,
            horizon_ms=90000.0, dp=16, gpus_per_node=8,
            repair_ms=60000.0,
            failure_type_weights={"source_worker_gpu": 0.5, "source_node": 0.5},
        ) for rate in rates
    }
    checks.append((
        "all_pilot_windows_have_unique_deterministic_ids_seeds_and_frozen_hashes",
        all(len(rows) == 2800 for rows in windows.values())
        and all(
            len({row["window_id"] for row in rows}) == 2800
            and len({row["fault_seed"] for row in rows}) == 2800
            and all(
                row["event_ledger_sha256"]
                == batch.canonical_sha256(row["event_ledger"])
                for row in rows
            )
            for rows in windows.values()
        ),
    ))
    checks.append((
        "zero_rate_empty_and_nonzero_events_are_timestamped_in_window",
        sum(row["candidate_event_count"] for row in windows[0.0]) == 0
        and sum(row["candidate_event_count"] for row in windows[5.952]) > 0
        and sum(row["candidate_event_count"] for row in windows[20.833]) > 0
        and all(
            0.0 <= event["time_ms"] <= 90000.0
            and event["failure_type"] in {"source_worker_gpu", "source_node"}
            and event["domain"].startswith(("gpu:", "node:"))
            for rate in rates for row in windows[rate]
            for event in row["event_ledger"]
        ),
    ))

    nominal, _metadata = core.build_workload(
        protocol, seed=2026082600, workflows=None, active_per_gpu=1,
        arrival_cutoff_ms=2000.0,
    )
    exact_equal = True
    for policy in ("Native Mooncake", "AegisKV"):
        healthy_replay = v313.CausalPhysicalTargetReplay(
            deepcopy(protocol), policy, capacity
        )
        healthy, _delays, _mechanism, _trace = healthy_replay.run_coupled(
            nominal, [], control_horizon_ms=2000.0
        )
        injected_replay = batch.BatchedPoissonFaultInjectedReplay(
            deepcopy(protocol), policy, capacity
        )
        injected, injected_mechanism, _trace = injected_replay.run_rate_injected(
            nominal, [], control_horizon_ms=2000.0
        )
        exact_equal &= (
            {row.request_id: (row.start_ms, row.finish_ms) for row in healthy}
            == {row.request_id: (row.start_ms, row.finish_ms) for row in injected}
            and injected_mechanism["poisson_fault_batch_v2"]["accepted_event_count"] == 0
        )
    checks.append((
        "zero_event_memoization_matches_executed_des_exactly", exact_equal
    ))

    repeated = v1.resolve_fault_candidates(
        [
            v1.FaultCandidate(500.0, "source_worker_gpu", 0),
            v1.FaultCandidate(600.0, "source_worker_gpu", 0),
        ],
        dp=16, gpus_per_node=8, repair_ms=60000.0,
    )
    checks.append((
        "repeat_domain_is_auditable_noop_before_repair",
        [row["accepted"] for row in repeated] == [True, False]
        and repeated[1]["outcome"] == "NOOP_WORKER_ALREADY_DOWN"
        and repeated[1]["repair_at_ms"] == repeated[0]["repair_at_ms"],
    ))

    synthetic = [
        {
            "workload_seed": 1,
            "wct_mean_ms": value,
            "candidate_event_count": int(index > 0),
            "accepted_event_count": int(index > 0),
            "noop_event_count": 0,
            "memoized_zero_event": index == 0,
        }
        for index, value in enumerate((10.0, 20.0, 30.0))
    ]
    summary = batch.summarize_window_samples(synthetic)
    checks.append((
        "estimator_is_literal_all_window_sample_mean_including_zero_windows",
        summary["window_count"] == 3
        and summary["memoized_zero_event_window_count"] == 1
        and summary["wct_mean_ms"] == 20.0,
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
