#!/usr/bin/env python3
"""Explicit fail-closed structure checks for the S2 full campaign."""

import json
from pathlib import Path

import active_hbm_v31 as v31
import poisson_fault_batch_v2 as batch
import run_active_hbm_v313_s2_full_concurrency as runner


ROOT = Path(__file__).resolve().parent


def main() -> None:
    spec = json.loads((ROOT / "active_hbm_v313_s2_full_concurrency_protocol.json").read_text())
    checks = []
    checks.append(("exact_concurrency_grid", spec["workload"]["concurrent_workflows_per_worker"] == [1, 2, 4, 8, 16]))
    checks.append(("exact_rate_grid", spec["failure_exposure"]["failure_rates_per_1000_gpu_shard_hour"] == [0.0, 5.952, 20.833]))
    checks.append(("ten_paired_seeds", len(spec["workload"]["paired_workload_seeds"]) == 10 and len(set(spec["workload"]["paired_workload_seeds"])) == 10))
    checks.append(("exact_policies", spec["policies"] == ["Native Mooncake", "AegisKV"]))
    checks.append(("exact_300_seed_policy_rows", spec["matrix"]["seed_policy_rate_concurrency_summary_rows"] == 5 * 3 * 2 * 10))
    windows = spec["failure_exposure"]["independent_windows_per_workload_seed"]
    exposure = 16 * 90_000 / 3_600_000 * windows * 10
    checks.append(("exposure_derivation", windows == 840 and abs(exposure - 3360.0) < 1e-12))
    checks.append(("expected_mid_high_events", abs(5.952 / 1000 * exposure - 19.99872) < 1e-10 and abs(20.833 / 1000 * exposure - 69.99888) < 1e-10))
    checks.append(("a100_capacity_frozen", spec["hardware_topology"]["hbm_bytes_per_gpu"] == 40 * 1024**3 and spec["capacity"]["analytical_max_serving_requests_without_peer_backup"] == 7))
    checks.append(("two_node_topology", spec["hardware_topology"]["system_dp"] == 16 and spec["hardware_topology"]["gpus_per_node"] == 8 and spec["hardware_topology"]["node_count"] == 2))
    checks.append(("directional_bandwidths", len(set(spec["hardware_topology"]["copy_bandwidth_gb_s"].values())) > 1 and len(set(spec["hardware_topology"]["recovery_bandwidth_gb_s"].values())) > 1))
    protocol, _capacity = runner.configured_protocol(4)
    tasks, metadata = v31.build_workload(protocol, seed=2026082600, active_per_gpu=4, arrival_cutoff_ms=1000.0)
    checks.append(("no_oracle_workload_and_slack_hashed", metadata["slack_predictor"]["reads_realized_output_tokens"] is False and metadata["slack_predictor"]["reads_realized_service_ms"] is False and runner.canonical_workload_hash(tasks) != batch.canonical_sha256([(row.request_id, row.output_tokens) for row in tasks])))
    checks.append(("no_selection_and_50_workers", spec["selection"].startswith("NONE") and spec["matrix"]["max_parallel_workers"] == 50 and spec["paper_ready"] is False))
    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS")
    if failed:
        raise SystemExit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
