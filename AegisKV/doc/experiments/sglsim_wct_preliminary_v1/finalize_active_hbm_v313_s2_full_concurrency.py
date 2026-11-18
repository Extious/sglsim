#!/usr/bin/env python3
"""Linear-time fail-closed finalizer for a complete immutable S2 raw stage."""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from pathlib import Path

from PIL import Image

import run_active_hbm_v313_s2_full_concurrency as runner
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "active_hbm_v313_s2_full_concurrency_protocol.json"
GIB = 1024 ** 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: finalize...py STAGE")
    stage = Path(sys.argv[1]).resolve()
    target = ROOT / "results" / "active_hbm_v313_s2_full_concurrency_physical_rate_v1"
    if target.exists():
        raise FileExistsError(target)
    spec = json.loads(SPEC.read_text())
    freeze = json.loads((stage / "execution_freeze.json").read_text())
    jobs = json.loads((stage / "job_results.json").read_text())
    failures = json.loads((stage / "job_failures.json").read_text())
    seed_rows = read_csv(stage / "seed_policy_metrics.csv")
    paired_rows = read_csv(stage / "paired_seed_metrics.csv")
    ci_rows = read_csv(stage / "paired_ci.csv")
    workload_rows = read_csv(stage / "workload_metadata.csv")

    zero = []
    minimum_deltas = []
    for result in jobs:
        path = stage / result["cell_path"]
        if core.sha256(path) != result["cell_sha256"]:
            raise AssertionError(f"cell hash changed: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        zero.extend(payload["zero_event_equivalence"].values())
        value = payload["minimum_fault_minus_healthy_workflow_wct_ms"]
        if value is not None:
            minimum_deltas.append(float(value))

    ledger_hashes: dict[tuple[int, int, float], set[str]] = {}
    window_count = 0
    with (stage / "window_samples.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            window_count += 1
            key = (
                int(row["workload_seed"]), int(row["window_index"]),
                float(row["rate_per_1000_gpu_shard_hour"]),
            )
            ledger_hashes.setdefault(key, set()).add(row["input_event_ledger_sha256"])

    figures = {
        "main_pdf": "figures/s2_v313_full_concurrency_observed.pdf",
        "main_png": "figures/s2_v313_full_concurrency_observed.png",
        "mechanism_pdf": "figures/s2_v313_full_concurrency_mechanisms.pdf",
        "mechanism_png": "figures/s2_v313_full_concurrency_mechanisms.png",
    }
    for key in ("main_png", "mechanism_png"):
        width, height = Image.open(stage / figures[key]).size
        figures[f"{key}_width_px"] = width
        figures[f"{key}_height_px"] = height
    for key in ("main_pdf", "mechanism_pdf"):
        figures[f"{key}_min_font_pt"] = runner.pdf_min_font_pt(stage / figures[key])
    source_checks = runner.source_hash_checks(spec)
    expected_seed_rows = spec["matrix"]["seed_policy_rate_concurrency_summary_rows"]
    expected_windows = 5 * 10 * 2 * 3 * 840
    expected_ledger_keys = 10 * 3 * 840
    slack_ok = True
    for row in workload_rows:
        metadata = json.loads(row["slack_predictor"])
        slack_ok &= metadata["reads_realized_output_tokens"] is False
        slack_ok &= metadata["reads_realized_service_ms"] is False
    checks = {
        **source_checks,
        "complete_raw_jobs_50_of_50": len(jobs) == 50,
        "no_worker_failures": failures == [],
        "immutable_cell_hashes_verified": True,
        "exact_300_seed_policy_rate_concurrency_rows": len(seed_rows) == expected_seed_rows,
        "exact_150_paired_seed_rows": len(paired_rows) == 150,
        "exact_15_paired_ci_rows": len(ci_rows) == 15,
        "exact_252000_window_samples": window_count == expected_windows,
        "old_core_build_workload_oracle_lineage_fail_closed": (
            freeze["workload_builder"] == "active_hbm_v31.build_workload"
            and freeze["old_core_build_workload_lineage_rejected"]
            and freeze["predicted_slack_in_workload_universe_hash"]
            and slack_ok
        ),
        "ledger_generated_and_hashed_before_policy_replay": freeze["generated_before_policy_replay"],
        "event_ledgers_paired_across_policies_and_concurrency": (
            len(ledger_hashes) == expected_ledger_keys
            and all(len(values) == 1 for values in ledger_hashes.values())
        ),
        "workload_universe_paired_across_policies": all(
            truth(row["paired_universe_hash_equal"]) for row in paired_rows
        ),
        "zero_event_memoization_exact": all(zero) and len(zero) == 100,
        "fault_replay_never_reduces_workflow_wct": min(minimum_deltas) >= -1e-7,
        "no_capacity_violations": all(int(row["capacity_violation_count_max"]) == 0 for row in seed_rows),
        "copy_byte_conservation": all(int(row["copy_byte_conservation_error_max_abs"]) == 0 for row in seed_rows),
        "asynchronous_copy_not_directly_charged_to_wct": all(int(row["health_burden_application_count_max"]) == 0 for row in seed_rows),
        "no_future_backup_or_seal_oracle": all(int(row["future_backup_projection_dispatch_count_max"]) == 0 and int(row["seal_before_dispatch_count_max"]) == 0 for row in seed_rows),
        "exact_three_aegis_control_opportunities": all(int(row["registered_control_opportunity_count"]) == 3 and (int(row["executed_policy_control_count"]) == 3 if row["policy"] == "AegisKV" else True) for row in seed_rows),
        "resource_integrals_nonnegative": all(float(row[key]) >= -1e-12 for row in seed_rows for key in ("serving_hbm_time_mean_gib_per_gpu", "peer_hbm_backup_time_mean_gib_per_gpu", "dram_time_mean_gib_per_gpu")),
        "shared_hbm_peak_within_frozen_budget": all(float(row["peer_plus_serving_peak_mean_gib_per_gpu"]) <= spec["capacity"]["serving_plus_peer_backup_budget_bytes_per_gpu"] / GIB + 1e-9 for row in seed_rows),
        "reuse_fraction_exactly_zero": all(float(row["reuse_fraction"]) == 0.0 for row in seed_rows),
        "all_registered_cells_and_negative_results_retained": len({(row["concurrent_workflows_per_worker"], row["workload_seed"], row["policy"], row["rate_per_1000_gpu_shard_hour"]) for row in seed_rows}) == 300,
        "png_width_at_least_1800": min(figures["main_png_width_px"], figures["mechanism_png_width_px"]) >= 1800,
        "pdf_min_font_at_least_7pt": min(figures["main_pdf_min_font_pt"], figures["mechanism_pdf_min_font_pt"]) >= 6.99,
        "vector_pdf_and_png_present": all((stage / figures[key]).is_file() for key in ("main_pdf", "main_png", "mechanism_pdf", "mechanism_png")),
        "no_performance_direction_gate": spec["selection"].startswith("NONE"),
    }
    lineage = {
        "schema_version": 1,
        "raw_execution_runner_sha256": freeze["runner_sha256"],
        "postprocessor_sha256": core.sha256(Path(__file__)),
        "raw_mutated": False,
        "reason": "The original postprocessor used a quadratic ledger-pairing scan and was interrupted only after all 50 immutable cell files, all 900 event raw files, summaries, and figures were complete. This finalizer replaces only that scan with an equivalent linear grouping and verifies every cell hash.",
    }
    (stage / "postprocess_lineage.json").write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "protocol_sha256": core.sha256(SPEC),
        "raw_execution_runner_sha256": freeze["runner_sha256"],
        "postprocessor_sha256": core.sha256(Path(__file__)),
        "execution_freeze_sha256": core.sha256(stage / "execution_freeze.json"),
        "ledger_sha256": core.sha256(stage / "preregistered_window_ledgers.jsonl"),
        "old_fixed_c16_lineage": "INVALIDATED_ORACLE_SLACK_DIAGNOSTIC_ONLY",
        "minimum_fault_minus_healthy_workflow_wct_ms": min(minimum_deltas),
        "actual_candidate_events_by_rate": freeze["candidate_event_counts_by_rate"],
        "figures": figures,
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(audit["failed_checks"])
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "paper_ready": False,
        "protocol": SPEC.name,
        "protocol_sha256": core.sha256(SPEC),
        "raw_execution_runner_sha256": freeze["runner_sha256"],
        "postprocessor": Path(__file__).name,
        "postprocessor_sha256": core.sha256(Path(__file__)),
        "audit": "audit.json",
        "audit_sha256": core.sha256(stage / "audit.json"),
        "execution_freeze": "execution_freeze.json",
        "postprocess_lineage": "postprocess_lineage.json",
        "ledger": "preregistered_window_ledgers.jsonl",
        "window_samples": "window_samples.csv",
        "seed_policy_metrics": "seed_policy_metrics.csv",
        "paired_seed_metrics": "paired_seed_metrics.csv",
        "paired_ci": "paired_ci.csv",
        "event_mechanisms": "event_mechanisms.csv",
        "workload_metadata": "workload_metadata.csv",
        "figures": figures,
        "seed_policy_rate_concurrency_rows": 300,
        "paired_seed_rows": 150,
        "paired_ci_rows": 15,
        "window_sample_rows": 252000,
        "execution_workers": 50,
        "old_fixed_c16_lineage": "INVALIDATED_ORACLE_SLACK_DIAGNOSTIC_ONLY",
        "workload_builder": "active_hbm_v31.build_workload",
        "predicted_slack_in_universe_hash": True,
        "no_result_or_seed_selection": True,
        "s9_status": "AWAITING_NEW_INDEPENDENT_PHYSICAL_HOLDOUT_NO_POSTHOC_141298_SPLIT",
        "evidence_boundary": spec["evidence_boundary"],
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (stage / SPEC.name).write_text(SPEC.read_text(), encoding="utf-8")
    (stage / "figure_caption.txt").write_text(
        "S2 uses actual 90 s Poisson exposure windows and ten paired workload seeds. All-workflow WCT is primary; error bars are paired Student-t 95% CIs. A100 path bandwidths, activation, replacement, and repair are modeled/unmeasured, and Native Mooncake is a modeled Router+Store approximation.\n",
        encoding="utf-8",
    )
    artifacts = {
        str(path.relative_to(stage)): core.sha256(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items())), encoding="utf-8")
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    print(f"manifest_sha256={core.sha256(target / 'manifest.json')}")
    print(f"sha256sums_sha256={core.sha256(target / 'SHA256SUMS')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
