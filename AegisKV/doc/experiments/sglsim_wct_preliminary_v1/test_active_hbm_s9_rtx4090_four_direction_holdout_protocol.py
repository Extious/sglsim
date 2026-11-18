#!/usr/bin/env python3
"""Fail-closed feasibility checks for the S9 RTX4090 holdout protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "active_hbm_s9_rtx4090_four_direction_holdout_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    source = protocol["training_source"]
    calibration_path = Path(source["calibration_path"])
    manifest_path = Path(source["manifest_path"])
    contract_path = Path(source["contract_path"])
    calibration = json.loads(calibration_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    contract = json.loads(contract_path.read_text())
    samples = calibration["samples"]
    identities = [
        (
            row["sequence"], row["repeat_ordinal"], row["path"],
            row["payload_bytes"], row["probe_artifact"]["sha256"],
        )
        for row in samples
    ]
    identity_sha = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cell_counts = Counter((row["path"], row["payload_bytes"]) for row in samples)
    split_terms = ("train", "holdout", "split")
    source_documents = (manifest, calibration, contract)
    split_declared = any(
        any(term in key.lower() for term in split_terms)
        for document in source_documents
        for key in document
    )
    checks = [
        ("status_fail_closed_until_new_holdout", protocol["status"] == "FROZEN_FEASIBILITY_FAIL_CLOSED_AWAITING_NEW_HOLDOUT_JOB"),
        ("physical_source_hashes_exact", sha256(calibration_path) == source["calibration_sha256"] and sha256(manifest_path) == source["manifest_sha256"] and sha256(contract_path) == source["contract_sha256"]),
        ("exact_80_physical_samples", len(samples) == source["sample_count"] == 80),
        ("exact_four_paths_four_payloads_five_repeats", len(cell_counts) == 16 and set(cell_counts.values()) == {5}),
        ("sample_identity_hash_exact", identity_sha == source["sample_identity_sha256"]),
        ("no_predeclared_split_in_physical_sources", not split_declared and source["predeclared_train_holdout_split_present"] is False),
        ("existing_samples_fit_only", source["role"] == "FIT_ONLY_NEVER_HOLDOUT" and protocol["execution_gate"]["existing_80_can_be_used_as_holdout"] is False),
        ("four_independent_direction_curves", protocol["fit"]["independent_per_direction"] is True and len(protocol["path_mapping"]) == 4),
        ("new_holdout_exact_80_required", protocol["holdout"]["required_sample_count"] == 80 and protocol["holdout"]["required_cell_count"] == 16 and protocol["holdout"]["repeats_per_cell"] == 5),
        ("per_path_payload_errors_required", protocol["holdout_statistics"]["required_per_direction_payload_rows"] == 16),
        ("ten_percent_gate_forbids_masking", "All 16" in protocol["holdout_statistics"]["reference_acceptance_gate"] and "cannot mask" in protocol["holdout_statistics"]["aggregate_reporting"]),
        ("no_sample_direction_or_threshold_selection", "No physical sample" in protocol["selection_policy"]),
        ("a100_profile_separate_and_hash_pinned", sha256(Path(protocol["hardware_separation"]["main_a100_protocol_path"])) == protocol["hardware_separation"]["main_a100_protocol_sha256"]),
        ("no_fit_or_holdout_started", protocol["execution_gate"]["calibration_fit_started"] is False and protocol["execution_gate"]["holdout_evaluation_started"] is False),
    ]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    passed = sum(value for _name, value in checks)
    print(f"RESULT {passed}/{len(checks)} PASS")
    print("FEASIBILITY BLOCKED_NEW_INDEPENDENT_HOLDOUT_REQUIRED")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
