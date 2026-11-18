#!/usr/bin/env python3
"""Independent structural regression harness for the S9 job141411 bundle."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "results" / "active_hbm_s9_rtx4090_four_direction_holdout_141411_v1"
PREREG = ROOT / "active_hbm_s9_rtx4090_four_direction_holdout_protocol.json"
EVAL = ROOT / "active_hbm_s9_rtx4090_four_direction_holdout_141411_evaluation_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(name: str) -> list[dict]:
    with (BUNDLE / "data" / name).open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    evaluation = json.loads(EVAL.read_text())
    audit = json.loads((BUNDLE / "audit.json").read_text())
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    train = read_csv("training_samples.csv")
    holdout = read_csv("holdout_samples.csv")
    cells = read_csv("holdout_errors_by_path_payload.csv")
    counts = Counter((row["path"], row["payload_bytes"]) for row in holdout)
    checks = [
        ("preregistered_protocol_hash_bound", sha256(PREREG) == evaluation["preregistered_protocol"]["sha256"] == manifest["preregistered_protocol_sha256"]),
        ("holdout_input_hash_bound", sha256(Path(evaluation["holdout_input"]["calibration_path"])) == evaluation["holdout_input"]["calibration_sha256"] == manifest["holdout_input_sha256"]),
        ("negative_mutated_holdout_hash_rejected", hashlib.sha256(Path(evaluation["holdout_input"]["calibration_path"]).read_bytes() + b"x").hexdigest() != evaluation["holdout_input"]["calibration_sha256"]),
        ("distinct_fit_and_holdout_jobs", manifest["fit_job"] == "141298" and manifest["holdout_job"] == "141411"),
        ("fit_rows_exact_80_and_fit_only", len(train) == 80 and {row["role"] for row in train} == {"FIT_ONLY"}),
        ("holdout_rows_exact_80_and_holdout_only", len(holdout) == 80 and {row["role"] for row in holdout} == {"HOLDOUT_ONLY"}),
        ("cells_exact_16_five_repeats", len(cells) == 16 and len(counts) == 16 and set(counts.values()) == {5}),
        ("four_paths_exact", {row["path"] for row in cells} == set(evaluation["paths"])),
        ("four_payloads_exact", {int(row["payload_bytes"]) for row in cells} == set(evaluation["payload_bytes"])),
        ("all_predictions_positive", all(float(row["predicted_transfer_ms"]) > 0 for row in holdout)),
        ("acceptance_uses_all_cells", sum(row["cell_acceptance_pass"] == "True" for row in cells) == 15 and sum(row["cell_acceptance_pass"] == "False" for row in cells) == 1),
        ("failed_cell_preserved", audit["failed_cells"] == [{"mape_percent": 10.635417146496447, "path": "peer_hbm_recovery", "payload_mib": 64.0}]),
        ("integrity_pass_acceptance_fail_separate", audit["audit_status"] == "PASS" and audit["calibration_acceptance"] == "FAIL"),
        ("a100_profile_unchanged", manifest["main_a100_profile_unchanged"] is True and audit["checks"]["a100_profile_not_mutated"] is True),
    ]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    passed = sum(value for _name, value in checks)
    print(f"RESULT {passed}/{len(checks)} PASS")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
