#!/usr/bin/env python3
"""Correct and independently re-evaluate the paired S7 repair-delay gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.result_dir
    audit_path = base / "audit.json"
    summary_path = base / "summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if audit.get("selection") != "S7" or summary.get("selection") != "S7":
        raise ValueError("reauditor only accepts S7 artifacts")
    prior_failures = [name for name, passed in audit["checks"].items() if not passed]
    if prior_failures != ["repair_delay_changes_executed_fault_wct"]:
        raise ValueError(f"unexpected prior audit failures: {prior_failures}")

    repair_cells = [
        cell for cell in summary["cells"]
        if any(axis["axis"] == "repair" for axis in cell["extension_spec"]["axes"])
    ]
    paired_ok = True
    evidence = []
    for policy in ("Native Mooncake", "AegisKV"):
        for seed in (2026082600, 2026082601, 2026082602):
            rows = {
                float(cell["extension_spec"]["repair_ms"]): float(cell["fault_conditioned_wct_mean_ms"])
                for cell in repair_cells
                if cell["policy"] == policy and int(cell["seed"]) == seed
            }
            passed = set(rows) == {30000.0, 60000.0} and rows[60000.0] > rows[30000.0]
            paired_ok &= passed
            evidence.append({
                "policy": policy, "seed": seed,
                "fault_wct_30s_ms": rows.get(30000.0),
                "fault_wct_60s_ms": rows.get(60000.0), "passed": bool(passed),
            })

    raw_ok = True
    for cell in summary["cells"]:
        raw = base / cell["raw_npz"]
        raw_ok &= raw.is_file() and sha256(raw) == cell["raw_npz_sha256"]
        with np.load(raw, allow_pickle=False) as payload:
            raw_ok &= bool(np.all(payload["base_wct_ms"] > 0.0))
            raw_ok &= bool(np.all(payload["fault_wct_ms"] > 0.0))
    audit["checks"]["repair_delay_changes_executed_fault_wct"] = bool(paired_ok)
    audit["checks"]["reaudit_raw_hashes_and_wct"] = bool(raw_ok)
    audit["repair_delay_paired_evidence"] = evidence
    audit["status"] = "PASS" if all(audit["checks"].values()) else "FAIL"
    audit["reaudit"] = {
        "reason": "original gate incorrectly counted distinct WCT values across three seeds instead of pairing 30/60-second cells by policy and seed",
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256(Path(__file__).resolve()),
        "summary_sha256_reverified": sha256(summary_path),
    }
    temporary = audit_path.with_name(audit_path.name + ".reaudit-tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, audit_path)
    if audit["status"] != "PASS":
        raise RuntimeError([name for name, passed in audit["checks"].items() if not passed])
    print(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
