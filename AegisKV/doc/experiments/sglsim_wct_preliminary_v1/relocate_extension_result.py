#!/usr/bin/env python3
"""Make raw paths relocation-safe and re-audit an extension result directory."""

from __future__ import annotations

import argparse
import csv
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


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".relocation-tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.result_dir
    summary_path = base / "summary.json"
    seed_path = base / "summary.csv"
    audit_path = base / "audit.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    pre_summary_sha = sha256(summary_path)
    for cell in summary["cells"]:
        raw = base / "raw" / Path(cell["raw_npz"]).name
        if not raw.is_file() or sha256(raw) != cell["raw_npz_sha256"]:
            raise ValueError(f"raw hash mismatch during relocation: {raw}")
        with np.load(raw, allow_pickle=False) as payload:
            if np.any(payload["base_wct_ms"] <= 0) or np.any(payload["fault_wct_ms"] <= 0):
                raise ValueError(f"invalid WCT during relocation: {raw}")
        cell["raw_npz"] = str(Path("raw") / raw.name)
    atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with seed_path.open(newline="", encoding="utf-8") as stream:
        seed_rows = list(csv.DictReader(stream))
        fields = list(stream and seed_rows[0].keys())
    for row in seed_rows:
        row["raw_npz"] = str(Path("raw") / Path(row["raw_npz"]).name)
    temporary_csv = seed_path.with_name(seed_path.name + ".relocation-tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(seed_rows)
    os.replace(temporary_csv, seed_path)

    audit["checks"]["relocatable_raw_paths_resolve"] = all(
        not Path(cell["raw_npz"]).is_absolute()
        and (base / cell["raw_npz"]).is_file()
        and sha256(base / cell["raw_npz"]) == cell["raw_npz_sha256"]
        for cell in summary["cells"]
    )
    audit["summary_sha256"] = sha256(summary_path)
    audit["seed_summary_sha256"] = sha256(seed_path)
    audit["status"] = "PASS" if all(audit["checks"].values()) else "FAIL"
    audit["relocation"] = {
        "kind": "raw_paths_made_relative_without_changing_raw_arrays",
        "pre_relocation_summary_sha256": pre_summary_sha,
        "auditor": str(Path(__file__).resolve()),
        "auditor_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_text(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError([name for name, passed in audit["checks"].items() if not passed])
    print(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
