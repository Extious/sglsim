#!/usr/bin/env python3
"""Validate the migrated AegisKV supporting figure bundle.

This is an artifact-integrity check. Scientific claims and final experiment
requirements are defined by the current paper outline, not by legacy gate
labels embedded in migrated results.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SUPPORT_FIGURES = SCRIPT_DIR.parent
PACKAGE_ROOT = SUPPORT_FIGURES.parent
PAPER_FIGURES = PACKAGE_ROOT / "Figures"
MEASURED = SUPPORT_FIGURES / "data" / "measured"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _valid_png(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 24:
        return False
    with path.open("rb") as handle:
        header = handle.read(24)
    return header.startswith(b"\x89PNG\r\n\x1a\n") and header[12:16] == b"IHDR"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []

    status_path = SUPPORT_FIGURES / "evidence_status.json"
    manifest_path = SUPPORT_FIGURES / "manifest.json"
    if not status_path.is_file():
        problems.append(f"missing {status_path}")
        status: dict[str, Any] = {}
    else:
        value = _read_json(status_path)
        status = value if isinstance(value, dict) else {}
        if not status:
            problems.append("evidence_status.json must contain an object")

    if not manifest_path.is_file():
        problems.append(f"missing {manifest_path}")
        manifest: list[Any] = []
    else:
        value = _read_json(manifest_path)
        manifest = value if isinstance(value, list) else []
        if not isinstance(value, list):
            problems.append("manifest.json must contain a list")

    rows = status.get("figures") if isinstance(status.get("figures"), list) else []
    stems: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            problems.append("evidence_status.json contains a non-object figure")
            continue
        stem = str(row.get("stem") or "")
        if not stem:
            problems.append("figure entry without stem")
            continue
        if stem in stems:
            problems.append(f"duplicate figure stem: {stem}")
        stems.add(stem)

        png_rel = str(row.get("png") or "")
        if png_rel:
            png = SUPPORT_FIGURES / png_rel
            if not _valid_png(png):
                problems.append(f"{stem}: missing or invalid PNG {png_rel}")

        csv_name = str(row.get("measured_csv") or "")
        if csv_name:
            measured = MEASURED / csv_name
            if not measured.is_file() or measured.stat().st_size == 0:
                problems.append(f"{stem}: missing measured CSV {csv_name}")
            else:
                actual_rows = _csv_rows(measured)
                recorded_rows = row.get("row_count")
                if recorded_rows not in (None, "") and int(recorded_rows) != actual_rows:
                    warnings.append(
                        f"{stem}: recorded row_count={recorded_rows}, "
                        f"current CSV rows={actual_rows}"
                    )

    manifest_stems = {
        str(row.get("stem") or "")
        for row in manifest
        if isinstance(row, dict) and row.get("stem")
    }
    unknown_manifest_stems = sorted(manifest_stems - stems)
    if unknown_manifest_stems:
        problems.append(f"manifest has unknown stems: {unknown_manifest_stems}")

    canonical_hashes = {
        _sha256(path): path.name
        for path in PAPER_FIGURES.glob("*.png")
        if path.is_file()
    }
    duplicates = [
        (path.name, canonical_hashes[_sha256(path)])
        for path in (SUPPORT_FIGURES / "png").glob("*.png")
        if path.is_file() and _sha256(path) in canonical_hashes
    ]
    if duplicates:
        problems.append(f"duplicate PNGs across Figures/ and figures/png/: {duplicates}")

    return {
        "status": "pass" if not problems else "fail",
        "figures": len(stems),
        "manifest_entries": len(manifest_stems),
        "problems": problems,
        "warnings": warnings,
    }


def main() -> int:
    payload = validate()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
