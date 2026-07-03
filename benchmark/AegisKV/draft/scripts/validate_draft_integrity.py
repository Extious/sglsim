#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = ROOT / "figures"


def repo_root() -> Path:
    for path in (ROOT, *ROOT.parents):
        if (path / ".git").exists():
            return path
    return ROOT.parents[2]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root(), text=True, capture_output=True)


def check_links() -> list[str]:
    problems: list[str] = []
    md_files = [
        ROOT / "README.md",
        PAPER / "README.md",
        PAPER / "current_results_overview.md",
        PAPER / "readiness_audit.md",
        FIGURES / "README.md",
        FIGURES / "gallery.md",
        ROOT / "experiments" / "README.md",
        ROOT / "related_works" / "README.md",
        *sorted((PAPER / "sections").glob("*.md")),
    ]
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in md_files:
        if not path.exists():
            problems.append(f"missing markdown file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            target = match.group(1)
            if "://" in target or target.startswith("#"):
                continue
            target_path = (path.parent / target.split("#", 1)[0]).resolve()
            if not target_path.exists():
                problems.append(f"broken link in {path.relative_to(ROOT)}: {target}")
    return problems


def check_figures() -> list[str]:
    problems: list[str] = []
    manifest_path = FIGURES / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path.relative_to(ROOT)}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        problems.append("figure manifest has no entries")
    for entry in manifest:
        stem = entry["stem"]
        for kind in ("png",):
            rel = entry.get("outputs", {}).get(kind)
            if not rel:
                problems.append(f"{stem}: missing {kind} output in manifest")
                continue
            p = FIGURES / rel
            if not p.exists() or p.stat().st_size == 0:
                problems.append(f"{stem}: missing/empty {kind} output {rel}")
        if entry.get("kind") == "mechanism_diagram":
            outputs = entry.get("outputs", {})
            source_rel = outputs.get("drawio") or outputs.get("generated_prompt")
            source_kind = "drawio/generated prompt"
            if not source_rel:
                problems.append(f"{stem}: missing {source_kind} source in manifest")
            else:
                p = FIGURES / source_rel
                if not p.exists() or p.stat().st_size == 0:
                    problems.append(f"{stem}: missing/empty {source_kind} source {source_rel}")
        if entry.get("kind") != "mechanism_diagram":
            measured = FIGURES / "data" / "measured" / f"{stem}.csv"
            if not measured.exists() or measured.stat().st_size == 0:
                problems.append(f"{stem}: missing measured/preliminary CSV")
    return problems


def check_final_evidence_matrix() -> list[str]:
    problems: list[str] = []
    manifest_path = FIGURES / "manifest.json"
    matrix_path = ROOT / "experiments" / "final_evidence_matrix.csv"
    if not manifest_path.exists() or not matrix_path.exists():
        if not matrix_path.exists():
            problems.append(f"missing {matrix_path.relative_to(ROOT)}")
        return problems

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_stems = {
        str(entry["stem"]): f"{entry['stem']}.csv"
        for entry in manifest
        if entry.get("kind") != "mechanism_diagram"
    }

    with matrix_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for index, row in enumerate(rows, start=2):
        if None in row:
            problems.append(
                "final_evidence_matrix.csv row "
                f"{index}: extra CSV fields; quote fields that contain commas"
            )

    required_columns = {
        "figure",
        "figure_stem",
        "measured_csv",
        "primary_final_source",
        "current_evidence",
        "sglang_small_requirement",
        "sglsim_large_requirement",
        "final_gate",
    }
    fields = set(rows[0].keys()) if rows else set()
    missing_columns = sorted(required_columns - fields)
    if missing_columns:
        problems.append(
            "final_evidence_matrix.csv missing columns: "
            + ", ".join(missing_columns)
        )

    matrix_stems = {str(row.get("figure_stem", "")).strip() for row in rows}
    missing_stems = sorted(set(result_stems) - matrix_stems)
    extra_stems = sorted(matrix_stems - set(result_stems))
    if missing_stems:
        problems.append(
            "final_evidence_matrix.csv missing result figures: "
            + ", ".join(missing_stems)
        )
    if extra_stems:
        problems.append(
            "final_evidence_matrix.csv has unknown figure stems: "
            + ", ".join(extra_stems)
        )

    for row in rows:
        stem = str(row.get("figure_stem", "")).strip()
        expected_csv = result_stems.get(stem)
        if expected_csv and str(row.get("measured_csv", "")).strip() != expected_csv:
            problems.append(
                f"final_evidence_matrix.csv {stem}: measured_csv should be {expected_csv}"
            )

    return problems


def check_evidence(strict_final: bool) -> list[str]:
    args = [
        sys.executable,
        str(FIGURES / "scripts" / "validate_figure_evidence.py"),
    ]
    if strict_final:
        args.append("--strict-final")
    result = run(args)
    if result.returncode != 0:
        return [
            "figure evidence validation failed",
            result.stdout.strip(),
            result.stderr.strip(),
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-final", action="store_true")
    args = parser.parse_args()

    problems: list[str] = []
    problems.extend(check_links())
    problems.extend(check_figures())
    problems.extend(check_final_evidence_matrix())
    problems.extend(check_evidence(args.strict_final))

    if problems:
        print("draft_integrity FAIL")
        for problem in problems:
            if problem:
                print(f"- {problem}")
        return 1
    print("draft_integrity PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
