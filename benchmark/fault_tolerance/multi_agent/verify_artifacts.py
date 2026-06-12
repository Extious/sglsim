#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


EXPECTED_FIGURES = (
    "fig_strategy_recovery",
    "fig_wct_failure_site",
    "fig_budget_efficiency",
    "fig_backup_lag",
    "fig_recovery_path_distribution",
    "fig_scale",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(rows: list[dict[str, str]], experiment: str, strategy: str, name: str) -> float:
    values = [
        float(row[name])
        for row in rows
        if row["experiment"] == experiment and row["strategy"] == strategy
    ]
    return mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify multi-agent AegisKV artifacts")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    args = parser.parse_args()
    failures: list[str] = []
    required = ["manifest.json", "summary.json", "runs.csv", "segments.csv", "recovery_paths.csv"]
    if args.profile == "paper":
        required += ["paper_tables.json", "paper_tables.md", "paper_tables.tex"]
    for name in required:
        path = args.results_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty {name}")
    if failures:
        payload = {"passed": False, "failures": failures}
        (args.results_dir / "artifact_verification.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
        return 1

    rows = read_csv(args.results_dir / "runs.csv")
    paths = read_csv(args.results_dir / "recovery_paths.csv")
    manifest = json.loads((args.results_dir / "manifest.json").read_text(encoding="utf-8"))
    if any(entry.get("status") != "completed" for entry in manifest):
        failures.append("manifest has non-completed entries")
    if args.profile == "paper":
        point_counts: dict[tuple[str, str, str, str, str, str], int] = {}
        for row in rows:
            key = (
                row["experiment"],
                row["strategy"],
                row["failure_site"],
                row["backup_budget_ratio"],
                row["peer_backup_lag_tokens"],
                row["num_workflows"],
            )
            point_counts[key] = point_counts.get(key, 0) + 1
        short = {key: count for key, count in point_counts.items() if count < 5}
        if short:
            failures.append(f"points with <5 repeats: {len(short)}")

    base_recompute = metric(rows, "strategy", "baseline", "total_recomputed_tokens")
    aegis_recompute = metric(rows, "strategy", "aegiskv_segment", "total_recomputed_tokens")
    agent_recompute = metric(rows, "strategy", "gpu_backup_agent", "total_recomputed_tokens")
    if base_recompute <= 0:
        failures.append("baseline recompute is zero")
    reduction = (base_recompute - aegis_recompute) / base_recompute if base_recompute else 0.0
    if reduction < (0.05 if args.profile == "smoke" else 0.30):
        failures.append(f"aegiskv recompute reduction too low: {reduction:.3f}")
    if args.profile == "paper" and aegis_recompute > agent_recompute:
        failures.append("aegiskv recompute worse than agent backup")
    survival = metric(rows, "strategy", "aegiskv_segment", "critical_segment_survival_rate")
    if survival < (0.30 if args.profile == "smoke" else 0.70):
        failures.append(f"critical segment survival too low: {survival:.3f}")
    if not any(row["recovery_path"] != "no_valid_checkpoint" for row in paths):
        failures.append("all recoveries degraded to no_valid_checkpoint")
    if args.profile == "paper":
        for stem in EXPECTED_FIGURES:
            for suffix in (".png", ".pdf"):
                path = args.results_dir / f"{stem}{suffix}"
                if not path.is_file() or path.stat().st_size == 0:
                    failures.append(f"missing figure {path.name}")

    payload = {
        "profile": args.profile,
        "passed": not failures,
        "failures": failures,
        "strategy_summary": {
            "baseline_recompute_tokens": base_recompute,
            "agent_recompute_tokens": agent_recompute,
            "aegiskv_recompute_tokens": aegis_recompute,
            "aegiskv_recompute_reduction_pct": reduction * 100.0,
            "aegiskv_critical_survival_pct": survival * 100.0,
        },
    }
    (args.results_dir / "artifact_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
