#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


EXPECTED_FIGURES = (
    "fig_wct_cdf",
    "fig_agent_timeline",
    "fig_agent_timeline_compare",
    "fig_agent_duration_violin",
    "fig_strategy_recovery",
    "fig_backup_cost",
    "fig_recovery_vs_cost",
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


def median_wct(
    workflows: list[dict[str, str]],
    strategy: str,
    *,
    impacted_only: bool = True,
) -> float:
    values = [
        float(row["wct_s"])
        for row in workflows
        if row["experiment"] == "strategy"
        and row["strategy"] == strategy
        and row["failure_site"] == "S5_ToolBlocked"
        and (not impacted_only or row.get("impacted_by_failure") == "1")
    ]
    return float(median(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify multi-agent AegisKV artifacts")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    args = parser.parse_args()
    failures: list[str] = []
    required = [
        "manifest.json",
        "summary.json",
        "runs.csv",
        "segments.csv",
        "recovery_paths.csv",
        "workflows.csv",
        "agent_timelines.csv",
    ]
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
    workflows = read_csv(args.results_dir / "workflows.csv")
    manifest = json.loads((args.results_dir / "manifest.json").read_text(encoding="utf-8"))
    if any(entry.get("status") != "completed" for entry in manifest):
        failures.append("manifest has non-completed entries")

    if args.profile == "paper":
        point_counts: dict[tuple[str, str, str, str, str], int] = {}
        for row in rows:
            key = (
                row["experiment"],
                row["strategy"],
                row["failure_site"],
                row["peer_backup_lag_tokens"],
                row["num_workflows"],
            )
            point_counts[key] = point_counts.get(key, 0) + 1
        short = {key: count for key, count in point_counts.items() if count < 5}
        if short:
            failures.append(f"points with <5 repeats: {len(short)}")

    base_recompute = metric(rows, "strategy", "baseline", "total_recomputed_tokens")
    aegis_recompute = metric(rows, "strategy", "aegiskv", "total_recomputed_tokens")
    naive_recompute = metric(rows, "strategy", "naive_backup", "total_recomputed_tokens")
    aegis_backup = metric(rows, "strategy", "aegiskv", "backed_up_tokens")
    naive_backup = metric(rows, "strategy", "naive_backup", "backed_up_tokens")

    if base_recompute <= 0:
        failures.append("baseline recompute is zero")
    reduction = (base_recompute - aegis_recompute) / base_recompute if base_recompute else 0.0
    if reduction < (0.05 if args.profile == "smoke" else 0.30):
        failures.append(f"aegiskv recompute reduction too low: {reduction:.3f}")
    if naive_backup > 0 and aegis_backup >= naive_backup * (1.0 if args.profile == "smoke" else 0.78):
        failures.append(
            f"aegiskv backup not sufficiently lower than naive: {aegis_backup:.0f} vs {naive_backup:.0f}"
        )
    if naive_recompute > 0 and aegis_recompute > naive_recompute * (1.25 if args.profile == "smoke" else 1.15):
        failures.append(
            f"aegiskv recompute too far above naive on critical failure: {aegis_recompute:.0f} vs {naive_recompute:.0f}"
        )

    wct_base = median_wct(workflows, "baseline")
    wct_aegis = median_wct(workflows, "aegiskv")
    wct_naive = median_wct(workflows, "naive_backup")
    wct_slack_vs_baseline = 1.05 if args.profile == "smoke" else 1.02
    wct_slack_vs_naive = 1.10 if args.profile == "smoke" else 1.05
    if wct_base > 0 and wct_aegis > wct_base * wct_slack_vs_baseline:
        failures.append(
            f"aegiskv impacted WCT median too high vs baseline: {wct_aegis:.2f} vs {wct_base:.2f}"
        )
    if wct_naive > 0 and wct_aegis > wct_naive * wct_slack_vs_naive:
        failures.append(
            f"aegiskv impacted WCT median too far above naive: {wct_aegis:.2f} vs {wct_naive:.2f}"
        )

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
            "naive_recompute_tokens": naive_recompute,
            "aegiskv_recompute_tokens": aegis_recompute,
            "aegiskv_recompute_reduction_pct": reduction * 100.0,
            "naive_backup_tokens": naive_backup,
            "aegiskv_backup_tokens": aegis_backup,
            "wct_median_baseline_s": wct_base,
            "wct_median_naive_s": wct_naive,
            "wct_median_aegiskv_s": wct_aegis,
        },
    }
    (args.results_dir / "artifact_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
