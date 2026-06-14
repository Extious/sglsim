#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def mean_for(rows: list[dict[str, str]], **where: str) -> dict[str, float]:
    selected = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in where.items())
    ]
    if not selected:
        return {}
    metrics = (
        "workflow_completion_time_s",
        "total_recomputed_tokens",
        "backed_up_tokens",
        "gpu_replica_reuse_tokens",
        "suffix_replay_tokens",
        "kv_reuse_rate",
        "critical_segment_survival_rate",
        "backup_overhead_s",
    )
    return {metric: mean(f(row, metric) for row in selected) for metric in metrics}


def median_wct(workflows: list[dict[str, str]], *, impacted_only: bool = True, **where: str) -> float:
    selected = [
        row
        for row in workflows
        if all(str(row.get(key)) == str(value) for key, value in where.items())
        and (not impacted_only or row.get("impacted_by_failure") == "1")
    ]
    if not selected:
        return 0.0
    return float(median(f(row, "wct_s") for row in selected))


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def build_tables(
    rows: list[dict[str, str]],
    paths: list[dict[str, str]],
    workflows: list[dict[str, str]],
) -> dict:
    base = mean_for(rows, experiment="strategy", strategy="baseline")
    naive = mean_for(rows, experiment="strategy", strategy="naive_backup")
    aegis = mean_for(rows, experiment="strategy", strategy="aegiskv")

    aegis_recompute_reduction = -pct_change(
        aegis["total_recomputed_tokens"], base["total_recomputed_tokens"]
    )
    naive_recompute_reduction = -pct_change(
        naive["total_recomputed_tokens"], base["total_recomputed_tokens"]
    )
    backup_reduction = -pct_change(aegis["backed_up_tokens"], naive["backed_up_tokens"])
    recompute_gap = pct_change(
        aegis["total_recomputed_tokens"], naive["total_recomputed_tokens"]
    )
    wct_change = pct_change(
        aegis["workflow_completion_time_s"], base["workflow_completion_time_s"]
    )
    wct_median_base = median_wct(
        workflows,
        experiment="strategy",
        strategy="baseline",
        failure_site="S5_ToolBlocked",
    )
    wct_median_aegis = median_wct(
        workflows,
        experiment="strategy",
        strategy="aegiskv",
        failure_site="S5_ToolBlocked",
    )
    wct_median_naive = median_wct(
        workflows,
        experiment="strategy",
        strategy="naive_backup",
        failure_site="S5_ToolBlocked",
    )

    path_counts: dict[str, int] = defaultdict(int)
    for row in paths:
        if row["experiment"] == "strategy" and row["strategy"] == "aegiskv":
            path_counts[row["recovery_path"]] += 1

    return {
        "strategy": {
            "baseline_recompute_tokens": base["total_recomputed_tokens"],
            "naive_recompute_reduction_pct": naive_recompute_reduction,
            "aegiskv_recompute_reduction_pct": aegis_recompute_reduction,
            "naive_backup_tokens": naive["backed_up_tokens"],
            "aegiskv_backup_tokens": aegis["backed_up_tokens"],
            "aegiskv_backup_reduction_vs_naive_pct": backup_reduction,
            "aegiskv_vs_naive_recompute_gap_pct": recompute_gap,
            "aegiskv_kv_reuse_rate_pct": aegis["kv_reuse_rate"] * 100.0,
            "aegiskv_critical_survival_pct": aegis["critical_segment_survival_rate"] * 100.0,
            "aegiskv_wct_change_pct": wct_change,
            "wct_median_baseline_s": wct_median_base,
            "wct_median_naive_s": wct_median_naive,
            "wct_median_aegiskv_s": wct_median_aegis,
            "aegiskv_suffix_replay_tokens": aegis["suffix_replay_tokens"],
        },
        "recovery_paths": dict(path_counts),
    }


def write_markdown(path: Path, tables: dict) -> None:
    s = tables["strategy"]
    lines = [
        "# Multi-Agent AegisKV Paper Tables",
        "",
        "## Strategy Summary (S5 critical failure)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline recompute tokens | {s['baseline_recompute_tokens']:.1f} |",
        f"| NaiveBackup recompute reduction | {s['naive_recompute_reduction_pct']:.1f}% |",
        f"| AegisKV recompute reduction | {s['aegiskv_recompute_reduction_pct']:.1f}% |",
        f"| NaiveBackup backed-up tokens | {s['naive_backup_tokens']:.1f} |",
        f"| AegisKV backed-up tokens | {s['aegiskv_backup_tokens']:.1f} |",
        f"| AegisKV backup reduction vs Naive | {s['aegiskv_backup_reduction_vs_naive_pct']:.1f}% |",
        f"| AegisKV vs Naive recompute gap | {s['aegiskv_vs_naive_recompute_gap_pct']:.1f}% |",
        f"| WCT median (Baseline) | {s['wct_median_baseline_s']:.1f}s |",
        f"| WCT median (NaiveBackup) | {s['wct_median_naive_s']:.1f}s |",
        f"| WCT median (AegisKV) | {s['wct_median_aegiskv_s']:.1f}s |",
        f"| AegisKV KV reuse rate | {s['aegiskv_kv_reuse_rate_pct']:.1f}% |",
        "",
        "## AegisKV Recovery Paths",
        "",
        "| Path | Count |",
        "|---|---:|",
    ]
    for key, value in sorted(tables["recovery_paths"].items()):
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(path: Path, tables: dict) -> None:
    s = tables["strategy"]
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"AegisKV recompute reduction & {s['aegiskv_recompute_reduction_pct']:.1f}\\% \\\\",
        f"AegisKV backup reduction vs Naive & {s['aegiskv_backup_reduction_vs_naive_pct']:.1f}\\% \\\\",
        f"AegisKV vs Naive recompute gap & {s['aegiskv_vs_naive_recompute_gap_pct']:.1f}\\% \\\\",
        f"WCT median AegisKV & {s['wct_median_aegiskv_s']:.1f}s \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build multi-agent AegisKV paper tables")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_csv(args.results_dir / "runs.csv")
    paths = read_csv(args.results_dir / "recovery_paths.csv")
    workflows = read_csv(args.results_dir / "workflows.csv")
    tables = build_tables(rows, paths, workflows)
    (args.results_dir / "paper_tables.json").write_text(
        json.dumps(tables, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.results_dir / "paper_tables.md", tables)
    write_tex(args.results_dir / "paper_tables.tex", tables)
    print(json.dumps(tables, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
