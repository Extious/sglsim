#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


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
        "gpu_replica_reuse_tokens",
        "suffix_replay_tokens",
        "kv_reuse_rate",
        "critical_segment_survival_rate",
    )
    return {metric: mean(f(row, metric) for row in selected) for metric in metrics}


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def build_tables(rows: list[dict[str, str]], paths: list[dict[str, str]]) -> dict:
    base = mean_for(rows, experiment="strategy", strategy="baseline")
    agent = mean_for(rows, experiment="strategy", strategy="gpu_backup_agent")
    aegis = mean_for(rows, experiment="strategy", strategy="aegiskv_segment")
    recompute_reduction = -pct_change(
        aegis["total_recomputed_tokens"], base["total_recomputed_tokens"]
    )
    agent_reduction = -pct_change(
        agent["total_recomputed_tokens"], base["total_recomputed_tokens"]
    )
    wct_change = pct_change(
        aegis["workflow_completion_time_s"], base["workflow_completion_time_s"]
    )
    path_counts: dict[str, int] = defaultdict(int)
    for row in paths:
        if row["experiment"] == "strategy" and row["strategy"] == "aegiskv_segment":
            path_counts[row["recovery_path"]] += 1
    return {
        "strategy": {
            "baseline_recompute_tokens": base["total_recomputed_tokens"],
            "agent_recompute_reduction_pct": agent_reduction,
            "aegiskv_recompute_reduction_pct": recompute_reduction,
            "aegiskv_kv_reuse_rate_pct": aegis["kv_reuse_rate"] * 100.0,
            "aegiskv_critical_survival_pct": aegis["critical_segment_survival_rate"] * 100.0,
            "aegiskv_wct_change_pct": wct_change,
            "aegiskv_suffix_replay_tokens": aegis["suffix_replay_tokens"],
        },
        "recovery_paths": dict(path_counts),
    }


def write_markdown(path: Path, tables: dict) -> None:
    s = tables["strategy"]
    lines = [
        "# Multi-Agent AegisKV Paper Tables",
        "",
        "## Strategy Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline recompute tokens | {s['baseline_recompute_tokens']:.1f} |",
        f"| Agent GPU backup recompute reduction | {s['agent_recompute_reduction_pct']:.1f}% |",
        f"| AegisKV recompute reduction | {s['aegiskv_recompute_reduction_pct']:.1f}% |",
        f"| AegisKV KV reuse rate | {s['aegiskv_kv_reuse_rate_pct']:.1f}% |",
        f"| Critical segment survival | {s['aegiskv_critical_survival_pct']:.1f}% |",
        f"| Workflow completion time change | {s['aegiskv_wct_change_pct']:.1f}% |",
        f"| Suffix replay tokens | {s['aegiskv_suffix_replay_tokens']:.1f} |",
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
        f"Critical segment survival & {s['aegiskv_critical_survival_pct']:.1f}\\% \\\\",
        f"AegisKV KV reuse rate & {s['aegiskv_kv_reuse_rate_pct']:.1f}\\% \\\\",
        f"WCT change & {s['aegiskv_wct_change_pct']:.1f}\\% \\\\",
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
    tables = build_tables(rows, paths)
    (args.results_dir / "paper_tables.json").write_text(
        json.dumps(tables, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.results_dir / "paper_tables.md", tables)
    write_tex(args.results_dir / "paper_tables.tex", tables)
    print(json.dumps(tables, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
