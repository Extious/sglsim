from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(rows: list[dict[str, Any]], *, axis: str, value: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("axis")) == axis and str(row.get("value")) == value:
            return row
    raise KeyError(f"Missing summary row axis={axis!r} value={value!r}")


def _pct_reduction(old: float, new: float) -> float:
    if old <= 0:
        return 0.0
    return max((old - new) / old, 0.0) * 100.0


def _pct_change(old: float, new: float) -> float:
    if old <= 0:
        return 0.0
    return (new - old) / old * 100.0


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def build_tables(summary_path: Path) -> dict[str, Any]:
    rows = _read_json(summary_path)
    baseline = _row(rows, axis="strategy", value="baseline")
    gpu = _row(rows, axis="strategy", value="gpu_critical")

    strategy = {
        "baseline_recompute_tokens": baseline["recompute_tokens"],
        "gpu_backup_recompute_tokens": gpu["recompute_tokens"],
        "recompute_reduction_pct": _pct_reduction(
            float(baseline["recompute_tokens"]),
            float(gpu["recompute_tokens"]),
        ),
        "critical_recompute_reduction_pct": _pct_reduction(
            float(baseline["critical_recompute_tokens"]),
            float(gpu["critical_recompute_tokens"]),
        ),
        "gpu_backup_reuse_rate_pct": float(gpu["reuse_rate"]) * 100.0,
        "gpu_backup_backed_up_tokens": gpu["backed_up_tokens"],
        "mean_overhead_change_pct": _pct_change(
            float(baseline["mean_overhead_ms"]),
            float(gpu["mean_overhead_ms"]),
        ),
        "output_throughput_change_pct": _pct_change(
            float(baseline["output_throughput_tps"]),
            float(gpu["output_throughput_tps"]),
        ),
        "mean_e2e_change_pct": _pct_change(
            float(baseline["mean_e2e_ms"]),
            float(gpu["mean_e2e_ms"]),
        ),
    }

    lag_rows = [
        row
        for row in rows
        if str(row.get("axis")) == "backup_lag"
    ]
    lag_rows.sort(key=lambda r: float(r["value"]))
    lag = [
        {
            "lag_tokens": float(row["value"]),
            "reuse_rate_pct": float(row["reuse_rate"]) * 100.0,
            "recompute_tokens": row["recompute_tokens"],
            "backed_up_tokens": row["backed_up_tokens"],
        }
        for row in lag_rows
    ]

    ratio_rows = [
        row
        for row in rows
        if str(row.get("axis")) == "critical_ratio"
    ]
    ratio_rows.sort(key=lambda r: float(r["value"]))
    critical = [
        {
            "critical_ratio": float(row["value"]),
            "critical_backed_up_tokens": row["critical_backed_up_tokens"],
            "critical_recompute_tokens": row["critical_recompute_tokens"],
            "noncritical_recompute_tokens": row["noncritical_recompute_tokens"],
            "critical_mean_e2e_ms": row["critical_mean_e2e_ms"],
            "noncritical_mean_e2e_ms": row["noncritical_mean_e2e_ms"],
        }
        for row in ratio_rows
    ]

    timing_rows = [
        row
        for row in rows
        if str(row.get("axis")) == "failure_timing_s"
    ]
    timing_rows.sort(key=lambda r: float(r["value"]))
    timing = [
        {
            "failure_timing_s": float(row["value"]),
            "mean_overhead_ms": row["mean_overhead_ms"],
            "affected_requests": row["affected"],
            "recompute_tokens": row["recompute_tokens"],
            "backed_up_tokens": row["backed_up_tokens"],
        }
        for row in timing_rows
    ]

    return {
        "strategy": strategy,
        "backup_lag": lag,
        "critical_ratio": critical,
        "failure_timing": timing,
    }


def _markdown_table(tables: dict[str, Any]) -> str:
    s = tables["strategy"]
    lines = [
        "# Fault Tolerance Paper Tables",
        "",
        "## Strategy Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Recompute reduction | {_fmt(s['recompute_reduction_pct'])}% |",
        f"| Critical-agent recompute reduction | {_fmt(s['critical_recompute_reduction_pct'])}% |",
        f"| GPU backup KV reuse | {_fmt(s['gpu_backup_reuse_rate_pct'])}% |",
        f"| Output throughput change | {_fmt(s['output_throughput_change_pct'])}% |",
        f"| Mean e2e latency change | {_fmt(s['mean_e2e_change_pct'])}% |",
        f"| Mean failover overhead change | {_fmt(s['mean_overhead_change_pct'])}% |",
        "",
        "## Backup Lag",
        "",
        "| Lag tokens | KV reuse (%) | Recompute tokens | Backed-up tokens |",
        "|---:|---:|---:|---:|",
    ]
    for row in tables["backup_lag"]:
        lines.append(
            f"| {_fmt(row['lag_tokens'], 0)} | {_fmt(row['reuse_rate_pct'])} | "
            f"{_fmt(row['recompute_tokens'])} | {_fmt(row['backed_up_tokens'])} |"
        )
    lines.extend(
        [
            "",
            "## Critical-Agent Coverage",
            "",
            "| Critical ratio | Critical reused | Critical recompute | Non-critical recompute |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in tables["critical_ratio"]:
        lines.append(
            f"| {_fmt(row['critical_ratio'], 2)} | "
            f"{_fmt(row['critical_backed_up_tokens'])} | "
            f"{_fmt(row['critical_recompute_tokens'])} | "
            f"{_fmt(row['noncritical_recompute_tokens'])} |"
        )
    return "\n".join(lines) + "\n"


def _latex_table(tables: dict[str, Any]) -> str:
    s = tables["strategy"]
    return "\n".join(
        [
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Metric & Value \\",
            r"\midrule",
            f"Recompute reduction & {_fmt(s['recompute_reduction_pct'])}\\% \\\\",
            f"Critical-agent recompute reduction & {_fmt(s['critical_recompute_reduction_pct'])}\\% \\\\",
            f"GPU backup KV reuse & {_fmt(s['gpu_backup_reuse_rate_pct'])}\\% \\\\",
            f"Output throughput change & {_fmt(s['output_throughput_change_pct'])}\\% \\\\",
            f"Mean e2e latency change & {_fmt(s['mean_e2e_change_pct'])}\\% \\\\",
            f"Mean failover overhead change & {_fmt(s['mean_overhead_change_pct'])}\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def write_tables(summary_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(summary_path)
    paths = {
        "json": output_dir / "fault_tolerance_paper_tables.json",
        "markdown": output_dir / "fault_tolerance_paper_tables.md",
        "latex": output_dir / "fault_tolerance_paper_tables.tex",
    }
    paths["json"].write_text(json.dumps(tables, indent=2), encoding="utf-8")
    paths["markdown"].write_text(_markdown_table(tables), encoding="utf-8")
    paths["latex"].write_text(_latex_table(tables), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready FT tables")
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/synthesis_requests/results/figures"),
    )
    args = parser.parse_args(argv)
    paths = write_tables(args.summary, args.output_dir)
    print(paths["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
