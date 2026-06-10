from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _truthy(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _summarize(entry: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(entry["run_dir"])
    metrics = _read_json(run_dir / "metrics.json")
    rows = _read_rows(run_dir / "request_details.csv")
    affected = [r for r in rows if _truthy(r.get("is_failover_retried", ""))]
    critical = [r for r in affected if _truthy(r.get("critical_agent", ""))]
    noncritical = [r for r in affected if not _truthy(r.get("critical_agent", ""))]
    all_critical = [r for r in rows if _truthy(r.get("critical_agent", ""))]
    all_noncritical = [r for r in rows if not _truthy(r.get("critical_agent", ""))]

    def sum_int(items: list[dict[str, str]], key: str) -> int:
        return sum(int(float(r.get(key, 0) or 0)) for r in items)

    def mean_s(items: list[dict[str, str]], key: str) -> float:
        vals = [float(r.get(key, 0) or 0) for r in items]
        return sum(vals) / len(vals) if vals else 0.0

    def mean_ms(items: list[dict[str, str]], key: str) -> float:
        return mean_s(items, key) * 1000.0

    def pct_ms(items: list[dict[str, str]], key: str, pct: float) -> float:
        vals = sorted(float(r.get(key, 0) or 0) for r in items)
        if not vals:
            return 0.0
        return vals[int(pct * (len(vals) - 1))] * 1000.0

    recompute = sum_int(affected, "recompute_tokens")
    backed = sum_int(affected, "pre_failover_backed_up_tokens")
    total = recompute + backed
    reuse_rate = backed / total if total > 0 else 0.0
    duration_s = float(metrics.get("duration", 0.0) or 0.0)
    total_output = float(metrics.get("total_output", 0.0) or 0.0)
    completed = int(metrics.get("completed", 0))
    return {
        **entry,
        "completed": completed,
        "failed": int(metrics.get("failed", 0)),
        "duration_s": duration_s,
        "affected": len(affected),
        "mean_e2e_ms": float(metrics.get("mean_e2e_latency_ms", 0.0)),
        "request_throughput_rps": completed / duration_s if duration_s > 0 else 0.0,
        "output_throughput_tps": total_output / duration_s if duration_s > 0 else 0.0,
        "recompute_tokens": recompute,
        "backed_up_tokens": backed,
        "reuse_rate": reuse_rate,
        "mean_overhead_ms": mean_s(affected, "failover_overhead_s") * 1000.0,
        "critical_mean_e2e_ms": mean_ms(all_critical, "e2e_latency_s"),
        "noncritical_mean_e2e_ms": mean_ms(all_noncritical, "e2e_latency_s"),
        "critical_p90_e2e_ms": pct_ms(all_critical, "e2e_latency_s", 0.9),
        "noncritical_p90_e2e_ms": pct_ms(all_noncritical, "e2e_latency_s", 0.9),
        "critical_recompute_tokens": sum_int(critical, "recompute_tokens"),
        "noncritical_recompute_tokens": sum_int(noncritical, "recompute_tokens"),
        "critical_backed_up_tokens": sum_int(
            critical, "pre_failover_backed_up_tokens"
        ),
        "noncritical_backed_up_tokens": sum_int(
            noncritical, "pre_failover_backed_up_tokens"
        ),
    }


def _num(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["axis"]), str(row["value"]), str(row["name"]))
        grouped.setdefault(key, []).append(row)

    metrics = [
        "completed",
        "failed",
        "duration_s",
        "affected",
        "mean_e2e_ms",
        "request_throughput_rps",
        "output_throughput_tps",
        "recompute_tokens",
        "backed_up_tokens",
        "reuse_rate",
        "mean_overhead_ms",
        "critical_mean_e2e_ms",
        "noncritical_mean_e2e_ms",
        "critical_p90_e2e_ms",
        "noncritical_p90_e2e_ms",
        "critical_recompute_tokens",
        "noncritical_recompute_tokens",
        "critical_backed_up_tokens",
        "noncritical_backed_up_tokens",
    ]
    out: list[dict[str, Any]] = []
    for (axis, value, name), items in grouped.items():
        row: dict[str, Any] = {
            "name": name,
            "axis": axis,
            "value": value,
            "n": len(items),
            "seeds": [int(x.get("seed", 0)) for x in items if "seed" in x],
            "run_dirs": [str(x.get("run_dir", "")) for x in items],
        }
        for metric in metrics:
            vals = [float(x.get(metric, 0.0) or 0.0) for x in items]
            row[metric] = _mean(vals)
            row[f"{metric}_stderr"] = _stderr(vals)
        out.append(row)
    out.sort(key=lambda r: (str(r["axis"]), _num(str(r["value"])), str(r["name"])))
    return out


def plot(manifest_path: Path, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    entries = _read_json(manifest_path)
    summaries = [_summarize(entry) for entry in entries]
    aggregates = _aggregate(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fault_tolerance_sweep_raw_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    (output_dir / "fault_tolerance_sweep_summary.json").write_text(
        json.dumps(aggregates, indent=2),
        encoding="utf-8",
    )

    by_axis: dict[str, list[dict[str, Any]]] = {}
    for row in aggregates:
        by_axis.setdefault(str(row["axis"]), []).append(row)
    for rows in by_axis.values():
        rows.sort(key=lambda r: _num(str(r["value"])))

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    ax = axes[0][0]
    strategy = by_axis.get("strategy", [])
    labels = [str(r["value"]) for r in strategy]
    recompute = [r["recompute_tokens"] for r in strategy]
    backed = [r["backed_up_tokens"] for r in strategy]
    totals = [r + b for r, b in zip(recompute, backed, strict=False)]
    total_err = [
        math.sqrt(
            r.get("recompute_tokens_stderr", 0.0) ** 2
            + r.get("backed_up_tokens_stderr", 0.0) ** 2
        )
        for r in strategy
    ]
    ax.bar(labels, recompute, color="#4c78a8", label="recompute")
    ax.bar(labels, backed, bottom=recompute, color="#59a14f", label="reused")
    if strategy:
        ax.errorbar(labels, totals, yerr=total_err, fmt="none", ecolor="#333333", capsize=3, linewidth=0.8)
    ax.set_title("(a) Strategy failover tokens")
    ax.set_ylabel("Tokens")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0][1]
    lag = by_axis.get("backup_lag", [])
    ax.errorbar(
        [_num(str(r["value"])) for r in lag],
        [r["reuse_rate"] * 100 for r in lag],
        yerr=[r.get("reuse_rate_stderr", 0.0) * 100 for r in lag],
        marker="o",
        color="#59a14f",
        capsize=3,
    )
    ax.set_title("(b) Backup lag sensitivity")
    ax.set_xlabel("Lag tokens")
    ax.set_ylabel("KV reuse (%)")

    ax = axes[1][0]
    ratio = by_axis.get("critical_ratio", [])
    x_ratio = [_num(str(r["value"])) for r in ratio]
    ax.errorbar(
        x_ratio,
        [r["critical_backed_up_tokens"] for r in ratio],
        yerr=[r.get("critical_backed_up_tokens_stderr", 0.0) for r in ratio],
        marker="o",
        color="#59a14f",
        capsize=3,
        label="critical reused",
    )
    ax.errorbar(
        x_ratio,
        [r["noncritical_recompute_tokens"] for r in ratio],
        yerr=[r.get("noncritical_recompute_tokens_stderr", 0.0) for r in ratio],
        marker="s",
        color="#9c755f",
        capsize=3,
        label="non-critical recompute",
    )
    ax.set_title("(c) Critical-agent coverage")
    ax.set_xlabel("Critical agent ratio")
    ax.set_ylabel("Tokens")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    timing = by_axis.get("failure_timing_s", [])
    ax.errorbar(
        [_num(str(r["value"])) for r in timing],
        [r["mean_overhead_ms"] for r in timing],
        yerr=[r.get("mean_overhead_ms_stderr", 0.0) for r in timing],
        marker="o",
        color="#f28e2b",
        capsize=3,
    )
    ax.set_title("(d) Failure timing")
    ax.set_xlabel("Failure injection time (s)")
    ax.set_ylabel("Mean overhead (ms)")

    for ax in axes.ravel():
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)

    fig.tight_layout()
    png = output_dir / "fault_tolerance_sweep_hpca.png"
    pdf = output_dir / "fault_tolerance_sweep_hpca.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    fig2, axes2 = plt.subplots(2, 2, figsize=(9.5, 6.2))
    ax = axes2[0][0]
    ax.bar(
        labels,
        [r["output_throughput_tps"] for r in strategy],
        yerr=[r.get("output_throughput_tps_stderr", 0.0) for r in strategy],
        color="#4c78a8",
        capsize=3,
    )
    ax.set_title("(a) Output throughput")
    ax.set_ylabel("tokens/s")

    ax = axes2[0][1]
    ax.bar(
        labels,
        [r["mean_e2e_ms"] for r in strategy],
        yerr=[r.get("mean_e2e_ms_stderr", 0.0) for r in strategy],
        color="#f28e2b",
        capsize=3,
    )
    ax.set_title("(b) End-to-end latency")
    ax.set_ylabel("ms")

    ax = axes2[1][0]
    ax.errorbar(
        x_ratio,
        [r["critical_mean_e2e_ms"] for r in ratio],
        yerr=[r.get("critical_mean_e2e_ms_stderr", 0.0) for r in ratio],
        marker="o",
        color="#59a14f",
        capsize=3,
        label="critical",
    )
    ax.errorbar(
        x_ratio,
        [r["noncritical_mean_e2e_ms"] for r in ratio],
        yerr=[r.get("noncritical_mean_e2e_ms_stderr", 0.0) for r in ratio],
        marker="s",
        color="#9c755f",
        capsize=3,
        label="non-critical",
    )
    ax.set_title("(c) Agent-class latency")
    ax.set_xlabel("Critical agent ratio")
    ax.set_ylabel("mean e2e (ms)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes2[1][1]
    ax.errorbar(
        [_num(str(r["value"])) for r in timing],
        [r["affected"] for r in timing],
        yerr=[r.get("affected_stderr", 0.0) for r in timing],
        marker="o",
        color="#b07aa1",
        capsize=3,
    )
    ax.set_title("(d) Requests retried")
    ax.set_xlabel("Failure injection time (s)")
    ax.set_ylabel("requests")

    for ax in axes2.ravel():
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)

    fig2.tight_layout()
    system_png = output_dir / "fault_tolerance_sweep_system.png"
    system_pdf = output_dir / "fault_tolerance_sweep_system.pdf"
    fig2.savefig(system_png, dpi=220, bbox_inches="tight")
    fig2.savefig(system_pdf, bbox_inches="tight")
    return png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot FT sweep matrix")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/synthesis_requests/results/figures"))
    args = parser.parse_args(argv)
    out = plot(args.manifest, args.output_dir)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
