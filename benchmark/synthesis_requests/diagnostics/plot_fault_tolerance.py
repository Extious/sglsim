from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _strategy_label(path: Path) -> str:
    cfg = path / "simulator_config.json"
    if cfg.is_file():
        raw = _read_json(cfg)
        ft = raw.get("fault_tolerance", {})
        if isinstance(ft, dict):
            return str(ft.get("strategy") or path.name)
    return path.name


def _summarize(run_dir: Path) -> dict[str, Any]:
    metrics = _read_json(run_dir / "metrics.json")
    rows = _read_rows(run_dir / "request_details.csv")
    affected = [r for r in rows if str(r.get("is_failover_retried", "")).lower() in {"1", "true"}]
    critical = [
        r
        for r in affected
        if str(r.get("critical_agent", "")).lower() in {"1", "true"}
    ]
    noncritical = [r for r in affected if r not in critical]

    def _sum_int(items: list[dict[str, str]], key: str) -> int:
        return sum(int(float(r.get(key, 0) or 0)) for r in items)

    def _mean_ms(items: list[dict[str, str]], key: str) -> float:
        vals = [float(r.get(key, 0) or 0) * 1000.0 for r in items]
        return sum(vals) / len(vals) if vals else 0.0

    recompute = sum(int(float(r.get("recompute_tokens", 0) or 0)) for r in affected)
    backed = sum(int(float(r.get("pre_failover_backed_up_tokens", 0) or 0)) for r in affected)
    overhead = [
        float(r.get("failover_overhead_s", 0) or 0) * 1000.0
        for r in affected
    ]
    return {
        "label": _strategy_label(run_dir),
        "completed": int(metrics.get("completed", 0)),
        "affected": len(affected),
        "recompute_tokens": recompute,
        "backed_up_tokens": backed,
        "mean_overhead_ms": sum(overhead) / len(overhead) if overhead else 0.0,
        "mean_e2e_latency_ms": float(metrics.get("mean_e2e_latency_ms", 0.0)),
        "critical_affected": len(critical),
        "noncritical_affected": len(noncritical),
        "critical_recompute_tokens": _sum_int(critical, "recompute_tokens"),
        "noncritical_recompute_tokens": _sum_int(noncritical, "recompute_tokens"),
        "critical_backed_up_tokens": _sum_int(
            critical, "pre_failover_backed_up_tokens"
        ),
        "noncritical_backed_up_tokens": _sum_int(
            noncritical, "pre_failover_backed_up_tokens"
        ),
        "critical_mean_overhead_ms": _mean_ms(critical, "failover_overhead_s"),
        "noncritical_mean_overhead_ms": _mean_ms(noncritical, "failover_overhead_s"),
    }


def plot(run_dirs: list[Path], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_summarize(path) for path in run_dirs]
    labels = [s["label"] for s in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#b07aa1"]

    axes[0].bar(labels, [s["recompute_tokens"] for s in summaries], color=colors[: len(labels)])
    axes[0].set_title("(a) Failover recompute")
    axes[0].set_ylabel("Tokens")

    axes[1].bar(labels, [s["backed_up_tokens"] for s in summaries], color=colors[: len(labels)])
    axes[1].set_title("(b) GPU backup reuse")
    axes[1].set_ylabel("Tokens")

    axes[2].bar(labels, [s["mean_overhead_ms"] for s in summaries], color=colors[: len(labels)])
    axes[2].set_title("(c) Failover overhead")
    axes[2].set_ylabel("ms")

    for ax in axes:
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    png = output_dir / "fault_tolerance_summary.png"
    pdf = output_dir / "fault_tolerance_summary.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    fig2, axes2 = plt.subplots(1, 3, figsize=(10.5, 3.0))
    x = range(len(labels))
    width = 0.65
    axes2[0].bar(
        x,
        [s["critical_recompute_tokens"] for s in summaries],
        width,
        label="critical",
        color="#59a14f",
    )
    axes2[0].bar(
        x,
        [s["noncritical_recompute_tokens"] for s in summaries],
        width,
        bottom=[s["critical_recompute_tokens"] for s in summaries],
        label="non-critical",
        color="#9c755f",
    )
    axes2[0].set_title("(a) Recompute by agent class")
    axes2[0].set_ylabel("Tokens")

    axes2[1].bar(
        x,
        [s["critical_backed_up_tokens"] for s in summaries],
        width,
        label="critical",
        color="#59a14f",
    )
    axes2[1].bar(
        x,
        [s["noncritical_backed_up_tokens"] for s in summaries],
        width,
        bottom=[s["critical_backed_up_tokens"] for s in summaries],
        label="non-critical",
        color="#9c755f",
    )
    axes2[1].set_title("(b) Backup reuse by agent class")
    axes2[1].set_ylabel("Tokens")

    axes2[2].bar(
        x,
        [s["critical_mean_overhead_ms"] for s in summaries],
        width,
        label="critical",
        color="#59a14f",
    )
    axes2[2].bar(
        x,
        [s["noncritical_mean_overhead_ms"] for s in summaries],
        width * 0.55,
        label="non-critical",
        color="#9c755f",
        alpha=0.75,
    )
    axes2[2].set_title("(c) Mean failover overhead")
    axes2[2].set_ylabel("ms")

    for ax in axes2:
        ax.set_xticks(list(x), labels, rotation=20)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
    axes2[0].legend(frameon=False, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(output_dir / "fault_tolerance_critical_breakdown.png", dpi=200, bbox_inches="tight")
    fig2.savefig(output_dir / "fault_tolerance_critical_breakdown.pdf", bbox_inches="tight")
    (output_dir / "fault_tolerance_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    return png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot synthesis FT comparison")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/synthesis_requests/results/figures"))
    args = parser.parse_args(argv)
    out = plot(args.run_dirs, args.output_dir)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
