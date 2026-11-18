#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import socket
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RESULTS_ROOT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "results"
    / "agentic_workflows"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_aegiskv_alpha_policy_wct_cdf.png"
)
DEFAULT_DRAFT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "png"
    / "fig_aegiskv_alpha_policy_wct_cdf.png"
)
DEFAULT_CSV = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_aegiskv_alpha_policy_wct_cdf.csv"
)

POLICY_SPECS = (
    ("fix_a010", "AegisKV (Fix alpha = 0.1)", "#2563EB"),
    ("fix_a030", "AegisKV (Fix alpha = 0.3)", "#059669"),
    ("fix_a050", "AegisKV (Fix alpha = 0.5)", "#D97706"),
    ("mdp", "AegisKV (withMDP+online)", "#7C3AED"),
)
RUN_PREFIX = "aegiskv-alpha-policy-cdf-dp8-48w-fail8"
COLOR_GRID = "#E5E7EB"
COLOR_MUTED = "#64748B"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def workflow_wcts_s(rows: list[dict[str, str]]) -> list[float]:
    by_workflow: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        workflow_id = row.get("workflow_id") or row.get("rid") or "workflow"
        by_workflow.setdefault(workflow_id, []).append(row)

    values: list[float] = []
    for workflow_rows in by_workflow.values():
        created = [
            _float(row.get("created_time_s"), -1.0)
            for row in workflow_rows
            if _float(row.get("created_time_s"), -1.0) >= 0.0
        ]
        finished = [
            _float(row.get("finish_time_s"), -1.0)
            for row in workflow_rows
            if _float(row.get("finish_time_s"), -1.0) >= 0.0
        ]
        if created and finished:
            values.append(max(finished) - min(created))
            continue
        latencies = [
            _float(row.get("client_wall_e2e_latency_s") or row.get("e2e_latency_s"))
            for row in workflow_rows
        ]
        if latencies:
            values.append(max(latencies))
    return [value for value in values if value > 0.0]


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    ordered = sorted(values)
    ys = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    return ordered, ys


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latest_run_dir(results_root: Path, run_id: str) -> Path | None:
    run_root = Path(results_root) / run_id / "runs"
    if not run_root.is_dir():
        return None
    candidates = sorted(
        path
        for path in run_root.glob("trial_*/*")
        if path.is_dir() and (path / "request_details.csv").is_file()
    )
    return candidates[-1] if candidates else None


def collect_measured_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for suffix, label, _color in POLICY_SPECS:
        run_id = f"{RUN_PREFIX}-{suffix}-r1"
        run_dir = latest_run_dir(results_root, run_id)
        if run_dir is None:
            continue
        request_rows = _read_csv(run_dir / "request_details.csv")
        by_workflow: dict[str, list[dict[str, str]]] = {}
        for row in request_rows:
            workflow_id = row.get("workflow_id") or row.get("rid") or "workflow"
            by_workflow.setdefault(workflow_id, []).append(row)
        metrics = {}
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload if isinstance(payload, dict) else {}
        wcts: list[float] = []
        for workflow_id, workflow_rows in sorted(by_workflow.items()):
            created = [
                _float(row.get("created_time_s"), -1.0)
                for row in workflow_rows
                if _float(row.get("created_time_s"), -1.0) >= 0.0
            ]
            finished = [
                _float(row.get("finish_time_s"), -1.0)
                for row in workflow_rows
                if _float(row.get("finish_time_s"), -1.0) >= 0.0
            ]
            if created and finished:
                wct_s = max(finished) - min(created)
            else:
                latencies = [
                    _float(row.get("client_wall_e2e_latency_s") or row.get("e2e_latency_s"))
                    for row in workflow_rows
                ]
                wct_s = max(latencies) if latencies else 0.0
            if wct_s <= 0.0:
                continue
            wcts.append(wct_s)
            rows.append(
                {
                    "policy_suffix": suffix,
                    "policy_label": label,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "workflow_id": workflow_id,
                    "wct_s": round(wct_s, 6),
                }
            )
        rows.append(
            {
                "policy_suffix": suffix,
                "policy_label": label,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "workflow_id": "__summary__",
                "wct_s": "",
                "workflow_count": len(wcts),
                "mean_wct_s": round(mean(wcts), 6) if wcts else "",
                "p90_wct_s": round(_percentile(wcts, 0.90), 6) if wcts else "",
                "p99_wct_s": round(_percentile(wcts, 0.99), 6) if wcts else "",
                "max_wct_s": round(max(wcts), 6) if wcts else "",
                "mean_e2e_latency_ms": metrics.get("mean_e2e_latency_ms", ""),
            }
        )
    return rows


def _curve_rows(measured_rows: list[dict[str, object]]) -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {}
    for suffix, label, _color in POLICY_SPECS:
        values = [
            float(row["wct_s"])
            for row in measured_rows
            if row.get("policy_suffix") == suffix
            and row.get("workflow_id") != "__summary__"
            and row.get("wct_s") not in ("", None)
        ]
        if values:
            curves[label] = values
    return curves


def render_figure(
    *,
    measured_rows: list[dict[str, object]],
    output_path: Path,
    draft_output_path: Path,
) -> None:
    curves = _curve_rows(measured_rows)
    if not curves:
        raise ValueError("no measured WCT curves found; run experiments first")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for suffix, label, color in POLICY_SPECS:
        values = curves.get(label)
        if not values:
            continue
        xs, ys = _ecdf(values)
        ax.step(
            xs,
            ys,
            where="post",
            color=color,
            linewidth=2.0,
            label=label,
            zorder=2,
        )

    ax.set_xlabel("Workflow completion time (s)", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=COLOR_MUTED, labelsize=10)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        borderpad=0.6,
    )
    ax.set_title(
        "DP8 / 48 workflows / concurrency 24 / 8 injected failures",
        fontsize=11,
        color=COLOR_MUTED,
        pad=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draft_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    fig.savefig(draft_output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def find_free_port(start: int = 39010, span: int = 200, step: int = 10) -> int:
    for port in range(start, start + span, step):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found in range {start}-{start + span}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render AegisKV alpha-policy WCT CDF comparison."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draft-output", type=Path, default=DEFAULT_DRAFT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    measured_rows = collect_measured_rows(args.results_root)
    if not measured_rows:
        raise SystemExit(
            "missing measured runs; expected run ids like "
            f"{RUN_PREFIX}-<suffix>-r1 under {args.results_root}"
        )
    _write_csv(measured_rows, args.csv)
    render_figure(
        measured_rows=measured_rows,
        output_path=args.output,
        draft_output_path=args.draft_output,
    )
    print(f"wrote {args.csv}")
    print(f"wrote {args.output}")
    print(f"wrote {args.draft_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
