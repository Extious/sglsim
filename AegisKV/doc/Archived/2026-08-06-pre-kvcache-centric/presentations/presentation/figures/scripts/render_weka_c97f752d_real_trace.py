#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
TRACE_RESULTS_ROOT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "results"
    / "weka_c97f752d_parallel_dp8"
)
TRACE_WORKLOAD = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "weka_c97f752d_parallel_workload.jsonl"
)
CSV_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_weka_c97f752d_real_trace.csv"
)
OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_weka_c97f752d_real_trace.png"
)
DRAFT_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "png"
    / "fig_weka_c97f752d_real_trace.png"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_trace_dir() -> Path:
    candidates = sorted(
        path
        for path in TRACE_RESULTS_ROOT.glob("*")
        if path.is_dir() and (path / "request_details.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"Missing Weka trace results under {TRACE_RESULTS_ROOT}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


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


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    ordered = sorted(values)
    ys = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    return ordered, ys


def _plot_ecdf(ax: plt.Axes, values: list[float], *, label: str, color: str) -> None:
    xs, ys = _ecdf(values)
    if xs:
        ax.step(xs, ys, where="post", label=label, color=color, linewidth=1.7)


def _completed_request_rows() -> list[dict[str, str]]:
    run_dir = _latest_trace_dir()
    return [
        row
        for row in _read_csv(run_dir / "request_details.csv")
        if str(row.get("status") or "") == "completed"
    ]


def _workload_token_map() -> dict[str, tuple[int, int]]:
    mapping: dict[str, tuple[int, int]] = {}
    if not TRACE_WORKLOAD.exists():
        return mapping
    with TRACE_WORKLOAD.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            rid = str(payload.get("rid") or "")
            mapping[rid] = (
                int(payload.get("prompt_len") or 0),
                int(payload.get("output_len") or 0),
            )
    return mapping


def _token_lengths(request_rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    source = _workload_token_map()
    input_tokens: list[float] = []
    output_tokens: list[float] = []
    for row in request_rows:
        rid = str(row.get("rid") or "")
        src = source.get(rid)
        inp = _float(row.get("input_length"))
        out = _float(row.get("output_length"), _float(row.get("target_output_tokens")))
        if inp <= 0 and src is not None:
            inp = float(src[0])
        if out <= 0 and src is not None:
            out = float(src[1])
        input_tokens.append(inp)
        output_tokens.append(out)
    return input_tokens, output_tokens


def _source_by_phase() -> dict[int, int]:
    counts: Counter[int] = Counter()
    if not TRACE_WORKLOAD.exists():
        return {}
    with TRACE_WORKLOAD.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            counts[int(payload.get("stage_id") or 0)] += 1
    return dict(counts)


def build_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    run_dir = _latest_trace_dir()
    request_rows = _completed_request_rows()
    by_workflow: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in request_rows:
        by_workflow[str(row.get("workflow_id") or "workflow")].append(row)

    input_tokens, output_tokens = _token_lengths(request_rows)
    latencies_ms = [
        _float(row.get("client_wall_e2e_latency_s")) * 1000.0
        for row in request_rows
        if _float(row.get("client_wall_e2e_latency_s")) > 0
    ]
    workflow_wct_ms: list[float] = []
    for rows in by_workflow.values():
        created = [_float(r.get("created_time_s"), -1.0) for r in rows if _float(r.get("created_time_s"), -1.0) >= 0]
        finished = [_float(r.get("finish_time_s"), -1.0) for r in rows if _float(r.get("finish_time_s"), -1.0) >= 0]
        if created and finished:
            workflow_wct_ms.append((max(finished) - min(created)) * 1000.0)

    parallel_rows = sum(
        1 for row in request_rows if int(_float(row.get("stage_width"))) >= 2
    )
    max_fanout = max((int(_float(row.get("stage_width"))) for row in request_rows), default=0)

    totals = {
        "run_id": run_dir.name,
        "workflows": len(by_workflow),
        "request_rows": len(request_rows),
        "parallel_rows": parallel_rows,
        "max_fanout": max_fanout,
        "critical_rows": sum(1 for row in request_rows if str(row.get("critical_agent")).lower() == "true"),
        "candidate_backup_tokens": int(
            sum(_float(row.get("aegis_candidate_backed_up_tokens")) for row in request_rows)
        ),
        "gpu_replica_tokens": int(sum(_float(row.get("aegis_gpu_replica_tokens")) for row in request_rows)),
        "p50_input_tokens": _percentile(input_tokens, 0.50),
        "p90_input_tokens": _percentile(input_tokens, 0.90),
        "p90_latency_ms": _percentile(latencies_ms, 0.90),
        "p90_wct_ms": _percentile(workflow_wct_ms, 0.90),
    }

    rows: list[dict[str, object]] = []
    for key, value in totals.items():
        rows.append({"row_type": "summary", "metric": key, "value": value})

    stage_counts = Counter(int(_float(row.get("stage_id"))) for row in request_rows)
    for stage_id, count in sorted(stage_counts.items()):
        rows.append({"row_type": "stage_count", "bucket": f"p{stage_id}", "value": count})

    source_phase = _source_by_phase()
    for stage_id, count in sorted(source_phase.items()):
        rows.append({"row_type": "source_stage_count", "bucket": f"p{stage_id}", "value": count})

    return rows, totals


def write_csv(rows: list[dict[str, object]]) -> None:
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict[str, object]]) -> None:
    request_rows = _completed_request_rows()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "figure.dpi": 180})
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.0))

    stage_rows = [row for row in rows if row.get("row_type") == "source_stage_count"]
    stage_rows = sorted(stage_rows, key=lambda row: str(row.get("bucket")))
    labels = [str(row.get("bucket") or "").replace("p", "P") for row in stage_rows]
    counts = [_float(row.get("value")) for row in stage_rows]
    colors = ["#2878B5" if lbl in {"P1", "P3", "P5", "P7"} else "#2D8B57" for lbl in labels]
    axes[0, 0].bar(range(len(counts)), counts, color=colors, edgecolor="#334155", linewidth=0.6)
    axes[0, 0].set_title("(a) request count by workflow phase")
    axes[0, 0].set_xlabel("phase (blue=main, green=fan-out)")
    axes[0, 0].set_ylabel("completed requests")
    axes[0, 0].set_xticks(range(len(labels)))
    axes[0, 0].set_xticklabels(labels, rotation=0)

    input_tokens, output_tokens = _token_lengths(request_rows)
    _plot_ecdf(axes[0, 1], input_tokens, label="input", color="#2878B5")
    _plot_ecdf(axes[0, 1], output_tokens, label="output", color="#D97706")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("(b) token length ECDF")
    axes[0, 1].set_xlabel("tokens / request")
    axes[0, 1].set_ylabel("ECDF")
    axes[0, 1].legend(frameon=False, loc="lower right")

    main_lat = [
        _float(row.get("client_wall_e2e_latency_s")) * 1000.0
        for row in request_rows
        if str(row.get("critical_agent")).lower() == "true"
    ]
    sub_lat = [
        _float(row.get("client_wall_e2e_latency_s")) * 1000.0
        for row in request_rows
        if str(row.get("critical_agent")).lower() != "true"
    ]
    _plot_ecdf(axes[1, 0], main_lat, label="main agent", color="#2878B5")
    _plot_ecdf(axes[1, 0], sub_lat, label="parallel subagent", color="#2D8B57")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("(c) latency ECDF by agent role")
    axes[1, 0].set_xlabel("client e2e latency (ms)")
    axes[1, 0].set_ylabel("ECDF")
    axes[1, 0].legend(frameon=False, loc="lower right")

    fanout = Counter(int(_float(row.get("stage_width"))) for row in request_rows)
    xs = sorted(fanout)
    ys = [fanout[x] for x in xs]
    axes[1, 1].bar([str(x) for x in xs], ys, color="#6A5ACD", edgecolor="#334155", linewidth=0.6)
    axes[1, 1].set_title("(d) fan-out width distribution")
    axes[1, 1].set_xlabel("stage width")
    axes[1, 1].set_ylabel("requests")

    summary = {str(row.get("metric")): row.get("value") for row in rows if row.get("row_type") == "summary"}
    fig.suptitle(
        "Weka trace c97f752d on AegisKV DP8 online path | "
        f"{int(_float(summary.get('request_rows')))} completed rows | "
        f"parallel rows={int(_float(summary.get('parallel_rows')))} | "
        f"max fan-out={int(_float(summary.get('max_fanout')))}",
        y=1.02,
        fontsize=11.0,
    )
    fig.tight_layout(pad=1.0, w_pad=1.5, h_pad=1.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DRAFT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    fig.savefig(DRAFT_OUTPUT, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows, totals = build_rows()
    write_csv(rows)
    render(rows)
    print(json.dumps({"output": str(OUTPUT), **totals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
