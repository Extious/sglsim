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
    / "claude_code_trace_dp8"
)
TRACE_WORKLOAD = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "claude_code_trace_workload.jsonl"
)
CSV_OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "figures"
    / "data"
    / "measured"
    / "fig_real_trace_characterization.csv"
)
OUTPUT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "Figures"
    / "fig_real_trace_characterization.png"
)
DRAFT_OUTPUT = OUTPUT


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
        raise FileNotFoundError(f"Missing Claude trace results under {TRACE_RESULTS_ROOT}")
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


def _hist_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    if count <= 15:
        return "11-15"
    return "16-20"


def _source_trace_summary() -> dict[str, object]:
    repos: Counter[str] = Counter()
    source_tables: Counter[str] = Counter()
    models: Counter[str] = Counter()
    stages: Counter[tuple[int, str]] = Counter()
    model_roles: Counter[str] = Counter()
    workflow_shapes: Counter[str] = Counter()
    rows = 0
    if not TRACE_WORKLOAD.exists():
        return {
            "source_trace_rows": 0,
            "source_repos": 0,
            "source_tables": 0,
            "source_models": 0,
            "source_stages": 0,
            "source_workflow_shapes": 0,
            "max_source_stage_width": 0,
            "repo_counts": {},
            "source_table_counts": {},
            "source_stage_counts": {},
            "model_role_counts": {},
        }
    with TRACE_WORKLOAD.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            payload = json.loads(line)
            repos[str(payload.get("source_repo") or "unknown")] += 1
            source_tables[str(payload.get("source_table") or "unknown")] += 1
            model = str(payload.get("model") or "unknown")
            models[model] += 1
            stage = str(payload.get("source_stage") or payload.get("stage_id") or "unknown")
            stage_id = int(float(payload.get("stage_id") or 0))
            stages[(stage_id, stage)] += 1
            role = str(payload.get("source_model_role") or "unknown")
            model_roles[role] += 1
            shape = str(payload.get("workflow_shape") or "")
            if shape:
                workflow_shapes[shape] += 1
    max_stage_width = 0
    for (_stage_id, _stage), count in stages.items():
        max_stage_width = max(max_stage_width, int(count))
    return {
        "source_trace_rows": rows,
        "source_repos": len(repos),
        "source_tables": len(source_tables),
        "source_models": len(models),
        "source_stages": len(stages),
        "source_workflow_shapes": len(workflow_shapes),
        "max_source_stage_width": max_stage_width,
        "top_workflow_shape": workflow_shapes.most_common(1)[0][0]
        if workflow_shapes
        else "",
        "top_repo": repos.most_common(1)[0][0] if repos else "",
        "top_source_table": source_tables.most_common(1)[0][0]
        if source_tables
        else "",
        "top_model": models.most_common(1)[0][0] if models else "",
        "repo_counts": dict(repos),
        "source_table_counts": dict(source_tables),
        "source_stage_counts": {
            f"s{stage_id}:{stage}": count
            for (stage_id, stage), count in sorted(stages.items())
        },
        "model_role_counts": dict(model_roles),
    }


def build_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    run_dir = _latest_trace_dir()
    request_rows = [
        row
        for row in _read_csv(run_dir / "request_details.csv")
        if str(row.get("status") or "") == "completed"
    ]
    by_workflow: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in request_rows:
        by_workflow[str(row.get("workflow_id") or "workflow")].append(row)

    request_counts = [len(rows) for rows in by_workflow.values()]
    input_tokens = [_float(row.get("input_length")) for row in request_rows]
    output_tokens = [_float(row.get("output_length")) for row in request_rows]
    latencies_ms = [
        _float(row.get("client_wall_e2e_latency_s")) * 1000.0
        for row in request_rows
        if _float(row.get("client_wall_e2e_latency_s")) > 0
    ]
    workflow_wct_ms: list[float] = []
    for rows in by_workflow.values():
        created = [
            _float(row.get("created_time_s"), -1.0)
            for row in rows
            if _float(row.get("created_time_s"), -1.0) >= 0.0
        ]
        finished = [
            _float(row.get("finish_time_s"), -1.0)
            for row in rows
            if _float(row.get("finish_time_s"), -1.0) >= 0.0
        ]
        if created and finished:
            workflow_wct_ms.append((max(finished) - min(created)) * 1000.0)

    source_summary = _source_trace_summary()
    totals = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "workflows": len(by_workflow),
        "request_rows": len(request_rows),
        "source_trace_rows": source_summary.get("source_trace_rows", 0),
        "source_repos": source_summary.get("source_repos", 0),
        "source_tables": source_summary.get("source_tables", 0),
        "source_models": source_summary.get("source_models", 0),
        "source_stages": source_summary.get("source_stages", 0),
        "source_workflow_shapes": source_summary.get("source_workflow_shapes", 0),
        "max_source_stage_width": source_summary.get("max_source_stage_width", 0),
        "top_workflow_shape": source_summary.get("top_workflow_shape", ""),
        "critical_rows": sum(
            1 for row in request_rows if str(row.get("critical_agent")).lower() == "true"
        ),
        "decode_dominant_rows": sum(
            1 for row in request_rows if str(row.get("decode_dominant")).lower() == "true"
        ),
        "input_tokens": int(sum(input_tokens)),
        "output_tokens": int(sum(output_tokens)),
        "gpu_replica_tokens": int(
            sum(_float(row.get("aegis_gpu_replica_tokens")) for row in request_rows)
        ),
        "candidate_backup_tokens": int(
            sum(
                _float(row.get("aegis_candidate_backed_up_tokens"))
                for row in request_rows
            )
        ),
        "candidate_recompute_tokens": int(
            sum(
                _float(row.get("aegis_candidate_recompute_tokens"))
                for row in request_rows
            )
        ),
        "p50_input_tokens": _percentile(input_tokens, 0.50),
        "p90_input_tokens": _percentile(input_tokens, 0.90),
        "p50_output_tokens": _percentile(output_tokens, 0.50),
        "p90_output_tokens": _percentile(output_tokens, 0.90),
        "p90_latency_ms": _percentile(latencies_ms, 0.90),
        "p90_wct_ms": _percentile(workflow_wct_ms, 0.90),
        "mean_requests_per_workflow": statistics.mean(request_counts)
        if request_counts
        else 0.0,
        "max_requests_per_workflow": max(request_counts) if request_counts else 0,
    }

    rows: list[dict[str, object]] = []
    common = {
        "run_id": totals["run_id"],
        "source": "sglsim_claude_code_real_trace_characterization",
        "evidence_tier": "sglsim-real-trace-characterization",
        "note": (
            "Public Claude Code trace rows are converted into SGLSim online "
            "serving requests for real-trace characterization."
        ),
    }
    for key, value in totals.items():
        rows.append(
            {
                **common,
                "row_type": "summary",
                "metric": key,
                "bucket": "",
                "value": value,
                "unit": "count" if isinstance(value, int) else "value",
            }
        )

    request_hist = Counter(_hist_bucket(count) for count in request_counts)
    for bucket in ["1", "2-5", "6-10", "11-15", "16-20"]:
        rows.append(
            {
                **common,
                "row_type": "workflow_request_histogram",
                "metric": "workflows",
                "bucket": bucket,
                "value": request_hist.get(bucket, 0),
                "unit": "workflow_count",
            }
        )

    for repo, count in sorted(
        dict(source_summary.get("repo_counts", {})).items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        rows.append(
            {
                **common,
                "row_type": "source_repo_count",
                "metric": "source_trace_rows",
                "bucket": repo,
                "value": count,
                "unit": "trace_rows",
            }
        )

    for table, count in sorted(
        dict(source_summary.get("source_table_counts", {})).items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        rows.append(
            {
                **common,
                "row_type": "source_table_count",
                "metric": "source_trace_rows",
                "bucket": table,
                "value": count,
                "unit": "trace_rows",
            }
        )

    for stage, count in sorted(
        dict(source_summary.get("source_stage_counts", {})).items(),
        key=lambda item: int(str(item[0]).split(":", 1)[0].lstrip("s") or 0),
    ):
        rows.append(
            {
                **common,
                "row_type": "source_stage_count",
                "metric": "source_trace_rows",
                "bucket": stage,
                "value": count,
                "unit": "agent_sessions",
            }
        )

    for metric, values in (
        ("input_tokens_per_request", input_tokens),
        ("output_tokens_per_request", output_tokens),
        ("workflow_wct_ms", workflow_wct_ms),
    ):
        for pct in (0.5, 0.9, 1.0):
            rows.append(
                {
                    **common,
                    "row_type": "percentile",
                    "metric": metric,
                    "bucket": f"p{int(pct * 100)}",
                    "value": round(_percentile(values, pct), 3),
                    "unit": "tokens" if "tokens" in metric else "ms",
                }
            )
    return rows, totals


def write_csv(rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(rows: list[dict[str, object]], name: str) -> float:
    for row in rows:
        if row.get("row_type") == "summary" and row.get("metric") == name:
            return _float(row.get("value"))
    return 0.0


def _completed_request_rows() -> list[dict[str, str]]:
    run_dir = _latest_trace_dir()
    return [
        row
        for row in _read_csv(run_dir / "request_details.csv")
        if str(row.get("status") or "") == "completed"
    ]


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(value for value in values if value >= 0.0)
    if not ordered:
        return [], []
    n = len(ordered)
    return ordered, [(idx + 1) / n for idx in range(n)]


def _plot_ecdf(
    ax: plt.Axes,
    values: list[float],
    *,
    label: str,
    color: str,
    lw: float = 1.7,
) -> None:
    xs, ys = _ecdf(values)
    if xs:
        ax.step(xs, ys, where="post", label=label, color=color, linewidth=lw)


def render(rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 180,
        }
    )
    request_rows = _completed_request_rows()
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.0))

    stage_rows = [row for row in rows if row.get("row_type") == "source_stage_count"]
    stage_rows = sorted(
        stage_rows,
        key=lambda row: int(str(row.get("bucket") or "s0").split(":", 1)[0].lstrip("s") or 0),
    )
    stage_labels = [
        str(row.get("bucket") or "").split(":", 1)[-1].replace("_", "\n")
        for row in stage_rows
    ]
    stage_counts = [_float(row.get("value")) for row in stage_rows]
    axes[0, 0].bar(
        list(range(len(stage_counts))),
        stage_counts,
        color="#2878B5",
        edgecolor="#1F4E79",
        linewidth=0.7,
    )
    axes[0, 0].set_title("(a) real workflow fan-out by stage")
    axes[0, 0].set_xlabel("Claude Code workflow stage")
    axes[0, 0].set_ylabel("agent sessions")
    axes[0, 0].set_xticks(list(range(len(stage_labels))))
    axes[0, 0].set_xticklabels(stage_labels, rotation=30, ha="right", fontsize=7.5)
    axes[0, 0].set_ylim(0, max(stage_counts or [1.0]) * 1.22)
    for idx, count in enumerate(stage_counts):
        axes[0, 0].text(
            idx,
            count + max(stage_counts or [1.0]) * 0.035,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )

    input_tokens = [_float(row.get("input_length")) for row in request_rows]
    output_tokens = [_float(row.get("output_length")) for row in request_rows]
    _plot_ecdf(axes[0, 1], input_tokens, label="input", color="#2D8B57")
    _plot_ecdf(axes[0, 1], output_tokens, label="output", color="#D97706")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_ylim(0, 1.03)
    axes[0, 1].set_xlabel("tokens / request")
    axes[0, 1].set_ylabel("ECDF")
    axes[0, 1].set_title("(b) token length ECDF")
    axes[0, 1].legend(frameon=False, loc="lower right")

    decode_latencies = [
        _float(row.get("client_wall_e2e_latency_s"), _float(row.get("total_latency_s"))) * 1000.0
        for row in request_rows
        if str(row.get("decode_dominant") or "").lower() == "true"
    ]
    other_latencies = [
        _float(row.get("client_wall_e2e_latency_s"), _float(row.get("total_latency_s"))) * 1000.0
        for row in request_rows
        if str(row.get("decode_dominant") or "").lower() != "true"
    ]
    _plot_ecdf(axes[1, 0], decode_latencies, label="decode-heavy", color="#8B5CF6")
    _plot_ecdf(axes[1, 0], other_latencies, label="other Critical", color="#4B5563")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(0, 1.03)
    axes[1, 0].set_xlabel("client wall latency (ms)")
    axes[1, 0].set_ylabel("ECDF")
    axes[1, 0].set_title("(c) online latency ECDF by request type")
    axes[1, 0].legend(frameon=False, loc="lower right")

    exposure_series = [
        (
            "candidate backup",
            [_float(row.get("aegis_candidate_backed_up_tokens")) + 1.0 for row in request_rows],
            "#2D8B57",
        ),
        (
            "candidate recompute",
            [_float(row.get("aegis_candidate_recompute_tokens")) + 1.0 for row in request_rows],
            "#DC2626",
        ),
        (
            "GPUReplica placed",
            [_float(row.get("aegis_gpu_replica_tokens")) + 1.0 for row in request_rows],
            "#31688E",
        ),
    ]
    for name, values, color in exposure_series:
        _plot_ecdf(axes[1, 1], values, label=name, color=color)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_ylim(0, 1.03)
    axes[1, 1].set_xlabel("tokens / request (+1 for zeros)")
    axes[1, 1].set_ylabel("ECDF")
    axes[1, 1].set_title("(d) AegisKV token exposure ECDF")
    axes[1, 1].legend(frameon=False, loc="lower right")

    for ax in axes.flat:
        ax.grid(True, color="#E5E7EB", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    workflows = int(_metric(rows, "workflows"))
    reqs = int(_metric(rows, "request_rows"))
    stages = int(_metric(rows, "source_stages"))
    max_width = int(_metric(rows, "max_source_stage_width"))
    decode = int(_metric(rows, "decode_dominant_rows"))
    p90_wct = _metric(rows, "p90_wct_ms")
    fig.suptitle(
        "Claude Code real multi-subagent trace characterization: "
        f"{workflows} workflow, {reqs} agent sessions, {stages} stages, "
        f"max fan-out {max_width}; decode-heavy {decode}/{reqs}, p90 WCT {p90_wct:.0f} ms",
        y=1.02,
        fontsize=11.5,
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
    print(
        json.dumps(
            {
                "csv": str(CSV_OUTPUT),
                "output": str(OUTPUT),
                "draft_output": str(DRAFT_OUTPUT),
                "rows": len(rows),
                "workflows": totals["workflows"],
                "request_rows": totals["request_rows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
