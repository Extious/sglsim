#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_DIR = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "cc-traces-weka-062126"
)
DEFAULT_INPUT = DEFAULT_DATASET_DIR / "traces.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "analysis"


def _percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def _distribution(values: Iterable[float]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {
            "count": 0,
            "sum": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    return {
        "count": len(materialized),
        "sum": sum(materialized),
        "min": min(materialized),
        "p50": _percentile(materialized, 50),
        "p90": _percentile(materialized, 90),
        "p95": _percentile(materialized, 95),
        "p99": _percentile(materialized, 99),
        "max": max(materialized),
    }


def _max_overlap(intervals: Iterable[tuple[float, float]]) -> int:
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    current = 0
    maximum = 0
    for _time, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def analyze(path: Path) -> dict[str, object]:
    trace_rows: list[dict[str, object]] = []
    main_type_counts: Counter[str] = Counter()
    inner_type_counts: Counter[str] = Counter()
    subagent_type_counts: Counter[str] = Counter()
    subagent_status_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    trace_model_counts: Counter[int] = Counter()
    hash_scope_counts: Counter[str] = Counter()
    block_size_counts: Counter[str] = Counter()

    group_inner_counts: list[int] = []
    group_total_tokens: list[int] = []
    group_duration_seconds: list[float] = []
    main_input_tokens: list[int] = []
    main_output_tokens: list[int] = []
    inner_input_tokens: list[int] = []
    inner_output_tokens: list[int] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            trace_id = str(row["id"])
            requests = list(row.get("requests") or [])
            models = list(row.get("models") or [])
            hash_scope_counts[str(row.get("hash_id_scope") or "")] += 1
            block_size_counts[str(row.get("block_size") or "")] += 1
            trace_model_counts[len(models)] += 1

            main_count = 0
            subagent_groups = 0
            inner_count = 0
            max_inner_per_group = 0
            first_t: Optional[float] = None
            last_t: Optional[float] = None
            group_intervals: list[tuple[float, float]] = []
            inner_intervals: list[tuple[float, float]] = []

            for request in requests:
                request_type = str(request.get("type") or "")
                request_t = _as_float(request.get("t"))
                if request_t is not None:
                    first_t = request_t if first_t is None else min(first_t, request_t)
                    last_t = request_t if last_t is None else max(last_t, request_t)

                if request_type == "subagent":
                    subagent_groups += 1
                    subagent_type_counts[str(request.get("subagent_type") or "")] += 1
                    subagent_status_counts[str(request.get("status") or "")] += 1
                    inner_requests = list(request.get("requests") or [])
                    inner_count += len(inner_requests)
                    max_inner_per_group = max(max_inner_per_group, len(inner_requests))
                    group_inner_counts.append(len(inner_requests))
                    group_total_tokens.append(int(request.get("total_tokens") or 0))
                    duration_s = float(request.get("duration_ms") or 0.0) / 1000.0
                    group_duration_seconds.append(duration_s)
                    if request_t is not None:
                        group_intervals.append((request_t, request_t + duration_s))

                    for inner in inner_requests:
                        inner_type_counts[str(inner.get("type") or "")] += 1
                        model = str(inner.get("model") or "")
                        if model:
                            model_counts[model] += 1
                        if isinstance(inner.get("in"), (int, float)):
                            inner_input_tokens.append(int(inner["in"]))
                        if isinstance(inner.get("out"), (int, float)):
                            inner_output_tokens.append(int(inner["out"]))
                        inner_t = _as_float(inner.get("t"))
                        api_time = _as_float(inner.get("api_time"))
                        if inner_t is not None and api_time is not None:
                            inner_intervals.append((inner_t, inner_t + api_time))
                            first_t = inner_t if first_t is None else min(first_t, inner_t)
                            last_t = (
                                inner_t + api_time
                                if last_t is None
                                else max(last_t, inner_t + api_time)
                            )
                else:
                    main_count += 1
                    main_type_counts[request_type] += 1
                    model = str(request.get("model") or "")
                    if model:
                        model_counts[model] += 1
                    if isinstance(request.get("in"), (int, float)):
                        main_input_tokens.append(int(request["in"]))
                    if isinstance(request.get("out"), (int, float)):
                        main_output_tokens.append(int(request["out"]))
                    api_time = _as_float(request.get("api_time"))
                    if request_t is not None and api_time is not None:
                        last_t = (
                            request_t + api_time
                            if last_t is None
                            else max(last_t, request_t + api_time)
                        )

            timeline_seconds = (
                0.0 if first_t is None or last_t is None else round(last_t - first_t, 3)
            )
            agent_hierarchy_depth = 2 if subagent_groups else 1
            request_chain_depth_estimate = len(requests) + max(
                0, max_inner_per_group - 1
            )
            trace_rows.append(
                {
                    "trace_id": trace_id,
                    "top_level_entries": len(requests),
                    "main_requests": main_count,
                    "subagent_groups": subagent_groups,
                    "subagent_inner_requests": inner_count,
                    "agent_count": 1 + subagent_groups,
                    "agent_hierarchy_depth": agent_hierarchy_depth,
                    "main_stream_depth": len(requests),
                    "max_subagent_local_depth": max_inner_per_group,
                    "request_chain_depth_estimate": request_chain_depth_estimate,
                    "model_requests": main_count + inner_count,
                    "max_inner_requests_per_group": max_inner_per_group,
                    "max_concurrent_subagent_groups": _max_overlap(group_intervals),
                    "max_concurrent_inner_requests": _max_overlap(inner_intervals),
                    "timeline_seconds": timeline_seconds,
                    "model_count": len(models),
                }
            )

    return {
        "dataset": "semianalysisai/cc-traces-weka-062126",
        "input": str(path),
        "traces": len(trace_rows),
        "traces_with_subagents": sum(
            1 for row in trace_rows if int(row["subagent_groups"]) > 0
        ),
        "top_level_entries": sum(int(row["top_level_entries"]) for row in trace_rows),
        "main_requests": sum(int(row["main_requests"]) for row in trace_rows),
        "subagent_groups": sum(int(row["subagent_groups"]) for row in trace_rows),
        "subagent_inner_requests": sum(
            int(row["subagent_inner_requests"]) for row in trace_rows
        ),
        "total_model_requests": sum(int(row["model_requests"]) for row in trace_rows),
        "main_type_counts": dict(main_type_counts),
        "inner_type_counts": dict(inner_type_counts),
        "subagent_type_counts": dict(subagent_type_counts),
        "subagent_status_counts": dict(subagent_status_counts),
        "model_counts": dict(model_counts),
        "trace_model_count_distribution": dict(trace_model_counts),
        "hash_id_scope_counts": dict(hash_scope_counts),
        "block_size_counts": dict(block_size_counts),
        "distributions": {
            "trace_subagent_groups": _distribution(
                int(row["subagent_groups"]) for row in trace_rows
            ),
            "trace_inner_requests": _distribution(
                int(row["subagent_inner_requests"]) for row in trace_rows
            ),
            "trace_model_requests": _distribution(
                int(row["model_requests"]) for row in trace_rows
            ),
            "trace_agent_count": _distribution(
                int(row["agent_count"]) for row in trace_rows
            ),
            "trace_agent_hierarchy_depth": _distribution(
                int(row["agent_hierarchy_depth"]) for row in trace_rows
            ),
            "trace_main_stream_depth": _distribution(
                int(row["main_stream_depth"]) for row in trace_rows
            ),
            "trace_max_subagent_local_depth": _distribution(
                int(row["max_subagent_local_depth"]) for row in trace_rows
            ),
            "trace_request_chain_depth_estimate": _distribution(
                int(row["request_chain_depth_estimate"]) for row in trace_rows
            ),
            "trace_max_concurrent_subagent_groups": _distribution(
                int(row["max_concurrent_subagent_groups"]) for row in trace_rows
            ),
            "trace_max_concurrent_inner_requests": _distribution(
                int(row["max_concurrent_inner_requests"]) for row in trace_rows
            ),
            "trace_timeline_seconds": _distribution(
                float(row["timeline_seconds"]) for row in trace_rows
            ),
            "group_inner_requests": _distribution(group_inner_counts),
            "group_total_tokens": _distribution(group_total_tokens),
            "group_duration_seconds": _distribution(group_duration_seconds),
            "main_input_tokens": _distribution(main_input_tokens),
            "main_output_tokens": _distribution(main_output_tokens),
            "inner_input_tokens": _distribution(inner_input_tokens),
            "inner_output_tokens": _distribution(inner_output_tokens),
        },
        "top_by_concurrent_subagent_groups": sorted(
            trace_rows,
            key=lambda row: (
                int(row["max_concurrent_subagent_groups"]),
                int(row["subagent_groups"]),
            ),
            reverse=True,
        )[:20],
        "top_by_subagent_inner_requests": sorted(
            trace_rows,
            key=lambda row: int(row["subagent_inner_requests"]),
            reverse=True,
        )[:20],
        "trace_rows": trace_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_int(value: object) -> str:
    return f"{int(float(value)):,}"


def _fmt_float(value: object) -> str:
    return f"{float(value):,.1f}"


def _write_report(path: Path, summary: dict[str, object]) -> None:
    dists = summary["distributions"]
    assert isinstance(dists, dict)
    group_dist = dists["trace_subagent_groups"]
    inner_dist = dists["trace_inner_requests"]
    concurrency_dist = dists["trace_max_concurrent_subagent_groups"]
    group_inner_dist = dists["group_inner_requests"]
    agent_depth_dist = dists["trace_agent_hierarchy_depth"]
    main_depth_dist = dists["trace_main_stream_depth"]
    subagent_depth_dist = dists["trace_max_subagent_local_depth"]
    chain_depth_dist = dists["trace_request_chain_depth_estimate"]
    main_input_dist = dists["main_input_tokens"]
    inner_input_dist = dists["inner_input_tokens"]
    assert isinstance(group_dist, dict)
    assert isinstance(inner_dist, dict)
    assert isinstance(concurrency_dist, dict)
    assert isinstance(group_inner_dist, dict)
    assert isinstance(agent_depth_dist, dict)
    assert isinstance(main_depth_dist, dict)
    assert isinstance(subagent_depth_dist, dict)
    assert isinstance(chain_depth_dist, dict)
    assert isinstance(main_input_dist, dict)
    assert isinstance(inner_input_dist, dict)

    lines = [
        "# Weka Claude Code Workflow Analysis",
        "",
        f"Dataset: `{summary['dataset']}`",
        f"Input: `{summary['input']}`",
        "",
        "## Interpretation",
        "",
        "- One JSONL row is one Claude Code session and should be treated as one workflow.",
        "- `requests` is the top-level main-agent timeline.",
        "- Top-level `type=subagent` entries are Task-tool spawned subagent groups.",
        "- Each subagent group contains its own `requests` list of inner model requests.",
        "- `hash_ids` are local 64-token KV block IDs; they are comparable only inside the same trace.",
        "- The dataset has timing (`t`, `duration_ms`, `api_time`) but no explicit parent request ID for each subagent group, so parent linkage should be reconstructed conservatively from timeline order.",
        "",
        "## Dataset Scale",
        "",
        f"- Traces/workflows: {_fmt_int(summary['traces'])}",
        f"- Traces with subagents: {_fmt_int(summary['traces_with_subagents'])}",
        f"- Main-agent model requests: {_fmt_int(summary['main_requests'])}",
        f"- Subagent groups: {_fmt_int(summary['subagent_groups'])}",
        f"- Subagent inner model requests: {_fmt_int(summary['subagent_inner_requests'])}",
        f"- Total individual model requests: {_fmt_int(summary['total_model_requests'])}",
        "",
        "## Workflow Shape",
        "",
        f"- Agent hierarchy depth: p50={_fmt_float(agent_depth_dist['p50'])}, p90={_fmt_float(agent_depth_dist['p90'])}, max={_fmt_int(agent_depth_dist['max'])}. This is 1 for main-only sessions and 2 for sessions with top-level subagents.",
        f"- Main-stream depth: p50={_fmt_float(main_depth_dist['p50'])}, p90={_fmt_float(main_depth_dist['p90'])}, p95={_fmt_float(main_depth_dist['p95'])}, max={_fmt_int(main_depth_dist['max'])}.",
        f"- Max subagent local depth: p50={_fmt_float(subagent_depth_dist['p50'])}, p90={_fmt_float(subagent_depth_dist['p90'])}, p95={_fmt_float(subagent_depth_dist['p95'])}, max={_fmt_int(subagent_depth_dist['max'])}.",
        f"- Estimated request-chain depth: p50={_fmt_float(chain_depth_dist['p50'])}, p90={_fmt_float(chain_depth_dist['p90'])}, p95={_fmt_float(chain_depth_dist['p95'])}, max={_fmt_int(chain_depth_dist['max'])}.",
        f"- Subagent groups per trace: p50={_fmt_float(group_dist['p50'])}, p90={_fmt_float(group_dist['p90'])}, p95={_fmt_float(group_dist['p95'])}, p99={_fmt_float(group_dist['p99'])}, max={_fmt_int(group_dist['max'])}.",
        f"- Subagent inner requests per trace: p50={_fmt_float(inner_dist['p50'])}, p90={_fmt_float(inner_dist['p90'])}, p95={_fmt_float(inner_dist['p95'])}, p99={_fmt_float(inner_dist['p99'])}, max={_fmt_int(inner_dist['max'])}.",
        f"- Concurrent subagent groups per trace: p50={_fmt_float(concurrency_dist['p50'])}, p90={_fmt_float(concurrency_dist['p90'])}, p95={_fmt_float(concurrency_dist['p95'])}, p99={_fmt_float(concurrency_dist['p99'])}, max={_fmt_int(concurrency_dist['max'])}.",
        f"- Inner requests per subagent group: p50={_fmt_float(group_inner_dist['p50'])}, p90={_fmt_float(group_inner_dist['p90'])}, p95={_fmt_float(group_inner_dist['p95'])}, p99={_fmt_float(group_inner_dist['p99'])}, max={_fmt_int(group_inner_dist['max'])}.",
        "",
        "## Token Pressure",
        "",
        f"- Main-agent input tokens: p50={_fmt_int(main_input_dist['p50'])}, p90={_fmt_int(main_input_dist['p90'])}, p95={_fmt_int(main_input_dist['p95'])}, max={_fmt_int(main_input_dist['max'])}.",
        f"- Subagent inner input tokens: p50={_fmt_int(inner_input_dist['p50'])}, p90={_fmt_int(inner_input_dist['p90'])}, p95={_fmt_int(inner_input_dist['p95'])}, max={_fmt_int(inner_input_dist['max'])}.",
        "",
        "## Recommended Experimental Use",
        "",
        "- Use this dataset as the real main/subagent trace source.",
        "- Treat each JSONL row as an independent workflow sample.",
        "- Use subagent group intervals to model fan-out and failure exposure.",
        "- For SGLSim replay, preserve `in`, `out`, `hash_ids`, model, and relative timestamps; synthesize prompt text only as a carrier because natural-language prompt content is not present.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze semianalysisai/cc-traces-weka-062126 workflows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = analyze(args.input)
    rows = summary.pop("trace_rows")
    assert isinstance(rows, list)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "workflow_summary.json"
    csv_path = args.output_dir / "trace_workflow_summary.csv"
    report_path = args.output_dir / "workflow_analysis.md"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    _write_csv(csv_path, rows)
    _write_report(report_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "csv": str(csv_path),
                "report": str(report_path),
                "traces": summary["traces"],
                "subagent_groups": summary["subagent_groups"],
                "subagent_inner_requests": summary["subagent_inner_requests"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
