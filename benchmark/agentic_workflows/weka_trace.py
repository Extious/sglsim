from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

WEKA_DATASET = "semianalysisai/cc-traces-weka-062126"
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "cc-traces-weka-062126"
    / "traces.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "weka_trace_workload.jsonl"
)
DEFAULT_BATCH_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "weka_subagent_cdf_24w_workload.jsonl"
)
DEFAULT_SUMMARY_CSV = (
    Path(__file__).resolve().parents[1]
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "cc-traces-weka-062126"
    / "analysis"
    / "trace_workflow_summary.csv"
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _segments(reqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    seg: dict[str, Any] | None = None
    for r in reqs:
        kind = "subagent" if r.get("type") == "subagent" else "main"
        if seg is None or seg["kind"] != kind:
            if seg is not None:
                segments.append(seg)
            seg = {"kind": kind, "items": []}
        seg["items"].append(r)
    if seg is not None:
        segments.append(seg)
    return segments


def _decode_dominant(input_tokens: int, output_tokens: int) -> bool:
    if input_tokens <= 0:
        return output_tokens > 64
    return (output_tokens / input_tokens) >= 0.02 and output_tokens >= 256


def _workload_row(
    *,
    trace_id: str,
    rid: str,
    prompt_len: int,
    output_len: int,
    group_id: int,
    agent_id: str,
    phase_id: int,
    stage_width: int,
    stage_local_agent_index: int,
    critical_agent: bool,
    model: str,
    role: str,
    arrival_t: float,
    api_time_s: float,
    ttft_s: float,
    parallel_fanout: bool,
) -> dict[str, object]:
    return {
        "rid": rid,
        "prompt": f"weka-carrier-{trace_id}-{rid}",
        "prompt_len": max(int(prompt_len), 1),
        "output_len": max(int(output_len), 1),
        "group_id": group_id,
        "agent_id": agent_id,
        "workflow_id": trace_id,
        "stage_id": phase_id,
        "stage_width": stage_width,
        "stage_local_agent_index": stage_local_agent_index,
        "critical_agent": critical_agent,
        "decode_dominant": _decode_dominant(prompt_len, output_len),
        "workflow_shape": "weka-claude-code-trace",
        "source_repo": "weka",
        "source_table": trace_id[:8],
        "source_dataset": WEKA_DATASET,
        "source_stage": f"phase_{phase_id}",
        "source_model_role": role,
        "model": model,
        "arrival_t": round(float(arrival_t), 3),
        "trace_api_time_s": round(max(float(api_time_s), 0.0), 6),
        "trace_ttft_s": round(max(float(ttft_s), 0.0), 6),
        "trace_completion_t": round(
            float(arrival_t) + max(float(api_time_s), 0.0), 6
        ),
        "parallel_fanout": parallel_fanout,
        "source": WEKA_DATASET,
    }


def annotate_trace_oracle_slack(
    rows: list[dict[str, object]],
    *,
    critical_threshold_ms: float = 1.0,
) -> list[dict[str, object]]:
    """Attach an offline-oracle CPM result derived from measured API times.

    The Weka source records request start ``t`` and end-to-end ``api_time``.
    A stage branch completes at the latest ``t + api_time`` among its
    sequential requests.  The branch Slack is the gap to the latest sibling
    completion in that stage.  This consumes future completion information and
    is therefore an explicit oracle upper bound, never an Algorithm-1 result.
    """

    if not rows:
        return rows
    by_stage_agent: dict[
        tuple[str, int, str], list[dict[str, object]]
    ] = {}
    for row in rows:
        if "trace_api_time_s" not in row:
            raise ValueError(
                f"request {row.get('rid')} has no measured trace_api_time_s"
            )
        key = (
            str(row.get("workflow_id", "")),
            int(row.get("stage_id", 0)),
            str(row.get("agent_id", "")),
        )
        by_stage_agent.setdefault(key, []).append(row)

    branch: dict[tuple[str, int, str], dict[str, float | int]] = {}
    stage_finishes: dict[tuple[str, int], float] = {}
    for key, branch_rows in by_stage_agent.items():
        started_at = min(float(row.get("arrival_t", 0.0)) for row in branch_rows)
        completed_at = max(
            float(row.get("arrival_t", 0.0))
            + max(float(row.get("trace_api_time_s", 0.0)), 0.0)
            for row in branch_rows
        )
        branch[key] = {
            "started_at_s": started_at,
            "completed_at_s": completed_at,
            "duration_ms": max(completed_at - started_at, 0.0) * 1000.0,
            "request_count": len(branch_rows),
            "input_tokens": sum(int(row.get("prompt_len", 0)) for row in branch_rows),
            "output_tokens": sum(int(row.get("output_len", 0)) for row in branch_rows),
        }
        stage_key = (key[0], key[1])
        stage_finishes[stage_key] = max(
            stage_finishes.get(stage_key, 0.0), completed_at
        )

    threshold = max(float(critical_threshold_ms), 0.0)
    for row in rows:
        key = (
            str(row.get("workflow_id", "")),
            int(row.get("stage_id", 0)),
            str(row.get("agent_id", "")),
        )
        stats = branch[key]
        slack_ms = max(
            (
                stage_finishes[(key[0], key[1])]
                - float(stats["completed_at_s"])
            )
            * 1000.0,
            0.0,
        )
        row.update(
            {
                "trace_oracle_slack_ms": round(slack_ms, 3),
                "trace_oracle_critical_agent": slack_ms <= threshold,
                "trace_oracle_branch_started_at_s": round(
                    float(stats["started_at_s"]), 6
                ),
                "trace_oracle_branch_completed_at_s": round(
                    float(stats["completed_at_s"]), 6
                ),
                "trace_oracle_branch_duration_ms": round(
                    float(stats["duration_ms"]), 3
                ),
                "trace_oracle_branch_request_count": int(stats["request_count"]),
                "trace_oracle_branch_input_tokens": int(stats["input_tokens"]),
                "trace_oracle_branch_output_tokens": int(stats["output_tokens"]),
                "trace_oracle_semantics": "weka_api_completion_cpm_v1",
                "trace_oracle_uses_future_information": True,
            }
        )
    return rows


def trace_to_workload(
    row: dict[str, Any],
    *,
    parallel_subagents_only: bool = False,
) -> list[dict[str, object]]:
    trace_id = str(row["id"])
    reqs = row["requests"]
    segments = _segments(reqs)
    out: list[dict[str, object]] = []
    seq = 0

    for phase_id, seg in enumerate(segments, start=1):
        parallel_fanout = seg["kind"] == "subagent" and len(seg["items"]) >= 2
        if parallel_subagents_only and seg["kind"] == "subagent" and not parallel_fanout:
            continue

        if seg["kind"] == "main":
            stage_width = 1
            for turn_idx, r in enumerate(seg["items"]):
                seq += 1
                out.append(
                    _workload_row(
                        trace_id=trace_id,
                        rid=f"{trace_id[:8]}-p{phase_id}-m{turn_idx}",
                        prompt_len=int(r.get("in", 0)),
                        output_len=int(r.get("out", 0)),
                        group_id=seq,
                        agent_id=f"{trace_id[:8]}-main",
                        phase_id=phase_id,
                        stage_width=stage_width,
                        stage_local_agent_index=0,
                        critical_agent=True,
                        model=str(r.get("model", "main")),
                        role="main",
                        arrival_t=float(r.get("t", 0.0)),
                        api_time_s=float(r.get("api_time", 0.0)),
                        ttft_s=float(r.get("ttft", 0.0)),
                        parallel_fanout=False,
                    )
                )
            continue

        stage_width = len(seg["items"])
        for sub_idx, group in enumerate(seg["items"]):
            inner = group.get("requests") or []
            agent_id = f"{trace_id[:8]}-sub{sub_idx + 1}"
            for turn_idx, r in enumerate(inner):
                seq += 1
                out.append(
                    _workload_row(
                        trace_id=trace_id,
                        rid=f"{trace_id[:8]}-p{phase_id}-s{sub_idx}-t{turn_idx}",
                        prompt_len=int(r.get("in", 0)),
                        output_len=int(r.get("out", 0)),
                        group_id=seq,
                        agent_id=agent_id,
                        phase_id=phase_id,
                        stage_width=stage_width,
                        stage_local_agent_index=sub_idx,
                        critical_agent=False,
                        model=str(r.get("model", "subagent")),
                        role="subagent",
                        arrival_t=float(r.get("t", group.get("t", 0.0))),
                        api_time_s=float(r.get("api_time", 0.0)),
                        ttft_s=float(r.get("ttft", 0.0)),
                        parallel_fanout=parallel_fanout,
                    )
                )
    return annotate_trace_oracle_slack(out)


def build_workload(
    input_path: Path,
    *,
    trace_id: str | None = None,
    min_parallel_subagents: int = 0,
    parallel_subagents_only: bool = False,
    limit: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _iter_jsonl(input_path):
        if trace_id is not None and not str(row["id"]).startswith(trace_id):
            continue
        if min_parallel_subagents > 0:
            segs = _segments(row["requests"])
            max_parallel = max(
                (len(s["items"]) for s in segs if s["kind"] == "subagent"),
                default=0,
            )
            if max_parallel < min_parallel_subagents:
                continue
        rows.extend(
            trace_to_workload(
                row,
                parallel_subagents_only=parallel_subagents_only,
            )
        )
        if limit is not None and len({r["workflow_id"] for r in rows}) >= limit:
            break
    return rows


def _select_trace_ids_from_summary(
    summary_csv: Path,
    *,
    limit: int,
    min_parallel_subagents: int = 1,
    min_model_requests: int = 15,
    max_model_requests: int = 120,
    max_timeline_seconds: float = 15000.0,
    required_prefixes: tuple[str, ...] = (),
) -> list[str]:
    rows = list(csv.DictReader(summary_csv.open(newline="", encoding="utf-8")))
    selected: list[str] = []
    seen: set[str] = set()
    for prefix in required_prefixes:
        for row in rows:
            trace_id = str(row["trace_id"])
            if trace_id.startswith(prefix) and trace_id not in seen:
                selected.append(trace_id)
                seen.add(trace_id)
                break
    candidates: list[tuple[int, int, float, str]] = []
    for row in rows:
        trace_id = str(row["trace_id"])
        if trace_id in seen:
            continue
        if int(row["subagent_groups"]) < min_parallel_subagents:
            continue
        model_requests = int(row["model_requests"])
        timeline_seconds = float(row["timeline_seconds"])
        if model_requests < min_model_requests or model_requests > max_model_requests:
            continue
        if timeline_seconds > max_timeline_seconds:
            continue
        candidates.append(
            (
                -int(row["subagent_groups"]),
                model_requests,
                timeline_seconds,
                trace_id,
            )
        )
    candidates.sort()
    for _neg_groups, _requests, _timeline, trace_id in candidates:
        if len(selected) >= limit:
            break
        selected.append(trace_id)
        seen.add(trace_id)
    return selected[:limit]


def build_batch_workload(
    input_path: Path,
    *,
    summary_csv: Path,
    limit: int = 24,
    required_prefixes: tuple[str, ...] = ("8c0579868fbd",),
) -> tuple[list[dict[str, object]], list[str]]:
    trace_ids = _select_trace_ids_from_summary(
        summary_csv,
        limit=limit,
        required_prefixes=required_prefixes,
    )
    wanted = set(trace_ids)
    rows: list[dict[str, object]] = []
    found: set[str] = set()
    for row in _iter_jsonl(input_path):
        trace_id = str(row["id"])
        if trace_id not in wanted:
            continue
        rows.extend(trace_to_workload(row))
        found.add(trace_id)
        if found == wanted:
            break
    missing = [trace_id for trace_id in trace_ids if trace_id not in found]
    if missing:
        raise ValueError(f"missing trace ids in dataset: {missing[:5]}")
    return rows, trace_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Weka cc-traces JSONL to SGLSim workload.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-id", default="c97f752d")
    parser.add_argument("--min-parallel-subagents", type=int, default=0)
    parser.add_argument(
        "--parallel-subagents-only",
        action="store_true",
        help="Keep only fan-out phases with stage_width>=2 plus all main-agent turns.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of workflows.")
    parser.add_argument(
        "--batch-cdf",
        action="store_true",
        help="Build the 24-workflow Weka subagent CDF batch workload.",
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    args = parser.parse_args()

    if args.batch_cdf:
        rows, trace_ids = build_batch_workload(
            args.input,
            summary_csv=args.summary_csv,
            limit=int(args.limit or 24),
        )
        output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_BATCH_OUTPUT
        manifest = {
            "dataset": WEKA_DATASET,
            "workflows": trace_ids,
            "request_rows": len(rows),
            "output": str(output),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        parallel_rows = sum(1 for row in rows if row.get("parallel_fanout"))
        print(
            f"Wrote {len(rows)} request rows across {len(trace_ids)} workflow(s) "
            f"({parallel_rows} parallel fan-out rows) to {output}"
        )
        print(f"Wrote manifest to {manifest_path}")
        return

    rows = build_workload(
        args.input,
        trace_id=args.trace_id or None,
        min_parallel_subagents=args.min_parallel_subagents,
        parallel_subagents_only=args.parallel_subagents_only,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    workflows = len({row["workflow_id"] for row in rows})
    parallel_rows = sum(1 for row in rows if row.get("parallel_fanout"))
    print(
        f"Wrote {len(rows)} request rows across {workflows} workflow(s) "
        f"({parallel_rows} parallel fan-out rows) to {args.output}"
    )


if __name__ == "__main__":
    main()
