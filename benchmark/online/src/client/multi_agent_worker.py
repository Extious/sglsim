from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from client.fixed_worker import (  # noqa: E402
    PromptBuilder,
    _build_completion_payload,
    _extract_cache_details,
    _extract_failover_details,
    _extract_usage_int,
    _post_control_event,
)
from client.metrics_utils import (  # noqa: E402
    write_cache_hit_records,
    write_job_summary_records,
    write_task_summary_records,
    write_trace_records,
)
from sglsim.fault_tolerance.aegiskv_simulator import (  # noqa: E402
    MultiAgentWorkflowConfig,
    build_two_round_coding_workflow,
)


def _load_workflow_config(raw: str) -> MultiAgentWorkflowConfig:
    data = json.loads(raw) if raw.strip() else {}
    if not isinstance(data, dict):
        data = {}
    known = {
        field: data[field]
        for field in asdict(MultiAgentWorkflowConfig()).keys()
        if field in data
    }
    return MultiAgentWorkflowConfig(**known)


def _completion_endpoint(server_url: str) -> str:
    endpoint = server_url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    return endpoint + "/completions"


def _request_cache_row(
    *,
    workflow_id: str,
    local_segment: str,
    role: str,
    rid: str,
    response_body: dict[str, Any],
    usage: dict[str, Any],
) -> dict[str, Any]:
    cache_details = _extract_cache_details(response_body)
    if not cache_details:
        cache_details = _extract_cache_details({"usage": usage})
    failover = _extract_failover_details(response_body)
    if not failover:
        failover = _extract_failover_details({"usage": usage})
    pre_output = _extract_usage_int(failover, "pre_failover_output_tokens")
    pre_backed_up = _extract_usage_int(failover, "pre_failover_backed_up_tokens")
    is_retried = bool(pre_output > 0 or pre_backed_up > 0)
    return {
        "job_id": workflow_id,
        "task_label": local_segment,
        "agent_role": role,
        "rid": rid,
        "l1_match": int(cache_details.get("device", 0) or 0),
        "l2_match": int(cache_details.get("host", 0) or 0),
        "remote_match": int(cache_details.get("storage_query", 0) or 0),
        "remote_prefetch": int(cache_details.get("storage", 0) or 0),
        "reused_device": int(cache_details.get("reused_device", 0) or 0),
        "reused_host": int(cache_details.get("reused_host", 0) or 0),
        "reused_storage": int(cache_details.get("reused_storage", 0) or 0),
        "is_failover_retried": "True" if is_retried else "",
        "pre_failover_output_tokens": str(pre_output) if pre_output > 0 else "",
        "pre_failover_backed_up_tokens": (
            str(pre_backed_up) if pre_backed_up > 0 else ""
        ),
    }


def _write_segment_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "workflow_id",
        "segment_id",
        "local_segment",
        "phase_id",
        "agent_id",
        "parent_agent_id",
        "role",
        "segment_type",
        "lifecycle_state",
        "tokens",
        "output_tokens",
        "duration_s",
        "status",
        "rid",
        "prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _run_segment_request(
    *,
    endpoint: str,
    model_path: str,
    prompt_builder: PromptBuilder,
    workflow_id: str,
    segment,
    request_index: int,
    output_len: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    local_segment = segment.segment_id.rsplit(":", 1)[1]
    prompt = prompt_builder.build(request_index)
    payload = _build_completion_payload(
        model_path=model_path,
        prompt=prompt,
        output_len=output_len,
        ignore_eos=ignore_eos,
    )
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY_API_KEY",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    wall_start = time.time()
    status = "completed"
    error = ""
    response_body: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    rid = ""
    try:
        with opener.open(req, timeout=180) as resp:
            raw_body = json.loads(resp.read().decode("utf-8"))
        response_body = raw_body if isinstance(raw_body, dict) else {}
        usage = response_body.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        rid = str(response_body.get("id", "") or "")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        status = "failed"
        error = str(exc)[:200]
    duration = round(time.time() - wall_start, 3)
    prompt_tokens = _extract_usage_int(usage, "prompt_tokens") or int(segment.tokens)
    completion_tokens = _extract_usage_int(usage, "completion_tokens") or int(output_len)
    prompt_details = usage.get("prompt_tokens_details", {})
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    cached_prompt_tokens = _extract_usage_int(prompt_details, "cached_tokens")
    return {
        "workflow_id": workflow_id,
        "segment_id": segment.segment_id,
        "local_segment": local_segment,
        "phase_id": segment.phase_id,
        "agent_id": segment.agent_id,
        "parent_agent_id": segment.parent_agent_id,
        "role": segment.role,
        "segment_type": segment.segment_type.value,
        "lifecycle_state": segment.lifecycle_state.value,
        "tokens": int(segment.tokens),
        "output_tokens": int(output_len),
        "duration_s": duration,
        "status": status,
        "rid": rid,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "error": error,
        "response_body": response_body,
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="AegisKV multi-agent online worker")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--workflow-config-json", default="{}")
    parser.add_argument("--num-workflows", type=int, default=0)
    parser.add_argument("--app-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ignore-eos", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-url", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_workflow_config(args.workflow_config_json)
    if args.num_workflows > 0:
        cfg = MultiAgentWorkflowConfig(**{**asdict(cfg), "num_workflows": args.num_workflows})
    output_len = int(args.output_len or cfg.output_tokens_per_invocation)
    workflows = [
        build_two_round_coding_workflow(f"wf-{index}", cfg)
        for index in range(max(1, cfg.num_workflows))
    ]
    if args.dry_run:
        planned = [
            {
                "workflow_id": segments[0].workflow_id,
                "segments": [
                    {
                        "local_segment": segment.segment_id.rsplit(":", 1)[1],
                        "agent_id": segment.agent_id,
                        "role": segment.role,
                        "tokens": segment.tokens,
                    }
                    for segment in segments
                ],
            }
            for segments in workflows
        ]
        print(json.dumps(planned, indent=2))
        return 0

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    endpoint = _completion_endpoint(args.server_url)
    workflow_records: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    cache_rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    semaphore = threading.Semaphore(max(1, int(args.app_workers)))

    def run_workflow(index: int, segments) -> None:
        workflow_id = f"wf-{index}"
        time.sleep(max(0.0, float(cfg.workflow_start_stagger_s)) * index)
        with semaphore:
            prompt_builders: dict[int, PromptBuilder] = {}
            timings: list[dict[str, Any]] = []
            workflow_status = "completed"
            workflow_error = ""
            start = time.time()
            for seg_index, segment in enumerate(segments):
                builder = prompt_builders.setdefault(
                    int(segment.tokens),
                    PromptBuilder(args.model_path, int(segment.tokens), args.seed + index),
                )
                segment_output_len = int(getattr(segment, "output_tokens", output_len) or output_len)
                result = _run_segment_request(
                    endpoint=endpoint,
                    model_path=args.model_path,
                    prompt_builder=builder,
                    workflow_id=workflow_id,
                    segment=segment,
                    request_index=index * 1000 + seg_index,
                    output_len=segment_output_len,
                    ignore_eos=bool(args.ignore_eos),
                )
                local_segment = result["local_segment"]
                _post_control_event(
                    args.control_url,
                    "task_completed" if result["status"] == "completed" else "task_failed",
                    {
                        "job_id": workflow_id,
                        "job": workflow_id,
                        "worker_id": str(index),
                        "task_label": local_segment,
                        "agent_role": result["role"],
                        "duration_s": result["duration_s"],
                        **({"error": result["error"]} if result["error"] else {}),
                    },
                )
                timing = {
                    "agent": result["agent_id"],
                    "task_label": local_segment,
                    "duration_s": result["duration_s"],
                    "llm_calls": 1 if result["status"] == "completed" else 0,
                    "prompt_tokens": result["prompt_tokens"],
                    "cached_prompt_tokens": result["cached_prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "prefill_cache_hit_rate": (
                        round(result["cached_prompt_tokens"] / result["prompt_tokens"], 4)
                        if result["prompt_tokens"]
                        else ""
                    ),
                    "rid": result["rid"],
                    "error": result["error"],
                }
                timings.append(timing)
                cache_row = _request_cache_row(
                    workflow_id=workflow_id,
                    local_segment=local_segment,
                    role=result["role"],
                    rid=result["rid"],
                    response_body=result["response_body"],
                    usage=result["usage"],
                )
                row_for_csv = dict(result)
                row_for_csv.pop("response_body", None)
                row_for_csv.pop("usage", None)
                with lock:
                    segment_rows.append(row_for_csv)
                    cache_rows.append(cache_row)
                if result["status"] != "completed":
                    workflow_status = "failed"
                    workflow_error = result["error"]
                    break
            wall_time = round(time.time() - start, 3)
            record = {
                "topic_id": workflow_id,
                "topic": workflow_id,
                "year": "",
                "worker_id": str(index),
                "status": workflow_status,
                "error": workflow_error,
                "agent_timings": timings,
                "summary": {
                    "wall_time_s": wall_time,
                    "llm_calls": sum(int(t["llm_calls"]) for t in timings),
                    "prompt_tokens": sum(int(t["prompt_tokens"]) for t in timings),
                    "prefill_cached_tokens": sum(
                        int(t["cached_prompt_tokens"]) for t in timings
                    ),
                    "total_tokens": sum(int(t["total_tokens"]) for t in timings),
                },
            }
            _post_control_event(
                args.control_url,
                "job_completed" if workflow_status == "completed" else "job_failed",
                {
                    "job_id": workflow_id,
                    "job": workflow_id,
                    "worker_id": str(index),
                    "status": workflow_status,
                    "wall_time_s": wall_time,
                    **({"error": workflow_error} if workflow_error else {}),
                },
            )
            with lock:
                workflow_records.append(record)

    threads = [
        threading.Thread(target=run_workflow, args=(index, segments), daemon=True)
        for index, segments in enumerate(workflows)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    workflow_records.sort(key=lambda row: int(str(row["topic_id"]).split("-")[-1]))
    segment_rows.sort(
        key=lambda row: (
            int(str(row["workflow_id"]).split("-")[-1]),
            int(str(row["local_segment"]).lstrip("S") or 0),
        )
    )
    write_trace_records(workflow_records, out / "online_workflow_trace.jsonl")
    write_job_summary_records(workflow_records, out / "job_summary.csv")
    write_task_summary_records(workflow_records, out / "task_summary.csv")
    write_cache_hit_records(cache_rows, out / "cache_hits.csv")
    _write_segment_rows(out / "online_segments.csv", segment_rows)
    (out / "online_manifest.json").write_text(
        json.dumps(
            {
                "client_mode": "aegiskv_workflow",
                "num_workflows": cfg.num_workflows,
                "segments_per_workflow": len(workflows[0]) if workflows else 0,
                "app_workers": int(args.app_workers),
                "output_len": output_len,
                "completed_workflows": sum(
                    1 for row in workflow_records if row["status"] == "completed"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if all(row["status"] == "completed" for row in workflow_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
