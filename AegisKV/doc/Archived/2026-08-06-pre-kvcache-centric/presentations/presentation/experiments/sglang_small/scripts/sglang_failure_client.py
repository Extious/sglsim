#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_json_or_text(raw: bytes) -> object:
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


def request_json(
    url: str,
    payload: dict[str, object] | None,
    timeout_s: float,
) -> tuple[int, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="GET" if payload is None else "POST",
    )
    try:
        with LOCAL_OPENER.open(req, timeout=timeout_s) as resp:
            return resp.status, parse_json_or_text(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, parse_json_or_text(exc.read())


def safe_request_json(
    url: str,
    payload: dict[str, object] | None,
    timeout_s: float,
) -> tuple[int, object]:
    try:
        return request_json(url, payload, timeout_s)
    except Exception as exc:  # noqa: BLE001 - diagnostic client
        return 0, {"error": repr(exc)}


def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            status, _ = request_json(f"{base_url}/health", None, timeout_s=3)
            if status < 500:
                return
        except Exception as exc:  # noqa: BLE001 - diagnostic client
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready in {timeout_s}s; last_error={last_error}")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[idx])


def build_prompt(
    request_index: int, prompt_repeat: int, unique_request_prefix: bool = False
) -> str:
    if unique_request_prefix:
        prefix = (
            f"{request_index} unique AegisKV failover probe. "
            "You are a systems researcher evaluating resilient KV-cache serving. "
        )
    else:
        prefix = (
            "You are a systems researcher evaluating resilient KV-cache serving. "
            f"Request index {request_index}. "
        )
    unit = (
        "Explain why multi-agent serving needs backup, restore, and retry paths "
        "that preserve user-visible completion continuity. "
    )
    return prefix + unit * prompt_repeat


def extract_response_fields(body: object) -> tuple[str, dict[str, object], dict[str, object], dict[str, object]]:
    text = ""
    usage: dict[str, object] = {}
    sglext: dict[str, object] = {}
    failover: dict[str, object] = {}
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            text = str(choices[0].get("text", ""))
        if isinstance(body.get("usage"), dict):
            usage = dict(body["usage"])
        ext_obj = body.get("sglext") or body.get("sgl_ext") or {}
        if isinstance(ext_obj, dict):
            sglext = dict(ext_obj)
            maybe_failover = ext_obj.get("failover")
            if isinstance(maybe_failover, dict):
                failover = dict(maybe_failover)
    return text, usage, sglext, failover


def completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_index: int,
    rid_prefix: str,
    routed_dp_rank: int,
    timeout_s: float,
    force_full_generation: bool,
) -> dict[str, object]:
    rid = f"{rid_prefix}-{request_index:04d}"
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "rid": rid,
        "routed_dp_rank": routed_dp_rank,
    }
    if force_full_generation:
        payload["ignore_eos"] = True
        payload["min_tokens"] = max_tokens

    started_at_s = time.time()
    try:
        status, body = request_json(f"{base_url}/v1/completions", payload, timeout_s=timeout_s)
        finished_at_s = time.time()
        text, usage, sglext, failover = extract_response_fields(body)
        return {
            "request_index": request_index,
            "rid": rid,
            "routed_dp_rank": routed_dp_rank,
            "status": status,
            "ok": 200 <= status < 300,
            "latency_s": finished_at_s - started_at_s,
            "started_at_s": started_at_s,
            "finished_at_s": finished_at_s,
            "prompt_chars": len(prompt),
            "response_chars": len(text),
            "usage": usage,
            "sglext": sglext,
            "failover": failover,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic client
        finished_at_s = time.time()
        return {
            "request_index": request_index,
            "rid": rid,
            "routed_dp_rank": routed_dp_rank,
            "status": 0,
            "ok": False,
            "latency_s": finished_at_s - started_at_s,
            "started_at_s": started_at_s,
            "finished_at_s": finished_at_s,
            "prompt_chars": len(prompt),
            "response_chars": 0,
            "usage": {},
            "sglext": {},
            "failover": {},
            "error": repr(exc),
        }


def read_jsonl(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def summarize(
    rows: list[dict[str, object]],
    events: list[dict[str, object]],
    admin_failure: dict[str, object],
    admin_recovery: dict[str, object] | None,
) -> dict[str, object]:
    ok_rows = [row for row in rows if bool(row.get("ok"))]
    latencies = [float(row["latency_s"]) for row in ok_rows]
    failover_rows = [row for row in ok_rows if isinstance(row.get("failover"), dict) and row["failover"]]
    event_counts: dict[str, int] = {}
    captured_request_count = 0
    dispatched_request_count = 0
    backed_up_tokens = 0
    pre_failover_tokens = 0
    for event in events:
        event_name = str(event.get("event", ""))
        event_counts[event_name] = event_counts.get(event_name, 0) + 1
        if event_name == "gpu_failure_simulated":
            captured_request_count += int(event.get("num_requests") or 0)
        if event_name == "failover_dispatched":
            dispatched_request_count += int(event.get("num_requests") or 0)
            requests = event.get("requests")
            if isinstance(requests, list):
                for req in requests:
                    if isinstance(req, dict):
                        backed_up_tokens += int(req.get("pre_failover_backed_up_tokens") or 0)
                        pre_failover_tokens += int(req.get("pre_failover_output_tokens") or 0)

    response_backed_up_tokens = 0
    response_pre_failover_tokens = 0
    for row in failover_rows:
        failover = row.get("failover") if isinstance(row.get("failover"), dict) else {}
        response_backed_up_tokens += int(failover.get("pre_failover_backed_up_tokens") or 0)
        response_pre_failover_tokens += int(failover.get("pre_failover_output_tokens") or 0)

    return {
        "count": len(rows),
        "success_count": len(ok_rows),
        "failure_count": len(rows) - len(ok_rows),
        "failover_response_count": len(failover_rows),
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "event_count": len(events),
        "event_counts": event_counts,
        "captured_request_count": captured_request_count,
        "dispatched_request_count": dispatched_request_count,
        "event_pre_failover_output_tokens": pre_failover_tokens,
        "event_pre_failover_backed_up_tokens": backed_up_tokens,
        "response_pre_failover_output_tokens": response_pre_failover_tokens,
        "response_pre_failover_backed_up_tokens": response_backed_up_tokens,
        "failure_injection_status": admin_failure.get("status", 0),
        "failure_injection_body": admin_failure.get("body"),
        "recovery_status": 0 if admin_recovery is None else admin_recovery.get("status", 0),
        "recovery_body": None if admin_recovery is None else admin_recovery.get("body"),
    }


def write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    flat_summary = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in summary.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_summary.keys()))
        writer.writeheader()
        writer.writerow(flat_summary)


def write_candidate_csv(path: Path, run_id: str, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "policy",
                "request_count",
                "success_count",
                "failover_response_count",
                "captured_request_count",
                "dispatched_request_count",
                "latency_p95_s",
                "latency_p99_s",
                "pre_failover_backed_up_tokens",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "policy": "aegiskv_sglang_dp2_failure_smoke",
                "request_count": summary["count"],
                "success_count": summary["success_count"],
                "failover_response_count": summary["failover_response_count"],
                "captured_request_count": summary["captured_request_count"],
                "dispatched_request_count": summary["dispatched_request_count"],
                "latency_p95_s": summary["latency_p95_s"],
                "latency_p99_s": summary["latency_p99_s"],
                "pre_failover_backed_up_tokens": summary[
                    "response_pre_failover_backed_up_tokens"
                ],
                "note": "small-scale SGLang failure smoke; calibrates control path, not final large-scale figure evidence",
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-repeat", type=int, default=64)
    parser.add_argument("--fail-dp-rank", type=int, default=0)
    parser.add_argument("--failure-delay-s", type=float, default=4.0)
    parser.add_argument("--request-timeout-s", type=float, default=900)
    parser.add_argument("--ready-timeout-s", type=float, default=900)
    parser.add_argument("--event-flush-wait-s", type=float, default=1.0)
    parser.add_argument("--rid-prefix", default="aegiskv-failure-smoke")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--allow-eos-stop", action="store_true")
    parser.add_argument("--unique-request-prefix", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--failover-events-jsonl", type=Path)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    wait_ready(base_url, args.ready_timeout_s)

    prompts = [
        build_prompt(idx, args.prompt_repeat, args.unique_request_prefix)
        for idx in range(args.num_requests)
    ]
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.num_requests) as executor:
        futures = [
            executor.submit(
                completion,
                base_url,
                args.model,
                prompt,
                args.max_tokens,
                idx,
                args.rid_prefix,
                args.fail_dp_rank,
                args.request_timeout_s,
                not args.allow_eos_stop,
            )
            for idx, prompt in enumerate(prompts)
        ]
        time.sleep(args.failure_delay_s)
        failure_t0 = time.time()
        failure_status, failure_body = safe_request_json(
            f"{base_url}/simulate_gpu_failure",
            {"dp_rank": args.fail_dp_rank},
            timeout_s=60,
        )
        failure_t1 = time.time()
        admin_failure = {
            "status": failure_status,
            "body": failure_body,
            "latency_s": failure_t1 - failure_t0,
        }
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: int(row.get("request_index", 0)))
    with args.output_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    admin_recovery = None
    if args.recover:
        recovery_t0 = time.time()
        recovery_status, recovery_body = safe_request_json(
            f"{base_url}/simulate_gpu_recovery",
            {"dp_rank": args.fail_dp_rank},
            timeout_s=60,
        )
        recovery_t1 = time.time()
        admin_recovery = {
            "status": recovery_status,
            "body": recovery_body,
            "latency_s": recovery_t1 - recovery_t0,
        }

    if args.event_flush_wait_s > 0:
        time.sleep(args.event_flush_wait_s)

    events = read_jsonl(args.failover_events_jsonl)
    summary = summarize(rows, events, admin_failure, admin_recovery)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.summary_csv is not None:
        write_summary_csv(args.summary_csv, summary)
    if args.candidate_csv is not None:
        run_id = args.run_id or f"dp2_failure_smoke_{int(time.time())}"
        write_candidate_csv(args.candidate_csv, run_id, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    success = (
        int(summary["success_count"]) == args.num_requests
        and int(summary["failure_injection_status"]) < 300
        and int(summary["captured_request_count"]) > 0
        and int(summary["failover_response_count"]) > 0
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
