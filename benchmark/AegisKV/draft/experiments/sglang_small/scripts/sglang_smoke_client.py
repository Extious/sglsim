#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(url: str, payload: dict[str, object] | None, timeout_s: float) -> tuple[int, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    with LOCAL_OPENER.open(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        if not body:
            return resp.status, None
        return resp.status, json.loads(body)


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


def completion(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, object]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.time()
    status, body = request_json(f"{base_url}/v1/completions", payload, timeout_s=180)
    t1 = time.time()
    text = ""
    usage = {}
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            text = str(choices[0].get("text", ""))
        if isinstance(body.get("usage"), dict):
            usage = dict(body["usage"])
    return {
        "status": status,
        "latency_s": t1 - t0,
        "prompt_chars": len(prompt),
        "response_chars": len(text),
        "usage": usage,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return float(ordered[idx])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--ready-timeout-s", type=float, default=600)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    wait_ready(base_url, args.ready_timeout_s)

    prompts = [
        (
            "You are a concise systems researcher. Summarize why workflow-aware "
            f"KV backup matters for multi-agent serving. Request {idx}."
        )
        for idx in range(args.num_requests)
    ]
    rows: list[dict[str, object]] = []
    with args.output_jsonl.open("w") as f:
        for idx, prompt in enumerate(prompts):
            row = completion(base_url, args.model, prompt, args.max_tokens)
            row["request_index"] = idx
            row["created_time_s"] = time.time()
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)

    latencies = [float(row["latency_s"]) for row in rows if int(row["status"]) < 500]
    summary = {
        "count": len(rows),
        "success_count": len(latencies),
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["success_count"] == args.num_requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
