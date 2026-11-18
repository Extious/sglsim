#!/usr/bin/env python3
"""Measure SGLang request latency with cache-unique token inputs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def post(url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30177")
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    for request_index in range(args.requests):
        if args.mode == "prefill":
            input_len, output_len = 2048, 1
        else:
            input_len, output_len = 1, 64
        # Independent random sequences avoid SGLang radix-prefix cache hits.
        input_ids = [rng.randrange(1000, 10000) for _ in range(input_len)]
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": False,
        }
        body, elapsed = post(args.url, payload)
        meta = body.get("meta_info", {})
        records.append(
            {
                "request_index": request_index,
                "client_latency_s": elapsed,
                "server_e2e_latency_s": meta.get("e2e_latency"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "cached_tokens": meta.get("cached_tokens"),
            }
        )

    latencies = [float(record["client_latency_s"]) for record in records]
    server_latencies = [
        float(record["server_e2e_latency_s"])
        for record in records
        if record["server_e2e_latency_s"] is not None
    ]
    result = {
        "label": args.label,
        "mode": args.mode,
        "requests": args.requests,
        "seed": args.seed,
        "input_tokens": 2048 if args.mode == "prefill" else 1,
        "max_new_tokens": 1 if args.mode == "prefill" else 64,
        "ignore_eos": True,
        "client_latency_s": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": p95(latencies),
            "min": min(latencies),
            "max": max(latencies),
        },
        "server_e2e_latency_s": {
            "mean": statistics.mean(server_latencies),
            "median": statistics.median(server_latencies),
            "p95": p95(server_latencies),
            "min": min(server_latencies),
            "max": max(server_latencies),
        },
        "all_cached_tokens": [record["cached_tokens"] for record in records],
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
