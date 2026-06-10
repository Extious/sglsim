from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sglsim.srt.simulation.types import RequestStats


@dataclass
class BenchmarkMetrics:
    completed: int
    duration: float
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    total_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    prefix_cache_reused_ratio: float = 0.0


def calc_bench_metrics(
    requests: list[RequestStats],
    duration_s: float,
) -> dict:
    ttfts: list[float] = []
    tpots: list[float] = []
    itls: list[float] = []
    e2e: list[float] = []
    total_input = 0
    total_output = 0
    completed = 0
    total_reused = 0
    measured = 0
    total_dur = max(duration_s, 1e-9)

    for req in requests:
        if not req.is_complete() or not req.gen_token_latencies:
            continue
        measured += 1
        completed += 1
        ttfts.append(req.gen_token_latencies[0])
        if len(req.gen_token_latencies) > 1:
            tpots.append(float(np.mean(req.gen_token_latencies[1:])))
        itls.extend(req.gen_token_latencies[1:])
        if req.created_time >= 0:
            e2e.append(max(req.last_event_time - req.created_time, 0.0))
        else:
            e2e.append(sum(req.gen_token_latencies))
        total_input += req.input_length
        total_output += req.output_length
        total_reused += req.cache_hit_tokens
        total_dur = max(total_dur, req.last_event_time)

    m = BenchmarkMetrics(
        completed=completed,
        duration=total_dur,
        total_input=total_input,
        total_output=total_output,
        request_throughput=measured / total_dur,
        input_throughput=total_input / total_dur,
        output_throughput=total_output / total_dur,
        total_throughput=(total_input + total_output) / total_dur,
        mean_ttft_ms=float(np.mean(ttfts or 0)) * 1000,
        median_ttft_ms=float(np.median(ttfts or 0)) * 1000,
        p99_ttft_ms=float(np.percentile(ttfts or 0, 99)) * 1000,
        mean_tpot_ms=float(np.mean(tpots or 0)) * 1000,
        median_tpot_ms=float(np.median(tpots or 0)) * 1000,
        p99_tpot_ms=float(np.percentile(tpots or 0, 99)) * 1000,
        mean_itl_ms=float(np.mean(itls or 0)) * 1000,
        median_itl_ms=float(np.median(itls or 0)) * 1000,
        p95_itl_ms=float(np.percentile(itls or 0, 95)) * 1000,
        p99_itl_ms=float(np.percentile(itls or 0, 99)) * 1000,
        max_itl_ms=float(np.max(itls or 0)) * 1000,
        mean_e2e_latency_ms=float(np.mean(e2e or 0)) * 1000,
        median_e2e_latency_ms=float(np.median(e2e or 0)) * 1000,
        p90_e2e_latency_ms=float(np.percentile(e2e or 0, 90)) * 1000,
        p99_e2e_latency_ms=float(np.percentile(e2e or 0, 99)) * 1000,
        prefix_cache_reused_ratio=(
            0.0 if total_input == 0 else total_reused / total_input
        ),
    )
    return {k: getattr(m, k) for k in m.__dataclass_fields__}
