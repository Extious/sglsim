from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class RequestStats:
    rid: str = ""
    last_event_time: float = 0
    input_length: int = 0
    output_length: int = 0
    final_reused_tokens: int = 0
    prefetch_complete_tokens: int = 0
    queue_start: float = -1
    queue_end: float = -1
    created_time: float = -1
    gen_token_latencies: List[float] = field(default_factory=list)

    def e2e_latency(self) -> float:
        return sum(self.gen_token_latencies)

    def ttft(self) -> float:
        return self.gen_token_latencies[0] if self.gen_token_latencies else 0

    def itls(self) -> List[float]:
        return self.gen_token_latencies[1:]


def _mean_ms(values: List[float]) -> float:
    return float(np.mean(values) * 1000) if values else 0.0


def _median_ms(values: List[float]) -> float:
    return float(np.median(values) * 1000) if values else 0.0


def _percentile_ms(values: List[float], percentile: float) -> float:
    return float(np.percentile(values, percentile) * 1000) if values else 0.0


def calc_metrics(stats: List[RequestStats], num_warmup: int = 0) -> Dict[str, Any]:
    metric_stats = stats[num_warmup:] if len(stats) > num_warmup else stats
    completed = len(metric_stats)
    total_input = sum(item.input_length for item in metric_stats)
    total_output = sum(item.output_length for item in metric_stats)
    total_reused_tokens = sum(item.final_reused_tokens for item in metric_stats)
    total_disk_hit_tokens = sum(item.prefetch_complete_tokens for item in metric_stats)
    queue_durs = [
        item.queue_end - item.queue_start
        for item in metric_stats
        if item.queue_start >= 0 and item.queue_end >= 0
    ]
    ttfts = [item.ttft() for item in metric_stats if item.gen_token_latencies]
    itls = [lat for item in metric_stats for lat in item.itls()]
    e2e_latencies = [item.e2e_latency() for item in metric_stats]
    duration = max(
        (item.created_time + item.e2e_latency() for item in metric_stats),
        default=0,
    )
    duration = max(duration, 1e-9)

    return {
        "num_requests": len(metric_stats),
        "completed": completed,
        "total_input": total_input,
        "total_input_text": -1,
        "total_input_vision": -1,
        "total_output": total_output,
        "total_output_retokenized": -1,
        "duration": duration,
        "request_throughput": completed / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "output_throughput_retokenized": -1,
        "total_throughput": (total_input + total_output) / duration,
        "total_throughput_retokenized": -1,
        "prefix_cache_reused_ratio": (
            0 if total_input == 0 else total_reused_tokens / total_input
        ),
        "disk_prefetch_ratio": (
            0 if total_input == 0 else total_disk_hit_tokens / total_input
        ),
        "mean_queue_ms": _mean_ms(queue_durs),
        "mean_ttft_ms": _mean_ms(ttfts),
        "median_ttft_ms": _median_ms(ttfts),
        "std_ttft_ms": float(np.std(ttfts) * 1000) if ttfts else 0.0,
        "p90_ttft_ms": _percentile_ms(ttfts, 90),
        "p95_ttft_ms": _percentile_ms(ttfts, 95),
        "p99_ttft_ms": _percentile_ms(ttfts, 99),
        "mean_tpot_ms": _mean_ms(itls),
        "median_tpot_ms": _median_ms(itls),
        "std_tpot_ms": float(np.std(itls) * 1000) if itls else 0.0,
        "p90_tpot_ms": _percentile_ms(itls, 90),
        "p95_tpot_ms": _percentile_ms(itls, 95),
        "p99_tpot_ms": _percentile_ms(itls, 99),
        "mean_itl_ms": _mean_ms(itls),
        "median_itl_ms": _median_ms(itls),
        "std_itl_ms": float(np.std(itls) * 1000) if itls else 0.0,
        "p90_itl_ms": _percentile_ms(itls, 90),
        "p95_itl_ms": _percentile_ms(itls, 95),
        "p99_itl_ms": _percentile_ms(itls, 99),
        "max_itl_ms": float(max(itls) * 1000) if itls else 0.0,
        "mean_e2e_latency_ms": _mean_ms(e2e_latencies),
        "median_e2e_latency_ms": _median_ms(e2e_latencies),
        "std_e2e_latency_ms": float(np.std(e2e_latencies) * 1000)
        if e2e_latencies
        else 0.0,
        "p90_e2e_latency_ms": _percentile_ms(e2e_latencies, 90),
        "p95_e2e_latency_ms": _percentile_ms(e2e_latencies, 95),
        "p99_e2e_latency_ms": _percentile_ms(e2e_latencies, 99),
        "concurrency": sum(e2e_latencies) / duration,
        "max_output_tokens_per_s": -1,
        "max_concurrent_requests": -1,
    }


def dump_simulator_outputs(
    output_dir: str,
    stats: List[RequestStats],
    iterations: List[Dict[str, Any]],
    num_warmup: int = 0,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    metrics = calc_metrics(stats, num_warmup=num_warmup)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
        f.write("\n")

    with open(os.path.join(output_dir, "request.jsonl"), "w") as f:
        for item in stats:
            f.write(json.dumps(asdict(item)) + "\n")

    with open(os.path.join(output_dir, "iteration.jsonl"), "w") as f:
        for item in iterations:
            f.write(json.dumps(item) + "\n")

    return metrics
