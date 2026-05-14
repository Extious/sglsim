from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetRow:
    prompt: Any
    prompt_len: int
    output_len: int
    extra_request_body: dict[str, Any] | None = field(default_factory=dict)
    timestamp: float | None = None


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_input_text: int
    total_input_vision: int
    total_output: int
    total_output_retokenized: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    output_throughput_retokenized: float
    total_throughput: float
    total_throughput_retokenized: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    max_itl_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    std_e2e_latency_ms: float
    p90_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float
    max_output_tokens_per_s: float = 0.0
    max_concurrent_requests: int = 0


def inject_simulation_params(
    row: DatasetRow,
    *,
    created_time: float,
    total_request: int,
    queue_start: float | None = None,
) -> None:
    if row.extra_request_body is None:
        row.extra_request_body = {}
    sampling_params = row.extra_request_body.setdefault("sampling_params", {})
    custom_params = sampling_params.setdefault("custom_params", {})
    simulation = {
        "created_time": created_time,
        "total_request": total_request,
    }
    if queue_start is not None:
        simulation["queue_start"] = queue_start
    custom_params["simulation"] = simulation


def load_simulator_metrics(output_dir: str) -> BenchmarkMetrics:
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Simulator metrics file not found: {metrics_path}")
    with open(metrics_path) as f:
        data = json.load(f)
    metric_fields = BenchmarkMetrics.__dataclass_fields__
    return BenchmarkMetrics(**{name: data.get(name, -1) for name in metric_fields})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SGLSim benchmark helper")
    parser.add_argument("--simulator-output-dir", default="/tmp/sglsim/output")
    parser.add_argument("--load-metrics", action="store_true")
    args = parser.parse_args(argv)
    if args.load_metrics:
        print(json.dumps(load_simulator_metrics(args.simulator_output_dir).__dict__))


if __name__ == "__main__":
    main()

