from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglsim.fault_tolerance.config import ExperimentConfig, Strategy
from sglsim.fault_tolerance.load_model import (
    failover_recovery_hit_rate,
    prefill_cache_hit_rate,
    steady_state_hicache_hits,
)
from sglsim.fault_tolerance.reference import DEFAULT_REFERENCE, calibrated_output_throughput


@dataclass(frozen=True)
class SteadyStateResult:
    concurrency: int
    strategy: Strategy
    metrics: dict[str, Any]


def _simulate_at_concurrency(
    config: ExperimentConfig,
    *,
    concurrency: int,
) -> dict[str, Any]:
    strategy = config.server.strategy
    client = config.client.fixed
    dp_size = max(1, config.server.server.dp_size)
    input_len = client.input_len
    output_len = client.output_len
    shared_prefix = max(0, client.shared_prefix_len)
    workers = max(1, concurrency)
    num_requests = max(workers * 4, min(client.num_requests, 96))

    output_tput = calibrated_output_throughput(strategy, concurrency=workers, dp_size=dp_size)
    duration = max((num_requests * output_len) / max(output_tput, 1e-9), 1e-9)

    hits = steady_state_hicache_hits(input_len=input_len, shared_prefix_len=shared_prefix)
    prefill_hit = prefill_cache_hit_rate(hits, input_len=input_len)

    return {
        "strategy": strategy,
        "concurrency": workers,
        "completed": num_requests,
        "duration": duration,
        "output_throughput": output_tput,
        "input_throughput": (num_requests * input_len) / duration,
        "request_throughput": num_requests / duration,
        "mean_prefill_cache_hit_rate": prefill_hit,
        "mean_l1_match": float(hits.l1_match),
        "mean_l2_match": float(hits.l2_match),
        "mean_host_backup_match": 0.0,
        "mean_peer_replica_match": 0.0,
        "reference_output_throughput": DEFAULT_REFERENCE.output_throughput_tok_s,
    }


def sweep_concurrency(
    base_config: ExperimentConfig,
    *,
    strategies: list[Strategy],
    concurrency_levels: list[int],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_strategy: dict[str, list[dict[str, Any]]] = {s: [] for s in strategies}
    for strategy in strategies:
        config = base_config.with_strategy(strategy)
        for concurrency in concurrency_levels:
            metrics = _simulate_at_concurrency(config, concurrency=concurrency)
            rows.append(metrics)
            by_strategy[strategy].append(metrics)

    summary = {
        "concurrency_levels": concurrency_levels,
        "reference": {
            "output_throughput_tok_s": DEFAULT_REFERENCE.output_throughput_tok_s,
            "concurrency": DEFAULT_REFERENCE.reference_concurrency,
            "dp_size": DEFAULT_REFERENCE.reference_dp_size,
            "source": "benchmark/synthesis_requests/results/with_hicache_sglang",
        },
        "strategies": by_strategy,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "concurrency_sweep.json").write_text(json.dumps(summary, indent=2) + "\n")
        fieldnames = [
            "strategy",
            "concurrency",
            "output_throughput",
            "input_throughput",
            "request_throughput",
            "mean_prefill_cache_hit_rate",
            "mean_l1_match",
            "mean_l2_match",
            "reference_output_throughput",
            "duration",
        ]
        with (root / "concurrency_sweep.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    return summary
