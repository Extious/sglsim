# Native Simulator Phase 2 State Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add simulator state and metrics primitives independent of `Scheduler` and `ModelRunner`.

**Architecture:** Keep logical clock and request accounting in small standalone modules. Later phases call these modules from runtime integration points.

**Tech Stack:** Python dataclasses, JSON, pytest.

---

## Dependencies

Complete Phase 1 first.

## Files

- Create: `python/sglang/srt/simulator/state.py`
- Create: `python/sglang/srt/simulator/metrics.py`
- Test: `test/registered/unit/simulator/test_state_metrics.py`

## Task 1: Simulator State

- [ ] **Step 1: Write failing state tests**

Create `test/registered/unit/simulator/test_state_metrics.py` with this first section:

```python
import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.simulator.metrics import RequestStats, calc_metrics, dump_simulator_outputs
from sglang.srt.simulator.state import SimulatorState


class TestSimulatorState(unittest.TestCase):
    def test_clock_and_latency_accumulators(self):
        state = SimulatorState()

        state.step_global_clock(0.5)
        state.set_current_inference_dur(0.2)
        state.inc_hicache_l2_load_dur(0.03)
        state.inc_hicache_l2_backup_dur(0.04)

        self.assertEqual(state.global_clock, 0.5)
        self.assertEqual(state.current_inference_dur, 0.2)
        self.assertEqual(state.pop_hicache_l2_load_dur(), 0.03)
        self.assertEqual(state.pop_hicache_l2_backup_dur(), 0.04)
        self.assertEqual(state.pop_hicache_l2_load_dur(), 0)
        self.assertEqual(state.pop_hicache_l2_backup_dur(), 0)

        state.set_current_inference_dur(0.7)
        self.assertEqual(state.last_inference_dur, 0.2)

    def test_reset_clears_state(self):
        state = SimulatorState()
        state.step_global_clock(1)
        state.inc_iteration()
        state.set_current_inference_dur(2)
        state.reset()

        self.assertEqual(state.global_clock, 0)
        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.current_inference_dur, 0)
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_state_metrics.py::TestSimulatorState -q
```

Expected: FAIL because `state.py` does not exist.

- [ ] **Step 3: Implement `SimulatorState`**

Create `python/sglang/srt/simulator/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulatorState:
    iteration: int = 0
    global_clock: float = 0
    last_inference_dur: float = 0
    current_inference_dur: float = 0
    hicache_l2_load_dur: float = 0
    hicache_l2_backup_dur: float = 0

    def reset(self) -> None:
        self.iteration = 0
        self.global_clock = 0
        self.last_inference_dur = 0
        self.current_inference_dur = 0
        self.hicache_l2_load_dur = 0
        self.hicache_l2_backup_dur = 0

    def inc_iteration(self) -> None:
        self.iteration += 1

    def step_global_clock(self, dur: float) -> None:
        self.global_clock += dur

    def set_global_clock(self, clock: float) -> None:
        self.global_clock = clock

    def set_current_inference_dur(self, dur: float) -> None:
        self.last_inference_dur = self.current_inference_dur
        self.current_inference_dur = dur

    def inc_hicache_l2_load_dur(self, dur: float) -> None:
        self.hicache_l2_load_dur += dur

    def inc_hicache_l2_backup_dur(self, dur: float) -> None:
        self.hicache_l2_backup_dur += dur

    def pop_hicache_l2_load_dur(self) -> float:
        dur = self.hicache_l2_load_dur
        self.hicache_l2_load_dur = 0
        return dur

    def pop_hicache_l2_backup_dur(self) -> float:
        dur = self.hicache_l2_backup_dur
        self.hicache_l2_backup_dur = 0
        return dur
```

- [ ] **Step 4: Run state tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_state_metrics.py::TestSimulatorState -q
```

Expected: PASS.

## Task 2: Metrics Calculation and Dumping

- [ ] **Step 1: Extend failing metrics tests**

Append to `test/registered/unit/simulator/test_state_metrics.py`:

```python

class TestSimulatorMetrics(unittest.TestCase):
    def test_calc_metrics_excludes_warmup_and_reports_ms(self):
        stats = [
            RequestStats(
                rid="warmup",
                created_time=0.0,
                queue_start=0.0,
                queue_end=0.1,
                input_length=10,
                output_length=2,
                gen_token_latencies=[0.2, 0.3],
            ),
            RequestStats(
                rid="req",
                created_time=1.0,
                queue_start=1.0,
                queue_end=1.1,
                input_length=20,
                output_length=3,
                gen_token_latencies=[0.4, 0.5, 0.6],
            ),
        ]

        metrics = calc_metrics(stats, num_warmup=1)

        self.assertEqual(metrics["completed"], 1)
        self.assertEqual(metrics["total_input"], 20)
        self.assertEqual(metrics["total_output"], 3)
        self.assertEqual(metrics["mean_ttft_ms"], 400)
        self.assertEqual(metrics["mean_itl_ms"], 550)
        self.assertEqual(metrics["mean_e2e_latency_ms"], 1500)

    def test_dump_outputs_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats = [
                RequestStats(
                    rid="req",
                    created_time=0.0,
                    queue_start=0.0,
                    queue_end=0.1,
                    input_length=5,
                    output_length=2,
                    gen_token_latencies=[0.2, 0.3],
                )
            ]
            iterations = [{"requests": [{"extend_length": 5}], "forward_latency": 0.2}]

            dump_simulator_outputs(tmp_dir, stats, iterations, num_warmup=0)

            self.assertTrue((Path(tmp_dir) / "metrics.json").exists())
            self.assertTrue((Path(tmp_dir) / "request.jsonl").exists())
            self.assertTrue((Path(tmp_dir) / "iteration.jsonl").exists())
            with open(Path(tmp_dir) / "metrics.json") as f:
                data = json.load(f)
            self.assertEqual(data["completed"], 1)
```

- [ ] **Step 2: Run metrics tests and verify failure**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_state_metrics.py::TestSimulatorMetrics -q
```

Expected: FAIL because `metrics.py` does not exist.

- [ ] **Step 3: Implement metrics module**

Create `python/sglang/srt/simulator/metrics.py`:

```python
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
    ttfts = [item.ttft() for item in metric_stats if item.gen_token_latencies]
    itls = [lat for item in metric_stats for lat in item.itls()]
    e2e_latencies = [item.e2e_latency() for item in metric_stats]
    duration = max((item.created_time + item.e2e_latency() for item in metric_stats), default=0)
    duration = max(duration, 1e-9)

    return {
        "completed": completed,
        "total_input": total_input,
        "total_input_text": -1,
        "total_input_vision": -1,
        "total_output": total_output,
        "total_output_retokenized": -1,
        "request_throughput": completed / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "output_throughput_retokenized": -1,
        "total_throughput": (total_input + total_output) / duration,
        "total_throughput_retokenized": -1,
        "mean_ttft_ms": _mean_ms(ttfts),
        "median_ttft_ms": _median_ms(ttfts),
        "std_ttft_ms": float(np.std(ttfts) * 1000) if ttfts else 0.0,
        "p99_ttft_ms": _percentile_ms(ttfts, 99),
        "mean_tpot_ms": _mean_ms(itls),
        "median_tpot_ms": _median_ms(itls),
        "std_tpot_ms": float(np.std(itls) * 1000) if itls else 0.0,
        "p99_tpot_ms": _percentile_ms(itls, 99),
        "mean_itl_ms": _mean_ms(itls),
        "median_itl_ms": _median_ms(itls),
        "std_itl_ms": float(np.std(itls) * 1000) if itls else 0.0,
        "p95_itl_ms": _percentile_ms(itls, 95),
        "p99_itl_ms": _percentile_ms(itls, 99),
        "max_itl_ms": float(max(itls) * 1000) if itls else 0.0,
        "mean_e2e_latency_ms": _mean_ms(e2e_latencies),
        "median_e2e_latency_ms": _median_ms(e2e_latencies),
        "std_e2e_latency_ms": float(np.std(e2e_latencies) * 1000) if e2e_latencies else 0.0,
        "p90_e2e_latency_ms": _percentile_ms(e2e_latencies, 90),
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
```

- [ ] **Step 4: Run metrics tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_state_metrics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit phase 2**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
git add python/sglang/srt/simulator/state.py python/sglang/srt/simulator/metrics.py test/registered/unit/simulator/test_state_metrics.py
git commit -m "feat: add simulator state and metrics"
```
