# Native Simulator Phase 4 Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Scheduler` run simulator request timing, predicted latency, logical clock, and metrics dump natively.

**Architecture:** Add `SchedulerSimulatorRuntime` as a helper owned by each scheduler. The scheduler calls helper methods at narrow lifecycle points.

**Tech Stack:** Python, heapq, SGLang scheduler, pytest with lightweight request/batch fakes.

---

## Dependencies

Complete Phases 1, 2, and 3 first.

## Files

- Create: `python/sglang/srt/simulator/scheduler.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/managers/scheduler_profiler_mixin.py`
- Test: `test/registered/unit/simulator/test_scheduler_simulator.py`

## Task 1: Runtime Queue and Request Stats

- [ ] **Step 1: Write failing runtime tests**

Create `test/registered/unit/simulator/test_scheduler_simulator.py`:

```python
import types
import unittest

from sglang.srt.simulator.scheduler import SchedulerSimulatorRuntime


def make_req(rid, created_time, total_request=2, input_len=4, output_len=3):
    sampling_params = types.SimpleNamespace(
        max_new_tokens=output_len,
        custom_params={
            "simulation": {
                "created_time": created_time,
                "total_request": total_request,
            }
        },
    )
    return types.SimpleNamespace(
        rid=rid,
        input_ids=list(range(input_len)),
        sampling_params=sampling_params,
        __class__=types.SimpleNamespace(__name__="TokenizedGenerateReqInput"),
    )


class TestSchedulerSimulatorRuntime(unittest.TestCase):
    def test_offline_buffers_until_all_requests_arrive(self):
        runtime = SchedulerSimulatorRuntime(
            mode="offline",
            output_dir="/tmp/sim",
            num_warmup=0,
            overlap_schedule=False,
            predictor=None,
        )
        r1 = make_req("r1", 0.0)
        r2 = make_req("r2", 2.0)

        self.assertEqual(runtime.filter_recv_requests([r1]), [])
        arrived = runtime.filter_recv_requests([r2])

        self.assertEqual([req.rid for req in arrived], ["r1"])
        runtime.state.set_global_clock(2.0)
        arrived = runtime.filter_recv_requests([])
        self.assertEqual([req.rid for req in arrived], ["r2"])

    def test_blocking_returns_requests_immediately_and_records_stats(self):
        runtime = SchedulerSimulatorRuntime(
            mode="blocking",
            output_dir="/tmp/sim",
            num_warmup=0,
            overlap_schedule=False,
            predictor=None,
        )
        req = make_req("r1", 0.0, total_request=1)

        arrived = runtime.filter_recv_requests([req])

        self.assertEqual(arrived, [req])
        self.assertEqual(runtime.request_stats["r1"].input_length, 4)
        self.assertEqual(runtime.request_stats["r1"].output_length, 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_scheduler_simulator.py -q
```

Expected: FAIL because runtime does not exist.

- [ ] **Step 3: Implement `SchedulerSimulatorRuntime` base**

Create `python/sglang/srt/simulator/scheduler.py`:

```python
from __future__ import annotations

import heapq
import time
from collections import defaultdict
from typing import Any, Dict, List

from sglang.srt.simulator.metrics import RequestStats, dump_simulator_outputs
from sglang.srt.simulator.state import SimulatorState


class SchedulerSimulatorRuntime:
    def __init__(
        self,
        *,
        mode: str,
        output_dir: str,
        num_warmup: int,
        overlap_schedule: bool,
        predictor: Any,
    ):
        self.mode = mode
        self.output_dir = output_dir
        self.num_warmup = num_warmup
        self.overlap_schedule = overlap_schedule
        self.predictor = predictor
        self.state = SimulatorState()
        self.request_stats: Dict[str, RequestStats] = defaultdict(RequestStats)
        self.iteration_stats: List[Dict[str, Any]] = []
        self.future_queue: List[tuple[float, int, Any]] = []
        self.offline_recv_all_request = False
        self.last_cpu_ts = 0.0

    @classmethod
    def from_scheduler(cls, scheduler, overlap_schedule: bool):
        settings = scheduler.simulator_settings
        predictor = getattr(scheduler, "simulator_predictor", None)
        return cls(
            mode=settings.mode,
            output_dir=settings.output_dir,
            num_warmup=settings.num_warmup,
            overlap_schedule=overlap_schedule,
            predictor=predictor,
        )

    def reset(self) -> None:
        self.state.reset()
        self.request_stats.clear()
        self.iteration_stats.clear()
        self.future_queue.clear()
        self.offline_recv_all_request = False
        self.last_cpu_ts = 0.0

    def _simulation_params(self, req) -> dict | None:
        sampling_params = getattr(req, "sampling_params", None)
        custom_params = getattr(sampling_params, "custom_params", None)
        if custom_params is None:
            return None
        return custom_params.get("simulation")

    def _is_generate_req(self, req) -> bool:
        return hasattr(req, "rid") and hasattr(req, "input_ids") and hasattr(req, "sampling_params")

    def filter_recv_requests(self, recv_reqs: List[Any]) -> List[Any]:
        if self.mode == "blocking":
            self._record_arrived_requests(recv_reqs)
            return recv_reqs

        immediate = []
        for req in recv_reqs:
            sim_params = self._simulation_params(req)
            if not self._is_generate_req(req) or sim_params is None:
                immediate.append(req)
                continue
            enqueue_time = sim_params.get("queue_start") or sim_params["created_time"]
            heapq.heappush(self.future_queue, (enqueue_time, time.time_ns(), req))
            total_request = sim_params["total_request"]
            if len(self.future_queue) == total_request:
                self.offline_recv_all_request = True

        arrived = list(immediate)
        while self.offline_recv_all_request and self.future_queue:
            enqueue_time, _, req = self.future_queue[0]
            if enqueue_time > self.state.global_clock:
                break
            heapq.heappop(self.future_queue)
            arrived.append(req)

        self._record_arrived_requests(arrived)
        return arrived

    def _record_arrived_requests(self, recv_reqs: List[Any]) -> None:
        now = time.time()
        for req in recv_reqs:
            if not self._is_generate_req(req):
                continue
            sim_params = self._simulation_params(req) or {}
            stats = self.request_stats[req.rid]
            stats.rid = req.rid
            stats.input_length = len(req.input_ids)
            stats.output_length = req.sampling_params.max_new_tokens
            if self.mode == "blocking":
                stats.created_time = sim_params.get("server_created_time", now)
                stats.last_event_time = stats.created_time
                stats.queue_start = now
            else:
                stats.created_time = sim_params["created_time"]
                stats.last_event_time = stats.created_time
                queue_start = sim_params.get("queue_start")
                if queue_start is not None:
                    self.state.set_global_clock(queue_start)
                stats.queue_start = self.state.global_clock

    def dump_outputs(self) -> dict:
        stats = sorted(
            [item for item in self.request_stats.values() if item.rid],
            key=lambda item: item.created_time,
        )
        return dump_simulator_outputs(
            self.output_dir,
            stats,
            self.iteration_stats,
            num_warmup=self.num_warmup,
        )
```

- [ ] **Step 4: Run runtime tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_scheduler_simulator.py -q
```

Expected: PASS.

## Task 2: Batch Latency and Metrics Hooks

- [ ] **Step 1: Extend runtime with batch methods**

Add these methods to `SchedulerSimulatorRuntime`:

```python
    def on_new_batch_prefill(self, new_batch, waiting_queue_len: int, running_batch_len: int) -> None:
        if new_batch is not None:
            for req in new_batch.reqs:
                stats = self.request_stats[req.rid]
                stats.final_reused_tokens = getattr(req, "cached_tokens", 0)
                if stats.queue_end == -1:
                    stats.queue_end = time.time() if self.mode == "blocking" else self.state.global_clock
        elif self.mode == "offline" and self.future_queue and running_batch_len == 0:
            next_created_time, _, _ = self.future_queue[0]
            self.state.set_global_clock(next_created_time + 1e-6)
        elif running_batch_len == 0 and waiting_queue_len > 0:
            self.state.step_global_clock(0.005)
            self.state.set_current_inference_dur(0.005)

    def _build_prediction_batch(self, batch):
        requests = []
        for req in batch.reqs:
            if batch.forward_mode.is_extend():
                extend_length = req.extend_input_len
            elif batch.forward_mode.is_decode():
                extend_length = 1
            else:
                continue
            requests.append(
                {
                    "extend_length": extend_length,
                    "past_kv_length": len(req.prefix_indices) + len(req.output_ids),
                }
            )
        return requests

    def on_run_batch(self, batch, result) -> None:
        if result.__class__.__name__ != "GenerationBatchResult":
            return
        requests = self._build_prediction_batch(batch)
        if not requests:
            return
        self.state.inc_iteration()
        predicted_latency = self._predict_latency(requests)
        if self.mode == "blocking":
            time.sleep(abs(predicted_latency))
            now = time.time()
            forward_latency = now - self.last_cpu_ts if self.last_cpu_ts else predicted_latency
            self.last_cpu_ts = now
        else:
            forward_latency = predicted_latency
        self.state.set_current_inference_dur(float(forward_latency))
        self._last_prediction_requests = requests

    def _predict_latency(self, requests: List[dict]) -> float:
        if self.predictor is None:
            total_tokens = sum(item["extend_length"] for item in requests)
            return max(total_tokens, 1) * 0.001
        return float(self.predictor.predict_requests(requests))

    def on_process_batch_result(self, batch) -> None:
        if batch is None or len(batch.reqs) == 0:
            return
        load_dur = self.state.pop_hicache_l2_load_dur()
        backup_dur = self.state.pop_hicache_l2_backup_dur()
        inference_dur = self.state.current_inference_dur
        if self.overlap_schedule:
            self.state.step_global_clock(max(load_dur - self.state.last_inference_dur, 0))
            self.state.step_global_clock(inference_dur)
            response_time = self.state.global_clock + backup_dur
        else:
            self.state.step_global_clock(load_dur + inference_dur + backup_dur)
            response_time = self.state.global_clock
        for req in batch.reqs:
            if getattr(req, "is_chunked", 0) == 0:
                stats = self.request_stats[req.rid]
                stats.gen_token_latencies.append(response_time - stats.last_event_time)
                stats.last_event_time = response_time
        self.iteration_stats.append(
            {
                "requests": getattr(self, "_last_prediction_requests", []),
                "forward_latency": inference_dur,
                "l2_load_latency": load_dur,
                "l2_backup_latency": backup_dur,
            }
        )
        self.last_cpu_ts = time.time()
```

- [ ] **Step 2: Patch `Scheduler.__init__`**

In `python/sglang/srt/managers/scheduler.py`, import:

```python
from sglang.srt.simulator.config import SimulatorSettings
from sglang.srt.simulator.scheduler import SchedulerSimulatorRuntime
```

Near the top of `Scheduler.__init__`, after `self.server_args = server_args`, add:

```python
        self.simulator_settings = SimulatorSettings.from_server_args(server_args)
        simulator_overlap_schedule = not server_args.disable_overlap_schedule
        if self.simulator_settings.enabled:
            server_args.disable_overlap_schedule = True
```

After normal initialization of profiler or before `self.is_initializing = False`, add:

```python
        self.simulator_runtime = (
            SchedulerSimulatorRuntime.from_scheduler(self, simulator_overlap_schedule)
            if self.simulator_settings.enabled
            else None
        )
```

- [ ] **Step 3: Patch `event_loop_overlap`**

At the start of `event_loop_overlap`, add:

```python
        if self.simulator_settings.enabled:
            return self.event_loop_normal()
```

- [ ] **Step 4: Patch `recv_requests`**

At the end of `recv_requests()`, replace `return recv_reqs` with:

```python
        if self.simulator_runtime is not None:
            recv_reqs = self.simulator_runtime.filter_recv_requests(recv_reqs)

        return recv_reqs
```

- [ ] **Step 5: Patch `get_new_batch_prefill`**

After `ret = self._get_new_batch_prefill_raw(...)` and prefill delayer finalize, before `return ret`, add:

```python
        if self.simulator_runtime is not None:
            self.simulator_runtime.on_new_batch_prefill(
                ret,
                waiting_queue_len=len(self.waiting_queue),
                running_batch_len=len(self.running_batch.reqs),
            )
```

- [ ] **Step 6: Patch `run_batch`**

Before `return ret` in `run_batch()`, add:

```python
        if self.simulator_runtime is not None:
            self.simulator_runtime.on_run_batch(batch, ret)
```

- [ ] **Step 7: Patch `process_batch_result`**

At the end of `process_batch_result()`, after existing side effects, add:

```python
        if self.simulator_runtime is not None:
            self.simulator_runtime.on_process_batch_result(batch)
```

- [ ] **Step 8: Patch profiler mixin**

In `python/sglang/srt/managers/scheduler_profiler_mixin.py::profile`, add at the top:

```python
        if getattr(self, "simulator_runtime", None) is not None:
            from sglang.srt.managers.io_struct import ProfileReqOutput

            if recv_req.type == ProfileReqType.START_PROFILE:
                self.simulator_runtime.reset()
                return ProfileReqOutput(True, "Simulator profiling started")

            metrics = self.simulator_runtime.dump_outputs()
            return ProfileReqOutput(True, json.dumps(metrics))
```

Ensure `json` is imported in the file.

- [ ] **Step 9: Compile modified files**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python python -m py_compile \
  python/sglang/srt/simulator/scheduler.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/managers/scheduler_profiler_mixin.py
```

Expected: compile succeeds.

- [ ] **Step 10: Run simulator unit tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_scheduler_simulator.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit phase 4**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
git add python/sglang/srt/simulator/scheduler.py python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/scheduler_profiler_mixin.py test/registered/unit/simulator/test_scheduler_simulator.py
git commit -m "feat: add simulator scheduler runtime"
```
