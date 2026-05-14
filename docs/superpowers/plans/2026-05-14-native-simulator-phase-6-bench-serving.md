# Native Simulator Phase 6 Bench Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python -m sglang.bench_serving --simulator` inject logical request timestamps and report backend-generated simulator metrics.

**Architecture:** Add small helper functions in `bench_serving.py` and thread a `simulator` flag through benchmark request generation and metric calculation. Avoid monkey patching aiohttp.

**Tech Stack:** Python asyncio, argparse, JSON, pytest.

---

## Dependencies

Complete Phases 1 through 4 before this phase. Phase 5 is only required if benchmarking HiCache simulation.

## Files

- Modify: `python/sglang/bench_serving.py`
- Test: `test/registered/unit/simulator/test_bench_serving_simulator.py`

## Task 1: Metadata Injection Helpers

- [ ] **Step 1: Write failing helper tests**

Create `test/registered/unit/simulator/test_bench_serving_simulator.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from sglang.bench_serving import (
    BenchmarkMetrics,
    inject_simulation_params,
    load_simulator_metrics,
)
from sglang.benchmark.datasets.common import DatasetRow


class TestBenchServingSimulator(unittest.TestCase):
    def test_inject_simulation_params(self):
        request = DatasetRow(prompt="hi", prompt_len=1, output_len=2)

        inject_simulation_params(request, created_time=1.5, total_request=3)

        sim = request.extra_request_body["sampling_params"]["custom_params"]["simulation"]
        self.assertEqual(sim["created_time"], 1.5)
        self.assertEqual(sim["total_request"], 3)

    def test_load_simulator_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = {field: 0 for field in BenchmarkMetrics.__dataclass_fields__}
            data["completed"] = 7
            with open(Path(tmp_dir) / "metrics.json", "w") as f:
                json.dump(data, f)

            metrics = load_simulator_metrics(tmp_dir)

        self.assertEqual(metrics.completed, 7)

    def test_load_simulator_metrics_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                load_simulator_metrics(tmp_dir)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_bench_serving_simulator.py -q
```

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Add helpers to `bench_serving.py`**

In `python/sglang/bench_serving.py`, after `BenchmarkMetrics`, add:

```python
def inject_simulation_params(
    request: DatasetRow,
    *,
    created_time: float,
    total_request: int,
    queue_start: Optional[float] = None,
) -> None:
    if request.extra_request_body is None:
        request.extra_request_body = {}
    sampling_params = request.extra_request_body.setdefault("sampling_params", {})
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
        raise FileNotFoundError(
            f"Simulator metrics file not found: {metrics_path}. "
            "Make sure the server was launched with --simulator and benchmark profiling completed."
        )
    with open(metrics_path) as f:
        data = json.load(f)
    metric_fields = BenchmarkMetrics.__dataclass_fields__
    return BenchmarkMetrics(**{name: data.get(name, -1) for name in metric_fields})
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_bench_serving_simulator.py -q
```

Expected: PASS.

## Task 2: Request Generator Simulator Path

- [ ] **Step 1: Modify `get_request` signature**

Change:

```python
async def get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
    use_trace_timestamps: bool = False,
    slowdown_factor: float = 1.0,
) -> AsyncGenerator[DatasetRow, None]:
```

to:

```python
async def get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
    use_trace_timestamps: bool = False,
    slowdown_factor: float = 1.0,
    simulator: bool = False,
    simulator_timestamp_scale_s: float = 1000.0,
) -> AsyncGenerator[DatasetRow, None]:
```

- [ ] **Step 2: Add simulator no-sleep path**

At the start of `get_request`, add:

```python
    if simulator:
        total_request = len(input_requests)
        if use_trace_timestamps:
            input_requests.sort(key=lambda r: r.timestamp)
            trace_start = input_requests[0].timestamp if input_requests else 0
            for request in input_requests:
                created_time = (
                    (request.timestamp - trace_start)
                    / simulator_timestamp_scale_s
                    * slowdown_factor
                )
                inject_simulation_params(
                    request,
                    created_time=created_time,
                    total_request=total_request,
                )
                yield request
            return

        created_time = 0.0
        for request in input_requests:
            inject_simulation_params(
                request,
                created_time=created_time,
                total_request=total_request,
            )
            yield request
            if request_rate != float("inf"):
                created_time += float(np.random.exponential(1.0 / request_rate))
        return
```

- [ ] **Step 3: Pass simulator flags from benchmark loop**

In `benchmark(...)`, add parameters:

```python
    simulator: bool = False,
    simulator_timestamp_scale_s: float = 1000.0,
    simulator_output_dir: str = "/tmp/sglang_simulator/output",
```

When building `request_generator`, change:

```python
        request_generator = get_request(input_requests, request_rate)
```

to:

```python
        request_generator = get_request(
            input_requests,
            request_rate,
            use_trace_timestamps=use_trace_timestamps,
            slowdown_factor=mooncake_slowdown_factor,
            simulator=simulator,
            simulator_timestamp_scale_s=simulator_timestamp_scale_s,
        )
```

- [ ] **Step 4: Force profiling for simulator**

In `run_benchmark`, after default args setup, add:

```python
    if getattr(args, "simulator", False):
        args.profile = True
        os.environ["SGLANG_SIMULATOR_OUTPUT_DIR"] = args.simulator_output_dir
```

- [ ] **Step 5: Read backend metrics**

After `calculate_metrics(...)` in `benchmark`, add:

```python
    if simulator:
        metrics = load_simulator_metrics(simulator_output_dir)
```

Keep `output_lens` from client calculation for output-file compatibility.

- [ ] **Step 6: Thread args through `run_benchmark`**

In the `benchmark(...)` call, add:

```python
            simulator=getattr(args, "simulator", False),
            simulator_timestamp_scale_s=args.simulator_timestamp_scale_s,
            simulator_output_dir=args.simulator_output_dir,
```

## Task 3: CLI Parser

- [ ] **Step 1: Add CLI args**

In `get_args_parser()`, add near general benchmark options:

```python
    parser.add_argument(
        "--simulator",
        action="store_true",
        help="Enable benchmark support for native SGLang simulator mode.",
    )
    parser.add_argument(
        "--simulator-output-dir",
        type=str,
        default="/tmp/sglang_simulator/output",
        help="Directory where the simulator server writes metrics.json.",
    )
    parser.add_argument(
        "--simulator-timestamp-scale-s",
        type=float,
        default=1000.0,
        help="Scale used to convert trace timestamps to seconds in simulator mode.",
    )
```

- [ ] **Step 2: Compile and run tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python python -m py_compile python/sglang/bench_serving.py
PYTHONPATH=python pytest test/registered/unit/simulator/test_bench_serving_simulator.py -q
```

Expected: compile succeeds and tests pass.

- [ ] **Step 3: Help smoke**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python python -m sglang.bench_serving --help | grep simulator
```

Expected: output includes `--simulator`, `--simulator-output-dir`, and `--simulator-timestamp-scale-s`.

- [ ] **Step 4: Commit phase 6**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
git add python/sglang/bench_serving.py test/registered/unit/simulator/test_bench_serving_simulator.py
git commit -m "feat: add simulator bench serving mode"
```
