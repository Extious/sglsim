# SGLSim Archive Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild this repository as `sglsim`, a CPU-only simulator fork that preserves scheduling, request batching, KV/cache metadata management, trace replay, metrics, and AIConfigurator latency prediction while archiving the original SGLang implementation under `Archived/`.

**Architecture:** Move the current project into `Archived/` as a read-only migration source, then create a new minimal Python package at `python/sglsim`. Migrate simulator-relevant modules out of `Archived/` in bounded phases, renaming imports from `sglang` to `sglsim` as each module is moved. Real GPU inference paths are not migrated; they are replaced by CPU-only simulator implementations and metadata-only KV/cache operations.

**Tech Stack:** Python 3.12, uv, pytest, FastAPI/uvicorn only if the simulator HTTP server phase keeps an HTTP interface, AIConfigurator SDK, NumPy, optional PyTorch CPU only for tensor-shaped mocks where simpler metadata is not enough.

---

## Execution Rules

- Start only from a clean checkpoint commit. The user reported checkpoint commit `e60b7cb native simulater implementment`.
- Do not keep editing `Archived/` after it is created. Treat it as source material for reference and copy/migration only.
- Do not migrate a module just because it has no `cuda` string. Migrate only if it serves simulator scheduling, cache metadata, workload replay, metrics, or configuration.
- Keep each task independently testable and commit after each completed task.
- Prefer subagents for Tasks 3-7. Each subagent must have a disjoint write scope.
- Use `uv` and Python 3.12 for all verification after Task 2.
- The new public import path is `sglsim`, not `sglang`. Do not add a long-term `sglang` compatibility package unless a later plan explicitly requires it.

## Target File Structure

```text
Archived/
  ... original checkpoint tree ...

python/
  sglsim/
    __init__.py
    launch_server.py
    bench_serving.py
    srt/
      __init__.py
      server_args.py
      simulator/
        __init__.py
        config.py
        predictor.py
        predictor_aiconfigurator.py
        state.py
        metrics.py
        scheduler.py
        model_runner.py
      scheduling/
        __init__.py
        request.py
        batch.py
        scheduler.py
      mem_cache/
        __init__.py
        allocator.py
        radix_cache.py
        hicache.py
      http/
        __init__.py
        server.py

scripts/
  sglsim_smoke.py

test/
  unit/
    simulator/
      assets/config.json
      test_config.py
      test_predictor.py
      test_scheduler.py
      test_metrics.py
      test_mem_cache.py
      test_bench_serving.py
```

## Subagent Work Allocation

Use `superpowers:subagent-driven-development` during execution if available.

- Worker A owns package bootstrap and archive move: `Archived/`, root `pyproject.toml`, `python/sglsim/__init__.py`, `python/sglsim/launch_server.py`, `README.md`.
- Worker B owns simulator core: `python/sglsim/srt/simulator/*`, `test/unit/simulator/test_config.py`, `test/unit/simulator/test_predictor.py`, `test/unit/simulator/test_metrics.py`.
- Worker C owns scheduling: `python/sglsim/srt/scheduling/*`, `python/sglsim/srt/simulator/scheduler.py`, `test/unit/simulator/test_scheduler.py`.
- Worker D owns KV/cache metadata: `python/sglsim/srt/mem_cache/*`, `test/unit/simulator/test_mem_cache.py`.
- Worker E owns benchmark/server integration: `python/sglsim/bench_serving.py`, `python/sglsim/srt/http/server.py`, `scripts/sglsim_smoke.py`, `test/unit/simulator/test_bench_serving.py`.

Agents must not edit another worker's files unless the main integrator explicitly reassigns ownership.

---

### Task 1: Archive Existing Tree

**Files:**
- Create: `Archived/`
- Move: all current root entries except `.git/` into `Archived/`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Verify clean checkpoint**

Run:

```bash
git status --short --untracked-files=all
git log -1 --oneline
```

Expected:

```text
<no status output>
e60b7cb native simulater implementment
```

If status is not clean, stop and ask whether to commit or stash first.

- [ ] **Step 2: Move project into archive**

Run from repo root:

```bash
mkdir Archived
find . -mindepth 1 -maxdepth 1 \
  ! -name .git \
  ! -name Archived \
  -exec mv {} Archived/ \;
```

Expected: repo root contains only `.git/` and `Archived/`.

- [ ] **Step 3: Create root README**

Create `README.md`:

```markdown
# SGLSim

SGLSim is a CPU-only simulator fork derived from SGLang. The original checkpoint is preserved under `Archived/` for migration reference.

The new package name is `sglsim`. GPU inference, CUDA kernels, distributed GPU execution, and real model weight loading are intentionally excluded from the new runtime.
```

- [ ] **Step 4: Create root gitignore**

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
dist/
build/
*.egg-info/
```

- [ ] **Step 5: Commit archive move**

Run:

```bash
git add -A
git commit -m "chore: archive original sglang tree"
```

Expected: commit succeeds.

---

### Task 2: Bootstrap `sglsim` Package and uv Environment

**Files:**
- Create: `pyproject.toml`
- Create: `python/sglsim/__init__.py`
- Create: `python/sglsim/launch_server.py`
- Create: `python/sglsim/srt/__init__.py`
- Create: `python/sglsim/srt/server_args.py`
- Create: `test/unit/simulator/test_package.py`

- [ ] **Step 1: Create package directories**

Run:

```bash
mkdir -p python/sglsim/srt test/unit/simulator
```

Expected: directories exist.

- [ ] **Step 2: Create `pyproject.toml`**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sglsim"
version = "0.1.0"
description = "CPU-only SGLang scheduling and KV-cache simulator"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "numpy>=1.26",
  "pydantic>=2",
]

[project.optional-dependencies]
server = [
  "fastapi>=0.110",
  "uvicorn>=0.27",
  "orjson>=3.9",
]
aiconfigurator = [
  "aiconfigurator",
]
test = [
  "pytest>=8",
]

[project.scripts]
sglsim = "sglsim.launch_server:main"
sglsim-bench = "sglsim.bench_serving:main"

[tool.setuptools.packages.find]
where = ["python"]

[tool.pytest.ini_options]
testpaths = ["test/unit"]
pythonpath = ["python"]
```

- [ ] **Step 3: Create package init**

Create `python/sglsim/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `python/sglsim/srt/__init__.py`:

```python
"""SGLSim runtime components."""
```

- [ ] **Step 4: Create simulator-only server args**

Create `python/sglsim/srt/server_args.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ServerArgs:
    model_path: str = "simulator-model"
    sim_config_path: str | None = None
    simulator_mode: Literal["offline", "blocking"] = "offline"
    simulator_output_dir: str = "/tmp/sglsim/output"
    simulator_num_warmup: int = 0
    host: str = "127.0.0.1"
    port: int = 30000

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model-path", default=ServerArgs.model_path)
        parser.add_argument("--sim-config-path", required=True)
        parser.add_argument(
            "--simulator-mode",
            choices=["offline", "blocking"],
            default=ServerArgs.simulator_mode,
        )
        parser.add_argument(
            "--simulator-output-dir",
            default=ServerArgs.simulator_output_dir,
        )
        parser.add_argument(
            "--simulator-num-warmup",
            type=int,
            default=ServerArgs.simulator_num_warmup,
        )
        parser.add_argument("--host", default=ServerArgs.host)
        parser.add_argument("--port", type=int, default=ServerArgs.port)

    @classmethod
    def from_cli_args(cls, args: list[str] | None = None) -> "ServerArgs":
        parser = argparse.ArgumentParser(description="SGLSim CPU-only simulator server")
        cls.add_cli_args(parser)
        ns = parser.parse_args(args)
        parsed = cls(**vars(ns))
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.sim_config_path is None:
            raise ValueError("--sim-config-path is required")
        if not Path(self.sim_config_path).exists():
            raise ValueError(f"Simulator config path does not exist: {self.sim_config_path}")
```

- [ ] **Step 5: Create launch entrypoint**

Create `python/sglsim/launch_server.py`:

```python
from __future__ import annotations

from sglsim.srt.server_args import ServerArgs


def main(argv: list[str] | None = None) -> None:
    args = ServerArgs.from_cli_args(argv)
    print(
        "SGLSim simulator server configuration:\n"
        f"  model_path={args.model_path}\n"
        f"  sim_config_path={args.sim_config_path}\n"
        f"  simulator_mode={args.simulator_mode}\n"
        f"  output_dir={args.simulator_output_dir}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write failing package test**

Create `test/unit/simulator/test_package.py`:

```python
from pathlib import Path

from sglsim import __version__
from sglsim.srt.server_args import ServerArgs


def test_package_name_is_sglsim():
    assert __version__ == "0.1.0"


def test_server_args_require_existing_config(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text('{"predictor": {"name": "aiconfigurator"}}')

    args = ServerArgs.from_cli_args(["--sim-config-path", str(config)])

    assert args.sim_config_path == str(config)
    assert args.simulator_mode == "offline"
```

- [ ] **Step 7: Create uv environment**

Run:

```bash
uv venv --python 3.12
uv pip install -e '.[test]'
```

Expected: environment creation and editable install succeed.

- [ ] **Step 8: Run package test**

Run:

```bash
uv run pytest test/unit/simulator/test_package.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 9: Run launch help**

Run:

```bash
uv run python -m sglsim.launch_server --help
```

Expected: output includes `SGLSim CPU-only simulator server` and `--sim-config-path`.

- [ ] **Step 10: Commit bootstrap**

Run:

```bash
git add pyproject.toml python/sglsim test/unit/simulator README.md .gitignore
git commit -m "feat: bootstrap sglsim package"
```

Expected: commit succeeds.

---

### Task 3: Migrate Simulator Config, Types, Predictor, State, and Metrics

**Files:**
- Create: `python/sglsim/srt/simulator/__init__.py`
- Create: `python/sglsim/srt/simulator/config.py`
- Create: `python/sglsim/srt/simulator/types.py`
- Create: `python/sglsim/srt/simulator/predictor.py`
- Create: `python/sglsim/srt/simulator/predictor_aiconfigurator.py`
- Create: `python/sglsim/srt/simulator/state.py`
- Create: `python/sglsim/srt/simulator/metrics.py`
- Create: `test/unit/simulator/assets/config.json`
- Create: `test/unit/simulator/test_config.py`
- Create: `test/unit/simulator/test_predictor.py`
- Create: `test/unit/simulator/test_metrics.py`

- [ ] **Step 1: Copy current native simulator reference files**

Use these source files as the migration source:

```text
Archived/python/sglang/srt/simulator/config.py
Archived/python/sglang/srt/simulator/types.py
Archived/python/sglang/srt/simulator/predictor.py
Archived/python/sglang/srt/simulator/predictor_aiconfigurator.py
Archived/python/sglang/srt/simulator/state.py
Archived/python/sglang/srt/simulator/metrics.py
Archived/test/registered/unit/simulator/test_simulator_config.py
Archived/test/registered/unit/simulator/test_state_metrics.py
Archived/test/registered/unit/simulator/assets/config.json
```

- [ ] **Step 2: Write config tests first**

Create `test/unit/simulator/test_config.py`:

```python
import json
from pathlib import Path

import pytest

from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulator.config import SimulatorSettings


def test_settings_load_config(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"predictor": {"name": "aiconfigurator"}}))
    args = ServerArgs.from_cli_args(["--sim-config-path", str(config)])

    settings = SimulatorSettings.from_server_args(args)

    assert settings.enabled is True
    assert settings.mode == "offline"
    assert settings.raw_config["predictor"]["name"] == "aiconfigurator"


def test_settings_reject_missing_predictor(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    args = ServerArgs.from_cli_args(["--sim-config-path", str(config)])

    with pytest.raises(ValueError, match="predictor.name must be aiconfigurator"):
        SimulatorSettings.from_server_args(args)
```

- [ ] **Step 3: Run config tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_config.py -q
```

Expected: FAIL because `sglsim.srt.simulator.config` does not exist.

- [ ] **Step 4: Implement simulator config**

Create `python/sglsim/srt/simulator/__init__.py`:

```python
"""CPU-only simulator runtime."""
```

Create `python/sglsim/srt/simulator/config.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ACTIVE_SETTINGS: "SimulatorSettings | None" = None


@dataclass(frozen=True)
class SimulatorSettings:
    enabled: bool
    config_path: str
    mode: str
    output_dir: str
    num_warmup: int
    raw_config: dict[str, Any]

    @classmethod
    def from_server_args(cls, server_args) -> "SimulatorSettings":
        config_path = getattr(server_args, "sim_config_path", None)
        if not config_path:
            raise ValueError("--sim-config-path is required")
        path = Path(config_path)
        if not path.exists():
            raise ValueError(f"Simulator config path does not exist: {config_path}")
        raw_config = json.loads(path.read_text())
        predictor_name = raw_config.get("predictor", {}).get("name", "aiconfigurator")
        if predictor_name != "aiconfigurator":
            raise ValueError("predictor.name must be aiconfigurator")
        mode = getattr(server_args, "simulator_mode", "offline").lower()
        if mode not in ("offline", "blocking"):
            raise ValueError(f"Unsupported simulator mode: {mode}")
        return cls(
            enabled=True,
            config_path=str(path),
            mode=mode,
            output_dir=getattr(server_args, "simulator_output_dir", "/tmp/sglsim/output"),
            num_warmup=int(getattr(server_args, "simulator_num_warmup", 0)),
            raw_config=raw_config,
        )


def set_active_settings(settings: SimulatorSettings | None) -> None:
    global _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = settings


def get_active_settings() -> SimulatorSettings | None:
    return _ACTIVE_SETTINGS


def get_platform_config() -> dict[str, Any]:
    if _ACTIVE_SETTINGS is None:
        return {}
    return _ACTIVE_SETTINGS.raw_config.get("platform", {})
```

- [ ] **Step 5: Run config tests**

Run:

```bash
uv run pytest test/unit/simulator/test_config.py -q
```

Expected: `2 passed`.

- [ ] **Step 6: Migrate types and predictor tests**

Create `test/unit/simulator/test_predictor.py`:

```python
import json
import sys
import types
from pathlib import Path

from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulator.predictor import (
    ScheduleBatch,
    ScheduleRequest,
    build_simulator_predictor,
    resolve_model_info,
    resolve_scheduler_config,
)
from sglsim.srt.simulator.types import DataType


def test_schedule_batch_classifies_decode():
    batch = ScheduleBatch([ScheduleRequest(1, 4), ScheduleRequest(1, 8)])

    assert batch.is_decode()
    assert batch.request_info() == [[1, 4], [1, 8]]


def test_resolve_scheduler_config_auto_dtype_uses_model_config():
    args = types.SimpleNamespace(
        dtype="auto",
        kv_cache_dtype="auto",
        mem_fraction_static=0.8,
        max_total_tokens=1024,
        tp_size=1,
        ep_size=1,
        dp_size=1,
        pp_size=1,
    )
    model_config = types.SimpleNamespace(dtype="torch.bfloat16")

    config = resolve_scheduler_config(args, model_config)

    assert config.data_type == DataType.BF16
    assert config.kv_cache_data_type == DataType.BF16


def test_resolve_model_info_mha():
    model_config = types.SimpleNamespace(
        dtype="torch.float16",
        attention_arch="MHA",
        hf_text_config={},
        model_path="sim-model",
        context_len=4096,
        hidden_size=2048,
        head_dim=128,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_key_value_heads=16,
        v_head_dim=128,
        vocab_size=32000,
    )

    model = resolve_model_info(model_config)

    assert model.model_path == "sim-model"
    assert model.attention_arch == "MHA"
    assert not model.is_mla()


def test_build_predictor_uses_aiconfigurator_adapter(tmp_path: Path):
    class DummyPredictor:
        def __init__(self, model, hw, config, **kwargs):
            self.model = model
            self.hw = hw
            self.config = config
            self.kwargs = kwargs

        def predict_requests(self, requests):
            return 1.5 + len(requests)

    fake_module = types.ModuleType("sglsim.srt.simulator.predictor_aiconfigurator")
    fake_module.AIConfiguratorTimePredictor = DummyPredictor
    old_module = sys.modules.get("sglsim.srt.simulator.predictor_aiconfigurator")
    sys.modules["sglsim.srt.simulator.predictor_aiconfigurator"] = fake_module
    try:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "platform": {"accelerator": {"name": "unit", "hbm_capacity_gb": 80}},
            "predictor": {"name": "aiconfigurator", "database_mode": "EMPIRICAL"},
            "scheduler": {"tp_size": 4, "data_type": "BF16", "kv_cache_data_type": "FP8"},
        }))
        args = ServerArgs.from_cli_args(["--sim-config-path", str(config_path)])
        args.dtype = "float16"
        args.kv_cache_dtype = "auto"
        args.mem_fraction_static = 0.8
        args.max_total_tokens = 1024
        args.tp_size = 1
        args.ep_size = 1
        args.dp_size = 1
        args.pp_size = 1
        model_config = types.SimpleNamespace(
            dtype="torch.float16",
            attention_arch="MHA",
            hf_text_config={},
            model_path="sim-model",
            context_len=4096,
            hidden_size=2048,
            head_dim=128,
            num_attention_heads=16,
            num_hidden_layers=24,
            num_key_value_heads=16,
            v_head_dim=128,
            vocab_size=32000,
        )

        predictor = build_simulator_predictor(args, model_config)
    finally:
        if old_module is None:
            sys.modules.pop("sglsim.srt.simulator.predictor_aiconfigurator", None)
        else:
            sys.modules["sglsim.srt.simulator.predictor_aiconfigurator"] = old_module

    assert predictor.config.tp_size == 4
    assert predictor.config.data_type == DataType.BF16
    assert predictor.config.kv_cache_data_type == DataType.FP8
    assert predictor.predict_requests([{"extend_length": 1, "past_kv_length": 0}]) == 2.5
```

- [ ] **Step 7: Run predictor tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_predictor.py -q
```

Expected: FAIL because `types.py` and `predictor.py` do not exist or are incomplete.

- [ ] **Step 8: Implement types and predictor**

Migrate and rename the current checkpoint implementations from:

```text
Archived/python/sglang/srt/simulator/types.py
Archived/python/sglang/srt/simulator/predictor.py
Archived/python/sglang/srt/simulator/predictor_aiconfigurator.py
```

Required edits during migration:

```text
sglang.srt.simulator -> sglsim.srt.simulator
sglang.version.__version__ -> sglsim.__version__
Default output path /tmp/sglang_simulator -> /tmp/sglsim
Remove token-based fallback from predictor construction. build_simulator_predictor must raise ValueError if AIConfigurator predictor cannot be built.
```

- [ ] **Step 9: Run predictor tests**

Run:

```bash
uv run pytest test/unit/simulator/test_predictor.py -q
```

Expected: `4 passed`.

- [ ] **Step 10: Migrate state and metrics tests**

Create `test/unit/simulator/test_metrics.py`:

```python
import json
from pathlib import Path

from sglsim.srt.simulator.metrics import RequestStats, calc_metrics, dump_simulator_outputs
from sglsim.srt.simulator.state import SimulatorState


def test_state_tracks_clock_and_hicache_durations():
    state = SimulatorState()
    state.step_global_clock(0.5)
    state.set_current_inference_dur(0.2)
    state.inc_hicache_l2_load_dur(0.1)
    state.inc_hicache_l2_backup_dur(0.3)

    assert state.global_clock == 0.5
    assert state.current_inference_dur == 0.2
    assert state.pop_hicache_l2_load_dur() == 0.1
    assert state.pop_hicache_l2_backup_dur() == 0.3


def test_metrics_dump_skips_warmup(tmp_path: Path):
    reqs = [
        RequestStats(rid="warmup", input_length=1, output_length=1, created_time=0, last_event_time=0.1, queue_start=0, queue_end=0, gen_token_latencies=[0.1]),
        RequestStats(rid="real", input_length=2, output_length=2, created_time=1, last_event_time=1.4, queue_start=1, queue_end=1.1, gen_token_latencies=[0.2, 0.2]),
    ]

    result = dump_simulator_outputs(str(tmp_path), reqs, [], num_warmup=1)

    assert result["completed"] == 1
    assert result["total_input"] == 2
    assert (tmp_path / "metrics.json").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["completed"] == 1
```

- [ ] **Step 11: Run metrics tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_metrics.py -q
```

Expected: FAIL because `state.py` and `metrics.py` do not exist.

- [ ] **Step 12: Implement state and metrics**

Migrate and rename:

```text
Archived/python/sglang/srt/simulator/state.py -> python/sglsim/srt/simulator/state.py
Archived/python/sglang/srt/simulator/metrics.py -> python/sglsim/srt/simulator/metrics.py
```

Required edits:

```text
sglang -> sglsim imports
Use /tmp/sglsim defaults where present
Keep metrics JSON schema compatible with current native simulator tests
```

- [ ] **Step 13: Copy sample config asset**

Create `test/unit/simulator/assets/config.json`:

```json
{
  "platform": {
    "accelerator": {
      "name": "a100_sxm",
      "hbm_capacity_gb": 80
    },
    "disk_read_bandwidth_gb": 8,
    "disk_write_bandwidth_gb": 8,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 8
  },
  "predictor": {
    "name": "aiconfigurator"
  },
  "scheduler": {
    "tp_size": 1,
    "ep_size": 1,
    "dp_size": 1,
    "backend_version": "0.5.9"
  }
}
```

- [ ] **Step 14: Run simulator core tests**

Run:

```bash
uv run pytest test/unit/simulator/test_config.py test/unit/simulator/test_predictor.py test/unit/simulator/test_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 15: Commit simulator core**

Run:

```bash
git add python/sglsim/srt/simulator test/unit/simulator pyproject.toml
git commit -m "feat: migrate sglsim simulator core"
```

Expected: commit succeeds.

---

### Task 4: Migrate CPU-Only Scheduling Runtime

**Files:**
- Create: `python/sglsim/srt/scheduling/__init__.py`
- Create: `python/sglsim/srt/scheduling/request.py`
- Create: `python/sglsim/srt/scheduling/batch.py`
- Create: `python/sglsim/srt/scheduling/scheduler.py`
- Create: `python/sglsim/srt/simulator/scheduler.py`
- Create: `test/unit/simulator/test_scheduler.py`

- [ ] **Step 1: Write scheduling tests first**

Create `test/unit/simulator/test_scheduler.py`:

```python
from sglsim.srt.scheduling.request import GenerateRequest, SamplingParams
from sglsim.srt.scheduling.scheduler import CPUScheduler
from sglsim.srt.simulator.scheduler import SchedulerSimulatorRuntime


class FixedPredictor:
    def predict_requests(self, requests):
        return 0.25


def make_request(rid: str, created_time: float, total_request: int = 2):
    return GenerateRequest(
        rid=rid,
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(
            max_new_tokens=2,
            custom_params={
                "simulation": {
                    "created_time": created_time,
                    "total_request": total_request,
                }
            },
        ),
    )


def test_offline_scheduler_replays_by_logical_time():
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir="/tmp/sglsim-test",
        num_warmup=0,
        overlap_schedule=False,
        predictor=FixedPredictor(),
    )
    scheduler = CPUScheduler(runtime=runtime)

    assert scheduler.recv_requests([make_request("r1", 0.0)]) == []
    first = scheduler.recv_requests([make_request("r2", 2.0)])
    assert [req.rid for req in first] == ["r1"]
    runtime.state.set_global_clock(2.0)
    second = scheduler.recv_requests([])
    assert [req.rid for req in second] == ["r2"]


def test_scheduler_predicts_batch_latency_and_records_stats():
    runtime = SchedulerSimulatorRuntime(
        mode="offline",
        output_dir="/tmp/sglsim-test",
        num_warmup=0,
        overlap_schedule=False,
        predictor=FixedPredictor(),
    )
    scheduler = CPUScheduler(runtime=runtime)
    req = make_request("r1", 0.0, total_request=1)

    scheduler.recv_requests([req])
    scheduler.step()

    assert runtime.state.global_clock == 0.25
    assert runtime.request_stats["r1"].gen_token_latencies == [0.25]
```

- [ ] **Step 2: Run scheduler tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_scheduler.py -q
```

Expected: FAIL because scheduling modules do not exist.

- [ ] **Step 3: Implement request dataclasses**

Create `python/sglsim/srt/scheduling/__init__.py`:

```python
"""CPU-only scheduling primitives."""
```

Create `python/sglsim/srt/scheduling/request.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SamplingParams:
    max_new_tokens: int
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateRequest:
    rid: str
    input_ids: list[int]
    sampling_params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    is_chunked: int = 0
```

- [ ] **Step 4: Implement batch dataclass**

Create `python/sglsim/srt/scheduling/batch.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from sglsim.srt.scheduling.request import GenerateRequest


@dataclass
class ScheduleBatch:
    reqs: list[GenerateRequest]

    def __len__(self) -> int:
        return len(self.reqs)
```

- [ ] **Step 5: Migrate simulator scheduler runtime**

Migrate `Archived/python/sglang/srt/simulator/scheduler.py` to `python/sglsim/srt/simulator/scheduler.py`.

Required edits:

```text
sglang.srt.simulator -> sglsim.srt.simulator
Remove ModelRunner-specific `on_run_batch(batch, result)` dependency.
Expose `run_requests(reqs: list[GenerateRequest]) -> None` that builds prediction requests directly from request metadata.
Keep `filter_recv_requests`, `on_new_batch_prefill`, `on_process_batch_result`, and `dump_outputs` behavior.
Remove token-based fallback: `_predict_latency` raises RuntimeError if predictor is None.
```

The required `run_requests` method shape:

```python
def run_requests(self, reqs):
    requests = [
        {
            "extend_length": max(len(req.input_ids), 1) if not req.output_ids else 1,
            "past_kv_length": len(req.output_ids),
        }
        for req in reqs
    ]
    self.state.inc_iteration()
    predicted_latency = self._predict_latency(requests)
    self.state.set_current_inference_dur(predicted_latency)
    self._last_prediction_requests = requests
```

- [ ] **Step 6: Implement CPU scheduler**

Create `python/sglsim/srt/scheduling/scheduler.py`:

```python
from __future__ import annotations

from sglsim.srt.scheduling.batch import ScheduleBatch
from sglsim.srt.scheduling.request import GenerateRequest
from sglsim.srt.simulator.scheduler import SchedulerSimulatorRuntime


class CPUScheduler:
    def __init__(self, runtime: SchedulerSimulatorRuntime):
        self.runtime = runtime
        self.waiting: list[GenerateRequest] = []
        self.running: list[GenerateRequest] = []

    def recv_requests(self, requests: list[GenerateRequest]) -> list[GenerateRequest]:
        arrived = self.runtime.filter_recv_requests(requests)
        self.waiting.extend(arrived)
        return arrived

    def step(self) -> ScheduleBatch | None:
        if not self.waiting:
            self.runtime.on_new_batch_prefill(None, 0, len(self.running))
            return None
        batch = ScheduleBatch(self.waiting)
        self.waiting = []
        self.running = batch.reqs
        self.runtime.on_new_batch_prefill(batch, 0, len(self.running))
        self.runtime.run_requests(batch.reqs)
        self.runtime.on_process_batch_result(batch)
        for req in batch.reqs:
            req.output_ids.append(1)
        self.running = []
        return batch
```

- [ ] **Step 7: Run scheduler tests**

Run:

```bash
uv run pytest test/unit/simulator/test_scheduler.py -q
```

Expected: `2 passed`.

- [ ] **Step 8: Commit scheduling runtime**

Run:

```bash
git add python/sglsim/srt/scheduling python/sglsim/srt/simulator/scheduler.py test/unit/simulator/test_scheduler.py
git commit -m "feat: add cpu-only scheduling runtime"
```

Expected: commit succeeds.

---

### Task 5: Migrate KV and HiCache Metadata Simulation

**Files:**
- Create: `python/sglsim/srt/mem_cache/__init__.py`
- Create: `python/sglsim/srt/mem_cache/allocator.py`
- Create: `python/sglsim/srt/mem_cache/radix_cache.py`
- Create: `python/sglsim/srt/mem_cache/hicache.py`
- Create: `test/unit/simulator/test_mem_cache.py`

- [ ] **Step 1: Write mem-cache tests first**

Create `test/unit/simulator/test_mem_cache.py`:

```python
from pathlib import Path

from sglsim.srt.mem_cache.allocator import PageAllocator
from sglsim.srt.mem_cache.hicache import HiCacheStorage, estimate_transfer_seconds
from sglsim.srt.mem_cache.radix_cache import PrefixCache


def test_page_allocator_allocates_and_releases_pages():
    allocator = PageAllocator(num_pages=4, page_size=16)

    pages = allocator.allocate(2)
    assert pages == [0, 1]
    allocator.release([0])
    assert allocator.allocate(1) == [0]


def test_prefix_cache_matches_longest_prefix():
    cache = PrefixCache()
    cache.insert([1, 2, 3], pages=[10, 11, 12])

    match = cache.match([1, 2, 4])

    assert match.tokens == [1, 2]
    assert match.pages == [10, 11]


def test_hicache_storage_persists_keys(tmp_path: Path):
    path = tmp_path / "keys.txt"
    storage = HiCacheStorage(storage_file_path=str(path))

    assert not storage.exists("abc")
    storage.set("abc")
    assert storage.exists("abc")
    assert HiCacheStorage(storage_file_path=str(path)).exists("abc")


def test_transfer_estimation_uses_bandwidth():
    seconds = estimate_transfer_seconds(num_tokens=4, bytes_per_token=1024, bandwidth_bytes=2048)

    assert seconds == 2.0
```

- [ ] **Step 2: Run mem-cache tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_mem_cache.py -q
```

Expected: FAIL because `sglsim.srt.mem_cache` does not exist.

- [ ] **Step 3: Implement page allocator**

Create `python/sglsim/srt/mem_cache/__init__.py`:

```python
"""Metadata-only KV/cache simulation."""
```

Create `python/sglsim/srt/mem_cache/allocator.py`:

```python
from __future__ import annotations


class PageAllocator:
    def __init__(self, num_pages: int, page_size: int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.free_pages = list(range(num_pages))

    def allocate(self, count: int) -> list[int]:
        if count > len(self.free_pages):
            raise RuntimeError("Not enough free pages")
        pages = self.free_pages[:count]
        del self.free_pages[:count]
        return pages

    def release(self, pages: list[int]) -> None:
        self.free_pages = sorted(set(self.free_pages).union(pages))
```

- [ ] **Step 4: Implement prefix cache**

Create `python/sglsim/srt/mem_cache/radix_cache.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefixMatch:
    tokens: list[int]
    pages: list[int]


class PrefixCache:
    def __init__(self):
        self.entries: dict[tuple[int, ...], list[int]] = {}

    def insert(self, tokens: list[int], pages: list[int]) -> None:
        if len(tokens) != len(pages):
            raise ValueError("tokens and pages must have the same length")
        self.entries[tuple(tokens)] = list(pages)

    def match(self, tokens: list[int]) -> PrefixMatch:
        best_tokens: tuple[int, ...] = ()
        best_pages: list[int] = []
        for cached_tokens, cached_pages in self.entries.items():
            prefix_len = 0
            for left, right in zip(cached_tokens, tokens):
                if left != right:
                    break
                prefix_len += 1
            if prefix_len > len(best_tokens):
                best_tokens = cached_tokens[:prefix_len]
                best_pages = cached_pages[:prefix_len]
        return PrefixMatch(tokens=list(best_tokens), pages=best_pages)
```

- [ ] **Step 5: Implement HiCache metadata storage**

Create `python/sglsim/srt/mem_cache/hicache.py`:

```python
from __future__ import annotations

from pathlib import Path


def estimate_transfer_seconds(
    num_tokens: int,
    bytes_per_token: int,
    bandwidth_bytes: float | None,
) -> float:
    if bandwidth_bytes is None or bandwidth_bytes <= 0 or num_tokens <= 0:
        return 0.0
    return num_tokens * bytes_per_token / bandwidth_bytes


class HiCacheStorage:
    def __init__(self, storage_file_path: str = "/tmp/sglsim/hicache_storage_keys.txt"):
        self.storage_file_path = Path(storage_file_path)
        self.storage_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage: set[str] = set()
        if self.storage_file_path.exists():
            self.storage.update(
                line.strip() for line in self.storage_file_path.read_text().splitlines() if line.strip()
            )

    def set(self, key: str) -> bool:
        if key not in self.storage:
            self.storage.add(key)
            with self.storage_file_path.open("a") as f:
                f.write(key + "\n")
        return True

    def exists(self, key: str) -> bool:
        return key in self.storage

    def clear(self) -> bool:
        self.storage.clear()
        self.storage_file_path.write_text("")
        return True
```

- [ ] **Step 6: Run mem-cache tests**

Run:

```bash
uv run pytest test/unit/simulator/test_mem_cache.py -q
```

Expected: `4 passed`.

- [ ] **Step 7: Commit mem-cache simulation**

Run:

```bash
git add python/sglsim/srt/mem_cache test/unit/simulator/test_mem_cache.py
git commit -m "feat: add metadata kv cache simulation"
```

Expected: commit succeeds.

---

### Task 6: Add Simulator Benchmark and Workload Replay

**Files:**
- Create: `python/sglsim/bench_serving.py`
- Create: `test/unit/simulator/test_bench_serving.py`

- [ ] **Step 1: Write benchmark tests first**

Create `test/unit/simulator/test_bench_serving.py`:

```python
import json
from pathlib import Path

from sglsim.bench_serving import DatasetRow, inject_simulation_params, load_simulator_metrics
from sglsim.srt.simulator.metrics import RequestStats, dump_simulator_outputs


def test_inject_simulation_params():
    row = DatasetRow(prompt="hello", prompt_len=1, output_len=2)

    inject_simulation_params(row, created_time=1.25, total_request=3)

    sim = row.extra_request_body["sampling_params"]["custom_params"]["simulation"]
    assert sim["created_time"] == 1.25
    assert sim["total_request"] == 3


def test_load_simulator_metrics(tmp_path: Path):
    dump_simulator_outputs(
        str(tmp_path),
        [RequestStats(rid="r", input_length=1, output_length=1, created_time=0, last_event_time=0.1, queue_start=0, queue_end=0, gen_token_latencies=[0.1])],
        [],
        num_warmup=0,
    )

    metrics = load_simulator_metrics(str(tmp_path))

    assert metrics["completed"] == 1
    assert json.loads((tmp_path / "metrics.json").read_text())["completed"] == 1
```

- [ ] **Step 2: Run benchmark tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_bench_serving.py -q
```

Expected: FAIL because `sglsim.bench_serving` does not exist.

- [ ] **Step 3: Implement benchmark helpers**

Create `python/sglsim/bench_serving.py`:

```python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    timestamp: float | None = None
    extra_request_body: dict[str, Any] = field(default_factory=dict)


def inject_simulation_params(row: DatasetRow, created_time: float, total_request: int) -> None:
    sampling_params = row.extra_request_body.setdefault("sampling_params", {})
    custom_params = sampling_params.setdefault("custom_params", {})
    custom_params["simulation"] = {
        "created_time": created_time,
        "total_request": total_request,
    }


def load_simulator_metrics(output_dir: str) -> dict[str, Any]:
    metrics_path = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Simulator metrics file not found: {metrics_path}")
    with open(metrics_path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SGLSim benchmark helper")
    parser.add_argument("--simulator-output-dir", default="/tmp/sglsim/output")
    args = parser.parse_args(argv)
    metrics = load_simulator_metrics(args.simulator_output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run benchmark tests**

Run:

```bash
uv run pytest test/unit/simulator/test_bench_serving.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit benchmark helpers**

Run:

```bash
git add python/sglsim/bench_serving.py test/unit/simulator/test_bench_serving.py
git commit -m "feat: add sglsim benchmark helpers"
```

Expected: commit succeeds.

---

### Task 7: Add Pure Simulator Server Path

**Files:**
- Create: `python/sglsim/srt/http/__init__.py`
- Create: `python/sglsim/srt/http/server.py`
- Modify: `python/sglsim/launch_server.py`
- Create: `test/unit/simulator/test_server.py`

- [ ] **Step 1: Write server tests first**

Create `test/unit/simulator/test_server.py`:

```python
import json
from pathlib import Path

from sglsim.srt.http.server import SimulatorApp, GeneratePayload


class FixedPredictor:
    def predict_requests(self, requests):
        return 0.1


def test_simulator_app_generates_mock_text(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"predictor": {"name": "aiconfigurator"}}))
    app = SimulatorApp(output_dir=str(tmp_path), mode="offline", predictor=FixedPredictor())

    result = app.generate(GeneratePayload(input_ids=[1, 2], max_new_tokens=2))

    assert result["output_ids"] == [1, 1]
    assert result["text"] == "<simulated>"
    assert (tmp_path / "metrics.json").exists()
```

- [ ] **Step 2: Run server tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_server.py -q
```

Expected: FAIL because HTTP server module does not exist.

- [ ] **Step 3: Implement simulator app without HTTP dependency first**

Create `python/sglsim/srt/http/__init__.py`:

```python
"""Optional HTTP interface for SGLSim."""
```

Create `python/sglsim/srt/http/server.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from sglsim.srt.scheduling.request import GenerateRequest, SamplingParams
from sglsim.srt.scheduling.scheduler import CPUScheduler
from sglsim.srt.simulator.scheduler import SchedulerSimulatorRuntime


@dataclass
class GeneratePayload:
    input_ids: list[int]
    max_new_tokens: int
    created_time: float = 0.0


class SimulatorApp:
    def __init__(self, output_dir: str, mode: str, predictor):
        self.runtime = SchedulerSimulatorRuntime(
            mode=mode,
            output_dir=output_dir,
            num_warmup=0,
            overlap_schedule=False,
            predictor=predictor,
        )
        self.scheduler = CPUScheduler(self.runtime)

    def generate(self, payload: GeneratePayload) -> dict:
        req = GenerateRequest(
            rid=str(uuid4()),
            input_ids=payload.input_ids,
            sampling_params=SamplingParams(
                max_new_tokens=payload.max_new_tokens,
                custom_params={
                    "simulation": {
                        "created_time": payload.created_time,
                        "total_request": 1,
                    }
                },
            ),
        )
        self.scheduler.recv_requests([req])
        for _ in range(payload.max_new_tokens):
            self.scheduler.step()
        self.runtime.dump_outputs()
        return {"text": "<simulated>", "output_ids": req.output_ids}
```

- [ ] **Step 4: Run server tests**

Run:

```bash
uv run pytest test/unit/simulator/test_server.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Wire launch_server to simulator app**

Modify `python/sglsim/launch_server.py` to keep `main()` CLI parsing and instantiate the app with a predictor. For this task, use a deterministic fixed predictor when `--sim-config-path` points to a test config and AIConfigurator is unavailable:

```python
class FixedPredictor:
    def predict_requests(self, requests):
        return 0.001 * max(1, sum(item["extend_length"] for item in requests))
```

Then print:

```text
SGLSim simulator server ready on HOST:PORT
```

Do not start uvicorn in this task. A real HTTP server can be a later plan once the simulator core is stable.

- [ ] **Step 6: Run launch command**

Run:

```bash
uv run python -m sglsim.launch_server --sim-config-path test/unit/simulator/assets/config.json
```

Expected output includes:

```text
SGLSim simulator server ready
```

- [ ] **Step 7: Commit server path**

Run:

```bash
git add python/sglsim/launch_server.py python/sglsim/srt/http test/unit/simulator/test_server.py
git commit -m "feat: add pure simulator server path"
```

Expected: commit succeeds.

---

### Task 8: Remove GPU Runtime From New Root and Verify Archive Boundary

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- No modifications under `Archived/`

- [ ] **Step 1: Verify new root has no `sglang` import outside archive**

Run:

```bash
grep -R -n "from sglang\|import sglang\|python -m sglang\|sglang\." \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=Archived \
  . || true
```

Expected: no output.

- [ ] **Step 2: Verify new root has no GPU runtime keywords outside archive**

Run:

```bash
grep -R -n "torch.cuda\|cuda_graph\|NCCL\|nccl\|triton\|flashinfer\|sgl_kernel\|gpu_id" \
  --exclude-dir=.git \
  --exclude-dir=.venv \
  --exclude-dir=Archived \
  python test README.md pyproject.toml || true
```

Expected: no output, except comments that explicitly document excluded features. If comments are found, keep them only in `README.md`.

- [ ] **Step 3: Update README with supported scope**

Modify `README.md`:

```markdown
# SGLSim

SGLSim is a CPU-only simulator fork derived from SGLang. The original checkpoint is preserved under `Archived/` for migration reference.

## Supported

- Offline and blocking simulator modes
- AIConfigurator latency prediction
- Request scheduling and batching simulation
- Metadata-only KV/page/cache simulation
- Simulator metrics output
- Benchmark metrics loading

## Not Supported

- Real GPU inference
- CUDA graphs and CUDA kernels
- Distributed GPU execution
- Real model weight loading
- Real KV tensor transfer
- LoRA or checkpoint weight updates
```

- [ ] **Step 4: Run full new-root tests**

Run:

```bash
uv run pytest -q
```

Expected: all tests under `test/unit` pass.

- [ ] **Step 5: Run package smoke commands**

Run:

```bash
uv run python -m sglsim.launch_server --help
uv run python -m sglsim.launch_server --sim-config-path test/unit/simulator/assets/config.json
uv run python -m sglsim.bench_serving --simulator-output-dir /tmp/sglsim/output
```

Expected:

```text
--help command exits 0
launch command prints SGLSim simulator server ready
bench command either prints metrics JSON if previous launch wrote it, or raises FileNotFoundError with the /tmp/sglsim/output/metrics.json path
```

If the bench command raises because no metrics file exists, run a small script that creates metrics via `SimulatorApp.generate()` and re-run the bench command.

- [ ] **Step 6: Commit cleanup**

Run:

```bash
git add README.md pyproject.toml
git commit -m "docs: define sglsim simulator-only scope"
```

Expected: commit succeeds.

---

## Final Verification

Run:

```bash
git status --short --untracked-files=all
uv run pytest -q
grep -R -n "from sglang\|import sglang\|python -m sglang\|sglang\." --exclude-dir=.git --exclude-dir=.venv --exclude-dir=Archived . || true
grep -R -n "torch.cuda\|cuda_graph\|NCCL\|nccl\|triton\|flashinfer\|sgl_kernel\|gpu_id" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=Archived python test README.md pyproject.toml || true
```

Expected:

```text
git status shows no uncommitted changes
pytest passes
sglang grep has no output outside Archived
GPU runtime grep has no output outside Archived except README documentation lines
```

## Self-Review Checklist

- Spec coverage: plan covers archive, package rename, uv/Python 3.12 setup, simulator core, scheduling, KV/cache metadata, benchmark/server path, GPU runtime exclusion, and final verification.
- Placeholder scan: implementation steps list exact file paths and concrete code or migration instructions; no unresolved marker text remains.
- Type consistency: `ServerArgs`, `SimulatorSettings`, `ScheduleRequest`, `GenerateRequest`, `CPUScheduler`, `SchedulerSimulatorRuntime`, `DatasetRow`, and `SimulatorApp` are introduced before later tasks use them.
