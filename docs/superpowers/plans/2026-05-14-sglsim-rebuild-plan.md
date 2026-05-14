# SGLSim Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sglsim` outside `Archived/` as a CPU-only simulator fork, migrating only scheduling, cache metadata, simulator metrics, workload replay, and AIConfigurator prediction from the archived SGLang checkpoint.

**Architecture:** `Archived/` is the immutable reference tree. New source lives under `python/sglsim`, tests under `test/unit`, and plans under root `docs/superpowers/plans`. The new runtime does not import `sglang`; it uses `sglsim` imports and replaces real GPU inference with CPU-only metadata simulation.

**Tech Stack:** Python 3.12, uv, pytest, NumPy, optional FastAPI/uvicorn for a later HTTP layer, AIConfigurator SDK as the default predictor provider.

---

## Current State

The checkpoint project has already been moved under `Archived/`.

Root should contain only:

```text
Archived/
docs/
.git/
.gitignore
README.md
```

All future implementation and plan files must be created outside `Archived/`. Do not edit files under `Archived/` except by explicit user request.

## Migration Sources

Use these archived files as reference material:

```text
Archived/python/sglang/srt/simulator/
Archived/test/registered/unit/simulator/
Archived/python/sglang/bench_serving.py
Archived/python/sglang/srt/managers/scheduler.py
Archived/python/sglang/srt/managers/schedule_batch.py
Archived/python/sglang/srt/mem_cache/
Archived/python/sglang/srt/server_args.py
```

Do not copy GPU runtime modules wholesale. Extract simulator-relevant logic and rename imports to `sglsim`.

## Subagent Work Allocation

Use subagents only after Task 1 creates the package scaffold. Suggested ownership:

- Worker A: package bootstrap and CLI, files under `pyproject.toml`, `python/sglsim/__init__.py`, `python/sglsim/launch_server.py`, `python/sglsim/srt/server_args.py`, `test/unit/simulator/test_package.py`.
- Worker B: simulator core, files under `python/sglsim/srt/simulator/*` and tests `test/unit/simulator/test_config.py`, `test_predictor.py`, `test_metrics.py`.
- Worker C: scheduling, files under `python/sglsim/srt/scheduling/*` and `test/unit/simulator/test_scheduler.py`.
- Worker D: KV/cache metadata, files under `python/sglsim/srt/mem_cache/*` and `test/unit/simulator/test_mem_cache.py`.
- Worker E: benchmark/server integration, files `python/sglsim/bench_serving.py`, `python/sglsim/srt/http/*`, `scripts/sglsim_smoke.py`, and related tests.

Agents must not write outside their owned paths.

---

### Task 1: Bootstrap Package and Python 3.12 uv Environment

**Files:**
- Create: `pyproject.toml`
- Create: `python/sglsim/__init__.py`
- Create: `python/sglsim/launch_server.py`
- Create: `python/sglsim/srt/__init__.py`
- Create: `python/sglsim/srt/server_args.py`
- Create: `test/unit/simulator/test_package.py`

- [ ] **Step 1: Create directories**

Run:

```bash
mkdir -p python/sglsim/srt test/unit/simulator
```

Expected: directories exist outside `Archived/`.

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

- [ ] **Step 3: Create package init files**

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
        parser.add_argument("--simulator-output-dir", default=ServerArgs.simulator_output_dir)
        parser.add_argument("--simulator-num-warmup", type=int, default=0)
        parser.add_argument("--host", default=ServerArgs.host)
        parser.add_argument("--port", type=int, default=ServerArgs.port)

    @classmethod
    def from_cli_args(cls, args: list[str] | None = None) -> "ServerArgs":
        parser = argparse.ArgumentParser(description="SGLSim CPU-only simulator")
        cls.add_cli_args(parser)
        parsed = cls(**vars(parser.parse_args(args)))
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
        f"SGLSim simulator server ready on {args.host}:{args.port}\n"
        f"mode={args.simulator_mode}\n"
        f"config={args.sim_config_path}\n"
        f"output_dir={args.simulator_output_dir}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Create package tests**

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

- [ ] **Step 8: Run package tests**

Run:

```bash
uv run pytest test/unit/simulator/test_package.py -q
```

Expected: `2 passed`.

- [ ] **Step 9: Commit bootstrap**

Run:

```bash
git add -A
git commit -m "chore: archive original tree and bootstrap sglsim"
```

Expected: commit succeeds and includes the archive move plus bootstrap files.

---

### Task 2: Migrate Simulator Core

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

- [ ] **Step 1: Write config tests**

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


def test_settings_reject_non_aiconfigurator(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"predictor": {"name": "dummy"}}))
    args = ServerArgs.from_cli_args(["--sim-config-path", str(config)])

    with pytest.raises(ValueError, match="predictor.name must be aiconfigurator"):
        SimulatorSettings.from_server_args(args)
```

- [ ] **Step 2: Run config tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_config.py -q
```

Expected: FAIL because `sglsim.srt.simulator.config` does not exist.

- [ ] **Step 3: Implement config**

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

- [ ] **Step 4: Run config tests**

Run:

```bash
uv run pytest test/unit/simulator/test_config.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Migrate types/predictor/state/metrics**

Use these files as reference and rename imports from `sglang` to `sglsim`:

```text
Archived/python/sglang/srt/simulator/types.py
Archived/python/sglang/srt/simulator/predictor.py
Archived/python/sglang/srt/simulator/predictor_aiconfigurator.py
Archived/python/sglang/srt/simulator/state.py
Archived/python/sglang/srt/simulator/metrics.py
Archived/test/registered/unit/simulator/test_simulator_config.py
Archived/test/registered/unit/simulator/test_state_metrics.py
```

Required behavior changes:

```text
Default paths use /tmp/sglsim instead of /tmp/sglang_simulator.
build_simulator_predictor raises ValueError if AIConfigurator cannot be constructed.
No token-count latency fallback is allowed in the final predictor path.
```

- [ ] **Step 6: Create predictor and metrics tests**

Create tests equivalent to archived simulator tests but import from `sglsim`, and include these assertions:

```python
assert config.data_type.name == "BF16"
assert config.kv_cache_data_type.name == "BF16"
assert metrics["completed"] == 1
assert state.global_clock == 0.5
```

- [ ] **Step 7: Run simulator core tests**

Run:

```bash
uv run pytest test/unit/simulator/test_config.py test/unit/simulator/test_predictor.py test/unit/simulator/test_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit simulator core**

Run:

```bash
git add python/sglsim/srt/simulator test/unit/simulator
git commit -m "feat: migrate sglsim simulator core"
```

Expected: commit succeeds.

---

### Task 3: Migrate CPU-Only Scheduling

**Files:**
- Create: `python/sglsim/srt/scheduling/__init__.py`
- Create: `python/sglsim/srt/scheduling/request.py`
- Create: `python/sglsim/srt/scheduling/batch.py`
- Create: `python/sglsim/srt/scheduling/scheduler.py`
- Create or Modify: `python/sglsim/srt/simulator/scheduler.py`
- Create: `test/unit/simulator/test_scheduler.py`

- [ ] **Step 1: Create scheduling tests**

Create `test/unit/simulator/test_scheduler.py` with offline replay and blocking immediate-arrival tests derived from:

```text
Archived/test/registered/unit/simulator/test_scheduler_simulator.py
```

Required assertions:

```python
assert [req.rid for req in first_arrival] == ["r1"]
assert [req.rid for req in second_arrival] == ["r2"]
assert runtime.request_stats["r1"].input_length == 3
```

- [ ] **Step 2: Run scheduler tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_scheduler.py -q
```

Expected: FAIL because scheduling modules do not exist.

- [ ] **Step 3: Implement request and batch dataclasses**

Create minimal CPU-only request and batch classes with these public fields:

```python
rid: str
input_ids: list[int]
sampling_params.max_new_tokens: int
sampling_params.custom_params: dict
output_ids: list[int]
cached_tokens: int
is_chunked: int
```

- [ ] **Step 4: Implement `SchedulerSimulatorRuntime`**

Migrate simulator scheduler logic from:

```text
Archived/python/sglang/srt/simulator/scheduler.py
```

Required edits:

```text
Use sglsim imports.
Remove dependencies on real ModelRunner result classes.
Add run_requests(reqs) that builds prediction requests from CPU request metadata.
Raise RuntimeError if predictor is missing.
```

- [ ] **Step 5: Implement `CPUScheduler`**

Create a simple scheduler that:

```text
receives requests through runtime.filter_recv_requests
stores arrived requests in waiting queue
forms one batch from all waiting requests per step
calls runtime.run_requests
calls runtime.on_process_batch_result
appends mock token id 1 to each request output_ids
```

- [ ] **Step 6: Run scheduler tests**

Run:

```bash
uv run pytest test/unit/simulator/test_scheduler.py -q
```

Expected: tests pass.

- [ ] **Step 7: Commit scheduling**

Run:

```bash
git add python/sglsim/srt/scheduling python/sglsim/srt/simulator/scheduler.py test/unit/simulator/test_scheduler.py
git commit -m "feat: add cpu-only scheduling runtime"
```

Expected: commit succeeds.

---

### Task 4: Migrate Metadata-Only KV and HiCache

**Files:**
- Create: `python/sglsim/srt/mem_cache/__init__.py`
- Create: `python/sglsim/srt/mem_cache/allocator.py`
- Create: `python/sglsim/srt/mem_cache/radix_cache.py`
- Create: `python/sglsim/srt/mem_cache/hicache.py`
- Create: `test/unit/simulator/test_mem_cache.py`

- [ ] **Step 1: Create mem-cache tests**

Tests must cover:

```text
page allocation and release
longest prefix match
HiCache key persistence
transfer duration = num_tokens * bytes_per_token / bandwidth_bytes
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest test/unit/simulator/test_mem_cache.py -q
```

Expected: FAIL because mem-cache modules do not exist.

- [ ] **Step 3: Implement metadata allocator**

Implement `PageAllocator` with:

```python
allocate(count: int) -> list[int]
release(pages: list[int]) -> None
```

No torch tensors are required.

- [ ] **Step 4: Implement prefix cache**

Implement `PrefixCache.insert(tokens, pages)` and `PrefixCache.match(tokens)` returning matched tokens and pages.

- [ ] **Step 5: Implement HiCache storage and transfer estimator**

Implement persistent key storage at `/tmp/sglsim/hicache_storage_keys.txt` by default and a pure arithmetic transfer estimator.

- [ ] **Step 6: Run mem-cache tests**

Run:

```bash
uv run pytest test/unit/simulator/test_mem_cache.py -q
```

Expected: tests pass.

- [ ] **Step 7: Commit mem-cache**

Run:

```bash
git add python/sglsim/srt/mem_cache test/unit/simulator/test_mem_cache.py
git commit -m "feat: add metadata kv cache simulation"
```

Expected: commit succeeds.

---

### Task 5: Add Benchmark and Simulator Server Integration

**Files:**
- Create: `python/sglsim/bench_serving.py`
- Create: `python/sglsim/srt/http/__init__.py`
- Create: `python/sglsim/srt/http/server.py`
- Modify: `python/sglsim/launch_server.py`
- Create: `test/unit/simulator/test_bench_serving.py`
- Create: `test/unit/simulator/test_server.py`

- [ ] **Step 1: Create benchmark tests**

Migrate core behavior from:

```text
Archived/test/registered/unit/simulator/test_bench_serving_simulator.py
```

Required assertions:

```python
simulation_params["created_time"] == 1.5
simulation_params["total_request"] == 3
loaded_metrics["completed"] == 1
```

- [ ] **Step 2: Implement benchmark helpers**

Implement:

```python
DatasetRow
inject_simulation_params(row, created_time, total_request)
load_simulator_metrics(output_dir)
main(argv=None)
```

Do not import the archived benchmark module.

- [ ] **Step 3: Create server integration tests**

Test that `SimulatorApp.generate()`:

```text
accepts input_ids and max_new_tokens
runs CPUScheduler for max_new_tokens steps
returns output_ids filled with mock token id 1
writes metrics.json
```

- [ ] **Step 4: Implement simulator app**

Create `python/sglsim/srt/http/server.py` with a dependency-free `SimulatorApp` class first. Do not require FastAPI in this task.

- [ ] **Step 5: Wire launch_server**

`python -m sglsim.launch_server --sim-config-path test/unit/simulator/assets/config.json` should build settings and print `SGLSim simulator server ready` without importing GPU runtime.

- [ ] **Step 6: Run integration tests**

Run:

```bash
uv run pytest test/unit/simulator/test_bench_serving.py test/unit/simulator/test_server.py -q
uv run python -m sglsim.launch_server --sim-config-path test/unit/simulator/assets/config.json
```

Expected: tests pass and launch command exits 0.

- [ ] **Step 7: Commit integration**

Run:

```bash
git add python/sglsim/bench_serving.py python/sglsim/launch_server.py python/sglsim/srt/http test/unit/simulator/test_bench_serving.py test/unit/simulator/test_server.py
git commit -m "feat: add sglsim benchmark and server integration"
```

Expected: commit succeeds.

---

### Task 6: Final Cleanup and Boundary Verification

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml` if dependency cleanup is needed

- [ ] **Step 1: Check no archived package imports outside archive**

Run:

```bash
python - <<'PY'
from pathlib import Path
needles = ("from " + "sglang", "import " + "sglang", "python -m " + "sglang", "sglang" + ".")
for path in Path('.').rglob('*'):
    if not path.is_file() or any(part in {'.git', '.venv', 'Archived'} for part in path.parts):
        continue
    text = path.read_text(errors='ignore')
    for index, line in enumerate(text.splitlines(), 1):
        if any(needle in line for needle in needles):
            print(f"{path}:{index}:{line}")
PY
```

Expected: no output.

- [ ] **Step 2: Check no GPU runtime keywords outside archive**

Run:

```bash
python - <<'PY'
from pathlib import Path
needles = ("torch" + ".cuda", "cuda" + "_graph", "NC" + "CL", "nc" + "cl", "tri" + "ton", "flash" + "infer", "sgl" + "_kernel", "gpu" + "_id")
for root in (Path('python'), Path('test'), Path('README.md'), Path('pyproject.toml')):
    paths = [root] if root.is_file() else root.rglob('*')
    for path in paths:
        if not path.is_file() or any(part in {'.git', '.venv', 'Archived'} for part in path.parts):
            continue
        text = path.read_text(errors='ignore')
        for index, line in enumerate(text.splitlines(), 1):
            if any(needle in line for needle in needles):
                print(f"{path}:{index}:{line}")
PY
```

Expected: no output except README documentation lines that explicitly list unsupported GPU features.

- [ ] **Step 3: Update README scope**

Ensure `README.md` includes:

```markdown
## Supported

- Offline and blocking simulator modes
- AIConfigurator latency prediction
- CPU-only request scheduling and batching simulation
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

- [ ] **Step 4: Run full tests**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit cleanup**

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
python - <<'PY'
from pathlib import Path
needles = ("from " + "sglang", "import " + "sglang", "python -m " + "sglang", "sglang" + ".")
for path in Path('.').rglob('*'):
    if path.is_file() and not any(part in {'.git', '.venv', 'Archived'} for part in path.parts):
        for index, line in enumerate(path.read_text(errors='ignore').splitlines(), 1):
            if any(needle in line for needle in needles):
                print(f"{path}:{index}:{line}")
PY
python - <<'PY'
from pathlib import Path
needles = ("torch" + ".cuda", "cuda" + "_graph", "NC" + "CL", "nc" + "cl", "tri" + "ton", "flash" + "infer", "sgl" + "_kernel", "gpu" + "_id")
for root in (Path('python'), Path('test'), Path('README.md'), Path('pyproject.toml')):
    paths = [root] if root.is_file() else root.rglob('*')
    for path in paths:
        if path.is_file():
            for index, line in enumerate(path.read_text(errors='ignore').splitlines(), 1):
                if any(needle in line for needle in needles):
                    print(f"{path}:{index}:{line}")
PY
```

Expected:

```text
git status shows no uncommitted changes
pytest passes
sglang grep has no output outside Archived
GPU runtime grep has no output outside Archived except README documentation lines
```
