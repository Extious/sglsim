# Strict SGLang Runtime Parity Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `sglsim` so that HTTP, tokenizer, scheduler, batch lifecycle, KV/cache management, HiCache, remote backup, failure injection, failover retry, and benchmark metadata follow `/home/comp/zhanzhao/sglang` behavior, while only GPU model forward, CUDA kernels, real KV tensor transfer, and GPU/distributed communication are simulated on CPU.

**Architecture:** Use a controlled vendoring model: copy selected upstream SGLang runtime modules into `python/sglsim/srt` with `sglang` imports rewritten to `sglsim`, then keep simulation-specific behavior in a narrow patch layer under `python/sglsim/srt/simulator/native/`. The current standalone FastAPI `RuntimeService` path becomes a temporary compatibility fallback and is removed from authoritative online experiments once native mode passes parity tests.

**Tech Stack:** Python 3.12, uv, pytest, FastAPI/uvicorn, pyzmq, torch CPU tensors where unavoidable, HuggingFace tokenizer when available, AIConfigurator latency predictor, upstream SGLang source tree at `/home/comp/zhanzhao/sglang`.

---

## Non-Negotiable Invariants

These invariants define success:

```text
1. benchmark/online sends real HTTP requests to sglsim.
2. sglsim.launch_server starts a SGLang-shaped runtime, not RuntimeService.submit.
3. TokenizerManager, DataParallelController, Scheduler, ScheduleBatch, Req, SchedulePolicy, cache controller, radix cache, HiCache, and remote backup semantics come from upstream SGLang logic.
4. GPU model forward is replaced by CPU simulated ModelRunner output.
5. CUDA/Triton allocation kernels are replaced by CPU allocation functions with identical input/output contracts.
6. Real KV tensor transfer is replaced by metadata movement plus latency accounting.
7. Failure API enters the same control path as SGLang: HTTP -> TokenizerManager -> DPC -> target Scheduler -> snapshot -> tokenizer failover dispatch -> healthy Scheduler retry.
8. cache_hits.csv values are derived from runtime cache metadata, never from fixed ratios.
9. Remote backup / host backup / baseline differences arise from cache controller, HiCache, and remote backup state.
10. A guard test fails if authoritative online experiments use python/sglsim/srt/managers/runtime_service.py.
```

## Reference Sources

Use `/home/comp/zhanzhao/sglang` on branch `multi-level-backup` unless the user pins a commit.

Critical upstream files:

```text
/home/comp/zhanzhao/sglang/python/sglang/srt/entrypoints/http_server.py
/home/comp/zhanzhao/sglang/python/sglang/srt/entrypoints/engine.py
/home/comp/zhanzhao/sglang/python/sglang/srt/entrypoints/openai/
/home/comp/zhanzhao/sglang/python/sglang/srt/managers/tokenizer_manager.py
/home/comp/zhanzhao/sglang/python/sglang/srt/managers/data_parallel_controller.py
/home/comp/zhanzhao/sglang/python/sglang/srt/managers/scheduler.py
/home/comp/zhanzhao/sglang/python/sglang/srt/managers/schedule_batch.py
/home/comp/zhanzhao/sglang/python/sglang/srt/managers/schedule_policy.py
/home/comp/zhanzhao/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py
/home/comp/zhanzhao/sglang/python/sglang/srt/managers/scheduler_runtime_checker_mixin.py
/home/comp/zhanzhao/sglang/python/sglang/srt/managers/cache_controller.py
/home/comp/zhanzhao/sglang/python/sglang/srt/mem_cache/
/home/comp/zhanzhao/sglang/python/sglang/srt/model_executor/model_runner.py
/home/comp/zhanzhao/sglang/python/sglang/srt/server_args.py
/home/comp/zhanzhao/sglang/tools/sglang-simulator/src/sglang_simulator/simulation/sglang/
```

## Target File Structure

New or substantially changed files:

```text
tools/sync_sglang_native_runtime.py
tools/check_sglang_native_parity.py

docs/sglang-native-parity-contract.md
docs/sglang-native-parity-status.md

python/sglsim/launch_server.py
python/sglsim/srt/server_args.py
python/sglsim/srt/entrypoints/
python/sglsim/srt/managers/
python/sglsim/srt/mem_cache/
python/sglsim/srt/model_executor/
python/sglsim/srt/simulator/native/bootstrap.py
python/sglsim/srt/simulator/native/config.py
python/sglsim/srt/simulator/native/model_runner_patch.py
python/sglsim/srt/simulator/native/scheduler_patch.py
python/sglsim/srt/simulator/native/cache_controller_patch.py
python/sglsim/srt/simulator/native/cpu_allocator_kernels.py
python/sglsim/srt/simulator/native/memory_pool_patch.py
python/sglsim/srt/simulator/native/remote_comm.py
python/sglsim/srt/simulator/native/guards.py

test/parity/test_native_file_manifest.py
test/parity/test_no_authoritative_runtime_service.py
test/unit/native/test_cpu_allocator_kernels.py
test/unit/native/test_simulated_model_runner_patch.py
test/unit/native/test_native_cache_controller_patch.py
test/integration/test_native_http_completion.py
test/integration/test_native_failure_failover.py
test/integration/test_native_remote_backup_cache_hits.py
test/integration/test_online_fixed_native_sglang_parity.py
```

Existing files to retire from the authoritative path:

```text
python/sglsim/srt/managers/runtime_service.py
python/sglsim/srt/http/server.py
python/sglsim/srt/managers/scheduler.py        # current simplified implementation
python/sglsim/srt/mem_cache/radix_cache.py     # current simplified implementation
```

They can remain temporarily for old tests, but native online tests must not import or call them.

---

### Task 1: Freeze The Upstream Reference And Add Drift Checks

**Files:**
- Create: `docs/sglang-native-parity-contract.md`
- Create: `tools/check_sglang_native_parity.py`
- Create: `test/parity/test_native_file_manifest.py`

- [x] **Step 1: Record the exact upstream reference**

Run:

```bash
git -C /home/comp/zhanzhao/sglang rev-parse HEAD
git -C /home/comp/zhanzhao/sglang branch --show-current
```

Expected: branch is `multi-level-backup`; write the commit hash into `docs/sglang-native-parity-contract.md`.

- [x] **Step 2: Create the parity contract document**

Create `docs/sglang-native-parity-contract.md` with this content:

```markdown
# SGLang Native Parity Contract

Reference tree: `/home/comp/zhanzhao/sglang`
Reference branch: `multi-level-backup`
Reference commit: `55044f182cd84544a03b50414644a2fa7a54640a`

Authoritative runtime rule:

`benchmark/online` must exercise a SGLang-shaped runtime. The online path must not use `RuntimeService.submit`.

Preserved upstream modules:

- `srt.entrypoints.http_server`
- `srt.entrypoints.engine`
- `srt.entrypoints.openai`
- `srt.managers.tokenizer_manager`
- `srt.managers.data_parallel_controller`
- `srt.managers.scheduler`
- `srt.managers.schedule_batch`
- `srt.managers.schedule_policy`
- `srt.managers.scheduler_output_processor_mixin`
- `srt.managers.scheduler_runtime_checker_mixin`
- `srt.managers.cache_controller`
- `srt.mem_cache`
- `srt.model_executor.model_runner` public contracts

Allowed simulation replacements:

- `ModelRunner.initialize`
- `ModelRunner.forward`
- `ModelRunner.sample`
- CUDA/Triton allocation kernels
- KV host/device/storage transfer payloads
- inter-rank process communication when local logical ranks are used
- latency prediction and wall-clock sleep

Forbidden shortcuts:

- fixed cache recovery ratios
- request-end synthetic failover metadata
- bypassing TokenizerManager/DataParallelController/Scheduler for failure handling
- online benchmark results produced by offline post-processing
```

- [x] **Step 3: Add a manifest checker**

Create `tools/check_sglang_native_parity.py`:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/home/comp/zhanzhao/sglang/python/sglang")

PARITY_FILES = [
    "srt/entrypoints/http_server.py",
    "srt/entrypoints/engine.py",
    "srt/managers/tokenizer_manager.py",
    "srt/managers/data_parallel_controller.py",
    "srt/managers/scheduler.py",
    "srt/managers/schedule_batch.py",
    "srt/managers/schedule_policy.py",
    "srt/managers/scheduler_output_processor_mixin.py",
    "srt/managers/scheduler_runtime_checker_mixin.py",
    "srt/managers/cache_controller.py",
    "srt/mem_cache/radix_cache.py",
    "srt/mem_cache/hiradix_cache.py",
    "srt/mem_cache/hicache_storage.py",
    "srt/mem_cache/memory_pool.py",
    "srt/mem_cache/memory_pool_host.py",
    "srt/mem_cache/allocator.py",
    "srt/model_executor/model_runner.py",
    "srt/server_args.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest() -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}
    for rel in PARITY_FILES:
        upstream = UPSTREAM / rel
        local = REPO / "python" / "sglsim" / rel
        data[rel] = {
            "upstream_exists": str(upstream.is_file()),
            "local_exists": str(local.is_file()),
            "upstream_sha256": _sha256(upstream) if upstream.is_file() else "",
            "local_sha256": _sha256(local) if local.is_file() else "",
        }
    return data


def main() -> int:
    print(json.dumps(build_manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4: Add the parity manifest test**

Create `test/parity/test_native_file_manifest.py`:

```python
from tools.check_sglang_native_parity import PARITY_FILES, build_manifest


def test_upstream_reference_files_exist():
    manifest = build_manifest()
    missing = [
        rel
        for rel in PARITY_FILES
        if manifest[rel]["upstream_exists"] != "True"
    ]
    assert missing == []
```

- [x] **Step 5: Run the parity test**

Run:

```bash
uv run pytest -q test/parity/test_native_file_manifest.py
```

Expected: pass.

---

### Task 2: Add Controlled Upstream Sync With Import Rewrite

**Files:**
- Create: `tools/sync_sglang_native_runtime.py`
- Modify: `.gitignore`
- Test: `test/parity/test_native_file_manifest.py`

- [x] **Step 1: Create the sync script**

Create `tools/sync_sglang_native_runtime.py`:

```python
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path("/home/comp/zhanzhao/sglang/python/sglang")
LOCAL_ROOT = REPO / "python" / "sglsim"

COPY_DIRS = [
    "srt/entrypoints",
    "srt/managers",
    "srt/mem_cache",
    "srt/model_executor",
]

COPY_FILES = [
    "srt/server_args.py",
]

EXCLUDE_SUFFIXES = {".pyc", ".so"}
EXCLUDE_DIR_NAMES = {"__pycache__"}


def rewrite_imports(text: str) -> str:
    text = text.replace("from sglang.", "from sglsim.")
    text = text.replace("import sglang.", "import sglsim.")
    text = text.replace("sglang.srt.", "sglsim.srt.")
    text = text.replace("sglang.lang.", "sglsim.lang.")
    return text


def should_copy(path: Path) -> bool:
    if path.name in EXCLUDE_DIR_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return path.suffix == ".py"


def copy_file(rel: str) -> None:
    src = UPSTREAM_ROOT / rel
    dst = LOCAL_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(rewrite_imports(src.read_text()), encoding="utf-8")


def copy_dir(rel_dir: str) -> None:
    src_dir = UPSTREAM_ROOT / rel_dir
    dst_dir = LOCAL_ROOT / rel_dir
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    for src in src_dir.rglob("*"):
        if src.is_dir() or not should_copy(src):
            continue
        rel = src.relative_to(UPSTREAM_ROOT)
        copy_file(str(rel))


def main() -> int:
    for rel_dir in COPY_DIRS:
        copy_dir(rel_dir)
    for rel in COPY_FILES:
        copy_file(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 2: Run the sync script**

Run:

```bash
uv run python tools/sync_sglang_native_runtime.py
```

Expected: upstream runtime files are copied into `python/sglsim/srt/...` and imports use `sglsim`.

- [x] **Step 3: Update `.gitignore` only for generated cache files**

Ensure `.gitignore` contains:

```gitignore
__pycache__/
*.pyc
.pytest_cache/
```

- [x] **Step 4: Run import smoke tests**

Run:

```bash
uv run python -c "import sglsim.srt.server_args; import sglsim.srt.managers.scheduler; import sglsim.srt.mem_cache.radix_cache"
```

Expected: imports either pass or fail on a concrete missing dependency. If a dependency is missing, add it in Task 3, not by editing upstream logic.

---

### Task 3: Add Native Runtime Dependencies And CPU Platform Shims

**Files:**
- Modify: `pyproject.toml`
- Create: `python/sglsim/srt/simulator/native/config.py`
- Create: `python/sglsim/srt/simulator/native/guards.py`
- Test: `test/unit/native/test_native_config.py`

- [x] **Step 1: Extend optional dependencies**

Modify `pyproject.toml`:

```toml
[project.optional-dependencies]
native = [
  "aiohttp>=3.9",
  "fastapi>=0.110",
  "httpx>=0.27",
  "orjson>=3.9",
  "psutil>=5.9",
  "pyzmq>=25",
  "requests>=2.31",
  "torch",
  "transformers>=4.51",
  "uvicorn>=0.27",
]
```

Keep existing `server`, `aiconfigurator`, and `test` extras.

- [x] **Step 2: Add native simulator config**

Create `python/sglsim/srt/simulator/native/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeSimulationConfig:
    sim_config_path: str
    output_dir: str
    mode: str = "blocking"
    num_warmup: int = 0
    enable_native_runtime: bool = True

    @classmethod
    def from_server_args(cls, server_args) -> "NativeSimulationConfig":
        config_path = str(getattr(server_args, "sim_config_path", "") or "")
        output_dir = str(getattr(server_args, "simulator_output_dir", "/tmp/sglsim/output"))
        mode = str(getattr(server_args, "simulator_mode", "blocking") or "blocking")
        num_warmup = int(getattr(server_args, "simulator_num_warmup", 0) or 0)
        if config_path and not Path(config_path).is_file():
            raise ValueError(f"sim_config_path does not exist: {config_path}")
        if mode not in {"blocking", "offline"}:
            raise ValueError(f"unsupported simulator mode: {mode}")
        return cls(
            sim_config_path=config_path,
            output_dir=output_dir,
            mode=mode,
            num_warmup=num_warmup,
        )
```

- [x] **Step 3: Add authoritative path guard**

Create `python/sglsim/srt/simulator/native/guards.py`:

```python
from __future__ import annotations


class AuthoritativePathViolation(RuntimeError):
    pass


def assert_not_runtime_service_path(component: str) -> None:
    raise AuthoritativePathViolation(
        f"{component} is using RuntimeService; native SGLang parity path must use "
        "TokenizerManager/DataParallelController/Scheduler instead."
    )
```

- [x] **Step 4: Add config tests**

Create `test/unit/native/test_native_config.py`:

```python
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglsim.srt.simulator.native.config import NativeSimulationConfig


def test_native_config_from_server_args(tmp_path: Path):
    cfg = tmp_path / "sim.json"
    cfg.write_text('{"predictor": {"name": "aiconfigurator"}}')
    args = SimpleNamespace(
        sim_config_path=str(cfg),
        simulator_output_dir=str(tmp_path / "out"),
        simulator_mode="blocking",
        simulator_num_warmup=3,
    )
    native = NativeSimulationConfig.from_server_args(args)
    assert native.sim_config_path == str(cfg)
    assert native.output_dir == str(tmp_path / "out")
    assert native.mode == "blocking"
    assert native.num_warmup == 3


def test_native_config_rejects_bad_mode(tmp_path: Path):
    args = SimpleNamespace(
        sim_config_path="",
        simulator_output_dir=str(tmp_path / "out"),
        simulator_mode="fast",
        simulator_num_warmup=0,
    )
    with pytest.raises(ValueError, match="unsupported simulator mode"):
        NativeSimulationConfig.from_server_args(args)
```

- [x] **Step 5: Run tests**

Run:

```bash
uv run pytest -q test/unit/native/test_native_config.py
```

Expected: pass.

---

### Task 4: Bootstrap Native Launch Instead Of The Standalone FastAPI Server

**Files:**
- Modify: `python/sglsim/launch_server.py`
- Create: `python/sglsim/srt/simulator/native/bootstrap.py`
- Test: `test/integration/test_native_launch_contract.py`

- [x] **Step 1: Add native bootstrap**

Create `python/sglsim/srt/simulator/native/bootstrap.py`:

```python
from __future__ import annotations

from sglsim.srt.simulator.native.config import NativeSimulationConfig


def install_native_simulation(server_args) -> NativeSimulationConfig:
    config = NativeSimulationConfig.from_server_args(server_args)

    from sglsim.srt.simulator.native import cache_controller_patch
    from sglsim.srt.simulator.native import memory_pool_patch
    from sglsim.srt.simulator.native import model_runner_patch
    from sglsim.srt.simulator.native import scheduler_patch

    model_runner_patch.install(config)
    scheduler_patch.install(config)
    cache_controller_patch.install(config)
    memory_pool_patch.install(config)
    return config
```

- [x] **Step 2: Replace `sglsim.launch_server` entrypoint**

Modify `python/sglsim/launch_server.py` so `main()` does this:

```python
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    from sglsim.srt.entrypoints.http_server import launch_server
    from sglsim.srt.server_args import ServerArgs
    from sglsim.srt.simulator.native.bootstrap import install_native_simulation
    from sglsim.srt.utils import kill_process_tree

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    server_args = ServerArgs.from_cli_args(raw_args)

    config_path = getattr(server_args, "sim_config_path", None)
    if config_path:
        os.environ["SGLSIM_SIMULATOR_CONFIG_PATH"] = str(config_path)

    install_native_simulation(server_args)
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
```

- [x] **Step 3: Add missing parser fields to native `ServerArgs`**

After Task 2 sync, update `python/sglsim/srt/server_args.py` to parse these SGLSim-only flags if upstream does not already include them:

```text
--sim-config-path
--simulator-output-dir
--simulator-mode
--simulator-num-warmup
```

The parser must store attributes with these exact names:

```python
sim_config_path: str | None
simulator_output_dir: str
simulator_mode: str
simulator_num_warmup: int
```

- [x] **Step 4: Add launch parser test**

Create `test/integration/test_native_launch_contract.py`:

```python
from pathlib import Path

from sglsim.srt.server_args import ServerArgs


def test_native_server_args_accept_simulator_flags(tmp_path: Path):
    cfg = tmp_path / "sim.json"
    cfg.write_text('{"predictor": {"name": "aiconfigurator"}}')
    parser = ServerArgs.add_cli_args
    assert parser is not None

    import argparse

    ap = argparse.ArgumentParser()
    ServerArgs.add_cli_args(ap)
    raw = ap.parse_args(
        [
            "--model-path",
            "Qwen/Qwen3-8B",
            "--host",
            "127.0.0.1",
            "--port",
            "28000",
            "--dp-size",
            "2",
            "--sim-config-path",
            str(cfg),
            "--simulator-output-dir",
            str(tmp_path / "out"),
            "--simulator-mode",
            "blocking",
            "--simulator-num-warmup",
            "0",
        ]
    )
    args = ServerArgs.from_cli_args(raw)
    assert args.sim_config_path == str(cfg)
    assert args.simulator_mode == "blocking"
```

- [x] **Step 5: Run launch contract test**

Run:

```bash
uv run --extra native pytest -q test/integration/test_native_launch_contract.py
```

Expected: pass.

---

### Task 5: Replace ModelRunner GPU Work With CPU Simulated Outputs

**Files:**
- Create: `python/sglsim/srt/simulator/native/model_runner_patch.py`
- Test: `test/unit/native/test_simulated_model_runner_patch.py`

- [x] **Step 1: Add ModelRunner patch module**

Create `python/sglsim/srt/simulator/native/model_runner_patch.py`:

```python
from __future__ import annotations

import torch

_INSTALLED = False


def install(config) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from sglsim.srt.model_executor import model_runner as model_runner_mod

    ModelRunner = model_runner_mod.ModelRunner

    def initialize(self, *args, **kwargs):
        from sglsim.srt.mem_cache.allocator import (
            PagedTokenToKVPoolAllocator,
            TokenToKVPoolAllocator,
        )
        from sglsim.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

        class MockModel:
            start_layer = 0

            def forward(self):
                return None

        self.model = MockModel()
        self.dtype = getattr(self.model_config, "dtype", torch.float16)
        self.kv_cache_dtype = self.dtype
        self.max_total_num_tokens = int(
            getattr(self.server_args, "max_total_tokens", None) or 262144
        )
        self.page_size = int(getattr(self, "page_size", 1) or 1)
        if self.page_size > 1:
            self.max_total_num_tokens = (
                self.max_total_num_tokens // self.page_size * self.page_size
            )
        context_len = int(getattr(self.model_config, "context_len", 32768) or 32768)
        max_num_reqs = min(max(int(self.max_total_num_tokens / context_len * 512), 2048), 4096)
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=context_len,
            device="cpu",
            enable_memory_saver=False,
        )
        num_layers = max(
            int(getattr(self.model_config, "num_hidden_layers", 1) or 1),
            int(getattr(self.model_config, "num_attention_layers", 1) or 1),
        )
        self.start_layer = 0
        self.end_layer = num_layers
        self.num_effective_layers = num_layers
        self.token_to_kv_pool = MHATokenToKVPool(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=1,
            head_dim=1,
            layer_num=self.num_effective_layers,
            device="cpu",
            enable_memory_saver=False,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            enable_alt_stream=False,
        )
        allocator_cls = TokenToKVPoolAllocator if self.page_size == 1 else PagedTokenToKVPoolAllocator
        self.token_to_kv_pool_allocator = allocator_cls(
            size=self.max_total_num_tokens,
            dtype=self.kv_cache_dtype,
            device="cpu",
            kvcache=self.token_to_kv_pool,
            need_sort=False,
            **({"page_size": self.page_size} if allocator_cls is PagedTokenToKVPoolAllocator else {}),
        )
        self.attn_backend = None
        self.graph_mem_usage = 0
        self.weight_load_mem_usage = 10
        self.max_running_requests = min(self.max_total_num_tokens // 2, self.req_to_token_pool.size)
        self.use_ngram_embedding = False

    def forward(self, batch, *args, **kwargs):
        from sglsim.srt.layers.logits_processor import LogitsProcessorOutput
        from sglsim.srt.model_executor.model_runner import ModelRunnerOutput

        vocab_size = int(getattr(self.model_config, "vocab_size", 32000) or 32000)
        logits = torch.empty((batch.batch_size, vocab_size), dtype=torch.float32, device="cpu")
        return ModelRunnerOutput(
            logits_output=LogitsProcessorOutput(next_token_logits=logits),
            can_run_graph=False,
            expert_distribution_metrics=None,
        )

    def sample(self, logits_output, *args, **kwargs):
        batch_size = int(logits_output.next_token_logits.shape[0])
        return torch.ones((batch_size,), dtype=torch.int64, device="cpu")

    def compute_logprobs_only(self, *args, **kwargs):
        return None

    ModelRunner.initialize = initialize
    ModelRunner.forward = forward
    ModelRunner.sample = sample
    ModelRunner.compute_logprobs_only = compute_logprobs_only
```

- [x] **Step 2: Add patch installation test**

Create `test/unit/native/test_simulated_model_runner_patch.py`:

```python
from types import SimpleNamespace

from sglsim.srt.simulator.native.model_runner_patch import install


def test_model_runner_patch_installs_once():
    install(SimpleNamespace())
    install(SimpleNamespace())
```

- [x] **Step 3: Run test**

Run:

```bash
uv run --extra native pytest -q test/unit/native/test_simulated_model_runner_patch.py
```

Expected: pass, or fail on a concrete missing upstream import that is added by syncing the needed module from `/home/comp/zhanzhao/sglang/python/sglang`.

---

### Task 6: Replace CUDA/Triton KV Allocation Kernels With CPU Functions

**Files:**
- Create: `python/sglsim/srt/simulator/native/cpu_allocator_kernels.py`
- Create: `python/sglsim/srt/simulator/native/memory_pool_patch.py`
- Test: `test/unit/native/test_cpu_allocator_kernels.py`

- [x] **Step 1: Add CPU allocator kernels**

Create `python/sglsim/srt/simulator/native/cpu_allocator_kernels.py` by porting the CPU logic from:

```text
/home/comp/zhanzhao/sglang/tools/sglang-simulator/src/sglang_simulator/simulation/sglang/mem_cache_allocator.py
```

The exported functions must be:

```python
def alloc_extend_cpu(pre_lens_ptr, seq_lens_ptr, last_loc_ptr, free_page_ptr, out_indices, bs_upper, page_size): ...
def alloc_decode_cpu(seq_lens_ptr, last_loc_ptr, free_page_ptr, out_indices, bs_upper, page_size): ...
```

- [x] **Step 2: Add allocator patch installer**

Create `python/sglsim/srt/simulator/native/memory_pool_patch.py`:

```python
from __future__ import annotations

from sglsim.srt.simulator.native.cpu_allocator_kernels import alloc_decode_cpu, alloc_extend_cpu

_INSTALLED = False


class IndexableWrapper:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def install(config) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from sglsim.srt.mem_cache import allocator

    allocator.alloc_extend_kernel = IndexableWrapper(alloc_extend_cpu)
    allocator.alloc_decode_kernel = IndexableWrapper(alloc_decode_cpu)
```

- [x] **Step 3: Add allocator tests**

Create `test/unit/native/test_cpu_allocator_kernels.py`:

```python
import torch

from sglsim.srt.simulator.native.cpu_allocator_kernels import alloc_decode_cpu, alloc_extend_cpu


def test_alloc_decode_reuses_current_page():
    seq_lens = torch.tensor([2], dtype=torch.int64)
    last_loc = torch.tensor([0], dtype=torch.int64)
    free_pages = torch.tensor([], dtype=torch.int64)
    out = torch.empty((1,), dtype=torch.int64)
    alloc_decode_cpu(seq_lens, last_loc, free_pages, out, 1, 16)
    assert out.tolist() == [1]


def test_alloc_decode_uses_new_page_at_boundary():
    seq_lens = torch.tensor([17], dtype=torch.int64)
    last_loc = torch.tensor([15], dtype=torch.int64)
    free_pages = torch.tensor([7], dtype=torch.int64)
    out = torch.empty((1,), dtype=torch.int64)
    alloc_decode_cpu(seq_lens, last_loc, free_pages, out, 1, 16)
    assert out.tolist() == [112]


def test_alloc_extend_spans_new_page():
    pre_lens = torch.tensor([15], dtype=torch.int64)
    seq_lens = torch.tensor([18], dtype=torch.int64)
    last_loc = torch.tensor([14], dtype=torch.int64)
    free_pages = torch.tensor([3], dtype=torch.int64)
    out = torch.empty((3,), dtype=torch.int64)
    alloc_extend_cpu(pre_lens, seq_lens, last_loc, free_pages, out, 1, 16)
    assert out.tolist() == [15, 48, 49]
```

- [x] **Step 4: Run allocator tests**

Run:

```bash
uv run --extra native pytest -q test/unit/native/test_cpu_allocator_kernels.py
```

Expected: pass.

---

### Task 7: Integrate AIConfigurator Latency Into Native Scheduler `run_batch`

**Files:**
- Create: `python/sglsim/srt/simulator/native/scheduler_patch.py`
- Modify: `python/sglsim/srt/simulator/predictor.py`
- Test: `test/unit/native/test_native_scheduler_latency_patch.py`

- [x] **Step 1: Add scheduler patch**

Create `python/sglsim/srt/simulator/native/scheduler_patch.py`:

```python
from __future__ import annotations

import time
from collections import defaultdict

from sglsim.srt.simulator.metrics import RequestStats, dump_simulator_outputs
from sglsim.srt.simulator.predictor import build_simulator_predictor
from sglsim.srt.simulator.state import SimulatorState

_INSTALLED = False


class NativeSchedulerState:
    def __init__(self, config, predictor):
        self.config = config
        self.predictor = predictor
        self.clock = SimulatorState()
        self.request_stats = defaultdict(RequestStats)
        self.iteration_stats: list[dict] = []
        self.last_cpu_ts = 0.0


def _prediction_requests(batch) -> list[dict[str, int]]:
    requests: list[dict[str, int]] = []
    if batch.forward_mode.is_extend():
        for req in batch.reqs:
            requests.append(
                {
                    "extend_length": int(req.extend_input_len),
                    "past_kv_length": int(len(req.prefix_indices) + len(req.output_ids)),
                }
            )
    elif batch.forward_mode.is_decode():
        for req in batch.reqs:
            requests.append(
                {
                    "extend_length": 1,
                    "past_kv_length": int(len(req.prefix_indices) + len(req.output_ids)),
                }
            )
    return requests


def install(config) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from sglsim.srt.managers import scheduler as scheduler_mod

    Scheduler = scheduler_mod.Scheduler
    original_init = Scheduler.__init__
    original_recv_requests = Scheduler.recv_requests
    original_get_new_batch_prefill = Scheduler.get_new_batch_prefill
    original_run_batch = Scheduler.run_batch
    original_process_batch_result = Scheduler.process_batch_result
    original_profile = Scheduler.profile

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        predictor = build_simulator_predictor(self.server_args, self.model_config)
        self.sglsim_native = NativeSchedulerState(config, predictor)

    def recv_requests(self, *args, **kwargs):
        reqs = original_recv_requests(self, *args, **kwargs)
        now = time.time()
        for req in reqs:
            if req.__class__.__name__ not in {"TokenizedGenerateReqInput", "BatchTokenizedGenerateReqInput"}:
                continue
            stats = self.sglsim_native.request_stats[req.rid]
            stats.rid = req.rid
            stats.input_length = len(req.input_ids)
            stats.output_length = req.sampling_params.max_new_tokens
            sim = (req.sampling_params.custom_params or {}).get("simulation", {})
            stats.created_time = float(sim.get("server_created_time", now))
            stats.last_event_time = stats.created_time
            stats.queue_start = now
        if reqs and self.sglsim_native.last_cpu_ts == 0:
            self.sglsim_native.last_cpu_ts = time.time()
        return reqs

    def get_new_batch_prefill(self, *args, **kwargs):
        batch = original_get_new_batch_prefill(self, *args, **kwargs)
        if batch is not None:
            for req in batch.reqs:
                stats = self.sglsim_native.request_stats[req.rid]
                stats.final_reused_tokens = getattr(req, "cached_tokens", 0)
                if stats.queue_end == -1:
                    stats.queue_end = time.time()
        return batch

    def run_batch(self, batch, *args, **kwargs):
        result = original_run_batch(self, batch, *args, **kwargs)
        requests = _prediction_requests(batch)
        if requests:
            latency = float(self.sglsim_native.predictor.predict_requests(requests))
            time.sleep(max(0.0, latency))
            self.sglsim_native.clock.set_current_inference_dur(latency)
            self.sglsim_native.iteration_stats.append(
                {
                    "requests": requests,
                    "forward_latency": latency,
                    "l2_load_latency": self.sglsim_native.clock.pop_hicache_l2_load_dur(),
                    "l2_backup_latency": self.sglsim_native.clock.pop_hicache_l2_backup_dur(),
                }
            )
        return result

    def process_batch_result(self, batch, *args, **kwargs):
        result = original_process_batch_result(self, batch, *args, **kwargs)
        now = time.time()
        if batch is not None:
            for req in getattr(batch, "reqs", []):
                stats = self.sglsim_native.request_stats[req.rid]
                if stats.rid:
                    stats.gen_token_latencies.append(now - stats.last_event_time)
                    stats.last_event_time = now
        return result

    def profile(self, req, *args, **kwargs):
        stats = [item for item in self.sglsim_native.request_stats.values() if item.rid]
        dump_simulator_outputs(
            self.sglsim_native.config.output_dir,
            sorted(stats, key=lambda item: item.created_time),
            self.sglsim_native.iteration_stats,
            num_warmup=self.sglsim_native.config.num_warmup,
        )
        return original_profile(self, req, *args, **kwargs)

    Scheduler.__init__ = __init__
    Scheduler.recv_requests = recv_requests
    Scheduler.get_new_batch_prefill = get_new_batch_prefill
    Scheduler.run_batch = run_batch
    Scheduler.process_batch_result = process_batch_result
    Scheduler.profile = profile
```

- [x] **Step 2: Add scheduler patch smoke test**

Create `test/unit/native/test_native_scheduler_latency_patch.py`:

```python
from types import SimpleNamespace

from sglsim.srt.simulator.native.scheduler_patch import _prediction_requests


class Mode:
    def __init__(self, name):
        self.name = name

    def is_extend(self):
        return self.name == "extend"

    def is_decode(self):
        return self.name == "decode"


def test_prediction_requests_for_extend():
    req = SimpleNamespace(extend_input_len=32, prefix_indices=[1, 2], output_ids=[3])
    batch = SimpleNamespace(forward_mode=Mode("extend"), reqs=[req])
    assert _prediction_requests(batch) == [{"extend_length": 32, "past_kv_length": 3}]


def test_prediction_requests_for_decode():
    req = SimpleNamespace(extend_input_len=32, prefix_indices=[1, 2], output_ids=[3, 4])
    batch = SimpleNamespace(forward_mode=Mode("decode"), reqs=[req])
    assert _prediction_requests(batch) == [{"extend_length": 1, "past_kv_length": 4}]
```

- [x] **Step 3: Run test**

Run:

```bash
uv run --extra native pytest -q test/unit/native/test_native_scheduler_latency_patch.py
```

Expected: pass.

---

### Task 8: Port Native Cache Controller, HiCache, And Host Transfer Latency

**Files:**
- Create: `python/sglsim/srt/simulator/native/cache_controller_patch.py`
- Modify: `python/sglsim/srt/mem_cache/memory_pool_host.py`
- Test: `test/unit/native/test_native_cache_controller_patch.py`

- [x] **Step 1: Add cache controller patch**

Create `python/sglsim/srt/simulator/native/cache_controller_patch.py` using the same public behavior as:

```text
/home/comp/zhanzhao/sglang/tools/sglang-simulator/src/sglang_simulator/simulation/sglang/cache_controller.py
/home/comp/zhanzhao/sglang/tools/sglang-simulator/src/sglang_simulator/simulation/sglang/hiradix_cache.py
/home/comp/zhanzhao/sglang/tools/sglang-simulator/src/sglang_simulator/simulation/sglang/mem_pool_host.py
```

The module must export:

```python
def install(config) -> None: ...
def estimate_host_transfer_seconds(num_tokens: int, bytes_per_token: int, bandwidth_bytes: float) -> float: ...
def estimate_storage_transfer_seconds(num_tokens: int, bytes_per_token: int, bandwidth_bytes: float) -> float: ...
```

- [x] **Step 2: Patch HiCacheController methods**

Inside `install(config)`, patch upstream-compatible `HiCacheController`:

```python
HiCacheController.backup_thread_func = lambda self, *args, **kwargs: None
HiCacheController.prefetch_thread_func = lambda self, *args, **kwargs: None
HiCacheController.handle_backup_operation = handle_backup_operation
HiCacheController.handle_prefetch_operation = handle_prefetch_operation
```

`handle_backup_operation` must drain `self.backup_queue`, call `self._page_backup(operation)` unless `self.backup_skip` is true, and push the operation into `self.ack_backup_queue`.

`handle_prefetch_operation` must use the current inference duration budget to compute how many storage tokens are prefetched, set `operation.completed_tokens`, terminate finished operations, and update request stats.

- [x] **Step 3: Patch HiRadixCache event checks**

Patch `HiRadixCache.check_hicache_events`:

```python
original = HiRadixCache.check_hicache_events

def check_hicache_events(self, *args, **kwargs):
    self.cache_controller.handle_backup_operation()
    self.cache_controller.handle_prefetch_operation()
    return original(self, *args, **kwargs)
```

- [x] **Step 4: Patch host memory pool transfer**

Patch `MHATokenToKVPoolHost.backup_from_device_all_layer` and `load_to_device_per_layer` to avoid real tensor transfer and record simulated duration through the native scheduler state.

- [x] **Step 5: Add transfer estimator tests**

Create `test/unit/native/test_native_cache_controller_patch.py`:

```python
from sglsim.srt.simulator.native.cache_controller_patch import (
    estimate_host_transfer_seconds,
    estimate_storage_transfer_seconds,
)


def test_host_transfer_seconds_is_size_over_bandwidth():
    assert estimate_host_transfer_seconds(100, 10, 1000) == 1.0


def test_storage_transfer_seconds_zero_for_no_tokens():
    assert estimate_storage_transfer_seconds(0, 10, 1000) == 0.0
```

- [x] **Step 6: Run tests**

Run:

```bash
uv run --extra native pytest -q test/unit/native/test_native_cache_controller_patch.py
```

Expected: pass.

---

### Task 9: Restore Native Failure And Failover Control Path

**Files:**
- Modify: `python/sglsim/srt/entrypoints/http_server.py`
- Modify: `python/sglsim/srt/managers/tokenizer_manager.py`
- Modify: `python/sglsim/srt/managers/data_parallel_controller.py`
- Modify: `python/sglsim/srt/managers/scheduler.py`
- Test: `test/integration/test_native_failure_failover.py`

- [x] **Step 1: Verify upstream failure endpoints exist**

Check that synced `python/sglsim/srt/entrypoints/http_server.py` contains:

```python
@app.post("/simulate_gpu_failure")
@app.post("/simulate_gpu_recovery")
```

If missing after sync, copy the endpoint bodies from:

```text
/home/comp/zhanzhao/sglang/python/sglang/srt/entrypoints/http_server.py
```

- [x] **Step 2: Verify TokenizerManager failure forwarding**

Ensure `python/sglsim/srt/managers/tokenizer_manager.py` contains methods with these exact names:

```python
async def simulate_gpu_failure(self, obj): ...
async def simulate_gpu_recovery(self, obj): ...
```

They must send `SimulateGpuFailureReqInput` and `SimulateGpuRecoveryReqInput` into scheduler/DPC using the same queue used by normal SGLang control messages.

- [x] **Step 3: Verify DPC failure routing**

Ensure `python/sglsim/srt/managers/data_parallel_controller.py` routes failure and recovery requests to the target DP rank worker and records failed ranks.

- [x] **Step 4: Verify Scheduler snapshot path**

Ensure `python/sglsim/srt/managers/scheduler.py` contains:

```python
def handle_simulate_gpu_failure(self, recv_req): ...
def handle_simulate_gpu_recovery(self, recv_req): ...
def _collect_failover_reqs(self): ...
def _clear_failed_rank_scheduler_state(self): ...
def _snapshot_req(self, req): ...
```

The implementation must cold-clear failed-rank scheduler state, tree cache, request pool, and KV allocator.

- [x] **Step 5: Add integration test**

Create `test/integration/test_native_failure_failover.py`:

```python
import csv
import json
import subprocess
import time
from pathlib import Path

import requests


def test_native_failure_api_writes_failover_events(tmp_path: Path):
    out = tmp_path / "native-failure"
    cmd = [
        "uv",
        "run",
        "--extra",
        "native",
        "python",
        "-m",
        "sglsim.launch_server",
        "--model-path",
        "simulator-model",
        "--host",
        "127.0.0.1",
        "--port",
        "28111",
        "--dp-size",
        "2",
        "--tp-size",
        "1",
        "--pp-size",
        "1",
        "--simulator-output-dir",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(100):
            try:
                if requests.get("http://127.0.0.1:28111/health", timeout=0.2).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            raise AssertionError("server did not become healthy")

        r = requests.post("http://127.0.0.1:28111/simulate_gpu_failure", json={"dp_rank": 0}, timeout=5)
        assert r.status_code == 200
        r = requests.post("http://127.0.0.1:28111/simulate_gpu_recovery", json={"dp_rank": 0}, timeout=5)
        assert r.status_code == 200

        events = out / "failover_events.jsonl"
        assert events.is_file()
        names = [json.loads(line)["event"] for line in events.read_text().splitlines() if line.strip()]
        assert "fault_injected" in names or "gpu_failure_simulated" in names
        assert "gpu_recovered" in names
    finally:
        proc.terminate()
        proc.wait(timeout=10)
```

- [x] **Step 6: Run integration test**

Run:

```bash
uv run --extra native pytest -q test/integration/test_native_failure_failover.py
```

Expected: pass.

---

### Task 10: Restore Native Remote Backup Server And Metadata Semantics

**Files:**
- Modify: `python/sglsim/srt/mem_cache/storage/backend_factory.py`
- Copy/modify: `python/sglsim/srt/mem_cache/storage/remote_backup/`
- Modify: `benchmark/online/src/server/deploy_server.py`
- Test: `test/integration/test_native_remote_backup_cache_hits.py`

- [x] **Step 1: Sync remote backup storage files**

Copy with import rewrite:

```text
/home/comp/zhanzhao/sglang/python/sglang/srt/mem_cache/storage/remote_backup/__init__.py
/home/comp/zhanzhao/sglang/python/sglang/srt/mem_cache/storage/remote_backup/launch_remote_backup_server.py
/home/comp/zhanzhao/sglang/python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_server.py
/home/comp/zhanzhao/sglang/python/sglang/srt/mem_cache/storage/remote_backup/remote_backup_storage.py
```

- [x] **Step 2: Preserve request-aware lease fields**

Verify the synced remote backup code supports:

```text
start_request
finish_request
insert_pages / put pages with token context
match_prefix
lease finish reasons: normal, abort, failover_old, failover_supersede
remote_backup_request_aware
```

- [x] **Step 3: Replace tensor payloads with metadata pages**

Where remote backup expects real KV tensor pages, use small CPU tensors or page-id placeholders while keeping:

```text
dp_rank
rid
generation
page_start
token_ids
hash values
page count
```

- [x] **Step 4: Update online deploy wrapper**

Modify `benchmark/online/src/server/deploy_server.py` so `remote_backup-fixed-a100` starts the native remote backup server when:

```json
"kv_backup": "remote_backup"
```

The command must use:

```bash
uv run --extra native python -m sglsim.srt.mem_cache.storage.remote_backup.launch_remote_backup_server
```

- [x] **Step 5: Add remote backup integration test**

Create `test/integration/test_native_remote_backup_cache_hits.py`:

```python
import csv
import subprocess
from pathlib import Path


def test_native_remote_backup_produces_storage_cache_hits(tmp_path: Path):
    out = tmp_path / "online"
    cmd = [
        "uv",
        "run",
        "--extra",
        "native",
        "python",
        "benchmark/online/src/main.py",
        "--config",
        "remote_backup-fixed-a100",
        "--dp-size",
        "2",
        "--pp-size",
        "1",
        "--tp-size",
        "1",
        "--fixed-input-len",
        "128",
        "--fixed-output-len",
        "32",
        "--fixed-num-requests",
        "8",
        "--fixed-app-workers",
        "2",
        "--inject-after-job",
        "1",
        "--recover-after-job",
        "6",
        "--fault-dp-rank",
        "0",
        "--inject-delay",
        "0",
        "--output-dir",
        str(out),
    ]
    subprocess.run(cmd, check=True, timeout=180)
    cache = out / "remote_backup-fixed-a100" / "cache_hits.csv"
    rows = list(csv.DictReader(cache.open()))
    assert rows
    storage_rows = [
        r
        for r in rows
        if int(r.get("reused_storage") or r.get("remote_match") or 0) > 0
    ]
    assert storage_rows
```

- [x] **Step 6: Run test**

Run:

```bash
uv run --extra native pytest -q test/integration/test_native_remote_backup_cache_hits.py
```

Expected: pass.

---

### Task 11: Make Online Benchmark Use Native Runtime By Default

**Files:**
- Modify: `benchmark/online/src/server/deploy_server.py`
- Modify: `benchmark/online/src/server/config.py`
- Modify: `benchmark/online/src/config/remote_backup-fixed-a100/server.json`
- Modify: `benchmark/online/src/config/baseline-fixed-a100/server.json`
- Test: `test/integration/test_online_fixed_native_sglang_parity.py`

- [x] **Step 1: Add `runtime_mode` to online server config**

Modify `benchmark/online/src/server/config.py` so server JSON supports:

```json
"runtime_mode": "native"
```

Valid values:

```text
native
legacy
```

Default must be `native`.

- [x] **Step 2: Update deploy command**

Modify `benchmark/online/src/server/deploy_server.py` so native mode launches:

```bash
uv run --extra native --extra aiconfigurator python -m sglsim.launch_server
```

Legacy mode can keep the old command only for debugging.

- [x] **Step 3: Update fixed A100 configs**

Add to `benchmark/online/src/config/remote_backup-fixed-a100/server.json`:

```json
"runtime_mode": "native"
```

Add the same key to `benchmark/online/src/config/baseline-fixed-a100/server.json`.

- [x] **Step 4: Add native fixed smoke test**

Create `test/integration/test_online_fixed_native_sglang_parity.py`:

```python
import csv
import subprocess
from pathlib import Path


def test_online_fixed_uses_native_runtime_and_writes_outputs(tmp_path: Path):
    out = tmp_path / "native-fixed"
    cmd = [
        "uv",
        "run",
        "--extra",
        "native",
        "--extra",
        "aiconfigurator",
        "python",
        "benchmark/online/src/main.py",
        "--config",
        "remote_backup-fixed-a100",
        "--dp-size",
        "2",
        "--pp-size",
        "1",
        "--tp-size",
        "1",
        "--fixed-input-len",
        "64",
        "--fixed-output-len",
        "8",
        "--fixed-num-requests",
        "4",
        "--fixed-app-workers",
        "2",
        "--inject-after-job",
        "1",
        "--recover-after-job",
        "3",
        "--fault-dp-rank",
        "0",
        "--inject-delay",
        "0",
        "--output-dir",
        str(out),
    ]
    subprocess.run(cmd, check=True, timeout=180)
    run_dir = out / "remote_backup-fixed-a100"
    assert (run_dir / "task_summary.csv").is_file()
    assert (run_dir / "cache_hits.csv").is_file()
    rows = list(csv.DictReader((run_dir / "task_summary.csv").open()))
    assert len(rows) == 4
```

- [x] **Step 5: Run test**

Run:

```bash
uv run --extra native --extra aiconfigurator pytest -q test/integration/test_online_fixed_native_sglang_parity.py
```

Expected: pass.

---

### Task 12: Add Guard Tests Against Current Simplified Runtime

**Files:**
- Create: `test/parity/test_no_authoritative_runtime_service.py`
- Modify: `python/sglsim/srt/managers/runtime_service.py`
- Test: `test/parity/test_no_authoritative_runtime_service.py`

- [x] **Step 1: Add guard test**

Create `test/parity/test_no_authoritative_runtime_service.py`:

```python
from pathlib import Path


def test_online_deploy_defaults_to_native_runtime():
    config_files = [
        Path("benchmark/online/src/config/remote_backup-fixed-a100/server.json"),
        Path("benchmark/online/src/config/baseline-fixed-a100/server.json"),
    ]
    for path in config_files:
        text = path.read_text()
        assert '"runtime_mode": "native"' in text


def test_runtime_service_not_imported_by_native_launch():
    launch = Path("python/sglsim/launch_server.py").read_text()
    assert "RuntimeService" not in launch
    assert "sglsim.srt.http.server" not in launch
```

- [x] **Step 2: Mark RuntimeService as legacy**

At the top of `python/sglsim/srt/managers/runtime_service.py`, add:

```python
"""Legacy compatibility runtime.

This module is not part of the authoritative native SGLang parity path.
Online benchmark configs must use runtime_mode=native.
"""
```

- [x] **Step 3: Run guard test**

Run:

```bash
uv run pytest -q test/parity/test_no_authoritative_runtime_service.py
```

Expected: pass.

---

### Task 13: Port Upstream Parity Tests And Activate Them

**Files:**
- Modify/Create: `test/upstream_multi_agent/`
- Create: `docs/sglang-native-parity-status.md`

- [x] **Step 1: Copy upstream tests with package rename**

Copy these files from `/home/comp/zhanzhao/sglang/benchmark/multi_agent/tests/` to `test/upstream_multi_agent/` and rewrite imports from benchmark/multi_agent to benchmark/online where needed:

```text
test_fault_injection_recovery.py
test_remote_backup_cache_details.py
test_remote_backup_prefetch.py
test_chunked_prefill_backup_policy.py
test_remote_backup_cross_rank.py
test_failover_routing.py
test_remote_backup_server_logging.py
test_process_log_contract.py
```

- [x] **Step 2: Create status document**

Create `docs/sglang-native-parity-status.md`:

```markdown
# SGLang Native Parity Status

Active parity tests:

- `test/upstream_multi_agent/test_fault_injection_recovery.py`
- `test/upstream_multi_agent/test_remote_backup_cache_details.py`
- `test/upstream_multi_agent/test_remote_backup_prefetch.py`
- `test/upstream_multi_agent/test_chunked_prefill_backup_policy.py`
- `test/upstream_multi_agent/test_remote_backup_cross_rank.py`
- `test/upstream_multi_agent/test_failover_routing.py`
- `test/upstream_multi_agent/test_remote_backup_server_logging.py`
- `test/upstream_multi_agent/test_process_log_contract.py`

Runtime mode:

- authoritative online runtime: native SGLang parity
- legacy RuntimeService: compatibility only
```

- [x] **Step 3: Run upstream parity tests**

Run:

```bash
uv run --extra native pytest -q test/upstream_multi_agent
```

Expected: active tests pass. Any unsupported test must fail and drive a concrete implementation task; do not leave broad xfail markers.

---

### Task 14: Validate 32K Context A100/Qwen3 Experiment End-To-End

**Files:**
- No code file changes in the passing case.
- Outputs under: `benchmark/online/results/native_rb_ctx32k_w16_f1/`

- [ ] **Step 1: Run 32K remote backup experiment**

Run:

```bash
SGLSIM_RUN_ID=native_rb_ctx32k_w16_f1 uv run --extra native --extra aiconfigurator python benchmark/online/src/main.py \
  --config remote_backup-fixed-a100 \
  --dp-size 8 \
  --pp-size 1 \
  --tp-size 1 \
  --fixed-input-len 32000 \
  --fixed-output-len 1000 \
  --fixed-num-requests 200 \
  --fixed-app-workers 16 \
  --inject-after-job 40 \
  --recover-after-job 160 \
  --fault-dp-rank 0 \
  --inject-delay 0 \
  --output-dir benchmark/online/results/native_rb_ctx32k_w16_f1
```

- [ ] **Step 2: Verify predictor is enabled**

Run:

```bash
grep -R "SGLSIM_PREDICTOR enabled=aiconfigurator" benchmark/online/results/native_rb_ctx32k_w16_f1/remote_backup-fixed-a100/logs
```

Expected: at least one match.

- [ ] **Step 3: Verify failure events**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("benchmark/online/results/native_rb_ctx32k_w16_f1/remote_backup-fixed-a100/logs/failover_events.jsonl")
events = [json.loads(x)["event"] for x in p.read_text().splitlines() if x.strip()]
print(events)
assert any(x in events for x in ["fault_injected", "gpu_failure_simulated"])
assert "gpu_recovered" in events
assert any(x in events for x in ["failover_dispatched", "runtime_failover_reaction"])
PY
```

Expected: assertions pass.

- [ ] **Step 4: Verify cache hits are runtime-derived and nonzero**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path("benchmark/online/results/native_rb_ctx32k_w16_f1/remote_backup-fixed-a100/cache_hits.csv")
rows = list(csv.DictReader(p.open()))
keys = ["remote_match", "remote_prefetch", "reused_storage", "l1_match", "l2_match", "reused_host", "reused_device"]
nonzero = [r for r in rows if any(int(r.get(k) or 0) > 0 for k in keys)]
failover = [r for r in rows if str(r.get("is_failover_retried", "")).lower() == "true"]
print("rows", len(rows), "nonzero_cache", len(nonzero), "failover", len(failover))
assert rows
assert nonzero
assert failover
PY
```

Expected: assertions pass.

- [ ] **Step 5: Verify 32K latency is plausible**

Run:

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path("benchmark/online/results/native_rb_ctx32k_w16_f1/remote_backup-fixed-a100/task_summary.csv")
rows = list(csv.DictReader(p.open()))
lat = [float(r["task_completion_time_s"]) for r in rows]
print("min", min(lat), "p50", sorted(lat)[len(lat)//2], "max", max(lat))
assert max(lat) > 20
PY
```

Expected: max completion time exceeds 20 seconds for 32K + 1000 output with A100/Qwen3 predictor.

---

### Task 15: Remove Simplified Cache/Failover Shortcuts From Authoritative Path

**Files:**
- Modify: `python/sglsim/srt/managers/runtime_service.py`
- Modify: `python/sglsim/srt/http/server.py`
- Modify: docs that mention legacy runtime
- Test: full targeted suite

- [x] **Step 1: Ensure no online config uses legacy runtime**

Run:

```bash
grep -R '"runtime_mode": "legacy"' benchmark/online/src/config
```

Expected: no output.

- [x] **Step 2: Ensure fixed recovery ratios are not used by native path**

Run:

```bash
grep -R "_recovered_tokens\\|ratio=0.8\\|ratio=0.55" python/sglsim benchmark/online/src test
```

Expected: matches may appear only in `runtime_service.py` legacy tests. No native test or native launch file may import `RuntimeService`.

- [x] **Step 3: Run targeted suite**

Run:

```bash
uv run --extra native --extra aiconfigurator pytest -q \
  test/parity \
  test/unit/native \
  test/integration/test_native_http_completion.py \
  test/integration/test_native_failure_failover.py \
  test/integration/test_native_remote_backup_cache_hits.py \
  test/integration/test_online_fixed_native_sglang_parity.py \
  test/upstream_multi_agent
```

Expected: pass.

---

## Milestone Acceptance Criteria

The plan is complete when all statements are true:

```text
1. `python -m sglsim.launch_server` imports and launches `sglsim.srt.entrypoints.http_server.launch_server`.
2. `benchmark/online` default runtime mode is `native`.
3. Native online tests fail if `RuntimeService.submit` is called.
4. A tiny fixed workload completes through native TokenizerManager/DataParallelController/Scheduler.
5. `/simulate_gpu_failure` reaches the native Scheduler and snapshots live requests.
6. Remote backup cache hits appear in `cache_hits.csv` from runtime metadata.
7. 32K + 1000 output latency uses A100/Qwen3 AIConfigurator predictor and is not sub-second.
8. Upstream multi-agent parity tests selected in Task 13 pass with imports rewritten to `sglsim` / `benchmark/online`.
9. `docs/sglang-native-parity-status.md` lists no failing active parity tests.
```

## Suggested Commit Sequence

Use these commit boundaries:

```bash
git add docs/sglang-native-parity-contract.md tools/check_sglang_native_parity.py test/parity/test_native_file_manifest.py
git commit -m "test: add native sglang parity manifest"

git add tools/sync_sglang_native_runtime.py python/sglsim/srt
git commit -m "chore: vendor native sglang runtime modules"

git add pyproject.toml python/sglsim/srt/simulator/native test/unit/native
git commit -m "feat: add native simulator patch layer"

git add python/sglsim/launch_server.py test/integration/test_native_launch_contract.py
git commit -m "feat: launch native sglang-compatible runtime"

git add benchmark/online/src benchmark/online/src/config test/integration/test_online_fixed_native_sglang_parity.py
git commit -m "feat: run online fixed benchmark through native runtime"

git add test/upstream_multi_agent docs/sglang-native-parity-status.md
git commit -m "test: activate upstream online parity tests"
```
