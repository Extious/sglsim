# Native Simulator Phase 5 HiCache Allocator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HiCache and allocator monkey patches with native simulator branches.

**Architecture:** Add simulator helpers for storage and CPU allocation fallbacks. Existing HiCache and allocator classes call those helpers only when `get_global_server_args().simulator` is true.

**Tech Stack:** Python, torch, numpy, SGLang memory cache classes, pytest.

---

## Dependencies

Complete Phases 1 through 4 first.

## Files

- Create: `python/sglang/srt/simulator/hicache.py`
- Create: `python/sglang/srt/simulator/allocator.py`
- Modify: `python/sglang/srt/mem_cache/storage/backend_factory.py`
- Modify: `python/sglang/srt/managers/cache_controller.py`
- Modify: `python/sglang/srt/mem_cache/hiradix_cache.py`
- Modify: `python/sglang/srt/mem_cache/memory_pool_host.py`
- Modify: `python/sglang/srt/mem_cache/allocator.py`
- Test: `test/registered/unit/simulator/test_hicache_allocator.py`

## Task 1: Simulator Storage Backend

- [ ] **Step 1: Write storage tests**

Create `test/registered/unit/simulator/test_hicache_allocator.py`:

```python
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.simulator.allocator import alloc_decode_cpu, alloc_extend_cpu
from sglang.srt.simulator.hicache import SimulatorHiCacheStorage


class TestSimulatorHiCacheStorage(unittest.TestCase):
    def test_storage_set_exists_clear(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = str(Path(tmp_dir) / "keys.txt")
            storage = SimulatorHiCacheStorage(storage_file_path=path)

            self.assertFalse(storage.exists("a"))
            self.assertTrue(storage.set("a"))
            self.assertTrue(storage.exists("a"))
            self.assertEqual(storage.batch_exists(["a", "b"], extra_info=None), 1)
            self.assertTrue(storage.clear())
            self.assertFalse(storage.exists("a"))


class TestSimulatorAllocator(unittest.TestCase):
    def test_alloc_decode_cpu_reuses_current_page(self):
        out = torch.empty((1,), dtype=torch.int64)
        alloc_decode_cpu(
            torch.tensor([2]),
            torch.tensor([0]),
            torch.tensor([], dtype=torch.int64),
            out,
            1,
            4,
        )
        self.assertEqual(out.item(), 1)

    def test_alloc_extend_cpu_allocates_new_page(self):
        out = torch.empty((4,), dtype=torch.int64)
        alloc_extend_cpu(
            torch.tensor([0]),
            torch.tensor([4]),
            torch.tensor([-1]),
            torch.tensor([3]),
            out,
            1,
            4,
        )
        self.assertEqual(out.tolist(), [12, 13, 14, 15])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_hicache_allocator.py -q
```

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement storage helper**

Create `python/sglang/srt/simulator/hicache.py`:

```python
from __future__ import annotations

import os
from typing import Any, List, Optional


class SimulatorHiCacheStorage:
    def __init__(self, storage_file_path: str = "/tmp/sglang_simulator/hicache_storage_keys.txt"):
        self.storage: set[str] = set()
        self.storage_file_path = storage_file_path
        os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)
        if os.path.exists(self.storage_file_path):
            with open(self.storage_file_path) as f:
                for line in f:
                    self.storage.add(line.strip())

    def register_mem_pool_host(self, mem_pool_host):
        return None

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if key in self.storage:
            return True
        self.storage.add(key)
        with open(self.storage_file_path, "a") as f:
            f.write(key + "\n")
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        extra_info=None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key in keys:
            if not self.set(key):
                return False
        return True

    def exists(self, key: str) -> bool:
        return key in self.storage

    def batch_exists(self, keys: List[str], extra_info=None) -> int:
        for i, key in enumerate(keys):
            if key not in self.storage:
                return i
        return len(keys)

    def clear(self) -> bool:
        self.storage.clear()
        with open(self.storage_file_path, "w"):
            pass
        return True
```

- [ ] **Step 4: Implement allocator helper**

Create `python/sglang/srt/simulator/allocator.py` with the native CPU paged-allocation helpers:

```python
from __future__ import annotations

import torch


class IndexableWrapper:
    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, _):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def ceil_div(a, b):
    return (a + b - 1) // b


def alloc_extend_cpu(
    pre_lens_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
):
    pre_lens = pre_lens_ptr.cpu().tolist()
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()
    batch_size = len(pre_lens)
    output_offset = 0
    free_page_offset = 0
    for pid in range(batch_size):
        pre_len = pre_lens[pid]
        seq_len = seq_lens[pid]
        extend_len = seq_len - pre_len
        num_new_pages = ceil_div(seq_len, page_size) - ceil_div(pre_len, page_size)
        if extend_len <= 0:
            free_page_offset += num_new_pages
            continue
        current_page_end = ceil_div(pre_len, page_size) * page_size
        part1_end = min(seq_len, current_page_end)
        for i in range(max(part1_end - pre_len, 0)):
            out_indices[output_offset + i] = last_loc[pid] + 1 + i
        output_offset += max(part1_end - pre_len, 0)
        if output_offset >= extend_len:
            free_page_offset += num_new_pages
            continue
        full_pages_start = current_page_end
        full_pages_end = (seq_len // page_size) * page_size
        for i in range(max(full_pages_end - full_pages_start, 0)):
            page_id = free_pages[free_page_offset + (i // page_size)]
            out_indices[output_offset + i] = page_id * page_size + (i % page_size)
        output_offset += max(full_pages_end - full_pages_start, 0)
        tail = seq_len - full_pages_end
        if tail > 0:
            page_id = free_pages[free_page_offset + num_new_pages - 1]
            for i in range(tail):
                out_indices[output_offset + i] = page_id * page_size + i
            output_offset += tail
        free_page_offset += num_new_pages


def alloc_decode_cpu(
    seq_lens_ptr: torch.Tensor,
    last_loc_ptr: torch.Tensor,
    free_page_ptr: torch.Tensor,
    out_indices: torch.Tensor,
    bs_upper: int,
    page_size: int,
):
    seq_lens = seq_lens_ptr.cpu().tolist()
    last_loc = last_loc_ptr.cpu().tolist()
    free_pages = free_page_ptr.cpu().tolist()
    free_page_offset = 0
    for pid, seq_len in enumerate(seq_lens):
        pre_len = seq_len - 1
        num_new_pages = ceil_div(seq_len, page_size) - ceil_div(pre_len, page_size)
        if num_new_pages == 0:
            out_indices[pid] = last_loc[pid] + 1
        else:
            out_indices[pid] = free_pages[free_page_offset] * page_size
        free_page_offset += num_new_pages
```

- [ ] **Step 5: Run helper tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_hicache_allocator.py -q
```

Expected: PASS.

## Task 2: Wire Native Branches

- [ ] **Step 1: Patch storage backend factory**

In `python/sglang/srt/mem_cache/storage/backend_factory.py`, before normal backend creation, add:

```python
        from sglang.srt.server_args import get_global_server_args

        if get_global_server_args().simulator:
            from sglang.srt.simulator.hicache import SimulatorHiCacheStorage

            return SimulatorHiCacheStorage()
```

- [ ] **Step 2: Patch paged allocator kernel selection**

In `python/sglang/srt/mem_cache/allocator.py`, inside `PagedTokenToKVPoolAllocator.__init__`, add before kernel use:

```python
        from sglang.srt.server_args import get_global_server_args

        if get_global_server_args().simulator:
            from sglang.srt import mem_cache
            from sglang.srt.simulator.allocator import (
                IndexableWrapper,
                alloc_decode_cpu,
                alloc_extend_cpu,
            )

            allocator_module = __import__(
                "sglang.srt.mem_cache.allocator", fromlist=["alloc_extend_kernel"]
            )
            allocator_module.alloc_extend_kernel = IndexableWrapper(alloc_extend_cpu)
            allocator_module.alloc_decode_kernel = IndexableWrapper(alloc_decode_cpu)
```

- [ ] **Step 3: Patch HostKVCache pin memory**

In `python/sglang/srt/mem_cache/memory_pool_host.py`, in `HostKVCache.__init__`, force `pin_memory=False` when simulator is enabled:

```python
        from sglang.srt.server_args import get_global_server_args

        if get_global_server_args().simulator:
            pin_memory = False
```

- [ ] **Step 4: Patch host transfer methods**

In `MHATokenToKVPoolHost.load_to_device_per_layer` and `backup_from_device_all_layer`, add simulator branches that compute transfer duration and call scheduler runtime state. Use helper functions in `simulator/hicache.py`:

```python
def estimate_segment_transfer_seconds(host_indices, device_indices, bytes_per_token, bandwidth_bytes):
    ...
```

Then call:

```python
runtime = getattr(get_global_server_args(), "simulator_runtime", None)
```

If direct global runtime access is not practical, store the active runtime in `sglang.srt.simulator.state` as module-level `set_active_state()` / `get_active_state()` during Scheduler runtime initialization.

- [ ] **Step 5: Patch cache controller threads**

In `python/sglang/srt/managers/cache_controller.py`, when simulator enabled:

```python
        if self.server_args.simulator:
            return
```

in background thread target startup paths, and expose explicit methods:

```python
    def handle_simulator_backup_operation(self):
        ...

    def handle_simulator_prefetch_operation(self):
        ...
```

Implement the simulator backup/prefetch branches directly in `cache_controller.py`.

- [ ] **Step 6: Patch HiRadixCache event check**

In `python/sglang/srt/mem_cache/hiradix_cache.py::check_hicache_events`, add:

```python
        if get_global_server_args().simulator:
            self.cache_controller.handle_simulator_backup_operation()
            self.cache_controller.handle_simulator_prefetch_operation()
```

- [ ] **Step 7: Compile modified files**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python python -m py_compile \
  python/sglang/srt/simulator/hicache.py \
  python/sglang/srt/simulator/allocator.py \
  python/sglang/srt/mem_cache/storage/backend_factory.py \
  python/sglang/srt/managers/cache_controller.py \
  python/sglang/srt/mem_cache/hiradix_cache.py \
  python/sglang/srt/mem_cache/memory_pool_host.py \
  python/sglang/srt/mem_cache/allocator.py
```

Expected: compile succeeds.

- [ ] **Step 8: Run helper tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_hicache_allocator.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit phase 5**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
git add python/sglang/srt/simulator/hicache.py python/sglang/srt/simulator/allocator.py python/sglang/srt/mem_cache/storage/backend_factory.py python/sglang/srt/managers/cache_controller.py python/sglang/srt/mem_cache/hiradix_cache.py python/sglang/srt/mem_cache/memory_pool_host.py python/sglang/srt/mem_cache/allocator.py test/registered/unit/simulator/test_hicache_allocator.py
git commit -m "feat: add simulator hicache allocator paths"
```
