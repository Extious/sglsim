# Native Simulator Phase 3 Model Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ModelRunner` support simulator mode without loading real model weights.

**Architecture:** Put simulator-specific setup and mock forward helpers in `sglang.srt.simulator.model_runner`; `ModelRunner` only contains narrow guard clauses.

**Tech Stack:** Python, torch, SGLang `ModelRunner`, pytest with lightweight fakes.

---

## Dependencies

Complete Phase 1 and Phase 2 first.

## Files

- Create: `python/sglang/srt/simulator/model_runner.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py`
- Test: `test/registered/unit/simulator/test_model_runner_simulator.py`

## Task 1: Mock Forward Helpers

- [ ] **Step 1: Write failing helper tests**

Create `test/registered/unit/simulator/test_model_runner_simulator.py`:

```python
import types
import unittest

import torch

from sglang.srt.simulator.model_runner import (
    SimulatorMockModel,
    build_mock_logits_output,
    build_mock_next_token_ids,
)


class TestSimulatorModelRunnerHelpers(unittest.TestCase):
    def test_mock_model_has_forward(self):
        self.assertIsNone(SimulatorMockModel().forward())

    def test_build_mock_logits_output_uses_batch_size_and_vocab(self):
        runner = types.SimpleNamespace(
            device="cpu",
            model_config=types.SimpleNamespace(vocab_size=32000),
        )
        forward_batch = types.SimpleNamespace(batch_size=3)

        output = build_mock_logits_output(runner, forward_batch)

        self.assertEqual(tuple(output.next_token_logits.shape), (3, 32000))
        self.assertEqual(output.next_token_logits.device.type, "cpu")

    def test_build_mock_next_token_ids(self):
        runner = types.SimpleNamespace(device="cpu")
        forward_batch = types.SimpleNamespace(batch_size=4)

        ids = build_mock_next_token_ids(runner, forward_batch)

        self.assertTrue(torch.equal(ids, torch.ones(4, dtype=torch.int64)))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_model_runner_simulator.py -q
```

Expected: FAIL because `sglang.srt.simulator.model_runner` does not exist.

- [ ] **Step 3: Implement helper module**

Create `python/sglang/srt/simulator/model_runner.py`:

```python
from __future__ import annotations

import logging

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.model_runner import ModelRunnerOutput

logger = logging.getLogger(__name__)


class SimulatorMockModel:
    start_layer = 0

    def forward(self):
        return None


def build_mock_logits_output(model_runner, forward_batch) -> LogitsProcessorOutput:
    batch_size = int(getattr(forward_batch, "batch_size", 0))
    vocab_size = int(model_runner.model_config.vocab_size)
    return LogitsProcessorOutput(
        next_token_logits=torch.empty(
            size=(batch_size, vocab_size),
            device=model_runner.device,
        )
    )


def build_mock_model_runner_output(model_runner, forward_batch) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        logits_output=build_mock_logits_output(model_runner, forward_batch),
        can_run_graph=False,
        expert_distribution_metrics=None,
    )


def build_mock_next_token_ids(model_runner, forward_batch) -> torch.Tensor:
    return torch.ones(
        size=(int(getattr(forward_batch, "batch_size", 0)),),
        device=model_runner.device,
        dtype=torch.int64,
    )
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_model_runner_simulator.py -q
```

Expected: PASS.

## Task 2: Simulator Initialization

- [ ] **Step 1: Extend helper module with initialization function**

Add to `python/sglang/srt/simulator/model_runner.py`:

```python
def initialize_simulator_model_runner(model_runner, pre_model_load_memory: float) -> None:
    model_runner.model = SimulatorMockModel()
    model_runner.dtype = model_runner.model_config.dtype
    model_runner.kv_cache_dtype = model_runner.dtype

    if model_runner.server_args.max_total_tokens is not None:
        model_runner.max_total_num_tokens = model_runner.server_args.max_total_tokens
    else:
        model_runner.max_total_num_tokens = int(
            getattr(model_runner.server_args, "max_total_tokens", None) or 1 << 20
        )

    if hasattr(model_runner, "page_size") and model_runner.page_size > 1:
        model_runner.max_total_num_tokens = (
            model_runner.max_total_num_tokens
            // model_runner.page_size
            * model_runner.page_size
        )

    max_num_reqs = min(
        max(
            int(model_runner.max_total_num_tokens / model_runner.model_config.context_len * 512),
            2048,
        ),
        4096,
    )

    from sglang.srt.mem_cache.allocator import (
        PagedTokenToKVPoolAllocator,
        TokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

    model_runner.req_to_token_pool = ReqToTokenPool(
        size=max_num_reqs,
        max_context_len=model_runner.model_config.context_len,
        device=model_runner.device,
        enable_memory_saver=False,
    )
    model_runner.token_to_kv_pool = MHATokenToKVPool(
        model_runner.max_total_num_tokens,
        page_size=model_runner.page_size,
        dtype=model_runner.kv_cache_dtype,
        head_num=1,
        head_dim=1,
        layer_num=max(
            model_runner.model_config.num_hidden_layers,
            model_runner.model_config.num_attention_layers,
        ),
        device=model_runner.device,
        enable_memory_saver=model_runner.server_args.enable_memory_saver,
        start_layer=0,
        end_layer=max(
            model_runner.model_config.num_hidden_layers,
            model_runner.model_config.num_attention_layers,
        ),
        enable_alt_stream=False,
    )

    if model_runner.page_size == 1:
        model_runner.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=model_runner.max_total_num_tokens,
            dtype=model_runner.kv_cache_dtype,
            device=model_runner.device,
            kvcache=model_runner.token_to_kv_pool,
            need_sort=False,
        )
    else:
        model_runner.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            size=model_runner.max_total_num_tokens,
            page_size=model_runner.page_size,
            dtype=model_runner.kv_cache_dtype,
            device=model_runner.device,
            kvcache=model_runner.token_to_kv_pool,
            need_sort=False,
        )

    model_runner.attn_backend = None
    model_runner.graph_mem_usage = 0
    model_runner.weight_load_mem_usage = 0
    model_runner.max_running_requests = min(
        model_runner.server_args.max_running_requests
        or model_runner.max_total_num_tokens // 2,
        model_runner.req_to_token_pool.size,
    )
    model_runner.use_ngram_embedding = False
    logger.info(
        "Simulator ModelRunner initialized with %s tokens and %s max requests.",
        model_runner.max_total_num_tokens,
        model_runner.max_running_requests,
    )
```

- [ ] **Step 2: Patch `ModelRunner.initialize`**

In `python/sglang/srt/model_executor/model_runner.py`, at the start of `initialize()` after `server_args = self.server_args`, add:

```python
        if self.server_args.simulator:
            from sglang.srt.simulator.model_runner import (
                initialize_simulator_model_runner,
            )

            initialize_simulator_model_runner(self, pre_model_load_memory)
            return
```

- [ ] **Step 3: Patch `ModelRunner.forward`**

At the start of `ModelRunner.forward()`, before incrementing `forward_pass_id`, add:

```python
        if self.server_args.simulator:
            from sglang.srt.simulator.model_runner import (
                build_mock_model_runner_output,
            )

            return build_mock_model_runner_output(self, forward_batch)
```

- [ ] **Step 4: Patch `ModelRunner.sample`**

At the start of `ModelRunner.sample()`, add:

```python
        if self.server_args.simulator:
            from sglang.srt.simulator.model_runner import build_mock_next_token_ids

            return build_mock_next_token_ids(self, forward_batch)
```

- [ ] **Step 5: Patch `ModelRunner.compute_logprobs_only`**

At the start of `compute_logprobs_only()`, add:

```python
        if self.server_args.simulator:
            return
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
PYTHONPATH=python pytest test/registered/unit/simulator/test_model_runner_simulator.py -q
PYTHONPATH=python python -m py_compile python/sglang/srt/model_executor/model_runner.py python/sglang/srt/simulator/model_runner.py
```

Expected: PASS and compile succeeds.

- [ ] **Step 7: Commit phase 3**

Run:

```bash
cd /home/comp/zhanzhao/sglsim
git add python/sglang/srt/model_executor/model_runner.py python/sglang/srt/simulator/model_runner.py test/registered/unit/simulator/test_model_runner_simulator.py
git commit -m "feat: add simulator model runner path"
```
