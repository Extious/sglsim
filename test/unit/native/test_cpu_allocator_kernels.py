from __future__ import annotations

import torch

from sglsim.srt.simulator.native.cpu_allocator_kernels import (
    alloc_decode_cpu,
    alloc_extend_cpu,
)
from sglsim.srt.simulator.native.memory_pool_patch import install


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


def test_memory_pool_patch_installs_indexable_kernel_wrappers():
    install(object())

    from sglsim.srt.mem_cache import allocator

    out = torch.empty((1,), dtype=torch.int64)
    allocator.alloc_decode_kernel[(1,)](
        torch.tensor([17], dtype=torch.int64),
        torch.tensor([15], dtype=torch.int64),
        torch.tensor([7], dtype=torch.int64),
        out,
        1,
        16,
    )
    assert out.tolist() == [112]
