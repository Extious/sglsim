"""Tests for suffix-only finalize_prefill_in_tree."""

from __future__ import annotations

import time

from sglsim.srt.config import CacheConfig
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.managers.schedule_batch import Req, SamplingParams


def _make_pool() -> KVCachePool:
    return KVCachePool(CacheConfig(page_size=1, l1_capacity_pages=500_000))


def test_finalize_reuses_locked_prefix_node() -> None:
    pool = _make_pool()
    shared_prefix = list(range(20_000))
    req1 = Req(
        rid="r1",
        input_ids=shared_prefix + list(range(20_000, 25_377)),
        sampling_params=SamplingParams(max_new_tokens=64),
    )
    req2 = Req(
        rid="r2",
        input_ids=shared_prefix + list(range(30_000, 35_377)),
        sampling_params=SamplingParams(max_new_tokens=64),
    )

    pool.prepare_prefill(req1)
    pool.extend_prefill(req1, req1.prefill_length)
    pool.finalize_prefill_in_tree(req1)
    assert pool.radix.match_prefix(req1.input_ids).matched_length >= len(shared_prefix)

    pool.prepare_prefill(req2)
    assert req2.cached_length >= len(shared_prefix)
    pool.extend_prefill(req2, req2.prefill_length - req2.kv_committed_length)

    start = time.perf_counter()
    pool.finalize_prefill_in_tree(req2)
    finalize_ms = (time.perf_counter() - start) * 1e3

    assert req2.prefill_alloc_page_ids == []
    assert finalize_ms < 800.0
