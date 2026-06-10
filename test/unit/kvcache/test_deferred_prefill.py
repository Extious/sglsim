"""Tests for deferred prefill KV allocation (O(1) per chunk, single tree finalize)."""

from __future__ import annotations

import time

from sglsim.srt.config import CacheConfig
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.managers.schedule_batch import Req, SamplingParams


def _make_pool() -> KVCachePool:
    return KVCachePool(CacheConfig(page_size=1, l1_capacity_pages=500_000))


def _make_req(n_input: int = 25_377) -> Req:
    return Req(
        rid="r1",
        input_ids=list(range(n_input)),
        sampling_params=SamplingParams(max_new_tokens=1024),
    )


def test_chunked_prefill_defers_tree_until_finalize() -> None:
    pool = _make_pool()
    req = _make_req()
    pool.prepare_prefill(req)

    for _ in range(3):
        pool.extend_prefill(req, 8192)
    assert req.kv_committed_length == 24_576
    assert len(req.prefill_alloc_page_ids) == 24_576
    assert req.last_node_id is not None

    pool.finalize_prefill_in_tree(req)
    assert req.prefill_alloc_page_ids == []
    assert req.last_node_id is not None


def test_chunked_prefill_wall_time_drops_vs_full_tree_per_chunk() -> None:
    pool = _make_pool()
    req = _make_req()
    pool.prepare_prefill(req)

    start = time.perf_counter()
    for _ in range(3):
        pool.extend_prefill(req, 8192)
    pool.extend_prefill(req, req.prefill_length - req.kv_committed_length)
    chunk_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    pool.finalize_prefill_in_tree(req)
    finalize_ms = (time.perf_counter() - start) * 1e3

    # Chunk extends avoid per-chunk radix walks; finalize is one O(n) insert.
    assert chunk_ms < 200.0
    assert finalize_ms < 2500.0


def test_hicache_chunked_prefill_syncs_tree_per_finalize() -> None:
    pool = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=500_000,
            l2_capacity_pages=256_000,
        )
    )
    req = _make_req()
    pool.prepare_prefill(req)

    for _ in range(3):
        pool.extend_prefill(req, 8192)
    assert req.kv_committed_length == 24_576
    assert len(req.prefill_alloc_page_ids) == 24_576

    pool.finalize_prefill_in_tree(req)
    assert req.prefill_alloc_page_ids == []
    assert req.cached_length >= req.kv_committed_length
    assert pool.hicache.pending_write_pages >= 0
