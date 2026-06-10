"""Regression: decode KV allocation must be O(1) and skip radix rematch."""

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool


def test_extend_decode_allocates_without_radix_growth():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=100_000, l2_capacity_pages=0)
    )
    prefill_len = 5000
    req = Req("r0", list(range(prefill_len)), SamplingParams(max_new_tokens=64))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, prefill_len)
    pool.finalize_prefill_in_tree(req)

    nodes_before = len(pool.radix._nodes)
    for _ in range(32):
        pool.extend_decode(req, 1)

    assert len(pool.radix._nodes) == nodes_before
    assert len(req.decode_pages) == 32
    assert req.kv_committed_length == prefill_len + 32

    used_before_finish, _, _, _ = pool.kv_token_usage_details()
    pool.finish_request(req)
    used_after_finish, _, _, _ = pool.kv_token_usage_details()
    assert used_after_finish < used_before_finish
    assert req.decode_pages == []


def test_extend_decode_page_boundary_with_page_size_gt_one():
    page_size = 16
    pool = KVCachePool(
        CacheConfig(page_size=page_size, l1_capacity_pages=10_000, l2_capacity_pages=0)
    )
    prefill_len = 32
    req = Req("r0", list(range(prefill_len)), SamplingParams(max_new_tokens=32))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, prefill_len)

    assert len(req.decode_pages) == 0
    pool.extend_decode(req, 1)
    assert req.kv_committed_length == prefill_len + 1
    assert len(req.decode_pages) == 1

    for _ in range(page_size - 1):
        pool.extend_decode(req, 1)
    assert len(req.decode_pages) == 1

    pool.extend_decode(req, 1)
    assert len(req.decode_pages) == 2
