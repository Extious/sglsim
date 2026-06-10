"""Regression: shared-prefix extend_prefill must not double-free KV pages."""

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool


def _full_prefill(controller: KVCachePool, rid: str, token_ids: list[int]) -> None:
    req = Req(rid, token_ids, SamplingParams(max_new_tokens=8))
    controller.prepare_prefill(req)
    controller.extend_prefill(req, len(token_ids))
    controller.finish_request(req)


def test_shared_prefix_many_requests_page_size_16():
    page_size = 16
    shared = list(range(2048))
    controller = KVCachePool(
        CacheConfig(
            page_size=page_size,
            l1_capacity_pages=4096,
            l2_capacity_pages=0,
        )
    )
    for i in range(32):
        suffix_start = 3000 + i * 500
        token_ids = shared + list(range(suffix_start, suffix_start + 1800))
        _full_prefill(controller, str(i + 1), token_ids)
        controller.check_hicache_events()
