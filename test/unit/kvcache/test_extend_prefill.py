from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool


def test_extend_prefill_grows_kv_incrementally():
    controller = KVCachePool(
        CacheConfig(page_size=4, l1_capacity_pages=32, l2_capacity_pages=32)
    )
    req = Req("r0", list(range(12)), SamplingParams(max_new_tokens=1))
    controller.prepare_prefill(req)
    controller.extend_prefill(req, 4)
    assert req.kv_committed_length == 4
    controller.extend_prefill(req, 4)
    assert req.kv_committed_length == 8
    controller.extend_prefill(req, 4)
    assert req.kv_committed_length == 12
    controller.finalize_prefill_in_tree(req)
    assert controller.radix.match_prefix(req.input_ids).matched_length == 12


def test_chunked_extend_without_finish_bulk():
    controller = KVCachePool(
        CacheConfig(page_size=2, l1_capacity_pages=16, l2_capacity_pages=16)
    )
    req = Req("r0", [1, 2, 3, 4, 5, 6], SamplingParams(max_new_tokens=1))
    controller.prepare_prefill(req)
    controller.extend_prefill(req, 2)
    controller.extend_prefill(req, 2)
    controller.extend_prefill(req, 2)
    assert req.kv_committed_length == 6
