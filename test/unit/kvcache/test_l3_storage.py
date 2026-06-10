from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.hicache_storage import MockHiCacheStorage
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.simulation.manager.state import StateManager


def test_mock_storage_hit_query():
    storage = MockHiCacheStorage(capacity_pages=32, page_size=16)
    tokens = list(range(32))
    storage.batch_set(tokens, [0, 1])
    assert storage.storage_hit_query(tokens) == 32
    assert storage.storage_hit_query([99, 100]) == 0


def test_l3_prefetch_skipped_without_storage_hit():
    controller = KVCachePool(
        CacheConfig(
            page_size=16,
            l1_capacity_pages=32,
            l2_capacity_pages=64,
            l3_capacity_pages=64,
            hicache_storage_backend="mock",
            prefetch_threshold=16,
        )
    )
    req = Req("r0", list(range(32)), SamplingParams(max_new_tokens=1))
    status = controller.prefetch_from_storage(req)
    assert status == "skipped"


def test_l3_prefetch_dispatched_when_storage_has_prefix():
    StateManager.reset()
    controller = KVCachePool(
        CacheConfig(
            page_size=16,
            l1_capacity_pages=32,
            l2_capacity_pages=64,
            l3_capacity_pages=64,
            hicache_storage_backend="mock",
            prefetch_threshold=16,
        )
    )
    assert controller._cache.storage is not None
    tokens = list(range(32))
    controller._cache.storage.batch_set(tokens, [0, 1])
    req = Req("r0", tokens, SamplingParams(max_new_tokens=1))
    status = controller.prefetch_from_storage(req)
    assert status == "dispatched"
    result = controller.check_hicache_events(remain_dur=1.0)
    assert result.get("l3_prefetch_tokens", 0) >= 0
