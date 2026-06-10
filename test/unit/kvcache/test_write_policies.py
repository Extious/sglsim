from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.simulation.manager.state import StateManager


def _extend(controller: KVCachePool, req: Req) -> None:
    controller.prepare_prefill(req)
    controller.extend_prefill(req, len(req.input_ids))
    controller.finalize_prefill_in_tree(req)
    controller.check_hicache_events()


def test_write_through_selective_defers_backup_until_second_hit():
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
            hicache_write_policy="write_through_selective",
            write_through_threshold=2,
        )
    )
    req1 = Req("r0", [1, 2], SamplingParams(max_new_tokens=1))
    _extend(controller, req1)
    assert controller.hicache.pending_backup_dur == 0

    req2 = Req("r1", [1, 2, 3], SamplingParams(max_new_tokens=1))
    _extend(controller, req2)
    assert controller.hicache.pending_backup_dur > 0


def test_write_through_backups_on_every_extend():
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
            hicache_write_policy="write_through",
            write_through_threshold=1,
        )
    )
    req = Req("r0", [1, 2], SamplingParams(max_new_tokens=1))
    _extend(controller, req)
    assert controller.metrics["l2_write_pages"] == 2


def test_write_back_only_backups_on_evict():
    StateManager.reset()
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=1,
            l2_capacity_pages=8,
            hicache_write_policy="write_back",
        )
    )
    req0 = Req("r0", [1], SamplingParams(max_new_tokens=0))
    _extend(controller, req0)
    assert controller.hicache.pending_backup_dur == 0
    controller.finish_request(req0)

    req1 = Req("r1", [2], SamplingParams(max_new_tokens=0))
    _extend(controller, req1)
    assert controller.metrics["l2_write_pages"] >= 1
