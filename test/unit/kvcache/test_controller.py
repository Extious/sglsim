import pytest

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool


def _hicache_config(**kwargs) -> CacheConfig:
    defaults = {"hicache_write_policy": "write_through", "write_through_threshold": 1}
    defaults.update(kwargs)
    return CacheConfig(**defaults)


def _flush_cache(controller: KVCachePool) -> None:
    controller.check_hicache_events()


def _extend_full_prefill(controller: KVCachePool, req: Req) -> None:
    controller.prepare_prefill(req)
    controller.extend_prefill(req, len(req.input_ids))
    controller.finalize_prefill_in_tree(req)


def test_prepare_prefill_sets_cached_length_from_radix_match():
    controller = KVCachePool(
        _hicache_config(
            page_size=1,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
        )
    )
    controller.radix.insert([1, 2, 3], page_ids=[10, 11, 12])
    req = Req("r0", [1, 2, 4], SamplingParams(max_new_tokens=1))

    controller.prepare_prefill(req)

    assert req.cached_length == 2
    assert controller.cache_prefix_tokens == 2


def test_prepare_prefill_preserves_recovered_cached_length():
    controller = KVCachePool(
        _hicache_config(
            page_size=1,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
        )
    )
    controller.radix.insert([1, 2], page_ids=[10, 11])
    req = Req("r0", [1, 2, 3, 4], SamplingParams(max_new_tokens=1), cached_length=3)

    controller.prepare_prefill(req)

    assert req.cached_length == 3
    assert controller.cache_prefix_tokens == 0


def test_extend_prefill_allocates_l1_pages_and_inserts_radix():
    controller = KVCachePool(
        _hicache_config(
            page_size=1,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
        )
    )
    req = Req("r0", [1, 2, 3], SamplingParams(max_new_tokens=1))

    _extend_full_prefill(controller, req)
    _flush_cache(controller)

    assert controller.allocator.num_free_pages == 5
    match = controller.radix.match_prefix([1, 2, 3])
    assert match.page_ids == [0, 1, 2]
    assert controller.metrics["l2_write_pages"] > 0


def test_write_through_queues_l2_backup():
    controller = KVCachePool(
        _hicache_config(
            page_size=1,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
        )
    )
    req = Req("r0", [1, 2], SamplingParams(max_new_tokens=1))
    _extend_full_prefill(controller, req)
    assert controller.hicache.write_queue
    _flush_cache(controller)
    assert controller.metrics["l2_write_pages"] == 2
    nodes = controller.radix.collect_nodes_on_path([1, 2])
    assert all(node.backuped for node in nodes)


def test_prefetch_and_l3_write_ignore_empty_matches():
    controller = KVCachePool(
        CacheConfig(
            page_size=2,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
            l3_capacity_pages=8,
        )
    )
    req = Req("r0", [9, 8, 7], SamplingParams(max_new_tokens=1))

    controller.prefetch(req)
    controller.write_to_l3(req)
    _flush_cache(controller)

    assert controller.metrics["prefetch_pages"] == 0
    assert controller.metrics["l3_write_pages"] == 0


def test_finish_request_keeps_radix_pages_and_releases_lock():
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=2,
            l2_capacity_pages=0,
        )
    )
    req = Req("r0", [1, 2], SamplingParams(max_new_tokens=0))
    _extend_full_prefill(controller, req)
    node = controller.radix.get_node(req.last_node_id)
    assert node is not None
    assert node.lock_ref >= 1

    controller.finish_request(req)

    assert node.lock_ref == 0
    assert controller.allocator.num_free_pages == 0
    assert controller.radix.match_prefix([1, 2]).matched_length == 2


def test_l3_write_records_backup_pages():
    controller = KVCachePool(
        _hicache_config(
            page_size=1,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
            l3_capacity_pages=8,
            hicache_storage_backend="mock",
        )
    )
    req = Req("r0", [1, 2], SamplingParams(max_new_tokens=1))
    _extend_full_prefill(controller, req)
    controller.write_to_l3(req)
    _flush_cache(controller)
    assert controller.metrics["l3_write_pages"] == 2


def test_eviction_frees_l1_when_locked_refs_released():
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=1,
            l2_capacity_pages=0,
        )
    )
    req0 = Req("r0", [1], SamplingParams(max_new_tokens=0))
    req1 = Req("r1", [2], SamplingParams(max_new_tokens=0))
    _extend_full_prefill(controller, req0)
    controller.finish_request(req0)
    _extend_full_prefill(controller, req1)
    assert controller.radix.match_prefix([2]).matched_length == 1


def test_extend_prefill_is_idempotent_when_kv_committed():
    controller = KVCachePool(
        _hicache_config(
            page_size=2,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
        )
    )
    req = Req("r0", [1, 2, 3], SamplingParams(max_new_tokens=1))
    _extend_full_prefill(controller, req)
    free_pages = controller.allocator.num_free_pages

    controller.extend_prefill(req, 0)

    assert controller.allocator.num_free_pages == free_pages


def test_check_hicache_events_fast_path_when_idle():
    controller = KVCachePool(
        _hicache_config(
            page_size=1,
            l1_capacity_pages=8,
            l2_capacity_pages=8,
        )
    )

    result = controller.check_hicache_events()

    assert result == {
        "l2_write_pages": 0,
        "prefetch_pages": 0,
        "l3_write_pages": 0,
        "l3_prefetch_tokens": 0,
        "load_completed_tokens": 0,
    }
    assert controller.metrics["l2_write_pages"] == 0
    assert controller.metrics["prefetch_pages"] == 0
