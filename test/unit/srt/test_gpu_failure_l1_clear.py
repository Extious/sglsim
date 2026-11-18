from types import SimpleNamespace

from sglsim.srt.managers.aegis_l1_l2 import AegisL1L2Config
from sglsim.srt.managers.scheduler import Scheduler
from sglsim.srt.mem_cache.hiradix_cache import HiRadixCache
from sglsim.srt.simulation.types import RequestStats


def test_gpu_failure_clears_device_pages_and_preserves_host_prefix() -> None:
    cache = HiRadixCache(
        page_size=2,
        l1_capacity_pages=8,
        l2_capacity_pages=16,
    )
    device_page = cache.mem_pool_device.alloc(1, owner="req-0")[0]
    inserted = cache.tree.insert([10, 11], [device_page.page_id] * 2)
    host_page = cache.mem_pool_host.alloc(1, owner="req-0")[0]
    cache.tree.set_host_value(inserted.last_node, [host_page.page_id] * 2)

    before = cache.match_prefix([10, 11])
    assert before.device_length == 2
    assert cache.mem_pool_device.used_pages == 1
    assert cache.mem_pool_host.used_pages == 1

    result = cache.clear_l1_for_gpu_failure()
    after = cache.match_prefix([10, 11])

    assert result["cleared_device_tokens"] == 2
    assert result["cleared_device_pages"] == 1
    assert result["preserved_host_pages"] == 1
    assert after.device_length == 0
    assert after.host_length == 2
    assert cache.mem_pool_device.used_pages == 0
    assert cache.mem_pool_host.used_pages == 1


def test_restore_materialization_is_capped_by_current_l1_capacity() -> None:
    cache = HiRadixCache(
        page_size=1,
        l1_capacity_pages=4,
        l2_capacity_pages=8,
    )
    protected_pages = cache.mem_pool_device.alloc(3, owner="active")
    protected = cache.tree.insert(
        [1, 2, 3],
        [page.page_id for page in protected_pages],
    )
    cache.tree.inc_lock_ref(protected.last_node)

    restored, _last_node_id = cache.materialize_device_prefix(
        "retry",
        [10, 11, 12, 13],
        4,
        last_node_id=None,
    )

    assert restored == 1
    assert cache.match_prefix([10, 11, 12, 13]).device_length == 1
    assert cache.mem_pool_device.used_pages == 4


def test_scheduler_converts_unmaterialized_backup_tokens_to_recompute() -> None:
    class CapacityLimitedCache:
        @staticmethod
        def materialize_restored_prefix(req, restored_tokens: int) -> int:
            assert restored_tokens == 200
            req.cache_hit_tokens = 80
            return 80

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.cache = CapacityLimitedCache()
    scheduler.aegis_backup_manager = SimpleNamespace(
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=2048,
            recompute_setup_ms=0.0,
            recompute_ms_per_token=0.22,
        )
    )
    req = SimpleNamespace(cache_hit_tokens=0)
    stats = RequestStats(
        rid="retry",
        aegis_failover_replay=True,
        aegis_replay_restore_action="l2_backup",
        aegis_replay_restored_tokens=200,
        aegis_replay_recompute_tokens=50,
        aegis_host_backup_lookup_tokens=200,
        aegis_replay_restore_timing_source="aegiskv_l1_l2_hint",
    )

    scheduler._materialize_aegis_replay_restore(req, stats)

    assert stats.aegis_replay_restored_tokens == 80
    assert stats.aegis_replay_recompute_tokens == 170
    assert stats.aegis_replay_capacity_limited_tokens == 120
    assert stats.aegis_replay_restored_prefill_tokens == 80
    assert stats.aegis_replay_restore_overhead_s > 0
    assert stats.aegis_replay_restore_timing_source.endswith("+capacity_cap")
