import pytest

from sglsim.srt.mem_cache.tier import CacheTier, TieredKVCache


def test_cache_tier_order_is_explicit():
    assert [tier.value for tier in CacheTier] == ["l1", "l2", "l3"]


def test_tiered_kv_cache_tracks_page_residency():
    cache = TieredKVCache(capacities={CacheTier.L1: 2})

    cache.put(page_id=7, tier=CacheTier.L1)

    assert cache.tier_of(7) == CacheTier.L1


def test_tiered_kv_cache_put_rejects_full_tier():
    cache = TieredKVCache(capacities={CacheTier.L1: 1})
    cache.put(page_id=7, tier=CacheTier.L1)

    with pytest.raises(RuntimeError):
        cache.put(page_id=8, tier=CacheTier.L1)


def test_tiered_kv_cache_move_updates_residency_and_rejects_full_target_tier():
    cache = TieredKVCache(capacities={CacheTier.L1: 1, CacheTier.L2: 2})
    cache.put(page_id=7, tier=CacheTier.L1)
    cache.put(page_id=8, tier=CacheTier.L2)

    with pytest.raises(RuntimeError):
        cache.move(page_id=8, target_tier=CacheTier.L1)

    assert cache.tier_of(8) == CacheTier.L2

    cache.move(page_id=7, target_tier=CacheTier.L2)

    assert cache.tier_of(7) == CacheTier.L2
