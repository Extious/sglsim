"""Regression tests for HiCache load-back evictable_size_ accounting."""

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.simulation.manager.state import StateManager


def _measured_evictable_tokens(controller: KVCachePool) -> int:
    return controller.radix.measured_evictable_tokens()


def _seed_host_backed_prefix(
    controller: KVCachePool,
    tokens: list[int],
    req_id: str,
) -> None:
    req = Req(req_id, tokens, SamplingParams(max_new_tokens=0))
    controller.prepare_prefill(req)
    controller.extend_prefill(req, len(tokens))
    controller.cache_unfinished_req(req)
    controller.check_hicache_events()
    controller.finish_request(req)


def _evict_all_device(controller: KVCachePool, tokens: list[int]) -> None:
    cache = controller._cache
    # Drain all evictable tokens so eviction cascades leaf->parent and reaches
    # interior (shared-prefix) nodes too, not just the immediate leaves.
    total = cache.tree.evictable_size()
    pool_pages = cache.mem_pool_device._allocator._pages_by_id
    write_through = cache.cache_controller.write_policy in (
        "write_through",
        "write_through_selective",
    )
    freed = cache.tree.evict(total, pool_pages, write_through=write_through)
    if freed:
        cache.mem_pool_device.free(freed)
    for node in cache.tree.collect_nodes_on_path(tokens):
        assert node.value is None
        assert node.host_value is not None


def _hicache_pool(*, l1_pages: int = 32, l2_pages: int = 64) -> KVCachePool:
    StateManager.reset()
    return KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=l1_pages,
            l2_capacity_pages=l2_pages,
            load_back_threshold=10,
            hicache_write_policy="write_through",
            write_through_threshold=1,
        )
    )


def test_load_back_restores_non_negative_evictable():
    controller = _hicache_pool()
    tokens = list(range(16))
    _seed_host_backed_prefix(controller, tokens, "seed")
    _evict_all_device(controller, tokens)

    req = Req("r1", tokens, SamplingParams(max_new_tokens=1))
    assert controller.init_load_back(req) == 16
    assert controller.load_back(req) == 16
    controller.check_hicache_events()

    assert controller.radix.evictable_size() >= 0
    assert controller.radix.protected_size_ >= 0
    controller.radix.assert_evictable_invariant(strict=True)


def test_second_request_hits_device_after_load_back_no_leak():
    """After load_back restores device residency (sglang-aligned, synchronous),
    a second request on the same prefix is a device hit (host_hit==0), so the
    old duplicate-overwrite leak path is unreachable; no device page leaks.
    """
    controller = _hicache_pool(l1_pages=48)
    tokens = list(range(16))
    _seed_host_backed_prefix(controller, tokens, "seed")
    _evict_all_device(controller, tokens)

    avail_before = controller.available_device_tokens()
    req1 = Req("r1", tokens, SamplingParams(max_new_tokens=1))
    assert controller.init_load_back(req1) == 16
    assert controller.load_back(req1) == 16
    # Prefix is now device-resident: second request needs no load-back.
    req2 = Req("r2", tokens, SamplingParams(max_new_tokens=1))
    assert controller.init_load_back(req2) == 0
    assert controller.load_back(req2) == 0

    controller.check_hicache_events()

    assert controller.radix.evictable_size() >= 0
    assert controller.radix.protected_size_ >= 0
    last_node = controller.radix.collect_nodes_on_path(tokens)[-1]
    assert last_node.lock_ref == 0
    controller.radix.assert_evictable_invariant(strict=True)
    # No device-page leak: exactly the 16 loaded-back tokens are now resident.
    assert controller.available_device_tokens() == avail_before - 16


def test_evict_load_back_cycle_keeps_available_and_evictable_sane():
    controller = _hicache_pool(l1_pages=24, l2_pages=128)
    tokens = list(range(16))

    for idx in range(3):
        _seed_host_backed_prefix(controller, tokens, f"seed-{idx}")
        _evict_all_device(controller, tokens)
        req = Req(f"r-{idx}", tokens, SamplingParams(max_new_tokens=1))
        assert controller.init_load_back(req) == 16
        assert controller.load_back(req) == 16
        controller.check_hicache_events()
        assert controller.radix.evictable_size() >= 0
        assert controller.available_and_evictable_tokens() >= 0


def test_multiturn_prefix_reuse_load_back_decode_finalize_accounting():
    """Mirror the real benchmark loop: shared prefix across turns, evict to host,
    load-back, decode, finalize. evictable_size_ must stay counter==measured
    throughout (no drift -> no phantom usage -> no premature retract)."""
    controller = _hicache_pool(l1_pages=96, l2_pages=256)
    shared = list(range(32))

    def _run_turn(req_id: str, turn_tokens: list[int]) -> None:
        req = Req(req_id, turn_tokens, SamplingParams(max_new_tokens=4))
        controller.init_load_back(req)
        if req.host_hit_length > 0:
            controller.load_back(req)
        controller.prepare_prefill(req)
        extend = len(turn_tokens) - req.cached_length
        if extend > 0:
            controller.extend_prefill(req, extend)
        controller.cache_unfinished_req(req)
        controller.check_hicache_events()
        controller.radix.assert_evictable_invariant(strict=True, context=f"{req_id}-prefill")
        for _ in range(4):
            controller.extend_decode(req, 1)
            req.append_output_token(0)
        controller.finish_request(req)
        controller.check_hicache_events()
        controller.radix.assert_evictable_invariant(strict=True, context=f"{req_id}-finish")

    # Turn 0 establishes the shared prefix.
    turn0 = shared + [100, 101, 102, 103]
    _run_turn("c0-t0", turn0)
    # Evict turn 0's full sequence to host; eviction cascades leaf->parent so the
    # shared prefix also goes to host and the next turn must load it back.
    _evict_all_device(controller, turn0)
    # Turns 1-2 reuse the shared prefix (with distinct suffixes) and load it back.
    turn1 = shared + [200, 201, 202, 203, 204, 205]
    _run_turn("c0-t1", turn1)
    _evict_all_device(controller, turn1)
    _run_turn("c0-t2", shared + [210, 211, 212, 213, 214, 215, 216, 217])

    assert controller.radix.evictable_size() >= 0
    controller.radix.assert_evictable_invariant(strict=True, context="final")


def test_inc_lock_ref_on_evicted_node_does_not_drop_evictable():
    controller = _hicache_pool()
    tokens = list(range(16))
    _seed_host_backed_prefix(controller, tokens, "seed")
    _evict_all_device(controller, tokens)
    last_node = controller.radix.collect_nodes_on_path(tokens)[-1]
    assert last_node.value is None
    before = controller.radix.evictable_size_
    controller.radix.inc_lock_ref(last_node)
    controller.radix.dec_lock_ref(last_node)
    assert controller.radix.evictable_size_ == before
    controller.radix.assert_evictable_invariant(strict=True)


def test_burst_load_back_and_hicache_check_keeps_evictable_non_negative():
    """Simulate benchmark: many load_back + check_hicache_events without async ack queue."""
    controller = _hicache_pool(l1_pages=48, l2_pages=128)

    for idx in range(12):
        tokens = list(range(idx * 16, idx * 16 + 16))
        _seed_host_backed_prefix(controller, tokens, f"seed-{idx}")
        _evict_all_device(controller, tokens)
        req = Req(f"burst-{idx}", tokens, SamplingParams(max_new_tokens=2))
        assert controller.init_load_back(req) == 16
        assert controller.load_back(req) == 16
        controller.check_hicache_events()
        controller.radix.assert_evictable_invariant(strict=True, context=f"burst-{idx}")
        assert controller.radix.evictable_size() >= 0


def test_write_through_acks_release_lock_refs():
    controller = _hicache_pool(l1_pages=32, l2_pages=64)
    tokens = list(range(16))
    req = Req("wt-lock", tokens, SamplingParams(max_new_tokens=0))
    controller.prepare_prefill(req)
    controller.extend_prefill(req, len(tokens))
    controller.cache_unfinished_req(req)
    last = controller.radix.collect_nodes_on_path(tokens)[-1]
    before_lock = last.lock_ref
    assert before_lock > 0
    assert controller._cache.ongoing_write_through

    controller.check_hicache_events()

    assert controller._cache.ongoing_write_through == {}
    assert last.lock_ref == max(before_lock - 1, 0)


def test_cache_unfinished_req_stale_old_lock_release_does_not_crash(monkeypatch):
    controller = _hicache_pool(l1_pages=64, l2_pages=64)
    tokens = list(range(8))
    req = Req("stale-unfinished", tokens, SamplingParams(max_new_tokens=0))

    controller.prepare_prefill(req)
    controller.extend_prefill(req, 4)
    controller.cache_unfinished_req(req)

    old_node_id = req.last_node_id
    assert old_node_id is not None
    old_node = controller._cache.tree.get_node(old_node_id)
    assert old_node is not None

    original_dec = controller._cache.tree.dec_lock_ref
    injected = {"hit": False}

    def flaky_dec(node):
        if node.id == old_node.id and not injected["hit"]:
            injected["hit"] = True
            raise RuntimeError("Lock reference counter is already zero.")
        return original_dec(node)

    monkeypatch.setattr(controller._cache.tree, "dec_lock_ref", flaky_dec)

    controller.extend_prefill(req, 4)
    controller.cache_unfinished_req(req)

    assert injected["hit"]
    assert req.cached_length == len(tokens)
    controller.radix.assert_evictable_invariant(strict=True)
