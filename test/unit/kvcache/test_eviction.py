from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.mem_cache.radix_cache import RadixCache


def test_evictable_size_counts_unlocked_path():
    tree = RadixCache(page_size=1)
    tree.insert([1], page_ids=[10])
    node = tree.get_node(tree.match_prefix([1]).last_node.id)
    assert tree.evictable_size() == 1
    tree.inc_lock_ref(node)
    assert tree.evictable_size() == 0
    tree.dec_lock_ref(node)
    assert tree.evictable_size() == 1


def test_evict_backuped_clears_l1_only():
    tree = RadixCache(page_size=1)
    tree.insert([1], page_ids=[10])
    node = tree.match_prefix([1]).last_node
    tree.set_host_value(node, [100])
    pool = {10: type("P", (), {"page_id": 10})()}
    freed = tree.evict(1, pool, write_through=True)
    assert len(freed) == 1
    assert node.value is None
    assert node.host_value == [100]


def test_evict_regular_removes_leaf():
    tree = RadixCache(page_size=1)
    tree.insert([1], page_ids=[10])
    pool = {10: type("P", (), {"page_id": 10})()}
    freed = tree.evict(1, pool, write_through=False)
    assert len(freed) == 1
    assert tree.match_prefix([1]).matched_length == 0


def test_deep_chain_unlocked_evictable_size():
    tree = RadixCache(page_size=16)
    tokens = list(range(256))
    page_ids = []
    for i in range(0, 256, 16):
        pid = i // 16
        page_ids.extend([pid] * 16)
    tree.insert(tokens, page_ids)
    assert tree.evictable_size() == 256


def test_available_and_evictable_after_unlock():
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=1,
            l2_capacity_pages=0,
        )
    )
    req0 = Req("r0", [1], SamplingParams(max_new_tokens=0))
    req1 = Req("r1", [2], SamplingParams(max_new_tokens=0))
    controller.prepare_prefill(req0)
    controller.extend_prefill(req0, 1)
    controller.finish_request(req0)
    assert controller.available_and_evictable_tokens() >= 1
    controller.prepare_prefill(req1)
    controller.extend_prefill(req1, 1)


def test_evict_from_tree_cache_uses_incremental_accounting(monkeypatch):
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=1,
            l2_capacity_pages=0,
        )
    )
    req0 = Req("r0", [1], SamplingParams(max_new_tokens=0))
    controller.prepare_prefill(req0)
    controller.extend_prefill(req0, 1)
    controller.finish_request(req0)
    assert controller.available_device_tokens() == 0
    assert controller.radix.evictable_size() == 1

    reconcile_calls = 0
    original_reconcile = controller.radix.reconcile_accounting

    def counted_reconcile():
        nonlocal reconcile_calls
        reconcile_calls += 1
        return original_reconcile()

    monkeypatch.setattr(controller.radix, "reconcile_accounting", counted_reconcile)

    controller.evict_from_tree_cache(1, owner="unit")

    assert reconcile_calls == 0
    assert controller.available_device_tokens() == 1
