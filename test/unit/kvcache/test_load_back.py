from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.simulation.manager.state import StateManager


def _seed_evicted_host_only(
    controller: KVCachePool,
    tokens: list[int],
    req_id: str,
) -> None:
    req = Req(req_id, tokens, SamplingParams(max_new_tokens=0))
    controller.prepare_prefill(req)
    controller.extend_prefill(req, len(tokens))
    controller.check_hicache_events()
    controller.finish_request(req)
    for node in controller.radix.collect_nodes_on_path(tokens):
        if node.host_value is not None:
            if node.value is not None and node.lock_ref == 0:
                controller.radix.evictable_size_ -= len(node.key)
            node.value = None
            if node.lock_ref == 0 and node not in controller.radix.evictable_host_leaves:
                controller.radix._update_host_leaf_status(node)


def test_load_back_skips_small_host_hit():
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=16,
            l2_capacity_pages=32,
            load_back_threshold=10,
            hicache_write_policy="write_through",
            write_through_threshold=1,
        )
    )
    _seed_evicted_host_only(controller, list(range(9)), "seed")
    req = Req("r1", list(range(9)), SamplingParams(max_new_tokens=1))
    host_hit = controller.init_load_back(req)
    assert host_hit == 0
    assert controller.load_back(req) == 0


def test_load_back_restores_value_immediately_and_queues_h2d():
    """Aligned with sglang load_back: node.value (device metadata) is restored
    synchronously; only the H2D transfer timing is deferred to drain/loading_check.
    """
    StateManager.reset()
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=32,
            l2_capacity_pages=64,
            load_back_threshold=10,
            hicache_write_policy="write_through",
            write_through_threshold=1,
        )
    )
    tokens = list(range(16))
    _seed_evicted_host_only(controller, tokens, "seed")
    req = Req("r1", tokens, SamplingParams(max_new_tokens=1))
    host_hit = controller.init_load_back(req)
    # SGLang matches at most input_len - 1 in the extend path so logits for the
    # last input token are still computed.
    assert host_hit == 15
    assert controller.load_back(req) == 15
    for node in controller.radix.collect_nodes_on_path(tokens[:-1]):
        if node.host_value is not None:
            assert node.value is not None


def test_load_back_ongoing_lock_released_after_hicache_check():
    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=32,
            l2_capacity_pages=64,
            load_back_threshold=10,
            hicache_write_policy="write_through",
            write_through_threshold=1,
        )
    )
    tokens = list(range(16))
    _seed_evicted_host_only(controller, tokens, "seed")
    req = Req("r-lock", tokens, SamplingParams(max_new_tokens=1))
    assert controller.init_load_back(req) == 15
    assert req.last_host_node_id is not None
    assert controller.load_back(req) == 15
    last_host = controller._cache.tree.get_node(req.last_host_node_id)
    assert last_host is not None
    assert last_host.lock_ref > 0
    assert controller._cache.ongoing_load_back

    controller.check_hicache_events()

    assert controller._cache.ongoing_load_back == {}
    assert last_host.lock_ref == 0

    controller.check_hicache_events()
    assert controller.metrics["prefetch_pages"] > 0
    assert StateManager.pop_hicache_l2_load_dur() > 0
    for node in controller.radix.collect_nodes_on_path(tokens[:-1]):
        if node.host_value is not None:
            assert node.value is not None
