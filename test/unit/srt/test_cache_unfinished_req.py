"""Unit tests for per-chunk cache_unfinished_req (sglang alignment)."""

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.chunked_prefill import split_prefill_length
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.mem_cache.radix_cache import RadixCache


def _chunked_prefill(pool: KVCachePool, req: Req) -> None:
    pool.prepare_prefill(req)
    for chunk in list(req.prefill_chunks_remaining):
        req.prefill_chunks_remaining.pop(0)
        pool.extend_prefill(req, chunk)
        pool.cache_unfinished_req(req)


def test_cache_unfinished_per_chunk_clears_deferred_pages():
    page_size = 16
    pool = KVCachePool(
        CacheConfig(
            page_size=page_size,
            l1_capacity_pages=256,
            l2_capacity_pages=64,
        )
    )
    token_ids = list(range(page_size * 12))
    req = Req("r0", token_ids, SamplingParams(max_new_tokens=8))
    req.prefill_chunks_remaining = split_prefill_length(len(token_ids), page_size * 4)
    _chunked_prefill(pool, req)
    assert req.prefill_alloc_page_ids == []
    assert req.cache_protected_len == len(token_ids)
    assert req.last_node_id is not None


def test_chunked_cache_unfinished_keeps_new_pages_in_radix():
    page_size = 1
    pool = KVCachePool(
        CacheConfig(page_size=page_size, l1_capacity_pages=10, l2_capacity_pages=0)
    )
    req = Req("chunked-release", list(range(8)), SamplingParams(max_new_tokens=4))
    req.prefill_chunks_remaining = [4, 4]

    pool.prepare_prefill(req)
    req.prefill_chunks_remaining.pop(0)
    pool.extend_prefill(req, 4)
    assert pool.available_device_tokens() == 6

    pool.cache_unfinished_req(req, chunked=True)

    assert req.prefill_alloc_page_ids == []
    assert req.cache_protected_len == 4
    assert pool.available_device_tokens() == 6

    pool.finish_request(req)
    pool.evict_from_tree_cache(10, owner="test-evict-chunked-tree-pages")
    assert pool.available_device_tokens() == 10


def test_shared_prefix_chunked_prefill_frees_duplicate_pages():
    page_size = 16
    pool = KVCachePool(
        CacheConfig(page_size=page_size, l1_capacity_pages=128, l2_capacity_pages=0)
    )
    shared = list(range(page_size * 4))
    for rid in ("a", "b"):
        req = Req(rid, shared + list(range(page_size * 8)), SamplingParams(max_new_tokens=4))
        req.prefill_chunks_remaining = split_prefill_length(len(req.input_ids), page_size * 4)
        _chunked_prefill(pool, req)
        pool.finish_request(req)
    assert pool.evictable_size() > 0


def test_finish_request_frees_lock_and_raises_available():
    page_size = 16
    pool = KVCachePool(
        CacheConfig(page_size=page_size, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    req = Req("r0", list(range(page_size * 8)), SamplingParams(max_new_tokens=4))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, len(req.input_ids))
    pool.cache_unfinished_req(req)
    before = pool.available_and_evictable_tokens()
    released = pool.finish_request(req)
    after = pool.available_and_evictable_tokens()
    assert released >= 0
    assert after >= before


def test_prepare_prefill_caps_full_cache_match_at_last_token():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    cached = Req("cached", list(range(8)), SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(cached)
    pool.extend_prefill(cached, len(cached.input_ids))
    pool.cache_unfinished_req(cached)
    pool.finish_request(cached)

    req = Req("same", list(range(8)), SamplingParams(max_new_tokens=1))
    pool.prepare_prefill(req)
    node = pool.radix.get_node(req.last_node_id)

    assert req.cached_length == len(req.input_ids) - 1
    assert req.cache_protected_len == len(req.input_ids) - 1
    assert node is not None
    assert pool.radix.prefix_length_to_node(node) == len(req.input_ids) - 1


def test_cache_unfinished_uses_fill_ids_length_not_reported_input_count():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    req = Req(
        "reported-longer",
        list(range(8)),
        SamplingParams(max_new_tokens=1),
        input_token_count=9,
    )

    pool.prepare_prefill(req)
    pool.extend_prefill(req, req.extend_length)
    pool.cache_unfinished_req(req)

    assert req.kv_committed_length == len(req.fill_ids)
    assert req.cache_protected_len == len(req.fill_ids)
    assert req.prefill_alloc_page_ids == []


def test_prepare_prefill_marks_device_prefix_protected():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    cached = Req("cached-prefix", list(range(12)), SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(cached)
    pool.extend_prefill(cached, len(cached.input_ids))
    pool.cache_unfinished_req(cached)
    pool.finish_request(cached)

    req = Req("reuse-prefix", list(range(12)) + [999], SamplingParams(max_new_tokens=1))
    pool.prepare_prefill(req)

    assert req.cached_length == len(cached.input_ids)
    assert req.cache_protected_len == len(cached.input_ids)
    assert req.kv_committed_length == 0


def test_match_prefix_device_length_is_contiguous():
    tree = RadixCache(page_size=1)
    token_ids = list(range(8))
    tree.insert(token_ids, token_ids)
    nodes = tree.collect_nodes_on_path(token_ids)

    # Simulate a HiCache state with a host-only gap followed by device-resident
    # descendants.  ``prefix_indices`` in SGLang is a contiguous device prefix,
    # so later device pages must not extend device_length across this gap.
    nodes[2].host_value = list(nodes[2].value or [])
    nodes[2].value = None
    match = tree.match_prefix(token_ids)

    assert match.device_length == 2
    assert match.page_ids[: match.device_length] == token_ids[:2]


def test_insert_rehydrates_host_only_gap_with_new_device_pages():
    tree = RadixCache(page_size=1)
    token_ids = [0, 1, 2, 3]
    tree.insert(token_ids, [10, 11, 12, 13])
    nodes = tree.collect_nodes_on_path(token_ids)

    nodes[2].host_value = [120]
    nodes[2].value = None
    tree.reconcile_accounting()
    assert tree.match_prefix(token_ids).device_length == 2

    result = tree.insert([0, 1, 2, 3, 4], [10, 11, 20, 21, 22])
    match = tree.match_prefix([0, 1, 2, 3, 4])

    assert result.prefix_len == 2
    assert result.replaced_page_ids == [13]
    assert match.device_length == 5
    assert match.page_ids == [10, 11, 20, 21, 22]


def test_cache_unfinished_req_missing_deferred_pages_does_not_raise():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=16, l2_capacity_pages=0)
    )
    cached = Req("cached-prefix", [0, 1, 2], SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(cached)
    pool.extend_prefill(cached, len(cached.input_ids))
    pool.cache_unfinished_req(cached)
    pool.finish_request(cached)

    req = Req("missing-pages", [0, 1, 2, 3, 4, 5], SamplingParams(max_new_tokens=1))
    pool.prepare_prefill(req)
    original_alloc = pool._cache._alloc_device_pages

    def _raise_alloc(num_pages: int, owner: str):
        raise RuntimeError("insufficient L1 KV cache pages")

    pool._cache._alloc_device_pages = _raise_alloc  # type: ignore[method-assign]
    try:
        kv_len, protected, last_node_id = pool._cache.cache_unfinished_req(
            req.rid,
            req.fill_ids,
            kv_committed_length=6,
            cache_protected_len=req.cache_protected_len,
            last_node_id=req.last_node_id,
            cached_length=req.cached_length,
            prefill_alloc_page_ids=[999],
            chunked=False,
        )
    finally:
        pool._cache._alloc_device_pages = original_alloc  # type: ignore[method-assign]

    assert kv_len == 3
    assert protected == 3
    assert last_node_id == req.last_node_id


def test_retract_sync_clears_lock_and_prefills_retained_output():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    req = Req("r-retract-fill", list(range(8)), SamplingParams(max_new_tokens=4))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, len(req.input_ids))
    pool.cache_unfinished_req(req)
    for token_id in (100, 101):
        pool.extend_decode(req, 1)
        req.append_output_token(token_id)

    pool.release_request(req)
    req.kv_released = False
    pool.sync_retract_prefix(req, chunk_size=4)

    assert req.last_node_id is None
    assert req.kv_committed_length == len(req.input_ids)
    assert req.prefill_length == len(req.input_ids) + len(req.output_ids)
    assert req.prefill_chunks_remaining == [2]

    pool.prepare_prefill(req)
    pool.extend_prefill(req, req.extend_length)
    pool.cache_unfinished_req(req)
    match = pool._cache.match_prefix(req.fill_ids)

    assert req.kv_committed_length == len(req.fill_ids)
    assert match.device_length == len(req.fill_ids)


def test_cache_unfinished_full_insert_preserves_existing_same_suffix_child():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    token_ids = list(range(8))
    first = Req("first", token_ids, SamplingParams(max_new_tokens=1))
    pool.prepare_prefill(first)
    pool.extend_prefill(first, len(first.input_ids))
    pool.cache_unfinished_req(first)
    first_last_node = pool.radix.get_node(first.last_node_id)
    assert first_last_node is not None

    second = Req("second", token_ids, SamplingParams(max_new_tokens=1))
    pool.prepare_prefill(second)
    pool.extend_prefill(second, second.extend_length)
    pool.cache_unfinished_req(second)

    match = pool._cache.match_prefix(token_ids)
    assert match.device_length == len(token_ids)
    assert pool.radix.get_node(first.last_node_id) is first_last_node


def test_finish_request_inserts_decode_pages_without_l1_realloc(monkeypatch):
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    req = Req("r-decode", list(range(8)), SamplingParams(max_new_tokens=3))

    pool.prepare_prefill(req)
    pool.extend_prefill(req, len(req.input_ids))
    pool.cache_unfinished_req(req)
    for token_id in (100, 101, 102):
        pool.extend_decode(req, 1)
        req.append_output_token(token_id)

    def fail_alloc(num_pages, owner):
        raise AssertionError(
            f"finish_request must not allocate L1 pages: {num_pages=} {owner=}"
        )

    monkeypatch.setattr(pool._cache.mem_pool_device, "alloc", fail_alloc)

    released = pool.finish_request(req)
    match = pool._cache.match_prefix(req.input_ids + req.output_ids)

    assert released >= 0
    assert req.decode_pages == []
    assert match.device_length == len(req.input_ids) + len(req.output_ids)


def test_release_request_does_not_insert_decode_suffix():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    req = Req("r-retract", list(range(8)), SamplingParams(max_new_tokens=3))

    pool.prepare_prefill(req)
    pool.extend_prefill(req, len(req.input_ids))
    pool.cache_unfinished_req(req)
    for token_id in (100, 101, 102):
        pool.extend_decode(req, 1)
        req.append_output_token(token_id)

    released = pool.release_request(req)
    duplicate = pool.release_request(req)
    input_match = pool._cache.match_prefix(req.input_ids)
    full_match = pool._cache.match_prefix(req.input_ids + req.output_ids)

    assert released >= 0
    assert duplicate == 0
    assert req.decode_pages == []
    assert input_match.device_length == len(req.input_ids)
    assert full_match.device_length == len(req.input_ids)


def test_pressure_evict_rebuilds_stale_evictable_leaves():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=10, l2_capacity_pages=0)
    )
    cached = Req("cached", list(range(8)), SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(cached)
    pool.extend_prefill(cached, len(cached.input_ids))
    pool.cache_unfinished_req(cached)
    pool.finish_request(cached)
    assert pool.evictable_size() >= len(cached.input_ids)

    # Simulate a stale leaf set after retract/finalize churn: aggregate
    # accounting says pages are evictable, but the eviction heap would be empty
    # unless pressure eviction rebuilds the radix leaf status.
    pool.radix.evictable_leaves.clear()

    req = Req("new", list(range(100, 106)), SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, len(req.input_ids))

    assert req.kv_committed_length == len(req.input_ids)


def test_cache_unfinished_chunked_skips_hit_count():
    pool = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=128,
            l2_capacity_pages=64,
            hicache_write_policy="write_through_selective",
            write_through_threshold=3,
        )
    )
    req = Req("r-hit", list(range(12)), SamplingParams(max_new_tokens=2))
    req.prefill_chunks_remaining = [6, 6]
    pool.prepare_prefill(req)

    first = req.prefill_chunks_remaining.pop(0)
    pool.extend_prefill(req, first)
    pool.cache_unfinished_req(req, chunked=True)
    first_nodes = pool.radix.collect_nodes_on_path(req.input_ids[:first])
    assert first_nodes[-1].hit_count == 0

    second = req.prefill_chunks_remaining.pop(0)
    pool.extend_prefill(req, second)
    pool.cache_unfinished_req(req, chunked=False)
    full_nodes = pool.radix.collect_nodes_on_path(req.input_ids)
    assert full_nodes[-1].hit_count >= 1


def test_write_through_chunked_cache_unfinished_skips_backup_until_final_insert():
    pool = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=128,
            l2_capacity_pages=128,
            hicache_write_policy="write_through",
        )
    )
    req = Req("r-wt-chunk", list(range(12)), SamplingParams(max_new_tokens=2))
    req.prefill_chunks_remaining = [6, 6]
    pool.prepare_prefill(req)

    first = req.prefill_chunks_remaining.pop(0)
    pool.extend_prefill(req, first)
    pool.cache_unfinished_req(req, chunked=True)
    first_nodes = pool.radix.collect_nodes_on_path(req.input_ids[:first])

    assert first_nodes
    assert all(not node.backuped for node in first_nodes)

    second = req.prefill_chunks_remaining.pop(0)
    pool.extend_prefill(req, second)
    pool.cache_unfinished_req(req, chunked=False)
    pool.flush_write_through_acks()
    full_nodes = pool.radix.collect_nodes_on_path(req.input_ids)

    assert all(node.backuped for node in first_nodes)
    assert any(node.backuped for node in full_nodes)


def test_stale_leaf_lock_release_frees_live_ancestor_lock():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=128, l2_capacity_pages=0)
    )
    first = pool.radix.insert([1, 2, 3, 4], [1, 2, 3, 4])
    pool.radix.insert([1, 2, 5, 6], [1, 2, 5, 6])
    match = pool.radix.match_prefix([1, 2, 3, 4])
    stale_leaf = match.last_node
    assert stale_leaf is not None
    parent = stale_leaf.parent
    assert parent is not None and parent is not pool.radix.root_node

    pool.radix.inc_lock_ref(stale_leaf)
    stale_leaf.lock_ref = 0
    pool.radix.reconcile_accounting()
    expected_released = 0
    cur = stale_leaf.parent
    while cur is not None and cur is not pool.radix.root_node:
        if cur.lock_ref > 0 and cur.value is not None:
            expected_released += len(cur.key)
        cur = cur.parent
    protected_before = pool.radix.protected_size_

    released = pool._cache._dec_request_lock_ref_tolerant(
        stale_leaf,
        req_id="stale",
        context="test",
    )

    assert released == expected_released
    assert parent.lock_ref == 0
    assert pool.radix.protected_size_ == protected_before - expected_released
    assert first.last_node is not None


def test_stale_leaf_lock_release_allows_pressure_prefill_allocation():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=5, l2_capacity_pages=0)
    )
    pages = pool._cache._alloc_device_pages(4, owner="cached")
    inserted = pool.radix.insert(
        [1, 2, 3, 4],
        [page.page_id for page in pages],
    )
    pool.radix.inc_lock_ref(inserted.last_node)

    # Simulate a stale leaf after radix split/finalize churn: the request's
    # stored leaf has lost its ref, while live ancestors still protect pages.
    inserted.last_node.lock_ref = 0
    pool.radix.reconcile_accounting()
    assert pool.available_device_tokens() == 1
    assert pool.available_and_evictable_tokens() < 5

    released = pool._cache._dec_request_lock_ref_tolerant(
        inserted.last_node,
        req_id="stale-pressure",
        context="test",
    )
    assert released == 3
    assert pool.available_and_evictable_tokens() == 5

    req = Req("new", [10, 11, 12, 13], SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, len(req.input_ids))
    pool.cache_unfinished_req(req)

    assert req.kv_committed_length == len(req.input_ids)
    assert pool._cache.match_prefix(req.input_ids).device_length == len(req.input_ids)


def test_cache_unfinished_uses_cache_protected_len_when_cached_length_stale():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=128, l2_capacity_pages=0)
    )
    req = Req("r-stale", list(range(20)), SamplingParams(max_new_tokens=2))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, 10)
    pool.cache_unfinished_req(req)
    assert req.cache_protected_len == 10
    assert req.prefill_alloc_page_ids == []

    # Admission-time cached_length can lag behind incremental tree inserts.
    req.cached_length = 4
    pool.extend_prefill(req, 6)
    assert len(req.prefill_alloc_page_ids) == 6
    pool.cache_unfinished_req(req)
    assert req.cache_protected_len == 16
    assert req.kv_committed_length == 16
    assert req.prefill_alloc_page_ids == []


def test_alloc_only_prefill_reuses_device_prefix_without_active_lock():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=16, l2_capacity_pages=0)
    )
    cached = Req("cached-device", list(range(8)), SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(cached)
    pool.extend_prefill(cached, len(cached.input_ids))
    pool.cache_unfinished_req(cached)
    pool.finish_request(cached)
    assert pool._cache.match_prefix(cached.input_ids).device_length == 8

    req = Req(
        "retry-with-prefix",
        cached.input_ids + [100, 101, 102],
        SamplingParams(max_new_tokens=0),
    )
    req.kv_committed_length = len(cached.input_ids)
    req.cached_length = len(cached.input_ids)
    req.cache_protected_len = 0
    req.last_node_id = None

    pool.extend_prefill(req, 3)

    assert req.kv_committed_length == len(req.input_ids)
    assert len(req.prefill_alloc_page_ids) == 3
    assert pool._cache.match_prefix(cached.input_ids).device_length == 8


def test_cache_unfinished_ignores_overestimated_cached_length_for_device_prefix():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=128, l2_capacity_pages=0)
    )
    req = Req("r-host-stale", list(range(20)), SamplingParams(max_new_tokens=2))
    pool.prepare_prefill(req)
    pool.extend_prefill(req, 4)
    pool.cache_unfinished_req(req)
    assert req.cache_protected_len == 4
    assert req.prefill_alloc_page_ids == []

    # A later match can overestimate ``cached_length`` with host-side hits, but
    # cache_unfinished must only use device-protected KV as prefix_indices.
    req.cached_length = 8
    pool.extend_prefill(req, 4)
    assert len(req.prefill_alloc_page_ids) == 4
    pool.cache_unfinished_req(req)

    assert req.cache_protected_len == 8
    assert req.kv_committed_length == 8
    assert req.cached_length == 8
    assert req.prefill_alloc_page_ids == []
    assert pool._cache.match_prefix(req.fill_ids[:8]).device_length == 8


def test_cache_unfinished_backfills_missing_deferred_pages():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=64, l2_capacity_pages=0)
    )
    req = Req("r-missing", list(range(10)), SamplingParams(max_new_tokens=0))

    pool.prepare_prefill(req)
    pool.extend_prefill(req, 4)
    assert len(req.prefill_alloc_page_ids) == 4

    # Simulate a stale committed length after blocked/retract retries. The
    # simulator's deferred-prefill path must restore the missing device page so
    # the radix insert preserves the full committed prefix, matching SGLang's
    # invariant that committed KV already exists in req_to_token.
    req.kv_committed_length = 8
    pool.cache_unfinished_req(req)

    match = pool._cache.match_prefix(req.fill_ids[:8])

    assert req.kv_committed_length == 8
    assert req.cache_protected_len == 8
    assert req.cached_length == 8
    assert req.prefill_alloc_page_ids == []
    assert match.device_length == 8


def test_prepare_prefill_does_not_commit_host_only_prefix():
    pool = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=12,
            l2_capacity_pages=64,
            hicache_write_policy="write_through",
        )
    )
    cached = Req("cached-host", list(range(12)), SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(cached)
    pool.extend_prefill(cached, len(cached.input_ids))
    pool.cache_unfinished_req(cached)
    pool.finish_request(cached)
    pool.flush_write_through_acks()
    pool.evict_from_tree_cache(12, owner="unit-host-only")

    req = Req("host-hit", list(range(12)), SamplingParams(max_new_tokens=1))
    host_match = pool._cache.match_prefix(req.fill_ids[:-1])
    assert host_match.host_length > 0
    assert host_match.device_length == 0
    assert pool.matched_prefix_length(req) == 0

    req.cached_length = 11
    req.cache_protected_len = 11
    pool.prepare_prefill(req)
    node = pool.radix.get_node(req.last_node_id)

    assert req.cached_length == 0
    assert req.cache_protected_len == 0
    assert node is pool.radix.root_node


def test_init_load_back_defers_quota_check_to_load_back():
    pool = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=12,
            l2_capacity_pages=64,
            hicache_write_policy="write_through",
        )
    )
    cached = Req("cached-host-quota", list(range(12)), SamplingParams(max_new_tokens=0))
    pool.prepare_prefill(cached)
    pool.extend_prefill(cached, len(cached.input_ids))
    pool.cache_unfinished_req(cached)
    pool.finish_request(cached)
    pool.flush_write_through_acks()
    pool.evict_from_tree_cache(12, owner="unit-host-quota")

    host_hit, last_host = pool._cache.init_load_back(cached.input_ids, mem_quota=1)

    assert host_hit == len(cached.input_ids)
    assert last_host is not None


def test_write_through_ack_does_not_change_request_lock_accounting():
    pool = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=128,
            l2_capacity_pages=128,
            hicache_write_policy="write_through",
        )
    )
    req = Req("r-wt", list(range(16)), SamplingParams(max_new_tokens=2))

    pool.prepare_prefill(req)
    pool.extend_prefill(req, len(req.input_ids))
    pool.cache_unfinished_req(req)

    protected_before_ack = pool.radix.protected_size_
    evictable_before_ack = pool.radix.evictable_size_
    flushed = pool.flush_write_through_acks()

    assert flushed > 0
    assert pool.radix.protected_size_ == protected_before_ack
    assert pool.radix.evictable_size_ == evictable_before_ack
    pool.radix.assert_evictable_invariant(context="unit", strict=True)


def test_radix_accounting_reconciles_from_node_state():
    pool = KVCachePool(
        CacheConfig(page_size=1, l1_capacity_pages=128, l2_capacity_pages=0)
    )
    req = Req("r-reconcile", list(range(16)), SamplingParams(max_new_tokens=2))

    pool.prepare_prefill(req)
    pool.extend_prefill(req, len(req.input_ids))
    pool.cache_unfinished_req(req)
    measured = pool.radix.measured_protected_tokens()
    assert measured > 0

    pool.radix.protected_size_ = -123
    pool.radix.assert_evictable_invariant(context="unit", strict=True)

    assert pool.radix.protected_size_ == measured
