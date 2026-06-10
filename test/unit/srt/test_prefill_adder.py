from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.prefill_adder import AddReqResult, PrefillAdder
from sglsim.srt.managers.schedule_batch import ForwardMode, Req, SamplingParams, ScheduleBatch
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool


def _running_with(req: Req) -> ScheduleBatch:
    return ScheduleBatch(
        batch_id=0,
        rank_id=0,
        mode=ForwardMode.DECODE,
        reqs=[req],
    )


def test_decode_reserve_reduces_rem_total_tokens():
    cache = KVCachePool(CacheConfig(page_size=16, l1_capacity_pages=100, l2_capacity_pages=0))
    decode_req = Req("d0", [1], SamplingParams(max_new_tokens=512))
    adder = PrefillAdder(
        page_size=16,
        cache=cache,
        running_batch=_running_with(decode_req),
        max_prefill_tokens=16384,
        max_running_requests=16,
    )
    assert adder.rem_total_tokens < cache.available_and_evictable_tokens()


def test_try_add_rejects_when_tokens_insufficient():
    cache = KVCachePool(CacheConfig(page_size=16, l1_capacity_pages=1, l2_capacity_pages=0))
    req = Req("r0", list(range(4096)), SamplingParams(max_new_tokens=512))
    req.prefill_chunks_remaining = [4096]
    adder = PrefillAdder(
        page_size=16,
        cache=cache,
        running_batch=ScheduleBatch(0, 0, ForwardMode.DECODE, []),
        max_prefill_tokens=16384,
        max_running_requests=16,
    )
    assert adder.try_add(req) == AddReqResult.NO_TOKEN


def test_hicache_no_token_leaves_batch_not_full_when_running_empty():
    cache = KVCachePool(
        CacheConfig(page_size=16, l1_capacity_pages=1, l2_capacity_pages=4)
    )
    req = Req("r0", list(range(4096)), SamplingParams(max_new_tokens=512))
    req.prefill_chunks_remaining = [4096]
    running = ScheduleBatch(0, 0, ForwardMode.DECODE, [])
    running.batch_is_full = True
    adder = PrefillAdder(
        page_size=16,
        cache=cache,
        running_batch=running,
        max_prefill_tokens=16384,
        max_running_requests=16,
    )
    assert adder.try_add(req) == AddReqResult.NO_TOKEN
    if cache.hierarchical_enabled:
        running.batch_is_full = (
            len(adder.can_run_list) > 0 or not running.is_empty()
        )
    assert running.batch_is_full is False


def test_proactive_evict_admits_waiting_after_fill():
    page_size = 16
    cache = KVCachePool(
        CacheConfig(page_size=page_size, l1_capacity_pages=8, l2_capacity_pages=0)
    )
    filler = Req("fill", list(range(page_size * 6)), SamplingParams(max_new_tokens=1))
    cache.prepare_prefill(filler)
    cache.extend_prefill(filler, len(filler.input_ids))
    cache.cache_unfinished_req(filler)
    cache.finish_request(filler)

    req = Req("r0", list(range(page_size * 4)), SamplingParams(max_new_tokens=8))
    req.prefill_chunks_remaining = [page_size * 4]
    adder = PrefillAdder(
        page_size=page_size,
        cache=cache,
        running_batch=ScheduleBatch(0, 0, ForwardMode.DECODE, []),
        max_prefill_tokens=16384,
        max_running_requests=16,
        new_token_ratio=0.0,
    )
    # Admission may return NO_TOKEN once KV is committed (budget exhausted) while
    # still keeping the request in can_run_list — matches SGLang budget_state().
    adder.try_add(req)
    assert adder.can_run_list == [req]


def test_sync_extend_after_load_back_allows_host_only_kv_to_shrink():
    class _CacheStub:
        hierarchical_enabled = True

        def available_and_evictable_tokens(self) -> int:
            return 10_000

        def device_resident_length(self, req: Req) -> int:
            return 3

    req = Req("r0", list(range(8)), SamplingParams(max_new_tokens=4))
    req.cached_length = 8
    req.kv_committed_length = 8
    req.prefill_chunks_remaining = [8]
    req.prefill_alloc_page_ids = []
    adder = PrefillAdder(
        page_size=1,
        cache=_CacheStub(),  # type: ignore[arg-type]
        running_batch=ScheduleBatch(0, 0, ForwardMode.DECODE, []),
        max_prefill_tokens=64,
        max_running_requests=8,
        chunked_prefill_size=4,
    )

    adder._sync_extend_after_load_back(req)

    assert req.kv_committed_length == 3
    assert req.prefill_chunks_remaining == [4, 1]


def test_add_one_req_does_not_keep_host_hit_on_load_back_failure():
    class _CacheStub:
        hierarchical_enabled = True

        def __init__(self) -> None:
            self.prepared: list[str] = []

        def available_and_evictable_tokens(self) -> int:
            return 10_000

        def init_load_back(self, req: Req, *, mem_quota: int | None = None) -> int:
            req.host_hit_length = 32
            req.last_host_node_id = 123
            return 32

        def load_back(self, req: Req, *, mem_quota: int | None = None) -> int:
            return 0

        def device_resident_length(self, req: Req) -> int:
            return 0

        def prepare_prefill(self, req: Req) -> None:
            self.prepared.append(req.rid)

        def evict_from_tree_cache(self, num_tokens: int, *, owner: str = "evict") -> None:
            return None

    req = Req("r_hh", list(range(64)), SamplingParams(max_new_tokens=8))
    req.prefill_chunks_remaining = [64]
    adder = PrefillAdder(
        page_size=16,
        cache=_CacheStub(),  # type: ignore[arg-type]
        running_batch=ScheduleBatch(0, 0, ForwardMode.DECODE, []),
        max_prefill_tokens=4096,
        max_running_requests=8,
        rem_chunk_tokens=None,
    )

    result = adder.add_one_req(req)

    assert result in (AddReqResult.CONTINUE, AddReqResult.NO_TOKEN, AddReqResult.OTHER)
    assert req in adder.can_run_list
    assert req.host_hit_length == 0
    assert req.last_host_node_id is None
    # Budget uses full extend length when load_back fails.
    assert adder.log_input_tokens == 64


def test_active_chunked_req_waits_when_decode_reserve_exhausts_budget():
    class _CacheStub:
        hierarchical_enabled = False

        def available_and_evictable_tokens(self) -> int:
            return 128

    decode_req = Req("decode", [1], SamplingParams(max_new_tokens=512))
    req = Req("chunk", list(range(4096)), SamplingParams(max_new_tokens=256))
    req.prefill_chunks_remaining = [4096]
    req.kv_committed_length = 0
    req.cache_protected_len = 1024
    req.last_node_id = 1
    adder = PrefillAdder(
        page_size=16,
        cache=_CacheStub(),  # type: ignore[arg-type]
        running_batch=ScheduleBatch(0, 0, ForwardMode.DECODE, [decode_req]),
        max_prefill_tokens=8192,
        max_running_requests=8,
        rem_chunk_tokens=8192,
        new_token_ratio=1.0,
    )

    assert adder.add_chunked_req(req) is req
    assert adder.can_run_list == []
    assert req.prefill_chunks_remaining == [4096]
