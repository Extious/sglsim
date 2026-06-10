"""PrefillAdder parity checks vs SGLang schedule_policy.add_one_req."""

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.prefill_adder import (
    AddReqResult,
    PrefillAdder,
    _new_request_decode_reserve,
)
from sglsim.srt.managers.schedule_batch import ForwardMode, Req, SamplingParams, ScheduleBatch
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool


def _adder(
    *,
    page_size: int = 16,
    rem_chunk_tokens: int | None = 8192,
    new_token_ratio: float = 0.5,
    l1_pages: int = 10_000,
) -> PrefillAdder:
    cache = KVCachePool(
        CacheConfig(page_size=page_size, l1_capacity_pages=l1_pages, l2_capacity_pages=0)
    )
    running = ScheduleBatch(0, 0, ForwardMode.DECODE, [])
    return PrefillAdder(
        page_size=page_size,
        cache=cache,
        running_batch=running,
        max_prefill_tokens=16384,
        max_running_requests=16,
        rem_chunk_tokens=rem_chunk_tokens,
        new_token_ratio=new_token_ratio,
    )


def _req(
    rid: str,
    extend: int,
    *,
    host_hit: int = 0,
    max_new: int = 64,
) -> Req:
    req = Req(rid, [0] * extend, SamplingParams(max_new_tokens=max_new))
    req.prefill_chunks_remaining = [extend]
    req.host_hit_length = host_hit
    return req


def test_new_request_decode_reserve_ignores_new_token_ratio():
    req = _req("a", 100, max_new=128)
    assert _new_request_decode_reserve(req) == 128
    adder = _adder(new_token_ratio=0.1)
    assert adder.add_one_req(req) in (AddReqResult.CONTINUE, AddReqResult.NO_TOKEN)
    assert req in adder.can_run_list


def test_real_input_tokens_subtracts_host_hit_length():
    page_size = 16
    req = _req("x", page_size * 4, host_hit=49)
    adder = _adder(page_size=page_size)
    real = adder._ceil_paged_tokens(max(req.extend_length - req.host_hit_length, 0))
    assert real == page_size


def test_page_aligned_first_chunk_truncation():
    page_size = 16
    adder = _adder(page_size=page_size, rem_chunk_tokens=100)
    req = _req("chunk", 200)
    adder.add_one_req(req)
    assert adder.new_chunked_req is req
    assert req.prefill_chunks_remaining[0] == 96  # 100 // 16 * 16
    assert sum(req.prefill_chunks_remaining) == 200


def test_full_first_chunk_with_remaining_chunks_is_still_chunked():
    page_size = 16
    adder = _adder(page_size=page_size, rem_chunk_tokens=96)
    req = _req("chunk", 192)
    req.prefill_chunks_remaining = [96, 96]

    result = adder.add_one_req(req)

    assert adder.new_chunked_req is req
    assert req in adder.can_run_list
    assert result == AddReqResult.OTHER


def test_chunked_continuation_does_not_log_prefix_as_cache_hit():
    page_size = 1
    adder = _adder(page_size=page_size, rem_chunk_tokens=8192)
    req = _req("chunk-cont", 14847)
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.prefill_chunks_remaining = [6655]

    assert adder.add_chunked_req(req) is None

    assert req in adder.can_run_list
    assert adder.log_input_tokens == 6655
    assert adder.log_hit_tokens == 0


def test_fresh_request_budget_uses_full_extend_before_chunking():
    page_size = 1
    cache = KVCachePool(
        CacheConfig(page_size=page_size, l1_capacity_pages=100_000, l2_capacity_pages=0)
    )
    adder = PrefillAdder(
        page_size=page_size,
        cache=cache,
        running_batch=ScheduleBatch(0, 0, ForwardMode.DECODE, []),
        max_prefill_tokens=16384,
        max_running_requests=4096,
        rem_chunk_tokens=8192,
        new_token_ratio=0.65,
    )
    active = Req("active", list(range(14847)), SamplingParams(max_new_tokens=256))
    active.kv_committed_length = 8192
    active.cached_length = 8192
    active.cache_protected_len = 8192
    active.last_node_id = 1
    active.prefill_chunks_remaining = [6655]
    assert adder.add_chunked_req(active) is None
    assert active in adder.can_run_list

    fresh = Req("fresh", list(range(14850)), SamplingParams(max_new_tokens=256))
    fresh.prefill_chunks_remaining = [8192, 6658]

    result = adder.add_one_req(fresh, has_chunked_req=False)

    assert result == AddReqResult.OTHER
    assert fresh not in adder.can_run_list
    assert fresh.prefill_chunks_remaining == [8192, 6658]


def test_rem_input_tokens_boundary_allows_first_req():
    page_size = 16
    adder = _adder(page_size=page_size, rem_chunk_tokens=None, l1_pages=500)
    req = _req("only", page_size * 2)
    result = adder.add_one_req(req)
    assert req in adder.can_run_list
    assert result in (AddReqResult.CONTINUE, AddReqResult.NO_TOKEN, AddReqResult.OTHER)


def test_second_req_other_when_rem_input_exhausted():
    page_size = 16
    adder = PrefillAdder(
        page_size=page_size,
        cache=KVCachePool(
            CacheConfig(page_size=page_size, l1_capacity_pages=500, l2_capacity_pages=0)
        ),
        running_batch=ScheduleBatch(0, 0, ForwardMode.DECODE, []),
        max_prefill_tokens=32,
        max_running_requests=16,
        rem_chunk_tokens=None,
        new_token_ratio=0.0,
    )
    r1 = _req("a", 32)
    r2 = _req("b", 32)
    adder.add_one_req(r1)
    assert r1 in adder.can_run_list
    assert adder.add_one_req(r2) == AddReqResult.OTHER
    assert r2 not in adder.can_run_list
