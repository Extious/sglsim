from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.prefill_adder import AddReqResult, PrefillAdder
from sglsim.srt.managers.schedule_batch import ForwardMode, Req, SamplingParams, ScheduleBatch
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool


def _adder(
    *,
    max_prefill_tokens: int = 16384,
    rem_chunk_tokens: int | None = 8192,
    l1_pages: int = 100_000,
) -> PrefillAdder:
    cache = KVCachePool(
        CacheConfig(page_size=16, l1_capacity_pages=l1_pages, l2_capacity_pages=0)
    )
    return PrefillAdder(
        page_size=16,
        cache=cache,
        running_batch=ScheduleBatch(0, 0, ForwardMode.DECODE, []),
        max_prefill_tokens=max_prefill_tokens,
        max_running_requests=16,
        rem_chunk_tokens=rem_chunk_tokens,
        new_token_ratio=0.0,
    )


def _req(rid: str, extend: int) -> Req:
    req = Req(rid, [0] * extend, SamplingParams(max_new_tokens=64))
    req.prefill_chunks_remaining = [extend]
    return req


def test_two_8k_extends_split_across_prefill_batches():
    # chunked_prefill_size (rem_chunk_tokens) is a per-batch token budget, not a
    # per-request cap. A single 8192-token extend exhausts the 8192 chunk budget,
    # so the second request cannot join the same prefill batch (matches sglang,
    # where #new-seq stays at 1 for back-to-back 8k prompts).
    adder = _adder()
    assert adder.add_one_req(_req("a", 8192)) == AddReqResult.OTHER
    assert adder.add_one_req(_req("b", 8192)) == AddReqResult.OTHER
    assert len(adder.can_run_list) == 1


def test_single_22k_sets_new_chunked_req():
    adder = _adder()
    req = _req("long", 22074)
    adder.add_one_req(req)
    assert adder.new_chunked_req is req
    assert req.prefill_chunks_remaining[0] == 8192
    assert req.prefill_chunks_remaining[1] == 22074 - 8192
