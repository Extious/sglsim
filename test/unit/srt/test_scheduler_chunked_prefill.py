from sglsim.srt.managers.schedule_batch import (
    ForwardMode,
    Req,
    RequestStatus,
    SamplingParams,
    ScheduleBatch,
)
from sglsim.srt.managers.prefill_adder import PrefillAdder
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.types import RequestStats
from sglsim.srt.utils.network import SimIpcRegistry


def _scheduler(l2_capacity_pages: int = 0) -> Scheduler:
    SimIpcRegistry.install_global(None)
    ConfigManager._raw = {
        "scheduler": {
            "max_prefill_tokens": 16384,
            "chunked_prefill_size": 8192,
            "pp_max_micro_batch_size": 8,
            "init_new_token_ratio": 0.0,
        },
        "cache": {
            "page_size": 16,
            "l1_capacity_pages": 100_000,
            "l2_capacity_pages": l2_capacity_pages,
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    server_args = ServerArgs()
    return Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)


def _make_req(rid: str, num_chunks: int) -> Req:
    extend = 8192 * num_chunks
    req = Req(rid, list(range(extend)), SamplingParams(max_new_tokens=8))
    req.prefill_chunks_remaining = [8192] * num_chunks
    REQUEST_STATS[rid] = RequestStats(rid=rid)
    REQUEST_STATS[rid].input_length = extend
    return req


def _enqueue(scheduler: Scheduler, rid: str, num_chunks: int) -> None:
    scheduler.waiting_queue.append(_make_req(rid, num_chunks))


def _drain(scheduler: Scheduler, max_steps: int = 2000) -> None:
    scheduler.last_batch = None
    for _ in range(max_steps):
        if scheduler.last_batch is not None:
            scheduler.process_batch_result(scheduler.last_batch)
        batch = scheduler.get_next_batch_to_run()
        if batch is None:
            if (
                not scheduler.waiting_queue
                and scheduler.chunked_req is None
                and scheduler.running_batch.is_empty()
            ):
                break
            scheduler.last_batch = None
            continue
        scheduler.run_batch(batch)
        scheduler.last_batch = batch


def test_chunked_prefill_runs_one_sequence_at_a_time():
    """Aligns with sglang's single in-flight chunked_req: a request that has
    started chunked prefill is finished serially before the next request enters
    prefill, so every prefill batch carries exactly one new sequence
    (#new-seq: 1) and the two requests never share a prefill batch."""
    scheduler = _scheduler()
    _enqueue(scheduler, "r0", num_chunks=3)
    _enqueue(scheduler, "r1", num_chunks=2)

    prefill_batches: list[list[str]] = []
    r0_done_chunk_steps = 0
    r1_started = False
    r1_overlapped_r0 = False

    scheduler.last_batch = None
    for _ in range(200):
        if scheduler.last_batch is not None:
            batch = scheduler.last_batch
            if batch.mode == ForwardMode.PREFILL:
                rids = [req.rid for req in batch.reqs]
                prefill_batches.append(rids)
                assert len(rids) == 1, prefill_batches
                if rids == ["r1"]:
                    r1_started = True
                if "r0" in rids:
                    r0_done_chunk_steps += 1
                    if r1_started:
                        r1_overlapped_r0 = True
            scheduler.process_batch_result(batch)

        batch = scheduler.get_next_batch_to_run()
        if batch is None:
            if (
                not scheduler.waiting_queue
                and scheduler.chunked_req is None
                and scheduler.running_batch.is_empty()
            ):
                break
            scheduler.last_batch = None
            continue

        scheduler.run_batch(batch)
        scheduler.last_batch = batch

    assert r1_started, prefill_batches
    assert not r1_overlapped_r0, prefill_batches
    assert prefill_batches == [["r0"], ["r0"], ["r0"], ["r1"], ["r1"]], prefill_batches


def test_eight_chunked_reqs_decode_together():
    scheduler = _scheduler()
    for i in range(8):
        _enqueue(scheduler, f"r{i}", num_chunks=3)

    max_decode_bs = 0
    scheduler.last_batch = None
    for _ in range(800):
        if scheduler.last_batch is not None:
            scheduler.process_batch_result(scheduler.last_batch)
        batch = scheduler.get_next_batch_to_run()
        if batch is None:
            if (
                not scheduler.waiting_queue
                and scheduler.chunked_req is None
                and scheduler.running_batch.is_empty()
            ):
                break
            scheduler.last_batch = None
            continue
        if batch.mode == ForwardMode.DECODE:
            max_decode_bs = max(max_decode_bs, len(batch.reqs))
        scheduler.run_batch(batch)
        scheduler.last_batch = batch

    assert max_decode_bs == 8


def test_stash_chunked_request_noop_after_deferred_pages_flushed():
    scheduler = _scheduler()
    chunked = _make_req("chunked", num_chunks=2)
    chunked.kv_committed_length = len(chunked.input_ids) // 2
    chunked.cached_length = chunked.kv_committed_length
    chunked.prefill_alloc_page_ids = []
    scheduler.chunked_req = chunked
    scheduler.stash_chunked_request(chunked)


def test_process_batch_result_chunked_prefill_stashes_progress():
    scheduler = _scheduler()
    req = _make_req("chunked_defer", num_chunks=2)
    req.status = RequestStatus.PREFILL
    req.kv_committed_length = 8192
    req.cached_length = 0
    req.cache_protected_len = 0
    req.prefill_alloc_page_ids = [1] * 8192
    batch = ScheduleBatch(14, 0, ForwardMode.PREFILL, [req])
    batch.inference_dur_s = 0.01
    batch.metadata["prefill_extend_tokens"] = {req.rid: 8192}
    batch.metadata["prefill_chunks_consumed_in_run_batch"] = False

    called = {"n": 0}

    def _cache_unfinished(_req, *, chunked=False):
        called["n"] += 1

    scheduler.cache.cache_unfinished_req = _cache_unfinished  # type: ignore[method-assign]

    scheduler.process_batch_result(batch)

    assert called["n"] == 1
    assert batch.reqs == []
    assert scheduler.chunked_req is req
    assert req.prefill_chunks_remaining == [8192]


def test_complete_request_purges_duplicate_scheduler_refs_by_rid():
    scheduler = _scheduler()
    req = _make_req("done_rid", num_chunks=1)
    req.output_ids = [1] * req.sampling_params.max_new_tokens
    duplicate = _make_req("done_rid", num_chunks=1)
    peer = _make_req("peer", num_chunks=1)

    scheduler.cache.finish_request = lambda _req: 0  # type: ignore[method-assign]
    scheduler.waiting_queue.append(duplicate)
    scheduler.waiting_queue.append(peer)
    scheduler.chunked_req = duplicate
    scheduler.running_batch.reqs = [duplicate, peer]
    scheduler.last_batch = ScheduleBatch(21, 0, ForwardMode.PREFILL, [duplicate, peer])
    scheduler._result_queue.append(
        ScheduleBatch(22, 0, ForwardMode.PREFILL, [duplicate, peer])
    )

    scheduler._complete_request(req)

    assert req.status == RequestStatus.COMPLETED
    assert [queued.rid for queued in scheduler.waiting_queue] == ["peer"]
    assert scheduler.chunked_req is None
    assert [queued.rid for queued in scheduler.running_batch.reqs] == ["peer"]
    assert [queued.rid for queued in scheduler.last_batch.reqs] == ["peer"]
    assert [queued.rid for queued in scheduler._result_queue[0].reqs] == ["peer"]


def test_complete_request_finalize_payload_uses_final_generated_tokens():
    scheduler = _scheduler()
    req = _make_req("done_payload", num_chunks=1)
    req.output_ids = [1] * req.sampling_params.max_new_tokens
    REQUEST_STATS[req.rid].generated_tokens = req.sampling_params.max_new_tokens - 1
    REQUEST_STATS[req.rid].last_event_time = 10.0

    sent: list[dict] = []
    scheduler.cache.finish_request = lambda _req: 0  # type: ignore[method-assign]

    class Sender:
        def send_pyobj(self, obj):
            sent.append(obj)

    scheduler.send_to_detokenizer = Sender()  # type: ignore[assignment]
    SimIpcRegistry.install_global(SimIpcRegistry())
    try:
        scheduler._complete_request(req)
    finally:
        SimIpcRegistry.install_global(None)

    assert sent
    payload = sent[-1]
    assert payload["op"] == "finalize"
    assert payload["generated_tokens"] == req.sampling_params.max_new_tokens
    assert payload["last_event_time"] > 10.0
    assert REQUEST_STATS[req.rid].generated_tokens == req.sampling_params.max_new_tokens


def test_chunked_refresh_keeps_inflight_prefill_progress():
    scheduler = _scheduler()
    req = _make_req("chunked_progress", num_chunks=3)
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.prefill_chunks_remaining = [8192, 8192]

    scheduler.cache.matched_prefix_length = lambda _req: 0  # type: ignore[method-assign]
    scheduler._refresh_req_next_round_input(req, adder=None)  # type: ignore[arg-type]

    assert req.kv_committed_length == 8192
    assert req.cached_length == 8192
    assert req.prefill_chunks_remaining == [8192, 8192]


def test_chunked_refresh_preserves_blocked_retry_chunk_split():
    scheduler = _scheduler(l2_capacity_pages=100_000)
    req = Req("chunked_retry", list(range(20_000)), SamplingParams(max_new_tokens=8))
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.prefill_chunks_remaining = [2435, 9373]

    scheduler.cache.matched_prefix_length = lambda _req: 0  # type: ignore[method-assign]
    scheduler._refresh_req_next_round_input(req, adder=None)  # type: ignore[arg-type]

    assert req.kv_committed_length == 8192
    assert req.cached_length == 8192
    assert req.prefill_chunks_remaining == [2435, 9373]


def test_load_back_sync_keeps_inflight_prefill_progress():
    scheduler = _scheduler()
    req = _make_req("chunked_load_back_progress", num_chunks=3)
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.prefill_chunks_remaining = [8192, 8192]

    scheduler.cache.device_resident_length = lambda _req: 0  # type: ignore[method-assign]
    adder = PrefillAdder(
        page_size=scheduler.cache.config.page_size,
        cache=scheduler.cache,
        running_batch=scheduler.running_batch,
        max_prefill_tokens=scheduler.sched_cfg.max_prefill_tokens,
        max_running_requests=scheduler.max_running_requests,
        rem_chunk_tokens=scheduler.sched_cfg.chunked_prefill_size,
        new_token_ratio=scheduler.new_token_ratio,
        chunked_prefill_size=scheduler.sched_cfg.chunked_prefill_size,
    )
    adder._sync_extend_after_load_back(req)

    assert req.kv_committed_length == 8192
    assert req.cached_length == 8192
    assert req.prefill_chunks_remaining == [8192, 8192]


def test_chunked_req_is_admitted_even_when_allocator_may_retract_later():
    scheduler = _scheduler()
    running = Req("decode", list(range(32)), SamplingParams(max_new_tokens=8))
    scheduler.running_batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [running])
    req = _make_req("chunked_no_page", num_chunks=2)
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.prefill_chunks_remaining = [8192]

    scheduler.cache.available_device_tokens = lambda: 0  # type: ignore[method-assign]
    scheduler.cache.evictable_size = lambda: 0  # type: ignore[method-assign]
    adder = PrefillAdder(
        page_size=scheduler.cache.config.page_size,
        cache=scheduler.cache,
        running_batch=scheduler.running_batch,
        max_prefill_tokens=scheduler.sched_cfg.max_prefill_tokens,
        max_running_requests=scheduler.max_running_requests,
        rem_chunk_tokens=scheduler.sched_cfg.chunked_prefill_size,
        new_token_ratio=scheduler.new_token_ratio,
        chunked_prefill_size=scheduler.sched_cfg.chunked_prefill_size,
    )

    assert adder.add_chunked_req(req) is None
    assert adder.can_run_list == [req]


def test_blocked_retry_keeps_active_prefix_lock_for_load_back():
    scheduler = _scheduler()
    req = _make_req("chunked_idle_no_page", num_chunks=2)
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.last_node_id = 7
    req.prefill_chunks_remaining = [scheduler.cache.config.page_size]
    req.prefill_blocked_retries = scheduler.PREFILL_BLOCKED_FLOOR_RETRY_LIMIT - 1
    scheduler.chunked_req = req

    calls: list[str] = []
    scheduler.cache.available_device_tokens = lambda: 0  # type: ignore[method-assign]
    scheduler.cache.evictable_size = lambda: 0  # type: ignore[method-assign]
    scheduler.cache.release_deferred_prefill_pages = lambda _req: calls.append(  # type: ignore[method-assign]
        "release_deferred"
    )
    scheduler.cache.release_prefill_lock = lambda _req: calls.append("release_lock")  # type: ignore[method-assign]

    scheduler.cache.sync_retract_prefix = lambda _req, *, chunk_size: calls.append(  # type: ignore[method-assign]
        f"sync:{chunk_size}"
    )
    scheduler.cache.evict_from_tree_cache = lambda _tokens, *, owner: calls.append(  # type: ignore[method-assign]
        owner
    )

    scheduler._handle_prefill_extend_blocked(
        req,
        extend_tokens=scheduler.cache.config.page_size,
        error=RuntimeError("insufficient L1 KV cache pages"),
        available_before=0,
        evictable_before=0,
    )

    assert "release_lock" not in calls
    assert f"sync:{scheduler.sched_cfg.chunked_prefill_size}" not in calls
    assert scheduler.chunked_req is req
    assert list(scheduler.waiting_queue) == [req]
    assert req.status == RequestStatus.WAITING
    assert req.prefill_blocked_retries == 0
    assert req.kv_committed_length == 8192
    assert req.cached_length == 8192


def test_blocked_floor_retry_does_not_retract_running_decode():
    scheduler = _scheduler()
    req = _make_req("chunked_with_decode_no_page", num_chunks=2)
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.last_node_id = 7
    req.prefill_chunks_remaining = [scheduler.cache.config.page_size]
    req.prefill_blocked_retries = scheduler.PREFILL_BLOCKED_FLOOR_RETRY_LIMIT - 1
    scheduler.chunked_req = req
    running = Req("decode", list(range(32)), SamplingParams(max_new_tokens=8))
    scheduler.running_batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [running])

    scheduler.cache.available_device_tokens = lambda: 0  # type: ignore[method-assign]
    scheduler.cache.evictable_size = lambda: 0  # type: ignore[method-assign]
    scheduler.cache.release_deferred_prefill_pages = lambda _req: None  # type: ignore[method-assign]
    scheduler.cache.release_prefill_lock = lambda _req: 0  # type: ignore[method-assign]
    scheduler.cache.sync_retract_prefix = lambda _req, *, chunk_size: None  # type: ignore[method-assign]
    scheduler.cache.evict_from_tree_cache = lambda _tokens, *, owner: None  # type: ignore[method-assign]
    scheduler._retract_decode_requests = lambda _batch: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("prefill blocked handler must not retract decode")
    )

    scheduler._handle_prefill_extend_blocked(
        req,
        extend_tokens=scheduler.cache.config.page_size,
        error=RuntimeError("insufficient L1 KV cache pages"),
        available_before=0,
        evictable_before=0,
    )

    assert scheduler.running_batch.reqs == [running]
    assert scheduler._prioritize_decode_after_retract
    assert scheduler.running_batch.batch_is_full


def test_blocked_active_chunk_runs_decode_before_retrying_prefill():
    scheduler = _scheduler()
    req = _make_req("chunked_wait_for_decode", num_chunks=2)
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.prefill_chunks_remaining = [8192]
    scheduler.chunked_req = req
    running = Req("decode_first", list(range(32)), SamplingParams(max_new_tokens=8))
    scheduler.running_batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [running])

    scheduler.cache.release_deferred_prefill_pages = lambda _req: None  # type: ignore[method-assign]
    scheduler.cache.release_prefill_lock = lambda _req: 0  # type: ignore[method-assign]
    scheduler.cache.sync_retract_prefix = lambda _req, *, chunk_size: None  # type: ignore[method-assign]
    scheduler.cache.evict_from_tree_cache = lambda _tokens, *, owner: None  # type: ignore[method-assign]

    scheduler._handle_prefill_extend_blocked(
        req,
        extend_tokens=8192,
        error=RuntimeError("insufficient L1 KV cache pages"),
        available_before=0,
        evictable_before=0,
    )

    calls: list[str] = []
    expected_batch = scheduler.running_batch
    scheduler.update_running_batch = lambda batch: calls.append("decode") or batch  # type: ignore[method-assign]
    scheduler.get_new_batch_prefill = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("prefill retry must wait for decode/retract")
    )

    assert scheduler.get_next_batch_to_run() is expected_batch
    assert calls == ["decode"]


def test_blocked_handler_preserves_deferred_chunk_progress():
    scheduler = _scheduler()
    req = _make_req("blocked_preserve_deferred", num_chunks=2)
    scheduler.cache.prepare_prefill(req)
    scheduler.cache.extend_prefill(req, scheduler.sched_cfg.chunked_prefill_size)
    req.prefill_chunks_remaining = [scheduler.sched_cfg.chunked_prefill_size]
    scheduler.chunked_req = req

    scheduler._handle_prefill_extend_blocked(
        req,
        extend_tokens=scheduler.sched_cfg.chunked_prefill_size,
        error=RuntimeError("insufficient L1 KV cache pages"),
        available_before=0,
        evictable_before=0,
    )

    assert req.prefill_alloc_page_ids == []
    assert req.cached_length == scheduler.sched_cfg.chunked_prefill_size
    assert req.kv_committed_length == scheduler.sched_cfg.chunked_prefill_size
    assert req.prefill_chunks_remaining == [4096, 4096]
    assert scheduler.chunked_req is req


def test_chunked_req_excluded_from_last_batch_merge():
    scheduler = _scheduler()
    chunked = _make_req("chunked", num_chunks=2)
    decode_peer = Req(
        "peer",
        list(range(32)),
        SamplingParams(max_new_tokens=4),
    )
    scheduler.chunked_req = chunked
    scheduler.last_batch = ScheduleBatch(
        1,
        0,
        ForwardMode.PREFILL,
        [chunked, decode_peer],
    )
    scheduler.running_batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [])
    scheduler.get_next_batch_to_run()
    assert chunked.rid not in [r.rid for r in scheduler.running_batch.reqs]
    assert decode_peer.rid in [r.rid for r in scheduler.running_batch.reqs]


def test_chunked_reqs_not_dropped_when_second_starts_later():
    """With a single chunked_req pointer, extra long requests wait in
    waiting_queue until the current chunked prefill finishes; none are dropped."""
    scheduler = _scheduler()
    rids = [f"r{i}" for i in range(4)]
    for rid in rids:
        _enqueue(scheduler, rid, num_chunks=2)

    _drain(scheduler)

    assert not scheduler.waiting_queue
    assert scheduler.chunked_req is None
    assert scheduler.running_batch.is_empty()
    for rid in rids:
        assert REQUEST_STATS[rid].status == "completed", rid


def test_prefill_blocked_retry_splits_chunk():
    scheduler = _scheduler()
    req = _make_req("r_split", num_chunks=1)
    batch = ScheduleBatch(9, 0, ForwardMode.PREFILL, [req])
    batch.inference_dur_s = 0.01

    called = {"n": 0}
    original_extend = scheduler.cache.extend_prefill

    def _fail_once(target_req, extend_tokens):
        called["n"] += 1
        if called["n"] == 1:
            raise RuntimeError("synthetic oom")
        return original_extend(target_req, extend_tokens)

    scheduler.cache.extend_prefill = _fail_once  # type: ignore[method-assign]
    scheduler.process_batch_result(batch)

    assert called["n"] == 1
    assert batch.reqs == []
    assert scheduler.waiting_queue
    requeued = scheduler.chunked_req
    assert requeued is not None
    assert requeued.rid == "r_split"
    assert requeued.prefill_chunks_remaining[0] == 4096
    assert requeued.prefill_chunks_remaining[1] == 4096


def test_blocked_active_chunked_req_is_requeued_once():
    scheduler = _scheduler()
    req = _make_req("blocked_active", num_chunks=1)
    scheduler.chunked_req = req
    scheduler.waiting_queue.append(req)

    scheduler._handle_prefill_extend_blocked(req, extend_tokens=8192)

    assert scheduler.chunked_req is req
    assert sum(1 for queued in scheduler.waiting_queue if queued is req) == 1
    assert req.prefill_chunks_remaining == [4096, 4096]


def test_blocked_retry_evicts_extend_with_allocator_page_overhead():
    scheduler = _scheduler()
    req = _make_req("blocked_allocator_overhead", num_chunks=1)
    req.prefill_chunks_remaining = [8192]
    scheduler.chunked_req = req

    calls: list[tuple[int, str]] = []
    scheduler.cache.release_deferred_prefill_pages = lambda _req: None  # type: ignore[method-assign]
    scheduler.cache.release_prefill_lock = lambda _req: 0  # type: ignore[method-assign]
    scheduler.cache.sync_retract_prefix = lambda _req, *, chunk_size: None  # type: ignore[method-assign]
    scheduler.cache.evict_from_tree_cache = lambda tokens, *, owner: calls.append(  # type: ignore[method-assign]
        (tokens, owner)
    )

    scheduler._handle_prefill_extend_blocked(req, extend_tokens=8192)

    assert calls == [
        (8192 + scheduler.cache.config.page_size, f"prefill-retry-{req.rid}")
    ]


def test_blocked_retry_reserves_one_decode_step_before_readmission():
    scheduler = _scheduler()
    req = _make_req("blocked_decode_reserve", num_chunks=1)
    req.prefill_chunks_remaining = [8192]
    scheduler.chunked_req = req
    scheduler.running_batch = ScheduleBatch(
        0,
        0,
        ForwardMode.DECODE,
        [
            Req(f"decode-{idx}", list(range(32)), SamplingParams(max_new_tokens=8))
            for idx in range(3)
        ],
    )

    calls: list[tuple[int, str]] = []
    scheduler.cache.release_deferred_prefill_pages = lambda _req: None  # type: ignore[method-assign]
    scheduler.cache.release_prefill_lock = lambda _req: 0  # type: ignore[method-assign]
    scheduler.cache.sync_retract_prefix = lambda _req, *, chunk_size: None  # type: ignore[method-assign]
    scheduler.cache.evict_from_tree_cache = lambda tokens, *, owner: calls.append(  # type: ignore[method-assign]
        (tokens, owner)
    )

    scheduler._handle_prefill_extend_blocked(req, extend_tokens=8192)

    page = scheduler.cache.config.page_size
    assert calls == [(8192 + page + 3 * page, f"prefill-retry-{req.rid}")]
    assert scheduler._prioritize_decode_after_retract


def test_blocked_final_chunk_keeps_active_prefix_lock_for_readmission():
    scheduler = _scheduler()
    req = Req("blocked_tail", list(range(12_082)), SamplingParams(max_new_tokens=8))
    req.kv_committed_length = 11_807
    req.cached_length = 11_807
    req.cache_protected_len = 11_807
    req.last_node_id = 123
    req.prefill_chunks_remaining = [275]
    released = {"lock": 0, "sync": 0}

    def _release_lock(_req):
        released["lock"] += 1

    def _sync(_req, *, chunk_size):
        released["sync"] += 1
        _req.cache_protected_len = 0
        _req.last_node_id = None

    scheduler.cache.release_prefill_lock = _release_lock  # type: ignore[method-assign]
    scheduler.cache.sync_retract_prefix = _sync  # type: ignore[method-assign]

    scheduler._handle_prefill_extend_blocked(req, extend_tokens=275)

    assert released == {"lock": 0, "sync": 0}
    assert req.kv_committed_length == 11_807
    assert req.cached_length == 11_807
    assert req.cache_protected_len == 11_807
    assert req.last_node_id == 123
    assert req.prefill_blocked_retries == 1
    assert scheduler.chunked_req is req
    assert list(scheduler.waiting_queue) == [req]


def test_blocked_middle_chunk_keeps_active_prefix_lock_for_readmission():
    scheduler = _scheduler()
    req = Req("blocked_mid", list(range(24_576)), SamplingParams(max_new_tokens=8))
    req.kv_committed_length = 16_384
    req.cached_length = 16_384
    req.cache_protected_len = 16_384
    req.last_node_id = 456
    req.prefill_chunks_remaining = [8192, 8192]
    released = {"lock": 0, "sync": 0}

    def _release_lock(_req):
        released["lock"] += 1

    def _sync(_req, *, chunk_size):
        released["sync"] += 1
        _req.cache_protected_len = 0
        _req.last_node_id = None

    scheduler.cache.release_prefill_lock = _release_lock  # type: ignore[method-assign]
    scheduler.cache.sync_retract_prefix = _sync  # type: ignore[method-assign]

    scheduler._handle_prefill_extend_blocked(req, extend_tokens=8192)

    assert released == {"lock": 0, "sync": 0}
    assert req.kv_committed_length == 16_384
    assert req.cached_length == 16_384
    assert req.cache_protected_len == 16_384
    assert req.last_node_id == 456
    assert req.prefill_chunks_remaining == [4096, 4096, 8192]
    assert scheduler.chunked_req is req
    assert list(scheduler.waiting_queue) == [req]


def test_active_chunked_req_duplicate_is_not_admitted_twice():
    scheduler = _scheduler()
    req = _make_req("active_duplicate", num_chunks=2)
    scheduler.chunked_req = req
    scheduler.waiting_queue.append(req)

    batch = scheduler.get_next_batch_to_run()

    assert batch is not None
    assert batch.reqs.count(req) == 1
    assert all(queued is not req for queued in scheduler.waiting_queue)


def test_run_batch_prefill_alloc_blocked_does_not_enter_forward():
    scheduler = _scheduler()
    req = _make_req("r_alloc_block", num_chunks=1)
    batch = ScheduleBatch(11, 0, ForwardMode.PREFILL, [req])
    called = {"predict": 0}

    def _predict(_pred_batch):
        called["predict"] += 1
        return 0.01

    scheduler.predictor.predict_infer_time = _predict  # type: ignore[method-assign]

    def _always_fail(*_args, **_kwargs):
        raise RuntimeError("synthetic alloc oom")

    scheduler.cache.extend_prefill = _always_fail  # type: ignore[method-assign]
    scheduler.run_batch(batch)

    assert called["predict"] == 0
    assert batch.reqs == []
    assert scheduler.chunked_req is req
    assert scheduler.waiting_queue
    assert scheduler.waiting_queue[-1].rid == "r_alloc_block"


def test_prefill_blocked_floor_retry_resets_counter():
    scheduler = _scheduler()
    rid = "r_floor"
    req = Req(rid, list(range(16)), SamplingParams(max_new_tokens=4))
    req.prefill_chunks_remaining = [16]
    REQUEST_STATS[rid] = RequestStats(rid=rid)
    REQUEST_STATS[rid].input_length = 16

    def _always_fail(*_args, **_kwargs):
        raise RuntimeError("synthetic floor oom")

    scheduler.cache.extend_prefill = _always_fail  # type: ignore[method-assign]
    for _ in range(scheduler.PREFILL_BLOCKED_FLOOR_RETRY_LIMIT):
        batch = ScheduleBatch(12, 0, ForwardMode.PREFILL, [req])
        batch.inference_dur_s = 0.01
        scheduler.process_batch_result(batch)
        assert batch.reqs == []
        assert scheduler.chunked_req is req
        assert scheduler.waiting_queue
        req = scheduler.waiting_queue.pop()
        scheduler.chunked_req = None

    assert req.prefill_blocked_retries == 0


def test_final_chunk_rollback_requeues_from_device_prefix():
    scheduler = _scheduler()
    rid = "rollback"
    req = _make_req(rid, num_chunks=2)
    req.status = RequestStatus.PREFILL
    req.kv_committed_length = 8192
    req.cached_length = 8192
    req.cache_protected_len = 8192
    req.last_node_id = 10
    req.prefill_chunks_remaining = []
    batch = ScheduleBatch(13, 0, ForwardMode.PREFILL, [req])
    batch.inference_dur_s = 0.01
    batch.metadata["prefill_extend_tokens"] = {rid: 4096}
    batch.metadata["prefill_chunks_consumed_in_run_batch"] = True

    def _rollback(target_req, *, chunked=False):
        assert target_req is req
        assert chunked is False
        target_req.kv_committed_length = 4096
        target_req.cached_length = 4096
        target_req.cache_protected_len = 4096
        target_req.last_node_id = None
        target_req.prefill_alloc_page_ids.clear()

    scheduler.cache.cache_unfinished_req = _rollback  # type: ignore[method-assign]

    scheduler.process_batch_result(batch)

    assert batch.reqs == []
    assert scheduler.chunked_req is req
    assert req.kv_committed_length == 0
    assert req.cached_length == 0
    assert req.cache_protected_len == 0
    assert req.last_node_id is None
    assert req.prefill_chunks_remaining == [8192, 8192]
    assert req.remaining_decode_tokens > 0
