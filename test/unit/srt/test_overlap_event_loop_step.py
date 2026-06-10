"""Overlap event-loop step must match SGLang queue/last_batch semantics."""

from sglsim.srt.managers.schedule_batch import ForwardMode, Req, RequestStatus, SamplingParams
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.types import RequestStats
from sglsim.srt.utils.network import SimIpcRegistry


def _overlap_scheduler() -> Scheduler:
    SimIpcRegistry.install_global(None)
    ConfigManager._raw = {
        "scheduler": {
            "max_prefill_tokens": 8192,
            "chunked_prefill_size": 8192,
        },
        "cache": {"page_size": 16, "l1_capacity_pages": 1000, "l2_capacity_pages": 0},
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    server_args = ServerArgs(disable_overlap_schedule=False)
    sched = Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)
    sched.enable_overlap = True
    return sched


def test_overlap_step_clears_last_batch_when_no_forward_batch():
    scheduler = _overlap_scheduler()
    scheduler.last_batch = scheduler._make_batch(ForwardMode.PREFILL, [], 0.0)
    scheduler.running_batch.reqs.clear()
    scheduler.waiting_queue.clear()
    scheduler.chunked_req = None

    progressed = scheduler._event_loop_step_overlap()

    assert progressed is False
    assert scheduler.last_batch is None
    assert scheduler._rank_idle() is True


def test_prefill_result_drained_before_decode_schedule():
    scheduler = _overlap_scheduler()
    processed: list[ForwardMode] = []

    req = Req("r0", list(range(32)), SamplingParams(max_new_tokens=4))
    REQUEST_STATS["r0"] = RequestStats(rid="r0", input_length=32, output_length=4)
    prefill = scheduler._make_batch(ForwardMode.PREFILL, [req], 0.0)
    decode = scheduler._make_batch(ForwardMode.DECODE, [req], 0.0)

    scheduler._result_queue.append(prefill)
    scheduler.last_batch = prefill
    scheduler.run_batch = lambda batch: None  # type: ignore[method-assign]
    scheduler.process_batch_result = lambda batch: processed.append(  # type: ignore[method-assign]
        batch.mode
    )
    scheduler.get_next_batch_to_run = lambda: decode  # type: ignore[method-assign]

    assert scheduler._event_loop_step_overlap() is True
    assert processed[0] == ForwardMode.PREFILL
    assert scheduler.last_batch is decode


def test_prefill_chunk_state_updates_before_last_batch_merge():
    scheduler = _overlap_scheduler()
    req = Req("r_prefill", list(range(32)), SamplingParams(max_new_tokens=2))
    req.prefill_chunks_remaining = [32]
    REQUEST_STATS.clear()
    REQUEST_STATS[req.rid] = RequestStats(
        rid=req.rid,
        input_length=32,
        output_length=2,
    )
    scheduler.predictor.predict_infer_time = lambda _batch: 0.01  # type: ignore[method-assign]

    prefill = scheduler._make_batch(ForwardMode.PREFILL, [req], 0.0)
    scheduler.run_batch(prefill)

    assert req.prefill_chunks_remaining == []

    scheduler.last_batch = prefill
    next_batch = scheduler.get_next_batch_to_run()

    assert next_batch is not None
    assert next_batch.mode == ForwardMode.DECODE
    assert [r.rid for r in next_batch.reqs] == [req.rid]


def test_overlap_result_queue_snapshots_chunked_prefill_before_filter():
    scheduler = _overlap_scheduler()
    req = Req("r_chunked", list(range(10_000)), SamplingParams(max_new_tokens=2))
    req.prefill_chunks_remaining = [8192, 1808]
    REQUEST_STATS.clear()
    REQUEST_STATS[req.rid] = RequestStats(
        rid=req.rid,
        input_length=10_000,
        output_length=2,
    )
    scheduler.predictor.predict_infer_time = lambda _batch: 0.01  # type: ignore[method-assign]

    prefill = scheduler._make_batch(ForwardMode.PREFILL, [req], 0.0)
    scheduler.run_batch(prefill)
    scheduler.chunked_req = req
    scheduler._result_queue.append(prefill.copy())
    scheduler.last_batch = prefill

    # SGLang calls get_next_batch_to_run before processing the prior overlap
    # result. This filters last_batch in-place, so the result queue must hold a
    # snapshot rather than the same ScheduleBatch object.
    next_batch = scheduler.get_next_batch_to_run()

    assert next_batch is not None
    assert [r.rid for r in next_batch.reqs] == [req.rid]
    assert prefill.reqs == []
    assert [r.rid for r in scheduler._result_queue[0].reqs] == [req.rid]

    scheduler._drain_one_result()

    assert scheduler.chunked_req is req
    assert req.prefill_chunks_remaining == [1808]


def test_overlap_filter_keeps_nonexcluded_inflight_prefill_reqs():
    scheduler = _overlap_scheduler()
    chunked = Req("r_chunked", list(range(10_000)), SamplingParams(max_new_tokens=2))
    chunked.prefill_chunks_remaining = [1808]
    peer = Req("r_peer", list(range(2048)), SamplingParams(max_new_tokens=2))
    peer.prefill_chunks_remaining = [1024]

    prefill = scheduler._make_batch(ForwardMode.PREFILL, [chunked, peer], 0.0)
    scheduler.chunked_req = chunked
    scheduler.last_batch = prefill
    scheduler._result_queue.append(prefill.copy())

    next_batch = scheduler.get_next_batch_to_run()

    assert next_batch is not None
    assert [req.rid for req in next_batch.reqs] == [chunked.rid]
    assert [req.rid for req in prefill.reqs] == [peer.rid]
    assert [req.rid for req in scheduler._result_queue[0].reqs] == [
        chunked.rid,
        peer.rid,
    ]


def test_disable_overlap_batch_processes_previous_before_run():
    scheduler = _overlap_scheduler()
    scheduler._disable_consecutive_prefill_overlap = True
    events: list[str] = []

    req = Req("r1", list(range(16)), SamplingParams(max_new_tokens=2))
    REQUEST_STATS["r1"] = RequestStats(rid="r1", input_length=16, output_length=2)
    prev_prefill = scheduler._make_batch(ForwardMode.PREFILL, [req], 0.0)
    next_prefill = scheduler._make_batch(ForwardMode.PREFILL, [req], 0.0)

    scheduler._result_queue.append(prev_prefill)
    scheduler.last_batch = prev_prefill
    scheduler.get_next_batch_to_run = lambda: next_prefill  # type: ignore[method-assign]
    scheduler.process_batch_result = lambda batch: events.append(  # type: ignore[method-assign]
        f"process:{batch.batch_id}"
    )
    scheduler.run_batch = lambda batch: events.append(  # type: ignore[method-assign]
        f"run:{batch.batch_id}"
    )

    assert scheduler._event_loop_step_overlap() is True
    assert events[:2] == [f"process:{prev_prefill.batch_id}", f"run:{next_prefill.batch_id}"]
    assert scheduler.last_batch is next_prefill
    assert list(scheduler._result_queue) == [next_prefill]


def test_disable_overlap_for_batch_includes_spec_grammar_decode_guard():
    scheduler = _overlap_scheduler()
    req = Req("r2", list(range(8)), SamplingParams(max_new_tokens=2))
    decode = scheduler._make_batch(ForwardMode.DECODE, [req], 0.0)
    decode.is_spec_v2 = True  # type: ignore[attr-defined]
    decode.has_grammar = True  # type: ignore[attr-defined]

    scheduler._result_queue.append(
        scheduler._make_batch(ForwardMode.PREFILL, [req], 0.0)
    )

    assert scheduler.is_disable_overlap_for_batch(decode) is True
