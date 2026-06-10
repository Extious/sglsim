import logging
from collections import deque
from unittest.mock import MagicMock

from sglsim.srt.managers.schedule_batch import ForwardMode, Req, RequestStatus, SamplingParams, ScheduleBatch
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.types import RequestStats


class _CacheStub:
    def __init__(self, *, available_tokens: int = 1000) -> None:
        self.released: list[str] = []
        self.prepared: list[str] = []
        self.flush_calls = 0
        self.init_load_back_mem_quotas: list[int | None] = []
        self._available_tokens = available_tokens
        self.config = type("Cfg", (), {"page_size": 16})()
        self.hierarchical_enabled = False
        mem_pool = type("MemPool", (), {"available_size": available_tokens // 16})()
        self._cache = type("InnerCache", (), {"mem_pool_device": mem_pool})()

    def device_resident_length(self, req: Req) -> int:
        return req.prefill_length

    def matched_prefix_length(self, req: Req) -> int:
        return req.cached_length

    def release_request(self, req: Req) -> int:
        self.released.append(req.rid)
        return 940

    def prepare_prefill(self, req: Req) -> None:
        self.prepared.append(req.rid)
        req.last_node_id = 9

    def release_deferred_prefill_pages(self, req: Req) -> None:
        req.prefill_alloc_page_ids.clear()

    def release_prefill_lock(self, req: Req) -> None:
        req.last_node_id = None

    def sync_retract_prefix(self, req: Req, *, chunk_size: int | None) -> None:
        fill_len = len(req.input_ids) + len(req.output_ids)
        req.cached_length = fill_len
        req.kv_committed_length = fill_len
        req.last_node_id = None
        req.cache_protected_len = 0
        req.prefill_chunks_remaining = []

    def available_and_evictable_tokens(self) -> int:
        return self._available_tokens

    def available_device_tokens(self) -> int:
        return self._available_tokens

    def evictable_size(self) -> int:
        return 0

    def finish_request(self, req: Req) -> None:
        self.release_request(req)

    def check_hicache_events(self, *, remain_dur: float = 0.0) -> dict[str, int]:
        return {}

    def consume_backup_overlap(self, forward: float) -> None:
        return None

    def consume_backup_serial(self) -> None:
        return None

    def evict_from_tree_cache(self, num_tokens: int, *, owner: str = "evict") -> None:
        return None

    def extend_decode(self, req: Req, extend_tokens: int = 1) -> None:
        req.kv_committed_length += extend_tokens

    def check_prefetch_progress(self, req: Req) -> bool:
        return True

    def init_load_back(self, req: Req, *, mem_quota: int | None = None) -> int:
        self.init_load_back_mem_quotas.append(mem_quota)
        return 0

    def ready_to_load_host_cache(self) -> int:
        return 0

    def flush_write_through_acks(self) -> int:
        self.flush_calls += 1
        return 0

    def kv_token_usage_details(self) -> tuple[int, int, int, int]:
        total = 1000
        used = total - self._available_tokens
        return used, total, 0, 0


def _make_scheduler(*, cache: _CacheStub | None = None) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.running_batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [])
    scheduler.waiting_queue = deque()
    scheduler.chunked_req = None
    scheduler.last_batch = None
    scheduler._result_queue = deque()
    scheduler._req_registry = {}
    scheduler.cache = cache or _CacheStub()
    scheduler.logger = logging.getLogger("test_scheduler_retraction")
    scheduler.sched_cfg = type(
        "Cfg",
        (),
        {"chunked_prefill_size": 8192, "init_new_token_ratio": 0.7},
    )()
    scheduler.init_new_token_ratio = 0.7
    scheduler.min_new_token_ratio = 0.098
    scheduler.new_token_ratio_decay = (0.7 - 0.098) / 600.0
    scheduler.new_token_ratio = 0.28
    scheduler.max_running_requests = 4
    scheduler.RETRACT_DECODE_STEPS = 20
    scheduler.server_args = ServerArgs()
    scheduler.send_to_detokenizer = MagicMock()
    scheduler._prioritize_decode_after_retract = False
    scheduler._blocked_chunked_waiting_for_decode = False
    return scheduler


def test_retract_decode_moves_request_to_waiting_and_rebuilds_prefill(caplog):
    scheduler = _make_scheduler(cache=_CacheStub(available_tokens=0))
    # sglang retract_decode evicts the least-progressed request first (fewest
    # decode tokens), so r0 (1 output token) is retracted before keep (2).
    req = Req("r0", [1] * 24096, SamplingParams(max_new_tokens=1024))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    req.kv_committed_length = 24096
    req.cached_length = 24096
    req.cache_protected_len = 24096
    req.last_node_id = 7
    keep = Req("r1", [1] * 24096, SamplingParams(max_new_tokens=1024))
    keep.status = RequestStatus.DECODE
    keep.output_ids = [0, 0]
    keep.kv_committed_length = 24096
    keep.cached_length = 24096
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req, keep])
    scheduler.running_batch = batch

    with caplog.at_level(logging.INFO, logger="test_scheduler_retraction"):
        gained = scheduler._retract_decode_requests(batch)

    assert gained >= 0
    assert scheduler.cache.released == ["r0"]
    assert batch.reqs == [keep]
    assert len(scheduler.waiting_queue) == 1
    requeued = scheduler.waiting_queue[0]
    assert requeued.is_retracted is True
    assert requeued.status == RequestStatus.WAITING
    assert requeued.output_ids == [0]
    assert requeued.kv_committed_length == 24097
    assert requeued.cached_length == 24097
    assert requeued.prefill_chunks_remaining == []
    assert requeued.last_node_id is None
    assert "KV cache pool is full. Retract requests. #retracted_reqs: 1" in caplog.text
    assert scheduler._prioritize_decode_after_retract is True
    expected_ratio = min(1.0, (2 + scheduler.RETRACT_DECODE_STEPS) / (1024 + 1))
    assert scheduler.new_token_ratio == expected_ratio
    assert scheduler.new_token_ratio < 0.28 - scheduler.new_token_ratio_decay


def test_reset_for_retract_preserves_request_cache_hit_accounting():
    scheduler = _make_scheduler()
    req = Req("r_cached", [1] * 256, SamplingParams(max_new_tokens=16))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    req.kv_committed_length = req.prefill_length
    req.cached_length = req.prefill_length
    req.cache_hit_tokens = 128
    req.cache_hit_recorded = True
    req.cache_hit_log_pending = 128

    scheduler._reset_req_for_retract(req)

    assert req.cache_hit_tokens == 128
    assert req.cache_hit_recorded is True
    assert req.cache_hit_log_pending == 0


def test_prefill_admission_records_load_back_from_first_breakdown_only():
    scheduler = _make_scheduler()
    req = Req("r_host", [1] * 100, SamplingParams(max_new_tokens=16))
    req.cache_hit_tokens = 80
    REQUEST_STATS.clear()
    REQUEST_STATS[req.rid] = RequestStats(rid=req.rid, input_length=100)

    req.admit_load_back_tokens = 40
    scheduler._record_prefill_admission_stats(req, ts=10.0)
    req.admit_load_back_tokens = 70
    scheduler._record_prefill_admission_stats(req, ts=11.0)
    req.admit_load_back_tokens = 120
    scheduler._record_prefill_admission_stats(req, ts=12.0)

    stats = REQUEST_STATS[req.rid]
    assert stats.queue_end == 10.0
    assert req.queue_end_time == 10.0
    assert stats.cache_hit_tokens == 80
    assert stats.load_back_tokens == 40


def test_retract_raises_new_token_ratio_when_kept_batch_has_progress():
    scheduler = _make_scheduler(cache=_CacheStub(available_tokens=0))
    req = Req("r0", [1] * 24096, SamplingParams(max_new_tokens=1024))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    keep = Req("r1", [1] * 24096, SamplingParams(max_new_tokens=1024))
    keep.status = RequestStatus.DECODE
    keep.output_ids = [0] * 900
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req, keep])
    scheduler.running_batch = batch
    old_ratio = scheduler.new_token_ratio

    scheduler._retract_decode_requests(batch)

    expected = min(
        1.0,
        (900 + scheduler.RETRACT_DECODE_STEPS * 1) / (1024 + 1),
    )
    assert scheduler.new_token_ratio == expected
    assert scheduler.new_token_ratio > old_ratio


def test_single_request_oom_aborts_instead_of_requeue(caplog):
    scheduler = _make_scheduler(cache=_CacheStub(available_tokens=0))
    req = Req("r0", [1] * 24096, SamplingParams(max_new_tokens=1024))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    req.kv_committed_length = 24096
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req])
    scheduler.running_batch = batch
    REQUEST_STATS.clear()
    REQUEST_STATS["r0"] = RequestStats(rid="r0")

    with caplog.at_level(logging.WARNING, logger="test_scheduler_retraction"):
        gained = scheduler._retract_decode_requests(batch)

    assert gained >= 0
    assert batch.reqs == []
    assert scheduler.waiting_queue == deque()
    assert scheduler.cache.released == ["r0"]
    assert req.status == RequestStatus.ABORTED
    assert REQUEST_STATS["r0"].status == "aborted"
    assert scheduler.send_to_detokenizer.send_pyobj.called
    assert "aborted last request r0 due to OOM" in caplog.text


def test_single_request_oom_requeues_when_chunked_prefill_is_active():
    scheduler = _make_scheduler(cache=_CacheStub(available_tokens=0))
    req = Req("r0", [1] * 24096, SamplingParams(max_new_tokens=1024))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    req.kv_committed_length = 24096
    chunked = Req("chunked", [2] * 4096, SamplingParams(max_new_tokens=8))
    chunked.prefill_chunks_remaining = [16]
    scheduler.chunked_req = chunked
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req])
    scheduler.running_batch = batch

    gained = scheduler._retract_decode_requests(batch)

    assert gained >= 0
    assert batch.reqs == []
    assert list(scheduler.waiting_queue) == [req]
    assert scheduler.cache.released == ["r0"]
    assert req.status == RequestStatus.WAITING
    assert scheduler.chunked_req is chunked
    assert not scheduler.send_to_detokenizer.send_pyobj.called


def test_prioritize_decode_after_retract_skips_prefill():
    scheduler = _make_scheduler()
    req = Req("r0", [1] * 100, SamplingParams(max_new_tokens=256))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    req.kv_committed_length = 100
    req.cached_length = 100
    scheduler.running_batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req])
    scheduler.waiting_queue = deque(
        [Req("w1", [1] * 50, SamplingParams(max_new_tokens=256))]
    )
    scheduler._prioritize_decode_after_retract = True
    scheduler.cache = _CacheStub(available_tokens=1000)
    scheduler.last_batch = None
    scheduler.chunked_req = None

    batch = scheduler.get_next_batch_to_run()

    assert batch is not None
    assert batch.mode == ForwardMode.DECODE
    assert scheduler._prioritize_decode_after_retract is False


def test_blocked_chunked_retry_yields_to_decode_before_prefill():
    scheduler = _make_scheduler()
    running = Req("decode", [1] * 100, SamplingParams(max_new_tokens=256))
    running.status = RequestStatus.DECODE
    running.output_ids = [0]
    running.kv_committed_length = 100
    running.cached_length = 100
    scheduler.running_batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [running])

    chunked = Req("chunked", [1] * 20_000, SamplingParams(max_new_tokens=256))
    chunked.status = RequestStatus.WAITING
    chunked.cached_length = 8192
    chunked.kv_committed_length = 8192
    chunked.prefill_chunks_remaining = [8192]
    scheduler.chunked_req = chunked
    scheduler.waiting_queue = deque([chunked])
    scheduler._blocked_chunked_waiting_for_decode = True

    batch = scheduler.get_next_batch_to_run()

    assert batch is not None
    assert batch.mode == ForwardMode.DECODE
    assert scheduler._blocked_chunked_waiting_for_decode is False


def test_idle_check_resets_new_token_ratio():
    scheduler = _make_scheduler()
    scheduler.running_batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [])
    scheduler.chunked_req = None
    scheduler.new_token_ratio = 0.04

    scheduler._self_check_during_idle()

    assert scheduler.new_token_ratio == scheduler.init_new_token_ratio


def test_retracted_request_with_pending_prefill_is_not_promoted():
    scheduler = _make_scheduler()
    req = Req("r1", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    req.status = RequestStatus.WAITING
    req.is_retracted = True
    req.output_ids = [0]
    req.cached_length = 0
    req.kv_committed_length = 0
    req.prefill_chunks_remaining = [4]
    scheduler.running_batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [])
    scheduler.waiting_queue = deque([req])

    assert scheduler._promote_decode_ready_waiting() is False

    assert list(scheduler.waiting_queue) == [req]
    assert scheduler.running_batch.reqs == []
    assert scheduler.cache.prepared == []


def test_retracted_decode_ready_request_is_promoted():
    scheduler = _make_scheduler()
    req = Req("r1", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    req.status = RequestStatus.WAITING
    req.is_retracted = True
    req.output_ids = [0]
    req.cached_length = req.prefill_length
    req.kv_committed_length = req.prefill_length
    req.prefill_chunks_remaining = []
    scheduler.running_batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [])
    scheduler.waiting_queue = deque([req])

    assert scheduler._promote_decode_ready_waiting() is True

    assert list(scheduler.waiting_queue) == []
    assert scheduler.running_batch.reqs == [req]
    assert req.is_retracted is False
    assert req.status == RequestStatus.DECODE


def test_retracted_decode_ready_readmitted_when_kv_saturated_no_stall():
    """A retracted decode-ready victim must re-enter running even when KV is
    saturated (no cooldown), so running never stays empty -> no stall deadlock.
    """
    scheduler = _make_scheduler(cache=_CacheStub(available_tokens=100))
    req = Req("r1", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    req.status = RequestStatus.WAITING
    req.is_retracted = True
    req.output_ids = [0]
    req.cached_length = req.prefill_length
    req.kv_committed_length = req.prefill_length
    req.prefill_chunks_remaining = []
    scheduler.running_batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [])
    scheduler.waiting_queue = deque([req])

    assert scheduler._promote_decode_ready_waiting() is True

    assert list(scheduler.waiting_queue) == []
    assert scheduler.running_batch.reqs == [req]
    assert req.is_retracted is False
    assert req.status == RequestStatus.DECODE


def test_decode_ready_promotion_prepares_prefill_when_last_node_missing():
    scheduler = _make_scheduler()
    req = Req("r1", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    req.status = RequestStatus.WAITING
    req.output_ids = [0]
    req.cached_length = req.prefill_length
    req.kv_committed_length = req.prefill_length
    req.last_node_id = None
    scheduler.running_batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [])
    scheduler.waiting_queue = deque([req])

    assert scheduler._promote_decode_ready_waiting() is True

    assert list(scheduler.waiting_queue) == []
    assert scheduler.running_batch.reqs == [req]
    assert req.status == RequestStatus.DECODE
    assert req.last_node_id == 9
    assert scheduler.cache.prepared == ["r1"]


def test_decode_alloc_block_retracts_before_requeue():
    class _BlockingDecodeCache(_CacheStub):
        def extend_decode(self, req: Req, extend_tokens: int = 1) -> None:
            raise RuntimeError("decode alloc blocked")

    cache = _BlockingDecodeCache(available_tokens=100)
    scheduler = _make_scheduler(cache=cache)
    req = Req("r_block", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    req.kv_committed_length = 4
    req.cached_length = 4
    req.cache_protected_len = 4
    req.last_node_id = 7
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req])

    updated = scheduler.update_running_batch(batch)

    assert updated is None
    assert cache.released == ["r_block"]
    assert list(scheduler.waiting_queue) == [req]
    assert req.status == RequestStatus.WAITING
    assert req.is_retracted is True
    assert req.last_node_id is None
    assert req.cache_protected_len == 0


def test_prefill_alloc_block_requeues_and_reopens_admission():
    scheduler = _make_scheduler()
    req = Req("p_block", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    req.status = RequestStatus.PREFILL
    req.prefill_chunks_remaining = [4]
    scheduler.running_batch.batch_is_full = True

    scheduler._handle_prefill_extend_blocked(
        req,
        extend_tokens=4,
        error=RuntimeError("prefill alloc blocked"),
        available_before=0,
        evictable_before=0,
    )

    assert list(scheduler.waiting_queue) == [req]
    assert req.status == RequestStatus.WAITING
    assert scheduler.running_batch.batch_is_full is False


def test_idle_pending_request_is_recovered_to_waiting_queue():
    scheduler = _make_scheduler()
    req = Req("lost", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    req.kv_committed_length = req.prefill_length
    scheduler._req_registry[req.rid] = req
    REQUEST_STATS.clear()
    REQUEST_STATS[req.rid] = RequestStats(rid=req.rid)

    assert scheduler._recover_idle_pending_requests() is True

    assert list(scheduler.waiting_queue) == [req]
    assert req.status == RequestStatus.WAITING
    assert req.is_retracted is True
    assert scheduler.cache.released == ["lost"]


def test_idle_pending_length_done_request_completes():
    scheduler = _make_scheduler()
    req = Req("lost_done", [1, 2, 3, 4], SamplingParams(max_new_tokens=1))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    scheduler._req_registry[req.rid] = req
    REQUEST_STATS.clear()
    REQUEST_STATS[req.rid] = RequestStats(rid=req.rid)

    assert scheduler._recover_idle_pending_requests() is True

    assert REQUEST_STATS[req.rid].status == "completed"
    assert list(scheduler.waiting_queue) == []
    assert req.rid not in scheduler._req_registry


def test_idle_aborted_request_is_terminal_not_recovered():
    scheduler = _make_scheduler()
    req = Req("aborted", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    scheduler._req_registry[req.rid] = req
    REQUEST_STATS.clear()
    REQUEST_STATS[req.rid] = RequestStats(rid=req.rid, status="aborted")

    assert scheduler._recover_idle_pending_requests() is False
    assert scheduler._all_done() is True
    assert list(scheduler.waiting_queue) == []


def test_stale_decode_result_does_not_exceed_max_new_tokens():
    scheduler = _make_scheduler()
    scheduler.enable_overlap = True
    scheduler.last_batch = None
    scheduler.chunked_req = None
    scheduler._result_queue = deque()
    req = Req("r0", [1, 2, 3], SamplingParams(max_new_tokens=2))
    req.status = RequestStatus.DECODE
    req.output_ids = [10, 11]
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req])
    batch.inference_dur_s = 0.01
    REQUEST_STATS.clear()
    REQUEST_STATS["r0"] = RequestStats(rid="r0")

    scheduler.process_batch_result(batch)

    assert req.output_ids == [10, 11]
    assert req.status == RequestStatus.COMPLETED
    assert REQUEST_STATS["r0"].status == "completed"
    assert batch.reqs == []


def test_original_max_new_tokens_caps_retry_decode_progress():
    req = Req("r_retry", [1, 2, 3], SamplingParams(max_new_tokens=10_000))
    req.original_max_new_tokens = 256
    req.pre_failover_output_tokens = 200
    req.output_ids = [0] * 56

    assert req.remaining_decode_tokens == 0
    assert req.reached_decode_limit() is True
    assert req.finished() is False


def test_filter_batch_does_not_drop_length_done_before_completion():
    req = Req("r_done", [1, 2, 3], SamplingParams(max_new_tokens=2))
    req.status = RequestStatus.DECODE
    req.output_ids = [10, 11]
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req])

    batch.filter_batch()

    assert batch.reqs == [req]
    batch.prepare_for_decode()
    assert req.status == RequestStatus.DECODE


def test_merge_batch_preserves_batch_is_full_flag():
    running = ScheduleBatch(
        1,
        0,
        ForwardMode.DECODE,
        [Req("r0", [1], SamplingParams(max_new_tokens=1))],
        batch_is_full=True,
    )
    prefill = ScheduleBatch(
        2,
        0,
        ForwardMode.PREFILL,
        [Req("r1", [2], SamplingParams(max_new_tokens=1))],
    )

    running.merge_batch(prefill)

    assert [req.rid for req in running.reqs] == ["r0", "r1"]
    assert running.batch_is_full is True


def test_refresh_waiting_prefill_plans_only_when_dirty():
    class _HierCache(_CacheStub):
        def __init__(self) -> None:
            super().__init__()
            self.hierarchical_enabled = True
            self.match_calls = 0

        def matched_prefix_length(self, req: Req) -> int:
            self.match_calls += 1
            return 3

    scheduler = _make_scheduler(cache=_HierCache())
    scheduler.sched_cfg.chunked_prefill_size = 4
    scheduler.waiting_queue = deque(
        [Req("w0", [1, 2, 3, 4, 5, 6], SamplingParams(max_new_tokens=2))]
    )
    req = scheduler.waiting_queue[0]
    req.prefill_plan_synced = False
    req.is_retracted = False
    req.kv_committed_length = 0

    scheduler._waiting_plan_dirty = False
    scheduler._refresh_waiting_prefill_plans()
    assert scheduler.cache.match_calls == 0
    assert req.prefill_plan_synced is False

    scheduler._waiting_plan_dirty = True
    scheduler._refresh_waiting_prefill_plans()
    assert scheduler.cache.match_calls == 1
    assert req.prefill_plan_synced is True
    assert req.prefill_chunks_remaining == [3]

    scheduler._refresh_waiting_prefill_plans()
    assert scheduler.cache.match_calls == 1


def test_refresh_req_next_round_input_replans_without_loadback_probe():
    class _HierCache(_CacheStub):
        def __init__(self) -> None:
            super().__init__()
            self.hierarchical_enabled = True
            self.match_calls = 0

        def matched_prefix_length(self, req: Req) -> int:
            self.match_calls += 1
            return 3

    scheduler = _make_scheduler(cache=_HierCache())
    scheduler.sched_cfg.chunked_prefill_size = 4
    req = Req("w1", [1, 2, 3, 4, 5, 6], SamplingParams(max_new_tokens=2))
    req.prefill_chunks_remaining = [6]
    req.cached_length = 0
    req.kv_committed_length = 0
    req.is_retracted = False
    req.prefill_plan_synced = False
    adder = type("Adder", (), {"rem_total_tokens": 77.9})()

    scheduler._refresh_req_next_round_input(req, adder=adder)

    assert scheduler.cache.match_calls == 1
    assert req.cached_length == 3
    assert req.prefill_chunks_remaining == [3]
    assert req.prefill_plan_synced is True
    assert scheduler.cache.init_load_back_mem_quotas == []


def test_refresh_req_next_round_input_refreshes_fresh_synced_request_at_admission():
    class _HierCache(_CacheStub):
        def __init__(self) -> None:
            super().__init__()
            self.hierarchical_enabled = True
            self.match_calls = 0

        def matched_prefix_length(self, req: Req) -> int:
            self.match_calls += 1
            return 5

    scheduler = _make_scheduler(cache=_HierCache())
    scheduler.sched_cfg.chunked_prefill_size = 4
    req = Req("w2", [1, 2, 3, 4, 5, 6], SamplingParams(max_new_tokens=2))
    req.prefill_chunks_remaining = [2]
    req.cached_length = 4
    req.kv_committed_length = 0
    req.is_retracted = False
    req.prefill_plan_synced = True
    adder = type("Adder", (), {"rem_total_tokens": 99.0})()

    scheduler._refresh_req_next_round_input(req, adder=adder)

    assert scheduler.cache.match_calls == 1
    assert req.cached_length == 5
    assert req.prefill_chunks_remaining == [1]
    assert scheduler.cache.init_load_back_mem_quotas == []


def test_refresh_req_next_round_input_refreshes_chunked_state_each_round():
    class _HierCache(_CacheStub):
        def __init__(self) -> None:
            super().__init__()
            self.hierarchical_enabled = True
            self.match_calls = 0

        def matched_prefix_length(self, req: Req) -> int:
            self.match_calls += 1
            return 5

    scheduler = _make_scheduler(cache=_HierCache())
    scheduler.sched_cfg.chunked_prefill_size = 4
    req = Req("w3", [1, 2, 3, 4, 5, 6, 7, 8], SamplingParams(max_new_tokens=2))
    req.prefill_chunks_remaining = [2]
    req.cached_length = 6
    req.kv_committed_length = 7
    req.is_retracted = False
    req.prefill_plan_synced = True
    adder = type("Adder", (), {"rem_total_tokens": 88.0})()

    scheduler._refresh_req_next_round_input(req, adder=adder)

    assert scheduler.cache.match_calls == 1
    assert req.cached_length == 7
    assert req.kv_committed_length == 7
    assert req.prefill_chunks_remaining == [1]


def test_update_running_batch_flushes_write_through_acks_when_hierarchical():
    cache = _CacheStub(available_tokens=1000)
    cache.hierarchical_enabled = True
    scheduler = _make_scheduler(cache=cache)
    req = Req("d0", [1, 2, 3, 4], SamplingParams(max_new_tokens=4))
    req.status = RequestStatus.DECODE
    req.output_ids = [0]
    req.kv_committed_length = req.prefill_length
    req.cached_length = req.prefill_length
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [req])
    scheduler.running_batch = batch

    updated = scheduler.update_running_batch(batch)

    assert updated is not None
    assert cache.flush_calls == 1
