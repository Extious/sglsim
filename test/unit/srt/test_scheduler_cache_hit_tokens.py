from sglsim.srt.managers.schedule_batch import ForwardMode, Req, SamplingParams
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.state import StateManager
from sglsim.srt.simulation.types import RequestStats
from sglsim.srt.utils.network import SimIpcRegistry


def _scheduler() -> Scheduler:
    SimIpcRegistry.install_global(None)
    StateManager.reset()
    REQUEST_STATS.clear()
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
            "l2_capacity_pages": 0,
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    server_args = ServerArgs()
    return Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)


def _enqueue(scheduler: Scheduler, rid: str, token_ids: list[int]) -> Req:
    req = Req(rid, token_ids, SamplingParams(max_new_tokens=1))
    req.prefill_chunks_remaining = [8192, 8192, 8192, max(len(token_ids) - 24576, 0)]
    req.prefill_chunks_remaining = [chunk for chunk in req.prefill_chunks_remaining if chunk > 0]
    REQUEST_STATS[rid] = RequestStats(rid=rid, input_length=len(token_ids), output_length=1)
    scheduler.waiting_queue.append(req)
    return req


def _drain_prefill(scheduler: Scheduler, rid: str, *, max_steps: int = 20) -> None:
    for _ in range(max_steps):
        batch = scheduler.get_next_batch_to_run()
        if batch is None:
            continue
        assert batch.mode == ForwardMode.PREFILL
        scheduler.process_batch_result(batch)
        req = next((req for req in batch.reqs if req.rid == rid), None)
        if req is not None and not req.prefill_chunks_remaining:
            return
    raise AssertionError(f"rid={rid} did not finish prefill")


def test_chunked_prefill_self_progress_is_not_cache_hit():
    scheduler = _scheduler()
    _enqueue(scheduler, "r0", list(range(25320)))

    _drain_prefill(scheduler, "r0")

    assert REQUEST_STATS["r0"].cache_hit_tokens == 0


def test_shared_prefix_reports_external_cache_hit_once():
    scheduler = _scheduler()
    shared_prefix = list(range(1000))
    _enqueue(scheduler, "r0", shared_prefix + list(range(10_000, 34_320)))
    _drain_prefill(scheduler, "r0")

    _enqueue(scheduler, "r1", shared_prefix + list(range(20_000, 44_320)))
    first_batch = scheduler.get_next_batch_to_run()
    assert first_batch is not None
    assert first_batch.mode == ForwardMode.PREFILL

    # Page size is 16, so the 1000-token shared prefix is truncated to 992.
    assert REQUEST_STATS["r1"].cache_hit_tokens == 992
    scheduler.process_batch_result(first_batch)

    _drain_prefill(scheduler, "r1")
    assert REQUEST_STATS["r1"].cache_hit_tokens == 992
