import os

from sglsim.srt.managers.schedule_batch import ForwardMode, Req, RequestStatus, SamplingParams
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.types import RequestStats, SimulationMode


def _scheduler() -> Scheduler:
    from sglsim.srt.simulation.manager.state import StateManager
    from sglsim.srt.utils.network import SimIpcRegistry

    REQUEST_STATS.clear()
    StateManager.reset()
    SimIpcRegistry.install_global(None)
    os.environ["SGLSIM_SIMULATION_MODE"] = SimulationMode.STREAMING.value
    ConfigManager._raw = {
        "scheduler": {"dp_size": 1},
        "cache": {
            "page_size": 16,
            "l1_capacity_pages": 10_000,
            "l2_capacity_pages": 0,
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    server_args = ServerArgs(streaming_arrival=True)
    return Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)


def test_rank_idle_after_decode_without_stale_last_batch():
    scheduler = _scheduler()
    sent: list[dict] = []
    scheduler.send_to_detokenizer.send_pyobj = lambda msg: sent.append(msg)

    req = Req("r0", list(range(8)), SamplingParams(max_new_tokens=2))
    REQUEST_STATS["r0"] = RequestStats(rid="r0", input_length=8, output_length=2)
    scheduler.waiting_queue.append(req)
    scheduler.last_batch = None

    for _ in range(200):
        if not scheduler.event_loop_step() and scheduler._all_done():
            break

    finished = [m for m in sent if m.get("finished")]
    assert len(finished) == 1
    assert len(sent) >= 2
    assert scheduler.last_batch is None
    assert scheduler._rank_idle() is True

    before = len(sent)
    for _ in range(5):
        scheduler.event_loop_step()
    assert len(sent) == before


def test_finalize_last_batch_clears_completed_reqs():
    scheduler = _scheduler()
    r = Req("r1", [1, 2, 3, 4], SamplingParams(max_new_tokens=1))
    r.status = RequestStatus.COMPLETED
    scheduler.last_batch = scheduler._make_batch(ForwardMode.DECODE, [r], 0.0)
    assert not scheduler.last_batch.is_empty()
    scheduler._finalize_last_batch()
    assert scheduler.last_batch is None


def test_rank_idle_ignores_stale_stats_without_active_work():
    scheduler = _scheduler()
    REQUEST_STATS["stale"] = RequestStats(rid="stale", status="running")

    assert scheduler._rank_idle() is True


def test_completed_decode_records_final_latency(monkeypatch):
    scheduler = _scheduler()
    req = Req("r2", [1, 2, 3, 4], SamplingParams(max_new_tokens=1))
    stats = RequestStats(
        rid="r2",
        queue_end=2.0,
        last_event_time=2.0,
        input_length=4,
        output_length=1,
    )
    REQUEST_STATS["r2"] = stats
    batch = scheduler._make_batch(ForwardMode.DECODE, [req], 2.0)

    monkeypatch.setattr(scheduler, "_response_time_after_batch", lambda batch: 5.0)

    def complete(request):
        request.status = RequestStatus.COMPLETED
        REQUEST_STATS[request.rid].status = "completed"

    monkeypatch.setattr(scheduler, "_complete_request", complete)

    scheduler.process_batch_result(batch)

    assert stats.gen_token_latencies == [3.0]
    assert stats.last_event_time == 5.0
