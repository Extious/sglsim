from unittest.mock import patch

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
            "max_prefill_tokens": 8192,
            "chunked_prefill_size": 8192,
            "pp_max_micro_batch_size": 4,
            "init_new_token_ratio": 0.0,
            "disable_overlap_schedule": True,
        },
        "cache": {
            "page_size": 16,
            "l1_capacity_pages": 100_000,
            "l2_capacity_pages": 0,
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    server_args = ServerArgs(disable_overlap_schedule=True)
    return Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)


def test_chunked_prefill_ttft_uses_service_time_not_unrelated_gaps():
    scheduler = _scheduler()
    req = Req("r0", list(range(16384)), SamplingParams(max_new_tokens=1))
    req.prefill_chunks_remaining = [8192, 8192]
    REQUEST_STATS["r0"] = RequestStats(
        rid="r0",
        input_length=16384,
        output_length=1,
        created_time=0.0,
        queue_start=0.0,
    )
    scheduler.waiting_queue.append(req)

    clock = iter([0.0, 1.0, 101.0, 102.0])

    def _fake_time() -> float:
        return next(clock)

    with patch("sglsim.srt.managers.scheduler.time.time", side_effect=_fake_time):
        first = scheduler.get_next_batch_to_run()
        assert first is not None
        assert first.mode == ForwardMode.PREFILL
        first.inference_dur_s = 1.0
        scheduler.process_batch_result(first)

        second = scheduler.get_next_batch_to_run()
        assert second is not None
        assert second.mode == ForwardMode.PREFILL
        second.inference_dur_s = 1.0
        scheduler.process_batch_result(second)

    stats = REQUEST_STATS["r0"]
    assert stats.queue_end == 0.0
    assert stats.service_time_s == 2.0
    assert stats.gen_token_latencies == [2.0]
    assert stats.last_event_time == 102.0
