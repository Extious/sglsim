"""Wall-clock pipeline: one sleep per step via _advance_wall_clock_step."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sglsim.srt.managers.schedule_batch import ForwardMode, Req, SamplingParams, ScheduleBatch
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.types import RequestStats
from sglsim.srt.utils.network import SimIpcRegistry


def _wall_clock_scheduler() -> Scheduler:
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
    return Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)


def test_run_batch_does_not_sleep():
    scheduler = _wall_clock_scheduler()
    batch = scheduler._make_batch(ForwardMode.DECODE, [], 0.0)
    batch.inference_dur_s = 0.0

    with patch("sglsim.srt.managers.scheduler.time.sleep") as sleep_mock:
        scheduler.run_batch(batch)
        sleep_mock.assert_not_called()


def test_advance_wall_clock_step_sleeps_once_with_pipeline_total():
    scheduler = _wall_clock_scheduler()
    batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [])
    batch.inference_dur_s = 0.05

    with patch("sglsim.srt.managers.scheduler.time.sleep") as sleep_mock:
        scheduler._advance_wall_clock_step(
            batch=batch,
            sched_elapsed=0.01,
            overlap=True,
        )
        sleep_mock.assert_called_once()
        slept = sleep_mock.call_args[0][0]
        assert slept >= 0.05


def test_advance_wall_clock_step_overlap_sleeps_forward_only():
    scheduler = _wall_clock_scheduler()
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [])
    batch.inference_dur_s = 0.02
    scheduler._forward_slot.last_infer_s = 0.10
    scheduler.cache.hicache.pending_backup_dur = 0.01

    with patch(
        "sglsim.srt.managers.scheduler.StateManager.pop_hicache_l2_load_dur",
        return_value=0.03,
    ):
        with patch("sglsim.srt.managers.scheduler.time.sleep") as sleep_mock:
            scheduler._advance_wall_clock_step(
                batch=batch,
                sched_elapsed=0.0,
                overlap=True,
            )
            assert sleep_mock.call_args[0][0] == 0.02


def test_advance_wall_clock_step_no_overlap_serializes_hicache_transfer():
    scheduler = _wall_clock_scheduler()
    batch = ScheduleBatch(1, 0, ForwardMode.DECODE, [])
    batch.inference_dur_s = 0.02
    scheduler._forward_slot.last_infer_s = 0.10
    scheduler.cache.hicache.pending_backup_dur = 0.01

    with patch(
        "sglsim.srt.managers.scheduler.StateManager.pop_hicache_l2_load_dur",
        return_value=0.03,
    ):
        with patch("sglsim.srt.managers.scheduler.time.sleep") as sleep_mock:
            scheduler._advance_wall_clock_step(
                batch=batch,
                sched_elapsed=0.0,
                overlap=True,
            )
            overlap_sleep = sleep_mock.call_args[0][0]

            scheduler._forward_slot = scheduler._forward_slot.__class__()
            scheduler.cache.hicache.pending_backup_dur = 0.01
            sleep_mock.reset_mock()
            scheduler._advance_wall_clock_step(
                batch=batch,
                sched_elapsed=0.0,
                overlap=False,
            )
            no_overlap_sleep = sleep_mock.call_args[0][0]

    assert overlap_sleep == 0.02
    assert no_overlap_sleep == pytest.approx(0.06)


def test_advance_wall_clock_step_no_overlap_does_not_add_cpu_overflow():
    scheduler = _wall_clock_scheduler()
    batch = ScheduleBatch(2, 0, ForwardMode.DECODE, [])
    batch.inference_dur_s = 0.02
    scheduler._forward_slot.record_forward(now=10.0, forward_s=0.01, sleep_s=0.01)

    with patch("sglsim.srt.managers.scheduler.time.time", return_value=10.0):
        with patch("sglsim.srt.managers.scheduler.time.sleep") as sleep_mock:
            scheduler._advance_wall_clock_step(
                batch=batch,
                sched_elapsed=0.10,
                overlap=False,
            )

    assert sleep_mock.call_args[0][0] == 0.02


def test_advance_wall_clock_step_does_not_consume_backup():
    scheduler = _wall_clock_scheduler()
    batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [])
    batch.inference_dur_s = 0.05
    scheduler.cache.hicache.pending_backup_dur = 0.08

    with patch.object(scheduler.cache, "consume_backup_overlap") as overlap_mock:
        with patch.object(scheduler.cache, "consume_backup_serial") as serial_mock:
            with patch("sglsim.srt.managers.scheduler.time.sleep"):
                scheduler._advance_wall_clock_step(
                    batch=batch,
                    sched_elapsed=0.0,
                    overlap=True,
                )
        overlap_mock.assert_not_called()
        serial_mock.assert_not_called()
    assert scheduler.cache.hicache.pending_backup_dur == 0.08


def test_response_time_checks_hicache_and_consumes_backup():
    scheduler = _wall_clock_scheduler()
    batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [])
    batch.inference_dur_s = 0.05
    batch.pipeline_overlap = True
    scheduler.cache.hicache.pending_backup_dur = 0.08

    with patch.object(scheduler.cache, "check_hicache_events") as check_mock:
        with patch.object(
            scheduler.cache,
            "consume_backup_overlap",
            wraps=scheduler.cache.consume_backup_overlap,
        ) as consume_mock:
            scheduler._response_time_after_batch(batch)
        check_mock.assert_called_once_with(remain_dur=0.05)
        consume_mock.assert_called_once_with(0.05)
    assert scheduler.cache.hicache.pending_backup_dur == 0.03


def test_overlap_step_advances_after_predict_only_run_batch():
    scheduler = _wall_clock_scheduler()
    req = Req("r0", list(range(8)), SamplingParams(max_new_tokens=2))
    REQUEST_STATS["r0"] = RequestStats(rid="r0", input_length=8, output_length=2)
    decode = scheduler._make_batch(ForwardMode.DECODE, [req], 0.0)

    sleeps: list[float] = []

    def _capture_sleep(duration: float) -> None:
        sleeps.append(duration)

    with patch.object(scheduler, "get_next_batch_to_run", return_value=decode):
        with patch.object(
            scheduler.predictor,
            "predict_infer_time",
            return_value=0.04,
        ):
            with patch("sglsim.srt.managers.scheduler.time.sleep", side_effect=_capture_sleep):
                assert scheduler._event_loop_step_overlap() is True

    assert len(sleeps) == 1
    assert sleeps[0] >= 0.04
