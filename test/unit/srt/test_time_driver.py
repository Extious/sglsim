from __future__ import annotations

from unittest.mock import patch

import pytest

from sglsim.srt.managers.schedule_batch import ForwardMode, ScheduleBatch
from sglsim.srt.managers.aegis_l1_l2 import (
    AegisL1L2Config,
    AegisRankBackupManager,
)
from sglsim.srt.managers.scheduler import Scheduler
from sglsim.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.managers.scheduler import REQUEST_STATS
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.forward_slot import ForwardSlot
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.pipeline_accounting import PipelineStages
from sglsim.srt.simulation.time_driver import LogicalClockDriver
from sglsim.srt.simulation.offline_coordinator import OfflineEventCoordinator
from sglsim.srt.utils.network import SimIpcRegistry


def _logical_scheduler(driver: LogicalClockDriver) -> Scheduler:
    SimIpcRegistry.install_global(None)
    ConfigManager._raw = {
        "scheduler": {
            "max_prefill_tokens": 8192,
            "chunked_prefill_size": 8192,
        },
        "cache": {
            "page_size": 16,
            "l1_capacity_pages": 1000,
            "l2_capacity_pages": 0,
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    server_args = ServerArgs(disable_overlap_schedule=False)
    return Scheduler(
        server_args,
        PortArgs.init_new(server_args),
        dp_rank=0,
        time_driver=driver,
    )


def test_logical_driver_advances_pipeline_without_cpu_overflow() -> None:
    driver = LogicalClockDriver(start_s=10.0)
    slot = ForwardSlot()

    result = driver.advance_pipeline(
        PipelineStages(h2d_s=3.0, forward_s=2.0, backup_pending_s=1.0),
        last_infer_s=5.0,
        sched_elapsed_s=99.0,
        overlap=True,
        forward_slot=slot,
    )

    assert result.elapsed_s == 2.0
    assert result.cpu_overflow_s == 0.0
    assert result.completed_at == 12.0
    assert driver.now() == 12.0
    assert slot.last_infer_s == 2.0


def test_logical_pipeline_can_be_planned_then_completed_by_parent_event() -> None:
    driver = LogicalClockDriver(start_s=4.0)
    slot = ForwardSlot()
    plan = driver.plan_pipeline(
        PipelineStages(h2d_s=0.0, forward_s=3.0, backup_pending_s=0.0),
        last_infer_s=0.0,
        sched_elapsed_s=8.0,
        overlap=True,
    )

    assert driver.now() == 4.0
    assert plan.completed_at == 7.0
    driver.set_time(plan.completed_at)
    result = driver.complete_pipeline(plan, forward_slot=slot)
    assert result.completed_at == 7.0
    assert result.cpu_overflow_s == 0.0


def test_logical_driver_clocks_are_independent() -> None:
    rank0 = LogicalClockDriver()
    rank1 = LogicalClockDriver()

    rank0.delay(7.0)
    rank1.delay(2.0)

    assert rank0.now() == 7.0
    assert rank1.now() == 2.0
    rank1.set_time(5.0)
    assert rank0.now() == 7.0
    assert rank1.now() == 5.0
    with pytest.raises(ValueError, match="backwards"):
        rank1.set_time(4.0)


def test_scheduler_logical_step_uses_pipeline_v2_without_sleep() -> None:
    driver = LogicalClockDriver()
    scheduler = _logical_scheduler(driver)
    batch = ScheduleBatch(0, 0, ForwardMode.DECODE, [])
    batch.inference_dur_s = 0.02
    batch.pipeline_overlap = False
    scheduler.cache.hicache.pending_backup_dur = 0.01

    with patch(
        "sglsim.srt.managers.scheduler.StateManager.pop_hicache_l2_load_dur",
        return_value=0.03,
    ):
        with patch("sglsim.srt.managers.scheduler.time.sleep") as sleep_mock:
            scheduler._advance_time_step(
                batch=batch,
                sched_elapsed=100.0,
                overlap=False,
            )

    # H2D is serial; D2H is fully hidden by the longer forward.
    assert driver.now() == pytest.approx(0.05)
    sleep_mock.assert_not_called()
    assert scheduler._response_time_after_batch(batch) == pytest.approx(0.05)


def test_logical_restore_delay_advances_only_own_rank_clock() -> None:
    rank0 = LogicalClockDriver(start_s=3.0)
    rank1 = LogicalClockDriver(start_s=3.0)

    assert rank0.delay(0.25) == pytest.approx(3.25)
    assert rank1.now() == 3.0


def test_algorithm3_window_uses_injected_rank_clock() -> None:
    driver = LogicalClockDriver()
    manager = AegisRankBackupManager(
        dp_rank=0,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=2048,
        ),
        now_fn=driver.now,
        monotonic_fn=driver.now,
    )

    driver.delay(0.125)
    assert manager.snapshot()["window_elapsed_ms"] == pytest.approx(125.0)


def test_split_offline_scheduler_step_runs_request_without_sleep() -> None:
    coordinator = OfflineEventCoordinator(dp_size=1)
    driver = coordinator.rank_drivers[0]
    scheduler = _logical_scheduler(driver)
    scheduler.enable_overlap = False
    REQUEST_STATS.clear()
    tokenized = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid="offline-r0",
            input_ids=list(range(8)),
            max_new_tokens=2,
            simulation={"server_created_time": 0.0},
        )
    )
    scheduler._register_incoming_request(tokenized, now=0.0)
    telemetry: list[dict] = []

    with patch.object(
        scheduler.predictor,
        "predict_infer_time",
        return_value=0.01,
    ):
        with patch("sglsim.srt.managers.scheduler.time.sleep") as sleep_mock:
            for _ in range(20):
                step = scheduler.prepare_offline_step()
                if step is None:
                    if scheduler._all_done():
                        break
                    continue
                event = coordinator.schedule_rank_completion(
                    0,
                    duration_s=step.duration_s,
                    payload=step,
                )
                assert coordinator.pop_next() == event
                telemetry.append(scheduler.complete_offline_step(step))
            else:
                raise AssertionError("offline request did not finish")

    sleep_mock.assert_not_called()
    assert REQUEST_STATS["offline-r0"].status == "completed"
    assert driver.now() > 0.0
    assert {row["batch_mode"] for row in telemetry} >= {"prefill", "decode"}
    assert all(row["cpu_overflow_s"] == 0.0 for row in telemetry)
