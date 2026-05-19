from __future__ import annotations

import json
from pathlib import Path

from sglsim.srt.managers.io_struct import (
    FailoverBatchReqInput,
    ReqSnapshot,
    SimulateGpuFailureReqInput,
    SimulateGpuRecoveryReqInput,
)
from sglsim.srt.utils.failover_event_logger import append_failover_event


def test_native_failure_control_path_symbols_are_present():
    from sglsim.srt.entrypoints import http_server
    from sglsim.srt.managers import data_parallel_controller
    from sglsim.srt.managers import scheduler
    from sglsim.srt.managers import tokenizer_manager

    assert hasattr(http_server, "simulate_gpu_failure")
    assert hasattr(http_server, "simulate_gpu_recovery")
    assert hasattr(tokenizer_manager.TokenizerManager, "simulate_gpu_failure")
    assert hasattr(tokenizer_manager.TokenizerManager, "simulate_gpu_recovery")
    assert hasattr(data_parallel_controller.DataParallelController, "handle_simulate_gpu_failure")
    assert hasattr(data_parallel_controller.DataParallelController, "handle_simulate_gpu_recovery")
    for name in (
        "handle_simulate_gpu_failure",
        "handle_simulate_gpu_recovery",
        "_collect_failover_reqs",
        "_clear_failed_rank_scheduler_state",
        "_snapshot_req",
    ):
        assert hasattr(scheduler.Scheduler, name)


def test_native_failure_io_structs_preserve_failover_snapshot_contract():
    failure = SimulateGpuFailureReqInput(dp_rank=2)
    recovery = SimulateGpuRecoveryReqInput(dp_rank=2)
    snapshot = ReqSnapshot(
        rid="rid-1",
        output_ids=[7, 8],
        backed_up_tokens=3,
        origin_input_ids=[1, 2, 3],
        original_max_new_tokens=16,
    )
    batch = FailoverBatchReqInput(failed_dp_rank=2, snapshots=[snapshot])

    assert failure.dp_rank == 2
    assert recovery.dp_rank == 2
    assert batch.failed_dp_rank == 2
    assert batch.snapshots[0].rid == "rid-1"
    assert batch.snapshots[0].backed_up_tokens == 3


def test_failover_event_logger_writes_configured_jsonl(monkeypatch, tmp_path: Path):
    events = tmp_path / "failover_events.jsonl"
    monkeypatch.setenv("SGLANG_FAILOVER_EVENTS_FILE", str(events))

    append_failover_event("gpu_failure_simulated", dp_rank=0, num_requests=1)
    append_failover_event("gpu_recovered", dp_rank=0)

    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "gpu_failure_simulated",
        "gpu_recovered",
    ]
    assert rows[0]["dp_rank"] == 0
