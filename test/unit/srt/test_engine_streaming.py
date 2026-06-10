import threading

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.types import RequestStats

from test.unit.srt.engine_helpers import drive_engine


def _workload(n: int, input_len: int = 3, output_len: int = 1) -> list[GenerateReqInput]:
    return [
        GenerateReqInput(
            rid=f"r{i}",
            input_len=input_len,
            max_new_tokens=output_len,
            created_time=float(i),
            simulation={"created_time": float(i)},
        )
        for i in range(n)
    ]


def test_engine_completes_multiple_requests():
    engine = Engine(ServerArgs())
    drive_engine(engine, _workload(2))
    assert all(REQUEST_STATS[f"r{i}"].status == "completed" for i in range(2))


def test_engine_decodes_until_max_new_tokens():
    engine = Engine(ServerArgs())
    drive_engine(
        engine,
        [GenerateReqInput("r0", input_len=3, max_new_tokens=3, simulation={})],
    )
    assert len(REQUEST_STATS["r0"].gen_token_latencies) >= 1


def test_wait_for_rid_preserves_already_completed_event(monkeypatch):
    engine = Engine.__new__(Engine)
    engine._completion_events = {}
    engine._completion_text = {}
    engine._completion_lock = threading.Lock()
    monkeypatch.setattr(engine, "_drain_detokenizer", lambda: 0)

    REQUEST_STATS.clear()
    REQUEST_STATS["r0"] = RequestStats(rid="r0", status="completed")
    engine._register_rid("r0")
    engine._apply_detokenizer_message({"rid": "r0", "finished": True, "text": "ok"})

    stats = engine.wait_for_rid("r0", timeout_s=0.0)

    assert stats.status == "completed"
    assert engine.completion_text("r0") == "ok"


def test_finished_abort_message_preserves_aborted_status(monkeypatch):
    engine = Engine.__new__(Engine)
    engine._completion_events = {}
    engine._completion_text = {}
    engine._completion_lock = threading.Lock()
    monkeypatch.setattr(engine, "_drain_detokenizer", lambda: 0)

    REQUEST_STATS.clear()
    REQUEST_STATS["r0"] = RequestStats(rid="r0", status="running")
    engine._register_rid("r0")
    engine._apply_detokenizer_message(
        {"rid": "r0", "status": "aborted", "finished": True}
    )

    stats = engine.wait_for_rid("r0", timeout_s=0.0)

    assert stats.status == "aborted"
