import io
import logging
import re
from datetime import datetime

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.utils.logging import scheduler_logger

from test.unit.srt.engine_helpers import drive_engine

_WALL_TIME_RE = re.compile(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def _capture_scheduler_log(server_args: ServerArgs) -> tuple[str, logging.Logger]:
    stream = io.StringIO()
    log = scheduler_logger("sglsim.srt.managers.scheduler", server_args, dp_rank=0)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(log.handlers[0].formatter)
    log.addHandler(handler)
    return stream, log


def test_scheduler_logs_prefill_and_decode_batches():
    engine = Engine(ServerArgs())
    result = drive_engine(
        engine,
        [
            GenerateReqInput(
                rid="r0",
                input_len=4,
                max_new_tokens=2,
                simulation={"created_time": 0.0},
            )
        ],
    )
    engine.shutdown()
    assert result.request_stats["r0"].status == "completed"


def test_decode_log_running_req_can_exceed_one_with_multiple_reqs():
    engine = Engine(ServerArgs())
    result = drive_engine(
        engine,
        [
            GenerateReqInput(
                rid=f"r{i}",
                input_len=4,
                max_new_tokens=3,
                simulation={"created_time": 0.0},
            )
            for i in range(3)
        ],
    )
    engine.shutdown()
    assert len(result.request_stats) == 3


def test_multi_dp_streaming_submit_completes():
    engine = Engine(ServerArgs(dp_size=2))
    engine.start_background_scheduler()
    for i in range(4):
        engine.submit(
            GenerateReqInput(
                rid=f"r{i}",
                input_len=4,
                max_new_tokens=1,
                created_time=0.0,
                simulation={"created_time": 0.0},
            )
        )
    result = engine.run_until_idle()
    engine.shutdown()
    assert len(result.request_stats) == 4
    assert all(s.status == "completed" for s in result.request_stats.values())


def test_multi_dp_completes_requests_in_subprocess():
    engine = Engine(ServerArgs(dp_size=2))
    assert engine.schedulers == []
    result = drive_engine(
        engine,
        [
            GenerateReqInput(
                rid="r0",
                input_len=4,
                max_new_tokens=1,
                created_time=0.0,
                simulation={"created_time": 0.0},
            ),
            GenerateReqInput(
                rid="r1",
                input_len=4,
                max_new_tokens=1,
                created_time=0.0,
                simulation={"created_time": 0.0},
            ),
        ],
    )
    engine.shutdown()
    assert len(result.request_stats) == 2
    assert all(s.status == "completed" for s in result.request_stats.values())


def test_engine_startup_logs_sglang_compat_events(capsys):
    engine = Engine(ServerArgs(model_path="mock-model"))
    engine.shutdown()

    text = capsys.readouterr().out
    assert "Load weight begin." in text
    assert "Load weight end." in text
    assert "KV Cache is allocated." in text
    assert "Capture CUDA graph begin." in text
    assert "Capture CUDA graph end." in text
    assert "max_total_num_tokens=" in text


def test_logs_use_wall_clock_timestamp():
    stream, log = _capture_scheduler_log(ServerArgs())
    year = str(datetime.now().year)
    log.info("Prefill batch, #running-req: 0")
    text = stream.getvalue()
    assert _WALL_TIME_RE.search(text)
    assert year in text


def test_sglang_projected_logs_use_base_epoch_when_no_sim_time(monkeypatch):
    base_epoch = 1_700_000_000.0
    monkeypatch.setenv("SGLSIM_LOG_PROFILE", "sglang_projected")
    monkeypatch.setenv("SGLSIM_LOG_BASE_TIME_EPOCH_S", str(base_epoch))
    stream, log = _capture_scheduler_log(ServerArgs())

    log.info("Prefill batch, #running-req: 0")

    expected = datetime.fromtimestamp(base_epoch).strftime("%Y-%m-%d %H:%M:%S")
    text = stream.getvalue()
    assert f" | {expected} DP0]" in text
    assert _WALL_TIME_RE.search(text)
