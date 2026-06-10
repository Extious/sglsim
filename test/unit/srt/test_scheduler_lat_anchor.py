from __future__ import annotations

import os

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.types import SimulationMode

from test.unit.srt.engine_helpers import drive_engine


def test_register_sets_last_event_time_to_created_time():
    os.environ["SGLSIM_SIMULATION_MODE"] = SimulationMode.STREAMING.value
    ConfigManager._raw = {"scheduler": {"dp_size": 1}}
    ConfigManager._scheduler = None
    REQUEST_STATS.clear()

    server_args = ServerArgs(
        streaming_arrival=True,
        model_path="mock-model",
    )
    scheduler = Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)
    tok = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid="r0",
            input_len=4,
            max_new_tokens=1,
            simulation={"server_created_time": 2.0},
        )
    )
    scheduler._register_incoming_request(tok, now=5.0)
    stats = REQUEST_STATS["r0"]
    assert stats.created_time == 2.0
    assert stats.last_event_time == stats.created_time
    assert stats.queue_start == 5.0


def test_total_latency_on_t_includes_pre_queue_gap():
    tokenize_dur = 0.05
    engine = Engine(
        ServerArgs(
            streaming_arrival=True,
            model_path="mock-model",
        )
    )
    drive_engine(
        engine,
        [
            GenerateReqInput(
                rid="r0",
                input_ids=[1, 2, 3],
                max_new_tokens=2,
                simulation={"created_time": 0.0, "tokenize_dur": tokenize_dur},
            )
        ],
    )
    stats = REQUEST_STATS["r0"]
    pre_queue = stats.queue_start - stats.created_time
    assert pre_queue >= tokenize_dur
    assert stats.last_event_time - stats.created_time >= pre_queue
    engine.shutdown()
