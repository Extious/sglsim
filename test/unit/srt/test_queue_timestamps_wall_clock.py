"""Queue timestamps align with SGLang wait_queue_entry_time."""

from __future__ import annotations

from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.network import SimIpcRegistry


def test_register_sets_queue_start_at_enqueue():
    SimIpcRegistry.install_global(None)
    ConfigManager._raw = {"scheduler": {"dp_size": 1}}
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    REQUEST_STATS.clear()

    server_args = ServerArgs(model_path="mock-model")
    scheduler = Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)
    tok = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid="r0",
            input_len=4,
            max_new_tokens=1,
            simulation={"server_created_time": 100.0},
        )
    )
    scheduler._register_incoming_request(tok, now=105.0)
    stats = REQUEST_STATS["r0"]
    assert stats.created_time == 100.0
    assert stats.queue_start == 105.0
    assert stats.queue_start >= stats.created_time
