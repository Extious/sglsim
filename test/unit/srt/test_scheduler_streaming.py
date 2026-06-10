import os

from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.managers.scheduler import Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.types import SimulationMode
from sglsim.srt.utils.network import SimZmqContext, _QUEUES


def test_streaming_recv_enqueues_on_arrival():
    os.environ["SGLSIM_SIMULATION_MODE"] = SimulationMode.STREAMING.value
    ConfigManager._raw = {"scheduler": {"dp_size": 1}}
    ConfigManager._scheduler = None
    server_args = ServerArgs(streaming_arrival=True)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()

    scheduler = Scheduler(server_args, port_args, dp_rank=0)
    assert scheduler.sim_mode == SimulationMode.STREAMING

    gen = GenerateReqInput(
        rid="r0",
        input_len=4,
        max_new_tokens=1,
        simulation={"created_time": 0.0},
    )
    tok = TokenizedGenerateReqInput.from_generate(gen)
    endpoint = port_args.scheduler_input_ipc_name_for_dp(0)
    _QUEUES[endpoint].append(tok)

    scheduler.recv_requests()
    assert len(scheduler.waiting_queue) == 1
