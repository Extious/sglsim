from __future__ import annotations

import os

from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.state import StateManager
from sglsim.srt.simulation.types import SimulationMode
from sglsim.srt.utils.network import SimIpcRegistry, _QUEUES


def drive_scheduler(
    server_args: ServerArgs,
    reqs: list[GenerateReqInput],
) -> Scheduler:
    if server_args.sim_config_path:
        os.environ["SGLSIM_CONFIG_PATH"] = server_args.sim_config_path
    SimIpcRegistry.install_global(None)
    ConfigManager.load(server_args.sim_config_path)
    ConfigManager.apply_server_args_overrides(server_args)
    os.environ["SGLSIM_SIMULATION_MODE"] = SimulationMode.STREAMING.value
    StateManager.reset()
    REQUEST_STATS.clear()

    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    scheduler = Scheduler(server_args, port_args, dp_rank=0)
    endpoint = port_args.scheduler_input_ipc_name_for_dp(0)
    for req in reqs:
        _QUEUES[endpoint].append(TokenizedGenerateReqInput.from_generate(req))
    scheduler.event_loop()
    return scheduler
