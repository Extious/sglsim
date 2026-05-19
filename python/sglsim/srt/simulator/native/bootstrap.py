from __future__ import annotations

from sglsim.srt.simulator.native.config import NativeSimulationConfig


def install_native_simulation(server_args) -> NativeSimulationConfig:
    config = NativeSimulationConfig.from_server_args(server_args)

    from sglsim.srt.simulator.native import cache_controller_patch
    from sglsim.srt.simulator.native import memory_pool_patch
    from sglsim.srt.simulator.native import model_config_patch
    from sglsim.srt.simulator.native import model_runner_patch
    from sglsim.srt.simulator.native import scheduler_patch

    model_config_patch.install(config)
    model_runner_patch.install(config)
    scheduler_patch.install(config)
    cache_controller_patch.install(config)
    memory_pool_patch.install(config)
    return config


def run_native_scheduler_process(*args, **kwargs):
    """Scheduler subprocess entrypoint that installs native patches before init."""
    server_args = args[0] if args else kwargs.get("server_args")
    if server_args is not None:
        install_native_simulation(server_args)

    from sglsim.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)
