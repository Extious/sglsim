from __future__ import annotations

import os

from sglsim.srt.simulation.types import SimulationMode


def config_path() -> str | None:
    return os.environ.get("SGLSIM_CONFIG_PATH")


def simulation_mode() -> SimulationMode:
    raw = os.getenv("SGLSIM_SIMULATION_MODE", SimulationMode.STREAMING.value).upper()
    if raw != SimulationMode.STREAMING.value:
        raise ValueError(
            f"Unsupported SGLSIM_SIMULATION_MODE={raw!r}; only STREAMING is supported"
        )
    return SimulationMode.STREAMING
