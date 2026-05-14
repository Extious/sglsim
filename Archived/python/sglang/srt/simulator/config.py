from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict


_ACTIVE_SETTINGS: "SimulatorSettings | None" = None


@dataclass(frozen=True)
class SimulatorSettings:
    enabled: bool
    config_path: str | None
    mode: str
    output_dir: str
    num_warmup: int
    raw_config: Dict[str, Any]

    @classmethod
    def from_server_args(cls, server_args) -> "SimulatorSettings":
        if not getattr(server_args, "simulator", False):
            return cls(
                enabled=False,
                config_path=None,
                mode="offline",
                output_dir="/tmp/sglang_simulator/output",
                num_warmup=0,
                raw_config={},
            )

        config_path = getattr(server_args, "sim_config_path", None)
        if not config_path:
            raise ValueError("--sim-config-path is required when simulator mode is enabled.")
        if not os.path.exists(config_path):
            raise ValueError(f"Simulator config path does not exist: {config_path}")

        with open(config_path) as f:
            raw_config = json.load(f)

        mode = getattr(server_args, "simulator_mode", "offline").lower()
        if mode not in ("offline", "blocking"):
            raise ValueError(f"Unsupported simulator mode: {mode}")

        return cls(
            enabled=True,
            config_path=config_path,
            mode=mode,
            output_dir=getattr(
                server_args,
                "simulator_output_dir",
                "/tmp/sglang_simulator/output",
            ),
            num_warmup=int(getattr(server_args, "simulator_num_warmup", 0)),
            raw_config=raw_config,
        )


def set_active_settings(settings: SimulatorSettings | None) -> None:
    global _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = settings


def get_active_settings() -> SimulatorSettings | None:
    return _ACTIVE_SETTINGS


def get_platform_config() -> dict:
    settings = get_active_settings()
    if settings is not None:
        return settings.raw_config.get("platform", {})
    try:
        from sglang.srt.server_args import get_global_server_args

        config_path = getattr(get_global_server_args(), "sim_config_path", None)
        if not config_path or not os.path.exists(config_path):
            return {}
        with open(config_path) as f:
            return json.load(f).get("platform", {})
    except Exception:
        return {}
