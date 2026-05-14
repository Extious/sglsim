from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ACTIVE_SETTINGS: "SimulatorSettings | None" = None


@dataclass(frozen=True)
class SimulatorSettings:
    enabled: bool
    config_path: str
    mode: str
    output_dir: str
    num_warmup: int
    raw_config: dict[str, Any]

    @classmethod
    def from_server_args(cls, server_args) -> "SimulatorSettings":
        config_path = getattr(server_args, "sim_config_path", None)
        if not config_path:
            raise ValueError("--sim-config-path is required")

        path = Path(config_path)
        if not path.exists():
            raise ValueError(f"Simulator config path does not exist: {config_path}")

        raw_config = json.loads(path.read_text())
        predictor_name = raw_config.get("predictor", {}).get("name", "aiconfigurator")
        if predictor_name != "aiconfigurator":
            raise ValueError("predictor.name must be aiconfigurator")

        mode = getattr(server_args, "simulator_mode", "offline").lower()
        if mode not in ("offline", "blocking"):
            raise ValueError(f"Unsupported simulator mode: {mode}")

        return cls(
            enabled=True,
            config_path=str(path),
            mode=mode,
            output_dir=getattr(server_args, "simulator_output_dir", "/tmp/sglsim/output"),
            num_warmup=int(getattr(server_args, "simulator_num_warmup", 0)),
            raw_config=raw_config,
        )


def set_active_settings(settings: SimulatorSettings | None) -> None:
    global _ACTIVE_SETTINGS
    _ACTIVE_SETTINGS = settings


def get_active_settings() -> SimulatorSettings | None:
    return _ACTIVE_SETTINGS


def get_platform_config() -> dict[str, Any]:
    if _ACTIVE_SETTINGS is None:
        return {}
    return _ACTIVE_SETTINGS.raw_config.get("platform", {})

