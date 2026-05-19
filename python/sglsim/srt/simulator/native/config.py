from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeSimulationConfig:
    sim_config_path: str
    output_dir: str
    mode: str = "blocking"
    num_warmup: int = 0
    enable_native_runtime: bool = True

    @classmethod
    def from_server_args(cls, server_args) -> "NativeSimulationConfig":
        config_path = str(getattr(server_args, "sim_config_path", "") or "")
        output_dir = str(getattr(server_args, "simulator_output_dir", "/tmp/sglsim/output"))
        mode = str(getattr(server_args, "simulator_mode", "blocking") or "blocking")
        num_warmup = int(getattr(server_args, "simulator_num_warmup", 0) or 0)
        if config_path and not Path(config_path).is_file():
            raise ValueError(f"sim_config_path does not exist: {config_path}")
        if mode not in {"blocking", "offline"}:
            raise ValueError(f"unsupported simulator mode: {mode}")
        return cls(
            sim_config_path=config_path,
            output_dir=output_dir,
            mode=mode,
            num_warmup=num_warmup,
        )
