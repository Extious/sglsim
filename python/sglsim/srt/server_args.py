from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ServerArgs:
    model_path: str = "simulator-model"
    sim_config_path: str | None = None
    simulator_mode: Literal["offline", "blocking"] = "offline"
    simulator_output_dir: str = "/tmp/sglsim/output"
    simulator_num_warmup: int = 0
    host: str = "127.0.0.1"
    port: int = 30000

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model-path", default=ServerArgs.model_path)
        parser.add_argument("--sim-config-path", required=True)
        parser.add_argument(
            "--simulator-mode",
            choices=["offline", "blocking"],
            default=ServerArgs.simulator_mode,
        )
        parser.add_argument("--simulator-output-dir", default=ServerArgs.simulator_output_dir)
        parser.add_argument("--simulator-num-warmup", type=int, default=0)
        parser.add_argument("--host", default=ServerArgs.host)
        parser.add_argument("--port", type=int, default=ServerArgs.port)

    @classmethod
    def from_cli_args(cls, args: list[str] | None = None) -> "ServerArgs":
        parser = argparse.ArgumentParser(description="SGLSim CPU-only simulator")
        cls.add_cli_args(parser)
        parsed = cls(**vars(parser.parse_args(args)))
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.sim_config_path is None:
            raise ValueError("--sim-config-path is required")
        if not Path(self.sim_config_path).exists():
            raise ValueError(f"Simulator config path does not exist: {self.sim_config_path}")
