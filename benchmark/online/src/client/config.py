"""Fixed workload client configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FixedClientConfig:
    """Fixed-length, non-agent client config from client.json -> fixed."""

    input_len: int = 1024
    output_len: int = 128
    num_requests: int = 64
    app_workers: int = 4
    seed: int = 1
    ignore_eos: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "FixedClientConfig":
        fixed = data.get("fixed", {})
        if not isinstance(fixed, dict):
            fixed = {}
        return cls(
            input_len=int(fixed.get("input_len", 1024)),
            output_len=int(fixed.get("output_len", 128)),
            num_requests=int(fixed.get("num_requests", 64)),
            app_workers=int(fixed.get("app_workers", 4)),
            seed=int(fixed.get("seed", 1)),
            ignore_eos=int(fixed.get("ignore_eos", data.get("ignore_eos", 1))),
        )


@dataclass
class FixedRunnerConfig:
    """Runtime parameters for the fixed-length workload runner."""

    server_url: str
    model_path: str
    input_len: int = 1024
    output_len: int = 128
    num_requests: int = 64
    app_workers: int = 4
    seed: int = 1
    ignore_eos: int = 1
    output_dir: Path = Path(".")
    control_url: str = ""

    def build_worker_flags(self) -> list[str]:
        flags = [
            "--server-url",
            self.server_url,
            "--model-path",
            self.model_path,
            "--input-len",
            str(self.input_len),
            "--output-len",
            str(self.output_len),
            "--num-requests",
            str(self.num_requests),
            "--app-workers",
            str(self.app_workers),
            "--seed",
            str(self.seed),
            "--ignore-eos",
            str(self.ignore_eos),
            "--output-dir",
            str(self.output_dir),
        ]
        if self.control_url:
            flags.extend(["--control-url", self.control_url])
        return flags


@dataclass(frozen=True)
class MultiAgentRunnerConfig:
    """Runtime parameters for the AegisKV multi-agent online worker."""

    server_url: str
    model_path: str
    workflow_config: dict
    output_dir: Path = Path(".")
    control_url: str = ""
    num_workflows: int = 0
    app_workers: int = 4
    seed: int = 1
    ignore_eos: int = 1
    output_len: int = 0

    def build_worker_flags(self) -> list[str]:
        flags = [
            "--server-url",
            self.server_url,
            "--model-path",
            self.model_path,
            "--workflow-config-json",
            json.dumps(self.workflow_config, separators=(",", ":")),
            "--output-dir",
            str(self.output_dir),
            "--app-workers",
            str(self.app_workers),
            "--seed",
            str(self.seed),
            "--ignore-eos",
            str(self.ignore_eos),
        ]
        if self.num_workflows > 0:
            flags.extend(["--num-workflows", str(self.num_workflows)])
        if self.output_len > 0:
            flags.extend(["--output-len", str(self.output_len)])
        if self.control_url:
            flags.extend(["--control-url", self.control_url])
        return flags
