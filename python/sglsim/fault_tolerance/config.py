from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast


Strategy = Literal["baseline", "remote_backup", "host_backup", "gpu_backup"]


@dataclass(frozen=True)
class FailureConfig:
    inject_after_job: str = ""
    inject_after_task: str = ""
    timeline_after_start_s: str = ""
    delay: int = 0
    dp_rank: str = "0"
    pp_rank: str = "0"
    tp_rank: str = "0"
    recovery_delay: str = ""
    recover_after_job: str = ""
    recover_after_task: str = ""
    fault_fraction: float = 0.35

    @classmethod
    def from_dict(cls, data: dict) -> "FailureConfig":
        return cls(
            inject_after_job=str(data.get("inject_after_job", "") or ""),
            inject_after_task=str(data.get("inject_after_task", "") or ""),
            timeline_after_start_s=str(data.get("timeline_after_start_s", "") or ""),
            delay=int(data.get("delay", 0) or 0),
            dp_rank=str(data.get("dp_rank", "0") or "0"),
            pp_rank=str(data.get("pp_rank", "0") or "0"),
            tp_rank=str(data.get("tp_rank", "0") or "0"),
            recovery_delay=str(data.get("recovery_delay", "") or ""),
            recover_after_job=str(data.get("recover_after_job", "") or ""),
            recover_after_task=str(data.get("recover_after_task", "") or ""),
            fault_fraction=float(data.get("fault_fraction", 0.35) or 0.35),
        )

    @property
    def trigger_kind(self) -> str:
        if self.inject_after_task:
            return "task"
        if self.timeline_after_start_s:
            return "timeline"
        return "job"

    @property
    def thresholds(self) -> list[int]:
        raw = self.inject_after_task or self.inject_after_job
        return [int(item.strip()) for item in raw.split(",") if item.strip()]

    @property
    def timeline_thresholds(self) -> list[float]:
        return [float(item.strip()) for item in self.timeline_after_start_s.split(",") if item.strip()]

    def recovery_job_threshold(self) -> int | None:
        return int(self.recover_after_job) if self.recover_after_job.isdigit() else None

    def recovery_task_threshold(self) -> int | None:
        return int(self.recover_after_task) if self.recover_after_task.isdigit() else None

    def recovery_delay_seconds(self) -> float:
        return float(self.recovery_delay) if self.recovery_delay else 0.0


@dataclass(frozen=True)
class HicacheStorageConfig:
    backend: str = ""
    prefetch_policy: str = ""
    remote_backup_port_base: int = 0
    remote_backup_buffer_size_gb: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "HicacheStorageConfig":
        return cls(
            backend=str(data.get("backend", "") or ""),
            prefetch_policy=str(data.get("prefetch_policy", "") or ""),
            remote_backup_port_base=int(data.get("remote_backup_port_base", 0) or 0),
            remote_backup_buffer_size_gb=float(data.get("remote_backup_buffer_size_gb", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class ServerConfig:
    dp_size: int = 2
    pp_size: int = 1
    tp_size: int = 1
    nnodes: int = 1
    hicache_size_gb: int = 0
    model_path: str = ""
    load_balance_method: str = ""
    enable_hicache: bool = False
    hicache_ratio: float = 0.0
    context_length: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "ServerConfig":
        return cls(
            dp_size=int(data.get("dp_size", 2) or 2),
            pp_size=int(data.get("pp_size", 1) or 1),
            tp_size=int(data.get("tp_size", 1) or 1),
            nnodes=int(data.get("nnodes", 1) or 1),
            hicache_size_gb=int(data.get("hicache_size_gb", 0) or 0),
            model_path=str(data.get("model_path", "") or ""),
            load_balance_method=str(data.get("load_balance_method", "") or ""),
            enable_hicache=bool(data.get("enable_hicache", False)),
            hicache_ratio=float(data.get("hicache_ratio", 0.0) or 0.0),
            context_length=int(data.get("context_length", 0) or 0),
        )


@dataclass(frozen=True)
class SimConfigRef:
    sim_config_path: str = ""
    peer_backup_lag_tokens: int = 8
    gpu_recovery_delay_s: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfigRef":
        return cls(
            sim_config_path=str(data.get("sim_config_path", "") or ""),
            peer_backup_lag_tokens=int(data.get("peer_backup_lag_tokens", 8) or 8),
            gpu_recovery_delay_s=float(data.get("gpu_recovery_delay_s", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class ServerExperimentConfig:
    name: str = "baseline"
    description: str = ""
    platform: str = ""
    kv_backup: str = "none"
    server: ServerConfig = ServerConfig()
    hicache_storage: HicacheStorageConfig = HicacheStorageConfig()
    sim: SimConfigRef = SimConfigRef()

    @classmethod
    def from_dict(cls, data: dict) -> "ServerExperimentConfig":
        return cls(
            name=str(data.get("name", "baseline") or "baseline"),
            description=str(data.get("description", "") or ""),
            platform=str(data.get("platform", "") or ""),
            kv_backup=str(data.get("kv_backup", "none") or "none"),
            server=ServerConfig.from_dict(data.get("server", {}) or {}),
            hicache_storage=HicacheStorageConfig.from_dict(data.get("hicache_storage", {}) or {}),
            sim=SimConfigRef.from_dict(data.get("sim", {}) or {}),
        )

    @property
    def strategy(self) -> Strategy:
        if self.kv_backup in ("remote_backup", "host_backup", "gpu_backup"):
            return cast(Strategy, self.kv_backup)
        if self.kv_backup == "host":
            return "host_backup"
        if self.kv_backup in ("none", "baseline"):
            return "baseline"
        return "baseline"

    def with_strategy(self, strategy: Strategy) -> "ServerExperimentConfig":
        kv_backup = "none" if strategy == "baseline" else strategy
        if strategy == "baseline":
            backend = ""
        elif strategy == "gpu_backup":
            backend = "gpu_backup"
        else:
            backend = strategy
        return replace(
            self,
            name=strategy,
            kv_backup=kv_backup,
            hicache_storage=replace(self.hicache_storage, backend=backend),
        )


@dataclass(frozen=True)
class FixedClientConfig:
    input_len: int = 1024
    output_len: int = 128
    num_requests: int = 64
    app_workers: int = 4
    shared_prefix_len: int = 0
    seed: int = 1
    ignore_eos: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "FixedClientConfig":
        fixed = data.get("fixed", {}) if "fixed" in data else data
        return cls(
            input_len=int(fixed.get("input_len", 1024) or 1024),
            output_len=int(fixed.get("output_len", 128) or 128),
            num_requests=int(fixed.get("num_requests", 64) or 64),
            app_workers=int(fixed.get("app_workers", 4) or 4),
            shared_prefix_len=int(fixed.get("shared_prefix_len", 0) or 0),
            seed=int(fixed.get("seed", 1) or 1),
            ignore_eos=int(fixed.get("ignore_eos", data.get("ignore_eos", 1)) or 1),
        )


@dataclass(frozen=True)
class CrewAIClientConfig:
    job_limit: int = 6
    app_workers: int = 2
    short_max_tokens: int = 1024
    long_max_tokens: int = 2048
    worker_start_stagger_s: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "CrewAIClientConfig":
        crew = data.get("crewai", {}) if "crewai" in data else data
        return cls(
            job_limit=int(crew.get("job_limit", 6) or 6),
            app_workers=int(crew.get("app_workers", 2) or 2),
            short_max_tokens=int(crew.get("short_max_tokens", 1024) or 1024),
            long_max_tokens=int(crew.get("long_max_tokens", 2048) or 2048),
            worker_start_stagger_s=float(crew.get("worker_start_stagger_s", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class ClientConfig:
    client_mode: str = "fixed"
    fixed: FixedClientConfig = FixedClientConfig()
    crewai: CrewAIClientConfig = CrewAIClientConfig()

    @classmethod
    def from_dict(cls, data: dict) -> "ClientConfig":
        return cls(
            client_mode=str(data.get("client_mode", "fixed") or "fixed"),
            fixed=FixedClientConfig.from_dict(data),
            crewai=CrewAIClientConfig.from_dict(data),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    server: ServerExperimentConfig
    client: ClientConfig
    failure: FailureConfig
    sim_config_path: str = ""

    def with_strategy(self, strategy: Strategy) -> "ExperimentConfig":
        return replace(self, server=self.server.with_strategy(strategy))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_experiment_config(path: str | Path, *, sim_config_path: str = "") -> ExperimentConfig:
    root = Path(path)
    server = ServerExperimentConfig.from_dict(_load_json(root / "server.json"))
    scenario_path = root / "scenario.json"
    scenario_sim = ""
    if scenario_path.is_file():
        scenario_sim = str(_load_json(scenario_path).get("sim_config_path", "") or "")
    resolved_sim = sim_config_path or server.sim.sim_config_path or scenario_sim
    return ExperimentConfig(
        server=server,
        client=ClientConfig.from_dict(_load_json(root / "client.json")),
        failure=FailureConfig.from_dict(_load_json(root / "failure.json")),
        sim_config_path=resolved_sim,
    )
