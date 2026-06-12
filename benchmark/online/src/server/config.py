# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Server configuration dataclass mirroring config/<name>/server.json."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Corresponds to config/<name>/server.json."""
    dp_size: int = 2
    pp_size: int = 1
    tp_size: int = 1
    nnodes: int = 1
    hicache_size_gb: int = 0
    quantization: str = ""
    model_path: str = ""
    tool_call_parser: str = ""
    enable_cache_report: bool = False
    enable_hicache: bool = False
    hicache_ratio: float = 0.0
    hicache_storage_backend: str = ""
    hicache_storage_prefetch_policy: str = ""
    remote_backup_port_base: int = 0
    remote_backup_buffer_size_gb: float = 0.0
    # Authoritative runtime mode. Native uses SGLang-shaped sglsim runtime.
    runtime_mode: str = "native"
    # SGLSim --load-balance-method (e.g. total_tokens); empty = server default.
    load_balance_method: str = ""
    # SGLSim --reasoning-parser; empty = no override.
    reasoning_parser: str = ""
    # SGLang --context-length override; <=0 = use model derived
    context_length: int = 0
    # Raw JSON string forwarded to SGLSim --json-model-override-args.
    json_model_override_args: str = ""
    # If true, retry_queue and waiting_queue use the same scheduling priority.
    retry_queue_same_priority_as_waiting: bool = False
    # SGLSim --chat-template. May be relative to the benchmark src root.
    chat_template: str = ""
    # SGLSim latency predictor config.
    sim_config_path: str = ""
    simulator_mode: str = "blocking"
    simulator_num_warmup: int = 0
    peer_backup_lag_tokens: int = 8
    wait_timeout: int = 1800

    @classmethod
    def from_dict(cls, d: dict) -> ServerConfig:
        hs = d.get("hicache_storage", {})
        sim = d.get("simulator", {})
        if not isinstance(sim, dict):
            sim = {}
        aegiskv = d.get("aegiskv", {})
        if not isinstance(aegiskv, dict):
            aegiskv = {}
        return cls(
            dp_size=int(d.get("dp_size", 2)),
            pp_size=int(d.get("pp_size", 1)),
            tp_size=int(d.get("tp_size", 1)),
            nnodes=int(d.get("nnodes", 1)),
            hicache_size_gb=int(d.get("hicache_size_gb", 0)),
            quantization=str(d.get("quantization", "")),
            model_path=str(d.get("model_path", "")),
            tool_call_parser=str(d.get("tool_call_parser", "")),
            enable_cache_report=bool(d.get("enable_cache_report", False)),
            enable_hicache=bool(d.get("enable_hicache", False)),
            hicache_ratio=float(d.get("hicache_ratio", 0.0)),
            hicache_storage_backend=str(hs.get("backend", "")),
            hicache_storage_prefetch_policy=str(hs.get("prefetch_policy", "")),
            remote_backup_port_base=int(hs.get("remote_backup_port_base", 0)),
            remote_backup_buffer_size_gb=float(hs.get("remote_backup_buffer_size_gb", 0.0)),
            runtime_mode=str(d.get("runtime_mode", "native") or "native"),
            load_balance_method=str(d.get("load_balance_method", "")),
            reasoning_parser=str(d.get("reasoning_parser", "")),
            context_length=int(d.get("context_length", 0) or 0),
            json_model_override_args=str(d.get("json_model_override_args", "")),
            retry_queue_same_priority_as_waiting=bool(
                d.get("retry_queue_same_priority_as_waiting", False)
            ),
            chat_template=str(d.get("chat_template", "")),
            sim_config_path=str(
                sim.get("config_path", d.get("sim_config_path", ""))
            ),
            simulator_mode=str(
                sim.get("mode", d.get("simulator_mode", "blocking"))
            ),
            simulator_num_warmup=int(
                sim.get("num_warmup", d.get("simulator_num_warmup", 0)) or 0
            ),
            peer_backup_lag_tokens=int(
                aegiskv.get(
                    "peer_backup_lag_tokens",
                    sim.get("peer_backup_lag_tokens", d.get("peer_backup_lag_tokens", 8)),
                )
                or 8
            ),
            wait_timeout=int(d.get("wait_timeout", 1800) or 1800),
        )

    def build_deploy_flags(
        self,
        model_path: str,
        server_port: int,
        kv_backup: str,
        default_hicache_size_gb: int = 16,
        venv_dir: str = "",
    ) -> list[str]:
        flags = [
            "--dp-size", str(self.dp_size),
            "--pp-size", str(self.pp_size),
            "--tp-size", str(self.tp_size),
            "--model-path", model_path or "Qwen/Qwen3-8B",
            "--server-port-base", str(server_port),
            "--kv-backup", kv_backup,
            "--runtime-mode", self.runtime_mode,
        ]
        # Match deploy_server.py: explicit --hicache-size only when GB > 0; else --hicache-ratio.
        # Note: hicache_size_gb==0 must not fall back to a positive size here, or host pool stays
        # too small vs device_pool and HiRadixCache asserts (host must be strictly larger).
        # deploy_server.py only parses --hicache-size-gb (not --hicache-size). When using ratio,
        # pass --hicache-size-gb 0 so platform default (e.g. gpuhome 16) is overridden if --config
        # is missing.
        if self.hicache_size_gb > 0:
            flags.extend(["--hicache-size-gb", str(self.hicache_size_gb)])
        elif self.hicache_ratio > 0:
            flags.extend(
                ["--hicache-size-gb", "0", "--hicache-ratio", str(self.hicache_ratio)]
            )
        else:
            flags.extend(["--hicache-size-gb", str(default_hicache_size_gb)])
        if venv_dir:
            flags.extend(["--venv-dir", venv_dir])
        flags.extend(["--wait-timeout", str(self.wait_timeout)])
        if self.quantization:
            flags.extend(["--quantization", self.quantization])
        if self.hicache_storage_backend:
            flags.extend(["--hicache-storage-backend", self.hicache_storage_backend])
        if self.hicache_storage_prefetch_policy:
            flags.extend(["--hicache-storage-prefetch-policy", self.hicache_storage_prefetch_policy])
        if kv_backup == "remote_backup" and self.remote_backup_port_base > 0:
            flags.extend(["--remote-backup-port-base", str(self.remote_backup_port_base)])
            if self.remote_backup_buffer_size_gb > 0:
                flags.extend(["--remote-backup-buffer-size-gb",
                              str(self.remote_backup_buffer_size_gb)])
        if self.load_balance_method:
            flags.extend(["--load-balance-method", self.load_balance_method])
        if kv_backup == "gpu_backup":
            flags.extend(["--peer-backup-lag-tokens", str(self.peer_backup_lag_tokens)])
        if self.reasoning_parser:
            flags.extend(["--reasoning-parser", self.reasoning_parser])
        if self.context_length and self.context_length > 0:
            flags.extend(["--context-length", str(self.context_length)])
        if self.json_model_override_args and self.json_model_override_args != "{}":
            flags.extend(
                ["--json-model-override-args", self.json_model_override_args]
            )
        if self.chat_template:
            flags.extend(["--chat-template", self.chat_template])
        if self.retry_queue_same_priority_as_waiting:
            flags.extend(["--retry-queue-same-priority-as-waiting"])
        if self.sim_config_path:
            flags.extend(["--sim-config-path", self.sim_config_path])
            if self.simulator_mode:
                flags.extend(["--simulator-mode", self.simulator_mode])
            flags.extend(["--simulator-num-warmup", str(self.simulator_num_warmup)])
        return flags

    def topology_tag(self) -> str:
        return f"DP{self.dp_size}_PP{self.pp_size}_TP{self.tp_size}_N{self.nnodes}"
