"""Translate deployment JSON into the complete metadata-DES configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sglsim.srt.simulation.aegis_complete import CompleteSimulationConfig, CompleteWorker
from sglsim.srt.simulation.aegis_full_design import AegisFullDesignConfig, CalibratedTransferPath
from sglsim.srt.simulation.aegis_independent_tiers import RecoveryTiming


GIB = 1024**3


@dataclass(frozen=True)
class DeploymentRuntime:
    model_name: str
    model: dict[str, Any]
    policy_name: str
    policy: dict[str, Any]
    effective_bandwidth_gbytes_per_s: float
    host_dram_gib_per_node: float
    workers: tuple[CompleteWorker, ...]
    simulation_config: CompleteSimulationConfig

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(worker.worker_id for worker in self.workers)


def build_deployment(
    deployment: Mapping[str, Any],
    failure: Mapping[str, Any],
    axes: Mapping[str, Any],
    *,
    seed: int,
) -> DeploymentRuntime:
    model_name = str(axes["model"])
    policy_name = str(axes["policy"])
    model = dict(deployment["models"][model_name])
    policy = dict(deployment["policies"][policy_name])
    cluster = deployment["cluster"]
    protection = deployment["protection"]
    bandwidth_gbytes_per_s = float(
        axes.get(
            "effective_bandwidth_GB_per_s",
            cluster["effective_cross_node_bandwidth_GB_per_s"],
        )
    )
    host_dram_gib = float(
        axes.get("host_dram_gib_per_node", cluster["aegiskv_host_dram_gib_per_node"])
    )
    if bandwidth_gbytes_per_s <= 0:
        raise ValueError("effective bandwidth must be positive")
    if host_dram_gib <= 0 or host_dram_gib > float(cluster["host_memory_gib_per_node"]):
        raise ValueError("invalid AegisKV host DRAM allocation")

    dp_size = int(model["dp_size"])
    node_count = int(cluster["node_count"])
    if dp_size % node_count:
        raise ValueError("DP replicas must divide evenly across nodes")
    replicas_per_node = dp_size // node_count
    workers = tuple(
        CompleteWorker(f"worker-{replica}", f"node-{replica // replicas_per_node}")
        for replica in range(dp_size)
    )

    bytes_per_ms = bandwidth_gbytes_per_s * 1_000_000.0
    kv_bytes = int(model["kv_bytes_per_token"])
    service = model["service_time"]
    design = AegisFullDesignConfig(
        hbm_capacity_bytes=int(float(model["hbm_reserve_gib_per_replica"]) * GIB),
        dram_capacity_bytes=int(host_dram_gib * node_count * GIB),
        kv_bytes_per_token=kv_bytes,
        hbm_copy=CalibratedTransferPath(
            float(protection["copy_setup_ms"]),
            bytes_per_ms,
            f"modeled-cross-node-peer-hbm-{bandwidth_gbytes_per_s:g}GBps",
        ),
        dram_copy=CalibratedTransferPath(
            float(protection["copy_setup_ms"]),
            bytes_per_ms,
            f"modeled-source-store-dram-{bandwidth_gbytes_per_s:g}GBps",
        ),
        recompute=RecoveryTiming(
            float(service["setup_ms"]),
            float(service["prefill_ms_per_uncached_token"]),
        ),
        hbm_load=RecoveryTiming(
            float(protection["load_setup_ms"]), kv_bytes / bytes_per_ms
        ),
        dram_load=RecoveryTiming(
            float(protection["load_setup_ms"]), kv_bytes / bytes_per_ms
        ),
        prefix_chunk_tokens=int(protection["prefix_chunk_tokens"]),
        decode_chunk_tokens=int(protection["decode_chunk_tokens"]),
        initial_hbm_alpha=float(protection["initial_hbm_alpha"]),
        minimum_hbm_alpha=float(protection["minimum_hbm_alpha"]),
        maximum_hbm_alpha=float(protection["maximum_hbm_alpha"]),
        aimd_decrease_factor=float(protection["aimd_decrease_factor"]),
        aimd_increase_step=float(protection["aimd_increase_step"]),
        aimd_stable_windows=int(protection["aimd_stable_windows"]),
        tp_size=int(model["tp_size"]),
        pp_size=1,
        event_log_limit=500_000,
    )
    fault = failure["fault_model"]
    repair_ms = float(axes.get("repair_time_ms", fault["repair_time_ms"]))
    sim_config = CompleteSimulationConfig(
        workers=workers,
        design=design,
        replacement_delay_ms=float(fault["replacement_delay_ms"]),
        queue_outage_ms=float(fault["queue_outage_ms"]),
        repair_time_ms=repair_ms,
        silent_replay_setup_ms=0.25,
        silent_replay_ms_per_token=0.0005,
        control_window_ms=float(protection["control_window_ms"]),
        primary_pressure_queue_depth=1,
        protection_mode=str(policy["protection_mode"]),
        recovery_priority=str(policy["recovery_priority"]),
        adaptive_hbm_quota=bool(policy["adaptive_hbm_quota"]),
        model_revision=str(model["model_id"]),
        kv_abi=f"bf16-kv-{kv_bytes}-bytes-per-token",
        parallel_layout=f"tp{model['tp_size']}-pp1-dp{model['dp_size']}",
        random_salt=f"aegiskv-paper-v2:{seed}:{policy_name}",
        event_log_limit=500_000,
    )
    return DeploymentRuntime(
        model_name=model_name,
        model=model,
        policy_name=policy_name,
        policy=policy,
        effective_bandwidth_gbytes_per_s=bandwidth_gbytes_per_s,
        host_dram_gib_per_node=host_dram_gib,
        workers=workers,
        simulation_config=sim_config,
    )


__all__ = ["DeploymentRuntime", "build_deployment"]
