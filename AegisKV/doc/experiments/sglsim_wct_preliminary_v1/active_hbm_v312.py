"""Topology-aware v3.1.2 recovery contract for the S1/S2 campaign.

The published v3.1.1 implementation is intentionally left unchanged.  This
module fixes one demonstrated semantic problem: a source-node fault must not
hide a committed replica in another node behind a common replacement wait.
Failover activation is selected per request from the actual committed,
failure-domain-valid protection state.  Same-node Peer-HBM and local DRAM are
lost with the source node; cross-node Peer-HBM and remote DRAM survive.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import active_hbm_v31 as v31
import run_preliminary as core


def topology_placement(protocol: dict[str, Any], rank: int) -> dict[str, Any]:
    hardware = protocol["hardware"]
    gpus_per_node = int(hardware["gpus_per_node"])
    dp = int(protocol["models"]["qwen3_8b"]["center_system_dp"])
    if dp % gpus_per_node:
        raise ValueError("TP1/PP1 system DP must fill complete 8-GPU nodes")
    node_count = dp // gpus_per_node
    source_node = int(rank) // gpus_per_node
    local_gpu_index = int(rank) % gpus_per_node
    peer_mode, dram_mode = core.topology_for_rank(rank, protocol)
    peer_node = (
        source_node
        if peer_mode == "same_node"
        else (source_node + 1) % node_count
    )
    peer_local_gpu_index = (
        (local_gpu_index + 1) % gpus_per_node
        if peer_mode == "same_node"
        else local_gpu_index
    )
    dram_node = (
        source_node
        if dram_mode == "local"
        else (source_node + 1) % node_count
    )
    return {
        "rank": int(rank),
        "source_node": source_node,
        "local_gpu_index": local_gpu_index,
        "peer_hbm_mode": peer_mode,
        "peer_hbm_target_node": peer_node,
        "peer_hbm_target_rank": peer_node * gpus_per_node + peer_local_gpu_index,
        "dram_mode": dram_mode,
        "dram_target_node": dram_node,
    }


class TopologyAwareExactAdmissionProjectionReplay(
    v31.ExactAdmissionProjectionReplay
):
    """v3.1 exact admission plus per-replica node-failure activation."""

    def __init__(
        self, protocol: dict[str, Any], policy: str,
        capacity_protocol: dict[str, Any],
    ) -> None:
        super().__init__(protocol, policy, capacity_protocol)
        self.topology_rows = [topology_placement(protocol, rank) for rank in range(self.dp)]
        self.surviving_cross_domain_committed_count = 0
        self.node_replacement_no_survivor_count = 0
        self.node_prebound_survivor_count = 0

    def _profile(self, task: core.Task) -> core.RequestRecoveryProfile:
        placement = topology_placement(self.protocol, task.source_rank)
        slack = (
            0.0
            if self.policy == "AegisKV no Algorithm 1"
            else task.predicted_slack_ms
        )
        return core.RequestRecoveryProfile(
            request_id=task.request_id,
            source_worker=f"source-{task.source_rank}",
            source_failure_domain=f"node-{placement['source_node']}",
            hbm_target=(
                f"peer-node{placement['peer_hbm_target_node']}"
                f"-gpu{placement['local_gpu_index']}"
            ),
            hbm_failure_domain=f"node-{placement['peer_hbm_target_node']}",
            route_version=0,
            source_token_end=task.prompt_tokens,
            slack_ms=slack,
            model_revision="Qwen3-8B-frozen",
            kv_abi="qwen3-fp16-kv-v1",
            parallel_layout="TP1-PP1",
        )

    def _modeled_failover(
        self, probe: core.FaultProbe, *, surviving_committed: bool,
    ) -> tuple[float, str]:
        contract = self.protocol["failure_model"]["policy_failover"]
        if self.policy == "Native Mooncake":
            return float(contract["replacement_delay_ms"]), "replacement_wait"
        if probe.failure_type == "source_worker_gpu":
            return (
                float(contract["fence_activation_delay_ms"]),
                "prebound_fence_activation",
            )
        if probe.failure_type != "source_node":
            raise ValueError(f"unsupported failure type {probe.failure_type!r}")
        if surviving_committed:
            self.node_prebound_survivor_count += 1
            return (
                float(contract["fence_activation_delay_ms"]),
                "cross_domain_committed_fence_activation",
            )
        self.node_replacement_no_survivor_count += 1
        return float(contract["replacement_delay_ms"]), "replacement_no_survivor"

    def recovery(
        self, task: core.Task, probe: core.FaultProbe,
    ) -> tuple[float, str]:
        placement = topology_placement(self.protocol, task.source_rank)
        candidates: list[tuple[float, str, int, int]] = []
        surviving_committed = False
        recompute = core.RecoveryTiming(
            4.0, float(self.model["prefill_ms_per_token"])
        )
        candidates.append((
            recompute.setup_ms + task.prompt_tokens * recompute.ms_per_token,
            "RECOMPUTE",
            0,
            task.prompt_tokens,
        ))

        if self.is_aegis:
            controller = self.aegis[task.source_rank]
            hbm_prefix = controller.committed_prefix_tokens(
                task.request_id, core.ProtectionTier.HBM
            )
            dram_prefix = controller.committed_prefix_tokens(
                task.request_id, core.ProtectionTier.DRAM
            )
            node_failure = probe.failure_type == "source_node"
            hbm_valid = hbm_prefix > 0 and not (
                node_failure
                and placement["peer_hbm_target_node"] == placement["source_node"]
            )
            dram_valid = dram_prefix > 0 and not (
                node_failure
                and placement["dram_target_node"] == placement["source_node"]
            )
            hbm_load_bw, dram_load_bw = core.load_bandwidth(
                self.protocol, task.source_rank
            )
            if hbm_valid:
                value = (
                    0.08
                    + hbm_prefix * self.kv
                    / (hbm_load_bw * core.GB_PER_S_TO_BYTES_PER_MS)
                )
                suffix = task.prompt_tokens - hbm_prefix
                if suffix:
                    value += recompute.setup_ms + suffix * recompute.ms_per_token
                action = f"peer_hbm_{placement['peer_hbm_mode']}"
                if suffix:
                    action += "+RECOMPUTE"
                candidates.append((value, action, hbm_prefix, suffix))
            if dram_valid:
                value = (
                    0.15
                    + dram_prefix * self.kv
                    / (dram_load_bw * core.GB_PER_S_TO_BYTES_PER_MS)
                )
                suffix = task.prompt_tokens - dram_prefix
                if suffix:
                    value += recompute.setup_ms + suffix * recompute.ms_per_token
                action = f"dram_{placement['dram_mode']}"
                if suffix:
                    action += "+RECOMPUTE"
                candidates.append((value, action, dram_prefix, suffix))
            surviving_committed = hbm_valid or dram_valid
        else:
            for runtime in self.baselines[task.source_rank]:
                tier = runtime.config.tier.value
                target_node = (
                    placement["peer_hbm_target_node"]
                    if tier == "HBM"
                    else placement["dram_target_node"]
                )
                valid = not (
                    probe.failure_type == "source_node"
                    and target_node == placement["source_node"]
                )
                if not valid:
                    continue
                decision = runtime.estimate_source_failure(
                    task.request_id,
                    produced_tokens=task.prompt_tokens,
                    remaining_slack_ms=0.0,
                    time_ms=probe.time_ms,
                )
                mode = (
                    f"peer_hbm_{placement['peer_hbm_mode']}"
                    if tier == "HBM"
                    else f"dram_{placement['dram_mode']}"
                )
                action = mode if decision.restored_tokens else "RECOMPUTE"
                if decision.restored_tokens and decision.recompute_tokens:
                    action += "+RECOMPUTE"
                candidates.append((
                    decision.recovery_ms,
                    action,
                    int(decision.restored_tokens),
                    int(decision.recompute_tokens),
                ))
                surviving_committed |= bool(decision.restored_tokens)

        if probe.failure_type == "source_node" and surviving_committed:
            self.surviving_cross_domain_committed_count += 1
        failover_delay, failover_mode = self._modeled_failover(
            probe, surviving_committed=surviving_committed
        )
        value, action, restored, recomputed = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        self.failover_mode_counts[f"{probe.failure_type}:{failover_mode}"] += 1
        self.failover_delay_values_ms.append(failover_delay)
        self.recovery_service_values_ms.append(value)
        self.recovery_restored_tokens += int(restored)
        self.recovery_recomputed_tokens += int(recomputed)
        self.recovery_counts[action] += 1
        self.recovery_type_path_counts[f"{probe.failure_type}:{action}"] += 1
        return failover_delay + value, action

    def summarize_mechanism(self, control_horizon_ms: float) -> dict[str, Any]:
        mechanism = super().summarize_mechanism(control_horizon_ms)
        mechanism["active_hbm_v312"] = {
            "node_failure_semantics": (
                "same-node Peer-HBM/local DRAM invalid; committed cross-node "
                "Peer-HBM/remote DRAM enables modeled prebound activation"
            ),
            "topology_rows": self.topology_rows,
            "surviving_cross_domain_committed_count": (
                self.surviving_cross_domain_committed_count
            ),
            "node_prebound_survivor_count": self.node_prebound_survivor_count,
            "node_replacement_no_survivor_count": (
                self.node_replacement_no_survivor_count
            ),
            "copy_bandwidth_gb_s": dict(
                self.protocol["hardware"]["copy_bandwidth_gb_s"]
            ),
            "recovery_bandwidth_gb_s": dict(
                self.protocol["hardware"]["load_bandwidth_gb_s"]
            ),
        }
        return mechanism


def run_v312_cell(
    protocol: dict[str, Any], capacity_protocol: dict[str, Any], *,
    policy: str, seed: int, active_per_gpu: int, probe_count: int,
    output: Path, arrival_cutoff_ms: float,
) -> dict[str, Any]:
    """Run v3.1's audited output path with the v3.1.2 replay class.

    The scoped class replacement occurs inside one worker process and is
    restored even on failure.  It avoids mutating the hash-addressed v3.1.1
    source used by earlier result bundles.
    """
    original = v31.ExactAdmissionProjectionReplay
    v31.ExactAdmissionProjectionReplay = TopologyAwareExactAdmissionProjectionReplay
    try:
        cell = v31.run_v31_cell(
            protocol,
            capacity_protocol,
            policy=policy,
            seed=seed,
            active_per_gpu=active_per_gpu,
            probe_count=probe_count,
            output=output,
            arrival_cutoff_ms=arrival_cutoff_ms,
        )
    finally:
        v31.ExactAdmissionProjectionReplay = original
    cell["des_version"] = "active_hbm_v312_topology_aware_recovery"
    return cell
