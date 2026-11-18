#!/usr/bin/env python3
"""Small, auditable AegisKV workflow-completion-time DES campaign.

Unlike the older large-scale harm-surrogate campaign, this runner executes the
request lane queues and recomputes workflow completion after a fault.  The
fault-rate aggregation is deliberately sparse-rate/single-fault preliminary;
every one-fault sample itself is an executed queue/service/recovery replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PYTHON = REPO / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from sglsim.srt.simulation.aegis_baselines import (  # noqa: E402
    BaselinePolicy,
    BaselineRequestProfile,
    ProtectionBaselineConfig,
    ProtectionBaselineRuntime,
)
from sglsim.srt.simulation.aegis_full_design import (  # noqa: E402
    CalibratedTransferPath,
    SealedChunk,
)
from sglsim.srt.simulation.aegis_independent_tiers import (  # noqa: E402
    DESIGN_SEMANTICS,
    AegisIndependentTierController,
    ChunkSpec,
    CopyState,
    IndependentTierConfig,
    ProtectionTier,
    RecoveryTiming,
    RequestRecoveryProfile,
)


GIB = 1024**3
GB_PER_S_TO_BYTES_PER_MS = 1_000_000.0
PRIMARY_POLICIES = (
    "Native Mooncake",
    "Static FCFS",
    "DejaVu",
    "RAIDServe",
    "KevlarFlow",
    "GhostServe",
    "AegisKV",
)
ABLATION_POLICIES = (
    "AegisKV",
    "AegisKV no Algorithm 1",
    "AegisKV no Algorithm 2",
    "AegisKV no Algorithm 3",
)


@dataclass(frozen=True)
class Task:
    request_id: str
    workflow_id: str
    workflow_arrival_ms: float
    arrival_ms: float
    source_rank: int
    lane: int
    start_ms: float
    finish_ms: float
    service_ms: float
    prompt_tokens: int
    output_tokens: int
    predicted_slack_ms: float


@dataclass(frozen=True)
class FaultProbe:
    probe_id: str
    time_ms: float
    source_rank: int
    failure_type: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: Iterable[float], q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    index = (len(rows) - 1) * float(q)
    left = int(math.floor(index))
    right = int(math.ceil(index))
    if left == right:
        return rows[left]
    return rows[left] * (right - index) + rows[right] * (index - left)


def load_protocol() -> dict[str, Any]:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    if protocol["design_semantics"] != DESIGN_SEMANTICS:
        raise ValueError("protocol design semantics do not match runtime")
    return protocol


def topology_for_rank(
    rank: int, protocol: dict[str, Any] | None = None,
) -> tuple[str, str]:
    if protocol is not None:
        forced = protocol.get("runtime_overrides", {}).get("topology")
        if forced is not None:
            peer = str(forced["peer_hbm"])
            dram = str(forced["dram"])
            if peer not in {"same_node", "cross_node"}:
                raise ValueError(f"invalid forced Peer-HBM topology {peer!r}")
            if dram not in {"local", "remote"}:
                raise ValueError(f"invalid forced DRAM topology {dram!r}")
            return peer, dram
    return (
        ("same_node", "local")
        if int(rank) % 2 == 0
        else ("cross_node", "remote")
    )


def bandwidth(protocol: dict[str, Any], rank: int) -> tuple[float, float]:
    """Legacy shared copy/load bandwidth lookup.

    New mechanism-audit protocols provide phase-specific tables through
    :func:`copy_bandwidth` and :func:`load_bandwidth`.  Keep this helper for
    old protocols and external analysis code, but do not use it on a new
    source-copy or recovery critical path.
    """

    return load_bandwidth(protocol, rank)


def _phase_bandwidth(
    protocol: dict[str, Any], rank: int, *, phase: str,
) -> tuple[float, float]:
    peer, dram = topology_for_rank(rank, protocol)
    hardware = protocol["hardware"]
    key = f"{phase}_bandwidth_gb_s"
    table = hardware.get(key, hardware["bandwidth_gb_s"])
    return float(table[f"peer_hbm_{peer}"]), float(table[f"dram_{dram}"])


def copy_bandwidth(protocol: dict[str, Any], rank: int) -> tuple[float, float]:
    return _phase_bandwidth(protocol, rank, phase="copy")


def load_bandwidth(protocol: dict[str, Any], rank: int) -> tuple[float, float]:
    return _phase_bandwidth(protocol, rank, phase="load")


def policy_failover_delay_ms(
    protocol: dict[str, Any], policy: str, failure_type: str,
) -> tuple[float, str]:
    """Return the architecture-contract delay before KV recovery can start.

    Historical protocols have no policy contract and therefore retain the
    former common replacement delay.  New protocols must explicitly map each
    policy and failure type to either an already bound target activation or a
    replacement wait.  The modeled activation value is not a physical timing.
    """

    failure_model = protocol["failure_model"]
    contract = failure_model.get("policy_failover")
    if not contract:
        return float(failure_model["replacement_delay_ms"]), "legacy_common_replacement"
    policy_modes = contract["policy_modes"]
    if policy not in policy_modes:
        raise KeyError(f"missing failover contract for policy {policy!r}")
    entry = policy_modes[policy]
    mode = entry.get(failure_type) if isinstance(entry, dict) else entry
    if mode == "replacement_wait":
        return float(contract["replacement_delay_ms"]), mode
    if mode == "prebound_fence_activation":
        return float(contract["fence_activation_delay_ms"]), mode
    raise ValueError(
        f"invalid failover mode {mode!r} for {policy!r}/{failure_type!r}"
    )


def build_workload(
    protocol: dict[str, Any], *, seed: int, workflows: int | None,
    active_per_gpu: int, arrival_cutoff_ms: float | None = None,
) -> tuple[list[Task], dict[str, Any]]:
    rng = random.Random(int(seed) ^ 0xA3615C7)
    model = protocol["models"]["qwen3_8b"]
    workload = protocol["workload"]
    dp = int(model["center_system_dp"])
    branches = int(workload["requests_per_workflow"])
    lanes = int(workload["max_batch_lanes_per_gpu"])
    context = int(workload["center_context_tokens"])
    reuse_fraction = float(workload.get("reuse_fraction", 0.0))
    if not 0.0 <= reuse_fraction < 1.0:
        raise ValueError("reuse_fraction must be in [0, 1)")
    fresh_context = max(int(round(context * (1.0 - reuse_fraction))), 1)
    outputs = [int(value) for value in workload["output_tokens"]]
    output_weights = [float(value) for value in workload["output_weights"]]
    slowdown = 1.0 + 0.035 * max(int(active_per_gpu) - 1, 0)
    expected_output = sum(v * w for v, w in zip(outputs, output_weights))
    expected_service = slowdown * (
        float(model["prefill_setup_ms"])
        + fresh_context * float(model["prefill_ms_per_token"])
        + expected_output * float(model["decode_ms_per_token"])
    )
    # Little's-law offered stream.  Four simultaneous branches form one
    # workflow; lanes model SGLang continuous-batch service slots.
    mean_workflow_gap = (
        branches * expected_service / max(dp * int(active_per_gpu), 1)
    )
    lane_free = [[0.0] * lanes for _ in range(dp)]
    tasks: list[Task] = []
    workflow_arrivals = [0.0]
    if arrival_cutoff_ms is None:
        if workflows is None or int(workflows) < 1:
            raise ValueError("workflows must be positive without an arrival cutoff")
        while len(workflow_arrivals) < int(workflows):
            workflow_arrivals.append(
                workflow_arrivals[-1] + rng.expovariate(1.0 / mean_workflow_gap)
            )
    else:
        if float(arrival_cutoff_ms) <= 0:
            raise ValueError("arrival_cutoff_ms must be positive")
        while True:
            candidate = workflow_arrivals[-1] + rng.expovariate(1.0 / mean_workflow_gap)
            if candidate > float(arrival_cutoff_ms):
                break
            workflow_arrivals.append(candidate)
    for workflow_index, workflow_arrival in enumerate(workflow_arrivals):
        service_by_branch: list[float] = []
        selected_outputs: list[int] = []
        for _ in range(branches):
            output = rng.choices(outputs, weights=output_weights, k=1)[0]
            service = slowdown * (
                float(model["prefill_setup_ms"])
                + fresh_context * float(model["prefill_ms_per_token"])
                + output * float(model["decode_ms_per_token"])
            )
            service *= rng.uniform(0.96, 1.04)
            selected_outputs.append(output)
            service_by_branch.append(service)
        critical = max(service_by_branch)
        for branch, service in enumerate(service_by_branch):
            request_index = workflow_index * branches + branch
            rank = request_index % dp
            lane = min(range(lanes), key=lambda value: (lane_free[rank][value], value))
            start = max(workflow_arrival, lane_free[rank][lane])
            finish = start + service
            lane_free[rank][lane] = finish
            tasks.append(
                Task(
                    request_id=f"s{seed}-w{workflow_index:06d}-b{branch}",
                    workflow_id=f"s{seed}-w{workflow_index:06d}",
                    workflow_arrival_ms=workflow_arrival,
                    arrival_ms=workflow_arrival,
                    source_rank=rank,
                    lane=lane,
                    start_ms=start,
                    finish_ms=finish,
                    service_ms=service,
                    prompt_tokens=fresh_context,
                    output_tokens=selected_outputs[branch],
                    predicted_slack_ms=max(critical - service, 0.0),
                )
            )
    tasks.sort(key=lambda row: (row.start_ms, row.request_id))
    realized = realized_active(tasks, dp)
    return tasks, {
        "workflow_count": len(workflow_arrivals),
        "request_count": len(tasks),
        "target_active_requests_per_gpu": int(active_per_gpu),
        "realized_active_requests_per_gpu_time_mean": realized["mean"] / dp,
        "realized_active_requests_per_gpu_p95": realized["p95"] / dp,
        "realized_active_requests_per_gpu_peak": realized["peak"] / dp,
        "base_horizon_ms": max(row.finish_ms for row in tasks),
        "arrival_horizon_ms": (
            float(arrival_cutoff_ms)
            if arrival_cutoff_ms is not None
            else float(workflow_arrivals[-1])
        ),
        "actual_last_arrival_ms": float(workflow_arrivals[-1]),
        "fixed_arrival_cutoff": arrival_cutoff_ms is not None,
        "mean_workflow_gap_ms": float(mean_workflow_gap),
        "logical_context_tokens": context,
        "fresh_context_tokens": fresh_context,
        "reuse_fraction": reuse_fraction,
        "batch_lanes_per_gpu": lanes,
        "service_model": "A100 synthetic coefficient with fixed concurrency slowdown; not physical calibration",
    }


def realized_active(tasks: list[Task], dp: int) -> dict[str, float]:
    events = sorted(
        [*((row.start_ms, 1) for row in tasks), *((row.finish_ms, -1) for row in tasks)],
        key=lambda item: (item[0], item[1]),
    )
    active = 0
    prior = events[0][0]
    spans: list[tuple[int, float]] = []
    for at, delta in events:
        if at > prior:
            spans.append((active, at - prior))
        active += delta
        prior = at
    total = sum(span for _level, span in spans)
    weighted = sum(level * span for level, span in spans) / max(total, 1e-12)
    expanded = sorted(spans, key=lambda row: row[0])
    threshold = total * 0.95
    seen = 0.0
    p95 = 0.0
    for level, span in expanded:
        seen += span
        if seen >= threshold:
            p95 = float(level)
            break
    return {"mean": weighted, "p95": p95, "peak": float(max(level for level, _ in spans))}


def workflow_wct(tasks: list[Task], completion: dict[str, float] | None = None) -> tuple[list[str], list[float]]:
    arrivals: dict[str, float] = {}
    finishes: dict[str, float] = {}
    for row in tasks:
        arrivals[row.workflow_id] = min(arrivals.get(row.workflow_id, math.inf), row.workflow_arrival_ms)
        finish = row.finish_ms if completion is None else completion[row.request_id]
        finishes[row.workflow_id] = max(finishes.get(row.workflow_id, 0.0), finish)
    ids = sorted(arrivals)
    return ids, [finishes[key] - arrivals[key] for key in ids]


def make_fault_probes(
    tasks: list[Task], *, seed: int, count: int, gpus_per_node: int,
    right_ms: float | None = None,
    failure_types: list[str] | tuple[str, ...] | None = None,
) -> list[FaultProbe]:
    rng = random.Random(int(seed) ^ 0xFA017)
    left = min(row.start_ms for row in tasks)
    right = (
        float(right_ms)
        if right_ms is not None
        else max(row.finish_ms for row in tasks)
    )
    ranks = sorted({row.source_rank for row in tasks})
    scheduled_types = tuple(failure_types or ())
    if scheduled_types and any(
        failure_type not in {"source_worker_gpu", "source_node"}
        for failure_type in scheduled_types
    ):
        raise ValueError(f"invalid failure type schedule: {scheduled_types!r}")
    probes: list[FaultProbe] = []
    for index in range(int(count)):
        lo = left + (right - left) * index / count
        hi = left + (right - left) * (index + 1) / count
        at = rng.uniform(lo, hi)
        rank = rng.choice(ranks)
        failure_type = (
            scheduled_types[index % len(scheduled_types)]
            if scheduled_types
            else ("source_worker_gpu" if index % 2 == 0 else "source_node")
        )
        probes.append(FaultProbe(f"probe-{index:03d}", at, rank, failure_type))
    return sorted(probes, key=lambda row: row.time_ms)


def transfer_path(gb_s: float, name: str) -> CalibratedTransferPath:
    return CalibratedTransferPath(
        setup_ms=0.10,
        bandwidth_bytes_per_ms=float(gb_s) * GB_PER_S_TO_BYTES_PER_MS,
        provenance=f"protocol-modeled-{name}",
    )


def recovery_timing(gb_s: float, kv_bytes_per_token: int, setup_ms: float) -> RecoveryTiming:
    return RecoveryTiming(
        setup_ms=float(setup_ms),
        ms_per_token=float(kv_bytes_per_token) / (float(gb_s) * GB_PER_S_TO_BYTES_PER_MS),
    )


class ProtectionReplay:
    def __init__(self, protocol: dict[str, Any], policy: str) -> None:
        self.protocol = protocol
        self.policy = policy
        self.model = protocol["models"]["qwen3_8b"]
        self.dp = int(self.model["center_system_dp"])
        self.kv = int(self.model["kv_bytes_per_token"])
        self.aegis: list[AegisIndependentTierController] = []
        self.aegis_time = [0.0] * self.dp
        self.baselines: list[list[ProtectionBaselineRuntime]] = []
        self.aegis_specs: dict[tuple[str, int, ProtectionTier], ChunkSpec] = {}
        self.aegis_lane = defaultdict(float)
        self.active: set[str] = set()
        self.task_by_id: dict[str, Task] = {}
        self.path_counts: Counter[str] = Counter()
        self.residency_counts: Counter[str] = Counter()
        self.recovery_counts: Counter[str] = Counter()
        self.recovery_type_path_counts: Counter[str] = Counter()
        self.failover_mode_counts: Counter[str] = Counter()
        self.failover_delay_values_ms: list[float] = []
        self.recovery_service_values_ms: list[float] = []
        self.recovery_restored_tokens = 0
        self.recovery_recomputed_tokens = 0
        self.health_burden_ms: dict[str, float] = {}
        self.copy_service_ms: dict[str, float] = {}
        self.copy_queue_wait_ms: dict[str, float] = {}
        self.slack_nonzero_request_count = 0
        self.slack_used_total_ms = 0.0
        self.control_window_count = 0
        self.control_opportunity_times_ms: list[float] = []
        self.control_trace: list[dict[str, float]] = []
        self.control_target_evaluations = 0
        self.pressure_target_evaluations = 0
        self.occupancy_observation_horizon_ms: float | None = None
        self.occupancy_last_ms: list[float] = []
        self.occupancy_current_bytes: list[dict[str, int]] = []
        self.occupancy_total_bytes = {"HBM": 0, "DRAM": 0}
        self.occupancy_area_byte_ms = {"HBM": 0.0, "DRAM": 0.0}
        self.occupancy_peak_total_bytes = {"HBM": 0, "DRAM": 0}
        self.occupancy_trace: list[tuple[float, int, int]] = []
        self.events: list[tuple[float, int, int, str, Any]] = []
        self.sequence = 0
        self._build_runtimes()
        if self.is_aegis:
            self.occupancy_last_ms = [0.0] * self.dp
            self.occupancy_current_bytes = [
                {"HBM": 0, "DRAM": 0} for _ in range(self.dp)
            ]

    @property
    def is_aegis(self) -> bool:
        return self.policy in {
            "AegisKV",
            "AegisKV no Algorithm 1",
            "AegisKV no Algorithm 3",
        }

    def _build_runtimes(self) -> None:
        if self.is_aegis:
            settings = self.protocol["aegiskv"]
            for rank in range(self.dp):
                hbm_copy_bw, dram_copy_bw = copy_bandwidth(self.protocol, rank)
                hbm_load_bw, dram_load_bw = load_bandwidth(self.protocol, rank)
                self.aegis.append(
                    AegisIndependentTierController(
                        IndependentTierConfig(
                            hbm_capacity_bytes=int(40 * GIB),
                            dram_capacity_bytes=int(64 * GIB),
                            initial_hbm_alpha=float(settings["initial_alpha"]),
                            minimum_hbm_alpha=float(settings["minimum_alpha"]),
                            maximum_hbm_alpha=float(settings["maximum_alpha"]),
                            aimd_decrease_factor=float(settings["multiplicative_decrease"]),
                            aimd_increase_step=float(settings["additive_increase"]),
                            aimd_stable_windows=int(settings["stable_windows_before_increase"]),
                            recompute=RecoveryTiming(4.0, float(self.model["prefill_ms_per_token"])),
                            hbm_load=recovery_timing(hbm_load_bw, self.kv, 0.08),
                            dram_load=recovery_timing(dram_load_bw, self.kv, 0.15),
                            hbm_copy=recovery_timing(hbm_copy_bw, self.kv, 0.05),
                            dram_copy=recovery_timing(dram_copy_bw, self.kv, 0.10),
                            hbm_pool_id=f"peer-hbm-target-{rank}",
                            dram_pool_id=f"store-dram-target-{rank}",
                            dram_failure_domain=f"store-node-{rank // 8}",
                            retain_event_log=False,
                            retain_record_history=False,
                        )
                    )
                )
            return
        for rank in range(self.dp):
            hbm_copy_bw, dram_copy_bw = copy_bandwidth(self.protocol, rank)
            hbm_load_bw, dram_load_bw = load_bandwidth(self.protocol, rank)
            common = dict(
                hbm_capacity_bytes=int(10 * GIB),
                dram_capacity_bytes=int(64 * GIB),
                kv_bytes_per_token=self.kv,
                hbm_copy=transfer_path(hbm_copy_bw, "peer-hbm"),
                dram_copy=transfer_path(dram_copy_bw, "store-dram"),
                recompute=RecoveryTiming(4.0, float(self.model["prefill_ms_per_token"])),
                hbm_load=recovery_timing(hbm_load_bw, self.kv, 0.08),
                dram_load=recovery_timing(dram_load_bw, self.kv, 0.15),
                pp_size=1,
                tp_size=1,
                event_log_limit=0,
            )
            if self.policy == "Native Mooncake":
                kinds = [BaselinePolicy.RAIDSERVE_HOST]
            elif self.policy in {"Static FCFS", "AegisKV no Algorithm 2"}:
                kinds = [BaselinePolicy.DEJAVU_NEIGHBOR, BaselinePolicy.RAIDSERVE_HOST]
            else:
                kinds = [{
                    "DejaVu": BaselinePolicy.DEJAVU_NEIGHBOR,
                    "RAIDServe": BaselinePolicy.RAIDSERVE_HOST,
                    "KevlarFlow": BaselinePolicy.KEVLARFLOW_RING,
                    "GhostServe": BaselinePolicy.GHOSTSERVE_PARITY,
                }[self.policy]]
            self.baselines.append([
                ProtectionBaselineRuntime(
                    ProtectionBaselineConfig(policy=kind, **common),
                    target_id=f"target-{rank}-{kind.value}",
                )
                for kind in kinds
            ])

    def push(self, at: float, priority: int, kind: str, payload: Any) -> None:
        self.sequence += 1
        heapq.heappush(self.events, (float(at), int(priority), self.sequence, kind, payload))

    def advance_aegis(self, rank: int, at: float) -> None:
        self.integrate_occupancy(rank, at)
        delta = float(at) - self.aegis_time[rank]
        if delta < -1e-8:
            raise ValueError("controller time regressed")
        if delta > 0:
            self.aegis[rank].advance_time(delta)
            self.aegis_time[rank] = float(at)

    def integrate_occupancy(self, rank: int, at: float) -> None:
        if not self.is_aegis or self.occupancy_observation_horizon_ms is None:
            return
        capped = min(float(at), float(self.occupancy_observation_horizon_ms))
        prior = self.occupancy_last_ms[rank]
        if capped <= prior:
            return
        delta = capped - prior
        for tier in ("HBM", "DRAM"):
            self.occupancy_area_byte_ms[tier] += (
                self.occupancy_current_bytes[rank][tier] * delta
            )
        self.occupancy_last_ms[rank] = capped

    def refresh_occupancy(self, rank: int, at: float) -> None:
        if not self.is_aegis:
            return
        snapshot = self.aegis[rank].telemetry_snapshot()
        for tier in ("HBM", "DRAM"):
            previous = self.occupancy_current_bytes[rank][tier]
            current = int(snapshot["pools"][tier]["used_bytes"])
            self.occupancy_current_bytes[rank][tier] = current
            self.occupancy_total_bytes[tier] += current - previous
            self.occupancy_peak_total_bytes[tier] = max(
                self.occupancy_peak_total_bytes[tier],
                self.occupancy_total_bytes[tier],
            )
        if (
            bool(self.protocol.get("runtime_overrides", {}).get("record_occupancy_trace"))
            and self.occupancy_observation_horizon_ms is not None
            and float(at) <= self.occupancy_observation_horizon_ms + 1e-9
        ):
            self.occupancy_trace.append((
                float(at),
                self.occupancy_total_bytes["HBM"],
                self.occupancy_total_bytes["DRAM"],
            ))

    def start(self, task: Task, at: float) -> None:
        self.active.add(task.request_id)
        burden_contract = self.protocol["normal_path_copy_burden"]
        burden_mode = str(burden_contract.get("mode", "queue_tail"))
        if burden_mode not in {"queue_tail", "own_transfer_service"}:
            raise ValueError(f"invalid normal-path copy burden mode {burden_mode!r}")
        if self.is_aegis:
            rank = task.source_rank
            self.advance_aegis(rank, at)
            slack = 0.0 if self.policy == "AegisKV no Algorithm 1" else task.predicted_slack_ms
            if slack > 0.0:
                self.slack_nonzero_request_count += 1
                self.slack_used_total_ms += slack
            peer_mode, dram_mode = topology_for_rank(rank, self.protocol)
            profile = RequestRecoveryProfile(
                request_id=task.request_id,
                source_worker=f"source-{rank}",
                source_failure_domain=f"node-{rank // 8}",
                hbm_target=f"peer-{rank}-{peer_mode}",
                hbm_failure_domain=(f"node-{rank // 8}" if peer_mode == "same_node" else f"node-{(rank // 8 + 1) % 8}"),
                route_version=0,
                source_token_end=task.prompt_tokens,
                slack_ms=slack,
                model_revision="Qwen3-8B-frozen",
                kv_abi="qwen3-fp16-kv-v1",
                parallel_layout="TP1-PP1",
            )
            controller = self.aegis[rank]
            controller.register_request(profile)
            chunk_tokens = int(self.protocol["aegiskv"]["prefix_chunk_tokens"])
            copy_bw = copy_bandwidth(self.protocol, rank)
            own_service_by_tier = {
                ProtectionTier.HBM: 0.0,
                ProtectionTier.DRAM: 0.0,
            }
            queue_wait_by_tier = {
                ProtectionTier.HBM: 0.0,
                ProtectionTier.DRAM: 0.0,
            }
            cursor = 0
            ordinal = 0
            while cursor < task.prompt_tokens:
                end = min(cursor + chunk_tokens, task.prompt_tokens)
                spec = ChunkSpec(
                    request_id=task.request_id,
                    ordinal=ordinal,
                    token_start=cursor,
                    token_end=end,
                    byte_size=(end - cursor) * self.kv,
                    route_version=0,
                    source_worker=f"source-{rank}",
                    hbm_target=f"peer-{rank}-{peer_mode}",
                    model_revision="Qwen3-8B-frozen",
                    kv_abi="qwen3-fp16-kv-v1",
                    parallel_layout="TP1-PP1",
                )
                decision = controller.admit_chunk(spec)
                self.residency_counts[decision.intended_residency.value] += 1
                for tier, admitted, mode, setup in (
                    (ProtectionTier.HBM, decision.hbm.admitted, f"peer_hbm_{peer_mode}", 0.05),
                    (ProtectionTier.DRAM, decision.dram.admitted, f"dram_{dram_mode}", 0.10),
                ):
                    if not admitted:
                        continue
                    bw = copy_bw[0 if tier is ProtectionTier.HBM else 1]
                    duration = setup + spec.byte_size / (bw * GB_PER_S_TO_BYTES_PER_MS)
                    key = (rank, tier)
                    lane_before = self.aegis_lane[key]
                    queue_wait_by_tier[tier] = max(
                        queue_wait_by_tier[tier], max(lane_before - at, 0.0)
                    )
                    own_service_by_tier[tier] += duration
                    done = max(at, lane_before) + duration
                    self.aegis_lane[key] = done
                    self.aegis_specs[(task.request_id, ordinal, tier)] = spec
                    self.path_counts[f"copy_{mode}"] += 1
                    self.push(done, 0, "aegis_copy", (rank, task.request_id, ordinal, tier))
                cursor = end
                ordinal += 1
            critical_done = max(
                (
                    self.aegis_lane[(rank, tier)]
                    for tier in (ProtectionTier.HBM, ProtectionTier.DRAM)
                ),
                default=at,
            )
            own_copy_critical = max(own_service_by_tier.values(), default=0.0)
            queue_tail_critical = max(critical_done - at, 0.0)
            selected_critical = (
                own_copy_critical
                if burden_mode == "own_transfer_service"
                else queue_tail_critical
            )
            self.copy_service_ms[task.request_id] = own_copy_critical
            self.copy_queue_wait_ms[task.request_id] = max(
                queue_wait_by_tier.values(), default=0.0
            )
            self.health_burden_ms[task.request_id] = self.visible_copy_burden(
                task, selected_critical
            )
            self.refresh_occupancy(rank, at)
            return
        profile = BaselineRequestProfile(
            request_id=task.request_id,
            source_worker=f"source-{task.source_rank}",
            source_failure_domain=f"node-{task.source_rank // 8}",
            target_worker=f"target-{task.source_rank}",
            target_failure_domain=f"target-node-{task.source_rank // 8}",
            route_version=0,
            total_tokens=task.prompt_tokens,
            slack_ms=task.predicted_slack_ms,
        )
        chunk = SealedChunk(0, 0, task.prompt_tokens, "prefill")
        own_service: list[float] = []
        queue_wait: list[float] = []
        for runtime in self.baselines[task.source_rank]:
            runtime.register_request(profile, time_ms=at)
            lane_before = runtime.lane_available_ms
            admission = runtime.seal_chunk(task.request_id, chunk, time_ms=at)
            if admission.admitted:
                own_service.append(
                    runtime.lane_available_ms - max(at, lane_before)
                )
                queue_wait.append(max(lane_before - at, 0.0))
                peer_mode, dram_mode = topology_for_rank(task.source_rank, self.protocol)
                mode = f"peer_hbm_{peer_mode}" if admission.tier.value == "HBM" else f"dram_{dram_mode}"
                self.path_counts[f"copy_{mode}"] += 1
        critical_done = max(
            (runtime.lane_available_ms for runtime in self.baselines[task.source_rank]),
            default=at,
        )
        own_copy_critical = max(own_service, default=0.0)
        queue_tail_critical = max(critical_done - at, 0.0)
        selected_critical = (
            own_copy_critical
            if burden_mode == "own_transfer_service"
            else queue_tail_critical
        )
        self.copy_service_ms[task.request_id] = own_copy_critical
        self.copy_queue_wait_ms[task.request_id] = max(queue_wait, default=0.0)
        self.health_burden_ms[task.request_id] = self.visible_copy_burden(
            task, selected_critical
        )

    def visible_copy_burden(self, task: Task, copy_critical_ms: float) -> float:
        contract = self.protocol["normal_path_copy_burden"]
        return min(
            float(copy_critical_ms)
            * float(contract["visible_fraction_of_copy_critical_path"]),
            task.service_ms
            * float(contract["maximum_fraction_of_request_service"]),
        )

    def finish(self, task: Task, at: float) -> None:
        if task.request_id not in self.active:
            return
        self.active.remove(task.request_id)
        if self.is_aegis:
            self.advance_aegis(task.source_rank, at)
            self.aegis[task.source_rank].complete_request(task.request_id, force_abort_pending=True)
            self.refresh_occupancy(task.source_rank, at)
        else:
            for runtime in self.baselines[task.source_rank]:
                if runtime.has_request(task.request_id):
                    runtime.finish_request(task.request_id, time_ms=at)

    def complete_copy(self, payload: tuple[int, str, int, ProtectionTier], at: float) -> None:
        rank, request_id, ordinal, tier = payload
        if request_id not in self.active:
            return
        self.advance_aegis(rank, at)
        controller = self.aegis[rank]
        try:
            if controller.copy_state(request_id, ordinal, tier) is not CopyState.PENDING:
                return
            spec = self.aegis_specs[(request_id, ordinal, tier)]
            controller.complete_copy(tier, request_id, ordinal, controller.completion_proof(spec, tier))
            self.refresh_occupancy(rank, at)
        except KeyError:
            return

    def control(
        self, at: float, active_by_rank: dict[int, int],
        external_pressure_by_rank: dict[int, bool] | None = None,
    ) -> None:
        if not self.is_aegis or self.policy == "AegisKV no Algorithm 3":
            return
        self.control_window_count += 1
        self.control_opportunity_times_ms.append(float(at))
        pressure_count = 0
        external_pressure_count = 0
        alpha_after: list[float] = []
        hbm_used = 0
        dram_used = 0
        for rank, controller in enumerate(self.aegis):
            self.advance_aegis(rank, at)
            snap = controller.telemetry_snapshot()
            pool = snap["pools"]["HBM"]
            occupancy = int(pool["used_bytes"]) / max(int(pool["limit_bytes"]), 1)
            external_pressure = bool(
                (external_pressure_by_rank or {}).get(rank, False)
            )
            external_pressure_count += int(external_pressure)
            pressure = (
                external_pressure
                or active_by_rank.get(rank, 0) > 2
                or occupancy >= 0.95
            )
            self.control_target_evaluations += 1
            self.pressure_target_evaluations += int(pressure)
            pressure_count += int(pressure)
            controller.control_hbm_quota(pressure=pressure)
            self.refresh_occupancy(rank, at)
            post = controller.telemetry_snapshot()
            alpha_after.append(float(post["hbm_alpha"]))
            hbm_used += int(post["pools"]["HBM"]["used_bytes"])
            dram_used += int(post["pools"]["DRAM"]["used_bytes"])
        self.control_trace.append({
            "time_ms": float(at),
            "alpha_after_mean": statistics.fmean(alpha_after),
            "pressure_fraction": pressure_count / self.dp,
            "external_memory_pressure_fraction": external_pressure_count / self.dp,
            "active_requests_per_gpu": sum(active_by_rank.values()) / self.dp,
            "hbm_used_bytes_total": float(hbm_used),
            "dram_used_bytes_total": float(dram_used),
        })

    def _aegis_recovery(
        self, task: Task, probe: FaultProbe,
    ) -> tuple[float, str, int, int]:
        rank = task.source_rank
        controller = self.aegis[rank]
        hbm_prefix = controller.committed_prefix_tokens(task.request_id, ProtectionTier.HBM)
        dram_prefix = controller.committed_prefix_tokens(task.request_id, ProtectionTier.DRAM)
        peer_mode, dram_mode = topology_for_rank(rank, self.protocol)
        node_failure = probe.failure_type == "source_node"
        hbm_valid = hbm_prefix > 0 and not (node_failure and peer_mode == "same_node")
        dram_valid = dram_prefix > 0 and not (node_failure and dram_mode == "local")
        recompute = RecoveryTiming(4.0, float(self.model["prefill_ms_per_token"]))
        candidates = [(
            recompute.setup_ms + task.prompt_tokens * recompute.ms_per_token,
            "RECOMPUTE",
            0,
            task.prompt_tokens,
        )]
        hbm_load_bw, dram_load_bw = load_bandwidth(self.protocol, rank)
        if hbm_valid:
            value = 0.08 + hbm_prefix * self.kv / (hbm_load_bw * GB_PER_S_TO_BYTES_PER_MS)
            suffix = task.prompt_tokens - hbm_prefix
            if suffix:
                value += recompute.setup_ms + suffix * recompute.ms_per_token
            action = f"peer_hbm_{peer_mode}"
            if suffix:
                action += "+RECOMPUTE"
            candidates.append((value, action, hbm_prefix, suffix))
        if dram_valid:
            value = 0.15 + dram_prefix * self.kv / (dram_load_bw * GB_PER_S_TO_BYTES_PER_MS)
            suffix = task.prompt_tokens - dram_prefix
            if suffix:
                value += recompute.setup_ms + suffix * recompute.ms_per_token
            action = f"dram_{dram_mode}"
            if suffix:
                action += "+RECOMPUTE"
            candidates.append((value, action, dram_prefix, suffix))
        return min(candidates, key=lambda row: (row[0], row[1]))

    def recovery(self, task: Task, probe: FaultProbe) -> tuple[float, str]:
        failover_delay, failover_mode = policy_failover_delay_ms(
            self.protocol, self.policy, probe.failure_type
        )
        if self.is_aegis:
            value, action, restored, recomputed = self._aegis_recovery(task, probe)
        else:
            candidates: list[tuple[float, str, int, int]] = []
            peer_mode, dram_mode = topology_for_rank(task.source_rank, self.protocol)
            for runtime in self.baselines[task.source_rank]:
                tier = runtime.config.tier.value
                valid = not (
                    probe.failure_type == "source_node"
                    and ((tier == "HBM" and peer_mode == "same_node") or (tier == "DRAM" and dram_mode == "local"))
                )
                if not valid:
                    continue
                decision = runtime.estimate_source_failure(
                    task.request_id,
                    produced_tokens=task.prompt_tokens,
                    remaining_slack_ms=0.0,
                    time_ms=probe.time_ms,
                )
                mode = f"peer_hbm_{peer_mode}" if tier == "HBM" else f"dram_{dram_mode}"
                action = mode if decision.restored_tokens else "RECOMPUTE"
                if decision.restored_tokens and decision.recompute_tokens:
                    action += "+RECOMPUTE"
                candidates.append((
                    decision.recovery_ms,
                    action,
                    int(decision.restored_tokens),
                    int(decision.recompute_tokens),
                ))
            if not candidates:
                value = 4.0 + task.prompt_tokens * float(self.model["prefill_ms_per_token"])
                action = "RECOMPUTE"
                restored = 0
                recomputed = task.prompt_tokens
            else:
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

    def run(
        self, tasks: list[Task], probes: list[FaultProbe],
        *, control_horizon_ms: float | None = None,
    ) -> tuple[list[dict[str, float]], dict[str, float], dict[str, Any]]:
        self.task_by_id = {row.request_id: row for row in tasks}
        self.occupancy_observation_horizon_ms = (
            float(control_horizon_ms)
            if control_horizon_ms is not None
            else max(row.finish_ms for row in tasks)
        )
        if self.is_aegis and bool(
            self.protocol.get("runtime_overrides", {}).get("record_occupancy_trace")
        ):
            self.occupancy_trace = [(0.0, 0, 0)]
        active_by_rank: dict[int, set[str]] = defaultdict(set)
        recoveries: list[dict[str, float]] = []
        for row in tasks:
            self.push(row.start_ms, 1, "start", row)
            self.push(row.finish_ms, 4, "finish", row)
        for probe in probes:
            self.push(probe.time_ms, 2, "probe", probe)
        horizon = (
            float(control_horizon_ms)
            if control_horizon_ms is not None
            else max(row.finish_ms for row in tasks)
        )
        window = float(self.protocol["aegiskv"]["window_ms"])
        at = window
        while at <= horizon + 1e-9:
            self.push(at, 3, "control", None)
            at += window
        while self.events:
            at, _priority, _seq, kind, payload = heapq.heappop(self.events)
            if kind == "aegis_copy":
                self.complete_copy(payload, at)
            elif kind == "start":
                task = payload
                active_by_rank[task.source_rank].add(task.request_id)
                self.start(task, at)
            elif kind == "finish":
                task = payload
                active_by_rank[task.source_rank].discard(task.request_id)
                self.finish(task, at)
            elif kind == "control":
                self.control(at, {rank: len(rows) for rank, rows in active_by_rank.items()})
            elif kind == "probe":
                probe = payload
                ranks = [probe.source_rank]
                if probe.failure_type == "source_node":
                    base = (probe.source_rank // 8) * 8
                    ranks = list(range(base, base + 8))
                row: dict[str, float] = {}
                for rank in ranks:
                    for request_id in sorted(active_by_rank.get(rank, ())):
                        task = self.task_by_id[request_id]
                        delay, _action = self.recovery(task, probe)
                        row[request_id] = delay
                recoveries.append(row)
        if self.is_aegis:
            for rank in range(self.dp):
                self.integrate_occupancy(
                    rank, float(self.occupancy_observation_horizon_ms)
                )
        snapshots = [controller.telemetry_snapshot() for controller in self.aegis] if self.is_aegis else []
        alpha = [float(row["hbm_alpha"]) for row in snapshots]
        burden_values = list(self.health_burden_ms.values())
        copy_service_values = list(self.copy_service_ms.values())
        copy_queue_wait_values = list(self.copy_queue_wait_ms.values())
        tier_churn = {}
        for tier in ("HBM", "DRAM"):
            rows = [snapshot["pools"][tier]["churn"] for snapshot in snapshots]
            tier_churn[tier] = {
                key: sum(int(row[key]) for row in rows)
                for key in (
                    "admission_attempt_count", "admission_success_count",
                    "admission_rejection_count", "admitted_bytes",
                    "eviction_count", "evicted_bytes", "removal_count",
                    "removed_bytes",
                )
            } if rows else {}
        occupancy_horizon = float(self.occupancy_observation_horizon_ms or 0.0)
        return recoveries, dict(self.health_burden_ms), {
            "path_event_counts": dict(sorted(self.path_counts.items())),
            "residency_event_counts": dict(sorted(self.residency_counts.items())),
            "recovery_path_counts": dict(sorted(self.recovery_counts.items())),
            "recovery_type_path_counts": dict(
                sorted(self.recovery_type_path_counts.items())
            ),
            "failover_mode_counts": dict(sorted(self.failover_mode_counts.items())),
            "failover_delay_mean_ms": (
                statistics.fmean(self.failover_delay_values_ms)
                if self.failover_delay_values_ms else None
            ),
            "recovery_service_mean_ms": (
                statistics.fmean(self.recovery_service_values_ms)
                if self.recovery_service_values_ms else None
            ),
            "recovery_restored_tokens": self.recovery_restored_tokens,
            "recovery_recomputed_tokens": self.recovery_recomputed_tokens,
            "alpha_final_mean": statistics.fmean(alpha) if alpha else None,
            "alpha_final_min": min(alpha) if alpha else None,
            "target_owned_copy_events": 0,
            "hbm_to_dram_copy_events": 0,
            "normal_path_copy_burden_mean_ms": (
                statistics.fmean(burden_values) if burden_values else 0.0
            ),
            "normal_path_copy_burden_p95_ms": quantile(burden_values, 0.95),
            "copy_transfer_service_mean_ms": (
                statistics.fmean(copy_service_values)
                if copy_service_values else 0.0
            ),
            "copy_transfer_service_p95_ms": quantile(copy_service_values, 0.95),
            "copy_queue_wait_mean_ms": (
                statistics.fmean(copy_queue_wait_values)
                if copy_queue_wait_values else 0.0
            ),
            "copy_queue_wait_p95_ms": quantile(copy_queue_wait_values, 0.95),
            "copy_queue_wait_positive_count": sum(
                value > 0.0 for value in copy_queue_wait_values
            ),
            "algorithm1_nonzero_slack_request_count": self.slack_nonzero_request_count,
            "algorithm1_total_slack_used_ms": self.slack_used_total_ms,
            "normal_path_copy_burden_contract": dict(
                self.protocol["normal_path_copy_burden"]
            ),
            "phase_specific_bandwidth_contract": {
                "copy_bandwidth_gb_s": dict(
                    self.protocol["hardware"].get(
                        "copy_bandwidth_gb_s",
                        self.protocol["hardware"]["bandwidth_gb_s"],
                    )
                ),
                "load_bandwidth_gb_s": dict(
                    self.protocol["hardware"].get(
                        "load_bandwidth_gb_s",
                        self.protocol["hardware"]["bandwidth_gb_s"],
                    )
                ),
            },
            "control_window_count": self.control_window_count,
            "control_opportunity_times_ms": self.control_opportunity_times_ms,
            "control_target_evaluations": self.control_target_evaluations,
            "pressure_target_evaluations": self.pressure_target_evaluations,
            "pressure_target_evaluation_fraction": (
                self.pressure_target_evaluations / self.control_target_evaluations
                if self.control_target_evaluations else None
            ),
            "tier_churn": tier_churn,
            "occupancy_observation_horizon_ms": occupancy_horizon,
            "capacity_integral_byte_ms": dict(self.occupancy_area_byte_ms),
            "capacity_integral_gib_seconds_total": {
                tier: value / GIB / 1000.0
                for tier, value in self.occupancy_area_byte_ms.items()
            },
            "occupancy_time_mean_gib_total": {
                tier: value / max(occupancy_horizon, 1e-12) / GIB
                for tier, value in self.occupancy_area_byte_ms.items()
            },
            "occupancy_time_mean_gib_per_gpu": {
                tier: value / max(occupancy_horizon, 1e-12) / GIB / self.dp
                for tier, value in self.occupancy_area_byte_ms.items()
            },
            "occupancy_peak_gib_total": {
                tier: value / GIB
                for tier, value in self.occupancy_peak_total_bytes.items()
            },
            "occupancy_peak_gib_per_gpu": {
                tier: value / GIB / self.dp
                for tier, value in self.occupancy_peak_total_bytes.items()
            },
            "_occupancy_trace": self.occupancy_trace,
            "_control_trace": self.control_trace,
        }


def replay_one_fault(tasks: list[Task], delays: dict[str, float]) -> dict[str, float]:
    lanes: dict[tuple[int, int], list[Task]] = defaultdict(list)
    for row in tasks:
        lanes[(row.source_rank, row.lane)].append(row)
    completion: dict[str, float] = {}
    for rows in lanes.values():
        available = 0.0
        for row in sorted(rows, key=lambda item: (item.start_ms, item.request_id)):
            # Preserve all already-executed no-fault gating (serving lanes,
            # protection burden, and v3 shared-HBM admission) while allowing a
            # fault delay to propagate down the same serving lane.  Replaying
            # from arrival would erase memory-admission gaps just as the old
            # service-only replay erased copy burden.
            start = max(row.start_ms, available)
            # ``tasks`` is already the protection-aware serving schedule.  Its
            # start-to-finish duration includes normal-path copy/fabric burden;
            # using nominal ``service_ms`` here silently erased that burden in
            # every fault replay, including an empty-delay replay.
            scheduled_duration = row.finish_ms - row.start_ms
            if scheduled_duration < -1e-9:
                raise AssertionError("scheduled request duration became negative")
            finish = start + scheduled_duration + float(delays.get(row.request_id, 0.0))
            completion[row.request_id] = finish
            available = finish
    return completion


def reschedule_with_burden(tasks: list[Task], burdens: dict[str, float]) -> list[Task]:
    lanes: dict[tuple[int, int], list[Task]] = defaultdict(list)
    for row in tasks:
        lanes[(row.source_rank, row.lane)].append(row)
    adjusted: list[Task] = []
    for rows in lanes.values():
        available = 0.0
        for row in sorted(rows, key=lambda item: (item.start_ms, item.request_id)):
            start = max(row.arrival_ms, available)
            finish = start + row.service_ms + float(burdens.get(row.request_id, 0.0))
            adjusted.append(replace(row, start_ms=start, finish_ms=finish))
            available = finish
    return sorted(adjusted, key=lambda row: (row.start_ms, row.request_id))


def solve_schedule_with_protection_burden(
    protocol: dict[str, Any], policy: str, nominal_tasks: list[Task],
    *, control_horizon_ms: float | None,
) -> tuple[list[Task], dict[str, Any]]:
    """Solve the serving/protection schedule coupling declared by the protocol.

    Legacy protocols retain their historical one-pass behavior.  A mechanism
    audit can explicitly require a fixed point, preventing the final serving
    schedule from being paired with burdens computed from a stale schedule.
    """

    contract = protocol["normal_path_copy_burden"]
    fixed = contract.get("schedule_fixed_point")
    if not fixed or not bool(fixed.get("enabled", False)):
        _delays, burdens, _mechanism = ProtectionReplay(protocol, policy).run(
            nominal_tasks, [], control_horizon_ms=control_horizon_ms,
        )
        del _delays, _mechanism
        return reschedule_with_burden(nominal_tasks, burdens), {
            "enabled": False,
            "converged": None,
            "iterations": 1,
            "residual_finish_ms": None,
        }
    maximum_iterations = int(fixed["maximum_iterations"])
    tolerance_ms = float(fixed["tolerance_ms"])
    if maximum_iterations < 1 or tolerance_ms < 0.0:
        raise ValueError("invalid normal-path burden fixed-point contract")
    tasks = list(nominal_tasks)
    residual = math.inf
    for iteration in range(1, maximum_iterations + 1):
        _delays, burdens, _mechanism = ProtectionReplay(protocol, policy).run(
            tasks, [], control_horizon_ms=control_horizon_ms,
        )
        del _delays, _mechanism
        updated = reschedule_with_burden(nominal_tasks, burdens)
        previous = {row.request_id: row.finish_ms for row in tasks}
        residual = max(
            abs(row.finish_ms - previous[row.request_id]) for row in updated
        )
        tasks = updated
        if residual <= tolerance_ms:
            return tasks, {
                "enabled": True,
                "converged": True,
                "iterations": iteration,
                "residual_finish_ms": residual,
                "tolerance_ms": tolerance_ms,
                "maximum_iterations": maximum_iterations,
            }
    return tasks, {
        "enabled": True,
        "converged": False,
        "iterations": maximum_iterations,
        "residual_finish_ms": residual,
        "tolerance_ms": tolerance_ms,
        "maximum_iterations": maximum_iterations,
    }


def run_mechanism_gate(protocol: dict[str, Any]) -> dict[str, Any]:
    model = protocol["models"]["qwen3_8b"]
    rows = []
    for index, phase in enumerate(protocol["mechanism_gate"]["fixed_tier_limit_phases"]):
        controller = AegisIndependentTierController(
            IndependentTierConfig(
                hbm_capacity_bytes=int(40 * GIB),
                dram_capacity_bytes=int(64 * GIB),
                initial_hbm_alpha=0.25,
                minimum_hbm_alpha=0.0,
                maximum_hbm_alpha=0.3,
                recompute=RecoveryTiming(4.0, float(model["prefill_ms_per_token"])),
                hbm_load=recovery_timing(50.0, int(model["kv_bytes_per_token"]), 0.08),
                dram_load=recovery_timing(16.0, int(model["kv_bytes_per_token"]), 0.15),
                hbm_copy=recovery_timing(50.0, int(model["kv_bytes_per_token"]), 0.05),
                dram_copy=recovery_timing(16.0, int(model["kv_bytes_per_token"]), 0.10),
                retain_event_log=False,
                retain_record_history=False,
            )
        )
        controller.set_pool_limit(ProtectionTier.HBM, int(float(phase["hbm_limit_gib"]) * GIB))
        controller.set_pool_limit(ProtectionTier.DRAM, int(float(phase["dram_limit_gib"]) * GIB))
        request_id = f"mechanism-gate-{index}"
        profile = RequestRecoveryProfile(
            request_id=request_id,
            source_worker="source-gate",
            source_failure_domain="node-gate",
            hbm_target="peer-gate",
            hbm_failure_domain="peer-node-gate",
            route_version=0,
            source_token_end=4096,
            slack_ms=0.0,
            model_revision="Qwen3-8B-frozen",
            kv_abi="qwen3-fp16-kv-v1",
            parallel_layout="TP1-PP1",
        )
        controller.register_request(profile)
        spec = ChunkSpec(
            request_id=request_id,
            ordinal=0,
            token_start=0,
            token_end=4096,
            byte_size=4096 * int(model["kv_bytes_per_token"]),
            route_version=0,
            source_worker="source-gate",
            hbm_target="peer-gate",
            model_revision="Qwen3-8B-frozen",
            kv_abi="qwen3-fp16-kv-v1",
            parallel_layout="TP1-PP1",
        )
        decision = controller.admit_chunk(spec)
        observed = decision.intended_residency.value
        rows.append({**dict(phase), "observed": observed, "passed": observed == phase["expected"]})
    return {
        "status": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "separate_from_performance_matrix": True,
        "posthoc_performance_selection": False,
        "rows": rows,
    }


def run_cell(
    protocol: dict[str, Any], *, policy: str, seed: int, workflows: int | None,
    active_per_gpu: int, probe_count: int, output: Path,
    arrival_cutoff_ms: float | None = None,
) -> dict[str, Any]:
    nominal_tasks, workload_meta = build_workload(
        protocol, seed=seed, workflows=workflows, active_per_gpu=active_per_gpu,
        arrival_cutoff_ms=arrival_cutoff_ms,
    )
    probes = make_fault_probes(
        nominal_tasks,
        seed=seed,
        count=probe_count,
        gpus_per_node=int(protocol["hardware"]["gpus_per_node"]),
        right_ms=arrival_cutoff_ms,
        failure_types=protocol["failure_model"].get("probe_type_schedule"),
    )
    tasks, schedule_solver = solve_schedule_with_protection_burden(
        protocol,
        policy,
        nominal_tasks,
        control_horizon_ms=arrival_cutoff_ms,
    )
    workload_meta["policy_horizon_ms"] = max(row.finish_ms for row in tasks)
    workload_meta["drain_horizon_ms"] = workload_meta["policy_horizon_ms"]
    protection = ProtectionReplay(protocol, policy)
    delay_maps, final_burdens, mechanism = protection.run(
        tasks, probes, control_horizon_ms=arrival_cutoff_ms,
    )
    final_reschedule = reschedule_with_burden(nominal_tasks, final_burdens)
    scheduled_finish = {row.request_id: row.finish_ms for row in tasks}
    schedule_solver["final_burden_residual_finish_ms"] = max(
        abs(row.finish_ms - scheduled_finish[row.request_id])
        for row in final_reschedule
    )
    mechanism["normal_path_schedule_solver"] = schedule_solver
    occupancy_trace = mechanism.pop("_occupancy_trace", [])
    control_trace = mechanism.pop("_control_trace", [])
    mechanism["control_step_trace"] = control_trace
    workflow_ids, base_wct = workflow_wct(tasks)
    no_fault_completion = replay_one_fault(tasks, {})
    mechanism["empty_fault_replay_max_completion_delta_ms"] = max(
        abs(no_fault_completion[row.request_id] - row.finish_ms) for row in tasks
    )
    fault_wct: list[list[float]] = []
    effective = 0
    for delays in delay_maps:
        if delays:
            effective += 1
        completion = replay_one_fault(tasks, delays)
        ids, values = workflow_wct(tasks, completion)
        if ids != workflow_ids:
            raise AssertionError("workflow identities changed during fault replay")
        fault_wct.append(values)
    changed_wct: list[float] = []
    changed_added_wct: list[float] = []
    changed_by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"wct": [], "added": []}
    )
    changed_workflow_count_by_probe: list[int] = []
    for probe, values in zip(probes, fault_wct):
        changed_count = 0
        for base, fault in zip(base_wct, values):
            delta = float(fault) - float(base)
            if delta < -1e-7:
                raise AssertionError("fault replay produced WCT below the no-fault WCT")
            if delta > 1e-7:
                changed_count += 1
                changed_wct.append(float(fault))
                changed_added_wct.append(delta)
                changed_by_type[probe.failure_type]["wct"].append(float(fault))
                changed_by_type[probe.failure_type]["added"].append(delta)
        changed_workflow_count_by_probe.append(changed_count)
    horizon_ms = float(
        workload_meta["arrival_horizon_ms"]
        if arrival_cutoff_ms is not None
        else workload_meta["base_horizon_ms"]
    )
    gpu_hours = int(protocol["models"]["qwen3_8b"]["center_system_dp"]) * horizon_ms / 3_600_000.0
    rates = [float(value) for value in protocol["failure_model"]["rates_per_1000_gpu_shard_hour"]]
    base_mean = statistics.fmean(base_wct)
    fault_mean = statistics.fmean(value for row in fault_wct for value in row)
    logical_new_kv_bytes = sum(
        int(row.prompt_tokens) * int(protocol["models"]["qwen3_8b"]["kv_bytes_per_token"])
        for row in tasks
    )
    admitted_copy_bytes = sum(
        int(mechanism.get("tier_churn", {}).get(tier, {}).get("admitted_bytes", 0))
        for tier in ("HBM", "DRAM")
    )
    mechanism["logical_new_kv_bytes"] = logical_new_kv_bytes
    mechanism["admitted_copy_bytes"] = admitted_copy_bytes
    mechanism["copy_amplification"] = (
        admitted_copy_bytes / logical_new_kv_bytes
        if logical_new_kv_bytes else 0.0
    )
    mechanism["eviction_count_total"] = sum(
        int(mechanism.get("tier_churn", {}).get(tier, {}).get("eviction_count", 0))
        for tier in ("HBM", "DRAM")
    )
    recovery_total = sum(int(value) for value in mechanism["recovery_path_counts"].values())
    recovery_hits = recovery_total - int(mechanism["recovery_path_counts"].get("RECOMPUTE", 0))
    mechanism["recovery_observation_count"] = recovery_total
    mechanism["recovery_hit_count"] = recovery_hits
    mechanism["recovery_hit_fraction"] = (
        recovery_hits / recovery_total if recovery_total else None
    )
    failure_conditioned_changed = {
        failure_type: {
            "changed_workflow_observations": len(values["wct"]),
            "changed_wct_mean_ms": (
                statistics.fmean(values["wct"]) if values["wct"] else None
            ),
            "changed_wct_p95_ms": quantile(values["wct"], 0.95),
            "added_wct_mean_ms": (
                statistics.fmean(values["added"]) if values["added"] else None
            ),
            "added_wct_p95_ms": quantile(values["added"], 0.95),
        }
        for failure_type, values in sorted(changed_by_type.items())
    }
    rate_rows = []
    for rate in rates:
        expected = rate * gpu_hours / 1000.0
        probability = 1.0 - math.exp(-expected)
        mixed_mean = (1.0 - probability) * base_mean + probability * fault_mean
        rate_rows.append({
            "failure_rate_per_1000_gpu_shard_hour": rate,
            "expected_faults_per_trace": expected,
            "single_or_more_fault_probability": probability,
            "hazard_integrated_mean_wct_ms": mixed_mean,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    trace_array = np.asarray(occupancy_trace, dtype=np.float64).reshape((-1, 3))
    control_array = np.asarray([
        (
            row["time_ms"], row["alpha_after_mean"], row["pressure_fraction"],
            row["active_requests_per_gpu"], row["hbm_used_bytes_total"],
            row["dram_used_bytes_total"],
        )
        for row in control_trace
    ], dtype=np.float64).reshape((-1, 6))
    np.savez_compressed(
        output,
        workflow_id=np.asarray(workflow_ids),
        base_wct_ms=np.asarray(base_wct, dtype=np.float64),
        fault_wct_ms=np.asarray(fault_wct, dtype=np.float64),
        probe_time_ms=np.asarray([row.time_ms for row in probes], dtype=np.float64),
        probe_source_rank=np.asarray([row.source_rank for row in probes], dtype=np.int16),
        probe_failure_type=np.asarray([row.failure_type for row in probes]),
        changed_workflow_count_by_probe=np.asarray(
            changed_workflow_count_by_probe, dtype=np.int32
        ),
        occupancy_time_ms=trace_array[:, 0],
        occupancy_hbm_total_bytes=trace_array[:, 1],
        occupancy_dram_total_bytes=trace_array[:, 2],
        control_time_ms=control_array[:, 0],
        control_alpha_after_mean=control_array[:, 1],
        control_pressure_fraction=control_array[:, 2],
        control_active_requests_per_gpu=control_array[:, 3],
        control_hbm_used_bytes_total=control_array[:, 4],
        control_dram_used_bytes_total=control_array[:, 5],
    )
    return {
        "policy": policy,
        "seed": int(seed),
        "active_requests_per_gpu": int(active_per_gpu),
        "workflow_count": len(base_wct),
        "base_wct_mean_ms": base_mean,
        "base_wct_p95_ms": quantile(base_wct, 0.95),
        "fault_conditioned_wct_mean_ms": fault_mean,
        "fault_conditioned_wct_p95_ms": quantile((v for row in fault_wct for v in row), 0.95),
        "failure_changed_workflow_observation_count": len(changed_wct),
        "failure_changed_wct_mean_ms": (
            statistics.fmean(changed_wct) if changed_wct else None
        ),
        "failure_changed_wct_p95_ms": quantile(changed_wct, 0.95),
        "failure_changed_added_wct_mean_ms": (
            statistics.fmean(changed_added_wct) if changed_added_wct else None
        ),
        "failure_changed_added_wct_p95_ms": quantile(changed_added_wct, 0.95),
        "failure_conditioned_changed_by_type": failure_conditioned_changed,
        "fault_probe_count": len(probes),
        "effective_fault_probe_count": effective,
        "simulated_gpu_shard_hours": gpu_hours,
        "rate_rows": rate_rows,
        "workload": workload_meta,
        "mechanism": mechanism,
        "raw_npz": str(output),
        "raw_npz_sha256": sha256(output),
    }


def bootstrap_ci(values: list[float], *, seed: int, samples: int = 10000) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap input cannot be empty")
    rng = random.Random(int(seed))
    means = []
    for _ in range(int(samples)):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    return statistics.fmean(values), quantile(means, 0.025), quantile(means, 0.975)


def write_aggregate_and_audit(
    protocol: dict[str, Any], summaries: list[dict[str, Any]], output_dir: Path,
    *, mode: str, runner_path: Path,
) -> None:
    by_key: dict[tuple[str, int, float], list[tuple[int, float]]] = defaultdict(list)
    for cell in summaries:
        for rate in cell["rate_rows"]:
            key = (
                str(cell["policy"]),
                int(cell["active_requests_per_gpu"]),
                float(rate["failure_rate_per_1000_gpu_shard_hour"]),
            )
            by_key[key].append((int(cell["seed"]), float(rate["hazard_integrated_mean_wct_ms"])))
    aggregate_rows = []
    for (policy, concurrency, rate), seed_values in sorted(by_key.items()):
        seed_values.sort()
        values = [value for _seed, value in seed_values]
        mean, low, high = bootstrap_ci(values, seed=20260826 ^ concurrency ^ int(rate * 1000))
        native = dict(by_key.get(("Native Mooncake", concurrency, rate), ()))
        paired_ratio = [value / native[seed] for seed, value in seed_values if seed in native]
        if paired_ratio:
            ratio, ratio_low, ratio_high = bootstrap_ci(
                paired_ratio, seed=20260827 ^ concurrency ^ int(rate * 1000)
            )
        else:
            ratio = ratio_low = ratio_high = math.nan
        aggregate_rows.append({
            "policy": policy,
            "active_requests_per_gpu": concurrency,
            "failure_rate_per_1000_gpu_shard_hour": rate,
            "seed_count": len(values),
            "mean_wct_ms": mean,
            "ci95_low_ms": low,
            "ci95_high_ms": high,
            "normalized_to_native": ratio,
            "normalized_ci95_low": ratio_low,
            "normalized_ci95_high": ratio_high,
        })
    aggregate_path = output_dir / "aggregate_ci.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    import numpy as np
    checks: dict[str, bool] = {}
    expected_center_policies = set(dict.fromkeys((*PRIMARY_POLICIES, *ABLATION_POLICIES)))
    center = int(protocol["workload"]["center_active_requests_per_gpu"])
    center_cells = [cell for cell in summaries if cell["active_requests_per_gpu"] == center]
    checks["design_semantics_exact"] = protocol["design_semantics"] == DESIGN_SEMANTICS
    checks["a100_40gib_exact"] = float(protocol["hardware"]["gpu_hbm_gib"]) == 40.0
    checks["eight_gpus_per_node_exact"] = int(protocol["hardware"]["gpus_per_node"]) == 8
    checks["alpha0_quota_exact"] = (
        float(protocol["aegiskv"]["initial_alpha"]) == 0.25
        and float(protocol["aegiskv"]["hbm_nominal_quota_gib"]) == 10.0
    )
    checks["failure_rates_exact"] = protocol["failure_model"]["rates_per_1000_gpu_shard_hour"] == [0.0, 5.952, 20.833]
    checks["center_three_paired_seeds"] = all(
        {cell["seed"] for cell in center_cells if cell["policy"] == policy}
        == set(protocol["paired_seeds"])
        for policy in expected_center_policies
    )
    checks["center_policy_set_exact"] = {cell["policy"] for cell in center_cells} == expected_center_policies
    checks["raw_hashes_match"] = all(sha256(Path(cell["raw_npz"])) == cell["raw_npz_sha256"] for cell in summaries)
    checks["raw_wct_shapes_and_values"] = True
    checks["paired_workload_and_fault_tuples"] = True
    paired_signature: dict[tuple[int, int], tuple[Any, ...]] = {}
    combined_paths: Counter[str] = Counter()
    for cell in summaries:
        with np.load(cell["raw_npz"], allow_pickle=False) as raw:
            base = raw["base_wct_ms"]
            fault = raw["fault_wct_ms"]
            if (
                base.shape != (cell["workflow_count"],)
                or fault.shape != (cell["fault_probe_count"], cell["workflow_count"])
                or not np.all(np.isfinite(base))
                or not np.all(np.isfinite(fault))
                or np.any(base <= 0)
                or np.any(fault <= 0)
            ):
                checks["raw_wct_shapes_and_values"] = False
            signature = (
                tuple(raw["workflow_id"].tolist()),
                raw["probe_time_ms"].tobytes(),
                raw["probe_source_rank"].tobytes(),
                tuple(raw["probe_failure_type"].tolist()),
            )
            key = (int(cell["seed"]), int(cell["active_requests_per_gpu"]))
            if key in paired_signature and paired_signature[key] != signature:
                checks["paired_workload_and_fault_tuples"] = False
            paired_signature[key] = signature
        combined_paths.update(cell["mechanism"]["path_event_counts"])
    checks["four_paths_exercised"] = all(
        combined_paths[name] > 0
        for name in (
            "copy_peer_hbm_same_node", "copy_peer_hbm_cross_node",
            "copy_dram_local", "copy_dram_remote",
        )
    )
    checks["no_target_owned_or_hbm_to_dram"] = all(
        cell["mechanism"]["target_owned_copy_events"] == 0
        and cell["mechanism"]["hbm_to_dram_copy_events"] == 0
        for cell in summaries
    )
    checks["workflow_completion_is_des_not_slack_harm"] = "queueing" in protocol["wct_definition"] and "recovery" in protocol["wct_definition"]
    center_base_by_policy = {
        policy: statistics.fmean(
            cell["base_wct_mean_ms"] for cell in center_cells if cell["policy"] == policy
        )
        for policy in expected_center_policies
    }
    checks["normal_path_copy_burden_nonzero"] = all(
        cell["mechanism"]["normal_path_copy_burden_mean_ms"] > 0.0
        for cell in summaries
    )
    checks["policy_normal_path_wct_not_all_identical"] = len(
        {round(value, 9) for value in center_base_by_policy.values()}
    ) > 1
    mechanism_gate = run_mechanism_gate(protocol)
    checks["four_residency_mechanism_gate"] = mechanism_gate["status"] == "PASS"
    mechanism_gate_path = output_dir / "mechanism_gate.json"
    mechanism_gate_path.write_text(
        json.dumps(mechanism_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": mode,
        "paper_ready": False,
        "preliminary_simulation_result": True,
        "physical_result": False,
        "workflow_completion_time_result": True,
        "single_fault_sparse_rate_approximation": True,
        "checks": checks,
        "cell_count": len(summaries),
        "protocol_sha256": sha256(ROOT / "protocol.json"),
        "runner_sha256": sha256(runner_path),
        "aggregate_ci_sha256": sha256(aggregate_path),
        "combined_path_event_counts": dict(sorted(combined_paths.items())),
        "mechanism_gate": {
            "path": str(mechanism_gate_path),
            "sha256": sha256(mechanism_gate_path),
            "status": mechanism_gate["status"],
        },
        "modeled_native_mooncake_approximation": protocol["baseline_fidelity"]["Native Mooncake"],
        "evidence_boundary": protocol["provenance_boundary"],
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"preliminary audit failed: {failed}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    protocol = load_protocol()
    mode = "smoke" if args.smoke else "preliminary"
    output_dir = args.output_dir or ROOT / "results" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow_count = int(protocol["workload"]["center_workflows_smoke" if args.smoke else "center_workflows_main"])
    probe_count = int(protocol["workload"]["fault_probes_per_seed_smoke" if args.smoke else "fault_probes_per_seed_main"])
    seeds = [int(value) for value in protocol["paired_seeds"]]
    center = int(protocol["workload"]["center_active_requests_per_gpu"])
    cells: list[tuple[str, int]] = []
    all_center = tuple(dict.fromkeys((*PRIMARY_POLICIES, *ABLATION_POLICIES)))
    cells.extend((policy, center) for policy in all_center)
    if not args.smoke:
        for concurrency in protocol["workload"]["active_requests_per_gpu"]:
            if int(concurrency) == center:
                continue
            cells.extend((policy, int(concurrency)) for policy in ("Native Mooncake", "AegisKV"))
    summaries = []
    for policy, concurrency in cells:
        for seed in seeds:
            label = policy.lower().replace(" ", "_").replace("/", "_")
            raw = output_dir / "raw" / f"{label}__c{concurrency}__s{seed}.npz"
            print(f"RUN policy={policy!r} c={concurrency} seed={seed}", flush=True)
            summaries.append(run_cell(
                protocol,
                policy=policy,
                seed=seed,
                workflows=workflow_count,
                active_per_gpu=concurrency,
                probe_count=probe_count,
                output=raw,
            ))
    summary_path = output_dir / "summary.json"
    payload = {
        "schema_version": 1,
        "status": "PRELIMINARY_COMPLETE",
        "mode": mode,
        "evidence_class": protocol["evidence_class"],
        "design_semantics": protocol["design_semantics"],
        "workflow_completion_time_result": True,
        "protocol": {"path": str(ROOT / "protocol.json"), "sha256": sha256(ROOT / "protocol.json")},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "cells": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "policy", "seed", "active_requests_per_gpu", "failure_rate_per_1000_gpu_shard_hour",
            "hazard_integrated_mean_wct_ms", "expected_faults_per_trace", "base_wct_mean_ms",
            "fault_conditioned_wct_mean_ms", "effective_fault_probe_count", "raw_npz",
        ])
        writer.writeheader()
        for cell in summaries:
            for rate in cell["rate_rows"]:
                writer.writerow({
                    "policy": cell["policy"],
                    "seed": cell["seed"],
                    "active_requests_per_gpu": cell["active_requests_per_gpu"],
                    "failure_rate_per_1000_gpu_shard_hour": rate["failure_rate_per_1000_gpu_shard_hour"],
                    "hazard_integrated_mean_wct_ms": rate["hazard_integrated_mean_wct_ms"],
                    "expected_faults_per_trace": rate["expected_faults_per_trace"],
                    "base_wct_mean_ms": cell["base_wct_mean_ms"],
                    "fault_conditioned_wct_mean_ms": cell["fault_conditioned_wct_mean_ms"],
                    "effective_fault_probe_count": cell["effective_fault_probe_count"],
                    "raw_npz": cell["raw_npz"],
                })
    write_aggregate_and_audit(
        protocol, summaries, output_dir, mode=mode, runner_path=Path(__file__).resolve()
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
