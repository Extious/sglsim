"""Event-level serving/Peer-HBM shared-capacity coupling for the v3 audit."""

from __future__ import annotations

import heapq
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import run_active_hbm_v3_capacity_sanity as capacity
import run_preliminary as core


GIB = 1024**3
PROTOCOL_PATH = Path(__file__).resolve().parent / "active_hbm_v3_protocol.json"


class ActiveHBMCoupledReplay(core.ProtectionReplay):
    """Joint FCFS serving scheduler and source-owned protection replay."""

    def __init__(self, protocol: dict[str, Any], policy: str, capacity_protocol: dict[str, Any]) -> None:
        super().__init__(protocol, policy)
        self.capacity_protocol = capacity_protocol
        self.shared_budget_bytes = int(
            capacity_protocol["shared_capacity"][
                "serving_plus_peer_backup_budget_bytes"
            ]
        )
        self.weight_bytes = int(
            capacity_protocol["frozen_model"]["weight_bytes_per_tp1_pp1_rank"]
        )
        self.runtime_reserve_bytes = int(
            capacity_protocol["runtime_reserve"]["runtime_reserve_bytes"]
        )
        self.total_hbm_bytes = int(capacity_protocol["hardware"]["total_hbm_bytes"])
        self.max_engine_lanes = int(
            capacity_protocol["shared_capacity"]["max_engine_lanes_per_gpu"]
        )
        self.waiting = [deque() for _ in range(self.dp)]
        self.free_lanes = [list(range(self.max_engine_lanes)) for _ in range(self.dp)]
        for lanes in self.free_lanes:
            heapq.heapify(lanes)
        self.active_by_rank: dict[int, set[str]] = defaultdict(set)
        self.active_serving_bytes = [0] * self.dp
        self.serving_bytes_by_request: dict[str, int] = {}
        self.scheduled_tasks: list[core.Task] = []
        self.queue_wait_values_ms: list[float] = []
        self.memory_blocked_observation_count = 0
        self.lane_blocked_observation_count = 0
        self.memory_blocked_request_ids: set[str] = set()
        self.dispatch_count = 0
        self.dispatch_at_control_count = 0
        self.capacity_violation_count = 0
        self.capacity_min_headroom_bytes = self.shared_budget_bytes
        self.capacity_peak_serving_bytes = 0
        self.capacity_peak_backup_bytes = 0
        self.capacity_peak_shared_used_bytes = 0
        self.capacity_peak_total_hbm_bytes = self.weight_bytes + self.runtime_reserve_bytes
        self.capacity_trace: list[dict[str, Any]] = []
        self.control_capacity_trace: list[dict[str, Any]] = []
        self.backup_bytes_released_by_control = 0

    def serving_reservation_bytes(self, task: core.Task) -> int:
        # Deliberately do not read task.output_tokens: it is the trace-realized
        # final length.  The no-oracle helper accepts only prompt_tokens.
        return capacity.serving_reservation_bytes(
            self.capacity_protocol, task.prompt_tokens
        )

    def hbm_backup_used_bytes(self, rank: int) -> int:
        if self.is_aegis:
            return int(
                self.aegis[rank].telemetry_snapshot()["pools"]["HBM"]["used_bytes"]
            )
        return sum(
            int(runtime.used_bytes)
            for runtime in self.baselines[rank]
            if runtime.config.tier.value == "HBM"
        )

    def hbm_backup_limit_bytes(self, rank: int) -> int:
        if self.is_aegis:
            return int(
                self.aegis[rank].telemetry_snapshot()["pools"]["HBM"]["limit_bytes"]
            )
        return sum(
            int(runtime.config.capacity_bytes)
            for runtime in self.baselines[rank]
            if runtime.config.tier.value == "HBM"
        )

    def projected_backup_after_start(self, rank: int, task: core.Task) -> int:
        used = self.hbm_backup_used_bytes(rank)
        limit = self.hbm_backup_limit_bytes(rank)
        prompt_backup = int(task.prompt_tokens) * self.kv
        return min(limit, used + prompt_backup)

    def dispatch_block_reason(self, rank: int) -> str | None:
        if not self.waiting[rank]:
            return None
        if not self.free_lanes[rank]:
            return "ENGINE_LANE"
        task = self.waiting[rank][0]
        projected = (
            self.active_serving_bytes[rank]
            + self.serving_reservation_bytes(task)
            + self.projected_backup_after_start(rank, task)
        )
        if projected > self.shared_budget_bytes:
            return "SHARED_HBM"
        return None

    def record_capacity(self, rank: int, at: float, event: str) -> None:
        serving = int(self.active_serving_bytes[rank])
        backup = self.hbm_backup_used_bytes(rank)
        shared_used = serving + backup
        total = self.weight_bytes + self.runtime_reserve_bytes + shared_used
        headroom = self.shared_budget_bytes - shared_used
        if shared_used > self.shared_budget_bytes:
            self.capacity_violation_count += 1
        self.capacity_min_headroom_bytes = min(self.capacity_min_headroom_bytes, headroom)
        self.capacity_peak_serving_bytes = max(self.capacity_peak_serving_bytes, serving)
        self.capacity_peak_backup_bytes = max(self.capacity_peak_backup_bytes, backup)
        self.capacity_peak_shared_used_bytes = max(
            self.capacity_peak_shared_used_bytes, shared_used
        )
        self.capacity_peak_total_hbm_bytes = max(self.capacity_peak_total_hbm_bytes, total)
        self.capacity_trace.append({
            "time_ms": float(at),
            "rank": int(rank),
            "event": event,
            "active_requests": len(self.active_by_rank.get(rank, ())),
            "waiting_requests": len(self.waiting[rank]),
            "active_serving_kv_bytes": serving,
            "peer_hbm_backup_used_bytes": backup,
            "shared_used_bytes": shared_used,
            "shared_headroom_bytes": headroom,
            "total_hbm_used_bytes": total,
        })
        if shared_used > self.shared_budget_bytes:
            raise AssertionError("v3 shared serving/backup HBM capacity exceeded")

    def try_dispatch(self, rank: int, at: float, *, event: str) -> int:
        dispatched = 0
        while self.waiting[rank]:
            reason = self.dispatch_block_reason(rank)
            if reason is not None:
                if reason == "SHARED_HBM":
                    self.memory_blocked_observation_count += 1
                    self.memory_blocked_request_ids.add(
                        self.waiting[rank][0].request_id
                    )
                else:
                    self.lane_blocked_observation_count += 1
                self.record_capacity(rank, at, f"{event}_blocked_{reason.lower()}")
                break
            nominal = self.waiting[rank].popleft()
            lane = heapq.heappop(self.free_lanes[rank])
            serving_bytes = self.serving_reservation_bytes(nominal)
            runtime_task = replace(
                nominal, lane=lane, start_ms=float(at), finish_ms=float(at)
            )
            self.active_by_rank[rank].add(runtime_task.request_id)
            self.active_serving_bytes[rank] += serving_bytes
            self.serving_bytes_by_request[runtime_task.request_id] = serving_bytes
            self.start(runtime_task, at)
            burden = float(self.health_burden_ms[runtime_task.request_id])
            runtime_task = replace(
                runtime_task, finish_ms=float(at) + runtime_task.service_ms + burden
            )
            self.task_by_id[runtime_task.request_id] = runtime_task
            self.scheduled_tasks.append(runtime_task)
            self.push(runtime_task.finish_ms, 4, "coupled_finish", runtime_task)
            self.queue_wait_values_ms.append(float(at) - runtime_task.arrival_ms)
            self.dispatch_count += 1
            dispatched += 1
            self.record_capacity(rank, at, f"{event}_dispatch")
        return dispatched

    def summarize_mechanism(self, control_horizon_ms: float) -> dict[str, Any]:
        snapshots = [
            controller.telemetry_snapshot() for controller in self.aegis
        ] if self.is_aegis else []
        alpha = [float(row["hbm_alpha"]) for row in snapshots]
        tier_churn = {}
        for tier in ("HBM", "DRAM"):
            tier_rows = [snapshot["pools"][tier]["churn"] for snapshot in snapshots]
            tier_churn[tier] = {
                key: sum(int(row[key]) for row in tier_rows)
                for key in (
                    "admission_attempt_count", "admission_success_count",
                    "admission_rejection_count", "admitted_bytes",
                    "eviction_count", "evicted_bytes", "removal_count",
                    "removed_bytes",
                )
            } if tier_rows else {}
        baseline_snapshots = [
            runtime.snapshot()
            for rank in self.baselines
            for runtime in rank
        ]
        burden = list(self.health_burden_ms.values())
        return {
            "path_event_counts": dict(sorted(self.path_counts.items())),
            "residency_event_counts": dict(sorted(self.residency_counts.items())),
            "recovery_path_counts": dict(sorted(self.recovery_counts.items())),
            "recovery_type_path_counts": dict(sorted(self.recovery_type_path_counts.items())),
            "failover_mode_counts": dict(sorted(self.failover_mode_counts.items())),
            "recovery_restored_tokens": self.recovery_restored_tokens,
            "recovery_recomputed_tokens": self.recovery_recomputed_tokens,
            "alpha_final_mean": statistics.fmean(alpha) if alpha else None,
            "alpha_final_min": min(alpha) if alpha else None,
            "control_window_count": self.control_window_count,
            "control_opportunity_times_ms": self.control_opportunity_times_ms,
            "control_target_evaluations": self.control_target_evaluations,
            "pressure_target_evaluations": self.pressure_target_evaluations,
            "pressure_target_evaluation_fraction": (
                self.pressure_target_evaluations / self.control_target_evaluations
                if self.control_target_evaluations else None
            ),
            "algorithm1_nonzero_slack_request_count": self.slack_nonzero_request_count,
            "normal_path_copy_burden_mean_ms": statistics.fmean(burden) if burden else 0.0,
            "normal_path_copy_burden_p95_ms": core.quantile(burden, 0.95),
            "copy_transfer_service_mean_ms": (
                statistics.fmean(self.copy_service_ms.values())
                if self.copy_service_ms else 0.0
            ),
            "copy_queue_wait_mean_ms": (
                statistics.fmean(self.copy_queue_wait_ms.values())
                if self.copy_queue_wait_ms else 0.0
            ),
            "tier_churn": tier_churn,
            "baseline_runtime_snapshots": baseline_snapshots,
            "control_step_trace": self.control_trace,
            "active_hbm_v3": {
                "dispatch_uses_no_realized_output_oracle": True,
                "request_max_output_tokens": int(
                    self.capacity_protocol["shared_capacity"]["request_max_output_tokens"]
                ),
                "total_hbm_bytes": self.total_hbm_bytes,
                "weight_bytes": self.weight_bytes,
                "runtime_reserve_bytes": self.runtime_reserve_bytes,
                "shared_budget_bytes": self.shared_budget_bytes,
                "dispatch_count": self.dispatch_count,
                "queue_wait_mean_ms": (
                    statistics.fmean(self.queue_wait_values_ms)
                    if self.queue_wait_values_ms else 0.0
                ),
                "queue_wait_p95_ms": core.quantile(self.queue_wait_values_ms, 0.95),
                "memory_blocked_observation_count": self.memory_blocked_observation_count,
                "memory_blocked_unique_request_count": len(self.memory_blocked_request_ids),
                "lane_blocked_observation_count": self.lane_blocked_observation_count,
                "dispatch_at_control_count": self.dispatch_at_control_count,
                "backup_bytes_released_by_control": self.backup_bytes_released_by_control,
                "capacity_record_count": len(self.capacity_trace),
                "capacity_violation_count": self.capacity_violation_count,
                "capacity_min_headroom_bytes": self.capacity_min_headroom_bytes,
                "capacity_peak_serving_bytes": self.capacity_peak_serving_bytes,
                "capacity_peak_backup_bytes": self.capacity_peak_backup_bytes,
                "capacity_peak_shared_used_bytes": self.capacity_peak_shared_used_bytes,
                "capacity_peak_total_hbm_bytes": self.capacity_peak_total_hbm_bytes,
                "control_capacity_trace": self.control_capacity_trace,
            },
            "target_owned_copy_events": 0,
            "hbm_to_dram_copy_events": 0,
            "occupancy_observation_horizon_ms": float(control_horizon_ms),
        }

    def run_coupled(
        self, nominal_tasks: list[core.Task], probes: list[core.FaultProbe],
        *, control_horizon_ms: float,
    ) -> tuple[list[core.Task], list[dict[str, float]], dict[str, Any], list[dict[str, Any]]]:
        self.task_by_id = {row.request_id: row for row in nominal_tasks}
        self.occupancy_observation_horizon_ms = float(control_horizon_ms)
        for rank in range(self.dp):
            self.record_capacity(rank, 0.0, "initial")
        for row in sorted(nominal_tasks, key=lambda item: (item.arrival_ms, item.request_id)):
            self.push(row.arrival_ms, 1, "coupled_arrival", row)
        for probe in probes:
            self.push(probe.time_ms, 2, "probe", probe)
        window = float(self.protocol["aegiskv"]["window_ms"])
        at = window
        while at <= float(control_horizon_ms) + 1e-9:
            self.push(at, 3, "coupled_control", None)
            at += window
        recoveries: list[dict[str, float]] = []
        while self.events:
            at, _priority, _sequence, kind, payload = heapq.heappop(self.events)
            if kind == "aegis_copy":
                self.complete_copy(payload, at)
            elif kind == "coupled_arrival":
                task = payload
                self.waiting[task.source_rank].append(task)
                self.try_dispatch(task.source_rank, at, event="arrival")
            elif kind == "coupled_finish":
                task = payload
                rank = task.source_rank
                self.active_by_rank[rank].discard(task.request_id)
                self.active_serving_bytes[rank] -= self.serving_bytes_by_request.pop(
                    task.request_id
                )
                heapq.heappush(self.free_lanes[rank], task.lane)
                self.finish(task, at)
                self.record_capacity(rank, at, "finish_release")
                self.try_dispatch(rank, at, event="finish")
            elif kind == "coupled_control":
                external = {
                    rank: self.dispatch_block_reason(rank) == "SHARED_HBM"
                    for rank in range(self.dp)
                }
                before_backup = sum(
                    self.hbm_backup_used_bytes(rank) for rank in range(self.dp)
                )
                waiting_before = sum(len(rows) for rows in self.waiting)
                active_before = sum(len(rows) for rows in self.active_by_rank.values())
                self.control(
                    at,
                    {rank: len(rows) for rank, rows in self.active_by_rank.items()},
                    external_pressure_by_rank=external,
                )
                after_backup = sum(
                    self.hbm_backup_used_bytes(rank) for rank in range(self.dp)
                )
                released = max(before_backup - after_backup, 0)
                self.backup_bytes_released_by_control += released
                dispatched = 0
                for rank in range(self.dp):
                    self.record_capacity(rank, at, "control_after_quota")
                    dispatched += self.try_dispatch(rank, at, event="control")
                self.dispatch_at_control_count += dispatched
                self.control_capacity_trace.append({
                    "time_ms": float(at),
                    "memory_pressure_rank_count": sum(external.values()),
                    "backup_before_bytes": before_backup,
                    "backup_after_bytes": after_backup,
                    "backup_released_bytes": released,
                    "waiting_before": waiting_before,
                    "waiting_after": sum(len(rows) for rows in self.waiting),
                    "active_before": active_before,
                    "active_after": sum(len(rows) for rows in self.active_by_rank.values()),
                    "dispatched_after_control": dispatched,
                })
            elif kind == "probe":
                probe = payload
                ranks = [probe.source_rank]
                if probe.failure_type == "source_node":
                    base = (probe.source_rank // 8) * 8
                    ranks = list(range(base, base + 8))
                delays: dict[str, float] = {}
                for rank in ranks:
                    for request_id in sorted(self.active_by_rank.get(rank, ())):
                        task = self.task_by_id[request_id]
                        delay, _action = self.recovery(task, probe)
                        delays[request_id] = delay
                recoveries.append(delays)
            else:
                raise AssertionError(f"unknown coupled event {kind!r}")
        if any(self.waiting) or any(self.active_by_rank.values()):
            raise AssertionError("coupled scheduler failed to drain")
        if len(self.scheduled_tasks) != len(nominal_tasks):
            raise AssertionError("coupled scheduler lost requests")
        mechanism = self.summarize_mechanism(control_horizon_ms)
        return (
            sorted(self.scheduled_tasks, key=lambda row: (row.start_ms, row.request_id)),
            recoveries,
            mechanism,
            list(self.capacity_trace),
        )


def run_v3_cell(
    protocol: dict[str, Any], capacity_protocol: dict[str, Any], *,
    policy: str, seed: int, active_per_gpu: int, probe_count: int,
    output: Path, arrival_cutoff_ms: float,
) -> dict[str, Any]:
    nominal, workload = core.build_workload(
        protocol,
        seed=seed,
        workflows=None,
        active_per_gpu=active_per_gpu,
        arrival_cutoff_ms=arrival_cutoff_ms,
    )
    probes = core.make_fault_probes(
        nominal,
        seed=seed,
        count=probe_count,
        gpus_per_node=int(protocol["hardware"]["gpus_per_node"]),
        right_ms=arrival_cutoff_ms,
        failure_types=protocol["failure_model"].get("probe_type_schedule"),
    )
    replay = ActiveHBMCoupledReplay(protocol, policy, capacity_protocol)
    tasks, delay_maps, mechanism, capacity_trace = replay.run_coupled(
        nominal, probes, control_horizon_ms=arrival_cutoff_ms
    )
    ids, base_wct = core.workflow_wct(tasks)
    no_fault = core.replay_one_fault(tasks, {})
    empty_delta = max(
        abs(no_fault[row.request_id] - row.finish_ms) for row in tasks
    )
    mechanism["empty_fault_replay_max_completion_delta_ms"] = empty_delta
    fault_wct = []
    changed_by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"wct": [], "added": [], "all_wct": [], "all_added": []}
    )
    changed_counts = []
    for probe, delays in zip(probes, delay_maps):
        completion = core.replay_one_fault(tasks, delays)
        replay_ids, values = core.workflow_wct(tasks, completion)
        if replay_ids != ids:
            raise AssertionError("v3 workflow identity changed")
        changed = 0
        for base, fault in zip(base_wct, values):
            delta = fault - base
            if delta < -1e-7:
                raise AssertionError("v3 fault replay reduced WCT")
            if delta > 1e-7:
                changed += 1
                changed_by_type[probe.failure_type]["wct"].append(fault)
                changed_by_type[probe.failure_type]["added"].append(delta)
            changed_by_type[probe.failure_type]["all_wct"].append(fault)
            changed_by_type[probe.failure_type]["all_added"].append(max(delta, 0.0))
        changed_counts.append(changed)
        fault_wct.append(values)
    failure_stats = {
        failure_type: {
            "changed_workflow_observations": len(values["wct"]),
            "changed_wct_mean_ms": (
                statistics.fmean(values["wct"]) if values["wct"] else None
            ),
            "changed_wct_p95_ms": core.quantile(values["wct"], 0.95),
            "added_wct_mean_ms": (
                statistics.fmean(values["added"]) if values["added"] else None
            ),
            "added_wct_p95_ms": core.quantile(values["added"], 0.95),
            "all_workflow_observations": len(values["all_wct"]),
            "all_workflow_wct_mean_ms": statistics.fmean(values["all_wct"]),
            "all_workflow_wct_p95_ms": core.quantile(values["all_wct"], 0.95),
            "all_workflow_added_wct_mean_ms": statistics.fmean(values["all_added"]),
            "all_workflow_added_wct_p95_ms": core.quantile(values["all_added"], 0.95),
            "changed_workflow_fraction": (
                len(values["wct"]) / len(values["all_wct"])
            ),
        }
        for failure_type, values in sorted(changed_by_type.items())
        if values["all_wct"]
    }
    trace_array = np.asarray([
        (
            row["time_ms"], row["rank"], row["active_requests"],
            row["waiting_requests"], row["active_serving_kv_bytes"],
            row["peer_hbm_backup_used_bytes"], row["shared_used_bytes"],
            row["shared_headroom_bytes"], row["total_hbm_used_bytes"],
        )
        for row in capacity_trace
    ], dtype=np.float64).reshape((-1, 9))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        workflow_id=np.asarray(ids),
        base_wct_ms=np.asarray(base_wct, dtype=np.float64),
        fault_wct_ms=np.asarray(fault_wct, dtype=np.float64),
        probe_id=np.asarray([row.probe_id for row in probes]),
        probe_time_ms=np.asarray([row.time_ms for row in probes], dtype=np.float64),
        probe_source_rank=np.asarray([row.source_rank for row in probes], dtype=np.int16),
        probe_failure_type=np.asarray([row.failure_type for row in probes]),
        changed_workflow_count_by_probe=np.asarray(changed_counts, dtype=np.int32),
        capacity_time_ms=trace_array[:, 0],
        capacity_rank=trace_array[:, 1],
        capacity_active_requests=trace_array[:, 2],
        capacity_waiting_requests=trace_array[:, 3],
        capacity_active_serving_kv_bytes=trace_array[:, 4],
        capacity_peer_hbm_backup_used_bytes=trace_array[:, 5],
        capacity_shared_used_bytes=trace_array[:, 6],
        capacity_shared_headroom_bytes=trace_array[:, 7],
        capacity_total_hbm_used_bytes=trace_array[:, 8],
    )
    workload["policy_horizon_ms"] = max(row.finish_ms for row in tasks)
    workload["drain_horizon_ms"] = workload["policy_horizon_ms"]
    active = core.realized_active(tasks, int(protocol["models"]["qwen3_8b"]["center_system_dp"]))
    workload["coupled_realized_active_requests_per_gpu_time_mean"] = (
        active["mean"] / int(protocol["models"]["qwen3_8b"]["center_system_dp"])
    )
    workload["coupled_realized_active_requests_per_gpu_peak"] = (
        active["peak"] / int(protocol["models"]["qwen3_8b"]["center_system_dp"])
    )
    workflow_universe_sha256 = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    probe_tuple_payload = [
        {
            "probe_id": row.probe_id,
            "time_ms": row.time_ms,
            "source_rank": row.source_rank,
            "failure_type": row.failure_type,
        }
        for row in probes
    ]
    fault_probe_tuple_sha256 = hashlib.sha256(
        json.dumps(
            probe_tuple_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "policy": policy,
        "seed": int(seed),
        "active_requests_per_gpu": int(active_per_gpu),
        "context_tokens": int(protocol["workload"]["center_context_tokens"]),
        "failure_type": probes[0].failure_type if probes else None,
        "workflow_count": len(ids),
        "base_wct_mean_ms": statistics.fmean(base_wct),
        "base_wct_p95_ms": core.quantile(base_wct, 0.95),
        "failure_conditioned_changed_by_type": failure_stats,
        "fault_probe_count": len(probes),
        "effective_fault_probe_count": sum(bool(row) for row in delay_maps),
        "workflow_universe_count": len(ids),
        "workflow_universe_sha256": workflow_universe_sha256,
        "fault_probe_tuple_count": len(probes),
        "fault_probe_tuple_sha256": fault_probe_tuple_sha256,
        "workload": workload,
        "mechanism": mechanism,
        "raw_npz": str(output),
        "raw_npz_sha256": core.sha256(output),
    }
