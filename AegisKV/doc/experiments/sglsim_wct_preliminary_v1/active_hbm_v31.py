"""V3.1 correction: exact admission projection and non-oracle modeled Slack.

This module intentionally leaves ``active_hbm_v3.py`` unchanged so its
published diagnostic remains hash-addressable.  V3.1 changes only two
mechanisms shown to disagree with the current-design contract:

* serving admission projects the HBM bytes that the current tier controller
  would actually retain after per-chunk admission, rather than assuming that
  an entire prompt will necessarily occupy Peer HBM; and
* modeled Slack is generated from an independent prediction stream and never
  reads the trace-realized output length or realized service perturbation.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random
import statistics
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import active_hbm_v3 as v3
import run_active_hbm_v3_capacity_sanity as capacity
import run_preliminary as core


def failure_domain_ranks(
    probe: core.FaultProbe, *, gpus_per_node: int, dp: int,
) -> tuple[int, ...]:
    if probe.failure_type == "source_worker_gpu":
        return (int(probe.source_rank),)
    if probe.failure_type == "source_node":
        node_base = (int(probe.source_rank) // int(gpus_per_node)) * int(gpus_per_node)
        return tuple(range(node_base, min(node_base + int(gpus_per_node), int(dp))))
    raise ValueError(f"unsupported failure type {probe.failure_type!r}")


def apply_nonoracle_modeled_slack(
    tasks: list[core.Task], protocol: dict[str, Any], *, seed: int,
) -> list[core.Task]:
    """Attach a causal modeled prediction which is independent of final output."""
    model = protocol["models"]["qwen3_8b"]
    workload = protocol["workload"]
    outputs = [int(value) for value in workload["output_tokens"]]
    weights = [float(value) for value in workload["output_weights"]]
    active_per_gpu = int(workload["center_active_requests_per_gpu"])
    slowdown = 1.0 + 0.035 * max(active_per_gpu - 1, 0)
    rng = random.Random(int(seed) ^ 0x51ACCA11)
    by_workflow: dict[str, list[core.Task]] = defaultdict(list)
    for task in tasks:
        by_workflow[task.workflow_id].append(task)
    rewritten: dict[str, core.Task] = {}
    for workflow_id in sorted(by_workflow):
        predictions: list[tuple[core.Task, float]] = []
        for task in sorted(by_workflow[workflow_id], key=lambda row: row.request_id):
            predicted_output = rng.choices(outputs, weights=weights, k=1)[0]
            predicted_service = slowdown * (
                float(model["prefill_setup_ms"])
                + int(task.prompt_tokens) * float(model["prefill_ms_per_token"])
                + predicted_output * float(model["decode_ms_per_token"])
            )
            # Independent prediction residual; it is not the realized request
            # service multiplier sampled by build_workload().
            predicted_service *= rng.uniform(0.96, 1.04)
            predictions.append((task, predicted_service))
        predicted_critical = max(value for _task, value in predictions)
        for task, predicted_service in predictions:
            rewritten[task.request_id] = replace(
                task,
                predicted_slack_ms=max(predicted_critical - predicted_service, 0.0),
            )
    return [rewritten[row.request_id] for row in tasks]


def build_workload(
    protocol: dict[str, Any], *, seed: int, active_per_gpu: int,
    arrival_cutoff_ms: float,
) -> tuple[list[core.Task], dict[str, Any]]:
    tasks, metadata = core.build_workload(
        protocol,
        seed=seed,
        workflows=None,
        active_per_gpu=active_per_gpu,
        arrival_cutoff_ms=arrival_cutoff_ms,
    )
    tasks = apply_nonoracle_modeled_slack(tasks, protocol, seed=seed)
    metadata["slack_predictor"] = {
        "kind": "independent_registered_output_distribution_v1",
        "seed_derivation": "seed XOR 0x51ACCA11",
        "reads_realized_output_tokens": False,
        "reads_realized_service_ms": False,
        "physical_gbdt_calibration": False,
    }
    return tasks, metadata


class ExactAdmissionProjectionReplay(v3.ActiveHBMCoupledReplay):
    """Shared-HBM gate based on exact policy admission, not potential bytes."""

    def __init__(self, protocol: dict[str, Any], policy: str, capacity_protocol: dict[str, Any]) -> None:
        super().__init__(protocol, policy, capacity_protocol)
        self.exact_projection_evaluation_count = 0
        self.old_projection_false_block_request_ids: set[str] = set()
        self.old_projection_excess_bytes_max = 0
        self.projection_prediction_mismatch_count = 0

    def _profile(self, task: core.Task) -> core.RequestRecoveryProfile:
        rank = task.source_rank
        peer_mode, _dram_mode = core.topology_for_rank(rank, self.protocol)
        slack = 0.0 if self.policy == "AegisKV no Algorithm 1" else task.predicted_slack_ms
        return core.RequestRecoveryProfile(
            request_id=task.request_id,
            source_worker=f"source-{rank}",
            source_failure_domain=f"node-{rank // 8}",
            hbm_target=f"peer-{rank}-{peer_mode}",
            hbm_failure_domain=(
                f"node-{rank // 8}"
                if peer_mode == "same_node"
                else f"node-{(rank // 8 + 1) % 8}"
            ),
            route_version=0,
            source_token_end=task.prompt_tokens,
            slack_ms=slack,
            model_revision="Qwen3-8B-frozen",
            kv_abi="qwen3-fp16-kv-v1",
            parallel_layout="TP1-PP1",
        )

    def exact_projected_backup_after_start(
        self, rank: int, task: core.Task, at: float,
    ) -> int:
        if self.is_aegis:
            controller = deepcopy(self.aegis[rank])
            delta = float(at) - float(self.aegis_time[rank])
            if delta < -1e-8:
                raise ValueError("projection time regressed")
            if delta > 0:
                controller.advance_time(delta)
            controller.register_request(self._profile(task))
            chunk_tokens = int(self.protocol["aegiskv"]["prefix_chunk_tokens"])
            cursor = 0
            ordinal = 0
            while cursor < task.prompt_tokens:
                end = min(cursor + chunk_tokens, task.prompt_tokens)
                spec = core.ChunkSpec(
                    request_id=task.request_id,
                    ordinal=ordinal,
                    token_start=cursor,
                    token_end=end,
                    byte_size=(end - cursor) * self.kv,
                    route_version=0,
                    source_worker=f"source-{rank}",
                    hbm_target=self._profile(task).hbm_target,
                    model_revision="Qwen3-8B-frozen",
                    kv_abi="qwen3-fp16-kv-v1",
                    parallel_layout="TP1-PP1",
                )
                controller.admit_chunk(spec)
                cursor = end
                ordinal += 1
            return int(controller.telemetry_snapshot()["pools"]["HBM"]["used_bytes"])

        runtimes = deepcopy(self.baselines[rank])
        profile = core.BaselineRequestProfile(
            request_id=task.request_id,
            source_worker=f"source-{rank}",
            source_failure_domain=f"node-{rank // 8}",
            target_worker=f"target-{rank}",
            target_failure_domain=f"target-node-{rank // 8}",
            route_version=0,
            total_tokens=task.prompt_tokens,
            slack_ms=task.predicted_slack_ms,
        )
        chunk = core.SealedChunk(0, 0, task.prompt_tokens, "prefill")
        for runtime in runtimes:
            runtime.register_request(profile, time_ms=at)
            runtime.seal_chunk(task.request_id, chunk, time_ms=at)
        return sum(
            int(runtime.used_bytes)
            for runtime in runtimes
            if runtime.config.tier.value == "HBM"
        )

    def dispatch_block_reason(self, rank: int, at: float) -> str | None:
        if not self.waiting[rank]:
            return None
        if not self.free_lanes[rank]:
            return "ENGINE_LANE"
        task = self.waiting[rank][0]
        exact_backup = self.exact_projected_backup_after_start(rank, task, at)
        conservative_backup = super().projected_backup_after_start(rank, task)
        serving_after = (
            self.active_serving_bytes[rank] + self.serving_reservation_bytes(task)
        )
        exact_total = serving_after + exact_backup
        conservative_total = serving_after + conservative_backup
        self.exact_projection_evaluation_count += 1
        self.old_projection_excess_bytes_max = max(
            self.old_projection_excess_bytes_max,
            conservative_backup - exact_backup,
        )
        if conservative_total > self.shared_budget_bytes and exact_total <= self.shared_budget_bytes:
            self.old_projection_false_block_request_ids.add(task.request_id)
        if exact_total > self.shared_budget_bytes:
            return "SHARED_HBM"
        return None

    def try_dispatch(self, rank: int, at: float, *, event: str) -> int:
        dispatched = 0
        while self.waiting[rank]:
            reason = self.dispatch_block_reason(rank, at)
            if reason is not None:
                if reason == "SHARED_HBM":
                    self.memory_blocked_observation_count += 1
                    self.memory_blocked_request_ids.add(self.waiting[rank][0].request_id)
                else:
                    self.lane_blocked_observation_count += 1
                self.record_capacity(rank, at, f"{event}_blocked_{reason.lower()}")
                break
            nominal = self.waiting[rank].popleft()
            projected_backup = self.exact_projected_backup_after_start(rank, nominal, at)
            lane = heapq.heappop(self.free_lanes[rank])
            serving_bytes = self.serving_reservation_bytes(nominal)
            runtime_task = replace(nominal, lane=lane, start_ms=float(at), finish_ms=float(at))
            self.active_by_rank[rank].add(runtime_task.request_id)
            self.active_serving_bytes[rank] += serving_bytes
            self.serving_bytes_by_request[runtime_task.request_id] = serving_bytes
            self.start(runtime_task, at)
            if self.hbm_backup_used_bytes(rank) != projected_backup:
                self.projection_prediction_mismatch_count += 1
            burden = float(self.health_burden_ms[runtime_task.request_id])
            runtime_task = replace(
                runtime_task,
                finish_ms=float(at) + runtime_task.service_ms + burden,
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
        mechanism = super().summarize_mechanism(control_horizon_ms)
        mechanism["active_hbm_v31"] = {
            "projection_semantics": "exact_dry_run_policy_admission",
            "projection_reads_realized_output_tokens": False,
            "exact_projection_evaluation_count": self.exact_projection_evaluation_count,
            "old_projection_false_block_unique_request_count": len(
                self.old_projection_false_block_request_ids
            ),
            "old_projection_excess_bytes_max": self.old_projection_excess_bytes_max,
            "projection_prediction_mismatch_count": self.projection_prediction_mismatch_count,
        }
        return mechanism

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
                self.active_serving_bytes[rank] -= self.serving_bytes_by_request.pop(task.request_id)
                heapq.heappush(self.free_lanes[rank], task.lane)
                self.finish(task, at)
                self.record_capacity(rank, at, "finish_release")
                self.try_dispatch(rank, at, event="finish")
            elif kind == "coupled_control":
                external = {
                    rank: self.dispatch_block_reason(rank, at) == "SHARED_HBM"
                    for rank in range(self.dp)
                }
                before_backup = sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
                waiting_before = sum(len(rows) for rows in self.waiting)
                active_before = sum(len(rows) for rows in self.active_by_rank.values())
                self.control(
                    at,
                    {rank: len(rows) for rank, rows in self.active_by_rank.items()},
                    external_pressure_by_rank=external,
                )
                after_backup = sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
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
                delays = {}
                for rank in failure_domain_ranks(
                    probe,
                    gpus_per_node=int(self.protocol["hardware"]["gpus_per_node"]),
                    dp=self.dp,
                ):
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


def run_v31_cell(
    protocol: dict[str, Any], capacity_protocol: dict[str, Any], *,
    policy: str, seed: int, active_per_gpu: int, probe_count: int,
    output: Path, arrival_cutoff_ms: float,
) -> dict[str, Any]:
    nominal, workload = build_workload(
        protocol,
        seed=seed,
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
    replay = ExactAdmissionProjectionReplay(protocol, policy, capacity_protocol)
    tasks, delay_maps, mechanism, capacity_trace = replay.run_coupled(
        nominal, probes, control_horizon_ms=arrival_cutoff_ms
    )
    ids, base_wct = core.workflow_wct(tasks)
    no_fault = core.replay_one_fault(tasks, {})
    mechanism["empty_fault_replay_max_completion_delta_ms"] = max(
        abs(no_fault[row.request_id] - row.finish_ms) for row in tasks
    )
    fault_wct = []
    changed_by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"wct": [], "added": [], "all_wct": [], "all_added": []}
    )
    changed_counts = []
    for probe, delays in zip(probes, delay_maps):
        completion = core.replay_one_fault(tasks, delays)
        replay_ids, values = core.workflow_wct(tasks, completion)
        if replay_ids != ids:
            raise AssertionError("v3.1 workflow identity changed")
        changed = 0
        for base_value, fault_value in zip(base_wct, values):
            delta = fault_value - base_value
            if delta < -1e-7:
                raise AssertionError("v3.1 fault replay reduced WCT")
            if delta > 1e-7:
                changed += 1
                changed_by_type[probe.failure_type]["wct"].append(fault_value)
                changed_by_type[probe.failure_type]["added"].append(delta)
            changed_by_type[probe.failure_type]["all_wct"].append(fault_value)
            changed_by_type[probe.failure_type]["all_added"].append(max(delta, 0.0))
        changed_counts.append(changed)
        fault_wct.append(values)
    failure_stats = {
        failure_type: {
            "changed_workflow_observations": len(values["wct"]),
            "changed_wct_mean_ms": statistics.fmean(values["wct"]) if values["wct"] else None,
            "changed_wct_p95_ms": core.quantile(values["wct"], 0.95),
            "added_wct_mean_ms": statistics.fmean(values["added"]) if values["added"] else None,
            "added_wct_p95_ms": core.quantile(values["added"], 0.95),
            "all_workflow_observations": len(values["all_wct"]),
            "same_probe_failure_conditioned_wct_mean_ms": statistics.fmean(values["all_wct"]),
            "all_workflow_wct_mean_ms": statistics.fmean(values["all_wct"]),
            "all_workflow_wct_p95_ms": core.quantile(values["all_wct"], 0.95),
            "all_workflow_added_wct_mean_ms": statistics.fmean(values["all_added"]),
            "all_workflow_added_wct_p95_ms": core.quantile(values["all_added"], 0.95),
            "changed_workflow_fraction": len(values["wct"]) / len(values["all_wct"]),
        }
        for failure_type, values in sorted(changed_by_type.items())
        if values["all_wct"]
    }
    trace_array = np.asarray([
        (
            row["time_ms"], row["rank"], row["active_requests"], row["waiting_requests"],
            row["active_serving_kv_bytes"], row["peer_hbm_backup_used_bytes"],
            row["shared_used_bytes"], row["shared_headroom_bytes"], row["total_hbm_used_bytes"],
        )
        for row in capacity_trace
    ], dtype=np.float64).reshape((-1, 9))
    domain_rank_lists = [
        failure_domain_ranks(
            probe,
            gpus_per_node=int(protocol["hardware"]["gpus_per_node"]),
            dp=int(protocol["models"]["qwen3_8b"]["center_system_dp"]),
        )
        for probe in probes
    ]
    maximum_domain_width = int(protocol["hardware"]["gpus_per_node"])
    domain_rank_array = np.full(
        (len(domain_rank_lists), maximum_domain_width), -1, dtype=np.int16
    )
    for index, ranks in enumerate(domain_rank_lists):
        domain_rank_array[index, :len(ranks)] = ranks
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
        probe_failure_domain_rank_ids=domain_rank_array,
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
    workflow_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    probe_payload = [
        {
            "probe_id": row.probe_id,
            "time_ms": row.time_ms,
            "source_rank": row.source_rank,
            "failure_type": row.failure_type,
        }
        for row in probes
    ]
    probe_hash = hashlib.sha256(json.dumps(
        probe_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    failure_domain_payload = [
        {"probe_id": probe.probe_id, "rank_ids": list(ranks)}
        for probe, ranks in zip(probes, domain_rank_lists)
    ]
    failure_domain_hash = hashlib.sha256(json.dumps(
        failure_domain_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
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
        "workflow_universe_sha256": workflow_hash,
        "fault_probe_tuple_count": len(probes),
        "fault_probe_tuple_sha256": probe_hash,
        "fault_domain_tuple_sha256": failure_domain_hash,
        "workload": workload,
        "mechanism": mechanism,
        "raw_npz": str(output),
        "raw_npz_sha256": core.sha256(output),
    }
