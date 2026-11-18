"""Independent 90 s Poisson exposure windows for v3.1.3.

Every window owns a frozen event ledger.  Zero-event windows may reuse the
matching executed healthy DES after an exact equivalence gate; event-bearing
windows always execute the full persistent multi-fault state machine.  The
reported estimator is the arithmetic mean of the resulting window samples,
not a conditional result multiplied by an analytic probability.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import statistics
from typing import Any, Iterable

import active_hbm_v313 as v313
import poisson_fault_injection_v1 as v1
import run_preliminary as core


EXPERIMENT_ID = "active_hbm_v313_accounting_batched_s2"


def window_seed(workload_seed: int, window_index: int, rate: float) -> int:
    payload = (
        f"{EXPERIMENT_ID}|{int(workload_seed)}|{int(window_index)}|"
        f"{format(float(rate), '.9f')}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def generate_batch_windows(
    *, rate: float, workload_seeds: Iterable[int], windows_per_seed: int,
    horizon_ms: float, dp: int, gpus_per_node: int, repair_ms: float,
    failure_type_weights: dict[str, float],
) -> list[dict[str, Any]]:
    """Generate and resolve every independent window before any DES replay."""

    if int(windows_per_seed) <= 0:
        raise ValueError("windows_per_seed must be positive")
    rows: list[dict[str, Any]] = []
    for workload_seed in tuple(int(value) for value in workload_seeds):
        for index in range(int(windows_per_seed)):
            seed = window_seed(workload_seed, index, rate)
            ledger = v1.generate_poisson_fault_ledger(
                rate_per_1000_gpu_shard_hour=float(rate),
                seed=seed,
                horizon_ms=float(horizon_ms),
                dp=int(dp),
                gpus_per_node=int(gpus_per_node),
                repair_ms=float(repair_ms),
                failure_type_weights=dict(failure_type_weights),
            )
            rows.append({
                "window_id": f"s{workload_seed}-r{format(float(rate), '.3f')}-w{index:04d}",
                "workload_seed": workload_seed,
                "window_index": index,
                "fault_seed": seed,
                "rate_per_1000_gpu_shard_hour": float(rate),
                "horizon_ms": float(horizon_ms),
                "candidate_event_count": len(ledger),
                "accepted_event_count": sum(bool(event["accepted"]) for event in ledger),
                "noop_event_count": sum(not bool(event["accepted"]) for event in ledger),
                "event_ledger_sha256": canonical_sha256(ledger),
                "event_ledger": ledger,
            })
    return rows


def expected_events(
    rate_per_1000_gpu_shard_hour: float, *, dp: int, horizon_ms: float,
    window_count: int,
) -> float:
    return (
        float(rate_per_1000_gpu_shard_hour)
        * int(dp) / 1000.0
        * float(horizon_ms) / 3_600_000.0
        * int(window_count)
    )


def summarize_window_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("window samples are empty")
    values = [float(row["wct_mean_ms"]) for row in rows]
    seeds = sorted({int(row["workload_seed"]) for row in rows})
    mean = statistics.fmean(values)
    standard_error = (
        statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    )
    return {
        "window_count": len(rows),
        "unique_workload_seed_count": len(seeds),
        "workload_seeds": seeds,
        "event_bearing_window_count": sum(int(row["candidate_event_count"]) > 0 for row in rows),
        "memoized_zero_event_window_count": sum(bool(row["memoized_zero_event"]) for row in rows),
        "candidate_event_count": sum(int(row["candidate_event_count"]) for row in rows),
        "accepted_event_count": sum(int(row["accepted_event_count"]) for row in rows),
        "noop_event_count": sum(int(row["noop_event_count"]) for row in rows),
        "wct_mean_ms": mean,
        "window_sample_standard_error_ms": standard_error,
        "window_sample_normal_95_half_width_ms": 1.96 * standard_error,
        "statistical_boundary": (
            "window-level fault-randomness interval; workload-seed uncertainty "
            "requires the preregistered three-seed batch"
        ),
    }


class BatchedPoissonFaultInjectedReplay(
    v1.PoissonFaultInjectedReplay,
    v313.CausalPhysicalTargetReplay,
):
    """v1 multi-fault state transitions on the v3.1.3 healthy scheduler."""

    def _snapshot_replicas(self, task: core.Task, at: float) -> None:
        original = v1.v312.topology_placement
        v1.v312.topology_placement = v313.topology_placement
        try:
            return super()._snapshot_replicas(task, at)
        finally:
            v1.v312.topology_placement = original

    def run_rate_injected(
        self, nominal_tasks: list[core.Task], event_ledger: list[dict[str, Any]], *,
        control_horizon_ms: float,
    ) -> tuple[list[core.Task], dict[str, Any], list[dict[str, Any]]]:
        self.task_by_id = {row.request_id: row for row in nominal_tasks}
        self.occupancy_observation_horizon_ms = float(control_horizon_ms)
        for rank in range(self.dp):
            self.record_capacity(rank, 0.0, "initial")
        for row in sorted(nominal_tasks, key=lambda item: (item.arrival_ms, item.request_id)):
            self.push(row.arrival_ms, 1, "coupled_arrival", row)
        for event in event_ledger:
            self.push(float(event["time_ms"]), 2, "injected_fault", event)
        window = float(self.protocol["aegiskv"]["window_ms"])
        at = window
        while at <= float(control_horizon_ms) + 1e-9:
            self.scheduled_control_opportunity_times_ms.append(float(at))
            self.push(at, 3, "coupled_control", None)
            at += window
        while self.events:
            at, _priority, _sequence, kind, payload = heapq.heappop(self.events)
            if kind == "aegis_copy":
                self.complete_copy(payload, at)
            elif kind == "protection_seal":
                self.handle_protection_seal(payload, at)
            elif kind == "coupled_arrival":
                task = payload
                self.waiting[task.source_rank].append(task)
                self.try_dispatch(task.source_rank, at, event="arrival")
            elif kind == "coupled_finish":
                task = payload
                current = self.task_by_id[task.request_id]
                if not math.isclose(
                    float(task.finish_ms), float(current.finish_ms),
                    rel_tol=0.0, abs_tol=1e-8,
                ):
                    self.stale_finish_event_count += 1
                    continue
                rank = current.source_rank
                if current.request_id not in self.active_by_rank.get(rank, set()):
                    self.stale_finish_event_count += 1
                    continue
                self.active_by_rank[rank].discard(current.request_id)
                self.active_serving_bytes[rank] -= self.serving_bytes_by_request.pop(
                    current.request_id
                )
                heapq.heappush(self.free_lanes[rank], current.lane)
                self.finish(current, at)
                self.record_capacity(rank, at, "finish_release")
                self._retry_all(at, "finish")
            elif kind == "injected_fault":
                self._apply_fault_event(payload, at)
                self._retry_all(at, "fault")
            elif kind == "logical_route_ready":
                rank = int(payload)
                if at + 1e-9 >= self.logical_unavailable_until[rank]:
                    self.try_dispatch(rank, at, event="route_ready")
            elif kind == "coupled_control":
                external = {
                    rank: self.dispatch_block_reason(rank, at) == "SHARED_HBM"
                    for rank in range(self.dp)
                }
                before = sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
                self.control(
                    at,
                    {rank: len(rows) for rank, rows in self.active_by_rank.items()},
                    external_pressure_by_rank=external,
                )
                after = sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
                self.backup_bytes_released_by_control += max(before - after, 0)
                self._retry_all(at, "control")
            else:
                raise AssertionError(f"unknown batched injected event {kind!r}")
        if any(self.waiting) or any(self.active_by_rank.values()):
            raise AssertionError("batched fault-injected scheduler failed to drain")
        tasks = sorted(
            self.task_by_id.values(), key=lambda row: (row.start_ms, row.request_id)
        )
        if len(tasks) != len(nominal_tasks):
            raise AssertionError("batched fault-injected scheduler lost requests")
        mechanism = self.summarize_mechanism(control_horizon_ms)
        mechanism["poisson_fault_batch_v2"] = {
            "candidate_event_count": len(event_ledger),
            "accepted_event_count": sum(bool(row["accepted"]) for row in event_ledger),
            "noop_event_count": sum(not bool(row["accepted"]) for row in event_ledger),
            "stale_finish_event_count": self.stale_finish_event_count,
            "replica_invalidated_bytes": self.replica_invalidated_bytes,
            "recovery_action_counts": dict(sorted(self.injected_recovery_counts.items())),
            "event_ledger": self.injected_event_ledger,
        }
        return tasks, mechanism, list(self.capacity_trace)
