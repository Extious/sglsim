"""Auditable Poisson multi-fault injection for the v3.1.2 DES.

Fault candidates are generated over the fixed arrival horizon.  A fail-stop
domain remains down for the registered repair interval.  Repeated hits on an
already-down worker or node are ledgered no-ops and never apply damage twice.
The event replay updates serving completion times, lane queues, protection
replica validity, and protection-capacity accounting before later events.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable

import active_hbm_v312 as v312
import run_preliminary as core


@dataclass(frozen=True)
class FaultCandidate:
    time_ms: float
    failure_type: str
    source_rank: int


def _state_hash(nodes: dict[int, float], ranks: dict[int, float]) -> str:
    payload = {
        "nodes": sorted((int(key), float(value)) for key, value in nodes.items()),
        "ranks": sorted((int(key), float(value)) for key, value in ranks.items()),
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_fault_candidates(
    candidates: Iterable[FaultCandidate], *, dp: int, gpus_per_node: int,
    repair_ms: float,
) -> list[dict[str, Any]]:
    """Resolve fail-stop state, retaining duplicate candidates as no-op rows."""
    if dp <= 0 or dp % gpus_per_node:
        raise ValueError("DP must contain complete nodes")
    node_count = dp // gpus_per_node
    node_down_until: dict[int, float] = {}
    rank_down_until: dict[int, float] = {}
    ledger: list[dict[str, Any]] = []
    for index, candidate in enumerate(sorted(
        candidates, key=lambda row: (row.time_ms, row.failure_type, row.source_rank)
    )):
        at = float(candidate.time_ms)
        if not 0 <= int(candidate.source_rank) < dp:
            raise ValueError("fault source rank outside DP")
        node_down_until = {
            key: value for key, value in node_down_until.items() if value > at
        }
        rank_down_until = {
            key: value for key, value in rank_down_until.items() if value > at
        }
        rank = int(candidate.source_rank)
        node = rank // gpus_per_node
        before_hash = _state_hash(node_down_until, rank_down_until)
        accepted = False
        outcome = ""
        repair_at = at
        if candidate.failure_type == "source_worker_gpu":
            if node in node_down_until:
                outcome = "NOOP_NODE_ALREADY_DOWN"
                repair_at = node_down_until[node]
            elif rank in rank_down_until:
                outcome = "NOOP_WORKER_ALREADY_DOWN"
                repair_at = rank_down_until[rank]
            else:
                accepted = True
                outcome = "APPLIED_WORKER_FAIL_STOP"
                repair_at = at + float(repair_ms)
                rank_down_until[rank] = repair_at
            domain = f"gpu:{rank}"
            ranks = [rank]
        elif candidate.failure_type == "source_node":
            if node in node_down_until:
                outcome = "NOOP_NODE_ALREADY_DOWN"
                repair_at = node_down_until[node]
            else:
                accepted = True
                outcome = "APPLIED_NODE_FAIL_STOP"
                repair_at = at + float(repair_ms)
                node_down_until[node] = repair_at
                # The broader node outage supersedes individual worker state.
                for member in range(node * gpus_per_node, (node + 1) * gpus_per_node):
                    rank_down_until.pop(member, None)
            domain = f"node:{node}"
            ranks = list(range(node * gpus_per_node, (node + 1) * gpus_per_node))
        else:
            raise ValueError(f"unsupported failure type {candidate.failure_type!r}")
        if not 0 <= node < node_count:
            raise AssertionError("fault node outside topology")
        ledger.append({
            "event_id": f"fault-{index:04d}",
            "candidate_index": index,
            "time_ms": at,
            "failure_type": candidate.failure_type,
            "source_rank": rank,
            "domain": domain,
            "domain_rank_ids": ranks,
            "accepted": accepted,
            "outcome": outcome,
            "repair_at_ms": repair_at,
            "state_before_sha256": before_hash,
            "down_nodes_after": sorted(node_down_until),
            "down_ranks_after": sorted(rank_down_until),
            "state_after_sha256": _state_hash(node_down_until, rank_down_until),
        })
    return ledger


def generate_poisson_fault_ledger(
    *, rate_per_1000_gpu_shard_hour: float, seed: int, horizon_ms: float,
    dp: int, gpus_per_node: int, repair_ms: float,
    failure_type_weights: dict[str, float],
) -> list[dict[str, Any]]:
    rate = float(rate_per_1000_gpu_shard_hour)
    if rate < 0 or horizon_ms <= 0:
        raise ValueError("invalid Poisson rate or horizon")
    if rate == 0:
        return []
    names = sorted(failure_type_weights)
    weights = [float(failure_type_weights[name]) for name in names]
    if names != ["source_node", "source_worker_gpu"] or sum(weights) <= 0:
        raise ValueError("worker/node failure weights are required")
    seed_value = int(seed) ^ 0xF4117E5 ^ int(round(rate * 1_000_000))
    rng = random.Random(seed_value)
    events_per_ms = rate * int(dp) / 1000.0 / 3_600_000.0
    at = rng.expovariate(events_per_ms)
    candidates: list[FaultCandidate] = []
    while at <= float(horizon_ms):
        candidates.append(FaultCandidate(
            time_ms=at,
            failure_type=rng.choices(names, weights=weights, k=1)[0],
            source_rank=rng.randrange(int(dp)),
        ))
        at += rng.expovariate(events_per_ms)
    return resolve_fault_candidates(
        candidates, dp=dp, gpus_per_node=gpus_per_node, repair_ms=repair_ms
    )


class PoissonFaultInjectedReplay(
    v312.TopologyAwareExactAdmissionProjectionReplay
):
    """Fault-aware event loop with persistent replica and route state."""

    def __init__(self, protocol: dict, policy: str, capacity_protocol: dict) -> None:
        super().__init__(protocol, policy, capacity_protocol)
        self.request_route_rank: dict[str, int] = {}
        self.logical_route_rank = list(range(self.dp))
        self.logical_unavailable_until = [0.0] * self.dp
        self.replica_registry: dict[str, list[dict[str, Any]]] = {}
        self.frozen_after_source_failure: set[str] = set()
        self.injected_event_ledger: list[dict[str, Any]] = []
        self.injected_recovery_counts: Counter[str] = Counter()
        self.stale_finish_event_count = 0
        self.replica_invalidated_bytes = 0

    def dispatch_block_reason(self, rank: int, at: float) -> str | None:
        if self.waiting[rank] and float(at) < self.logical_unavailable_until[rank] - 1e-9:
            return "FAILURE_DOMAIN"
        return super().dispatch_block_reason(rank, at)

    def try_dispatch(self, rank: int, at: float, *, event: str) -> int:
        before = set(self.active_by_rank.get(rank, ()))
        count = super().try_dispatch(rank, at, event=event)
        for request_id in set(self.active_by_rank.get(rank, ())) - before:
            self.request_route_rank[request_id] = self.logical_route_rank[rank]
        return count

    def _dram_target_rank(self, placement: dict[str, Any]) -> int:
        local = int(placement["local_gpu_index"])
        if placement["dram_mode"] == "local":
            local = (local + 1) % int(self.protocol["hardware"]["gpus_per_node"])
        return (
            int(placement["dram_target_node"])
            * int(self.protocol["hardware"]["gpus_per_node"])
            + local
        )

    def _snapshot_replicas(self, task: core.Task, at: float) -> None:
        if task.request_id in self.frozen_after_source_failure:
            return
        placement = v312.topology_placement(self.protocol, task.source_rank)
        replicas: list[dict[str, Any]] = []
        if self.is_aegis:
            self.advance_aegis(task.source_rank, at)
            controller = self.aegis[task.source_rank]
            for tier, target_node, target_rank, mode in (
                (
                    core.ProtectionTier.HBM,
                    placement["peer_hbm_target_node"],
                    placement["peer_hbm_target_rank"],
                    placement["peer_hbm_mode"],
                ),
                (
                    core.ProtectionTier.DRAM,
                    placement["dram_target_node"],
                    self._dram_target_rank(placement),
                    placement["dram_mode"],
                ),
            ):
                prefix = controller.committed_prefix_tokens(task.request_id, tier)
                replicas.append({
                    "tier": tier.value,
                    "prefix_tokens": int(prefix),
                    "target_node": int(target_node),
                    "target_rank": int(target_rank),
                    "mode": str(mode),
                    "valid": bool(prefix),
                })
        else:
            for runtime in self.baselines[task.source_rank]:
                decision = runtime.estimate_source_failure(
                    task.request_id,
                    produced_tokens=task.prompt_tokens,
                    remaining_slack_ms=0.0,
                    time_ms=at,
                )
                tier = runtime.config.tier.value
                target_node = (
                    placement["peer_hbm_target_node"]
                    if tier == "HBM" else placement["dram_target_node"]
                )
                target_rank = (
                    placement["peer_hbm_target_rank"]
                    if tier == "HBM" else self._dram_target_rank(placement)
                )
                replicas.append({
                    "tier": tier,
                    "prefix_tokens": int(decision.restored_tokens),
                    "target_node": int(target_node),
                    "target_rank": int(target_rank),
                    "mode": (
                        placement["peer_hbm_mode"]
                        if tier == "HBM" else placement["dram_mode"]
                    ),
                    "valid": bool(decision.restored_tokens),
                    "runtime": runtime,
                })
        self.replica_registry[task.request_id] = replicas

    @staticmethod
    def _replica_hit(replica: dict[str, Any], event: dict[str, Any]) -> bool:
        if event["failure_type"] == "source_node":
            return replica["target_node"] == event["source_rank"] // 8
        return replica["tier"] == "HBM" and replica["target_rank"] == event["source_rank"]

    def _invalidate_replicas(self, task: core.Task, event: dict[str, Any], at: float) -> int:
        released = 0
        for replica in self.replica_registry.get(task.request_id, []):
            if not replica["valid"] or not self._replica_hit(replica, event):
                continue
            replica["valid"] = False
            replica["prefix_tokens"] = 0
            if self.is_aegis:
                self.advance_aegis(task.source_rank, at)
                result = self.aegis[task.source_rank].invalidate_tier(
                    task.request_id,
                    core.ProtectionTier(replica["tier"]),
                    reason=f"injected_{event['domain']}_failed",
                )
                released += int(result["released_bytes"])
                self.refresh_occupancy(task.source_rank, at)
            else:
                runtime = replica["runtime"]
                runtime.advance_to(at)
                state = runtime._requests.get(task.request_id)  # component adapter has no public tier invalidation
                before = 0 if state is None else int(state.used_bytes)
                if state is not None:
                    runtime._release_request(
                        task.request_id, reason=f"injected_{event['domain']}_failed"
                    )
                released += before
        self.replica_invalidated_bytes += released
        return released

    def _recovery_from_registry(
        self, task: core.Task, event: dict[str, Any]
    ) -> tuple[float, str, int, int, int]:
        valid = [
            row for row in self.replica_registry.get(task.request_id, [])
            if row["valid"] and row["prefix_tokens"] > 0
        ]
        recompute_ms = 4.0 + task.prompt_tokens * float(self.model["prefill_ms_per_token"])
        candidates: list[tuple[float, str, int, int, int]] = [
            (recompute_ms, "RECOMPUTE", 0, task.prompt_tokens, task.source_rank)
        ]
        hbm_bw, dram_bw = core.load_bandwidth(self.protocol, task.source_rank)
        for replica in valid:
            prefix = min(int(replica["prefix_tokens"]), task.prompt_tokens)
            bw = hbm_bw if replica["tier"] == "HBM" else dram_bw
            setup = 0.08 if replica["tier"] == "HBM" else 0.15
            service = setup + prefix * self.kv / (bw * core.GB_PER_S_TO_BYTES_PER_MS)
            suffix = task.prompt_tokens - prefix
            if suffix:
                service += 4.0 + suffix * float(self.model["prefill_ms_per_token"])
            action = (
                f"peer_hbm_{replica['mode']}"
                if replica["tier"] == "HBM"
                else f"dram_{replica['mode']}"
            )
            if suffix:
                action += "+RECOMPUTE"
            candidates.append((
                service, action, prefix, suffix, int(replica["target_rank"])
            ))
        if self.policy == "Native Mooncake":
            activation = float(
                self.protocol["failure_model"]["policy_failover"]["replacement_delay_ms"]
            )
            mode = "replacement_wait"
        elif event["failure_type"] == "source_worker_gpu" or valid:
            activation = float(
                self.protocol["failure_model"]["policy_failover"]["fence_activation_delay_ms"]
            )
            mode = "prebound_fence_activation"
        else:
            activation = float(
                self.protocol["failure_model"]["policy_failover"]["replacement_delay_ms"]
            )
            mode = "replacement_no_survivor"
        service, action, restored, recomputed, route_rank = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        if action == "RECOMPUTE" and activation >= 10000.0:
            route_rank = task.source_rank
        elif action == "RECOMPUTE" and valid:
            route_rank = int(valid[0]["target_rank"])
        self.failover_mode_counts[f"{event['failure_type']}:{mode}"] += 1
        self.failover_delay_values_ms.append(activation)
        self.recovery_service_values_ms.append(service)
        self.recovery_restored_tokens += restored
        self.recovery_recomputed_tokens += recomputed
        self.recovery_counts[action] += 1
        self.recovery_type_path_counts[f"{event['failure_type']}:{action}"] += 1
        return activation + service, action, restored, recomputed, route_rank

    @staticmethod
    def _route_hit(route_rank: int, event: dict[str, Any]) -> bool:
        if event["failure_type"] == "source_worker_gpu":
            return int(route_rank) == int(event["source_rank"])
        return int(route_rank) // 8 == int(event["source_rank"]) // 8

    def _apply_fault_event(self, event: dict[str, Any], at: float) -> None:
        row = dict(event)
        row.update({
            "active_request_count_before": sum(len(v) for v in self.active_by_rank.values()),
            "queue_count_before": sum(len(v) for v in self.waiting),
            "affected_request_count": 0,
            "delayed_request_count": 0,
            "replica_invalidated_bytes": 0,
            "surviving_replica_count": 0,
            "recovery_actions": {},
        })
        if not event["accepted"]:
            row["queue_count_after"] = row["queue_count_before"]
            row["capacity_shared_used_bytes_after"] = sum(
                self.active_serving_bytes[rank] + self.hbm_backup_used_bytes(rank)
                for rank in range(self.dp)
            )
            self.injected_event_ledger.append(row)
            return

        active_ids = sorted(
            request_id
            for values in self.active_by_rank.values()
            for request_id in values
        )
        for request_id in active_ids:
            self._snapshot_replicas(self.task_by_id[request_id], at)
        for request_id in active_ids:
            row["replica_invalidated_bytes"] += self._invalidate_replicas(
                self.task_by_id[request_id], event, at
            )
        actions: Counter[str] = Counter()
        for request_id in active_ids:
            route = self.request_route_rank.get(
                request_id, self.task_by_id[request_id].source_rank
            )
            if not self._route_hit(route, event):
                continue
            task = self.task_by_id[request_id]
            row["affected_request_count"] += 1
            if self.is_aegis:
                self.advance_aegis(task.source_rank, at)
                self.aegis[task.source_rank].abort_source_pending(
                    f"source-{task.source_rank}"
                )
                self.refresh_occupancy(task.source_rank, at)
            self.frozen_after_source_failure.add(request_id)
            delay, action, _restored, _recomputed, route_rank = (
                self._recovery_from_registry(task, event)
            )
            current = self.task_by_id[request_id]
            updated = replace(current, finish_ms=current.finish_ms + delay)
            self.task_by_id[request_id] = updated
            self.request_route_rank[request_id] = route_rank
            self.logical_route_rank[current.source_rank] = route_rank
            unavailable = at + min(
                delay,
                float(self.protocol["failure_model"]["policy_failover"]["replacement_delay_ms"]),
            )
            self.logical_unavailable_until[current.source_rank] = max(
                self.logical_unavailable_until[current.source_rank], unavailable
            )
            self.push(updated.finish_ms, 4, "coupled_finish", updated)
            self.push(
                self.logical_unavailable_until[current.source_rank],
                3,
                "logical_route_ready",
                current.source_rank,
            )
            actions[action] += 1
            self.injected_recovery_counts[action] += 1
            row["delayed_request_count"] += 1
        row["surviving_replica_count"] = sum(
            int(replica["valid"] and replica["prefix_tokens"] > 0)
            for replicas in self.replica_registry.values()
            for replica in replicas
        )
        row["recovery_actions"] = dict(sorted(actions.items()))
        row["queue_count_after"] = sum(len(v) for v in self.waiting)
        row["capacity_shared_used_bytes_after"] = sum(
            self.active_serving_bytes[rank] + self.hbm_backup_used_bytes(rank)
            for rank in range(self.dp)
        )
        for rank in range(self.dp):
            self.record_capacity(rank, at, f"fault_{event['event_id']}")
        self.injected_event_ledger.append(row)

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
            self.push(at, 3, "coupled_control", None)
            at += window
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
                self.try_dispatch(rank, at, event="finish")
            elif kind == "injected_fault":
                self._apply_fault_event(payload, at)
            elif kind == "logical_route_ready":
                rank = int(payload)
                if at + 1e-9 >= self.logical_unavailable_until[rank]:
                    self.try_dispatch(rank, at, event="route_ready")
            elif kind == "coupled_control":
                external = {
                    rank: self.dispatch_block_reason(rank, at) == "SHARED_HBM"
                    for rank in range(self.dp)
                }
                self.control(
                    at,
                    {rank: len(rows) for rank, rows in self.active_by_rank.items()},
                    external_pressure_by_rank=external,
                )
                for rank in range(self.dp):
                    self.record_capacity(rank, at, "control_after_quota")
                    self.try_dispatch(rank, at, event="control")
            else:
                raise AssertionError(f"unknown injected event {kind!r}")
        if any(self.waiting) or any(self.active_by_rank.values()):
            raise AssertionError("fault-injected scheduler failed to drain")
        tasks = sorted(
            self.task_by_id.values(), key=lambda row: (row.start_ms, row.request_id)
        )
        if len(tasks) != len(nominal_tasks):
            raise AssertionError("fault-injected scheduler lost requests")
        mechanism = self.summarize_mechanism(control_horizon_ms)
        mechanism["poisson_fault_injection_v1"] = {
            "candidate_event_count": len(event_ledger),
            "accepted_event_count": sum(bool(row["accepted"]) for row in event_ledger),
            "noop_event_count": sum(not bool(row["accepted"]) for row in event_ledger),
            "stale_finish_event_count": self.stale_finish_event_count,
            "replica_invalidated_bytes": self.replica_invalidated_bytes,
            "recovery_action_counts": dict(sorted(self.injected_recovery_counts.items())),
            "event_ledger": self.injected_event_ledger,
        }
        return tasks, mechanism, list(self.capacity_trace)

