"""Explicit shared-prefix identity/refcount model for the v3.1.3 S6 pilot.

The serving scheduler and source-owned independent HBM/DRAM transfer paths are
inherited unchanged from :mod:`active_hbm_v313`.  This experiment-only layer
changes only protection-object identity:

* requests in a deterministic same-source cohort share content-addressed
  prefix chunks;
* the first causal producer creates one physical protection object while later
  producers attach logical references without scheduling duplicate copies;
* each request's unique suffix remains an independent controller object; and
* recovery concatenates a still-COMMITTED shared prefix with a still-COMMITTED
  per-request suffix.  A missing shared chunk makes later suffix chunks
  unusable, preserving continuous-prefix recovery.

Logical references do not pin a fault-tolerance replica.  Capacity control may
evict a shared backup while requests still reference its identity; those
requests then correctly fall back to another committed tier or recomputation.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable

import active_hbm_v313 as v313
import run_preliminary as core


@dataclass(frozen=True)
class PrefixAssignment:
    request_id: str
    source_rank: int
    group_id: str
    shared_tokens: int
    unique_tokens: int
    cohort_position: int
    cohort_size: int


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_prefix_assignments(
    tasks: Iterable[core.Task], *, reuse_fraction: float,
    group_size: int, chunk_tokens: int,
) -> dict[str, PrefixAssignment]:
    """Assign deterministic same-source content identities without seed filtering."""

    reuse = float(reuse_fraction)
    if reuse < 0.0 or reuse > 1.0:
        raise ValueError("reuse_fraction must be in [0, 1]")
    if int(group_size) <= 0 or int(chunk_tokens) <= 0:
        raise ValueError("group_size and chunk_tokens must be positive")
    by_rank: dict[int, list[core.Task]] = defaultdict(list)
    for task in tasks:
        by_rank[int(task.source_rank)].append(task)
    assignments: dict[str, PrefixAssignment] = {}
    for rank, rows in sorted(by_rank.items()):
        ordered = sorted(rows, key=lambda row: (row.arrival_ms, row.request_id))
        for offset in range(0, len(ordered), int(group_size)):
            cohort = ordered[offset: offset + int(group_size)]
            prompts = {int(row.prompt_tokens) for row in cohort}
            if len(prompts) != 1:
                raise ValueError("a shared-prefix cohort must have one prompt length")
            prompt_tokens = prompts.pop()
            shared_float = prompt_tokens * reuse
            shared_tokens = int(round(shared_float))
            if abs(shared_float - shared_tokens) > 1e-9:
                raise ValueError("reuse point does not map to an integral token count")
            if shared_tokens % int(chunk_tokens):
                raise ValueError("shared prefix must end on a registered chunk boundary")
            group_index = offset // int(group_size)
            group_id = (
                f"rank{rank:04d}-group{group_index:06d}-"
                f"prompt{prompt_tokens}-shared{shared_tokens}"
            )
            for position, task in enumerate(cohort):
                assignments[task.request_id] = PrefixAssignment(
                    request_id=task.request_id,
                    source_rank=rank,
                    group_id=group_id,
                    shared_tokens=shared_tokens,
                    unique_tokens=prompt_tokens - shared_tokens,
                    cohort_position=position,
                    cohort_size=len(cohort),
                )
    if len(assignments) != sum(len(rows) for rows in by_rank.values()):
        raise AssertionError("prefix assignment lost requests")
    return assignments


def assignment_hash(assignments: dict[str, PrefixAssignment]) -> str:
    return digest_json([
        (
            row.request_id, row.source_rank, row.group_id,
            row.shared_tokens, row.unique_tokens,
            row.cohort_position, row.cohort_size,
        )
        for row in sorted(assignments.values(), key=lambda item: item.request_id)
    ])


class SharedPrefixReplay(v313.CausalPhysicalTargetReplay):
    """v3.1.3 replay with explicit content identity and logical references."""

    def __init__(
        self, protocol: dict[str, Any], policy: str,
        capacity_protocol: dict[str, Any], *,
        assignments: dict[str, PrefixAssignment], reuse_fraction: float,
        group_size: int,
    ) -> None:
        super().__init__(protocol, policy, capacity_protocol)
        self.assignments = dict(assignments)
        self.reuse_fraction = float(reuse_fraction)
        self.group_size = int(group_size)
        self.shared_owner_by_group: dict[str, str] = {}
        self.shared_owner_task: dict[str, core.Task] = {}
        self.shared_materialized_keys: set[tuple[str, int]] = set()
        self.shared_ref_members: dict[tuple[str, int], set[str]] = defaultdict(set)
        self.shared_refcount_value: dict[tuple[str, int], int] = defaultdict(int)
        self.request_shared_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
        self.shared_reference_count = 0
        self.shared_reference_alias_hit_count = 0
        self.refcount_peak = 0
        self.refcount_max_abs_error = 0
        self.refcount_underflow_count = 0
        self.logical_useful_bytes = 0
        self.logical_shared_reference_bytes = 0
        self.logical_unique_reference_bytes = 0
        self.recovery_call_count = 0
        self.recovery_request_hit_count = 0
        self.shared_recovery_restored_tokens = 0
        self.unique_recovery_restored_tokens = 0
        self.physical_occupancy_last_ms = 0.0
        self.physical_occupancy_current_bytes = {"HBM": 0, "DRAM": 0}
        self.physical_occupancy_area_byte_ms = {"HBM": 0.0, "DRAM": 0.0}
        self.physical_occupancy_peak_bytes = {"HBM": 0, "DRAM": 0}
        self.physical_occupancy_trace: list[tuple[float, int, int]] = []

    def _assignment(self, request_id: str) -> PrefixAssignment:
        try:
            return self.assignments[str(request_id)]
        except KeyError as exc:
            raise KeyError(f"missing shared-prefix assignment for {request_id!r}") from exc

    def _owner_id(self, assignment: PrefixAssignment) -> str:
        return f"__s6_shared_prefix__{assignment.group_id}"

    def _physical_tier_used_bytes(self) -> dict[str, int]:
        if self.is_aegis:
            return {
                tier: sum(
                    int(controller.telemetry_snapshot()["pools"][tier]["used_bytes"])
                    for controller in self.aegis
                )
                for tier in ("HBM", "DRAM")
            }
        values = {"HBM": 0, "DRAM": 0}
        for runtimes in self.baselines:
            for runtime in runtimes:
                values[runtime.config.tier.value] += int(runtime.used_bytes)
        return values

    def _sample_physical_occupancy(self, at: float) -> None:
        if self.occupancy_observation_horizon_ms is None:
            return
        capped = min(float(at), float(self.occupancy_observation_horizon_ms))
        if capped < self.physical_occupancy_last_ms - 1e-9:
            raise ValueError("S6 occupancy time regressed")
        delta = max(capped - self.physical_occupancy_last_ms, 0.0)
        for tier in ("HBM", "DRAM"):
            self.physical_occupancy_area_byte_ms[tier] += (
                self.physical_occupancy_current_bytes[tier] * delta
            )
        self.physical_occupancy_last_ms = capped
        current = self._physical_tier_used_bytes()
        changed = current != self.physical_occupancy_current_bytes
        self.physical_occupancy_current_bytes = current
        for tier in current:
            self.physical_occupancy_peak_bytes[tier] = max(
                self.physical_occupancy_peak_bytes[tier], current[tier]
            )
        if changed or not self.physical_occupancy_trace:
            self.physical_occupancy_trace.append(
                (capped, current["HBM"], current["DRAM"])
            )

    def record_capacity(self, rank: int, at: float, event: str) -> None:
        super().record_capacity(rank, at, event)
        self._sample_physical_occupancy(at)

    def _register_actual(self, task: core.Task, assignment: PrefixAssignment, at: float) -> None:
        rank = task.source_rank
        if self.is_aegis:
            self.advance_aegis(rank, at)
            profile = replace(
                self._profile(task), source_token_end=int(assignment.unique_tokens)
            )
            self.aegis[rank].register_request(profile)
            self.refresh_occupancy(rank, at)
            return
        placement = v313.topology_placement(self.protocol, rank)
        profile = core.BaselineRequestProfile(
            request_id=task.request_id,
            source_worker=f"source-{rank}",
            source_failure_domain=f"node-{placement['source_node']}",
            target_worker=f"target-{rank}",
            target_failure_domain=f"target-node-{placement['peer_hbm_target_node']}",
            route_version=0,
            total_tokens=int(assignment.unique_tokens),
            slack_ms=task.predicted_slack_ms,
        )
        for runtime in self.baselines[rank]:
            runtime.register_request(profile, time_ms=at)

    def _ensure_shared_owner(
        self, task: core.Task, assignment: PrefixAssignment, at: float,
    ) -> str:
        current = self.shared_owner_by_group.get(assignment.group_id)
        if current is not None:
            return current
        owner_id = self._owner_id(assignment)
        owner_task = replace(
            task,
            request_id=owner_id,
            workflow_id=owner_id,
            prompt_tokens=int(assignment.shared_tokens),
            output_tokens=0,
        )
        self.shared_owner_by_group[assignment.group_id] = owner_id
        self.shared_owner_task[assignment.group_id] = owner_task
        self.task_by_id[owner_id] = owner_task
        self.active.add(owner_id)
        self.health_burden_ms[owner_id] = 0.0
        self.copy_service_ms[owner_id] = 0.0
        self.copy_queue_wait_ms[owner_id] = 0.0
        rank = task.source_rank
        if self.is_aegis:
            self.advance_aegis(rank, at)
            self.aegis[rank].register_request(
                replace(
                    self._profile(owner_task),
                    request_id=owner_id,
                    source_token_end=int(assignment.shared_tokens),
                )
            )
            self.refresh_occupancy(rank, at)
        else:
            placement = v313.topology_placement(self.protocol, rank)
            profile = core.BaselineRequestProfile(
                request_id=owner_id,
                source_worker=f"source-{rank}",
                source_failure_domain=f"node-{placement['source_node']}",
                target_worker=f"target-{rank}",
                target_failure_domain=f"target-node-{placement['peer_hbm_target_node']}",
                route_version=0,
                total_tokens=int(assignment.shared_tokens),
                slack_ms=task.predicted_slack_ms,
            )
            for runtime in self.baselines[rank]:
                runtime.register_request(profile, time_ms=at)
        return owner_id

    def start(self, task: core.Task, at: float) -> None:
        assignment = self._assignment(task.request_id)
        if assignment.source_rank != task.source_rank:
            raise AssertionError("prefix identity crossed source-rank placement")
        if assignment.shared_tokens + assignment.unique_tokens != task.prompt_tokens:
            raise AssertionError("shared plus unique tokens do not equal prompt")
        self.active.add(task.request_id)
        self.health_burden_ms[task.request_id] = 0.0
        self.copy_service_ms[task.request_id] = 0.0
        self.copy_queue_wait_ms[task.request_id] = 0.0
        prompt_bytes = int(task.prompt_tokens) * self.kv
        self.logical_useful_bytes += prompt_bytes
        self.logical_shared_reference_bytes += int(assignment.shared_tokens) * self.kv
        self.logical_unique_reference_bytes += int(assignment.unique_tokens) * self.kv
        self._register_actual(task, assignment, at)
        chunk_tokens = int(self.protocol["aegiskv"]["prefix_chunk_tokens"])
        for ordinal, start in enumerate(range(0, assignment.shared_tokens, chunk_tokens)):
            end = min(start + chunk_tokens, assignment.shared_tokens)
            self.push(
                self._seal_time(task, end), 0, "protection_seal",
                ("shared", task.request_id, ordinal, start, end),
            )
        for ordinal, start in enumerate(range(0, assignment.unique_tokens, chunk_tokens)):
            end = min(start + chunk_tokens, assignment.unique_tokens)
            original_end = assignment.shared_tokens + end
            self.push(
                self._seal_time(task, original_end), 0, "protection_seal",
                ("unique", task.request_id, ordinal, start, end),
            )
        self._sample_physical_occupancy(at)

    def _check_refcount(self, key: tuple[str, int]) -> None:
        observed = len(self.shared_ref_members[key])
        declared = int(self.shared_refcount_value[key])
        self.refcount_max_abs_error = max(
            self.refcount_max_abs_error, abs(declared - observed)
        )
        self.refcount_peak = max(
            self.refcount_peak,
            sum(len(rows) for rows in self.shared_ref_members.values()),
        )

    def _attach_shared_reference(
        self, request_id: str, key: tuple[str, int], byte_size: int,
    ) -> None:
        members = self.shared_ref_members[key]
        if request_id in members:
            raise AssertionError("request attached the same shared chunk twice")
        if members:
            self.shared_reference_alias_hit_count += 1
        members.add(request_id)
        self.shared_refcount_value[key] += 1
        self.request_shared_keys[request_id].add(key)
        self.shared_reference_count += 1
        self._check_refcount(key)

    def handle_protection_seal(self, payload: tuple[Any, ...], at: float) -> None:
        if len(payload) != 5:
            raise AssertionError("S6 seal payload must include shared/unique kind")
        kind, request_id, ordinal, token_start, token_end = payload
        if request_id not in self.active:
            return
        assignment = self._assignment(request_id)
        if kind == "unique":
            super().handle_protection_seal(
                (request_id, int(ordinal), int(token_start), int(token_end)), at
            )
            self._sample_physical_occupancy(at)
            return
        if kind != "shared":
            raise AssertionError(f"unknown S6 seal kind {kind!r}")
        if int(token_end) > assignment.shared_tokens:
            raise AssertionError("shared chunk exceeds registered prefix")
        key = (assignment.group_id, int(ordinal))
        self._attach_shared_reference(
            request_id, key, (int(token_end) - int(token_start)) * self.kv
        )
        if key not in self.shared_materialized_keys:
            owner_id = self._ensure_shared_owner(
                self.task_by_id[request_id], assignment, at
            )
            super().handle_protection_seal(
                (owner_id, int(ordinal), int(token_start), int(token_end)), at
            )
            self.shared_materialized_keys.add(key)
        self._sample_physical_occupancy(at)

    def finish(self, task: core.Task, at: float) -> None:
        if task.request_id not in self.active:
            return
        for key in sorted(self.request_shared_keys.pop(task.request_id, set())):
            members = self.shared_ref_members[key]
            if task.request_id not in members:
                self.refcount_underflow_count += 1
                raise AssertionError("shared-prefix refcount underflow")
            members.remove(task.request_id)
            self.shared_refcount_value[key] -= 1
            if self.shared_refcount_value[key] < 0:
                self.refcount_underflow_count += 1
                raise AssertionError("shared-prefix refcount became negative")
            self._check_refcount(key)
        self.active.remove(task.request_id)
        if self.is_aegis:
            self.advance_aegis(task.source_rank, at)
            self.aegis[task.source_rank].complete_request(
                task.request_id, force_abort_pending=True
            )
            self.refresh_occupancy(task.source_rank, at)
        else:
            for runtime in self.baselines[task.source_rank]:
                if runtime.has_request(task.request_id):
                    runtime.finish_request(task.request_id, time_ms=at)
        self._sample_physical_occupancy(at)

    def _aegis_tier_parts(
        self, task: core.Task, tier: core.ProtectionTier,
    ) -> tuple[int, int]:
        assignment = self._assignment(task.request_id)
        controller = self.aegis[task.source_rank]
        shared = 0
        if assignment.shared_tokens:
            owner = self.shared_owner_by_group.get(assignment.group_id)
            if owner is not None:
                shared = min(
                    controller.committed_prefix_tokens(owner, tier),
                    assignment.shared_tokens,
                )
        unique = min(
            controller.committed_prefix_tokens(task.request_id, tier),
            assignment.unique_tokens,
        )
        if shared < assignment.shared_tokens:
            return shared, 0
        return shared, unique

    def _baseline_tier_parts(
        self, task: core.Task, runtime: Any, probe: core.FaultProbe,
    ) -> tuple[int, int]:
        assignment = self._assignment(task.request_id)
        shared = 0
        owner = self.shared_owner_by_group.get(assignment.group_id)
        if assignment.shared_tokens and owner is not None and runtime.has_request(owner):
            decision = runtime.estimate_source_failure(
                owner,
                produced_tokens=assignment.shared_tokens,
                remaining_slack_ms=0.0,
                time_ms=probe.time_ms,
            )
            shared = min(int(decision.restored_tokens), assignment.shared_tokens)
        unique = 0
        if runtime.has_request(task.request_id):
            decision = runtime.estimate_source_failure(
                task.request_id,
                produced_tokens=assignment.unique_tokens,
                remaining_slack_ms=0.0,
                time_ms=probe.time_ms,
            )
            unique = min(int(decision.restored_tokens), assignment.unique_tokens)
        if shared < assignment.shared_tokens:
            return shared, 0
        return shared, unique

    def recovery(
        self, task: core.Task, probe: core.FaultProbe,
    ) -> tuple[float, str]:
        assignment = self._assignment(task.request_id)
        placement = v313.topology_placement(self.protocol, task.source_rank)
        recompute = core.RecoveryTiming(
            4.0, float(self.model["prefill_ms_per_token"])
        )
        candidates: list[tuple[float, str, int, int, int, int]] = [(
            recompute.setup_ms + task.prompt_tokens * recompute.ms_per_token,
            "RECOMPUTE", 0, task.prompt_tokens, 0, 0,
        )]
        node_failure = probe.failure_type == "source_node"
        if self.is_aegis:
            tier_rows = []
            for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM):
                shared, unique = self._aegis_tier_parts(task, tier)
                tier_rows.append((tier.value, shared, unique))
        else:
            tier_rows = []
            for runtime in self.baselines[task.source_rank]:
                shared, unique = self._baseline_tier_parts(task, runtime, probe)
                tier_rows.append((runtime.config.tier.value, shared, unique))

        hbm_bw, dram_bw = core.load_bandwidth(self.protocol, task.source_rank)
        for tier_name, shared, unique in tier_rows:
            restored = int(shared) + int(unique)
            if restored <= 0:
                continue
            if tier_name == "HBM":
                valid = not (
                    node_failure
                    and placement["peer_hbm_target_node"] == placement["source_node"]
                )
                mode = f"peer_hbm_{placement['peer_hbm_mode']}"
                setup, bandwidth = 0.08, hbm_bw
            else:
                valid = not (
                    node_failure
                    and placement["dram_target_node"] == placement["source_node"]
                )
                mode = f"dram_{placement['dram_mode']}"
                setup, bandwidth = 0.15, dram_bw
            if not valid:
                continue
            suffix = max(task.prompt_tokens - restored, 0)
            value = setup + restored * self.kv / (
                bandwidth * core.GB_PER_S_TO_BYTES_PER_MS
            )
            if suffix:
                value += recompute.setup_ms + suffix * recompute.ms_per_token
            action = mode + ("+RECOMPUTE" if suffix else "")
            candidates.append((value, action, restored, suffix, shared, unique))

        value, action, restored, recomputed, shared_restored, unique_restored = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        surviving = bool(restored)
        failover_delay, failover_mode = self._modeled_failover(
            probe, surviving_committed=surviving
        )
        if probe.failure_type == "source_node" and surviving:
            self.surviving_cross_domain_committed_count += 1
        self.failover_mode_counts[f"{probe.failure_type}:{failover_mode}"] += 1
        self.failover_delay_values_ms.append(failover_delay)
        self.recovery_service_values_ms.append(value)
        self.recovery_restored_tokens += int(restored)
        self.recovery_recomputed_tokens += int(recomputed)
        self.recovery_counts[action] += 1
        self.recovery_type_path_counts[f"{probe.failure_type}:{action}"] += 1
        self.recovery_call_count += 1
        self.recovery_request_hit_count += int(restored > 0)
        self.shared_recovery_restored_tokens += int(shared_restored)
        self.unique_recovery_restored_tokens += int(unique_restored)
        return failover_delay + value, action

    def _shared_scheduled_copy_bytes(self) -> dict[str, int]:
        values = {"HBM": 0, "DRAM": 0}
        owners = set(self.shared_owner_by_group.values())
        for request_id, tiers in self.copy_accounting_by_request.items():
            if request_id not in owners:
                continue
            for tier in values:
                values[tier] += int(tiers[tier]["bytes"])
        return values

    def _baseline_evicted_bytes(self) -> dict[str, int]:
        values = {"HBM": 0, "DRAM": 0}
        for runtimes in self.baselines:
            for runtime in runtimes:
                snapshot = runtime.snapshot()
                values[runtime.config.tier.value] += int(
                    snapshot["metrics"].get("evicted_bytes", 0)
                )
        return values

    def summarize_mechanism(self, control_horizon_ms: float) -> dict[str, Any]:
        self._sample_physical_occupancy(control_horizon_ms)
        mechanism = super().summarize_mechanism(control_horizon_ms)
        accounting = mechanism["active_hbm_v313"]
        scheduled = {
            tier: int(accounting["scheduled_copy_bytes"][tier])
            for tier in ("HBM", "DRAM")
        }
        shared_scheduled = self._shared_scheduled_copy_bytes()
        unique_scheduled = {
            tier: scheduled[tier] - shared_scheduled[tier]
            for tier in scheduled
        }
        groups = {
            row.group_id: row for row in self.assignments.values()
            if row.shared_tokens > 0
        }
        theoretical_shared = sum(
            int(row.shared_tokens) * self.kv for row in groups.values()
        )
        theoretical_unique = sum(
            int(row.unique_tokens) * self.kv for row in self.assignments.values()
        )
        theoretical_per_tier = theoretical_shared + theoretical_unique
        logical = int(self.logical_useful_bytes)
        if logical != (
            self.logical_shared_reference_bytes + self.logical_unique_reference_bytes
        ):
            raise AssertionError("logical shared/unique byte conservation failed")
        if any(value < 0 for value in unique_scheduled.values()):
            raise AssertionError("shared scheduled bytes exceed total scheduled bytes")
        final_refcount = sum(self.shared_refcount_value.values())
        horizon = float(control_horizon_ms)
        if self.is_aegis:
            evicted = {
                tier: int(mechanism["tier_churn"][tier]["evicted_bytes"])
                for tier in ("HBM", "DRAM")
            }
        else:
            evicted = self._baseline_evicted_bytes()
        restored_total = (
            int(mechanism["recovery_restored_tokens"])
            + int(mechanism["recovery_recomputed_tokens"])
        )
        mechanism["active_hbm_v313_s6"] = {
            "identity_semantics": (
                "deterministic same-source content ID; first causal producer copies; "
                "later aliases add logical references; refcount does not pin backup"
            ),
            "reuse_fraction": self.reuse_fraction,
            "group_size": self.group_size,
            "assignment_sha256": assignment_hash(self.assignments),
            "request_count": len(self.assignments),
            "shared_prefix_group_count": len(groups),
            "shared_physical_chunk_identity_count": len(self.shared_materialized_keys),
            "shared_reference_count": self.shared_reference_count,
            "shared_reference_alias_hit_count": self.shared_reference_alias_hit_count,
            "shared_reference_alias_hit_fraction": (
                self.shared_reference_alias_hit_count / self.shared_reference_count
                if self.shared_reference_count else 0.0
            ),
            "refcount_peak_total": self.refcount_peak,
            "refcount_final_total": final_refcount,
            "refcount_max_abs_error": self.refcount_max_abs_error,
            "refcount_underflow_count": self.refcount_underflow_count,
            "logical_useful_bytes": logical,
            "logical_shared_reference_bytes": int(self.logical_shared_reference_bytes),
            "logical_unique_reference_bytes": int(self.logical_unique_reference_bytes),
            "theoretical_deduplicated_physical_bytes_per_tier": theoretical_per_tier,
            "theoretical_shared_physical_bytes_per_tier": theoretical_shared,
            "theoretical_unique_physical_bytes_per_tier": theoretical_unique,
            "theoretical_copy_ratio_per_tier": (
                theoretical_per_tier / logical if logical else 0.0
            ),
            "scheduled_physical_copy_bytes": scheduled,
            "scheduled_shared_physical_copy_bytes": shared_scheduled,
            "scheduled_unique_physical_copy_bytes": unique_scheduled,
            "copy_amplification_physical_over_logical": (
                sum(scheduled.values()) / logical if logical else 0.0
            ),
            "copy_ratio_by_tier": {
                tier: scheduled[tier] / logical if logical else 0.0
                for tier in scheduled
            },
            "recovery_call_count": self.recovery_call_count,
            "recovery_request_hit_count": self.recovery_request_hit_count,
            "recovery_request_hit_fraction": (
                self.recovery_request_hit_count / self.recovery_call_count
                if self.recovery_call_count else 0.0
            ),
            "shared_recovery_restored_tokens": self.shared_recovery_restored_tokens,
            "unique_recovery_restored_tokens": self.unique_recovery_restored_tokens,
            "recovery_prefix_token_hit_fraction": (
                int(mechanism["recovery_restored_tokens"]) / restored_total
                if restored_total else 0.0
            ),
            "recoverable_byte_fraction": (
                int(mechanism["recovery_restored_tokens"]) / restored_total
                if restored_total else 0.0
            ),
            "physical_evicted_bytes": evicted,
            "physical_occupancy_observation_horizon_ms": horizon,
            "physical_capacity_integral_byte_ms": dict(
                self.physical_occupancy_area_byte_ms
            ),
            "physical_occupancy_time_mean_gib_per_gpu": {
                tier: self.physical_occupancy_area_byte_ms[tier]
                / max(horizon, 1e-12) / core.GIB / self.dp
                for tier in ("HBM", "DRAM")
            },
            "physical_occupancy_peak_gib_per_gpu": {
                tier: self.physical_occupancy_peak_bytes[tier] / core.GIB / self.dp
                for tier in ("HBM", "DRAM")
            },
            "physical_occupancy_trace_change_points": len(
                self.physical_occupancy_trace
            ),
            "future_member_count_used_for_admission": 0,
        }
        return mechanism
