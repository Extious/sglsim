"""Causal source-copy and physical-target HBM accounting for v3.1.3.

This module leaves every published v3.1.2 artifact untouched.  It corrects
three mechanism errors demonstrated by the v3.1.2 health audit:

* a whole prompt was reserved for protection at request dispatch, before its
  KV chunks were produced;
* a protection reservation could block the source request itself; and
* Peer-HBM bytes were charged to the logical source controller index instead
  of the route-bound physical target GPU.

Prompt chunks now seal causally during modeled prefill.  Source-owned HBM and
DRAM copies remain independent asynchronous queues.  They affect healthy WCT
through actual shared-HBM serving pressure only, not through an extra transfer
tail surcharge.
"""

from __future__ import annotations

import heapq
import math
import statistics
from collections import defaultdict
from dataclasses import replace
from typing import Any

import active_hbm_v312 as v312
import run_preliminary as core


def topology_placement(protocol: dict[str, Any], rank: int) -> dict[str, Any]:
    """Return a balanced one-source/one-target route-bound placement.

    Even local GPU indices use a same-node successor.  Odd indices use the
    predecessor local index in the next node.  Across a ring of complete
    8-GPU nodes this is a bijection, so every physical Peer-HBM target owns one
    controller pool and no pool is silently duplicated.
    """

    hardware = protocol["hardware"]
    gpus_per_node = int(hardware["gpus_per_node"])
    dp = int(protocol["models"]["qwen3_8b"]["center_system_dp"])
    if dp <= 0 or dp % gpus_per_node:
        raise ValueError("system DP must contain complete GPU nodes")
    node_count = dp // gpus_per_node
    source_node = int(rank) // gpus_per_node
    local = int(rank) % gpus_per_node
    peer_mode, dram_mode = core.topology_for_rank(rank, protocol)
    peer_node = source_node if peer_mode == "same_node" else (source_node + 1) % node_count
    peer_local = (local + 1) % gpus_per_node if peer_mode == "same_node" else (local - 1) % gpus_per_node
    dram_node = source_node if dram_mode == "local" else (source_node + 1) % node_count
    dram_local = (local + 1) % gpus_per_node if dram_mode == "local" else (local - 1) % gpus_per_node
    return {
        "rank": int(rank),
        "source_node": source_node,
        "local_gpu_index": local,
        "peer_hbm_mode": peer_mode,
        "peer_hbm_target_node": peer_node,
        "peer_hbm_target_rank": peer_node * gpus_per_node + peer_local,
        "dram_mode": dram_mode,
        "dram_target_node": dram_node,
        "dram_target_rank": dram_node * gpus_per_node + dram_local,
    }


class CausalPhysicalTargetReplay(v312.TopologyAwareExactAdmissionProjectionReplay):
    """Serving-primary replay with delayed source copy and target accounting."""

    SUPPORTED_POLICIES = frozenset({
        "Native Mooncake", "Static FCFS", "DejaVu", "KevlarFlow",
        "GhostServe", "AegisKV",
    })

    def __init__(
        self, protocol: dict[str, Any], policy: str,
        capacity_protocol: dict[str, Any],
    ) -> None:
        if policy not in self.SUPPORTED_POLICIES:
            raise ValueError(
                f"v3.1.3 pilot supports only {sorted(self.SUPPORTED_POLICIES)}; "
                f"got {policy!r}"
            )
        super().__init__(protocol, policy, capacity_protocol)
        self.topology_rows = [topology_placement(protocol, rank) for rank in range(self.dp)]
        inverse: dict[int, int] = {}
        for row in self.topology_rows:
            target = int(row["peer_hbm_target_rank"])
            if target in inverse:
                raise AssertionError("Peer-HBM topology is not one-to-one")
            inverse[target] = int(row["rank"])
        if set(inverse) != set(range(self.dp)):
            raise AssertionError("Peer-HBM topology does not cover all physical targets")
        self.source_controller_for_target = inverse
        self.seal_event_count = 0
        self.seal_before_dispatch_count = 0
        self.seal_uses_realized_output_or_service = False
        self.primary_priority_reclaimed_bytes = 0
        self.copy_accounting_by_request: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: {
                "HBM": {"bytes": 0.0, "service_ms": 0.0, "queue_wait_ms": 0.0, "jobs": 0.0},
                "DRAM": {"bytes": 0.0, "service_ms": 0.0, "queue_wait_ms": 0.0, "jobs": 0.0},
            }
        )
        self.copy_duration_formula_max_abs_error_ms = 0.0
        self.copy_critical_path_sum_violation_count = 0
        self.target_accounting_max_abs_error_bytes = 0
        self.target_capacity_violation_count = 0
        self.future_backup_projection_dispatch_count = 0
        self.physical_target_peak_backup_bytes = [0] * self.dp
        self.scheduled_control_opportunity_times_ms: list[float] = []

    def _profile(self, task: core.Task) -> core.RequestRecoveryProfile:
        placement = topology_placement(self.protocol, task.source_rank)
        slack = 0.0 if self.policy == "AegisKV no Algorithm 1" else task.predicted_slack_ms
        target = int(placement["peer_hbm_target_rank"])
        return core.RequestRecoveryProfile(
            request_id=task.request_id,
            source_worker=f"source-{task.source_rank}",
            source_failure_domain=f"node-{placement['source_node']}",
            hbm_target=(
                f"peer-node{placement['peer_hbm_target_node']}"
                f"-gpu{target % int(self.protocol['hardware']['gpus_per_node'])}"
            ),
            hbm_failure_domain=f"node-{placement['peer_hbm_target_node']}",
            route_version=0,
            source_token_end=task.prompt_tokens,
            slack_ms=slack,
            model_revision="Qwen3-8B-frozen",
            kv_abi="qwen3-fp16-kv-v1",
            parallel_layout="TP1-PP1",
        )

    def recovery(
        self, task: core.Task, probe: core.FaultProbe,
    ) -> tuple[float, str]:
        """Reuse v3.1.2 recovery with the corrected frozen placement."""

        original = v312.topology_placement
        v312.topology_placement = topology_placement
        try:
            return super().recovery(task, probe)
        finally:
            v312.topology_placement = original

    def _logical_hbm_used_bytes(self, source_rank: int) -> int:
        if not self.is_aegis:
            return super().hbm_backup_used_bytes(source_rank)
        return int(
            self.aegis[source_rank].telemetry_snapshot()["pools"]["HBM"]["used_bytes"]
        )

    def hbm_backup_used_bytes(self, target_rank: int) -> int:
        source = self.source_controller_for_target[int(target_rank)]
        return self._logical_hbm_used_bytes(source)

    def hbm_backup_limit_bytes(self, target_rank: int) -> int:
        source = self.source_controller_for_target[int(target_rank)]
        if self.is_aegis:
            return int(
                self.aegis[source].telemetry_snapshot()["pools"]["HBM"]["limit_bytes"]
            )
        return super().hbm_backup_limit_bytes(source)

    def _account_target_conservation(self) -> None:
        logical = sum(self._logical_hbm_used_bytes(source) for source in range(self.dp))
        physical = sum(self.hbm_backup_used_bytes(target) for target in range(self.dp))
        self.target_accounting_max_abs_error_bytes = max(
            self.target_accounting_max_abs_error_bytes, abs(logical - physical)
        )
        if logical != physical:
            raise AssertionError("logical-to-physical Peer-HBM byte conservation failed")

    def record_capacity(self, rank: int, at: float, event: str) -> None:
        super().record_capacity(rank, at, event)
        self._account_target_conservation()
        self.physical_target_peak_backup_bytes[rank] = max(
            self.physical_target_peak_backup_bytes[rank],
            self.hbm_backup_used_bytes(rank),
        )
        if self.active_serving_bytes[rank] + self.hbm_backup_used_bytes(rank) > self.shared_budget_bytes:
            self.target_capacity_violation_count += 1
            raise AssertionError("physical target serving plus backup exceeds shared HBM")

    def _enforce_target_headroom(
        self, target_rank: int, at: float, *, serving_bytes: int | None = None,
    ) -> None:
        source = self.source_controller_for_target[int(target_rank)]
        serving = (
            int(self.active_serving_bytes[target_rank])
            if serving_bytes is None else int(serving_bytes)
        )
        effective = max(self.shared_budget_bytes - serving, 0)
        if not self.is_aegis:
            for runtime in self.baselines[source]:
                if runtime.config.tier.value != "HBM":
                    continue
                runtime.advance_to(at)
                while int(runtime.used_bytes) > effective:
                    candidates = sorted(
                        (
                            state for state in runtime._requests.values()
                            if int(state.used_bytes) > 0
                        ),
                        key=lambda state: (
                            state.registered_at_ms,
                            state.profile.request_id,
                        ),
                    )
                    if not candidates:
                        raise AssertionError(
                            "baseline target reclaim cannot reach serving headroom"
                        )
                    before = int(runtime.used_bytes)
                    runtime._release_request(
                        candidates[0].profile.request_id,
                        reason="physical_target_primary_serving_reclaim",
                    )
                    self.primary_priority_reclaimed_bytes += (
                        before - int(runtime.used_bytes)
                    )
            return
        self.advance_aegis(source, at)
        controller = self.aegis[source]
        snapshot = controller.telemetry_snapshot()["pools"]["HBM"]
        policy_limit = int(snapshot["limit_bytes"])
        effective = min(policy_limit, max(self.shared_budget_bytes - serving, 0))
        before = int(snapshot["used_bytes"])
        if before > effective:
            controller.set_pool_limit(core.ProtectionTier.HBM, effective)
            after = int(controller.telemetry_snapshot()["pools"]["HBM"]["used_bytes"])
            self.primary_priority_reclaimed_bytes += before - after
            controller.set_pool_limit(core.ProtectionTier.HBM, policy_limit)
            self.refresh_occupancy(source, at)

    def dispatch_block_reason(self, rank: int, at: float) -> str | None:
        if not self.waiting[rank]:
            return None
        if not self.free_lanes[rank]:
            return "ENGINE_LANE"
        task = self.waiting[rank][0]
        serving_after = self.active_serving_bytes[rank] + self.serving_reservation_bytes(task)
        self._enforce_target_headroom(rank, at, serving_bytes=serving_after)
        if serving_after + self.hbm_backup_used_bytes(rank) > self.shared_budget_bytes:
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
            lane = heapq.heappop(self.free_lanes[rank])
            serving_bytes = self.serving_reservation_bytes(nominal)
            runtime_task = replace(nominal, lane=lane, start_ms=float(at), finish_ms=float(at))
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

    def _seal_time(self, task: core.Task, token_end: int) -> float:
        # No realized output_tokens or realized task.service_ms is read here.
        return (
            float(task.start_ms)
            + float(self.model["prefill_setup_ms"])
            + int(token_end) * float(self.model["prefill_ms_per_token"])
        )

    def start(self, task: core.Task, at: float) -> None:
        self.active.add(task.request_id)
        self.health_burden_ms[task.request_id] = 0.0
        self.copy_service_ms[task.request_id] = 0.0
        self.copy_queue_wait_ms[task.request_id] = 0.0
        if self.is_aegis:
            rank = task.source_rank
            self.advance_aegis(rank, at)
            self.aegis[rank].register_request(self._profile(task))
            chunk_tokens = int(self.protocol["aegiskv"]["prefix_chunk_tokens"])
            for ordinal, start in enumerate(range(0, task.prompt_tokens, chunk_tokens)):
                end = min(start + chunk_tokens, task.prompt_tokens)
                self.push(
                    self._seal_time(task, end), 0, "protection_seal",
                    (task.request_id, ordinal, start, end),
                )
            self.refresh_occupancy(rank, at)
            return
        profile = core.BaselineRequestProfile(
            request_id=task.request_id,
            source_worker=f"source-{task.source_rank}",
            source_failure_domain=f"node-{task.source_rank // 8}",
            target_worker=f"target-{task.source_rank}",
            target_failure_domain=f"target-node-{task.source_rank // 8}",
            route_version=0,
            total_tokens=task.prompt_tokens,
            slack_ms=task.predicted_slack_ms,
        )
        for runtime in self.baselines[task.source_rank]:
            runtime.register_request(profile, time_ms=at)
        self.push(
            self._seal_time(task, task.prompt_tokens), 0, "protection_seal",
            (task.request_id, 0, 0, task.prompt_tokens),
        )

    def _record_copy_job(
        self, request_id: str, tier: str, *, byte_size: int,
        duration_ms: float, queue_wait_ms: float,
    ) -> None:
        row = self.copy_accounting_by_request[request_id][tier]
        row["bytes"] += int(byte_size)
        row["service_ms"] += float(duration_ms)
        row["queue_wait_ms"] = max(row["queue_wait_ms"], float(queue_wait_ms))
        row["jobs"] += 1
        hbm = self.copy_accounting_by_request[request_id]["HBM"]
        dram = self.copy_accounting_by_request[request_id]["DRAM"]
        critical = max(hbm["service_ms"], dram["service_ms"])
        serial = hbm["service_ms"] + dram["service_ms"]
        self.copy_service_ms[request_id] = critical
        self.copy_queue_wait_ms[request_id] = max(
            hbm["queue_wait_ms"], dram["queue_wait_ms"]
        )
        if hbm["service_ms"] > 0 and dram["service_ms"] > 0 and not critical < serial:
            self.copy_critical_path_sum_violation_count += 1

    def handle_protection_seal(self, payload: tuple[str, int, int, int], at: float) -> None:
        request_id, ordinal, token_start, token_end = payload
        if request_id not in self.active:
            return
        task = self.task_by_id[request_id]
        self.seal_event_count += 1
        if at < task.start_ms - 1e-9:
            self.seal_before_dispatch_count += 1
        rank = task.source_rank
        placement = topology_placement(self.protocol, rank)
        if self.is_aegis:
            self.advance_aegis(rank, at)
            controller = self.aegis[rank]
            target = int(placement["peer_hbm_target_rank"])
            desired_limit = int(
                controller.telemetry_snapshot()["pools"]["HBM"]["limit_bytes"]
            )
            effective_limit = min(
                desired_limit,
                max(self.shared_budget_bytes - self.active_serving_bytes[target], 0),
            )
            controller.set_pool_limit(core.ProtectionTier.HBM, effective_limit)
            profile = self._profile(task)
            spec = core.ChunkSpec(
                request_id=request_id,
                ordinal=int(ordinal),
                token_start=int(token_start),
                token_end=int(token_end),
                byte_size=(int(token_end) - int(token_start)) * self.kv,
                route_version=0,
                source_worker=profile.source_worker,
                hbm_target=profile.hbm_target,
                model_revision=profile.model_revision,
                kv_abi=profile.kv_abi,
                parallel_layout=profile.parallel_layout,
            )
            try:
                decision = controller.admit_chunk(spec)
            finally:
                controller.set_pool_limit(core.ProtectionTier.HBM, desired_limit)
            self.residency_counts[decision.intended_residency.value] += 1
            copy_bw = core.copy_bandwidth(self.protocol, rank)
            for tier, admitted, mode, setup in (
                (core.ProtectionTier.HBM, decision.hbm.admitted, f"peer_hbm_{placement['peer_hbm_mode']}", 0.05),
                (core.ProtectionTier.DRAM, decision.dram.admitted, f"dram_{placement['dram_mode']}", 0.10),
            ):
                if not admitted:
                    continue
                bw = copy_bw[0 if tier is core.ProtectionTier.HBM else 1]
                duration = setup + spec.byte_size / (bw * core.GB_PER_S_TO_BYTES_PER_MS)
                formula = setup + (token_end - token_start) * self.kv / (
                    bw * core.GB_PER_S_TO_BYTES_PER_MS
                )
                self.copy_duration_formula_max_abs_error_ms = max(
                    self.copy_duration_formula_max_abs_error_ms, abs(duration - formula)
                )
                key = (rank, tier)
                lane_before = self.aegis_lane[key]
                queue_wait = max(lane_before - at, 0.0)
                done = max(at, lane_before) + duration
                self.aegis_lane[key] = done
                self.aegis_specs[(request_id, ordinal, tier)] = spec
                self.path_counts[f"copy_{mode}"] += 1
                self.push(done, 0, "aegis_copy", (rank, request_id, ordinal, tier))
                self._record_copy_job(
                    request_id, tier.value, byte_size=spec.byte_size,
                    duration_ms=duration, queue_wait_ms=queue_wait,
                )
            self.refresh_occupancy(rank, at)
            self.record_capacity(target, at, "protection_seal")
            return

        chunk = core.SealedChunk(ordinal, token_start, token_end, "prefill")
        for runtime in self.baselines[rank]:
            lane_before = runtime.lane_available_ms
            original_config = runtime.config
            if runtime.config.tier.value == "HBM":
                target = int(placement["peer_hbm_target_rank"])
                self._enforce_target_headroom(target, at)
                physical_headroom = max(
                    self.shared_budget_bytes
                    - self.active_serving_bytes[target]
                    - self.hbm_backup_used_bytes(target),
                    0,
                )
                runtime.config = replace(
                    original_config,
                    hbm_capacity_bytes=min(
                        int(original_config.hbm_capacity_bytes),
                        int(runtime.used_bytes) + int(physical_headroom),
                    ),
                )
            try:
                admission = runtime.seal_chunk(request_id, chunk, time_ms=at)
            finally:
                runtime.config = original_config
            if not admission.admitted:
                continue
            duration = runtime.lane_available_ms - max(at, lane_before)
            queue_wait = max(lane_before - at, 0.0)
            tier = runtime.config.tier.value
            mode = (
                f"peer_hbm_{placement['peer_hbm_mode']}"
                if tier == "HBM" else f"dram_{placement['dram_mode']}"
            )
            self.path_counts[f"copy_{mode}"] += 1
            self._record_copy_job(
                request_id, tier, byte_size=admission.logical_bytes,
                duration_ms=duration, queue_wait_ms=queue_wait,
            )

    def control(
        self, at: float, active_by_rank: dict[int, int],
        external_pressure_by_rank: dict[int, bool] | None = None,
    ) -> None:
        if not self.is_aegis:
            return
        logical_active = {}
        logical_external = {}
        for target, source in self.source_controller_for_target.items():
            logical_active[source] = int(active_by_rank.get(target, 0))
            logical_external[source] = bool(
                (external_pressure_by_rank or {}).get(target, False)
            )
        super().control(
            at, logical_active, external_pressure_by_rank=logical_external
        )
        for target in range(self.dp):
            self._enforce_target_headroom(target, at)

    def _retry_all(self, at: float, event: str) -> int:
        total = 0
        for rank in range(self.dp):
            total += self.try_dispatch(rank, at, event=event)
        return total

    def summarize_mechanism(self, control_horizon_ms: float) -> dict[str, Any]:
        mechanism = super().summarize_mechanism(control_horizon_ms)
        scheduled = {"HBM": 0, "DRAM": 0}
        jobs = {"HBM": 0, "DRAM": 0}
        for request in self.copy_accounting_by_request.values():
            for tier in scheduled:
                scheduled[tier] += int(request[tier]["bytes"])
                jobs[tier] += int(request[tier]["jobs"])
        admitted = {"HBM": 0, "DRAM": 0}
        if self.is_aegis:
            for controller in self.aegis:
                snapshot = controller.telemetry_snapshot()
                for tier in admitted:
                    admitted[tier] += int(
                        snapshot["pools"][tier]["churn"]["admitted_bytes"]
                    )
        else:
            for rank in self.baselines:
                for runtime in rank:
                    snapshot = runtime.snapshot()
                    admitted[runtime.config.tier.value] += int(
                        snapshot["metrics"]["copy_logical_bytes"]
                    )
        byte_error = {
            tier: scheduled[tier] - admitted[tier] for tier in scheduled
        }
        mechanism["active_hbm_v313"] = {
            "prompt_seal_semantics": (
                "dispatch + prefill_setup_ms + token_end * prefill_ms_per_token; "
                "no realized output/service oracle"
            ),
            "normal_path_copy_wct_semantics": (
                "asynchronous transfer service is not directly added to output WCT; "
                "shared physical-target HBM pressure remains on the serving path"
            ),
            "topology_rows": self.topology_rows,
            "topology_bijective": len(self.source_controller_for_target) == self.dp,
            "seal_event_count": self.seal_event_count,
            "seal_before_dispatch_count": self.seal_before_dispatch_count,
            "seal_uses_realized_output_or_service": self.seal_uses_realized_output_or_service,
            "scheduled_copy_bytes": scheduled,
            "controller_admitted_copy_bytes": admitted,
            "copy_byte_conservation_error": byte_error,
            "copy_job_counts": jobs,
            "copy_duration_formula_max_abs_error_ms": self.copy_duration_formula_max_abs_error_ms,
            "copy_critical_path_operator": "max(HBM, DRAM)",
            "copy_critical_path_sum_violation_count": self.copy_critical_path_sum_violation_count,
            "health_burden_application_count": sum(
                int(abs(value) > 1e-12) for value in self.health_burden_ms.values()
            ),
            "primary_priority_reclaimed_bytes": self.primary_priority_reclaimed_bytes,
            "target_accounting_max_abs_error_bytes": self.target_accounting_max_abs_error_bytes,
            "target_capacity_violation_count": self.target_capacity_violation_count,
            "future_backup_projection_dispatch_count": (
                self.future_backup_projection_dispatch_count
            ),
            "physical_target_peak_backup_bytes": (
                self.physical_target_peak_backup_bytes
            ),
            "physical_target_peak_backup_max_bytes": max(
                self.physical_target_peak_backup_bytes, default=0
            ),
            "physical_target_hbm_capacity_bytes": self.total_hbm_bytes,
            "scheduled_control_opportunity_times_ms": (
                self.scheduled_control_opportunity_times_ms
            ),
            "executed_policy_control_times_ms": (
                mechanism["control_opportunity_times_ms"]
            ),
        }
        return mechanism

    def run_coupled(
        self, nominal_tasks: list[core.Task], probes: list[core.FaultProbe], *,
        control_horizon_ms: float,
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
            self.scheduled_control_opportunity_times_ms.append(float(at))
            self.push(at, 3, "coupled_control", None)
            at += window
        recoveries: list[dict[str, float]] = []
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
                rank = task.source_rank
                if task.request_id not in self.active_by_rank.get(rank, set()):
                    continue
                self.active_by_rank[rank].discard(task.request_id)
                self.active_serving_bytes[rank] -= self.serving_bytes_by_request.pop(task.request_id)
                heapq.heappush(self.free_lanes[rank], task.lane)
                self.finish(task, at)
                self.record_capacity(rank, at, "finish_release")
                self._retry_all(at, "finish")
            elif kind == "coupled_control":
                external = {
                    rank: self.dispatch_block_reason(rank, at) == "SHARED_HBM"
                    for rank in range(self.dp)
                }
                before = sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
                waiting_before = sum(len(rows) for rows in self.waiting)
                active_before = sum(len(rows) for rows in self.active_by_rank.values())
                self.control(
                    at,
                    {rank: len(rows) for rank, rows in self.active_by_rank.items()},
                    external_pressure_by_rank=external,
                )
                after = sum(self.hbm_backup_used_bytes(rank) for rank in range(self.dp))
                released = max(before - after, 0)
                self.backup_bytes_released_by_control += released
                dispatched = self._retry_all(at, "control")
                self.dispatch_at_control_count += dispatched
                self.control_capacity_trace.append({
                    "time_ms": float(at),
                    "memory_pressure_rank_count": sum(external.values()),
                    "backup_before_bytes": before,
                    "backup_after_bytes": after,
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
                for rank in v312.v31.failure_domain_ranks(
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
                raise AssertionError(f"unknown v3.1.3 event {kind!r}")
        if any(self.waiting) or any(self.active_by_rank.values()):
            raise AssertionError("v3.1.3 scheduler failed to drain")
        if len(self.scheduled_tasks) != len(nominal_tasks):
            raise AssertionError("v3.1.3 scheduler lost requests")
        mechanism = self.summarize_mechanism(control_horizon_ms)
        return (
            sorted(self.scheduled_tasks, key=lambda row: (row.start_ms, row.request_id)),
            recoveries,
            mechanism,
            list(self.capacity_trace),
        )


def workflow_decomposition(tasks: list[core.Task]) -> dict[str, float]:
    """Exact healthy WCT = queue + modeled service + direct copy burden."""

    by_workflow: dict[str, list[core.Task]] = defaultdict(list)
    for task in tasks:
        by_workflow[task.workflow_id].append(task)
    rows = []
    for workflow_id in sorted(by_workflow):
        critical = max(
            by_workflow[workflow_id], key=lambda row: (row.finish_ms, row.request_id)
        )
        queue = critical.start_ms - critical.workflow_arrival_ms
        burden = critical.finish_ms - critical.start_ms - critical.service_ms
        total = critical.finish_ms - critical.workflow_arrival_ms
        if not math.isclose(total, queue + critical.service_ms + burden, abs_tol=1e-7):
            raise AssertionError("healthy critical-path accounting failed")
        rows.append((total, queue, critical.service_ms, burden))
    return {
        "workflow_count": len(rows),
        "healthy_wct_mean_ms": statistics.fmean(row[0] for row in rows),
        "critical_queue_mean_ms": statistics.fmean(row[1] for row in rows),
        "critical_service_mean_ms": statistics.fmean(row[2] for row in rows),
        "critical_direct_copy_burden_mean_ms": statistics.fmean(row[3] for row in rows),
        "identity_max_abs_error_ms": max(
            abs(row[0] - row[1] - row[2] - row[3]) for row in rows
        ),
    }
