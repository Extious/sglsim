"""S6 v2 shared-prefix residency lifetime and rematerialization semantics.

This module leaves the published S6 v1 implementation unchanged.  A logical
shared chunk keeps a stable content identity ``(group_id, ordinal)`` while each
physical tier owns an independent materialization epoch.  Within one tier:

* PENDING or COMMITTED means a later source attaches without duplicate copy;
* a COMMITTED object remains a resident hit even when active refcount is zero;
* after eviction, a later causal source may create a new epoch and readmit that
  tier; and
* HBM and DRAM are retried independently, so one surviving tier never masks a
  missing tier.

Recovery consumes only a continuous chain of currently COMMITTED logical
chunks on the selected tier.  An old epoch can never remain recoverable after
the controller/runtime has evicted its physical bytes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

import active_hbm_v313 as v313
import active_hbm_v313_s6 as s6v1
import run_preliminary as core


PrefixAssignment = s6v1.PrefixAssignment
build_prefix_assignments = s6v1.build_prefix_assignments
assignment_hash = s6v1.assignment_hash
digest_json = s6v1.digest_json


class SharedPrefixReplay(s6v1.SharedPrefixReplay):
    """Per-tier physical epochs for stable shared content identities."""

    RESIDENT_STATES = frozenset({"PENDING", "COMMITTED"})

    def __init__(
        self, protocol: dict[str, Any], policy: str,
        capacity_protocol: dict[str, Any], *,
        assignments: dict[str, PrefixAssignment], reuse_fraction: float,
        group_size: int,
    ) -> None:
        super().__init__(
            protocol, policy, capacity_protocol,
            assignments=assignments, reuse_fraction=reuse_fraction,
            group_size=group_size,
        )
        self.shared_current_owner_by_key_tier: dict[
            tuple[tuple[str, int], str], str
        ] = {}
        self.shared_owner_key_tier: dict[
            str, tuple[tuple[str, int], str]
        ] = {}
        self.shared_owner_ids: set[str] = set()
        self.shared_group_source_rank = {
            row.group_id: int(row.source_rank) for row in self.assignments.values()
        }
        self.shared_key_token_bounds: dict[
            tuple[str, int], tuple[int, int]
        ] = {}
        self.shared_epoch_counter: dict[
            tuple[tuple[str, int], str], int
        ] = defaultdict(int)
        self.shared_ever_admitted_key_tier: set[
            tuple[tuple[str, int], str]
        ] = set()
        self.shared_tier_materialization_attempt_count = {
            "HBM": 0, "DRAM": 0,
        }
        self.shared_tier_admitted_epoch_count = {"HBM": 0, "DRAM": 0}
        self.shared_tier_rejected_epoch_count = {"HBM": 0, "DRAM": 0}
        self.shared_tier_rematerialization_attempt_count = {
            "HBM": 0, "DRAM": 0,
        }
        self.shared_tier_rematerialization_admitted_count = {
            "HBM": 0, "DRAM": 0,
        }
        self.shared_tier_pending_no_duplicate_count = {"HBM": 0, "DRAM": 0}
        self.shared_tier_committed_no_duplicate_count = {"HBM": 0, "DRAM": 0}
        self.shared_active_overlap_hit_count = 0
        self.shared_resident_hit_count = 0
        self.shared_refcount_zero_resident_hit_count = 0
        self.shared_pending_only_hit_count = 0
        self.shared_epoch_audit: dict[str, dict[str, Any]] = {}
        self.shared_owner_priority_slack_ms: dict[str, float] = {}
        self.shared_priority_metadata_update_count = 0
        self.shared_priority_zero_refcount_demotion_count = 0
        self.shared_priority_active_reactivation_count = 0
        self.shared_priority_copy_byte_delta_max = 0
        self.shared_priority_used_byte_delta_max = 0
        self.shared_priority_eviction_count_delta_max = 0

    @staticmethod
    def _tier_name(tier: core.ProtectionTier | str) -> str:
        return core.ProtectionTier(tier).value

    def _baseline_runtime(self, rank: int, tier: core.ProtectionTier) -> Any:
        wanted = self._tier_name(tier)
        matches = [
            runtime for runtime in self.baselines[int(rank)]
            if runtime.config.tier.value == wanted
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one baseline runtime for {wanted}")
        return matches[0]

    def _tier_is_supported(self, rank: int, tier: core.ProtectionTier) -> bool:
        if self.is_aegis:
            return True
        wanted = self._tier_name(tier)
        return any(
            runtime.config.tier.value == wanted
            for runtime in self.baselines[int(rank)]
        )

    def _tier_state_for_owner(
        self, owner_id: str, rank: int, tier: core.ProtectionTier, *,
        at: float | None = None,
    ) -> str:
        tier = core.ProtectionTier(tier)
        if self.is_aegis:
            if at is not None:
                self.advance_aegis(int(rank), float(at))
            state = self.aegis[int(rank)].copy_state(owner_id, 0, tier)
            return "ABSENT" if state is None else str(state.value)
        runtime = self._baseline_runtime(rank, tier)
        if at is not None:
            runtime.advance_to(float(at))
        return str(runtime.request_replica_state(owner_id)["presence"])

    def shared_tier_state(
        self, key: tuple[str, int], tier: core.ProtectionTier | str, *,
        at: float | None = None,
    ) -> str:
        """Return UNMATERIALIZED/PENDING/COMMITTED/EVICTED for one tier."""

        tier_value = self._tier_name(tier)
        source_rank = self.shared_group_source_rank[key[0]]
        if not self._tier_is_supported(source_rank, core.ProtectionTier(tier_value)):
            return "UNAVAILABLE"
        map_key = (key, tier_value)
        owner_id = self.shared_current_owner_by_key_tier.get(map_key)
        if owner_id is None:
            return "UNMATERIALIZED"
        state = self._tier_state_for_owner(
            owner_id, source_rank,
            core.ProtectionTier(tier_value),
            at=at,
        )
        return "EVICTED" if state == "ABSENT" else state

    def _new_owner(
        self, task: core.Task, assignment: PrefixAssignment,
        key: tuple[str, int], tier: core.ProtectionTier,
        token_start: int, token_end: int, at: float,
    ) -> str:
        tier_value = tier.value
        map_key = (key, tier_value)
        self.shared_epoch_counter[map_key] += 1
        epoch = self.shared_epoch_counter[map_key]
        owner_id = (
            f"__s6v2_shared__{assignment.group_id}__chunk{key[1]}__"
            f"{tier_value.lower()}__epoch{epoch}"
        )
        chunk_tokens = int(token_end) - int(token_start)
        priority_slack = self._aggregate_priority_slack_ms(key)
        owner_task = replace(
            task,
            request_id=owner_id,
            workflow_id=owner_id,
            prompt_tokens=chunk_tokens,
            output_tokens=0,
            predicted_slack_ms=priority_slack,
        )
        self.task_by_id[owner_id] = owner_task
        self.active.add(owner_id)
        self.health_burden_ms[owner_id] = 0.0
        self.copy_service_ms[owner_id] = 0.0
        self.copy_queue_wait_ms[owner_id] = 0.0
        self.shared_owner_ids.add(owner_id)
        self.shared_owner_key_tier[owner_id] = map_key
        self.shared_owner_priority_slack_ms[owner_id] = priority_slack
        self.shared_epoch_audit[owner_id] = {
            "owner_id": owner_id,
            "content_group_id": key[0],
            "content_chunk_ordinal": int(key[1]),
            "tier": tier_value,
            "epoch": epoch,
            "byte_size": chunk_tokens * self.kv,
            "admitted": None,
            "finalized": False,
            "finalize_reason": "",
        }
        rank = int(task.source_rank)
        if self.is_aegis:
            self.advance_aegis(rank, at)
            self.aegis[rank].register_request(
                replace(
                    self._profile(owner_task), request_id=owner_id,
                    source_token_end=chunk_tokens,
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
                target_failure_domain=(
                    f"target-node-{placement['peer_hbm_target_node']}"
                    if tier is core.ProtectionTier.HBM
                    else f"target-node-{placement['dram_target_node']}"
                ),
                route_version=0,
                total_tokens=chunk_tokens,
                slack_ms=task.predicted_slack_ms,
            )
            self._baseline_runtime(rank, tier).register_request(profile, time_ms=at)
        return owner_id

    def _aggregate_priority_slack_ms(self, key: tuple[str, int]) -> float:
        """Most-critical visible member, or zero-benefit slack when idle."""

        members = self.shared_ref_members[key]
        if members:
            return min(
                float(self.task_by_id[request_id].predicted_slack_ms)
                for request_id in members
            )
        start, end = self.shared_key_token_bounds[key]
        chunk_tokens = int(end) - int(start)
        return 4.0 + chunk_tokens * float(self.model["prefill_ms_per_token"])

    def _refresh_shared_priority(self, key: tuple[str, int], at: float) -> None:
        """Refresh residual-DAG value metadata without physical movement."""

        if not self.is_aegis:
            return
        rank = self.shared_group_source_rank[key[0]]
        self.advance_aegis(rank, at)
        controller = self.aegis[rank]
        desired = self._aggregate_priority_slack_ms(key)
        has_members = bool(self.shared_ref_members[key])
        for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM):
            owner_id = self.shared_current_owner_by_key_tier.get(
                (key, tier.value)
            )
            if owner_id is None:
                continue
            if self._tier_state_for_owner(owner_id, rank, tier) not in self.RESIDENT_STATES:
                continue
            prior = self.shared_owner_priority_slack_ms[owner_id]
            if abs(prior - desired) <= 1e-12:
                continue
            before_copy = sum(
                int(values[t]["bytes"])
                for values in self.copy_accounting_by_request.values()
                for t in ("HBM", "DRAM")
            )
            before = controller.telemetry_snapshot()
            before_used = sum(
                int(before["pools"][name]["used_bytes"])
                for name in ("HBM", "DRAM")
            )
            before_evictions = sum(
                int(before["pools"][name]["churn"]["eviction_count"])
                for name in ("HBM", "DRAM")
            )
            controller.update_slack(owner_id, desired)
            after = controller.telemetry_snapshot()
            after_copy = sum(
                int(values[t]["bytes"])
                for values in self.copy_accounting_by_request.values()
                for t in ("HBM", "DRAM")
            )
            after_used = sum(
                int(after["pools"][name]["used_bytes"])
                for name in ("HBM", "DRAM")
            )
            after_evictions = sum(
                int(after["pools"][name]["churn"]["eviction_count"])
                for name in ("HBM", "DRAM")
            )
            self.shared_priority_copy_byte_delta_max = max(
                self.shared_priority_copy_byte_delta_max,
                abs(after_copy - before_copy),
            )
            self.shared_priority_used_byte_delta_max = max(
                self.shared_priority_used_byte_delta_max,
                abs(after_used - before_used),
            )
            self.shared_priority_eviction_count_delta_max = max(
                self.shared_priority_eviction_count_delta_max,
                abs(after_evictions - before_evictions),
            )
            self.shared_owner_priority_slack_ms[owner_id] = desired
            self.shared_priority_metadata_update_count += 1
            if has_members:
                self.shared_priority_active_reactivation_count += 1
            else:
                self.shared_priority_zero_refcount_demotion_count += 1

    def _finalize_epoch_owner(
        self, owner_id: str, rank: int, tier: core.ProtectionTier, *,
        at: float, reason: str,
    ) -> None:
        """Remove a dead/rejected epoch from every live registry."""

        tier = core.ProtectionTier(tier)
        if self.is_aegis:
            controller = self.aegis[int(rank)]
            if controller.has_request(owner_id):
                controller.complete_request(owner_id, force_abort_pending=True)
                self.refresh_occupancy(int(rank), at)
        else:
            runtime = self._baseline_runtime(rank, tier)
            if runtime.has_request(owner_id):
                runtime.finish_request(owner_id, time_ms=at)
        self.active.discard(owner_id)
        self.health_burden_ms.pop(owner_id, None)
        self.copy_service_ms.pop(owner_id, None)
        self.copy_queue_wait_ms.pop(owner_id, None)
        self.shared_owner_priority_slack_ms.pop(owner_id, None)
        audit = self.shared_epoch_audit[owner_id]
        audit["finalized"] = True
        audit["finalize_reason"] = str(reason)

    def _admit_shared_tier(
        self, task: core.Task, assignment: PrefixAssignment,
        key: tuple[str, int], tier: core.ProtectionTier,
        token_start: int, token_end: int, at: float,
    ) -> bool:
        """Create a fresh physical epoch and attempt only the missing tier."""

        tier = core.ProtectionTier(tier)
        tier_value = tier.value
        map_key = (key, tier_value)
        old_owner = self.shared_current_owner_by_key_tier.get(map_key)
        was_evicted = map_key in self.shared_ever_admitted_key_tier
        if old_owner is not None:
            old_state = self._tier_state_for_owner(
                old_owner, int(task.source_rank), tier, at=at
            )
            if old_state in self.RESIDENT_STATES:
                raise AssertionError("attempted rematerialization of a live tier epoch")
            self._finalize_epoch_owner(
                old_owner, int(task.source_rank), tier, at=at,
                reason="evicted_epoch_replaced",
            )
            self.shared_current_owner_by_key_tier.pop(map_key, None)
        self.shared_tier_materialization_attempt_count[tier_value] += 1
        if was_evicted:
            self.shared_tier_rematerialization_attempt_count[tier_value] += 1
        owner_id = self._new_owner(
            task, assignment, key, tier, token_start, token_end, at
        )
        owner_task = self.task_by_id[owner_id]
        chunk_tokens = int(token_end) - int(token_start)
        byte_size = chunk_tokens * self.kv
        rank = int(task.source_rank)
        placement = v313.topology_placement(self.protocol, rank)
        admitted = False
        if self.is_aegis:
            controller = self.aegis[rank]
            target = int(placement["peer_hbm_target_rank"])
            desired_limit = int(
                controller.telemetry_snapshot()["pools"]["HBM"]["limit_bytes"]
            )
            spec = core.ChunkSpec(
                request_id=owner_id,
                ordinal=0,
                token_start=0,
                token_end=chunk_tokens,
                byte_size=byte_size,
                route_version=0,
                source_worker=f"source-{rank}",
                hbm_target=self._profile(owner_task).hbm_target,
                model_revision="Qwen3-8B-frozen",
                kv_abi="qwen3-fp16-kv-v1",
                parallel_layout="TP1-PP1",
            )
            if tier is core.ProtectionTier.HBM:
                effective_limit = min(
                    desired_limit,
                    max(
                        self.shared_budget_bytes
                        - self.active_serving_bytes[target],
                        0,
                    ),
                )
                controller.set_pool_limit(tier, effective_limit)
            try:
                decision = controller.admit_chunk(spec, allowed_tiers=(tier,))
            finally:
                if tier is core.ProtectionTier.HBM:
                    controller.set_pool_limit(tier, desired_limit)
            admission = decision.hbm if tier is core.ProtectionTier.HBM else decision.dram
            admitted = bool(admission.admitted)
            self.residency_counts[decision.intended_residency.value] += 1
            if admitted:
                peer_bw, dram_bw = core.copy_bandwidth(self.protocol, rank)
                if tier is core.ProtectionTier.HBM:
                    bandwidth = peer_bw
                    mode = f"peer_hbm_{placement['peer_hbm_mode']}"
                    setup_ms = 0.05
                else:
                    bandwidth = dram_bw
                    mode = f"dram_{placement['dram_mode']}"
                    setup_ms = 0.10
                duration = setup_ms + byte_size / (
                    bandwidth * core.GB_PER_S_TO_BYTES_PER_MS
                )
                formula = setup_ms + chunk_tokens * self.kv / (
                    bandwidth * core.GB_PER_S_TO_BYTES_PER_MS
                )
                self.copy_duration_formula_max_abs_error_ms = max(
                    self.copy_duration_formula_max_abs_error_ms,
                    abs(duration - formula),
                )
                lane_key = (rank, tier)
                lane_before = self.aegis_lane[lane_key]
                queue_wait = max(lane_before - at, 0.0)
                done = max(at, lane_before) + duration
                self.aegis_lane[lane_key] = done
                self.aegis_specs[(owner_id, 0, tier)] = spec
                self.path_counts[f"copy_{mode}"] += 1
                self.push(done, 0, "aegis_copy", (rank, owner_id, 0, tier))
                self._record_copy_job(
                    owner_id, tier_value, byte_size=byte_size,
                    duration_ms=duration, queue_wait_ms=queue_wait,
                )
            self.refresh_occupancy(rank, at)
            self.record_capacity(target, at, "s6v2_shared_tier_seal")
        else:
            runtime = self._baseline_runtime(rank, tier)
            lane_before = runtime.lane_available_ms
            original_config = runtime.config
            if tier is core.ProtectionTier.HBM:
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
                admission = runtime.seal_chunk(
                    owner_id,
                    core.SealedChunk(0, 0, chunk_tokens, "prefill"),
                    time_ms=at,
                )
            finally:
                runtime.config = original_config
            admitted = bool(admission.admitted)
            if admitted:
                duration = runtime.lane_available_ms - max(at, lane_before)
                queue_wait = max(lane_before - at, 0.0)
                mode = (
                    f"peer_hbm_{placement['peer_hbm_mode']}"
                    if tier is core.ProtectionTier.HBM
                    else f"dram_{placement['dram_mode']}"
                )
                self.path_counts[f"copy_{mode}"] += 1
                self._record_copy_job(
                    owner_id, tier_value,
                    byte_size=int(admission.logical_bytes),
                    duration_ms=duration, queue_wait_ms=queue_wait,
                )
        if admitted:
            self.shared_current_owner_by_key_tier[map_key] = owner_id
            self.shared_ever_admitted_key_tier.add(map_key)
            self.shared_tier_admitted_epoch_count[tier_value] += 1
            self.shared_epoch_audit[owner_id]["admitted"] = True
            if was_evicted:
                self.shared_tier_rematerialization_admitted_count[tier_value] += 1
        else:
            self.shared_tier_rejected_epoch_count[tier_value] += 1
            self.shared_epoch_audit[owner_id]["admitted"] = False
            self._finalize_epoch_owner(
                owner_id, rank, tier, at=at, reason="admission_rejected"
            )
        return admitted

    def _attach_shared_reference_v2(
        self, request_id: str, key: tuple[str, int], *,
        states: dict[str, str], byte_size: int,
    ) -> None:
        members = self.shared_ref_members[key]
        active_overlap = bool(members)
        resident = any(state == "COMMITTED" for state in states.values())
        pending = any(state == "PENDING" for state in states.values())
        if active_overlap:
            self.shared_active_overlap_hit_count += 1
        if resident:
            self.shared_resident_hit_count += 1
            if not members:
                self.shared_refcount_zero_resident_hit_count += 1
        if pending and not resident:
            self.shared_pending_only_hit_count += 1
        if any(state in self.RESIDENT_STATES for state in states.values()):
            self.shared_reference_alias_hit_count += 1
        for tier_value, state in states.items():
            if state == "PENDING":
                self.shared_tier_pending_no_duplicate_count[tier_value] += 1
            elif state == "COMMITTED":
                self.shared_tier_committed_no_duplicate_count[tier_value] += 1
        if request_id in members:
            raise AssertionError("request attached the same shared chunk twice")
        members.add(request_id)
        self.shared_refcount_value[key] += 1
        self.request_shared_keys[request_id].add(key)
        self.shared_reference_count += 1
        self._check_refcount(key)

    def handle_protection_seal(self, payload: tuple[Any, ...], at: float) -> None:
        if len(payload) != 5:
            raise AssertionError("S6 v2 seal payload must include kind")
        kind, request_id, ordinal, token_start, token_end = payload
        if request_id not in self.active:
            return
        if kind == "unique":
            v313.CausalPhysicalTargetReplay.handle_protection_seal(
                self,
                (request_id, int(ordinal), int(token_start), int(token_end)),
                at,
            )
            self._sample_physical_occupancy(at)
            return
        if kind != "shared":
            raise AssertionError(f"unknown S6 v2 seal kind {kind!r}")
        assignment = self._assignment(request_id)
        if int(token_end) > assignment.shared_tokens:
            raise AssertionError("shared chunk exceeds registered prefix")
        self.seal_event_count += 1
        task = self.task_by_id[request_id]
        if at < task.start_ms - 1e-9:
            self.seal_before_dispatch_count += 1
        key = (assignment.group_id, int(ordinal))
        bounds = (int(token_start), int(token_end))
        prior_bounds = self.shared_key_token_bounds.setdefault(key, bounds)
        if prior_bounds != bounds:
            raise AssertionError("content identity changed token bounds")
        states = {
            tier.value: self.shared_tier_state(key, tier, at=at)
            for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM)
        }
        self._attach_shared_reference_v2(
            request_id, key, states=states,
            byte_size=(int(token_end) - int(token_start)) * self.kv,
        )
        admitted_any = False
        for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM):
            if states[tier.value] == "UNAVAILABLE":
                continue
            if states[tier.value] in self.RESIDENT_STATES:
                continue
            admitted_any = self._admit_shared_tier(
                task, assignment, key, tier,
                int(token_start), int(token_end), at,
            ) or admitted_any
        if admitted_any or key in self.shared_materialized_keys:
            self.shared_materialized_keys.add(key)
        self._refresh_shared_priority(key, at)
        self._sample_physical_occupancy(at)

    def finish(self, task: core.Task, at: float) -> None:
        keys = tuple(self.request_shared_keys.get(task.request_id, ()))
        super().finish(task, at)
        for key in keys:
            self._refresh_shared_priority(key, at)

    def _aegis_tier_parts(
        self, task: core.Task, tier: core.ProtectionTier,
    ) -> tuple[int, int]:
        assignment = self._assignment(task.request_id)
        tier = core.ProtectionTier(tier)
        controller = self.aegis[task.source_rank]
        chunk_tokens = int(self.protocol["aegiskv"]["prefix_chunk_tokens"])
        shared = 0
        for ordinal, start in enumerate(
            range(0, assignment.shared_tokens, chunk_tokens)
        ):
            end = min(start + chunk_tokens, assignment.shared_tokens)
            key = (assignment.group_id, ordinal)
            owner = self.shared_current_owner_by_key_tier.get((key, tier.value))
            if (
                owner is None
                or controller.copy_state(owner, 0, tier) is not core.CopyState.COMMITTED
            ):
                break
            shared += end - start
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
        runtime.advance_to(probe.time_ms)
        tier_value = runtime.config.tier.value
        chunk_tokens = int(self.protocol["aegiskv"]["prefix_chunk_tokens"])
        shared = 0
        for ordinal, start in enumerate(
            range(0, assignment.shared_tokens, chunk_tokens)
        ):
            end = min(start + chunk_tokens, assignment.shared_tokens)
            key = (assignment.group_id, ordinal)
            owner = self.shared_current_owner_by_key_tier.get((key, tier_value))
            if owner is None:
                break
            state = runtime.request_replica_state(owner)
            if (
                state["presence"] != "COMMITTED"
                or int(state["committed_frontier"]) < end - start
            ):
                break
            shared += end - start
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

    def _shared_scheduled_copy_bytes(self) -> dict[str, int]:
        values = {"HBM": 0, "DRAM": 0}
        for request_id, tiers in self.copy_accounting_by_request.items():
            if request_id not in self.shared_owner_ids:
                continue
            for tier in values:
                values[tier] += int(tiers[tier]["bytes"])
        return values

    def _sweep_dead_epoch_owners(self) -> None:
        """Finalize evicted epochs even when no later source rematerializes."""

        for map_key, owner_id in list(
            self.shared_current_owner_by_key_tier.items()
        ):
            key, tier_value = map_key
            rank = self.shared_group_source_rank[key[0]]
            tier = core.ProtectionTier(tier_value)
            state = self._tier_state_for_owner(owner_id, rank, tier)
            if state in self.RESIDENT_STATES:
                continue
            if self.is_aegis:
                at = float(self.aegis_time[rank])
            else:
                at = float(self._baseline_runtime(rank, tier).now_ms)
            self._finalize_epoch_owner(
                owner_id, rank, tier, at=at,
                reason="evicted_epoch_terminal_sweep",
            )
            self.shared_current_owner_by_key_tier.pop(map_key, None)

    def summarize_mechanism(self, control_horizon_ms: float) -> dict[str, Any]:
        self._sweep_dead_epoch_owners()
        mechanism = super().summarize_mechanism(control_horizon_ms)
        row = mechanism["active_hbm_v313_s6"]
        row.update({
            "schema_version": 2,
            "identity_semantics": (
                "stable (group,ordinal) content identity with independent per-tier "
                "physical epoch owners; PENDING/COMMITTED suppress same-tier "
                "duplicates; EVICTED tiers are causally rematerialized"
            ),
            "shared_physical_epoch_owner_count": len(self.shared_owner_ids),
            "shared_tier_materialization_attempt_count": dict(
                self.shared_tier_materialization_attempt_count
            ),
            "shared_tier_admitted_epoch_count": dict(
                self.shared_tier_admitted_epoch_count
            ),
            "shared_tier_rejected_epoch_count": dict(
                self.shared_tier_rejected_epoch_count
            ),
            "shared_tier_rematerialization_attempt_count": dict(
                self.shared_tier_rematerialization_attempt_count
            ),
            "shared_tier_rematerialization_admitted_count": dict(
                self.shared_tier_rematerialization_admitted_count
            ),
            "shared_tier_pending_no_duplicate_count": dict(
                self.shared_tier_pending_no_duplicate_count
            ),
            "shared_tier_committed_no_duplicate_count": dict(
                self.shared_tier_committed_no_duplicate_count
            ),
            "shared_active_overlap_hit_count": self.shared_active_overlap_hit_count,
            "shared_resident_hit_count": self.shared_resident_hit_count,
            "shared_refcount_zero_resident_hit_count": (
                self.shared_refcount_zero_resident_hit_count
            ),
            "shared_pending_only_hit_count": self.shared_pending_only_hit_count,
            "content_identity_epoch_owner_sha256": digest_json([
                (owner, key, tier)
                for owner, (key, tier) in sorted(self.shared_owner_key_tier.items())
            ]),
            "shared_epoch_audit_sha256": digest_json([
                self.shared_epoch_audit[owner]
                for owner in sorted(self.shared_epoch_audit)
            ]),
            "shared_live_epoch_owner_count": sum(
                int(owner in self.active)
                for owner in self.shared_owner_ids
            ),
            "shared_current_resident_tier_count": len(
                self.shared_current_owner_by_key_tier
            ),
            "shared_orphan_active_epoch_owner_count": sum(
                int(
                    owner in self.active
                    and owner not in self.shared_current_owner_by_key_tier.values()
                )
                for owner in self.shared_owner_ids
            ),
            "per_tier_rematerialization_independent": True,
            "shared_priority_aggregate_rule": (
                "minimum predicted residual-DAG slack across current logical "
                "members; at refcount zero set slack to full chunk recompute "
                "cost so current protection benefit is zero"
            ),
            "shared_priority_metadata_update_count": (
                self.shared_priority_metadata_update_count
            ),
            "shared_priority_zero_refcount_demotion_count": (
                self.shared_priority_zero_refcount_demotion_count
            ),
            "shared_priority_active_reactivation_count": (
                self.shared_priority_active_reactivation_count
            ),
            "shared_priority_copy_byte_delta_max": (
                self.shared_priority_copy_byte_delta_max
            ),
            "shared_priority_used_byte_delta_max": (
                self.shared_priority_used_byte_delta_max
            ),
            "shared_priority_eviction_count_delta_max": (
                self.shared_priority_eviction_count_delta_max
            ),
        })
        if len(self.shared_owner_ids) != sum(
            self.shared_epoch_counter.values()
        ):
            raise AssertionError("physical epoch owner accounting mismatch")
        if row["shared_orphan_active_epoch_owner_count"]:
            raise AssertionError("dead/rejected shared epoch remains active")
        if (
            row["shared_live_epoch_owner_count"]
            != row["shared_current_resident_tier_count"]
        ):
            raise AssertionError("live shared epochs do not match resident tiers")
        if any((
            row["shared_priority_copy_byte_delta_max"],
            row["shared_priority_used_byte_delta_max"],
            row["shared_priority_eviction_count_delta_max"],
        )):
            raise AssertionError("priority metadata refresh moved physical state")
        return mechanism
