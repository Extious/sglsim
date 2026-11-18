#!/usr/bin/env python3
"""Executable hard gates for S6 v2 per-tier shared residency epochs."""

from __future__ import annotations

from copy import deepcopy

import active_hbm_v313_s6_v2 as s6v2
import run_preliminary as core
import test_active_hbm_v313 as v313_tests


def inputs() -> tuple[dict, dict]:
    protocol, capacity = v313_tests.inputs(context=4096, active=2)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    protocol["hardware"]["gpus_per_node"] = 8
    protocol["hardware"]["node_count"] = 2
    protocol.setdefault("runtime_overrides", {})["topology"] = {
        "peer_hbm": "same_node", "dram": "local",
    }
    return protocol, capacity


def tasks(count: int = 2) -> list[core.Task]:
    return [
        core.Task(
            request_id=f"request-{index}", workflow_id=f"workflow-{index}",
            workflow_arrival_ms=0.0, arrival_ms=0.0, source_rank=0,
            lane=index, start_ms=0.0, finish_ms=20000.0,
            service_ms=20000.0, prompt_tokens=4096, output_tokens=256,
            predicted_slack_ms=0.0,
        )
        for index in range(count)
    ]


def make_replay(policy: str = "AegisKV", count: int = 2):
    protocol, capacity = inputs()
    nominal = tasks(count)
    assignments = s6v2.build_prefix_assignments(
        nominal, reuse_fraction=1.0, group_size=count, chunk_tokens=4096
    )
    replay = s6v2.SharedPrefixReplay(
        deepcopy(protocol), policy, capacity,
        assignments=assignments, reuse_fraction=1.0, group_size=count,
    )
    replay.task_by_id = {task.request_id: task for task in nominal}
    replay.occupancy_observation_horizon_ms = 30000.0
    return protocol, nominal, assignments, replay


def seal(replay, task: core.Task, at: float) -> tuple[str, int]:
    replay.start(task, at)
    replay.handle_protection_seal(
        ("shared", task.request_id, 0, 0, task.prompt_tokens), at + 1000.0
    )
    return (replay.assignments[task.request_id].group_id, 0)


def complete_aegis_tiers(replay, key: tuple[str, int]) -> float:
    events = []
    for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM):
        owner = replay.shared_current_owner_by_key_tier[(key, tier.value)]
        done = replay.aegis_lane[(0, tier)]
        events.append((done, owner, tier))
    for done, owner, tier in sorted(events, key=lambda row: row[0]):
        replay.complete_copy((0, owner, 0, tier), done)
    return max(done for done, _owner, _tier in events)


def complete_native_dram(replay, key: tuple[str, int]) -> float:
    runtime = replay._baseline_runtime(0, core.ProtectionTier.DRAM)
    done = runtime.lane_available_ms
    runtime.advance_to(done)
    owner = replay.shared_current_owner_by_key_tier[
        (key, core.ProtectionTier.DRAM.value)
    ]
    if runtime.request_replica_state(owner)["presence"] != "COMMITTED":
        raise AssertionError("native DRAM shared epoch did not commit")
    return done


def rejection_retry_case(rejected_tiers: tuple[core.ProtectionTier, ...]):
    _protocol, nominal, _assignments, replay = make_replay()
    controller = replay.aegis[0]
    for tier in rejected_tiers:
        controller.set_pool_limit(tier, 0)
    key = seal(replay, nominal[0], 0.0)
    rejected = [
        owner for owner in replay.shared_owner_ids
        if replay.shared_epoch_audit[owner]["admitted"] is False
    ]
    no_orphans_after_reject = all(
        owner not in replay.active and not controller.has_request(owner)
        for owner in rejected
    )
    for tier in rejected_tiers:
        restored = 10 * core.GIB if tier is core.ProtectionTier.HBM else 64 * core.GIB
        controller.set_pool_limit(tier, restored)
    seal(replay, nominal[1], 1001.0)
    retried_admitted = all(
        (key, tier.value) in replay.shared_current_owner_by_key_tier
        for tier in rejected_tiers
    )
    no_orphans_after_retry = all(
        owner in replay.shared_current_owner_by_key_tier.values()
        for owner in replay.shared_owner_ids if owner in replay.active
    )
    return replay, key, rejected, (
        no_orphans_after_reject and retried_admitted and no_orphans_after_retry
    )


def main() -> None:
    checks: list[tuple[str, bool]] = []

    # A same-time follower aliases each tier's PENDING epoch and schedules no
    # duplicate physical transfer in that tier.
    _protocol, nominal, _assignments, pending = make_replay()
    pending.start(nominal[0], 0.0)
    pending.start(nominal[1], 0.0)
    pending.handle_protection_seal(
        ("shared", nominal[0].request_id, 0, 0, 4096), 1000.0
    )
    key = (pending.assignments[nominal[0].request_id].group_id, 0)
    first_owners = dict(pending.shared_current_owner_by_key_tier)
    first_bytes = {
        tier: sum(
            int(row[tier]["bytes"])
            for row in pending.copy_accounting_by_request.values()
        )
        for tier in ("HBM", "DRAM")
    }
    pending.handle_protection_seal(
        ("shared", nominal[1].request_id, 0, 0, 4096), 1000.0
    )
    second_bytes = {
        tier: sum(
            int(row[tier]["bytes"])
            for row in pending.copy_accounting_by_request.values()
        )
        for tier in ("HBM", "DRAM")
    }
    checks.append((
        "same_tier_pending_epoch_suppresses_duplicate_copy",
        first_owners == pending.shared_current_owner_by_key_tier
        and first_bytes == second_bytes
        and pending.shared_tier_pending_no_duplicate_count
        == {"HBM": 1, "DRAM": 1},
    ))

    # Refcount zero does not evict a resident cache object.  The late request
    # must be a resident hit rather than a fresh producer.
    _protocol, nominal, _assignments, resident = make_replay()
    key = seal(resident, nominal[0], 0.0)
    done = complete_aegis_tiers(resident, key)
    owners_before = dict(resident.shared_current_owner_by_key_tier)
    copy_before_priority = resident._shared_scheduled_copy_bytes()
    resident.finish(nominal[0], done + 1.0)
    zero_benefit_slack = 4.0 + 4096 * float(
        resident.model["prefill_ms_per_token"]
    )
    demoted_slacks = [
        resident.aegis[0].request_profile(owner).slack_ms
        for owner in owners_before.values()
    ]
    seal(resident, nominal[1], done + 2.0)
    reactivated_slacks = [
        resident.aegis[0].request_profile(owner).slack_ms
        for owner in owners_before.values()
    ]
    checks.append((
        "refcount_zero_committed_cache_is_late_resident_hit",
        resident.shared_current_owner_by_key_tier == owners_before
        and resident.shared_refcount_zero_resident_hit_count == 1
        and resident.shared_resident_hit_count == 1
        and resident.shared_active_overlap_hit_count == 0
        and resident.shared_tier_committed_no_duplicate_count
        == {"HBM": 1, "DRAM": 1},
    ))
    checks.append((
        "aggregate_priority_demotes_idle_and_reactivates_metadata_only",
        all(abs(value - zero_benefit_slack) <= 1e-12 for value in demoted_slacks)
        and reactivated_slacks == [0.0, 0.0]
        and resident.shared_priority_zero_refcount_demotion_count == 2
        and resident.shared_priority_active_reactivation_count == 2
        and resident.shared_priority_metadata_update_count == 4
        and resident._shared_scheduled_copy_bytes() == copy_before_priority
        and resident.shared_priority_copy_byte_delta_max == 0
        and resident.shared_priority_used_byte_delta_max == 0
        and resident.shared_priority_eviction_count_delta_max == 0,
    ))

    # The same resident-cache rule is required for modeled Native's Store-only
    # DRAM tier; HBM is explicitly unavailable rather than silently emulated.
    _protocol, nominal, _assignments, native = make_replay("Native Mooncake")
    key = seal(native, nominal[0], 0.0)
    done = complete_native_dram(native, key)
    owner_before = native.shared_current_owner_by_key_tier[
        (key, core.ProtectionTier.DRAM.value)
    ]
    native.finish(nominal[0], done + 1.0)
    seal(native, nominal[1], done + 2.0)
    checks.append((
        "native_store_refcount_zero_resident_hit_without_fake_hbm",
        native.shared_current_owner_by_key_tier[
            (key, core.ProtectionTier.DRAM.value)
        ] == owner_before
        and (key, core.ProtectionTier.HBM.value)
        not in native.shared_current_owner_by_key_tier
        and native.shared_refcount_zero_resident_hit_count == 1,
    ))

    # Evict only HBM.  The next source rematerializes HBM with a new epoch
    # while retaining the still-COMMITTED DRAM epoch.
    _protocol, nominal, _assignments, single = make_replay()
    key = seal(single, nominal[0], 0.0)
    done = complete_aegis_tiers(single, key)
    hbm_map = (key, core.ProtectionTier.HBM.value)
    dram_map = (key, core.ProtectionTier.DRAM.value)
    old_hbm = single.shared_current_owner_by_key_tier[hbm_map]
    old_dram = single.shared_current_owner_by_key_tier[dram_map]
    single.finish(nominal[0], done + 1.0)
    single.aegis[0].invalidate_tier(
        old_hbm, core.ProtectionTier.HBM, reason="test_single_tier_eviction"
    )
    state_before = {
        tier.value: single.shared_tier_state(key, tier, at=done + 2.0)
        for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM)
    }
    seal(single, nominal[1], done + 3.0)
    new_hbm = single.shared_current_owner_by_key_tier[hbm_map]
    new_dram = single.shared_current_owner_by_key_tier[dram_map]
    checks.append((
        "single_tier_eviction_rematerializes_only_missing_tier",
        state_before == {"HBM": "EVICTED", "DRAM": "COMMITTED"}
        and new_hbm != old_hbm and new_dram == old_dram
        and single.shared_tier_rematerialization_admitted_count
        == {"HBM": 1, "DRAM": 0}
        and single.shared_tier_committed_no_duplicate_count["DRAM"] == 1,
    ))
    probe = core.FaultProbe(
        "single-tier-survival", done + 3.1, 0, "source_worker_gpu"
    )
    _delay, action = single.recovery(nominal[1], probe)
    checks.append((
        "single_surviving_committed_tier_remains_recoverable",
        action.startswith("dram_local")
        and single.shared_recovery_restored_tokens == nominal[1].prompt_tokens,
    ))

    # With both tiers evicted, the next source becomes the producer of one new
    # epoch per tier.  Logical content identity stays unchanged.
    _protocol, nominal, _assignments, full = make_replay()
    key = seal(full, nominal[0], 0.0)
    done = complete_aegis_tiers(full, key)
    old = dict(full.shared_current_owner_by_key_tier)
    full.finish(nominal[0], done + 1.0)
    for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM):
        full.aegis[0].invalidate_tier(
            old[(key, tier.value)], tier, reason="test_full_eviction"
        )
    seal(full, nominal[1], done + 2.0)
    checks.append((
        "full_eviction_creates_new_per_tier_epochs_same_content_identity",
        all(
            full.shared_current_owner_by_key_tier[(key, tier.value)]
            != old[(key, tier.value)]
            for tier in (core.ProtectionTier.HBM, core.ProtectionTier.DRAM)
        )
        and full.shared_key_token_bounds == {key: (0, 4096)}
        and full.shared_tier_rematerialization_admitted_count
        == {"HBM": 1, "DRAM": 1},
    ))

    chunk_bytes = 4096 * full.kv
    scheduled = full._shared_scheduled_copy_bytes()
    checks.append((
        "rematerialized_physical_bytes_and_epoch_counts_conserve",
        scheduled == {"HBM": 2 * chunk_bytes, "DRAM": 2 * chunk_bytes}
        and len(full.shared_owner_ids) == 4
        and sum(full.shared_epoch_counter.values()) == 4
        and full.refcount_max_abs_error == 0
        and full.refcount_underflow_count == 0,
    ))

    # End-to-end event-loop gate: no direct healthy WCT surcharge, exact
    # reference drain, and copy/controller byte conservation.
    protocol, capacity = inputs()
    nominal = tasks(4)
    assignments = s6v2.build_prefix_assignments(
        nominal, reuse_fraction=1.0, group_size=4, chunk_tokens=4096
    )
    coupled = s6v2.SharedPrefixReplay(
        deepcopy(protocol), "AegisKV", capacity,
        assignments=assignments, reuse_fraction=1.0, group_size=4,
    )
    scheduled_tasks, delays, mechanism, trace = coupled.run_coupled(
        nominal,
        [core.FaultProbe("probe", 5000.0, 0, "source_worker_gpu")],
        control_horizon_ms=10000.0,
    )
    row = mechanism["active_hbm_v313_s6"]
    checks.append((
        "coupled_refcount_copy_capacity_and_recovery_conserve",
        row["schema_version"] == 2
        and row["refcount_final_total"] == 0
        and row["refcount_max_abs_error"] == 0
        and row["refcount_underflow_count"] == 0
        and mechanism["active_hbm_v313"]["copy_byte_conservation_error"]
        == {"HBM": 0, "DRAM": 0}
        and mechanism["active_hbm_v313"]["target_capacity_violation_count"] == 0
        and all(
            int(point["shared_used_bytes"]) <= int(coupled.shared_budget_bytes)
            for point in trace
        )
        and len(delays) == 1
        and all(
            task.finish_ms == task.start_ms + task.service_ms
            for task in scheduled_tasks
        ),
    ))

    # Aegis completion state and Native baseline state must both be derived
    # from the physical tier object, never from the logical identity set.
    checks.append((
        "per_tier_state_is_physical_not_permanent_identity",
        resident.shared_tier_state(key, core.ProtectionTier.HBM) == "COMMITTED"
        and full.shared_materialized_keys == {key}
        and full.shared_current_owner_by_key_tier[hbm_map] != old[hbm_map],
    ))

    hbm_reject, reject_key, rejected, passed = rejection_retry_case(
        (core.ProtectionTier.HBM,)
    )
    checks.append((
        "hbm_only_reject_has_no_orphan_and_later_retries",
        passed and len(rejected) == 1
        and hbm_reject.shared_tier_rejected_epoch_count
        == {"HBM": 1, "DRAM": 0}
        and hbm_reject.shared_refcount_value[reject_key] == 2,
    ))

    dram_reject, reject_key, rejected, passed = rejection_retry_case(
        (core.ProtectionTier.DRAM,)
    )
    checks.append((
        "dram_only_reject_has_no_orphan_and_later_retries",
        passed and len(rejected) == 1
        and dram_reject.shared_tier_rejected_epoch_count
        == {"HBM": 0, "DRAM": 1}
        and dram_reject.shared_refcount_value[reject_key] == 2,
    ))

    both_reject, reject_key, rejected, passed = rejection_retry_case(
        (core.ProtectionTier.HBM, core.ProtectionTier.DRAM)
    )
    checks.append((
        "both_reject_cleanup_retry_and_legacy_materialized_set_are_exact",
        passed and len(rejected) == 2
        and both_reject.shared_tier_rejected_epoch_count
        == {"HBM": 1, "DRAM": 1}
        and both_reject.shared_materialized_keys == {reject_key}
        and both_reject.shared_refcount_value[reject_key] == 2,
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
