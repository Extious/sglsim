from __future__ import annotations

import copy

import pytest

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.aegis_recovery_snapshot import (
    SNAPSHOT_SCHEMA,
    audit_algorithm2_inputs,
    audit_recovery_laxity_inputs,
    audit_snapshot_capabilities,
    build_causal_recovery_snapshot,
    validate_causal_snapshot,
)


def _causal_state(dp_rank: int, request: dict) -> dict:
    return {
        "schema_version": "aegiskv-causal-scheduler-state-v1",
        "information_scope": "causal_scheduler_state_after_step",
        "forbidden_future_fields_present": False,
        "dp_rank": dp_rank,
        "pp_rank": 0,
        "tp_rank": 0,
        "model_path": "mock-model",
        "waiting_request_ids": [],
        "running_request_ids": [request["rid"]],
        "inflight_request_ids": [],
        "chunked_request_id": None,
        "active_requests": [request],
        "max_running_requests": 8,
        "l1_total_tokens": 100,
        "l1_used_tokens": 20,
        "l1_available_tokens": 80,
        "l1_available_and_evictable_tokens": 80,
        "l2_total_tokens": 200,
        "l2_used_tokens": 10,
        "l2_available_tokens": 190,
    }


def _request(rid: str, workflow_id: str, object_id: str) -> dict:
    return {
        "rid": rid,
        "workflow_id": workflow_id,
        "agent_id": f"agent-{rid}",
        "stage_id": 1,
        "stage_width": 2,
        "queue_location": "running",
        "request_status": "decode",
        "created_time_s": 1.0,
        "queue_start_time_s": 1.0,
        "prompt_tokens": 32,
        "requested_max_new_tokens": 16,
        "generated_tokens_observed": 4,
        "kv_committed_tokens_observed": 36,
        "cache_hit_tokens_observed": 0,
        "host_hit_tokens_observed": 0,
        "restored_prefill_tokens_observed": 0,
        "slack_lcb_ms": 3.0,
        "workflow_deadline_s": None,
        "kv_object_id": object_id,
        "object_id_observed": True,
        "assigned_dp_rank": int(rid[-1]),
    }


def _telemetry_row(dp_rank: int, completed_at: float, request: dict) -> dict:
    return {
        "dp_rank": dp_rank,
        "pp_rank": 0,
        "tp_rank": 0,
        "batch_id": 3,
        "batch_mode": "decode",
        "request_ids": [request["rid"]],
        "waiting_depth": 0,
        "running_depth": 1,
        "waiting_depth_after": 0,
        "running_depth_after": 1,
        "pipeline_base_s": 0.02,
        "completed_at": completed_at,
        "causal_state": _causal_state(dp_rank, request),
    }


def test_snapshot_uses_only_latest_state_before_failure_cutoff() -> None:
    victim = _request("r0", "workflow-victim", "object-v")
    bystander = _request("r1", "workflow-bystander", "object-b")
    future = copy.deepcopy(victim)
    future["generated_tokens_observed"] = 16
    artifact = {
        "simulation_time_mode": "wall",
        "evidence_tier": "sglsim-online-instrumented",
        "failure_events": [
            {
                "event_id": "failure-1",
                "observed_at_s": 10.0,
                "operation": "inject",
                "failure_mode": "hard",
                "dp_rank": 0,
                "failure_domain": "dp_rank:0",
                "accepted": True,
            }
        ],
        "scheduler_telemetry": [
            _telemetry_row(0, 9.0, victim),
            _telemetry_row(1, 9.5, bystander),
            _telemetry_row(0, 11.0, future),
        ],
        "aegis_status": {
            "events": [
                {"kind": "backup_ack", "time": 8.0, "backup_key": "object-v"},
                {"kind": "backup_ack", "time": 12.0, "backup_key": "future"},
            ]
        },
    }

    snapshot = build_causal_recovery_snapshot(
        artifact, failure_event_id="failure-1"
    )

    assert snapshot["schema_version"] == SNAPSHOT_SCHEMA
    assert snapshot["posthoc_outcomes_included"] is False
    rows = {row["rid"]: row for row in snapshot["active_requests"]}
    assert rows["r0"]["generated_tokens_observed"] == 4
    assert rows["r0"]["failure_role"] == "interrupted"
    assert rows["r1"]["failure_role"] == "bystander"
    assert [row["backup_key"] for row in snapshot["copy_events_observed_before_cutoff"]] == [
        "object-v"
    ]
    audit = audit_snapshot_capabilities(snapshot)
    assert audit["capabilities"]["source_failure_domain"] == "SUPPORTED"
    assert audit["capabilities"]["target_queue_batch_state"] == "SUPPORTED"
    assert audit["capabilities"]["bystander_identity"] == "SUPPORTED"
    assert audit["capabilities"]["target_reusable_prefix"] == "MISSING"
    assert audit["g1_ready"] is False


def test_snapshot_replays_only_live_target_prefixes_at_failure() -> None:
    victim = _request("r0", "workflow-victim", "object-v")
    bystander = _request("r1", "workflow-bystander", "object-b")

    def hint(
        *,
        rid: str,
        key: str,
        source: int,
        target: int,
        version: int,
        l1: int,
        l2: int,
    ) -> dict:
        return {
            "rid": rid,
            "backup_key": key,
            "source_dp_rank": source,
            "target_dp_rank": target,
            "version": version,
            "l1_tokens": l1,
            "l2_tokens": l2,
            "l3_tokens": 0,
            "recompute_tokens": 0,
            "materialization_scope": "committed_prefix",
            "source_committed_token_end": 64,
            "group_commit_verified": True,
            "copy_completion_evidence": "modeled_unit_group_commit",
            "execution_id": f"exec-{key}-v{version}",
            "block_generation": version,
        }

    events = [
        {
            "kind": "backup_ack",
            "sequence": 1,
            "time": 7.0,
            "hint_updates": [
                hint(
                    rid="r0",
                    key="object-v",
                    source=0,
                    target=1,
                    version=2,
                    l1=32,
                    l2=16,
                )
            ],
        },
        # A delayed lower-version ACK cannot resurrect older state.
        {
            "kind": "backup_ack",
            "sequence": 2,
            "time": 8.0,
            "hint_updates": [
                hint(
                    rid="r0",
                    key="object-v",
                    source=0,
                    target=1,
                    version=1,
                    l1=999,
                    l2=0,
                )
            ],
        },
        # This is a real copy, but its target is the failed rank.
        {
            "kind": "backup_ack",
            "sequence": 3,
            "time": 8.5,
            "hint_updates": [
                hint(
                    rid="r1",
                    key="object-b",
                    source=1,
                    target=0,
                    version=1,
                    l1=20,
                    l2=0,
                )
            ],
        },
        {
            "kind": "host_pool_lease_evicted",
            "sequence": 4,
            "time": 9.0,
            "backup_key": "object-v",
            "version": 2,
            "target_dp_rank": 1,
            "lost_l2_tokens": 8,
        },
        {
            "kind": "backup_ack",
            "sequence": 5,
            "time": 9.2,
            "hint_updates": [
                hint(
                    rid="old",
                    key="old-object",
                    source=0,
                    target=1,
                    version=1,
                    l1=12,
                    l2=0,
                )
            ],
        },
        {
            "kind": "backup_frontier_superseded",
            "sequence": 6,
            "time": 9.3,
            "backup_key": "old-object",
            "version": 1,
            "superseded_by": "object-v",
        },
        # Present in the complete journal, but causally excluded by the cutoff.
        {
            "kind": "backup_ack",
            "sequence": 7,
            "time": 12.0,
            "hint_updates": [
                hint(
                    rid="future",
                    key="future-object",
                    source=0,
                    target=1,
                    version=1,
                    l1=100,
                    l2=0,
                )
            ],
        },
    ]
    victim["kv_committed_tokens_observed"] = 64
    artifact = {
        "simulation_time_mode": "wall",
        "evidence_tier": "unit",
        "failure_events": [
            {
                "event_id": "failure-1",
                "observed_at_s": 10.0,
                "operation": "inject",
                "failure_mode": "hard",
                "dp_rank": 0,
                "failure_domain": "dp_rank:0",
                "accepted": True,
            }
        ],
        "scheduler_telemetry": [
            _telemetry_row(0, 9.7, victim),
            _telemetry_row(1, 9.8, bystander),
        ],
        "aegis_status": {
            "event_history": {
                "mode": "unbounded_in_memory",
                "event_count": len(events),
            },
            "events": events,
        },
    }
    snapshot = build_causal_recovery_snapshot(
        artifact, failure_event_id="failure-1"
    )

    prefixes = {
        row["backup_key"]: row
        for row in snapshot["recoverable_prefixes_at_cutoff"]
    }
    assert set(prefixes) == {"object-v", "object-b"}
    assert prefixes["object-v"]["continuous_recoverable_prefix_tokens"] == 40
    assert prefixes["object-v"]["survives_failure"] is True
    assert prefixes["object-b"]["survives_failure"] is False
    assert snapshot["copy_state_replay"]["causal_replay_complete"] is True
    rows = {row["rid"]: row for row in snapshot["active_requests"]}
    assert rows["r0"]["target_prefix_observation_status"] == "OBSERVED_POSITIVE"
    assert rows["r0"]["target_recovery_candidates"] == [
        {
            "target_id": "dp:1",
            "target_dp_rank": 1,
            "backup_key": "object-v",
            "version": 2,
            "continuous_recoverable_prefix_tokens": 40,
            "l1_tokens": 32,
            "l2_tokens": 8,
            "l3_tokens": 0,
            "state_exact": True,
            "evidence_source": "controller_event_replay",
            "materialization_scope": "committed_prefix",
            "source_committed_token_end": 64,
            "group_commit_verified": True,
            "copy_completion_evidence": "modeled_unit_group_commit",
            "execution_id": "exec-object-v-v2",
            "block_generation": 2,
            "protected_recovery_service_s": None,
            "unprotected_recovery_service_s": None,
            "recovery_service_cost_source": "",
            "recovery_service_cost_calibration_id": "",
            "model_layout_id": "",
            "layout_compatible": None,
        }
    ]
    audit = audit_snapshot_capabilities(snapshot)
    assert audit["capabilities"]["target_reusable_prefix"] == "SUPPORTED"
    assert audit["positive_target_prefix_count"] == 1
    assert audit["g1_ready"] is False
    alg2 = audit_algorithm2_inputs(snapshot)
    assert alg2["checks"]["committed_target_prefix"] is True
    assert alg2["checks"]["interrupted_workflow_and_domain"] is True
    assert alg2["algorithm2_runnable"] is False
    assert alg2["safe_result"] == "REFUSED_TO_SCORE"
    assert "target_queue_work_seconds" in alg2["blocking_inputs"]
    laxity = audit_recovery_laxity_inputs(snapshot)
    assert laxity["algorithm2_runnable"] is False
    assert laxity["safe_result"] == "REFUSED_TO_SCORE"
    assert "identified_bystander_outcome_model" not in laxity["checks"]
    assert "bystander_counterfactual_completion_model" in laxity[
        "removed_online_requirements"
    ]


def test_recovery_laxity_audit_accepts_only_enriched_causal_inputs() -> None:
    victim = _request("r0", "workflow-victim", "object-v")
    victim.update(
        {
            "kv_committed_tokens_observed": 64,
            "slack_lcb_ms": 2.5,
            "slack_target_coverage": 0.9,
            "slack_calibration_id": "slack-heldout-v3",
            "slack_source": "online_revealed_dag_quantile_model",
            "slack_kind": "CALIBRATED",
            "need_state": "ACTIVE",
            "slack_semantics": "revealed_dag_causal_slack_v1",
            "slack_uses_future_information": False,
            "slack_usable": True,
            "slack_input_provenance_sha256": "a" * 64,
            "recovery_target_id": "dp:1",
            "recovery_target_dp_rank": 1,
            "recovery_target_assignment_source": "router_failure_plan",
            "recovery_route_plan_version": "route-v7",
        }
    )
    bystander = _request("r1", "workflow-bystander", "object-b")
    victim_row = _telemetry_row(0, 9.7, victim)
    target_row = _telemetry_row(1, 9.8, bystander)
    target_row["causal_state"].update(
        {
            "target_availability_offset_s": 0.15,
            "target_availability_source": "router_queue_service_estimator",
            "target_availability_plan_version": "route-v7",
        }
    )
    event = {
        "kind": "backup_ack",
        "sequence": 1,
        "time": 9.0,
        "hint_updates": [
            {
                "rid": "r0",
                "backup_key": "object-v",
                "source_dp_rank": 0,
                "target_dp_rank": 1,
                "version": 3,
                "l1_tokens": 32,
                "l2_tokens": 16,
                "l3_tokens": 0,
                "recompute_tokens": 16,
                "materialization_scope": "committed_prefix",
                "source_committed_token_end": 64,
                "group_commit_verified": True,
                "copy_completion_evidence": "unit_group_commit",
                "execution_id": "exec-r0-v3",
                "block_generation": 8,
                "protected_recovery_service_s": 0.20,
                "unprotected_recovery_service_s": 0.80,
                "recovery_service_cost_source": "heldout_physical_profile",
                "recovery_service_cost_calibration_id": "recovery-v4",
                "model_layout_id": "model=mock,tp=1,pp=1,kv=float16",
                "layout_compatible": True,
            }
        ],
    }
    artifact = {
        "simulation_time_mode": "wall",
        "evidence_tier": "unit",
        "failure_events": [
            {
                "event_id": "failure-1",
                "observed_at_s": 10.0,
                "operation": "inject",
                "failure_mode": "hard",
                "dp_rank": 0,
                "failure_domain": "dp_rank:0",
                "accepted": True,
            }
        ],
        "scheduler_telemetry": [victim_row, target_row],
        "aegis_status": {
            "event_history": {
                "mode": "unbounded_in_memory",
                "event_count": 1,
            },
            "events": [event],
        },
        "protection_resource_state": {
            "observed_at_s": 9.9,
            "peer_hbm_budget_bytes": 1024,
            "store_protection_budget_bytes": 4096,
            "copy_rate_cap_bytes_per_s": 8192,
            "foreground_reserve_source": "no_aegis_profile_v2",
            "resource_budget_version": "aimd-window-11",
        },
    }

    snapshot = build_causal_recovery_snapshot(
        artifact, failure_event_id="failure-1"
    )
    audit = audit_recovery_laxity_inputs(snapshot)

    assert audit["safe_result"] == "READY_TO_SCORE"
    assert audit["algorithm2_runnable"] is True
    assert audit["blocking_inputs"] == []
    assert all(audit["checks"].values())

    unknown_fallback = copy.deepcopy(snapshot)
    unknown_row = unknown_fallback["active_requests"][0]
    unknown_row.update(
        {
            "slack_lcb_ms": 0.0,
            "slack_target_coverage": 0.0,
            "slack_calibration_id": "",
            "slack_source": "causal_information_insufficient",
            "slack_kind": "UNKNOWN",
            "need_state": "ACTIVE",
            "slack_usable": False,
        }
    )
    unknown_audit = audit_recovery_laxity_inputs(unknown_fallback)
    assert unknown_audit["safe_result"] == "READY_TO_SCORE"
    assert unknown_audit["algorithm2_runnable"] is True

    invalid_unknown = copy.deepcopy(unknown_fallback)
    invalid_unknown["active_requests"][0]["slack_lcb_ms"] = 1.0
    invalid_unknown_audit = audit_recovery_laxity_inputs(invalid_unknown)
    assert invalid_unknown_audit["safe_result"] == "REFUSED_TO_SCORE"
    assert "causal_slack_due_date" in invalid_unknown_audit["blocking_inputs"]

    default_costs = copy.deepcopy(snapshot)
    default_costs["active_requests"][0]["target_recovery_candidates"][0][
        "recovery_service_cost_source"
    ] = "default_coefficients"
    refused = audit_recovery_laxity_inputs(default_costs)
    assert refused["safe_result"] == "REFUSED_TO_SCORE"
    assert "calibrated_recovery_service_costs" in refused["blocking_inputs"]


def test_snapshot_rejects_capacity_reservation_as_recoverable_copy() -> None:
    victim = _request("r0", "workflow-victim", "object-v")
    event = {
        "kind": "backup_ack",
        "sequence": 1,
        "time": 8.0,
        "hint_updates": [
            {
                "rid": "r0",
                "backup_key": "object-v",
                "source_dp_rank": 0,
                "target_dp_rank": 1,
                "version": 1,
                "l1_tokens": 224,
                "l2_tokens": 0,
                "l3_tokens": 0,
                "materialization_scope": "metadata_reservation_only",
            }
        ],
    }
    artifact = {
        "simulation_time_mode": "wall",
        "evidence_tier": "unit",
        "failure_events": [
            {
                "event_id": "failure-1",
                "observed_at_s": 10.0,
                "operation": "inject",
                "failure_mode": "soft",
                "dp_rank": 0,
                "accepted": True,
            }
        ],
        "scheduler_telemetry": [
            _telemetry_row(0, 9.0, victim),
            _telemetry_row(1, 9.0, _request("r1", "other", "object-b")),
        ],
        "aegis_status": {
            "event_history": {
                "mode": "unbounded_in_memory",
                "event_count": 1,
            },
            "events": [event],
        },
    }

    snapshot = build_causal_recovery_snapshot(
        artifact, failure_event_id="failure-1"
    )

    assert snapshot["recoverable_prefixes_at_cutoff"] == []
    rows = {row["rid"]: row for row in snapshot["active_requests"]}
    assert rows["r0"]["target_recovery_candidates"] == []
    assert rows["r0"]["target_prefix_observation_status"] == (
        "UNVERIFIED_MATERIALIZATION"
    )
    assert rows["r0"]["target_metadata_reservations"][0][
        "reserved_prefix_tokens"
    ] == 224
    audit = audit_snapshot_capabilities(snapshot)
    assert audit["positive_target_prefix_count"] == 0
    assert audit["metadata_reservation_count"] == 1
    assert audit["capabilities"]["target_reusable_prefix"] == (
        "UNVERIFIED_MATERIALIZATION"
    )


def test_snapshot_rejects_prefix_beyond_source_commit() -> None:
    victim = _request("r0", "workflow-victim", "object-v")
    event = {
        "kind": "backup_ack",
        "sequence": 1,
        "time": 8.0,
        "hint_updates": [
            {
                "rid": "r0",
                "backup_key": "object-v",
                "source_dp_rank": 0,
                "target_dp_rank": 1,
                "version": 1,
                "l1_tokens": 224,
                "l2_tokens": 0,
                "l3_tokens": 0,
                "materialization_scope": "committed_prefix",
                "source_committed_token_end": 7,
                "group_commit_verified": True,
            }
        ],
    }
    artifact = {
        "simulation_time_mode": "wall",
        "evidence_tier": "unit",
        "failure_events": [
            {
                "event_id": "failure-1",
                "observed_at_s": 10.0,
                "operation": "inject",
                "failure_mode": "soft",
                "dp_rank": 0,
                "accepted": True,
            }
        ],
        "scheduler_telemetry": [
            _telemetry_row(0, 9.0, victim),
            _telemetry_row(1, 9.0, _request("r1", "other", "object-b")),
        ],
        "aegis_status": {
            "event_history": {
                "mode": "unbounded_in_memory",
                "event_count": 1,
            },
            "events": [event],
        },
    }

    snapshot = build_causal_recovery_snapshot(
        artifact, failure_event_id="failure-1"
    )

    assert snapshot["recoverable_prefixes_at_cutoff"] == []
    rows = {row["rid"]: row for row in snapshot["active_requests"]}
    assert rows["r0"]["target_prefix_observation_status"] == (
        "INVALID_COMMIT_PROOF"
    )
    assert rows["r0"]["target_invalid_commit_records"] == [
        {
            "target_id": "dp:1",
            "target_dp_rank": 1,
            "backup_key": "object-v",
            "version": 1,
            "claimed_prefix_tokens": 224,
            "source_committed_token_end": 7,
            "materialization_error": "prefix_exceeds_source_commit",
        }
    ]
    audit = audit_snapshot_capabilities(snapshot)
    assert audit["invalid_commit_record_count"] == 1
    assert audit["capabilities"]["target_reusable_prefix"] == (
        "INVALID_COMMIT_PROOF"
    )


def test_snapshot_validator_rejects_future_outcome_fields() -> None:
    victim = _request("r0", "workflow-victim", "object-v")
    artifact = {
        "simulation_time_mode": "wall",
        "evidence_tier": "unit",
        "failure_events": [
            {
                "event_id": "failure-1",
                "observed_at_s": 10.0,
                "operation": "inject",
                "failure_mode": "soft",
                "dp_rank": 0,
                "accepted": True,
            }
        ],
        "scheduler_telemetry": [_telemetry_row(0, 9.0, victim)],
        "aegis_status": {"events": []},
    }
    snapshot = build_causal_recovery_snapshot(
        artifact, failure_event_id="failure-1"
    )
    snapshot["active_requests"][0]["actual_output_tokens"] = 16

    with pytest.raises(ValueError, match="forbidden future fields"):
        validate_causal_snapshot(snapshot)


def test_opt_in_scheduler_telemetry_contains_causal_request_state() -> None:
    engine = Engine(
        ServerArgs(
            streaming_arrival=True,
            model_path="mock-model",
            scheduler_telemetry=True,
            log_level="error",
        )
    )
    engine.start_background_scheduler()
    try:
        engine.submit(
            GenerateReqInput(
                rid="causal-state-row",
                input_len=8,
                max_new_tokens=4,
            )
        )
        engine.wait_for_rid("causal-state-row", timeout_s=5.0)
        payload = engine.calibration_telemetry()
        states = [
            row["causal_state"]
            for row in payload["scheduler_telemetry"]
            if "causal_state" in row
        ]
        assert states
        observed = [
            request
            for state in states
            for request in state["active_requests"]
            if request["rid"] == "causal-state-row"
        ]
        assert observed
        assert all(
            request["generated_tokens_observed"]
            <= request["requested_max_new_tokens"]
            for request in observed
        )
        assert all("finish_time_s" not in request for request in observed)
        assert all(
            state["information_scope"] == "causal_scheduler_state_after_step"
            for state in states
        )
    finally:
        engine.shutdown()


def test_engine_records_online_failure_control_timestamp() -> None:
    engine = Engine(
        ServerArgs(
            streaming_arrival=True,
            model_path="mock-model",
            dp_size=2,
            scheduler_telemetry=True,
            log_level="error",
        )
    )
    try:
        result = engine.soft_fail_dp_rank(1)
        assert result["accepted"] is True
        payload = engine.calibration_telemetry()
        assert len(payload["failure_events"]) == 1
        event = payload["failure_events"][0]
        assert event["operation"] == "inject"
        assert event["failure_mode"] == "soft"
        assert event["dp_rank"] == 1
        assert event["failure_domain"] == "dp_rank:1"
        assert event["observed_at_s"] > 0.0
        assert payload["calibration_metrics"]["failure_event_order"] == [
            event["event_id"]
        ]
    finally:
        engine.shutdown()
