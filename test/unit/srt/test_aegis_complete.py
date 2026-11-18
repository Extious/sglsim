from __future__ import annotations

from dataclasses import replace

import pytest

from sglsim.srt.simulation.aegis_complete import (
    AegisCompleteSimulator,
    CompleteFailure,
    CompleteRequest,
    CompleteSimulationConfig,
    CompleteWorker,
    audit_complete_simulation,
)
from sglsim.srt.simulation.aegis_full_design import (
    AegisFullDesignConfig,
    CalibratedTransferPath,
)
from sglsim.srt.simulation.aegis_independent_tiers import RecoveryTiming
from sglsim.srt.simulation.aegis_residual_dag import (
    RESIDUAL_DAG_SLACK_SEMANTICS,
)


def _slack_evidence(request_id: str, slack_ms: float) -> dict:
    return {
        "workflow_slack_ms": slack_ms,
        "aegis_slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
        "aegis_slack_uses_future_information": False,
        "aegis_slack_residual_dag_visible": True,
        "aegis_slack_metadata_only": True,
        "aegis_slack_usable": True,
        "aegis_slack_reason": "unit_visible_residual_dag",
        "aegis_slack_graph_version": "unit-graph-v1",
        "aegis_slack_node_id": request_id,
        "aegis_slack_agent_id": request_id,
        "aegis_slack_input_provenance_sha256": "a" * 64,
    }


def _design(
    *,
    slow_copy: bool = False,
    hbm_capacity_bytes: int = 1 << 20,
    dram_capacity_bytes: int = 1 << 22,
) -> AegisFullDesignConfig:
    return AegisFullDesignConfig(
        hbm_capacity_bytes=hbm_capacity_bytes,
        dram_capacity_bytes=dram_capacity_bytes,
        kv_bytes_per_token=1024,
        hbm_copy=CalibratedTransferPath(
            100.0 if slow_copy else 0.1,
            1024.0 if slow_copy else 10_240.0,
            "unit-hbm-copy",
        ),
        dram_copy=CalibratedTransferPath(
            100.0 if slow_copy else 0.2,
            512.0 if slow_copy else 5_120.0,
            "unit-dram-copy",
        ),
        recompute=RecoveryTiming(1.0, 0.1),
        hbm_load=RecoveryTiming(0.5, 0.01),
        dram_load=RecoveryTiming(1.0, 0.02),
        prefix_chunk_tokens=4,
        decode_chunk_tokens=2,
        event_log_limit=10_000,
    )


def _config(*, slow_copy: bool = False) -> CompleteSimulationConfig:
    return CompleteSimulationConfig(
        workers=tuple(
            CompleteWorker(f"w{index}", f"node-{index}")
            for index in range(4)
        ),
        design=_design(slow_copy=slow_copy),
        replacement_delay_ms=2.0,
        queue_outage_ms=3.0,
        repair_time_ms=20.0,
        control_window_ms=5.0,
    )


def _complete_result() -> dict:
    requests = (
        CompleteRequest(
            "r1",
            "workflow",
            0.0,
            12,
            4,
            80.0,
            20.0,
            _slack_evidence("r1", 20.0),
            "w0",
        ),
        CompleteRequest(
            "r2",
            "workflow",
            1.0,
            8,
            2,
            40.0,
            10.0,
            _slack_evidence("r2", 10.0),
            "w0",
        ),
    )
    return AegisCompleteSimulator(_config()).run(
        requests,
        failures=(
            CompleteFailure(22.0, "w1"),
            CompleteFailure(45.0, "w0"),
        ),
        max_time_ms=500.0,
    )


def test_complete_current_design_failure_path_passes_fail_closed_audit() -> None:
    result = _complete_result()
    audit = audit_complete_simulation(result)

    assert audit["status"] == "PASS"
    assert audit["failed_checks"] == []
    assert result["gates"]["complete_current_design_simulation"] is True
    assert result["gates"][
        "complete_current_design_target_failover_evaluated"
    ] is True
    assert result["physical_result"] is False
    assert all(row["status"] == "COMPLETED" for row in result["requests"].values())


@pytest.mark.parametrize(
    ("protection_mode", "expected_action"),
    (
        ("hbm_and_dram", "PEER_HBM"),
        ("hbm_only", "PEER_HBM"),
        ("dram_only", "DRAM"),
        ("none", "RECOMPUTE"),
    ),
)
def test_complete_path_supports_capacity_matched_protection_ablations(
    protection_mode: str, expected_action: str
) -> None:
    config = CompleteSimulationConfig(
        workers=tuple(
            CompleteWorker(f"w{index}", f"node-{index}") for index in range(4)
        ),
        design=replace(_design(), recompute=RecoveryTiming(5.0, 1.0)),
        protection_mode=protection_mode,
        recovery_priority="fifo",
        adaptive_hbm_quota=False,
        replacement_delay_ms=2.0,
        repair_time_ms=20.0,
        control_window_ms=5.0,
    )
    request = CompleteRequest(
        "policy",
        "workflow",
        0.0,
        12,
        4,
        80.0,
        0.0,
        _slack_evidence("policy", 0.0),
        "w0",
    )
    result = AegisCompleteSimulator(config).run(
        (request,), failures=(CompleteFailure(45.0, "w0"),), max_time_ms=500.0
    )

    assert result["requests"]["policy"]["status"] == "COMPLETED"
    assert result["requests"]["policy"]["recovery_action"] == expected_action
    assert result["policy"] == {
        "protection_mode": protection_mode,
        "recovery_priority": "fifo",
        "adaptive_hbm_quota": False,
    }


def test_explicit_prefill_phase_seals_prompt_before_short_decode_tail() -> None:
    request = CompleteRequest(
        "two-phase",
        "workflow",
        0.0,
        12,
        4,
        80.0,
        0.0,
        _slack_evidence("two-phase", 0.0),
        "w0",
        prefill_service_ms=8.0,
    )
    result = AegisCompleteSimulator(_config()).run(
        (request,), failures=(CompleteFailure(20.0, "w0"),), max_time_ms=500.0
    )
    first_seal = next(
        event
        for event in result["events"]
        if event["kind"] == "chunk_sealed_and_admitted"
    )
    fenced = next(
        event for event in result["events"] if event["kind"] == "source_failure_fenced"
    )

    assert first_seal["time_ms"] == pytest.approx(8.0 / 3.0)
    assert fenced["restored_tokens"] == 12
    assert fenced["recovery_action"] == "PEER_HBM"


def test_same_epoch_recovery_ready_cohort_is_ordered_by_remaining_slack() -> None:
    requests = tuple(
        CompleteRequest(
            f"cohort-{index}",
            f"workflow-{index}",
            0.0,
            12,
            4,
            80.0,
            slack_ms,
            _slack_evidence(f"cohort-{index}", slack_ms),
            f"w{index}",
            prefill_service_ms=8.0,
        )
        for index, slack_ms in enumerate((100.0, 50.0, 0.0))
    )
    failures = tuple(CompleteFailure(20.0, f"w{index}") for index in range(3))
    base = replace(_config(), adaptive_hbm_quota=False)

    fifo = AegisCompleteSimulator(replace(base, recovery_priority="fifo")).run(
        requests, failures=failures, max_time_ms=500.0
    )
    slack = AegisCompleteSimulator(
        replace(base, recovery_priority="remaining_slack")
    ).run(requests, failures=failures, max_time_ms=500.0)

    def selected(result: dict) -> list[tuple[str, float]]:
        return [
            (row["request_id"], row["remaining_slack_due_at_ms"])
            for row in result["events"]
            if row["kind"] == "recovery_selected_and_started"
        ]

    assert [request_id for request_id, _due in selected(fifo)] == [
        "cohort-0",
        "cohort-1",
        "cohort-2",
    ]
    assert [request_id for request_id, _due in selected(slack)] == [
        "cohort-2",
        "cohort-1",
        "cohort-0",
    ]
    assert [due for _request_id, due in selected(slack)] == sorted(
        due for _request_id, due in selected(slack)
    )


def test_target_first_rotation_uses_new_real_pool_and_preserves_route_and_dram() -> None:
    result = _complete_result()
    rotation = next(
        row
        for row in result["events"]
        if row["kind"] == "standby_target_rotated" and row["request_id"] == "r1"
    )
    reprotected = next(
        row
        for row in result["events"]
        if row["kind"] == "chunk_reprotected"
        and row["request_id"] == "r1"
        and row["target_worker"] == rotation["replacement_target"]
    )

    assert rotation["route_version_changed"] is False
    assert rotation["route_version_before"] == rotation["route_version"] == 1
    assert rotation["dram_preserved"] is True
    assert rotation["replacement_uses_real_target_pool"] is True
    assert reprotected["independent_tier_outcomes"][0]["pool_id"] == (
        f"peer-hbm-{rotation['replacement_target']}"
    )
    assert reprotected["hbm_to_dram"] is False


def test_source_failure_fences_output_then_rebinds_contiguous_route_version() -> None:
    result = _complete_result()
    fenced = next(
        row
        for row in result["events"]
        if row["kind"] == "source_failure_fenced" and row["request_id"] == "r1"
    )
    activated = next(
        row
        for row in result["events"]
        if row["kind"] == "recovery_completed_route_activated"
        and row["request_id"] == "r1"
    )

    assert fenced["output_fenced"] is True
    assert fenced["hbm_pending_aborted"] >= 0
    assert fenced["dram_pending_aborted"] >= 0
    assert fenced["actual_load_bytes"] == (
        fenced["restored_tokens"] * _design().kv_bytes_per_token
    )
    assert activated["route_version_contiguous"] is True
    assert activated["route_version"] == fenced["next_route_version"]
    assert activated["silent_replay_completed"] is True
    assert activated["all_shards_completed"] is True
    assert activated["output_fenced"] is False


def test_failed_source_queue_has_explicit_outage_and_rescheduling() -> None:
    result = _complete_result()
    started = next(
        row
        for row in result["events"]
        if row["kind"] == "queue_outage_started" and row["request_id"] == "r2"
    )
    resumed = next(
        row
        for row in result["events"]
        if row["kind"] == "queue_outage_rescheduled" and row["request_id"] == "r2"
    )

    assert resumed["time_ms"] - started["time_ms"] == 3.0
    assert resumed["replacement_source"] != started["failed_source"]
    assert result["requests"]["r2"]["redispatch_count"] == 1
    assert result["requests"]["r2"]["outage_delay_ms"] == 3.0


def test_pending_copies_never_contribute_recovery_value() -> None:
    request = CompleteRequest(
        "pending",
        "workflow",
        0.0,
        12,
        4,
        80.0,
        0.0,
        _slack_evidence("pending", 0.0),
        "w0",
    )
    result = AegisCompleteSimulator(_config(slow_copy=True)).run(
        (request,),
        failures=(CompleteFailure(25.0, "w0"),),
        max_time_ms=500.0,
    )
    fenced = next(
        row for row in result["events"] if row["kind"] == "source_failure_fenced"
    )

    assert fenced["hbm_pending_aborted"] > 0
    assert fenced["dram_pending_aborted"] > 0
    assert fenced["recovery_action"] == "RECOMPUTE"
    assert fenced["restored_tokens"] == 0


def test_unproven_positive_slack_fails_closed_without_copy_movement() -> None:
    request = CompleteRequest(
        "unsafe-slack",
        "workflow",
        0.0,
        4,
        0,
        10.0,
        999.0,
        {},
        "w0",
    )
    result = AegisCompleteSimulator(_config()).run((request,), max_time_ms=100.0)
    arrived = next(
        row
        for row in result["events"]
        if row["kind"] == "request_arrived_and_route_bound"
    )

    assert arrived["causal_slack_ms"] == 0.0
    assert arrived["slack_metadata_only"] is True
    assert arrived["slack_reason"].startswith("missing_or_invalid_")


def test_recovery_target_can_fail_and_rebind_while_output_remains_fenced() -> None:
    request = CompleteRequest(
        "nested-fault",
        "workflow",
        0.0,
        12,
        4,
        80.0,
        0.0,
        _slack_evidence("nested-fault", 0.0),
        "w0",
    )
    result = AegisCompleteSimulator(_config()).run(
        (request,),
        failures=(
            CompleteFailure(22.0, "w1"),
            CompleteFailure(45.0, "w0"),
            CompleteFailure(47.5, "w2"),
        ),
        max_time_ms=500.0,
    )
    rebound = next(
        row
        for row in result["events"]
        if row["kind"] == "recovery_target_failed_and_rebound"
    )

    assert rebound["output_fenced"] is True
    assert rebound["replacement_recovery_worker"] != rebound["failed_recovery_worker"]
    assert result["requests"]["nested-fault"]["status"] == "COMPLETED"
    assert result["requests"]["nested-fault"]["output_fenced"] is False


@pytest.mark.parametrize(
    ("hbm_capacity", "dram_capacity", "expected"),
    (
        (1 << 20, 1 << 20, {"HBM": True, "DRAM": True}),
        (1 << 20, 2_048, {"HBM": True, "DRAM": False}),
        (2_048, 1 << 20, {"HBM": False, "DRAM": True}),
        (2_048, 2_048, {"HBM": False, "DRAM": False}),
    ),
)
def test_unified_path_preserves_all_four_independent_residency_outcomes(
    hbm_capacity: int,
    dram_capacity: int,
    expected: dict[str, bool],
) -> None:
    config = CompleteSimulationConfig(
        workers=tuple(
            CompleteWorker(f"w{index}", f"node-{index}")
            for index in range(2)
        ),
        design=_design(
            hbm_capacity_bytes=hbm_capacity,
            dram_capacity_bytes=dram_capacity,
        ),
        control_window_ms=5.0,
    )
    request = CompleteRequest(
        "residency",
        "workflow",
        0.0,
        4,
        0,
        10.0,
        0.0,
        _slack_evidence("residency", 0.0),
        "w0",
    )
    result = AegisCompleteSimulator(config).run((request,), max_time_ms=100.0)
    decision = next(
        row
        for row in result["events"]
        if row["kind"] == "chunk_sealed_and_admitted"
    )
    observed = {
        row["tier"]: row["admitted"]
        for row in decision["independent_tier_outcomes"]
    }

    assert observed == expected
