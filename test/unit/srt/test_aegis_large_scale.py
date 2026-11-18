from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sglsim.srt.simulation import aegis_large_scale as large
from sglsim.srt.simulation.aegis_large_scale import (
    ALPHA_DRIVER_CONTRACT,
    AlgorithmToggles,
    LargeScaleCellConfig,
    RequestInput,
    run_cell,
)


def _workload(count: int, *, prompt: int = 4_096) -> list[RequestInput]:
    return [
        RequestInput(
            request_id=f"r-{index}",
            workflow_id=f"w-{index // 4}",
            prompt_tokens=prompt,
            output_tokens=256,
            slack_ms=float((index % 4) * 50),
            prefix_group=f"p-{index}",
            arrival_ms=0.0,
            input_reuse_tokens=0,
        )
        for index in range(count)
    ]


def test_protocol_compute_profiles_are_the_runtime_source_of_truth() -> None:
    eight = LargeScaleCellConfig(model="qwen3_8b").model_profile
    thirty_two = LargeScaleCellConfig(model="qwen3_32b", dp_size=16).model_profile

    assert eight.prefill_setup_ms == pytest.approx(4.0)
    assert eight.prefill_ms_per_token == pytest.approx(0.060)
    assert eight.decode_ms_per_token == pytest.approx(0.34)
    assert eight.kv_bytes_per_token == 147_456
    assert eight.pp_size == 1
    assert thirty_two.prefill_setup_ms == pytest.approx(9.0)
    assert thirty_two.prefill_ms_per_token == pytest.approx(0.160)
    assert thirty_two.decode_ms_per_token == pytest.approx(0.92)
    assert thirty_two.kv_bytes_per_token == 262_144
    assert thirty_two.pp_size == 4

    protocol_path = (
        Path(__file__).resolve().parents[3]
        / "AegisKV/experiments/sglsim_large_scale_v2/protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for protocol_key, profile in (
        ("qwen3_8b", eight),
        ("qwen3_32b", thirty_two),
    ):
        declared = protocol["models"][protocol_key]["simulated_compute_profile"]
        assert profile.prefill_setup_ms == declared["prefill_setup_ms"]
        assert profile.prefill_ms_per_token == declared["prefill_ms_per_token"]
        assert profile.decode_ms_per_token == declared["decode_ms_per_token"]


def test_all_eight_algorithm_compositions_are_distinct() -> None:
    observed = {}
    for bits in (f"{value:03b}" for value in range(8)):
        _label, toggles, _kind = large._resolve_policy(
            f"aegiskv_alg{bits}", None
        )
        observed[bits] = (
            toggles.algorithm1,
            toggles.algorithm2,
            toggles.algorithm3,
        )

    assert len(set(observed.values())) == 8
    assert observed["000"] == (False, False, False)
    assert observed["111"] == (True, True, True)
    assert AlgorithmToggles.from_value("none") == AlgorithmToggles(
        False, False, False
    )


@pytest.mark.parametrize("bits", [f"{value:03b}" for value in range(8)])
def test_all_eight_algorithm_compositions_execute_declared_effects(bits: str) -> None:
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=2,
            request_count=2,
            workflow_count=1,
            control_window_ms=100.0,
            failure_rate_per_1000_worker_hours=0.0,
        ),
        9,
        f"aegiskv_alg{bits}",
        workload=_workload(2),
    )

    assert tuple(result["algorithms"].values()) == tuple(
        value == "1" for value in bits
    )
    assert result["algorithm_effects"]["algorithm2_admission"] == (
        "slack_landlord_independent_hbm_dram"
        if bits[1] == "1"
        else "static_independent_hbm_dram_fcfs"
    )
    assert result["algorithm_effects"]["algorithm3_alpha"] == (
        "target_local_pressure_only_aimd"
        if bits[2] == "1"
        else "fixed_configured_alpha"
    )


@pytest.mark.parametrize(
    "policy",
    [*large.BASELINE_POLICIES, "static_independent_fcfs", "recompute_only"],
)
def test_baseline_and_static_smoke_use_total_tokens_property(policy: str) -> None:
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=2,
            request_count=2,
            workflow_count=1,
            control_window_ms=100.0,
            failure_rate_per_1000_worker_hours=0.0,
        ),
        2,
        policy,
        workload=_workload(2),
    )

    assert result["summary"]["request_count"] == 2
    assert result["summary"]["copy_amplification"] >= 0.0
    if policy == "recompute_only":
        assert result["summary"]["protection_copy_attempted_bytes"] == 0
    else:
        assert result["summary"]["protection_copy_attempted_bytes"] > 0


def test_measurement_window_tags_telemetry_but_does_not_clip_fault_process() -> None:
    base = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=2,
        request_count=2,
        workflow_count=1,
        control_window_ms=100.0,
        failure_rate_per_1000_worker_hours=1_000_000.0,
    )
    tagged = replace(base, measurement_start_ms=100.0, measurement_end_ms=200.0)

    assert large._generate_faults(
        base, horizon_ms=1_000.0, seed=3
    ) == large._generate_faults(tagged, horizon_ms=1_000.0, seed=3)
    result = run_cell(tagged, 3, workload=_workload(2))
    assert result["gates"]["fault_process_covers_full_run_horizon"] is True
    assert result["config"]["resolved_measurement_window"][
        "fault_process_clipped_to_window"
    ] is False
    assert result["summary"]["measurement_window_sample_count"] == sum(
        row["in_measurement_window"] for row in result["window_rows"]
    )


def test_realized_workflow_concurrency_is_computed_from_join_intervals() -> None:
    rows = [
        {
            "arrival_ms": 0.0,
            "base_join_completion_ms": 10.0,
            "simulated_join_completion_ms": 12.0,
        },
        {
            "arrival_ms": 5.0,
            "base_join_completion_ms": 15.0,
            "simulated_join_completion_ms": 15.0,
        },
    ]

    base = large._active_interval_stats(
        rows, finish_key="base_join_completion_ms"
    )
    simulated = large._active_interval_stats(
        rows, finish_key="simulated_join_completion_ms"
    )

    assert base == {
        "mean": pytest.approx(20.0 / 15.0),
        "p50": 1.0,
        "p95": 2.0,
        "peak": 2,
        "duration_ms": 15.0,
    }
    assert simulated["mean"] == pytest.approx(22.0 / 15.0)
    assert simulated["peak"] == 2


def test_actual_slack_is_ground_truth_and_policy_slack_only_drives_decision() -> None:
    request = RequestInput(
        request_id="different-slacks",
        workflow_id="workflow",
        prompt_tokens=8_192,
        output_tokens=256,
        slack_ms=1_000_000.0,
        actual_slack_ms=0.0,
        prefix_group="prefix",
        arrival_ms=0.0,
        input_reuse_tokens=0,
    )
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=1,
            request_count=1,
            workflow_count=1,
            failure_rate_per_1000_worker_hours=0.0,
            minimum_faults_per_cell=1,
            control_window_ms=100.0,
        ),
        4,
        workload=[request],
    )
    failure = result["failure_rows"][0]
    workflow = result["workflow_rows"][0]

    assert failure["policy_remaining_slack_ms"] > failure["recovery_ms"]
    assert failure["policy_modeled_visible_harm_ms"] == 0.0
    assert failure["actual_remaining_slack_ms"] == 0.0
    assert failure["actual_modeled_visible_harm_ms"] == pytest.approx(
        failure["recovery_ms"]
    )
    assert workflow["failure_added_workflow_wct_ms"] == pytest.approx(
        failure["actual_modeled_visible_harm_ms"]
    )
    assert workflow["simulated_workflow_wct_ms"] == pytest.approx(
        workflow["base_workflow_wct_ms"]
        + failure["actual_modeled_visible_harm_ms"]
    )
    assert result["gates"]["end_to_end_dag_failure_execution_replay"] is False
    summary = result["summary"]
    assert summary[
        "actual_failure_added_workflow_harm_surrogate_mean_ms"
    ] == pytest.approx(summary["failure_added_workflow_wct_mean_ms"])
    assert summary[
        "actual_failure_added_workflow_harm_surrogate_p95_ms"
    ] == pytest.approx(summary["failure_added_workflow_wct_p95_ms"])
    assert summary["legacy_metric_aliases_forbidden_for_formal_claims"] is True
    assert result["formal_metric_contract"][
        "primary_failure_endpoint_prefix"
    ] == "actual_failure_added_workflow_harm_surrogate_"


def test_source_failure_cell_rejects_compound_target_first_probability() -> None:
    with pytest.raises(ValueError, match="target_failure_probability=0"):
        LargeScaleCellConfig(target_failure_probability=0.2)


def test_paired_fault_schedule_is_policy_independent() -> None:
    config = LargeScaleCellConfig(
        dp_size=4,
        gpus_per_node=4,
        concurrency=4,
        request_count=8,
        workflow_count=2,
        failure_rate_per_1000_worker_hours=1_000_000.0,
        control_window_ms=100.0,
    )
    workload = _workload(8, prompt=8_192)
    aegis = run_cell(config, 13, "aegiskv", workload=workload)
    baseline = run_cell(config, 13, "recompute_only", workload=workload)

    identity = lambda row: (
        row["time_ms"],
        row["source_rank"],
        row["shard_rank"],
        row["forced_floor_event"],
    )
    assert list(map(identity, aegis["failure_rows"])) == list(
        map(identity, baseline["failure_rows"])
    )
    assert all(
        row["compound_target_first_failure_injected"] is False
        for row in aegis["failure_rows"]
    )
    assert all(
        row["forced_synthetic_target_first_injection"] is False
        for row in aegis["failure_rows"]
    )
    assert aegis["gates"][
        "forced_synthetic_target_first_injection_disabled"
    ]
    assert "compound_target_first_failure_disabled" not in aegis["gates"]
    assert aegis["gates"]["standby_rotation_modeled"] is False
    assert aegis["gates"][
        "complete_current_design_target_failover_evaluated"
    ] is False
    assert aegis["gates"][
        "target_failure_history_is_no_standby_rotation_conservative_sensitivity"
    ] is True


@pytest.mark.parametrize(
    "policy", ["aegiskv", "dejavu_neighbor", "static_independent_fcfs"]
)
def test_worker_fail_stop_invalidates_colocated_peer_hbm_without_target_first(
    policy: str,
) -> None:
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=2,
            request_count=2,
            workflow_count=1,
            failure_rate_per_1000_worker_hours=0.0,
            minimum_faults_per_cell=1,
            control_window_ms=100.0,
        ),
        4,
        policy,
        workload=_workload(2, prompt=8_192),
    )
    failure = result["failure_rows"][0]

    assert failure["compound_target_first_failure_injected"] is False
    assert failure["invalidated_hbm_requests"] >= 1
    assert failure["invalidated_hbm_bytes"] > 0


@pytest.mark.parametrize(
    "policy", ["aegiskv", "dejavu_neighbor", "static_independent_fcfs"]
)
def test_repair_event_reenables_capacity_without_restoring_old_copies(
    policy: str,
) -> None:
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=2,
            request_count=2,
            workflow_count=1,
            failure_rate_per_1000_worker_hours=0.0,
            minimum_faults_per_cell=1,
            repair_time_ms=50.0,
            control_window_ms=25.0,
        ),
        4,
        policy,
        workload=_workload(2, prompt=8_192),
    )
    repairs = result["repair_rows"]

    assert len(repairs) == 1
    assert repairs[0]["future_hbm_admission_reenabled"] is True
    assert repairs[0]["old_hbm_copies_restored"] is False
    assert repairs[0]["post_repair_hbm_reprotection_scheduled"] is False
    assert any(row["failed_hbm_target_count"] == 1 for row in result["window_rows"])
    assert result["window_rows"][-1]["failed_hbm_target_count"] == 0
    assert result["window_rows"][-1][
        "logical_replica_targets_ever_failed_count"
    ] == 1
    assert result["summary"]["worker_repair_events_effective"] == 1
    history = result["final_state"]["failure_history"]
    assert history["current_logical_replica_target_outage_count"] == 0
    assert history["logical_replica_targets_ever_failed_count"] == 1
    assert history["physical_gpu_shards_ever_failed_count"] == 1
    assert history[
        "repair_clears_current_outage_not_ever_failed_history"
    ] is True


def test_horizon_outage_state_is_frozen_before_post_horizon_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def faults(*_args: object, **_kwargs: object) -> tuple[large._FaultEvent, ...]:
        return (large._FaultEvent(100.0, source_rank=0, shard_rank=0),)

    monkeypatch.setattr(large, "_generate_faults", faults)
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=1,
            request_count=1,
            workflow_count=1,
            failure_rate_per_1000_worker_hours=1.0,
            repair_time_ms=60_000.0,
            control_window_ms=100.0,
        ),
        4,
        workload=_workload(1, prompt=8_192),
    )

    summary = result["summary"]
    horizon = result["failure_state_at_horizon"]
    post_all = result["final_state"]["failure_history"]
    repair = result["repair_rows"][0]

    assert repair["time_ms"] > summary["base_run_horizon_ms"]
    assert repair["after_base_run_horizon"] is True
    assert summary["repair_events_after_horizon"] == 1
    assert summary["post_horizon_event_count"] >= 1
    assert horizon["snapshot_scope"] == (
        "base_run_horizon_after_all_events_at_or_before_horizon"
    )
    assert horizon["snapshot_time_ms"] == pytest.approx(
        summary["base_run_horizon_ms"]
    )
    assert horizon["current_physical_gpu_shard_outage_count"] == 1
    assert horizon["current_logical_replica_target_outage_count"] == 1
    assert summary["current_physical_gpu_shard_outage_count_at_horizon"] == 1
    assert summary[
        "current_logical_replica_target_outage_count_at_horizon"
    ] == 1

    assert post_all["snapshot_scope"] == (
        "post_all_scheduled_events_including_repairs_after_horizon"
    )
    assert post_all["snapshot_time_ms"] == pytest.approx(
        summary["post_all_events_last_event_time_ms"]
    )
    assert post_all["current_physical_gpu_shard_outage_count"] == 0
    assert post_all["current_logical_replica_target_outage_count"] == 0
    assert summary[
        "current_physical_gpu_shard_outage_count_post_all_events"
    ] == 0
    assert summary[
        "current_logical_replica_target_outage_count_post_all_events"
    ] == 0
    for field in (
        "physical_gpu_shards_ever_failed_count",
        "logical_replica_targets_ever_failed_count",
        "old_hbm_copy_invalidation_event_count_cumulative",
        "old_hbm_copy_invalidated_requests_cumulative",
        "old_hbm_copy_invalidated_bytes_cumulative",
    ):
        assert horizon[field] == post_all[field]

    history_contract = result["formal_metric_contract"][
        "failure_history_contract"
    ]
    assert history_contract["horizon_state_source"] == (
        "failure_state_at_horizon"
    )
    assert history_contract["post_all_events_state_source"] == (
        "final_state.failure_history"
    )
    assert history_contract[
        "events_after_horizon_are_cleanup_not_horizon_state"
    ] is True
    assert result["gates"][
        "summary_at_horizon_fields_derive_from_horizon_failure_state"
    ] is True
    assert result["gates"][
        "final_state_is_post_all_events_not_horizon_state"
    ] is True


def test_pp4_physical_shard_history_is_distinct_from_logical_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def faults(*_args: object, **_kwargs: object) -> tuple[large._FaultEvent, ...]:
        return (
            large._FaultEvent(50.0, source_rank=0, shard_rank=0),
            large._FaultEvent(60.0, source_rank=0, shard_rank=1),
        )

    monkeypatch.setattr(large, "_generate_faults", faults)
    result = run_cell(
        LargeScaleCellConfig(
            model="qwen3-32b",
            dp_size=2,
            gpus_per_node=4,
            concurrency=2,
            request_count=2,
            workflow_count=1,
            failure_rate_per_1000_worker_hours=1.0,
            repair_time_ms=100.0,
            control_window_ms=25.0,
        ),
        4,
        workload=_workload(2, prompt=8_192),
    )
    first, second = result["failure_rows"]

    assert first["physical_gpu_shard_ever_failed_before_event"] is False
    assert second["physical_gpu_shard_ever_failed_before_event"] is False
    assert second[
        "logical_replica_target_currently_unavailable_before_event"
    ] is True
    assert second["current_physical_gpu_shard_outage_count_after_event"] == 2
    assert second[
        "current_logical_replica_target_outage_count_after_event"
    ] == 1
    history = result["final_state"]["failure_history"]
    assert history["physical_gpu_shards_ever_failed_count"] == 2
    assert history["logical_replica_targets_ever_failed_count"] == 1
    assert history["current_physical_gpu_shard_outage_count"] == 0
    assert history["current_logical_replica_target_outage_count"] == 0
    assert result["summary"][
        "failure_events_on_previously_failed_physical_gpu_shard"
    ] == 0


def test_aegis_repair_allows_a_new_request_hbm_without_resurrecting_old_copy() -> None:
    config = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=2,
        request_count=2,
        workflow_count=1,
        failure_rate_per_1000_worker_hours=0.0,
    )
    workload = [
        RequestInput(
            request_id="old",
            workflow_id="w",
            prompt_tokens=4_096,
            output_tokens=256,
            slack_ms=0.0,
            actual_slack_ms=0.0,
            prefix_group="same-source",
            arrival_ms=0.0,
            input_reuse_tokens=0,
        ),
        RequestInput(
            request_id="new",
            workflow_id="w",
            prompt_tokens=4_096,
            output_tokens=256,
            slack_ms=0.0,
            actual_slack_ms=0.0,
            prefix_group="same-source",
            arrival_ms=70.0,
            input_reuse_tokens=0,
        ),
    ]
    old, new = large._schedule_requests(config, workload)
    adapter = large._AegisAdapter(config, AlgorithmToggles())
    adapter.register(old, slack_enabled=True)
    adapter.seal(old, old.chunks[0], 0.0)
    target = old.target_rank
    adapter.fail_worker(target, 1.0)

    assert (
        adapter.runtimes[target].controller.copy_state(
            "old", 0, large.ProtectionTier.HBM
        )
        is None
    )
    assert adapter.repair_worker(target, 51.0)[
        "future_hbm_admission_reenabled"
    ] is True
    adapter.register(new, slack_enabled=True)
    adapter.seal(new, new.chunks[0], 70.0)
    assert adapter.runtimes[target].controller.copy_state(
        "new", 0, large.ProtectionTier.HBM
    ).value == "PENDING"


def test_quota_occupancy_pressure_threshold_is_configurable() -> None:
    base = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        failure_rate_per_1000_worker_hours=0.0,
        pressure_queue_depth=999,
        control_window_ms=100.0,
        hbm_capacity_gib_per_target=4.0,
    )
    workload = _workload(1, prompt=8_192)
    low = run_cell(
        replace(base, hbm_quota_pressure_fraction=0.0001),
        6,
        workload=workload,
    )
    high = run_cell(
        replace(base, hbm_quota_pressure_fraction=1.0),
        6,
        workload=workload,
    )

    assert max(row["pressure_targets"] for row in low["window_rows"]) > max(
        row["pressure_targets"] for row in high["window_rows"]
    )
    assert low["config"]["resolved_pressure_signals"] == {
        "queue_depth_threshold": 999,
        "hbm_quota_occupancy_fraction_threshold": 0.0001,
        "failure_rate_is_direct_input": False,
    }


def test_cell_is_deterministic_and_emits_workflow_not_request_harm_as_wct() -> None:
    config = LargeScaleCellConfig(
        dp_size=4,
        gpus_per_node=4,
        concurrency=8,
        request_count=24,
        workflow_count=6,
        control_window_ms=500.0,
        failure_rate_per_1000_worker_hours=0.0,
    )

    first = run_cell(config, 17, "aegiskv")
    second = run_cell(config, 17, "aegiskv")

    assert first == second
    assert first["evidence_class"] == "simulated_sglsim_only"
    assert len(first["workflow_rows"]) == 6
    assert "simulated_workflow_wct_mean_ms" in first["summary"]
    assert "modeled_visible_harm_mean_ms" in first["summary"]
    assert first["gates"]["full_pinned_sglang_scheduler_replay"] is False


def test_exact_trace_reuse_override_reduces_compute_but_not_copy_bytes() -> None:
    workload = [
        RequestInput(
            request_id="no-reuse",
            workflow_id="w",
            prompt_tokens=4_096,
            output_tokens=256,
            slack_ms=0.0,
            prefix_group="same",
            arrival_ms=0.0,
            input_reuse_tokens=0,
        ),
        RequestInput(
            request_id="exact-reuse",
            workflow_id="w",
            prompt_tokens=4_096,
            output_tokens=256,
            slack_ms=0.0,
            prefix_group="same",
            arrival_ms=1.0,
            input_reuse_tokens=2_048,
        ),
    ]
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=2,
            request_count=2,
            workflow_count=1,
            failure_rate_per_1000_worker_hours=0.0,
            control_window_ms=100.0,
        ),
        11,
        "aegiskv",
        workload=workload,
    )
    rows = {row["request_id"]: row for row in result["request_rows"]}

    assert rows["no-reuse"]["input_reuse_tokens"] == 0
    assert rows["exact-reuse"]["input_reuse_tokens"] == 2_048
    assert rows["exact-reuse"]["prefill_ms"] < rows["no-reuse"]["prefill_ms"]
    assert result["summary"]["input_reuse_tokens"] == 2_048
    assert result["summary"]["protection_copy_deduplicated_by_input_reuse"] is False
    assert all(
        row["protection_copy_deduplicated_by_reuse"] is False
        for row in result["request_rows"]
    )


def test_algorithm1_error_multiplier_interpolates_actual_and_predicted_slack() -> None:
    request = RequestInput(
        request_id="slack",
        workflow_id="w",
        prompt_tokens=4_096,
        output_tokens=256,
        slack_ms=300.0,
        actual_slack_ms=100.0,
        prefix_group="p",
        arrival_ms=0.0,
        input_reuse_tokens=0,
    )
    base = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        failure_rate_per_1000_worker_hours=0.0,
        control_window_ms=100.0,
    )

    oracle = run_cell(
        replace(base, algorithm1_prediction_error_multiplier=0.0),
        1,
        workload=[request],
    )
    central = run_cell(base, 1, workload=[request])
    doubled = run_cell(
        replace(base, algorithm1_prediction_error_multiplier=2.0),
        1,
        workload=[request],
    )

    assert oracle["request_rows"][0]["policy_slack_ms"] == 100.0
    assert central["request_rows"][0]["policy_slack_ms"] == 300.0
    assert doubled["request_rows"][0]["policy_slack_ms"] == 500.0
    assert oracle["request_rows"][0]["predicted_policy_slack_ms"] == 300.0
    assert central["request_rows"][0]["effective_policy_slack_ms"] == 300.0
    assert central["summary"]["predicted_policy_slack_nonzero_count"] == 1
    assert central["summary"]["effective_policy_slack_nonzero_count"] == 1
    assert central["summary"]["algorithm1_mechanism_state"] == (
        "enabled_nonzero_effective_slack"
    )
    assert central["gates"][
        "algorithm1_policy_input_is_request_slack_when_multiplier_one"
    ]


def test_preapplied_policy_slack_is_not_interpolated_twice() -> None:
    request = RequestInput(
        request_id="preapplied-slack",
        workflow_id="w",
        prompt_tokens=4_096,
        output_tokens=256,
        slack_ms=50.0,
        actual_slack_ms=10.0,
        predicted_slack_ms=30.0,
        policy_slack_multiplier_applied=2.0,
        prefix_group="p",
        arrival_ms=0.0,
        input_reuse_tokens=0,
    )
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=1,
            request_count=1,
            workflow_count=1,
            algorithm1_prediction_error_multiplier=2.0,
            failure_rate_per_1000_worker_hours=0.0,
            control_window_ms=100.0,
        ),
        1,
        workload=[request],
    )
    row = result["request_rows"][0]
    assert row["predicted_policy_slack_ms"] == 30.0
    assert row["caller_supplied_policy_slack_ms"] == 50.0
    assert row["policy_slack_ms"] == 50.0
    assert row["policy_slack_multiplier_preapplied"] is True
    assert result["summary"]["predicted_policy_slack_nonzero_count"] == 1
    assert result["summary"]["requests_with_preapplied_policy_slack_multiplier"] == 1
    assert result["gates"][
        "algorithm1_preapplied_multiplier_matches_cell_config"
    ]

    with pytest.raises(ValueError, match="does not match"):
        run_cell(
            LargeScaleCellConfig(
                dp_size=2,
                gpus_per_node=2,
                concurrency=1,
                request_count=1,
                workflow_count=1,
                algorithm1_prediction_error_multiplier=1.0,
                failure_rate_per_1000_worker_hours=0.0,
            ),
            1,
            workload=[request],
        )


def test_algorithm1_full_and_disabled_states_change_admission_mechanism() -> None:
    request = RequestInput(
        request_id="alg1-mechanism",
        workflow_id="w",
        prompt_tokens=8_192,
        output_tokens=512,
        slack_ms=1e9,
        actual_slack_ms=1e9,
        prefix_group="p",
        arrival_ms=0.0,
        input_reuse_tokens=0,
    )
    config = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        hbm_capacity_gib_per_target=1.0,
        dram_capacity_gib_per_target=1.0,
        failure_rate_per_1000_worker_hours=0.0,
        control_window_ms=100.0,
    )

    full = run_cell(config, 1, "aegiskv_alg111", workload=[request])
    disabled = run_cell(
        config, 1, "aegiskv_alg011_no_alg1", workload=[request]
    )

    assert full["summary"]["algorithm1_mechanism_state"] == (
        "enabled_nonzero_effective_slack"
    )
    assert disabled["summary"]["algorithm1_mechanism_state"] == (
        "disabled_zero_effective_slack"
    )
    assert disabled["request_rows"][0]["effective_policy_slack_ms"] == 0.0
    assert full["summary"]["admission_outcomes"] != disabled["summary"][
        "admission_outcomes"
    ]
    assert full["summary"]["protection_copy_attempted_bytes"] != disabled[
        "summary"
    ]["protection_copy_attempted_bytes"]


def test_alpha_is_pressure_only_and_fixed_pressure_ignores_failure_metadata() -> None:
    base = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=2,
        request_count=2,
        workflow_count=1,
        control_window_ms=100.0,
        failure_rate_per_1000_worker_hours=0.0,
    )
    workload = _workload(2)
    no_fault_rate = run_cell(base, 5, "aegiskv", workload=workload)
    metadata_rate = run_cell(
        replace(base, failure_rate_per_1000_worker_hours=41.667),
        5,
        "aegiskv",
        workload=workload,
    )

    assert not no_fault_rate["failure_rows"]
    assert not metadata_rate["failure_rows"]
    assert [row["controller_alpha_mean"] for row in no_fault_rate["window_rows"]] == [
        row["controller_alpha_mean"] for row in metadata_rate["window_rows"]
    ]
    assert all(
        row["alpha_driver_contract"] == ALPHA_DRIVER_CONTRACT
        and row["failure_rate_is_direct_alpha_input"] is False
        for row in metadata_rate["window_rows"]
    )


def test_t0_window_is_snapshot_only_and_first_aimd_update_is_periodic() -> None:
    workload = [
        RequestInput(
            request_id="initial-sample",
            workflow_id="w0",
            prompt_tokens=4_096,
            output_tokens=256,
            slack_ms=0.0,
            actual_slack_ms=0.0,
            prefix_group="w0",
            arrival_ms=0.0,
            input_reuse_tokens=0,
        ),
        RequestInput(
            request_id="extends-horizon",
            workflow_id="w1",
            prompt_tokens=4_096,
            output_tokens=256,
            slack_ms=0.0,
            actual_slack_ms=0.0,
            prefix_group="w1",
            arrival_ms=31_000.0,
            input_reuse_tokens=0,
        ),
    ]
    result = run_cell(
        LargeScaleCellConfig(
            dp_size=2,
            gpus_per_node=2,
            concurrency=1,
            request_count=2,
            workflow_count=2,
            failure_rate_per_1000_worker_hours=0.0,
            control_window_ms=30_000.0,
            initial_hbm_alpha=0.5,
            aimd_stable_windows=3,
        ),
        1,
        workload=workload,
    )
    initial = result["window_rows"][0]
    assert initial["time_ms"] == 0.0
    assert initial["initial_sample"] is True
    assert initial["aimd_update_applied"] is False
    assert initial["controller_alpha_mean"] == 0.5
    assert initial["controller_alpha_min"] == 0.5
    assert initial["controller_alpha_max"] == 0.5
    assert initial["controller_stable_windows_mean"] == 0.0
    assert initial["controller_stable_windows_min"] == 0
    assert initial["controller_stable_windows_max"] == 0

    updates = [
        row for row in result["window_rows"] if row["aimd_update_applied"]
    ]
    assert updates[0]["time_ms"] == 30_000.0
    assert updates[0]["initial_sample"] is False
    assert updates[0]["controller_stable_windows_mean"] == 1.0


def test_high_pressure_reaches_operational_zero_then_stable_windows_recover() -> None:
    config = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=20,
        request_count=20,
        workflow_count=5,
        control_window_ms=100.0,
        hbm_capacity_gib_per_target=1.0,
        failure_rate_per_1000_worker_hours=0.0,
        initial_hbm_alpha=0.5,
        minimum_hbm_alpha=0.0,
        aimd_decrease_factor=0.5,
        aimd_increase_step=0.05,
        aimd_stable_windows=3,
        pressure_queue_depth=2,
    )
    result = run_cell(config, 7, "aegiskv", workload=_workload(20, prompt=8_192))
    effective = [row["effective_alpha_mean"] for row in result["window_rows"]]
    zero = next(index for index, value in enumerate(effective) if value == 0.0)

    assert result["summary"]["effective_alpha_min"] == 0.0
    assert any(value > 0.0 for value in effective[zero + 1 :])
    assert min(row["controller_alpha_min"] for row in result["window_rows"]) >= 0.0


def test_dp_model_and_bandwidth_are_operational_not_labels() -> None:
    config = LargeScaleCellConfig(
        model="qwen3_32b",
        dp_size=16,
        gpus_per_node=4,
        concurrency=32,
        request_count=8,
        workflow_count=2,
        hbm_copy_bandwidth_gbps=10.0,
        failure_rate_per_1000_worker_hours=0.0,
    )
    result = run_cell(config, 3, "aegiskv", workload=_workload(8))
    fast = large._full_design_config(config)
    slow = large._full_design_config(
        replace(config, hbm_copy_bandwidth_gbps=1.0)
    )

    assert result["config"]["system_dp_size"] == 16
    assert result["config"]["total_gpu_count"] == 64
    assert result["config"]["node_count"] == 16
    assert result["config"]["model_profile"]["pp_size"] == 4
    assert result["config"]["resolved_capacity_accounting"][
        "logical_dp_target_shard_count"
    ] == 4
    assert result["config"]["resolved_capacity_accounting"][
        "aggregate_hbm_capacity_bytes_per_logical_dp_target"
    ] == large._gib(config.hbm_capacity_gib_per_target) * 4
    assert fast.shard_count == 4
    assert fast.kv_bytes_per_token == 262_144
    assert slow.hbm_copy.duration_ms(1_000_000) > fast.hbm_copy.duration_ms(
        1_000_000
    )


def test_peer_binding_prefers_local_8b_and_uses_network_for_32b() -> None:
    eight = LargeScaleCellConfig(
        model="qwen3_8b",
        dp_size=8,
        gpus_per_node=4,
        request_count=1,
        workflow_count=1,
        concurrency=1,
        hbm_copy_bandwidth_gbps=25.0,
        network_effective_bandwidth_gbps=10.0,
    )
    thirty_two = replace(eight, model="qwen3_32b")

    assert [large._peer_target_rank(eight, rank) for rank in range(4)] == [
        1,
        2,
        3,
        0,
    ]
    assert large._peer_target_rank(eight, 7) == 4
    assert large._peer_source_rank(eight, 0) == 3
    assert large._peer_source_rank(eight, 4) == 7
    assert all(
        large._dp_rank_node(eight, source)
        == large._dp_rank_node(eight, large._peer_target_rank(eight, source))
        for source in range(eight.dp_size)
    )
    assert large._peer_target_rank(thirty_two, 0) == 1
    assert large._dp_rank_node(thirty_two, 0) == 0
    assert large._dp_rank_node(thirty_two, 1) == 1
    assert large._dp_rank_node(thirty_two, 0) != large._dp_rank_node(
        thirty_two, 1
    )
    local_path = large._full_design_config(eight, target_rank=1).hbm_copy
    remote_path = large._full_design_config(
        thirty_two, target_rank=1
    ).hbm_copy
    assert local_path.bandwidth_bytes_per_ms == pytest.approx(25_000_000.0)
    # Qwen3-32B has four PP stage lanes, each capped by the 10 GB/s
    # cross-node effective rate in this aggregate model.
    assert remote_path.bandwidth_bytes_per_ms == pytest.approx(40_000_000.0)

    baseline = large._BaselineAdapter(
        thirty_two, large.BaselinePolicy.DEJAVU_NEIGHBOR
    )
    static = large._StaticIndependentAdapter(thirty_two)
    assert all(
        runtime.config.hbm_copy.bandwidth_bytes_per_ms
        == pytest.approx(remote_path.bandwidth_bytes_per_ms)
        for runtime in baseline.runtimes
    )
    assert all(
        pool.copy_path.bandwidth_bytes_per_ms
        == pytest.approx(remote_path.bandwidth_bytes_per_ms)
        for pool in static.hbm
    )


@pytest.mark.parametrize(
    "policy",
    [large.BaselinePolicy.DEJAVU_NEIGHBOR, large.BaselinePolicy.KEVLARFLOW_RING],
)
def test_hbm_baseline_outage_preserves_progressive_chunk_ordinals(
    policy: large.BaselinePolicy,
) -> None:
    """An outage BYPASS is an observed chunk, not a dropped seal event.

    This is the minimal state-machine regression for the Qwen3-32B full-matrix
    failures: a target failure landed between two seals, the adapter returned
    early during the outage, and a post-repair seal then appeared out of order.
    """

    config = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=1,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        failure_rate_per_1000_worker_hours=0.0,
    )
    request_input = RequestInput(
        request_id="ordinal-regression",
        workflow_id="workflow",
        prompt_tokens=768,
        output_tokens=0,
        slack_ms=0.0,
        prefix_group="prefix",
        arrival_ms=0.0,
        input_reuse_tokens=0,
    )
    chunks = tuple(
        large.SealedChunk(index, index * 256, (index + 1) * 256, "prefix")
        for index in range(3)
    )
    request = large._ScheduledRequest(
        request=request_input,
        source_rank=0,
        target_rank=1,
        arrival_ms=0.0,
        start_ms=0.0,
        base_finish_ms=100.0,
        queue_ms=0.0,
        service_ms=100.0,
        prefill_ms=100.0,
        decode_ms=0.0,
        reused_tokens=0,
        chunks=chunks,
        policy_slack_ms=0.0,
    )
    adapter = large._BaselineAdapter(config, policy)
    adapter.register(request, slack_enabled=False)

    assert adapter.seal(request, chunks[0], 0.0) == "HBM_ONLY"
    first_failure = adapter.fail_worker(1, 2.0)
    state = adapter._runtime(request)._requests[request_input.request_id]
    assert first_failure["invalidated_hbm_requests"] == 1
    assert state.generation == 1
    assert state.blocked is True

    # A repeated physical failure before repair is an idempotent invalidation,
    # not a new request generation.
    repeated_failure = adapter.fail_worker(1, 2.1)
    assert repeated_failure["invalidated_hbm_requests"] == 0
    assert state.generation == 1

    # The outage chunk and the later suffix are both rejected, but both seals
    # advance the strict ordinal frontier.  Repair only enables new replicas;
    # it must not fill the hole or resurrect this request's old prefix.
    assert adapter.seal(request, chunks[1], 3.0) == "BYPASS"
    assert state.next_ordinal == 2
    assert state.generation == 1
    assert adapter.repair_worker(1, 4.0)["future_hbm_admission_reenabled"]
    assert adapter.seal(request, chunks[2], 5.0) == "BYPASS"
    assert state.next_ordinal == 3
    assert state.generation == 1
    assert state.used_bytes == 0

    # Preserve the safety check: retries and future ordinals are not silently
    # accepted merely because an outage occurred.
    with pytest.raises(ValueError, match="chunk ordinal is not progressive"):
        adapter.seal(request, chunks[2], 6.0)
    with pytest.raises(ValueError, match="chunk ordinal is not progressive"):
        adapter.seal(
            request,
            large.SealedChunk(4, 1_024, 1_280, "prefix"),
            7.0,
        )


@pytest.mark.parametrize(
    "policy",
    [large.BaselinePolicy.DEJAVU_NEIGHBOR, large.BaselinePolicy.KEVLARFLOW_RING],
)
def test_hbm_baseline_request_arriving_during_outage_consumes_every_ordinal(
    policy: large.BaselinePolicy,
) -> None:
    """Lock the exact ordering observed in the failed Qwen3-32B seeds."""

    config = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=1,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        failure_rate_per_1000_worker_hours=0.0,
    )
    request_input = RequestInput(
        request_id="arrived-while-target-down",
        workflow_id="workflow",
        prompt_tokens=768,
        output_tokens=0,
        slack_ms=0.0,
        prefix_group="prefix",
        arrival_ms=1.0,
        input_reuse_tokens=0,
    )
    chunks = tuple(
        large.SealedChunk(index, index * 256, (index + 1) * 256, "prefix")
        for index in range(3)
    )
    request = large._ScheduledRequest(
        request=request_input,
        source_rank=0,
        target_rank=1,
        arrival_ms=1.0,
        start_ms=1.0,
        base_finish_ms=100.0,
        queue_ms=0.0,
        service_ms=99.0,
        prefill_ms=99.0,
        decode_ms=0.0,
        reused_tokens=0,
        chunks=chunks,
        policy_slack_ms=0.0,
    )
    adapter = large._BaselineAdapter(config, policy)
    adapter.fail_worker(1, 0.0)
    adapter.register(request, slack_enabled=False)

    assert adapter.seal(request, chunks[0], 2.0) == "BYPASS"
    assert adapter.seal(request, chunks[1], 3.0) == "BYPASS"
    state = adapter._runtime(request)._requests[request_input.request_id]
    assert state.next_ordinal == 2
    assert state.generation == 1
    assert sum(
        row["kind"] == "request_state_released"
        for row in adapter._runtime(request).snapshot()["events"]
    ) == 1

    assert adapter.repair_worker(1, 4.0)["future_hbm_admission_reenabled"]
    assert adapter.seal(request, chunks[2], 5.0) == "BYPASS"
    assert state.next_ordinal == 3
    assert state.generation == 1
    with pytest.raises(ValueError, match="chunk ordinal is not progressive"):
        adapter.seal(request, chunks[2], 6.0)
    with pytest.raises(ValueError, match="chunk ordinal is not progressive"):
        adapter.seal(
            request,
            large.SealedChunk(4, 1_024, 1_280, "prefix"),
            7.0,
        )


def test_cluster_fault_exposure_scales_with_physical_worker_count() -> None:
    four = LargeScaleCellConfig(
        dp_size=4,
        gpus_per_node=4,
        request_count=1,
        workflow_count=1,
        concurrency=1,
        failure_rate_per_1000_worker_hours=1_000_000.0,
    )
    eight = replace(four, dp_size=8)

    four_faults = large._generate_faults(four, horizon_ms=100_000.0, seed=19)
    eight_faults = large._generate_faults(eight, horizon_ms=100_000.0, seed=19)

    assert len(four_faults) > 0
    assert len(eight_faults) >= len(four_faults)
    assert max(row.source_rank for row in four_faults) < 4
    assert max(row.source_rank for row in eight_faults) < 8


def test_failure_delays_protection_reclaim_until_adjusted_finish() -> None:
    config = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        control_window_ms=100.0,
        failure_rate_per_1000_worker_hours=0.0,
        minimum_faults_per_cell=1,
    )
    workload = _workload(1, prompt=8_192)
    result = run_cell(config, 22, "aegiskv", workload=workload)
    request = result["request_rows"][0]
    final_base_window = max(
        result["window_rows"], key=lambda row: row["time_ms"]
    )

    assert request["failure_count"] == 1
    assert request["simulated_finish_ms"] > request["base_finish_ms"]
    assert (
        final_base_window["hbm_used_bytes"] > 0
        or final_base_window["dram_used_bytes"] > 0
    )
    assert result["final_state"]["hbm_used_bytes"] == 0
    assert result["final_state"]["dram_used_bytes"] == 0


def _assert_shadow_probe_primary_state_is_identical(
    disabled: dict[str, object], enabled: dict[str, object]
) -> None:
    """Compare every primary surface while excluding probe-only evidence."""

    for key in (
        "request_rows",
        "workflow_rows",
        "window_rows",
        "failure_rows",
        "repair_rows",
        "failure_state_at_horizon",
        "final_state",
    ):
        assert disabled[key] == enabled[key]
    disabled_summary = {
        key: value
        for key, value in disabled["summary"].items()
        if not key.startswith("active_source_probe_")
    }
    enabled_summary = {
        key: value
        for key, value in enabled["summary"].items()
        if not key.startswith("active_source_probe_")
    }
    assert disabled_summary == enabled_summary


@pytest.mark.parametrize("minimum_faults", [0, 1])
def test_active_source_shadow_probes_do_not_change_primary_state(
    minimum_faults: int,
) -> None:
    base = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=2,
        request_count=4,
        workflow_count=2,
        control_window_ms=50.0,
        failure_rate_per_1000_worker_hours=0.0,
        minimum_faults_per_cell=minimum_faults,
    )
    workload = _workload(4, prompt=8_192)
    disabled = run_cell(base, 23, "aegiskv", workload=workload)
    enabled = run_cell(
        replace(base, active_source_probe_count=64),
        23,
        "aegiskv",
        workload=workload,
    )

    _assert_shadow_probe_primary_state_is_identical(disabled, enabled)
    evidence = enabled["active_source_fault_conditioned_recovery"]
    assert len(enabled["active_source_probe_rows"]) == 64
    assert evidence["realized_probe_count"] == 64
    assert evidence["formal_ci_unit"] == "paired_seed_mean_over_64_strata"
    assert evidence["probe_inner_bootstrap_allowed"] is False
    assert enabled["gates"][
        "active_source_probe_one_per_equal_exposure_stratum"
    ]
    if minimum_faults == 0:
        assert all(
            not snapshot["recovery_estimate_action_counts"]
            for snapshot in enabled["final_state"]["target_snapshots"]
        )


@pytest.mark.parametrize(
    "policy",
    [
        "dejavu_neighbor",
        "raidserve_host",
        "kevlarflow_ring",
        "ghostserve_parity",
        "static_independent_fcfs",
        "recompute_only",
    ],
)
@pytest.mark.parametrize("minimum_faults", [0, 1])
def test_shadow_probe_noninterference_holds_for_every_adapter(
    policy: str, minimum_faults: int
) -> None:
    base = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=2,
        request_count=4,
        workflow_count=2,
        control_window_ms=50.0,
        failure_rate_per_1000_worker_hours=0.0,
        minimum_faults_per_cell=minimum_faults,
    )
    workload = _workload(4, prompt=8_192)

    disabled = run_cell(base, 23, policy, workload=workload)
    enabled = run_cell(
        replace(base, active_source_probe_count=64),
        23,
        policy,
        workload=workload,
    )

    _assert_shadow_probe_primary_state_is_identical(disabled, enabled)


def test_active_source_probe_tuple_and_universe_are_policy_and_rate_independent() -> None:
    base = LargeScaleCellConfig(
        dp_size=4,
        gpus_per_node=4,
        concurrency=4,
        request_count=8,
        workflow_count=2,
        control_window_ms=100.0,
        failure_rate_per_1000_worker_hours=0.0,
        active_source_probe_count=64,
    )
    workload = _workload(8, prompt=8_192)
    full = run_cell(base, 31, "aegiskv_alg111", workload=workload)
    no_alg1 = run_cell(
        base, 31, "aegiskv_alg011_no_alg1", workload=workload
    )
    high_rate = run_cell(
        replace(base, failure_rate_per_1000_worker_hours=41.667),
        31,
        "recompute_only",
        workload=workload,
    )

    tuple_keys = full["active_source_probe_metadata"]["probe_tuple_fields"]
    expected_rows = [
        {key: row[key] for key in tuple_keys}
        for row in full["active_source_probe_rows"]
    ]
    assert large._sha256_json(expected_rows) == full[
        "active_source_probe_metadata"
    ]["probe_tuple_sha256"]
    for result in (no_alg1, high_rate):
        assert result["active_source_probe_metadata"][
            "probe_tuple_sha256"
        ] == full["active_source_probe_metadata"]["probe_tuple_sha256"]
        assert result["active_source_probe_metadata"][
            "eligible_exposure_universe_sha256"
        ] == full["active_source_probe_metadata"][
            "eligible_exposure_universe_sha256"
        ]
        assert result["active_source_probe_metadata"][
            "sampling_prng_key_sha256"
        ] == full["active_source_probe_metadata"][
            "sampling_prng_key_sha256"
        ]
        assert [
            {key: row[key] for key in tuple_keys}
            for row in result["active_source_probe_rows"]
        ] == expected_rows
    assert [row["stratum_index"] for row in expected_rows] == list(range(64))
    assert all(
        row["produced_tokens"] >= row["sealed_frontier_tokens"] > 0
        for row in expected_rows
    )


def test_controller_shadow_recovery_query_is_pure() -> None:
    config = LargeScaleCellConfig(
        dp_size=2,
        gpus_per_node=2,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        failure_rate_per_1000_worker_hours=0.0,
    )
    request = large._schedule_requests(config, _workload(1, prompt=8_192))[0]
    adapter = large._AegisAdapter(config, AlgorithmToggles())
    adapter.register(request, slack_enabled=True)
    adapter.seal(request, request.chunks[0], request.start_ms)
    runtime = adapter._runtime(request)
    runtime.flush()
    before = runtime.snapshot(include_controller=True)

    decision = runtime.estimate_recovery_at(
        request.request.request_id,
        activated_target=f"target-{request.target_rank}",
        source_token_end=request.total_tokens,
        slack_ms=17.0,
        time_ms=runtime.snapshot()["time_ms"],
    )
    after = runtime.snapshot(include_controller=True)

    assert decision.request_id == request.request.request_id
    assert after == before
    assert not after["recovery_estimate_action_counts"]


def test_probe_reuse_fields_and_full_workload_bins_are_auditable() -> None:
    reuse_tokens = (0, 1_024, 3_800, 4_096)
    workload = [
        RequestInput(
            request_id=f"reuse-{index}",
            workflow_id=f"w-{index}",
            prompt_tokens=4_096,
            output_tokens=256,
            slack_ms=0.0,
            actual_slack_ms=0.0,
            prefix_group=f"identity-a-{index}",
            arrival_ms=0.0,
            input_reuse_tokens=reused,
        )
        for index, reused in enumerate(reuse_tokens)
    ]
    relabeled = [
        replace(row, prefix_group=f"identity-b-{index}")
        for index, row in enumerate(workload)
    ]
    config = LargeScaleCellConfig(
        dp_size=4,
        gpus_per_node=4,
        concurrency=4,
        request_count=4,
        workflow_count=4,
        failure_rate_per_1000_worker_hours=0.0,
        active_source_probe_count=64,
    )

    result = run_cell(config, 37, workload=workload)
    same_base_service = run_cell(config, 37, workload=relabeled)

    assert result["active_source_probe_metadata"][
        "probe_tuple_sha256"
    ] == same_base_service["active_source_probe_metadata"][
        "probe_tuple_sha256"
    ]
    assert result["summary"]["workload_reuse_bin_request_counts"] == {
        "none": 1,
        "partial": 1,
        "high": 1,
        "near_complete": 1,
    }
    assert sum(
        result["summary"]["active_source_probe_reuse_bin_counts"].values()
    ) == 64
    assert all(
        row["input_reuse_tokens"] in reuse_tokens
        and 0.0 <= row["input_reuse_fraction"] <= 1.0
        and row["reuse_outcome_does_not_filter_probe_selection"] is True
        for row in result["active_source_probe_rows"]
    )
    intended = result["summary"][
        "workload_reuse_bin_intended_residency_chunk_counts"
    ]
    assert set(intended) == {"none", "partial", "high", "near_complete"}
    assert sum(
        sum(counts.values()) for counts in intended.values()
    ) == sum(
        len(row.chunks) for row in large._schedule_requests(config, workload)
    )


def test_positive_shadow_probe_count_fails_closed_without_active_sealed_exposure() -> None:
    config = LargeScaleCellConfig(
        dp_size=1,
        gpus_per_node=1,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        failure_rate_per_1000_worker_hours=0.0,
        active_source_probe_count=1,
    )
    workload = [
        RequestInput(
            request_id="no-post-seal-service",
            workflow_id="w",
            prompt_tokens=128,
            output_tokens=0,
            slack_ms=0.0,
            actual_slack_ms=0.0,
            prefix_group="p",
            arrival_ms=0.0,
            input_reuse_tokens=0,
        )
    ]

    with pytest.raises(ValueError, match=r"no active\+sealed GPU-shard exposure"):
        run_cell(config, 1, workload=workload)
