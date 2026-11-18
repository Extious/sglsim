#!/usr/bin/env python3
"""Independent fail-closed audit for raw, analysis, and figure artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sglsim.srt.simulation.aegis_large_scale import REUSE_BIN_CONTRACT


ROOT = Path(__file__).resolve().parent
PROBE_COUNT = 64
PROBE_TUPLE_FIELDS = (
    "probe_id",
    "stratum_index",
    "request_id",
    "workflow_id",
    "time_ms",
    "source_rank",
    "target_rank",
    "shard_rank",
    "produced_tokens",
    "sealed_frontier_tokens",
)
PROBE_GATES = (
    "active_source_probe_schedule_policy_independent",
    "active_source_probe_schedule_failure_rate_independent",
    "active_source_probe_uses_physical_gpu_shard_time",
    "active_source_probe_requires_active_and_sealed",
    "active_source_probe_does_not_filter_policy_outcomes",
    "active_source_probe_count_realized_or_disabled",
    "active_source_probe_one_per_equal_exposure_stratum",
    "active_source_probe_tuple_hash_present",
    "active_source_probe_query_non_mutating",
    "active_source_probe_excluded_from_primary_counters",
    "active_source_probe_ground_truth_harm_uses_actual_slack",
    "active_source_probe_ci_uses_paired_seed_outer_bootstrap_only",
    "active_source_probe_inner_bootstrap_forbidden",
)
PROBE_MATCH_FIELDS = (
    "model_id",
    "prompt_scale",
    "active_workflow_concurrency",
    "total_gpu_count",
    "tp_size",
    "pp_size",
    "system_dp_size",
    "network_effective_gb_s",
    "hbm_quota_gib_per_target",
    "dram_quota_gib_per_target",
    "algorithm1_prediction_error_multiplier",
)
PREFIGURE_REQUIRED_CHECKS = frozenset(
    {
        "protocol_sha",
        "matrix_sha",
        "matrix_canonical_sha",
        "producer_source_hashes",
        "cache_manifest_internal_refs_and_generator_sha",
        "raw_contract",
        "canonical_campaign_index_is_zero_failure_complete",
        "t0_aimd_snapshot_and_periodic_contract",
        "failure_history_horizon_and_post_event_drain_accounting_exact",
        "complete_raw_retention",
        "failure_tuple_schedule_identical_across_policies",
        "active_source_probe_raw_contract",
        "active_source_probe_tuple_identical_across_policy_and_rate",
        "selection_provenance_matches_raw_tuples",
        "exact_3110",
        "exact_matrix_run_ids",
        "exact_ten_paired_seeds_per_policy_scenario",
        "analysis_status_exact_for_campaign_mode",
        "analysis_files_match",
        "analysis_contract_pass",
        "analysis_rationality_matches_mode",
        "analysis_algorithm1_identifiability_matches_mode",
        "analysis_paired_comparison_coverage_matches_mode",
        "analysis_paired_probe_comparison_coverage_matches_mode",
        "analysis_scenario_diagnostics_count_exact",
        "analysis_paper_eligibility_exact",
    }
)


def _aimd_snapshot_and_periodic_contract(
    artifact: dict[str, Any], protocol: dict[str, Any]
) -> bool:
    """Independently validate initialization and periodic AIMD raw evidence."""

    rows = list(artifact.get("window_rows") or [])
    initial = [row for row in rows if bool(row.get("initial_sample"))]
    if len(initial) != 1:
        return False
    first = initial[0]
    algorithm3_enabled = bool(first.get("algorithm3_enabled"))
    if not (
        math.isclose(float(first.get("time_ms", math.nan)), 0.0, abs_tol=1e-12)
        and first.get("aimd_update_applied") is False
        and (
            not algorithm3_enabled
            or (
                math.isclose(
                    float(first.get("controller_alpha_mean", math.nan)),
                    float(protocol["algorithm3"]["initial_alpha"]),
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(first.get("controller_stable_windows_mean", math.nan)),
                    0.0,
                    abs_tol=1e-12,
                )
                and int(first.get("controller_stable_windows_max", -1)) == 0
            )
        )
    ):
        return False
    summary = artifact.get("summary") or {}
    applied = [row for row in rows if bool(row.get("aimd_update_applied"))]
    if not all(
        bool(row.get("aimd_update_applied"))
        == bool(algorithm3_enabled and float(row.get("time_ms", 0.0)) > 0.0)
        for row in rows
    ):
        return False
    if not (
        int(summary.get("initial_snapshot_sample_count", -1)) == 1
        and int(summary.get("aimd_periodic_updates_applied", -1)) == len(applied)
        and summary.get("first_aimd_periodic_update_ms")
        == (float(applied[0]["time_ms"]) if applied else None)
    ):
        return False
    gates = artifact.get("gates") or {}
    if not (
        gates.get("t0_control_window_is_snapshot_only") is True
        and gates.get("aimd_updates_only_at_positive_periodic_windows") is True
        and gates.get("initial_sample_does_not_advance_stable_counter") is True
    ):
        return False
    highest = int(artifact["scenario"]["active_workflow_concurrency"]) == max(
        int(value)
        for value in protocol["workload"]["active_workflow_concurrency_sweep"]
    )
    if highest:
        horizon_ms = float(artifact["workload"]["steady_measurement_end_ms"])
        opportunities = [
            row
            for row in rows
            if 0.0 < float(row.get("time_ms", 0.0)) <= horizon_ms + 1e-9
        ]
        applied_before_horizon = [
            row
            for row in applied
            if float(row.get("time_ms", 0.0)) <= horizon_ms + 1e-9
        ]
        minimum = int(
            protocol["workload"][
                "highest_concurrency_minimum_periodic_aimd_updates"
            ]
        )
        if len(opportunities) < minimum:
            return False
        if algorithm3_enabled:
            if len(applied_before_horizon) < minimum:
                return False
        elif applied_before_horizon:
            return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def verify_manifest_files(root: Path, records: dict[str, Any]) -> bool:
    return all(
        (root / name).is_file()
        and sha256(root / name) == record["sha256"]
        and (root / name).stat().st_size == int(record["size_bytes"])
        for name, record in records.items()
    )


def _canonical_manifest_matches(path: Path) -> tuple[bool, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    expected = str(payload.pop("canonical_sha256", ""))
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return bool(expected) and expected == actual, manifest


def _cache_bundle_contract(index: dict[str, Any]) -> bool:
    """Reverify generated-cache refs, canonical hashes, and producer binding."""

    try:
        lineage_files = index["lineage"]["files"]

        def lineage_record(name: str) -> dict[str, Any]:
            record = lineage_files[name]
            path = Path(record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(name)
            return record

        compact_record = lineage_record("compact_manifest")
        alg1_record = lineage_record("modeled_algorithm1_manifest")
        compact_ok, compact = _canonical_manifest_matches(
            Path(compact_record["path"])
        )
        alg1_ok, alg1 = _canonical_manifest_matches(Path(alg1_record["path"]))
        if not compact_ok or not alg1_ok or int(alg1.get("schema_version", 0)) != 2:
            return False

        for name, record in {
            **dict(compact.get("arrays") or {}),
            "workflow_reuse": compact["workflow_reuse"],
        }.items():
            bound = lineage_record(f"compact_cache:{name}")
            if (
                bound["sha256"] != record["sha256"]
                or Path(bound["path"]).resolve()
                != (
                    Path(compact_record["path"]).parent / str(record["path"])
                ).resolve()
            ):
                return False
        for name, record in dict(alg1.get("arrays") or {}).items():
            bound = lineage_record(f"modeled_algorithm1_cache:{name}")
            if (
                bound["sha256"] != record["sha256"]
                or Path(bound["path"]).resolve()
                != (
                    Path(alg1_record["path"]).parent / str(record["path"])
                ).resolve()
            ):
                return False

        producer = lineage_record("modeled_algorithm1_generator")
        protocol = lineage_record("protocol")
        trace_generator = lineage_record("compact_trace_generator")
        if not trace_generator["sha256"]:
            return False
        return bool(
            alg1.get("producer", {}).get("sha256") == producer["sha256"]
            and alg1.get("protocol", {}).get("sha256") == protocol["sha256"]
            and alg1.get("compact_workload", {}).get("sha256")
            == compact_record["sha256"]
            and alg1.get("compact_workload", {}).get("canonical_sha256")
            == compact["canonical_sha256"]
            and alg1.get("mechanism_gate", {}).get("passed") is True
            and alg1.get("policy_input", {}).get("array")
            == "predicted_slack_ms"
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def audit_raw(campaign_dir: Path, *, require_full: bool) -> tuple[dict[str, Any], dict[str, bool]]:
    index_path = campaign_dir / "campaign-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    matrix = json.loads(Path(index["matrix"]["path"]).read_text(encoding="utf-8"))
    protocol = json.loads(Path(index["protocol"]["path"]).read_text(encoding="utf-8"))
    records = {row["run_id"]: row for row in index["records"]}
    expected_matrix_ids = {row["run_id"] for row in matrix["runs"]}
    observed_ids: set[str] = set()
    evidence: set[str] = set()
    designs: set[str] = set()
    lineages: set[str] = set()
    scenario_policy_seed: set[tuple[str, str, int]] = set()
    schedule_by_scenario_seed: dict[tuple[str, int], set[str]] = {}
    failures_by_scenario_seed: dict[tuple[str, int], set[str]] = {}
    artifact_hashes: list[dict[str, str]] = []
    raw_contract_pass = True
    full_retention_pass = True
    schedule_contract_pass = True
    resolved_network_pass = True
    probe_contract_pass = True
    aimd_snapshot_contract_pass = True
    probe_tuple_by_match: dict[tuple[Any, ...], set[str]] = {}
    probe_universe_by_match: dict[tuple[Any, ...], set[str]] = {}
    probe_prng_by_match: dict[tuple[Any, ...], set[str]] = {}
    probe_exposure_by_match: dict[tuple[Any, ...], set[float]] = {}
    failure_history_contract_pass = True
    for path in sorted((campaign_dir / "raw").glob("**/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            artifact = json.load(stream)
        run_id = artifact["run_id"]
        digest = sha256(path)
        observed_ids.add(run_id)
        evidence.add(artifact.get("evidence_class", ""))
        designs.add(artifact.get("design_semantics", ""))
        lineages.add(artifact.get("lineage", {}).get("lineage_sha256", ""))
        scenario_policy_seed.add(
            (
                artifact["scenario"]["scenario_id"],
                artifact["policy"],
                int(artifact["seed"]),
            )
        )
        schedule_by_scenario_seed.setdefault(
            (
                artifact["scenario"]["scenario_id"],
                int(artifact["seed"]),
            ),
            set(),
        ).add(
            artifact["workload"][
                "policy_independent_projected_workload_sha256"
            ]
        )
        failure_schedule = [
            (
                float(row["time_ms"]),
                int(row["source_rank"]),
                int(row["shard_rank"]),
                bool(row["forced_floor_event"]),
            )
            for row in artifact["failure_rows"]
        ]
        failure_schedule_sha = hashlib.sha256(
            json.dumps(
                failure_schedule, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        summary = artifact["summary"]
        horizon_history = artifact.get("failure_state_at_horizon", {})
        final_history = artifact["final_state"].get("failure_history", {})
        physical_repeat = sum(
            bool(row["physical_gpu_shard_ever_failed_before_event"])
            for row in artifact["failure_rows"]
        )
        effective_bound_target_current = sum(
            bool(row["effective"])
            and bool(
                row[
                    "bound_peer_hbm_target_currently_unavailable_before_source_event"
                ]
            )
            for row in artifact["failure_rows"]
        )
        current_physical_horizon = int(
            horizon_history.get("current_physical_gpu_shard_outage_count", 0)
        )
        current_physical_post = int(
            final_history.get("current_physical_gpu_shard_outage_count", 0)
        )
        physical_ever = int(
            horizon_history.get("physical_gpu_shards_ever_failed_count", 0)
        )
        current_logical_horizon = int(
            horizon_history.get(
                "current_logical_replica_target_outage_count", 0
            )
        )
        current_logical_post = int(
            final_history.get(
                "current_logical_replica_target_outage_count", 0
            )
        )
        logical_ever = int(
            horizon_history.get("logical_replica_targets_ever_failed_count", 0)
        )
        stable_history_fields = (
            "physical_gpu_shards_ever_failed_count",
            "logical_replica_targets_ever_failed_count",
            "old_hbm_copy_invalidation_event_count_cumulative",
            "old_hbm_copy_invalidated_requests_cumulative",
            "old_hbm_copy_invalidated_bytes_cumulative",
        )
        repairs_after_horizon = sum(
            bool(row.get("after_base_run_horizon"))
            for row in artifact["repair_rows"]
        )
        history_contract = artifact["formal_metric_contract"][
            "failure_history_contract"
        ]
        failure_history_contract_pass &= bool(
            physical_repeat
            == int(
                summary[
                    "failure_events_on_previously_failed_physical_gpu_shard"
                ]
            )
            and effective_bound_target_current
            == int(
                summary[
                    "effective_failures_with_bound_peer_hbm_target_prior_unavailable"
                ]
            )
            and int(
                horizon_history.get(
                    "old_hbm_copy_invalidation_event_count_cumulative", 0
                )
            )
            == int(summary["old_hbm_copy_invalidation_event_count_cumulative"])
            and int(
                horizon_history.get(
                    "old_hbm_copy_invalidated_requests_cumulative", 0
                )
            )
            == int(summary["old_hbm_copy_invalidated_requests_cumulative"])
            and int(
                horizon_history.get(
                    "old_hbm_copy_invalidated_bytes_cumulative", 0
                )
            )
            == int(summary["old_hbm_copy_invalidated_bytes_cumulative"])
            and current_physical_horizon
            == int(summary["current_physical_gpu_shard_outage_count_at_horizon"])
            and current_physical_post
            == int(
                summary[
                    "current_physical_gpu_shard_outage_count_post_all_events"
                ]
            )
            and physical_ever
            == int(summary["physical_gpu_shards_ever_failed_count"])
            and current_logical_horizon
            == int(
                summary[
                    "current_logical_replica_target_outage_count_at_horizon"
                ]
            )
            and current_logical_post
            == int(
                summary[
                    "current_logical_replica_target_outage_count_post_all_events"
                ]
            )
            and logical_ever
            == int(summary["logical_replica_targets_ever_failed_count"])
            and current_physical_horizon <= physical_ever
            and current_physical_post <= physical_ever
            and current_logical_horizon <= logical_ever
            and current_logical_post <= logical_ever
            and horizon_history.get("snapshot_scope")
            == "base_run_horizon_after_all_events_at_or_before_horizon"
            and math.isclose(
                float(horizon_history.get("snapshot_time_ms", math.nan)),
                float(summary["base_run_horizon_ms"]),
                abs_tol=1e-9,
            )
            and final_history.get("snapshot_scope")
            == "post_all_scheduled_events_including_repairs_after_horizon"
            and math.isclose(
                float(final_history.get("snapshot_time_ms", math.nan)),
                float(summary["post_all_events_last_event_time_ms"]),
                abs_tol=1e-9,
            )
            and all(
                horizon_history.get(field) == final_history.get(field)
                for field in stable_history_fields
            )
            and int(summary["repair_events_after_horizon"])
            == repairs_after_horizon
            and int(summary["post_horizon_event_count"])
            >= repairs_after_horizon
            and history_contract.get("horizon_state_source")
            == "failure_state_at_horizon"
            and history_contract.get("post_all_events_state_source")
            == "final_state.failure_history"
            and history_contract.get(
                "events_after_horizon_are_cleanup_not_horizon_state"
            )
            is True
            and artifact["gates"].get(
                "horizon_failure_state_frozen_before_post_horizon_events"
            )
            is True
            and artifact["gates"].get(
                "summary_at_horizon_fields_derive_from_horizon_failure_state"
            )
            is True
            and artifact["gates"].get(
                "final_state_is_post_all_events_not_horizon_state"
            )
            is True
            and final_history.get(
                "repair_clears_current_outage_not_ever_failed_history"
            )
            is True
        )
        metadata = artifact.get("active_source_probe_metadata", {})
        probe_rows = artifact.get("active_source_probe_rows", [])
        nested_probe = artifact.get(
            "active_source_fault_conditioned_recovery", {}
        )
        tuple_fields = tuple(metadata.get("probe_tuple_fields", ()))
        tuple_payload = (
            [
                {field: row[field] for field in tuple_fields}
                for row in sorted(
                    probe_rows, key=lambda row: int(row["stratum_index"])
                )
            ]
            if tuple_fields == PROBE_TUPLE_FIELDS
            else []
        )
        probe_digest = canonical_json_sha(tuple_payload)
        probe_contract_pass &= bool(
            int(metadata.get("requested_probe_count", -1)) == PROBE_COUNT
            and int(metadata.get("realized_probe_count", -1)) == PROBE_COUNT
            and int(artifact.get("active_source_probe_row_count", -1))
            == PROBE_COUNT
            and len(probe_rows) == PROBE_COUNT
            and tuple_fields == PROBE_TUPLE_FIELDS
            and probe_digest == metadata.get("probe_tuple_sha256")
            and nested_probe.get("probe_rows") == probe_rows
            and nested_probe.get("probe_inner_bootstrap_allowed") is False
            and nested_probe.get("formal_ci_unit")
            == "paired_seed_mean_over_64_strata"
            and sorted(int(row["stratum_index"]) for row in probe_rows)
            == list(range(PROBE_COUNT))
            and all(
                row.get("eligible_active_source") is True
                and row.get("eligible_complete_sealed_frontier") is True
                and int(row.get("sealed_frontier_tokens", 0)) > 0
                and row.get("eligibility_uses_policy_state") is False
                and row.get(
                    "eligibility_filters_admission_or_recovery_outcome"
                )
                is False
                and row.get("primary_counters_updated") is False
                for row in probe_rows
            )
            and all(artifact.get("gates", {}).get(name) is True for name in PROBE_GATES)
            and artifact["summary"].get("workload_reuse_bin_contract")
            == REUSE_BIN_CONTRACT
            and artifact["summary"].get(
                "active_source_probe_reuse_bin_contract"
            )
            == REUSE_BIN_CONTRACT
        )
        match_key = (
            int(artifact["seed"]),
            *(artifact["scenario"][field] for field in PROBE_MATCH_FIELDS),
        )
        probe_tuple_by_match.setdefault(match_key, set()).add(probe_digest)
        probe_universe_by_match.setdefault(match_key, set()).add(
            str(metadata.get("eligible_exposure_universe_sha256"))
        )
        probe_prng_by_match.setdefault(match_key, set()).add(
            str(metadata.get("sampling_prng_key_sha256"))
        )
        probe_exposure_by_match.setdefault(match_key, set()).add(
            float(metadata.get("eligible_exposure_gpu_shard_hours", float("nan")))
        )
        failures_by_scenario_seed.setdefault(
            (
                artifact["scenario"]["scenario_id"],
                int(artifact["seed"]),
            ),
            set(),
        ).add(failure_schedule_sha)
        artifact_hashes.append(
            {"run_id": run_id, "path": str(path), "sha256": digest}
        )
        raw_contract_pass &= (
            run_id in records
            and records[run_id].get("sha256") == digest
            and artifact["summary"]["request_count"] == 98_827
            and artifact["simulator_cell_schema_version"]
            == "aegiskv_sglsim_large_scale_cell_v2"
            and artifact["summary"]["workflow_count"] == 393
            and artifact["summary"]["failure_rate_exposure_unit"]
            == "total_gpu_shard_hours"
            and float(
                artifact["summary"]["modeled_failure_exposure_gpu_hours"]
            )
            > 0.0
            and int(artifact["summary"]["failure_events_injected"])
            == len(artifact["failure_rows"])
            and int(artifact["summary"]["worker_repair_events_scheduled"])
            == len(artifact["repair_rows"])
            and artifact["resolved_config"]["failure_rate_exposure"]["unit"]
            == "total_gpu_shard_hours"
            and int(
                artifact["resolved_config"]["failure_rate_exposure"][
                    "poisson_process_count"
                ]
            )
            == int(artifact["resolved_config"]["total_gpu_count"])
            and artifact["gates"]["simulated_only_label_present"] is True
            and artifact["gates"]["fault_process_covers_full_run_horizon"] is True
            and artifact["gates"][
                "measurement_window_only_filters_steady_telemetry"
            ]
            is True
            and artifact["gates"]["alpha_pressure_only"] is True
            and artifact["gates"][
                "failure_rate_direct_alpha_dependency_absent"
            ]
            is True
            and artifact["gates"]["single_worker_shard_failure_counterfactual"]
            is True
            and protocol["failure"][
                "forced_or_synthetic_target_first_injection"
            ]
            is False
            and artifact["gates"][
                "forced_synthetic_target_first_injection_disabled"
            ]
            is True
            and all(
                row.get("compound_target_first_failure_injected") is False
                and row.get("forced_synthetic_target_first_injection") is False
                for row in artifact["failure_rows"]
            )
            and artifact["gates"][
                "physical_gpu_shard_and_logical_replica_history_separated"
            ]
            is True
            and artifact["gates"][
                "current_outage_and_ever_failed_counters_separated"
            ]
            is True
            and artifact["gates"][
                "old_hbm_copy_invalidation_counters_cumulative"
            ]
            is True
            and protocol["failure"]["standby_rotation_modeled"] is False
            and protocol["failure"][
                "complete_current_design_target_failover_evaluated"
            ]
            is False
            and protocol["failure"][
                "target_failure_history_is_no_standby_rotation_conservative_sensitivity"
            ]
            is True
            and artifact["gates"]["standby_rotation_modeled"] is False
            and artifact["gates"][
                "complete_current_design_target_failover_evaluated"
            ]
            is False
            and artifact["gates"][
                "target_failure_history_is_no_standby_rotation_conservative_sensitivity"
            ]
            is True
            and artifact["formal_metric_contract"]["failure_history_contract"][
                "standby_rotation_modeled"
            ]
            is False
            and artifact["gates"]["executed_physical_fail_stop"] is False
            and artifact["gates"][
                "failure_queue_outage_replacement_rescheduling_modeled"
            ]
            is False
            and artifact["gates"][
                "failed_worker_peer_hbm_invalidated_for_all_policies"
            ]
            is True
            and artifact["gates"]["explicit_worker_repair_event_modeled"]
            is True
            and artifact["gates"][
                "repair_reenables_future_hbm_admission_only"
            ]
            is True
            and artifact["gates"]["repair_does_not_restore_old_hbm_copies"]
            is True
            and artifact["gates"][
                "repair_does_not_schedule_hbm_reprotection"
            ]
            is True
            and artifact["gates"]["ground_truth_harm_uses_actual_slack"]
            is True
            and artifact["gates"][
                "policy_predicted_slack_not_used_as_ground_truth"
            ]
            is True
            and artifact["gates"]["workflow_actual_harm_surrogate_aggregated"]
            is True
            and artifact["gates"]["end_to_end_dag_failure_execution_replay"]
            is False
            and artifact["gates"][
                "formal_analysis_uses_explicit_actual_harm_surrogate_fields"
            ]
            is True
            and artifact["gates"][
                "legacy_wct_aliases_forbidden_for_formal_claims"
            ]
            is True
            and artifact["formal_metric_contract"][
                "primary_failure_endpoint_prefix"
            ]
            == "actual_failure_added_workflow_harm_surrogate_"
            and {
                row["request_id"]
                for row in artifact["failure_rows"]
                if bool(row["effective"])
            }.issubset(
                {
                    row["request_id"]
                    for row in artifact["request_rows_sample"]
                }
            )
        )
        aimd_snapshot_contract_pass &= _aimd_snapshot_and_periodic_contract(
            artifact, protocol
        )
        full_retention_pass &= (
            len(artifact["workflow_rows"]) == 393
            and len(artifact["window_rows"]) == artifact["window_row_count"]
            and len(artifact["failure_rows"]) == artifact["failure_row_count"]
            and len(artifact["repair_rows"]) == artifact["repair_row_count"]
            and artifact["raw_retention"]["full_workflow_rows_retained"] is True
            and artifact["raw_retention"]["full_window_rows_retained"] is True
            and artifact["raw_retention"]["full_failure_rows_retained"] is True
            and artifact["raw_retention"]["full_repair_rows_retained"] is True
            and artifact["raw_retention"][
                "full_active_source_probe_rows_retained"
            ]
            is True
        )
        schedule_contract_pass &= (
            int(protocol["workload"]["wct_measurement_workflows"]) == 393
            and artifact["summary"]["workflow_count"] == 393
            and artifact["workload"]["all_393_workflows_in_wct"] is True
            and artifact["workload"]["stabilization_prefix_excluded_from_wct"]
            is False
            and artifact["workload"][
                "achieved_offered_trace_steady_concurrency_relative_error"
            ]
            <= float(
                protocol["workload"][
                    "achieved_steady_concurrency_relative_error_gate"
                ]
            )
            + 1e-12
            and "policy-independent"
            in artifact["workload"]["offered_schedule_completion_semantics"]
            and (
                int(artifact["scenario"]["active_workflow_concurrency"])
                != max(
                    int(value)
                    for value in protocol["workload"][
                        "active_workflow_concurrency_sweep"
                    ]
                )
                or sum(
                    bool(row.get("in_measurement_window"))
                    for row in artifact["window_rows"]
                )
                >= int(
                    protocol["workload"][
                        "highest_concurrency_minimum_steady_window_samples"
                    ]
                )
                and float(artifact["workload"]["steady_measurement_end_ms"])
                >= float(
                    protocol["workload"][
                        "highest_concurrency_minimum_last_launch_ms"
                    ]
                )
                and sum(
                    float(row.get("time_ms", 0.0)) > 0.0
                    and float(row.get("time_ms", 0.0))
                    <= float(
                        artifact["workload"]["steady_measurement_end_ms"]
                    )
                    + 1e-9
                    for row in artifact["window_rows"]
                )
                >= int(
                    protocol["workload"][
                        "highest_concurrency_minimum_periodic_aimd_updates"
                    ]
                )
            )
        )
        rates = artifact["resolved_config"]["resolved_transfer_rates_gbps"]
        scenario_rate = float(artifact["scenario"]["network_effective_gb_s"])
        resolved_network_pass &= (
            float(rates["network_effective"]) == scenario_rate
            and float(rates["cross_node_source_d2d"])
            == min(25.0, scenario_rate)
            and float(rates["source_d2h_store"]) == scenario_rate
            and float(rates["peer_hbm_load"]) == scenario_rate
            and float(rates["dram_load"]) == scenario_rate
            and rates["shared_nic_contention_modeled"] is False
        )

    selection = dict(index.get("selection") or {})
    selection_canonical = str(selection.get("canonical_sha256", ""))
    selection_payload = dict(selection)
    selection_payload.pop("canonical_sha256", None)
    selection_mode = str(selection.get("mode", "MISSING"))
    indexed_selection_tuples = {
        (str(row[0]), str(row[1]), int(row[2]))
        for row in selection.get("selected_tuples", [])
        if isinstance(row, (list, tuple)) and len(row) == 3
    }
    selection_contract_pass = bool(
        selection_canonical == canonical_json_sha(selection_payload)
        and
        selection_mode
        in {
            "DEFAULT_NINE_RUN_MECHANISM_PILOT",
            "FULL_FROZEN_MATRIX",
            "CUSTOM_PARTIAL_DIAGNOSTIC",
        }
        and int(selection.get("selected_run_count", -1)) == len(observed_ids)
        and indexed_selection_tuples == scenario_policy_seed
    )
    default_pilot_contract_pass = True
    if selection_mode == "DEFAULT_NINE_RUN_MECHANISM_PILOT":
        scenarios = {
            row["scenario_id"]: row for row in matrix["scenarios"]
        }
        central = next(
            row
            for row in scenarios.values()
            if row["model_id"] == "qwen3_8b"
            and row["active_workflow_concurrency"] == 128
            and float(row["failure_rate_per_1000_worker_hours"]) == 5.952
            and row["total_gpu_count"] == 64
            and float(row["prompt_scale"]) == 0.016
            and float(row["network_effective_gb_s"]) == 10.0
            and float(row["hbm_quota_gib_per_target"]) == 4.0
            and float(row["dram_quota_gib_per_target"]) == 64.0
            and float(row["algorithm1_prediction_error_multiplier"]) == 1.0
        )
        fields = (
            "model_id",
            "prompt_scale",
            "active_workflow_concurrency",
            "total_gpu_count",
            "network_effective_gb_s",
            "hbm_quota_gib_per_target",
            "dram_quota_gib_per_target",
            "algorithm1_prediction_error_multiplier",
            "tp_size",
            "pp_size",
            "system_dp_size",
        )
        high_failure = max(
            (
                row
                for row in scenarios.values()
                if all(row[field] == central[field] for field in fields)
            ),
            key=lambda row: float(row["failure_rate_per_1000_worker_hours"]),
        )
        rate_zero = next(
            row
            for row in scenarios.values()
            if all(row[field] == central[field] for field in fields)
            and float(row["failure_rate_per_1000_worker_hours"]) == 0.0
        )
        expected_seed = min(int(seed) for seed in protocol["statistics"]["paired_seeds"])
        expected_tuples = {
            (row["scenario_id"], policy, expected_seed)
            for row in (rate_zero, central, high_failure)
            for policy in {
                "aegiskv_alg111",
                "aegiskv_alg011_no_alg1",
                "recompute_only",
            }
        }
        default_pilot_contract_pass = bool(
            index.get("pilot") is True
            and len(expected_tuples) == 9
            and scenario_policy_seed == expected_tuples
            and float(central["failure_rate_per_1000_worker_hours"]) == 5.952
            and float(rate_zero["failure_rate_per_1000_worker_hours"]) == 0.0
            and float(high_failure["failure_rate_per_1000_worker_hours"])
            == 41.667
            and selection.get("requested_filters")
            == {"scenario_ids": [], "policies": [], "max_runs": None}
            and selection.get("pilot_contract", {}).get("status") == "PASS"
            and selection.get("allowed_analysis_status")
            == "PILOT_DIAGNOSTIC"
        )
    elif selection_mode == "FULL_FROZEN_MATRIX":
        selection_contract_pass &= bool(
            index.get("pilot") is False
            and selection.get("allowed_analysis_status") == "COMPLETE"
        )
    elif selection_mode == "CUSTOM_PARTIAL_DIAGNOSTIC":
        selection_contract_pass &= bool(
            selection.get("allowed_analysis_status")
            == "CUSTOM_PARTIAL_DIAGNOSTIC"
        )

    checks = {
        "canonical_campaign_index_is_zero_failure_complete": (
            index.get("status") == "COMPLETE"
            and index.get("canonical_campaign_index_written") is True
            and int(index.get("failure_count", -1)) == 0
        ),
        "protocol_sha": sha256(Path(index["protocol"]["path"]))
        == index["protocol"]["sha256"],
        "matrix_sha": sha256(Path(index["matrix"]["path"]))
        == index["matrix"]["sha256"],
        "matrix_canonical_sha": index["matrix"]["canonical_sha256"]
        == matrix["canonical_sha256"],
        "producer_source_hashes": all(
            sha256(Path(record["path"])) == record["sha256"]
            for record in index["lineage"]["files"].values()
        ),
        "cache_manifest_internal_refs_and_generator_sha": (
            _cache_bundle_contract(index)
        ),
        "index_pass_count": int(index["pass_count"]) == len(observed_ids),
        "index_failure_count_zero": int(index["failure_count"]) == 0,
        "raw_ids_match_index": observed_ids == set(records),
        "run_tuple_unique": len(scenario_policy_seed) == len(observed_ids),
        "evidence_exact": evidence == {"simulated_sglsim_only"},
        "design_exact": designs == {protocol["design_semantics"]},
        "single_lineage": lineages == {index["lineage"]["lineage_sha256"]},
        "raw_contract": raw_contract_pass,
        "t0_aimd_snapshot_and_periodic_contract": aimd_snapshot_contract_pass,
        "failure_history_raw_summary_final_state_accounting_exact": (
            failure_history_contract_pass
        ),
        "failure_history_horizon_and_post_event_drain_accounting_exact": (
            failure_history_contract_pass
        ),
        "complete_raw_retention": full_retention_pass,
        "policy_independent_offered_schedule_contract": schedule_contract_pass,
        "projected_workload_hash_identical_across_policies": all(
            len(values) == 1 for values in schedule_by_scenario_seed.values()
        ),
        "failure_tuple_schedule_identical_across_policies": all(
            len(values) == 1 for values in failures_by_scenario_seed.values()
        ),
        "active_source_probe_raw_contract": probe_contract_pass,
        "active_source_probe_tuple_identical_across_policy_and_rate": all(
            len(values) == 1 for values in probe_tuple_by_match.values()
        ),
        "active_source_probe_universe_identical_across_policy_and_rate": all(
            len(values) == 1 for values in probe_universe_by_match.values()
        ),
        "active_source_probe_prng_identical_across_policy_and_rate": all(
            len(values) == 1 for values in probe_prng_by_match.values()
        ),
        "active_source_probe_exposure_identical_across_policy_and_rate": all(
            len(values) == 1
            and all(value > 0.0 for value in values)
            for values in probe_exposure_by_match.values()
        ),
        "active_source_probe_output_retention_preregistered": protocol[
            "output"
        ].get("retain_active_source_probe_rows_per_run")
        == "all_64",
        "reuse_bin_contract_matches_core": {
            name: protocol["fault_estimands"]["reuse_bin_contract"][name]
            for name in REUSE_BIN_CONTRACT
        }
        == REUSE_BIN_CONTRACT,
        "reuse_bin_fraction_and_preregistration_explicit": protocol[
            "fault_estimands"
        ]["reuse_bin_contract"].get("selected_before_results")
        is True
        and "clip"
        in str(
            protocol["fault_estimands"]["reuse_bin_contract"].get(
                "fraction", ""
            )
        ),
        "resolved_network_paths_operational": resolved_network_pass,
        "selection_provenance_matches_raw_tuples": selection_contract_pass,
        "default_pilot_exact_nine_tuple_contract": (
            default_pilot_contract_pass
        ),
    }
    if require_full:
        expected_seeds = set(protocol["statistics"]["paired_seeds"])
        seed_sets = {
            (scenario["scenario_id"], policy): {
                seed
                for observed_scenario, observed_policy, seed in scenario_policy_seed
                if observed_scenario == scenario["scenario_id"]
                and observed_policy == policy
            }
            for scenario in matrix["scenarios"]
            for policy in scenario["policies"]
        }
        checks.update(
            {
                "exact_3110": len(observed_ids) == 3110,
                "exact_matrix_run_ids": observed_ids == expected_matrix_ids,
                "exact_ten_paired_seeds_per_policy_scenario": all(
                    seeds == expected_seeds for seeds in seed_sets.values()
                ),
                "full_selection_mode_exact": (
                    selection_mode == "FULL_FROZEN_MATRIX"
                ),
            }
        )
    detail = {
        "campaign_index": str(index_path),
        "campaign_index_sha256": sha256(index_path),
        "pilot": bool(index["pilot"]),
        "selection_mode": selection_mode,
        "raw_artifacts": artifact_hashes,
        "observed_run_count": len(observed_ids),
        "matrix_run_count": matrix["run_count"],
    }
    return detail, checks


def audit_analysis(
    analysis_dir: Path,
    campaign_index_sha: str,
    *,
    selection_mode: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    path = analysis_dir / "analysis_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_status = {
        "DEFAULT_NINE_RUN_MECHANISM_PILOT": "PILOT_DIAGNOSTIC",
        "FULL_FROZEN_MATRIX": "COMPLETE",
        "CUSTOM_PARTIAL_DIAGNOSTIC": "CUSTOM_PARTIAL_DIAGNOSTIC",
    }.get(selection_mode, "INVALID_SELECTION_MODE")
    formal = selection_mode != "CUSTOM_PARTIAL_DIAGNOSTIC"
    expected_scenario_count = {
        "DEFAULT_NINE_RUN_MECHANISM_PILOT": 3,
        "FULL_FROZEN_MATRIX": 33,
    }.get(selection_mode)
    paper_summary_path = analysis_dir / "paper_summary.json"
    paper_summary = (
        json.loads(paper_summary_path.read_text(encoding="utf-8"))
        if paper_summary_path.is_file()
        else {}
    )
    checks = {
        "analysis_status_exact_for_campaign_mode": (
            manifest["status"] == expected_status
        ),
        "analysis_simulated_only": manifest["evidence_class"]
        == "simulated_sglsim_only_analysis"
        and manifest["physical_result"] is False,
        "analysis_binds_campaign_index": manifest["campaign_index"]["sha256"]
        == campaign_index_sha,
        "analysis_files_match": verify_manifest_files(
            analysis_dir, manifest["files"]
        ),
        "analysis_sources_match": all(
            sha256(Path(record["path"])) == record["sha256"]
            for record in manifest["analysis_sources"].values()
        ),
        "analysis_raw_hashes_complete": len(manifest["raw_artifacts"])
        == manifest["observed_run_count"],
        "analysis_contract_pass": manifest["contract_status"] == "PASS",
        "analysis_rationality_matches_mode": (not formal)
        or manifest["rationality_status"] == "PASS",
        "analysis_algorithm1_identifiability_matches_mode": (not formal)
        or manifest["algorithm1_identifiability_status"] == "PASS",
        "analysis_paired_comparison_coverage_matches_mode": (
            manifest["paired_comparison_coverage_status"] == "PASS"
            if formal
            else manifest["paired_comparison_coverage_status"]
            == "NOT_APPLICABLE_CUSTOM_PARTIAL"
        ),
        "analysis_paired_probe_comparison_coverage_matches_mode": (
            manifest.get("paired_probe_comparison_coverage_status") == "PASS"
            if formal
            else manifest.get("paired_probe_comparison_coverage_status")
            == "NOT_APPLICABLE_CUSTOM_PARTIAL"
        ),
        "analysis_active_source_probe_contract_pass": manifest.get(
            "active_source_probe_contract_status"
        )
        == "PASS",
        "analysis_probe_inner_bootstrap_forbidden": manifest.get(
            "probe_inner_bootstrap_used"
        )
        is False,
        "analysis_scenario_diagnostics_pass": (not formal)
        or manifest.get("scenario_diagnostics_status") == "PASS",
        "analysis_scenario_diagnostics_count_exact": (
            expected_scenario_count is None
            or int(manifest.get("scenario_diagnostics_row_count", -1))
            == expected_scenario_count
        ),
        "analysis_paper_summary_bound": "paper_summary.json"
        in manifest.get("files", {}),
        "analysis_paper_summary_evidence_boundary_exact": bool(paper_summary)
        and paper_summary.get("analysis_status") == expected_status
        and paper_summary.get("selection_mode") == selection_mode
        and paper_summary.get("paper_eligible")
        is (expected_status == "COMPLETE"),
        "analysis_paper_summary_records_bind_shared_source_hashes": bool(
            paper_summary.get("records")
        )
        and all(
            row.get("source_hashes") == paper_summary.get("source_hashes")
            for row in paper_summary["records"]
        ),
        "analysis_paper_eligibility_exact": manifest.get("paper_eligible")
        is (expected_status == "COMPLETE")
        and manifest.get("selection_mode") == selection_mode,
    }
    return {"path": str(path), "sha256": sha256(path)}, checks


def audit_figures(figures_dir: Path, analysis_manifest_sha: str) -> tuple[dict[str, Any], dict[str, bool]]:
    path = figures_dir / "figure_manifest.json"
    manifest_canonical_ok, manifest = _canonical_manifest_matches(path)
    prefigure_record = manifest.get("prefigure_full_audit", {})
    prefigure_path = Path(str(prefigure_record.get("path", "")))
    try:
        prefigure_ok, prefigure = _canonical_manifest_matches(prefigure_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        prefigure_ok, prefigure = False, {}
    checks = {
        "figure_manifest_canonical": manifest_canonical_ok,
        "figures_simulated_only": manifest["evidence_class"]
        == "simulated_sglsim_only_figures"
        and manifest["physical_result"] is False,
        "figures_bind_analysis": manifest["analysis_manifest"]["sha256"]
        == analysis_manifest_sha,
        "figures_bind_passed_full_prefigure_audit": prefigure_ok
        and prefigure_record.get("sha256") == sha256(prefigure_path)
        and prefigure_record.get("canonical_sha256")
        == prefigure.get("canonical_sha256")
        and prefigure.get("status") == "PASS"
        and prefigure.get("require_full") is True
        and prefigure.get("failed_checks") == []
        and PREFIGURE_REQUIRED_CHECKS.issubset(prefigure.get("checks", {}))
        and all(bool(value) for value in prefigure.get("checks", {}).values())
        and prefigure.get("figures") is None
        and prefigure.get("analysis", {}).get("sha256")
        == analysis_manifest_sha
        and prefigure.get("auditor", {}).get("sha256")
        == sha256(Path(__file__).resolve()),
        "figure_evidence_status_explicit": manifest.get("analysis_status")
        == "COMPLETE"
        and manifest.get("paper_eligible") is True
        and manifest.get("selection_mode") == "FULL_FROZEN_MATRIX",
        "figure_files_match": verify_manifest_files(figures_dir, manifest["files"]),
        "both_pdf_png_per_figure": {
            Path(name).stem for name in manifest["files"] if name.endswith(".pdf")
        }
        == {
            Path(name).stem for name in manifest["files"] if name.endswith(".png")
        },
    }
    return {"path": str(path), "sha256": sha256(path)}, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign_dir = args.campaign_dir.resolve()
    detail, checks = audit_raw(campaign_dir, require_full=args.require_full)
    analysis_detail = None
    if args.analysis_dir:
        analysis_detail, analysis_checks = audit_analysis(
            args.analysis_dir.resolve(),
            detail["campaign_index_sha256"],
            selection_mode=str(detail["selection_mode"]),
        )
        checks.update(analysis_checks)
    figure_detail = None
    if args.figures_dir:
        if not analysis_detail:
            raise ValueError("--figures-dir requires --analysis-dir")
        figure_detail, figure_checks = audit_figures(
            args.figures_dir.resolve(), analysis_detail["sha256"]
        )
        checks.update(figure_checks)
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    payload = {
        "schema_version": 1,
        "evidence_class": "simulated_sglsim_only_campaign_audit",
        "status": "PASS" if not failed else "FAIL",
        "physical_result": False,
        "require_full": bool(args.require_full),
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "checks": checks,
        "failed_checks": failed,
        "campaign": detail,
        "analysis": analysis_detail,
        "figures": figure_detail,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") != serialized:
                raise FileExistsError(
                    f"refusing to overwrite immutable audit: {args.output}"
                )
        else:
            args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
