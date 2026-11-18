#!/usr/bin/env python3
"""Recompute campaign statistics from immutable raw simulation artifacts.

All workflow comparisons are rebuilt from the complete 393-row workflow
tables. Workflow harm and fault exposure cover the full run; Algorithm 3/resource
telemetry uses only the policy-independent offered-load steady interval.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sglsim.srt.simulation.aegis_large_scale import REUSE_BIN_CONTRACT


ROOT = Path(__file__).resolve().parent
EXPECTED_EVIDENCE = "simulated_sglsim_only"
EXPECTED_DESIGN = "source_owned_independent_hbm_dram_v2"
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
HAZARD_MATCH_FIELDS = (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_manifest_matches(manifest: dict[str, Any]) -> bool:
    payload = dict(manifest)
    expected = str(payload.pop("canonical_sha256", ""))
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return bool(expected) and expected == actual


def _lineage_file(index: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    record = index["lineage"]["files"][name]
    path = Path(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"campaign lineage file changed: {name}")
    return path, record


def load_lineage_cache_manifests(
    index: dict[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Load predictor/trace diagnostics only from the producer-bound lineage."""

    compact_path, compact_record = _lineage_file(index, "compact_manifest")
    modeled_path, modeled_record = _lineage_file(
        index, "modeled_algorithm1_manifest"
    )
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    modeled = json.loads(modeled_path.read_text(encoding="utf-8"))
    if not _canonical_manifest_matches(compact):
        raise ValueError("lineage compact manifest canonical SHA mismatch")
    if not _canonical_manifest_matches(modeled):
        raise ValueError("lineage Algorithm-1 manifest canonical SHA mismatch")
    if int(modeled.get("schema_version", 0)) != 2:
        raise ValueError("lineage Algorithm-1 manifest schema mismatch")

    compact_payloads = {
        **dict(compact.get("arrays") or {}),
        "workflow_reuse": compact.get("workflow_reuse"),
    }
    for name, manifest_record in compact_payloads.items():
        payload_path, lineage_record = _lineage_file(
            index, f"compact_cache:{name}"
        )
        if not isinstance(manifest_record, dict) or (
            lineage_record["sha256"] != manifest_record.get("sha256")
            or payload_path.resolve()
            != (compact_path.parent / str(manifest_record.get("path"))).resolve()
        ):
            raise ValueError(f"compact cache lineage mismatch: {name}")
    for name, manifest_record in dict(modeled.get("arrays") or {}).items():
        payload_path, lineage_record = _lineage_file(
            index, f"modeled_algorithm1_cache:{name}"
        )
        if (
            lineage_record["sha256"] != manifest_record.get("sha256")
            or payload_path.resolve()
            != (modeled_path.parent / str(manifest_record.get("path"))).resolve()
        ):
            raise ValueError(f"Algorithm-1 cache lineage mismatch: {name}")

    _producer_path, producer = _lineage_file(
        index, "modeled_algorithm1_generator"
    )
    _protocol_path, protocol = _lineage_file(index, "protocol")
    _trace_generator_path, trace_generator = _lineage_file(
        index, "compact_trace_generator"
    )
    if not trace_generator.get("sha256"):
        raise ValueError("compact trace generator is not hash-bound")
    if modeled.get("producer", {}).get("sha256") != producer["sha256"]:
        raise ValueError("Algorithm-1 producer lineage mismatch")
    if modeled.get("protocol", {}).get("sha256") != protocol["sha256"]:
        raise ValueError("Algorithm-1 protocol lineage mismatch")
    if (
        modeled.get("compact_workload", {}).get("sha256")
        != compact_record["sha256"]
        or modeled.get("compact_workload", {}).get("canonical_sha256")
        != compact["canonical_sha256"]
    ):
        raise ValueError("Algorithm-1 compact-workload lineage mismatch")
    if modeled_record["sha256"] != sha256(modeled_path):
        raise ValueError("Algorithm-1 manifest lineage mismatch")
    if modeled.get("mechanism_gate", {}).get("passed") is not True:
        raise ValueError("Algorithm-1 mechanism gate is not PASS")
    if modeled.get("policy_input", {}).get("array") != "predicted_slack_ms":
        raise ValueError("Algorithm-1 policy input lineage mismatch")
    return compact_path, compact, modeled_path, modeled


def flatten(prefix: str, value: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            output.update(flatten(name + ".", item))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            output[name] = item
    return output


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


def json_native(value: Any) -> Any:
    """Recursively convert only known numpy/pandas scalars to strict JSON."""

    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_native(item) for item in sorted(value, key=repr)]
    return value


def strict_json_dumps(value: Any, **kwargs: Any) -> str:
    """Serialize with explicit scalar normalization and no NaN/string fallback."""

    return json.dumps(json_native(value), allow_nan=False, **kwargs)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a deterministic pipe table without pandas' tabulate extra."""

    def cell(value: Any) -> str:
        if value is None or (
            isinstance(value, (float, np.floating)) and math.isnan(float(value))
        ):
            text = "NA"
        elif isinstance(value, (float, np.floating)):
            text = format(float(value), ".8g")
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\r\n", "<br>").replace(
            "\n", "<br>"
        )

    headers = [cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def probe_tuple_payload(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    fields = tuple(artifact["active_source_probe_metadata"]["probe_tuple_fields"])
    if fields != PROBE_TUPLE_FIELDS:
        raise ValueError(f"probe tuple field contract drift: {fields}")
    rows = sorted(
        artifact["active_source_probe_rows"],
        key=lambda row: int(row["stratum_index"]),
    )
    return [{field: row[field] for field in fields} for row in rows]


def probe_group_key(artifact: dict[str, Any]) -> tuple[Any, ...]:
    scenario = artifact["scenario"]
    return (
        int(artifact["seed"]),
        *(scenario[field] for field in HAZARD_MATCH_FIELDS),
    )


def audit_probe_contract(
    artifacts: list[tuple[Path, dict[str, Any]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    expected_bins = dict(REUSE_BIN_CONTRACT)
    protocol_reuse = protocol["fault_estimands"]["reuse_bin_contract"]
    protocol_bins = {
        name: protocol_reuse[name] for name in REUSE_BIN_CONTRACT
    }
    per_artifact: list[dict[str, Any]] = []
    tuple_hashes: dict[tuple[Any, ...], set[str]] = {}
    universe_hashes: dict[tuple[Any, ...], set[str]] = {}
    prng_hashes: dict[tuple[Any, ...], set[str]] = {}
    exposures: dict[tuple[Any, ...], set[float]] = {}
    for _path, artifact in artifacts:
        metadata = artifact.get("active_source_probe_metadata", {})
        rows = artifact.get("active_source_probe_rows", [])
        nested = artifact.get("active_source_fault_conditioned_recovery", {})
        tuple_payload = probe_tuple_payload(artifact)
        digest = canonical_json_sha(tuple_payload)
        indices = sorted(int(row["stratum_index"]) for row in rows)
        checks = {
            "exact_k64": (
                int(metadata.get("requested_probe_count", -1)) == PROBE_COUNT
                and int(metadata.get("realized_probe_count", -1)) == PROBE_COUNT
                and int(artifact.get("active_source_probe_row_count", -1))
                == PROBE_COUNT
                and len(rows) == PROBE_COUNT
            ),
            "all_rows_retained": artifact.get("raw_retention", {}).get(
                "full_active_source_probe_rows_retained"
            )
            is True,
            "top_level_nested_rows_identical": nested.get("probe_rows") == rows,
            "tuple_fields_exact": tuple(
                metadata.get("probe_tuple_fields", ())
            )
            == PROBE_TUPLE_FIELDS,
            "tuple_hash_recomputed": digest
            == metadata.get("probe_tuple_sha256"),
            "one_row_per_equal_exposure_stratum": indices
            == list(range(PROBE_COUNT)),
            "active_and_sealed_evidence": all(
                row.get("eligible_active_source") is True
                and row.get("eligible_complete_sealed_frontier") is True
                and int(row.get("sealed_frontier_tokens", 0)) > 0
                and float(row["eligible_interval_start_ms"])
                <= float(row["time_ms"])
                < float(row["eligible_interval_end_ms"])
                for row in rows
            ),
            "query_non_mutating": all(
                row.get("primary_counters_updated") is False for row in rows
            ),
            "selection_does_not_read_policy_outcome": all(
                row.get("eligibility_uses_policy_state") is False
                and row.get(
                    "eligibility_filters_admission_or_recovery_outcome"
                )
                is False
                for row in rows
            ),
            "all_required_core_gates": all(
                artifact.get("gates", {}).get(name) is True
                for name in PROBE_GATES
            ),
            "outer_seed_only_ci_contract": (
                nested.get("probe_inner_bootstrap_allowed") is False
                and nested.get("formal_ci_unit")
                == "paired_seed_mean_over_64_strata"
            ),
            "reuse_bin_contract_exact": (
                artifact["summary"].get("workload_reuse_bin_contract")
                == expected_bins
                and artifact["summary"].get(
                    "active_source_probe_reuse_bin_contract"
                )
                == expected_bins
                and nested.get("reuse_analysis_contract", {}).get("bins")
                == expected_bins
            ),
        }
        per_artifact.append(
            {
                "run_id": artifact["run_id"],
                "status": "PASS" if all(checks.values()) else "FAIL",
                "checks": checks,
            }
        )
        key = probe_group_key(artifact)
        tuple_hashes.setdefault(key, set()).add(digest)
        universe_hashes.setdefault(key, set()).add(
            str(metadata.get("eligible_exposure_universe_sha256"))
        )
        prng_hashes.setdefault(key, set()).add(
            str(metadata.get("sampling_prng_key_sha256"))
        )
        exposures.setdefault(key, set()).add(
            float(metadata.get("eligible_exposure_gpu_shard_hours", math.nan))
        )
    cross_checks = {
        "protocol_reuse_bins_equal_core": protocol_bins == expected_bins,
        "protocol_reuse_fraction_and_preregistration_explicit": (
            protocol_reuse.get("selected_before_results") is True
            and "clip" in str(protocol_reuse.get("fraction", ""))
        ),
        "probe_tuple_hash_same_across_policy_and_failure_rate": all(
            len(values) == 1 for values in tuple_hashes.values()
        ),
        "eligible_universe_same_across_policy_and_failure_rate": all(
            len(values) == 1 for values in universe_hashes.values()
        ),
        "sampling_prng_same_across_policy_and_failure_rate": all(
            len(values) == 1 for values in prng_hashes.values()
        ),
        "eligible_exposure_same_across_policy_and_failure_rate": all(
            len(values) == 1 and all(math.isfinite(value) and value > 0 for value in values)
            for values in exposures.values()
        ),
    }
    status = all(row["status"] == "PASS" for row in per_artifact) and all(
        cross_checks.values()
    )
    return {
        "status": "PASS" if status else "FAIL",
        "probe_count_per_run": PROBE_COUNT,
        "formal_ci_unit": "paired seed mean over 64 strata; outer bootstrap only",
        "inner_probe_bootstrap_forbidden": True,
        "cross_artifact_checks": cross_checks,
        "per_artifact": per_artifact,
    }


def load_campaign(
    input_dir: Path,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    index_path = input_dir / "campaign-index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing campaign index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("status") != "COMPLETE"
        or index.get("canonical_campaign_index_written") is not True
        or int(index.get("failure_count", -1)) != 0
    ):
        raise ValueError("campaign index is not a zero-failure canonical completion")
    load_lineage_cache_manifests(index)
    if sha256(Path(index["protocol"]["path"])) != index["protocol"]["sha256"]:
        raise ValueError("campaign protocol hash no longer matches its index")
    if sha256(Path(index["matrix"]["path"])) != index["matrix"]["sha256"]:
        raise ValueError("campaign matrix hash no longer matches its index")
    lineage_sha = index["lineage"]["lineage_sha256"]
    records = {row["run_id"]: row for row in index["records"]}
    artifacts: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((input_dir / "raw").glob("**/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            artifact = json.load(stream)
        run_id = artifact.get("run_id")
        if run_id not in records:
            raise ValueError(f"unindexed raw artifact: {path}")
        if sha256(path) != records[run_id].get("sha256"):
            raise ValueError(f"raw artifact SHA mismatch: {path}")
        if artifact.get("lineage", {}).get("lineage_sha256") != lineage_sha:
            raise ValueError(f"mixed producer lineage: {path}")
        artifacts.append((path, artifact))
    if len(artifacts) != int(index["pass_count"]):
        raise ValueError("campaign index/raw artifact cardinality mismatch")
    return index, artifacts


def aimd_snapshot_contract_checks(
    artifact: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, bool]:
    """Re-audit t=0 snapshot and positive-time AIMD markings from raw rows."""

    rows = list(artifact.get("window_rows") or [])
    initial = [row for row in rows if bool(row.get("initial_sample"))]
    algorithm3_enabled = bool(
        initial and initial[0].get("algorithm3_enabled")
    )
    initial_exact = bool(
        len(initial) == 1
        and math.isclose(float(initial[0].get("time_ms", math.nan)), 0.0, abs_tol=1e-12)
        and initial[0].get("aimd_update_applied") is False
        and (
            not algorithm3_enabled
            or math.isclose(
                float(initial[0].get("controller_alpha_mean", math.nan)),
                float(protocol["algorithm3"]["initial_alpha"]),
                abs_tol=1e-12,
            )
        )
    )
    stable_counter_unchanged = bool(
        initial_exact
        and (
            not algorithm3_enabled
            or (
                math.isclose(
                    float(initial[0].get("controller_stable_windows_mean", math.nan)),
                    0.0,
                    abs_tol=1e-12,
                )
                and int(initial[0].get("controller_stable_windows_max", -1)) == 0
            )
        )
    )
    update_markings_exact = bool(
        all(
            bool(row.get("aimd_update_applied"))
            == bool(algorithm3_enabled and float(row.get("time_ms", 0.0)) > 0.0)
            for row in rows
        )
    )
    applied_rows = [row for row in rows if bool(row.get("aimd_update_applied"))]
    summary_counts_exact = bool(
        int(artifact.get("summary", {}).get("initial_snapshot_sample_count", -1))
        == len(initial)
        and int(artifact.get("summary", {}).get("aimd_periodic_updates_applied", -1))
        == len(applied_rows)
        and artifact.get("summary", {}).get("first_aimd_periodic_update_ms")
        == (float(applied_rows[0]["time_ms"]) if applied_rows else None)
    )
    gates = artifact.get("gates") or {}
    core_gates_exact = bool(
        gates.get("t0_control_window_is_snapshot_only") is True
        and gates.get("aimd_updates_only_at_positive_periodic_windows") is True
        and gates.get("initial_sample_does_not_advance_stable_counter") is True
    )
    highest_concurrency = int(artifact["scenario"]["active_workflow_concurrency"]) == max(
        int(value)
        for value in protocol["workload"]["active_workflow_concurrency_sweep"]
    )
    periodic_opportunity_gate = True
    if highest_concurrency:
        horizon_ms = float(artifact["workload"]["steady_measurement_end_ms"])
        opportunities = [
            row
            for row in rows
            if 0.0 < float(row.get("time_ms", 0.0)) <= horizon_ms + 1e-9
        ]
        applied_before_horizon = [
            row
            for row in applied_rows
            if float(row.get("time_ms", 0.0)) <= horizon_ms + 1e-9
        ]
        minimum = int(
            protocol["workload"][
                "highest_concurrency_minimum_periodic_aimd_updates"
            ]
        )
        periodic_opportunity_gate = bool(
            len(opportunities) >= minimum
            and (
                len(applied_before_horizon) >= minimum
                if algorithm3_enabled
                else len(applied_before_horizon) == 0
            )
        )
    return {
        "t0_control_window_snapshot_exact": initial_exact,
        "initial_sample_stable_counter_unchanged": stable_counter_unchanged,
        "aimd_updates_only_at_positive_periodic_windows": update_markings_exact,
        "aimd_summary_counts_exact": summary_counts_exact,
        "core_t0_aimd_gates_exact": core_gates_exact,
        "highest_concurrency_periodic_aimd_opportunities_exact": periodic_opportunity_gate,
    }


def audit_artifact_contract(
    index: dict[str, Any],
    artifacts: list[tuple[Path, dict[str, Any]]],
    matrix: dict[str, Any],
    protocol: dict[str, Any],
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    probe_contract = audit_probe_contract(artifacts, protocol)
    aimd_contracts = [
        aimd_snapshot_contract_checks(artifact, protocol)
        for _path, artifact in artifacts
    ]
    expected_index_ids = {row["run_id"] for row in index["records"]}
    observed_ids = {artifact["run_id"] for _path, artifact in artifacts}
    observed_tuples = {
        (
            artifact["scenario"]["scenario_id"],
            artifact["policy"],
            int(artifact["seed"]),
        )
        for _path, artifact in artifacts
    }
    selection = dict(index.get("selection") or {})
    selection_canonical = str(selection.get("canonical_sha256", ""))
    selection_payload = dict(selection)
    selection_payload.pop("canonical_sha256", None)
    selection_mode = str(selection.get("mode", "MISSING"))
    selection_tuples = {
        (str(row[0]), str(row[1]), int(row[2]))
        for row in selection.get("selected_tuples", [])
        if isinstance(row, (list, tuple)) and len(row) == 3
    }
    selection_contract = bool(
        selection_canonical == canonical_json_sha(selection_payload)
        and
        selection_mode
        in {
            "DEFAULT_NINE_RUN_MECHANISM_PILOT",
            "FULL_FROZEN_MATRIX",
            "CUSTOM_PARTIAL_DIAGNOSTIC",
        }
        and selection_tuples == observed_tuples
        and int(selection.get("selected_run_count", -1)) == len(artifacts)
        and (
            selection_mode != "DEFAULT_NINE_RUN_MECHANISM_PILOT"
            or (
                len(observed_tuples) == 9
                and selection.get("pilot_contract", {}).get("status")
                == "PASS"
                and selection.get("allowed_analysis_status")
                == "PILOT_DIAGNOSTIC"
            )
        )
    )
    checks: dict[str, Any] = {
        "index_records_unique": len(expected_index_ids) == len(index["records"]),
        "raw_run_ids_exactly_match_index": observed_ids == expected_index_ids,
        "index_has_no_failed_runs": int(index["failure_count"]) == 0,
        "selection_provenance_matches_raw": selection_contract,
        "producer_lineage_single": len(
            {
                artifact["lineage"]["lineage_sha256"]
                for _path, artifact in artifacts
            }
        )
        == 1,
        "evidence_class_exact": all(
            artifact.get("evidence_class") == EXPECTED_EVIDENCE
            for _path, artifact in artifacts
        ),
        "simulator_cell_schema_v2_exact": all(
            artifact.get("simulator_cell_schema_version")
            == "aegiskv_sglsim_large_scale_cell_v2"
            for _path, artifact in artifacts
        ),
        "design_semantics_exact": all(
            artifact.get("design_semantics") == EXPECTED_DESIGN
            for _path, artifact in artifacts
        ),
        "t0_control_window_snapshot_exact": all(
            row["t0_control_window_snapshot_exact"] for row in aimd_contracts
        ),
        "initial_sample_stable_counter_unchanged": all(
            row["initial_sample_stable_counter_unchanged"]
            for row in aimd_contracts
        ),
        "aimd_updates_only_at_positive_periodic_windows": all(
            row["aimd_updates_only_at_positive_periodic_windows"]
            for row in aimd_contracts
        ),
        "aimd_summary_counts_exact": all(
            row["aimd_summary_counts_exact"] for row in aimd_contracts
        ),
        "core_t0_aimd_gates_exact": all(
            row["core_t0_aimd_gates_exact"] for row in aimd_contracts
        ),
        "highest_concurrency_periodic_aimd_opportunities_exact": all(
            row["highest_concurrency_periodic_aimd_opportunities_exact"]
            for row in aimd_contracts
        ),
        "simulator_gates_exact": all(
            artifact.get("gates", {}).get("simulated_only_label_present") is True
            and artifact["gates"].get("fault_process_covers_full_run_horizon") is True
            and artifact["gates"].get(
                "measurement_window_only_filters_steady_telemetry"
            )
            is True
            and artifact["gates"].get("alpha_pressure_only") is True
            and artifact["gates"].get(
                "failure_rate_direct_alpha_dependency_absent"
            )
            is True
            and artifact["gates"].get(
                "input_reuse_does_not_deduplicate_protection_copy"
            )
            is True
            and artifact["gates"].get(
                "single_worker_shard_failure_counterfactual"
            )
            is True
            and artifact["gates"].get("executed_physical_fail_stop") is False
            and artifact["gates"].get(
                "failure_queue_outage_replacement_rescheduling_modeled"
            )
            is False
            and artifact["gates"].get(
                "failed_worker_peer_hbm_invalidated_for_all_policies"
            )
            is True
            and artifact["gates"].get("explicit_worker_repair_event_modeled")
            is True
            and artifact["gates"].get(
                "repair_reenables_future_hbm_admission_only"
            )
            is True
            and artifact["gates"].get(
                "repair_does_not_restore_old_hbm_copies"
            )
            is True
            and artifact["gates"].get(
                "repair_does_not_schedule_hbm_reprotection"
            )
            is True
            and artifact["gates"].get("ground_truth_harm_uses_actual_slack")
            is True
            and artifact["gates"].get(
                "policy_predicted_slack_not_used_as_ground_truth"
            )
            is True
            and artifact["gates"].get(
                "workflow_actual_harm_surrogate_aggregated"
            )
            is True
            and artifact["gates"].get("end_to_end_dag_failure_execution_replay")
            is False
            and artifact["gates"].get(
                "formal_analysis_uses_explicit_actual_harm_surrogate_fields"
            )
            is True
            and artifact["gates"].get(
                "legacy_wct_aliases_forbidden_for_formal_claims"
            )
            is True
            and artifact["formal_metric_contract"][
                "primary_failure_endpoint_prefix"
            ]
            == "actual_failure_added_workflow_harm_surrogate_"
            for _path, artifact in artifacts
        ),
        "forced_target_and_no_rotation_evidence_boundary_exact": (
            protocol["failure"]["forced_or_synthetic_target_first_injection"]
            is False
            and protocol["failure"]["standby_rotation_modeled"] is False
            and protocol["failure"][
                "complete_current_design_target_failover_evaluated"
            ]
            is False
            and protocol["failure"][
                "target_failure_history_is_no_standby_rotation_conservative_sensitivity"
            ]
            is True
            and all(
                artifact["gates"].get(
                    "forced_synthetic_target_first_injection_disabled"
                )
                is True
                and artifact["gates"].get("standby_rotation_modeled") is False
                and artifact["gates"].get(
                    "complete_current_design_target_failover_evaluated"
                )
                is False
                and artifact["gates"].get(
                    "target_failure_history_is_no_standby_rotation_conservative_sensitivity"
                )
                is True
                and artifact["formal_metric_contract"][
                    "failure_history_contract"
                ]["standby_rotation_modeled"]
                is False
                and all(
                    row.get("compound_target_first_failure_injected") is False
                    for row in artifact["failure_rows"]
                )
                for _path, artifact in artifacts
            )
        ),
        "failure_history_raw_summary_final_state_accounting_exact": all(
            uptime_history_diagnostics(artifact)[
                "history_accounting_matches_raw_summary_and_final_state"
            ]
            is True
            for _path, artifact in artifacts
        ),
        "failure_history_horizon_and_post_event_drain_accounting_exact": all(
            uptime_history_diagnostics(artifact)[
                "history_accounting_matches_raw_summary_and_final_state"
            ]
            is True
            for _path, artifact in artifacts
        ),
        "all_393_workflows_retained": all(
            len(artifact.get("workflow_rows", [])) == 393
            for _path, artifact in artifacts
        ),
        "offered_concurrency_gate_uses_protocol_threshold": all(
            float(
                artifact["workload"][
                    "achieved_offered_trace_steady_concurrency_relative_error"
                ]
            )
            <= float(
                protocol["workload"][
                    "achieved_steady_concurrency_relative_error_gate"
                ]
            )
            + 1e-12
            for _path, artifact in artifacts
        ),
        "full_workflow_aggregate_count_exact_393": (
            int(protocol["workload"]["wct_measurement_workflows"]) == 393
            and all(
                int(artifact["summary"]["workflow_count"]) == 393
                for _path, artifact in artifacts
            )
        ),
        "highest_concurrency_minimum_steady_samples": all(
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
            for _path, artifact in artifacts
        ),
        "highest_concurrency_minimum_update_horizon": all(
            int(artifact["scenario"]["active_workflow_concurrency"])
            != max(
                int(value)
                for value in protocol["workload"][
                    "active_workflow_concurrency_sweep"
                ]
            )
            or float(artifact["workload"]["steady_measurement_end_ms"])
            >= float(
                protocol["workload"][
                    "highest_concurrency_minimum_last_launch_ms"
                ]
            )
            for _path, artifact in artifacts
        ),
        "highest_concurrency_minimum_periodic_aimd_updates": all(
            int(artifact["scenario"]["active_workflow_concurrency"])
            != max(
                int(value)
                for value in protocol["workload"][
                    "active_workflow_concurrency_sweep"
                ]
            )
            or sum(
                float(row.get("time_ms", 0.0)) > 0.0
                and float(row.get("time_ms", 0.0))
                <= float(artifact["workload"]["steady_measurement_end_ms"])
                + 1e-9
                for row in artifact["window_rows"]
            )
            >= int(
                protocol["workload"][
                    "highest_concurrency_minimum_periodic_aimd_updates"
                ]
            )
            for _path, artifact in artifacts
        ),
        "all_windows_retained": all(
            len(artifact.get("window_rows", []))
            == int(artifact.get("window_row_count", -1))
            for _path, artifact in artifacts
        ),
        "all_failures_retained": all(
            len(artifact.get("failure_rows", []))
            == int(artifact.get("failure_row_count", -1))
            for _path, artifact in artifacts
        ),
        "all_repairs_retained": all(
            len(artifact.get("repair_rows", []))
            == int(artifact.get("repair_row_count", -1))
            for _path, artifact in artifacts
        ),
        "active_source_probe_contract_pass": probe_contract["status"]
        == "PASS",
    }
    if not bool(index["pilot"]) and not allow_partial:
        matrix_ids = {row["run_id"] for row in matrix["runs"]}
        checks.update(
            {
                "full_exact_3110_runs": len(observed_ids) == 3110,
                "full_run_ids_exact_matrix": observed_ids == matrix_ids,
                "full_ten_paired_seeds": all(
                    len(
                        {
                            artifact["seed"]
                            for _path, artifact in artifacts
                            if artifact["scenario"]["scenario_id"]
                            == scenario["scenario_id"]
                            and artifact["policy"] == policy
                        }
                    )
                    == 10
                    for scenario in matrix["scenarios"]
                    for policy in scenario["policies"]
                ),
            }
        )
    required = sorted(checks)
    return {
        "status": "PASS" if all(bool(checks[key]) for key in required) else "FAIL",
        "checks": checks,
        "required_checks": required,
        "active_source_probe_contract": probe_contract,
    }


def steady_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in artifact["window_rows"]
        if bool(row.get("in_measurement_window"))
    ]
    if not rows:
        return {
            "sample_count": 0,
            "effective_alpha_mean": 0.0,
            "effective_alpha_min": 0.0,
            "effective_alpha_max": 0.0,
            "effective_alpha_final": 0.0,
            "pressure_fraction_mean": 0.0,
            "pressure_fraction_peak": 0.0,
            "hbm_used_bytes_mean": 0.0,
            "hbm_used_bytes_peak": 0,
            "dram_used_bytes_mean": 0.0,
            "dram_used_bytes_peak": 0,
            "stable_alpha_increase_transitions": 0,
            "pressure_alpha_decrease_transitions": 0,
            "alpha_zero_window_fraction": 0.0,
        }
    alpha = np.asarray([float(row["effective_alpha_mean"]) for row in rows])
    pressure = np.asarray([float(row["pressure_fraction"]) for row in rows])
    hbm = np.asarray([float(row["hbm_used_bytes"]) for row in rows])
    dram = np.asarray([float(row["dram_used_bytes"]) for row in rows])
    delta = np.diff(alpha)
    prior_pressure = pressure[:-1]
    return {
        "sample_count": len(rows),
        "effective_alpha_mean": float(np.mean(alpha)),
        "effective_alpha_min": float(np.min(alpha)),
        "effective_alpha_max": float(np.max(alpha)),
        "effective_alpha_final": float(alpha[-1]),
        "pressure_fraction_mean": float(np.mean(pressure)),
        "pressure_fraction_peak": float(np.max(pressure)),
        "hbm_used_bytes_mean": float(np.mean(hbm)),
        "hbm_used_bytes_peak": int(np.max(hbm)),
        "dram_used_bytes_mean": float(np.mean(dram)),
        "dram_used_bytes_peak": int(np.max(dram)),
        "stable_alpha_increase_transitions": int(
            np.sum((prior_pressure == 0.0) & (delta > 1e-12))
        ),
        "pressure_alpha_decrease_transitions": int(
            np.sum((prior_pressure > 0.0) & (delta < -1e-12))
        ),
        "alpha_zero_window_fraction": float(np.mean(alpha <= 1e-12)),
    }


def uptime_history_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    """Recompute horizon and post-event-drain history from raw evidence."""

    failures = artifact["failure_rows"]
    summary = artifact["summary"]
    horizon_history = artifact.get("failure_state_at_horizon", {})
    final_history = artifact["final_state"].get("failure_history", {})
    physical_repeat = sum(
        bool(row["physical_gpu_shard_ever_failed_before_event"])
        for row in failures
    )
    effective_physical_repeat = sum(
        bool(row["effective"])
        and bool(row["physical_gpu_shard_ever_failed_before_event"])
        for row in failures
    )
    effective_bound_target_currently_unavailable = sum(
        bool(row["effective"])
        and bool(
            row[
                "bound_peer_hbm_target_currently_unavailable_before_source_event"
            ]
        )
        for row in failures
    )
    effective_bound_target_ever_failed = sum(
        bool(row["effective"])
        and bool(row["bound_peer_hbm_target_ever_failed_before_source_event"])
        for row in failures
    )
    old_copy_events = int(
        horizon_history.get(
            "old_hbm_copy_invalidation_event_count_cumulative", 0
        )
    )
    old_copy_requests = int(
        horizon_history.get("old_hbm_copy_invalidated_requests_cumulative", 0)
    )
    old_copy_bytes = int(
        horizon_history.get("old_hbm_copy_invalidated_bytes_cumulative", 0)
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
        horizon_history.get("current_logical_replica_target_outage_count", 0)
    )
    current_logical_post = int(
        final_history.get("current_logical_replica_target_outage_count", 0)
    )
    logical_ever = int(
        horizon_history.get("logical_replica_targets_ever_failed_count", 0)
    )
    stable_fields = (
        "physical_gpu_shards_ever_failed_count",
        "logical_replica_targets_ever_failed_count",
        "old_hbm_copy_invalidation_event_count_cumulative",
        "old_hbm_copy_invalidated_requests_cumulative",
        "old_hbm_copy_invalidated_bytes_cumulative",
    )
    repairs_after_horizon = sum(
        bool(row.get("after_base_run_horizon"))
        for row in artifact.get("repair_rows", ())
    )
    history_contract = artifact["formal_metric_contract"][
        "failure_history_contract"
    ]
    gates = artifact["gates"]
    accounting_matches = bool(
        physical_repeat
        == int(summary["failure_events_on_previously_failed_physical_gpu_shard"])
        and effective_bound_target_currently_unavailable
        == int(
            summary[
                "effective_failures_with_bound_peer_hbm_target_prior_unavailable"
            ]
        )
        and old_copy_events
        == int(summary["old_hbm_copy_invalidation_event_count_cumulative"])
        and old_copy_requests
        == int(summary["old_hbm_copy_invalidated_requests_cumulative"])
        and old_copy_bytes
        == int(summary["old_hbm_copy_invalidated_bytes_cumulative"])
        and current_physical_horizon
        == int(summary["current_physical_gpu_shard_outage_count_at_horizon"])
        and current_physical_post
        == int(summary["current_physical_gpu_shard_outage_count_post_all_events"])
        and physical_ever
        == int(summary["physical_gpu_shards_ever_failed_count"])
        and current_logical_horizon
        == int(summary["current_logical_replica_target_outage_count_at_horizon"])
        and current_logical_post
        == int(summary["current_logical_replica_target_outage_count_post_all_events"])
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
            for field in stable_fields
        )
        and int(summary["repair_events_after_horizon"]) == repairs_after_horizon
        and int(summary["post_horizon_event_count"]) >= repairs_after_horizon
        and history_contract.get("horizon_state_source")
        == "failure_state_at_horizon"
        and history_contract.get("post_all_events_state_source")
        == "final_state.failure_history"
        and history_contract.get("events_after_horizon_are_cleanup_not_horizon_state")
        is True
        and gates.get("horizon_failure_state_frozen_before_post_horizon_events")
        is True
        and gates.get("summary_at_horizon_fields_derive_from_horizon_failure_state")
        is True
        and gates.get("final_state_is_post_all_events_not_horizon_state") is True
    )
    return {
        "uptime_physical_gpu_shard_repeat_failure_events": physical_repeat,
        "uptime_effective_faults_on_previously_failed_physical_gpu_shard": (
            effective_physical_repeat
        ),
        "uptime_effective_faults_with_bound_target_currently_unavailable": (
            effective_bound_target_currently_unavailable
        ),
        "uptime_effective_faults_with_bound_target_ever_failed": (
            effective_bound_target_ever_failed
        ),
        "old_hbm_copy_invalidation_event_count_cumulative": old_copy_events,
        "old_hbm_copy_invalidated_requests_cumulative": old_copy_requests,
        "old_hbm_copy_invalidated_bytes_cumulative": old_copy_bytes,
        "current_physical_gpu_shard_outage_count_at_horizon": (
            current_physical_horizon
        ),
        "current_physical_gpu_shard_outage_count_post_all_events": (
            current_physical_post
        ),
        "physical_gpu_shards_ever_failed_count": physical_ever,
        "current_logical_replica_target_outage_count_at_horizon": (
            current_logical_horizon
        ),
        "current_logical_replica_target_outage_count_post_all_events": (
            current_logical_post
        ),
        "logical_replica_targets_ever_failed_count": logical_ever,
        "repair_events_after_horizon": repairs_after_horizon,
        "post_horizon_event_count": int(summary["post_horizon_event_count"]),
        "history_accounting_matches_raw_summary_and_final_state": accounting_matches,
    }


def summary_frame(
    artifacts: list[tuple[Path, dict[str, Any]]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path, artifact in artifacts:
        summary = artifact["summary"]
        scenario = artifact["scenario"]
        steady = steady_diagnostics(artifact)
        history = uptime_history_diagnostics(artifact)
        restored = int(summary.get("restored_tokens", 0))
        recomputed = int(summary.get("recomputed_tokens", 0))
        hbm_capacity = float(summary.get("hbm_capacity_bytes", 0))
        dram_capacity = float(summary.get("dram_capacity_bytes", 0))
        exposure_worker_hours = float(
            summary["modeled_failure_exposure_gpu_hours"]
        )
        row = {
            "path": str(path),
            "sha256": sha256(path),
            "run_id": artifact["run_id"],
            "scenario_id": scenario["scenario_id"],
            "campaign_axis": scenario["campaign_axis"],
            "seed": int(artifact["seed"]),
            "policy": artifact["policy"],
            **{
                key: value
                for key, value in scenario.items()
                if key not in {"scenario_id", "campaign_axis", "policies"}
            },
            **flatten("summary.", summary),
            **flatten("workload.", artifact["workload"]),
            **flatten("steady.", steady),
            **flatten("realized.", artifact["realized_concurrency"]),
            **flatten("history.", history),
        }
        row["failure_rate_per_1000_gpu_shard_hours"] = float(
            scenario["failure_rate_per_1000_worker_hours"]
        )
        row["failure_rate_legacy_input_field"] = (
            "failure_rate_per_1000_worker_hours; semantic unit is physical "
            "GPU-shard-hours"
        )
        recovery_tokens = restored + recomputed
        shard_count = int(scenario["tp_size"]) * int(scenario["pp_size"])
        row["hbm_quota_gib_per_gpu_shard"] = float(
            scenario["hbm_quota_gib_per_target"]
        )
        row["dram_quota_gib_per_gpu_shard"] = float(
            scenario["dram_quota_gib_per_target"]
        )
        row["hbm_quota_gib_per_logical_dp_target"] = (
            row["hbm_quota_gib_per_gpu_shard"] * shard_count
        )
        row["dram_quota_gib_per_logical_dp_target"] = (
            row["dram_quota_gib_per_gpu_shard"] * shard_count
        )
        row["summary.restore_hit_fraction"] = (
            restored / recovery_tokens if recovery_tokens else math.nan
        )
        row["summary.recompute_fraction"] = (
            recomputed / recovery_tokens if recovery_tokens else math.nan
        )
        row["steady.hbm_occupancy_mean_fraction"] = (
            steady["hbm_used_bytes_mean"] / hbm_capacity if hbm_capacity else 0.0
        )
        row["steady.hbm_occupancy_peak_fraction"] = (
            steady["hbm_used_bytes_peak"] / hbm_capacity if hbm_capacity else 0.0
        )
        row["steady.dram_occupancy_mean_fraction"] = (
            steady["dram_used_bytes_mean"] / dram_capacity if dram_capacity else 0.0
        )
        row["steady.dram_occupancy_peak_fraction"] = (
            steady["dram_used_bytes_peak"] / dram_capacity if dram_capacity else 0.0
        )
        row["fault.total_gpu_shard_exposure_hours"] = exposure_worker_hours
        row["fault.observed_events_per_1000_gpu_shard_hours"] = (
            int(summary["failure_events_injected"])
            / max(exposure_worker_hours, 1e-12)
            * 1000.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def probe_seed_frame(
    artifacts: list[tuple[Path, dict[str, Any]]]
) -> pd.DataFrame:
    """Recompute one equal-exposure-stratified K64 mean per run/seed."""

    output: list[dict[str, Any]] = []
    for path, artifact in artifacts:
        scenario = artifact["scenario"]
        metadata = artifact["active_source_probe_metadata"]
        rows = sorted(
            artifact["active_source_probe_rows"],
            key=lambda row: int(row["stratum_index"]),
        )
        if len(rows) != PROBE_COUNT:
            raise ValueError(f"expected K64 probe rows: {artifact['run_id']}")
        restored = sum(int(row["restored_tokens"]) for row in rows)
        recomputed = sum(int(row["recompute_tokens"]) for row in rows)
        total = restored + recomputed
        source_counts = {
            name: sum(row["recovery_source_class"] == name for row in rows)
            for name in ("PEER_HBM", "DRAM", "RECOMPUTE", "MIXED_OR_BASELINE_SPECIFIC")
        }
        output.append(
            {
                "path": str(path),
                "run_id": artifact["run_id"],
                "scenario_id": scenario["scenario_id"],
                "campaign_axis": scenario["campaign_axis"],
                "policy": artifact["policy"],
                "seed": int(artifact["seed"]),
                **{
                    field: scenario[field]
                    for field in HAZARD_MATCH_FIELDS
                },
                "failure_rate_per_1000_worker_hours": float(
                    scenario["failure_rate_per_1000_worker_hours"]
                ),
                "failure_rate_per_1000_gpu_shard_hours": float(
                    scenario["failure_rate_per_1000_worker_hours"]
                ),
                "probe_count": len(rows),
                "probe_tuple_sha256": metadata["probe_tuple_sha256"],
                "eligible_exposure_universe_sha256": metadata[
                    "eligible_exposure_universe_sha256"
                ],
                "sampling_prng_key_sha256": metadata[
                    "sampling_prng_key_sha256"
                ],
                "eligible_exposure_gpu_shard_hours": float(
                    metadata["eligible_exposure_gpu_shard_hours"]
                ),
                "conditional_recovery_mean_ms": float(
                    np.mean([float(row["recovery_ms"]) for row in rows])
                ),
                "conditional_recovery_p50_ms_descriptive": float(
                    np.quantile([float(row["recovery_ms"]) for row in rows], 0.5)
                ),
                "conditional_recovery_p95_ms_descriptive": float(
                    np.quantile([float(row["recovery_ms"]) for row in rows], 0.95)
                ),
                "conditional_actual_harm_mean_ms": float(
                    np.mean([float(row["actual_visible_harm_ms"]) for row in rows])
                ),
                "conditional_actual_harm_p50_ms_descriptive": float(
                    np.quantile(
                        [float(row["actual_visible_harm_ms"]) for row in rows],
                        0.5,
                    )
                ),
                "conditional_actual_harm_p95_ms_descriptive": float(
                    np.quantile(
                        [float(row["actual_visible_harm_ms"]) for row in rows],
                        0.95,
                    )
                ),
                "restored_tokens": restored,
                "peer_hbm_restored_tokens": sum(
                    int(row["peer_hbm_restored_tokens"]) for row in rows
                ),
                "dram_restored_tokens": sum(
                    int(row["dram_restored_tokens"]) for row in rows
                ),
                "mixed_or_baseline_specific_restored_tokens": sum(
                    int(row["mixed_or_baseline_specific_restored_tokens"])
                    for row in rows
                ),
                "recomputed_tokens": recomputed,
                "restore_hit_fraction_pooled_tokens": (
                    restored / total if total else math.nan
                ),
                "recompute_fraction_pooled_tokens": (
                    recomputed / total if total else math.nan
                ),
                **{
                    f"source_class_{name.lower()}_count": count
                    for name, count in source_counts.items()
                },
                "probe_rows_with_uptime_history": sum(
                    bool(row["probe_state_includes_uptime_poisson_history"])
                    for row in rows
                ),
                "normal_state_reference": all(
                    not bool(row["probe_state_includes_uptime_poisson_history"])
                    for row in rows
                ),
                "probe_state_semantics": (
                    "failure-free normal state"
                    if float(scenario["failure_rate_per_1000_worker_hours"]) == 0.0
                    else "rate-local live state including any preceding uptime Poisson history"
                ),
                "statistical_unit": (
                    "one seed-level mean over 64 equal-exposure strata; "
                    "outer seed bootstrap only"
                ),
            }
        )
    return pd.DataFrame(output)


def probe_reuse_frame(
    artifacts: list[tuple[Path, dict[str, Any]]]
) -> pd.DataFrame:
    """Full-workload reuse distribution plus descriptive K64 recovery by bin."""

    output: list[dict[str, Any]] = []
    for _path, artifact in artifacts:
        summary = artifact["summary"]
        scenario = artifact["scenario"]
        for reuse_bin in REUSE_BIN_CONTRACT:
            rows = [
                row
                for row in artifact["active_source_probe_rows"]
                if row["input_reuse_bin"] == reuse_bin
            ]
            restored = sum(int(row["restored_tokens"]) for row in rows)
            recomputed = sum(int(row["recompute_tokens"]) for row in rows)
            total = restored + recomputed
            residency = summary[
                "workload_reuse_bin_intended_residency_chunk_counts"
            ][reuse_bin]
            admission = summary[
                "workload_reuse_bin_tier_admission_chunk_counts"
            ][reuse_bin]
            output.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "campaign_axis": scenario["campaign_axis"],
                    "policy": artifact["policy"],
                    "seed": int(artifact["seed"]),
                    "input_reuse_bin": reuse_bin,
                    "request_count_full_workload": int(
                        summary["workload_reuse_bin_request_counts"][reuse_bin]
                    ),
                    "prompt_tokens_full_workload": int(
                        summary["workload_reuse_bin_prompt_tokens"][reuse_bin]
                    ),
                    "input_reuse_tokens_full_workload": int(
                        summary["workload_reuse_bin_input_reuse_tokens"][reuse_bin]
                    ),
                    **{
                        f"intended_residency_{name.lower()}_chunks": int(value)
                        for name, value in residency.items()
                    },
                    **{
                        f"tier_admission_{name.lower()}_chunks": int(value)
                        for name, value in admission.items()
                    },
                    "probe_count_descriptive": len(rows),
                    "probe_recovery_mean_ms_descriptive": (
                        float(np.mean([float(row["recovery_ms"]) for row in rows]))
                        if rows
                        else math.nan
                    ),
                    "probe_actual_harm_mean_ms_descriptive": (
                        float(
                            np.mean(
                                [float(row["actual_visible_harm_ms"]) for row in rows]
                            )
                        )
                        if rows
                        else math.nan
                    ),
                    "probe_restored_tokens": restored,
                    "probe_recomputed_tokens": recomputed,
                    "probe_restore_hit_fraction_pooled_tokens": (
                        restored / total if total else math.nan
                    ),
                    "inference_boundary": (
                        "probe recovery summaries are descriptive; 64 strata are "
                        "not IID inferential units"
                    ),
                }
            )
    return pd.DataFrame(output)


def normal_state_hazard_burden(probes: pd.DataFrame) -> pd.DataFrame:
    """Bind every row to its exact same-policy/seed rate-zero normal state."""

    references: dict[tuple[Any, ...], pd.Series] = {}
    for _index, row in probes[
        np.isclose(probes["failure_rate_per_1000_worker_hours"], 0.0)
    ].iterrows():
        key = (
            row["policy"],
            int(row["seed"]),
            *(row[field] for field in HAZARD_MATCH_FIELDS),
        )
        if key in references:
            raise ValueError(f"duplicate exact rate-zero probe reference: {key}")
        references[key] = row
    output: list[dict[str, Any]] = []
    for _index, row in probes.iterrows():
        key = (
            row["policy"],
            int(row["seed"]),
            *(row[field] for field in HAZARD_MATCH_FIELDS),
        )
        reference = references.get(key)
        rate = float(row["failure_rate_per_1000_worker_hours"])
        matched = reference is not None and bool(reference["normal_state_reference"])
        if matched:
            exposure = float(reference["eligible_exposure_gpu_shard_hours"])
            harm = float(reference["conditional_actual_harm_mean_ms"])
            total = 0.0 if rate == 0.0 else rate / 1000.0 * exposure * harm
            per_workflow = total / 393.0
        else:
            exposure = harm = total = per_workflow = math.nan
        output.append(
            {
                **row.to_dict(),
                "normal_state_match_status": (
                    "MATCHED_EXACT_RATE_ZERO" if matched else "NA_NO_EXACT_RATE_ZERO"
                ),
                "normal_state_reference_scenario_id": (
                    reference["scenario_id"] if matched else None
                ),
                "normal_state_reference_probe_tuple_sha256": (
                    reference["probe_tuple_sha256"] if matched else None
                ),
                "normal_state_eligible_exposure_gpu_shard_hours": exposure,
                "normal_state_mean_probe_actual_harm_ms_per_event": harm,
                "normal_state_burden_total_ms_per_trace_epoch": total,
                "normal_state_burden_ms_per_workflow_epoch": per_workflow,
                "estimand_name": (
                    "single-fault normal-state eligible hazard-integrated "
                    "additive burden"
                ),
                "estimand_boundary": (
                    "first-order active+sealed partial additive event-harm; not "
                    "WCT, workflow-max, MTBF, steady-state observed, or total burden; "
                    "not added to uptime Poisson incidence"
                ),
            }
        )
    return pd.DataFrame(output)


def paired_probe_outer_seed_bootstrap(
    burdens: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Pair same seed and K64 stratum means; bootstrap seeds only."""

    treatment = "aegiskv_alg111"
    metrics = (
        "conditional_recovery_mean_ms",
        "conditional_actual_harm_mean_ms",
        "normal_state_burden_total_ms_per_trace_epoch",
        "normal_state_burden_ms_per_workflow_epoch",
    )
    rng = np.random.default_rng(seed ^ 0x64A3)
    output: list[dict[str, Any]] = []
    for scenario_id, group in burdens.groupby("scenario_id"):
        policies = set(group["policy"])
        if treatment not in policies:
            continue
        for reference_policy in sorted(policies - {treatment}):
            treatment_rows = group[group["policy"] == treatment].set_index("seed")
            reference_rows = group[group["policy"] == reference_policy].set_index("seed")
            seeds = sorted(set(treatment_rows.index) & set(reference_rows.index))
            for metric in metrics:
                differences: list[float] = []
                for run_seed in seeds:
                    left = treatment_rows.loc[run_seed]
                    right = reference_rows.loc[run_seed]
                    if left["probe_tuple_sha256"] != right["probe_tuple_sha256"]:
                        raise ValueError(
                            f"unpaired K64 tuple schedule: {scenario_id}/{run_seed}"
                        )
                    left_value = float(left[metric])
                    right_value = float(right[metric])
                    if math.isfinite(left_value) and math.isfinite(right_value):
                        differences.append(left_value - right_value)
                if differences:
                    values = np.asarray(differences, dtype=float)
                    samples = rng.integers(
                        0, len(values), size=(resamples, len(values))
                    )
                    boot = np.mean(values[samples], axis=1)
                    mean = float(np.mean(values))
                    low = float(np.quantile(boot, 0.025))
                    high = float(np.quantile(boot, 0.975))
                else:
                    mean = low = high = math.nan
                output.append(
                    {
                        "scenario_id": scenario_id,
                        "policy": treatment,
                        "reference": reference_policy,
                        "metric": metric,
                        "paired_seed_count": len(differences),
                        "probe_strata_per_seed": PROBE_COUNT,
                        "mean_delta": mean,
                        "delta_ci95_low": low,
                        "delta_ci95_high": high,
                        "bootstrap_unit": (
                            "paired outer seed bootstrap over one K64 equal-exposure "
                            "stratified mean per seed; no probe-inner bootstrap"
                        ),
                    }
                )
    return pd.DataFrame(output)


def paired_probe_comparison_coverage(
    paired_probes: pd.DataFrame,
    paired_workflows: pd.DataFrame,
    burdens: pd.DataFrame,
    *,
    pilot: bool,
) -> dict[str, Any]:
    """Fail closed on every B/C same-seed direct comparison."""

    metrics = {
        "conditional_recovery_mean_ms": "B",
        "conditional_actual_harm_mean_ms": "B",
        "normal_state_burden_total_ms_per_trace_epoch": "C",
        "normal_state_burden_ms_per_workflow_epoch": "C",
    }
    comparisons = {
        (str(row.scenario_id), str(row.reference))
        for row in paired_workflows.itertuples(index=False)
    }
    expected = {
        (scenario, reference, metric)
        for scenario, reference in comparisons
        for metric in metrics
    }
    observed = {
        (str(row.scenario_id), str(row.reference), str(row.metric))
        for row in paired_probes.itertuples(index=False)
    }
    indexed = burdens.set_index(["scenario_id", "policy", "seed"])
    rate_zero_reference_keys = {
        (
            row.policy,
            int(row.seed),
            *(getattr(row, field) for field in HAZARD_MATCH_FIELDS),
        )
        for row in burdens[
            np.isclose(burdens["failure_rate_per_1000_worker_hours"], 0.0)
        ].itertuples(index=False)
    }
    row_checks: list[dict[str, Any]] = []
    for row in paired_probes.itertuples(index=False):
        treatment_seeds = {
            int(seed)
            for scenario, policy, seed in indexed.index
            if scenario == row.scenario_id and policy == "aegiskv_alg111"
        }
        reference_seeds = {
            int(seed)
            for scenario, policy, seed in indexed.index
            if scenario == row.scenario_id and policy == row.reference
        }
        common = sorted(treatment_seeds & reference_seeds)
        expected_finite = 0
        if metrics[row.metric] == "B":
            expected_finite = len(common)
        else:
            for run_seed in common:
                left = indexed.loc[(row.scenario_id, "aegiskv_alg111", run_seed)]
                right = indexed.loc[(row.scenario_id, row.reference, run_seed)]
                left_key = (
                    "aegiskv_alg111",
                    run_seed,
                    *(left[field] for field in HAZARD_MATCH_FIELDS),
                )
                right_key = (
                    row.reference,
                    run_seed,
                    *(right[field] for field in HAZARD_MATCH_FIELDS),
                )
                if (
                    left_key in rate_zero_reference_keys
                    and right_key in rate_zero_reference_keys
                ):
                    expected_finite += 1
        values = (
            float(row.mean_delta),
            float(row.delta_ci95_low),
            float(row.delta_ci95_high),
        )
        finite = all(math.isfinite(value) for value in values)
        row_pass = int(row.paired_seed_count) == expected_finite and (
            finite if expected_finite else not any(math.isfinite(value) for value in values)
        )
        row_checks.append(
            {
                "scenario_id": row.scenario_id,
                "reference": row.reference,
                "metric": row.metric,
                "expected_paired_seed_count": expected_finite,
                "observed_paired_seed_count": int(row.paired_seed_count),
                "status": "PASS" if row_pass else "FAIL",
            }
        )
    rate_zero = burdens[
        np.isclose(burdens["failure_rate_per_1000_worker_hours"], 0.0)
        & (burdens["normal_state_match_status"] == "MATCHED_EXACT_RATE_ZERO")
    ]
    rate_zero_absolute = bool(len(rate_zero)) and np.isclose(
        rate_zero["normal_state_burden_total_ms_per_trace_epoch"], 0.0
    ).all() and np.isclose(
        rate_zero["normal_state_burden_ms_per_workflow_epoch"], 0.0
    ).all()
    rate_zero_scenarios = set(rate_zero["scenario_id"])
    rate_zero_paired = paired_probes[
        paired_probes["scenario_id"].isin(rate_zero_scenarios)
        & paired_probes["metric"].str.startswith("normal_state_burden_")
    ]
    rate_zero_paired_exact = bool(len(rate_zero_paired)) and all(
        math.isclose(float(value), 0.0, abs_tol=0.0)
        for column in ("mean_delta", "delta_ci95_low", "delta_ci95_high")
        for value in rate_zero_paired[column]
    )
    expected_rows = len(expected)
    checks = {
        "expected_comparison_metric_set_exact": observed == expected,
        "expected_row_count_exact": len(paired_probes) == expected_rows,
        "all_rows_seed_count_and_ci_contract": all(
            row["status"] == "PASS" for row in row_checks
        ),
        "rate_zero_c_absolute_point_exact_zero": rate_zero_absolute,
        "rate_zero_c_paired_delta_ci_exact_zero": rate_zero_paired_exact,
        "probe_inner_bootstrap_absent": all(
            "no probe-inner bootstrap" in str(row.bootstrap_unit)
            for row in paired_probes.itertuples(index=False)
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": "pilot" if pilot else "full",
        "expected_row_count": expected_rows,
        "observed_row_count": len(paired_probes),
        "checks": checks,
        "rows": row_checks,
    }


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, math.nan
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, mean, mean
    half = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    return mean, mean - half, mean + half


def pooled_recovery_fraction(
    group: pd.DataFrame,
    *,
    resamples: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, float]:
    """Token-pooled restore/recompute fraction; zero denominator is NA."""

    restored = group["summary.restored_tokens"].to_numpy(dtype=float)
    recomputed = group["summary.recomputed_tokens"].to_numpy(dtype=float)
    denominator = float(np.sum(restored + recomputed))
    if denominator <= 0.0:
        return math.nan, math.nan, math.nan, math.nan
    hit = float(np.sum(restored) / denominator)
    if not resamples or rng is None:
        return hit, 1.0 - hit, hit, hit
    samples = rng.integers(0, len(group), size=(resamples, len(group)))
    boot_restored = np.sum(restored[samples], axis=1)
    boot_total = np.sum((restored + recomputed)[samples], axis=1)
    valid = boot_total > 0.0
    boot = boot_restored[valid] / boot_total[valid]
    if not len(boot):
        return hit, 1.0 - hit, math.nan, math.nan
    return (
        hit,
        1.0 - hit,
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def axis_aggregates(
    frame: pd.DataFrame, *, resamples: int, seed: int
) -> pd.DataFrame:
    group_columns = [
        "scenario_id",
        "campaign_axis",
        "policy",
        "model_id",
        "prompt_scale",
        "active_workflow_concurrency",
        "failure_rate_per_1000_gpu_shard_hours",
        "total_gpu_count",
        "network_effective_gb_s",
        "hbm_quota_gib_per_target",
        "dram_quota_gib_per_target",
        "algorithm1_prediction_error_multiplier",
    ]
    metrics = [
        "summary.actual_failure_added_workflow_harm_surrogate_mean_ms",
        "summary.actual_modeled_visible_harm_mean_ms",
        "summary.input_reuse_fraction",
        "steady.effective_alpha_mean",
        "steady.effective_alpha_final",
        "steady.pressure_fraction_mean",
        "steady.hbm_occupancy_mean_fraction",
        "steady.dram_occupancy_mean_fraction",
        "fault.total_gpu_shard_exposure_hours",
        "fault.observed_events_per_1000_gpu_shard_hours",
        "realized.base_full_horizon.time_mean",
        "realized.simulated_full_horizon.time_mean",
    ]
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed ^ 0xC011)
    for keys, group in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        row["seed_count"] = len(group)
        row["ci_note"] = "normal 1.96 SE across paired seeds; paired bootstrap is separate"
        for metric in metrics:
            mean, low, high = mean_ci(pd.to_numeric(group[metric], errors="coerce"))
            row[f"{metric}.mean"] = mean
            row[f"{metric}.ci95_low"] = low
            row[f"{metric}.ci95_high"] = high
        hit, recompute, low, high = pooled_recovery_fraction(
            group, resamples=resamples, rng=rng
        )
        row["summary.restore_hit_fraction.mean"] = hit
        row["summary.restore_hit_fraction.ci95_low"] = low
        row["summary.restore_hit_fraction.ci95_high"] = high
        row["summary.recompute_fraction.mean"] = recompute
        row["summary.recovery_fraction_aggregation"] = (
            "pooled tokens: sum(restored)/sum(restored+recomputed); "
            "zero denominator is NA"
        )
        logical_bytes = float(group["summary.logical_request_kv_bytes"].sum())
        row["summary.copy_amplification.mean"] = (
            float(group["summary.protection_copy_attempted_bytes"].sum())
            / logical_bytes
            if logical_bytes > 0.0
            else math.nan
        )
        row["summary.copy_amplification.aggregation"] = (
            "pooled bytes: sum(protection copy attempted)/sum(logical request KV)"
        )
        output.append(row)
    return pd.DataFrame(output)


def workflow_map(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["workflow_id"]: row for row in artifact["workflow_rows"]}


def paired_workflow_seed_bootstrap(
    artifacts: list[tuple[Path, dict[str, Any]]],
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Use every workflow row, then bootstrap paired per-seed means."""

    by_key = {
        (
            artifact["scenario"]["scenario_id"],
            artifact["policy"],
            int(artifact["seed"]),
        ): artifact
        for _path, artifact in artifacts
    }
    scenario_policies: dict[str, set[str]] = {}
    scenario_seeds: dict[str, set[int]] = {}
    for _path, artifact in artifacts:
        scenario = artifact["scenario"]["scenario_id"]
        scenario_policies.setdefault(scenario, set()).add(artifact["policy"])
        scenario_seeds.setdefault(scenario, set()).add(int(artifact["seed"]))
    metrics = ["actual_failure_added_workflow_harm_surrogate_ms"]
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    treatment_policy = "aegiskv_alg111"
    for scenario, policies in sorted(scenario_policies.items()):
        if treatment_policy not in policies:
            continue
        seeds = sorted(scenario_seeds[scenario])
        for reference_policy in sorted(policies - {treatment_policy}):
            paired_seed_means = {metric: [] for metric in metrics}
            for run_seed in seeds:
                reference = workflow_map(
                    by_key[(scenario, reference_policy, run_seed)]
                )
                treatment = workflow_map(
                    by_key[(scenario, treatment_policy, run_seed)]
                )
                ids = sorted(set(reference) & set(treatment))
                if len(ids) != 393:
                    raise ValueError(
                        f"expected 393 paired workflows: "
                        f"{scenario}/{treatment_policy}/{reference_policy}/{run_seed}"
                    )
                for metric in metrics:
                    paired_seed_means[metric].append(
                        float(
                            np.mean(
                                [
                                    float(treatment[row_id][metric])
                                    - float(reference[row_id][metric])
                                    for row_id in ids
                                ]
                            )
                        )
                    )
            samples = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
            for metric in metrics:
                difference = np.asarray(paired_seed_means[metric], dtype=float)
                boot = np.mean(difference[samples], axis=1)
                output.append(
                    {
                        "scenario_id": scenario,
                        "policy": treatment_policy,
                        "reference": reference_policy,
                        "metric": metric,
                        "paired_seed_count": len(seeds),
                        "paired_workflow_count_per_seed": 393,
                        "mean_delta_ms": float(np.mean(difference)),
                        "delta_ci95_low_ms": float(np.quantile(boot, 0.025)),
                        "delta_ci95_high_ms": float(np.quantile(boot, 0.975)),
                        "bootstrap_unit": (
                            "same-seed, same-workflow difference; paired seed "
                            "bootstrap; negative delta favors full AegisKV"
                        ),
                    }
                )
    return pd.DataFrame(output)


def central_scenario_id(frame: pd.DataFrame) -> str:
    candidate = frame[
        (frame["model_id"] == "qwen3_8b")
        & (frame["active_workflow_concurrency"] == 128)
        & np.isclose(frame["failure_rate_per_1000_worker_hours"], 5.952)
        & (frame["total_gpu_count"] == 64)
        & np.isclose(frame["prompt_scale"], 0.016)
        & np.isclose(frame["network_effective_gb_s"], 10.0)
        & np.isclose(frame["hbm_quota_gib_per_target"], 4.0)
        & np.isclose(frame["dram_quota_gib_per_target"], 64.0)
        & np.isclose(frame["algorithm1_prediction_error_multiplier"], 1.0)
    ]
    unique = candidate["scenario_id"].unique()
    if len(unique) != 1:
        raise ValueError(f"central scenario not unique: {list(unique)}")
    return str(unique[0])


def central_rate_zero_scenario_id(frame: pd.DataFrame) -> str:
    candidate = frame[
        (frame["model_id"] == "qwen3_8b")
        & (frame["active_workflow_concurrency"] == 128)
        & np.isclose(frame["failure_rate_per_1000_worker_hours"], 0.0)
        & (frame["total_gpu_count"] == 64)
        & np.isclose(frame["prompt_scale"], 0.016)
        & np.isclose(frame["network_effective_gb_s"], 10.0)
        & np.isclose(frame["hbm_quota_gib_per_target"], 4.0)
        & np.isclose(frame["dram_quota_gib_per_target"], 64.0)
        & np.isclose(frame["algorithm1_prediction_error_multiplier"], 1.0)
    ]
    unique = candidate["scenario_id"].unique()
    if len(unique) != 1:
        raise ValueError(f"central rate-zero scenario not unique: {list(unique)}")
    return str(unique[0])


def algorithm1_identifiability_audit(
    artifacts: list[tuple[Path, dict[str, Any]]], frame: pd.DataFrame
) -> dict[str, Any]:
    """Fail closed if full and no-Algorithm-1 are observationally identical."""

    central = central_scenario_id(frame)
    by_key = {
        (artifact["policy"], int(artifact["seed"])): artifact
        for _path, artifact in artifacts
        if artifact["scenario"]["scenario_id"] == central
    }
    full_seeds = {
        seed for policy, seed in by_key if policy == "aegiskv_alg111"
    }
    no_alg1_seeds = {
        seed
        for policy, seed in by_key
        if policy == "aegiskv_alg011_no_alg1"
    }
    common = sorted(full_seeds & no_alg1_seeds)
    nonzero_coverage: list[dict[str, Any]] = []
    state_differences: list[dict[str, Any]] = []
    effective_input_contrasts: list[dict[str, Any]] = []
    for run_seed in common:
        full = by_key[("aegiskv_alg111", run_seed)]
        no_alg1 = by_key[("aegiskv_alg011_no_alg1", run_seed)]
        count = int(
            full["workload"]["predicted_policy_slack_nonzero_count"]
        )
        fraction = float(
            full["workload"]["predicted_policy_slack_nonzero_fraction"]
        )
        nonzero_coverage.append(
            {"seed": run_seed, "count": count, "fraction": fraction}
        )

        def signature(artifact: dict[str, Any]) -> str:
            summary = artifact["summary"]
            payload = {
                "admission_outcomes": summary["admission_outcomes"],
                "protection_copy_attempted_bytes": summary[
                    "protection_copy_attempted_bytes"
                ],
                "measurement_hbm_used_bytes_time_mean": summary[
                    "measurement_hbm_used_bytes_time_mean"
                ],
                "measurement_dram_used_bytes_time_mean": summary[
                    "measurement_dram_used_bytes_time_mean"
                ],
            }
            return hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()

        full_signature = signature(full)
        no_alg1_signature = signature(no_alg1)
        state_differences.append(
            {
                "seed": run_seed,
                "full_signature": full_signature,
                "no_algorithm1_signature": no_alg1_signature,
                "different": full_signature != no_alg1_signature,
            }
        )
        effective_input_contrasts.append(
            {
                "seed": run_seed,
                "full_algorithm1_enabled": full["summary"][
                    "algorithm1_enabled"
                ]
                is True,
                "full_effective_nonzero_count": int(
                    full["summary"]["effective_policy_slack_nonzero_count"]
                ),
                "no_algorithm1_enabled": no_alg1["summary"][
                    "algorithm1_enabled"
                ]
                is True,
                "no_algorithm1_effective_nonzero_count": int(
                    no_alg1["summary"]["effective_policy_slack_nonzero_count"]
                ),
            }
        )
    checks = {
        "central_full_and_no_algorithm1_pairs_exist": bool(common),
        "central_predicted_policy_slack_nonzero_for_every_pair": bool(common)
        and all(row["count"] > 0 and row["fraction"] > 0.0 for row in nonzero_coverage),
        "central_algorithm1_effective_input_path_contrasts_every_pair": bool(common)
        and all(
            row["full_algorithm1_enabled"]
            and row["full_effective_nonzero_count"] > 0
            and not row["no_algorithm1_enabled"]
            and row["no_algorithm1_effective_nonzero_count"] == 0
            for row in effective_input_contrasts
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "paired_seed_count": len(common),
        "nonzero_policy_slack_coverage": nonzero_coverage,
        "effective_input_contrasts": effective_input_contrasts,
        "mechanism_state_differences": state_differences,
        "post_treatment_state_difference_is_diagnostic_not_gate": True,
        "post_treatment_state_difference_observed_every_pair": bool(common)
        and all(row["different"] for row in state_differences),
    }


def hierarchical_central_bootstrap(
    artifacts: list[tuple[Path, dict[str, Any]]],
    frame: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Two-stage paired seed/workflow bootstrap for the central comparison."""

    central = central_scenario_id(frame)
    by_key = {
        (
            artifact["scenario"]["scenario_id"],
            artifact["policy"],
            int(artifact["seed"]),
        ): artifact
        for _path, artifact in artifacts
    }
    central_rows = frame[frame["scenario_id"] == central]
    seeds = sorted(central_rows["seed"].unique())
    treatment_policy = "aegiskv_alg111"
    references = sorted(set(central_rows["policy"]) - {treatment_policy})
    metrics = ["actual_failure_added_workflow_harm_surrogate_ms"]
    rng = np.random.default_rng(seed ^ 0xA391)
    output: list[dict[str, Any]] = []
    for reference_policy in references:
        matrices: list[np.ndarray] = []
        canonical_ids: list[str] | None = None
        for run_seed in seeds:
            reference = workflow_map(
                by_key[(central, reference_policy, int(run_seed))]
            )
            treatment = workflow_map(
                by_key[(central, treatment_policy, int(run_seed))]
            )
            ids = sorted(set(reference) & set(treatment))
            if len(ids) != 393:
                raise ValueError("central workflow pair is incomplete")
            if canonical_ids is None:
                canonical_ids = ids
            elif canonical_ids != ids:
                raise ValueError("central workflow identities drift across seeds")
            matrices.append(
                np.asarray(
                    [
                        [
                            float(treatment[row_id][metric])
                            - float(reference[row_id][metric])
                            for metric in metrics
                        ]
                        for row_id in ids
                    ],
                    dtype=float,
                )
            )
        difference = np.stack(matrices)  # seed x workflow x metric
        bootstrap = np.empty((resamples, len(metrics)), dtype=float)
        for sample_index in range(resamples):
            seed_indices = rng.integers(0, len(seeds), size=len(seeds))
            workflow_indices = rng.integers(0, 393, size=393)
            sampled = difference[seed_indices][:, workflow_indices, :]
            bootstrap[sample_index] = np.mean(sampled, axis=(0, 1))
        means = np.mean(difference, axis=(0, 1))
        for metric_index, metric in enumerate(metrics):
            output.append(
                {
                    "scenario_id": central,
                    "policy": treatment_policy,
                    "reference": reference_policy,
                    "metric": metric,
                    "paired_seed_count": len(seeds),
                    "paired_workflow_count_per_seed": 393,
                    "mean_delta_ms": float(means[metric_index]),
                    "delta_ci95_low_ms": float(
                        np.quantile(bootstrap[:, metric_index], 0.025)
                    ),
                    "delta_ci95_high_ms": float(
                        np.quantile(bootstrap[:, metric_index], 0.975)
                    ),
                    "bootstrap_unit": (
                        "hierarchical same-seed, same-workflow bootstrap; "
                        "negative delta favors full AegisKV"
                    ),
                }
            )
    return pd.DataFrame(output)


def paired_comparison_coverage(
    paired: pd.DataFrame,
    hierarchical: pd.DataFrame,
    artifacts: list[tuple[Path, dict[str, Any]]],
    matrix: dict[str, Any],
    *,
    pilot: bool,
) -> dict[str, Any]:
    """Fail closed on missing or malformed direct paired comparisons."""

    treatment = "aegiskv_alg111"
    if pilot:
        scenarios = sorted(
            {
                artifact["scenario"]["scenario_id"]
                for _path, artifact in artifacts
            }
        )
        expected_references = {
            "aegiskv_alg011_no_alg1",
            "recompute_only",
        }
        expected = {
            (scenario, reference)
            for scenario in scenarios
            for reference in expected_references
        }
        expected_seed_count = 1
        expected_row_count = 6
    else:
        expected = {
            (scenario["scenario_id"], policy)
            for scenario in matrix["scenarios"]
            for policy in scenario["policies"]
            if policy != treatment
        }
        expected_seed_count = 10
        expected_row_count = 278

    observed = {
        (str(row.scenario_id), str(row.reference))
        for row in paired.itertuples(index=False)
    }
    finite_columns = (
        "mean_delta_ms",
        "delta_ci95_low_ms",
        "delta_ci95_high_ms",
    )
    direct_rows_valid = bool(len(paired)) and all(
        str(row.policy) == treatment
        and str(row.metric)
        == "actual_failure_added_workflow_harm_surrogate_ms"
        and int(row.paired_seed_count) == expected_seed_count
        and int(row.paired_workflow_count_per_seed) == 393
        and all(math.isfinite(float(getattr(row, key))) for key in finite_columns)
        for row in paired.itertuples(index=False)
    )

    if pilot:
        hierarchical_expected: set[str] = set()
        hierarchical_observed = set(hierarchical.get("reference", []))
        hierarchical_rows_valid = hierarchical.empty
    else:
        central = next(
            row
            for row in matrix["scenarios"]
            if row["model_id"] == "qwen3_8b"
            and row["active_workflow_concurrency"] == 128
            and math.isclose(
                float(row["failure_rate_per_1000_worker_hours"]), 5.952
            )
            and row["total_gpu_count"] == 64
            and math.isclose(float(row["prompt_scale"]), 0.016)
            and math.isclose(float(row["network_effective_gb_s"]), 10.0)
            and math.isclose(float(row["hbm_quota_gib_per_target"]), 4.0)
            and math.isclose(float(row["dram_quota_gib_per_target"]), 64.0)
            and math.isclose(
                float(row["algorithm1_prediction_error_multiplier"]), 1.0
            )
        )
        hierarchical_expected = set(central["policies"]) - {treatment}
        hierarchical_observed = set(hierarchical["reference"])
        hierarchical_rows_valid = bool(len(hierarchical)) and all(
            str(row.policy) == treatment
            and str(row.metric)
            == "actual_failure_added_workflow_harm_surrogate_ms"
            and int(row.paired_seed_count) == 10
            and int(row.paired_workflow_count_per_seed) == 393
            and all(
                math.isfinite(float(getattr(row, key)))
                for key in finite_columns
            )
            for row in hierarchical.itertuples(index=False)
        )

    checks = {
        "direct_expected_row_count_exact": len(paired) == expected_row_count,
        "direct_expected_comparison_set_exact": observed == expected,
        "direct_rows_seed_workflow_ci_finite": direct_rows_valid,
        "hierarchical_expected_reference_set_exact": (
            hierarchical_observed == hierarchical_expected
        ),
        "hierarchical_rows_mode_exact": hierarchical_rows_valid,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": "pilot_nine_run" if pilot else "full_3110_run",
        "checks": checks,
        "direct_expected_row_count": expected_row_count,
        "direct_observed_row_count": len(paired),
        "direct_missing": sorted(expected - observed),
        "direct_unexpected": sorted(observed - expected),
        "hierarchical_expected_reference_count": len(hierarchical_expected),
        "hierarchical_observed_reference_count": len(hierarchical_observed),
    }


def factorial_effects(
    artifacts: list[tuple[Path, dict[str, Any]]],
    frame: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Compute the hierarchical structural 2^3 closure and interactions."""

    central = central_scenario_id(frame)
    algorithms: dict[tuple[int, int, int], str] = {}
    for policy in frame.loc[frame["scenario_id"] == central, "policy"].unique():
        if not str(policy).startswith("aegiskv_alg"):
            continue
        bits = str(policy).split("aegiskv_alg", 1)[1][:3]
        algorithms[tuple(int(bit) for bit in bits)] = str(policy)
    if len(algorithms) != 8:
        raise ValueError(f"central 2^3 algorithm cube incomplete: {algorithms}")
    by_key = {
        (artifact["policy"], int(artifact["seed"])): artifact
        for _path, artifact in artifacts
        if artifact["scenario"]["scenario_id"] == central
    }
    seeds = sorted(frame.loc[frame["scenario_id"] == central, "seed"].unique())
    terms = {
        "Algorithm 1": (0,),
        "Algorithm 2": (1,),
        "Algorithm 3": (2,),
        "Algorithm 1 x 2": (0, 1),
        "Algorithm 1 x 3": (0, 2),
        "Algorithm 2 x 3": (1, 2),
        "Algorithm 1 x 2 x 3": (0, 1, 2),
    }
    metrics = ["actual_failure_added_workflow_harm_surrogate_ms"]
    rng = np.random.default_rng(seed ^ 0xFACA)
    output: list[dict[str, Any]] = []
    for metric in metrics:
        seed_effects = {term: [] for term in terms}
        for run_seed in seeds:
            values: dict[tuple[int, int, int], float] = {}
            for bits, policy in algorithms.items():
                rows = workflow_map(by_key[(policy, int(run_seed))])
                values[bits] = float(np.mean([float(row[metric]) for row in rows.values()]))
            for term, indices in terms.items():
                # Standard full-factorial effect: mean(high contrast)-mean(low contrast).
                contrast = []
                for bits, value in values.items():
                    sign = int(np.prod([1 if bits[index] else -1 for index in indices]))
                    contrast.append(sign * value)
                seed_effects[term].append(2.0 * float(np.mean(contrast)))
        samples = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
        for term, values in seed_effects.items():
            array = np.asarray(values, dtype=float)
            boot = np.mean(array[samples], axis=1)
            output.append(
                {
                    "scenario_id": central,
                    "term": term,
                    "metric": metric,
                    "effect_mean_ms": float(np.mean(array)),
                    "effect_ci95_low_ms": float(np.quantile(boot, 0.025)),
                    "effect_ci95_high_ms": float(np.quantile(boot, 0.975)),
                    "paired_seed_count": len(seeds),
                    "effect_sign": "negative favors enabling the factor/interaction",
                    "ablation_structure": "hierarchical_structural_2x2x2_closure",
                    "algorithm1_effect_interpretable_only_when_algorithm2_enabled": True,
                    "equivalent_cells_retained": "101==001; 100==000; 101==static_independent_fcfs",
                }
            )
    return pd.DataFrame(output)


def factorial_probe_effects(
    probes: pd.DataFrame,
    *,
    central_rate: float,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Hierarchical structural 2^3 closure from rate-zero K64 probes."""

    central = central_rate_zero_scenario_id(probes)
    data = probes[probes["scenario_id"] == central].copy()
    policies: dict[tuple[int, int, int], str] = {}
    for policy in data["policy"].unique():
        if str(policy).startswith("aegiskv_alg"):
            bits = str(policy).split("aegiskv_alg", 1)[1][:3]
            policies[tuple(int(bit) for bit in bits)] = str(policy)
    if len(policies) != 8:
        raise ValueError(f"rate-zero K64 2^3 cube incomplete: {policies}")
    metrics = {
        "conditional_actual_harm_mean_ms_per_event": lambda row: float(
            row["conditional_actual_harm_mean_ms"]
        ),
        "single_fault_normal_state_burden_ms_per_workflow_epoch_at_central_rate": (
            lambda row: central_rate
            / 1000.0
            * float(row["eligible_exposure_gpu_shard_hours"])
            * float(row["conditional_actual_harm_mean_ms"])
            / 393.0
        ),
    }
    terms = {
        "Algorithm 1": (0,),
        "Algorithm 2": (1,),
        "Algorithm 3": (2,),
        "Algorithm 1 x 2": (0, 1),
        "Algorithm 1 x 3": (0, 2),
        "Algorithm 2 x 3": (1, 2),
        "Algorithm 1 x 2 x 3": (0, 1, 2),
    }
    indexed = data.set_index(["policy", "seed"])
    seeds = sorted(data["seed"].unique())
    rng = np.random.default_rng(seed ^ 0xFAC064)
    output: list[dict[str, Any]] = []
    for metric, resolver in metrics.items():
        seed_effects = {term: [] for term in terms}
        for run_seed in seeds:
            values = {
                bits: resolver(indexed.loc[(policy, run_seed)])
                for bits, policy in policies.items()
            }
            for term, indices in terms.items():
                contrast = [
                    int(
                        np.prod(
                            [1 if bits[index] else -1 for index in indices]
                        )
                    )
                    * value
                    for bits, value in values.items()
                ]
                seed_effects[term].append(2.0 * float(np.mean(contrast)))
        for term, values in seed_effects.items():
            array = np.asarray(values, dtype=float)
            samples = rng.integers(0, len(array), size=(resamples, len(array)))
            boot = np.mean(array[samples], axis=1)
            output.append(
                {
                    "scenario_id": central,
                    "normal_state_failure_rate": 0.0,
                    "integrated_target_rate_per_1000_gpu_shard_hours": central_rate,
                    "term": term,
                    "metric": metric,
                    "effect_mean": float(np.mean(array)),
                    "effect_ci95_low": float(np.quantile(boot, 0.025)),
                    "effect_ci95_high": float(np.quantile(boot, 0.975)),
                    "paired_seed_count": len(array),
                    "probe_strata_per_seed": PROBE_COUNT,
                    "bootstrap_unit": "paired outer seed only; no probe-inner bootstrap",
                    "effect_sign": "negative favors enabling the factor/interaction",
                    "ablation_structure": "hierarchical_structural_2x2x2_closure",
                    "algorithm1_effect_interpretable_only_when_algorithm2_enabled": True,
                    "equivalent_cells_retained": "101==001; 100==000; 101==static_independent_fcfs",
                }
            )
    return pd.DataFrame(output)


def scenario_policy_audit(
    frame: pd.DataFrame, *, expected_seed_count: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario_id, policy), group in frame.groupby(["scenario_id", "policy"]):
        checks = {
            "expected_seed_count_met": len(group) == expected_seed_count,
            "offered_concurrency_gate": bool(
                (
                    group[
                        "workload.achieved_offered_trace_steady_concurrency_relative_error"
                    ]
                    <= 0.05 + 1e-12
                ).all()
            ),
            "alpha_bounded": bool(
                group["steady.effective_alpha_mean"].between(0.0, 1.0).all()
            ),
            "occupancy_bounded": bool(
                (group["steady.hbm_occupancy_peak_fraction"] <= 1.0 + 1e-9).all()
                and (
                    group["steady.dram_occupancy_peak_fraction"] <= 1.0 + 1e-9
                ).all()
            ),
            "reuse_bounded": bool(
                group["summary.input_reuse_fraction"].between(0.0, 1.0).all()
            ),
            "restore_bounded_when_exercised": bool(
                group["summary.restore_hit_fraction"]
                .dropna()
                .between(0.0, 1.0)
                .all()
            ),
            "copy_amplification_nonnegative": bool(
                (group["summary.copy_amplification"] >= 0.0).all()
                and np.isfinite(group["summary.copy_amplification"]).all()
            ),
        }
        status = "PASS" if all(checks.values()) else "WARN"
        first = group.iloc[0]
        rows.append(
            {
                "scenario_id": scenario_id,
                "policy": policy,
                "status": status,
                "checks_json": strict_json_dumps(checks, sort_keys=True),
                "seed_count": len(group),
                "expected_seed_count": expected_seed_count,
                "concurrency": first["active_workflow_concurrency"],
                "failure_rate_per_1000_gpu_shard_hours": first[
                    "failure_rate_per_1000_worker_hours"
                ],
                "fault_total_gpu_shard_exposure_hours_mean": group[
                    "fault.total_gpu_shard_exposure_hours"
                ].mean(),
                "observed_gpu_shard_fault_rate_mean": group[
                    "fault.observed_events_per_1000_gpu_shard_hours"
                ].mean(),
                "effective_alpha_mean": group["steady.effective_alpha_mean"].mean(),
                "effective_alpha_final_mean": group[
                    "steady.effective_alpha_final"
                ].mean(),
                "pressure_fraction_mean": group["steady.pressure_fraction_mean"].mean(),
                "hbm_occupancy_mean": group[
                    "steady.hbm_occupancy_mean_fraction"
                ].mean(),
                "dram_occupancy_mean": group[
                    "steady.dram_occupancy_mean_fraction"
                ].mean(),
                "input_reuse_fraction": group["summary.input_reuse_fraction"].mean(),
                "restore_hit_fraction": pooled_recovery_fraction(group)[0],
                "restore_fraction_aggregation": (
                    "pooled restored/recovered tokens; zero denominator is NA"
                ),
                "copy_amplification": (
                    float(
                        group["summary.protection_copy_attempted_bytes"].sum()
                    )
                    / float(group["summary.logical_request_kv_bytes"].sum())
                ),
                "repair_events_scheduled_mean": group[
                    "summary.worker_repair_events_scheduled"
                ].mean(),
                "repair_events_effective_mean": group[
                    "summary.worker_repair_events_effective"
                ].mean(),
                "repair_events_superseded_mean": group[
                    "summary.worker_repair_events_superseded"
                ].mean(),
                "realized_base_active_mean": group[
                    "realized.base_full_horizon.time_mean"
                ].mean(),
                "realized_simulated_active_mean": group[
                    "realized.simulated_full_horizon.time_mean"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def scenario_diagnostics(
    frame: pd.DataFrame,
    probes: pd.DataFrame,
    burdens: pd.DataFrame,
    artifacts: list[tuple[Path, dict[str, Any]]],
) -> pd.DataFrame:
    """One complete, machine-audited diagnostics row per scenario for Alg111."""

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for _path, artifact in artifacts:
        if artifact["policy"] == "aegiskv_alg111":
            by_scenario.setdefault(artifact["scenario"]["scenario_id"], []).append(
                artifact
            )
    output: list[dict[str, Any]] = []
    for scenario_id, group in frame[frame["policy"] == "aegiskv_alg111"].groupby(
        "scenario_id"
    ):
        raw = by_scenario[scenario_id]
        probe_group = probes[
            (probes["scenario_id"] == scenario_id)
            & (probes["policy"] == "aegiskv_alg111")
        ]
        burden_group = burdens[
            (burdens["scenario_id"] == scenario_id)
            & (burdens["policy"] == "aegiskv_alg111")
        ]
        admission: dict[str, int] = {}
        churn = {
            "HBM": {"eviction_count": 0, "admission_rejection_count": 0},
            "DRAM": {"eviction_count": 0, "admission_rejection_count": 0},
        }
        request_reclaims = 0
        for artifact in raw:
            for name, count in artifact["summary"]["admission_outcomes"].items():
                admission[name] = admission.get(name, 0) + int(count)
            for target in artifact["final_state"].get("target_snapshots", []):
                request_reclaims += int(
                    target.get("metrics", {}).get("request_reclaims", 0)
                )
                for pool in ("HBM", "DRAM"):
                    values = target.get("pool_churn", {}).get(pool, {})
                    for metric in churn[pool]:
                        churn[pool][metric] += int(values.get(metric, 0))
        restored = int(probe_group["restored_tokens"].sum())
        recomputed = int(probe_group["recomputed_tokens"].sum())
        recovery_total = restored + recomputed
        checks = {
            "seed_count_positive": len(group) > 0,
            "alpha_bounded": bool(
                group["steady.effective_alpha_min"].between(0.0, 1.0).all()
                and group["steady.effective_alpha_max"].between(0.0, 1.0).all()
            ),
            "occupancy_bounded": bool(
                (group["steady.hbm_occupancy_peak_fraction"] <= 1.0 + 1e-9).all()
                and (
                    group["steady.dram_occupancy_peak_fraction"] <= 1.0 + 1e-9
                ).all()
            ),
            "probe_k64_every_seed": bool((probe_group["probe_count"] == 64).all()),
            "reuse_bounded": bool(
                group["summary.input_reuse_fraction"].between(0.0, 1.0).all()
            ),
            "copy_amplification_nonnegative": bool(
                (group["summary.copy_amplification"] >= 0.0).all()
            ),
            "failure_history_accounting_exact": bool(
                group[
                    "history.history_accounting_matches_raw_summary_and_final_state"
                ]
                .astype(bool)
                .all()
            ),
            "failure_zero_exact": bool(
                not np.isclose(
                    float(group["failure_rate_per_1000_worker_hours"].iloc[0]),
                    0.0,
                )
                or (
                    (group["summary.failure_events_injected"] == 0).all()
                    and np.isclose(
                        burden_group[
                            "normal_state_burden_total_ms_per_trace_epoch"
                        ],
                        0.0,
                    ).all()
                )
            ),
        }
        reason_codes = [
            "SIMULATED_SYNTHETIC_COMPUTE_AND_BANDWIDTH",
            "FAILURE_RATE_NOT_DIRECT_ALGORITHM3_INPUT",
            (
                "EFFECTIVE_ALPHA_ZERO_OBSERVED"
                if (group["steady.alpha_zero_window_fraction"] > 0).any()
                else "EFFECTIVE_ALPHA_ZERO_NOT_OBSERVED"
            ),
            (
                "UPTIME_EFFECTIVE_FAULTS_OBSERVED"
                if int(group["summary.failure_events_effective"].sum()) > 0
                else "NO_EFFECTIVE_UPTIME_FAULTS_OBSERVED"
            ),
            (
                "BOUND_TARGET_CURRENTLY_UNAVAILABLE_HISTORY_OBSERVED"
                if int(
                    group[
                        "history.uptime_effective_faults_with_bound_target_currently_unavailable"
                    ].sum()
                )
                > 0
                else "NO_BOUND_TARGET_CURRENTLY_UNAVAILABLE_HISTORY_OBSERVED"
            ),
            (
                "REPEATED_PHYSICAL_GPU_SHARD_FAILURE_HISTORY_OBSERVED"
                if int(
                    group[
                        "history.uptime_physical_gpu_shard_repeat_failure_events"
                    ].sum()
                )
                > 0
                else "NO_REPEATED_PHYSICAL_GPU_SHARD_FAILURE_HISTORY_OBSERVED"
            ),
            (
                "BOUND_TARGET_EVER_FAILED_HISTORY_OBSERVED"
                if int(
                    group[
                        "history.uptime_effective_faults_with_bound_target_ever_failed"
                    ].sum()
                )
                > 0
                else "NO_BOUND_TARGET_EVER_FAILED_HISTORY_OBSERVED"
            ),
            (
                "OLD_HBM_COPY_INVALIDATION_OBSERVED"
                if int(
                    group[
                        "history.old_hbm_copy_invalidation_event_count_cumulative"
                    ].sum()
                )
                > 0
                else "NO_OLD_HBM_COPY_INVALIDATION_OBSERVED"
            ),
            (
                "EXACT_RATE_ZERO_NORMAL_STATE_REFERENCE_AVAILABLE"
                if (
                    burden_group["normal_state_match_status"]
                    == "MATCHED_EXACT_RATE_ZERO"
                ).all()
                else "NORMAL_STATE_BURDEN_NA_NO_EXACT_RATE_ZERO_MATCH"
            ),
        ]
        first = group.iloc[0]
        logical_bytes = float(group["summary.logical_request_kv_bytes"].sum())
        output.append(
            {
                "scenario_id": scenario_id,
                "campaign_axis": first["campaign_axis"],
                "model_id": first["model_id"],
                "prompt_scale": first["prompt_scale"],
                "active_workflow_concurrency": first[
                    "active_workflow_concurrency"
                ],
                "failure_rate_per_1000_gpu_shard_hours": first[
                    "failure_rate_per_1000_worker_hours"
                ],
                "total_gpu_count": first["total_gpu_count"],
                "tp_size": first["tp_size"],
                "pp_size": first["pp_size"],
                "system_dp_size": first["system_dp_size"],
                "network_effective_gb_s": first["network_effective_gb_s"],
                "hbm_quota_gib_per_gpu_shard": first[
                    "hbm_quota_gib_per_gpu_shard"
                ],
                "dram_quota_gib_per_gpu_shard": first[
                    "dram_quota_gib_per_gpu_shard"
                ],
                "hbm_quota_gib_per_logical_dp_target": first[
                    "hbm_quota_gib_per_logical_dp_target"
                ],
                "dram_quota_gib_per_logical_dp_target": first[
                    "dram_quota_gib_per_logical_dp_target"
                ],
                "seed_count": len(group),
                "effective_alpha_mean": float(
                    group["steady.effective_alpha_mean"].mean()
                ),
                "effective_alpha_min": float(
                    group["steady.effective_alpha_min"].min()
                ),
                "effective_alpha_max": float(
                    group["steady.effective_alpha_max"].max()
                ),
                "effective_alpha_zero_window_fraction": float(
                    group["steady.alpha_zero_window_fraction"].mean()
                ),
                "pressure_fraction_mean": float(
                    group["steady.pressure_fraction_mean"].mean()
                ),
                "pressure_fraction_peak": float(
                    group["steady.pressure_fraction_peak"].max()
                ),
                "hbm_occupancy_mean_fraction": float(
                    group["steady.hbm_occupancy_mean_fraction"].mean()
                ),
                "hbm_occupancy_peak_fraction": float(
                    group["steady.hbm_occupancy_peak_fraction"].max()
                ),
                "dram_occupancy_mean_fraction": float(
                    group["steady.dram_occupancy_mean_fraction"].mean()
                ),
                "dram_occupancy_peak_fraction": float(
                    group["steady.dram_occupancy_peak_fraction"].max()
                ),
                **{
                    f"admission_{name.lower()}_count": admission.get(name, 0)
                    for name in ("HBM_ONLY", "DRAM_ONLY", "HBM_AND_DRAM", "BYPASS")
                },
                "hbm_eviction_count": churn["HBM"]["eviction_count"],
                "hbm_admission_rejection_count": churn["HBM"][
                    "admission_rejection_count"
                ],
                "dram_eviction_count": churn["DRAM"]["eviction_count"],
                "dram_admission_rejection_count": churn["DRAM"][
                    "admission_rejection_count"
                ],
                "request_reclaims": request_reclaims,
                "protection_copy_attempted_bytes": int(
                    group["summary.protection_copy_attempted_bytes"].sum()
                ),
                "copy_amplification_pooled": (
                    float(group["summary.protection_copy_attempted_bytes"].sum())
                    / logical_bytes
                    if logical_bytes
                    else math.nan
                ),
                "input_reuse_request_fraction_mean": float(
                    group["summary.requests_with_input_reuse"].sum()
                    / group["summary.request_count"].sum()
                ),
                "input_reuse_token_fraction_pooled": float(
                    group["summary.input_reuse_tokens"].sum()
                    / group["summary.input_prompt_tokens"].sum()
                ),
                "probe_count_total": int(probe_group["probe_count"].sum()),
                "probe_peer_hbm_path_fraction": float(
                    probe_group["source_class_peer_hbm_count"].sum()
                    / max(int(probe_group["probe_count"].sum()), 1)
                ),
                "probe_dram_path_fraction": float(
                    probe_group["source_class_dram_count"].sum()
                    / max(int(probe_group["probe_count"].sum()), 1)
                ),
                "probe_recompute_path_fraction": float(
                    probe_group["source_class_recompute_count"].sum()
                    / max(int(probe_group["probe_count"].sum()), 1)
                ),
                "probe_restore_hit_fraction_pooled_tokens": (
                    restored / recovery_total if recovery_total else math.nan
                ),
                "probe_recompute_fraction_pooled_tokens": (
                    recomputed / recovery_total if recovery_total else math.nan
                ),
                "probe_recovery_mean_ms": float(
                    probe_group["conditional_recovery_mean_ms"].mean()
                ),
                "probe_actual_harm_mean_ms": float(
                    probe_group["conditional_actual_harm_mean_ms"].mean()
                ),
                "normal_state_burden_ms_per_workflow_epoch_mean": float(
                    burden_group[
                        "normal_state_burden_ms_per_workflow_epoch"
                    ].mean()
                ),
                "uptime_failure_events_injected": int(
                    group["summary.failure_events_injected"].sum()
                ),
                "uptime_failure_events_effective": int(
                    group["summary.failure_events_effective"].sum()
                ),
                "uptime_physical_gpu_shard_repeat_failure_events": int(
                    group[
                        "history.uptime_physical_gpu_shard_repeat_failure_events"
                    ].sum()
                ),
                "uptime_effective_faults_on_previously_failed_physical_gpu_shard": int(
                    group[
                        "history.uptime_effective_faults_on_previously_failed_physical_gpu_shard"
                    ].sum()
                ),
                "uptime_effective_faults_with_bound_target_currently_unavailable": int(
                    group[
                        "history.uptime_effective_faults_with_bound_target_currently_unavailable"
                    ].sum()
                ),
                "uptime_effective_faults_with_bound_target_ever_failed": int(
                    group[
                        "history.uptime_effective_faults_with_bound_target_ever_failed"
                    ].sum()
                ),
                "old_hbm_copy_invalidation_event_count_cumulative": int(
                    group[
                        "history.old_hbm_copy_invalidation_event_count_cumulative"
                    ].sum()
                ),
                "old_hbm_copy_invalidated_requests_cumulative": int(
                    group[
                        "history.old_hbm_copy_invalidated_requests_cumulative"
                    ].sum()
                ),
                "old_hbm_copy_invalidated_bytes_cumulative": int(
                    group[
                        "history.old_hbm_copy_invalidated_bytes_cumulative"
                    ].sum()
                ),
                "uptime_exposure_gpu_shard_hours": float(
                    group["fault.total_gpu_shard_exposure_hours"].sum()
                ),
                "machine_invariant_status": (
                    "PASS" if all(checks.values()) else "WARN"
                ),
                "machine_invariant_checks_json": strict_json_dumps(
                    checks, sort_keys=True
                ),
                "narrative_reason_codes": ";".join(reason_codes),
            }
        )
    return pd.DataFrame(output).sort_values("scenario_id").reset_index(drop=True)


def rationality_audit(frame: pd.DataFrame) -> dict[str, Any]:
    full = frame[frame["policy"] == "aegiskv_alg111"]
    zero_failure = frame[
        np.isclose(frame["failure_rate_per_1000_worker_hours"], 0.0)
    ]
    checks: dict[str, Any] = {
        "failure_rate_zero_has_exactly_zero_injected_events": bool(
            (zero_failure["summary.failure_events_injected"] == 0).all()
        ),
        "failure_rate_zero_has_exactly_zero_actual_harm_surrogate": bool(
            np.isclose(
                zero_failure[
                    "summary.actual_failure_added_workflow_harm_surrogate_mean_ms"
                ],
                0.0,
            ).all()
        ),
        "failure_rate_zero_has_exactly_zero_visible_harm": bool(
            np.isclose(
                zero_failure["summary.actual_modeled_visible_harm_mean_ms"],
                0.0,
            ).all()
        ),
        "repair_event_count_matches_injected_fault_count": bool(
            (
                frame["summary.worker_repair_events_scheduled"]
                == frame["summary.failure_events_injected"]
            ).all()
        ),
        "repair_event_effective_plus_superseded_not_above_scheduled": bool(
            (
                frame["summary.worker_repair_events_effective"]
                + frame["summary.worker_repair_events_superseded"]
                <= frame["summary.worker_repair_events_scheduled"]
            ).all()
        ),
        "offered_schedule_within_5pct": bool(
            (
                full[
                    "workload.achieved_offered_trace_steady_concurrency_relative_error"
                ]
                <= 0.05 + 1e-12
            ).all()
        ),
        "all_393_workflows_retained_in_aggregate_accounting": bool(
            full["workload.all_393_workflows_in_wct"].astype(bool).all()
            and (~full["workload.stabilization_prefix_excluded_from_wct"].astype(bool)).all()
        ),
        "steady_alpha_bounded": bool(
            full["steady.effective_alpha_mean"].between(0.0, 1.0).all()
            and full["steady.effective_alpha_min"].between(0.0, 1.0).all()
            and full["steady.effective_alpha_max"].between(0.0, 1.0).all()
        ),
        "steady_occupancy_bounded": bool(
            (full["steady.hbm_occupancy_peak_fraction"] <= 1.0 + 1e-9).all()
            and (full["steady.dram_occupancy_peak_fraction"] <= 1.0 + 1e-9).all()
        ),
    }
    response = full[
        full["campaign_axis"].str.contains("concurrency_failure", regex=False)
    ]
    alpha_metrics = (
        "steady.effective_alpha_mean",
        "steady.effective_alpha_final",
    )
    alpha_violations: list[dict[str, Any]] = []
    for metric in alpha_metrics:
        means = response.groupby(
            ["failure_rate_per_1000_worker_hours", "active_workflow_concurrency"]
        )[metric].mean()
        for failure_rate in sorted(
            response["failure_rate_per_1000_worker_hours"].unique()
        ):
            group = means.loc[failure_rate].sort_index()
            for (prior_c, prior), (current_c, current) in zip(
                group.items(), list(group.items())[1:]
            ):
                if float(current) > float(prior) + 0.02:
                    alpha_violations.append(
                        {
                            "metric": metric,
                            "failure_rate": float(failure_rate),
                            "prior_concurrency": int(prior_c),
                            "current_concurrency": int(current_c),
                            "prior_alpha": float(prior),
                            "current_alpha": float(current),
                        }
                    )
    required = sorted(checks)

    # Directional performance checks are diagnostics, never gates.  Negative
    # or non-monotone results must be retained and discussed.
    bandwidth = full[
        full["campaign_axis"].str.contains("network_bandwidth", regex=False)
    ]
    bandwidth_violations: list[dict[str, Any]] = []
    for run_seed, group in bandwidth.groupby("seed"):
        ordered = group.sort_values("network_effective_gb_s")
        values = ordered[
            "summary.actual_failure_added_workflow_harm_surrogate_mean_ms"
        ].to_numpy(float)
        rates = ordered["network_effective_gb_s"].to_numpy(float)
        for index, (prior, current) in enumerate(zip(values, values[1:])):
            if current > prior + max(abs(prior) * 1e-9, 1e-9):
                bandwidth_violations.append(
                    {
                        "seed": int(run_seed),
                        "prior_bandwidth": float(rates[index]),
                        "current_bandwidth": float(rates[index + 1]),
                        "prior_actual_harm_surrogate_ms": float(prior),
                        "current_actual_harm_surrogate_ms": float(current),
                    }
                )
    diagnostics = {
        "failure_rate_alpha_monotonicity_required": False,
        "stable_windows_additive_increase_observed": bool(
            (full["steady.stable_alpha_increase_transitions"] > 0).any()
        ),
        "pressure_multiplicative_decrease_observed": bool(
            (full["steady.pressure_alpha_decrease_transitions"] > 0).any()
        ),
        "aimd_transition_exercise_is_diagnostic_not_validity_gate": True,
        "failure_rate_alpha_interpretation": (
            "conditional correlation only; pressure/headroom is the direct Algorithm 3 input"
        ),
        "bandwidth_superiority_is_not_a_required_gate": True,
        "bandwidth_nonincreasing_actual_harm_surrogate_observed": (
            not bandwidth_violations
        ),
        "bandwidth_actual_harm_surrogate_violations": bandwidth_violations,
        "higher_load_alpha_violations": alpha_violations,
        "higher_offered_load_alpha_nonincreasing_tolerance_0p02_observed": (
            not alpha_violations
        ),
        "higher_load_alpha_direction_is_diagnostic_not_validity_gate": True,
        "alpha_zero_is_observation_not_validity_or_superiority_gate": True,
        "repair_effective_count_is_right_censorable": True,
        "repair_effective_or_superseded_observed_anywhere": bool(
            (
                full["summary.worker_repair_events_effective"]
                + full["summary.worker_repair_events_superseded"]
                > 0
            ).any()
        ),
        "high_pressure_effective_alpha_zero_observed": bool(
            (
                (full["steady.pressure_fraction_mean"] >= 0.85)
                & (full["steady.effective_alpha_min"] <= 1e-12)
            ).any()
        ),
        "highest_offered_load_effective_alpha_min": float(
            full.loc[
                full["active_workflow_concurrency"]
                == full["active_workflow_concurrency"].max(),
                "steady.effective_alpha_min",
            ].min()
        ),
        "highest_offered_load_alpha_zero_window_fraction_mean": float(
            full.loc[
                full["active_workflow_concurrency"]
                == full["active_workflow_concurrency"].max(),
                "steady.alpha_zero_window_fraction",
            ].mean()
        ),
    }
    return {
        "status": "PASS" if all(bool(checks[key]) for key in required) else "FAIL",
        "checks": checks,
        "required_checks": required,
        "directional_diagnostics_not_gates": diagnostics,
    }


def central_timelines(
    artifacts: list[tuple[Path, dict[str, Any]]], frame: pd.DataFrame
) -> pd.DataFrame:
    central = central_scenario_id(frame)
    keep = {"aegiskv_alg111", "aegiskv_alg110_no_alg3", "recompute_only"}
    rows: list[dict[str, Any]] = []
    for _path, artifact in artifacts:
        if artifact["scenario"]["scenario_id"] != central or artifact["policy"] not in keep:
            continue
        for row in artifact["window_rows"]:
            if not bool(row.get("in_measurement_window")):
                continue
            rows.append(
                {
                    "run_id": artifact["run_id"],
                    "seed": artifact["seed"],
                    "policy": artifact["policy"],
                    "time_ms": row["time_ms"],
                    "window_index": row["window_index"],
                    "controller_alpha_mean": row["controller_alpha_mean"],
                    "effective_alpha_mean": row["effective_alpha_mean"],
                    "pressure_fraction": row["pressure_fraction"],
                    "active_requests": row["active_requests"],
                    "queued_requests": row["queued_requests"],
                    "hbm_used_bytes": row["hbm_used_bytes"],
                    "dram_used_bytes": row["dram_used_bytes"],
                    "hbm_capacity_bytes": artifact["summary"]["hbm_capacity_bytes"],
                    "dram_capacity_bytes": artifact["summary"]["dram_capacity_bytes"],
                }
            )
    return pd.DataFrame(rows)


def paper_summary_payload(
    frame: pd.DataFrame,
    burdens: pd.DataFrame,
    paired_probe: pd.DataFrame,
    paired_workflow: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
    source_hashes: dict[str, str],
    analysis_status: str,
    selection_mode: str,
    paper_eligible: bool,
) -> dict[str, Any]:
    """Machine-readable source of every candidate paper number.

    This deliberately retains null and negative results; it performs no
    advantage-based filtering.
    """

    rng = np.random.default_rng(seed ^ 0xA6E15)
    records: list[dict[str, Any]] = []
    metrics = (
        "conditional_recovery_mean_ms",
        "conditional_actual_harm_mean_ms",
        "normal_state_burden_total_ms_per_trace_epoch",
        "normal_state_burden_ms_per_workflow_epoch",
    )
    absolute_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (scenario_id, policy), group in burdens.groupby(["scenario_id", "policy"]):
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                samples = rng.integers(0, len(values), size=(resamples, len(values)))
                boot = np.mean(values[samples], axis=1)
                absolute = float(np.mean(values))
                low = float(np.quantile(boot, 0.025))
                high = float(np.quantile(boot, 0.975))
            else:
                absolute = low = high = None
            record = {
                "estimand": (
                    "B active+sealed fault-conditioned recovery"
                    if metric.startswith("conditional_")
                    else "C single-fault normal-state eligible hazard-integrated additive burden"
                ),
                "metric": metric,
                "absolute": absolute,
                "paired_delta": None,
                "ci95_low": low,
                "ci95_high": high,
                "ci_target": "absolute",
                "n_paired_seeds": int(len(values)),
                "probe_strata_per_seed": PROBE_COUNT,
                "scenario_id": scenario_id,
                "policy": policy,
                "reference_policy": None,
                "source_hashes": source_hashes,
                "negative_or_null_retained": True,
            }
            records.append(record)
            absolute_lookup[(scenario_id, policy, metric)] = record
    for row in paired_probe.itertuples(index=False):
        treatment = absolute_lookup.get((row.scenario_id, row.policy, row.metric), {})
        records.append(
            {
                "estimand": (
                    "B active+sealed fault-conditioned recovery"
                    if str(row.metric).startswith("conditional_")
                    else "C single-fault normal-state eligible hazard-integrated additive burden"
                ),
                "metric": row.metric,
                "absolute": treatment.get("absolute"),
                "paired_delta": (
                    float(row.mean_delta) if math.isfinite(float(row.mean_delta)) else None
                ),
                "ci95_low": (
                    float(row.delta_ci95_low)
                    if math.isfinite(float(row.delta_ci95_low))
                    else None
                ),
                "ci95_high": (
                    float(row.delta_ci95_high)
                    if math.isfinite(float(row.delta_ci95_high))
                    else None
                ),
                "ci_target": "paired_delta",
                "n_paired_seeds": int(row.paired_seed_count),
                "probe_strata_per_seed": PROBE_COUNT,
                "scenario_id": row.scenario_id,
                "policy": row.policy,
                "reference_policy": row.reference,
                "source_hashes": source_hashes,
                "negative_or_null_retained": True,
            }
        )
    summary_lookup = frame.set_index(["scenario_id", "policy"])
    for row in paired_workflow.itertuples(index=False):
        values = summary_lookup.loc[(row.scenario_id, row.policy)][
            "summary.actual_failure_added_workflow_harm_surrogate_mean_ms"
        ]
        absolute = float(np.mean(np.atleast_1d(values).astype(float)))
        records.append(
            {
                "estimand": "A full-horizon uptime-Poisson per-workflow max-overrun join-harm surrogate",
                "metric": row.metric,
                "absolute": absolute,
                "paired_delta": float(row.mean_delta_ms),
                "ci95_low": float(row.delta_ci95_low_ms),
                "ci95_high": float(row.delta_ci95_high_ms),
                "ci_target": "paired_delta",
                "n_paired_seeds": int(row.paired_seed_count),
                "paired_workflows_per_seed": int(row.paired_workflow_count_per_seed),
                "scenario_id": row.scenario_id,
                "policy": row.policy,
                "reference_policy": row.reference,
                "source_hashes": source_hashes,
                "negative_or_null_retained": True,
            }
        )
    return {
        "schema_version": 1,
        "evidence_class": "simulated_sglsim_only_paper_number_source",
        "physical_result": False,
        "analysis_status": analysis_status,
        "selection_mode": selection_mode,
        "paper_eligible": paper_eligible,
        "single_seed_ci_semantics": (
            "n=1 pilot intervals are degenerate descriptive points, not inferential CIs"
        ),
        "selection_contract": (
            "all candidate absolute and paired numbers retained; no advantage "
            "filtering; null and negative results retained"
        ),
        "source_hashes": source_hashes,
        "records": records,
    }


def resolve_analysis_evidence_status(
    *,
    selection_mode: str,
    allowed_status: str,
    complete: bool,
    observed_run_count: int,
    allow_partial: bool,
    formal_gates_pass: bool,
) -> tuple[str, bool]:
    """Never let a partial or diagnostic selection self-promote to COMPLETE."""

    if selection_mode == "CUSTOM_PARTIAL_DIAGNOSTIC":
        status = (
            "CUSTOM_PARTIAL_DIAGNOSTIC"
            if complete
            else "FAILED_GATE"
        )
    elif selection_mode == "DEFAULT_NINE_RUN_MECHANISM_PILOT":
        status = (
            "PILOT_DIAGNOSTIC"
            if complete and formal_gates_pass
            else "FAILED_GATE"
        )
    elif selection_mode == "FULL_FROZEN_MATRIX":
        status = (
            "COMPLETE"
            if complete
            and observed_run_count == 3110
            and not allow_partial
            and formal_gates_pass
            and allowed_status == "COMPLETE"
            else "FAILED_GATE"
        )
    else:
        status = "FAILED_GATE"
    paper_eligible = bool(
        status == "COMPLETE" and selection_mode == "FULL_FROZEN_MATRIX"
    )
    return status, paper_eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir.parent / f"analysis-{input_dir.name}"
    )
    index, artifacts = load_campaign(input_dir)
    protocol = json.loads(
        Path(index["protocol"]["path"]).read_text(encoding="utf-8")
    )
    matrix = json.loads(
        Path(index["matrix"]["path"]).read_text(encoding="utf-8")
    )
    if not artifacts:
        raise FileNotFoundError(f"no artifacts under {input_dir}")
    if len(artifacts) != int(index["requested_run_count"]) and not args.allow_partial:
        raise ValueError("campaign incomplete; --allow-partial is diagnostic only")
    contract = audit_artifact_contract(
        index,
        artifacts,
        matrix,
        protocol,
        allow_partial=args.allow_partial,
    )
    if contract["status"] != "PASS":
        raise ValueError(f"campaign contract failed: {contract}")
    selection_mode = str(index["selection"]["mode"])

    frame = summary_frame(artifacts)
    probes = probe_seed_frame(artifacts)
    reuse = probe_reuse_frame(artifacts)
    burdens = normal_state_hazard_burden(probes)
    paired_probes = paired_probe_outer_seed_bootstrap(
        burdens,
        resamples=int(protocol["statistics"]["bootstrap_resamples"]),
        seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    aggregate = axis_aggregates(
        frame,
        resamples=int(protocol["statistics"]["bootstrap_resamples"]),
        seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    scenario_audit = scenario_policy_audit(
        frame, expected_seed_count=1 if bool(index["pilot"]) else 10
    )
    scenario_detail = scenario_diagnostics(frame, probes, burdens, artifacts)
    rationality = rationality_audit(frame)
    algorithm1_identifiability = algorithm1_identifiability_audit(
        artifacts, frame
    )
    workflows = paired_workflow_seed_bootstrap(
        artifacts,
        resamples=int(protocol["statistics"]["bootstrap_resamples"]),
        seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    complete = len(artifacts) == int(index["requested_run_count"])
    full_complete = (
        complete
        and not bool(index["pilot"])
        and selection_mode == "FULL_FROZEN_MATRIX"
        and len(artifacts) == 3110
        and not args.allow_partial
    )
    full_ten_seed = full_complete and frame["seed"].nunique() == 10
    if full_ten_seed:
        hierarchical = hierarchical_central_bootstrap(
            artifacts,
            frame,
            resamples=int(protocol["statistics"]["bootstrap_resamples"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]),
        )
        factorial = factorial_probe_effects(
            probes,
            central_rate=float(protocol["failure"]["central_rate"]),
            resamples=int(protocol["statistics"]["bootstrap_resamples"]),
            seed=int(protocol["statistics"]["bootstrap_seed"]),
        )
    else:
        hierarchical = pd.DataFrame()
        factorial = pd.DataFrame()
    if selection_mode == "CUSTOM_PARTIAL_DIAGNOSTIC":
        comparison_coverage = {
            "status": "NOT_APPLICABLE_CUSTOM_PARTIAL",
            "mode": selection_mode,
            "checks": {
                "formal_comparison_coverage_not_claimed": True,
            },
        }
    else:
        comparison_coverage = paired_comparison_coverage(
            workflows,
            hierarchical,
            artifacts,
            matrix,
            pilot=bool(index["pilot"]),
        )
    if selection_mode == "CUSTOM_PARTIAL_DIAGNOSTIC":
        probe_comparison_coverage = {
            "status": "NOT_APPLICABLE_CUSTOM_PARTIAL",
            "mode": selection_mode,
            "checks": {"formal_probe_comparison_coverage_not_claimed": True},
        }
    else:
        probe_comparison_coverage = paired_probe_comparison_coverage(
            paired_probes,
            workflows,
            burdens,
            pilot=bool(index["pilot"]),
        )
    timelines = central_timelines(artifacts, frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary_rows.csv": frame,
        "active_source_probe_seed_rows.csv": probes,
        "active_source_probe_reuse_bins.csv": reuse,
        "normal_state_hazard_burden.csv": burdens,
        "paired_probe_outer_seed_bootstrap.csv": paired_probes,
        "axis_aggregates.csv": aggregate,
        "scenario_policy_audit.csv": scenario_audit,
        "scenario_diagnostics.csv": scenario_detail,
        "paired_workflow_seed_bootstrap.csv": workflows,
        "central_hierarchical_bootstrap.csv": hierarchical,
        "factorial_effects.csv": factorial,
        "central_steady_timelines.csv": timelines,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / name, index=False)
    scenario_json_rows = json.loads(
        scenario_detail.to_json(orient="records", double_precision=15)
    )
    (output_dir / "scenario_diagnostics.json").write_text(
        strict_json_dumps(
            {
                "schema_version": 1,
                "evidence_class": "simulated_sglsim_only_scenario_diagnostics",
                "scenario_count": len(scenario_json_rows),
                "all_negative_and_warn_results_retained": True,
                "rows": scenario_json_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostic_columns = [
        "scenario_id",
        "campaign_axis",
        "model_id",
        "active_workflow_concurrency",
        "failure_rate_per_1000_gpu_shard_hours",
        "effective_alpha_mean",
        "effective_alpha_zero_window_fraction",
        "pressure_fraction_mean",
        "hbm_occupancy_mean_fraction",
        "dram_occupancy_mean_fraction",
        "probe_restore_hit_fraction_pooled_tokens",
        "probe_actual_harm_mean_ms",
        "machine_invariant_status",
        "narrative_reason_codes",
    ]
    (output_dir / "scenario_diagnostics.md").write_text(
        "# Per-scenario diagnostics\n\n"
        "All rows are simulated-only; directional trends and alpha=0 are "
        "diagnostics, not validity or superiority gates.\n\n"
        + dataframe_to_markdown(scenario_detail[diagnostic_columns])
        + "\n",
        encoding="utf-8",
    )
    for name, payload in (
        ("contract_audit.json", contract),
        ("rationality_audit.json", rationality),
        (
            "algorithm1_identifiability_audit.json",
            algorithm1_identifiability,
        ),
        ("paired_comparison_coverage.json", comparison_coverage),
        ("paired_probe_comparison_coverage.json", probe_comparison_coverage),
        (
            "active_source_probe_contract_audit.json",
            contract["active_source_probe_contract"],
        ),
    ):
        (output_dir / name).write_text(
            strict_json_dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    (
        _compact_manifest_path,
        compact_manifest,
        _modeled_manifest_path,
        modeled_manifest,
    ) = load_lineage_cache_manifests(index)
    lineage_files = index["lineage"]["files"]
    prediction_diagnostics = {
        "schema_version": 1,
        "raw_weka_api_think_time_crossfit": compact_manifest["algorithm1"],
        "sglsim_modeled_service_crossfit": {
            "evidence_class": modeled_manifest["evidence_class"],
            "evidence_boundary": modeled_manifest["evidence_boundary"],
            "workflow_disjoint_crossfit": modeled_manifest[
                "workflow_disjoint_crossfit"
            ],
            "remaining": modeled_manifest["remaining"],
            "slack": modeled_manifest["slack"],
        },
        "interpretation": (
            "raw Weka duration prediction is retained as negative robustness "
            "evidence; the modeled-service predictor measures fidelity only to "
            "the declared SGLSim service model and is not physical validation. "
            "The actual-Slack endpoint is an actual-duration-derived boundary-CPM "
            "label with elapsed due-time decrement, not an exact fault-time "
            "concurrent-branch residual-DAG recomputation. Internal ground_truth "
            "gate names mean policy-independent offline label only."
        ),
        "sources": {
            "compact_manifest_sha256": lineage_files["compact_manifest"][
                "sha256"
            ],
            "modeled_manifest_sha256": lineage_files[
                "modeled_algorithm1_manifest"
            ]["sha256"],
            "producer_lineage_sha256": index["lineage"]["lineage_sha256"],
        },
    }
    trace_diagnostics = {
        "schema_version": 1,
        "source": compact_manifest["source"],
        "dag": compact_manifest["dag"],
        "reuse": compact_manifest["reuse"],
        "workflow_reuse": compact_manifest["workflow_reuse"],
        "reuse_scope": "workflow-local 64-token hash-id prefix blocks",
        "protection_copy_deduplication_implemented": False,
        "interpretation": (
            "input prefix reuse is a workload characteristic; restore-hit and "
            "recompute fractions are separate failure-recovery outcomes"
        ),
    }
    for name, payload in (
        ("algorithm1_prediction_diagnostics.json", prediction_diagnostics),
        ("trace_workload_diagnostics.json", trace_diagnostics),
    ):
        (output_dir / name).write_text(
            strict_json_dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    analysis_sources = {}
    for name in ("analyze.py", "plot_results.py", "audit_campaign.py"):
        path = ROOT / name
        if path.is_file():
            analysis_sources[name] = {"path": str(path), "sha256": sha256(path)}
    raw_hashes = [
        {"path": str(path), "sha256": sha256(path), "run_id": artifact["run_id"]}
        for path, artifact in artifacts
    ]
    source_hashes = {
        "campaign_index_sha256": sha256(input_dir / "campaign-index.json"),
        "protocol_sha256": sha256(Path(index["protocol"]["path"])),
        "matrix_sha256": sha256(Path(index["matrix"]["path"])),
        "producer_lineage_sha256": index["lineage"]["lineage_sha256"],
        "compact_manifest_sha256": lineage_files["compact_manifest"]["sha256"],
        "modeled_algorithm1_manifest_sha256": lineage_files[
            "modeled_algorithm1_manifest"
        ]["sha256"],
        "simulator_core_sha256": lineage_files["core"]["sha256"],
        "authoritative_current_design_sha256": lineage_files[
            "authoritative_current_design"
        ]["sha256"],
        "analysis_script_sha256": analysis_sources["analyze.py"]["sha256"],
        "raw_artifact_set_sha256": canonical_json_sha(
            sorted((row["run_id"], row["sha256"]) for row in raw_hashes)
        ),
    }
    formal_gates_pass = (
        rationality["status"] == "PASS"
        and algorithm1_identifiability["status"] == "PASS"
        and contract["active_source_probe_contract"]["status"] == "PASS"
        and bool(
            (scenario_detail["machine_invariant_status"] == "PASS").all()
        )
        and comparison_coverage["status"]
        in {"PASS", "NOT_APPLICABLE_CUSTOM_PARTIAL"}
        and probe_comparison_coverage["status"]
        in {"PASS", "NOT_APPLICABLE_CUSTOM_PARTIAL"}
    )
    analysis_status, paper_eligible = resolve_analysis_evidence_status(
        selection_mode=selection_mode,
        allowed_status=str(index["selection"]["allowed_analysis_status"]),
        complete=complete,
        observed_run_count=len(artifacts),
        allow_partial=bool(args.allow_partial),
        formal_gates_pass=formal_gates_pass,
    )
    gates_pass = analysis_status != "FAILED_GATE"
    paper_summary = paper_summary_payload(
        frame,
        burdens,
        paired_probes,
        workflows,
        resamples=int(protocol["statistics"]["bootstrap_resamples"]),
        seed=int(protocol["statistics"]["bootstrap_seed"]),
        source_hashes=source_hashes,
        analysis_status=analysis_status,
        selection_mode=selection_mode,
        paper_eligible=paper_eligible,
    )
    (output_dir / "paper_summary.json").write_text(
        strict_json_dumps(paper_summary, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "evidence_class": "simulated_sglsim_only_analysis",
        "status": analysis_status,
        "selection_mode": selection_mode,
        "paper_eligible": paper_eligible,
        "physical_result": False,
        "campaign_index": {
            "path": str(input_dir / "campaign-index.json"),
            "sha256": sha256(input_dir / "campaign-index.json"),
        },
        "producer_lineage_sha256": index["lineage"]["lineage_sha256"],
        "analysis_sources": analysis_sources,
        "raw_artifacts": raw_hashes,
        "observed_run_count": len(artifacts),
        "workflow_rows_consumed": sum(len(a["workflow_rows"]) for _, a in artifacts),
        "window_rows_consumed": sum(len(a["window_rows"]) for _, a in artifacts),
        "failure_rows_consumed": sum(len(a["failure_rows"]) for _, a in artifacts),
        "repair_rows_consumed": sum(len(a["repair_rows"]) for _, a in artifacts),
        "active_source_probe_rows_consumed": sum(
            len(a["active_source_probe_rows"]) for _, a in artifacts
        ),
        "active_source_probe_contract_status": contract[
            "active_source_probe_contract"
        ]["status"],
        "scenario_diagnostics_row_count": len(scenario_detail),
        "scenario_diagnostics_status": (
            "PASS"
            if (scenario_detail["machine_invariant_status"] == "PASS").all()
            else "WARN"
        ),
        "contract_status": contract["status"],
        "rationality_status": rationality["status"],
        "algorithm1_identifiability_status": algorithm1_identifiability[
            "status"
        ],
        "paired_comparison_coverage_status": comparison_coverage["status"],
        "paired_probe_comparison_coverage_status": probe_comparison_coverage[
            "status"
        ],
        "formal_gates_pass": formal_gates_pass,
        "negative_results_retained": True,
        "probe_inner_bootstrap_used": False,
        "files": {},
    }
    for name in (
        *tables,
        "contract_audit.json",
        "rationality_audit.json",
        "algorithm1_identifiability_audit.json",
        "paired_comparison_coverage.json",
        "paired_probe_comparison_coverage.json",
        "active_source_probe_contract_audit.json",
        "algorithm1_prediction_diagnostics.json",
        "trace_workload_diagnostics.json",
        "scenario_diagnostics.json",
        "scenario_diagnostics.md",
        "paper_summary.json",
    ):
        path = output_dir / name
        manifest["files"][name] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    canonical = strict_json_dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    (output_dir / "analysis_manifest.json").write_text(
        strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(strict_json_dumps(manifest, indent=2))
    if not gates_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
