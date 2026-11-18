"""Fail-closed contract tests for the overnight campaign wrapper."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from analyze import (
    dataframe_to_markdown,
    json_native,
    strict_json_dumps,
    aimd_snapshot_contract_checks,
    load_lineage_cache_manifests,
    normal_state_hazard_burden,
    paired_comparison_coverage,
    paired_probe_comparison_coverage,
    paired_probe_outer_seed_bootstrap,
    paired_workflow_seed_bootstrap,
    pooled_recovery_fraction,
    resolve_analysis_evidence_status,
    uptime_history_diagnostics,
)
from audit_campaign import (
    PREFIGURE_REQUIRED_CHECKS as AUDIT_PREFIGURE_REQUIRED_CHECKS,
    audit_analysis,
    audit_figures,
    _aimd_snapshot_and_periodic_contract,
)
from matrix import build_scenarios, load_protocol, matrix_manifest, validate
import run_campaign as campaign
from run_campaign import (
    _active_workflow_stats,
    _scenario_config,
    _selection_provenance,
    _selection_output_leaf,
    _select_runs,
    _source_lineage,
    _persist_campaign_outcome,
    _write_immutable_json,
    _validate_t0_aimd_result,
    _validate_failure_state_snapshots,
)
from workload import (
    CompactWeka,
    _array_sha256,
    _policy_slack_for_multiplier,
    _scenario_specific_cpm,
)
import plot_results as plots
from plot_results import (
    render_all,
    style,
    verify_analysis,
    verify_prefigure_audit,
)
import matplotlib.pyplot as plt
from sglsim.srt.simulation.aegis_large_scale import (
    MODEL_PROFILES,
    LargeScaleCellConfig,
    RequestInput,
    REUSE_BIN_CONTRACT,
    run_cell,
)


ROOT = Path(__file__).resolve().parent


def test_markdown_renderer_does_not_require_tabulate(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise ImportError("tabulate intentionally unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", forbidden)
    rendered = dataframe_to_markdown(
        pd.DataFrame({"metric": ["a|b", "line\nbreak"], "value": [1.25, np.nan]})
    )
    assert "a\\|b" in rendered
    assert "line<br>break" in rendered
    assert "1.25" in rendered
    assert "NA" in rendered


def test_strict_json_normalizer_handles_numpy_without_string_fallback() -> None:
    payload = {
        "ok": np.bool_(True),
        "count": np.int64(7),
        "value": np.float64(1.25),
        "missing": np.float64(np.nan),
        "nested": np.asarray([np.int64(2), np.int64(3)]),
    }
    normalized = json_native(payload)
    assert normalized == {
        "ok": True,
        "count": 7,
        "value": 1.25,
        "missing": None,
        "nested": [2, 3],
    }
    encoded = strict_json_dumps(payload, sort_keys=True)
    assert '"ok": true' in encoded
    assert '"missing": null' in encoded
    class Unsupported:
        pass
    with pytest.raises(TypeError):
        strict_json_dumps({"unsupported": Unsupported()})


def _canonical_manifest(payload: dict) -> dict:
    frozen = dict(payload)
    frozen["canonical_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return frozen


def test_analysis_loads_cache_diagnostics_from_frozen_lineage(
    tmp_path: Path,
) -> None:
    def write(name: str, payload: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(payload)
        return path

    compact_array = write("compact.npy", b"compact-array")
    workflow_reuse = write("workflow-reuse.json", b"{}\n")
    modeled_array = write("modeled.npy", b"modeled-array")
    producer = write("prepare-modeled.py", b"# producer\n")
    protocol = write("protocol.json", b"{}\n")
    trace_generator = write("prepare-trace.py", b"# trace producer\n")

    compact = _canonical_manifest(
        {
            "schema_version": 1,
            "arrays": {
                "compact": {
                    "path": compact_array.name,
                    "sha256": hashlib.sha256(compact_array.read_bytes()).hexdigest(),
                }
            },
            "workflow_reuse": {
                "path": workflow_reuse.name,
                "sha256": hashlib.sha256(workflow_reuse.read_bytes()).hexdigest(),
            },
        }
    )
    compact_manifest = tmp_path / "compact-manifest.json"
    compact_manifest.write_text(
        json.dumps(compact, sort_keys=True) + "\n", encoding="utf-8"
    )
    modeled = _canonical_manifest(
        {
            "schema_version": 2,
            "producer": {"sha256": hashlib.sha256(producer.read_bytes()).hexdigest()},
            "protocol": {"sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()},
            "compact_workload": {
                "sha256": hashlib.sha256(compact_manifest.read_bytes()).hexdigest(),
                "canonical_sha256": compact["canonical_sha256"],
            },
            "arrays": {
                "modeled": {
                    "path": modeled_array.name,
                    "sha256": hashlib.sha256(modeled_array.read_bytes()).hexdigest(),
                }
            },
            "mechanism_gate": {"passed": True},
            "policy_input": {"array": "predicted_slack_ms"},
        }
    )
    modeled_manifest = tmp_path / "modeled-manifest.json"
    modeled_manifest.write_text(
        json.dumps(modeled, sort_keys=True) + "\n", encoding="utf-8"
    )

    def record(path: Path) -> dict[str, str]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    index = {
        "lineage": {
            "lineage_sha256": "frozen-lineage",
            "files": {
                "compact_manifest": record(compact_manifest),
                "modeled_algorithm1_manifest": record(modeled_manifest),
                "compact_cache:compact": record(compact_array),
                "compact_cache:workflow_reuse": record(workflow_reuse),
                "modeled_algorithm1_cache:modeled": record(modeled_array),
                "modeled_algorithm1_generator": record(producer),
                "protocol": record(protocol),
                "compact_trace_generator": record(trace_generator),
            },
        }
    }
    loaded = load_lineage_cache_manifests(index)
    assert loaded[0] == compact_manifest
    assert loaded[2] == modeled_manifest

    modeled_manifest.write_text(
        modeled_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lineage file changed"):
        load_lineage_cache_manifests(index)


def test_matrix_is_frozen_and_complete() -> None:
    protocol = load_protocol()
    payload = matrix_manifest(protocol)
    validate(payload, protocol)
    frozen = json.loads((ROOT / "matrix.json").read_text(encoding="utf-8"))
    assert payload["canonical_sha256"] == frozen["canonical_sha256"]
    assert payload["scenario_count"] == 33
    assert payload["policy_scenario_count"] == 311
    assert payload["run_count"] == 3110


def test_default_pilot_is_exact_nine_run_mechanism_gate() -> None:
    frozen = json.loads((ROOT / "matrix.json").read_text(encoding="utf-8"))
    runs = _select_runs(
        frozen,
        pilot=True,
        max_runs=None,
        scenario_ids=None,
        policies=None,
    )
    assert len(runs) == 9
    assert len({run["scenario_id"] for run in runs}) == 3
    assert {run["policy"] for run in runs} == {
        "aegiskv_alg111",
        "aegiskv_alg011_no_alg1",
        "recompute_only",
    }
    assert len({run["seed"] for run in runs}) == 1
    selection = _selection_provenance(
        frozen,
        runs,
        pilot=True,
        scenario_ids=None,
        policies=None,
        max_runs=None,
    )
    assert selection["mode"] == "DEFAULT_NINE_RUN_MECHANISM_PILOT"
    assert selection["pilot_contract"]["checks"]["rate_zero_reference_exact"]
    assert selection["pilot_contract"]["status"] == "PASS"
    assert selection["allowed_analysis_status"] == "PILOT_DIAGNOSTIC"
    assert _selection_output_leaf(selection) == "pilot"


def test_custom_pilot_is_never_labeled_default_pilot() -> None:
    frozen = json.loads((ROOT / "matrix.json").read_text(encoding="utf-8"))
    scenario = frozen["scenarios"][0]["scenario_id"]
    filters = {scenario}
    runs = _select_runs(
        frozen,
        pilot=True,
        max_runs=1,
        scenario_ids=filters,
        policies={"aegiskv_alg111"},
    )
    selection = _selection_provenance(
        frozen,
        runs,
        pilot=True,
        scenario_ids=filters,
        policies={"aegiskv_alg111"},
        max_runs=1,
    )
    assert selection["mode"] == "CUSTOM_PARTIAL_DIAGNOSTIC"
    assert selection["pilot_contract"]["status"] == (
        "CUSTOM_PARTIAL_DIAGNOSTIC"
    )
    assert selection["allowed_analysis_status"] == (
        "CUSTOM_PARTIAL_DIAGNOSTIC"
    )
    first_leaf = _selection_output_leaf(selection)
    assert first_leaf.startswith("custom-")
    other = _selection_provenance(
        frozen,
        _select_runs(
            frozen,
            pilot=True,
            max_runs=2,
            scenario_ids=filters,
            policies={"aegiskv_alg111"},
        ),
        pilot=True,
        scenario_ids=filters,
        policies={"aegiskv_alg111"},
        max_runs=2,
    )
    assert _selection_output_leaf(other) != first_leaf


def test_campaign_index_writer_refuses_different_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "campaign-index.json"
    _write_immutable_json(path, {"selection": "first"})
    original = path.read_bytes()
    _write_immutable_json(path, {"selection": "first"})
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError, match="immutable JSON"):
        _write_immutable_json(path, {"selection": "different"})
    assert path.read_bytes() == original


def test_failed_attempt_is_resumable_and_cannot_claim_complete(
    tmp_path: Path,
) -> None:
    passed_raw = tmp_path / "raw" / "completed.json.gz"
    passed_raw.parent.mkdir()
    passed_raw.write_bytes(b"completed-raw")
    raw_sha = hashlib.sha256(passed_raw.read_bytes()).hexdigest()
    failed_index = {
        "selection": {"canonical_sha256": "selection"},
        "pass_count": 1,
        "failure_count": 1,
        "records": [
            {"run_id": "done", "status": "PASS", "sha256": raw_sha},
            {"run_id": "retry", "status": "FAIL", "error": "injected"},
        ],
    }
    attempt = _persist_campaign_outcome(
        tmp_path, failed_index, [failed_index["records"][1]], attempt_token="unit"
    )
    assert attempt.name == "attempt-unit-failed.json"
    assert not (tmp_path / "campaign-index.json").exists()
    failed_payload = json.loads(attempt.read_text(encoding="utf-8"))
    assert failed_payload["status"] == "FAILED_ATTEMPT"
    assert failed_payload["canonical_campaign_index_written"] is False
    assert failed_payload["records"][0]["sha256"] == raw_sha

    successful_index = {
        **failed_index,
        "pass_count": 2,
        "failure_count": 0,
        "records": [
            failed_index["records"][0],
            {"run_id": "retry", "status": "PASS", "sha256": "retry-sha"},
        ],
    }
    canonical = _persist_campaign_outcome(tmp_path, successful_index, [])
    assert canonical.name == "campaign-index.json"
    complete_payload = json.loads(canonical.read_text(encoding="utf-8"))
    assert complete_payload["status"] == "COMPLETE"
    assert complete_payload["canonical_campaign_index_written"] is True
    assert attempt.read_text(encoding="utf-8") == json.dumps(
        failed_payload, indent=2, sort_keys=True
    ) + "\n"


def test_analysis_audit_requires_exact_pilot_or_full_status(tmp_path: Path) -> None:
    manifest = {
        "status": "PILOT_DIAGNOSTIC",
        "evidence_class": "simulated_sglsim_only_analysis",
        "physical_result": False,
        "campaign_index": {"sha256": "campaign-sha"},
        "files": {},
        "analysis_sources": {},
        "raw_artifacts": [],
        "observed_run_count": 0,
        "contract_status": "PASS",
        "rationality_status": "PASS",
        "algorithm1_identifiability_status": "PASS",
        "paired_comparison_coverage_status": "PASS",
    }
    path = tmp_path / "analysis_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _detail, pilot_checks = audit_analysis(
        tmp_path,
        "campaign-sha",
        selection_mode="DEFAULT_NINE_RUN_MECHANISM_PILOT",
    )
    _detail, wrong_full_checks = audit_analysis(
        tmp_path, "campaign-sha", selection_mode="FULL_FROZEN_MATRIX"
    )
    assert pilot_checks["analysis_status_exact_for_campaign_mode"] is True
    assert wrong_full_checks["analysis_status_exact_for_campaign_mode"] is False

    manifest["status"] = "COMPLETE"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _detail, full_checks = audit_analysis(
        tmp_path, "campaign-sha", selection_mode="FULL_FROZEN_MATRIX"
    )
    _detail, wrong_pilot_checks = audit_analysis(
        tmp_path,
        "campaign-sha",
        selection_mode="DEFAULT_NINE_RUN_MECHANISM_PILOT",
    )
    assert full_checks["analysis_status_exact_for_campaign_mode"] is True
    assert wrong_pilot_checks["analysis_status_exact_for_campaign_mode"] is False


def test_partial_formal_selection_can_never_resolve_complete() -> None:
    status, eligible = resolve_analysis_evidence_status(
        selection_mode="FULL_FROZEN_MATRIX",
        allowed_status="COMPLETE",
        complete=False,
        observed_run_count=3109,
        allow_partial=True,
        formal_gates_pass=True,
    )
    assert status == "FAILED_GATE"
    assert eligible is False

    status, eligible = resolve_analysis_evidence_status(
        selection_mode="FULL_FROZEN_MATRIX",
        allowed_status="COMPLETE",
        complete=True,
        observed_run_count=3110,
        allow_partial=True,
        formal_gates_pass=True,
    )
    assert status == "FAILED_GATE"
    assert eligible is False

    status, eligible = resolve_analysis_evidence_status(
        selection_mode="FULL_FROZEN_MATRIX",
        allowed_status="COMPLETE",
        complete=True,
        observed_run_count=3110,
        allow_partial=False,
        formal_gates_pass=True,
    )
    assert status == "COMPLETE"
    assert eligible is True


def test_pilot_paired_comparison_coverage_fails_on_missing_row() -> None:
    rows = [
        {
            "scenario_id": scenario,
            "policy": "aegiskv_alg111",
            "reference": reference,
            "metric": "actual_failure_added_workflow_harm_surrogate_ms",
            "paired_seed_count": 1,
            "paired_workflow_count_per_seed": 393,
            "mean_delta_ms": 1.0,
            "delta_ci95_low_ms": 0.5,
            "delta_ci95_high_ms": 1.5,
        }
        for scenario in ("zero", "central", "high")
        for reference in (
            "aegiskv_alg011_no_alg1",
            "recompute_only",
        )
    ]
    artifacts = [
        (Path(scenario), {"scenario": {"scenario_id": scenario}})
        for scenario in ("zero", "central", "high")
    ]
    passed = paired_comparison_coverage(
        pd.DataFrame(rows),
        pd.DataFrame(),
        artifacts,
        {},
        pilot=True,
    )
    missing = paired_comparison_coverage(
        pd.DataFrame(rows[:-1]),
        pd.DataFrame(),
        artifacts,
        {},
        pilot=True,
    )
    assert passed["status"] == "PASS"
    assert passed["direct_observed_row_count"] == 6
    assert missing["status"] == "FAIL"
    assert missing["direct_observed_row_count"] == 5


def test_protocol_and_core_compute_coefficients_are_identical() -> None:
    protocol = load_protocol()
    for protocol_name, core_name in (
        ("qwen3_8b", "qwen3-8b"),
        ("qwen3_32b", "qwen3-32b"),
    ):
        raw = protocol["models"][protocol_name]["simulated_compute_profile"]
        core = MODEL_PROFILES[core_name]
        assert raw["prefill_setup_ms"] == core.prefill_setup_ms
        assert raw["prefill_ms_per_token"] == core.prefill_ms_per_token
        assert raw["decode_ms_per_token"] == core.decode_ms_per_token


def test_algorithm1_cache_uses_non_degenerate_causal_crossfit_point_slack() -> None:
    cache = ROOT / "cache/modeled_algorithm1_v1"
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    policy_input = manifest["policy_input"]
    crossfit = manifest["crossfit_contract"]
    causal = manifest["causal_feature_contract"]
    gate = manifest["mechanism_gate"]
    predicted = np.load(cache / manifest["arrays"]["predicted_slack_ms"]["path"])
    eligible = np.load(
        cache / manifest["arrays"]["complete_visible_frontier"]["path"]
    ).astype(bool)
    lcb = np.load(cache / manifest["arrays"]["slack_lcb_ms"]["path"])

    assert manifest["schema_version"] == 2
    assert manifest["producer"]["sha256"] == hashlib.sha256(
        (ROOT / "prepare_modeled_algorithm1.py").read_bytes()
    ).hexdigest()
    assert manifest["protocol"]["sha256"] == hashlib.sha256(
        (ROOT / "protocol.json").read_bytes()
    ).hexdigest()
    assert policy_input["array"] == "predicted_slack_ms"
    assert policy_input["uses_future_truth"] is False
    assert policy_input["uses_actual_slack"] is False
    assert policy_input["uses_conformal_lcb"] is False
    assert crossfit["fit_calibration_target_workflows_disjoint_per_fold"]
    assert crossfit["target_prediction_uses_target_labels"] is False
    assert causal["uses_future_output_or_completion_as_feature"] is False
    assert causal["missing_or_incomplete_frontier_publishes_zero_slack"]
    assert gate["passed"]
    assert int(np.sum(predicted > 1e-9)) == policy_input["nonzero_count"]
    assert float(np.mean(predicted > 1e-9)) >= gate[
        "minimum_nonzero_fraction_all_requests"
    ]
    assert float(np.mean(predicted[eligible] > 1e-9)) >= gate[
        "minimum_nonzero_fraction_eligible_frontiers"
    ]
    assert np.all(predicted[~eligible] == 0.0)
    assert np.all(lcb == 0.0)
    assert manifest["uncertainty_diagnostic"]["policy_input"] is False
    assert manifest["uncertainty_diagnostic"]["vacuous_for_central_policy"]
    # Prediction quality is reported even when negative; it is not a hidden
    # performance-selection gate.
    assert math.isfinite(manifest["remaining"]["r2"])
    assert math.isfinite(manifest["slack"]["r2"])
    assert "signed_bias_ms" in manifest["slack"]


def test_algorithm1_error_multiplier_is_operational() -> None:
    request = RequestInput(
        request_id="r0",
        workflow_id="w0",
        prompt_tokens=64,
        output_tokens=1,
        slack_ms=10.0,
        actual_slack_ms=30.0,
        prefix_group="w0",
        arrival_ms=0.0,
        input_reuse_tokens=0,
    )
    observed = []
    for multiplier in (0.0, 0.5, 1.0, 2.0):
        result = run_cell(
            LargeScaleCellConfig(
                model="qwen3-8b",
                dp_size=2,
                concurrency=1,
                request_count=1,
                workflow_count=1,
                failure_rate_per_1000_worker_hours=0.0,
                algorithm1_prediction_error_multiplier=multiplier,
                control_window_ms=10.0,
            ),
            seed=1,
            policy="aegiskv_alg111",
            workload=(request,),
        )
        observed.append(result["request_rows"][0]["policy_slack_ms"])
    assert observed == [30.0, 20.0, 10.0, 0.0]


def test_every_scenario_uses_real_operational_axes() -> None:
    protocol = load_protocol()
    scenarios = build_scenarios(protocol)
    assert {row.network_effective_gb_s for row in scenarios} >= {1.0, 10.0, 20.0, 40.0}
    assert {row.hbm_quota_gib_per_target for row in scenarios} >= {2.0, 4.0, 8.0}
    assert {row.dram_quota_gib_per_target for row in scenarios} >= {16.0, 64.0, 256.0}
    assert {row.algorithm1_prediction_error_multiplier for row in scenarios} >= {
        0.0,
        0.5,
        1.0,
        2.0,
    }
    assert {row.active_workflow_concurrency for row in scenarios} >= {
        32,
        64,
        128,
        224,
    }
    assert 256 not in {row.active_workflow_concurrency for row in scenarios}
    assert 240 not in {row.active_workflow_concurrency for row in scenarios}
    assert protocol["workload"][
        "highest_concurrency_minimum_steady_window_samples"
    ] == 5
    assert protocol["workload"][
        "highest_concurrency_minimum_last_launch_ms"
    ] == 180000.0
    assert protocol["workload"][
        "highest_concurrency_minimum_periodic_aimd_updates"
    ] == 6


def test_network_and_measurement_axes_reach_core_config() -> None:
    protocol = load_protocol()
    scenario = next(
        row.as_dict()
        for row in build_scenarios(protocol)
        if row.network_effective_gb_s == 1.0
    )
    config = _scenario_config(
        scenario,
        protocol,
        request_count=1,
        reuse_fraction=0.0,
        measurement_start_ms=10.0,
        measurement_end_ms=20.0,
    )
    resolved = LargeScaleCellConfig.from_value(config)
    assert resolved.network_effective_bandwidth_gbps == 1.0
    assert resolved.dram_copy_bandwidth_gbps == 1.0
    assert resolved.dram_load_bandwidth_gbps == 1.0
    assert resolved.remote_hbm_load_bandwidth_gbps == 1.0
    assert resolved.measurement_start_ms == 10.0
    assert resolved.measurement_end_ms == 20.0
    assert resolved.target_failure_probability == 0.0
    assert resolved.pressure_queue_depth == 2
    assert resolved.hbm_quota_pressure_fraction == 0.95
    assert resolved.active_source_probe_count == 64


def test_offered_concurrency_schedule_is_named_and_calibrated() -> None:
    protocol = load_protocol()
    scenario = next(
        row.as_dict()
        for row in build_scenarios(protocol)
        if row.model_id == "qwen3_8b"
        and row.active_workflow_concurrency == 128
        and row.failure_rate_per_1000_worker_hours == 5.952
        and row.total_gpu_count == 64
        and row.prompt_scale == 0.016
        and row.network_effective_gb_s == 10.0
        and row.hbm_quota_gib_per_target == 4.0
        and row.dram_quota_gib_per_target == 64.0
        and row.algorithm1_prediction_error_multiplier == 1.0
    )
    projected = CompactWeka().project(scenario, protocol, seed=2026082600)
    metadata = projected.metadata
    assert metadata["target_offered_trace_workflow_concurrency"] == 128
    assert (
        metadata[
            "achieved_offered_trace_steady_concurrency_relative_error"
        ]
        <= 0.05
    )
    assert metadata["achieved_offered_trace_steady_active_workflows_peak"] >= 128
    assert "policy-independent" in metadata["offered_schedule_completion_semantics"]
    assert "achieved_steady_concurrency_relative_error" not in metadata
    assert len(metadata["policy_independent_projected_workload_sha256"]) == 64
    assert metadata["algorithm1_policy_input_array"] == (
        "effective_policy_slack_ms"
    )
    assert metadata["algorithm1_m1_policy_input_array"] == "predicted_slack_ms"
    assert metadata["algorithm1_policy_input_uses_future_truth"] is False
    assert metadata["algorithm1_m1_policy_path_reads_actual_slack"] is False
    assert metadata["algorithm1_realized_output_is_prediction_input"] is False
    assert metadata["algorithm1_actual_slack_is_ground_truth_only"]
    assert metadata["algorithm1_central_m1_nonzero_slack_gate_passed"]
    assert metadata["predicted_policy_slack_nonzero_count"] > 0
    assert metadata["predicted_policy_slack_nonzero_fraction"] >= 0.05
    assert metadata[
        "scenario_specific_cpm_central_regeneration_alignment"
    ]["passed"]


def test_scenario_specific_cpm_flips_critical_branch_without_future_truth() -> None:
    workflow = np.asarray([0, 0, 0, 0], dtype=np.int16)
    stage = np.asarray([0, 0, 0, 0], dtype=np.int16)
    branch = np.asarray([1, 2, 1, 2], dtype=np.int16)
    round_index = np.asarray([0, 0, 1, 1], dtype=np.int16)
    central_visible = np.asarray([10.0, 9.0, 10.0, 9.0])
    central_prediction = np.asarray([20.0, 18.0, 10.0, 9.0])

    central = _scenario_specific_cpm(
        workflow_index=workflow,
        stage_id=stage,
        branch=branch,
        round_index=round_index,
        actual_scenario_service_ms=central_visible,
        causal_visible_scenario_service_proxy_ms=central_visible,
        central_causal_visible_service_proxy_ms=central_visible,
        central_crossfit_predicted_remaining_ms=central_prediction,
    )
    noncentral_visible = np.asarray([1.0, 9.0, 1.0, 9.0])
    noncentral = _scenario_specific_cpm(
        workflow_index=workflow,
        stage_id=stage,
        branch=branch,
        round_index=round_index,
        actual_scenario_service_ms=noncentral_visible,
        causal_visible_scenario_service_proxy_ms=noncentral_visible,
        central_causal_visible_service_proxy_ms=central_visible,
        central_crossfit_predicted_remaining_ms=central_prediction,
    )

    # Central branch 1 is critical; the noncentral profile makes branch 2
    # critical.  Exact CPM therefore moves positive Slack to the other branch.
    assert central["actual_slack_ms"][:2].tolist() == [0.0, 2.0]
    assert noncentral["actual_slack_ms"][:2].tolist() == [16.0, 0.0]
    assert noncentral["predicted_slack_ms"][:2].tolist() == [16.0, 0.0]

    changed_future_truth = _scenario_specific_cpm(
        workflow_index=workflow,
        stage_id=stage,
        branch=branch,
        round_index=round_index,
        actual_scenario_service_ms=np.asarray([1.0, 9.0, 1000.0, 2.0]),
        causal_visible_scenario_service_proxy_ms=noncentral_visible,
        central_causal_visible_service_proxy_ms=central_visible,
        central_crossfit_predicted_remaining_ms=central_prediction,
    )
    assert np.array_equal(
        changed_future_truth[
            "causally_scaled_crossfit_predicted_remaining_ms"
        ],
        noncentral["causally_scaled_crossfit_predicted_remaining_ms"],
    )
    assert np.array_equal(
        changed_future_truth["predicted_slack_ms"],
        noncentral["predicted_slack_ms"],
    )
    assert not np.array_equal(
        changed_future_truth["actual_remaining_ms"],
        noncentral["actual_remaining_ms"],
    )
    assert not np.array_equal(
        changed_future_truth["actual_slack_ms"],
        noncentral["actual_slack_ms"],
    )


def test_algorithm1_multiplier_arrays_and_hashes_are_distinct_and_m1_is_causal() -> None:
    actual = np.asarray([10.0, 100.0])
    predicted = np.asarray([30.0, 50.0])
    expected = {
        0.0: [10.0, 100.0],
        0.5: [20.0, 75.0],
        1.0: [30.0, 50.0],
        2.0: [50.0, 0.0],
    }
    hashes = set()
    for multiplier, values in expected.items():
        policy = _policy_slack_for_multiplier(
            actual_slack_ms=actual,
            predicted_slack_ms=predicted,
            multiplier=multiplier,
        )
        assert policy.tolist() == values
        hashes.add(_array_sha256("effective_policy_slack_ms", policy))
    assert len(hashes) == 4

    # The central path must not inspect endpoint truth.  NaNs here would
    # contaminate the result if m=1 evaluated the interpolation expression.
    m1 = _policy_slack_for_multiplier(
        actual_slack_ms=np.asarray([np.nan, np.nan]),
        predicted_slack_ms=predicted,
        multiplier=1.0,
    )
    assert np.array_equal(m1, predicted)


def test_projected_workload_hash_binds_all_multiplier_sensitivity_arrays() -> None:
    protocol = load_protocol()
    scenario = next(
        row.as_dict()
        for row in build_scenarios(protocol)
        if row.model_id == "qwen3_8b"
        and row.prompt_scale == 0.016
        and row.active_workflow_concurrency == 128
        and row.failure_rate_per_1000_worker_hours == 5.952
        and row.total_gpu_count == 64
        and row.network_effective_gb_s == 10.0
        and row.hbm_quota_gib_per_target == 4.0
        and row.dram_quota_gib_per_target == 64.0
        and row.algorithm1_prediction_error_multiplier == 1.0
    )
    compact = CompactWeka()
    workload_hashes: set[str] = set()
    effective_hashes: set[str] = set()
    predicted_hashes: set[str] = set()
    actual_hashes: set[str] = set()
    for multiplier in (0.0, 0.5, 1.0, 2.0):
        sensitivity = dict(scenario)
        sensitivity["algorithm1_prediction_error_multiplier"] = multiplier
        projected = compact.project(sensitivity, protocol, seed=2026082600)
        arrays = projected.metadata[
            "scenario_specific_algorithm1_arrays_sha256"
        ]
        workload_hashes.add(
            projected.metadata[
                "policy_independent_projected_workload_sha256"
            ]
        )
        effective_hashes.add(arrays["effective_policy_slack_ms"])
        predicted_hashes.add(arrays["predicted_slack_ms"])
        actual_hashes.add(arrays["actual_slack_ms"])
        assert all(
            row.policy_slack_multiplier_applied == multiplier
            for row in projected.rows
        )
    assert len(workload_hashes) == 4
    assert len(effective_hashes) == 4
    assert len(predicted_hashes) == 1
    assert len(actual_hashes) == 1


def test_realized_workflow_concurrency_is_recomputed_from_intervals() -> None:
    rows = [
        {"arrival_ms": 0.0, "finish_ms": 10.0},
        {"arrival_ms": 5.0, "finish_ms": 15.0},
    ]
    stats = _active_workflow_stats(rows, finish_key="finish_ms")
    assert stats["time_mean"] == 4.0 / 3.0
    assert stats["time_p50"] == 1.0
    assert stats["time_p95"] == 2.0
    assert stats["peak"] == 2


def test_raw_producer_lineage_is_self_contained_and_excludes_analysis() -> None:
    lineage = _source_lineage(ROOT / "protocol.json", ROOT / "matrix.json")
    files = lineage["files"]
    assert "analyzer" not in files
    assert "transfer_engine" in files
    assert "residual_dag" in files
    assert "compact_trace_generator" in files
    assert "modeled_algorithm1_generator" in files
    compact = json.loads((ROOT / "cache/weka_compact_v1/manifest.json").read_text())
    modeled = json.loads(
        (ROOT / "cache/modeled_algorithm1_v1/manifest.json").read_text()
    )
    assert {
        f"compact_cache:{name}" for name in (*compact["arrays"], "workflow_reuse")
    }.issubset(files)
    assert {
        f"modeled_algorithm1_cache:{name}" for name in modeled["arrays"]
    }.issubset(files)
    assert modeled["producer"]["sha256"] == files[
        "modeled_algorithm1_generator"
    ]["sha256"]
    assert files["authoritative_current_design"]["path"] == str(
        (ROOT.parents[2] / "AegisKV/design/current_design.md").resolve()
    )
    assert (
        files["authoritative_current_design"]["sha256"]
        == "0fdfd097a536ab73cd9f7eb12099eb6f5b9c068a061da30af57a8e6798caacac"
    )


def test_execute_one_retains_core_schema_metric_and_repair_contract(
    tmp_path: Path, monkeypatch,
) -> None:
    protocol = json.loads(json.dumps(load_protocol()))
    protocol["source_trace"]["workflow_count"] = 1
    scenario = next(
        row.as_dict()
        for row in build_scenarios(protocol)
        if row.model_id == "qwen3_8b"
    )
    scenario.update(
        {
            "scenario_id": "contract-micro",
            "active_workflow_concurrency": 1,
            "failure_rate_per_1000_worker_hours": 0.0,
            "system_dp_size": 2,
            "total_gpu_count": 2,
        }
    )
    request = RequestInput(
        request_id="contract-r0",
        workflow_id="contract-w0",
        prompt_tokens=8192,
        output_tokens=512,
        slack_ms=1.0,
        actual_slack_ms=1.0,
        prefix_group="contract-w0",
        arrival_ms=0.0,
        input_reuse_tokens=0,
    )

    class MicroWorkload:
        def project(self, *_args, **_kwargs):
            return type(
                "Projected",
                (),
                {
                    "rows": (request,),
                        "metadata": {
                            "input_reuse_fraction": 0.0,
                            "steady_measurement_start_ms": 0.0,
                            "steady_measurement_end_ms": 10.0,
                            "achieved_offered_trace_steady_concurrency_relative_error": 0.0,
                            "all_393_workflows_in_wct": True,
                            "stabilization_prefix_excluded_from_wct": False,
                        },
                },
            )()

    monkeypatch.setattr(campaign, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(campaign, "CompactWeka", MicroWorkload)
    output = tmp_path / "contract.json.gz"
    run = {
        "run_id": "contract-micro__aegiskv_alg111__s1",
        "scenario_id": scenario["scenario_id"],
        "policy": "aegiskv_alg111",
        "seed": 1,
    }
    lineage = _source_lineage(ROOT / "protocol.json", ROOT / "matrix.json")
    record = campaign._execute_one(
        {
            "protocol_path": str(ROOT / "protocol.json"),
            "scenario": scenario,
            "run": run,
            "lineage": lineage,
            "output_path": str(output),
        }
    )
    assert record["status"] == "PASS"
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        artifact = json.load(stream)
    assert artifact["simulator_cell_schema_version"] == (
        "aegiskv_sglsim_large_scale_cell_v2"
    )
    assert artifact["formal_metric_contract"][
        "primary_failure_endpoint_prefix"
    ] == "actual_failure_added_workflow_harm_surrogate_"
    assert artifact["raw_retention"]["full_repair_rows_retained"] is True
    assert artifact["repair_row_count"] == len(artifact["repair_rows"])
    assert artifact["active_source_probe_row_count"] == 64
    assert len(artifact["active_source_probe_rows"]) == 64
    assert artifact["active_source_fault_conditioned_recovery"][
        "probe_rows"
    ] == artifact["active_source_probe_rows"]
    assert artifact["raw_retention"][
        "full_active_source_probe_rows_retained"
    ] is True
    assert artifact["failure_state_at_horizon"]["snapshot_scope"] == (
        "base_run_horizon_after_all_events_at_or_before_horizon"
    )
    assert artifact["final_state"]["failure_history"]["snapshot_scope"] == (
        "post_all_scheduled_events_including_repairs_after_horizon"
    )
    checks = aimd_snapshot_contract_checks(artifact, protocol)
    assert all(checks.values()), checks
    assert _aimd_snapshot_and_periodic_contract(artifact, protocol)
    assert artifact["summary"]["initial_snapshot_sample_count"] == 1
    assert artifact["window_rows"][0]["initial_sample"] is True
    assert artifact["window_rows"][0]["aimd_update_applied"] is False
    assert artifact["window_rows"][0]["controller_alpha_mean"] == 0.5
    assert artifact["window_rows"][0]["controller_stable_windows_mean"] == 0.0

    corrupted = json.loads(json.dumps(artifact))
    corrupted["window_rows"][0]["aimd_update_applied"] = True
    assert not all(aimd_snapshot_contract_checks(corrupted, protocol).values())
    assert not _aimd_snapshot_and_periodic_contract(corrupted, protocol)


@pytest.mark.parametrize(
    ("policy", "expected_alpha"),
    (
        ("recompute_only", 0.0),
        ("dejavu_neighbor", 1.0),
        ("raidserve_host", 0.0),
    ),
)
def test_t0_contract_preserves_baseline_specific_fixed_alpha(
    policy: str, expected_alpha: float
) -> None:
    protocol = load_protocol()
    result = run_cell(
        LargeScaleCellConfig(
            model="qwen3-8b",
            dp_size=2,
            concurrency=1,
            request_count=1,
            workflow_count=1,
            failure_rate_per_1000_worker_hours=0.0,
            active_source_probe_count=0,
            control_window_ms=100.0,
        ),
        seed=13,
        policy=policy,
        workload=(
            RequestInput(
                request_id="baseline-t0-r0",
                workflow_id="baseline-t0-w0",
                prompt_tokens=8192,
                output_tokens=512,
                slack_ms=1.0,
                actual_slack_ms=1.0,
                prefix_group="baseline-t0-w0",
                arrival_ms=0.0,
                input_reuse_tokens=0,
            ),
        ),
    )
    assert _validate_t0_aimd_result(result, protocol) is False
    initial = result["window_rows"][0]
    assert initial["initial_sample"] is True
    assert initial["aimd_update_applied"] is False
    assert initial["controller_alpha_mean"] == expected_alpha
    artifact = dict(result)
    artifact["scenario"] = {"active_workflow_concurrency": 1}
    artifact["workload"] = {"steady_measurement_end_ms": 100.0}
    assert all(aimd_snapshot_contract_checks(artifact, protocol).values())
    assert _aimd_snapshot_and_periodic_contract(artifact, protocol)


def test_protocol_fidelity_boundaries_are_explicit() -> None:
    protocol = load_protocol()
    failure = protocol["failure"]
    assert failure["target_first_fraction"] == 0.0
    assert failure["forced_or_synthetic_target_first_injection"] is False
    assert failure[
        "natural_prior_target_unavailability_from_uptime_history_is_preserved"
    ]
    assert "not failed by the same event" in failure["primary_failure_type"]
    assert failure["standby_rotation_modeled"] is False
    assert failure["complete_current_design_target_failover_evaluated"] is False
    assert failure[
        "target_failure_history_is_no_standby_rotation_conservative_sensitivity"
    ] is True
    assert failure["hazard_unit"] == (
        "single_worker_shard_failure_counterfactual_per_1000_total_gpu_shard_hours"
    )
    assert "not executed worker replacement" in failure["failure_execution_boundary"]
    assert protocol["algorithm3"]["queue_depth_pressure_threshold"] == 2
    assert protocol["algorithm3"]["quota_occupancy_pressure_threshold"] == 0.95
    assert "telemetry-only snapshot" in protocol["algorithm3"][
        "initial_sample_semantics"
    ]
    assert "positive-time periodic windows" in protocol["algorithm3"][
        "periodic_update_semantics"
    ]
    algorithm1 = protocol["algorithm1_sensitivity"]
    assert algorithm1["central_policy_input_array"] == "predicted_slack_ms"
    assert algorithm1[
        "one_means_workflow_disjoint_causal_visible_crossfit_point_prediction"
    ]
    assert algorithm1[
        "absolute_conformal_lcb_is_uncertainty_diagnostic_not_policy_input"
    ]
    assert algorithm1["actual_slack_is_label_and_ground_truth_endpoint_only"]
    assert algorithm1["fault_time_concurrent_branch_residual_cpm_recomputed"] is False
    assert "elapsed due-time decrement" in algorithm1["actual_slack_label_semantics"]
    assert protocol["output"]["retain_workflow_rows_per_run"] == "all_393"
    assert protocol["output"]["retain_window_rows_per_run"] == "all"
    assert protocol["output"]["retain_failure_rows_per_run"] == "all"
    assert protocol["output"]["retain_repair_rows_per_run"] == "all"
    assert protocol["output"]["retain_active_source_probe_rows_per_run"] == (
        "all_64"
    )
    assert {
        name: protocol["fault_estimands"]["reuse_bin_contract"][name]
        for name in REUSE_BIN_CONTRACT
    } == REUSE_BIN_CONTRACT
    assert "queue_safety_limit_requests" not in protocol["workload"]
    assert "central_target_utilization" not in protocol["workload"]
    assert protocol["workload"]["wct_measurement_workflows"] == 393
    estimand_a = protocol["fault_estimands"][
        "estimand_a_uptime_poisson_incidence"
    ]
    assert "per-workflow max-overrun" in estimand_a["name"]
    assert estimand_a["not_event_additive_sum"] is True
    assert estimand_a["not_executed_or_replayed_workflow_wct"] is True
    ablation = protocol["structural_ablation_contract"]
    assert ablation[
        "algorithm1_effect_interpretable_only_when_algorithm2_enabled"
    ] is True
    assert ablation["retain_equivalent_cells_for_preregistered_cube_closure"]


def test_paper_figure_fonts_are_at_least_ten_points() -> None:
    style()
    for key in (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "legend.fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
    ):
        assert float(plt.rcParams[key]) >= 10.0
    source = (ROOT / "plot_results.py").read_text(encoding="utf-8")
    assert source.count(
        "Rate-local live-state K64; may include prior uptime history"
    ) == 2
    colors = re.findall(r'#[0-9A-Fa-f]{6}', source)
    assert colors
    assert all(color[1:3] == color[3:5] == color[5:7] for color in colors)


def test_formal_endpoint_and_repair_closure_contract() -> None:
    result = run_cell(
        LargeScaleCellConfig(
            model="qwen3-8b",
            dp_size=2,
            concurrency=1,
            request_count=1,
            workflow_count=1,
            failure_rate_per_1000_worker_hours=0.0,
            minimum_faults_per_cell=1,
            repair_time_ms=60_000.0,
            control_window_ms=100.0,
            target_failure_probability=0.0,
        ),
        seed=7,
        policy="aegiskv_alg111",
        workload=(
            RequestInput(
                request_id="repair-r0",
                workflow_id="repair-w0",
                prompt_tokens=8192,
                output_tokens=512,
                slack_ms=1000.0,
                actual_slack_ms=0.0,
                prefix_group="repair-w0",
                arrival_ms=0.0,
                input_reuse_tokens=0,
            ),
        ),
    )
    assert result["schema_version"] == "aegiskv_sglsim_large_scale_cell_v2"
    assert result["formal_metric_contract"]["primary_failure_endpoint_prefix"] == (
        "actual_failure_added_workflow_harm_surrogate_"
    )
    assert "actual_failure_added_workflow_harm_surrogate_mean_ms" in result["summary"]
    assert result["summary"]["legacy_metric_aliases_forbidden_for_formal_claims"]
    assert len(result["failure_rows"]) == 1
    assert len(result["repair_rows"]) == 1
    assert result["repair_rows"][0]["old_hbm_copies_restored"] is False
    assert (
        result["repair_rows"][0]["post_repair_hbm_reprotection_scheduled"]
        is False
    )
    assert result["gates"]["explicit_worker_repair_event_modeled"]
    assert result["gates"]["repair_reenables_future_hbm_admission_only"]
    assert result["gates"]["repair_does_not_restore_old_hbm_copies"]
    assert result["gates"]["formal_analysis_uses_explicit_actual_harm_surrogate_fields"]
    _validate_failure_state_snapshots(result)
    horizon = result["failure_state_at_horizon"]
    post = result["final_state"]["failure_history"]
    assert result["repair_rows"][0]["after_base_run_horizon"] is True
    assert horizon["current_physical_gpu_shard_outage_count"] == 1
    assert post["current_physical_gpu_shard_outage_count"] == 0
    assert result["summary"][
        "current_physical_gpu_shard_outage_count_at_horizon"
    ] == 1
    assert result["summary"][
        "current_physical_gpu_shard_outage_count_post_all_events"
    ] == 0
    diagnostics = uptime_history_diagnostics(result)
    assert diagnostics[
        "history_accounting_matches_raw_summary_and_final_state"
    ] is True
    assert diagnostics["current_physical_gpu_shard_outage_count_at_horizon"] == 1
    assert diagnostics[
        "current_physical_gpu_shard_outage_count_post_all_events"
    ] == 0
    corrupted = json.loads(json.dumps(result))
    corrupted["summary"][
        "current_physical_gpu_shard_outage_count_at_horizon"
    ] = 0
    with pytest.raises(ValueError, match="horizon/post-all-events"):
        _validate_failure_state_snapshots(corrupted)
    assert uptime_history_diagnostics(corrupted)[
        "history_accounting_matches_raw_summary_and_final_state"
    ] is False


def test_restore_fraction_is_token_pooled_and_zero_denominator_is_na() -> None:
    exercised = pd.DataFrame(
        {
            "summary.restored_tokens": [3, 0],
            "summary.recomputed_tokens": [0, 1],
        }
    )
    hit, recompute, _low, _high = pooled_recovery_fraction(exercised)
    assert hit == 0.75
    assert recompute == 0.25
    empty = pd.DataFrame(
        {
            "summary.restored_tokens": [0, 0],
            "summary.recomputed_tokens": [0, 0],
        }
    )
    hit, recompute, _low, _high = pooled_recovery_fraction(empty)
    assert math.isnan(hit)
    assert math.isnan(recompute)


def test_full_vs_baseline_workflow_pairing_uses_all_393_rows() -> None:
    def artifact(policy: str, value: float) -> dict:
        return {
            "scenario": {"scenario_id": "central"},
            "policy": policy,
            "seed": 1,
            "workflow_rows": [
                {
                    "workflow_id": f"w{index:03d}",
                    "actual_failure_added_workflow_harm_surrogate_ms": value,
                }
                for index in range(393)
            ],
        }

    table = paired_workflow_seed_bootstrap(
        [
            (Path("full"), artifact("aegiskv_alg111", 1.0)),
            (Path("baseline"), artifact("ghostserve_parity", 3.0)),
        ],
        resamples=20,
        seed=1,
    )
    assert len(table) == 1
    assert table.iloc[0]["policy"] == "aegiskv_alg111"
    assert table.iloc[0]["reference"] == "ghostserve_parity"
    assert table.iloc[0]["paired_workflow_count_per_seed"] == 393
    assert table.iloc[0]["mean_delta_ms"] == -2.0


def test_failure_tuple_schedule_is_policy_independent() -> None:
    config = LargeScaleCellConfig(
        model="qwen3-8b",
        dp_size=2,
        concurrency=1,
        request_count=1,
        workflow_count=1,
        failure_rate_per_1000_worker_hours=0.0,
        minimum_faults_per_cell=2,
        control_window_ms=100.0,
    )
    workload = (
        RequestInput(
            request_id="pair-r0",
            workflow_id="pair-w0",
            prompt_tokens=8192,
            output_tokens=512,
            slack_ms=1.0,
            actual_slack_ms=1.0,
            prefix_group="pair-w0",
            arrival_ms=0.0,
            input_reuse_tokens=0,
        ),
    )
    schedules = []
    for policy in ("aegiskv_alg111", "ghostserve_parity"):
        result = run_cell(config, seed=9, policy=policy, workload=workload)
        schedules.append(
            [
                (
                    row["time_ms"],
                    row["source_rank"],
                    row["shard_rank"],
                    row["forced_floor_event"],
                )
                for row in result["failure_rows"]
            ]
        )
    assert schedules[0] == schedules[1]


def test_rate_zero_hazard_burden_and_probe_coverage_are_fail_closed() -> None:
    policies = (
        "aegiskv_alg111",
        "aegiskv_alg011_no_alg1",
        "recompute_only",
    )
    scenarios = (("zero", 0.0), ("central", 5.952), ("high", 41.667))
    rows = []
    for scenario_id, rate in scenarios:
        for policy_index, policy in enumerate(policies):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "campaign_axis": "concurrency_failure",
                    "policy": policy,
                    "seed": 1,
                    "model_id": "qwen3_8b",
                    "prompt_scale": 0.016,
                    "active_workflow_concurrency": 128,
                    "total_gpu_count": 64,
                    "tp_size": 1,
                    "pp_size": 1,
                    "system_dp_size": 64,
                    "network_effective_gb_s": 10.0,
                    "hbm_quota_gib_per_target": 4.0,
                    "dram_quota_gib_per_target": 64.0,
                    "algorithm1_prediction_error_multiplier": 1.0,
                    "failure_rate_per_1000_worker_hours": rate,
                    "failure_rate_per_1000_gpu_shard_hours": rate,
                    "probe_count": 64,
                    "probe_tuple_sha256": "same-tuple",
                    "eligible_exposure_gpu_shard_hours": 2.0,
                    "conditional_recovery_mean_ms": 4.0 + policy_index,
                    "conditional_actual_harm_mean_ms": 1.0 + policy_index,
                    "normal_state_reference": rate == 0.0,
                }
            )
    burden = normal_state_hazard_burden(pd.DataFrame(rows))
    central_full = burden[
        (burden["scenario_id"] == "central")
        & (burden["policy"] == "aegiskv_alg111")
    ].iloc[0]
    assert central_full["normal_state_burden_total_ms_per_trace_epoch"] == (
        5.952 / 1000 * 2.0 * 1.0
    )
    paired = paired_probe_outer_seed_bootstrap(burden, resamples=20, seed=1)
    workflow_pairs = pd.DataFrame(
        [
            {"scenario_id": scenario, "reference": reference}
            for scenario, _rate in scenarios
            for reference in policies[1:]
        ]
    )
    coverage = paired_probe_comparison_coverage(
        paired, workflow_pairs, burden, pilot=True
    )
    assert len(paired) == 24
    assert coverage["status"] == "PASS"
    zero_c = paired[
        (paired["scenario_id"] == "zero")
        & paired["metric"].str.startswith("normal_state_burden_")
    ]
    assert (zero_c[["mean_delta", "delta_ci95_low", "delta_ci95_high"]] == 0).all().all()


def test_renderer_wiring_uses_all_bc_inputs(tmp_path: Path, monkeypatch) -> None:
    called: list[str] = []

    def stub(name: str):
        def renderer(*_args, **_kwargs):
            called.append(name)
            return [tmp_path / f"{name}.pdf"]

        return renderer

    for name in (
        "plot_baselines",
        "plot_concurrency_failure",
        "plot_resources",
        "plot_ablation",
        "plot_timeline",
        "plot_reuse",
        "plot_model_alg1",
    ):
        monkeypatch.setattr(plots, name, stub(name))
    empty = pd.DataFrame()
    paths = render_all(
        frame=empty,
        probes=empty,
        burdens=empty,
        paired_probes=empty,
        reuse=empty,
        factorial=empty,
        timeline=empty,
        output_dir=tmp_path,
    )
    assert len(paths) == 7
    assert called == [
        "plot_baselines",
        "plot_concurrency_failure",
        "plot_resources",
        "plot_ablation",
        "plot_timeline",
        "plot_reuse",
        "plot_model_alg1",
    ]


def test_renderer_end_to_end_synthetic_complete_tables(tmp_path: Path) -> None:
    """Exercise every real filter/pivot/NA path and emit all paper formats."""

    common = {
        "model_id": "qwen3_8b",
        "prompt_scale": 0.016,
        "active_workflow_concurrency": 128,
        "failure_rate_per_1000_worker_hours": 5.952,
        "total_gpu_count": 64,
        "tp_size": 1,
        "pp_size": 1,
        "system_dp_size": 64,
        "network_effective_gb_s": 10.0,
        "hbm_quota_gib_per_target": 4.0,
        "dram_quota_gib_per_target": 64.0,
        "algorithm1_prediction_error_multiplier": 1.0,
    }

    def frame_row(
        scenario_id: str,
        policy: str,
        axis: str,
        **overrides,
    ) -> dict:
        return {
            "scenario_id": scenario_id,
            "policy": policy,
            "campaign_axis": axis,
            **common,
            "steady.effective_alpha_mean": 0.4,
            "steady.pressure_fraction_mean": 0.3,
            "summary.restored_tokens": 80,
            "summary.recomputed_tokens": 20,
            "summary.input_prompt_tokens": 1000,
            "summary.input_reuse_tokens": 400,
            "summary.logical_request_kv_bytes": 1000,
            "summary.protection_copy_attempted_bytes": 700,
            **overrides,
        }

    frame_rows = [
        frame_row("central-f5", policy, "concurrency_failure")
        for policy in plots.PALETTE
    ]
    for concurrency in (32, 128, 224):
        for rate in (0.0, 5.952, 41.667):
            if concurrency == 128 and math.isclose(rate, 5.952):
                continue
            frame_rows.append(
                frame_row(
                    f"cf-{concurrency}-{rate:g}",
                    "aegiskv_alg111",
                    "concurrency_failure",
                    active_workflow_concurrency=concurrency,
                    failure_rate_per_1000_worker_hours=rate,
                    **{
                        "steady.effective_alpha_mean": 0.5 - concurrency / 1000,
                        "steady.pressure_fraction_mean": concurrency / 256,
                    },
                )
            )
    frame = pd.DataFrame(frame_rows)

    def probe_row(
        scenario_id: str,
        policy: str,
        axis: str,
        **overrides,
    ) -> dict:
        return {
            "scenario_id": scenario_id,
            "policy": policy,
            "campaign_axis": axis,
            "seed": 1,
            **common,
            "conditional_recovery_mean_ms": 12.0,
            "conditional_actual_harm_mean_ms": 5.0,
            "eligible_exposure_gpu_shard_hours": 2.0,
            "restored_tokens": 80,
            "recomputed_tokens": 20,
            **overrides,
        }

    probes = [
        probe_row(
            "cf-128-0",
            policy,
            "concurrency_failure",
            failure_rate_per_1000_worker_hours=0.0,
            conditional_actual_harm_mean_ms=5.0 + index,
        )
        for index, policy in enumerate(plots.PALETTE)
    ]
    probes.append(probe_row("central-f5", "aegiskv_alg111", "concurrency_failure"))
    for axis, key, values in (
        ("cluster_scale", "total_gpu_count", (16, 128)),
        ("network_bandwidth", "network_effective_gb_s", (1.0, 40.0)),
        ("hbm_capacity", "hbm_quota_gib_per_target", (2.0, 8.0)),
        ("dram_capacity", "dram_quota_gib_per_target", (16.0, 256.0)),
    ):
        for index, value in enumerate(values):
            probes.append(
                probe_row(
                    f"{axis}-{value}",
                    "aegiskv_alg111",
                    axis,
                    **{
                        key: value,
                        "conditional_recovery_mean_ms": 20.0 - index,
                        "restored_tokens": 60 + 10 * index,
                    },
                )
            )
    for bits in ("000", "001", "010", "011", "100", "101", "110", "111"):
        policy = "aegiskv_alg111" if bits == "111" else f"aegiskv_alg{bits}"
        if bits == "111":
            continue  # already present at the exact rate-zero scenario
        probes.append(
            probe_row(
                "cf-128-0",
                policy,
                "concurrency_failure",
                failure_rate_per_1000_worker_hours=0.0,
                conditional_actual_harm_mean_ms=4.0 + int(bits, 2),
            )
        )
    for prompt in (0.004, 0.008, 0.032):
        probes.append(
            probe_row(
                f"context-{prompt}",
                "aegiskv_alg111",
                "context_scale",
                prompt_scale=prompt,
            )
        )
    probes.append(
        probe_row(
            "model-32b",
            "aegiskv_alg111",
            "model_scale",
            model_id="qwen3_32b",
            pp_size=4,
            system_dp_size=16,
        )
    )
    for multiplier in (0.0, 0.5, 2.0):
        probes.append(
            probe_row(
                f"alg1-{multiplier}",
                "aegiskv_alg111",
                "algorithm1_error",
                algorithm1_prediction_error_multiplier=multiplier,
            )
        )
    probes_frame = pd.DataFrame(probes)

    burdens = frame[
        (frame["policy"] == "aegiskv_alg111")
        & frame["campaign_axis"].str.contains("concurrency_failure")
    ].copy()
    burdens["normal_state_burden_ms_per_workflow_epoch"] = (
        0.1 + burdens["active_workflow_concurrency"] / 1000
    )
    paired = pd.DataFrame(
        [
            {
                "scenario_id": "central-f5",
                "reference": policy,
                "metric": "normal_state_burden_ms_per_workflow_epoch",
                "mean_delta": -0.2 - index / 100,
                "delta_ci95_low": -0.3 - index / 100,
                "delta_ci95_high": -0.1 - index / 100,
            }
            for index, policy in enumerate(plots.PALETTE)
            if policy != "aegiskv_alg111"
        ]
    )
    reuse = pd.DataFrame(
        [
            {
                "scenario_id": "cf-128-0",
                "policy": "aegiskv_alg111",
                "input_reuse_bin": name,
                "request_count_full_workload": count,
                "prompt_tokens_full_workload": count * 100,
                "intended_residency_hbm_only_chunks": 0 if name == "near_complete" else count,
                "intended_residency_dram_only_chunks": 0 if name == "near_complete" else count,
                "intended_residency_hbm_and_dram_chunks": 0,
                "intended_residency_bypass_chunks": 0,
                "probe_restored_tokens": 0 if name == "near_complete" else 8,
                "probe_recomputed_tokens": 0 if name == "near_complete" else 2,
            }
            for name, count in zip(
                ("none", "partial", "high", "near_complete"),
                (10, 20, 30, 40),
                strict=True,
            )
        ]
    )
    terms = (
        "Algorithm 1",
        "Algorithm 2",
        "Algorithm 3",
        "Algorithm 1 x 2",
        "Algorithm 1 x 3",
        "Algorithm 2 x 3",
        "Algorithm 1 x 2 x 3",
    )
    factorial = pd.DataFrame(
        [
            {
                "term": term,
                "metric": (
                    "single_fault_normal_state_burden_ms_per_workflow_epoch_at_central_rate"
                ),
                "effect_mean": 0.01 * (index - 3),
                "effect_ci95_low": 0.01 * (index - 4),
                "effect_ci95_high": 0.01 * (index - 2),
            }
            for index, term in enumerate(terms)
        ]
    )
    timeline = pd.DataFrame(
        [
            {
                "policy": "aegiskv_alg111",
                "seed": 1,
                "time_ms": time_ms,
                "effective_alpha_mean": alpha,
                "pressure_fraction": pressure,
                "hbm_used_bytes": 20,
                "hbm_capacity_bytes": 100,
                "dram_used_bytes": 40,
                "dram_capacity_bytes": 100,
            }
            for time_ms, alpha, pressure in (
                (0.0, 0.5, 0.0),
                (30_000.0, 0.25, 1.0),
                (60_000.0, 0.3, 0.0),
            )
        ]
    )

    style()
    paths = render_all(
        frame=frame,
        probes=probes_frame,
        burdens=burdens,
        paired_probes=paired,
        reuse=reuse,
        factorial=factorial,
        timeline=timeline,
        output_dir=tmp_path,
    )
    assert len(paths) == 14
    assert len({path.stem for path in paths}) == 7
    assert {path.suffix for path in paths} == {".pdf", ".png"}
    assert all(path.is_file() and path.stat().st_size > 100 for path in paths)
    assert all(len(hashlib.sha256(path.read_bytes()).hexdigest()) == 64 for path in paths)


def test_figure_consumer_rejects_pilot_and_failed_gate(tmp_path: Path) -> None:
    manifest = {
        "status": "FAILED_GATE",
        "selection_mode": "DEFAULT_NINE_RUN_MECHANISM_PILOT",
        "paper_eligible": False,
        "contract_status": "PASS",
        "files": {},
        "analysis_sources": {},
    }
    path = tmp_path / "analysis_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="COMPLETE"):
        verify_analysis(tmp_path, allow_partial=True)
    manifest["status"] = "PILOT_DIAGNOSTIC"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="COMPLETE"):
        verify_analysis(tmp_path, allow_partial=True)
    manifest.update(
        {
            "status": "COMPLETE",
            "selection_mode": "FULL_FROZEN_MATRIX",
            "paper_eligible": True,
        }
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_analysis(tmp_path, allow_partial=False)["paper_eligible"] is True


def test_forged_complete_analysis_cannot_bypass_failed_prefigure_audit(
    tmp_path: Path,
) -> None:
    analysis_manifest = {
        "status": "COMPLETE",
        "selection_mode": "FULL_FROZEN_MATRIX",
        "paper_eligible": True,
        "campaign_index": {"sha256": "campaign-sha"},
        "files": {},
        "analysis_sources": {},
        "canonical_sha256": "analysis-canonical",
    }
    analysis_path = tmp_path / "analysis_manifest.json"
    analysis_path.write_text(json.dumps(analysis_manifest), encoding="utf-8")
    assert verify_analysis(tmp_path, allow_partial=False)["status"] == "COMPLETE"

    auditor = ROOT / "audit_campaign.py"
    audit = _canonical_manifest(
        {
            "schema_version": 1,
            "evidence_class": "simulated_sglsim_only_campaign_audit",
            "status": "FAIL",
            "physical_result": False,
            "require_full": True,
            "auditor": {
                "path": str(auditor),
                "sha256": hashlib.sha256(auditor.read_bytes()).hexdigest(),
            },
            "checks": {
                name: False if name == "raw_contract" else True
                for name in plots.PREFIGURE_REQUIRED_CHECKS
            },
            "failed_checks": ["raw_contract"],
            "campaign": {"campaign_index_sha256": "campaign-sha"},
            "analysis": {
                "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest()
            },
            "figures": None,
        }
    )
    audit_path = tmp_path / "prefigure_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="prefigure full audit"):
        verify_prefigure_audit(
            audit_path,
            analysis_dir=tmp_path,
            analysis_manifest=analysis_manifest,
        )

    audit["status"] = "PASS"
    audit["checks"] = {
        name: True for name in plots.PREFIGURE_REQUIRED_CHECKS
    }
    audit["failed_checks"] = []
    audit.pop("canonical_sha256")
    audit = _canonical_manifest(audit)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    verified = verify_prefigure_audit(
        audit_path,
        analysis_dir=tmp_path,
        analysis_manifest=analysis_manifest,
    )
    assert verified["canonical_sha256"] == audit["canonical_sha256"]
    assert plots.PREFIGURE_REQUIRED_CHECKS == AUDIT_PREFIGURE_REQUIRED_CHECKS

    pdf = tmp_path / "figure.pdf"
    png = tmp_path / "figure.png"
    pdf.write_bytes(b"pdf-evidence")
    png.write_bytes(b"png-evidence")
    figure_manifest = _canonical_manifest(
        {
            "schema_version": 1,
            "evidence_class": "simulated_sglsim_only_figures",
            "physical_result": False,
            "analysis_status": "COMPLETE",
            "selection_mode": "FULL_FROZEN_MATRIX",
            "paper_eligible": True,
            "analysis_manifest": {
                "sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest()
            },
            "prefigure_full_audit": verified,
            "files": {
                path.name: {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
                for path in (pdf, png)
            },
        }
    )
    (tmp_path / "figure_manifest.json").write_text(
        json.dumps(figure_manifest), encoding="utf-8"
    )
    _detail, figure_checks = audit_figures(
        tmp_path, hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    )
    assert all(figure_checks.values())
