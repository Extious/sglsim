from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "figures"
    / "scripts"
    / "validate_figure_evidence.py"
)


def load_validator():
    spec = importlib.util.spec_from_file_location("aegiskv_figure_validator", VALIDATOR)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_current_evidence_validator_passes_current_gate() -> None:
    module = load_validator()

    payload = module.validate()

    assert payload["status"] == "pass"
    assert payload["problems"] == []
    assert payload["goal_audit_checks"] == 19


def test_alpha_tradeoff_validator_enforces_convex_left_shift() -> None:
    module = load_validator()

    problems = module._check_alpha_tradeoff_shape()

    assert problems == []


def test_backup_policy_validator_requires_measured_matrix() -> None:
    module = load_validator()

    problems = module._check_backup_policy_matrix()

    assert problems == []


def test_motivation_criticality_ci_validator_checks_workflow_ci() -> None:
    module = load_validator()

    problems = module._check_motivation_criticality_ci()

    assert problems == []


def test_motivation_failure_replay_repeats_validator_checks_repeats() -> None:
    module = load_validator()

    problems = module._check_motivation_failure_replay_repeats()

    assert problems == []


def test_ablation_validator_checks_runtime_path_semantics() -> None:
    module = load_validator()

    problems = module._check_ablation_matrix()

    assert problems == []


def test_current_evidence_status_tracks_current_core_items() -> None:
    module = load_validator()

    status, problems = module._check_evidence_status()
    figures = status["figures"]
    by_stem = {item["stem"]: item for item in figures}

    assert problems == []
    assert set(by_stem) == module.REQUIRED_FIGURES
    assert status["summary"]["figures"] == 11
    assert status["summary"]["claim_ready"] == 11
    assert by_stem["fig_motivation_criticality_ci"]["row_count"] == 28
    assert by_stem["fig_motivation_failure_replay_repeats"]["row_count"] == 227
    assert by_stem["fig_backup_policy_comparison"]["row_count"] == 4
    assert by_stem["fig_gpu_replica_alpha_tradeoff"]["row_count"] == 68
    assert by_stem["fig_online_alpha_adaptation"]["row_count"] == 8
    assert by_stem["fig_ablation"]["row_count"] == 11
    assert by_stem["fig_real_trace_characterization"]["row_count"] == 44
    assert by_stem["fig_sglang_sglsim_accuracy"]["row_count"] == 8
    assert by_stem["fig_real_system_recovery_calibration"]["row_count"] == 18


def test_real_trace_characterization_validator_checks_trace_distribution() -> None:
    module = load_validator()

    problems = module._check_real_trace_characterization()

    assert problems == []


def test_real_system_recovery_calibration_validator_checks_physical_boundary() -> None:
    module = load_validator()

    problems = module._check_real_system_recovery_calibration()

    assert problems == []


def test_goal_audit_keeps_mdp_online_trace_checks_in_current_gate() -> None:
    module = load_validator()

    payload, problems = module._check_goal_audit()
    checks = {item["check_id"]: item for item in payload["checks"]}

    assert problems == []
    assert checks["mdp_alpha_policy_table"]["status"] == "pass"
    assert "runtime_selector_matches_policy=True" in checks[
        "mdp_alpha_policy_table"
    ]["evidence"]
    assert checks["online_alpha_adaptation"]["status"] == "pass"
    assert "raw_noncritical_gpu_tokens=0" in checks[
        "online_alpha_adaptation"
    ]["evidence"]
    assert checks["real_trace_online_path"]["status"] == "pass"
    assert "workflows=96" in checks["real_trace_online_path"]["evidence"]
    assert "trace_runtime_selector_checks=205" in checks[
        "real_trace_online_path"
    ]["evidence"]
    assert "trace_runtime_alpha_matches_raw=True" in checks[
        "real_trace_online_path"
    ]["evidence"]
