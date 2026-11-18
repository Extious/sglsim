from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "scripts"
    / "audit_system_paper_readiness.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("aegiskv_system_paper_readiness", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_system_paper_readiness_keeps_submission_gate_strict() -> None:
    module = load_module()

    requirements = module.build_requirements()
    summary = module._summary(requirements)
    by_id = {item.req_id: item for item in requirements}

    assert summary == {
        "overall_status": "ready",
        "ready_for_submission": True,
        "requirements": 9,
        "ready": 9,
        "partial": 0,
        "missing": 0,
    }
    assert by_id["end_to_end_policy_matrix"].status == module.READY
    assert by_id["motivation_criticality"].status == module.READY
    assert "motivation_ci_rows=28" in (
        by_id["motivation_criticality"].current_evidence
    )
    assert "long_vs_noncritical_output_ratio=8.151" in (
        by_id["motivation_criticality"].current_evidence
    )
    assert "motivation_replay_rows=227" in (
        by_id["motivation_criticality"].current_evidence
    )
    assert "critical_repeat_mean_delta_s=2.428" in (
        by_id["motivation_criticality"].current_evidence
    )
    assert "real_trace_failure_positions=205" in (
        by_id["motivation_criticality"].current_evidence
    )
    assert "rows=28" in by_id["end_to_end_policy_matrix"].current_evidence
    assert "run_ids=12" in by_id["end_to_end_policy_matrix"].current_evidence
    assert "load_levels=['100K', 'lite', 'stress']" in (
        by_id["end_to_end_policy_matrix"].current_evidence
    )
    assert by_id["mdp_online_real_trace_wiring"].status == module.READY
    assert "mdp_real_trace_integration=pass" in (
        by_id["mdp_online_real_trace_wiring"].current_evidence
    )
    assert "mdp_real_trace_submission=pass" in (
        by_id["mdp_online_real_trace_wiring"].current_evidence
    )
    assert "runtime_trace_checks=324" in (
        by_id["mdp_online_real_trace_wiring"].current_evidence
    )
    assert "trace_runtime_selector_checks=205" in (
        by_id["mdp_online_real_trace_wiring"].current_evidence
    )
    assert by_id["multi_source_real_trace"].status == module.READY
    assert "trace_sources=3" in by_id["multi_source_real_trace"].current_evidence
    assert "source_tables=4" in by_id["multi_source_real_trace"].current_evidence
    assert "workflows=96" in by_id["multi_source_real_trace"].current_evidence
    assert by_id["ablation_matrix"].status == module.READY
    assert "rows=11" in by_id["ablation_matrix"].current_evidence
    assert "runtime_replay_rows=792" in by_id["ablation_matrix"].current_evidence
    assert "static alpha" in by_id["ablation_matrix"].current_evidence
    assert "no marginal accounting" in by_id["ablation_matrix"].current_evidence
    assert by_id["alpha_tradeoff"].status == module.READY
    assert "raw_measured_rows=54" in by_id["alpha_tradeoff"].current_evidence
    assert "raw_load_regimes=['base load', 'high load', 'low load']" in (
        by_id["alpha_tradeoff"].current_evidence
    )
    assert by_id["real_system_calibration"].status == module.READY
    assert "calibration_rows=18" in by_id["real_system_calibration"].current_evidence
    assert "physical_copy_smoke=True" in (
        by_id["real_system_calibration"].current_evidence
    )
    assert "physical_tensor_copy_rows=0 boundary" in (
        by_id["real_system_calibration"].current_evidence
    )


def test_system_paper_readiness_outputs_claim_evidence_matrix() -> None:
    module = load_module()

    payload = module.write_outputs(module.build_requirements())
    matrix_path = module.EXPERIMENTS / "system_paper_claim_evidence_matrix.csv"

    assert payload["summary"]["overall_status"] == "ready"
    assert matrix_path.exists()
    with matrix_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 9
    by_id = {row["req_id"]: row for row in rows}
    assert by_id["end_to_end_policy_matrix"]["status"] == "ready"
    assert by_id["motivation_criticality"]["status"] == "ready"
    assert by_id["mdp_online_real_trace_wiring"]["status"] == "ready"
    assert by_id["ablation_matrix"]["status"] == "ready"
