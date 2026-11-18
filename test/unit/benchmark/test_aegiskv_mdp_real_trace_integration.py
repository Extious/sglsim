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
    / "audit_mdp_real_trace_integration.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("aegiskv_mdp_trace_audit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mdp_real_trace_integration_keeps_core_wiring_strict() -> None:
    module = load_module()

    checks = module.build_checks()
    summary = module._summary(checks)
    by_id = {item.check_id: item for item in checks}

    assert summary == {
        "integration_status": "pass",
        "submission_readiness": "pass",
        "ready_for_submission": True,
        "checks": 6,
        "pass": 6,
        "partial": 0,
        "missing": 0,
        "core_checks_pass": True,
    }
    assert by_id["mdp_policy_table_selector"].status == module.PASS
    assert "selector_checks=15" in by_id["mdp_policy_table_selector"].evidence
    assert "selector_mismatches=0" in by_id["mdp_policy_table_selector"].evidence

    online = by_id["online_serving_selector_replay"]
    assert online.status == module.PASS
    assert "rows=324" in online.evidence
    assert "alpha_mismatches=0" in online.evidence
    assert "source_mismatches=0" in online.evidence

    critical_only = by_id["critical_only_gpu_replica_enforcement"]
    assert critical_only.status == module.PASS
    assert "raw_critical_gpu_tokens=1509300" in critical_only.evidence
    assert "raw_noncritical_gpu_tokens=0" in critical_only.evidence

    real_trace = by_id["real_trace_ingestion_selector_replay"]
    assert real_trace.status == module.PASS
    assert "completed_rows=205" in real_trace.evidence
    assert "workflows=96" in real_trace.evidence
    assert "selector_checks=205" in real_trace.evidence
    assert "alpha_mismatches=0" in real_trace.evidence
    assert "candidate_actions={'gpu_replica': 66, 'host_restore': 139}" in real_trace.evidence

    assert by_id["real_trace_source_coverage"].status == module.PASS
    assert "source_repos=3" in by_id["real_trace_source_coverage"].evidence
    assert "source_tables=4" in by_id["real_trace_source_coverage"].evidence
    assert by_id["policy_quality_and_ablation_gap"].status == module.PASS
    assert "ablation_matrix_status=ready" in by_id[
        "policy_quality_and_ablation_gap"
    ].evidence


def test_mdp_real_trace_integration_writes_machine_readable_outputs() -> None:
    module = load_module()

    payload = module.write_outputs(module.build_checks())

    assert payload["summary"]["integration_status"] == "pass"
    assert payload["summary"]["ready_for_submission"] is True
    assert module.OUT_JSON.exists()
    assert module.OUT_MD.exists()
    assert module.OUT_CSV.exists()
    with module.OUT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    by_id = {row["check_id"]: row for row in rows}
    assert by_id["real_trace_ingestion_selector_replay"]["status"] == "pass"
    assert by_id["policy_quality_and_ablation_gap"]["status"] == "pass"
