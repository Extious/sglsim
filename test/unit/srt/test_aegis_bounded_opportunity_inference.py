from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSESSOR_PATH = (
    _REPO_ROOT
    / "AegisKV"
    / "experiments"
    / "g0_g1"
    / "assess_bounded_opportunity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "aegis_bounded_opportunity_inference", _ASSESSOR_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_ASSESSOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ASSESSOR
_SPEC.loader.exec_module(_ASSESSOR)


def _preregistration(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "preregistration_id": "g1-formal-v1",
        "frozen_before_test_open": True,
        "input_schema_version": 3,
        "dataset_id": "dataset-v1",
        "split_id": "formal-test-v1",
        "sampling_policy_id": "every-graph-event-v1",
        "confidence_level": 0.95,
        "bootstrap_replicates": 1_000,
        "bootstrap_seed": 17,
        "minimum_all_workflows": 10,
        "minimum_eligible_workflows": 10,
        "eligible_workflow_fraction_lower_threshold": 0.30,
        "eligible_kv_byte_fraction_lower_threshold": 0.20,
    }
    value.update(updates)
    return value


def _row(
    workflow_index: int,
    *,
    decision_index: int = 0,
    eligible: bool = True,
    positive: bool = True,
    active_bytes: int = 100,
    positive_bytes: int | None = None,
    evidence_class: str = "held_out_formal",
) -> dict[str, object]:
    if positive_bytes is None:
        positive_bytes = active_bytes if positive else 0
    return {
        "evidence_class": evidence_class,
        "dataset_id": "dataset-v1",
        "split_id": "formal-test-v1",
        "sampling_policy_id": "every-graph-event-v1",
        "workflow_id": f"workflow-{workflow_index}",
        "decision_id": f"decision-{decision_index}",
        "failure_domain": "node-0",
        "eligible_declared_parallel_region": eligible,
        "proof_scope_status": "AVAILABLE" if eligible else "UNAVAILABLE",
        "workflow_template": f"template-{workflow_index % 2}",
        "model": f"model-{workflow_index % 2}",
        "layout_digest": f"layout-{workflow_index % 2}",
        "status": "SCORED" if eligible else "NO_PROOF_SCOPE",
        "failure_exposed_active_kv_bytes_total": active_bytes,
        "positive_bounded_kv_bytes": positive_bytes,
        "has_positive_bounded_deadline": positive,
    }


def _scored(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 3,
        "gate_assessment": "NOT_ASSESSED",
        "input_records": len(rows),
        "supported_records": len(rows),
        "unsupported_records": 0,
        "held_out_formal_records": sum(
            row["evidence_class"] == "held_out_formal" for row in rows
        ),
        "formal_input_consistency": "PASS",
        "formal_collection_contracts": [
            {
                "dataset_id": "dataset-v1",
                "split_id": "formal-test-v1",
                "sampling_policy_id": "every-graph-event-v1",
            }
        ],
        "rows": rows,
    }


def test_formal_opportunity_subgate_pass_never_claims_full_g1() -> None:
    scored = _scored([_row(index) for index in range(40)])

    result = _ASSESSOR.assess_scored_result(
        scored, _preregistration()
    )

    assert result["gate_assessment"] == "OPPORTUNITY_SUBGATE_PASS"
    assert (
        result["overall_g1_status"]
        == "NOT_ASSESSED_ACTIONABILITY_AND_EDF_CHANGE_MISSING"
    )
    assert result["eligible_workflow_fraction"]["lower"] > 0.30
    assert result["eligible_decision_snapshot_kv_byte_fraction"]["lower"] == 1.0
    assert result["unconditional_workflow_fraction"]["point"] == 1.0
    assert (
        result["unconditional_decision_snapshot_kv_byte_fraction"]["point"]
        == 1.0
    )
    assert "protected_vs_unprotected_action_crossing" in result["not_assessed"]


def test_many_snapshots_from_one_workflow_do_not_become_independent_samples() -> None:
    rows = [
        _row(0, decision_index=index, positive=True)
        for index in range(100)
    ]
    rows.extend(_row(index, positive=False) for index in range(1, 10))
    scored = _scored(rows)
    preregistration = _preregistration(
        eligible_workflow_fraction_lower_threshold=0.20,
        eligible_kv_byte_fraction_lower_threshold=0.20,
    )

    result = _ASSESSOR.assess_scored_result(scored, preregistration)

    assert result["eligible_workflows"] == 10
    assert result["eligible_positive_workflows"] == 1
    assert result["eligible_workflow_fraction"]["point"] == 0.1
    assert result["gate_assessment"] == "OPPORTUNITY_SUBGATE_FAIL"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("development", "development or mixed-evidence"),
        ("unsupported", "schema-rejected"),
        ("mixed_contract", "absent or mixed"),
        ("unfrozen", "frozen before"),
        ("contract_mismatch", "frozen collection contract"),
    ],
)
def test_formal_assessment_refuses_evidence_boundary_violations(
    mutation: str, message: str
) -> None:
    scored = _scored([_row(index) for index in range(10)])
    preregistration = _preregistration()
    if mutation == "development":
        scored["rows"][0]["evidence_class"] = "synthetic_development"
    elif mutation == "unsupported":
        scored["unsupported_records"] = 1
        scored["supported_records"] = 9
    elif mutation == "mixed_contract":
        scored["formal_input_consistency"] = (
            "REFUSED_MIXED_COLLECTION_CONTRACTS"
        )
    elif mutation == "unfrozen":
        preregistration["frozen_before_test_open"] = False
    elif mutation == "contract_mismatch":
        preregistration["split_id"] = "another-split"

    with pytest.raises(_ASSESSOR.AssessmentRefused, match=message):
        _ASSESSOR.assess_scored_result(scored, preregistration)


def test_zero_eligible_kv_denominator_is_refused_not_reported_as_zero() -> None:
    scored = _scored(
        [
            _row(
                index,
                positive=False,
                active_bytes=0,
                positive_bytes=0,
            )
            for index in range(10)
        ]
    )

    with pytest.raises(_ASSESSOR.AssessmentRefused, match="zero active KV"):
        _ASSESSOR.assess_scored_result(scored, _preregistration())


def test_cluster_bootstrap_is_reproducible_from_the_frozen_seed() -> None:
    rows = [
        _row(
            index,
            positive=index % 3 != 0,
            active_bytes=100 + index,
            positive_bytes=(80 + index if index % 3 != 0 else 0),
        )
        for index in range(30)
    ]
    scored = _scored(rows)
    preregistration = _preregistration()

    first = _ASSESSOR.assess_scored_result(scored, preregistration)
    second = _ASSESSOR.assess_scored_result(
        copy.deepcopy(scored), copy.deepcopy(preregistration)
    )

    assert first == second


def test_minimum_workflow_count_is_a_gate_not_a_silent_warning() -> None:
    scored = _scored([_row(index) for index in range(10)])
    preregistration = _preregistration(minimum_all_workflows=20)

    result = _ASSESSOR.assess_scored_result(scored, preregistration)

    assert result["sample_size_status"] == "FAIL_MINIMUM_ALL_WORKFLOWS"
    assert result["gate_assessment"] == "OPPORTUNITY_SUBGATE_FAIL"
