from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCORER_PATH = (
    _REPO_ROOT
    / "AegisKV"
    / "experiments"
    / "g0_g1"
    / "score_bounded_opportunity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "aegis_bounded_opportunity", _SCORER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_SCORER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCORER)


def _duration(milliseconds: float) -> dict[str, object]:
    return {
        "lower_ms": milliseconds,
        "upper_ms": milliseconds,
        "graph_version": "graph-v1",
        "service_envelope_version": "service-v1",
        "bound_id": f"bound-{milliseconds}",
        "source": "declared_work_service_envelope",
        "uses_future_information": False,
    }


def _record(
    *,
    evidence_class: str = "synthetic_development",
    workflow_disjoint_test: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "evidence_class": evidence_class,
        "dataset_id": "dataset-v1",
        "split_id": "test-v1",
        "sampling_policy_id": "every-graph-event-v1",
        "workflow_disjoint_test": workflow_disjoint_test,
        "workflow_id": "workflow-1",
        "decision_id": "decision-1",
        "workflow_template": "parallel-review",
        "model": "model-a",
        "layout_digest": "layout-a",
        "failure_domain": "node-a",
        "eligible_declared_parallel_region": True,
        "failure_exposed_active_kv_bytes_total": 300,
        "proof_scope_status": "AVAILABLE",
        "proof_scope_unavailable_reason": None,
        "proof_scope_kind": "CLOSED_MANDATORY_AND_JOIN",
        "observed_at_ms": 100.0,
        "graph_version": "graph-v1",
        "proof_scope_complete": True,
        "witness_contract_id": "scope-and-service-contract-v1",
        "witness_contract_valid_until_ms": 1_000.0,
        "completion_barrier_id": "join",
        "uses_future_information": False,
        "nodes": [
            {
                "node_id": "short",
                "agent_id": "agent-short",
                "need_state": "ACTIVE",
                "failure_exposed_kv_bytes": 100,
                "duration": _duration(1.0),
            },
            {
                "node_id": "long",
                "agent_id": "agent-long",
                "need_state": "ACTIVE",
                "failure_exposed_kv_bytes": 200,
                "duration": _duration(10.0),
            },
            {
                "node_id": "join",
                "agent_id": "join-agent",
                "need_state": "DEFERRED",
                "failure_exposed_kv_bytes": 0,
                "duration": _duration(0.0),
            },
        ],
        "edges": [
            {"source_id": "short", "target_id": "join"},
            {"source_id": "long", "target_id": "join"},
        ],
    }


def test_valid_closed_join_reports_point_estimates_without_passing_gate() -> None:
    result = _SCORER.score_records([_record()])

    assert result["gate_assessment"] == "NOT_ASSESSED"
    assert result["supported_records"] == 1
    row = result["rows"][0]
    assert row["status"] == "SCORED"
    assert row["positive_bounded_nodes"] == 1
    assert row["positive_bounded_kv_bytes"] == 100
    short_ticket = next(
        ticket for ticket in row["tickets"] if ticket["node_id"] == "short"
    )
    assert short_ticket["slack_ms"] == 9.0
    assert short_ticket["deadline_at_ms"] == 109.0

    summary = result["point_estimates_by_evidence_class"][
        "synthetic_development"
    ]
    assert summary["eligible_workflow_point_fraction"] == 1.0
    assert (
        summary["eligible_decision_snapshot_kv_byte_point_fraction"]
        == 1.0 / 3.0
    )


def test_synthetic_and_formal_rows_are_never_aggregated_together() -> None:
    formal = _record(
        evidence_class="held_out_formal", workflow_disjoint_test=True
    )
    formal["workflow_id"] = "workflow-formal"
    formal["decision_id"] = "decision-formal"

    result = _SCORER.score_records([_record(), formal])

    summaries = result["point_estimates_by_evidence_class"]
    assert summaries["synthetic_development"]["eligible_workflows"] == 1
    assert summaries["held_out_formal"]["eligible_workflows"] == 1
    assert result["held_out_formal_records"] == 1


def test_formal_row_requires_workflow_disjoint_test_split() -> None:
    result = _SCORER.score_records(
        [_record(evidence_class="held_out_formal")]
    )

    assert result["supported_records"] == 0
    assert result["rows"][0]["status"] == "UNSUPPORTED_BY_SCHEMA"
    assert "workflow_disjoint_test=true" in result["rows"][0]["error"]


def test_string_boolean_is_not_coerced_to_true() -> None:
    record = _record()
    record["proof_scope_complete"] = "false"

    result = _SCORER.score_records([record])

    assert result["supported_records"] == 0
    assert result["rows"][0]["status"] == "UNSUPPORTED_BY_SCHEMA"
    assert "must be a boolean" in result["rows"][0]["error"]


def test_extra_or_missing_fields_fail_closed() -> None:
    extra = _record()
    extra["hidden_default"] = 1
    missing = copy.deepcopy(_record())
    del missing["nodes"][0]["failure_exposed_kv_bytes"]

    result = _SCORER.score_records([extra, missing])

    assert result["supported_records"] == 0
    assert result["unsupported_records"] == 2
    assert all(
        row["status"] == "UNSUPPORTED_BY_SCHEMA" for row in result["rows"]
    )


def test_future_information_fails_closed() -> None:
    record = _record()
    record["nodes"][0]["duration"]["uses_future_information"] = True

    result = _SCORER.score_records([record])

    assert result["supported_records"] == 0
    assert result["rows"][0]["status"] == "UNSUPPORTED_BY_SCHEMA"
    assert "future information" in result["rows"][0]["error"]


def test_unavailable_scope_is_a_zero_opportunity_not_a_dropped_row() -> None:
    record = _record()
    record.update(
        {
            "proof_scope_status": "UNAVAILABLE",
            "proof_scope_unavailable_reason": "MISSING_HARD_DURATION_BOUND",
            "proof_scope_kind": None,
            "graph_version": "graph-v1",
            "proof_scope_complete": False,
            "witness_contract_id": None,
            "witness_contract_valid_until_ms": None,
            "completion_barrier_id": None,
            "nodes": [],
            "edges": [],
        }
    )

    result = _SCORER.score_records([record])

    assert result["supported_records"] == 1
    assert result["rows"][0]["status"] == "NO_PROOF_SCOPE"
    assert (
        result["rows"][0]["failure_exposed_active_kv_bytes_total"] == 300
    )
    summary = result["point_estimates_by_evidence_class"][
        "synthetic_development"
    ]
    assert summary["eligible_workflows"] == 1
    assert summary["eligible_workflow_point_fraction"] == 0.0
    assert summary["eligible_decision_snapshot_kv_byte_point_fraction"] == 0.0


def test_active_scope_bytes_cannot_exceed_sampling_frame_total() -> None:
    record = _record()
    record["failure_exposed_active_kv_bytes_total"] = 299

    result = _SCORER.score_records([record])

    assert result["supported_records"] == 0
    assert "exceed failure_exposed_active_kv_bytes_total" in (
        result["rows"][0]["error"]
    )


def test_positive_slack_on_unaffected_sibling_is_not_an_opportunity() -> None:
    record = _record()
    record["nodes"][0]["failure_exposed_kv_bytes"] = 0
    record["nodes"][1]["failure_exposed_kv_bytes"] = 300

    result = _SCORER.score_records([record])

    assert result["supported_records"] == 1
    assert result["rows"][0]["positive_bounded_nodes"] == 0
    assert result["rows"][0]["has_positive_bounded_deadline"] is False
    summary = result["point_estimates_by_evidence_class"][
        "synthetic_development"
    ]
    assert summary["eligible_workflow_point_fraction"] == 0.0


def test_duplicate_decision_identity_is_rejected() -> None:
    result = _SCORER.score_records([_record(), copy.deepcopy(_record())])

    assert result["supported_records"] == 1
    assert result["unsupported_records"] == 1
    assert result["rows"][1]["status"] == "UNSUPPORTED_BY_SCHEMA"
    assert "duplicate decision identity" in result["rows"][1]["error"]


def test_full_residual_dag_cannot_claim_a_closed_join_barrier() -> None:
    record = _record()
    record["proof_scope_kind"] = "COMPLETE_RESIDUAL_DAG"

    result = _SCORER.score_records([record])

    assert result["supported_records"] == 0
    assert "null completion barrier" in result["rows"][0]["error"]


def test_short_witness_contract_fails_closed_without_positive_opportunity() -> None:
    record = _record()
    record["witness_contract_valid_until_ms"] = 109.999

    result = _SCORER.score_records([record])

    assert result["supported_records"] == 1
    row = result["rows"][0]
    assert row["status"] == "SCORED"
    assert row["positive_bounded_nodes"] == 0
    assert all(
        ticket["slack_kind"] == "UNKNOWN" for ticket in row["tickets"]
    )
