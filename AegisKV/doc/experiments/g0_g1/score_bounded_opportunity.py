#!/usr/bin/env python3
"""Score deterministic BOUNDED-Slack opportunity without upgrading evidence.

The input is JSONL following ``bounded_scope.schema.json``.  This script uses
only explicit fields and emits ``UNSUPPORTED_BY_SCHEMA`` for malformed rows.
It intentionally does not compute a pass/fail confidence interval: formal G1
uses workflow-clustered inference on a frozen split, not snapshots as IID rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from sglsim.srt.simulation.aegis_bounded_slack import (
    AegisHardDurationBound,
    AegisProofScopeKind,
    AegisResidualDag,
    AegisResidualEdge,
    AegisResidualNode,
    build_bounded_slack_tickets,
)
from sglsim.srt.simulation.aegis_causal_slack import AegisNeedState


REQUIRED_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_class",
        "dataset_id",
        "split_id",
        "sampling_policy_id",
        "workflow_disjoint_test",
        "workflow_id",
        "decision_id",
        "workflow_template",
        "model",
        "layout_digest",
        "failure_domain",
        "eligible_declared_parallel_region",
        "failure_exposed_active_kv_bytes_total",
        "proof_scope_status",
        "proof_scope_unavailable_reason",
        "proof_scope_kind",
        "observed_at_ms",
        "graph_version",
        "proof_scope_complete",
        "witness_contract_id",
        "witness_contract_valid_until_ms",
        "completion_barrier_id",
        "uses_future_information",
        "nodes",
        "edges",
    }
)
PROOF_SCOPE_KINDS = frozenset(
    {"COMPLETE_RESIDUAL_DAG", "CLOSED_MANDATORY_AND_JOIN"}
)
PROOF_SCOPE_UNAVAILABLE_REASONS = frozenset(
    {
        "NO_DECLARED_PARALLEL_REGION",
        "NO_REVEALED_PROOF_SCOPE",
        "INCOMPLETE_SCOPE",
        "MISSING_HARD_DURATION_BOUND",
        "STALE_GRAPH_OR_BOUND_VERSION",
        "UNENFORCEABLE_SERVICE_ENVELOPE",
        "UNFENCED_WITNESS_CONTRACT",
        "WITNESS_CONTRACT_TOO_SHORT",
        "UNSUPPORTED_JOIN_SEMANTICS",
        "UNSUPPORTED_LAYOUT",
    }
)
REQUIRED_NODE_FIELDS = frozenset(
    {
        "node_id",
        "agent_id",
        "need_state",
        "failure_exposed_kv_bytes",
        "duration",
    }
)
REQUIRED_DURATION_FIELDS = frozenset(
    {
        "lower_ms",
        "upper_ms",
        "graph_version",
        "service_envelope_version",
        "bound_id",
        "source",
        "uses_future_information",
    }
)


class UnsupportedSchema(ValueError):
    pass


def _exact_fields(
    raw: dict[str, Any], fields: frozenset[str], scope: str
) -> None:
    missing = sorted(fields - raw.keys())
    if missing:
        raise UnsupportedSchema(f"{scope} missing required fields: {missing}")
    extra = sorted(raw.keys() - fields)
    if extra:
        raise UnsupportedSchema(f"{scope} has unsupported fields: {extra}")


def _nonempty(raw: dict[str, Any], field: str) -> str:
    if not isinstance(raw[field], str):
        raise UnsupportedSchema(f"{field} must be a string")
    value = raw[field].strip()
    if not value:
        raise UnsupportedSchema(f"{field} must be nonempty")
    return value


def _boolean(raw: dict[str, Any], field: str) -> bool:
    value = raw[field]
    if type(value) is not bool:
        raise UnsupportedSchema(f"{field} must be a boolean")
    return value


def _number(raw: dict[str, Any], field: str) -> float:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedSchema(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise UnsupportedSchema(f"{field} must be finite")
    return parsed


def _parse_record(
    raw: dict[str, Any],
) -> tuple[AegisResidualDag | None, dict[str, int], int]:
    _exact_fields(raw, REQUIRED_RECORD_FIELDS, "record")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 3:
        raise UnsupportedSchema("schema_version must equal 3")
    if raw["evidence_class"] not in {
        "synthetic_development",
        "held_out_formal",
    }:
        raise UnsupportedSchema("unsupported evidence_class")
    if raw["uses_future_information"] is not False:
        raise UnsupportedSchema("record uses future information")
    workflow_disjoint_test = _boolean(raw, "workflow_disjoint_test")
    _boolean(raw, "eligible_declared_parallel_region")
    proof_scope_complete = _boolean(raw, "proof_scope_complete")
    if raw["evidence_class"] == "held_out_formal" and not workflow_disjoint_test:
        raise UnsupportedSchema(
            "held_out_formal requires workflow_disjoint_test=true"
        )
    for field in (
        "dataset_id",
        "split_id",
        "sampling_policy_id",
        "workflow_id",
        "decision_id",
        "workflow_template",
        "model",
        "layout_digest",
        "failure_domain",
    ):
        _nonempty(raw, field)
    active_kv_bytes_total = raw["failure_exposed_active_kv_bytes_total"]
    if (
        isinstance(active_kv_bytes_total, bool)
        or not isinstance(active_kv_bytes_total, int)
        or active_kv_bytes_total < 0
    ):
        raise UnsupportedSchema(
            "failure_exposed_active_kv_bytes_total must be a nonnegative "
            "integer"
        )
    if not isinstance(raw["nodes"], list):
        raise UnsupportedSchema("nodes must be a list")
    if not isinstance(raw["edges"], list):
        raise UnsupportedSchema("edges must be a list")

    scope_status = raw["proof_scope_status"]
    if scope_status not in {"AVAILABLE", "UNAVAILABLE"}:
        raise UnsupportedSchema("unsupported proof_scope_status")
    if scope_status == "AVAILABLE" and not raw["nodes"]:
        raise UnsupportedSchema("available proof scope needs nonempty nodes")
    if scope_status == "UNAVAILABLE":
        if raw["proof_scope_unavailable_reason"] not in (
            PROOF_SCOPE_UNAVAILABLE_REASONS
        ):
            raise UnsupportedSchema(
                "unavailable proof scope needs a supported reason"
            )
        if raw["proof_scope_kind"] is not None:
            raise UnsupportedSchema(
                "unavailable proof scope must have null proof_scope_kind"
            )
        if proof_scope_complete:
            raise UnsupportedSchema(
                "unavailable proof scope cannot be marked complete"
            )
        if raw["completion_barrier_id"] is not None:
            raise UnsupportedSchema(
                "unavailable proof scope must have null completion barrier"
            )
        if raw["witness_contract_id"] is not None:
            raise UnsupportedSchema(
                "unavailable proof scope must have null witness contract"
            )
        if raw["witness_contract_valid_until_ms"] is not None:
            raise UnsupportedSchema(
                "unavailable proof scope must have null witness validity"
            )
        graph_version = raw["graph_version"]
        if graph_version is not None:
            _nonempty(raw, "graph_version")
        if raw["nodes"] or raw["edges"]:
            raise UnsupportedSchema(
                "unavailable proof scope must not carry partial graph data"
            )
        return None, {}, active_kv_bytes_total

    if raw["proof_scope_unavailable_reason"] is not None:
        raise UnsupportedSchema(
            "available proof scope must have null unavailable reason"
        )
    scope_kind = raw["proof_scope_kind"]
    if scope_kind not in PROOF_SCOPE_KINDS:
        raise UnsupportedSchema("available proof scope needs a supported kind")
    if not proof_scope_complete:
        raise UnsupportedSchema("available proof scope must be complete")
    graph_version = _nonempty(raw, "graph_version")
    witness_contract_id = _nonempty(raw, "witness_contract_id")
    witness_contract_valid_until_ms = _number(
        raw, "witness_contract_valid_until_ms"
    )
    barrier_value = raw["completion_barrier_id"]
    if scope_kind == "COMPLETE_RESIDUAL_DAG":
        if barrier_value is not None:
            raise UnsupportedSchema(
                "complete residual DAG must have a null completion barrier"
            )
    else:
        if not isinstance(barrier_value, str) or not barrier_value.strip():
            raise UnsupportedSchema(
                "closed mandatory AND-join scope needs a completion barrier"
            )

    nodes: list[AegisResidualNode] = []
    kv_bytes_by_node: dict[str, int] = {}
    for index, node_raw_value in enumerate(raw["nodes"]):
        if not isinstance(node_raw_value, dict):
            raise UnsupportedSchema(f"node[{index}] must be an object")
        node_raw = node_raw_value
        _exact_fields(node_raw, REQUIRED_NODE_FIELDS, f"node[{index}]")
        duration_value = node_raw["duration"]
        if not isinstance(duration_value, dict):
            raise UnsupportedSchema(f"node[{index}].duration must be an object")
        duration_raw = duration_value
        _exact_fields(
            duration_raw,
            REQUIRED_DURATION_FIELDS,
            f"node[{index}].duration",
        )
        if duration_raw["uses_future_information"] is not False:
            raise UnsupportedSchema(
                f"node[{index}] duration uses future information"
            )
        kv_bytes = node_raw["failure_exposed_kv_bytes"]
        if isinstance(kv_bytes, bool) or not isinstance(kv_bytes, int):
            raise UnsupportedSchema(
                f"node[{index}].failure_exposed_kv_bytes must be an integer"
            )
        if kv_bytes < 0:
            raise UnsupportedSchema(
                f"node[{index}].failure_exposed_kv_bytes must be nonnegative"
            )
        node_id = _nonempty(node_raw, "node_id")
        kv_bytes_by_node[node_id] = kv_bytes
        nodes.append(
            AegisResidualNode(
                node_id=node_id,
                agent_id=_nonempty(node_raw, "agent_id"),
                need_state=AegisNeedState(node_raw["need_state"]),
                duration=AegisHardDurationBound(
                    lower_ms=_number(duration_raw, "lower_ms"),
                    upper_ms=_number(duration_raw, "upper_ms"),
                    graph_version=_nonempty(duration_raw, "graph_version"),
                    service_envelope_version=_nonempty(
                        duration_raw, "service_envelope_version"
                    ),
                    bound_id=_nonempty(duration_raw, "bound_id"),
                    source=_nonempty(duration_raw, "source"),
                ),
            )
        )

    edges: list[AegisResidualEdge] = []
    for index, edge_value in enumerate(raw["edges"]):
        if not isinstance(edge_value, dict):
            raise UnsupportedSchema(f"edge[{index}] must be an object")
        if set(edge_value) != {"source_id", "target_id"}:
            raise UnsupportedSchema(
                f"edge[{index}] requires only source_id and target_id"
            )
        edges.append(
            AegisResidualEdge(
                _nonempty(edge_value, "source_id"),
                _nonempty(edge_value, "target_id"),
            )
        )
    dag = AegisResidualDag(
        workflow_id=_nonempty(raw, "workflow_id"),
        graph_version=graph_version,
        observed_at_ms=_number(raw, "observed_at_ms"),
        nodes=tuple(nodes),
        edges=tuple(edges),
        proof_scope_kind=AegisProofScopeKind(scope_kind),
        proof_scope_complete=proof_scope_complete,
        witness_contract_id=witness_contract_id,
        witness_contract_valid_until_ms=witness_contract_valid_until_ms,
        completion_barrier_id=(
            None if barrier_value is None else barrier_value.strip()
        ),
    )
    scoped_active_kv_bytes = sum(
        kv_bytes_by_node[node.node_id]
        for node in dag.nodes
        if node.need_state is AegisNeedState.ACTIVE
    )
    if scoped_active_kv_bytes > active_kv_bytes_total:
        raise UnsupportedSchema(
            "scoped ACTIVE failure-exposed bytes exceed "
            "failure_exposed_active_kv_bytes_total"
        )
    return dag, kv_bytes_by_node, active_kv_bytes_total


def _point_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["eligible_declared_parallel_region"] is True
    ]
    available = [
        row for row in rows if row["proof_scope_status"] == "AVAILABLE"
    ]
    all_workflows = {str(row["workflow_id"]) for row in rows}
    eligible_workflows = {str(row["workflow_id"]) for row in eligible}
    all_positive_workflows = {
        str(row["workflow_id"])
        for row in rows
        if row["has_positive_bounded_deadline"]
    }
    eligible_positive_workflows = {
        str(row["workflow_id"])
        for row in eligible
        if row["has_positive_bounded_deadline"]
    }
    all_active_bytes = sum(
        int(row["failure_exposed_active_kv_bytes_total"]) for row in rows
    )
    all_positive_bytes = sum(
        int(row["positive_bounded_kv_bytes"]) for row in rows
    )
    eligible_active_bytes = sum(
        int(row["failure_exposed_active_kv_bytes_total"])
        for row in eligible
    )
    eligible_positive_bytes = sum(
        int(row["positive_bounded_kv_bytes"]) for row in eligible
    )
    return {
        "scored_decision_snapshots": len(rows),
        "available_proof_scope_decision_snapshots": len(available),
        "eligible_decision_snapshots": len(eligible),
        "all_workflows": len(all_workflows),
        "all_positive_bounded_workflows": len(all_positive_workflows),
        "unconditional_workflow_point_fraction": (
            len(all_positive_workflows) / len(all_workflows)
            if all_workflows
            else None
        ),
        "eligible_workflows": len(eligible_workflows),
        "eligible_positive_bounded_workflows": len(
            eligible_positive_workflows
        ),
        "eligible_workflow_point_fraction": (
            len(eligible_positive_workflows) / len(eligible_workflows)
            if eligible_workflows
            else None
        ),
        "all_decision_snapshot_active_kv_bytes": all_active_bytes,
        "all_positive_bounded_decision_snapshot_kv_bytes": (
            all_positive_bytes
        ),
        "unconditional_decision_snapshot_kv_byte_point_fraction": (
            all_positive_bytes / all_active_bytes
            if all_active_bytes
            else None
        ),
        "eligible_decision_snapshot_active_kv_bytes": eligible_active_bytes,
        "eligible_positive_bounded_decision_snapshot_kv_bytes": (
            eligible_positive_bytes
        ),
        "eligible_decision_snapshot_kv_byte_point_fraction": (
            eligible_positive_bytes / eligible_active_bytes
            if eligible_active_bytes
            else None
        ),
    }


def score_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_decisions: set[tuple[str, ...]] = set()
    supported = 0
    unsupported = 0
    for row_index, raw in enumerate(records):
        identity = {
            "row_index": row_index,
            "workflow_id": raw.get("workflow_id"),
            "decision_id": raw.get("decision_id"),
            "evidence_class": raw.get("evidence_class"),
            "dataset_id": raw.get("dataset_id"),
            "split_id": raw.get("split_id"),
            "sampling_policy_id": raw.get("sampling_policy_id"),
            "failure_domain": raw.get("failure_domain"),
            "eligible_declared_parallel_region": raw.get(
                "eligible_declared_parallel_region"
            ),
            "proof_scope_status": raw.get("proof_scope_status"),
            "workflow_template": raw.get("workflow_template"),
            "model": raw.get("model"),
            "layout_digest": raw.get("layout_digest"),
        }
        try:
            dag, kv_bytes_by_node, active_kv_bytes_total = _parse_record(raw)
            decision_key = tuple(
                str(raw[field])
                for field in (
                    "evidence_class",
                    "dataset_id",
                    "split_id",
                    "sampling_policy_id",
                    "workflow_id",
                    "decision_id",
                    "failure_domain",
                )
            )
            if decision_key in seen_decisions:
                raise UnsupportedSchema("duplicate decision identity")
            seen_decisions.add(decision_key)
            tickets = (
                {} if dag is None else build_bounded_slack_tickets(dag)
            )
        except (KeyError, TypeError, ValueError) as exc:
            unsupported += 1
            rows.append(
                {
                    **identity,
                    "status": "UNSUPPORTED_BY_SCHEMA",
                    "error": str(exc),
                }
            )
            continue
        supported += 1
        if dag is None:
            rows.append(
                {
                    **identity,
                    "status": "NO_PROOF_SCOPE",
                    "active_nodes_in_scope": 0,
                    "positive_bounded_nodes": 0,
                    "failure_exposed_active_kv_bytes_total": (
                        active_kv_bytes_total
                    ),
                    "scoped_active_kv_bytes": 0,
                    "positive_bounded_kv_bytes": 0,
                    "has_positive_bounded_deadline": False,
                    "proof_scope_unavailable_reason": raw[
                        "proof_scope_unavailable_reason"
                    ],
                    "tickets": [],
                }
            )
            continue
        active_nodes = {
            node.node_id
            for node in dag.nodes
            if node.need_state is AegisNeedState.ACTIVE
        }
        positive_nodes = {
            node_id
            for node_id, ticket in tickets.items()
            if (
                ticket.slack_ms > 0.0
                and node_id in active_nodes
                and kv_bytes_by_node[node_id] > 0
            )
        }
        active_bytes = sum(kv_bytes_by_node[node_id] for node_id in active_nodes)
        positive_bytes = sum(
            kv_bytes_by_node[node_id] for node_id in positive_nodes
        )
        rows.append(
            {
                **identity,
                "status": "SCORED",
                "active_nodes_in_scope": len(active_nodes),
                "positive_bounded_nodes": len(positive_nodes),
                "failure_exposed_active_kv_bytes_total": (
                    active_kv_bytes_total
                ),
                "scoped_active_kv_bytes": active_bytes,
                "positive_bounded_kv_bytes": positive_bytes,
                "has_positive_bounded_deadline": bool(positive_nodes),
                "tickets": [
                    {
                        "node_id": node_id,
                        "slack_kind": ticket.slack_kind.value,
                        "slack_ms": ticket.slack_ms,
                        "deadline_at_ms": ticket.deadline_at_ms,
                        "failure_exposed_kv_bytes": kv_bytes_by_node[node_id],
                        "witness_sha256": ticket.witness_sha256,
                    }
                    for node_id, ticket in sorted(tickets.items())
                ],
            }
        )

    scored = [
        row
        for row in rows
        if row["status"] in {"SCORED", "NO_PROOF_SCOPE"}
    ]
    formal_rows = [
        row for row in scored if row["evidence_class"] == "held_out_formal"
    ]
    development_rows = [
        row
        for row in scored
        if row["evidence_class"] == "synthetic_development"
    ]
    formal_contracts = sorted(
        {
            (
                str(row["dataset_id"]),
                str(row["split_id"]),
                str(row["sampling_policy_id"]),
            )
            for row in formal_rows
        }
    )
    if not formal_contracts:
        formal_input_consistency = "NO_FORMAL_ROWS"
    elif len(formal_contracts) == 1:
        formal_input_consistency = "PASS"
    else:
        formal_input_consistency = "REFUSED_MIXED_COLLECTION_CONTRACTS"
    return {
        "schema_version": 3,
        "evidence_boundary": (
            "point_estimates_only_not_a_G1_pass_without_frozen_split_and_"
            "workflow_clustered_confidence_intervals"
        ),
        "gate_assessment": "NOT_ASSESSED",
        "input_records": len(rows),
        "supported_records": supported,
        "unsupported_records": unsupported,
        "held_out_formal_records": len(formal_rows),
        "formal_input_consistency": formal_input_consistency,
        "formal_collection_contracts": [
            {
                "dataset_id": dataset_id,
                "split_id": split_id,
                "sampling_policy_id": sampling_policy_id,
            }
            for dataset_id, split_id, sampling_policy_id in formal_contracts
        ],
        "point_estimates_by_evidence_class": {
            "synthetic_development": _point_summary(development_rows),
            "held_out_formal": _point_summary(formal_rows),
        },
        "rows": rows,
    }


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise UnsupportedSchema(f"line {line_number} must be a JSON object")
        records.append(value)
    return records, hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    records, input_sha256 = _read_jsonl(args.input_jsonl)
    result = score_records(records)
    result["input_jsonl"] = str(args.input_jsonl)
    result["input_sha256"] = input_sha256
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
