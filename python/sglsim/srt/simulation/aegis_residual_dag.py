"""Causal residual-DAG CPM for the current AegisKV design.

The workflow runtime owns prediction and CPM.  The serving simulator consumes
only the resulting metadata ticket and decays its Slack with elapsed time.  A
positive ticket is published only when the *currently visible* residual graph
is complete through the next join and every visible agent node has a causal
remaining-time prediction.  Missing information fails closed to Slack zero;
building or applying a ticket never moves KV bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS


RESIDUAL_DAG_SLACK_SEMANTICS = "visible_residual_dag_cpm_slack_v1"


def _finite_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


@dataclass(frozen=True)
class ResidualDagNode:
    """One causally visible node in the residual workflow graph."""

    node_id: str
    node_kind: str
    predicted_remaining_ms: float | None
    active: bool = False
    agent_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.node_id).strip():
            raise ValueError("residual DAG node_id must be nonempty")
        if self.node_kind not in {"agent", "control"}:
            raise ValueError("residual DAG node_kind must be agent or control")
        if self.node_kind == "control":
            if self.active:
                raise ValueError("control nodes cannot be active agent frontiers")
            if self.predicted_remaining_ms not in {None, 0, 0.0}:
                raise ValueError("control-node weight must be zero")
        else:
            if not str(self.agent_id).strip():
                raise ValueError("agent nodes require agent_id")
            if self.predicted_remaining_ms is not None:
                _finite_nonnegative(
                    "predicted_remaining_ms", self.predicted_remaining_ms
                )

    @property
    def weight_ms(self) -> float | None:
        if self.node_kind == "control":
            return 0.0
        if self.predicted_remaining_ms is None:
            return None
        return float(self.predicted_remaining_ms)


@dataclass(frozen=True)
class VisibleResidualDag:
    workflow_id: str
    graph_version: str
    observed_at_ms: float
    nodes: tuple[ResidualDagNode, ...]
    edges: tuple[tuple[str, str], ...]
    next_join_subgraph_complete: bool
    metadata_complete: bool = True
    uses_future_information: bool = False

    def __post_init__(self) -> None:
        if not str(self.workflow_id).strip() or not str(self.graph_version).strip():
            raise ValueError("residual DAG workflow/version identity is required")
        _finite_nonnegative("observed_at_ms", self.observed_at_ms)
        if not self.nodes:
            raise ValueError("residual DAG must contain at least one visible node")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("residual DAG node IDs must be unique")
        known = set(ids)
        seen_edges: set[tuple[str, str]] = set()
        for source, target in self.edges:
            edge = (str(source), str(target))
            if source not in known or target not in known:
                raise ValueError("residual DAG edge references an unknown node")
            if source == target:
                raise ValueError("residual DAG cannot contain self edges")
            if edge in seen_edges:
                raise ValueError("residual DAG edges must be unique")
            seen_edges.add(edge)


@dataclass(frozen=True)
class ResidualDagSlackTicket:
    workflow_id: str
    graph_version: str
    observed_at_ms: float
    node_id: str
    agent_id: str
    earliest_start_ms: float
    latest_start_ms: float
    critical_path_ms: float
    slack_ms: float
    usable_slack: bool
    reason: str
    input_provenance_sha256: str
    uses_future_information: bool = False
    metadata_only: bool = True
    semantics: str = RESIDUAL_DAG_SLACK_SEMANTICS

    def to_simulation_fields(self) -> dict[str, Any]:
        return {
            "workflow_slack_ms": self.slack_ms,
            "aegis_slack_semantics": self.semantics,
            "aegis_slack_uses_future_information": self.uses_future_information,
            "aegis_slack_usable": self.usable_slack,
            "aegis_slack_kind": (
                "CPM_EXACT_TOTAL_FLOAT" if self.usable_slack else "UNKNOWN"
            ),
            "aegis_slack_reason": self.reason,
            "aegis_slack_graph_version": self.graph_version,
            "aegis_slack_observed_at_ms": self.observed_at_ms,
            "aegis_slack_node_id": self.node_id,
            "aegis_slack_agent_id": self.agent_id,
            "aegis_slack_es_ms": self.earliest_start_ms,
            "aegis_slack_ls_ms": self.latest_start_ms,
            "aegis_slack_critical_path_ms": self.critical_path_ms,
            "aegis_slack_input_provenance_sha256": self.input_provenance_sha256,
            "aegis_slack_residual_dag_visible": True,
            "aegis_slack_metadata_only": self.metadata_only,
        }


def _topological_order(
    nodes: Mapping[str, ResidualDagNode], edges: Iterable[tuple[str, str]]
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    predecessors = {node_id: [] for node_id in nodes}
    successors = {node_id: [] for node_id in nodes}
    indegree = {node_id: 0 for node_id in nodes}
    for source, target in edges:
        predecessors[target].append(source)
        successors[source].append(target)
        indegree[target] += 1
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for target in sorted(successors[node_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("visible residual graph must be a DAG")
    return order, predecessors, successors


def _provenance_sha256(graph: VisibleResidualDag) -> str:
    payload = json.dumps(
        asdict(graph), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_residual_dag_slack_tickets(
    graph: VisibleResidualDag,
) -> dict[str, ResidualDagSlackTicket]:
    """Run standard forward/backward CPM on the visible residual DAG."""

    nodes = {node.node_id: node for node in graph.nodes}
    order, predecessors, successors = _topological_order(nodes, graph.edges)
    active = [node for node in graph.nodes if node.active]
    if not active:
        return {}

    missing_prediction = any(node.weight_ms is None for node in graph.nodes)
    fail_reason = ""
    if graph.uses_future_information:
        fail_reason = "future_information_forbidden"
    elif not graph.next_join_subgraph_complete:
        fail_reason = "next_join_subgraph_incomplete"
    elif not graph.metadata_complete:
        fail_reason = "residual_dag_metadata_incomplete"
    elif missing_prediction:
        fail_reason = "visible_agent_prediction_missing"

    provenance = _provenance_sha256(graph)
    if fail_reason:
        return {
            node.node_id: ResidualDagSlackTicket(
                workflow_id=graph.workflow_id,
                graph_version=graph.graph_version,
                observed_at_ms=float(graph.observed_at_ms),
                node_id=node.node_id,
                agent_id=node.agent_id,
                earliest_start_ms=0.0,
                latest_start_ms=0.0,
                critical_path_ms=0.0,
                slack_ms=0.0,
                usable_slack=False,
                reason=fail_reason,
                input_provenance_sha256=provenance,
            )
            for node in active
        }

    weights = {node_id: float(nodes[node_id].weight_ms or 0.0) for node_id in nodes}
    earliest: dict[str, float] = {}
    for node_id in order:
        earliest[node_id] = max(
            (
                earliest[parent] + weights[parent]
                for parent in predecessors[node_id]
            ),
            default=0.0,
        )
    ends = [node_id for node_id in order if not successors[node_id]]
    critical_path = max(earliest[node_id] + weights[node_id] for node_id in ends)
    latest: dict[str, float] = {
        node_id: critical_path - weights[node_id] for node_id in ends
    }
    for node_id in reversed(order):
        if node_id in latest:
            continue
        latest[node_id] = min(
            latest[child] - weights[node_id] for child in successors[node_id]
        )

    return {
        node.node_id: ResidualDagSlackTicket(
            workflow_id=graph.workflow_id,
            graph_version=graph.graph_version,
            observed_at_ms=float(graph.observed_at_ms),
            node_id=node.node_id,
            agent_id=node.agent_id,
            earliest_start_ms=round(earliest[node.node_id], 6),
            latest_start_ms=round(latest[node.node_id], 6),
            critical_path_ms=round(critical_path, 6),
            slack_ms=round(max(latest[node.node_id] - earliest[node.node_id], 0.0), 6),
            usable_slack=True,
            reason="visible_residual_dag_cpm",
            input_provenance_sha256=provenance,
        )
        for node in active
    }


def resolve_slack_for_design(
    simulation: Mapping[str, Any], *, design_semantics: str
) -> tuple[float, str]:
    """Return trusted Slack, or zero for incomplete current-design metadata.

    Legacy experiments retain their historical scalar behavior.  The current
    design accepts a positive value only from a complete metadata-only ticket
    produced by :func:`build_residual_dag_slack_tickets`.
    """

    raw = simulation.get("workflow_slack_ms", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value < 0.0:
        value = 0.0
    if str(design_semantics) != DESIGN_SEMANTICS:
        return value, "legacy_scalar"

    required = {
        "aegis_slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
        "aegis_slack_uses_future_information": False,
        "aegis_slack_residual_dag_visible": True,
        "aegis_slack_metadata_only": True,
    }
    for field, expected in required.items():
        if simulation.get(field) != expected:
            return 0.0, f"missing_or_invalid_{field}"
    for field in (
        "aegis_slack_graph_version",
        "aegis_slack_node_id",
        "aegis_slack_agent_id",
        "aegis_slack_input_provenance_sha256",
    ):
        if not str(simulation.get(field) or "").strip():
            return 0.0, f"missing_{field}"
    if simulation.get("aegis_slack_usable") is not True:
        return 0.0, str(simulation.get("aegis_slack_reason") or "slack_unusable")
    return value, "visible_residual_dag_cpm"


__all__ = [
    "RESIDUAL_DAG_SLACK_SEMANTICS",
    "ResidualDagNode",
    "ResidualDagSlackTicket",
    "VisibleResidualDag",
    "build_residual_dag_slack_tickets",
    "resolve_slack_for_design",
]
