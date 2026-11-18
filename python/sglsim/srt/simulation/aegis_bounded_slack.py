"""Deterministic positive-``BOUNDED`` Slack for a declared residual DAG.

The statistical model in :mod:`aegis_causal_slack` cannot turn a prediction
into a hard bound.  This module implements the smaller theorem-eligible path
from the AegisKV design.  The workflow runtime supplies either a complete,
versioned residual DAG or a closed region ending at the next mandatory
AND-join.  Supernodes already coalesce each recovery context's causal chain.
A closed join is sufficient because the common suffix after it cancels from
branch Slack; Aegis need not predict the entire future workflow.  Every node
supplies a finite lower and upper duration witness tied to the same graph
version and to a versioned service envelope.  That envelope must jointly admit
all simultaneous floors on each shared resource and provide composable queue
bounds; independent oversubscribed floors are not witnesses.  The graph and service witnesses
are covered by one fenced contract whose lifetime reaches the declared upper
completion horizon.  A weakening graph/service event must fence the old
contract before it becomes effective; an asynchronous best-effort invalidation
is not a hard-bound premise.

For context ``i`` we compute

``L_without_i``
    the longest original source-to-sink path avoiding ``i`` using lower node
    bounds, and

``U_through_i``
    the longest original source-to-sink path through ``i`` using upper node
    bounds.

Then ``max(L_without_i - U_through_i, 0)`` is a deterministic lower bound on
the structural Slack of ``i``.  Original sources and sinks are preserved when
``i`` is removed; otherwise a truncated path could be mistaken for a complete
workflow path.  An incomplete graph, missing bound, or stale bound fails every
ticket closed to ``UNKNOWN`` instead of borrowing a calibrated estimate.

This is a semantic/proof reference.  It does not prove that an application
actually provides useful hard witnesses or that a physical service envelope
is enforceable; those are G1 and implementation-conformance obligations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum

from sglsim.srt.simulation.aegis_causal_slack import (
    FORBIDDEN_PROVENANCE_FRAGMENTS,
    AegisNeedState,
    AegisSlackKind,
)


BOUNDED_SLACK_SEMANTICS = "declared_residual_dag_hard_bounds_v2"


class AegisProofScopeKind(str, Enum):
    COMPLETE_RESIDUAL_DAG = "COMPLETE_RESIDUAL_DAG"
    CLOSED_MANDATORY_AND_JOIN = "CLOSED_MANDATORY_AND_JOIN"


def _finite_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


@dataclass(frozen=True)
class AegisHardDurationBound:
    lower_ms: float
    upper_ms: float
    graph_version: str
    service_envelope_version: str
    bound_id: str
    source: str
    uses_future_information: bool = False

    def __post_init__(self) -> None:
        lower = _finite_nonnegative("lower_ms", self.lower_ms)
        upper = _finite_nonnegative("upper_ms", self.upper_ms)
        if lower > upper:
            raise ValueError("hard duration lower bound exceeds upper bound")
        for name in (
            "graph_version",
            "service_envelope_version",
            "bound_id",
            "source",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be nonempty")
        if self.uses_future_information:
            raise ValueError("hard duration bound cannot use future information")
        lowered = self.source.lower()
        if any(
            fragment in lowered
            for fragment in FORBIDDEN_PROVENANCE_FRAGMENTS
        ):
            raise ValueError(
                f"post-hoc hard-bound source is invalid: {self.source!r}"
            )

    @classmethod
    def from_work_service_envelope(
        cls,
        *,
        work_lower_units: float,
        work_upper_units: float,
        service_rate_lower_units_per_ms: float,
        service_rate_upper_units_per_ms: float,
        fixed_lower_ms: float,
        fixed_upper_ms: float,
        queue_wait_upper_ms: float,
        work_unit: str,
        service_rate_work_unit: str,
        graph_version: str,
        service_envelope_version: str,
        bound_id: str,
        source: str,
    ) -> "AegisHardDurationBound":
        """Derive a duration interval without fitting a hidden constant.

        ``service_rate_lower`` must be an enforceable reservation and
        ``service_rate_upper`` a physical/implementation maximum in the same
        work unit.  Queue wait has a zero lower bound and a finite hard upper
        bound.  Consequently

        ``L = fixed_lower + work_lower / rate_upper`` and
        ``U = fixed_upper + queue_wait_upper + work_upper / rate_lower``.
        """

        work_lower = _finite_nonnegative(
            "work_lower_units", work_lower_units
        )
        work_upper = _finite_nonnegative(
            "work_upper_units", work_upper_units
        )
        if not work_unit or work_unit != service_rate_work_unit:
            raise ValueError(
                "work and service-rate envelopes must use the same unit"
            )
        if work_lower > work_upper:
            raise ValueError("work lower bound exceeds upper bound")
        rate_lower = float(service_rate_lower_units_per_ms)
        rate_upper = float(service_rate_upper_units_per_ms)
        if (
            not math.isfinite(rate_lower)
            or not math.isfinite(rate_upper)
            or rate_lower <= 0.0
            or rate_upper <= 0.0
            or rate_lower > rate_upper
        ):
            raise ValueError(
                "service rates must be finite, positive, and ordered"
            )
        fixed_lower = _finite_nonnegative("fixed_lower_ms", fixed_lower_ms)
        fixed_upper = _finite_nonnegative("fixed_upper_ms", fixed_upper_ms)
        if fixed_lower > fixed_upper:
            raise ValueError("fixed-time lower bound exceeds upper bound")
        queue_upper = _finite_nonnegative(
            "queue_wait_upper_ms", queue_wait_upper_ms
        )
        return cls(
            lower_ms=fixed_lower + work_lower / rate_upper,
            upper_ms=(
                fixed_upper + queue_upper + work_upper / rate_lower
            ),
            graph_version=graph_version,
            service_envelope_version=service_envelope_version,
            bound_id=bound_id,
            source=source,
        )


@dataclass(frozen=True)
class AegisResidualNode:
    node_id: str
    agent_id: str
    need_state: AegisNeedState
    duration: AegisHardDurationBound | None

    def __post_init__(self) -> None:
        if not self.node_id or not self.agent_id:
            raise ValueError("node_id and agent_id must be nonempty")
        try:
            normalized = AegisNeedState(self.need_state)
        except ValueError as exc:
            raise ValueError("need_state must be ACTIVE or DEFERRED") from exc
        object.__setattr__(self, "need_state", normalized)


@dataclass(frozen=True)
class AegisResidualEdge:
    source_id: str
    target_id: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise ValueError("edge endpoints must be nonempty")
        if self.source_id == self.target_id:
            raise ValueError("residual DAG cannot contain self loops")


@dataclass(frozen=True)
class AegisResidualDag:
    workflow_id: str
    graph_version: str
    observed_at_ms: float
    nodes: tuple[AegisResidualNode, ...]
    edges: tuple[AegisResidualEdge, ...]
    proof_scope_kind: AegisProofScopeKind
    proof_scope_complete: bool
    witness_contract_id: str
    witness_contract_valid_until_ms: float
    completion_barrier_id: str | None = None

    def __post_init__(self) -> None:
        if not self.workflow_id or not self.graph_version:
            raise ValueError("workflow_id and graph_version must be nonempty")
        try:
            scope_kind = AegisProofScopeKind(self.proof_scope_kind)
        except ValueError as exc:
            raise ValueError("unsupported proof scope kind") from exc
        object.__setattr__(self, "proof_scope_kind", scope_kind)
        if type(self.proof_scope_complete) is not bool:
            raise ValueError("proof_scope_complete must be a boolean")
        _finite_nonnegative("observed_at_ms", self.observed_at_ms)
        if not str(self.witness_contract_id).strip():
            raise ValueError("witness_contract_id must be nonempty")
        _finite_nonnegative(
            "witness_contract_valid_until_ms",
            self.witness_contract_valid_until_ms,
        )
        if not self.nodes:
            raise ValueError("residual DAG must contain at least one node")
        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("residual DAG node IDs must be unique")
        endpoints = set(node_ids)
        edge_pairs = [
            (edge.source_id, edge.target_id) for edge in self.edges
        ]
        if len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError("residual DAG edges must be unique")
        for source_id, target_id in edge_pairs:
            if source_id not in endpoints or target_id not in endpoints:
                raise ValueError("residual DAG edge references an unknown node")
        _topological_order(self)
        if scope_kind is AegisProofScopeKind.COMPLETE_RESIDUAL_DAG:
            if self.completion_barrier_id is not None:
                raise ValueError(
                    "complete residual DAG must not declare a completion "
                    "barrier"
                )
        else:
            if self.completion_barrier_id is None:
                raise ValueError(
                    "closed mandatory AND-join scope requires a completion "
                    "barrier"
                )
            barrier = str(self.completion_barrier_id).strip()
            if not barrier or barrier not in endpoints:
                raise ValueError(
                    "completion barrier must reference a residual DAG node"
                )
            _node_by_id, _predecessors, successors = _graph_maps(self)
            if successors[barrier]:
                raise ValueError("completion barrier must be a sink")
            reaches_barrier = {barrier}
            changed = True
            while changed:
                changed = False
                for edge in self.edges:
                    if (
                        edge.target_id in reaches_barrier
                        and edge.source_id not in reaches_barrier
                    ):
                        reaches_barrier.add(edge.source_id)
                        changed = True
            if reaches_barrier != endpoints:
                raise ValueError(
                    "every closed-region node must reach the completion barrier"
                )


@dataclass(frozen=True)
class AegisBoundedSlackTicket:
    workflow_id: str
    node_id: str
    agent_id: str
    need_state: AegisNeedState
    slack_kind: AegisSlackKind
    slack_ms: float
    observed_at_ms: float
    deadline_at_ms: float
    lower_avoiding_ms: float | None
    upper_through_ms: float | None
    graph_version: str
    proof_scope_kind: AegisProofScopeKind
    proof_scope_complete: bool
    witness_contract_id: str
    witness_contract_valid_until_ms: float
    completion_barrier_id: str | None
    service_envelope_versions: tuple[str, ...]
    source: str
    usable_slack: bool
    witness_sha256: str
    uses_future_information: bool = False

    def to_simulation_fields(self) -> dict[str, object]:
        return {
            "workflow_slack_ms": self.slack_ms,
            "aegis_slack_lcb_ms": self.slack_ms,
            "aegis_slack_observed_at_ms": self.observed_at_ms,
            "aegis_deadline_at_ms": self.deadline_at_ms,
            "aegis_need_state": self.need_state.value,
            "aegis_slack_kind": self.slack_kind.value,
            "aegis_slack_semantics": BOUNDED_SLACK_SEMANTICS,
            "aegis_slack_source": self.source,
            "aegis_slack_uses_future_information": False,
            "aegis_slack_usable": self.usable_slack,
            "aegis_slack_graph_version": self.graph_version,
            "aegis_slack_proof_scope_kind": self.proof_scope_kind.value,
            "aegis_slack_proof_scope_complete": self.proof_scope_complete,
            "aegis_slack_witness_contract_id": self.witness_contract_id,
            "aegis_slack_witness_contract_valid_until_ms": (
                self.witness_contract_valid_until_ms
            ),
            "aegis_slack_completion_barrier_id": (
                self.completion_barrier_id
            ),
            "aegis_slack_node_id": self.node_id,
            "aegis_slack_agent_id": self.agent_id,
            "aegis_slack_lower_avoiding_ms": self.lower_avoiding_ms,
            "aegis_slack_upper_through_ms": self.upper_through_ms,
            "aegis_slack_service_envelope_versions": list(
                self.service_envelope_versions
            ),
            "aegis_slack_witness_sha256": self.witness_sha256,
        }


def _graph_maps(
    dag: AegisResidualDag,
) -> tuple[
    dict[str, AegisResidualNode],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    node_by_id = {node.node_id: node for node in dag.nodes}
    predecessors: dict[str, list[str]] = {
        node_id: [] for node_id in node_by_id
    }
    successors: dict[str, list[str]] = {
        node_id: [] for node_id in node_by_id
    }
    for edge in dag.edges:
        predecessors[edge.target_id].append(edge.source_id)
        successors[edge.source_id].append(edge.target_id)
    return (
        node_by_id,
        {
            node_id: tuple(sorted(values))
            for node_id, values in predecessors.items()
        },
        {
            node_id: tuple(sorted(values))
            for node_id, values in successors.items()
        },
    )


def _topological_order(dag: AegisResidualDag) -> tuple[str, ...]:
    node_by_id, predecessors, successors = _graph_maps(dag)
    indegree = {
        node_id: len(predecessors[node_id]) for node_id in node_by_id
    }
    ready = sorted(
        node_id for node_id, degree in indegree.items() if degree == 0
    )
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for successor in successors[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != len(node_by_id):
        raise ValueError("residual graph must be acyclic")
    return tuple(order)


def _witness_sha256(dag: AegisResidualDag, node_id: str) -> str:
    payload = json.dumps(
        {"dag": asdict(dag), "ticket_node_id": node_id},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unknown_reason(dag: AegisResidualDag) -> str | None:
    if not dag.proof_scope_complete:
        return "fail_closed_incomplete_proof_scope"
    if any(node.duration is None for node in dag.nodes):
        return "fail_closed_missing_hard_duration_bound"
    if any(
        node.duration is not None
        and node.duration.graph_version != dag.graph_version
        for node in dag.nodes
    ):
        return "fail_closed_stale_hard_duration_bound"
    _node_by_id, predecessors, successors = _graph_maps(dag)
    order = _topological_order(dag)
    upper_completion: dict[str, float] = {}
    for node_id in order:
        node = _node_by_id[node_id]
        assert node.duration is not None
        upper_completion[node_id] = node.duration.upper_ms + max(
            (upper_completion[parent] for parent in predecessors[node_id]),
            default=0.0,
        )
    completion_horizon_ms = dag.observed_at_ms + max(
        upper_completion[node_id]
        for node_id in order
        if not successors[node_id]
    )
    if dag.witness_contract_valid_until_ms < completion_horizon_ms:
        return "fail_closed_short_witness_contract"
    return None


def _longest_lower_path_avoiding(
    *,
    excluded: str,
    order: tuple[str, ...],
    sources: frozenset[str],
    sinks: frozenset[str],
    predecessors: dict[str, tuple[str, ...]],
    lower: dict[str, float],
) -> float | None:
    negative_infinity = float("-inf")
    distance = {node_id: negative_infinity for node_id in order}
    for node_id in order:
        if node_id == excluded:
            continue
        candidates = [
            distance[parent]
            for parent in predecessors[node_id]
            if parent != excluded and distance[parent] != negative_infinity
        ]
        if node_id in sources:
            candidates.append(0.0)
        if candidates:
            distance[node_id] = max(candidates) + lower[node_id]
    completed = [
        distance[node_id]
        for node_id in sinks
        if node_id != excluded and distance[node_id] != negative_infinity
    ]
    return max(completed) if completed else None


def build_bounded_slack_tickets(
    dag: AegisResidualDag,
) -> dict[str, AegisBoundedSlackTicket]:
    """Build deterministic Slack tickets or fail the whole witness closed."""

    node_by_id, predecessors, successors = _graph_maps(dag)
    order = _topological_order(dag)
    sources = frozenset(
        node_id for node_id in order if not predecessors[node_id]
    )
    sinks = frozenset(node_id for node_id in order if not successors[node_id])
    unknown_reason = _unknown_reason(dag)
    service_versions = tuple(
        sorted(
            {
                node.duration.service_envelope_version
                for node in dag.nodes
                if node.duration is not None
            }
        )
    )
    if unknown_reason is not None:
        return {
            node_id: AegisBoundedSlackTicket(
                workflow_id=dag.workflow_id,
                node_id=node_id,
                agent_id=node.agent_id,
                need_state=node.need_state,
                slack_kind=AegisSlackKind.UNKNOWN,
                slack_ms=0.0,
                observed_at_ms=dag.observed_at_ms,
                deadline_at_ms=dag.observed_at_ms,
                lower_avoiding_ms=None,
                upper_through_ms=None,
                graph_version=dag.graph_version,
                proof_scope_kind=dag.proof_scope_kind,
                proof_scope_complete=dag.proof_scope_complete,
                witness_contract_id=dag.witness_contract_id,
                witness_contract_valid_until_ms=(
                    dag.witness_contract_valid_until_ms
                ),
                completion_barrier_id=dag.completion_barrier_id,
                service_envelope_versions=service_versions,
                source=unknown_reason,
                usable_slack=False,
                witness_sha256=_witness_sha256(dag, node_id),
            )
            for node_id, node in sorted(node_by_id.items())
        }

    lower = {
        node_id: node.duration.lower_ms  # type: ignore[union-attr]
        for node_id, node in node_by_id.items()
    }
    upper = {
        node_id: node.duration.upper_ms  # type: ignore[union-attr]
        for node_id, node in node_by_id.items()
    }
    prefix_upper: dict[str, float] = {}
    for node_id in order:
        prefix_upper[node_id] = upper[node_id] + max(
            (prefix_upper[parent] for parent in predecessors[node_id]),
            default=0.0,
        )
    suffix_upper: dict[str, float] = {}
    for node_id in reversed(order):
        suffix_upper[node_id] = upper[node_id] + max(
            (suffix_upper[child] for child in successors[node_id]),
            default=0.0,
        )

    tickets: dict[str, AegisBoundedSlackTicket] = {}
    for node_id in order:
        avoiding = _longest_lower_path_avoiding(
            excluded=node_id,
            order=order,
            sources=sources,
            sinks=sinks,
            predecessors=predecessors,
            lower=lower,
        )
        lower_avoiding = 0.0 if avoiding is None else avoiding
        upper_through = (
            prefix_upper[node_id]
            + suffix_upper[node_id]
            - upper[node_id]
        )
        slack_ms = max(lower_avoiding - upper_through, 0.0)
        node = node_by_id[node_id]
        tickets[node_id] = AegisBoundedSlackTicket(
            workflow_id=dag.workflow_id,
            node_id=node_id,
            agent_id=node.agent_id,
            need_state=node.need_state,
            slack_kind=AegisSlackKind.BOUNDED,
            slack_ms=round(slack_ms, 6),
            observed_at_ms=dag.observed_at_ms,
            deadline_at_ms=round(dag.observed_at_ms + slack_ms, 6),
            lower_avoiding_ms=round(lower_avoiding, 6),
            upper_through_ms=round(upper_through, 6),
            graph_version=dag.graph_version,
            proof_scope_kind=dag.proof_scope_kind,
            proof_scope_complete=dag.proof_scope_complete,
            witness_contract_id=dag.witness_contract_id,
            witness_contract_valid_until_ms=(
                dag.witness_contract_valid_until_ms
            ),
            completion_barrier_id=dag.completion_barrier_id,
            service_envelope_versions=service_versions,
            source="declared_residual_dag_hard_bounds",
            usable_slack=True,
            witness_sha256=_witness_sha256(dag, node_id),
        )
    return tickets
