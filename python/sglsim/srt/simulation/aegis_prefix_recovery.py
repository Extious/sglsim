"""Serial shortest-path reference for composing AegisKV recovery tiers.

For one failure scenario and one recovery job, vertex ``k`` means that the
target has reconstructed a correct continuous KV prefix through group ``k``.
An edge ``u -> v`` is one validated *serial* operation that extends that
prefix: adopt peer HBM, copy a DRAM/SSD range, or recompute known tokens.
Edges always move
forward, so the minimum modeled serial recovery service is a DAG shortest
path.  This operation grammar intentionally does not model token/layer/GPU
overlap or cross-request pipelines.  Production Algorithm 2 treats it as one
Recovery Planner implementation, not as a general physical-restoration
optimum.

Only group-committed and failure-visible state may be represented by an
action-gated edge.  ``enabled_action_ids`` therefore means committed actions,
not in-flight intentions.  This DP runs after the failure snapshot: peer state
has a recovery pin and Store state a read lease.  A time-bounded access lease
is revalidated at edge dispatch and must cover that edge's service bound.
Normal-time protection expiry is a separate portfolio-horizon check.  This
module composes already-validated edges; it does not replace group commit,
FreezeAndPin/read leases, layout checks, or service-envelope enforcement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class PrefixRecoveryKind(str, Enum):
    PEER_ADOPT = "PEER_ADOPT"
    PEER_D2D = "PEER_D2D"
    STORE_DRAM = "STORE_DRAM"
    STORE_SSD = "STORE_SSD"
    RECOMPUTE = "RECOMPUTE"


def _finite_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


@dataclass(frozen=True)
class PrefixRecoveryEdge:
    edge_id: str
    start_group: int
    end_group: int
    service_upper_s: float
    kind: PrefixRecoveryKind
    action_id: str | None = None
    access_valid_for_s: float | None = None

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise ValueError("edge_id must be nonempty")
        if isinstance(self.start_group, bool) or not isinstance(
            self.start_group, int
        ):
            raise ValueError("start_group must be an integer")
        if isinstance(self.end_group, bool) or not isinstance(
            self.end_group, int
        ):
            raise ValueError("end_group must be an integer")
        if self.start_group < 0 or self.end_group <= self.start_group:
            raise ValueError("recovery edge must advance the prefix")
        _finite_nonnegative("service_upper_s", self.service_upper_s)
        try:
            normalized_kind = PrefixRecoveryKind(self.kind)
        except ValueError as exc:
            raise ValueError("unsupported recovery edge kind") from exc
        object.__setattr__(self, "kind", normalized_kind)
        if self.action_id is not None and not self.action_id.strip():
            raise ValueError("action_id must be nonempty when present")
        if self.access_valid_for_s is not None:
            access_valid_for = _finite_nonnegative(
                "access_valid_for_s", self.access_valid_for_s
            )
            if access_valid_for + 1e-12 < self.service_upper_s:
                raise ValueError(
                    "access lease must cover the edge service upper bound"
                )


@dataclass(frozen=True)
class PrefixRecoveryStep:
    edge_id: str
    kind: PrefixRecoveryKind
    start_group: int
    end_group: int
    starts_after_service_s: float
    completes_after_service_s: float


@dataclass(frozen=True)
class PrefixRecoveryPlan:
    total_service_upper_s: float
    steps: tuple[PrefixRecoveryStep, ...]


@dataclass(frozen=True)
class PrefixRecoveryProblem:
    terminal_group: int
    edges: tuple[PrefixRecoveryEdge, ...]
    resume_barrier_upper_s: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.terminal_group, bool) or not isinstance(
            self.terminal_group, int
        ):
            raise ValueError("terminal_group must be an integer")
        if self.terminal_group <= 0:
            raise ValueError("terminal_group must be positive")
        if not self.edges:
            raise ValueError("prefix recovery problem needs at least one edge")
        _finite_nonnegative(
            "resume_barrier_upper_s", self.resume_barrier_upper_s
        )
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("recovery edge IDs must be unique")
        for edge in self.edges:
            if edge.end_group > self.terminal_group:
                raise ValueError("recovery edge exceeds terminal_group")

    @property
    def action_ids(self) -> frozenset[str]:
        return frozenset(
            edge.action_id for edge in self.edges if edge.action_id is not None
        )

    def solve(
        self,
        *,
        enabled_action_ids: frozenset[str] = frozenset(),
    ) -> PrefixRecoveryPlan:
        """Return the deterministic minimum-upper-bound prefix plan.

        The DP key is ``(service, step_count, edge_ids)``.  Thus equal-service
        plans prefer fewer operations and then stable edge IDs.  Edge access
        leases have already been validated for at least one bounded operation;
        protection lifetime before the failure snapshot is not part of this
        service DP.
        """

        unknown_actions = enabled_action_ids - self.action_ids
        if unknown_actions:
            raise ValueError(
                f"unknown enabled actions: {sorted(unknown_actions)}"
            )
        outgoing: dict[int, list[PrefixRecoveryEdge]] = {
            group: [] for group in range(self.terminal_group)
        }
        for edge in self.edges:
            if edge.action_id is None or edge.action_id in enabled_action_ids:
                outgoing[edge.start_group].append(edge)
        for values in outgoing.values():
            values.sort(key=lambda edge: edge.edge_id)

        # group -> (service from failure, edge-id path, edge path)
        best: dict[
            int,
            tuple[float, tuple[str, ...], tuple[PrefixRecoveryEdge, ...]],
        ] = {0: (0.0, (), ())}
        for group in range(self.terminal_group):
            current = best.get(group)
            if current is None:
                continue
            service_so_far, path_ids, path_edges = current
            for edge in outgoing[group]:
                candidate_service = service_so_far + edge.service_upper_s
                candidate_ids = path_ids + (edge.edge_id,)
                candidate_edges = path_edges + (edge,)
                incumbent = best.get(edge.end_group)
                candidate_key = (
                    candidate_service,
                    len(candidate_ids),
                    candidate_ids,
                )
                incumbent_key = (
                    None
                    if incumbent is None
                    else (incumbent[0], len(incumbent[1]), incumbent[1])
                )
                if incumbent_key is None or candidate_key < incumbent_key:
                    best[edge.end_group] = (
                        candidate_service,
                        candidate_ids,
                        candidate_edges,
                    )

        terminal = best.get(self.terminal_group)
        if terminal is None:
            raise ValueError("no valid continuous recovery path")
        service_without_barrier, _path_ids, path_edges = terminal
        steps: list[PrefixRecoveryStep] = []
        elapsed = 0.0
        for edge in path_edges:
            start = elapsed
            elapsed += edge.service_upper_s
            steps.append(
                PrefixRecoveryStep(
                    edge_id=edge.edge_id,
                    kind=edge.kind,
                    start_group=edge.start_group,
                    end_group=edge.end_group,
                    starts_after_service_s=start,
                    completes_after_service_s=elapsed,
                )
            )
        total = service_without_barrier + self.resume_barrier_upper_s
        return PrefixRecoveryPlan(
            total_service_upper_s=total,
            steps=tuple(steps),
        )
