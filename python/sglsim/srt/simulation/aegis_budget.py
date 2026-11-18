"""Versioned DP policy budgets and node-shared Host capacity arbitration.

Policy ownership remains at the DP replica.  The node arbiter is deliberately
mechanical: it enforces the published per-DP beta limit and the physical node
capacity, but it does not run an independent heuristic or rewrite beta.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

from sglsim.srt.managers.pipeline_parallel import DPBudgetSnapshot


@dataclass(frozen=True)
class TierWindowState:
    """Unambiguous stock/flow/outcome metrics for one DP policy window."""

    dp_rank: int
    window_id: int
    budget_version: int
    unique_offered: int
    attempted_by_tier: tuple[tuple[str, int], ...]
    admitted_by_tier: tuple[tuple[str, int], ...]
    resident_start_by_tier: tuple[tuple[str, int], ...]
    resident_end_by_tier: tuple[tuple[str, int], ...]
    released_by_tier: tuple[tuple[str, int], ...]
    evicted_by_tier: tuple[tuple[str, int], ...]
    unmet_tokens_by_tier: tuple[tuple[str, int], ...]
    unmet_utility_by_tier: tuple[tuple[str, float], ...]
    normal_pressure: float
    recovery_p99_s: float
    slo_miss_rate: float

    def __post_init__(self) -> None:
        for name in (
            "dp_rank",
            "window_id",
            "budget_version",
            "unique_offered",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for field_name in (
            "attempted_by_tier",
            "admitted_by_tier",
            "resident_start_by_tier",
            "resident_end_by_tier",
            "released_by_tier",
            "evicted_by_tier",
            "unmet_tokens_by_tier",
        ):
            values = tuple(getattr(self, field_name))
            if len(dict(values)) != len(values):
                raise ValueError(f"{field_name} contains duplicate tiers")
            if any(int(value) < 0 for _tier, value in values):
                raise ValueError(f"{field_name} must be nonnegative")
        if len(dict(self.unmet_utility_by_tier)) != len(
            self.unmet_utility_by_tier
        ):
            raise ValueError("unmet_utility_by_tier contains duplicate tiers")
        if any(value < 0.0 for _tier, value in self.unmet_utility_by_tier):
            raise ValueError("unmet utility must be nonnegative")
        if not 0.0 <= self.normal_pressure <= 1.0:
            raise ValueError("normal_pressure must be in [0, 1]")
        if self.recovery_p99_s < 0.0:
            raise ValueError("recovery_p99_s must be nonnegative")
        if not 0.0 <= self.slo_miss_rate <= 1.0:
            raise ValueError("slo_miss_rate must be in [0, 1]")


class DPBudgetPublisher:
    """Publish one atomic alpha/beta version to every PP stage of a DP."""

    def __init__(self, *, dp_rank: int, pp_size: int) -> None:
        if dp_rank < 0 or pp_size <= 0:
            raise ValueError("dp_rank must be nonnegative and pp_size positive")
        self.dp_rank = int(dp_rank)
        self.pp_size = int(pp_size)
        self._current: DPBudgetSnapshot | None = None
        self._acknowledged_pp_ranks: set[int] = set()
        self._lock = threading.RLock()

    @property
    def current(self) -> DPBudgetSnapshot | None:
        with self._lock:
            return self._current

    def publish(self, snapshot: DPBudgetSnapshot) -> None:
        with self._lock:
            if snapshot.dp_rank != self.dp_rank:
                raise ValueError("budget snapshot belongs to another DP")
            if snapshot.pp_size != self.pp_size:
                raise ValueError("budget snapshot PP size does not match publisher")
            if self._current is not None:
                if not self.fully_acknowledged:
                    raise RuntimeError(
                        "cannot publish a new budget before every PP stage "
                        "acknowledges the current version"
                    )
                if snapshot.budget_version != self._current.budget_version + 1:
                    raise ValueError("budget versions must increase by exactly one")
                if snapshot.window_id <= self._current.window_id:
                    raise ValueError("budget window_id must increase")
                if (
                    snapshot.valid_from_logical_time
                    < self._current.valid_from_logical_time
                ):
                    raise ValueError("budget logical time cannot move backwards")
            self._current = snapshot
            self._acknowledged_pp_ranks.clear()

    def view_for_stage(self, pp_rank: int) -> DPBudgetSnapshot:
        with self._lock:
            self._validate_pp_rank(pp_rank)
            if self._current is None:
                raise RuntimeError("no budget has been published")
            return self._current

    def acknowledge(self, *, pp_rank: int, budget_version: int) -> bool:
        with self._lock:
            self._validate_pp_rank(pp_rank)
            if self._current is None:
                raise RuntimeError("no budget has been published")
            if budget_version != self._current.budget_version:
                raise ValueError("stage acknowledged a stale or future budget version")
            self._acknowledged_pp_ranks.add(pp_rank)
            return self.fully_acknowledged

    @property
    def fully_acknowledged(self) -> bool:
        with self._lock:
            return self._acknowledged_pp_ranks == set(range(self.pp_size))

    @property
    def acknowledged_pp_ranks(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._acknowledged_pp_ranks))

    def reset_acknowledgements(self) -> None:
        """Require the current version to be replayed after worker restart."""

        with self._lock:
            if self._current is None:
                raise RuntimeError("no budget has been published")
            self._acknowledged_pp_ranks.clear()

    def _validate_pp_rank(self, pp_rank: int) -> None:
        if not 0 <= pp_rank < self.pp_size:
            raise ValueError(
                f"pp_rank must be in [0, {self.pp_size}), got {pp_rank}"
            )


@dataclass(frozen=True)
class HostAdmissionRequest:
    lease_key: str
    request_id: str
    workflow_id: str
    chunk_id: int
    dp_rank: int
    pp_rank: int
    budget_version: int
    token_count: int
    utility: float

    def __post_init__(self) -> None:
        if not self.lease_key or not self.request_id or not self.workflow_id:
            raise ValueError("lease/request/workflow identifiers must be nonempty")
        for name in ("chunk_id", "dp_rank", "pp_rank"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.budget_version <= 0:
            raise ValueError("budget_version must be positive")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.utility < 0.0:
            raise ValueError("utility must be nonnegative")


@dataclass(frozen=True)
class HostLease:
    lease_version: int
    admitted_sequence: int
    request: HostAdmissionRequest


@dataclass(frozen=True)
class HostAdmissionResult:
    admitted: bool
    reason: str
    lease: HostLease | None
    dp_resident_tokens: int
    dp_policy_limit_tokens: int
    node_resident_tokens: int
    node_physical_capacity_tokens: int
    unmet_tokens: int
    unmet_utility: float


class NodeHostCapacityArbiter:
    """Enforce DP beta and node physical Host capacity without policy choice."""

    def __init__(
        self,
        *,
        node_id: int,
        physical_capacity_tokens: int,
        allowed_dp_ranks: tuple[int, ...],
    ) -> None:
        if node_id < 0 or physical_capacity_tokens < 0:
            raise ValueError("node_id/capacity must be nonnegative")
        if not allowed_dp_ranks or any(dp_rank < 0 for dp_rank in allowed_dp_ranks):
            raise ValueError("allowed_dp_ranks must be nonempty and nonnegative")
        if len(set(allowed_dp_ranks)) != len(allowed_dp_ranks):
            raise ValueError("allowed_dp_ranks must be unique")
        self.node_id = int(node_id)
        self.physical_capacity_tokens = int(physical_capacity_tokens)
        self.allowed_dp_ranks = tuple(sorted(allowed_dp_ranks))
        self._budgets: dict[int, DPBudgetSnapshot] = {}
        self._leases: dict[str, HostLease] = {}
        self._next_lease_version = 1
        self._admitted_sequence = 0
        self._lock = threading.RLock()

    def publish_dp_budget(self, snapshot: DPBudgetSnapshot) -> None:
        with self._lock:
            if snapshot.dp_rank not in self.allowed_dp_ranks:
                raise ValueError("budget DP is not mapped to this node arbiter")
            current = self._budgets.get(snapshot.dp_rank)
            if current is not None and snapshot.budget_version <= current.budget_version:
                raise ValueError("node arbiter requires a newer DP budget version")
            resident = self.dp_resident_tokens(snapshot.dp_rank)
            if resident > snapshot.host_policy_limit_tokens:
                raise RuntimeError(
                    "new beta would be lower than current DP Host residency; "
                    "release/evict leases before publication"
                )
            self._budgets[snapshot.dp_rank] = snapshot

    def admit(self, request: HostAdmissionRequest) -> HostAdmissionResult:
        with self._lock:
            if request.dp_rank not in self.allowed_dp_ranks:
                raise ValueError("admission DP is not mapped to this node")
            if request.lease_key in self._leases:
                raise ValueError(f"duplicate Host lease key: {request.lease_key}")
            budget = self._budgets.get(request.dp_rank)
            if budget is None:
                raise RuntimeError("no DP budget has been published to node arbiter")
            if request.budget_version != budget.budget_version:
                raise ValueError("Host admission carries a stale or future budget")
            if request.pp_rank >= budget.pp_size:
                raise ValueError("Host admission PP rank is outside budget snapshot")
            dp_resident = self.dp_resident_tokens(request.dp_rank)
            node_resident = self.resident_tokens
            dp_after = dp_resident + request.token_count
            node_after = node_resident + request.token_count
            if dp_after > budget.host_policy_limit_tokens:
                return self._denied_result(
                    reason="dp_policy_budget",
                    request=request,
                    budget=budget,
                    dp_resident=dp_resident,
                    node_resident=node_resident,
                )
            if node_after > self.physical_capacity_tokens:
                return self._denied_result(
                    reason="node_physical_capacity",
                    request=request,
                    budget=budget,
                    dp_resident=dp_resident,
                    node_resident=node_resident,
                )
            lease = HostLease(
                lease_version=self._next_lease_version,
                admitted_sequence=self._admitted_sequence,
                request=request,
            )
            self._next_lease_version += 1
            self._admitted_sequence += 1
            self._leases[request.lease_key] = lease
            return HostAdmissionResult(
                admitted=True,
                reason="admitted",
                lease=lease,
                dp_resident_tokens=dp_after,
                dp_policy_limit_tokens=budget.host_policy_limit_tokens,
                node_resident_tokens=node_after,
                node_physical_capacity_tokens=self.physical_capacity_tokens,
                unmet_tokens=0,
                unmet_utility=0.0,
            )

    def release(
        self, lease_key: str, *, expected_lease_version: int | None = None
    ) -> HostLease | None:
        with self._lock:
            lease = self._leases.get(lease_key)
            if lease is None:
                return None
            if (
                expected_lease_version is not None
                and lease.lease_version != expected_lease_version
            ):
                raise ValueError("Host lease version does not match release guard")
            return self._leases.pop(lease_key)

    @property
    def resident_tokens(self) -> int:
        with self._lock:
            return sum(lease.request.token_count for lease in self._leases.values())

    def dp_resident_tokens(self, dp_rank: int) -> int:
        with self._lock:
            return sum(
                lease.request.token_count
                for lease in self._leases.values()
                if lease.request.dp_rank == dp_rank
            )

    def snapshot(self) -> dict:
        with self._lock:
            by_dp = {
                dp_rank: self.dp_resident_tokens(dp_rank)
                for dp_rank in self.allowed_dp_ranks
            }
            return {
                "node_id": self.node_id,
                "physical_capacity_tokens": self.physical_capacity_tokens,
                "resident_tokens": self.resident_tokens,
                "resident_tokens_by_dp": by_dp,
                "budgets": {
                    dp_rank: asdict(snapshot)
                    for dp_rank, snapshot in sorted(self._budgets.items())
                },
                "leases": {
                    key: asdict(lease)
                    for key, lease in sorted(self._leases.items())
                },
            }

    def _denied_result(
        self,
        *,
        reason: str,
        request: HostAdmissionRequest,
        budget: DPBudgetSnapshot,
        dp_resident: int,
        node_resident: int,
    ) -> HostAdmissionResult:
        return HostAdmissionResult(
            admitted=False,
            reason=reason,
            lease=None,
            dp_resident_tokens=dp_resident,
            dp_policy_limit_tokens=budget.host_policy_limit_tokens,
            node_resident_tokens=node_resident,
            node_physical_capacity_tokens=self.physical_capacity_tokens,
            unmet_tokens=request.token_count,
            unmet_utility=request.utility,
        )
