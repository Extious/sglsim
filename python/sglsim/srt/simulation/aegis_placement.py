"""Topology-aware recovery paths and node-shared Host protection for AegisKV.

This module is simulator-only metadata.  It does not replace SGLang scheduling
or move real tensors.  Its purpose is to make Algorithm 2's physical contract
explicit before latency is charged:

* a GPU copy is reusable only when retry is routed to that exact DP replica;
* a Host copy records the node that owns the shared pool and therefore whether
  recovery is H2D or H2H + H2D;
* a candidate must survive the declared failure scope before it can receive
  positive recovery benefit;
* Host protection is a lease in one process-external node pool.  Existing
  cache payload can be promoted to a protection lease without another copy,
  while pinned bytes remain under a separate bounded quota.

The Host-pool rules mirror the useful control properties of Mooncake Store's
allocator and pinned-buffer managers: one bounded allocator, RAII-like leases,
reuse before allocation, a bounded pinned quota, and pageable fallback.  The
payload remains metadata-only in SGLSim.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping


FAILURE_SCOPES = {"gpu", "process", "node", "rack"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not math.isfinite(parsed) else parsed


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def normalize_failure_scope(value: Any) -> str:
    scope = str(value or "node").strip().lower().replace("-", "_")
    aliases = {
        "gpu_only": "gpu",
        "scheduler": "process",
        "worker": "process",
        "host": "node",
        "machine": "node",
    }
    scope = aliases.get(scope, scope)
    return scope if scope in FAILURE_SCOPES else "node"


@dataclass(frozen=True)
class AegisClusterTopology:
    """DP-replica placement and the PP GPUs contained in each replica."""

    dp_rank_to_node: tuple[int, ...]
    pp_stages_per_replica: int = 1
    node_to_rack: tuple[int, ...] = ()
    gpu_backup_requires_direct_retry_route: bool = True
    host_pool_scope: str = "node_shared_process_external"

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        dp_size: int,
        pp_size: int,
    ) -> "AegisClusterTopology":
        values = dict(raw or {})
        mapping_raw = values.get("dp_rank_to_node", tuple(range(dp_size)))
        mapping = tuple(_as_int(value, -1) for value in mapping_raw)
        if len(mapping) != int(dp_size) or any(node < 0 for node in mapping):
            raise ValueError(
                "topology.dp_rank_to_node must contain one non-negative node "
                f"id per DP rank (dp_size={dp_size})"
            )
        max_node = max(mapping, default=-1)
        racks_raw = values.get("node_to_rack")
        if racks_raw is None:
            racks = tuple(range(max_node + 1))
        elif isinstance(racks_raw, Mapping):
            racks = tuple(
                _as_int(racks_raw.get(str(node), racks_raw.get(node, node)), node)
                for node in range(max_node + 1)
            )
        else:
            racks = tuple(_as_int(value, -1) for value in racks_raw)
            if len(racks) <= max_node or any(rack < 0 for rack in racks):
                raise ValueError(
                    "topology.node_to_rack must cover every referenced node"
                )
        stages = max(
            _as_int(
                values.get("pp_stages_per_node", values.get("pp_stages_per_replica")),
                pp_size,
            ),
            1,
        )
        if stages != int(pp_size):
            raise ValueError(
                "topology PP stage count must match the deployment pp_size "
                f"({stages} != {pp_size})"
            )
        return cls(
            dp_rank_to_node=mapping,
            pp_stages_per_replica=stages,
            node_to_rack=racks,
            gpu_backup_requires_direct_retry_route=bool(
                values.get("gpu_backup_requires_direct_retry_route", True)
            ),
            host_pool_scope=str(
                values.get("host_pool_scope", "node_shared_process_external")
            ),
        )

    @property
    def dp_size(self) -> int:
        return len(self.dp_rank_to_node)

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.dp_rank_to_node)))

    def node_for_rank(self, dp_rank: int) -> int:
        if not 0 <= int(dp_rank) < self.dp_size:
            raise ValueError(f"invalid DP rank {dp_rank}")
        return self.dp_rank_to_node[int(dp_rank)]

    def rack_for_node(self, node_id: int) -> int:
        if not 0 <= int(node_id) < len(self.node_to_rack):
            raise ValueError(f"invalid node id {node_id}")
        return self.node_to_rack[int(node_id)]

    def copy_survives(
        self,
        *,
        source_dp_rank: int,
        copy_dp_rank: int | None = None,
        copy_node_id: int | None = None,
        failure_scope: str,
    ) -> bool:
        scope = normalize_failure_scope(failure_scope)
        source_node = self.node_for_rank(source_dp_rank)
        if copy_node_id is None:
            if copy_dp_rank is None:
                raise ValueError("copy rank or copy node is required")
            copy_node_id = self.node_for_rank(copy_dp_rank)
        if scope in {"gpu", "process"}:
            return copy_dp_rank is None or int(copy_dp_rank) != int(source_dp_rank)
        if scope == "node":
            return int(copy_node_id) != source_node
        return self.rack_for_node(int(copy_node_id)) != self.rack_for_node(source_node)

    def pp_gpu_mapping(
        self, *, source_dp_rank: int, target_dp_rank: int
    ) -> tuple[dict[str, Any], ...]:
        source_node = self.node_for_rank(source_dp_rank)
        target_node = self.node_for_rank(target_dp_rank)
        return tuple(
            {
                "pp_rank": pp_rank,
                "source_gpu": f"node{source_node}/gpu{pp_rank}",
                "target_gpu": f"node{target_node}/gpu{pp_rank}",
            }
            for pp_rank in range(self.pp_stages_per_replica)
        )


@dataclass(frozen=True)
class AegisPathCostModel:
    """Calibratable path components; all coefficients are milliseconds."""

    gpu_write_setup_ms: float = 0.25
    gpu_write_ms_per_token: float = 0.0019
    gpu_direct_reuse_setup_ms: float = 0.20
    d2h_setup_ms: float = 0.30
    d2h_ms_per_token: float = 0.0012
    h2h_setup_ms: float = 0.60
    h2h_ms_per_token: float = 0.0036
    h2d_setup_ms: float = 2.40
    h2d_ms_per_token: float = 0.0019
    l3_write_setup_ms: float = 1.20
    l3_write_ms_per_token: float = 0.0060
    l3_read_setup_ms: float = 8.00
    l3_read_ms_per_token: float = 0.0070
    recompute_setup_ms: float = 4.00
    recompute_ms_per_token: float = 0.0108
    # Historical/configured probability that the protected failure domain
    # fails while this frontier is useful.  Keep 1.0 as the compatibility
    # default; formal profiles must calibrate it without looking ahead at the
    # replay manifest.
    failure_hazard_probability: float = 1.0
    normal_path_cost_weight: float = 0.05
    pressure_cost_weight: float = 0.50

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "AegisPathCostModel":
        values = dict(raw or {})
        aliases = {
            "gpu_write_setup_ms": "l1_write_setup_ms",
            "gpu_write_ms_per_token": "l1_write_ms_per_token",
            "gpu_direct_reuse_setup_ms": "l1_restore_setup_ms",
            "h2h_setup_ms": "l2_write_setup_ms",
            "h2h_ms_per_token": "l2_write_ms_per_token",
            "h2d_setup_ms": "l2_restore_setup_ms",
            "h2d_ms_per_token": "l2_restore_ms_per_token",
            "normal_path_cost_weight": "cost_weight",
        }
        kwargs: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            source = name if name in values else aliases.get(name)
            if source is not None and source in values:
                kwargs[name] = _as_float(values[source])
        return cls(**kwargs)

    @staticmethod
    def _component(setup_ms: float, per_token_ms: float, tokens: int) -> float:
        return max(float(setup_ms), 0.0) + max(int(tokens), 0) * max(
            float(per_token_ms), 0.0
        )

    def recompute_ms(self, tokens: int) -> float:
        return self._component(
            self.recompute_setup_ms, self.recompute_ms_per_token, tokens
        )

    def path_ms(self, path: Iterable[str], tokens: int) -> float:
        total = 0.0
        for component in path:
            if component == "gpu_rdma":
                total += self._component(
                    self.gpu_write_setup_ms,
                    self.gpu_write_ms_per_token,
                    tokens,
                )
            elif component == "pcie_p2p":
                total += self._component(
                    self.gpu_write_setup_ms,
                    self.gpu_write_ms_per_token,
                    tokens,
                )
            elif component == "direct_hbm_reuse":
                total += max(self.gpu_direct_reuse_setup_ms, 0.0)
            elif component == "d2h":
                total += self._component(
                    self.d2h_setup_ms, self.d2h_ms_per_token, tokens
                )
            elif component == "h2h":
                total += self._component(
                    self.h2h_setup_ms, self.h2h_ms_per_token, tokens
                )
            elif component == "h2d":
                total += self._component(
                    self.h2d_setup_ms, self.h2d_ms_per_token, tokens
                )
            elif component == "l3_write":
                total += self._component(
                    self.l3_write_setup_ms,
                    self.l3_write_ms_per_token,
                    tokens,
                )
            elif component == "l3_read":
                total += self._component(
                    self.l3_read_setup_ms,
                    self.l3_read_ms_per_token,
                    tokens,
                )
            else:
                raise ValueError(f"unknown recovery-path component: {component}")
        return total


@dataclass(frozen=True)
class AegisPlacementRequest:
    rid: str
    token_count: int
    source_dp_rank: int
    critical_probability: float
    slack_lcb_ms: float
    failure_scope: str = "node"


@dataclass(frozen=True)
class AegisRecoveryCandidate:
    tier: str
    valid: bool
    invalid_reason: str
    source_dp_rank: int
    target_dp_rank: int
    source_node_id: int
    target_node_id: int
    host_pool_node_id: int
    failure_scope: str
    failure_scope_survives: bool
    direct_retry_route_required: bool
    reusable_dp_ranks: tuple[int, ...]
    pp_gpu_mapping: tuple[dict[str, Any], ...]
    backup_path: tuple[str, ...]
    recovery_path: tuple[str, ...]
    token_count: int
    write_ms: float
    recovery_ms: float
    recompute_ms: float
    deadline_benefit_ms: float
    utility: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AegisTopologyAwarePlanner:
    """Evaluate Algorithm 2 candidates without assuming every copy is reusable."""

    def __init__(
        self, *, topology: AegisClusterTopology, costs: AegisPathCostModel
    ) -> None:
        self.topology = topology
        self.costs = costs

    def _score(
        self,
        *,
        request: AegisPlacementRequest,
        write_ms: float,
        recovery_ms: float,
        capacity_pressure: float,
        path_pressure: float,
    ) -> tuple[float, float, float]:
        recompute_ms = self.costs.recompute_ms(request.token_count)
        slack = max(float(request.slack_lcb_ms), 0.0)
        recompute_overrun = max(recompute_ms - slack, 0.0)
        recovery_overrun = max(recovery_ms - slack, 0.0)
        deadline_benefit = (
            _clamp(self.costs.failure_hazard_probability)
            * _clamp(request.critical_probability)
            * max(recompute_overrun - recovery_overrun, 0.0)
        )
        pressure = _clamp(capacity_pressure) + _clamp(path_pressure)
        normal_cost = max(write_ms, 0.0) * (
            max(self.costs.normal_path_cost_weight, 0.0)
            + max(self.costs.pressure_cost_weight, 0.0) * pressure
        )
        return recompute_ms, deadline_benefit, deadline_benefit - normal_cost

    def gpu_candidate(
        self,
        request: AegisPlacementRequest,
        *,
        target_dp_rank: int,
        capacity_pressure: float = 0.0,
        path_pressure: float = 0.0,
    ) -> AegisRecoveryCandidate:
        source_node = self.topology.node_for_rank(request.source_dp_rank)
        target_node = self.topology.node_for_rank(target_dp_rank)
        survives = self.topology.copy_survives(
            source_dp_rank=request.source_dp_rank,
            copy_dp_rank=target_dp_rank,
            failure_scope=request.failure_scope,
        )
        distinct = int(target_dp_rank) != int(request.source_dp_rank)
        valid = distinct and survives
        reason = ""
        if not distinct:
            reason = "gpu copy cannot target the execution replica"
        elif not survives:
            reason = f"gpu copy does not survive {normalize_failure_scope(request.failure_scope)} failure"
        backup_path = ("pcie_p2p",) if source_node == target_node else ("gpu_rdma",)
        recovery_path = ("direct_hbm_reuse",)
        write_ms = self.costs.path_ms(backup_path, request.token_count)
        recovery_ms = self.costs.path_ms(recovery_path, request.token_count)
        recompute_ms, benefit, utility = self._score(
            request=request,
            write_ms=write_ms,
            recovery_ms=recovery_ms,
            capacity_pressure=capacity_pressure,
            path_pressure=path_pressure,
        )
        if not valid:
            benefit, utility = 0.0, float("-inf")
        return AegisRecoveryCandidate(
            tier="gpu",
            valid=valid,
            invalid_reason=reason,
            source_dp_rank=request.source_dp_rank,
            target_dp_rank=int(target_dp_rank),
            source_node_id=source_node,
            target_node_id=target_node,
            host_pool_node_id=-1,
            failure_scope=normalize_failure_scope(request.failure_scope),
            failure_scope_survives=survives,
            direct_retry_route_required=self.topology.gpu_backup_requires_direct_retry_route,
            reusable_dp_ranks=(int(target_dp_rank),),
            pp_gpu_mapping=self.topology.pp_gpu_mapping(
                source_dp_rank=request.source_dp_rank,
                target_dp_rank=target_dp_rank,
            ),
            backup_path=backup_path,
            recovery_path=recovery_path,
            token_count=max(int(request.token_count), 0),
            write_ms=round(write_ms, 6),
            recovery_ms=round(recovery_ms, 6),
            recompute_ms=round(recompute_ms, 6),
            deadline_benefit_ms=round(benefit, 6),
            utility=round(utility, 6),
        )

    def host_candidate(
        self,
        request: AegisPlacementRequest,
        *,
        host_pool_node_id: int,
        retry_dp_rank: int,
        capacity_pressure: float = 0.0,
        path_pressure: float = 0.0,
    ) -> AegisRecoveryCandidate:
        source_node = self.topology.node_for_rank(request.source_dp_rank)
        retry_node = self.topology.node_for_rank(retry_dp_rank)
        host_node = int(host_pool_node_id)
        survives = self.topology.copy_survives(
            source_dp_rank=request.source_dp_rank,
            copy_node_id=host_node,
            failure_scope=request.failure_scope,
        )
        backup_path = ("d2h",) if host_node == source_node else ("d2h", "h2h")
        recovery_path = ("h2d",) if retry_node == host_node else ("h2h", "h2d")
        write_ms = self.costs.path_ms(backup_path, request.token_count)
        recovery_ms = self.costs.path_ms(recovery_path, request.token_count)
        recompute_ms, benefit, utility = self._score(
            request=request,
            write_ms=write_ms,
            recovery_ms=recovery_ms,
            capacity_pressure=capacity_pressure,
            path_pressure=path_pressure,
        )
        if not survives:
            benefit, utility = 0.0, float("-inf")
        return AegisRecoveryCandidate(
            tier="host",
            valid=survives,
            invalid_reason=(
                ""
                if survives
                else f"Host pool does not survive {normalize_failure_scope(request.failure_scope)} failure"
            ),
            source_dp_rank=request.source_dp_rank,
            target_dp_rank=int(retry_dp_rank),
            source_node_id=source_node,
            target_node_id=retry_node,
            host_pool_node_id=host_node,
            failure_scope=normalize_failure_scope(request.failure_scope),
            failure_scope_survives=survives,
            direct_retry_route_required=False,
            reusable_dp_ranks=tuple(range(self.topology.dp_size)),
            pp_gpu_mapping=(),
            backup_path=backup_path,
            recovery_path=recovery_path,
            token_count=max(int(request.token_count), 0),
            write_ms=round(write_ms, 6),
            recovery_ms=round(recovery_ms, 6),
            recompute_ms=round(recompute_ms, 6),
            deadline_benefit_ms=round(benefit, 6),
            utility=round(utility, 6),
        )

    def l3_candidate(
        self,
        request: AegisPlacementRequest,
        *,
        retry_dp_rank: int,
        storage_available: bool,
        manifest_quorum: int,
        capacity_pressure: float = 0.0,
        path_pressure: float = 0.0,
    ) -> AegisRecoveryCandidate:
        """Evaluate cluster-external persistent storage with manifest quorum."""

        source_node = self.topology.node_for_rank(request.source_dp_rank)
        retry_node = self.topology.node_for_rank(retry_dp_rank)
        valid = bool(storage_available) and int(manifest_quorum) >= 1
        backup_path = ("d2h", "l3_write")
        recovery_path = ("l3_read", "h2d")
        write_ms = self.costs.path_ms(backup_path, request.token_count)
        recovery_ms = self.costs.path_ms(recovery_path, request.token_count)
        recompute_ms, benefit, utility = self._score(
            request=request,
            write_ms=write_ms,
            recovery_ms=recovery_ms,
            capacity_pressure=capacity_pressure,
            path_pressure=path_pressure,
        )
        if not valid:
            benefit, utility = 0.0, float("-inf")
        return AegisRecoveryCandidate(
            tier="l3",
            valid=valid,
            invalid_reason=(
                ""
                if valid
                else "persistent storage or manifest quorum is unavailable"
            ),
            source_dp_rank=request.source_dp_rank,
            target_dp_rank=int(retry_dp_rank),
            source_node_id=source_node,
            target_node_id=retry_node,
            host_pool_node_id=-1,
            failure_scope=normalize_failure_scope(request.failure_scope),
            failure_scope_survives=valid,
            direct_retry_route_required=False,
            reusable_dp_ranks=tuple(range(self.topology.dp_size)),
            pp_gpu_mapping=(),
            backup_path=backup_path,
            recovery_path=recovery_path,
            token_count=max(int(request.token_count), 0),
            write_ms=round(write_ms, 6),
            recovery_ms=round(recovery_ms, 6),
            recompute_ms=round(recompute_ms, 6),
            deadline_benefit_ms=round(benefit, 6),
            utility=round(utility, 6),
        )

    def recompute_candidate(
        self, request: AegisPlacementRequest, *, retry_dp_rank: int
    ) -> AegisRecoveryCandidate:
        source_node = self.topology.node_for_rank(request.source_dp_rank)
        retry_node = self.topology.node_for_rank(retry_dp_rank)
        recompute_ms = self.costs.recompute_ms(request.token_count)
        return AegisRecoveryCandidate(
            tier="recompute",
            valid=True,
            invalid_reason="",
            source_dp_rank=request.source_dp_rank,
            target_dp_rank=int(retry_dp_rank),
            source_node_id=source_node,
            target_node_id=retry_node,
            host_pool_node_id=-1,
            failure_scope=normalize_failure_scope(request.failure_scope),
            failure_scope_survives=True,
            direct_retry_route_required=False,
            reusable_dp_ranks=tuple(range(self.topology.dp_size)),
            pp_gpu_mapping=(),
            backup_path=(),
            recovery_path=("recompute",),
            token_count=max(int(request.token_count), 0),
            write_ms=0.0,
            recovery_ms=round(recompute_ms, 6),
            recompute_ms=round(recompute_ms, 6),
            deadline_benefit_ms=0.0,
            utility=0.0,
        )

    def choose(self, candidates: Iterable[AegisRecoveryCandidate]) -> AegisRecoveryCandidate:
        valid = [candidate for candidate in candidates if candidate.valid]
        if not valid:
            raise ValueError("Algorithm 2 has no valid candidate")
        return max(
            valid,
            key=lambda candidate: (
                candidate.utility,
                -candidate.recovery_ms,
                candidate.tier == "host",
            ),
        )


@dataclass(frozen=True)
class AegisHostLease:
    backup_key: str
    node_id: int
    token_count: int
    allocated_tokens: int
    internal_fragment_tokens: int
    pinned_tokens: int
    pageable_tokens: int
    cache_reused_tokens: int
    copied_tokens: int
    utility_density: float
    source_node_id: int
    failure_scope: str
    version: int
    created_time: float = field(default_factory=time.time)


@dataclass(frozen=True)
class AegisHostAdmission:
    admitted: bool
    reason: str
    lease: AegisHostLease | None
    evicted_backup_keys: tuple[str, ...] = ()
    evicted_leases: tuple[AegisHostLease, ...] = ()


class AegisNodeHostPool:
    """One node-scoped shared Host allocator with protected and pinned quotas."""

    def __init__(
        self,
        *,
        node_id: int,
        capacity_tokens: int,
        protection_ratio: float,
        pinned_quota_tokens: int,
        alignment_tokens: int = 1,
        max_regions: int | None = None,
        process_external_owner: str = "data_parallel_controller",
    ) -> None:
        self.node_id = int(node_id)
        self.capacity_tokens = max(int(capacity_tokens), 1)
        self.protection_ratio = _clamp(protection_ratio)
        self.pinned_quota_tokens = max(int(pinned_quota_tokens), 0)
        self.alignment_tokens = max(int(alignment_tokens), 1)
        self.max_regions = (
            max(int(max_regions), 1) if max_regions is not None else None
        )
        self.process_external_owner = str(process_external_owner)
        self._leases: dict[str, AegisHostLease] = {}
        self._lock = threading.RLock()

    @property
    def protected_budget_tokens(self) -> int:
        if self.protection_ratio <= 0.0:
            return 0
        return min(
            max(int(self.capacity_tokens * self.protection_ratio), 1),
            self.capacity_tokens,
        )

    @property
    def protected_used_tokens(self) -> int:
        return sum(lease.allocated_tokens for lease in self._leases.values())

    @property
    def pinned_used_tokens(self) -> int:
        return sum(lease.pinned_tokens for lease in self._leases.values())

    def allocation_tokens(self, token_count: int) -> int:
        """Return allocator occupancy for a logical payload token count."""

        tokens = max(int(token_count), 0)
        if tokens == 0:
            return 0
        return (
            (tokens + self.alignment_tokens - 1) // self.alignment_tokens
        ) * self.alignment_tokens

    def get(self, backup_key: str) -> AegisHostLease | None:
        with self._lock:
            return self._leases.get(str(backup_key))

    def release(
        self,
        backup_key: str,
        *,
        max_version: int | None = None,
    ) -> AegisHostLease | None:
        with self._lock:
            key = str(backup_key)
            lease = self._leases.get(key)
            if lease is None:
                return None
            if max_version is not None and lease.version > int(max_version):
                return None
            return self._leases.pop(key, None)

    def admit(
        self,
        *,
        backup_key: str,
        token_count: int,
        utility_density: float,
        source_node_id: int,
        failure_scope: str,
        cache_resident_tokens: int = 0,
        prefer_pinned: bool = True,
        version: int = 1,
    ) -> AegisHostAdmission:
        key = str(backup_key)
        tokens = max(int(token_count), 0)
        if not key or tokens <= 0:
            return AegisHostAdmission(False, "empty Host protection lease", None)
        allocated_tokens = self.allocation_tokens(tokens)
        with self._lock:
            previous = self._leases.get(key)
            requested_version = max(int(version), 1)
            if previous is not None and requested_version < previous.version:
                return AegisHostAdmission(
                    False,
                    "stale Host protection lease version",
                    None,
                )
            previous = self._leases.pop(key, None)
            available = self.protected_budget_tokens - self.protected_used_tokens
            token_deficit = max(allocated_tokens - available, 0)
            region_deficit = (
                max(len(self._leases) + 1 - self.max_regions, 0)
                if self.max_regions is not None
                else 0
            )
            victims = sorted(
                (
                    lease
                    for lease in self._leases.values()
                    if lease.utility_density < float(utility_density)
                ),
                key=lambda lease: (
                    lease.utility_density,
                    lease.version,
                    lease.backup_key,
                ),
            )
            evicted: list[str] = []
            evicted_leases: list[AegisHostLease] = []
            reclaimed = 0
            for victim in victims:
                if (
                    reclaimed >= token_deficit
                    and len(evicted_leases) >= region_deficit
                ):
                    break
                reclaimed += victim.allocated_tokens
                evicted.append(victim.backup_key)
                evicted_leases.append(victim)
            if (
                reclaimed < token_deficit
                or len(evicted_leases) < region_deficit
            ):
                if previous is not None:
                    self._leases[key] = previous
                reason = (
                    "node Host region limit exhausted"
                    if len(evicted_leases) < region_deficit
                    else "node Host protection budget exhausted"
                )
                return AegisHostAdmission(False, reason, None)
            for victim in evicted_leases:
                self._leases.pop(victim.backup_key, None)

            reusable = min(max(int(cache_resident_tokens), 0), tokens)
            pinned_available = max(
                self.pinned_quota_tokens - self.pinned_used_tokens, 0
            )
            pinned_available = (
                pinned_available // self.alignment_tokens
            ) * self.alignment_tokens
            pinned = (
                min(allocated_tokens, pinned_available) if prefer_pinned else 0
            )
            lease = AegisHostLease(
                backup_key=key,
                node_id=self.node_id,
                token_count=tokens,
                allocated_tokens=allocated_tokens,
                internal_fragment_tokens=allocated_tokens - tokens,
                pinned_tokens=pinned,
                pageable_tokens=allocated_tokens - pinned,
                cache_reused_tokens=reusable,
                copied_tokens=tokens - reusable,
                utility_density=float(utility_density),
                source_node_id=int(source_node_id),
                failure_scope=normalize_failure_scope(failure_scope),
                version=requested_version,
            )
            self._leases[key] = lease
            return AegisHostAdmission(
                True,
                "",
                lease,
                tuple(sorted(evicted)),
                tuple(sorted(evicted_leases, key=lambda item: item.backup_key)),
            )

    def update_protection_ratio(self, ratio: float) -> tuple[AegisHostLease, ...]:
        """Publish Algorithm 3's next-window Host ratio and shed low value leases."""

        with self._lock:
            self.protection_ratio = _clamp(ratio)
            evicted: list[AegisHostLease] = []
            while self.protected_used_tokens > self.protected_budget_tokens:
                victim = min(
                    self._leases.values(),
                    key=lambda lease: (
                        lease.utility_density,
                        lease.version,
                        lease.backup_key,
                    ),
                )
                evicted.append(victim)
                self._leases.pop(victim.backup_key, None)
            return tuple(sorted(evicted, key=lambda item: item.backup_key))

    def invalidate_for_node_failure(self, failed_node_id: int) -> tuple[str, ...]:
        """A node failure destroys that node's pool; GPU/process failure does not."""

        if int(failed_node_id) != self.node_id:
            return ()
        with self._lock:
            invalidated = tuple(sorted(self._leases))
            self._leases.clear()
            return invalidated

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.node_id,
                "scope": "node_shared_process_external",
                "owner": self.process_external_owner,
                "capacity_tokens": self.capacity_tokens,
                "alignment_tokens": self.alignment_tokens,
                "max_regions": self.max_regions,
                "protection_ratio": round(self.protection_ratio, 6),
                "protected_budget_tokens": self.protected_budget_tokens,
                "protected_used_tokens": self.protected_used_tokens,
                "pinned_quota_tokens": self.pinned_quota_tokens,
                "pinned_used_tokens": self.pinned_used_tokens,
                "pageable_used_tokens": sum(
                    lease.pageable_tokens for lease in self._leases.values()
                ),
                "internal_fragment_tokens": sum(
                    lease.internal_fragment_tokens
                    for lease in self._leases.values()
                ),
                "lease_count": len(self._leases),
            }


class AegisNodeHostBudgetController:
    """Adaptive node-scoped beta controller for the shared Host pool.

    Beta cannot be independently owned by every GPU because all schedulers on
    one node allocate from the same process-external Host pool.  This
    controller aggregates authoritative post-admission demand and closes a
    window on a bounded request count, a target wall-clock duration, or an
    overflow shock after the minimum sample count.
    """

    def __init__(
        self,
        *,
        node_id: int,
        capacity_tokens: int,
        initial_beta: float,
        window_min_requests: int = 4,
        window_max_requests: int = 32,
        window_target_ms: float = 5000.0,
        overflow_trigger_tokens: int = 512,
        demand_ewma: float = 0.35,
        publish_smoothing: float = 0.20,
        hysteresis: float = 0.01,
        safety_margin: float = 0.08,
        burst_reserve_tokens: int = 2048,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.node_id = int(node_id)
        self.capacity_tokens = max(int(capacity_tokens), 1)
        self.beta = _clamp(initial_beta)
        self.window_min_requests = max(int(window_min_requests), 1)
        self.window_max_requests = max(
            int(window_max_requests), self.window_min_requests
        )
        self.window_target_ms = max(float(window_target_ms), 0.0)
        self.overflow_trigger_tokens = max(int(overflow_trigger_tokens), 1)
        self.demand_ewma = _clamp(demand_ewma)
        self.publish_smoothing = _clamp(publish_smoothing)
        self.hysteresis = max(float(hysteresis), 0.0)
        self.safety_margin = _clamp(safety_margin)
        self.burst_reserve_tokens = max(int(burst_reserve_tokens), 0)
        self._now_fn = now_fn
        self.window_id = 0
        self._predicted_rate_tokens_per_s = 0.0
        self._reset_window(self._now_fn())

    def _reset_window(self, now: float) -> None:
        self._window_started = float(now)
        self._requests = 0
        self._demand_tokens = 0
        self._admitted_tokens = 0
        self._unmet_tokens = 0
        self._normal_pressure = 0.0

    def observe(
        self,
        *,
        demand_tokens: int,
        admitted_tokens: int,
        unmet_tokens: int,
        normal_pressure: float,
        protected_used_tokens: int,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        current = self._now_fn() if now is None else float(now)
        self._requests += 1
        self._demand_tokens += max(int(demand_tokens), 0)
        self._admitted_tokens += max(int(admitted_tokens), 0)
        self._unmet_tokens += max(int(unmet_tokens), 0)
        self._normal_pressure = max(
            self._normal_pressure, _clamp(normal_pressure)
        )
        elapsed_ms = max((current - self._window_started) * 1000.0, 0.0)
        trigger = ""
        if self._requests >= self.window_max_requests:
            trigger = "max_requests"
        elif self._requests >= self.window_min_requests:
            if self._unmet_tokens >= self.overflow_trigger_tokens:
                trigger = "overflow_shock"
            elif self.window_target_ms > 0.0 and elapsed_ms >= self.window_target_ms:
                trigger = "target_time"
        if not trigger:
            return None
        return self.close_window(
            protected_used_tokens=protected_used_tokens,
            now=current,
            trigger=trigger,
        )

    def close_window(
        self,
        *,
        protected_used_tokens: int,
        now: float | None = None,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        current = self._now_fn() if now is None else float(now)
        elapsed_s = max(current - self._window_started, 1e-6)
        required = max(
            self._demand_tokens,
            self._admitted_tokens + self._unmet_tokens,
        )
        observed_rate = required / elapsed_s
        mu = self.demand_ewma
        predicted_rate = (
            (1.0 - mu) * self._predicted_rate_tokens_per_s
            + mu * observed_rate
        )
        self._predicted_rate_tokens_per_s = predicted_rate
        target_s = (
            self.window_target_ms / 1000.0
            if self.window_target_ms > 0.0
            else elapsed_s
        )
        predicted_tokens = predicted_rate * target_s
        occupied_floor = max(int(protected_used_tokens), 0) / self.capacity_tokens
        safe_cap = _clamp(
            1.0 - self._normal_pressure - self.safety_margin
        )
        raw = min(
            safe_cap,
            _clamp(
                (predicted_tokens + self.burst_reserve_tokens)
                / self.capacity_tokens
            ),
        )
        old_beta = self.beta
        published = (
            (1.0 - self.publish_smoothing) * old_beta
            + self.publish_smoothing * raw
        )
        if abs(published - old_beta) < self.hysteresis:
            published = old_beta
        # Higher serving pressure must shrink Host protection share.
        if self._normal_pressure > 0.15:
            published = min(published, old_beta, safe_cap)
        # The pool can safely publish below current occupancy because the
        # controller immediately evicts the lowest-utility leases and revokes
        # their Router hints before the update becomes externally visible.
        self.beta = _clamp(published)
        self.window_id += 1
        update = {
            "scope": "node_shared_host_pool",
            "node_id": self.node_id,
            "window_id": self.window_id,
            "trigger": str(trigger),
            "elapsed_ms": round(elapsed_s * 1000.0, 3),
            "requests": self._requests,
            "D": self._demand_tokens,
            "U": self._admitted_tokens,
            "O": self._unmet_tokens,
            "normal_pressure": round(self._normal_pressure, 6),
            "observed_rate_tokens_per_s": round(observed_rate, 3),
            "predicted_rate_tokens_per_s": round(predicted_rate, 3),
            "predicted_next_window_tokens": round(predicted_tokens, 3),
            "occupied_floor": round(occupied_floor, 6),
            "safe_cap": round(safe_cap, 6),
            "raw_beta": round(raw, 6),
            "old_beta": round(old_beta, 6),
            "new_beta": round(self.beta, 6),
        }
        self._reset_window(current)
        return update

    def snapshot(self) -> dict[str, Any]:
        return {
            "scope": "node_shared_host_pool",
            "node_id": self.node_id,
            "window_id": self.window_id,
            "window_requests": self._requests,
            "beta": round(self.beta, 6),
            "D_host_tokens": self._demand_tokens,
            "U_host_tokens": self._admitted_tokens,
            "O_host_tokens": self._unmet_tokens,
            "predicted_rate_tokens_per_s": round(
                self._predicted_rate_tokens_per_s, 3
            ),
        }


@dataclass(frozen=True)
class AegisPersistentLease:
    backup_key: str
    token_count: int
    utility_density: float
    version: int
    manifest_version: int
    manifest_quorum: int
    committed: bool
    created_time: float = field(default_factory=time.time)


@dataclass(frozen=True)
class AegisPersistentAdmission:
    admitted: bool
    reason: str
    lease: AegisPersistentLease | None
    evicted_leases: tuple[AegisPersistentLease, ...] = ()


class AegisPersistentPool:
    """Cluster-external L3 capacity with committed manifest leases."""

    def __init__(self, *, capacity_tokens: int, manifest_quorum: int = 1) -> None:
        self.capacity_tokens = max(int(capacity_tokens), 0)
        self.manifest_quorum = max(int(manifest_quorum), 1)
        self._leases: dict[str, AegisPersistentLease] = {}
        self._lock = threading.RLock()

    @property
    def used_tokens(self) -> int:
        return sum(lease.token_count for lease in self._leases.values())

    def get(self, backup_key: str) -> AegisPersistentLease | None:
        with self._lock:
            return self._leases.get(str(backup_key))

    def release(
        self,
        backup_key: str,
        *,
        max_version: int | None = None,
    ) -> AegisPersistentLease | None:
        """Release one manifest lease without deleting a newer replacement."""

        key = str(backup_key)
        with self._lock:
            lease = self._leases.get(key)
            if lease is None:
                return None
            if max_version is not None and lease.version > int(max_version):
                return None
            return self._leases.pop(key)

    def admit(
        self,
        *,
        backup_key: str,
        token_count: int,
        utility_density: float,
        version: int,
    ) -> AegisPersistentAdmission:
        key = str(backup_key)
        tokens = max(int(token_count), 0)
        requested_version = max(int(version), 1)
        if not key or tokens <= 0 or self.capacity_tokens <= 0:
            return AegisPersistentAdmission(
                False, "persistent capacity is unavailable", None
            )
        with self._lock:
            previous = self._leases.get(key)
            if previous is not None and requested_version < previous.version:
                return AegisPersistentAdmission(
                    False, "stale persistent lease version", None
                )
            previous = self._leases.pop(key, None)
            deficit = max(self.used_tokens + tokens - self.capacity_tokens, 0)
            victims = sorted(
                (
                    lease
                    for lease in self._leases.values()
                    if lease.utility_density < float(utility_density)
                ),
                key=lambda lease: (
                    lease.utility_density,
                    lease.version,
                    lease.backup_key,
                ),
            )
            if sum(lease.token_count for lease in victims) < deficit:
                if previous is not None:
                    self._leases[key] = previous
                return AegisPersistentAdmission(
                    False, "persistent capacity is exhausted", None
                )
            evicted: list[AegisPersistentLease] = []
            reclaimed = 0
            for victim in victims:
                if reclaimed >= deficit:
                    break
                reclaimed += victim.token_count
                evicted.append(victim)
                self._leases.pop(victim.backup_key, None)
            lease = AegisPersistentLease(
                backup_key=key,
                token_count=tokens,
                utility_density=float(utility_density),
                version=requested_version,
                manifest_version=requested_version,
                manifest_quorum=self.manifest_quorum,
                committed=True,
            )
            self._leases[key] = lease
            return AegisPersistentAdmission(
                True,
                "",
                lease,
                tuple(sorted(evicted, key=lambda item: item.backup_key)),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scope": "cluster_external_persistent",
                "capacity_tokens": self.capacity_tokens,
                "used_tokens": self.used_tokens,
                "available_tokens": max(
                    self.capacity_tokens - self.used_tokens, 0
                ),
                "manifest_quorum": self.manifest_quorum,
                "committed_leases": sum(
                    int(lease.committed) for lease in self._leases.values()
                ),
                "leases": {
                    key: asdict(lease)
                    for key, lease in sorted(self._leases.items())
                },
            }
