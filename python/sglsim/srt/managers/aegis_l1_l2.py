"""Rank-local metadata model for the L1/L2 AegisKV design.

The SGLSim runtime keeps real SGLang process topology, but KV payloads are
metadata only.  This module therefore models Algorithm 2 (chunk action choice)
and Algorithm 3 (windowed alpha/beta control) without changing SGLang's batch
lifecycle.  A target DP scheduler owns one :class:`AegisRankBackupManager`.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from sglsim.srt.managers.pipeline_parallel import DPBudgetSnapshot


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def project_alpha_beta(
    alpha: float,
    beta: float,
    *,
    normal_pressure: float = 0.0,
    previous_alpha: float | None = None,
    previous_beta: float | None = None,
) -> tuple[float, float]:
    """Enforce beta>=alpha and shrink both under high serving pressure."""

    a = _clamp(alpha)
    b = _clamp(beta)
    pressure = _clamp(normal_pressure)
    # Higher normal-path pressure must not expand protection budgets.
    if pressure > 0.15:
        shrink = _clamp(1.0 - 0.75 * pressure)
        a = min(a, shrink)
        b = min(b, shrink)
        if previous_alpha is not None:
            a = min(a, _clamp(previous_alpha))
        if previous_beta is not None:
            b = min(b, _clamp(previous_beta))
    if b < a:
        b = a
    return a, b


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
    return float(default) if math.isnan(parsed) or math.isinf(parsed) else parsed


@dataclass(frozen=True)
class AegisPerNodeDramQuota:
    """One node-shared Aegis DRAM quota in logical-token units.

    The byte quota is authoritative experiment configuration.  Controller and
    Scheduler admission use the largest whole logical-token capacity that fits
    that byte quota; the sub-token remainder is recorded and is never silently
    allocated.
    """

    quota_bytes: int
    quota_tokens: int
    kv_bytes_per_logical_token: int
    unallocatable_tail_bytes: int
    source: str
    explicit: bool

    @property
    def effective_quota_bytes(self) -> int:
        return self.quota_tokens * self.kv_bytes_per_logical_token

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": "per_node",
            "quota_bytes": self.quota_bytes,
            "quota_tokens": self.quota_tokens,
            "kv_bytes_per_logical_token": self.kv_bytes_per_logical_token,
            "effective_quota_bytes": self.effective_quota_bytes,
            "unallocatable_tail_bytes": self.unallocatable_tail_bytes,
            "source": self.source,
            "explicit": self.explicit,
        }


def resolve_aegis_per_node_dram_quota(
    raw: Mapping[str, Any] | None,
    *,
    fallback_tokens: int,
) -> AegisPerNodeDramQuota:
    """Parse the shared runtime binding or retain the legacy L2 fallback.

    ``require_explicit_per_node_dram_quota`` makes absence fail closed.  If any
    explicit quota field is supplied, all three fields must be supplied and
    byte/token conversion must be exact under the declared floor rule.
    """

    values = dict(raw or {})
    field_names = (
        "per_node_aegis_dram_quota_bytes",
        "per_node_aegis_dram_quota_tokens",
        "kv_bytes_per_logical_token",
    )
    present = tuple(name in values for name in field_names)
    required = bool(values.get("require_explicit_per_node_dram_quota", False))
    if not any(present):
        if required:
            raise ValueError(
                "explicit per-node Aegis DRAM quota is required by runtime config"
            )
        tokens = max(_as_int(fallback_tokens), 1)
        return AegisPerNodeDramQuota(
            quota_bytes=0,
            quota_tokens=tokens,
            kv_bytes_per_logical_token=0,
            unallocatable_tail_bytes=0,
            source="legacy_cache_l2_capacity",
            explicit=False,
        )
    if not all(present):
        missing = [
            name for name, is_present in zip(field_names, present) if not is_present
        ]
        raise ValueError(
            "incomplete per-node Aegis DRAM quota binding; missing "
            + ", ".join(missing)
        )

    def positive_int(name: str) -> int:
        value = values[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    quota_bytes = positive_int("per_node_aegis_dram_quota_bytes")
    quota_tokens = positive_int("per_node_aegis_dram_quota_tokens")
    kv_bytes = positive_int("kv_bytes_per_logical_token")
    expected_tokens, tail = divmod(quota_bytes, kv_bytes)
    if quota_tokens != expected_tokens:
        raise ValueError(
            "per_node_aegis_dram_quota_tokens must equal floor("
            "per_node_aegis_dram_quota_bytes / kv_bytes_per_logical_token)"
        )
    return AegisPerNodeDramQuota(
        quota_bytes=quota_bytes,
        quota_tokens=quota_tokens,
        kv_bytes_per_logical_token=kv_bytes,
        unallocatable_tail_bytes=tail,
        source="explicit_runtime_config_v1",
        explicit=True,
    )


@dataclass(frozen=True)
class AegisL1L2Config:
    l1_total_tokens: int
    l2_total_tokens: int
    initial_alpha: float = 0.20
    max_alpha: float = 1.0
    initial_beta: float = 0.40
    chunk_tokens: int = 512
    window_requests: int = 6
    window_min_requests: int = 0
    window_max_requests: int = 0
    window_target_ms: float = 0.0
    window_overflow_trigger_tokens: int = 0
    demand_ewma: float = 0.50
    publish_smoothing: float = 0.50
    hysteresis: float = 0.02
    safety_margin: float = 0.05
    burst_reserve_tokens: int = 512
    cost_weight: float = 0.05
    failure_hazard_probability: float = 1.0
    critical_host_floor_fraction: float = 0.0
    critical_host_floor_probability: float = 0.75
    l1_write_setup_ms: float = 0.12
    l1_write_ms_per_token: float = 0.00045
    l2_write_setup_ms: float = 0.30
    l2_write_ms_per_token: float = 0.00120
    l1_restore_setup_ms: float = 2.40
    l1_restore_ms_per_token: float = 0.00190
    l2_restore_setup_ms: float = 5.00
    l2_restore_ms_per_token: float = 0.00360
    l3_read_setup_ms: float = 8.00
    l3_read_ms_per_token: float = 0.00700
    recompute_setup_ms: float = 4.00
    recompute_ms_per_token: float = 0.01080

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any] | None,
        *,
        l1_total_tokens: int,
        l2_total_tokens: int,
    ) -> "AegisL1L2Config":
        values = dict(raw or {})
        values["l1_total_tokens"] = max(_as_int(l1_total_tokens), 1)
        values["l2_total_tokens"] = max(_as_int(l2_total_tokens), 1)
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True)
class AegisBackupPlan:
    rid: str
    backup_key: str
    source_dp_rank: int
    target_dp_rank: int
    token_count: int
    critical_probability: float
    slack_ms: float
    workflow_id: str = ""
    agent_id: str = ""
    worker_seq: int = 0
    normal_l1_pressure: float = 0.0
    normal_l2_pressure: float = 0.0
    allowed_tiers: tuple[str, ...] = ("l1", "l2")
    failure_scope: str = "node"
    gpu_candidate: dict[str, Any] = field(default_factory=dict)
    host_candidate: dict[str, Any] = field(default_factory=dict)
    l3_candidate: dict[str, Any] = field(default_factory=dict)
    dp_host_beta: float | None = None
    placement_mode: str = "legacy_greedy"
    exact_chunk_placements: tuple[tuple[int, int, str], ...] = ()
    solver_instance_sha256: str = ""
    solver_decision_sha256: str = ""
    solver_certificate_id: str = ""
    route_version: int = 1
    required_pp_size: int = 1
    source_pp_rank: int = 0
    target_pp_rank: int = 0
    model_revision: str = ""
    kv_abi: str = ""
    parallel_layout: str = ""
    source_failure_domain: str = ""
    hbm_failure_domain: str = ""
    version: int = 1
    created_time: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AegisBackupPlan":
        allowed_raw = raw.get("allowed_tiers", ("l1", "l2"))
        if isinstance(allowed_raw, str):
            allowed_raw = (allowed_raw,)
        allowed_tiers = tuple(
            tier
            for tier in (str(value).strip().lower() for value in allowed_raw)
            if tier in {"l1", "l2"}
        )
        exact_rows: list[tuple[int, int, str]] = []
        for index, value in enumerate(raw.get("exact_chunk_placements", ())):
            if isinstance(value, dict):
                chunk_id = _as_int(value.get("chunk_id"), index)
                token_count = _as_int(value.get("token_count"), 0)
                tier = str(value.get("tier") or "").upper()
            elif isinstance(value, (tuple, list)) and len(value) == 3:
                chunk_id = _as_int(value[0], index)
                token_count = _as_int(value[1], 0)
                tier = str(value[2]).upper()
            else:
                raise ValueError(
                    f"invalid exact chunk placement at index {index}: {value!r}"
                )
            exact_rows.append((chunk_id, token_count, tier))
        return cls(
            rid=str(raw.get("rid") or ""),
            backup_key=str(raw.get("backup_key") or raw.get("rid") or ""),
            source_dp_rank=_as_int(raw.get("source_dp_rank"), -1),
            target_dp_rank=_as_int(raw.get("target_dp_rank"), -1),
            token_count=max(_as_int(raw.get("token_count"), 0), 0),
            critical_probability=_clamp(
                _as_float(raw.get("critical_probability"), 0.25)
            ),
            slack_ms=max(_as_float(raw.get("slack_ms"), 0.0), 0.0),
            workflow_id=str(raw.get("workflow_id") or ""),
            agent_id=str(raw.get("agent_id") or ""),
            worker_seq=max(_as_int(raw.get("worker_seq"), 0), 0),
            normal_l1_pressure=_clamp(
                _as_float(raw.get("normal_l1_pressure"), 0.0)
            ),
            normal_l2_pressure=_clamp(
                _as_float(raw.get("normal_l2_pressure"), 0.0)
            ),
            allowed_tiers=allowed_tiers,
            failure_scope=str(raw.get("failure_scope") or "node"),
            gpu_candidate=(
                dict(raw.get("gpu_candidate"))
                if isinstance(raw.get("gpu_candidate"), dict)
                else {}
            ),
            host_candidate=(
                dict(raw.get("host_candidate"))
                if isinstance(raw.get("host_candidate"), dict)
                else {}
            ),
            l3_candidate=(
                dict(raw.get("l3_candidate"))
                if isinstance(raw.get("l3_candidate"), dict)
                else {}
            ),
            dp_host_beta=(
                _clamp(
                    _as_float(
                        raw.get("dp_host_beta", raw.get("node_host_beta"))
                    )
                )
                if raw.get("dp_host_beta", raw.get("node_host_beta"))
                is not None
                else None
            ),
            placement_mode=str(
                raw.get("placement_mode") or "legacy_greedy"
            ),
            exact_chunk_placements=tuple(exact_rows),
            solver_instance_sha256=str(
                raw.get("solver_instance_sha256") or ""
            ),
            solver_decision_sha256=str(
                raw.get("solver_decision_sha256") or ""
            ),
            solver_certificate_id=str(
                raw.get("solver_certificate_id") or ""
            ),
            route_version=max(_as_int(raw.get("route_version"), 1), 1),
            required_pp_size=max(
                _as_int(raw.get("required_pp_size"), 1), 1
            ),
            source_pp_rank=max(_as_int(raw.get("source_pp_rank"), 0), 0),
            target_pp_rank=max(_as_int(raw.get("target_pp_rank"), 0), 0),
            model_revision=str(raw.get("model_revision") or ""),
            kv_abi=str(raw.get("kv_abi") or ""),
            parallel_layout=str(raw.get("parallel_layout") or ""),
            source_failure_domain=str(
                raw.get("source_failure_domain") or ""
            ),
            hbm_failure_domain=str(raw.get("hbm_failure_domain") or ""),
            version=max(_as_int(raw.get("version"), 1), 1),
            created_time=_as_float(raw.get("created_time"), time.time()),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AegisBackupAck:
    rid: str
    backup_key: str
    source_dp_rank: int
    target_dp_rank: int
    version: int
    ack: bool
    l1_tokens: int
    l2_tokens: int
    recompute_tokens: int
    safe_frontier_tokens: int
    recovery_action: str
    expected_recovery_ms: float
    alpha: float
    beta: float
    window_id: int
    d_l1_tokens: int
    d_l2_tokens: int
    u_l1_tokens: int
    u_l2_tokens: int
    o_l1_tokens: int
    o_l2_tokens: int
    sequence: int
    updated_time: float
    window_update: dict[str, Any] | None = None
    failure_scope: str = "node"
    gpu_candidate: dict[str, Any] = field(default_factory=dict)
    host_candidate: dict[str, Any] = field(default_factory=dict)
    l3_candidate: dict[str, Any] = field(default_factory=dict)
    decision_d_l1_tokens: int = 0
    decision_d_l2_tokens: int = 0
    decision_u_l1_tokens: int = 0
    decision_u_l2_tokens: int = 0
    decision_o_l1_tokens: int = 0
    decision_o_l2_tokens: int = 0
    alpha_scope: str = "dp_replica_pp_bottleneck"
    beta_scope: str = "dp_replica_policy"
    workflow_id: str = ""
    agent_id: str = ""
    worker_seq: int = 0
    critical_host_floor_tokens: int = 0
    critical_host_floor_fraction: float = 0.0
    placement_mode: str = "legacy_greedy"
    exact_chunk_placements: tuple[tuple[int, int, str], ...] = ()
    solver_instance_sha256: str = ""
    solver_decision_sha256: str = ""
    solver_certificate_id: str = ""
    design_semantics: str = "legacy_l1_l2_v1"
    residency: str = ""
    hbm_pending_tokens: int = 0
    dram_pending_tokens: int = 0
    reserved_l1_tokens: int = 0
    reserved_l2_tokens: int = 0
    hbm_copy_path: str = ""
    dram_copy_path: str = ""
    independent_tier_results: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Entry:
    backup_key: str
    rid: str
    version: int
    source_dp_rank: int
    workflow_id: str
    agent_id: str
    worker_seq: int
    token_count: int
    l1_tokens: int
    l2_tokens: int
    recompute_tokens: int
    l1_density: float
    l2_density: float
    failure_scope: str = "node"
    gpu_candidate: dict[str, Any] = field(default_factory=dict)
    host_candidate: dict[str, Any] = field(default_factory=dict)
    l3_candidate: dict[str, Any] = field(default_factory=dict)
    placement_mode: str = "legacy_greedy"
    exact_chunk_placements: tuple[tuple[int, int, str], ...] = ()
    solver_instance_sha256: str = ""
    solver_decision_sha256: str = ""
    solver_certificate_id: str = ""
    planned_chunk_placements: tuple[tuple[int, int, str], ...] = ()
    source_committed_token_end: int = 0
    materialized_token_end: int = 0
    materialized_l1_tokens: int = 0
    materialized_l2_tokens: int = 0
    last_block_generation: int = -1
    execution_id: str = ""


class AegisRankBackupManager:
    """Own Algorithm 2 and Algorithm 3 state for one target DP rank."""

    def __init__(
        self,
        *,
        dp_rank: int,
        config: AegisL1L2Config,
        now_fn: Callable[[], float] = time.time,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dp_rank = int(dp_rank)
        self.config = config
        self._now_fn = now_fn
        self._monotonic_fn = monotonic_fn
        self.alpha = _clamp(
            config.initial_alpha,
            high=_clamp(config.max_alpha),
        )
        self.beta = _clamp(config.initial_beta)
        self.window_id = 0
        self.sequence = 0
        self._window_requests = 0
        self._demand = {"l1": 0, "l2": 0}
        # Direct per-window Algorithm 2 admission misses.  This is the O
        # feedback consumed by Algorithm 3; resident U is cross-window state
        # and must not be subtracted from the current window's D.
        self._unmet = {"l1": 0, "l2": 0}
        self._predicted = {"l1": 0.0, "l2": 0.0}
        self._normal_pressure = {"l1": 0.0, "l2": 0.0}
        self._entries: dict[str, _Entry] = {}
        self._used_tokens = {"l1": 0, "l2": 0}
        self._pending_hint_updates: list[dict[str, Any]] = []
        self._window_started_monotonic = self._monotonic_fn()
        self._external_budget: DPBudgetSnapshot | None = None
        self._external_l1_limit_tokens: int | None = None
        self._external_l2_limit_tokens: int | None = None
        self._stage_observation_window_id = 0

    @property
    def l1_used_tokens(self) -> int:
        return self._used_tokens["l1"]

    @property
    def l2_used_tokens(self) -> int:
        return self._used_tokens["l2"]

    def _tier_budget(self, tier: str) -> int:
        if tier == "l1" and self._external_l1_limit_tokens is not None:
            return self._external_l1_limit_tokens
        if tier == "l2" and self._external_l2_limit_tokens is not None:
            return self._external_l2_limit_tokens
        if tier == "l1":
            return int(self.alpha * self.config.l1_total_tokens)
        return int(self.beta * self.config.l2_total_tokens)

    def apply_dp_budget(
        self,
        snapshot: DPBudgetSnapshot,
        *,
        pp_rank: int,
    ) -> dict[str, Any]:
        """Atomically consume one controller-owned DP alpha/beta version."""

        if snapshot.dp_rank != self.dp_rank:
            raise ValueError("DP budget belongs to another replica")
        if not 0 <= int(pp_rank) < snapshot.pp_size:
            raise ValueError("PP stage is outside DP budget topology")
        current = self._external_budget
        if current is not None:
            if snapshot.budget_version < current.budget_version:
                raise ValueError("cannot apply a stale DP budget version")
            if snapshot.budget_version == current.budget_version:
                if snapshot != current:
                    raise ValueError("conflicting DP budget payload for one version")
                return {
                    "applied": False,
                    "budget_version": snapshot.budget_version,
                    "window_id": snapshot.window_id,
                    "evicted_l1_backup_keys": [],
                    "evicted_l2_backup_keys": [],
                }

        self._external_budget = snapshot
        self._external_l1_limit_tokens = snapshot.hbm_limit_for_stage(pp_rank)
        self._external_l2_limit_tokens = snapshot.host_policy_limit_tokens
        self.alpha = float(snapshot.alpha)
        self.beta = float(snapshot.beta)
        self.window_id = int(snapshot.window_id)
        l1_evicted = self._shrink_tier_to_budget("l1")
        l2_evicted = self._shrink_tier_to_budget("l2")
        return {
            "applied": True,
            "budget_version": snapshot.budget_version,
            "window_id": snapshot.window_id,
            "alpha": snapshot.alpha,
            "beta": snapshot.beta,
            "hbm_limit_tokens": self._external_l1_limit_tokens,
            "host_policy_limit_tokens": self._external_l2_limit_tokens,
            "objective_id": snapshot.objective_id,
            "solver_certificate_id": snapshot.solver_certificate_id,
            "evicted_l1_backup_keys": list(l1_evicted),
            "evicted_l2_backup_keys": list(l2_evicted),
        }

    def _tier_available(self, tier: str) -> int:
        used = self.l1_used_tokens if tier == "l1" else self.l2_used_tokens
        return max(self._tier_budget(tier) - used, 0)

    def _chunk_value(
        self,
        *,
        tier: str,
        tokens: int,
        critical_probability: float,
        slack_ms: float,
        topology_candidate: dict[str, Any] | None = None,
    ) -> tuple[float, float, float, float]:
        if topology_candidate:
            if not bool(topology_candidate.get("valid", False)):
                return 0.0, 0.0, float("-inf"), float("-inf")
            candidate_tokens = max(
                _as_int(topology_candidate.get("token_count"), tokens), 1
            )
            scale = max(int(tokens), 0) / candidate_tokens
            benefit = _as_float(
                topology_candidate.get("deadline_benefit_ms"), 0.0
            ) * scale
            write_ms = _as_float(
                topology_candidate.get("write_ms"), 0.0
            ) * scale
            net = _as_float(topology_candidate.get("utility"), 0.0) * scale
            return benefit, write_ms, net, net / max(int(tokens), 1)
        cfg = self.config
        recompute_ms = cfg.recompute_setup_ms + tokens * cfg.recompute_ms_per_token
        if tier == "l1":
            restore_ms = cfg.l1_restore_setup_ms + tokens * cfg.l1_restore_ms_per_token
            write_ms = cfg.l1_write_setup_ms + tokens * cfg.l1_write_ms_per_token
        else:
            restore_ms = cfg.l2_restore_setup_ms + tokens * cfg.l2_restore_ms_per_token
            write_ms = cfg.l2_write_setup_ms + tokens * cfg.l2_write_ms_per_token
        recompute_penalty = max(recompute_ms - slack_ms, 0.0)
        restore_penalty = max(restore_ms - slack_ms, 0.0)
        benefit = max(recompute_penalty - restore_penalty, 0.0)
        expected_benefit = (
            _clamp(cfg.failure_hazard_probability)
            * _clamp(critical_probability)
            * benefit
        )
        net = expected_benefit - cfg.cost_weight * write_ms
        density = net / max(tokens, 1)
        return expected_benefit, write_ms, net, density

    @staticmethod
    def _entry_density(entry: _Entry, tier: str) -> float:
        return entry.l1_density if tier == "l1" else entry.l2_density

    def _make_room_for_chunk(
        self,
        *,
        tier: str,
        chunk_tokens: int,
        selected_tokens: int,
        candidate_density: float,
        exclude_key: str,
    ) -> tuple[bool, set[str]]:
        """Admit directly or replace lower-density tier entries."""

        used = self.l1_used_tokens if tier == "l1" else self.l2_used_tokens
        deficit = max(
            used + max(int(selected_tokens), 0) + max(int(chunk_tokens), 0)
            - self._tier_budget(tier),
            0,
        )
        if deficit <= 0:
            return True, set()
        candidates = sorted(
            (
                entry
                for key, entry in self._entries.items()
                if key != exclude_key
                and (entry.l1_tokens if tier == "l1" else entry.l2_tokens) > 0
                and self._entry_density(entry, tier) < candidate_density
            ),
            key=lambda entry: (
                self._entry_density(entry, tier),
                entry.version,
                entry.backup_key,
            ),
        )
        removable = sum(
            entry.l1_tokens if tier == "l1" else entry.l2_tokens
            for entry in candidates
        )
        if removable < deficit:
            return False, set()

        affected: set[str] = set()
        remaining = deficit
        for entry in candidates:
            resident = entry.l1_tokens if tier == "l1" else entry.l2_tokens
            removed = min(resident, remaining)
            if tier == "l1":
                entry.l1_tokens -= removed
            else:
                entry.l2_tokens -= removed
            self._used_tokens[tier] -= removed
            entry.recompute_tokens += removed
            remaining -= removed
            affected.add(entry.backup_key)
            if remaining <= 0:
                break
        return True, affected

    def _entry_hint(self, entry: _Entry) -> dict[str, Any]:
        if entry.l1_tokens and entry.l2_tokens:
            action = "l1_l2_backup"
        elif entry.l1_tokens:
            action = "l1_backup"
        elif entry.l2_tokens:
            action = "l2_backup"
        else:
            action = "recompute"
        expected_ms = (
            (self.config.l1_restore_setup_ms if entry.l1_tokens else 0.0)
            + entry.l1_tokens * self.config.l1_restore_ms_per_token
            + (self.config.l2_restore_setup_ms if entry.l2_tokens else 0.0)
            + entry.l2_tokens * self.config.l2_restore_ms_per_token
            + (self.config.recompute_setup_ms if entry.recompute_tokens else 0.0)
            + entry.recompute_tokens * self.config.recompute_ms_per_token
        )
        self.sequence += 1
        return AegisBackupAck(
            rid=entry.rid,
            backup_key=entry.backup_key,
            source_dp_rank=entry.source_dp_rank,
            target_dp_rank=self.dp_rank,
            version=entry.version,
            ack=True,
            l1_tokens=entry.l1_tokens,
            l2_tokens=entry.l2_tokens,
            recompute_tokens=entry.recompute_tokens,
            safe_frontier_tokens=entry.l1_tokens + entry.l2_tokens,
            recovery_action=action,
            expected_recovery_ms=round(expected_ms, 3),
            alpha=round(self.alpha, 4),
            beta=round(self.beta, 4),
            window_id=self.window_id,
            d_l1_tokens=int(self._demand["l1"]),
            d_l2_tokens=int(self._demand["l2"]),
            u_l1_tokens=self.l1_used_tokens,
            u_l2_tokens=self.l2_used_tokens,
            o_l1_tokens=int(self._unmet["l1"]),
            o_l2_tokens=int(self._unmet["l2"]),
            sequence=self.sequence,
            updated_time=self._now_fn(),
            failure_scope=entry.failure_scope,
            gpu_candidate=dict(entry.gpu_candidate),
            host_candidate=dict(entry.host_candidate),
            l3_candidate=dict(entry.l3_candidate),
            workflow_id=entry.workflow_id,
            agent_id=entry.agent_id,
            worker_seq=entry.worker_seq,
            placement_mode=entry.placement_mode,
            exact_chunk_placements=entry.exact_chunk_placements,
            solver_instance_sha256=entry.solver_instance_sha256,
            solver_decision_sha256=entry.solver_decision_sha256,
            solver_certificate_id=entry.solver_certificate_id,
        ).to_dict()

    def supersede(
        self,
        backup_key: str,
        *,
        expected_version: int,
        superseded_by: str,
    ) -> dict[str, Any]:
        """Release one obsolete agent frontier from rank-local L1/L2.

        The controller is the lineage authority because a newer frontier may
        select a different target rank.  Version matching keeps a delayed
        command from deleting a newer replacement that reused the same key.
        """

        key = str(backup_key)
        entry = self._entries.get(key)
        if entry is None or entry.version != int(expected_version):
            return {
                "backup_key": key,
                "expected_version": int(expected_version),
                "superseded_by": str(superseded_by),
                "released": False,
                "released_l1_tokens": 0,
                "released_l2_tokens": 0,
            }
        entry = self._entries.pop(key)
        self._used_tokens["l1"] -= entry.l1_tokens
        self._used_tokens["l2"] -= entry.l2_tokens
        return {
            "backup_key": key,
            "expected_version": int(expected_version),
            "superseded_by": str(superseded_by),
            "released": True,
            "released_l1_tokens": max(int(entry.l1_tokens), 0),
            "released_l2_tokens": max(int(entry.l2_tokens), 0),
            "workflow_id": entry.workflow_id,
            "agent_id": entry.agent_id,
            "worker_seq": entry.worker_seq,
        }

    def _shrink_tier_to_budget(self, tier: str) -> tuple[str, ...]:
        """Evict lowest-density chunks before publishing a lower tier limit."""

        if tier not in {"l1", "l2"}:
            raise ValueError(f"unknown AegisKV tier {tier}")
        target = self._tier_budget(tier)
        affected: set[str] = set()
        used_tokens = (
            lambda: self.l1_used_tokens if tier == "l1" else self.l2_used_tokens
        )
        token_field = f"{tier}_tokens"
        density_field = f"{tier}_density"
        while used_tokens() > target:
            candidates = [
                entry
                for entry in self._entries.values()
                if int(getattr(entry, token_field)) > 0
            ]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda entry: (
                    float(getattr(entry, density_field)),
                    entry.version,
                    entry.backup_key,
                ),
            )
            victim_tokens = int(getattr(victim, token_field))
            remove = min(victim_tokens, used_tokens() - target)
            setattr(victim, token_field, victim_tokens - remove)
            victim.recompute_tokens += remove
            self._used_tokens[tier] -= remove
            affected.add(victim.backup_key)
        for backup_key in sorted(affected):
            entry = self._entries.get(backup_key)
            if entry is not None:
                self._pending_hint_updates.append(self._entry_hint(entry))
        return tuple(sorted(affected))

    def _shrink_l1_to_budget(self) -> tuple[str, ...]:
        """Compatibility wrapper for the legacy local alpha controller."""

        return self._shrink_tier_to_budget("l1")

    def pop_pending_hint_updates(self) -> list[dict[str, Any]]:
        updates = self._pending_hint_updates
        self._pending_hint_updates = []
        return updates

    def place(self, plan: AegisBackupPlan) -> AegisBackupAck:
        """Run Algorithm 2, then optionally publish an Algorithm 3 update."""

        if plan.target_dp_rank != self.dp_rank:
            raise ValueError(
                f"backup plan target {plan.target_dp_rank} != local rank {self.dp_rank}"
            )
        current = self._entries.get(plan.backup_key)
        if current is not None and plan.version < current.version:
            return AegisBackupAck(**self._entry_hint(current))
        exact_mode = plan.placement_mode == "certified_exact"
        if exact_mode:
            if self._external_budget is None:
                raise RuntimeError(
                    "certified exact placement requires a DP-owned budget"
                )
            if not plan.exact_chunk_placements:
                raise ValueError(
                    "certified exact placement has no stage-local chunks"
                )
            if not (
                plan.solver_instance_sha256
                and plan.solver_decision_sha256
                and plan.solver_certificate_id
            ):
                raise ValueError(
                    "certified exact placement is missing certificate identity"
                )
        elif plan.exact_chunk_placements:
            raise ValueError(
                "exact_chunk_placements require placement_mode=certified_exact"
            )
        self._normal_pressure["l1"] = _clamp(plan.normal_l1_pressure)
        self._normal_pressure["l2"] = _clamp(plan.normal_l2_pressure)
        if plan.dp_host_beta is not None:
            requested_beta = _clamp(plan.dp_host_beta)
            if self._external_budget is not None and not math.isclose(
                requested_beta,
                self._external_budget.beta,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "backup plan beta does not match DP budget snapshot"
                )
            self.beta = requested_beta

        # A newer version of the same logical backup replaces its old capacity
        # before admission. The new authoritative ACK supersedes the old hint.
        previous = self._entries.pop(plan.backup_key, None)
        if previous is not None:
            self._used_tokens["l1"] -= previous.l1_tokens
            self._used_tokens["l2"] -= previous.l2_tokens
        remaining = max(int(plan.token_count), 0)
        l1_tokens = 0
        l2_tokens = 0
        planned_chunks: list[tuple[int, int, str]] = []
        tier_densities = {"l1": 0.0, "l2": 0.0}
        decision_demand = {"l1": 0, "l2": 0}
        decision_admitted = {"l1": 0, "l2": 0}
        decision_unmet = {"l1": 0, "l2": 0}
        replaced_keys: set[str] = set()
        host_floor_fraction = _clamp(
            self.config.critical_host_floor_fraction
        )
        host_floor_enabled = bool(
            not exact_mode
            and
            host_floor_fraction > 0.0
            and plan.critical_probability
            >= _clamp(self.config.critical_host_floor_probability)
            and "l2" in plan.allowed_tiers
            and bool(plan.gpu_candidate.get("direct_retry_route_required", False))
        )
        critical_host_floor_tokens = (
            min(
                int(plan.token_count),
                math.ceil(
                    int(plan.token_count)
                    * host_floor_fraction
                    / max(int(self.config.chunk_tokens), 1)
                )
                * max(int(self.config.chunk_tokens), 1),
            )
            if host_floor_enabled
            else 0
        )
        if exact_mode:
            ordered = tuple(
                sorted(plan.exact_chunk_placements, key=lambda row: row[0])
            )
            chunk_ids = [row[0] for row in ordered]
            if len(set(chunk_ids)) != len(chunk_ids) or chunk_ids != list(
                range(len(chunk_ids))
            ):
                raise ValueError(
                    "certified exact chunk IDs must be unique and contiguous from zero"
                )
            if any(tokens <= 0 for _chunk_id, tokens, _tier in ordered):
                raise ValueError("certified exact chunk tokens must be positive")
            if sum(tokens for _chunk_id, tokens, _tier in ordered) != remaining:
                raise ValueError(
                    "certified exact stage chunks do not cover plan.token_count"
                )
            for _chunk_id, chunk, exact_tier in ordered:
                planned_chunks.append((_chunk_id, chunk, exact_tier))
                if exact_tier == "RECOMPUTE":
                    continue
                tier = "l1" if exact_tier == "HBM" else "l2" if exact_tier == "HOST" else ""
                if not tier or tier not in plan.allowed_tiers:
                    raise ValueError(
                        f"certified exact tier {exact_tier!r} is not allowed"
                    )
                topology_candidate = (
                    plan.gpu_candidate if tier == "l1" else plan.host_candidate
                )
                _benefit, _cost, _net, density = self._chunk_value(
                    tier=tier,
                    tokens=chunk,
                    critical_probability=plan.critical_probability,
                    slack_ms=plan.slack_ms,
                    topology_candidate=topology_candidate,
                )
                self._demand[tier] += chunk
                decision_demand[tier] += chunk
                selected = l1_tokens if tier == "l1" else l2_tokens
                if (
                    self._used_tokens[tier] + selected + chunk
                    > self._tier_budget(tier)
                ):
                    raise RuntimeError(
                        "certified exact placement exceeds deployed DP budget: "
                        f"tier={exact_tier} used={self._used_tokens[tier]} "
                        f"selected={selected} chunk={chunk} "
                        f"limit={self._tier_budget(tier)}"
                    )
                if tier == "l1":
                    l1_tokens += chunk
                else:
                    l2_tokens += chunk
                decision_admitted[tier] += chunk
                tier_densities[tier] = max(tier_densities[tier], density)
        else:
            chunk_id = 0
            while remaining > 0:
                chunk = min(max(int(self.config.chunk_tokens), 1), remaining)
                choices: list[tuple[float, str, float]] = []
                selected_tier = "RECOMPUTE"
                force_host = l2_tokens < critical_host_floor_tokens
                for tier in plan.allowed_tiers:
                    topology_candidate = (
                        plan.gpu_candidate if tier == "l1" else plan.host_candidate
                    )
                    _benefit, _cost, net, density = self._chunk_value(
                        tier=tier,
                        tokens=chunk,
                        critical_probability=plan.critical_probability,
                        slack_ms=plan.slack_ms,
                        topology_candidate=topology_candidate,
                    )
                    if net > 0.0 or (
                        force_host
                        and tier == "l2"
                        and math.isfinite(net)
                    ):
                        choices.append((density, tier, net))
                choices.sort(
                    key=lambda item: (
                        bool(force_host and item[1] == "l2"),
                        item[0],
                        item[1],
                    ),
                    reverse=True,
                )
                for density, tier, _net in choices:
                    self._demand[tier] += chunk
                    decision_demand[tier] += chunk
                    selected = l1_tokens if tier == "l1" else l2_tokens
                    admitted, affected = self._make_room_for_chunk(
                        tier=tier,
                        chunk_tokens=chunk,
                        selected_tokens=selected,
                        candidate_density=density,
                        exclude_key=plan.backup_key,
                    )
                    if not admitted:
                        self._unmet[tier] += chunk
                        decision_unmet[tier] += chunk
                        continue
                    replaced_keys.update(affected)
                    if tier == "l1":
                        l1_tokens += chunk
                        selected_tier = "HBM"
                    else:
                        l2_tokens += chunk
                        selected_tier = "HOST"
                    decision_admitted[tier] += chunk
                    tier_densities[tier] = max(tier_densities[tier], density)
                    break
                planned_chunks.append((chunk_id, chunk, selected_tier))
                chunk_id += 1
                remaining -= chunk

        recompute_tokens = max(plan.token_count - l1_tokens - l2_tokens, 0)
        if l1_tokens and l2_tokens:
            action = "l1_l2_backup"
        elif l1_tokens:
            action = "l1_backup"
        elif l2_tokens:
            action = "l2_backup"
        else:
            action = "recompute"
        expected_ms = (
            (self.config.l1_restore_setup_ms if l1_tokens else 0.0)
            + l1_tokens * self.config.l1_restore_ms_per_token
            + (self.config.l2_restore_setup_ms if l2_tokens else 0.0)
            + l2_tokens * self.config.l2_restore_ms_per_token
            + (self.config.recompute_setup_ms if recompute_tokens else 0.0)
            + recompute_tokens * self.config.recompute_ms_per_token
        )
        self._entries[plan.backup_key] = _Entry(
            backup_key=plan.backup_key,
            rid=plan.rid,
            version=plan.version,
            source_dp_rank=plan.source_dp_rank,
            workflow_id=plan.workflow_id,
            agent_id=plan.agent_id,
            worker_seq=plan.worker_seq,
            token_count=max(int(plan.token_count), 0),
            l1_tokens=l1_tokens,
            l2_tokens=l2_tokens,
            recompute_tokens=recompute_tokens,
            l1_density=tier_densities["l1"],
            l2_density=tier_densities["l2"],
            failure_scope=plan.failure_scope,
            gpu_candidate=dict(plan.gpu_candidate),
            host_candidate=dict(plan.host_candidate),
            l3_candidate=dict(plan.l3_candidate),
            placement_mode=plan.placement_mode,
            exact_chunk_placements=plan.exact_chunk_placements,
            solver_instance_sha256=plan.solver_instance_sha256,
            solver_decision_sha256=plan.solver_decision_sha256,
            solver_certificate_id=plan.solver_certificate_id,
            planned_chunk_placements=tuple(planned_chunks),
        )
        self._used_tokens["l1"] += l1_tokens
        self._used_tokens["l2"] += l2_tokens
        for backup_key in sorted(replaced_keys):
            replaced = self._entries.get(backup_key)
            if replaced is not None:
                self._pending_hint_updates.append(self._entry_hint(replaced))
        self._window_requests += 1
        d_l1 = int(self._demand["l1"])
        d_l2 = int(self._demand["l2"])
        u_l1 = self.l1_used_tokens
        u_l2 = self.l2_used_tokens
        o_l1 = int(self._unmet["l1"])
        o_l2 = int(self._unmet["l2"])
        window_update = None
        window_trigger = self._window_close_trigger()
        if window_trigger:
            window_update = self.close_window(trigger=window_trigger)
            authoritative = self._entries[plan.backup_key]
            l1_tokens = authoritative.l1_tokens
            l2_tokens = authoritative.l2_tokens
            recompute_tokens = authoritative.recompute_tokens
            if l1_tokens and l2_tokens:
                action = "l1_l2_backup"
            elif l1_tokens:
                action = "l1_backup"
            elif l2_tokens:
                action = "l2_backup"
            else:
                action = "recompute"
            expected_ms = (
                (self.config.l1_restore_setup_ms if l1_tokens else 0.0)
                + l1_tokens * self.config.l1_restore_ms_per_token
                + (self.config.l2_restore_setup_ms if l2_tokens else 0.0)
                + l2_tokens * self.config.l2_restore_ms_per_token
                + (
                    self.config.recompute_setup_ms
                    if recompute_tokens
                    else 0.0
                )
                + recompute_tokens * self.config.recompute_ms_per_token
            )

        self.sequence += 1
        return AegisBackupAck(
            rid=plan.rid,
            backup_key=plan.backup_key,
            source_dp_rank=plan.source_dp_rank,
            target_dp_rank=self.dp_rank,
            version=plan.version,
            ack=True,
            l1_tokens=l1_tokens,
            l2_tokens=l2_tokens,
            recompute_tokens=recompute_tokens,
            safe_frontier_tokens=l1_tokens + l2_tokens,
            recovery_action=action,
            expected_recovery_ms=round(expected_ms, 3),
            alpha=round(self.alpha, 4),
            beta=round(self.beta, 4),
            window_id=self.window_id,
            d_l1_tokens=d_l1,
            d_l2_tokens=d_l2,
            u_l1_tokens=u_l1,
            u_l2_tokens=u_l2,
            o_l1_tokens=o_l1,
            o_l2_tokens=o_l2,
            sequence=self.sequence,
            updated_time=self._now_fn(),
            window_update=window_update,
            failure_scope=plan.failure_scope,
            gpu_candidate=dict(plan.gpu_candidate),
            host_candidate=dict(plan.host_candidate),
            l3_candidate=dict(plan.l3_candidate),
            decision_d_l1_tokens=decision_demand["l1"],
            decision_d_l2_tokens=decision_demand["l2"],
            decision_u_l1_tokens=decision_admitted["l1"],
            decision_u_l2_tokens=decision_admitted["l2"],
            decision_o_l1_tokens=decision_unmet["l1"],
            decision_o_l2_tokens=decision_unmet["l2"],
            workflow_id=plan.workflow_id,
            agent_id=plan.agent_id,
            worker_seq=plan.worker_seq,
            critical_host_floor_tokens=critical_host_floor_tokens,
            critical_host_floor_fraction=(
                host_floor_fraction if host_floor_enabled else 0.0
            ),
            placement_mode=plan.placement_mode,
            exact_chunk_placements=plan.exact_chunk_placements,
            solver_instance_sha256=plan.solver_instance_sha256,
            solver_decision_sha256=plan.solver_decision_sha256,
            solver_certificate_id=plan.solver_certificate_id,
        )

    def materialize_committed_prefix(
        self,
        backup_key: str,
        *,
        expected_version: int,
        source_committed_token_end: int,
        execution_id: str,
        block_generation: int,
    ) -> dict[str, Any]:
        """Commit only a source-observed, contiguous KV prefix.

        ``place`` reserves capacity for a request's maximum token budget.  It
        does not prove that any KV exists.  This method is the separate target
        data-plane commit boundary: the source has completed the corresponding
        forward(s), and every protected chunk from token zero through the
        returned frontier fits in the still-live reserved tier capacity.

        Capacity shrink is interpreted conservatively as tail-first within an
        entry.  A hole (RECOMPUTE), a partial chunk, or insufficient current
        capacity stops the continuous frontier immediately.
        """

        key = str(backup_key)
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"unknown AegisKV backup key {key!r}")
        version = int(expected_version)
        if version != entry.version:
            raise ValueError(
                "committed-prefix version does not match reservation: "
                f"commit={version}, reservation={entry.version}"
            )
        source_end = int(source_committed_token_end)
        if source_end < 0 or source_end > entry.token_count:
            raise ValueError(
                "source committed prefix is outside the reserved token range: "
                f"source_end={source_end}, reserved={entry.token_count}"
            )
        execution = str(execution_id).strip()
        if not execution:
            raise ValueError("committed-prefix update requires execution_id")
        generation = int(block_generation)
        if generation < 0:
            raise ValueError("committed-prefix update requires block_generation>=0")
        if entry.execution_id and execution != entry.execution_id:
            raise ValueError(
                "committed-prefix execution_id differs from prior commit"
            )
        if (
            entry.last_block_generation >= 0
            and generation != entry.last_block_generation
        ):
            raise ValueError(
                "committed-prefix block_generation differs from prior commit"
            )
        if source_end < entry.source_committed_token_end:
            raise ValueError(
                "source committed prefix regressed: "
                f"commit={source_end}, prior={entry.source_committed_token_end}"
            )

        l1_remaining = max(int(entry.l1_tokens), 0)
        l2_remaining = max(int(entry.l2_tokens), 0)
        materialized_l1 = 0
        materialized_l2 = 0
        frontier = 0
        for _chunk_id, raw_tokens, raw_tier in entry.planned_chunk_placements:
            chunk_tokens = max(int(raw_tokens), 0)
            tier = str(raw_tier).upper()
            chunk_end = frontier + chunk_tokens
            if chunk_tokens <= 0 or chunk_end > source_end:
                break
            if tier == "HBM" and l1_remaining >= chunk_tokens:
                l1_remaining -= chunk_tokens
                materialized_l1 += chunk_tokens
            elif tier == "HOST" and l2_remaining >= chunk_tokens:
                l2_remaining -= chunk_tokens
                materialized_l2 += chunk_tokens
            else:
                # A recompute chunk or a revoked/partial reservation is a real
                # prefix hole; later bytes cannot be advertised as reusable.
                break
            frontier = chunk_end

        entry.execution_id = execution
        entry.last_block_generation = generation
        entry.source_committed_token_end = source_end
        entry.materialized_token_end = frontier
        entry.materialized_l1_tokens = materialized_l1
        entry.materialized_l2_tokens = materialized_l2

        if materialized_l1 and materialized_l2:
            action = "l1_l2_backup"
        elif materialized_l1:
            action = "l1_backup"
        elif materialized_l2:
            action = "l2_backup"
        else:
            action = "recompute"
        recompute_tokens = max(source_end - frontier, 0)
        expected_ms = (
            (self.config.l1_restore_setup_ms if materialized_l1 else 0.0)
            + materialized_l1 * self.config.l1_restore_ms_per_token
            + (self.config.l2_restore_setup_ms if materialized_l2 else 0.0)
            + materialized_l2 * self.config.l2_restore_ms_per_token
            + (self.config.recompute_setup_ms if recompute_tokens else 0.0)
            + recompute_tokens * self.config.recompute_ms_per_token
        )
        self.sequence += 1
        return {
            "rid": entry.rid,
            "backup_key": entry.backup_key,
            "source_dp_rank": entry.source_dp_rank,
            "target_dp_rank": self.dp_rank,
            "version": entry.version,
            "ack": True,
            "token_count": source_end,
            "l1_tokens": materialized_l1,
            "l2_tokens": materialized_l2,
            "l3_tokens": 0,
            "recompute_tokens": recompute_tokens,
            "safe_frontier_tokens": frontier,
            "continuous_recoverable_prefix_tokens": frontier,
            "recovery_action": action,
            "expected_recovery_ms": round(expected_ms, 3),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "window_id": self.window_id,
            "sequence": self.sequence,
            "updated_time": self._now_fn(),
            "failure_scope": entry.failure_scope,
            "gpu_candidate": dict(entry.gpu_candidate),
            "host_candidate": dict(entry.host_candidate),
            "l3_candidate": dict(entry.l3_candidate),
            "workflow_id": entry.workflow_id,
            "agent_id": entry.agent_id,
            "worker_seq": entry.worker_seq,
            "placement_mode": entry.placement_mode,
            "exact_chunk_placements": entry.exact_chunk_placements,
            "solver_instance_sha256": entry.solver_instance_sha256,
            "solver_decision_sha256": entry.solver_decision_sha256,
            "solver_certificate_id": entry.solver_certificate_id,
            "materialization_scope": "committed_prefix",
            "source_committed_token_end": source_end,
            "group_commit_verified": True,
            "copy_completion_evidence": "sglsim_modeled_group_commit",
            "execution_id": execution,
            "block_generation": generation,
            "reserved_token_count": entry.token_count,
            "reserved_l1_tokens": max(int(entry.l1_tokens), 0),
            "reserved_l2_tokens": max(int(entry.l2_tokens), 0),
        }

    def clear_l1_for_gpu_failure(self) -> dict[str, Any]:
        """Invalidate rank-local L1 backup copies and retain L2 entries.

        The returned ``updated_hints`` replace Router ACK state before the rank
        is made healthy again, so a retry can never consume a stale GPU copy.
        """

        cleared_tokens = 0
        affected_entries = 0
        updated_hints: list[dict[str, Any]] = []
        for entry in self._entries.values():
            lost = max(int(entry.l1_tokens), 0)
            if lost <= 0:
                continue
            affected_entries += 1
            cleared_tokens += lost
            entry.l1_tokens = 0
            self._used_tokens["l1"] -= lost
            entry.recompute_tokens += lost
            updated_hints.append(self._entry_hint(entry))
        return {
            "dp_rank": self.dp_rank,
            "cleared_l1_backup_tokens": cleared_tokens,
            "affected_backup_entries": affected_entries,
            "preserved_l2_backup_tokens": self.l2_used_tokens,
            "updated_hints": updated_hints,
        }

    def invalidate_l2_for_host_lease(
        self,
        backup_key: str,
        *,
        expected_version: int,
    ) -> dict[str, Any] | None:
        """Remove an evicted controller-owned Host lease from rank-local state.

        The shared node pool is authoritative.  Version matching prevents a
        delayed eviction command from clearing a newer replacement of the same
        logical backup key.
        """

        entry = self._entries.get(str(backup_key))
        if entry is None or entry.version != int(expected_version):
            return None
        lost = max(int(entry.l2_tokens), 0)
        if lost <= 0:
            return self._entry_hint(entry)
        entry.l2_tokens = 0
        self._used_tokens["l2"] -= lost
        entry.recompute_tokens += lost
        return self._entry_hint(entry)

    def _window_close_trigger(self) -> str:
        legacy = max(int(self.config.window_requests), 1)
        maximum = max(int(self.config.window_max_requests), 0) or legacy
        minimum = max(int(self.config.window_min_requests), 0)
        if minimum <= 0:
            minimum = min(maximum, max(maximum // 4, 1))
        minimum = min(minimum, maximum)
        if self._window_requests >= maximum:
            return "max_requests"
        if self._window_requests < minimum:
            return ""
        overflow_trigger = max(
            int(self.config.window_overflow_trigger_tokens), 0
        )
        if overflow_trigger > 0 and sum(self._unmet.values()) >= overflow_trigger:
            return "overflow_shock"
        target_ms = max(float(self.config.window_target_ms), 0.0)
        elapsed_ms = (
            self._monotonic_fn() - self._window_started_monotonic
        ) * 1000.0
        if target_ms > 0.0 and elapsed_ms >= target_ms:
            return "target_time"
        return ""

    def _algorithm3_publication_frozen(self) -> bool:
        """Exact/certified DP budgets freeze stage-local Algorithm 3."""

        budget = self._external_budget
        if budget is None:
            return False
        return str(budget.solver_certificate_id) != "not-applicable-bootstrap"

    def close_window(self, *, trigger: str = "manual") -> dict[str, Any]:
        """Run replica-scoped alpha/beta control with joint projection."""

        old_alpha, old_beta = self.alpha, self.beta
        frozen = self._algorithm3_publication_frozen()
        values: dict[str, dict[str, float | int]] = {}
        for tier, total in (
            ("l1", self.config.l1_total_tokens),
            ("l2", self.config.l2_total_tokens),
        ):
            used = self.l1_used_tokens if tier == "l1" else self.l2_used_tokens
            demand = int(self._demand[tier])
            overflow = int(self._unmet[tier])
            # Predict next-window demand from this window's attempted work,
            # not cross-window resident occupancy.  Resident entries are
            # separately handled by explicit value-ordered revocation when a
            # published budget shrinks.
            signal = demand + overflow
            mu = _clamp(self.config.demand_ewma)
            predicted = (1.0 - mu) * self._predicted[tier] + mu * signal
            self._predicted[tier] = predicted
            occupied_floor = used / max(total, 1)
            safe = max(
                occupied_floor,
                1.0 - self._normal_pressure[tier] - self.config.safety_margin,
            )
            safe = _clamp(safe)
            configured_cap = (
                _clamp(self.config.max_alpha) if tier == "l1" else 1.0
            )
            safe = min(safe, configured_cap)
            raw = min(
                safe,
                _clamp(
                    (predicted + max(self.config.burst_reserve_tokens, 0))
                    / max(total, 1)
                ),
            )
            current = self.alpha if tier == "l1" else self.beta
            if frozen:
                # Certified exact windows keep the DP-published version.
                published = current
            else:
                eta = _clamp(self.config.publish_smoothing)
                published = (1.0 - eta) * current + eta * raw
                if abs(published - current) < max(self.config.hysteresis, 0.0):
                    published = current
                published = _clamp(published, high=configured_cap)
            if tier == "l1":
                self.alpha = published
            else:
                self.beta = published
            values[tier] = {
                **(
                    {
                        "attempted_tokens": demand,
                        "resident_end_tokens": used,
                        "unmet_tokens": overflow,
                    }
                    if self._external_budget is not None
                    else {
                        "D": demand,
                        "U": used,
                        "O": overflow,
                        "Z": signal,
                    }
                ),
                "predicted_Z": round(predicted, 3),
                "normal_pressure": round(self._normal_pressure[tier], 4),
                "safe_cap": round(safe, 4),
                "configured_cap": round(configured_cap, 4),
                "raw_ratio": round(raw, 4),
                "published_ratio": round(published, 4),
                "controller_scope": (
                    "dp_replica_pp_bottleneck"
                    if tier == "l1"
                    else "dp_replica_host_policy"
                ),
            }

        if not frozen:
            # Joint projection: beta must dominate alpha; pressure shrinks both.
            pressure = max(
                self._normal_pressure.get("l1", 0.0),
                self._normal_pressure.get("l2", 0.0),
            )
            self.alpha, self.beta = project_alpha_beta(
                self.alpha,
                self.beta,
                normal_pressure=pressure,
                previous_alpha=old_alpha,
                previous_beta=old_beta,
            )
            values["l1"]["published_ratio"] = round(self.alpha, 4)
            values["l2"]["published_ratio"] = round(self.beta, 4)
            if self._external_budget is not None:
                # Bootstrap DP budgets remain editable by Algorithm 3 so the
                # hard token caps track the projected ratios on each stage.
                self._external_l1_limit_tokens = int(
                    self.alpha * self.config.l1_total_tokens
                )
                self._external_l2_limit_tokens = int(
                    self.beta * self.config.l2_total_tokens
                )

        l1_evicted_keys = self._shrink_l1_to_budget()
        l2_evicted_keys = (
            self._shrink_tier_to_budget("l2") if not frozen else ()
        )

        elapsed_ms = (
            self._monotonic_fn() - self._window_started_monotonic
        ) * 1000.0
        self._stage_observation_window_id += 1
        if self._external_budget is None or not frozen:
            self.window_id += 1
        self._window_requests = 0
        self._demand = {"l1": 0, "l2": 0}
        self._unmet = {"l1": 0, "l2": 0}
        self._window_started_monotonic = self._monotonic_fn()
        return {
            "window_id": self.window_id,
            "stage_observation_window_id": self._stage_observation_window_id,
            "trigger": str(trigger),
            "elapsed_ms": round(elapsed_ms, 3),
            "alpha_scope": "dp_replica_pp_bottleneck",
            "beta_scope": "dp_replica_policy",
            "budget_version": (
                self._external_budget.budget_version
                if self._external_budget is not None
                else 0
            ),
            "budget_owner": (
                "dp_controller"
                if self._external_budget is not None
                else "legacy_rank_local"
            ),
            "hbm_policy_limit_tokens": self._tier_budget("l1"),
            "host_policy_limit_tokens": self._tier_budget("l2"),
            "old_alpha": round(old_alpha, 4),
            "new_alpha": round(self.alpha, 4),
            "old_beta": round(old_beta, 4),
            "new_beta": round(self.beta, 4),
            "tiers": values,
            "l1_evicted_backup_keys": list(l1_evicted_keys),
            "l2_evicted_backup_keys": list(l2_evicted_keys),
        }

    def snapshot(self) -> dict[str, Any]:
        state = {
            "dp_rank": self.dp_rank,
            "window_id": self.window_id,
            "window_requests": self._window_requests,
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "l1_total_tokens": self.config.l1_total_tokens,
            "l2_total_tokens": self.config.l2_total_tokens,
            "l1_budget_tokens": self._tier_budget("l1"),
            "l2_budget_tokens": self._tier_budget("l2"),
            "hbm_policy_limit_tokens": self._tier_budget("l1"),
            "host_policy_limit_tokens": self._tier_budget("l2"),
            "l1_used_tokens": self.l1_used_tokens,
            "l2_used_tokens": self.l2_used_tokens,
            "normal_l1_pressure": round(self._normal_pressure["l1"], 4),
            "normal_l2_pressure": round(self._normal_pressure["l2"], 4),
            "entries": len(self._entries),
            "window_elapsed_ms": round(
                (self._monotonic_fn() - self._window_started_monotonic)
                * 1000.0,
                3,
            ),
            "alpha_scope": "dp_replica_pp_bottleneck",
            "beta_scope": "dp_replica_policy",
            "budget_version": (
                self._external_budget.budget_version
                if self._external_budget is not None
                else 0
            ),
            "budget_owner": (
                "dp_controller"
                if self._external_budget is not None
                else "legacy_rank_local"
            ),
            "stage_observation_window_id": self._stage_observation_window_id,
            "critical_host_floor_fraction": round(
                _clamp(self.config.critical_host_floor_fraction), 4
            ),
            "critical_host_floor_probability": round(
                _clamp(self.config.critical_host_floor_probability), 4
            ),
        }
        if self._external_budget is not None:
            state.update(
                {
                    "attempted_l1_tokens": int(self._demand["l1"]),
                    "attempted_l2_tokens": int(self._demand["l2"]),
                    "resident_end_l1_tokens": self.l1_used_tokens,
                    "resident_end_l2_tokens": self.l2_used_tokens,
                    "unmet_l1_tokens": int(self._unmet["l1"]),
                    "unmet_l2_tokens": int(self._unmet["l2"]),
                }
            )
        else:
            state.update(
                {
                    "D_l1_tokens": int(self._demand["l1"]),
                    "D_l2_tokens": int(self._demand["l2"]),
                    "O_l1_tokens": int(self._unmet["l1"]),
                    "O_l2_tokens": int(self._unmet["l2"]),
                }
            )
        return state
