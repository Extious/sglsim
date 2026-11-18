"""Rank-local adapter for the latest independent-tier AegisKV controller.

The scheduler already has a simulator-only metadata command path for AegisKV.
This adapter preserves that command interface while replacing the historical
L1/L2 either-or policy with ``source_owned_independent_hbm_dram_v2``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from sglsim.srt.managers.aegis_l1_l2 import AegisBackupAck, AegisBackupPlan
from sglsim.srt.managers.pipeline_parallel import DPBudgetSnapshot
from sglsim.srt.simulation.aegis_independent_tiers import (
    DESIGN_SEMANTICS,
    AegisIndependentTierController,
    ChunkSpec,
    CopyState,
    IndependentTierConfig,
    PlacementDecision,
    ProtectionTier,
    RecoveryTiming,
    RequestRecoveryProfile,
    Residency,
)


def _as_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


@dataclass
class _LatestEntry:
    plan: AegisBackupPlan
    specs: tuple[ChunkSpec, ...]
    decisions: tuple[PlacementDecision, ...]
    execution_id: str = ""
    block_generation: int = -1
    source_committed_token_end: int = 0
    tier_committed_token_ends: dict[ProtectionTier, int] = field(
        default_factory=lambda: {
            ProtectionTier.HBM: 0,
            ProtectionTier.DRAM: 0,
        }
    )
    last_hint_signature: tuple[Any, ...] = field(default_factory=tuple)


def _normalize_completed_tiers(
    completed_tiers: object,
) -> tuple[ProtectionTier, ...]:
    """Return a deterministic, non-empty set of independently completed tiers.

    The physical-DES bridge uses the storage names ``hbm`` and ``dram``.  The
    ``l1``/``l2`` aliases are retained for callers that still expose the
    scheduler-facing names.  No other tier is legal here: in particular there
    is deliberately no target-HBM-to-DRAM materialization path.
    """

    if isinstance(completed_tiers, (str, ProtectionTier)):
        raw_tiers = (completed_tiers,)
    else:
        try:
            raw_tiers = tuple(completed_tiers)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("completed_tiers must be a non-empty tier list") from exc
    if not raw_tiers:
        raise ValueError("completed_tiers must be a non-empty tier list")

    aliases = {
        "hbm": ProtectionTier.HBM,
        "l1": ProtectionTier.HBM,
        "dram": ProtectionTier.DRAM,
        "l2": ProtectionTier.DRAM,
    }
    selected: set[ProtectionTier] = set()
    for raw_tier in raw_tiers:
        if isinstance(raw_tier, ProtectionTier):
            tier = raw_tier
        else:
            tier = aliases.get(str(raw_tier).strip().lower())
        if tier not in (ProtectionTier.HBM, ProtectionTier.DRAM):
            raise ValueError(
                f"unsupported completed AegisKV tier {raw_tier!r}; "
                "expected hbm or dram"
            )
        selected.add(tier)
    return tuple(
        tier
        for tier in (ProtectionTier.HBM, ProtectionTier.DRAM)
        if tier in selected
    )


class AegisIndependentTierRankManager:
    """Duck-typed replacement for ``AegisRankBackupManager`` in latest mode."""

    def __init__(
        self,
        *,
        dp_rank: int,
        raw_config: dict[str, Any],
        hbm_total_tokens: int,
        dram_total_tokens: int,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        if str(raw_config.get("design_semantics") or "") != DESIGN_SEMANTICS:
            raise ValueError("latest AegisKV manager requires explicit design semantics")
        if not bool(raw_config.get("committed_copy_protocol", False)):
            raise ValueError("latest AegisKV requires committed_copy_protocol=true")
        self.dp_rank = int(dp_rank)
        self._raw = dict(raw_config)
        self._now_fn = now_fn
        self.disable_slack_ordering = bool(
            raw_config.get("disable_slack_ordering", False)
        )
        legacy_chunk_tokens = raw_config.get("chunk_tokens")
        self.prefix_chunk_tokens = max(
            _as_int(
                raw_config.get("prefix_chunk_tokens", legacy_chunk_tokens),
                4096,
            ),
            1,
        )
        self.decode_chunk_tokens = max(
            _as_int(
                raw_config.get("decode_chunk_tokens", legacy_chunk_tokens),
                256,
            ),
            1,
        )
        # Retain the legacy diagnostic attribute for callers that inspect it.
        self.chunk_tokens = self.decode_chunk_tokens
        self.window_requests = max(
            _as_int(raw_config.get("window_requests"), 6), 1
        )
        self.hbm_pressure_threshold = min(
            max(_as_float(raw_config.get("hbm_pressure_threshold"), 0.90), 0.0),
            1.0,
        )
        dram_fraction = min(
            max(_as_float(raw_config.get("dram_static_fraction"), 1.0), 0.0),
            1.0,
        )
        hbm_capacity = max(int(hbm_total_tokens), 1)
        dram_capacity = max(int(dram_total_tokens * dram_fraction), 1)
        minimum_alpha = min(
            max(_as_float(raw_config.get("min_alpha"), 0.10), 0.0),
            1.0,
        )
        maximum_alpha = min(
            max(_as_float(raw_config.get("max_alpha"), 1.0), 0.0),
            1.0,
        )
        if minimum_alpha > maximum_alpha:
            raise ValueError("min_alpha must not exceed max_alpha")
        initial_alpha = min(
            max(
                _as_float(raw_config.get("initial_alpha"), 1.0),
                minimum_alpha,
            ),
            maximum_alpha,
        )
        controller_config = IndependentTierConfig(
            hbm_capacity_bytes=hbm_capacity,
            dram_capacity_bytes=dram_capacity,
            initial_hbm_alpha=initial_alpha,
            minimum_hbm_alpha=minimum_alpha,
            maximum_hbm_alpha=maximum_alpha,
            aimd_decrease_factor=_as_float(
                raw_config.get("aimd_decrease_factor"), 0.50
            ),
            aimd_increase_step=_as_float(
                raw_config.get("aimd_increase_step"), 0.10
            ),
            aimd_stable_windows=max(
                _as_int(raw_config.get("aimd_stable_windows"), 3), 1
            ),
            recompute=RecoveryTiming(
                _as_float(raw_config.get("recompute_setup_ms"), 4.0),
                _as_float(raw_config.get("recompute_ms_per_token"), 0.0108),
            ),
            hbm_load=RecoveryTiming(
                _as_float(raw_config.get("l1_restore_setup_ms"), 2.4),
                _as_float(raw_config.get("l1_restore_ms_per_token"), 0.0019),
            ),
            dram_load=RecoveryTiming(
                _as_float(raw_config.get("l2_restore_setup_ms"), 7.5),
                _as_float(raw_config.get("l2_restore_ms_per_token"), 0.0036),
            ),
            hbm_copy=RecoveryTiming(
                _as_float(raw_config.get("l1_write_setup_ms"), 0.12),
                _as_float(raw_config.get("l1_write_ms_per_token"), 0.00045),
            ),
            dram_copy=RecoveryTiming(
                _as_float(raw_config.get("l2_write_setup_ms"), 0.30),
                _as_float(raw_config.get("l2_write_ms_per_token"), 0.00120),
            ),
            hbm_pool_id=str(
                raw_config.get("hbm_pool_id") or f"peer-hbm-dp-{self.dp_rank}"
            ),
            dram_pool_id=str(raw_config.get("dram_pool_id") or "store-dram"),
            dram_failure_domain=str(
                raw_config.get("dram_failure_domain") or "store-dram-domain"
            ),
        )
        self.controller = AegisIndependentTierController(controller_config)
        self._entries: dict[str, _LatestEntry] = {}
        self._pending_hint_updates: list[dict[str, Any]] = []
        self._pending_physical_releases: list[dict[str, Any]] = []
        self._pending_physical_release_keys: set[tuple[Any, ...]] = set()
        self._window_requests = 0
        self._window_id = 0
        self._normal_l1_pressure = 0.0
        self._normal_l2_pressure = 0.0
        self._external_budget: DPBudgetSnapshot | None = None

    @staticmethod
    def _request_key_and_pp_rank(
        backup_key: str, fallback_pp_rank: int
    ) -> tuple[str, int]:
        root, marker, suffix = str(backup_key).rpartition("::pp")
        if marker and root and suffix.isdigit():
            return root, int(suffix)
        return str(backup_key), max(int(fallback_pp_rank), 0)

    def _queue_entry_release(
        self,
        entry: _LatestEntry,
        *,
        tier: ProtectionTier | None,
        ordinal: int | None,
        reason: str,
    ) -> None:
        root_key, pp_rank = self._request_key_and_pp_rank(
            entry.plan.backup_key, entry.plan.target_pp_rank
        )
        spec = None
        if ordinal is not None:
            spec = next(
                (
                    item
                    for item in entry.specs
                    if item.ordinal == int(ordinal)
                ),
                None,
            )
            if spec is None:
                raise ValueError(
                    f"unknown AegisKV chunk ordinal {ordinal} for "
                    f"{entry.plan.backup_key!r}"
                )
        key = (
            root_key,
            int(entry.plan.version),
            pp_rank,
            None if tier is None else ProtectionTier(tier).value,
            None if spec is None else int(spec.ordinal),
            str(reason),
        )
        if key in self._pending_physical_release_keys:
            return
        self._pending_physical_release_keys.add(key)
        self._pending_physical_releases.append(
            {
                "request_backup_key": root_key,
                "stage_backup_key": entry.plan.backup_key,
                "version": int(entry.plan.version),
                "route_version": int(entry.plan.route_version),
                "target_dp_rank": int(entry.plan.target_dp_rank),
                "pp_rank": pp_rank,
                "tier": (
                    None if tier is None else ProtectionTier(tier).value
                ),
                "ordinal": None if spec is None else int(spec.ordinal),
                "token_start": (
                    None if spec is None else int(spec.token_start)
                ),
                "token_end": None if spec is None else int(spec.token_end),
                "reason": str(reason),
            }
        )

    def _queue_logical_ids(
        self,
        logical_ids: object,
        *,
        tier: ProtectionTier,
        reason: str,
    ) -> None:
        for logical_id in logical_ids or ():
            request_id, marker, raw_ordinal = str(logical_id).rpartition(":")
            if not marker or not raw_ordinal.isdigit():
                raise ValueError(f"malformed AegisKV logical chunk id {logical_id!r}")
            entry = self._entries.get(request_id)
            if entry is None:
                # The controller can evict a just-admitted candidate before its
                # adapter entry is published.  It has no physical group yet.
                continue
            self._queue_entry_release(
                entry,
                tier=tier,
                ordinal=int(raw_ordinal),
                reason=reason,
            )

    def pop_pending_physical_releases(self) -> list[dict[str, Any]]:
        releases = self._pending_physical_releases
        self._pending_physical_releases = []
        self._pending_physical_release_keys.clear()
        return releases

    @property
    def alpha(self) -> float:
        return self.controller.hbm_alpha

    @property
    def beta(self) -> float:
        # DRAM is a static, independent pool in the latest design.
        snapshot = self.controller.snapshot()["pools"][ProtectionTier.DRAM.value]
        return float(snapshot["limit_bytes"]) / max(
            int(snapshot["capacity_bytes"]), 1
        )

    @property
    def l1_used_tokens(self) -> int:
        return int(
            self.controller.snapshot()["pools"][ProtectionTier.HBM.value][
                "used_bytes"
            ]
        )

    @property
    def l2_used_tokens(self) -> int:
        return int(
            self.controller.snapshot()["pools"][ProtectionTier.DRAM.value][
                "used_bytes"
            ]
        )

    def sealed_source_frontier(
        self,
        source_token_end: int,
        *,
        request_finished: bool,
        reserved_tokens: int,
    ) -> int:
        """Project the physical source frontier onto latest chunk boundaries."""

        source_end = min(
            max(int(source_token_end), 0), max(int(reserved_tokens), 0)
        )
        if request_finished:
            return source_end
        prefix_end = min(self.prefix_chunk_tokens, int(reserved_tokens))
        if source_end < prefix_end:
            return 0
        if source_end == prefix_end:
            return prefix_end
        return min(
            prefix_end
            + ((source_end - prefix_end) // self.decode_chunk_tokens)
            * self.decode_chunk_tokens,
            int(reserved_tokens),
        )

    def _identity(self, plan: AegisBackupPlan) -> tuple[str, str, str, str, str]:
        model_revision = plan.model_revision or "sglsim-mock-model-revision-v1"
        kv_abi = plan.kv_abi or "sglsim-metadata-kv-v1"
        layout = plan.parallel_layout or (
            f"dp-route-pp{max(plan.required_pp_size, 1)}"
        )
        source_domain = plan.source_failure_domain or (
            f"dp-rank-{plan.source_dp_rank}"
        )
        hbm_domain = plan.hbm_failure_domain or (
            f"dp-rank-{plan.target_dp_rank}"
        )
        return model_revision, kv_abi, layout, source_domain, hbm_domain

    def _request_profile(self, plan: AegisBackupPlan) -> RequestRecoveryProfile:
        model_revision, kv_abi, layout, source_domain, hbm_domain = (
            self._identity(plan)
        )
        return RequestRecoveryProfile(
            request_id=plan.backup_key,
            source_worker=f"dp-rank-{plan.source_dp_rank}",
            source_failure_domain=source_domain,
            hbm_target=f"dp-rank-{plan.target_dp_rank}",
            hbm_failure_domain=hbm_domain,
            route_version=plan.route_version,
            source_token_end=max(int(plan.token_count), 0),
            slack_ms=(
                0.0
                if self.disable_slack_ordering
                else max(float(plan.slack_ms), 0.0)
            ),
            model_revision=model_revision,
            kv_abi=kv_abi,
            parallel_layout=layout,
        )

    def _chunk_specs(self, plan: AegisBackupPlan) -> tuple[ChunkSpec, ...]:
        model_revision, kv_abi, layout, _source_domain, _hbm_domain = (
            self._identity(plan)
        )
        specs = []
        token_start = 0
        ordinal = 0
        while token_start < plan.token_count:
            chunk_tokens = (
                self.prefix_chunk_tokens
                if ordinal == 0
                else self.decode_chunk_tokens
            )
            token_end = min(token_start + chunk_tokens, plan.token_count)
            specs.append(
                ChunkSpec(
                    request_id=plan.backup_key,
                    ordinal=ordinal,
                    token_start=token_start,
                    token_end=token_end,
                    # Existing rank-local pools are token-accounted metadata
                    # pools, so one token is one capacity unit in this adapter.
                    byte_size=token_end - token_start,
                    route_version=plan.route_version,
                    source_worker=f"dp-rank-{plan.source_dp_rank}",
                    hbm_target=f"dp-rank-{plan.target_dp_rank}",
                    model_revision=model_revision,
                    kv_abi=kv_abi,
                    parallel_layout=layout,
                    # This manager is stage-local.  The outer DP controller
                    # collects one strict completion ACK per PP stage before
                    # publishing the request-level GroupCommitAck.
                    shard_count=1,
                )
            )
            token_start = token_end
            ordinal += 1
        return tuple(specs)

    @staticmethod
    def _residency(hbm_tokens: int, dram_tokens: int) -> Residency:
        if hbm_tokens > 0 and dram_tokens > 0:
            return Residency.HBM_AND_DRAM
        if hbm_tokens > 0:
            return Residency.HBM_ONLY
        if dram_tokens > 0:
            return Residency.DRAM_ONLY
        return Residency.BYPASS

    def _recovery_summary(
        self,
        entry: _LatestEntry,
        *,
        source_end: int,
    ) -> tuple[int, int, int, str, float, Residency]:
        hbm = min(
            self.controller.committed_prefix_tokens(
                entry.plan.backup_key, ProtectionTier.HBM
            ),
            source_end,
        )
        dram = min(
            self.controller.committed_prefix_tokens(
                entry.plan.backup_key, ProtectionTier.DRAM
            ),
            source_end,
        )
        cfg = self.controller.config

        def recovery_ms(tier: ProtectionTier | None, prefix: int) -> float:
            suffix = max(source_end - prefix, 0)
            value = 0.0
            if prefix > 0 and tier is not None:
                timing = cfg.hbm_load if tier is ProtectionTier.HBM else cfg.dram_load
                value += timing.setup_ms + prefix * timing.ms_per_token
            if suffix:
                value += (
                    cfg.recompute.setup_ms
                    + suffix * cfg.recompute.ms_per_token
                )
            return value

        candidates = [(recovery_ms(None, 0), "recompute", 0)]
        if hbm:
            candidates.append((recovery_ms(ProtectionTier.HBM, hbm), "l1_backup", hbm))
        if dram:
            candidates.append((recovery_ms(ProtectionTier.DRAM, dram), "l2_backup", dram))
        expected_ms, action, restored = min(
            candidates, key=lambda value: (value[0], -value[2], value[1])
        )
        safe = max(hbm, dram)
        return (
            hbm,
            dram,
            max(source_end - safe, 0),
            action,
            expected_ms,
            self._residency(hbm, dram),
        )

    def _hint(self, entry: _LatestEntry, *, committed_only: bool) -> dict[str, Any]:
        source_end = (
            entry.source_committed_token_end
            if committed_only
            else max(int(entry.plan.token_count), 0)
        )
        hbm, dram, recompute, action, expected_ms, residency = (
            self._recovery_summary(entry, source_end=source_end)
        )
        pending_hbm = sum(
            spec.token_count
            for spec in entry.specs
            if self.controller.copy_state(
                entry.plan.backup_key, spec.ordinal, ProtectionTier.HBM
            )
            is CopyState.PENDING
        )
        pending_dram = sum(
            spec.token_count
            for spec in entry.specs
            if self.controller.copy_state(
                entry.plan.backup_key, spec.ordinal, ProtectionTier.DRAM
            )
            is CopyState.PENDING
        )
        return {
            "rid": entry.plan.rid,
            "backup_key": entry.plan.backup_key,
            "source_dp_rank": entry.plan.source_dp_rank,
            "target_dp_rank": entry.plan.target_dp_rank,
            "version": entry.plan.version,
            "ack": True,
            "token_count": source_end,
            "l1_tokens": hbm,
            "l2_tokens": dram,
            "l3_tokens": 0,
            "recompute_tokens": recompute,
            "safe_frontier_tokens": max(hbm, dram),
            "continuous_recoverable_prefix_tokens": max(hbm, dram),
            "recovery_action": action,
            "expected_recovery_ms": round(expected_ms, 6),
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
            "window_id": self._window_id,
            "updated_time": self._now_fn(),
            "failure_scope": entry.plan.failure_scope,
            "gpu_candidate": dict(entry.plan.gpu_candidate),
            "host_candidate": dict(entry.plan.host_candidate),
            "l3_candidate": dict(entry.plan.l3_candidate),
            "workflow_id": entry.plan.workflow_id,
            "agent_id": entry.plan.agent_id,
            "worker_seq": entry.plan.worker_seq,
            "design_semantics": DESIGN_SEMANTICS,
            "residency": residency.value,
            "independent_tier_results": True,
            "hbm_pending_tokens": pending_hbm,
            "dram_pending_tokens": pending_dram,
            "reserved_l1_tokens": hbm + pending_hbm,
            "reserved_l2_tokens": dram + pending_dram,
            "hbm_copy_path": "source_d2d",
            "dram_copy_path": "source_d2h",
            "route_version": entry.plan.route_version,
            "source_owned": True,
            "cross_tier_transfer": False,
            "tier_committed_source_frontiers": {
                tier.value.lower(): int(entry.tier_committed_token_ends[tier])
                for tier in (ProtectionTier.HBM, ProtectionTier.DRAM)
            },
        }

    def _hint_signature(self, hint: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            hint.get(key)
            for key in (
                "l1_tokens",
                "l2_tokens",
                "recompute_tokens",
                "safe_frontier_tokens",
                "residency",
                "hbm_pending_tokens",
                "dram_pending_tokens",
            )
        )

    def _queue_changed_hints(self, *, exclude_key: str = "") -> None:
        for backup_key, entry in self._entries.items():
            if backup_key == exclude_key:
                continue
            hint = self._hint(entry, committed_only=True)
            signature = self._hint_signature(hint)
            if signature != entry.last_hint_signature:
                entry.last_hint_signature = signature
                self._pending_hint_updates.append(hint)

    def place(self, plan: AegisBackupPlan) -> AegisBackupAck:
        if plan.target_dp_rank != self.dp_rank:
            raise ValueError("backup plan target differs from local DP rank")
        current = self._entries.get(plan.backup_key)
        if current is not None and plan.version <= current.plan.version:
            hint = self._hint(current, committed_only=True)
            return self._ack_from_hint(current, hint)
        if current is not None:
            self._queue_entry_release(
                current,
                tier=None,
                ordinal=None,
                reason="backup_version_replaced",
            )
            self.controller.complete_request(
                plan.backup_key, force_abort_pending=True
            )
            self._entries.pop(plan.backup_key, None)
        self._normal_l1_pressure = min(max(plan.normal_l1_pressure, 0.0), 1.0)
        self._normal_l2_pressure = min(max(plan.normal_l2_pressure, 0.0), 1.0)
        self.controller.register_request(self._request_profile(plan))
        specs = self._chunk_specs(plan)
        allowed_tiers = {
            tier
            for name, tier in (
                ("l1", ProtectionTier.HBM),
                ("l2", ProtectionTier.DRAM),
            )
            if name in plan.allowed_tiers
        }
        decisions = tuple(
            self.controller.admit_chunk(
                spec,
                allowed_tiers=allowed_tiers,
            )
            for spec in specs
        )
        for decision in decisions:
            self._queue_logical_ids(
                decision.hbm.victim_ids,
                tier=ProtectionTier.HBM,
                reason="slack_landlord_hbm_eviction",
            )
            self._queue_logical_ids(
                decision.dram.victim_ids,
                tier=ProtectionTier.DRAM,
                reason="slack_landlord_dram_eviction",
            )
        entry = _LatestEntry(plan=plan, specs=specs, decisions=decisions)
        self._entries[plan.backup_key] = entry
        self._window_requests += 1
        self._queue_changed_hints(exclude_key=plan.backup_key)

        reserved_hbm = sum(
            spec.token_count
            for spec, decision in zip(specs, decisions)
            if decision.hbm.admitted
        )
        reserved_dram = sum(
            spec.token_count
            for spec, decision in zip(specs, decisions)
            if decision.dram.admitted
        )
        demand = max(int(plan.token_count), 0)
        window_update = None
        if self._window_requests >= self.window_requests:
            window_update = self.close_window(trigger="max_requests")
        return AegisBackupAck(
            rid=plan.rid,
            backup_key=plan.backup_key,
            source_dp_rank=plan.source_dp_rank,
            target_dp_rank=plan.target_dp_rank,
            version=plan.version,
            ack=True,
            # These fields represent reserved final slots on the reservation
            # ACK.  ``safe_frontier_tokens=0`` and committed-copy validation
            # prevent them from being consumed as recovery evidence.
            l1_tokens=reserved_hbm,
            l2_tokens=reserved_dram,
            recompute_tokens=demand,
            safe_frontier_tokens=0,
            recovery_action="recompute",
            expected_recovery_ms=round(
                (
                    self.controller.config.recompute.setup_ms
                    + demand * self.controller.config.recompute.ms_per_token
                    if demand
                    else 0.0
                ),
                6,
            ),
            alpha=round(self.alpha, 6),
            beta=round(self.beta, 6),
            window_id=self._window_id,
            d_l1_tokens=demand,
            d_l2_tokens=demand,
            u_l1_tokens=self.l1_used_tokens,
            u_l2_tokens=self.l2_used_tokens,
            o_l1_tokens=max(demand - reserved_hbm, 0),
            o_l2_tokens=max(demand - reserved_dram, 0),
            sequence=len(self.controller.snapshot()["events"]),
            updated_time=self._now_fn(),
            window_update=window_update,
            failure_scope=plan.failure_scope,
            gpu_candidate=dict(plan.gpu_candidate),
            host_candidate=dict(plan.host_candidate),
            l3_candidate=dict(plan.l3_candidate),
            decision_d_l1_tokens=demand,
            decision_d_l2_tokens=demand,
            decision_u_l1_tokens=reserved_hbm,
            decision_u_l2_tokens=reserved_dram,
            decision_o_l1_tokens=max(demand - reserved_hbm, 0),
            decision_o_l2_tokens=max(demand - reserved_dram, 0),
            workflow_id=plan.workflow_id,
            agent_id=plan.agent_id,
            worker_seq=plan.worker_seq,
            placement_mode=(
                "cost_landlord_no_slack"
                if self.disable_slack_ordering
                else "slack_landlord_independent_tiers"
            ),
            design_semantics=DESIGN_SEMANTICS,
            residency=Residency.BYPASS.value,
            hbm_pending_tokens=reserved_hbm,
            dram_pending_tokens=reserved_dram,
            reserved_l1_tokens=reserved_hbm,
            reserved_l2_tokens=reserved_dram,
            hbm_copy_path="source_d2d",
            dram_copy_path="source_d2h",
            independent_tier_results=True,
        )

    def _ack_from_hint(
        self, entry: _LatestEntry, hint: dict[str, Any]
    ) -> AegisBackupAck:
        return AegisBackupAck(
            rid=entry.plan.rid,
            backup_key=entry.plan.backup_key,
            source_dp_rank=entry.plan.source_dp_rank,
            target_dp_rank=entry.plan.target_dp_rank,
            version=entry.plan.version,
            ack=True,
            l1_tokens=int(hint["l1_tokens"]),
            l2_tokens=int(hint["l2_tokens"]),
            recompute_tokens=int(hint["recompute_tokens"]),
            safe_frontier_tokens=int(hint["safe_frontier_tokens"]),
            recovery_action=str(hint["recovery_action"]),
            expected_recovery_ms=float(hint["expected_recovery_ms"]),
            alpha=float(hint["alpha"]),
            beta=float(hint["beta"]),
            window_id=self._window_id,
            d_l1_tokens=0,
            d_l2_tokens=0,
            u_l1_tokens=self.l1_used_tokens,
            u_l2_tokens=self.l2_used_tokens,
            o_l1_tokens=0,
            o_l2_tokens=0,
            sequence=len(self.controller.snapshot()["events"]),
            updated_time=self._now_fn(),
            failure_scope=entry.plan.failure_scope,
            gpu_candidate=dict(entry.plan.gpu_candidate),
            host_candidate=dict(entry.plan.host_candidate),
            l3_candidate=dict(entry.plan.l3_candidate),
            workflow_id=entry.plan.workflow_id,
            agent_id=entry.plan.agent_id,
            worker_seq=entry.plan.worker_seq,
            placement_mode=(
                "cost_landlord_no_slack"
                if self.disable_slack_ordering
                else "slack_landlord_independent_tiers"
            ),
            design_semantics=DESIGN_SEMANTICS,
            residency=str(hint["residency"]),
            hbm_pending_tokens=int(hint["hbm_pending_tokens"]),
            dram_pending_tokens=int(hint["dram_pending_tokens"]),
            reserved_l1_tokens=int(hint["reserved_l1_tokens"]),
            reserved_l2_tokens=int(hint["reserved_l2_tokens"]),
            hbm_copy_path="source_d2d",
            dram_copy_path="source_d2h",
            independent_tier_results=True,
        )

    def materialize_committed_prefix(
        self,
        backup_key: str,
        *,
        expected_version: int,
        source_committed_token_end: int,
        execution_id: str,
        block_generation: int,
        completed_tiers: object | None = None,
    ) -> dict[str, Any]:
        """Commit completed source-owned copies without coupling HBM and DRAM.

        ``completed_tiers=None`` preserves the historical simulator command:
        every admitted tier is modeled as completed at the supplied frontier.
        The offline physical-DES path must instead pass exactly the tier(s)
        whose atomic TP-shard group has completed.  Each tier has its own
        monotonic frontier, so differently paced HBM and DRAM groups can ACK
        out of order without one tier manufacturing evidence for the other.
        """

        entry = self._entries.get(str(backup_key))
        if entry is None:
            raise KeyError(f"unknown latest AegisKV backup key {backup_key!r}")
        if int(expected_version) != entry.plan.version:
            raise ValueError("committed-prefix version does not match reservation")
        source_end = int(source_committed_token_end)
        if source_end < 0 or source_end > entry.plan.token_count:
            raise ValueError("source committed prefix is outside reservation")
        compatibility_all_tiers = completed_tiers is None
        selected_tiers = (
            (ProtectionTier.HBM, ProtectionTier.DRAM)
            if compatibility_all_tiers
            else _normalize_completed_tiers(completed_tiers)
        )
        if compatibility_all_tiers and source_end < entry.source_committed_token_end:
            raise ValueError("source committed prefix regressed")
        for tier in selected_tiers:
            if source_end < entry.tier_committed_token_ends[tier]:
                raise ValueError(
                    f"{tier.value.lower()} committed prefix regressed: "
                    f"commit={source_end}, "
                    f"prior={entry.tier_committed_token_ends[tier]}"
                )
        if not str(execution_id):
            raise ValueError("committed-prefix update requires execution_id")
        if int(block_generation) < 0:
            raise ValueError("committed-prefix update requires block_generation>=0")
        if entry.execution_id and entry.execution_id != str(execution_id):
            raise ValueError("committed-prefix execution_id differs from prior commit")
        if entry.block_generation >= 0 and entry.block_generation != int(
            block_generation
        ):
            raise ValueError("committed-prefix block_generation differs from prior commit")
        entry.execution_id = str(execution_id)
        entry.block_generation = int(block_generation)
        for spec, decision in zip(entry.specs, entry.decisions):
            if spec.token_end > source_end:
                break
            for tier, admission in (
                (ProtectionTier.HBM, decision.hbm),
                (ProtectionTier.DRAM, decision.dram),
            ):
                if tier not in selected_tiers:
                    continue
                if not admission.admitted:
                    continue
                if self.controller.copy_state(
                    entry.plan.backup_key, spec.ordinal, tier
                ) is CopyState.PENDING:
                    self.controller.complete_copy(
                        tier,
                        entry.plan.backup_key,
                        spec.ordinal,
                        self.controller.completion_proof(spec, tier),
                    )
        for tier in selected_tiers:
            entry.tier_committed_token_ends[tier] = source_end
        entry.source_committed_token_end = max(
            entry.source_committed_token_end,
            source_end,
        )
        hint = self._hint(entry, committed_only=True)
        entry.last_hint_signature = self._hint_signature(hint)
        hint.update(
            {
                "materialization_scope": "committed_prefix",
                "source_committed_token_end": entry.source_committed_token_end,
                "tier_source_committed_token_end": source_end,
                "completed_tiers": [
                    tier.value.lower() for tier in selected_tiers
                ],
                "tier_commit_scope": (
                    "all_admitted_tiers_compatibility"
                    if compatibility_all_tiers
                    else "physical_completed_tiers"
                ),
                "group_commit_verified": True,
                "copy_completion_evidence": (
                    "sglsim_modeled_independent_source_tier_commit"
                    if compatibility_all_tiers
                    else "sglsim_physical_independent_tier_group_commit"
                ),
                "execution_id": entry.execution_id,
                "block_generation": entry.block_generation,
            }
        )
        return hint

    def apply_dp_budget(
        self,
        snapshot: DPBudgetSnapshot,
        *,
        pp_rank: int,
    ) -> dict[str, Any]:
        if snapshot.dp_rank != self.dp_rank:
            raise ValueError("DP budget belongs to another replica")
        if self._external_budget is not None:
            if snapshot.budget_version < self._external_budget.budget_version:
                raise ValueError("cannot apply a stale DP budget version")
            if snapshot.budget_version == self._external_budget.budget_version:
                if snapshot != self._external_budget:
                    raise ValueError("conflicting DP budget payload for one version")
                return {
                    "applied": False,
                    "budget_version": snapshot.budget_version,
                    "window_id": snapshot.window_id,
                    "evicted_l1_backup_keys": [],
                    "evicted_l2_backup_keys": [],
                }
        self._external_budget = snapshot
        hbm = self.controller.set_pool_limit(
            ProtectionTier.HBM, snapshot.hbm_limit_for_stage(pp_rank)
        )
        dram = self.controller.set_pool_limit(
            ProtectionTier.DRAM, snapshot.host_policy_limit_tokens
        )
        self._queue_logical_ids(
            (*hbm["aborted_pending"], *hbm["evicted_tails"]),
            tier=ProtectionTier.HBM,
            reason="dp_budget_hbm_reclaim",
        )
        self._queue_logical_ids(
            (*dram["aborted_pending"], *dram["evicted_tails"]),
            tier=ProtectionTier.DRAM,
            reason="dp_budget_dram_reclaim",
        )
        self._window_id = int(snapshot.window_id)
        self._queue_changed_hints()
        return {
            "applied": True,
            "budget_version": snapshot.budget_version,
            "window_id": snapshot.window_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "hbm_limit_tokens": hbm["limit_bytes"],
            "host_policy_limit_tokens": dram["limit_bytes"],
            "objective_id": snapshot.objective_id,
            "solver_certificate_id": snapshot.solver_certificate_id,
            "evicted_l1_backup_keys": list(hbm["evicted_tails"]),
            "evicted_l2_backup_keys": list(dram["evicted_tails"]),
        }

    def close_window(self, *, trigger: str = "manual") -> dict[str, Any]:
        pressure = (
            self._normal_l1_pressure >= self.hbm_pressure_threshold
            or self.l1_used_tokens
            > self.controller.snapshot()["pools"][ProtectionTier.HBM.value][
                "limit_bytes"
            ]
        )
        update = self.controller.control_hbm_quota(pressure=pressure)
        self._queue_logical_ids(
            (*update["aborted_pending"], *update["evicted_hbm_tails"]),
            tier=ProtectionTier.HBM,
            reason="hbm_quota_aimd_reclaim",
        )
        self._window_requests = 0
        self._window_id += 1
        self._queue_changed_hints()
        return {
            "window_id": self._window_id,
            "trigger": trigger,
            "old_alpha": update["old_alpha"],
            "new_alpha": update["new_alpha"],
            "old_beta": self.beta,
            "new_beta": self.beta,
            "hbm_policy_limit_tokens": update["quota_bytes"],
            "host_policy_limit_tokens": self.controller.snapshot()["pools"]
            [ProtectionTier.DRAM.value]["limit_bytes"],
            "tiers": {
                "l1": {
                    "normal_pressure": self._normal_l1_pressure,
                    "published_ratio": update["new_alpha"],
                    "controller_scope": "hbm_quota_aimd",
                },
                "l2": {
                    "normal_pressure": self._normal_l2_pressure,
                    "published_ratio": self.beta,
                    "controller_scope": "independent_static_dram_pool",
                },
            },
            "l1_evicted_backup_keys": list(update["evicted_hbm_tails"]),
            "l2_evicted_backup_keys": [],
            "design_semantics": DESIGN_SEMANTICS,
        }

    def pop_pending_hint_updates(self) -> list[dict[str, Any]]:
        updates = self._pending_hint_updates
        self._pending_hint_updates = []
        return updates

    def clear_l1_for_gpu_failure(self) -> dict[str, Any]:
        cleared = 0
        affected = 0
        hints = []
        for backup_key, entry in self._entries.items():
            reserved = sum(
                spec.token_count
                for spec in entry.specs
                if self.controller.copy_state(
                    backup_key, spec.ordinal, ProtectionTier.HBM
                )
                is not None
            )
            if reserved <= 0:
                continue
            affected += 1
            self._queue_entry_release(
                entry,
                tier=ProtectionTier.HBM,
                ordinal=None,
                reason="gpu_failure_invalidated_peer_hbm",
            )
            result = self.controller.invalidate_tier(
                backup_key,
                ProtectionTier.HBM,
                reason="gpu_failure_invalidated_peer_hbm",
            )
            cleared += int(result["released_bytes"])
            hint = self._hint(entry, committed_only=True)
            entry.last_hint_signature = self._hint_signature(hint)
            hints.append(hint)
        return {
            "dp_rank": self.dp_rank,
            "cleared_l1_backup_tokens": cleared,
            "affected_backup_entries": affected,
            "preserved_l2_backup_tokens": self.l2_used_tokens,
            "updated_hints": hints,
            "design_semantics": DESIGN_SEMANTICS,
        }

    def invalidate_l2_for_host_lease(
        self,
        backup_key: str,
        *,
        expected_version: int,
    ) -> dict[str, Any] | None:
        entry = self._entries.get(str(backup_key))
        if entry is None or entry.plan.version != int(expected_version):
            return None
        self._queue_entry_release(
            entry,
            tier=ProtectionTier.DRAM,
            ordinal=None,
            reason="controller_host_pool_eviction",
        )
        self.controller.invalidate_tier(
            entry.plan.backup_key,
            ProtectionTier.DRAM,
            reason="controller_host_pool_eviction",
        )
        hint = self._hint(entry, committed_only=True)
        entry.last_hint_signature = self._hint_signature(hint)
        return hint

    def supersede(
        self,
        backup_key: str,
        *,
        expected_version: int,
        superseded_by: str,
    ) -> dict[str, Any]:
        entry = self._entries.get(str(backup_key))
        if entry is None or entry.plan.version != int(expected_version):
            return {
                "backup_key": str(backup_key),
                "expected_version": int(expected_version),
                "superseded_by": str(superseded_by),
                "released": False,
                "released_l1_tokens": 0,
                "released_l2_tokens": 0,
            }
        hbm = self.l1_used_tokens
        dram = self.l2_used_tokens
        self._queue_entry_release(
            entry,
            tier=None,
            ordinal=None,
            reason="backup_frontier_superseded",
        )
        self.controller.complete_request(
            entry.plan.backup_key, force_abort_pending=True
        )
        self._entries.pop(entry.plan.backup_key, None)
        return {
            "backup_key": str(backup_key),
            "expected_version": int(expected_version),
            "superseded_by": str(superseded_by),
            "released": True,
            "released_l1_tokens": max(hbm - self.l1_used_tokens, 0),
            "released_l2_tokens": max(dram - self.l2_used_tokens, 0),
            "workflow_id": entry.plan.workflow_id,
            "agent_id": entry.plan.agent_id,
            "worker_seq": entry.plan.worker_seq,
            "design_semantics": DESIGN_SEMANTICS,
        }

    def complete_request(
        self,
        backup_key: str,
        *,
        expected_version: int,
    ) -> dict[str, Any]:
        """Apply Router ``CompleteRequest`` after physical groups are terminal."""

        entry = self._entries.get(str(backup_key))
        if entry is None or entry.plan.version != int(expected_version):
            return {
                "backup_key": str(backup_key),
                "expected_version": int(expected_version),
                "completed": False,
                "released_l1_tokens": 0,
                "released_l2_tokens": 0,
            }
        hbm_before = self.l1_used_tokens
        dram_before = self.l2_used_tokens
        self._queue_entry_release(
            entry,
            tier=None,
            ordinal=None,
            reason="request_completed",
        )
        result = self.controller.complete_request(
            entry.plan.backup_key,
            force_abort_pending=True,
        )
        self._entries.pop(entry.plan.backup_key, None)
        return {
            "backup_key": str(backup_key),
            "expected_version": int(expected_version),
            "completed": bool(result.get("completed", False)),
            "released_l1_tokens": max(hbm_before - self.l1_used_tokens, 0),
            "released_l2_tokens": max(dram_before - self.l2_used_tokens, 0),
            "workflow_id": entry.plan.workflow_id,
            "agent_id": entry.plan.agent_id,
            "worker_seq": entry.plan.worker_seq,
            "design_semantics": DESIGN_SEMANTICS,
        }

    def snapshot(self) -> dict[str, Any]:
        raw = self.controller.snapshot()
        hbm = raw["pools"][ProtectionTier.HBM.value]
        dram = raw["pools"][ProtectionTier.DRAM.value]
        return {
            **raw,
            "dp_rank": self.dp_rank,
            "window_id": self._window_id,
            "window_requests": self._window_requests,
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
            "l1_total_tokens": int(hbm["capacity_bytes"]),
            "l2_total_tokens": int(dram["capacity_bytes"]),
            "l1_budget_tokens": int(hbm["limit_bytes"]),
            "l2_budget_tokens": int(dram["limit_bytes"]),
            "hbm_policy_limit_tokens": int(hbm["limit_bytes"]),
            "host_policy_limit_tokens": int(dram["limit_bytes"]),
            "l1_used_tokens": int(hbm["used_bytes"]),
            "l2_used_tokens": int(dram["used_bytes"]),
            "entries": len(self._entries),
            "budget_version": (
                self._external_budget.budget_version
                if self._external_budget is not None
                else 0
            ),
            "budget_owner": (
                "dp_controller"
                if self._external_budget is not None
                else "hbm_quota_aimd_and_static_dram"
            ),
        }
