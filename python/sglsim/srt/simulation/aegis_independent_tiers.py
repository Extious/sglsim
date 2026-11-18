"""Latest AegisKV metadata control plane for independent HBM/DRAM copies.

This module implements the ``source_owned_independent_hbm_dram_v2`` design.
It deliberately does not change SGLang scheduling or batch lifecycle: SGLSim
models KV bytes as metadata, while this controller models the policy and
commit boundaries that determine whether those bytes are recoverable.

The important invariants are:

* HBM and DRAM admission, quota, rent, victims, copy state, and completion are
  independent.  Neither tier is a prerequisite or fallback for the other.
* every copy is owned by the live source and moves directly from source
  buffers to the final HBM or DRAM destination;
* PENDING copies reserve capacity but have no recovery value;
* a pool keeps a continuous per-request prefix and exposes at most one safe
  suffix tail per request to its global lazy Landlord heap;
* Slack updates only invalidate metadata keys; they never trigger movement;
* HBM quota contraction can directly remove HBM_ONLY or HBM_AND_DRAM tails;
* recovery uses only complete, version-compatible COMMITTED prefixes.

The implementation is intentionally self-contained so the legacy L1/L2 path
can remain available for historical experiments while new runs opt in to the
latest design explicitly.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Iterable


DESIGN_SEMANTICS = "source_owned_independent_hbm_dram_v2"

_STORAGE_EVICTION_REASONS = frozenset(
    {
        "slack_landlord_victim",
        "external_hbm_limit_contraction",
        "external_dram_limit_contraction",
        "hbm_quota_contraction",
    }
)


def _finite_nonnegative(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class ProtectionTier(str, Enum):
    HBM = "HBM"
    DRAM = "DRAM"


class CopyState(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


class Residency(str, Enum):
    BYPASS = "BYPASS"
    HBM_ONLY = "HBM_ONLY"
    DRAM_ONLY = "DRAM_ONLY"
    HBM_AND_DRAM = "HBM_AND_DRAM"


@dataclass(frozen=True)
class RecoveryTiming:
    setup_ms: float
    ms_per_token: float

    def __post_init__(self) -> None:
        _finite_nonnegative("setup_ms", self.setup_ms)
        _finite_nonnegative("ms_per_token", self.ms_per_token)


@dataclass(frozen=True)
class IndependentTierConfig:
    hbm_capacity_bytes: int
    dram_capacity_bytes: int
    initial_hbm_alpha: float = 1.0
    minimum_hbm_alpha: float = 0.10
    maximum_hbm_alpha: float = 1.0
    aimd_decrease_factor: float = 0.50
    aimd_increase_step: float = 0.10
    aimd_stable_windows: int = 3
    recompute: RecoveryTiming = RecoveryTiming(4.0, 0.0108)
    hbm_load: RecoveryTiming = RecoveryTiming(2.4, 0.0019)
    dram_load: RecoveryTiming = RecoveryTiming(7.5, 0.0036)
    hbm_copy: RecoveryTiming = RecoveryTiming(0.12, 0.00045)
    dram_copy: RecoveryTiming = RecoveryTiming(0.30, 0.00120)
    hbm_pool_id: str = "peer-hbm"
    dram_pool_id: str = "store-dram"
    dram_failure_domain: str = "store-dram-domain"
    retain_event_log: bool = True
    retain_record_history: bool = True
    event_log_limit: int = 0

    def __post_init__(self) -> None:
        _positive_int("hbm_capacity_bytes", self.hbm_capacity_bytes)
        _positive_int("dram_capacity_bytes", self.dram_capacity_bytes)
        for name, value in (
            ("initial_hbm_alpha", self.initial_hbm_alpha),
            ("minimum_hbm_alpha", self.minimum_hbm_alpha),
            ("maximum_hbm_alpha", self.maximum_hbm_alpha),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not (
            self.minimum_hbm_alpha
            <= self.initial_hbm_alpha
            <= self.maximum_hbm_alpha
        ):
            raise ValueError("initial HBM alpha must lie within its bounds")
        if not 0.0 < float(self.aimd_decrease_factor) < 1.0:
            raise ValueError("aimd_decrease_factor must be in (0, 1)")
        if not math.isfinite(float(self.aimd_increase_step)) or float(
            self.aimd_increase_step
        ) <= 0.0:
            raise ValueError("aimd_increase_step must be finite and positive")
        _positive_int("aimd_stable_windows", self.aimd_stable_windows)
        if not self.hbm_pool_id or not self.dram_pool_id:
            raise ValueError("pool IDs must be nonempty")
        if not self.dram_failure_domain:
            raise ValueError("dram_failure_domain must be nonempty")
        if isinstance(self.event_log_limit, bool) or int(self.event_log_limit) < 0:
            raise ValueError("event_log_limit must be a nonnegative integer")


@dataclass(frozen=True)
class RequestRecoveryProfile:
    request_id: str
    source_worker: str
    source_failure_domain: str
    hbm_target: str
    hbm_failure_domain: str
    route_version: int
    source_token_end: int
    slack_ms: float
    model_revision: str
    kv_abi: str
    parallel_layout: str

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("source_worker", self.source_worker),
            ("source_failure_domain", self.source_failure_domain),
            ("hbm_target", self.hbm_target),
            ("hbm_failure_domain", self.hbm_failure_domain),
            ("model_revision", self.model_revision),
            ("kv_abi", self.kv_abi),
            ("parallel_layout", self.parallel_layout),
        ):
            if not str(value):
                raise ValueError(f"{name} must be nonempty")
        if isinstance(self.route_version, bool) or int(self.route_version) < 0:
            raise ValueError("route_version must be a nonnegative integer")
        if isinstance(self.source_token_end, bool) or int(self.source_token_end) < 0:
            raise ValueError("source_token_end must be a nonnegative integer")
        _finite_nonnegative("slack_ms", self.slack_ms)


@dataclass(frozen=True)
class ChunkSpec:
    request_id: str
    ordinal: int
    token_start: int
    token_end: int
    byte_size: int
    route_version: int
    source_worker: str
    hbm_target: str
    model_revision: str
    kv_abi: str
    parallel_layout: str
    shard_count: int = 1

    def __post_init__(self) -> None:
        if not self.request_id or not self.source_worker or not self.hbm_target:
            raise ValueError("chunk request/source/target identities are required")
        if not self.model_revision or not self.kv_abi or not self.parallel_layout:
            raise ValueError("chunk compatibility identity is required")
        if isinstance(self.ordinal, bool) or int(self.ordinal) < 0:
            raise ValueError("chunk ordinal must be a nonnegative integer")
        if int(self.token_start) < 0 or int(self.token_end) <= int(self.token_start):
            raise ValueError("chunk token interval must be nonempty and ordered")
        _positive_int("byte_size", self.byte_size)
        if isinstance(self.route_version, bool) or int(self.route_version) < 0:
            raise ValueError("route_version must be a nonnegative integer")
        _positive_int("shard_count", self.shard_count)

    @property
    def token_count(self) -> int:
        return int(self.token_end) - int(self.token_start)

    @property
    def logical_id(self) -> str:
        return f"{self.request_id}:{self.ordinal}"


@dataclass(frozen=True)
class CopyCompletionProof:
    tier: ProtectionTier
    pending_id: str
    copy_kind: str
    source_worker: str
    destination_id: str
    route_version: int
    model_revision: str
    kv_abi: str
    parallel_layout: str
    transferred_bytes: int
    acked_shards: tuple[int, ...]
    transfer_succeeded: bool = True
    source_owned: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", ProtectionTier(self.tier))
        for name in ("pending_id", "copy_kind", "source_worker", "destination_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"copy completion {name} must be nonempty")


@dataclass
class TierCopyRecord:
    spec: ChunkSpec
    tier: ProtectionTier
    pool_id: str
    state: CopyState
    gain_ms: float
    admitted_at_ms: float
    copy_kind: str
    source_owned: bool = True
    landlord_key: float = 0.0
    last_loss_ms: float = 0.0
    committed_at_ms: float | None = None
    aborted_at_ms: float | None = None
    abort_reason: str = ""
    base_hbm_chunk_id: str | None = None
    writeback_cost_ms: float = 0.0
    completion_validated: bool = False

    @property
    def key(self) -> tuple[str, int]:
        return (self.spec.request_id, self.spec.ordinal)


@dataclass(frozen=True)
class TierAdmission:
    tier: ProtectionTier
    admitted: bool
    reason: str
    gain_ms: float
    threshold: float
    victim_ids: tuple[str, ...]
    pending_id: str
    copy_kind: str
    source_owned: bool = True


@dataclass(frozen=True)
class PlacementDecision:
    request_id: str
    ordinal: int
    hbm: TierAdmission
    dram: TierAdmission

    @property
    def intended_residency(self) -> Residency:
        if self.hbm.admitted and self.dram.admitted:
            return Residency.HBM_AND_DRAM
        if self.hbm.admitted:
            return Residency.HBM_ONLY
        if self.dram.admitted:
            return Residency.DRAM_ONLY
        return Residency.BYPASS


@dataclass(frozen=True)
class TierCompletion:
    tier: ProtectionTier
    request_id: str
    ordinal: int
    state: CopyState
    reason: str
    residency: Residency


@dataclass(frozen=True)
class VictimFrontier:
    """The sole globally exposed safe suffix tail for one request/pool."""

    tier: ProtectionTier
    pool_id: str
    request_id: str
    ordinal: int
    logical_id: str
    landlord_key: float
    slack_epoch: int


@dataclass(frozen=True)
class RecoveryDecision:
    request_id: str
    action: str
    restored_tokens: int
    recompute_tokens: int
    recovery_ms: float
    slack_ms: float
    visible_harm_ms: float
    route_version: int
    activated_target: str


@dataclass
class _RequestState:
    profile: RequestRecoveryProfile
    slack_epoch: int = 0


@dataclass
class _PoolState:
    tier: ProtectionTier
    pool_id: str
    capacity_bytes: int
    limit_bytes: int
    rent: float = 0.0
    used_bytes: int = 0
    records: dict[tuple[str, int], TierCopyRecord] = field(default_factory=dict)
    live_ordinals: dict[str, list[int]] = field(default_factory=dict)
    # These two request-local frontiers make the common append/commit path
    # O(1).  The live list remains authoritative for validation and suffix
    # eviction; the caches never change which chunk is admissible or
    # recoverable.
    committed_prefix_count: dict[str, int] = field(default_factory=dict)
    committed_frontier_tokens: dict[str, int] = field(default_factory=dict)
    tail_version: dict[str, int] = field(default_factory=dict)
    heap: list[tuple[float, int, str, int, int, int, int, int]] = field(
        default_factory=list
    )
    admission_attempt_count: int = 0
    admission_success_count: int = 0
    admission_rejection_count: int = 0
    admitted_bytes: int = 0
    eviction_count: int = 0
    evicted_bytes: int = 0
    removal_count: int = 0
    removed_bytes: int = 0
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)
    eviction_reason_counts: dict[str, int] = field(default_factory=dict)
    removal_reason_counts: dict[str, int] = field(default_factory=dict)


class AegisIndependentTierController:
    """Metadata implementation of latest AegisKV Algorithm 2 and 3."""

    def __init__(self, config: IndependentTierConfig) -> None:
        self.config = config
        hbm_limit = int(config.hbm_capacity_bytes * config.initial_hbm_alpha)
        self._pools = {
            ProtectionTier.HBM: _PoolState(
                tier=ProtectionTier.HBM,
                pool_id=config.hbm_pool_id,
                capacity_bytes=config.hbm_capacity_bytes,
                limit_bytes=hbm_limit,
            ),
            ProtectionTier.DRAM: _PoolState(
                tier=ProtectionTier.DRAM,
                pool_id=config.dram_pool_id,
                capacity_bytes=config.dram_capacity_bytes,
                limit_bytes=config.dram_capacity_bytes,
            ),
        }
        self._requests: dict[str, _RequestState] = {}
        self._history: list[TierCopyRecord] = []
        self._events: list[dict[str, Any]] = []
        self._event_counts: dict[str, int] = {}
        self._event_log_dropped = 0
        self._event_sequence = 0
        self._record_history_count = 0
        self._heap_sequence = 0
        self._selection_sequence = 0
        self.hbm_alpha = float(config.initial_hbm_alpha)
        self._stable_windows = 0
        self._hbm_admission_paused = False
        self._now_ms = 0.0

    def _emit(self, kind: str, **fields: Any) -> None:
        self._event_sequence += 1
        self._event_counts[kind] = self._event_counts.get(kind, 0) + 1
        event = {
            "sequence": self._event_sequence,
            "time_ms": round(self._now_ms, 6),
            "kind": kind,
            "design_semantics": DESIGN_SEMANTICS,
            **fields,
        }
        limit = int(self.config.event_log_limit)
        if self.config.retain_event_log and (limit <= 0 or len(self._events) < limit):
            self._events.append(event)
        else:
            self._event_log_dropped += 1

    def advance_time(self, delta_ms: float) -> None:
        self._now_ms += _finite_nonnegative("delta_ms", delta_ms)

    def register_request(self, profile: RequestRecoveryProfile) -> None:
        current = self._requests.get(profile.request_id)
        if current is None:
            self._requests[profile.request_id] = _RequestState(profile=profile)
            self._emit(
                "request_registered",
                request_id=profile.request_id,
                route_version=profile.route_version,
                hbm_target=profile.hbm_target,
            )
            return
        old = current.profile
        if profile.route_version < old.route_version:
            raise ValueError("request profile route version regressed")
        if profile.source_token_end < old.source_token_end:
            raise ValueError("request source frontier regressed")
        compatibility = (
            profile.model_revision,
            profile.kv_abi,
            profile.parallel_layout,
        )
        old_compatibility = (
            old.model_revision,
            old.kv_abi,
            old.parallel_layout,
        )
        if compatibility != old_compatibility:
            raise ValueError("request compatibility identity changed")
        slack_changed = not math.isclose(
            profile.slack_ms, old.slack_ms, rel_tol=0.0, abs_tol=1e-12
        )
        current.profile = profile
        if slack_changed:
            current.slack_epoch += 1
            self._emit(
                "slack_metadata_updated",
                request_id=profile.request_id,
                slack_ms=profile.slack_ms,
                slack_epoch=current.slack_epoch,
                physical_movement=False,
            )

    def has_request(self, request_id: str) -> bool:
        """Return whether request metadata is live in this controller."""

        return str(request_id) in self._requests

    def update_slack(self, request_id: str, slack_ms: float) -> None:
        state = self._request(request_id)
        self.register_request(
            replace(state.profile, slack_ms=_finite_nonnegative("slack_ms", slack_ms))
        )

    def update_source_frontier(self, request_id: str, source_token_end: int) -> None:
        state = self._request(request_id)
        self.register_request(
            replace(state.profile, source_token_end=int(source_token_end))
        )

    def _request(self, request_id: str) -> _RequestState:
        try:
            return self._requests[str(request_id)]
        except KeyError as exc:
            raise KeyError(f"unknown AegisKV request {request_id!r}") from exc

    def _timing(self, tier: ProtectionTier) -> RecoveryTiming:
        return self.config.hbm_load if tier is ProtectionTier.HBM else self.config.dram_load

    def _copy_timing(self, tier: ProtectionTier) -> RecoveryTiming:
        return self.config.hbm_copy if tier is ProtectionTier.HBM else self.config.dram_copy

    def _recovery_ms(
        self,
        profile: RequestRecoveryProfile,
        tier: ProtectionTier | None,
        prefix_tokens: int,
    ) -> float:
        total = max(int(profile.source_token_end), 0)
        prefix = min(max(int(prefix_tokens), 0), total)
        suffix = max(total - prefix, 0)
        value = 0.0
        if tier is not None and prefix > 0:
            timing = self._timing(tier)
            value += timing.setup_ms + prefix * timing.ms_per_token
        if suffix > 0:
            timing = self.config.recompute
            value += timing.setup_ms + suffix * timing.ms_per_token
        return value

    def _harm_ms(
        self,
        profile: RequestRecoveryProfile,
        tier: ProtectionTier | None,
        prefix_tokens: int,
    ) -> float:
        return max(
            self._recovery_ms(profile, tier, prefix_tokens) - profile.slack_ms,
            0.0,
        )

    def _live(self, pool: _PoolState, request_id: str) -> list[int]:
        pool.committed_prefix_count.setdefault(request_id, 0)
        pool.committed_frontier_tokens.setdefault(request_id, 0)
        return pool.live_ordinals.setdefault(request_id, [])

    def _frontier_tokens(self, pool: _PoolState, request_id: str) -> int:
        self._live(pool, request_id)
        return int(pool.committed_frontier_tokens[request_id])

    def _reserved_frontier_tokens(self, pool: _PoolState, request_id: str) -> int:
        """Return the contiguous prefix covered by committed or pending slots.

        Recovery always uses :meth:`_frontier_tokens`, which stops at the first
        non-committed record.  Admission needs a different frontier so several
        sealed chunks can be queued before their asynchronous copies finish.
        If an earlier copy later aborts, ``_abort_from`` removes its entire
        tier-local pending suffix, so this reservation frontier never creates
        a recoverable hole.
        """

        live = self._live(pool, request_id)
        if not live:
            return 0
        # Admission and suffix removal maintain ordinal and token continuity;
        # validate_invariants independently checks that contract.
        return int(pool.records[(request_id, live[-1])].spec.token_end)

    def _tail_record(
        self, pool: _PoolState, request_id: str, *, depth: int = 0
    ) -> TierCopyRecord | None:
        live = self._live(pool, request_id)
        index = len(live) - 1 - int(depth)
        if index < 0:
            return None
        record = pool.records.get((request_id, live[index]))
        if record is None or record.state is not CopyState.COMMITTED:
            return None
        return record

    def _tail_loss_ms(
        self, pool: _PoolState, request_id: str, *, depth: int = 0
    ) -> float:
        record = self._tail_record(pool, request_id, depth=depth)
        if record is None:
            return 0.0
        return self._record_loss_ms(pool, record)

    def _record_loss_ms(
        self,
        pool: _PoolState,
        record: TierCopyRecord,
    ) -> float:
        profile = self._request(record.spec.request_id).profile
        before = record.spec.token_end
        after = record.spec.token_start
        return max(
            self._harm_ms(profile, pool.tier, after)
            - self._harm_ms(profile, pool.tier, before),
            0.0,
        )

    def _effective_key(
        self, pool: _PoolState, record: TierCopyRecord, *, depth: int
    ) -> tuple[float, float]:
        loss = self._tail_loss_ms(pool, record.spec.request_id, depth=depth)
        key = max(
            pool.rent,
            record.landlord_key
            + (loss - record.last_loss_ms) / max(record.spec.byte_size, 1),
        )
        return key, loss

    def _invalidate_tail(self, pool: _PoolState, request_id: str) -> int:
        version = pool.tail_version.get(request_id, 0) + 1
        pool.tail_version[request_id] = version
        return version

    def _push_tail(
        self,
        pool: _PoolState,
        request_id: str,
        *,
        depth: int = 0,
        selection_id: int = 0,
        version: int | None = None,
        persist_refresh: bool = False,
    ) -> None:
        record = self._tail_record(pool, request_id, depth=depth)
        if record is None:
            return
        if version is None:
            version = pool.tail_version.get(request_id, 0)
        key, loss = self._effective_key(pool, record, depth=depth)
        if persist_refresh:
            record.landlord_key = key
            record.last_loss_ms = loss
        self._heap_sequence += 1
        epoch = self._request(request_id).slack_epoch
        heapq.heappush(
            pool.heap,
            (
                key,
                self._heap_sequence,
                request_id,
                record.spec.ordinal,
                int(version),
                int(selection_id),
                int(depth),
                int(epoch),
            ),
        )

    def _publish_actual_tail(self, pool: _PoolState, request_id: str) -> None:
        version = self._invalidate_tail(pool, request_id)
        self._push_tail(
            pool,
            request_id,
            version=version,
            persist_refresh=True,
        )

    def _pop_selection_tail(
        self,
        pool: _PoolState,
        *,
        selection_id: int,
        depths: dict[str, int],
        exclude_request: str,
        touched: set[str],
    ) -> tuple[TierCopyRecord, float] | None:
        while pool.heap:
            (
                queued_key,
                _serial,
                request_id,
                ordinal,
                version,
                queued_selection,
                depth,
                epoch,
            ) = heapq.heappop(pool.heap)
            if version != pool.tail_version.get(request_id, 0):
                continue
            expected_depth = depths.get(request_id, 0)
            if depth != expected_depth:
                continue
            if depth == 0:
                if queued_selection not in {0, selection_id}:
                    continue
            elif queued_selection != selection_id:
                continue
            record = self._tail_record(pool, request_id, depth=depth)
            if record is None or record.spec.ordinal != ordinal:
                continue
            touched.add(request_id)
            if request_id == exclude_request:
                continue
            effective_key, _loss = self._effective_key(pool, record, depth=depth)
            current_epoch = self._request(request_id).slack_epoch
            if epoch != current_epoch or not math.isclose(
                queued_key, effective_key, rel_tol=0.0, abs_tol=1e-12
            ):
                self._push_tail(
                    pool,
                    request_id,
                    depth=depth,
                    selection_id=selection_id if depth else 0,
                    version=version,
                )
                continue
            return record, effective_key
        return None

    def _selection_joint_loss(
        self,
        pool: _PoolState,
        depths: dict[str, int],
    ) -> float:
        total = 0.0
        for request_id, depth in depths.items():
            if depth <= 0:
                continue
            profile = self._request(request_id).profile
            live = self._live(pool, request_id)
            before = self._frontier_tokens(pool, request_id)
            remaining = len(live) - depth
            after = 0
            if remaining > 0:
                record = pool.records[(request_id, live[remaining - 1])]
                after = record.spec.token_end
            total += max(
                self._harm_ms(profile, pool.tier, after)
                - self._harm_ms(profile, pool.tier, before),
                0.0,
            )
        return total

    def _select_victims(
        self,
        pool: _PoolState,
        *,
        bytes_needed: int,
        threshold: float,
        candidate_gain_ms: float,
        exclude_request: str,
        force: bool = False,
    ) -> tuple[list[TierCopyRecord], list[float], float, set[str], str]:
        deficit = max(pool.used_bytes + int(bytes_needed) - pool.limit_bytes, 0)
        if deficit <= 0:
            return [], [], 0.0, set(), "capacity_available"
        self._selection_sequence += 1
        selection_id = self._selection_sequence
        depths: dict[str, int] = {}
        selected: list[TierCopyRecord] = []
        selected_keys: list[float] = []
        touched: set[str] = set()
        released = 0
        reason = "insufficient_safe_tail"
        while released < deficit:
            value = self._pop_selection_tail(
                pool,
                selection_id=selection_id,
                depths=depths,
                exclude_request=exclude_request,
                touched=touched,
            )
            if value is None:
                break
            record, key = value
            if not force and key >= threshold:
                reason = "landlord_threshold"
                break
            selected.append(record)
            selected_keys.append(key)
            released += record.spec.byte_size
            request_id = record.spec.request_id
            depths[request_id] = depths.get(request_id, 0) + 1
            self._push_tail(
                pool,
                request_id,
                depth=depths[request_id],
                selection_id=selection_id,
                version=pool.tail_version.get(request_id, 0),
            )
        joint_loss = self._selection_joint_loss(pool, depths)
        if released >= deficit:
            if force or candidate_gain_ms > joint_loss:
                reason = "selected"
            else:
                reason = "joint_loss"
        return selected, selected_keys, joint_loss, touched, reason

    def _remove_tail_record(
        self,
        pool: _PoolState,
        record: TierCopyRecord,
        *,
        reason: str,
    ) -> None:
        request_id = record.spec.request_id
        prior_residency = self.residency(request_id, record.spec.ordinal)
        live = self._live(pool, request_id)
        if not live or live[-1] != record.spec.ordinal:
            raise RuntimeError("AegisKV attempted a non-suffix eviction")
        live.pop()
        pool.records.pop(record.key, None)
        committed_count = int(pool.committed_prefix_count.get(request_id, 0))
        if committed_count > len(live):
            pool.committed_prefix_count[request_id] = len(live)
            pool.committed_frontier_tokens[request_id] = int(
                record.spec.token_start
            )
        pool.used_bytes -= record.spec.byte_size
        pool.removal_count += 1
        pool.removed_bytes += int(record.spec.byte_size)
        pool.removal_reason_counts[reason] = (
            pool.removal_reason_counts.get(reason, 0) + 1
        )
        if reason in _STORAGE_EVICTION_REASONS:
            pool.eviction_count += 1
            pool.evicted_bytes += int(record.spec.byte_size)
            pool.eviction_reason_counts[reason] = (
                pool.eviction_reason_counts.get(reason, 0) + 1
            )
        record.state = CopyState.ABORTED
        record.aborted_at_ms = self._now_ms
        record.abort_reason = reason
        self._record_history_count += 1
        if self.config.retain_record_history:
            self._history.append(record)
        self._emit(
            "tier_copy_removed",
            request_id=request_id,
            ordinal=record.spec.ordinal,
            tier=pool.tier.value,
            pool_id=pool.pool_id,
            reason=reason,
            prior_residency=prior_residency.value,
        )

    def _restore_touched_tails(
        self, pool: _PoolState, request_ids: Iterable[str]
    ) -> None:
        for request_id in sorted(set(request_ids)):
            self._publish_actual_tail(pool, request_id)

    def _validate_spec(self, spec: ChunkSpec) -> _RequestState:
        request = self._request(spec.request_id)
        profile = request.profile
        if spec.route_version != profile.route_version:
            raise ValueError("chunk route version differs from request")
        if spec.source_worker != profile.source_worker:
            raise ValueError("chunk source owner differs from request")
        if spec.hbm_target != profile.hbm_target:
            raise ValueError("chunk HBM target differs from route binding")
        if (
            spec.model_revision,
            spec.kv_abi,
            spec.parallel_layout,
        ) != (
            profile.model_revision,
            profile.kv_abi,
            profile.parallel_layout,
        ):
            raise ValueError("chunk compatibility identity differs from request")
        if spec.token_end > profile.source_token_end:
            raise ValueError("chunk exceeds the causally visible source frontier")
        return request

    def _candidate_gain_ms(
        self, pool: _PoolState, spec: ChunkSpec
    ) -> float:
        profile = self._request(spec.request_id).profile
        # Admission evaluates the eventual contiguous prefix after already
        # reserved predecessors commit.  PENDING bytes remain unusable by
        # recovery and a failed predecessor aborts this tier's full suffix.
        before = self._reserved_frontier_tokens(pool, spec.request_id)
        if spec.token_start != before:
            return 0.0
        return max(
            self._harm_ms(profile, pool.tier, before)
            - self._harm_ms(profile, pool.tier, spec.token_end),
            0.0,
        )

    def _try_admit(
        self, pool: _PoolState, spec: ChunkSpec
    ) -> TierAdmission:
        if (
            pool.tier is ProtectionTier.HBM
            and self._hbm_admission_paused
        ):
            return self._disabled_admission(
                ProtectionTier.HBM,
                reason="hbm_admission_paused_for_quota_reclaim",
            )
        live = self._live(pool, spec.request_id)
        if spec.ordinal != len(live):
            return TierAdmission(
                tier=pool.tier,
                admitted=False,
                reason="noncontiguous_frontier",
                gain_ms=0.0,
                threshold=pool.rent,
                victim_ids=(),
                pending_id="",
                copy_kind="source_d2d" if pool.tier is ProtectionTier.HBM else "source_d2h",
            )
        gain = self._candidate_gain_ms(pool, spec)
        threshold = pool.rent + gain / max(spec.byte_size, 1)
        copy_kind = "source_d2d" if pool.tier is ProtectionTier.HBM else "source_d2h"
        if gain <= 0.0:
            return TierAdmission(
                tier=pool.tier,
                admitted=False,
                reason="nonpositive_recovery_gain",
                gain_ms=gain,
                threshold=threshold,
                victim_ids=(),
                pending_id="",
                copy_kind=copy_kind,
            )
        victims, keys, joint_loss, touched, reason = self._select_victims(
            pool,
            bytes_needed=spec.byte_size,
            threshold=threshold,
            candidate_gain_ms=gain,
            exclude_request=spec.request_id,
        )
        if reason != "selected" and reason != "capacity_available":
            self._restore_touched_tails(pool, touched)
            return TierAdmission(
                tier=pool.tier,
                admitted=False,
                reason=reason,
                gain_ms=gain,
                threshold=threshold,
                victim_ids=tuple(item.spec.logical_id for item in victims),
                pending_id="",
                copy_kind=copy_kind,
            )
        for record in victims:
            self._remove_tail_record(pool, record, reason="slack_landlord_victim")
        if keys:
            pool.rent = max(pool.rent, max(keys))
        self._restore_touched_tails(pool, touched)
        if pool.used_bytes + spec.byte_size > pool.limit_bytes:
            raise RuntimeError("SlackLandlord admission overcommitted pool capacity")
        record = TierCopyRecord(
            spec=spec,
            tier=pool.tier,
            pool_id=pool.pool_id,
            state=CopyState.PENDING,
            gain_ms=gain,
            admitted_at_ms=self._now_ms,
            copy_kind=copy_kind,
            # Capture the post-victim rent at admission time.  A later copy
            # completion must not mint fresh credit from a rent that advanced
            # while this source-owned transfer was in flight.
            landlord_key=pool.rent + gain / max(spec.byte_size, 1),
            last_loss_ms=0.0,
        )
        pool.records[record.key] = record
        live.append(spec.ordinal)
        pool.used_bytes += spec.byte_size
        self._publish_actual_tail(pool, spec.request_id)
        pending_id = f"{pool.pool_id}:{spec.logical_id}"
        self._emit(
            "tier_copy_pending",
            request_id=spec.request_id,
            ordinal=spec.ordinal,
            tier=pool.tier.value,
            pool_id=pool.pool_id,
            pending_id=pending_id,
            copy_kind=copy_kind,
            source_worker=spec.source_worker,
            hbm_target=spec.hbm_target,
            route_version=spec.route_version,
            gain_ms=round(gain, 6),
            joint_victim_loss_ms=round(joint_loss, 6),
            victim_ids=[item.spec.logical_id for item in victims],
            source_owned=True,
            base_hbm_chunk_id=None,
            writeback_cost_ms=0.0,
        )
        return TierAdmission(
            tier=pool.tier,
            admitted=True,
            reason="admitted",
            gain_ms=gain,
            threshold=threshold,
            victim_ids=tuple(item.spec.logical_id for item in victims),
            pending_id=pending_id,
            copy_kind=copy_kind,
        )

    @staticmethod
    def _disabled_admission(
        tier: ProtectionTier,
        *,
        reason: str,
    ) -> TierAdmission:
        return TierAdmission(
            tier=tier,
            admitted=False,
            reason=reason,
            gain_ms=0.0,
            threshold=0.0,
            victim_ids=(),
            pending_id="",
            copy_kind=(
                "source_d2d"
                if tier is ProtectionTier.HBM
                else "source_d2h"
            ),
        )

    def admit_chunk(
        self,
        spec: ChunkSpec,
        *,
        allowed_tiers: Iterable[ProtectionTier] | None = None,
    ) -> PlacementDecision:
        self._validate_spec(spec)
        allowed = (
            {ProtectionTier(tier) for tier in allowed_tiers}
            if allowed_tiers is not None
            else {ProtectionTier.HBM, ProtectionTier.DRAM}
        )
        hbm = (
            self._try_admit(self._pools[ProtectionTier.HBM], spec)
            if ProtectionTier.HBM in allowed
            else self._disabled_admission(
                ProtectionTier.HBM, reason="tier_not_allowed"
            )
        )
        dram = (
            self._try_admit(self._pools[ProtectionTier.DRAM], spec)
            if ProtectionTier.DRAM in allowed
            else self._disabled_admission(
                ProtectionTier.DRAM, reason="tier_not_allowed"
            )
        )
        for tier, admission in (
            (ProtectionTier.HBM, hbm),
            (ProtectionTier.DRAM, dram),
        ):
            if tier not in allowed:
                continue
            pool = self._pools[tier]
            pool.admission_attempt_count += 1
            if admission.admitted:
                pool.admission_success_count += 1
                pool.admitted_bytes += int(spec.byte_size)
            else:
                pool.admission_rejection_count += 1
                pool.rejection_reason_counts[admission.reason] = (
                    pool.rejection_reason_counts.get(admission.reason, 0) + 1
                )
        decision = PlacementDecision(
            request_id=spec.request_id,
            ordinal=spec.ordinal,
            hbm=hbm,
            dram=dram,
        )
        self._emit(
            "independent_tier_decision",
            request_id=spec.request_id,
            ordinal=spec.ordinal,
            hbm_admitted=hbm.admitted,
            dram_admitted=dram.admitted,
            intended_residency=decision.intended_residency.value,
        )
        return decision

    def _abort_from(
        self,
        pool: _PoolState,
        request_id: str,
        ordinal: int,
        *,
        reason: str,
    ) -> None:
        live = self._live(pool, request_id)
        while live and live[-1] >= int(ordinal):
            record = pool.records[(request_id, live[-1])]
            self._remove_tail_record(pool, record, reason=reason)
        self._publish_actual_tail(pool, request_id)

    def complete_copy(
        self,
        tier: ProtectionTier,
        request_id: str,
        ordinal: int,
        proof: CopyCompletionProof,
    ) -> TierCompletion:
        tier = ProtectionTier(tier)
        pool = self._pools[tier]
        key = (str(request_id), int(ordinal))
        record = pool.records.get(key)
        if record is None:
            raise KeyError(f"unknown pending copy {tier.value}:{request_id}:{ordinal}")
        if record.state is not CopyState.PENDING:
            raise ValueError("copy completion requires a PENDING record")
        spec = record.spec
        expected_shards = tuple(range(spec.shard_count))
        expected_pending_id = f"{pool.pool_id}:{spec.logical_id}"
        expected_copy_kind = (
            "source_d2d" if tier is ProtectionTier.HBM else "source_d2h"
        )
        expected_destination = (
            spec.hbm_target if tier is ProtectionTier.HBM else pool.pool_id
        )
        valid = bool(proof.transfer_succeeded) and (
            proof.tier is tier
            and proof.pending_id == expected_pending_id
            and proof.copy_kind == expected_copy_kind
            and proof.source_worker == spec.source_worker
            and proof.destination_id == expected_destination
            and proof.source_owned is True
            and int(proof.route_version) == spec.route_version
            and proof.model_revision == spec.model_revision
            and proof.kv_abi == spec.kv_abi
            and proof.parallel_layout == spec.parallel_layout
            and int(proof.transferred_bytes) == spec.byte_size
            and tuple(sorted(set(proof.acked_shards))) == expected_shards
        )
        if not valid:
            self._abort_from(
                pool,
                spec.request_id,
                spec.ordinal,
                reason="copy_completion_validation_failed",
            )
            completion = TierCompletion(
                tier=tier,
                request_id=spec.request_id,
                ordinal=spec.ordinal,
                state=CopyState.ABORTED,
                reason="copy_completion_validation_failed",
                residency=self.residency(spec.request_id, spec.ordinal),
            )
            self._emit(
                "tier_copy_aborted",
                request_id=spec.request_id,
                ordinal=spec.ordinal,
                tier=tier.value,
                reason=completion.reason,
                presented_tier=proof.tier.value,
                presented_pending_id=proof.pending_id,
                presented_copy_kind=proof.copy_kind,
            )
            return completion
        record.completion_validated = True
        live = self._live(pool, spec.request_id)
        committed_now: list[TierCopyRecord] = []
        committed_count = int(
            pool.committed_prefix_count.get(spec.request_id, 0)
        )
        for candidate_ordinal in live[committed_count:]:
            candidate = pool.records[(spec.request_id, candidate_ordinal)]
            if not candidate.completion_validated:
                break
            candidate.state = CopyState.COMMITTED
            candidate.committed_at_ms = self._now_ms
            candidate.last_loss_ms = self._record_loss_ms(pool, candidate)
            committed_now.append(candidate)
            committed_count += 1
            pool.committed_frontier_tokens[spec.request_id] = int(
                candidate.spec.token_end
            )
        pool.committed_prefix_count[spec.request_id] = committed_count
        self._publish_actual_tail(pool, spec.request_id)
        for committed in committed_now:
            residency = self.residency(
                committed.spec.request_id, committed.spec.ordinal
            )
            self._emit(
                "tier_copy_committed",
                request_id=committed.spec.request_id,
                ordinal=committed.spec.ordinal,
                tier=tier.value,
                pool_id=pool.pool_id,
                route_version=committed.spec.route_version,
                transferred_bytes=committed.spec.byte_size,
                residency=residency.value,
                source_owned=True,
                source_worker=proof.source_worker,
                destination_id=proof.destination_id,
                copy_kind=proof.copy_kind,
                pending_id=proof.pending_id,
                independent_ack=True,
            )
        residency = self.residency(spec.request_id, spec.ordinal)
        if record.state is CopyState.PENDING:
            self._emit(
                "tier_copy_completion_waiting",
                request_id=spec.request_id,
                ordinal=spec.ordinal,
                tier=tier.value,
                reason="waiting_for_prefix_predecessor",
            )
        return TierCompletion(
            tier=tier,
            request_id=spec.request_id,
            ordinal=spec.ordinal,
            state=record.state,
            reason=(
                "committed"
                if record.state is CopyState.COMMITTED
                else "waiting_for_prefix_predecessor"
            ),
            residency=residency,
        )

    def completion_proof(
        self,
        spec: ChunkSpec,
        tier: ProtectionTier,
        *,
        succeeded: bool = True,
    ) -> CopyCompletionProof:
        tier = ProtectionTier(tier)
        pool = self._pools[tier]
        return CopyCompletionProof(
            tier=tier,
            pending_id=f"{pool.pool_id}:{spec.logical_id}",
            copy_kind=(
                "source_d2d" if tier is ProtectionTier.HBM else "source_d2h"
            ),
            source_worker=spec.source_worker,
            destination_id=(
                spec.hbm_target if tier is ProtectionTier.HBM else pool.pool_id
            ),
            route_version=spec.route_version,
            model_revision=spec.model_revision,
            kv_abi=spec.kv_abi,
            parallel_layout=spec.parallel_layout,
            transferred_bytes=spec.byte_size if succeeded else 0,
            acked_shards=tuple(range(spec.shard_count)) if succeeded else (),
            transfer_succeeded=succeeded,
            source_owned=True,
        )

    def residency(self, request_id: str, ordinal: int) -> Residency:
        states = {}
        for tier, pool in self._pools.items():
            record = pool.records.get((str(request_id), int(ordinal)))
            states[tier] = bool(record and record.state is CopyState.COMMITTED)
        if states[ProtectionTier.HBM] and states[ProtectionTier.DRAM]:
            return Residency.HBM_AND_DRAM
        if states[ProtectionTier.HBM]:
            return Residency.HBM_ONLY
        if states[ProtectionTier.DRAM]:
            return Residency.DRAM_ONLY
        return Residency.BYPASS

    def committed_prefix_tokens(
        self, request_id: str, tier: ProtectionTier
    ) -> int:
        return self._frontier_tokens(self._pools[ProtectionTier(tier)], request_id)

    def pool_used_bytes(self, tier: ProtectionTier) -> int:
        """Return maintained pool accounting without serializing records."""

        return int(self._pools[ProtectionTier(tier)].used_bytes)

    def copy_state(
        self, request_id: str, ordinal: int, tier: ProtectionTier
    ) -> CopyState | None:
        record = self._pools[ProtectionTier(tier)].records.get(
            (str(request_id), int(ordinal))
        )
        return record.state if record is not None else None

    def victim_frontiers(
        self, tier: ProtectionTier
    ) -> tuple[VictimFrontier, ...]:
        """Expose one and only one currently safe request tail per pool.

        Hidden predecessor chunks retain their stored Landlord metadata but do
        not participate in global victim ordering until the current tail is
        removed.  A request ending in PENDING has no safe committed frontier.
        """

        tier = ProtectionTier(tier)
        pool = self._pools[tier]
        frontiers: list[VictimFrontier] = []
        for request_id in sorted(pool.live_ordinals):
            record = self._tail_record(pool, request_id)
            if record is None:
                continue
            key, _loss = self._effective_key(pool, record, depth=0)
            frontiers.append(
                VictimFrontier(
                    tier=tier,
                    pool_id=pool.pool_id,
                    request_id=request_id,
                    ordinal=record.spec.ordinal,
                    logical_id=record.spec.logical_id,
                    landlord_key=key,
                    slack_epoch=self._request(request_id).slack_epoch,
                )
            )
        return tuple(frontiers)

    def request_profile(self, request_id: str) -> RequestRecoveryProfile:
        return self._request(request_id).profile

    def estimate_recovery(
        self,
        request_id: str,
        *,
        activated_target: str,
    ) -> RecoveryDecision:
        """Estimate fail-now recovery without mutating route or copy state."""

        profile = self._request(request_id).profile
        return self._estimate_recovery_for_profile(
            profile,
            activated_target=str(activated_target),
        )

    def estimate_recovery_at(
        self,
        request_id: str,
        *,
        source_token_end: int,
        slack_ms: float,
        activated_target: str,
    ) -> RecoveryDecision:
        """Pure recovery query at a counterfactual source frontier.

        Unlike :meth:`update_source_frontier`, :meth:`update_slack`, or the
        runtime's primary fail-now estimator, this method does not change the
        request profile, its slack epoch, the victim indices, event counters,
        or recovery action metrics.  It is the controller primitive used by
        policy-independent active-source shadow probes.
        """

        state = self._request(request_id)
        profile = replace(
            state.profile,
            source_token_end=int(source_token_end),
            slack_ms=_finite_nonnegative("slack_ms", slack_ms),
        )
        return self._estimate_recovery_for_profile(
            profile,
            activated_target=str(activated_target),
        )

    def _estimate_recovery_for_profile(
        self,
        profile: RequestRecoveryProfile,
        *,
        activated_target: str,
    ) -> RecoveryDecision:
        total = profile.source_token_end
        candidates: list[tuple[float, str, int]] = [
            (self._recovery_ms(profile, None, 0), "RECOMPUTE", 0)
        ]
        request_id = profile.request_id
        hbm_prefix = self.committed_prefix_tokens(request_id, ProtectionTier.HBM)
        if hbm_prefix > 0 and activated_target == profile.hbm_target:
            candidates.append(
                (
                    self._recovery_ms(profile, ProtectionTier.HBM, hbm_prefix),
                    "PEER_HBM",
                    hbm_prefix,
                )
            )
        dram_prefix = self.committed_prefix_tokens(request_id, ProtectionTier.DRAM)
        if (
            dram_prefix > 0
            and profile.source_failure_domain != self.config.dram_failure_domain
        ):
            candidates.append(
                (
                    self._recovery_ms(profile, ProtectionTier.DRAM, dram_prefix),
                    "DRAM",
                    dram_prefix,
                )
            )
        recovery_ms, action, restored = min(
            candidates, key=lambda item: (item[0], -item[2], item[1])
        )
        return RecoveryDecision(
            request_id=request_id,
            action=action,
            restored_tokens=restored,
            recompute_tokens=max(total - restored, 0),
            recovery_ms=recovery_ms,
            slack_ms=profile.slack_ms,
            visible_harm_ms=max(recovery_ms - profile.slack_ms, 0.0),
            route_version=profile.route_version,
            activated_target=activated_target,
        )

    def copy_cost_ms(self, decision: PlacementDecision) -> float:
        costs = []
        spec: ChunkSpec | None = None
        for tier, admission in (
            (ProtectionTier.HBM, decision.hbm),
            (ProtectionTier.DRAM, decision.dram),
        ):
            if not admission.admitted:
                continue
            pool = self._pools[tier]
            record = pool.records.get((decision.request_id, decision.ordinal))
            if record is None:
                continue
            spec = record.spec
            timing = self._copy_timing(tier)
            costs.append(timing.setup_ms + spec.token_count * timing.ms_per_token)
        # Source D2D and D2H may execute independently or in parallel.  The
        # conservative per-chunk critical path is their maximum, not a hidden
        # HBM->DRAM serial transfer.
        return max(costs, default=0.0)

    def rotate_hbm_target(
        self,
        request_id: str,
        *,
        failed_target: str,
        replacement_target: str,
        replacement_failure_domain: str,
    ) -> dict[str, Any]:
        if not replacement_target or not replacement_failure_domain:
            raise ValueError("replacement HBM target identity is required")
        state = self._request(request_id)
        profile = state.profile
        if profile.hbm_target != failed_target:
            raise ValueError("failed target does not match the route binding")
        pool = self._pools[ProtectionTier.HBM]
        removed = len(self._live(pool, request_id))
        if removed:
            self._abort_from(
                pool,
                request_id,
                0,
                reason="bound_hbm_target_failed",
            )
        state.profile = replace(
            profile,
            hbm_target=replacement_target,
            hbm_failure_domain=replacement_failure_domain,
        )
        self._emit(
            "standby_target_rotated",
            request_id=request_id,
            failed_target=failed_target,
            replacement_target=replacement_target,
            route_version=profile.route_version,
            route_version_changed=False,
            invalidated_hbm_records=removed,
            dram_records_preserved=len(
                self._live(self._pools[ProtectionTier.DRAM], request_id)
            ),
        )
        return {
            "request_id": request_id,
            "route_version": profile.route_version,
            "replacement_target": replacement_target,
            "invalidated_hbm_records": removed,
            "dram_records_preserved": len(
                self._live(self._pools[ProtectionTier.DRAM], request_id)
            ),
        }

    def abort_source_pending(self, source_worker: str) -> int:
        aborted = 0
        for pool in self._pools.values():
            request_ids = {
                record.spec.request_id
                for record in pool.records.values()
                if record.spec.source_worker == source_worker
                and record.state is CopyState.PENDING
            }
            for request_id in sorted(request_ids):
                pending = [
                    record.spec.ordinal
                    for record in pool.records.values()
                    if record.spec.request_id == request_id
                    and record.spec.source_worker == source_worker
                    and record.state is CopyState.PENDING
                ]
                if pending:
                    before = len(self._live(pool, request_id))
                    self._abort_from(
                        pool,
                        request_id,
                        min(pending),
                        reason="source_failed_before_copy_commit",
                    )
                    aborted += before - len(self._live(pool, request_id))
        return aborted

    def recover_after_source_failure(
        self,
        request_id: str,
        *,
        activated_target: str,
        next_route_version: int,
    ) -> RecoveryDecision:
        state = self._request(request_id)
        profile = state.profile
        if next_route_version <= profile.route_version:
            raise ValueError("source failure must advance the route version")
        self.abort_source_pending(profile.source_worker)
        estimate = self.estimate_recovery(
            request_id, activated_target=activated_target
        )
        result = replace(estimate, route_version=next_route_version)
        self._emit(
            "recovery_selected",
            request_id=request_id,
            action=result.action,
            restored_tokens=result.restored_tokens,
            recompute_tokens=result.recompute_tokens,
            recovery_ms=round(result.recovery_ms, 6),
            visible_harm_ms=round(result.visible_harm_ms, 6),
            prior_route_version=profile.route_version,
            route_version=next_route_version,
            activated_target=activated_target,
        )
        return result

    def invalidate_tier(
        self,
        request_id: str,
        tier: ProtectionTier,
        *,
        reason: str,
    ) -> dict[str, Any]:
        pool = self._pools[ProtectionTier(tier)]
        live = self._live(pool, request_id)
        released = sum(
            pool.records[(request_id, ordinal)].spec.byte_size
            for ordinal in live
        )
        if live:
            self._abort_from(pool, request_id, 0, reason=reason)
        return {
            "request_id": request_id,
            "tier": ProtectionTier(tier).value,
            "released_bytes": released,
            "reason": reason,
        }

    def set_pool_limit(
        self,
        tier: ProtectionTier,
        limit_bytes: int,
    ) -> dict[str, Any]:
        tier = ProtectionTier(tier)
        pool = self._pools[tier]
        limit = min(max(int(limit_bytes), 0), pool.capacity_bytes)
        pool.limit_bytes = limit
        pause_observed = False
        if tier is ProtectionTier.HBM:
            pause_observed = pool.used_bytes > pool.limit_bytes
            if pause_observed and not self._hbm_admission_paused:
                self._hbm_admission_paused = True
                self._emit(
                    "hbm_admission_paused",
                    reason="external_hbm_limit_contraction",
                    quota_bytes=pool.limit_bytes,
                    used_bytes=pool.used_bytes,
                )
            self.hbm_alpha = limit / max(pool.capacity_bytes, 1)
            aborted = self._abort_pending_until_limit(
                pool,
                reason="hbm_quota_contraction_pending_abort",
            )
            evicted = self._force_hbm_victims()
        else:
            aborted = self._abort_pending_until_limit(
                pool,
                reason="external_dram_limit_pending_abort",
            )
            evicted = []
            # DRAM has no AIMD, but an externally reduced physical/policy
            # limit still reclaims its own suffix tails without touching HBM.
            while pool.used_bytes > pool.limit_bytes:
                victims, _keys, _loss, touched, selected_reason = (
                    self._select_victims(
                        pool,
                        bytes_needed=0,
                        threshold=math.inf,
                        candidate_gain_ms=math.inf,
                        exclude_request="",
                        force=True,
                    )
                )
                if selected_reason != "selected" or not victims:
                    self._restore_touched_tails(pool, touched)
                    break
                for record in victims:
                    evicted.append(record.spec.logical_id)
                    self._remove_tail_record(
                        pool, record, reason="external_dram_limit_contraction"
                    )
                self._restore_touched_tails(pool, touched)
        if pool.used_bytes > pool.limit_bytes:
            raise RuntimeError("pool limit contraction could not reach quota")
        admission_resumed = False
        if tier is ProtectionTier.HBM and self._hbm_admission_paused:
            self._hbm_admission_paused = False
            admission_resumed = True
            self._emit(
                "hbm_admission_resumed",
                reason="external_hbm_limit_satisfied",
                quota_bytes=pool.limit_bytes,
                used_bytes=pool.used_bytes,
            )
        return {
            "tier": tier.value,
            "limit_bytes": pool.limit_bytes,
            "used_bytes": pool.used_bytes,
            "aborted_pending": aborted,
            "evicted_tails": evicted,
            "admission_pause_observed": pause_observed,
            "admission_resumed": admission_resumed,
        }

    def _abort_pending_until_limit(
        self,
        pool: _PoolState,
        *,
        reason: str,
    ) -> list[str]:
        aborted: list[str] = []
        while pool.used_bytes > pool.limit_bytes:
            tails = [
                record
                for request_id in pool.live_ordinals
                if (record := self._tail_record_any(pool, request_id)) is not None
                and record.state is CopyState.PENDING
            ]
            if not tails:
                break
            victim = min(
                tails,
                key=lambda item: (
                    item.gain_ms / max(item.spec.byte_size, 1),
                    -item.spec.ordinal,
                    item.spec.request_id,
                ),
            )
            aborted.append(victim.spec.logical_id)
            self._abort_from(
                pool,
                victim.spec.request_id,
                victim.spec.ordinal,
                reason=reason,
            )
        return aborted

    def _tail_record_any(
        self, pool: _PoolState, request_id: str
    ) -> TierCopyRecord | None:
        live = self._live(pool, request_id)
        if not live:
            return None
        return pool.records.get((request_id, live[-1]))

    def _force_hbm_victims(self) -> list[str]:
        pool = self._pools[ProtectionTier.HBM]
        evicted: list[str] = []
        while pool.used_bytes > pool.limit_bytes:
            victims, _keys, _loss, touched, reason = self._select_victims(
                pool,
                # No new candidate is being admitted.  The deficit is exactly
                # current usage minus the newly published quota.
                bytes_needed=0,
                threshold=math.inf,
                candidate_gain_ms=math.inf,
                exclude_request="",
                force=True,
            )
            if reason != "selected" or not victims:
                self._restore_touched_tails(pool, touched)
                break
            for record in victims:
                evicted.append(record.spec.logical_id)
                self._remove_tail_record(
                    pool, record, reason="hbm_quota_contraction"
                )
            self._restore_touched_tails(pool, touched)
        return evicted

    def control_hbm_quota(self, *, pressure: bool) -> dict[str, Any]:
        old_alpha = self.hbm_alpha
        if pressure:
            self.hbm_alpha = max(
                self.config.minimum_hbm_alpha,
                self.hbm_alpha * self.config.aimd_decrease_factor,
            )
            self._stable_windows = 0
        else:
            self._stable_windows += 1
            if self._stable_windows >= self.config.aimd_stable_windows:
                self.hbm_alpha = min(
                    self.config.maximum_hbm_alpha,
                    self.hbm_alpha + self.config.aimd_increase_step,
                )
                self._stable_windows = 0
        pool = self._pools[ProtectionTier.HBM]
        pool.limit_bytes = int(pool.capacity_bytes * self.hbm_alpha)
        pause_observed = pool.used_bytes > pool.limit_bytes
        if pause_observed and not self._hbm_admission_paused:
            self._hbm_admission_paused = True
            self._emit(
                "hbm_admission_paused",
                reason="aimd_quota_contraction",
                quota_bytes=pool.limit_bytes,
                used_bytes=pool.used_bytes,
            )
        aborted = self._abort_pending_until_limit(
            pool,
            reason="hbm_quota_contraction_pending_abort",
        )
        evicted = self._force_hbm_victims()
        if pool.used_bytes > pool.limit_bytes:
            raise RuntimeError("AIMD HBM reclaim could not satisfy quota")
        admission_resumed = False
        if self._hbm_admission_paused:
            self._hbm_admission_paused = False
            admission_resumed = True
            self._emit(
                "hbm_admission_resumed",
                reason="aimd_quota_satisfied",
                quota_bytes=pool.limit_bytes,
                used_bytes=pool.used_bytes,
            )
        result = {
            "old_alpha": old_alpha,
            "new_alpha": self.hbm_alpha,
            "pressure": bool(pressure),
            "stable_windows": self._stable_windows,
            "quota_bytes": pool.limit_bytes,
            "used_bytes": pool.used_bytes,
            "aborted_pending": aborted,
            "evicted_hbm_tails": evicted,
            "admission_pause_observed": pause_observed,
            "admission_resumed": admission_resumed,
            "reclaim_satisfied": pool.used_bytes <= pool.limit_bytes,
            "dram_quota_changed": False,
        }
        self._emit("hbm_quota_aimd_window", **result)
        return result

    def complete_request(
        self, request_id: str, *, force_abort_pending: bool = False
    ) -> dict[str, Any]:
        pending = []
        for tier, pool in self._pools.items():
            pending.extend(
                f"{tier.value}:{record.spec.ordinal}"
                for record in pool.records.values()
                if record.spec.request_id == request_id
                and record.state is CopyState.PENDING
            )
        if pending:
            if force_abort_pending:
                released = {
                    tier.value: sum(
                        record.spec.byte_size
                        for record in self._pools[tier].records.values()
                        if record.spec.request_id == request_id
                    )
                    for tier in (ProtectionTier.HBM, ProtectionTier.DRAM)
                }
                for tier in (ProtectionTier.HBM, ProtectionTier.DRAM):
                    self.invalidate_tier(
                        request_id,
                        tier,
                        reason="request_superseded_pending_abort",
                    )
                pending = []
            else:
                return {
                    "completed": False,
                    "reason": "pending_copy_not_terminal",
                    "pending": sorted(pending),
                }
        else:
            released = {
                ProtectionTier.HBM.value: 0,
                ProtectionTier.DRAM.value: 0,
            }
        for tier, pool in self._pools.items():
            while self._live(pool, request_id):
                record = self._tail_record(pool, request_id)
                if record is None:
                    raise RuntimeError("terminal request contains a noncommitted live record")
                released[tier.value] += record.spec.byte_size
                self._remove_tail_record(pool, record, reason="request_completed")
            self._publish_actual_tail(pool, request_id)
        self._requests.pop(request_id, None)
        self._emit("request_reclaimed", request_id=request_id, released_bytes=released)
        return {"completed": True, "released_bytes": released}

    def validate_invariants(self) -> None:
        for pool in self._pools.values():
            used = sum(record.spec.byte_size for record in pool.records.values())
            if used != pool.used_bytes:
                raise AssertionError("pool used bytes differ from live records")
            if pool.used_bytes > pool.limit_bytes:
                raise AssertionError("pool exceeds its current quota")
            if pool.admission_attempt_count != (
                pool.admission_success_count + pool.admission_rejection_count
            ):
                raise AssertionError("pool admission accounting does not reconcile")
            if pool.eviction_count > pool.removal_count:
                raise AssertionError("pool eviction count exceeds all removals")
            if pool.evicted_bytes > pool.removed_bytes:
                raise AssertionError("pool evicted bytes exceed all removed bytes")
            if sum(pool.eviction_reason_counts.values()) != pool.eviction_count:
                raise AssertionError("pool eviction reasons do not reconcile")
            if sum(pool.removal_reason_counts.values()) != pool.removal_count:
                raise AssertionError("pool removal reasons do not reconcile")
            if sum(pool.rejection_reason_counts.values()) != (
                pool.admission_rejection_count
            ):
                raise AssertionError("pool rejection reasons do not reconcile")
            for request_id, ordinals in pool.live_ordinals.items():
                if ordinals != list(range(len(ordinals))):
                    raise AssertionError("tier request prefix is not ordinal-contiguous")
                for ordinal in ordinals:
                    record = pool.records.get((request_id, ordinal))
                    if record is None:
                        raise AssertionError("live ordinal has no copy record")
                    if not record.source_owned:
                        raise AssertionError("copy path is not source owned")
                    if record.base_hbm_chunk_id is not None:
                        raise AssertionError("latest design has a cross-tier base")
                    if record.writeback_cost_ms != 0.0:
                        raise AssertionError("latest design contains writeback cost")
                committed = 0
                frontier = 0
                for ordinal in ordinals:
                    record = pool.records[(request_id, ordinal)]
                    if record.state is not CopyState.COMMITTED:
                        break
                    if record.spec.token_start != frontier:
                        break
                    committed += 1
                    frontier = record.spec.token_end
                if committed != pool.committed_prefix_count.get(request_id, 0):
                    raise AssertionError("cached committed prefix count drifted")
                if frontier != pool.committed_frontier_tokens.get(request_id, 0):
                    raise AssertionError("cached committed token frontier drifted")
            frontiers = self.victim_frontiers(pool.tier)
            if len({row.request_id for row in frontiers}) != len(frontiers):
                raise AssertionError("pool exposes multiple victim tails per request")
            for frontier in frontiers:
                live = pool.live_ordinals.get(frontier.request_id, [])
                if not live or live[-1] != frontier.ordinal:
                    raise AssertionError("victim frontier is not the safe request tail")

    def snapshot(self) -> dict[str, Any]:
        pools: dict[str, Any] = {}
        for tier, pool in self._pools.items():
            pools[tier.value] = {
                "pool_id": pool.pool_id,
                "capacity_bytes": pool.capacity_bytes,
                "limit_bytes": pool.limit_bytes,
                "used_bytes": pool.used_bytes,
                "rent": pool.rent,
                "churn": {
                    "admission_attempt_count": pool.admission_attempt_count,
                    "admission_success_count": pool.admission_success_count,
                    "admission_rejection_count": pool.admission_rejection_count,
                    "admitted_bytes": pool.admitted_bytes,
                    "eviction_count": pool.eviction_count,
                    "evicted_bytes": pool.evicted_bytes,
                    "removal_count": pool.removal_count,
                    "removed_bytes": pool.removed_bytes,
                    "churn_event_count": (
                        pool.admission_success_count + pool.eviction_count
                    ),
                    "churn_bytes": pool.admitted_bytes + pool.evicted_bytes,
                    "rejection_reason_counts": dict(
                        sorted(pool.rejection_reason_counts.items())
                    ),
                    "eviction_reason_counts": dict(
                        sorted(pool.eviction_reason_counts.items())
                    ),
                    "removal_reason_counts": dict(
                        sorted(pool.removal_reason_counts.items())
                    ),
                },
                "victim_frontiers": [
                    {
                        **asdict(frontier),
                        "tier": frontier.tier.value,
                    }
                    for frontier in self.victim_frontiers(tier)
                ],
                "records": [
                    {
                        **asdict(record),
                        "tier": record.tier.value,
                        "state": record.state.value,
                        "spec": asdict(record.spec),
                        "residency": self.residency(
                            record.spec.request_id, record.spec.ordinal
                        ).value,
                    }
                    for record in sorted(
                        pool.records.values(),
                        key=lambda item: (
                            item.spec.request_id,
                            item.spec.ordinal,
                        ),
                    )
                ],
            }
        return {
            "design_semantics": DESIGN_SEMANTICS,
            "time_ms": self._now_ms,
            "hbm_alpha": self.hbm_alpha,
            "stable_windows": self._stable_windows,
            "hbm_admission_paused": self._hbm_admission_paused,
            "pools": pools,
            "events": [dict(event) for event in self._events],
            "event_counts": dict(sorted(self._event_counts.items())),
            "event_sequence": self._event_sequence,
            "event_log_dropped": self._event_log_dropped,
            "record_history_count": self._record_history_count,
            "history": [
                {
                    **asdict(record),
                    "tier": record.tier.value,
                    "state": record.state.value,
                    "spec": asdict(record.spec),
                }
                for record in self._history
            ],
        }

    def telemetry_snapshot(self) -> dict[str, Any]:
        """Return bounded controller telemetry without materializing records.

        Large trace sweeps sample control windows frequently.  The complete
        :meth:`snapshot` intentionally serializes every record for mechanism
        audits, which would make telemetry cost grow with resident chunks.
        This bounded view exposes only state that is already maintained as
        counters and never changes controller semantics.
        """

        return {
            "design_semantics": DESIGN_SEMANTICS,
            "time_ms": self._now_ms,
            "hbm_alpha": self.hbm_alpha,
            "stable_windows": self._stable_windows,
            "hbm_admission_paused": self._hbm_admission_paused,
            "pools": {
                tier.value: {
                    "pool_id": pool.pool_id,
                    "capacity_bytes": pool.capacity_bytes,
                    "limit_bytes": pool.limit_bytes,
                    "used_bytes": pool.used_bytes,
                    "rent": pool.rent,
                    "record_count": len(pool.records),
                    "active_request_count": len(pool.live_ordinals),
                    "churn": {
                        "admission_attempt_count": pool.admission_attempt_count,
                        "admission_success_count": pool.admission_success_count,
                        "admission_rejection_count": pool.admission_rejection_count,
                        "admitted_bytes": pool.admitted_bytes,
                        "eviction_count": pool.eviction_count,
                        "evicted_bytes": pool.evicted_bytes,
                        "removal_count": pool.removal_count,
                        "removed_bytes": pool.removed_bytes,
                    },
                }
                for tier, pool in self._pools.items()
            },
            "event_counts": dict(sorted(self._event_counts.items())),
            "event_sequence": self._event_sequence,
        }
