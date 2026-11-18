from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_GPU_REPLICA_ALPHA = 0.30
# Mirror pinned SGLang sgl-model-gateway CacheAwareConfig defaults.  AegisKV
# extends the single-worker choice to an ordered Top-2, but does not change the
# cache/load switch boundary.
DEFAULT_CACHE_MATCH_THRESHOLD = 0.50
DEFAULT_BALANCE_ABS_THRESHOLD = 32
DEFAULT_BALANCE_REL_THRESHOLD = 1.0001


def normalized_fraction(value: object) -> float | None:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction < 0.0:
        return None
    return max(0.0, min(fraction, 1.0))


@dataclass(frozen=True)
class GpuReplicaReservation:
    rid: str
    source_dp_rank: int
    backup_dp_rank: int
    cache_key: str
    requested_tokens: int
    quota_tokens: int
    overlap_tokens: int
    unique_tokens: int
    reserved_tokens: int
    alpha: float
    evictable: bool = False


@dataclass(frozen=True)
class RankPlacementScore:
    rank: int
    match_rate: float
    covered_tokens: int
    active_unique_tokens: int
    inflight_requests: int
    inflight_tokens: int
    available_backup_tokens: int

    def for_cache_first(self) -> tuple[float, int, int, int, int, int]:
        return (
            -self.match_rate,
            -self.covered_tokens,
            self.inflight_tokens,
            self.inflight_requests,
            self.active_unique_tokens,
            self.rank,
        )

    def for_load_first(self) -> tuple[int, int, int, float, int, int]:
        return (
            self.inflight_tokens,
            self.inflight_requests,
            self.active_unique_tokens,
            -self.match_rate,
            -self.covered_tokens,
            self.rank,
        )


class GpuReplicaLedger:
    """Metadata ledger for AegisKV GPUReplica placement.

    Inference KV and backup KV share the same per-rank prefix map here.  Tokens
    already present for the same cache key count as overlap and do not consume
    the alpha-controlled backup quota.
    """

    def __init__(
        self,
        *,
        dp_size: int,
        total_cache_tokens: int,
        cache_match_threshold: float = DEFAULT_CACHE_MATCH_THRESHOLD,
    ) -> None:
        self.dp_size = max(int(dp_size), 1)
        self.total_cache_tokens = max(int(total_cache_tokens), 0)
        self.cache_match_threshold = max(0.0, min(float(cache_match_threshold), 1.0))
        self._prefix_tokens_by_rank: list[dict[str, int]] = [
            {} for _ in range(self.dp_size)
        ]
        # Keep execution locality separate from protection-only prefixes so a
        # recovered rank rebuilds its former work instead of adopting an
        # arbitrary backup key.
        self._inference_cache_keys_by_rank: list[set[str]] = [
            set() for _ in range(self.dp_size)
        ]
        self._active_unique_tokens_by_rank = [0 for _ in range(self.dp_size)]
        self._inflight_by_rank = [0 for _ in range(self.dp_size)]
        self._inflight_tokens_by_rank = [0 for _ in range(self.dp_size)]
        self._inference_rank_by_rid: dict[str, int] = {}
        self._inference_tokens_by_rid: dict[str, int] = {}
        self._reservations: dict[str, GpuReplicaReservation] = {}

    def quota_tokens(self, alpha: float) -> int:
        return int(self.total_cache_tokens * max(0.0, min(float(alpha), 1.0)))

    def active_unique_tokens(self, dp_rank: int) -> int:
        if not 0 <= dp_rank < self.dp_size:
            return 0
        return self._active_unique_tokens_by_rank[dp_rank]

    def inflight_requests(self, dp_rank: int) -> int:
        if not 0 <= dp_rank < self.dp_size:
            return 0
        return self._inflight_by_rank[dp_rank]

    def inflight_tokens(self, dp_rank: int) -> int:
        if not 0 <= dp_rank < self.dp_size:
            return 0
        return self._inflight_tokens_by_rank[dp_rank]

    def prefix_tokens(self, dp_rank: int, cache_key: str) -> int:
        if not 0 <= dp_rank < self.dp_size or not cache_key:
            return 0
        return max(int(self._prefix_tokens_by_rank[dp_rank].get(cache_key, 0)), 0)

    def clear_rank_l1(self, dp_rank: int) -> dict[str, Any]:
        """Forget Router-side GPU cache state after a rank loses L1 KV."""

        if not 0 <= dp_rank < self.dp_size:
            return {
                "cleared_prefix_keys": 0,
                "cleared_prefix_tokens": 0,
                "cleared_reservations": 0,
                "cleared_inference_cache_keys": (),
            }
        inference_cache_keys = tuple(
            sorted(self._inference_cache_keys_by_rank[dp_rank])
        )
        self._inference_cache_keys_by_rank[dp_rank].clear()
        prefixes = self._prefix_tokens_by_rank[dp_rank]
        cleared_prefix_tokens = sum(max(int(value), 0) for value in prefixes.values())
        cleared_prefix_keys = len(prefixes)
        prefixes.clear()

        stale_rids = [
            rid
            for rid, rank in self._inference_rank_by_rid.items()
            if rank == dp_rank
        ]
        for rid in stale_rids:
            self._inference_rank_by_rid.pop(rid, None)
            self._inference_tokens_by_rid.pop(rid, None)
        self._inflight_by_rank[dp_rank] = 0
        self._inflight_tokens_by_rank[dp_rank] = 0

        stale_reservations = [
            rid
            for rid, reservation in self._reservations.items()
            if reservation.backup_dp_rank == dp_rank
        ]
        for rid in stale_reservations:
            self._reservations.pop(rid, None)
        self._active_unique_tokens_by_rank[dp_rank] = 0
        return {
            "cleared_prefix_keys": cleared_prefix_keys,
            "cleared_prefix_tokens": cleared_prefix_tokens,
            "cleared_reservations": len(stale_reservations),
            "cleared_inference_cache_keys": inference_cache_keys,
        }

    def record_inference(
        self,
        *,
        rid: str,
        dp_rank: int,
        cache_key: str,
        token_count: int,
    ) -> None:
        if not rid or not 0 <= dp_rank < self.dp_size:
            return
        self._inference_rank_by_rid[rid] = dp_rank
        token_count = max(int(token_count), 0)
        self._inference_tokens_by_rid[rid] = token_count
        if cache_key:
            self._inference_cache_keys_by_rank[dp_rank].add(cache_key)
        self._inflight_by_rank[dp_rank] += 1
        self._inflight_tokens_by_rank[dp_rank] += token_count
        self._record_prefix(
            dp_rank=dp_rank,
            cache_key=cache_key,
            token_count=token_count,
        )

    def placement_score(
        self,
        *,
        rank: int,
        cache_key: str,
        requested_tokens: int,
        alpha: float,
    ) -> RankPlacementScore:
        requested = max(int(requested_tokens), 0)
        quota = self.quota_tokens(alpha)
        overlap = min(self.prefix_tokens(rank, cache_key), requested)
        active = self.active_unique_tokens(rank)
        available = max(quota - active, 0)
        covered = min(requested, overlap + available)
        match_rate = overlap / requested if requested > 0 else 0.0
        return RankPlacementScore(
            rank=int(rank),
            match_rate=match_rate,
            covered_tokens=covered,
            active_unique_tokens=active,
            inflight_requests=self.inflight_requests(rank),
            inflight_tokens=self.inflight_tokens(rank),
            available_backup_tokens=available,
        )

    def choose_execution_rank(
        self,
        *,
        cache_key: str,
        requested_tokens: int,
        healthy_ranks: list[int],
        alpha: float = DEFAULT_GPU_REPLICA_ALPHA,
    ) -> int:
        candidates = [
            int(rank)
            for rank in healthy_ranks
            if 0 <= int(rank) < self.dp_size
        ]
        if not candidates:
            return -1
        scores = [
            self.placement_score(
                rank=rank,
                cache_key=cache_key,
                requested_tokens=requested_tokens,
                alpha=alpha,
            )
            for rank in candidates
        ]
        if self._load_imbalanced(scores):
            return min(scores, key=lambda score: score.for_load_first()).rank
        matched = [
            score
            for score in scores
            if score.match_rate > self.cache_match_threshold
        ]
        if matched:
            return min(matched, key=lambda score: score.for_cache_first()).rank
        return min(scores, key=lambda score: score.for_load_first()).rank

    def rank_execution_candidates(
        self,
        *,
        cache_key: str,
        requested_tokens: int,
        healthy_ranks: list[int],
    ) -> list[int]:
        """Return Cache-Aware Top-K using only normal load and prefix state.

        This is the Router-side ranking used by the L1/L2 AegisKV path.  It
        deliberately does not inspect target backup-partition occupancy.
        """

        candidates = [
            int(rank)
            for rank in healthy_ranks
            if 0 <= int(rank) < self.dp_size
        ]
        scores = [
            self.placement_score(
                rank=rank,
                cache_key=cache_key,
                requested_tokens=requested_tokens,
                alpha=0.0,
            )
            for rank in candidates
        ]
        if self._load_imbalanced(scores):
            scores.sort(key=lambda score: score.for_load_first())
            return [score.rank for score in scores]

        matched = [
            score
            for score in scores
            if score.match_rate > self.cache_match_threshold
        ]
        unmatched = [score for score in scores if score not in matched]
        matched.sort(key=lambda score: score.for_cache_first())
        unmatched.sort(key=lambda score: score.for_load_first())
        return [score.rank for score in matched + unmatched]

    def _record_prefix(self, *, dp_rank: int, cache_key: str, token_count: int) -> None:
        if not 0 <= dp_rank < self.dp_size or not cache_key:
            return
        current = self._prefix_tokens_by_rank[dp_rank].get(cache_key, 0)
        self._prefix_tokens_by_rank[dp_rank][cache_key] = max(
            int(current),
            max(int(token_count), 0),
        )

    @staticmethod
    def _load_imbalanced(
        scores: list[RankPlacementScore],
        *,
        abs_threshold: int = DEFAULT_BALANCE_ABS_THRESHOLD,
        rel_threshold: float = DEFAULT_BALANCE_REL_THRESHOLD,
    ) -> bool:
        if not scores:
            return False
        loads = [score.inflight_requests for score in scores]
        min_load = min(loads)
        max_load = max(loads)
        return (
            max_load - min_load > int(abs_threshold)
            and float(max_load) > float(min_load) * float(rel_threshold)
        )

    def choose_backup_rank(
        self,
        *,
        source_dp_rank: int,
        cache_key: str,
        requested_tokens: int,
        alpha: float,
        healthy_ranks: list[int],
    ) -> int:
        requested = max(int(requested_tokens), 0)
        if requested <= 0 or self.dp_size <= 1:
            return -1
        candidates = [
            rank
            for rank in healthy_ranks
            if 0 <= int(rank) < self.dp_size and int(rank) != int(source_dp_rank)
        ]
        if not candidates:
            candidates = [
                rank for rank in healthy_ranks if 0 <= int(rank) < self.dp_size
            ]
        if not candidates:
            return -1

        quota = self.quota_tokens(alpha)

        def rank_score(rank: int) -> RankPlacementScore:
            return self.placement_score(
                rank=rank,
                cache_key=cache_key,
                requested_tokens=requested,
                alpha=alpha,
            )

        scores = [rank_score(rank) for rank in candidates]
        if self._load_imbalanced(scores):
            return min(scores, key=lambda score: score.for_load_first()).rank

        matched = [
            score for score in scores if score.match_rate > self.cache_match_threshold
        ]
        if matched:
            return min(matched, key=lambda score: score.for_cache_first()).rank

        return min(scores, key=lambda score: score.for_load_first()).rank

    def reserve(
        self,
        *,
        rid: str,
        source_dp_rank: int,
        cache_key: str,
        requested_tokens: int,
        alpha: float,
        healthy_ranks: list[int],
    ) -> GpuReplicaReservation | None:
        if not rid:
            return None
        requested = max(int(requested_tokens), 0)
        backup_dp_rank = self.choose_backup_rank(
            source_dp_rank=source_dp_rank,
            cache_key=cache_key,
            requested_tokens=requested,
            alpha=alpha,
            healthy_ranks=healthy_ranks,
        )
        if backup_dp_rank < 0:
            return None

        quota = self.quota_tokens(alpha)
        overlap = min(self.prefix_tokens(backup_dp_rank, cache_key), requested)
        active = self.active_unique_tokens(backup_dp_rank)
        available = max(quota - active, 0)
        unique = min(max(requested - overlap, 0), available)
        reserved = min(requested, overlap + unique)
        reservation = GpuReplicaReservation(
            rid=rid,
            source_dp_rank=int(source_dp_rank),
            backup_dp_rank=int(backup_dp_rank),
            cache_key=cache_key,
            requested_tokens=requested,
            quota_tokens=quota,
            overlap_tokens=overlap,
            unique_tokens=unique,
            reserved_tokens=reserved,
            alpha=max(0.0, min(float(alpha), 1.0)),
        )
        self._reservations[rid] = reservation
        self._active_unique_tokens_by_rank[backup_dp_rank] += unique
        self._record_prefix(
            dp_rank=backup_dp_rank,
            cache_key=cache_key,
            token_count=reserved,
        )
        return reservation

    def release_request(self, rid: str) -> GpuReplicaReservation | None:
        if not rid:
            return None
        inference_rank = self._inference_rank_by_rid.pop(rid, None)
        if inference_rank is not None and 0 <= inference_rank < self.dp_size:
            self._inflight_by_rank[inference_rank] = max(
                self._inflight_by_rank[inference_rank] - 1,
                0,
            )
            released_tokens = self._inference_tokens_by_rid.pop(rid, 0)
            self._inflight_tokens_by_rank[inference_rank] = max(
                self._inflight_tokens_by_rank[inference_rank]
                - max(int(released_tokens), 0),
                0,
            )

        reservation = self._reservations.get(rid)
        if reservation is None:
            return None
        if reservation.evictable:
            return reservation

        self._active_unique_tokens_by_rank[reservation.backup_dp_rank] = max(
            self._active_unique_tokens_by_rank[reservation.backup_dp_rank]
            - reservation.unique_tokens,
            0,
        )
        evictable = GpuReplicaReservation(
            rid=reservation.rid,
            source_dp_rank=reservation.source_dp_rank,
            backup_dp_rank=reservation.backup_dp_rank,
            cache_key=reservation.cache_key,
            requested_tokens=reservation.requested_tokens,
            quota_tokens=reservation.quota_tokens,
            overlap_tokens=reservation.overlap_tokens,
            unique_tokens=reservation.unique_tokens,
            reserved_tokens=reservation.reserved_tokens,
            alpha=reservation.alpha,
            evictable=True,
        )
        self._reservations[rid] = evictable
        return evictable

    def reservation(self, rid: str) -> GpuReplicaReservation | None:
        return self._reservations.get(rid)
