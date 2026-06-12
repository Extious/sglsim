from __future__ import annotations

from dataclasses import dataclass

from sglsim.fault_tolerance.config import Strategy
from sglsim.fault_tolerance.reference import (
    DEFAULT_REFERENCE,
    SynthesisReference,
    concurrency_utilization,
)


@dataclass(frozen=True)
class LoadOverhead:
    latency_multiplier: float
    backup_coverage_scale: float
    queue_wait_scale: float


@dataclass(frozen=True)
class CacheHitBreakdown:
    """HiCache tier hits. peer_match is failover-readiness only, not normal serving."""

    l1_match: int
    l2_match: int
    host_backup_match: int
    peer_replica_match: int

    @property
    def prefill_serving_hits(self) -> int:
        return self.l1_match + self.l2_match


def steady_state_hicache_hits(
    *,
    input_len: int,
    shared_prefix_len: int,
    reference: SynthesisReference = DEFAULT_REFERENCE,
) -> CacheHitBreakdown:
    """Normal serving: L1 radix + L2 host tier (HiCache), identical across strategies."""
    l1 = min(max(0, shared_prefix_len), input_len, reference.shared_prefix_len)
    return CacheHitBreakdown(l1_match=l1, l2_match=0, host_backup_match=0, peer_replica_match=0)


def estimate_failover_recovery_hits(
    strategy: Strategy,
    *,
    input_len: int,
    output_generated: int,
    shared_prefix_len: int,
    peer_backup_lag_tokens: int = 8,
    concurrency: int = 16,
) -> CacheHitBreakdown:
    """KV available to healthy rank after GPU loss on failed rank."""
    l1 = min(max(0, shared_prefix_len), input_len)
    total = input_len + max(0, output_generated)

    if strategy in ("baseline",):
        # GPU KV gone; only HiCache host-resident prefix may survive.
        return CacheHitBreakdown(
            l1_match=l1,
            l2_match=0,
            host_backup_match=0,
            peer_replica_match=0,
        )

    if strategy == "host_backup":
        load = concurrency_utilization(concurrency)
        coverage = max(0.70, 0.95 - 0.08 * max(0.0, load - 0.5))
        backed = int(total * coverage)
        return CacheHitBreakdown(
            l1_match=l1,
            l2_match=0,
            host_backup_match=max(0, backed - l1),
            peer_replica_match=0,
        )

    if strategy == "gpu_backup":
        decode_backed = max(0, output_generated - peer_backup_lag_tokens)
        return CacheHitBreakdown(
            l1_match=l1,
            l2_match=0,
            host_backup_match=0,
            peer_replica_match=max(0, input_len - l1) + decode_backed,
        )

    if strategy == "remote_backup":
        load = concurrency_utilization(concurrency)
        coverage = max(0.75, 0.98 - 0.05 * max(0.0, load - 0.5))
        backed = int(total * coverage)
        return CacheHitBreakdown(
            l1_match=l1,
            l2_match=0,
            host_backup_match=max(0, backed - l1),
            peer_replica_match=0,
        )

    return CacheHitBreakdown(l1_match=l1, l2_match=0, host_backup_match=0, peer_replica_match=0)


def prefill_cache_hit_rate(hits: CacheHitBreakdown, *, input_len: int) -> float:
    if input_len <= 0:
        return 0.0
    return min(1.0, hits.prefill_serving_hits / float(input_len))


def failover_recovery_hit_rate(
    hits: CacheHitBreakdown,
    *,
    input_len: int,
    output_generated: int,
) -> float:
    total = input_len + max(0, output_generated)
    if total <= 0:
        return 0.0
    reusable = hits.l1_match + hits.l2_match + hits.host_backup_match + hits.peer_replica_match
    return min(1.0, reusable / float(total))


def estimate_load_overhead(strategy: Strategy, *, concurrency: int, dp_size: int) -> LoadOverhead:
    load = concurrency_utilization(concurrency)
    if strategy == "baseline":
        return LoadOverhead(latency_multiplier=1.0, backup_coverage_scale=1.0, queue_wait_scale=1.0)
    if strategy == "host_backup":
        return LoadOverhead(
            latency_multiplier=1.0 + 0.025 + 0.01 * load,
            backup_coverage_scale=max(0.70, 0.95 - 0.08 * max(0.0, load - 0.5)),
            queue_wait_scale=1.0 + 0.02 * load,
        )
    if strategy == "gpu_backup":
        vram = 0.02 + 0.48 * load
        return LoadOverhead(
            latency_multiplier=1.0 + vram + 0.008 * load,
            backup_coverage_scale=max(0.55, 0.98 - 0.15 * max(0.0, load - 0.4)),
            queue_wait_scale=1.0 + 0.05 * load,
        )
    return LoadOverhead(latency_multiplier=1.0, backup_coverage_scale=1.0, queue_wait_scale=1.0)
