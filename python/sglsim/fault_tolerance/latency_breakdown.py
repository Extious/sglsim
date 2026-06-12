from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sglsim.fault_tolerance.config import Strategy
from sglsim.fault_tolerance.recovery import compute_failover_recovery_cost, estimate_output_progress
from sglsim.fault_tolerance.reference import DEFAULT_REFERENCE, SynthesisReference
from sglsim.srt.communication.transfer import TransferEstimator


@dataclass(frozen=True)
class LatencySegments:
    queue_s: float
    prefill_s: float
    decode_s: float
    prefill_before_s: float = 0.0
    decode_before_s: float = 0.0
    load_backup_s: float = 0.0
    recompute_after_s: float = 0.0
    decode_after_s: float = 0.0
    prefill_cached_tokens: int = 0
    prefill_total_tokens: int = 0
    affected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def service_s(self) -> float:
        if self.affected:
            return (
                self.prefill_before_s
                + self.decode_before_s
                + self.load_backup_s
                + self.recompute_after_s
                + self.decode_after_s
            )
        return self.prefill_s + self.decode_s

    @property
    def total_s(self) -> float:
        return self.queue_s + self.service_s


def _prefill_seconds(fresh_tokens: int, *, reference: SynthesisReference) -> float:
    if fresh_tokens <= 0:
        return 0.0
    return fresh_tokens / reference.cached_prefill_tok_s


def _decode_seconds(num_tokens: int, *, reference: SynthesisReference) -> float:
    if num_tokens <= 0:
        return 0.0
    return num_tokens * (reference.decode_ms_per_token / 1000.0)


def compute_latency_segments(
    *,
    strategy: Strategy,
    input_len: int,
    output_len: int,
    shared_prefix_len: int,
    affected: bool,
    latency_s: float,
    recompute_tokens: int = 0,
    restore_transfer_s: float = 0.0,
    failover_overhead_s: float = 0.0,
    output_generated: int | None = None,
    fault_fraction: float = 0.35,
    peer_backup_lag_tokens: int = 8,
    concurrency: int = 16,
    transfer: TransferEstimator | None = None,
    reference: SynthesisReference = DEFAULT_REFERENCE,
) -> LatencySegments:
    transfer = transfer or TransferEstimator()
    prefill_cached = min(max(0, shared_prefix_len), input_len)
    prefill_fresh = max(0, input_len - prefill_cached)
    prefill_s = _prefill_seconds(prefill_fresh, reference=reference)
    decode_s = _decode_seconds(output_len, reference=reference)

    if not affected:
        queue_s = max(0.0, latency_s - prefill_s - decode_s)
        return LatencySegments(
            queue_s=queue_s,
            prefill_s=prefill_s,
            decode_s=decode_s,
            prefill_cached_tokens=prefill_cached,
            prefill_total_tokens=input_len,
            affected=False,
        )

    output_generated = (
        estimate_output_progress(output_len, fault_fraction=fault_fraction)
        if output_generated is None
        else output_generated
    )
    recovery = compute_failover_recovery_cost(
        strategy,
        input_len=input_len,
        output_len=output_len,
        output_generated=output_generated,
        shared_prefix_len=shared_prefix_len,
        backed_up_tokens=None,
        transfer=transfer,
        peer_backup_lag_tokens=peer_backup_lag_tokens,
        failover_overhead_s=failover_overhead_s,
        concurrency=concurrency,
    )
    recompute_tokens = recovery.recompute_tokens
    restore_transfer_s = recovery.restore_transfer_s
    failover_overhead_s = recovery.failover_overhead_s

    prefill_before_s = prefill_s
    decode_before_s = _decode_seconds(output_generated, reference=reference)
    decode_after_s = _decode_seconds(max(0, output_len - output_generated), reference=reference)
    recompute_after_s = _prefill_seconds(recompute_tokens, reference=reference)

    if strategy in ("baseline", "none"):
        load_backup_s = 0.0
        recompute_after_s = _prefill_seconds(recompute_tokens, reference=reference) + failover_overhead_s
    else:
        load_backup_s = restore_transfer_s + failover_overhead_s
        recompute_after_s = _prefill_seconds(recompute_tokens, reference=reference)

    service_s = (
        prefill_before_s
        + decode_before_s
        + load_backup_s
        + recompute_after_s
        + decode_after_s
    )
    queue_s = max(0.0, latency_s - service_s)
    return LatencySegments(
        queue_s=queue_s,
        prefill_s=0.0,
        decode_s=0.0,
        prefill_before_s=prefill_before_s,
        decode_before_s=decode_before_s,
        load_backup_s=load_backup_s,
        recompute_after_s=recompute_after_s,
        decode_after_s=decode_after_s,
        prefill_cached_tokens=prefill_cached,
        prefill_total_tokens=input_len,
        affected=True,
    )
