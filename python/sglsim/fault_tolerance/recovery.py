from __future__ import annotations

from dataclasses import dataclass

from sglsim.fault_tolerance.load_model import estimate_failover_recovery_hits
from sglsim.srt.communication.transfer import TransferEstimator
from sglsim.srt.communication.types import CommPath


@dataclass(frozen=True)
class FailoverRecoveryCost:
    backed_up_tokens: int
    recompute_tokens: int
    restore_transfer_s: float
    failover_overhead_s: float
    reroute_rank: bool = True


def estimate_output_progress(output_len: int, *, fault_fraction: float = 0.35) -> int:
    if output_len <= 0:
        return 0
    return max(1, min(output_len, int(output_len * fault_fraction)))


def compute_failover_recovery_cost(
    strategy: str,
    *,
    input_len: int,
    output_len: int,
    output_generated: int | None = None,
    shared_prefix_len: int = 0,
    backed_up_tokens: int | None = None,
    transfer: TransferEstimator | None = None,
    peer_backup_lag_tokens: int = 8,
    failover_overhead_s: float = 0.5,
    concurrency: int = 16,
) -> FailoverRecoveryCost:
    """
    Failed rank loses all GPU KV. Healthy rank retries in-flight requests.

    - baseline: no backup; only HiCache L1/L2 prefix may remain on host.
    - host_backup: reload from failed rank's host-resident backup (H2D).
    - gpu_backup: reuse peer GPU replica; recompute decode gap only.
    """
    transfer = transfer or TransferEstimator()
    output_generated = (
        estimate_output_progress(output_len)
        if output_generated is None
        else output_generated
    )
    hits = estimate_failover_recovery_hits(
        "baseline" if strategy in ("none",) else strategy,
        input_len=input_len,
        output_generated=output_generated,
        shared_prefix_len=shared_prefix_len,
        peer_backup_lag_tokens=peer_backup_lag_tokens,
        concurrency=concurrency,
    )
    if backed_up_tokens is None:
        backed_up_tokens = (
            hits.l1_match + hits.l2_match + hits.host_backup_match + hits.peer_replica_match
        )

    total = input_len + output_generated

    if strategy in ("baseline", "none"):
        recompute = max(0, total - hits.l1_match)
        restore = transfer.estimate_seconds(CommPath.H2D, hits.l1_match) if hits.l1_match else 0.0
    elif strategy == "host_backup":
        recompute = max(0, total - backed_up_tokens)
        restore = transfer.estimate_seconds(CommPath.H2D, hits.host_backup_match + hits.l1_match)
    elif strategy == "gpu_backup":
        decode_backed = max(0, output_generated - peer_backup_lag_tokens)
        recompute = max(0, output_generated - decode_backed)
        restore = 0.0
        failover_overhead_s = 0.05
    elif strategy == "remote_backup":
        recompute = max(0, total - backed_up_tokens)
        restore = transfer.estimate_host_restore_seconds(backed_up_tokens)
    else:
        recompute = total
        restore = 0.0

    return FailoverRecoveryCost(
        backed_up_tokens=backed_up_tokens,
        recompute_tokens=recompute,
        restore_transfer_s=restore,
        failover_overhead_s=failover_overhead_s,
    )
