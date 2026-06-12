import pytest

from sglsim.fault_tolerance.recovery import compute_failover_recovery_cost
from sglsim.srt.communication.transfer import TransferEstimator
from sglsim.srt.communication.types import CommPath


def test_comm_path_includes_peer():
    assert CommPath.PEER.value == "peer"


def test_recovery_cost_ordering_none_host_gpu():
    transfer = TransferEstimator()
    base_kwargs = dict(input_len=100, output_len=40, output_generated=10, transfer=transfer)
    none = compute_failover_recovery_cost("none", **base_kwargs)
    host = compute_failover_recovery_cost("host_backup", **base_kwargs)
    gpu = compute_failover_recovery_cost("gpu_backup", **base_kwargs, peer_backup_lag_tokens=2)

    assert none.recompute_tokens > host.recompute_tokens > gpu.recompute_tokens
    assert host.restore_transfer_s > 0
    assert gpu.restore_transfer_s == 0


def test_peer_transfer_uses_peer_bandwidth():
    estimator = TransferEstimator()
    seconds = estimator.estimate_seconds(CommPath.PEER, num_tokens=100)
    assert seconds > 0
