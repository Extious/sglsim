from __future__ import annotations

from sglsim.srt.managers.io_struct import ReqSnapshot, TokenizedGenerateReqInput
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.managers.scheduler import Scheduler
from sglsim.srt.managers.tokenizer_manager import TokenizerManager
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.network import SimIpcRegistry


def _scheduler_with_gpu_backup() -> Scheduler:
    SimIpcRegistry.install_global(None)
    ConfigManager._raw = {
        "scheduler": {
            "max_prefill_tokens": 8192,
            "chunked_prefill_size": 8192,
            "dp_size": 2,
        },
        "cache": {
            "page_size": 1,
            "enable_hierarchical_cache": False,
            "l1_capacity_pages": 100_000,
            "l2_capacity_pages": 0,
        },
        "fault_tolerance": {
            "enabled": True,
            "strategy": "gpu_backup",
            "peer_backup_lag_tokens": 2,
            "backup_policy": "all",
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    ConfigManager._hicache = None
    ConfigManager._fault_tolerance = None
    server_args = ServerArgs()
    return Scheduler(server_args, PortArgs.init_new(server_args), dp_rank=0)


def test_gpu_backup_counts_prompt_and_lagged_decode_tokens_no_hicache():
    scheduler = _scheduler_with_gpu_backup()
    req = Req(
        "r0",
        list(range(16)),
        SamplingParams(max_new_tokens=8),
        original_prompt_len=16,
    )

    scheduler._update_peer_backup_after_prefill(req)
    assert req.peer_backed_up_tokens == 16

    req.output_ids.extend([101, 102, 103, 104, 105])
    scheduler._update_peer_backup_after_decode(req)

    assert req.peer_backed_up_tokens == 19
    assert scheduler._backed_up_tokens_for_snapshot(req) == 19


def test_failover_retry_rebuilds_same_rid_with_reduced_recompute_tokens():
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.rid_to_tokenized_state = {}
    manager.pending_failover_rids = set()
    manager.failover_failed_dp_by_rid = {}
    captured: list[TokenizedGenerateReqInput] = []
    manager.dispatch_tokenized_request = captured.append  # type: ignore[method-assign]

    sampling = SamplingParams(max_new_tokens=8)
    snap = ReqSnapshot(
        rid="agent-1-t0",
        origin_input_ids=list(range(16)),
        origin_input_token_count=16,
        output_ids=[201, 202, 203],
        backed_up_tokens=18,
        sampling_params=sampling,
        original_max_new_tokens=8,
        simulation={"server_created_time": 10.0, "failover_injected_time": 12.0},
        routing_key="agent-1",
    )

    manager._retry_snapshot(snap, failed_dp_rank=0)

    assert len(captured) == 1
    retry = captured[0]
    assert retry.rid == "agent-1-t0"
    assert retry.input_ids == list(range(16)) + [201, 202, 203]
    assert retry.input_token_count == 19
    assert retry.sampling_params.max_new_tokens == 5
    assert retry.pre_failover_output_tokens == 3
    assert retry.pre_failover_backed_up_tokens == 18
    assert retry.failover_recompute_tokens == 1
    assert retry.failed_dp_rank == 0
    assert retry.is_failover_retry is True


def test_gpu_backup_critical_agent_policy_skips_noncritical_requests():
    scheduler = _scheduler_with_gpu_backup()
    scheduler.ft_config = type(
        "Cfg",
        (),
        {
            "enabled": True,
            "strategy": "gpu_backup",
            "peer_backup_lag_tokens": 2,
            "backup_prefill_tokens": "all",
            "backup_decode": True,
            "backup_policy": "critical_agents",
        },
    )()
    critical = Req(
        "critical",
        list(range(8)),
        SamplingParams(
            max_new_tokens=4,
            custom_params={"simulation": {"critical_agent": True}},
        ),
        original_prompt_len=8,
    )
    normal = Req(
        "normal",
        list(range(8)),
        SamplingParams(
            max_new_tokens=4,
            custom_params={"simulation": {"critical_agent": False}},
        ),
        original_prompt_len=8,
    )

    scheduler._update_peer_backup_after_prefill(critical)
    scheduler._update_peer_backup_after_prefill(normal)

    assert critical.peer_backed_up_tokens == 8
    assert normal.peer_backed_up_tokens == 0
