from __future__ import annotations

from queue import Queue

from sglsim.srt.managers.data_parallel_controller import DataParallelController
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.network import _QUEUES


def _controller(*, dp_size: int = 3, pp_size: int = 1) -> DataParallelController:
    ConfigManager._raw = {
        "scheduler": {
            "dp_size": dp_size,
            "pp_size": pp_size,
            "load_balance_method": "round_robin",
        },
        "aegis_l1_l2": {
            "enabled": True,
            "committed_copy_protocol": True,
            "design_semantics": DESIGN_SEMANTICS,
            "host_alignment_tokens": 1,
            "topology": {
                "dp_rank_to_node": list(range(dp_size)),
                "pp_stages_per_node": pp_size,
                "gpu_backup_requires_direct_retry_route": True,
                "host_pool_scope": "node_shared_process_external",
            },
        },
    }
    ConfigManager._scheduler = None
    ConfigManager._cache = None
    ConfigManager._hicache = None
    server_args = ServerArgs(dp_size=dp_size, pp_size=pp_size)
    port_args = PortArgs.init_new(server_args)
    _QUEUES.clear()
    return DataParallelController(
        server_args,
        port_args,
        scheduler_cmd_queues=[Queue() for _ in range(dp_size)],
    )


def _committed_hint() -> dict:
    return {
        "rid": "latest-dpc",
        "backup_key": "latest-dpc",
        "source_dp_rank": 0,
        "target_dp_rank": 1,
        "version": 1,
        "ack": True,
        "token_count": 1024,
        "source_committed_token_end": 1024,
        "l1_tokens": 1024,
        "l2_tokens": 1024,
        "l3_tokens": 0,
        "recompute_tokens": 0,
        "safe_frontier_tokens": 1024,
        "materialization_scope": "committed_prefix",
        "group_commit_verified": True,
        "execution_id": "latest-copy-1",
        "block_generation": 0,
        "design_semantics": DESIGN_SEMANTICS,
        "failure_scope": "node",
        "gpu_candidate": {
            "valid": True,
            "reusable_dp_ranks": [1],
            "recovery_path": ["direct_hbm_reuse"],
            "backup_path": ["gpu_rdma"],
        },
        "host_candidate": {
            "valid": True,
            "host_pool_node_id": 1,
            "recovery_path": ["h2d"],
            "backup_path": ["d2h"],
        },
        "host_pool_lease": {"lease_version": 1},
        "updated_time": 1.0,
    }


def test_latest_refresh_counts_overlapping_committed_tiers_once() -> None:
    controller = _controller()
    payload = controller._refresh_aegis_recovery_hint(_committed_hint())

    assert payload["l1_tokens"] == 1024
    assert payload["l2_tokens"] == 1024
    assert payload["safe_frontier_tokens"] == 1024
    assert payload["recompute_tokens"] == 0
    assert payload["residency"] == "HBM_AND_DRAM"
    assert payload["recovery_action"] == "l1_backup"


def test_latest_refresh_never_promotes_reservation_to_committed_copy() -> None:
    controller = _controller()
    reservation = _committed_hint()
    reservation.update(
        {
            "materialization_scope": "metadata_reservation_only",
            "group_commit_verified": False,
            "source_committed_token_end": None,
        }
    )

    payload = controller._refresh_aegis_recovery_hint(reservation)

    assert payload["safe_frontier_tokens"] == 0
    assert payload["recompute_tokens"] == 1024
    assert payload["residency"] == "BYPASS"
    assert payload["recovery_action"] == "recompute"


def test_latest_retry_chooses_one_independent_path_not_tier_sum() -> None:
    controller = _controller()
    hint = controller._refresh_aegis_recovery_hint(_committed_hint())

    hbm = controller._aegis_retry_option(
        hint=hint,
        retry_dp_rank=1,
        requested_tokens=1024,
        failed_dp_rank=0,
        cache_key="",
    )
    dram = controller._aegis_retry_option(
        hint=hint,
        retry_dp_rank=2,
        requested_tokens=1024,
        failed_dp_rank=0,
        cache_key="",
    )

    assert hbm["restored_tokens"] == 1024
    assert hbm["l1_tokens"] == 1024
    assert hbm["l2_tokens"] == 0
    assert hbm["action"] == "l1_backup"
    assert dram["restored_tokens"] == 1024
    assert dram["l1_tokens"] == 0
    assert dram["l2_tokens"] == 1024
    assert dram["action"] == "l2_backup"


def test_latest_stage_ack_uses_source_frontier_not_hbm_plus_dram() -> None:
    controller = _controller(dp_size=2, pp_size=1)
    payload = _committed_hint()
    event = {
        "pp_rank": 0,
        "source_pp_rank": 0,
        "target_pp_rank": 0,
        "route_version": 1,
        "budget_version": 1,
    }

    ack = controller._stage_commit_ack_from_payload(
        event=event,
        payload=payload,
    )

    assert ack.stage_total_tokens == 1024
    assert ack.committed_frontier_tokens == 1024
    assert ack.hbm_tokens == 1024
    assert ack.host_tokens == 0
    assert ack.recompute_tokens == 0
    # Both direct source copies occurred even though recovery selected HBM.
    assert ack.copy_bytes == 0  # bytes/token is intentionally absent here.


def test_latest_pp_recovery_can_mix_independent_tiers_across_stages() -> None:
    controller = _controller(dp_size=3, pp_size=2)
    first = _committed_hint()
    first.update({"pp_rank": 0, "l1_tokens": 1024, "l2_tokens": 512})
    second = _committed_hint()
    second.update({"pp_rank": 1, "l1_tokens": 512, "l2_tokens": 1024})
    hint = {
        **_committed_hint(),
        "all_stages_committed": True,
        "stage_payloads": [first, second],
    }

    option = controller._pipeline_independent_retry_option(
        hint=hint,
        retry_dp_rank=1,
        requested_tokens=1024,
        failed_dp_rank=0,
        live_primary_tokens=0,
    )

    assert option["restored_tokens"] == 1024
    assert option["recompute_tokens"] == 0
    assert option["safe_frontier_tokens"] == 1024
    assert option["action"] == "stage_aligned_mixed_tier_backup"
    assert [row["selected_tier"] for row in option["per_stage"]] == [
        "hbm",
        "dram",
    ]
