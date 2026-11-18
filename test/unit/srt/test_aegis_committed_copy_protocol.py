from __future__ import annotations

import logging
from queue import Queue
from types import SimpleNamespace

from sglsim.srt.managers.aegis_l1_l2 import (
    AegisBackupPlan,
    AegisL1L2Config,
    AegisRankBackupManager,
)
from sglsim.srt.managers.aegis_independent_tier_manager import (
    AegisIndependentTierRankManager,
)
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.managers.scheduler import Scheduler
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS


def _source_req() -> Req:
    return Req(
        rid="source-request",
        input_ids=list(range(64)),
        sampling_params=SamplingParams(
            max_new_tokens=64,
            custom_params={
                "simulation": {
                    "aegis_backup_plan_sent": True,
                    "aegis_backup_key": "source-object",
                    "aegis_backup_dp_rank": 1,
                    "aegis_backup_version": 2,
                    "aegis_route_version": 3,
                    "aegis_budget_version": 4,
                    "aegis_reserved_token_count": 128,
                    "aegis_copy_execution_id": "execution-2",
                    "aegis_block_generation": 5,
                }
            },
        ),
    )


def test_source_publishes_full_chunks_and_one_terminal_partial_chunk() -> None:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.aegis_l1_l2_enabled = True
    scheduler.aegis_committed_copy_protocol = True
    scheduler._aegis_event_queue = Queue()
    scheduler._aegis_last_source_commit = {}
    scheduler.dp_rank = 0
    scheduler.pp_rank = 0
    scheduler.server_args = SimpleNamespace(pp_size=1)
    scheduler._time_driver = SimpleNamespace(now=lambda: 12.5)
    scheduler.aegis_backup_manager = AegisRankBackupManager(
        dp_rank=0,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            chunk_tokens=64,
        ),
    )
    req = _source_req()

    req.kv_committed_length = 63
    scheduler._emit_aegis_source_chunk_committed(req)
    assert scheduler._aegis_event_queue.empty()

    req.kv_committed_length = 64
    scheduler._emit_aegis_source_chunk_committed(req)
    first = scheduler._aegis_event_queue.get_nowait()
    assert first["source_committed_token_end"] == 64
    assert first["source_observed_token_end"] == 64

    req.kv_committed_length = 70
    scheduler._emit_aegis_source_chunk_committed(req)
    assert scheduler._aegis_event_queue.empty()
    scheduler._emit_aegis_source_chunk_committed(
        req,
        request_finished=True,
    )
    final = scheduler._aegis_event_queue.get_nowait()
    assert final["source_committed_token_end"] == 70
    assert final["request_finished"] is True


def test_target_emits_group_commit_only_after_materialization() -> None:
    manager = AegisRankBackupManager(
        dp_rank=1,
        config=AegisL1L2Config(
            l1_total_tokens=1024,
            l2_total_tokens=1024,
            initial_alpha=1.0,
            initial_beta=1.0,
            chunk_tokens=64,
        ),
    )
    manager.place(
        AegisBackupPlan(
            rid="target-request",
            backup_key="target-object",
            source_dp_rank=0,
            target_dp_rank=1,
            token_count=128,
            critical_probability=1.0,
            slack_ms=0.0,
            version=2,
        )
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.aegis_l1_l2_enabled = True
    scheduler.aegis_committed_copy_protocol = True
    scheduler.aegis_backup_manager = manager
    scheduler._aegis_event_queue = Queue()
    scheduler.dp_rank = 1
    scheduler.pp_rank = 0
    scheduler.tp_rank = 0
    scheduler.server_args = SimpleNamespace(pp_size=1)
    scheduler.logger = logging.getLogger("test-aegis-group-commit")

    should_stop = scheduler._handle_runtime_command(
        (
            "aegis_materialize_committed_prefix",
            {
                "request_backup_key": "target-object",
                "stage_backup_key": "target-object",
                "source_pp_rank": 0,
                "target_pp_rank": 0,
                "required_pp_size": 1,
                "version": 2,
                "route_version": 3,
                "budget_version": 4,
                "source_committed_token_end": 64,
                "execution_id": "execution-target-2",
                "block_generation": 5,
            },
        )
    )

    assert should_stop is False
    event = scheduler._aegis_event_queue.get_nowait()
    assert event["type"] == "aegis_l1_l2_group_commit_ack"
    assert event["payload"]["materialization_scope"] == "committed_prefix"
    assert event["payload"]["group_commit_verified"] is True
    assert event["payload"]["safe_frontier_tokens"] == 64
    assert event["payload"]["source_committed_token_end"] == 64


def test_latest_target_group_commit_keeps_hbm_and_dram_overlapping() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config={
            "design_semantics": DESIGN_SEMANTICS,
            "committed_copy_protocol": True,
            "chunk_tokens": 64,
                "initial_alpha": 1.0,
                "min_alpha": 0.1,
                "max_alpha": 1.0,
                "recompute_setup_ms": 0.0,
                "recompute_ms_per_token": 0.01,
                "l1_restore_setup_ms": 0.0,
                "l1_restore_ms_per_token": 0.001,
                "l2_restore_setup_ms": 0.0,
                "l2_restore_ms_per_token": 0.003,
            },
        hbm_total_tokens=1024,
        dram_total_tokens=1024,
    )
    manager.place(
        AegisBackupPlan(
            rid="latest-target-request",
            backup_key="latest-target-object",
            source_dp_rank=0,
            target_dp_rank=1,
            token_count=128,
            critical_probability=1.0,
            slack_ms=0.0,
            version=2,
            route_version=3,
            model_revision="model-rev",
            kv_abi="kv-abi-v1",
            parallel_layout="dp2-tp1-pp1",
        )
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.aegis_l1_l2_enabled = True
    scheduler.aegis_committed_copy_protocol = True
    scheduler.aegis_backup_manager = manager
    scheduler._aegis_event_queue = Queue()
    scheduler.dp_rank = 1
    scheduler.pp_rank = 0
    scheduler.tp_rank = 0
    scheduler.server_args = SimpleNamespace(pp_size=1)
    scheduler.logger = logging.getLogger("test-latest-aegis-group-commit")

    scheduler._handle_runtime_command(
        (
            "aegis_materialize_committed_prefix",
            {
                "request_backup_key": "latest-target-object",
                "stage_backup_key": "latest-target-object",
                "source_pp_rank": 0,
                "target_pp_rank": 0,
                "required_pp_size": 1,
                "version": 2,
                "route_version": 3,
                "budget_version": 4,
                "source_committed_token_end": 64,
                "execution_id": "latest-execution-target-2",
                "block_generation": 5,
            },
        )
    )

    event = scheduler._aegis_event_queue.get_nowait()
    payload = event["payload"]
    assert event["type"] == "aegis_l1_l2_group_commit_ack"
    assert payload["design_semantics"] == DESIGN_SEMANTICS
    assert payload["l1_tokens"] == 64
    assert payload["l2_tokens"] == 64
    assert payload["safe_frontier_tokens"] == 64
    assert payload["recompute_tokens"] == 0
    assert payload["residency"] == "HBM_AND_DRAM"


def test_latest_target_physical_group_commits_only_named_tier() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config={
            "design_semantics": DESIGN_SEMANTICS,
            "committed_copy_protocol": True,
            "chunk_tokens": 64,
            "initial_alpha": 1.0,
            "min_alpha": 0.1,
            "max_alpha": 1.0,
            "recompute_setup_ms": 0.0,
            "recompute_ms_per_token": 0.01,
            "l1_restore_setup_ms": 0.0,
            "l1_restore_ms_per_token": 0.001,
            "l2_restore_setup_ms": 0.0,
            "l2_restore_ms_per_token": 0.003,
        },
        hbm_total_tokens=1024,
        dram_total_tokens=1024,
    )
    manager.place(
        AegisBackupPlan(
            rid="physical-target-request",
            backup_key="physical-target-object",
            source_dp_rank=0,
            target_dp_rank=1,
            token_count=128,
            critical_probability=1.0,
            slack_ms=0.0,
            version=2,
            route_version=3,
            model_revision="model-rev",
            kv_abi="kv-abi-v1",
            parallel_layout="dp2-tp4-pp1",
        )
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.aegis_l1_l2_enabled = True
    scheduler.aegis_committed_copy_protocol = True
    scheduler.aegis_backup_manager = manager
    scheduler._aegis_event_queue = Queue()
    scheduler.dp_rank = 1
    scheduler.pp_rank = 0
    scheduler.tp_rank = 0
    scheduler.server_args = SimpleNamespace(pp_size=1)
    scheduler.logger = logging.getLogger("test-physical-tier-group-commit")

    scheduler._handle_runtime_command(
        (
            "aegis_materialize_committed_prefix",
            {
                "request_backup_key": "physical-target-object",
                "stage_backup_key": "physical-target-object",
                "source_pp_rank": 0,
                "target_pp_rank": 0,
                "required_pp_size": 1,
                "version": 2,
                "route_version": 3,
                "budget_version": 4,
                "source_committed_token_end": 64,
                "execution_id": "physical-execution-target-2",
                "block_generation": 5,
                "completed_tiers": ["hbm"],
                "physical_group_id": "physical-hbm-group-1",
                "physical_group_commit_verified": True,
            },
        )
    )

    event = scheduler._aegis_event_queue.get_nowait()
    payload = event["payload"]
    assert event["type"] == "aegis_l1_l2_group_commit_ack"
    assert payload["completed_tiers"] == ["hbm"]
    assert payload["physical_group_id"] == "physical-hbm-group-1"
    assert payload["physical_group_commit_verified"] is True
    assert payload["l1_tokens"] == 64
    assert payload["l2_tokens"] == 0
    assert payload["hbm_pending_tokens"] == 64
    assert payload["dram_pending_tokens"] == 128


def test_latest_target_rejects_unverified_physical_tier_command() -> None:
    manager = AegisIndependentTierRankManager(
        dp_rank=1,
        raw_config={
            "design_semantics": DESIGN_SEMANTICS,
            "committed_copy_protocol": True,
            "chunk_tokens": 64,
            "recompute_setup_ms": 0.0,
            "recompute_ms_per_token": 0.01,
            "l1_restore_setup_ms": 0.0,
            "l1_restore_ms_per_token": 0.001,
            "l2_restore_setup_ms": 0.0,
            "l2_restore_ms_per_token": 0.003,
        },
        hbm_total_tokens=1024,
        dram_total_tokens=1024,
    )
    manager.place(
        AegisBackupPlan(
            rid="unverified-target-request",
            backup_key="unverified-target-object",
            source_dp_rank=0,
            target_dp_rank=1,
            token_count=64,
            critical_probability=1.0,
            slack_ms=0.0,
            version=1,
        )
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.aegis_l1_l2_enabled = True
    scheduler.aegis_committed_copy_protocol = True
    scheduler.aegis_backup_manager = manager
    scheduler._aegis_event_queue = Queue()
    scheduler.dp_rank = 1
    scheduler.pp_rank = 0
    scheduler.tp_rank = 0
    scheduler.server_args = SimpleNamespace(pp_size=1)
    scheduler.logger = logging.getLogger("test-unverified-tier-group-commit")

    scheduler._handle_runtime_command(
        (
            "aegis_materialize_committed_prefix",
            {
                "request_backup_key": "unverified-target-object",
                "stage_backup_key": "unverified-target-object",
                "source_pp_rank": 0,
                "target_pp_rank": 0,
                "required_pp_size": 1,
                "version": 1,
                "source_committed_token_end": 64,
                "execution_id": "unverified-execution",
                "block_generation": 0,
                "completed_tiers": ["dram"],
            },
        )
    )

    assert scheduler._aegis_event_queue.empty()
    state = manager.snapshot()
    assert state["l1_used_tokens"] == 64
    assert state["l2_used_tokens"] == 64
    entry = manager._entries["unverified-target-object"]
    assert entry.source_committed_token_end == 0
