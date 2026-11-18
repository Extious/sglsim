from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from sglsim.srt.managers.chunked_prefill import split_prefill_length
from sglsim.srt.managers.aegis_l1_l2 import (
    AegisBackupPlan,
    AegisL1L2Config,
    AegisRankBackupManager,
    resolve_aegis_per_node_dram_quota,
)
from sglsim.srt.managers.aegis_independent_tier_manager import (
    AegisIndependentTierRankManager,
)
from sglsim.srt.managers.pipeline_parallel import (
    DPBudgetSnapshot,
    PPOutputEnvelope,
    PPProxyEnvelope,
    StageId,
    assert_batch_identity,
)
from sglsim.srt.managers.io_struct import TokenizedGenerateReqInput
from sglsim.srt.managers.prefill_adder import AddReqResult, PrefillAdder
from sglsim.srt.managers.schedule_batch import ForwardMode, Req, RequestStatus, ScheduleBatch
from sglsim.srt.managers.schedule_policy import SchedulePolicy
from sglsim.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _PendingPPBatch,
)
from sglsim.srt.mem_cache.common import extend_alloc_tokens
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.observability.scheduler_metrics_mixin import PrefillStats, SchedulerMetricsMixin
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.forward_slot import ForwardSlot
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.env import simulation_mode
from sglsim.srt.simulation.manager.state import StateManager
from sglsim.srt.simulation.pipeline_accounting import PipelineStages
from sglsim.srt.simulation.time_driver import (
    LogicalClockDriver,
    LogicalPipelinePlan,
    PipelineAdvanceResult,
    SchedulerTimeDriver,
    WallClockDriver,
)
from sglsim.srt.simulation.alpha_policy import (
    choose_alpha,
    resolve_failure_rate_per_window,
)
from sglsim.srt.simulation.recovery_model import aegis_restore_overhead_ms
from sglsim.srt.simulation.types import RequestStats, SimulationMode
from sglsim.srt.time_predictor.base import (
    InferTimePredictor,
    physical_pipeline_stage_latency,
)
from sglsim.srt.utils.network import SimIpcRegistry, SimZmqContext, SimZmqSocket, get_zmq_socket
from sglsim.srt.utils.logging import scheduler_logger

REQUEST_STATS: dict[str, RequestStats] = {}

OFFLINE_KV_RESOURCE_SNAPSHOT_SCHEMA = "sglsim.offline.rank-kv-resource.v1"


@dataclass(frozen=True)
class OfflinePreparedSchedulerStep:
    """One scheduler step split across a parent DES completion event."""

    dp_rank: int
    batch: ScheduleBatch | None
    pipeline_plan: LogicalPipelinePlan | None
    disable_overlap: bool
    pipeline_overlap: bool
    waiting_depth: int
    running_depth: int
    req_ids: tuple[str, ...]
    h2d_s: float = 0.0
    backup_pending_s: float = 0.0
    pp_rank: int = 0
    tp_rank: int = 0
    pp_sequence: int | None = None
    pp_slot: int | None = None
    pp_batch_envelope: object | None = None
    pp_received_requests: bool = False
    ingress_delay_s: float = 0.0

    @property
    def duration_s(self) -> float:
        pipeline_s = (
            0.0
            if self.pipeline_plan is None
            else self.pipeline_plan.accounting.sleep_s
        )
        return max(float(self.ingress_delay_s), 0.0) + pipeline_s

    def telemetry(self) -> dict:
        accounting = (
            self.pipeline_plan.accounting
            if self.pipeline_plan is not None
            else None
        )
        return {
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "pp_sequence": self.pp_sequence,
            "batch_id": self.batch.batch_id if self.batch is not None else -1,
            "batch_mode": (
                self.batch.mode.name.lower() if self.batch is not None else "idle"
            ),
            "request_ids": list(self.req_ids),
            "waiting_depth": self.waiting_depth,
            "running_depth": self.running_depth,
            "pipeline_overlap": self.pipeline_overlap,
            "ingress_delay_s": self.ingress_delay_s,
            "h2d_s": self.h2d_s,
            "forward_s": (
                accounting.forward_s if accounting is not None else 0.0
            ),
            "backup_pending_s": self.backup_pending_s,
            "pipeline_base_s": (
                accounting.sleep_s if accounting is not None else 0.0
            ),
            "comm_h2d_visible_s": (
                accounting.comm_h2d_visible_s
                if accounting is not None
                else 0.0
            ),
            "comm_d2h_remainder_s": (
                accounting.comm_d2h_remainder_s
                if accounting is not None
                else 0.0
            ),
            "cpu_overflow_s": 0.0,
            "started_at": (
                self.pipeline_plan.started_at
                if self.pipeline_plan is not None
                else 0.0
            ),
            "completed_at": (
                self.pipeline_plan.completed_at
                if self.pipeline_plan is not None
                else 0.0
            ),
        }


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_terminal_status(status: str) -> bool:
    return status in ("completed", "aborted")


def _normalized_fraction(value: object) -> float | None:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction < 0.0:
        return None
    return max(0.0, min(fraction, 1.0))


class Scheduler(SchedulerPPMixin, SchedulerMetricsMixin):
    LAST_CPU_TS: float = 0.0
    # Mirrors sglang envs.SGLANG_RETRACT_DECODE_STEPS (default 20): the assumed
    # number of additional decode steps per kept request when re-estimating
    # new_token_ratio after a retraction.
    RETRACT_DECODE_STEPS: int = 20
    PREFILL_BLOCKED_FLOOR_RETRY_LIMIT: int = 3

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        dp_rank: int = 0,
        *,
        pp_rank: int = 0,
        tp_rank: int = 0,
        time_driver: SchedulerTimeDriver | None = None,
    ) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.stage_id = StageId(dp_rank, pp_rank, tp_rank)
        if pp_rank >= server_args.pp_size:
            raise ValueError(
                f"pp_rank={pp_rank} must be smaller than pp_size={server_args.pp_size}"
            )
        if tp_rank >= server_args.tp_size:
            raise ValueError(
                f"tp_rank={tp_rank} must be smaller than tp_size={server_args.tp_size}"
            )
        # Pinned SGLang returns PP outputs to rank 0 before client-side result
        # processing.  Later stages update their own request/KV state but must
        # never emit duplicate HTTP/detokenizer messages.
        self._emit_client_outputs = pp_rank == 0
        self.logger = scheduler_logger(__name__, server_args, dp_rank)
        self.sim_mode = simulation_mode()
        sched_cfg = ConfigManager.get_scheduler_config()
        self.sched_cfg = sched_cfg
        self.max_running_requests = sched_cfg.effective_max_running_requests
        self.enable_overlap = not (
            server_args.disable_overlap_schedule
            or sched_cfg.disable_overlap_schedule
        )
        self.init_new_token_ratio = sched_cfg.init_new_token_ratio
        self.min_new_token_ratio = min(self.init_new_token_ratio * 0.14, 1.0)
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / 600.0
        self.new_token_ratio = self.init_new_token_ratio
        self._prioritize_decode_after_retract = False
        self._blocked_chunked_waiting_for_decode = False
        self.policy = SchedulePolicy(sched_cfg.schedule_policy)
        self.predictor: InferTimePredictor = ConfigManager.get_inference_time_predictor(
            server_args.model_path
        )
        self.cache = KVCachePool(
            ConfigManager.get_cache_config(server_args.model_path)
        )
        self.waiting_queue: deque[Req] = deque()
        self.running_batch = self._new_running_batch()
        self.last_batch: ScheduleBatch | None = None
        self.chunked_req: Req | None = None
        self._result_queue: deque[ScheduleBatch] = deque()
        self._req_registry: dict[str, Req] = {}
        self._next_batch_id = 0
        self._stalled = False
        self._last_spin_log_tic = 0.0
        self._prefill_blocked_spin_count = 0
        self._last_wall_heartbeat_tic = time.perf_counter()
        self._forward_slot = ForwardSlot()
        # Lambdas intentionally resolve this module's ``time`` at call time so
        # existing wall-clock tests and profiling hooks can patch it exactly as
        # before.  Offline injects one LogicalClockDriver per DP rank.
        self._time_driver = time_driver or WallClockDriver(
            now_fn=lambda: time.time(),
            sleep_fn=lambda duration_s: time.sleep(duration_s),
        )
        # Offline recovery admission may discover a restore/recompute delay
        # while receiving requests.  The child must not advance its private
        # clock at that point: the parent GlobalDES owns the only time cursor.
        # The delay is folded into the next prepared completion event instead.
        self._offline_pending_ingress_delay_s = 0.0
        self._disable_consecutive_prefill_overlap = _env_flag(
            "SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP",
            default=_env_flag(
                "SGLSIM_DISABLE_CONSECUTIVE_PREFILL_OVERLAP",
                default=False,
            ),
        )
        self._waiting_plan_dirty = False
        self._profile_scheduler = os.environ.get("SGLSIM_PROFILE_SCHEDULER", "") == "1"
        self._profile_totals: dict[str, float] = {}
        cache_cfg = ConfigManager.get_cache_config(server_args.model_path)
        aegis_raw = (ConfigManager._raw or {}).get("aegis_l1_l2", {})
        if not isinstance(aegis_raw, dict):
            aegis_raw = {}
        l1_pages = (
            cache_cfg.aegis_total_l1_capacity_pages
            if cache_cfg.aegis_total_l1_capacity_pages is not None
            else cache_cfg.l1_capacity_pages
        )
        l1_tokens = max(int(l1_pages) * int(cache_cfg.page_size), 1)
        legacy_l2_tokens = max(
            int(cache_cfg.l2_capacity_pages) * int(cache_cfg.page_size),
            l1_tokens * 2,
        )
        self.aegis_per_node_dram_quota = resolve_aegis_per_node_dram_quota(
            aegis_raw,
            fallback_tokens=legacy_l2_tokens,
        )
        l2_tokens = self.aegis_per_node_dram_quota.quota_tokens
        self.aegis_l1_l2_enabled = bool(aegis_raw.get("enabled", False))
        self.aegis_committed_copy_protocol = bool(
            aegis_raw.get("committed_copy_protocol", False)
        )
        self.aegis_design_semantics = str(
            aegis_raw.get("design_semantics") or "legacy_l1_l2_v1"
        )
        if self.aegis_design_semantics == DESIGN_SEMANTICS:
            self.aegis_backup_manager = AegisIndependentTierRankManager(
                dp_rank=self.dp_rank,
                raw_config=aegis_raw,
                hbm_total_tokens=l1_tokens,
                dram_total_tokens=l2_tokens,
                now_fn=self._time_driver.now,
            )
        else:
            self.aegis_backup_manager = AegisRankBackupManager(
                dp_rank=self.dp_rank,
                config=AegisL1L2Config.from_dict(
                    aegis_raw,
                    l1_total_tokens=l1_tokens,
                    l2_total_tokens=l2_tokens,
                ),
                now_fn=self._time_driver.now,
                monotonic_fn=self._time_driver.now,
            )
        self._aegis_event_queue = None
        self._aegis_last_source_commit: dict[
            tuple[str, int, str], int
        ] = {}
        # Optional calibration-only output.  Keeping this unset preserves the
        # normal online scheduler path and its pinned-SGLang control ordering.
        self._scheduler_telemetry_queue = None
        self._pending_gpu_l1_clears: deque[dict] = deque()
        self._aegis_budget_snapshot: DPBudgetSnapshot | None = None
        self._aegis_budget_version = 0
        self.init_metrics()
        ctx = SimZmqContext()
        self.recv_from_tokenizer = get_zmq_socket(
            ctx,
            SimZmqSocket.PULL,
            port_args.scheduler_input_ipc_name_for_stage(dp_rank, pp_rank),
            bind=False,
        )
        self.send_to_detokenizer = get_zmq_socket(
            ctx,
            SimZmqSocket.PUSH,
            port_args.detokenizer_ipc_name,
            bind=False,
        )
        self.recv_detokenizer_reply = get_zmq_socket(
            ctx,
            SimZmqSocket.PULL,
            port_args.detokenizer_reply_ipc_name_for_dp(dp_rank),
            bind=False,
        )
        self.init_pp_transport()

    def _new_running_batch(self) -> ScheduleBatch:
        return ScheduleBatch(
            batch_id=-1,
            rank_id=self.dp_rank,
            mode=ForwardMode.DECODE,
            reqs=[],
        )

    def recv_requests(self) -> list[TokenizedGenerateReqInput]:
        recv_reqs: list[TokenizedGenerateReqInput] = []
        try:
            while True:
                obj = self.recv_from_tokenizer.recv_pyobj(flags=1)
                recv_reqs.append(obj)
        except Exception:
            pass

        now = self._time_driver.now()
        if recv_reqs:
            self.logger.info(
                "Received %d request(s), waiting=%d running=%d",
                len(recv_reqs),
                len(self.waiting_queue),
                len(self.running_batch.reqs),
            )
        for tok_req in recv_reqs:
            self._register_incoming_request(tok_req, now=now)
        return recv_reqs

    def _register_incoming_request(
        self, tok_req: TokenizedGenerateReqInput, *, now: float
    ) -> None:
        req = tok_req.to_req()
        if req.rid not in REQUEST_STATS:
            REQUEST_STATS[req.rid] = RequestStats(rid=req.rid)
        stats = REQUEST_STATS[req.rid]
        stats.input_length = req.prefill_length
        stats.output_length = req.original_max_new_tokens or req.sampling_params.max_new_tokens
        stats.generated_tokens = req.total_generated_tokens
        sim = tok_req.sampling_params.custom_params.get("simulation", {})
        stats.created_time = float(sim.get("server_created_time", now))
        stats.last_event_time = stats.created_time
        stats.queue_start = now
        stats.status = "running"
        stats.workflow_id = str(sim.get("workflow_id", ""))
        try:
            stats.stage_id = int(sim.get("stage_id", -1))
        except (TypeError, ValueError):
            stats.stage_id = -1
        try:
            stats.stage_width = int(sim.get("stage_width", 1))
        except (TypeError, ValueError):
            stats.stage_width = 1
        try:
            stats.stage_local_agent_index = int(
                sim.get("stage_local_agent_index", 0)
            )
        except (TypeError, ValueError):
            stats.stage_local_agent_index = 0
        stats.decode_dominant = bool(sim.get("decode_dominant", False))
        stats.workflow_shape = str(sim.get("workflow_shape", ""))
        stats.agent_id = str(sim.get("agent_id", sim.get("worker_id", "")))
        stats.critical_agent = bool(sim.get("critical_agent", False))
        stats.aegis_backup_kv = bool(sim.get("backup_kv", False))
        stats.aegis_failover_replay = bool(sim.get("failover_replay", False))
        stats.aegis_soft_failure_injected = bool(
            sim.get("soft_failure_injected", False)
        )
        stats.aegis_hard_failure_injected = bool(
            sim.get("hard_failure_injected", False)
        )
        stats.aegis_host_backup_lookup_hit = bool(
            sim.get("host_backup_lookup_hit", False)
        )
        try:
            stats.aegis_host_backup_lookup_tokens = int(
                sim.get("host_backup_lookup_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_host_backup_lookup_tokens = 0
        stats.aegis_l3_backup_lookup_hit = bool(
            sim.get("l3_backup_lookup_hit", False)
        )
        try:
            stats.aegis_l3_backup_lookup_tokens = int(
                sim.get("l3_backup_lookup_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_l3_backup_lookup_tokens = 0
        stats.aegis_peer_gpu_lookup_hit = bool(
            sim.get("peer_gpu_lookup_hit", False)
        )
        try:
            stats.aegis_peer_gpu_lookup_tokens = int(
                sim.get("peer_gpu_lookup_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_peer_gpu_lookup_tokens = 0
        stats.aegis_policy = str(
            sim.get("aegis_policy")
            or ("aegiskv" if stats.aegis_backup_kv else "baseline")
        )
        stats.aegis_ablation = str(sim.get("aegis_ablation", ""))
        gpu_budget = _normalized_fraction(sim.get("aegis_gpu_budget_fraction", -1.0))
        stats.aegis_gpu_budget_fraction = -1.0 if gpu_budget is None else gpu_budget
        failure_rate_raw = sim.get(
            "aegis_failure_rate_per_window",
            sim.get("failure_rate_per_window"),
        )
        stats.aegis_failure_rate_per_window = (
            -1.0
            if failure_rate_raw is None
            else resolve_failure_rate_per_window(failure_rate_raw)
        )
        stats.aegis_alpha_policy_source = str(
            sim.get(
                "aegis_alpha_policy_source",
                "request_static" if gpu_budget is not None else "",
            )
        )
        request_pressure = _normalized_fraction(
            sim.get("aegis_critical_pressure", -1.0)
        )
        stats.aegis_critical_pressure = (
            -1.0 if request_pressure is None else request_pressure
        )
        for field_name in (
            "aegis_slack_lcb_ms",
            "aegis_slack_q50_ms",
            "aegis_slack_q90_ms",
            "aegis_slack_target_coverage",
        ):
            try:
                setattr(stats, field_name, float(sim.get(field_name, 0.0)))
            except (TypeError, ValueError):
                setattr(stats, field_name, 0.0)
        stats.aegis_slack_calibration_id = str(
            sim.get("aegis_slack_calibration_id", "")
        )
        stats.aegis_slack_calibration_group = str(
            sim.get("aegis_slack_calibration_group", "")
        )
        stats.aegis_slack_source = str(sim.get("aegis_slack_source", ""))
        stats.aegis_slack_kind = str(sim.get("aegis_slack_kind", ""))
        stats.aegis_need_state = str(sim.get("aegis_need_state", ""))
        stats.aegis_slack_role = str(sim.get("aegis_slack_role", ""))
        try:
            stats.aegis_slack_turns_completed = int(
                sim.get("aegis_slack_turns_completed", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_slack_turns_completed = 0
        try:
            stats.aegis_worker_seq = int(sim.get("worker_seq", 0))
        except (TypeError, ValueError):
            stats.aegis_worker_seq = 0
        try:
            stats.aegis_assigned_dp_rank = int(
                sim.get("assigned_dp_rank", getattr(self, "dp_rank", 0))
            )
        except (TypeError, ValueError):
            stats.aegis_assigned_dp_rank = int(getattr(self, "dp_rank", 0))
        try:
            stats.aegis_route_plan_dp_rank = int(
                sim.get("aegis_route_plan_dp_rank", -1)
            )
        except (TypeError, ValueError):
            stats.aegis_route_plan_dp_rank = -1
        route_plan_applied = sim.get("aegis_route_plan_applied")
        stats.aegis_route_plan_applied = (
            bool(route_plan_applied)
            if isinstance(route_plan_applied, bool)
            else None
        )
        stats.aegis_exact_admission_state_id = str(
            sim.get("aegis_exact_admission_state_id", "")
        )
        stats.aegis_exact_admission_information_mode = str(
            sim.get("aegis_exact_admission_information_mode", "")
        )
        raw_unavailable = sim.get(
            "aegis_exact_observed_unavailable_dp_ranks", ()
        )
        stats.aegis_exact_observed_unavailable_dp_ranks = [
            int(rank)
            for rank in raw_unavailable
        ] if isinstance(raw_unavailable, (list, tuple)) else []
        try:
            stats.aegis_gpu_replica_dp_rank = int(sim.get("gpu_replica_dp_rank", -1))
        except (TypeError, ValueError):
            stats.aegis_gpu_replica_dp_rank = -1
        try:
            stats.aegis_gpu_replica_quota_tokens = int(
                sim.get("gpu_replica_quota_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_gpu_replica_quota_tokens = 0
        try:
            stats.aegis_gpu_replica_overlap_tokens = int(
                sim.get("gpu_replica_overlap_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_gpu_replica_overlap_tokens = 0
        try:
            stats.aegis_gpu_replica_unique_tokens = int(
                sim.get("gpu_replica_unique_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_gpu_replica_unique_tokens = 0
        try:
            stats.aegis_gpu_replica_reserved_tokens = int(
                sim.get("gpu_replica_reserved_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_gpu_replica_reserved_tokens = 0
        stats.aegis_gpu_replica_evictable = bool(
            sim.get("gpu_replica_evictable", False)
        )
        try:
            stats.aegis_failed_source_dp_rank = int(
                sim.get("failed_dp_rank", stats.aegis_assigned_dp_rank)
            )
        except (TypeError, ValueError):
            stats.aegis_failed_source_dp_rank = stats.aegis_assigned_dp_rank
        try:
            stats.aegis_replay_restored_tokens = int(
                sim.get("restore_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_replay_restored_tokens = 0
        try:
            stats.aegis_replay_restored_prefill_tokens = int(
                sim.get("restored_prefill_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_replay_restored_prefill_tokens = 0
        try:
            stats.aegis_replay_recompute_tokens = int(
                sim.get("restore_recompute_tokens", 0)
            )
        except (TypeError, ValueError):
            stats.aegis_replay_recompute_tokens = 0
        stats.aegis_replay_restore_action = str(
            sim.get("restore_action", "none")
        )
        stats.aegis_replay_restore_data_plane = str(
            sim.get("restore_data_plane")
            or (
                "metadata_materialization"
                if stats.aegis_replay_restore_action
                in {
                    "host_restore",
                    "gpu_replica",
                    "stage_aligned_restore",
                    "l1_backup",
                    "l2_backup",
                    "l1_l2_backup",
                    "l3_backup",
                    "l1_l3_backup",
                    "l2_l3_backup",
                    "l1_l2_l3_backup",
                }
                else stats.aegis_replay_restore_action
            )
            or "none"
        )
        stats.aegis_replay_restore_timing_source = str(
            sim.get("restore_timing_source", "default_coefficients")
        )
        stats.aegis_replay_gpu_direct_reuse = bool(
            sim.get("aegis_retry_gpu_direct_reuse", False)
        )
        host_restore_path = sim.get("aegis_retry_host_restore_path", ())
        if isinstance(host_restore_path, (list, tuple)):
            stats.aegis_replay_host_restore_path = "+".join(
                str(component) for component in host_restore_path
            ) or "none"
        else:
            stats.aegis_replay_host_restore_path = str(
                host_restore_path or "none"
            )
        l3_restore_path = sim.get("aegis_retry_l3_restore_path", ())
        if isinstance(l3_restore_path, (list, tuple)):
            stats.aegis_replay_l3_restore_path = "+".join(
                str(component) for component in l3_restore_path
            ) or "none"
        else:
            stats.aegis_replay_l3_restore_path = str(
                l3_restore_path or "none"
            )
        try:
            stats.aegis_replay_restore_overhead_s = float(
                sim.get("restore_overhead_s", 0.0)
            )
        except (TypeError, ValueError):
            stats.aegis_replay_restore_overhead_s = 0.0
        stats.aegis_cross_rank_retry = (
            stats.aegis_failover_replay
            and stats.aegis_failed_source_dp_rank >= 0
            and stats.aegis_assigned_dp_rank >= 0
            and stats.aegis_failed_source_dp_rank != stats.aegis_assigned_dp_rank
        )
        stats.aegis_replay_restore_sleep_s = 0.0
        self._materialize_aegis_replay_restore(req, stats)
        if stats.aegis_failover_replay and stats.aegis_replay_restore_overhead_s > 0:
            restore_delay = max(
                float(stats.aegis_replay_restore_overhead_s), 0.0
            )
            if isinstance(self._time_driver, LogicalClockDriver):
                self._offline_pending_ingress_delay_s += restore_delay
                stats.aegis_replay_restore_sleep_s = restore_delay
                stats.last_event_time = (
                    self._time_driver.now()
                    + self._offline_pending_ingress_delay_s
                )
            else:
                restore_start = self._time_driver.now()
                restore_end = self._time_driver.delay(restore_delay)
                stats.aegis_replay_restore_sleep_s = max(
                    restore_end - restore_start,
                    0.0,
                )
                stats.last_event_time = restore_end
        if self.cache.config.l3_capacity_pages > 0:
            self.cache.prefetch_from_storage(req)
        self._req_registry[req.rid] = req
        self._enqueue_req(req, stats.queue_start)

    def _materialize_aegis_replay_restore(
        self, req: Req, stats: RequestStats
    ) -> None:
        if not stats.aegis_failover_replay:
            stats.aegis_replay_restored_prefill_tokens = 0
            return
        if stats.aegis_replay_restore_action not in {
            "host_restore",
            "gpu_replica",
            "stage_aligned_restore",
            "l1_backup",
            "l2_backup",
            "l1_l2_backup",
            "l3_backup",
            "l1_l3_backup",
            "l2_l3_backup",
            "l1_l2_l3_backup",
            "live_l1_reuse",
            "live_l1_l2_backup",
            "live_l1_l3_backup",
            "live_l1_l2_l3_backup",
        }:
            stats.aegis_replay_restored_prefill_tokens = 0
            return
        if stats.aegis_replay_restored_tokens <= 0:
            stats.aegis_replay_restored_prefill_tokens = 0
            return

        requested_restore = stats.aegis_replay_restored_tokens
        restored = self.cache.materialize_restored_prefix(
            req,
            requested_restore,
        )
        capacity_shortfall = max(requested_restore - restored, 0)
        stats.aegis_replay_capacity_limited_tokens = capacity_shortfall
        stats.aegis_replay_restored_tokens = restored
        stats.aegis_replay_recompute_tokens += capacity_shortfall
        stats.aegis_replay_restored_prefill_tokens = restored
        if restored > 0:
            stats.cache_hit_tokens = req.cache_hit_tokens
        else:
            stats.aegis_replay_restore_action = "recompute"
            stats.aegis_replay_restore_data_plane = "recompute"
        if capacity_shortfall > 0:
            manager = self.aegis_backup_manager
            if isinstance(manager, AegisIndependentTierRankManager):
                independent_cfg = manager.controller.config
                l1_restore_setup_ms = independent_cfg.hbm_load.setup_ms
                l1_restore_ms_per_token = (
                    independent_cfg.hbm_load.ms_per_token
                )
                l2_restore_setup_ms = independent_cfg.dram_load.setup_ms
                l2_restore_ms_per_token = (
                    independent_cfg.dram_load.ms_per_token
                )
                recompute_setup_ms = independent_cfg.recompute.setup_ms
                recompute_ms_per_token = (
                    independent_cfg.recompute.ms_per_token
                )
                l3_read_setup_ms = float(
                    manager._raw.get("l3_read_setup_ms", 8.0)
                )
                l3_read_ms_per_token = float(
                    manager._raw.get("l3_read_ms_per_token", 0.007)
                )
            else:
                cfg = manager.config
                l1_restore_setup_ms = cfg.l1_restore_setup_ms
                l1_restore_ms_per_token = cfg.l1_restore_ms_per_token
                l2_restore_setup_ms = cfg.l2_restore_setup_ms
                l2_restore_ms_per_token = cfg.l2_restore_ms_per_token
                l3_read_setup_ms = cfg.l3_read_setup_ms
                l3_read_ms_per_token = cfg.l3_read_ms_per_token
                recompute_setup_ms = cfg.recompute_setup_ms
                recompute_ms_per_token = cfg.recompute_ms_per_token
            l1_hint = min(
                max(stats.aegis_peer_gpu_lookup_tokens, 0),
                requested_restore,
            )
            actual_l1 = min(restored, l1_hint)
            l2_hint = min(
                max(stats.aegis_host_backup_lookup_tokens, 0),
                max(requested_restore - l1_hint, 0),
            )
            actual_l2 = min(max(restored - actual_l1, 0), l2_hint)
            actual_l3 = max(restored - actual_l1 - actual_l2, 0)
            recompute = max(stats.aegis_replay_recompute_tokens, 0)
            stats.aegis_replay_restore_overhead_s = (
                (l1_restore_setup_ms if actual_l1 else 0.0)
                + actual_l1 * l1_restore_ms_per_token
                + (l2_restore_setup_ms if actual_l2 else 0.0)
                + actual_l2 * l2_restore_ms_per_token
                + (l3_read_setup_ms if actual_l3 else 0.0)
                + actual_l3 * l3_read_ms_per_token
                + (recompute_setup_ms if recompute else 0.0)
                + recompute * recompute_ms_per_token
            ) / 1000.0
            stats.aegis_replay_restore_timing_source += "+capacity_cap"

    def _refresh_aegis_stats(self, req: Req, stats: RequestStats) -> None:
        total_tokens = max(stats.input_length + stats.generated_tokens, 0)
        if total_tokens <= 0:
            total_tokens = max(req.prefill_length, 0)

        # Fixed write-through HostBackup makes ACKed Host KV physically
        # available in the simulator. Policy labels only control which
        # metadata frontier is published for paper-facing AegisKV sweeps; they
        # do not change SGLang admission or batch scheduling.
        aegis_policy = stats.aegis_policy.lower().replace("-", "_").replace(" ", "_")
        aegis_ablation = (
            stats.aegis_ablation.lower().replace("-", "_").replace(" ", "_")
        )
        no_backup_policy = aegis_policy in {
            "baseline",
            "none",
            "no_backup",
            "nobackup",
        }
        host_only_policy = aegis_policy in {
            "host",
            "host_only",
            "hostbackup",
            "host_backup",
            "uniform_host",
        }
        gpu_only_policy = aegis_policy in {
            "gpu",
            "gpu_only",
            "gpu_replica",
            "gpureplica",
            "uniform_gpu",
        }
        aegis_policy_enabled = aegis_policy in {"aegis", "aegiskv", "dynamic"}
        disable_lifecycle = aegis_ablation in {
            "no_lifecycle",
            "disable_lifecycle",
        }
        disable_critical_restore = (
            aegis_ablation in {"no_critical_restore", "disable_critical_restore"}
            and stats.critical_agent
        )
        size_only_backup = aegis_ablation in {"size_only", "size_only_backup"}
        full_frontier_backup = aegis_ablation in {
            "no_marginal_accounting",
            "disable_marginal_accounting",
        }
        no_peer_locality = aegis_ablation in {
            "no_peer_locality",
            "disable_peer_locality",
        }

        host_enabled = (
            not no_backup_policy
            and not disable_lifecycle
            and not disable_critical_restore
            and not gpu_only_policy
            and (
                host_only_policy
                or aegis_policy_enabled
                or stats.aegis_backup_kv
                or stats.critical_agent
            )
        )
        gpu_enabled = (
            not no_backup_policy
            and not disable_lifecycle
            and not disable_critical_restore
            and not host_only_policy
            and (
                gpu_only_policy
                or (aegis_policy_enabled and (stats.critical_agent or size_only_backup))
                or (stats.aegis_backup_kv and (stats.critical_agent or size_only_backup))
            )
        )
        gpu_budget = _normalized_fraction(stats.aegis_gpu_budget_fraction)
        if gpu_budget is None and aegis_policy_enabled:
            failure_rate = (
                None
                if stats.aegis_failure_rate_per_window < 0
                else stats.aegis_failure_rate_per_window
            )
            decision = choose_alpha(
                critical_agent=stats.critical_agent,
                decode_dominant=stats.decode_dominant,
                stage_width=stats.stage_width,
                critical_pressure=(
                    None
                    if stats.aegis_critical_pressure < 0
                    else stats.aegis_critical_pressure
                ),
                failure_rate_per_window=failure_rate,
            )
            gpu_budget = decision.alpha
            stats.aegis_gpu_budget_fraction = decision.alpha
            stats.aegis_failure_rate_per_window = decision.failure_rate_per_window
            stats.aegis_alpha_policy_source = decision.source
            stats.aegis_critical_pressure = decision.critical_pressure
        elif gpu_budget is not None and not stats.aegis_alpha_policy_source:
            stats.aegis_alpha_policy_source = "request_static"

        if host_enabled:
            stats.aegis_host_visible_tokens = total_tokens
        else:
            stats.aegis_host_visible_tokens = 0

        if gpu_enabled:
            hot_tail = (
                total_tokens
                if full_frontier_backup
                else max(stats.generated_tokens, total_tokens // 8)
            )
            stats.aegis_gpu_replica_tokens = min(total_tokens, max(hot_tail, 1))
        else:
            stats.aegis_gpu_replica_tokens = 0
        if gpu_budget is not None:
            budget_tokens = int(total_tokens * gpu_budget)
            stats.aegis_gpu_replica_tokens = min(
                stats.aegis_gpu_replica_tokens,
                budget_tokens,
            )
        if stats.aegis_gpu_replica_reserved_tokens > 0:
            stats.aegis_gpu_replica_tokens = min(
                stats.aegis_gpu_replica_tokens,
                stats.aegis_gpu_replica_reserved_tokens,
            )
        if aegis_ablation == "no_segment_granularity":
            if stats.aegis_host_visible_tokens < total_tokens:
                stats.aegis_host_visible_tokens = 0
            if stats.aegis_gpu_replica_tokens < total_tokens:
                stats.aegis_gpu_replica_tokens = 0
        elif aegis_ablation == "no_pressure":
            stats.aegis_gpu_replica_tokens = 0
        elif aegis_ablation == "no_hysteresis" and stats.aegis_worker_seq % 2:
            stats.aegis_host_visible_tokens = 0
            stats.aegis_gpu_replica_tokens = 0

        stats.aegis_safe_frontier_tokens = max(
            stats.aegis_host_visible_tokens,
            stats.aegis_gpu_replica_tokens,
            min(total_tokens, stats.cache_hit_tokens + stats.load_back_tokens),
        )
        stats.aegis_recompute_tokens = max(
            total_tokens - stats.aegis_safe_frontier_tokens,
            0,
        )

        def best_recovery(
            token_count: int, host_frontier: int, gpu_frontier: int
        ) -> tuple[str, int, float]:
            if no_backup_policy:
                return (
                    "recompute",
                    0,
                    aegis_restore_overhead_ms(
                        action="recompute",
                        token_count=token_count,
                        restored_tokens=0,
                    ),
                )
            if host_only_policy and host_frontier > 0:
                return (
                    "host_restore",
                    host_frontier,
                    aegis_restore_overhead_ms(
                        action="host_restore",
                        token_count=token_count,
                        restored_tokens=host_frontier,
                    ),
                )
            if gpu_only_policy:
                if gpu_frontier > 0:
                    return (
                        "gpu_replica",
                        gpu_frontier,
                        aegis_restore_overhead_ms(
                            action="gpu_replica",
                            token_count=token_count,
                            restored_tokens=gpu_frontier,
                        ),
                    )
                return (
                    "recompute",
                    0,
                    aegis_restore_overhead_ms(
                        action="recompute",
                        token_count=token_count,
                        restored_tokens=0,
                    ),
                )
            candidates = [
                (
                    "recompute",
                    0,
                    aegis_restore_overhead_ms(
                        action="recompute",
                        token_count=token_count,
                        restored_tokens=0,
                    ),
                )
            ]
            if host_frontier > 0:
                candidates.append(
                    (
                        "host_restore",
                        host_frontier,
                        aegis_restore_overhead_ms(
                            action="host_restore",
                            token_count=token_count,
                            restored_tokens=host_frontier,
                        ),
                    )
                )
            if gpu_frontier > 0:
                candidates.append(
                    (
                        "gpu_replica",
                        gpu_frontier,
                        aegis_restore_overhead_ms(
                            action="gpu_replica",
                            token_count=token_count,
                            restored_tokens=gpu_frontier,
                        ),
                    )
                )
            return min(candidates, key=lambda item: item[2])

        (
            stats.aegis_recovery_action,
            _,
            stats.aegis_expected_recovery_ms,
        ) = best_recovery(
            total_tokens,
            min(stats.aegis_host_visible_tokens, total_tokens),
            min(stats.aegis_gpu_replica_tokens, total_tokens),
        )

        candidate_retry = (
            stats.critical_agent or stats.aegis_backup_kv or host_enabled or gpu_enabled
        )
        stats.aegis_candidate_retry = bool(candidate_retry)
        if not candidate_retry:
            stats.aegis_candidate_failed_dp_rank = -1
            stats.aegis_candidate_pre_failure_output_tokens = 0
            stats.aegis_candidate_pre_failure_total_tokens = 0
            stats.aegis_candidate_backed_up_tokens = 0
            stats.aegis_candidate_recompute_tokens = 0
            stats.aegis_candidate_recovery_action = "none"
            stats.aegis_candidate_failover_overhead_s = 0.0
            return

        sim = req.sampling_params.custom_params.get("simulation", {})
        try:
            failure_after_output = int(
                sim.get("failure_after_output_tokens", stats.generated_tokens // 2)
            )
        except (TypeError, ValueError):
            failure_after_output = stats.generated_tokens // 2
        if stats.generated_tokens > 0:
            failure_after_output = max(1, min(failure_after_output, stats.generated_tokens))
        else:
            failure_after_output = 0
        pre_failure_total = max(stats.input_length + failure_after_output, 0)

        try:
            failed_dp_rank = int(
                sim.get("failed_dp_rank", stats.aegis_assigned_dp_rank)
            )
        except (TypeError, ValueError):
            failed_dp_rank = stats.aegis_assigned_dp_rank

        candidate_gpu_tokens = (
            0
            if no_peer_locality
            else min(stats.aegis_gpu_replica_tokens, pre_failure_total)
        )
        action, backed_up, overhead_ms = best_recovery(
            pre_failure_total,
            min(stats.aegis_host_visible_tokens, pre_failure_total),
            candidate_gpu_tokens,
        )
        stats.aegis_candidate_failed_dp_rank = failed_dp_rank
        stats.aegis_candidate_pre_failure_output_tokens = failure_after_output
        stats.aegis_candidate_pre_failure_total_tokens = pre_failure_total
        stats.aegis_candidate_backed_up_tokens = backed_up
        stats.aegis_candidate_recompute_tokens = max(pre_failure_total - backed_up, 0)
        stats.aegis_candidate_recovery_action = action
        stats.aegis_candidate_failover_overhead_s = overhead_ms / 1000.0

    def _stats_payload(self, req: Req, *, finished: bool, last_event_time: float) -> dict:
        stats = REQUEST_STATS[req.rid]
        self._refresh_aegis_stats(req, stats)
        return {
            "dp_rank": getattr(self, "dp_rank", 0),
            "rid": req.rid,
            "finished": finished,
            "status": stats.status,
            "created_time": stats.created_time,
            "queue_start": stats.queue_start,
            "queue_end": stats.queue_end,
            "last_event_time": last_event_time,
            "service_time_s": stats.service_time_s,
            "input_length": stats.input_length,
            "output_length": stats.output_length,
            "generated_tokens": stats.generated_tokens,
            "gen_token_latencies": list(stats.gen_token_latencies),
            "cache_hit_tokens": stats.cache_hit_tokens,
            "load_back_tokens": stats.load_back_tokens,
            "prefetch_complete_tokens": stats.prefetch_complete_tokens,
            "workflow_id": stats.workflow_id,
            "stage_id": stats.stage_id,
            "stage_width": stats.stage_width,
            "stage_local_agent_index": stats.stage_local_agent_index,
            "decode_dominant": stats.decode_dominant,
            "workflow_shape": stats.workflow_shape,
            "agent_id": stats.agent_id,
            "critical_agent": stats.critical_agent,
            "aegis_policy": stats.aegis_policy,
            "aegis_ablation": stats.aegis_ablation,
            "aegis_gpu_budget_fraction": stats.aegis_gpu_budget_fraction,
            "aegis_failure_rate_per_window": stats.aegis_failure_rate_per_window,
            "aegis_alpha_policy_source": stats.aegis_alpha_policy_source,
            "aegis_critical_pressure": stats.aegis_critical_pressure,
            "aegis_slack_lcb_ms": stats.aegis_slack_lcb_ms,
            "aegis_slack_q50_ms": stats.aegis_slack_q50_ms,
            "aegis_slack_q90_ms": stats.aegis_slack_q90_ms,
            "aegis_slack_target_coverage": stats.aegis_slack_target_coverage,
            "aegis_slack_calibration_id": stats.aegis_slack_calibration_id,
            "aegis_slack_calibration_group": stats.aegis_slack_calibration_group,
            "aegis_slack_source": stats.aegis_slack_source,
            "aegis_slack_kind": stats.aegis_slack_kind,
            "aegis_need_state": stats.aegis_need_state,
            "aegis_slack_role": stats.aegis_slack_role,
            "aegis_slack_turns_completed": stats.aegis_slack_turns_completed,
            "aegis_backup_kv": stats.aegis_backup_kv,
            "aegis_worker_seq": stats.aegis_worker_seq,
            "aegis_assigned_dp_rank": stats.aegis_assigned_dp_rank,
            "aegis_route_plan_dp_rank": stats.aegis_route_plan_dp_rank,
            "aegis_route_plan_applied": stats.aegis_route_plan_applied,
            "aegis_exact_admission_state_id": (
                stats.aegis_exact_admission_state_id
            ),
            "aegis_exact_admission_information_mode": (
                stats.aegis_exact_admission_information_mode
            ),
            "aegis_exact_observed_unavailable_dp_ranks": list(
                stats.aegis_exact_observed_unavailable_dp_ranks
            ),
            "aegis_host_visible_tokens": stats.aegis_host_visible_tokens,
            "aegis_gpu_replica_tokens": stats.aegis_gpu_replica_tokens,
            "aegis_gpu_replica_dp_rank": stats.aegis_gpu_replica_dp_rank,
            "aegis_gpu_replica_quota_tokens": stats.aegis_gpu_replica_quota_tokens,
            "aegis_gpu_replica_overlap_tokens": stats.aegis_gpu_replica_overlap_tokens,
            "aegis_gpu_replica_unique_tokens": stats.aegis_gpu_replica_unique_tokens,
            "aegis_gpu_replica_reserved_tokens": stats.aegis_gpu_replica_reserved_tokens,
            "aegis_gpu_replica_evictable": stats.aegis_gpu_replica_evictable,
            "aegis_safe_frontier_tokens": stats.aegis_safe_frontier_tokens,
            "aegis_recompute_tokens": stats.aegis_recompute_tokens,
            "aegis_recovery_action": stats.aegis_recovery_action,
            "aegis_expected_recovery_ms": stats.aegis_expected_recovery_ms,
            "aegis_candidate_retry": stats.aegis_candidate_retry,
            "aegis_candidate_failed_dp_rank": stats.aegis_candidate_failed_dp_rank,
            "aegis_candidate_pre_failure_output_tokens": stats.aegis_candidate_pre_failure_output_tokens,
            "aegis_candidate_pre_failure_total_tokens": stats.aegis_candidate_pre_failure_total_tokens,
            "aegis_candidate_backed_up_tokens": stats.aegis_candidate_backed_up_tokens,
            "aegis_candidate_recompute_tokens": stats.aegis_candidate_recompute_tokens,
            "aegis_candidate_recovery_action": stats.aegis_candidate_recovery_action,
            "aegis_candidate_failover_overhead_s": stats.aegis_candidate_failover_overhead_s,
            "aegis_failover_replay": stats.aegis_failover_replay,
            "aegis_failed_source_dp_rank": stats.aegis_failed_source_dp_rank,
            "aegis_replay_restored_tokens": stats.aegis_replay_restored_tokens,
            "aegis_replay_restored_prefill_tokens": stats.aegis_replay_restored_prefill_tokens,
            "aegis_replay_recompute_tokens": stats.aegis_replay_recompute_tokens,
            "aegis_replay_capacity_limited_tokens": stats.aegis_replay_capacity_limited_tokens,
            "aegis_replay_restore_action": stats.aegis_replay_restore_action,
            "aegis_replay_restore_overhead_s": stats.aegis_replay_restore_overhead_s,
            "aegis_replay_restore_sleep_s": stats.aegis_replay_restore_sleep_s,
            "aegis_replay_restore_data_plane": stats.aegis_replay_restore_data_plane,
            "aegis_replay_restore_timing_source": stats.aegis_replay_restore_timing_source,
            "aegis_replay_gpu_direct_reuse": stats.aegis_replay_gpu_direct_reuse,
            "aegis_replay_host_restore_path": stats.aegis_replay_host_restore_path,
            "aegis_replay_l3_restore_path": stats.aegis_replay_l3_restore_path,
            "aegis_cross_rank_retry": stats.aegis_cross_rank_retry,
            "aegis_soft_failure_injected": stats.aegis_soft_failure_injected,
            "aegis_hard_failure_injected": stats.aegis_hard_failure_injected,
            "aegis_host_backup_lookup_hit": stats.aegis_host_backup_lookup_hit,
            "aegis_host_backup_lookup_tokens": stats.aegis_host_backup_lookup_tokens,
            "aegis_l3_backup_lookup_hit": stats.aegis_l3_backup_lookup_hit,
            "aegis_l3_backup_lookup_tokens": stats.aegis_l3_backup_lookup_tokens,
            "aegis_peer_gpu_lookup_hit": stats.aegis_peer_gpu_lookup_hit,
            "aegis_peer_gpu_lookup_tokens": stats.aegis_peer_gpu_lookup_tokens,
        }

    def _send_detokenizer_update(
        self, req: Req, *, finished: bool, last_event_time: float
    ) -> None:
        if not self._should_emit_client_outputs():
            return
        self.send_to_detokenizer.send_pyobj(
            self._stats_payload(req, finished=finished, last_event_time=last_event_time)
        )

    def _should_emit_client_outputs(self) -> bool:
        """Return whether this stage owns client-visible PP result emission.

        Low-level legacy tests construct Scheduler stubs through ``__new__``
        and therefore have no explicit PP identity.  Treat those as the old
        PP0 behavior; every real Scheduler initialized above has the field.
        """

        return bool(getattr(self, "_emit_client_outputs", True))

    def _recv_decode_ack(self, rid: str) -> float:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                ack = self.recv_detokenizer_reply.recv_pyobj(flags=1)
            except Exception:
                time.sleep(0.001)
                continue
            if str(ack.get("rid", "")) != rid:
                continue
            if ack.get("op") != "decode_ack":
                continue
            return float(ack.get("decode_dur", 0.0))
        raise TimeoutError(f"Timed out waiting for detokenizer ack rid={rid}")

    def _completion_decode_dur(self, req: Req) -> float:
        """Detokenization latency added to the clock at request completion.

        Default (fast) path uses the analytic estimate (0.0) instead of running
        the real tokenizer over dummy output ids.
        """
        if self.server_args.simulate_detokenizer_time:
            from sglsim.srt.utils.tokenizer_runtime import measure_decode

            _, decode_dur = measure_decode(
                self.server_args.model_path, list(req.output_ids)
            )
            return decode_dur
        from sglsim.srt.utils.tokenizer_runtime import estimate_decode_dur

        return estimate_decode_dur(req.output_ids)

    @staticmethod
    def _synthetic_output_token(req: Req) -> int:
        sim = req.sampling_params.custom_params.get("simulation", {})
        return int(sim.get("synthetic_output_token_id", 0))

    def _complete_request_inprocess(self, req: Req) -> None:
        stats = REQUEST_STATS[req.rid]
        decode_dur = self._completion_decode_dur(req)
        del decode_dur
        last_event_time = self._time_driver.now()
        stats.last_event_time = last_event_time
        stats.generated_tokens = req.total_generated_tokens
        self._send_detokenizer_update(
            req, finished=True, last_event_time=last_event_time
        )

    def _purge_req_from_scheduler_state(self, req: Req) -> None:
        """Drop completed/aborted requests from all scheduler-owned queues."""
        rid = req.rid
        target_id = id(req)

        self.waiting_queue = deque(
            queued
            for queued in self.waiting_queue
            if id(queued) != target_id and queued.rid != rid
        )

        chunked_req = getattr(self, "chunked_req", None)
        if chunked_req is not None and (
            id(chunked_req) == target_id or chunked_req.rid == rid
        ):
            self.chunked_req = None

        result_queue = getattr(self, "_result_queue", ())
        for batch in [
            getattr(self, "running_batch", None),
            getattr(self, "last_batch", None),
            *result_queue,
            *getattr(self, "_pp_running_mbs", ()),
            *getattr(self, "_pp_last_mbs", ()),
            *(
                pending.batch
                for pending in getattr(
                    self, "_pp_pending_batches", {}
                ).values()
            ),
        ]:
            if batch is None:
                continue
            before = len(batch.reqs)
            batch.reqs = [
                queued
                for queued in batch.reqs
                if id(queued) != target_id and queued.rid != rid
            ]
            if len(batch.reqs) != before:
                batch.batch_is_full = False

    def _abort_request(self, req: Req, *, reason: str = "OOM") -> None:
        self.cache.release_request(req)
        stats = REQUEST_STATS.get(req.rid)
        req.status = RequestStatus.ABORTED
        self._purge_req_from_scheduler_state(req)
        if stats is not None:
            stats.status = "aborted"
        self._req_registry.pop(req.rid, None)
        self.logger.warning(
            "retract_decode: aborted last request %s due to %s",
            req.rid,
            reason,
        )
        if stats is not None:
            last_event_time = self._time_driver.now()
            stats.last_event_time = last_event_time
            stats.generated_tokens = req.total_generated_tokens
            self._send_detokenizer_update(
                req, finished=True, last_event_time=last_event_time
            )

    def _complete_request(self, req: Req) -> None:
        stats = REQUEST_STATS[req.rid]
        req.status = RequestStatus.COMPLETED
        stats.status = "completed"
        last_event_time = self._time_driver.now()
        stats.last_event_time = last_event_time
        stats.generated_tokens = req.total_generated_tokens
        self.cache.finish_request(req)
        self._purge_req_from_scheduler_state(req)
        self._req_registry.pop(req.rid, None)
        if not self._should_emit_client_outputs():
            return
        if SimIpcRegistry.get_global() is None:
            self._complete_request_inprocess(req)
            return
        # Fast path: don't ship output ids (the detokenizer won't run the real
        # tokenizer) and don't block on a synchronous decode_ack round-trip.
        ship_output_ids = (
            list(req.output_ids) if self.server_args.simulate_detokenizer_time else []
        )
        sim = req.sampling_params.custom_params.get("simulation", {})
        self.send_to_detokenizer.send_pyobj(
            {
                "op": "finalize",
                "rid": req.rid,
                "output_ids": ship_output_ids,
                "synthetic_output_text_unit": sim.get(
                    "synthetic_output_text_unit", ""
                ),
                **self._stats_payload(
                    req,
                    finished=False,
                    last_event_time=last_event_time,
                ),
            }
        )
        if self.server_args.simulate_detokenizer_time:
            decode_dur = self._recv_decode_ack(req.rid)
        else:
            decode_dur = self._completion_decode_dur(req)
        del decode_dur

    def _enqueue_req(self, req: Req, timestamp: float) -> None:
        req.queue_start_time = timestamp
        stats = REQUEST_STATS.get(req.rid)
        if stats is not None and stats.queue_start < 0:
            stats.queue_start = timestamp
        req.status = RequestStatus.WAITING
        chunks = split_prefill_length(
            max(req.prefill_length - req.cached_length, 0),
            self.sched_cfg.chunked_prefill_size,
        )
        req.prefill_chunks_remaining = list(chunks)
        if self.cache.hierarchical_enabled:
            self._waiting_plan_dirty = True
        self.waiting_queue.append(req)

    def _refresh_waiting_prefill_plans(self) -> None:
        """Recompute each fresh waiting request's chunk plan from a read-only prefix
        match, mirroring sglang's match-at-schedule-time.

        At enqueue time no prefix match has run (``cached_length == 0``), so the
        chunk plan covers the *full* input. By schedule time a prior turn's KV may
        be cached (on device or host, the latter loaded back before extend), so the
        prefill must skip that matched prefix. Recomputing here keeps ``extend_length``
        / ``PrefillAdder`` budgeting / co-batch gating consistent with the prefix that
        will actually be skipped.

        The match is recomputed at most once per request (``prefill_plan_synced``).
        ``match_prefix`` walks the radix tree token-by-token (O(prefix); at
        ``page_size=1`` this is one Python step per cached token), so doing it for
        every waiting request on *every* scheduler step is quadratic in the hot loop:
        with multi-turn 30k+ contexts it added ~13s per decode log-interval whenever
        ``#queue-req > 0`` (measured 15.6s vs 2.5s), inflating wall-clock latency. A
        prior turn's KV is in the tree before its next turn is even enqueued (the
        closed-loop client waits for the response), so the matched prefix is stable by
        the time a fresh request first appears; the authoritative skip is recomputed
        from ``device_resident_length`` at admission anyway, so a one-shot estimate
        here only feeds admission budgeting / co-batch gating.

        Retracted requests (``kv_committed_length > 0`` after ``sync_retract_prefix``,
        or still flagged ``is_retracted``) keep their reconciled plan and are skipped.
        """
        if not self.cache.hierarchical_enabled or not self._waiting_plan_dirty:
            return
        chunk_size = self.sched_cfg.chunked_prefill_size
        for req in self.waiting_queue:
            if req.prefill_plan_synced:
                continue
            if req.is_retracted or req.kv_committed_length > 0:
                req.prefill_plan_synced = True
                continue
            req.prefill_plan_synced = True
            matched = self.cache.matched_prefix_length(req)
            # sglang always leaves >=1 token for extend (the model must compute the
            # last token's logits). Without this, a fully-cached input would have an
            # empty chunk plan + extend_length 0: the prefill loop skips it (extend<=0)
            # while a host-resident prefix is never loaded back -> the request stalls.
            if matched >= req.prefill_length:
                matched = max(req.prefill_length - 1, 0)
            uncached = max(req.prefill_length - matched, 0)
            req.prefill_chunks_remaining = list(
                split_prefill_length(uncached, chunk_size)
            )
        self._waiting_plan_dirty = False

    def _refresh_req_next_round_input(self, req: Req, *, adder: PrefillAdder) -> None:
        """Lightweight equivalent of sglang req.init_next_round_input for waiting prefill."""
        if not self.cache.hierarchical_enabled:
            return

        def refresh_matched_prefix() -> int:
            if hasattr(self.cache, "refresh_matched_prefix"):
                return self.cache.refresh_matched_prefix(req)
            req.host_hit_length = 0
            req.last_host_node_id = None
            return self.cache.matched_prefix_length(req)

        if req.prefill_chunks_remaining and req.kv_committed_length > 0 and not req.is_retracted:
            # Mirrors SGLang's req.prefix_indices for chunked prefill: once a
            # chunk has run, request-local KV progress is authoritative for the
            # next chunk. Radix matches can be smaller during HiCache load/evict
            # bookkeeping and must not roll the request back to an earlier chunk.
            matched = refresh_matched_prefix()
            if matched >= req.prefill_length:
                matched = max(req.prefill_length - 1, 0)
            committed = min(max(req.kv_committed_length, 0), req.prefill_length)
            req.cached_length = max(req.cached_length, matched, committed)
            req.kv_committed_length = committed
            remaining = max(req.prefill_length - req.kv_committed_length, 0)
            if sum(req.prefill_chunks_remaining) != remaining:
                req.prefill_chunks_remaining = list(
                    split_prefill_length(
                        remaining,
                        self.sched_cfg.chunked_prefill_size,
                    )
                )
            req.prefill_plan_synced = True
            return

        # SGLang calls Req.init_next_round_input immediately before admission.
        # Fresh waiting requests must see prefixes inserted by earlier chunked
        # prefill steps while they were still waiting.
        if req.kv_committed_length <= 0 and not req.is_retracted:
            matched = refresh_matched_prefix()
            if matched >= req.prefill_length:
                matched = max(req.prefill_length - 1, 0)
            req.cached_length = matched
            # Record cache hit discovered by the radix refresh so the
            # delta-based accounting in prepare_prefill does not zero it out.
            if not req.cache_hit_recorded and matched > 0:
                max_cached = max(req.prefill_length - 1, 0)
                req.cache_hit_tokens = min(matched, max_cached)
                req.cache_hit_log_pending = req.cache_hit_tokens
                req.cache_hit_recorded = True
                if hasattr(self.cache, "cache_prefix_tokens"):
                    self.cache.cache_prefix_tokens += req.cache_hit_tokens
                    self.cache.metrics["cache_prefix_tokens"] = self.cache.cache_prefix_tokens
            req.prefill_chunks_remaining = list(
                split_prefill_length(
                    max(req.prefill_length - matched, 0),
                    self.sched_cfg.chunked_prefill_size,
                )
            )
            req.prefill_plan_synced = True

    def get_num_allocatable_reqs(self, running_bs: int) -> int:
        cap = self.sched_cfg.pp_max_micro_batch_size or self.max_running_requests
        return max(0, cap - running_bs)

    def stash_chunked_request(self, req: Req) -> None:
        """Mirror sglang ``stash_chunked_request``: cache deferred chunk KV only."""
        if not req.prefill_alloc_page_ids:
            return
        self.cache.cache_unfinished_req(req, chunked=True)

    def _stash_chunked_req(self, req: Req) -> None:
        """Hold one in-flight chunked prefill pointer for the next scheduling round."""
        if self.chunked_req is None:
            self.chunked_req = req
            return
        if self.chunked_req is req:
            return
        self.waiting_queue.append(req)

    def is_disable_overlap_for_batch(self, batch: ScheduleBatch | None) -> bool:
        if batch is None:
            return False
        batch_is_extend = batch.is_extend()
        last_is_extend = self.last_batch is not None and self.last_batch.is_extend()
        disable_prefill_overlap = (
            self._disable_consecutive_prefill_overlap
            and batch_is_extend
            and last_is_extend
        )
        # Keep parity with SGLang's overlap guard:
        # overlap + spec-v2 + grammar decode should serialize the previous step.
        # SGLSim does not currently populate these fields, so this branch is
        # normally false but remains behaviorally aligned when set by tests/future paths.
        need_grammar_sync = (
            bool(getattr(batch, "is_spec_v2", False))
            and bool(getattr(batch, "has_grammar", False))
            and batch.mode == ForwardMode.DECODE
            and len(self._result_queue) > 0
        )
        return disable_prefill_overlap or need_grammar_sync

    def _drain_one_result(self) -> None:
        if not self._result_queue:
            return
        pending = self._result_queue.popleft()
        self.process_batch_result(pending)

    def _drain_result_queue(self) -> None:
        while self._result_queue:
            self._drain_one_result()

    @staticmethod
    def _consume_prefill_chunk(req: Req, extend_tokens: int) -> None:
        if extend_tokens <= 0 or not req.prefill_chunks_remaining:
            return
        first = req.prefill_chunks_remaining[0]
        if extend_tokens >= first:
            req.prefill_chunks_remaining.pop(0)
        else:
            req.prefill_chunks_remaining[0] = first - extend_tokens

    def _requeue_prefill_after_cache_rollback(
        self,
        req: Req,
        *,
        before_kv: int,
    ) -> bool:
        """Replan a prefill request when radix insertion exposes a shorter prefix.

        SGLang recomputes ``fill_ids`` / prefix indices before each extend round.
        If the simulator discovers during ``cache_unfinished_req`` that the device
        radix prefix is shorter than the just-planned chunk, the completed forward
        cannot be treated as committed KV. Replay from the actual device prefix
        instead of letting the request fall through into decode/completion.
        """
        committed = min(max(req.kv_committed_length, 0), req.prefill_length)
        before = min(max(before_kv, 0), req.prefill_length)
        rolled_back = committed < before
        missing_active_prefix = req.last_node_id is None and committed < req.prefill_length
        if not rolled_back and not missing_active_prefix:
            return False

        req.kv_committed_length = committed
        req.cached_length = min(max(req.cached_length, 0), committed)
        req.cache_protected_len = min(max(req.cache_protected_len, 0), committed)
        remaining = max(req.prefill_length - committed, 0)
        req.prefill_chunks_remaining = list(
            split_prefill_length(remaining, self.sched_cfg.chunked_prefill_size)
        )
        if not req.prefill_chunks_remaining:
            return False

        # This request is returning to the waiting queue. Mirror SGLang's
        # next-round admission model by dropping any active radix lock now and
        # rebuilding queue-facing prefix state from the tree on re-entry.
        self.cache.release_prefill_lock(req)
        self.cache.sync_retract_prefix(
            req,
            chunk_size=self.sched_cfg.chunked_prefill_size,
        )
        req.status = RequestStatus.WAITING
        req.prefill_plan_synced = True
        req.prefill_blocked_retries = 0
        self.logger.warning(
            "Requeue prefill after cache_unfinished rollback for %s: before_kv=%d committed=%d remaining=%d chunks=%s",
            req.rid,
            before,
            committed,
            remaining,
            req.prefill_chunks_remaining,
        )
        return True

    def _advance_time_step(
        self,
        *,
        batch: ScheduleBatch,
        sched_elapsed: float,
        overlap: bool,
    ) -> dict | None:
        forward_s = batch.inference_dur_s
        if forward_s <= 0:
            # Prefill-alloc blocked batches can yield forward_s==0; avoid a
            # tight scheduler spin that burns wall clock without sim progress.
            if sched_elapsed < 1e-3:
                self._time_driver.idle_wait(0.001)
            return None

        stages = self._pop_pipeline_stages(batch)
        advance = self._time_driver.advance_pipeline(
            stages,
            last_infer_s=self._forward_slot.last_infer_s,
            sched_elapsed_s=sched_elapsed,
            overlap=overlap,
            forward_slot=self._forward_slot,
        )
        result = advance.accounting
        self._profile_add("pipeline_sleep", advance.elapsed_s)
        self._profile_add("pipeline_sleep_base", result.sleep_s)
        self._profile_add("pipeline_cpu_overflow", advance.cpu_overflow_s)
        self._profile_add("pipeline_comm_h2d_visible", result.comm_h2d_visible_s)
        self._profile_add("pipeline_comm_d2h_remainder", result.comm_d2h_remainder_s)
        telemetry = self._wall_scheduler_telemetry(
            batch=batch,
            stages=stages,
            advance=advance,
            overlap=overlap,
        )
        if self._scheduler_telemetry_queue is not None:
            self._scheduler_telemetry_queue.put(
                {"type": "scheduler_telemetry", "row": telemetry}
            )
        return telemetry

    def _wall_scheduler_telemetry(
        self,
        *,
        batch: ScheduleBatch,
        stages: PipelineStages,
        advance: PipelineAdvanceResult,
        overlap: bool,
    ) -> dict:
        """Return the online row matching ``OfflinePreparedSchedulerStep``.

        This is deliberately observational: all pipeline accounting and time
        advancement have already happened in ``WallClockDriver``.
        """

        accounting = advance.accounting
        row = {
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "batch_id": batch.batch_id,
            "batch_mode": batch.mode.name.lower(),
            "request_ids": [req.rid for req in batch.reqs],
            "waiting_depth": len(self.waiting_queue),
            "running_depth": len(self.running_batch.reqs),
            "pipeline_overlap": bool(overlap),
            "h2d_s": stages.h2d_s,
            "forward_s": accounting.forward_s,
            "pipeline_inference_s": float(
                batch.metadata.get(
                    "predicted_pipeline_inference_dur_s",
                    accounting.forward_s,
                )
            ),
            "backup_pending_s": stages.backup_pending_s,
            "pipeline_base_s": accounting.sleep_s,
            "comm_h2d_visible_s": accounting.comm_h2d_visible_s,
            "comm_d2h_remainder_s": accounting.comm_d2h_remainder_s,
            "cpu_overflow_s": advance.cpu_overflow_s,
            "started_at": advance.completed_at - advance.elapsed_s,
            "completed_at": advance.completed_at,
            "waiting_depth_after": len(self.waiting_queue),
            "running_depth_after": len(self.running_batch.reqs),
        }
        # This is an opt-in observational snapshot.  It is deliberately built
        # after WallClockDriver has advanced, and it does not feed admission,
        # batching, cache allocation, or routing decisions.
        if self._scheduler_telemetry_queue is not None:
            row["causal_state"] = self._causal_scheduler_state()
        return row

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_or(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _causal_scheduler_state(self) -> dict:
        """Snapshot only state visible to this Scheduler at the current step.

        The snapshot intentionally excludes final output length, completion
        time, future arrivals, and any trace-oracle Slack.  Requested decode
        budget is included because it is supplied at admission; generated and
        committed token counts are the values observed at this event.
        """

        waiting_ids = {req.rid for req in self.waiting_queue}
        running_ids = {req.rid for req in self.running_batch.reqs}
        inflight_ids: set[str] = set()
        if self.last_batch is not None:
            inflight_ids.update(req.rid for req in self.last_batch.reqs)
        for pending_batch in self._result_queue:
            inflight_ids.update(req.rid for req in pending_batch.reqs)
        chunked_id = self.chunked_req.rid if self.chunked_req is not None else None

        active_requests: list[dict] = []
        for rid, req in sorted(self._req_registry.items()):
            if req.finished():
                continue
            if rid in waiting_ids:
                location = "waiting"
            elif chunked_id == rid:
                location = "chunked_prefill"
            elif rid in running_ids:
                location = "running"
            elif rid in inflight_ids:
                location = "inflight_result"
            else:
                location = "registered"
            sim = req.sampling_params.custom_params.get("simulation", {})
            if not isinstance(sim, dict):
                sim = {}
            prompt_tokens = (
                req.original_prompt_len
                if req.original_prompt_len is not None
                else req.input_token_count
                if req.input_token_count is not None
                else len(req.input_ids)
            )
            object_id = str(
                sim.get("aegis_cache_key")
                or sim.get("aegis_backup_key")
                or sim.get("cache_key")
                or ""
            )
            active_requests.append(
                {
                    "rid": rid,
                    "workflow_id": str(sim.get("workflow_id") or ""),
                    "agent_id": str(
                        sim.get("agent_id") or sim.get("worker_id") or ""
                    ),
                    "stage_id": self._int_or(sim.get("stage_id", -1), -1),
                    "stage_width": max(
                        self._int_or(sim.get("stage_width", 1), 1), 1
                    ),
                    "queue_location": location,
                    "request_status": req.status.value,
                    "created_time_s": float(req.created_time),
                    "queue_start_time_s": float(req.queue_start_time),
                    "prompt_tokens": max(int(prompt_tokens), 0),
                    "requested_max_new_tokens": max(
                        int(
                            req.original_max_new_tokens
                            if req.original_max_new_tokens is not None
                            else req.sampling_params.max_new_tokens
                        ),
                        0,
                    ),
                    "generated_tokens_observed": req.total_generated_tokens,
                    "kv_committed_tokens_observed": max(
                        int(req.kv_committed_length), 0
                    ),
                    "cache_hit_tokens_observed": max(
                        int(req.cache_hit_tokens), 0
                    ),
                    "host_hit_tokens_observed": max(int(req.host_hit_length), 0),
                    "restored_prefill_tokens_observed": max(
                        int(req.aegis_restored_prefill_tokens), 0
                    ),
                    "slack_lcb_ms": self._optional_float(
                        sim.get("slack_lcb_ms", sim.get("aegis_slack_lcb_ms"))
                    ),
                    "slack_q50_ms": self._optional_float(
                        sim.get("slack_q50_ms", sim.get("aegis_slack_q50_ms"))
                    ),
                    "slack_q90_ms": self._optional_float(
                        sim.get("slack_q90_ms", sim.get("aegis_slack_q90_ms"))
                    ),
                    "slack_target_coverage": self._optional_float(
                        sim.get(
                            "slack_target_coverage",
                            sim.get("aegis_slack_target_coverage"),
                        )
                    ),
                    "slack_calibration_id": str(
                        sim.get("slack_calibration_id")
                        or sim.get("aegis_slack_calibration_id")
                        or ""
                    ),
                    "slack_source": str(
                        sim.get("slack_source")
                        or sim.get("aegis_slack_source")
                        or ""
                    ),
                    "slack_kind": str(
                        sim.get("slack_kind")
                        or sim.get("aegis_slack_kind")
                        or ""
                    ),
                    "need_state": str(
                        sim.get("need_state")
                        or sim.get("aegis_need_state")
                        or ""
                    ),
                    "slack_semantics": str(
                        sim.get("slack_semantics")
                        or sim.get("aegis_slack_semantics")
                        or ""
                    ),
                    "slack_uses_future_information": bool(
                        sim.get(
                            "slack_uses_future_information",
                            sim.get(
                                "aegis_slack_uses_future_information", False
                            ),
                        )
                    ),
                    "slack_usable": bool(
                        sim.get(
                            "slack_usable",
                            sim.get("aegis_slack_usable", False),
                        )
                    ),
                    "slack_input_provenance_sha256": str(
                        sim.get("slack_input_provenance_sha256")
                        or sim.get("aegis_slack_input_provenance_sha256")
                        or ""
                    ),
                    "predicted_remaining_lcb_ms": self._optional_float(
                        sim.get(
                            "predicted_remaining_lcb_ms",
                            sim.get("aegis_predicted_remaining_lcb_ms"),
                        )
                    ),
                    "predicted_remaining_q50_ms": self._optional_float(
                        sim.get(
                            "predicted_remaining_q50_ms",
                            sim.get("aegis_predicted_remaining_q50_ms"),
                        )
                    ),
                    "predicted_remaining_q90_ms": self._optional_float(
                        sim.get(
                            "predicted_remaining_q90_ms",
                            sim.get("aegis_predicted_remaining_q90_ms"),
                        )
                    ),
                    "remaining_prediction_source": str(
                        sim.get("remaining_prediction_source")
                        or sim.get("aegis_remaining_prediction_source")
                        or ""
                    ),
                    "remaining_prediction_calibration_id": str(
                        sim.get("remaining_prediction_calibration_id")
                        or sim.get("aegis_remaining_prediction_calibration_id")
                        or ""
                    ),
                    "workflow_deadline_s": self._optional_float(
                        sim.get("workflow_deadline_s")
                    ),
                    "model_layout_id": str(
                        sim.get("model_layout_id")
                        or sim.get("aegis_model_layout_id")
                        or ""
                    ),
                    "recovery_target_id": str(
                        sim.get("recovery_target_id")
                        or sim.get("aegis_recovery_target_id")
                        or ""
                    ),
                    "recovery_target_dp_rank": self._int_or(
                        sim.get(
                            "recovery_target_dp_rank",
                            sim.get("aegis_recovery_target_dp_rank", -1),
                        ),
                        -1,
                    ),
                    "recovery_target_assignment_source": str(
                        sim.get("recovery_target_assignment_source")
                        or sim.get("aegis_recovery_target_assignment_source")
                        or ""
                    ),
                    "recovery_route_plan_version": str(
                        sim.get("recovery_route_plan_version")
                        or sim.get("aegis_recovery_route_plan_version")
                        or ""
                    ),
                    "kv_object_id": object_id or None,
                    "object_id_observed": bool(object_id),
                    "assigned_dp_rank": self.dp_rank,
                }
            )

        l1_used_tokens, l1_total_tokens = self.cache.kv_token_usage()
        host_pool = self.cache.hicache.mem_pool_host
        page_size = max(int(self.cache.config.page_size), 1)
        if host_pool is None:
            l2_total_tokens = 0
            l2_available_tokens = 0
        else:
            l2_total_tokens = max(int(host_pool.capacity_pages), 0) * page_size
            l2_available_tokens = max(int(host_pool.available_size), 0) * page_size
        return {
            "schema_version": "aegiskv-causal-scheduler-state-v1",
            "information_scope": "causal_scheduler_state_after_step",
            "forbidden_future_fields_present": False,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "model_path": self.server_args.model_path,
            # Router/Conductor owns recovery assignment and target readiness.
            # Do not convert queue depth into an invented service-time offset.
            "target_availability_offset_s": None,
            "target_availability_source": "unavailable",
            "target_availability_plan_version": "",
            "waiting_request_ids": sorted(waiting_ids),
            "running_request_ids": sorted(running_ids),
            "inflight_request_ids": sorted(inflight_ids),
            "chunked_request_id": chunked_id,
            "active_requests": active_requests,
            "max_running_requests": int(self.max_running_requests),
            "l1_total_tokens": max(int(l1_total_tokens), 0),
            "l1_used_tokens": max(int(l1_used_tokens), 0),
            "l1_available_tokens": max(
                int(self.cache.available_device_tokens()), 0
            ),
            "l1_available_and_evictable_tokens": max(
                int(self.cache.available_and_evictable_tokens()), 0
            ),
            "l2_total_tokens": l2_total_tokens,
            "l2_used_tokens": max(l2_total_tokens - l2_available_tokens, 0),
            "l2_available_tokens": l2_available_tokens,
        }

    def _pop_pipeline_stages(self, batch: ScheduleBatch) -> PipelineStages:
        stages = PipelineStages(
            h2d_s=StateManager.pop_hicache_l2_load_dur(),
            forward_s=batch.inference_dur_s,
            backup_pending_s=self.cache.hicache.pending_backup_dur,
        )
        self._profile_add("pipeline_forward", stages.forward_s)
        self._profile_add("pipeline_h2d", stages.h2d_s)
        self._profile_add("pipeline_backup_pending", stages.backup_pending_s)
        return stages

    def _plan_offline_pipeline(
        self,
        *,
        batch: ScheduleBatch,
        sched_elapsed: float,
        overlap: bool,
    ) -> tuple[LogicalPipelinePlan, PipelineStages]:
        driver = self._time_driver
        if not isinstance(driver, LogicalClockDriver):
            raise RuntimeError("offline scheduler step requires LogicalClockDriver")
        stages = self._pop_pipeline_stages(batch)
        plan = driver.plan_pipeline(
            stages,
            last_infer_s=self._forward_slot.last_infer_s,
            sched_elapsed_s=sched_elapsed,
            overlap=overlap,
        )
        return plan, stages

    def _advance_wall_clock_step(
        self,
        *,
        batch: ScheduleBatch,
        sched_elapsed: float,
        overlap: bool,
    ) -> None:
        """Compatibility name for the default wall-clock regression tests."""
        self._advance_time_step(
            batch=batch,
            sched_elapsed=sched_elapsed,
            overlap=overlap,
        )

    def _reset_req_for_retract(self, req: Req) -> None:
        # Mirrors SGLang Req.reset_for_retract: the same Req object can be
        # re-admitted after the current KV lifecycle was released.
        req.kv_released = False
        req.is_retracted = True
        req.prefill_plan_synced = True
        req.status = RequestStatus.WAITING
        req.host_hit_length = 0
        req.last_host_node_id = None
        req.cache_hit_log_pending = 0
        req.decode_pages = []
        self.cache.release_deferred_prefill_pages(req)
        self.cache.sync_retract_prefix(
            req, chunk_size=self.sched_cfg.chunked_prefill_size
        )

    def _record_prefill_admission_stats(self, req: Req, *, ts: float) -> None:
        stats = REQUEST_STATS[req.rid]
        first_admission = stats.queue_end < 0
        if stats.queue_end < 0:
            req.queue_end_time = ts
            stats.queue_end = ts
            stats.last_event_time = ts
        if first_admission:
            stats.cache_hit_tokens = req.cache_hit_tokens
            max_cached = max(req.prefill_length - 1, 0)
            prefix_limit = req.cache_hit_tokens if req.cache_hit_tokens > 0 else max_cached
            prefix_limit = min(prefix_limit, max_cached)
            stats.load_back_tokens = min(prefix_limit, req.admit_load_back_tokens)
            self._refresh_aegis_stats(req, stats)

    def _decay_new_token_ratio(self) -> None:
        self.new_token_ratio = max(
            self.new_token_ratio - self.new_token_ratio_decay,
            self.min_new_token_ratio,
        )

    def _retract_decode_requests(self, batch: ScheduleBatch) -> int:
        if len(batch.reqs) <= 0:
            return 0

        old_available = self.cache.available_and_evictable_tokens()
        old_ratio = self.new_token_ratio
        kept = list(batch.reqs)
        retracted: list[Req] = []
        aborted: list[Req] = []

        first_iter = True
        probe = ScheduleBatch(
            batch_id=batch.batch_id,
            rank_id=batch.rank_id,
            mode=ForwardMode.DECODE,
            reqs=kept,
        )
        while first_iter or not probe.check_decode_mem(self.cache):
            if len(kept) <= 1:
                break
            first_iter = False
            # Align with sglang ScheduleBatch.retract_decode: retract the request
            # that has made the least decode progress (cheapest to recompute),
            # breaking ties by the largest input (frees the most memory).
            victim = min(
                kept,
                key=lambda req: (len(req.output_ids), -req.prefill_length),
            )
            kept.remove(victim)
            probe.reqs = kept
            self.cache.release_request(victim)
            self.cache.evict_from_tree_cache(
                len(kept) * self.RETRACT_DECODE_STEPS,
                owner=f"retract-{victim.rid}",
            )
            self._reset_req_for_retract(victim)
            retracted.append(victim)
            self.waiting_queue.append(victim)

        if (
            not retracted
            and len(kept) == 1
            and not probe.check_decode_mem(self.cache)
        ):
            victim = kept[0]
            if self.chunked_req is not None:
                self.cache.release_request(victim)
                self.cache.evict_from_tree_cache(
                    self.RETRACT_DECODE_STEPS,
                    owner=f"retract-last-{victim.rid}",
                )
                self._reset_req_for_retract(victim)
                retracted.append(victim)
                self.waiting_queue.append(victim)
                self._prioritize_decode_after_retract = False
            else:
                self._abort_request(victim, reason="OOM")
                aborted.append(victim)
            kept = []

        if not retracted and not aborted:
            return 0

        batch.reqs = kept
        batch.batch_is_full = False
        self.running_batch.batch_is_full = False
        if retracted:
            self._prioritize_decode_after_retract = True

        if kept:
            total_decoded = sum(len(req.output_ids) for req in kept)
            total_max_new = sum(req.sampling_params.max_new_tokens for req in kept)
            self.new_token_ratio = min(
                1.0,
                (total_decoded + self.RETRACT_DECODE_STEPS * len(kept))
                / (total_max_new + 1),
            )
        else:
            self.new_token_ratio = self.init_new_token_ratio

        new_available = self.cache.available_and_evictable_tokens()
        tokens_gained = max(new_available - old_available, 0)
        self.logger.info(
            "KV cache pool is full. Retract requests. "
            "#retracted_reqs: %d, #aborted_reqs: %d, #new_tokens_gained: %d, "
            "#new_token_ratio: %.4f -> %.4f",
            len(retracted),
            len(aborted),
            tokens_gained,
            old_ratio,
            self.new_token_ratio,
        )
        return tokens_gained

    def _handle_prefill_extend_blocked(
        self,
        req: Req,
        *,
        extend_tokens: int,
        error: RuntimeError | None = None,
        available_before: int | None = None,
        evictable_before: int | None = None,
    ) -> None:
        active_chunked = bool(req.prefill_chunks_remaining)
        if self.chunked_req is req:
            self.chunked_req = None
        if req.prefill_alloc_page_ids and req.kv_committed_length > req.cached_length:
            self.cache.cache_unfinished_req(req, chunked=True)
        self.cache.release_deferred_prefill_pages(req)
        if not active_chunked:
            self.cache.release_prefill_lock(req)
            self.cache.sync_retract_prefix(
                req,
                chunk_size=self.sched_cfg.chunked_prefill_size,
            )
        req.prefill_blocked_retries += 1

        if req.prefill_chunks_remaining:
            first = req.prefill_chunks_remaining[0]
            page = max(self.cache.config.page_size, 1)
            if first > page:
                reduced = max((first // 2) // page * page, page)
                if reduced < first:
                    req.prefill_chunks_remaining[0] = reduced
                    req.prefill_chunks_remaining.insert(1, first - reduced)
            elif req.prefill_blocked_retries >= self.PREFILL_BLOCKED_FLOOR_RETRY_LIMIT:
                if self.running_batch.reqs:
                    self._prioritize_decode_after_retract = True
                # Convergence path for one-page floor retries: make at least one
                # page evictable, then retry through the normal waiting/decode path.
                self.cache.evict_from_tree_cache(
                    page,
                    owner=f"prefill-floor-{req.rid}",
                )
                req.prefill_plan_synced = False
                req.prefill_blocked_retries = 0

        page = max(self.cache.config.page_size, 1)
        decode_retry_reserve = len(self.running_batch.reqs) * page
        self.cache.evict_from_tree_cache(
            extend_alloc_tokens(extend_tokens, page) + decode_retry_reserve,
            owner=f"prefill-retry-{req.rid}",
        )
        req.status = RequestStatus.WAITING
        if self.cache.hierarchical_enabled:
            self._waiting_plan_dirty = True
        # This allocation failure is the run_batch equivalent of SGLang's
        # admission-time NO_TOKEN result.  If decodes are running, let the next
        # scheduler step run/update them first so decode retraction can free KV
        # before this prefill is retried.
        self.running_batch.batch_is_full = bool(
            self.running_batch.reqs
            or self.last_batch is not None
            or self._result_queue
        )
        if self.running_batch.reqs:
            self._prioritize_decode_after_retract = True
            if active_chunked:
                self._blocked_chunked_waiting_for_decode = True
        # Keep the partially-prefilled request on the chunked-prefill path.  A
        # blocked chunk is still the active chunked request; retrying it as an
        # ordinary waiting request lets other prefills co-batch ahead of it and
        # can reproduce the same allocation failure indefinitely.
        self.waiting_queue = deque(
            queued for queued in self.waiting_queue if queued is not req
        )
        if req.prefill_chunks_remaining:
            self.chunked_req = req
            self.waiting_queue.append(req)
        else:
            self.waiting_queue.append(req)
        self.logger.warning(
            "Prefill extend blocked for %s: extend=%d retries=%d available_tokens=%d evictable=%d before_available=%s before_evictable=%s error=%s",
            req.rid,
            extend_tokens,
            req.prefill_blocked_retries,
            self.cache._cache.mem_pool_device.available_size * self.cache.config.page_size,
            self.cache.evictable_size(),
            available_before,
            evictable_before,
            error,
        )

    def _promote_decode_ready_waiting(self) -> bool:
        if not self.waiting_queue:
            return False
        if len(self.running_batch.reqs) >= self.max_running_requests:
            return False

        promoted: list[Req] = []
        pending: deque[Req] = deque()
        capacity = self.max_running_requests - len(self.running_batch.reqs)
        for req in self.waiting_queue:
            decode_ready = (
                not req.prefill_chunks_remaining
                and req.extend_length <= 0
                and req.remaining_decode_tokens > 0
            )
            if decode_ready and len(promoted) < capacity:
                req.is_retracted = False
                self.cache.prepare_prefill(req)
                req.status = RequestStatus.DECODE
                promoted.append(req)
                continue
            pending.append(req)

        if not promoted:
            return False

        self.waiting_queue = pending
        self.running_batch.reqs.extend(promoted)
        self.running_batch.batch_is_full = False
        return True

    def _profile_add(self, key: str, elapsed: float) -> None:
        if not getattr(self, "_profile_scheduler", False):
            return
        self._profile_totals[key] = self._profile_totals.get(key, 0.0) + elapsed

    def get_new_batch_prefill(self) -> ScheduleBatch | None:
        self.waiting_queue = deque(
            req
            for req in self.waiting_queue
            if req.status not in (RequestStatus.COMPLETED, RequestStatus.ABORTED)
        )

        if self.cache.hierarchical_enabled:
            t0 = time.perf_counter()
            self.cache.check_hicache_events()
            self._profile_add("check_hicache_prefill", time.perf_counter() - t0)

        if self.running_batch.batch_is_full and not self.running_batch.reqs:
            self.running_batch.batch_is_full = False
        if (
            not self.waiting_queue
            and self.chunked_req is None
            and self.running_batch.batch_is_full
        ):
            return None
        if self.running_batch.batch_is_full and self.chunked_req is None:
            return None
        if (
            self._blocked_chunked_waiting_for_decode
            and not self.running_batch.is_empty()
            and not self.running_batch.is_prefill_only
        ):
            return None

        self.policy.calc_priority(self.waiting_queue, self.running_batch)
        t0 = time.perf_counter()
        self._refresh_waiting_prefill_plans()
        self._profile_add("refresh_waiting_plans", time.perf_counter() - t0)
        adder = PrefillAdder(
            page_size=self.cache.config.page_size,
            cache=self.cache,
            running_batch=self.running_batch,
            max_prefill_tokens=self.sched_cfg.max_prefill_tokens,
            max_running_requests=self.max_running_requests,
            rem_chunk_tokens=self.sched_cfg.chunked_prefill_size,
            new_token_ratio=self.new_token_ratio,
            prefill_max_requests=self.sched_cfg.prefill_max_requests,
            chunked_prefill_size=self.sched_cfg.chunked_prefill_size,
            enable_overlap=self.enable_overlap,
        )

        running_bs = len(self.running_batch.reqs)
        if self.chunked_req is not None:
            self.chunked_req = adder.add_chunked_req(self.chunked_req)
        active_chunked_id = id(self.chunked_req) if self.chunked_req is not None else None

        for req in self.waiting_queue:
            if active_chunked_id is not None and id(req) == active_chunked_id:
                continue
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True
                break
            if (
                self.sched_cfg.prefill_max_requests is not None
                and len(adder.can_run_list) >= self.sched_cfg.prefill_max_requests
            ):
                self.running_batch.batch_is_full = True
                break
            if req.extend_length <= 0:
                continue
            if not self.cache.check_prefetch_progress(req):
                continue
            self._refresh_req_next_round_input(req, adder=adder)
            result = adder.add_one_req(
                req, has_chunked_req=self.chunked_req is not None
            )
            if result != AddReqResult.CONTINUE:
                if result == AddReqResult.NO_TOKEN:
                    if self.cache.hierarchical_enabled:
                        self.running_batch.batch_is_full = (
                            len(adder.can_run_list) > 0
                            or not self.running_batch.is_empty()
                        )
                    else:
                        self.running_batch.batch_is_full = True
                break

        if not adder.can_run_list:
            return None

        hicache_consumer_index: int | None = None
        if self.cache.hierarchical_enabled:
            hicache_consumer_index = self.cache.ready_to_load_host_cache()

        if adder.new_chunked_req is not None:
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req
            adder.new_chunked_req = None

        can_run_ids = {id(x) for x in adder.can_run_list}
        self.waiting_queue = deque(
            x
            for x in self.waiting_queue
            if id(x) not in can_run_ids
            and (active_chunked_id is None or id(x) != active_chunked_id)
        )

        ts = self._time_driver.now()
        for req in adder.can_run_list:
            req.is_retracted = False
            req.status = RequestStatus.PREFILL
            self._record_prefill_admission_stats(req, ts=ts)

        decode_promoted: list[Req] = []
        still_prefill: list[Req] = []
        for req in adder.can_run_list:
            if req.extend_length <= 0:
                if req.remaining_decode_tokens > 0:
                    req.prefill_chunks_remaining = []
                    req.status = RequestStatus.DECODE
                    decode_promoted.append(req)
                continue
            still_prefill.append(req)
        if decode_promoted:
            self.running_batch.reqs.extend(decode_promoted)
            self.running_batch.batch_is_full = False
            if hicache_consumer_index is not None:
                self.running_batch.metadata["hicache_consumer_index"] = (
                    hicache_consumer_index
                )
        if not still_prefill:
            return None

        batch = self._make_batch(ForwardMode.PREFILL, still_prefill, ts)
        if hicache_consumer_index is not None:
            batch.metadata["hicache_consumer_index"] = hicache_consumer_index
        prefill_stats = PrefillStats.from_adder(adder, self.running_batch.reqs)
        batch.prefill_stats = prefill_stats
        batch.prepare_for_extend()
        return batch

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return None

        if self.cache.hierarchical_enabled:
            self.cache.flush_write_through_acks()

        t0 = time.perf_counter()
        if not batch.check_decode_mem(self.cache):
            self._profile_add("decode_check_mem", time.perf_counter() - t0)
            self._retract_decode_requests(batch)
            batch.filter_batch()
            if batch.is_empty():
                batch.batch_is_full = False
                return None
        else:
            self._profile_add("decode_check_mem", time.perf_counter() - t0)
            self._decay_new_token_ratio()

        t0 = time.perf_counter()
        batch.prepare_for_decode()
        self._profile_add("decode_prepare_batch", time.perf_counter() - t0)
        kept: list[Req] = []
        t0 = time.perf_counter()
        for req in batch.reqs:
            try:
                self.cache.extend_decode(req, 1)
                kept.append(req)
            except RuntimeError:
                # Keep queue progress under memory pressure: if a decode-ready request
                # cannot extend now, retract its current KV lifecycle before
                # putting it back to the waiting tail.
                self.cache.release_request(req)
                self._reset_req_for_retract(req)
                if not any(existing is req for existing in self.waiting_queue):
                    self.waiting_queue.append(req)
                self.logger.warning(
                    "Decode KV alloc blocked for %s: available_tokens=%d evictable=%d",
                    req.rid,
                    self.cache.available_device_tokens(),
                    self.cache.evictable_size(),
                )
        self._profile_add("decode_extend_kv", time.perf_counter() - t0)
        batch.reqs = kept
        if batch.is_empty():
            batch.batch_is_full = False
            return None
        return batch

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        chunked_req_to_exclude: list[Req] = []
        if self.chunked_req is not None:
            chunked_req_to_exclude.append(self.chunked_req)
            self.stash_chunked_request(self.chunked_req)

        if self.last_batch and self.last_batch.is_extend():
            last_bs = self.last_batch.batch_size
            self.last_batch.filter_batch(exclude=chunked_req_to_exclude)
            if self.last_batch.batch_size < last_bs:
                self.running_batch.batch_is_full = False
            if not self.last_batch.is_empty():
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        if self.running_batch.is_prefill_only:
            self.running_batch.filter_batch()

        if self._prioritize_decode_after_retract:
            self._prioritize_decode_after_retract = False
            self._promote_decode_ready_waiting()
            if (
                not self.running_batch.is_empty()
                and not self.running_batch.is_prefill_only
            ):
                updated = self.update_running_batch(self.running_batch)
                if updated is not None:
                    self._blocked_chunked_waiting_for_decode = False
                return updated

        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            return new_batch

        self._promote_decode_ready_waiting()

        if not self.running_batch.is_empty() and not self.running_batch.is_prefill_only:
            updated = self.update_running_batch(self.running_batch)
            if updated is not None:
                self._blocked_chunked_waiting_for_decode = False
            return updated
        if self.running_batch.is_empty() and self.chunked_req is None:
            self._self_check_during_idle()
        return None

    def run_batch(self, batch: ScheduleBatch) -> None:
        if batch.mode == ForwardMode.PREFILL:
            decode_ready = [
                req
                for req in batch.reqs
                if not req.prefill_chunks_remaining
                and req.extend_length <= 0
                and req.remaining_decode_tokens > 0
            ]
            if decode_ready:
                batch.reqs = [req for req in batch.reqs if req not in decode_ready]
                for req in decode_ready:
                    req.status = RequestStatus.DECODE
                self.running_batch.reqs.extend(decode_ready)
                self.running_batch.batch_is_full = False
            if batch.is_empty():
                return

        if batch.mode == ForwardMode.PREFILL:
            planned_extend_tokens: dict[str, int] = {}
            planned_past_kv_lengths: dict[str, int] = {}
            kept: list[Req] = []
            for req in batch.reqs:
                extend_tokens = req.extend_length
                if extend_tokens <= 0:
                    continue
                planned_past_kv_lengths[req.rid] = req.kv_committed_length
                available_before = self.cache.available_device_tokens()
                evictable_before = self.cache.evictable_size()
                try:
                    self.cache.extend_prefill(req, extend_tokens)
                except RuntimeError as exc:
                    self._handle_prefill_extend_blocked(
                        req,
                        extend_tokens=extend_tokens,
                        error=exc,
                        available_before=available_before,
                        evictable_before=evictable_before,
                    )
                    continue
                req.prefill_blocked_retries = 0
                planned_extend_tokens[req.rid] = extend_tokens
                self._consume_prefill_chunk(req, extend_tokens)
                kept.append(req)
            batch.reqs = kept
            batch.metadata["prefill_extend_tokens"] = planned_extend_tokens
            batch.metadata["prefill_past_kv_lengths"] = planned_past_kv_lengths
            batch.metadata["prefill_chunks_consumed_in_run_batch"] = True
            if batch.is_empty():
                return

        pred_batch = batch.to_predict_batch()
        batch.inference_dur_s = 0.0
        forward = 0.0
        if not pred_batch.is_empty():
            StateManager.inc_iteration()
            if batch.mode == ForwardMode.PREFILL:
                planned = batch.metadata.get("prefill_extend_tokens", {})
                self.logger.debug(
                    "Prefill forward start, #req: %d, #token: %d",
                    len(batch.reqs),
                    sum(int(planned.get(req.rid, req.extend_length)) for req in batch.reqs),
                )
            else:
                self.logger.debug(
                    "Decode forward start, #req: %d",
                    len(batch.reqs),
                )
            predicted_pipeline_s = float(
                self.predictor.predict_infer_time(pred_batch)
            )
            predicted_stage_s = physical_pipeline_stage_latency(
                predicted_pipeline_s,
                pp_size=int(self.server_args.pp_size),
            )
            batch.metadata["predicted_pipeline_inference_dur_s"] = (
                predicted_pipeline_s
            )
            batch.metadata["predicted_stage_inference_dur_s"] = predicted_stage_s
            batch.inference_dur_s = predicted_stage_s
            forward = predicted_stage_s
            StateManager.set_current_inference_dur(forward)

        if batch.mode == ForwardMode.PREFILL and batch.prefill_stats is not None:
            self.report_prefill_stats(batch, batch.prefill_stats)
        elif batch.mode == ForwardMode.DECODE:
            self.report_decode_stats(batch)

    def _response_time_after_batch(self, batch: ScheduleBatch) -> float:
        forward = batch.inference_dur_s
        hicache_result = self.cache.check_hicache_events(remain_dur=forward)
        self.cache._last_check_result = hicache_result
        prefetch_tokens = int(hicache_result.get("l3_prefetch_tokens", 0))
        if prefetch_tokens > 0:
            per_req = prefetch_tokens // max(len(batch.reqs), 1)
            for req in batch.reqs:
                stats = REQUEST_STATS.get(req.rid)
                if stats is None:
                    continue
                if req.prefetch_status == "dispatched":
                    stats.prefetch_complete_tokens += per_req
                    if stats.prefetch_complete_tokens >= req.prefill_length:
                        req.prefetch_status = "completed"
        if batch.pipeline_overlap:
            self.cache.consume_backup_overlap(forward)
        else:
            self.cache.consume_backup_serial()
        return self._time_driver.now()

    def process_batch_result(self, batch: ScheduleBatch) -> None:
        completed_reqs: list[Req] = []
        request_visible_forward_s = self._request_visible_forward_s(batch)
        planned = batch.metadata.get("prefill_extend_tokens", {})
        prefill_chunks_consumed = bool(
            batch.metadata.get("prefill_chunks_consumed_in_run_batch", False)
        )
        for req in list(batch.reqs):
            stats = REQUEST_STATS[req.rid]
            if batch.mode == ForwardMode.PREFILL:
                extend_tokens = int(planned.get(req.rid, 0))
                if extend_tokens <= 0 and req.prefill_chunks_remaining:
                    # Compatibility fallback for direct test paths that call
                    # process_batch_result without a preceding run_batch.
                    extend_tokens = req.prefill_chunks_remaining[0]
                    available_before = self.cache.available_device_tokens()
                    evictable_before = self.cache.evictable_size()
                    try:
                        self.cache.extend_prefill(req, extend_tokens)
                    except RuntimeError as exc:
                        self._handle_prefill_extend_blocked(
                            req,
                            extend_tokens=extend_tokens,
                            error=exc,
                            available_before=available_before,
                            evictable_before=evictable_before,
                        )
                        batch.reqs = [r for r in batch.reqs if r is not req]
                        continue
                    req.prefill_blocked_retries = 0
                if not prefill_chunks_consumed:
                    self._consume_prefill_chunk(req, extend_tokens)
                stats.service_time_s += request_visible_forward_s
                if req.prefill_chunks_remaining:
                    self.stash_chunked_request(req)
                    self._emit_aegis_source_chunk_committed(req)
                    self._stash_chunked_req(req)
                    batch.reqs = [r for r in batch.reqs if r is not req]
                    continue
                before_cache_kv = req.kv_committed_length
                self.cache.cache_unfinished_req(req, chunked=False)
                if self._requeue_prefill_after_cache_rollback(
                    req,
                    before_kv=before_cache_kv,
                ):
                    self._stash_chunked_req(req)
                    batch.reqs = [r for r in batch.reqs if r is not req]
                    continue
                if self.chunked_req is req:
                    self.chunked_req = None
                if self.cache.config.l3_capacity_pages > 0:
                    self.cache.write_to_l3(req)
                prefill_finished = req.mark_finished_length()
                self._emit_aegis_source_chunk_committed(
                    req,
                    request_finished=prefill_finished,
                )
                if prefill_finished:
                    completed_reqs.append(req)
            else:
                if self.enable_overlap and (
                    req.status in (RequestStatus.COMPLETED, RequestStatus.ABORTED)
                    or req.is_retracted
                ):
                    batch.reqs = [r for r in batch.reqs if r is not req]
                    continue
                if req.mark_finished_length():
                    self._emit_aegis_source_chunk_committed(
                        req,
                        request_finished=True,
                    )
                    completed_reqs.append(req)
                    continue
                stats.service_time_s += request_visible_forward_s
                req.append_output_token(self._synthetic_output_token(req))
                decode_finished = req.mark_finished_length()
                self._emit_aegis_source_chunk_committed(
                    req,
                    request_finished=decode_finished,
                )
                if decode_finished:
                    completed_reqs.append(req)

        response_time = self._response_time_after_batch(batch)
        completed_ids = {id(req) for req in completed_reqs}
        for req in batch.reqs:
            if batch.mode == ForwardMode.PREFILL and req.prefill_chunks_remaining:
                continue
            stats = REQUEST_STATS[req.rid]
            stats.generated_tokens = req.total_generated_tokens
            latency = response_time - stats.last_event_time
            if batch.mode == ForwardMode.PREFILL and not stats.gen_token_latencies:
                latency = stats.service_time_s
            elif batch.mode == ForwardMode.DECODE:
                latency = (
                    request_visible_forward_s
                    if request_visible_forward_s > 0
                    else latency
                )
            if latency > 0:
                stats.gen_token_latencies.append(latency)
            stats.last_event_time = response_time
            if id(req) in completed_ids:
                continue
            if self._should_emit_client_outputs():
                self.send_to_detokenizer.send_pyobj(
                    self._stats_payload(
                        req,
                        finished=req.status == RequestStatus.COMPLETED,
                        last_event_time=stats.last_event_time,
                    )
                )
        for req in completed_reqs:
            self._complete_request(req)
        if completed_ids:
            batch.reqs = [req for req in batch.reqs if id(req) not in completed_ids]

    def _emit_aegis_source_chunk_committed(
        self,
        req: Req,
        *,
        request_finished: bool = False,
    ) -> None:
        """Publish a causal source frontier only after a successful forward.

        The Router's admission-time plan reserves the maximum request budget.
        This event is intentionally later: ``process_batch_result`` has
        observed successful execution and the local KV allocator's committed
        length.  Full chunks are streamed while a request is live; only a
        terminal request may publish the plan's final partial chunk.
        """

        if not (
            getattr(self, "aegis_l1_l2_enabled", False)
            and getattr(self, "aegis_committed_copy_protocol", False)
            and getattr(self, "_aegis_event_queue", None) is not None
        ):
            return
        sim = req.sampling_params.custom_params.get("simulation", {})
        if not isinstance(sim, dict) or not bool(
            sim.get("aegis_backup_plan_sent", False)
        ):
            return
        backup_key = str(
            sim.get("aegis_backup_key")
            or sim.get("aegis_cache_key")
            or req.rid
        )
        target_dp_rank = self._int_or(sim.get("aegis_backup_dp_rank"), -1)
        if target_dp_rank < 0 or target_dp_rank == int(self.dp_rank):
            return
        version = max(self._int_or(sim.get("aegis_backup_version"), 1), 1)
        route_version = max(
            self._int_or(
                sim.get("aegis_route_version", sim.get("route_version", 1)),
                1,
            ),
            1,
        )
        budget_version = max(
            self._int_or(sim.get("aegis_budget_version"), 1), 1
        )
        reserved_tokens = max(
            self._int_or(
                sim.get("aegis_reserved_token_count"),
                req.prefill_length + req.remaining_decode_tokens,
            ),
            0,
        )
        source_end = min(max(int(req.kv_committed_length), 0), reserved_tokens)
        sealed_frontier = getattr(
            self.aegis_backup_manager, "sealed_source_frontier", None
        )
        if callable(sealed_frontier):
            chunk_tokens = max(
                int(self.aegis_backup_manager.decode_chunk_tokens), 1
            )
            publish_end = sealed_frontier(
                source_end,
                request_finished=bool(request_finished),
                reserved_tokens=reserved_tokens,
            )
        else:
            chunk_tokens = max(
                int(self.aegis_backup_manager.config.chunk_tokens), 1
            )
            publish_end = (
                source_end
                if request_finished
                else (source_end // chunk_tokens) * chunk_tokens
            )
        if publish_end <= 0:
            return
        execution_id = str(
            sim.get("aegis_copy_execution_id")
            or f"{req.rid}:v{version}:r{route_version}:dp{self.dp_rank}"
        )
        block_generation = max(
            self._int_or(sim.get("aegis_block_generation"), version - 1), 0
        )
        commit_key = (backup_key, version, execution_id)
        previous_end = int(
            getattr(self, "_aegis_last_source_commit", {}).get(commit_key, 0)
        )
        if publish_end <= previous_end:
            return
        self._aegis_last_source_commit[commit_key] = publish_end
        self._aegis_event_queue.put(
            {
                "type": "aegis_source_chunk_committed",
                "rid": req.rid,
                "request_backup_key": backup_key,
                "source_dp_rank": int(self.dp_rank),
                "target_dp_rank": target_dp_rank,
                "source_pp_rank": int(self.pp_rank),
                "target_pp_rank": int(self.pp_rank),
                "required_pp_size": int(self.server_args.pp_size),
                "version": version,
                "route_version": route_version,
                "budget_version": budget_version,
                "source_committed_token_end": publish_end,
                "source_observed_token_end": source_end,
                "reserved_token_count": reserved_tokens,
                "chunk_tokens": chunk_tokens,
                "prefix_chunk_tokens": int(
                    getattr(
                        self.aegis_backup_manager,
                        "prefix_chunk_tokens",
                        chunk_tokens,
                    )
                ),
                "decode_chunk_tokens": int(
                    getattr(
                        self.aegis_backup_manager,
                        "decode_chunk_tokens",
                        chunk_tokens,
                    )
                ),
                "request_finished": bool(request_finished),
                "execution_id": execution_id,
                "block_generation": block_generation,
                "observed_time": self._time_driver.now(),
            }
        )

    @staticmethod
    def _request_visible_forward_s(batch: ScheduleBatch) -> float:
        """Return full-pipeline latency for client/service-time accounting."""

        return max(
            float(
                batch.metadata.get(
                    "predicted_pipeline_inference_dur_s",
                    batch.inference_dur_s,
                )
            ),
            0.0,
        )

    def _batch_has_active_reqs(self, batch: ScheduleBatch) -> bool:
        for req in batch.reqs:
            if req.finished():
                continue
            if req.prefill_chunks_remaining:
                return True
            if req.remaining_decode_tokens > 0:
                return True
        return False

    def _finalize_last_batch(self) -> None:
        if self.last_batch is None:
            return
        self.last_batch.filter_batch()
        if self.last_batch.is_empty() or not self._batch_has_active_reqs(self.last_batch):
            self.last_batch = None

    def _log_stalled(self) -> None:
        self._stalled = True
        self.logger.warning(
            "Scheduler stalled: waiting=%d running=%d completed=%d",
            len(self.waiting_queue),
            len(self.running_batch.reqs),
            sum(1 for s in REQUEST_STATS.values() if s.status == "completed"),
        )

    def _self_check_during_idle(self) -> None:
        if self.chunked_req is not None:
            return
        if not self.running_batch.is_empty():
            return
        self.new_token_ratio = self.init_new_token_ratio

    def _maybe_log_scheduler_spin(self) -> None:
        now = time.perf_counter()
        if now - self._last_spin_log_tic < 5.0:
            return
        self._last_spin_log_tic = now
        self._self_check_during_idle()
        prefill_blocked = any(
            req.extend_length > 0 for req in self.waiting_queue
        )
        if prefill_blocked:
            self._prefill_blocked_spin_count += 1
            if self._prefill_blocked_spin_count >= 100:
                self.logger.warning(
                    "Scheduler prefill-blocked: waiting=%d running=%d evictable=%d",
                    len(self.waiting_queue),
                    len(self.running_batch.reqs),
                    self.cache.evictable_size(),
                )
                self._prefill_blocked_spin_count = 0
        else:
            self._prefill_blocked_spin_count = 0
        pending = [
            rid
            for rid, stats in REQUEST_STATS.items()
            if not _is_terminal_status(stats.status)
        ][:8]
        first_waiting = self.waiting_queue[0] if self.waiting_queue else None
        first_waiting_state = ""
        if first_waiting is not None:
            chunks = list(first_waiting.prefill_chunks_remaining[:4])
            first_waiting_state = (
                f" rid={first_waiting.rid}"
                f" extend={first_waiting.extend_length}"
                f" prefill={first_waiting.prefill_length}"
                f" committed={first_waiting.kv_committed_length}"
                f" cached={first_waiting.cached_length}"
                f" chunks_head={chunks}"
                f" chunks_total={sum(first_waiting.prefill_chunks_remaining)}"
                f" prefetch={first_waiting.prefetch_status}"
                f" blocked_retries={first_waiting.prefill_blocked_retries}"
            )
        self.logger.warning(
            "Scheduler busy-wait: waiting=%d running=%d chunked=%d last_batch=%s "
            "available=%d evictable=%d pending=%s first_waiting=%s",
            len(self.waiting_queue),
            len(self.running_batch.reqs),
            1 if self.chunked_req is not None else 0,
            "set" if self.last_batch is not None else "none",
            self.cache.available_device_tokens(),
            self.cache.evictable_size(),
            pending,
            first_waiting_state,
        )

    def _maybe_log_wall_heartbeat(self) -> None:
        now = time.perf_counter()
        if now - self._last_wall_heartbeat_tic < 30.0:
            return
        self._last_wall_heartbeat_tic = now
        self.logger.info(
            "Simulation progress: waiting=%d running=%d decode_ct=%d",
            len(self.waiting_queue),
            len(self.running_batch.reqs),
            self.forward_ct_decode,
        )

    def _scheduler_owned_req_ids(self) -> set[str]:
        owned = {req.rid for req in self.waiting_queue}
        if self.chunked_req is not None:
            owned.add(self.chunked_req.rid)
        for batch in (self.running_batch, self.last_batch):
            if batch is None:
                continue
            owned.update(req.rid for req in batch.reqs)
        for batch in self._result_queue:
            owned.update(req.rid for req in batch.reqs)
        return owned

    def _recover_idle_pending_requests(self) -> bool:
        if not self._rank_idle():
            return False
        pending = [
            rid
            for rid, stats in REQUEST_STATS.items()
            if not _is_terminal_status(stats.status)
        ]
        if not pending:
            return False

        owned = self._scheduler_owned_req_ids()
        recovered = 0
        missing: list[str] = []
        for rid in pending:
            if rid in owned:
                continue
            req = self._req_registry.get(rid)
            if req is None:
                missing.append(rid)
                continue
            if req.reached_decode_limit():
                self._complete_request(req)
                recovered += 1
                continue
            try:
                self.cache.release_request(req)
            except Exception:
                self.logger.exception("Failed to release lost request %s", rid)
            self._reset_req_for_retract(req)
            if not any(existing.rid == rid for existing in self.waiting_queue):
                self.waiting_queue.append(req)
                recovered += 1

        if recovered or missing:
            self.logger.warning(
                "Recovered idle pending scheduler state: recovered=%d missing=%s pending=%s",
                recovered,
                missing[:8],
                pending[:8],
            )
        return recovered > 0

    def offline_kv_resource_snapshot(
        self,
        *,
        timestamp_s: float | None = None,
    ) -> dict:
        """Return a read-only, rank-local view of the scheduler KV pools.

        One SGLSim ``Scheduler`` owns one metadata pool that represents the
        symmetric view of *one* TP rank.  ``kv_bytes_per_token_per_tp_rank``
        is therefore applied exactly once below; this snapshot deliberately
        does not multiply byte counts by ``logical_tp_group_size``.  A parent
        physical-topology DES can later expand this mirror onto the individual
        GPUs in the logical TP group without silently double-counting bytes.

        This helper only reads allocator/request state.  It does not allocate,
        evict, advance a clock, sleep, or otherwise alter scheduling semantics.
        """

        page_size = int(self.cache._cache.page_size)
        kv_bytes_per_token = int(
            ConfigManager.get_kv_cache_bytes_per_token(
                self.server_args.model_path
            )
        )

        def pool_snapshot(pool) -> dict:
            if pool is None:
                capacity_pages = 0
                free_pages = 0
                used_pages = 0
                owner_pages: dict[str, int] = {}
            else:
                allocator = pool._allocator
                capacity_pages = int(pool.capacity_pages)
                free_pages = int(pool.available_size)
                used_pages = int(pool.used_pages)
                owner_counts: dict[str, int] = {}
                for page_id, page in sorted(allocator._pages_by_id.items()):
                    if page_id in allocator._free_page_ids:
                        continue
                    owner = (
                        str(page.owner)
                        if page.owner is not None
                        else "<allocated-unowned>"
                    )
                    owner_counts[owner] = owner_counts.get(owner, 0) + 1
                owner_pages = {
                    owner: int(owner_counts[owner])
                    for owner in sorted(owner_counts)
                }

            capacity_tokens = capacity_pages * page_size
            used_tokens = used_pages * page_size
            free_tokens = free_pages * page_size
            return {
                "capacity_pages": capacity_pages,
                "used_pages": used_pages,
                "free_pages": free_pages,
                "capacity_tokens": capacity_tokens,
                "used_tokens": used_tokens,
                "free_tokens": free_tokens,
                "capacity_bytes": capacity_tokens * kv_bytes_per_token,
                "used_bytes": used_tokens * kv_bytes_per_token,
                "free_bytes": free_tokens * kv_bytes_per_token,
                "owner_pages": owner_pages,
            }

        running_ids = {req.rid for req in self.running_batch.reqs}
        if int(self.server_args.pp_size) > 1 and hasattr(
            self, "_pp_running_mbs"
        ):
            running_ids.update(
                req.rid
                for batch in self._pp_running_mbs
                for req in batch.reqs
            )
        waiting_ids = {req.rid for req in self.waiting_queue}
        if self.chunked_req is not None:
            waiting_ids.add(self.chunked_req.rid)

        resolved_timestamp = (
            self._time_driver.now()
            if timestamp_s is None
            else float(timestamp_s)
        )
        return {
            "schema": OFFLINE_KV_RESOURCE_SNAPSHOT_SCHEMA,
            "timestamp_s": float(resolved_timestamp),
            "dp_rank": int(self.dp_rank),
            "pp_rank": int(self.pp_rank),
            "scheduler_tp_rank": int(self.tp_rank),
            "logical_tp_group_size": int(self.server_args.tp_size),
            "scope": "one-symmetric-tp-rank-view",
            "page_size_tokens": page_size,
            "kv_bytes_per_token_per_tp_rank": kv_bytes_per_token,
            "l1_device": pool_snapshot(self.cache._cache.mem_pool_device),
            "l2_host": pool_snapshot(self.cache._cache.mem_pool_host),
            "running_request_ids": sorted(running_ids),
            "waiting_request_ids": sorted(waiting_ids),
        }

    def _take_offline_ingress_delay(self) -> float:
        """Consume recovery admission time for exactly one parent DES step."""

        delay_s = max(
            float(getattr(self, "_offline_pending_ingress_delay_s", 0.0)),
            0.0,
        )
        self._offline_pending_ingress_delay_s = 0.0
        return delay_s

    @staticmethod
    def _shift_offline_plan(
        plan: LogicalPipelinePlan,
        ingress_delay_s: float,
    ) -> LogicalPipelinePlan:
        """Serialize ingress recovery before compute without moving a clock."""

        delay_s = max(float(ingress_delay_s), 0.0)
        if delay_s == 0.0:
            return plan
        return LogicalPipelinePlan(
            accounting=plan.accounting,
            started_at=plan.started_at + delay_s,
            completed_at=plan.completed_at + delay_s,
        )

    def prepare_offline_step(self) -> OfflinePreparedSchedulerStep | None:
        """Prepare one rank-local logical step without advancing time.

        This preserves the pinned SGLang normal/overlap ordering.  The parent
        coordinator schedules ``step.duration_s`` and later calls
        :meth:`complete_offline_step`; no other step may be prepared for this
        rank in between.
        """

        if not isinstance(self._time_driver, LogicalClockDriver):
            raise RuntimeError("prepare_offline_step requires LogicalClockDriver")
        sched_start = time.perf_counter()
        self.recv_requests()
        ingress_delay_s = self._take_offline_ingress_delay()
        if self._all_done():
            if self.enable_overlap:
                self.last_batch = None
            return None

        if not self.enable_overlap:
            progressed = False
            if self.last_batch is not None:
                self.process_batch_result(self.last_batch)
                self._finalize_last_batch()
                progressed = True
            batch = self.get_next_batch_to_run()
            if batch is None:
                if self._is_stalled():
                    self._log_stalled()
                if not progressed:
                    return None
                return OfflinePreparedSchedulerStep(
                    dp_rank=self.dp_rank,
                    batch=None,
                    pipeline_plan=None,
                    disable_overlap=True,
                    pipeline_overlap=False,
                    waiting_depth=len(self.waiting_queue),
                    running_depth=len(self.running_batch.reqs),
                    req_ids=(),
                    ingress_delay_s=ingress_delay_s,
                )
            self.run_batch(batch)
            batch.pipeline_overlap = False
            plan, stages = self._plan_offline_pipeline(
                batch=batch,
                sched_elapsed=time.perf_counter() - sched_start,
                overlap=False,
            )
            plan = self._shift_offline_plan(plan, ingress_delay_s)
            return OfflinePreparedSchedulerStep(
                dp_rank=self.dp_rank,
                batch=batch,
                pipeline_plan=plan,
                disable_overlap=True,
                pipeline_overlap=False,
                waiting_depth=len(self.waiting_queue),
                running_depth=len(self.running_batch.reqs),
                req_ids=tuple(req.rid for req in batch.reqs),
                h2d_s=stages.h2d_s,
                backup_pending_s=stages.backup_pending_s,
                ingress_delay_s=ingress_delay_s,
            )

        batch = self.get_next_batch_to_run()
        disable_overlap = self.is_disable_overlap_for_batch(batch)
        progressed = False
        if disable_overlap and self.last_batch is not None and self._result_queue:
            self._drain_one_result()
            progressed = True

        if batch is None:
            if self.last_batch is not None:
                if not disable_overlap and self._result_queue:
                    self._drain_one_result()
                    progressed = True
            else:
                self._self_check_during_idle()
            self.last_batch = None
            if self._is_stalled():
                self._log_stalled()
            if not progressed:
                return None
            return OfflinePreparedSchedulerStep(
                dp_rank=self.dp_rank,
                batch=None,
                pipeline_plan=None,
                disable_overlap=disable_overlap,
                pipeline_overlap=False,
                waiting_depth=len(self.waiting_queue),
                running_depth=len(self.running_batch.reqs),
                req_ids=(),
                ingress_delay_s=ingress_delay_s,
            )

        self.run_batch(batch)
        batch.pipeline_overlap = self.enable_overlap and not disable_overlap
        plan, stages = self._plan_offline_pipeline(
            batch=batch,
            sched_elapsed=time.perf_counter() - sched_start,
            overlap=batch.pipeline_overlap,
        )
        plan = self._shift_offline_plan(plan, ingress_delay_s)
        return OfflinePreparedSchedulerStep(
            dp_rank=self.dp_rank,
            batch=batch,
            pipeline_plan=plan,
            disable_overlap=disable_overlap,
            pipeline_overlap=batch.pipeline_overlap,
            waiting_depth=len(self.waiting_queue),
            running_depth=len(self.running_batch.reqs),
            req_ids=tuple(req.rid for req in batch.reqs),
            h2d_s=stages.h2d_s,
            backup_pending_s=stages.backup_pending_s,
            ingress_delay_s=ingress_delay_s,
        )

    def complete_offline_step(
        self,
        step: OfflinePreparedSchedulerStep,
    ) -> dict:
        """Commit a prepared step at its parent-ordered completion event."""

        if step.dp_rank != self.dp_rank:
            raise ValueError(
                f"offline step rank {step.dp_rank} != scheduler rank {self.dp_rank}"
            )
        driver = self._time_driver
        if not isinstance(driver, LogicalClockDriver):
            raise RuntimeError("complete_offline_step requires LogicalClockDriver")
        if step.pipeline_plan is not None:
            driver.complete_pipeline(
                step.pipeline_plan,
                forward_slot=self._forward_slot,
            )
            accounting = step.pipeline_plan.accounting
            self._profile_add("pipeline_sleep", accounting.sleep_s)
            self._profile_add("pipeline_sleep_base", accounting.sleep_s)
            self._profile_add("pipeline_cpu_overflow", 0.0)
            self._profile_add(
                "pipeline_comm_h2d_visible", accounting.comm_h2d_visible_s
            )
            self._profile_add(
                "pipeline_comm_d2h_remainder", accounting.comm_d2h_remainder_s
            )

        batch = step.batch
        if not self.enable_overlap:
            if batch is not None:
                self.last_batch = batch
        elif batch is not None:
            self._result_queue.append(batch.copy())
            if self.last_batch is not None:
                if not step.disable_overlap and self._result_queue:
                    self._drain_one_result()
            self.last_batch = batch

        telemetry = step.telemetry()
        telemetry["completed_at"] = driver.now()
        telemetry["waiting_depth_after"] = len(self.waiting_queue)
        telemetry["running_depth_after"] = len(self.running_batch.reqs)
        telemetry["causal_state"] = self._causal_scheduler_state()
        return telemetry

    def prepare_offline_pp_step(self) -> OfflinePreparedSchedulerStep | None:
        """Prepare one physical PP-stage step for the parent logical DES.

        This is the clock-driver split of :meth:`run_event_loop_pp_until_stop`.
        It keeps the same request-forward, scheduling, proxy-validation, and
        local-forward order, but returns before advancing the stage clock.  The
        parent may therefore merge independent stage completions without a
        global max-sync barrier.

        ``None`` means only that this stage's next microbatch slot still awaits
        its output-ring result.  An idle pipeline bubble is represented by a
        real zero-duration step so its request/proxy/output envelopes retain the
        same causal sequence as the online PP loop.
        """

        if int(self.server_args.pp_size) <= 1:
            raise RuntimeError("prepare_offline_pp_step requires pp_size > 1")
        if not isinstance(self._time_driver, LogicalClockDriver):
            raise RuntimeError(
                "prepare_offline_pp_step requires LogicalClockDriver"
            )
        if not hasattr(self, "_pp_running_mbs"):
            self.init_pp_loop_state()

        self._pp_pump_outputs(cmd_queue=None, stop_event=None)
        self._maybe_clear_pending_gpu_l1()
        sequence = self._pp_step_sequence
        slot = sequence % self._pp_loop_size
        if any(
            pending.slot == slot
            for pending in self._pp_pending_batches.values()
        ):
            return None

        sched_start = time.perf_counter()
        received = self._pp_receive_and_forward_requests(
            sequence=sequence,
            cmd_queue=None,
            stop_event=None,
        )
        ingress_delay_s = self._take_offline_ingress_delay()
        self.running_batch = self._pp_running_mbs[slot]
        self.last_batch = self._pp_last_mbs[slot]
        batch = self.get_next_batch_to_run()
        self._pp_running_mbs[slot] = self.running_batch
        local_envelope = self._pp_batch_envelope(
            batch=batch,
            slot=slot,
            sequence=sequence,
        )

        if self.pp_rank > 0:
            upstream = self._pp_wait_for_proxy(
                sequence=sequence,
                cmd_queue=None,
                stop_event=None,
            )
            assert_batch_identity(upstream.batch, local_envelope)

        plan = None
        stages = PipelineStages(
            h2d_s=0.0,
            forward_s=0.0,
            backup_pending_s=0.0,
        )
        if batch is not None:
            self.run_batch(batch)
            batch.pipeline_overlap = False
            post_run_ids = tuple(req.rid for req in batch.reqs)
            if post_run_ids != local_envelope.request_ids:
                raise RuntimeError(
                    "run_batch changed offline PP request membership: "
                    f"planned={local_envelope.request_ids}, "
                    f"post_run={post_run_ids}"
                )
            plan, stages = self._plan_offline_pipeline(
                batch=batch,
                sched_elapsed=time.perf_counter() - sched_start,
                overlap=False,
            )
            plan = self._shift_offline_plan(plan, ingress_delay_s)

        return OfflinePreparedSchedulerStep(
            dp_rank=self.dp_rank,
            batch=batch,
            pipeline_plan=plan,
            disable_overlap=True,
            pipeline_overlap=False,
            waiting_depth=len(self.waiting_queue),
            running_depth=len(self.running_batch.reqs),
            req_ids=(
                tuple(req.rid for req in batch.reqs)
                if batch is not None
                else ()
            ),
            h2d_s=stages.h2d_s,
            backup_pending_s=stages.backup_pending_s,
            pp_rank=self.pp_rank,
            tp_rank=self.tp_rank,
            pp_sequence=sequence,
            pp_slot=slot,
            pp_batch_envelope=local_envelope,
            pp_received_requests=received,
            ingress_delay_s=ingress_delay_s,
        )

    def complete_offline_pp_step(
        self,
        step: OfflinePreparedSchedulerStep,
    ) -> dict:
        """Commit one prepared physical PP-stage step at its DES event."""

        if int(self.server_args.pp_size) <= 1:
            raise RuntimeError("complete_offline_pp_step requires pp_size > 1")
        if step.dp_rank != self.dp_rank or step.pp_rank != self.pp_rank:
            raise ValueError(
                "offline PP step identity does not match Scheduler stage: "
                f"step=({step.dp_rank}, {step.pp_rank}) "
                f"scheduler=({self.dp_rank}, {self.pp_rank})"
            )
        if step.pp_sequence is None or step.pp_slot is None:
            raise ValueError("offline PP step has no sequence/slot identity")
        if step.pp_sequence != self._pp_step_sequence:
            raise ValueError(
                "offline PP completion sequence mismatch: "
                f"prepared={step.pp_sequence}, current={self._pp_step_sequence}"
            )
        driver = self._time_driver
        if not isinstance(driver, LogicalClockDriver):
            raise RuntimeError(
                "complete_offline_pp_step requires LogicalClockDriver"
            )

        if step.pipeline_plan is not None:
            driver.complete_pipeline(
                step.pipeline_plan,
                forward_slot=self._forward_slot,
            )
            accounting = step.pipeline_plan.accounting
            self._profile_add("pipeline_sleep", accounting.sleep_s)
            self._profile_add("pipeline_sleep_base", accounting.sleep_s)
            self._profile_add("pipeline_cpu_overflow", 0.0)
            self._profile_add(
                "pipeline_comm_h2d_visible", accounting.comm_h2d_visible_s
            )
            self._profile_add(
                "pipeline_comm_d2h_remainder", accounting.comm_d2h_remainder_s
            )

        if step.batch is not None:
            self._pp_pending_batches[step.pp_sequence] = _PendingPPBatch(
                slot=step.pp_slot,
                batch=step.batch,
                envelope=step.pp_batch_envelope,
            )

        if self.pp_rank < int(self.server_args.pp_size) - 1:
            self._pp_send_proxy.send_pyobj(
                PPProxyEnvelope(
                    source_stage=self.stage_id,
                    target_stage=StageId(
                        self.dp_rank,
                        self.pp_rank + 1,
                        self.tp_rank,
                    ),
                    step_sequence=step.pp_sequence,
                    batch=step.pp_batch_envelope,
                )
            )
        else:
            self._pp_send_output.send_pyobj(
                PPOutputEnvelope.from_last_stage(
                    last_stage=self.stage_id,
                    pp_size=int(self.server_args.pp_size),
                    step_sequence=step.pp_sequence,
                    batch=step.pp_batch_envelope,
                    result_logical_time=driver.now(),
                )
            )

        self._pp_step_sequence += 1
        self._pp_pump_outputs(cmd_queue=None, stop_event=None)
        telemetry = step.telemetry()
        telemetry["completed_at"] = driver.now()
        telemetry["waiting_depth_after"] = len(self.waiting_queue)
        telemetry["running_depth_after"] = len(self.running_batch.reqs)
        telemetry["causal_state"] = self._causal_scheduler_state()
        return telemetry

    def _event_loop_step_normal(self) -> bool:
        sched_start = time.perf_counter()
        self.recv_requests()
        if self._all_done():
            return False

        if self.last_batch is not None:
            t0 = time.perf_counter()
            self.process_batch_result(self.last_batch)
            self._finalize_last_batch()
            self._profile_add("normal_process_last", time.perf_counter() - t0)

        t0 = time.perf_counter()
        batch = self.get_next_batch_to_run()
        self._profile_add("normal_get_next", time.perf_counter() - t0)
        if batch is None:
            if self._is_stalled():
                self._log_stalled()
            return False

        t0 = time.perf_counter()
        self.run_batch(batch)
        self._profile_add("normal_run_batch", time.perf_counter() - t0)
        sched_elapsed = time.perf_counter() - sched_start
        batch.pipeline_overlap = False
        t0 = time.perf_counter()
        self._advance_time_step(
            batch=batch,
            sched_elapsed=sched_elapsed,
            overlap=False,
        )
        self._profile_add("normal_advance_wall", time.perf_counter() - t0)
        self.last_batch = batch
        return True

    def _event_loop_step_overlap(self) -> bool:
        """Mirror sglang ``event_loop_overlap`` step order and ``last_batch`` semantics."""
        sched_start = time.perf_counter()
        self.recv_requests()
        if self._all_done():
            self.last_batch = None
            return False

        batch = self.get_next_batch_to_run()
        disable_overlap = self.is_disable_overlap_for_batch(batch)

        progressed = False
        # When this step must be serialized, process the previous result first.
        if disable_overlap and self.last_batch is not None and self._result_queue:
            self._drain_one_result()
            progressed = True

        if batch is not None:
            self.run_batch(batch)
            sched_elapsed = time.perf_counter() - sched_start
            batch.pipeline_overlap = self.enable_overlap and not disable_overlap
            self._advance_time_step(
                batch=batch,
                sched_elapsed=sched_elapsed,
                overlap=batch.pipeline_overlap,
            )
            self._result_queue.append(batch.copy())
            progressed = True

        if self.last_batch is not None:
            if not disable_overlap and self._result_queue:
                self._drain_one_result()
                progressed = True
        elif batch is None:
            # Match SGLang behavior: only run idle self-check when both current
            # and previous scheduled batches are absent.
            self._self_check_during_idle()

        self.last_batch = batch

        if batch is None and self._is_stalled():
            self._log_stalled()
        return progressed

    def event_loop_step(self) -> bool:
        t0 = time.perf_counter()
        if self.enable_overlap:
            progressed = self._event_loop_step_overlap()
        else:
            progressed = self._event_loop_step_normal()
        self._profile_add("event_loop_step", time.perf_counter() - t0)
        if self._profile_scheduler and self.forward_ct_decode % 40 == 0 and self.forward_ct_decode > 0:
            top = sorted(self._profile_totals.items(), key=lambda x: -x[1])[:6]
            self.logger.info("SGLSIM_PROFILE_SCHEDULER %s", top)
        return progressed

    def event_loop_normal(self, max_steps: int = 100_000) -> None:
        steps = 0
        while max_steps <= 0 or steps < max_steps:
            steps += 1
            if not self._event_loop_step_normal() and self._all_done():
                break
        if not self._all_done():
            self._stalled = True
            self.logger.warning(
                "Scheduler exited before idle: waiting=%d running=%d completed=%d steps=%d",
                len(self.waiting_queue),
                len(self.running_batch.reqs),
                sum(1 for s in REQUEST_STATS.values() if s.status == "completed"),
                steps,
            )

    def event_loop_overlap(self, max_steps: int = 100_000) -> None:
        steps = 0
        self.last_batch = None
        while max_steps <= 0 or steps < max_steps:
            steps += 1
            if not self._event_loop_step_overlap() and self._all_done():
                break
        self._drain_result_queue()
        if self.last_batch is not None:
            self.process_batch_result(self.last_batch)
            self._finalize_last_batch()
            self.last_batch = None
        if not self._all_done():
            self._stalled = True
            self.logger.warning(
                "Scheduler exited before idle: waiting=%d running=%d completed=%d steps=%d",
                len(self.waiting_queue),
                len(self.running_batch.reqs),
                sum(1 for s in REQUEST_STATS.values() if s.status == "completed"),
                steps,
            )

    def event_loop(self, max_steps: int = 100_000) -> None:
        if self.enable_overlap:
            self.event_loop_overlap(max_steps)
        else:
            self.event_loop_normal(max_steps)

    _IDLE_SLEEP_S = 0.01

    def _aegis_normal_tier_pressure(self) -> tuple[float, float]:
        """Read normal-serving occupancy from this rank's local cache pools."""

        l1_used, l1_total = self.cache.kv_token_usage()
        l1_pressure = l1_used / max(l1_total, 1)
        host_pool = self.cache.hicache.mem_pool_host
        if host_pool is None or host_pool.capacity_pages <= 0:
            l2_pressure = 0.0
        else:
            l2_used = host_pool.capacity_pages - host_pool.available_size
            l2_pressure = l2_used / host_pool.capacity_pages
        return (
            min(max(float(l1_pressure), 0.0), 1.0),
            min(max(float(l2_pressure), 0.0), 1.0),
        )

    def _pop_aegis_physical_releases(self) -> list[dict[str, Any]]:
        pop = getattr(
            self.aegis_backup_manager,
            "pop_pending_physical_releases",
            None,
        )
        if not callable(pop):
            return []
        return [dict(item) for item in pop() if isinstance(item, dict)]

    def _apply_aegis_dp_budget(
        self,
        raw: dict,
        *,
        emit_ack: bool,
    ) -> dict:
        snapshot = DPBudgetSnapshot.from_dict(raw)
        result = self.aegis_backup_manager.apply_dp_budget(
            snapshot,
            pp_rank=self.pp_rank,
        )
        self._aegis_budget_snapshot = snapshot
        self._aegis_budget_version = snapshot.budget_version
        if emit_ack and self._aegis_event_queue is not None:
            self._aegis_event_queue.put(
                {
                    "type": "aegis_dp_budget_ack",
                    "dp_rank": self.dp_rank,
                    "pp_rank": self.pp_rank,
                    "tp_rank": self.tp_rank,
                    "budget_version": snapshot.budget_version,
                    "window_id": snapshot.window_id,
                    "result": dict(result),
                    "physical_releases": self._pop_aegis_physical_releases(),
                    "rank_state": self.aegis_backup_manager.snapshot(),
                }
            )
        return result

    def _handle_runtime_command(self, msg: object) -> bool:
        """Handle simulator-only control messages outside the SGLang batch path.

        Returns True when the scheduler should stop.  Backup placement is a
        rank-local metadata operation and therefore does not alter admission or
        batch lifecycle ordering.
        """

        if not isinstance(msg, tuple) or not msg:
            return False
        op = msg[0]
        if op == "stop":
            return True
        if op == "simulate_gpu_recovery_clear_l1":
            if len(msg) < 2 or not isinstance(msg[1], dict):
                self.logger.warning("Ignored malformed GPU L1 clear: %r", msg)
                return False
            self._pending_gpu_l1_clears.append(dict(msg[1]))
            self._maybe_clear_pending_gpu_l1()
            return False
        if op == "aegis_publish_dp_budget":
            if len(msg) < 2 or not isinstance(msg[1], dict):
                self.logger.warning("Ignored malformed DP budget: %r", msg)
                return False
            try:
                self._apply_aegis_dp_budget(dict(msg[1]), emit_ack=True)
            except Exception:
                self.logger.exception("Rejected DP budget %r", msg[1])
            return False
        if op == "aegis_l1_l2_invalidate_host_lease":
            if not self.aegis_l1_l2_enabled:
                return False
            if len(msg) < 2 or not isinstance(msg[1], dict):
                self.logger.warning("Ignored malformed Host lease eviction: %r", msg)
                return False
            command = dict(msg[1])
            hint = self.aegis_backup_manager.invalidate_l2_for_host_lease(
                str(command.get("backup_key") or ""),
                expected_version=int(command.get("expected_version", -1)),
            )
            if hint is not None and self._aegis_event_queue is not None:
                self._aegis_event_queue.put(
                    {
                        "type": "aegis_l1_l2_invalidate_host_lease_ack",
                        "dp_rank": self.dp_rank,
                        "pp_rank": self.pp_rank,
                        "tp_rank": self.tp_rank,
                        "backup_key": str(command.get("backup_key") or ""),
                        "expected_version": int(
                            command.get("expected_version", -1)
                        ),
                        "result": hint,
                        "physical_releases": self._pop_aegis_physical_releases(),
                        "rank_state": self.aegis_backup_manager.snapshot(),
                        "reason": "controller_host_pool_eviction",
                    }
                )
            return False
        if op == "aegis_l1_l2_supersede_backup":
            if not self.aegis_l1_l2_enabled:
                return False
            if len(msg) < 2 or not isinstance(msg[1], dict):
                self.logger.warning("Ignored malformed frontier supersede: %r", msg)
                return False
            command = dict(msg[1])
            result = self.aegis_backup_manager.supersede(
                str(command.get("backup_key") or ""),
                expected_version=int(command.get("expected_version", -1)),
                superseded_by=str(command.get("superseded_by") or ""),
            )
            if self._aegis_event_queue is not None:
                self._aegis_event_queue.put(
                    {
                        "type": "aegis_l1_l2_supersede_ack",
                        "dp_rank": self.dp_rank,
                        "pp_rank": self.pp_rank,
                        "tp_rank": self.tp_rank,
                        "request_backup_key": str(
                            command.get("request_backup_key")
                            or command.get("backup_key")
                            or ""
                        ),
                        "stage_backup_key": str(
                            command.get("backup_key") or ""
                        ),
                        "required_pp_size": int(
                            command.get(
                                "required_pp_size", self.server_args.pp_size
                            )
                        ),
                        "result": result,
                        "physical_releases": self._pop_aegis_physical_releases(),
                        "rank_state": self.aegis_backup_manager.snapshot(),
                    }
                )
            return False
        if op == "aegis_l1_l2_complete_request":
            if not self.aegis_l1_l2_enabled:
                return False
            if len(msg) < 2 or not isinstance(msg[1], dict):
                self.logger.warning("Ignored malformed CompleteRequest: %r", msg)
                return False
            command = dict(msg[1])
            result = self.aegis_backup_manager.complete_request(
                str(command.get("stage_backup_key") or ""),
                expected_version=int(command.get("expected_version", -1)),
            )
            physical_releases = self._pop_aegis_physical_releases()
            if not physical_releases:
                physical_releases = [
                    {
                        "request_backup_key": str(
                            command.get("request_backup_key") or ""
                        ),
                        "stage_backup_key": str(
                            command.get("stage_backup_key") or ""
                        ),
                        "version": int(command.get("expected_version", -1)),
                        "pp_rank": self.pp_rank,
                        "tier": None,
                        "ordinal": None,
                        "token_start": None,
                        "token_end": None,
                        "reason": "request_completed",
                    }
                ]
            if self._aegis_event_queue is not None:
                self._aegis_event_queue.put(
                    {
                        "type": "aegis_l1_l2_complete_request_ack",
                        "dp_rank": self.dp_rank,
                        "pp_rank": self.pp_rank,
                        "tp_rank": self.tp_rank,
                        "request_backup_key": str(
                            command.get("request_backup_key") or ""
                        ),
                        "stage_backup_key": str(
                            command.get("stage_backup_key") or ""
                        ),
                        "expected_version": int(
                            command.get("expected_version", -1)
                        ),
                        "required_pp_size": int(
                            command.get(
                                "required_pp_size", self.server_args.pp_size
                            )
                        ),
                        "result": result,
                        "physical_releases": physical_releases,
                        "rank_state": self.aegis_backup_manager.snapshot(),
                    }
                )
            return False
        if op == "aegis_materialize_committed_prefix":
            if not (
                self.aegis_l1_l2_enabled
                and self.aegis_committed_copy_protocol
            ):
                return False
            if len(msg) < 2 or not isinstance(msg[1], dict):
                self.logger.warning(
                    "Ignored malformed committed-prefix command: %r", msg
                )
                return False
            command = dict(msg[1])
            source_pp_rank = int(
                command.get("source_pp_rank", self.pp_rank)
            )
            target_pp_rank = int(
                command.get("target_pp_rank", self.pp_rank)
            )
            if source_pp_rank != self.pp_rank or target_pp_rank != self.pp_rank:
                self.logger.warning(
                    "Rejected cross-stage committed-prefix command: %r", command
                )
                return False
            try:
                materialize_kwargs: dict[str, Any] = {
                    "expected_version": int(command.get("version", -1)),
                    "source_committed_token_end": int(
                        command.get("source_committed_token_end", -1)
                    ),
                    "execution_id": str(command.get("execution_id") or ""),
                    "block_generation": int(
                        command.get("block_generation", -1)
                    ),
                }
                physical_tiers = command.get("completed_tiers")
                if physical_tiers is not None:
                    if not isinstance(
                        self.aegis_backup_manager,
                        AegisIndependentTierRankManager,
                    ):
                        raise ValueError(
                            "tier-specific commit requires latest independent-tier "
                            "AegisKV semantics"
                        )
                    if not (
                        isinstance(physical_tiers, (list, tuple))
                        and len(physical_tiers) == 1
                        and str(physical_tiers[0]).strip().lower()
                        in {"hbm", "dram"}
                    ):
                        raise ValueError(
                            "physical completed_tiers must contain exactly one of "
                            "hbm or dram"
                        )
                    if not (
                        bool(command.get("physical_group_commit_verified", False))
                        and str(command.get("physical_group_id") or "")
                    ):
                        raise ValueError(
                            "tier-specific commit requires a verified physical "
                            "group identity"
                        )
                    materialize_kwargs["completed_tiers"] = physical_tiers
                payload = self.aegis_backup_manager.materialize_committed_prefix(
                    str(command.get("stage_backup_key") or ""),
                    **materialize_kwargs,
                )
                if physical_tiers is not None:
                    payload.update(
                        {
                            "physical_group_id": str(
                                command["physical_group_id"]
                            ),
                            "physical_group_commit_verified": True,
                        }
                    )
            except Exception:
                self.logger.exception(
                    "Rejected AegisKV committed-prefix command %r", command
                )
                return False
            event = {
                "type": "aegis_l1_l2_group_commit_ack",
                "payload": payload,
                "updated_hints": [],
                "physical_releases": self._pop_aegis_physical_releases(),
                "rank_state": self.aegis_backup_manager.snapshot(),
                "dp_rank": self.dp_rank,
                "pp_rank": self.pp_rank,
                "tp_rank": self.tp_rank,
                "request_backup_key": str(
                    command.get("request_backup_key") or payload["backup_key"]
                ),
                "stage_backup_key": str(
                    command.get("stage_backup_key") or payload["backup_key"]
                ),
                "source_pp_rank": source_pp_rank,
                "target_pp_rank": target_pp_rank,
                "required_pp_size": int(
                    command.get("required_pp_size", self.server_args.pp_size)
                ),
                "route_version": max(
                    int(command.get("route_version", 1)), 1
                ),
                "budget_version": max(
                    int(command.get("budget_version", 1)), 1
                ),
            }
            if self._aegis_event_queue is not None:
                self._aegis_event_queue.put(event)
            self.logger.info(
                "AegisKV GroupCommit rid=%s source=%d target=%d pp=%d "
                "source_end=%d frontier=%d",
                payload["rid"],
                payload["source_dp_rank"],
                payload["target_dp_rank"],
                self.pp_rank,
                payload["source_committed_token_end"],
                payload["safe_frontier_tokens"],
            )
            return False
        if op != "aegis_l1_l2_backup_plan" or not self.aegis_l1_l2_enabled:
            return False
        if len(msg) < 2 or not isinstance(msg[1], dict):
            self.logger.warning("Ignored malformed AegisKV backup plan: %r", msg)
            return False
        try:
            l1_pressure, l2_pressure = self._aegis_normal_tier_pressure()
            local_plan = dict(msg[1])
            budget_raw = local_plan.get("dp_budget_snapshot")
            if isinstance(budget_raw, dict):
                self._apply_aegis_dp_budget(budget_raw, emit_ack=False)
            source_pp_rank = int(local_plan.get("source_pp_rank", self.pp_rank))
            target_pp_rank = int(local_plan.get("target_pp_rank", self.pp_rank))
            if source_pp_rank != self.pp_rank or target_pp_rank != self.pp_rank:
                raise ValueError(
                    "stage-aligned Connector requires source_pp_rank == "
                    f"target_pp_rank == local pp_rank; source={source_pp_rank} "
                    f"target={target_pp_rank} local={self.pp_rank}"
                )
            local_plan["normal_l1_pressure"] = l1_pressure
            local_plan["normal_l2_pressure"] = l2_pressure
            plan_budget_version = max(
                int(local_plan.get("budget_version", 1)), 1
            )
            if (
                self._aegis_budget_version > 0
                and plan_budget_version != self._aegis_budget_version
            ):
                raise ValueError(
                    "backup plan budget version does not match stage policy: "
                    f"plan={plan_budget_version}, "
                    f"stage={self._aegis_budget_version}"
                )
            plan = AegisBackupPlan.from_dict(local_plan)
            ack = self.aegis_backup_manager.place(plan)
            updated_hints = self.aegis_backup_manager.pop_pending_hint_updates()
        except Exception:
            self.logger.exception("Algorithm 2 failed for backup plan %r", msg[1])
            return False
        event = {
            "type": "aegis_l1_l2_backup_ack",
            "payload": ack.to_dict(),
            "updated_hints": updated_hints,
            "physical_releases": self._pop_aegis_physical_releases(),
            "rank_state": self.aegis_backup_manager.snapshot(),
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "request_backup_key": str(
                local_plan.get("request_backup_key") or plan.backup_key
            ),
            "stage_backup_key": plan.backup_key,
            "source_pp_rank": source_pp_rank,
            "target_pp_rank": target_pp_rank,
            "required_pp_size": int(
                local_plan.get("required_pp_size", self.server_args.pp_size)
            ),
            "route_version": max(int(local_plan.get("route_version", 1)), 1),
            "budget_version": max(int(local_plan.get("budget_version", 1)), 1),
        }
        if self._aegis_event_queue is not None:
            self._aegis_event_queue.put(event)
        self.logger.info(
            "AegisKV ACK rid=%s source=%d target=%d action=%s "
            "L1=%d L2=%d R=%d alpha=%.3f beta=%.3f window=%d",
            ack.rid,
            ack.source_dp_rank,
            ack.target_dp_rank,
            ack.recovery_action,
            ack.l1_tokens,
            ack.l2_tokens,
            ack.recompute_tokens,
            ack.alpha,
            ack.beta,
            ack.window_id,
        )
        if ack.window_update:
            self.logger.info("AegisKV Algorithm 3 update %s", ack.window_update)
        return False

    def _maybe_clear_pending_gpu_l1(self) -> bool:
        """Clear GPU KV only at an idle boundary, preserving SGLang lifecycle."""

        if not self._pending_gpu_l1_clears or not self._rank_idle():
            return False
        payload = self._pending_gpu_l1_clears.popleft()
        request_id = str(payload.get("request_id") or "")
        self.cache.check_hicache_events()
        cache_result = self.cache.clear_l1_for_gpu_failure()
        backup_result = self.aegis_backup_manager.clear_l1_for_gpu_failure()
        event = {
            "type": "gpu_recovery_l1_clear_ack",
            "request_id": request_id,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "cache_result": cache_result,
            "backup_result": backup_result,
            "physical_releases": self._pop_aegis_physical_releases(),
            "rank_state": self.aegis_backup_manager.snapshot(),
            "cleared_time": self._time_driver.now(),
        }
        if self._aegis_event_queue is not None:
            self._aegis_event_queue.put(event)
        self.logger.warning(
            "Simulated GPU recovery cleared rank=%d L1_device=%d "
            "L1_backup=%d preserved_L2=%d",
            self.dp_rank,
            int(cache_result.get("cleared_device_tokens", 0)),
            int(backup_result.get("cleared_l1_backup_tokens", 0)),
            int(backup_result.get("preserved_l2_backup_tokens", 0)),
        )
        return True

    def run_event_loop_until_stop(
        self,
        *,
        cmd_queue=None,
        stop_event=None,
    ) -> None:
        """Run scheduler like sglang run_event_loop; exit on stop signal when idle."""
        import queue as queue_module

        if self.enable_overlap:
            self.last_batch = None
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if cmd_queue is not None:
                try:
                    msg = cmd_queue.get_nowait()
                    if self._handle_runtime_command(msg):
                        break
                except queue_module.Empty:
                    pass

            if self._maybe_clear_pending_gpu_l1():
                continue

            if self.event_loop_step():
                continue

            if self._recover_idle_pending_requests():
                continue

            if self._maybe_clear_pending_gpu_l1():
                continue

            if self._rank_idle():
                if stop_event is not None and stop_event.wait(timeout=self._IDLE_SLEEP_S):
                    break
                if cmd_queue is not None:
                    try:
                        msg = cmd_queue.get(timeout=self._IDLE_SLEEP_S)
                        if self._handle_runtime_command(msg):
                            break
                    except queue_module.Empty:
                        pass
                else:
                    time.sleep(self._IDLE_SLEEP_S)
            else:
                self._maybe_log_scheduler_spin()
                self._maybe_log_wall_heartbeat()
                time.sleep(0.001)

        if self.enable_overlap:
            self._drain_result_queue()
            if self.last_batch is not None:
                self.process_batch_result(self.last_batch)
                self._finalize_last_batch()
                self.last_batch = None

    def _rank_idle(self) -> bool:
        if self._result_queue:
            return False
        if self.waiting_queue or self.chunked_req is not None or self.running_batch.reqs:
            return False
        if self.last_batch is not None and self._batch_has_active_reqs(self.last_batch):
            return False
        if int(self.server_args.pp_size) > 1 and hasattr(self, "_pp_running_mbs"):
            if getattr(self, "_pp_pending_batches", None):
                return False
            if any(batch.reqs for batch in self._pp_running_mbs):
                return False
            if any(
                batch is not None and self._batch_has_active_reqs(batch)
                for batch in self._pp_last_mbs
            ):
                return False
        return True

    def _all_done(self) -> bool:
        if self._result_queue:
            return False
        if self.waiting_queue or self.chunked_req is not None or self.running_batch.reqs:
            return False
        if self.last_batch is not None and self._batch_has_active_reqs(self.last_batch):
            return False
        if not REQUEST_STATS:
            return False
        return all(_is_terminal_status(s.status) for s in list(REQUEST_STATS.values()))

    def _is_stalled(self) -> bool:
        if not self.waiting_queue or self.running_batch.reqs:
            return False
        for req in self.waiting_queue:
            if req.extend_length > 0:
                return False
        return True

    def _make_batch(
        self,
        mode: ForwardMode,
        reqs: list[Req],
        timestamp: float,
    ) -> ScheduleBatch:
        batch = ScheduleBatch(
            batch_id=self._next_batch_id,
            rank_id=self.dp_rank,
            mode=mode,
            reqs=reqs,
            started_time=timestamp,
        )
        self._next_batch_id += 1
        return batch


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: int,
    ipc_registry,
    cmd_queue,
    stats_queue,
    ready_event,
    pp_rank: int = 0,
    tp_rank: int = 0,
) -> None:
    import logging
    import os
    import signal

    from sglsim.srt.utils.logging import configure_logger

    # The HTTP parent owns lifecycle and converts Ctrl-C into Engine.shutdown().
    # Ignoring SIGINT here avoids eight child tracebacks in the three-terminal
    # demo while the parent still sends the normal command-queue stop message.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    configure_logger(server_args)
    if server_args.sim_config_path:
        os.environ["SGLSIM_CONFIG_PATH"] = server_args.sim_config_path
    from sglsim.srt.utils.network import SimIpcRegistry

    SimIpcRegistry.install_global(ipc_registry)
    ConfigManager.load(server_args.sim_config_path)
    ConfigManager.apply_server_args_overrides(server_args)
    from sglsim.srt.simulation.manager.env import simulation_mode

    if not server_args.streaming_arrival:
        raise ValueError("streaming_arrival must be enabled for scheduler subprocess")
    os.environ["SGLSIM_SIMULATION_MODE"] = SimulationMode.STREAMING.value
    StateManager.reset()
    REQUEST_STATS.clear()
    scheduler = Scheduler(
        server_args,
        port_args,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
    )
    scheduler._aegis_event_queue = stats_queue
    if server_args.scheduler_telemetry:
        scheduler._scheduler_telemetry_queue = stats_queue
    ready_event.set()
    # Move the large, process-lifetime heap (torch/transformers/aiconfigurator)
    # into the permanent generation so per-step decode allocations don't trigger
    # full gen2 rescans of millions of static objects.
    import gc

    gc.collect()
    gc.freeze()
    try:
        if server_args.pp_size > 1:
            scheduler.run_event_loop_pp_until_stop(cmd_queue=cmd_queue)
        else:
            scheduler.run_event_loop_until_stop(cmd_queue=cmd_queue)
    except Exception:
        scheduler.logger.exception(
            "Scheduler subprocess dp_rank=%d pp_rank=%d tp_rank=%d crashed",
            dp_rank,
            pp_rank,
            tp_rank,
        )
        raise
