from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass

from sglsim.srt.managers.data_parallel_controller import (
    DataParallelController,
    SchedulerWorkerDiedError,
    launch_scheduler_process,
    launch_scheduler_processes,
)
from sglsim.srt.managers.detokenizer_manager import run_detokenizer_process
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.managers.tokenizer_manager import TokenizerManager, resolve_dp_size
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.utils.logging import configure_logger

logger = logging.getLogger(__name__)
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.state import StateManager
from sglsim.srt.simulation.recovery_model import (
    aegis_restore_data_plane,
    aegis_restore_overhead_s,
    aegis_restore_timing_source,
)
from sglsim.srt.simulation.types import RequestStats, SimulationMode
from sglsim.srt.utils.network import SimIpcRegistry, build_dp_ipc_registry
from sglsim.srt.utils.request_validation import validate_generate_request


@dataclass
class EngineResult:
    metrics: dict
    request_stats: dict


@dataclass(frozen=True)
class AegisHostBackupEntry:
    rid: str
    agent_id: str
    worker_seq: int
    source_dp_rank: int
    peer_dp_rank: int
    host_visible_tokens: int
    gpu_replica_tokens: int
    input_length: int
    generated_tokens: int
    updated_time: float


@dataclass
class LaunchState:
    server_args: ServerArgs
    port_args: PortArgs
    dp_size: int
    ipc_registry: SimIpcRegistry
    scheduler_procs: list[mp.Process]
    cmd_queues: list[mp.Queue]
    detokenizer_proc: mp.Process
    tokenizer_manager: TokenizerManager
    dp_controller: DataParallelController | None


def _normalized_fraction(value: object) -> float | None:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction < 0.0:
        return None
    return max(0.0, min(fraction, 1.0))


def _resolve_simulation_mode(server_args: ServerArgs) -> SimulationMode:
    if not server_args.streaming_arrival:
        raise ValueError("streaming_arrival must be enabled")
    return SimulationMode.STREAMING


def _log_sglang_startup_events(server_args: ServerArgs, dp_size: int) -> None:
    sched = ConfigManager.get_scheduler_config()
    cache = ConfigManager.get_cache_config(server_args.model_path)
    model = ConfigManager.get_model_info(server_args.model_path)
    platform = ConfigManager.get_platform_config()
    max_total_num_tokens = cache.l1_capacity_pages * cache.page_size
    context_len = model.context_len or 40960
    logger.info("server_args=%s", server_args)
    logger.info("Load weight begin.")
    logger.info("Load weight end.")
    logger.info("KV Cache is allocated.")
    logger.info("Capture CUDA graph begin.")
    logger.info("Capture CUDA graph end.")
    for dp_rank in range(dp_size):
        logger.info(
            "max_total_num_tokens=%d, chunked_prefill_size=%s, "
            "max_prefill_tokens=%d, max_running_requests=%d, context_len=%d, "
            "available_gpu_mem=%.2f GB",
            max_total_num_tokens,
            sched.chunked_prefill_size,
            sched.max_prefill_tokens,
            sched.effective_max_running_requests,
            context_len,
            platform.accelerator.hbm_capacity_gb,
        )


def launch_subprocesses(server_args: ServerArgs) -> LaunchState:
    configure_logger(server_args)
    if server_args.sim_config_path:
        os.environ["SGLSIM_CONFIG_PATH"] = server_args.sim_config_path
    ConfigManager.load(server_args.sim_config_path)
    ConfigManager.apply_server_args_overrides(server_args)
    os.environ["SGLSIM_SIMULATION_MODE"] = _resolve_simulation_mode(server_args).value

    port_args = PortArgs.init_new(server_args)
    dp_size = resolve_dp_size(server_args)
    _log_sglang_startup_events(server_args, dp_size)
    ipc_registry = build_dp_ipc_registry(port_args, dp_size)
    SimIpcRegistry.install_global(ipc_registry)

    scheduler_procs, cmd_queues, _stats_queue, _ready_events = (
        launch_scheduler_processes(
            server_args,
            port_args,
            ipc_registry,
            dp_size,
        )
    )

    detokenizer_ready = mp.Event()
    detokenizer_proc = mp.Process(
        target=run_detokenizer_process,
        args=(server_args, port_args, ipc_registry, detokenizer_ready),
        daemon=True,
    )
    detokenizer_proc.start()
    if not detokenizer_ready.wait(timeout=60):
        raise RuntimeError("Detokenizer subprocess failed to become ready")

    dp_controller = (
        DataParallelController(
            server_args,
            port_args,
            scheduler_procs=scheduler_procs,
        )
        if dp_size > 1
        else None
    )
    tokenizer_manager = TokenizerManager(
        server_args,
        port_args,
        dp_controller=dp_controller,
    )
    logger.info(
        "launch_subprocesses ready dp_size=%d schedulers=%d",
        dp_size,
        len(scheduler_procs),
    )
    return LaunchState(
        server_args=server_args,
        port_args=port_args,
        dp_size=dp_size,
        ipc_registry=ipc_registry,
        scheduler_procs=scheduler_procs,
        cmd_queues=cmd_queues,
        detokenizer_proc=detokenizer_proc,
        tokenizer_manager=tokenizer_manager,
        dp_controller=dp_controller,
    )


def init_tokenizer_manager(
    server_args: ServerArgs,
    port_args: PortArgs,
) -> TokenizerManager:
    return TokenizerManager(server_args, port_args)


class Engine:
    def __init__(self, server_args: ServerArgs) -> None:
        state = launch_subprocesses(server_args)
        self._init_from_launch_state(state)

    @classmethod
    def from_launch_state(cls, state: LaunchState) -> Engine:
        engine = cls.__new__(cls)
        engine._init_from_launch_state(state)
        return engine

    def _init_from_launch_state(self, state: LaunchState) -> None:
        self.server_args = state.server_args
        self.sim_mode = _resolve_simulation_mode(state.server_args)
        self.port_args = state.port_args
        self.dp_size = state.dp_size
        self._ipc_registry = state.ipc_registry
        self._dp_controller = state.dp_controller
        self._scheduler_procs = state.scheduler_procs
        self._cmd_queues = state.cmd_queues
        self._detokenizer_proc = state.detokenizer_proc
        self.tokenizer_manager = state.tokenizer_manager
        self.scheduler: Scheduler | None = None
        self.schedulers: list[Scheduler] = []
        self.detokenizer = None
        self._ready = True
        self._completion_events: dict[str, threading.Event] = {}
        self._completion_text: dict[str, str] = {}
        self._completion_lock = threading.Lock()
        self._drain_stop = threading.Event()
        self._drain_thread: threading.Thread | None = None
        self._failure_control_lock = threading.Lock()
        self._aegis_host_backup_lock = threading.Lock()
        self._aegis_host_backup: dict[
            tuple[str, int, int], AegisHostBackupEntry
        ] = {}
        REQUEST_STATS.clear()

    def is_ready(self) -> bool:
        return self._ready

    def soft_fail_dp_rank(self, dp_rank: int) -> dict:
        if self._dp_controller is None:
            if dp_rank != 0:
                raise ValueError(f"invalid DP rank {dp_rank}")
            return {
                "accepted": False,
                "requested_dp_rank": int(dp_rank),
                "soft_failed_ranks": [],
                "dp_size": self.dp_size,
            }
        accepted = self._dp_controller.soft_fail_rank(int(dp_rank))
        return {
            "accepted": accepted,
            "requested_dp_rank": int(dp_rank),
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "dp_size": self.dp_size,
        }

    def recover_soft_failed_dp_rank(self, dp_rank: int) -> dict:
        if self._dp_controller is None:
            if dp_rank != 0:
                raise ValueError(f"invalid DP rank {dp_rank}")
            return {
                "released": False,
                "requested_dp_rank": int(dp_rank),
                "soft_failed_ranks": [],
                "dp_size": self.dp_size,
            }
        released = self._dp_controller.recover_soft_failed_rank(int(dp_rank))
        return {
            "released": released,
            "requested_dp_rank": int(dp_rank),
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "dp_size": self.dp_size,
        }

    def soft_failure_status(self) -> dict:
        if self._dp_controller is None:
            return {
                "soft_failed_ranks": [],
                "hard_failed_ranks": [],
                "dp_size": self.dp_size,
            }
        return {
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "hard_failed_ranks": self._dp_controller.hard_failed(),
            "dp_size": self.dp_size,
        }

    def hard_fail_dp_rank(self, dp_rank: int) -> dict:
        if self._dp_controller is None or self.dp_size <= 1:
            raise ValueError("hard DP failure injection requires dp_size > 1")
        dp_rank = int(dp_rank)
        if dp_rank < 0 or dp_rank >= self.dp_size:
            raise ValueError(f"invalid DP rank {dp_rank}")
        with self._failure_control_lock:
            accepted = self._dp_controller.mark_rank_hard_failed(dp_rank)
            proc = self._scheduler_procs[dp_rank]
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
            exitcode = proc.exitcode
        return {
            "accepted": accepted,
            "requested_dp_rank": dp_rank,
            "hard_failed_ranks": self._dp_controller.hard_failed(),
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "dp_size": self.dp_size,
            "exitcode": exitcode,
        }

    def restart_hard_failed_dp_rank(self, dp_rank: int) -> dict:
        if self._dp_controller is None or self.dp_size <= 1:
            raise ValueError("hard DP failure recovery requires dp_size > 1")
        dp_rank = int(dp_rank)
        if dp_rank < 0 or dp_rank >= self.dp_size:
            raise ValueError(f"invalid DP rank {dp_rank}")
        with self._failure_control_lock:
            old_proc = self._scheduler_procs[dp_rank]
            if old_proc.is_alive():
                old_proc.terminate()
                old_proc.join(timeout=2)
            if old_proc.is_alive():
                old_proc.kill()
                old_proc.join(timeout=2)
            proc, cmd_queue, ready_event = launch_scheduler_process(
                self.server_args,
                self.port_args,
                self._ipc_registry,
                dp_rank,
            )
            if not ready_event.wait(timeout=60):
                proc.terminate()
                proc.join(timeout=2)
                raise RuntimeError(
                    f"DP scheduler subprocess {dp_rank} failed to restart"
                )
            self._scheduler_procs[dp_rank] = proc
            self._cmd_queues[dp_rank] = cmd_queue
            restarted = self._dp_controller.mark_rank_restarted(dp_rank)
        return {
            "restarted": restarted,
            "requested_dp_rank": dp_rank,
            "hard_failed_ranks": self._dp_controller.hard_failed(),
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "dp_size": self.dp_size,
            "pid": proc.pid,
        }

    def _register_rid(self, rid: str) -> threading.Event:
        with self._completion_lock:
            event = self._completion_events.get(rid)
            if event is None:
                event = threading.Event()
                self._completion_events[rid] = event
            else:
                event.clear()
            self._completion_text.pop(rid, None)
            return event

    def _completion_event(self, rid: str) -> threading.Event:
        with self._completion_lock:
            event = self._completion_events.get(rid)
            if event is None:
                event = threading.Event()
                self._completion_events[rid] = event
            return event

    def _apply_detokenizer_message(self, msg: dict) -> None:
        rid = str(msg.get("rid", ""))
        if not rid:
            return
        stats = REQUEST_STATS.get(rid)
        if stats is None:
            stats = RequestStats(rid=rid)
            REQUEST_STATS[rid] = stats
        if "status" in msg:
            stats.status = str(msg["status"])
        if "last_event_time" in msg:
            stats.last_event_time = float(msg["last_event_time"])
        if "input_length" in msg:
            stats.input_length = int(msg["input_length"])
        if "output_length" in msg:
            stats.output_length = int(msg["output_length"])
        if "generated_tokens" in msg:
            stats.generated_tokens = int(msg["generated_tokens"])
        if "gen_token_latencies" in msg:
            stats.gen_token_latencies = list(msg["gen_token_latencies"])
        if "cache_hit_tokens" in msg:
            stats.cache_hit_tokens = int(msg["cache_hit_tokens"])
        if "created_time" in msg:
            stats.created_time = float(msg["created_time"])
        if "queue_start" in msg:
            stats.queue_start = float(msg["queue_start"])
        if "queue_end" in msg:
            stats.queue_end = float(msg["queue_end"])
        if "service_time_s" in msg:
            stats.service_time_s = float(msg["service_time_s"])
        if "load_back_tokens" in msg:
            stats.load_back_tokens = int(msg["load_back_tokens"])
        if "prefetch_complete_tokens" in msg:
            stats.prefetch_complete_tokens = int(msg["prefetch_complete_tokens"])
        if "workflow_id" in msg:
            stats.workflow_id = str(msg["workflow_id"])
        if "stage_id" in msg:
            stats.stage_id = int(msg["stage_id"])
        if "stage_width" in msg:
            stats.stage_width = int(msg["stage_width"])
        if "stage_local_agent_index" in msg:
            stats.stage_local_agent_index = int(msg["stage_local_agent_index"])
        if "decode_dominant" in msg:
            stats.decode_dominant = bool(msg["decode_dominant"])
        if "workflow_shape" in msg:
            stats.workflow_shape = str(msg["workflow_shape"])
        if "agent_id" in msg:
            stats.agent_id = str(msg["agent_id"])
        if "critical_agent" in msg:
            stats.critical_agent = bool(msg["critical_agent"])
        if "aegis_policy" in msg:
            stats.aegis_policy = str(msg["aegis_policy"])
        if "aegis_ablation" in msg:
            stats.aegis_ablation = str(msg["aegis_ablation"])
        if "aegis_gpu_budget_fraction" in msg:
            stats.aegis_gpu_budget_fraction = float(
                msg["aegis_gpu_budget_fraction"]
            )
        if "aegis_failure_rate_per_window" in msg:
            stats.aegis_failure_rate_per_window = float(
                msg["aegis_failure_rate_per_window"]
            )
        if "aegis_alpha_policy_source" in msg:
            stats.aegis_alpha_policy_source = str(msg["aegis_alpha_policy_source"])
        if "aegis_critical_pressure" in msg:
            stats.aegis_critical_pressure = float(msg["aegis_critical_pressure"])
        if "aegis_backup_kv" in msg:
            stats.aegis_backup_kv = bool(msg["aegis_backup_kv"])
        if "aegis_worker_seq" in msg:
            stats.aegis_worker_seq = int(msg["aegis_worker_seq"])
        if "aegis_assigned_dp_rank" in msg:
            stats.aegis_assigned_dp_rank = int(msg["aegis_assigned_dp_rank"])
        if "aegis_host_visible_tokens" in msg:
            stats.aegis_host_visible_tokens = int(msg["aegis_host_visible_tokens"])
        if "aegis_gpu_replica_tokens" in msg:
            stats.aegis_gpu_replica_tokens = int(msg["aegis_gpu_replica_tokens"])
        if "aegis_safe_frontier_tokens" in msg:
            stats.aegis_safe_frontier_tokens = int(msg["aegis_safe_frontier_tokens"])
        if "aegis_recompute_tokens" in msg:
            stats.aegis_recompute_tokens = int(msg["aegis_recompute_tokens"])
        if "aegis_recovery_action" in msg:
            stats.aegis_recovery_action = str(msg["aegis_recovery_action"])
        if "aegis_expected_recovery_ms" in msg:
            stats.aegis_expected_recovery_ms = float(msg["aegis_expected_recovery_ms"])
        if "aegis_candidate_retry" in msg:
            stats.aegis_candidate_retry = bool(msg["aegis_candidate_retry"])
        if "aegis_candidate_failed_dp_rank" in msg:
            stats.aegis_candidate_failed_dp_rank = int(
                msg["aegis_candidate_failed_dp_rank"]
            )
        if "aegis_candidate_pre_failure_output_tokens" in msg:
            stats.aegis_candidate_pre_failure_output_tokens = int(
                msg["aegis_candidate_pre_failure_output_tokens"]
            )
        if "aegis_candidate_pre_failure_total_tokens" in msg:
            stats.aegis_candidate_pre_failure_total_tokens = int(
                msg["aegis_candidate_pre_failure_total_tokens"]
            )
        if "aegis_candidate_backed_up_tokens" in msg:
            stats.aegis_candidate_backed_up_tokens = int(
                msg["aegis_candidate_backed_up_tokens"]
            )
        if "aegis_candidate_recompute_tokens" in msg:
            stats.aegis_candidate_recompute_tokens = int(
                msg["aegis_candidate_recompute_tokens"]
            )
        if "aegis_candidate_recovery_action" in msg:
            stats.aegis_candidate_recovery_action = str(
                msg["aegis_candidate_recovery_action"]
            )
        if "aegis_candidate_failover_overhead_s" in msg:
            stats.aegis_candidate_failover_overhead_s = float(
                msg["aegis_candidate_failover_overhead_s"]
            )
        if "aegis_failover_replay" in msg:
            stats.aegis_failover_replay = bool(msg["aegis_failover_replay"])
        if "aegis_failed_source_dp_rank" in msg:
            stats.aegis_failed_source_dp_rank = int(
                msg["aegis_failed_source_dp_rank"]
            )
        if "aegis_replay_restored_tokens" in msg:
            stats.aegis_replay_restored_tokens = int(
                msg["aegis_replay_restored_tokens"]
            )
        if "aegis_replay_restored_prefill_tokens" in msg:
            stats.aegis_replay_restored_prefill_tokens = int(
                msg["aegis_replay_restored_prefill_tokens"]
            )
        if "aegis_replay_recompute_tokens" in msg:
            stats.aegis_replay_recompute_tokens = int(
                msg["aegis_replay_recompute_tokens"]
            )
        if "aegis_replay_restore_action" in msg:
            stats.aegis_replay_restore_action = str(
                msg["aegis_replay_restore_action"]
            )
        if "aegis_replay_restore_overhead_s" in msg:
            stats.aegis_replay_restore_overhead_s = float(
                msg["aegis_replay_restore_overhead_s"]
            )
        if "aegis_replay_restore_sleep_s" in msg:
            stats.aegis_replay_restore_sleep_s = float(
                msg["aegis_replay_restore_sleep_s"]
            )
        if "aegis_replay_restore_data_plane" in msg:
            stats.aegis_replay_restore_data_plane = str(
                msg["aegis_replay_restore_data_plane"]
            )
        if "aegis_replay_restore_timing_source" in msg:
            stats.aegis_replay_restore_timing_source = str(
                msg["aegis_replay_restore_timing_source"]
            )
        if "aegis_cross_rank_retry" in msg:
            stats.aegis_cross_rank_retry = bool(msg["aegis_cross_rank_retry"])
        if "aegis_soft_failure_injected" in msg:
            stats.aegis_soft_failure_injected = bool(
                msg["aegis_soft_failure_injected"]
            )
        if "aegis_hard_failure_injected" in msg:
            stats.aegis_hard_failure_injected = bool(
                msg["aegis_hard_failure_injected"]
            )
        if "aegis_host_backup_lookup_hit" in msg:
            stats.aegis_host_backup_lookup_hit = bool(
                msg["aegis_host_backup_lookup_hit"]
            )
        if "aegis_host_backup_lookup_tokens" in msg:
            stats.aegis_host_backup_lookup_tokens = int(
                msg["aegis_host_backup_lookup_tokens"]
            )
        if "aegis_peer_gpu_lookup_hit" in msg:
            stats.aegis_peer_gpu_lookup_hit = bool(
                msg["aegis_peer_gpu_lookup_hit"]
            )
        if "aegis_peer_gpu_lookup_tokens" in msg:
            stats.aegis_peer_gpu_lookup_tokens = int(
                msg["aegis_peer_gpu_lookup_tokens"]
            )
        if "text" in msg:
            self._completion_text[rid] = str(msg["text"])
        if msg.get("finished"):
            if stats.status != "aborted":
                stats.status = "completed"
            self._record_aegis_host_backup(stats)
            with self._completion_lock:
                event = self._completion_events.get(rid)
                if event is not None:
                    event.set()

    def _record_aegis_host_backup(self, stats: RequestStats) -> None:
        if stats.status != "completed":
            return
        if (
            stats.aegis_host_visible_tokens <= 0
            and stats.aegis_gpu_replica_tokens <= 0
        ):
            return
        if not stats.agent_id:
            return
        if stats.aegis_assigned_dp_rank < 0:
            return
        key = (
            stats.agent_id,
            int(stats.aegis_worker_seq),
            int(stats.aegis_assigned_dp_rank),
        )
        entry = AegisHostBackupEntry(
            rid=stats.rid,
            agent_id=stats.agent_id,
            worker_seq=int(stats.aegis_worker_seq),
            source_dp_rank=int(stats.aegis_assigned_dp_rank),
            peer_dp_rank=(
                (int(stats.aegis_assigned_dp_rank) + 1) % max(int(self.dp_size), 1)
            ),
            host_visible_tokens=int(stats.aegis_host_visible_tokens),
            gpu_replica_tokens=int(stats.aegis_gpu_replica_tokens),
            input_length=int(stats.input_length),
            generated_tokens=int(stats.generated_tokens),
            updated_time=time.time(),
        )
        with self._aegis_host_backup_lock:
            self._aegis_host_backup[key] = entry

    @staticmethod
    def _aegis_restore_overhead_s(
        *, action: str, token_count: int, restored_tokens: int
    ) -> float:
        return aegis_restore_overhead_s(
            action=action,
            token_count=token_count,
            restored_tokens=restored_tokens,
        )

    @staticmethod
    def _sim_int(sim: dict, key: str, default: int = 0) -> int:
        try:
            return int(sim.get(key, default))
        except (TypeError, ValueError):
            return default

    def _apply_aegis_host_backup_lookup(self, req: GenerateReqInput) -> None:
        sim = req.simulation
        if not bool(sim.get("failover_replay", False)):
            return

        agent_id = str(sim.get("agent_id") or sim.get("worker_id") or "")
        worker_seq = self._sim_int(sim, "worker_seq", 0)
        failed_rank = self._sim_int(sim, "failed_dp_rank", -1)
        pre_failure_total = self._sim_int(sim, "pre_failure_total_tokens", 0)
        if pre_failure_total <= 0:
            pre_failure_total = (
                self._sim_int(sim, "input_length", 0)
                + self._sim_int(sim, "failure_after_output_tokens", 0)
            )

        entry = None
        if agent_id and failed_rank >= 0:
            key = (agent_id, worker_seq, failed_rank)
            with self._aegis_host_backup_lock:
                entry = self._aegis_host_backup.get(key)

        if entry is None:
            sim["host_backup_lookup_hit"] = False
            sim["host_backup_lookup_tokens"] = 0
            sim["peer_gpu_lookup_hit"] = False
            sim["peer_gpu_lookup_tokens"] = 0
            sim.setdefault("restore_tokens", 0)
            sim.setdefault("restore_recompute_tokens", max(pre_failure_total, 0))
            sim.setdefault("restore_action", "recompute")
            sim.setdefault("restore_data_plane", "recompute")
            sim.setdefault("restore_timing_source", aegis_restore_timing_source())
            sim.setdefault(
                "restore_overhead_s",
                self._aegis_restore_overhead_s(
                    action="recompute",
                    token_count=max(pre_failure_total, 0),
                    restored_tokens=0,
                ),
            )
            return

        host_tokens = min(
            max(int(entry.host_visible_tokens), 0),
            max(pre_failure_total, 0),
        )
        retry_rank = self._sim_int(sim, "retry_dp_rank", self._sim_int(sim, "force_dp_rank", -1))
        peer_tokens = 0
        if retry_rank == int(entry.peer_dp_rank):
            peer_tokens = min(
                max(int(entry.gpu_replica_tokens), 0),
                max(pre_failure_total, 0),
            )
        gpu_budget = _normalized_fraction(sim.get("aegis_gpu_budget_fraction", -1.0))
        if gpu_budget is not None:
            peer_tokens = min(peer_tokens, int(max(pre_failure_total, 0) * gpu_budget))
        ablation = (
            str(sim.get("aegis_ablation") or "")
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        def choose_recompute() -> tuple[str, int, float]:
            return (
                "recompute",
                0,
                self._aegis_restore_overhead_s(
                    action="recompute",
                    token_count=max(pre_failure_total, 0),
                    restored_tokens=0,
                ),
            )

        if ablation == "no_lifecycle" or (
            ablation == "no_critical_restore"
            and bool(sim.get("critical_agent", False))
        ):
            host_tokens = 0
            peer_tokens = 0
        elif ablation == "no_segment_granularity":
            if host_tokens < max(pre_failure_total, 0):
                host_tokens = 0
            if peer_tokens < max(pre_failure_total, 0):
                peer_tokens = 0
        elif ablation == "no_pressure":
            peer_tokens = 0
        elif ablation == "no_hysteresis" and worker_seq % 2:
            host_tokens = 0
            peer_tokens = 0

        candidates = [
            choose_recompute()
        ]
        if host_tokens > 0:
            candidates.append(
                (
                    "host_restore",
                    host_tokens,
                    self._aegis_restore_overhead_s(
                        action="host_restore",
                        token_count=max(pre_failure_total, 0),
                        restored_tokens=host_tokens,
                    ),
                )
            )
        if peer_tokens > 0:
            candidates.append(
                (
                    "gpu_replica",
                    peer_tokens,
                    self._aegis_restore_overhead_s(
                        action="gpu_replica",
                        token_count=max(pre_failure_total, 0),
                        restored_tokens=peer_tokens,
                    ),
                )
            )

        policy = str(sim.get("aegis_policy") or "").lower().replace("-", "_").replace(" ", "_")
        if ablation == "no_restore_recompute":
            if host_tokens > 0:
                action = "host_restore"
                restore_tokens = host_tokens
                restore_overhead_s = self._aegis_restore_overhead_s(
                    action=action,
                    token_count=max(pre_failure_total, 0),
                    restored_tokens=restore_tokens,
                )
            elif peer_tokens > 0:
                action = "gpu_replica"
                restore_tokens = peer_tokens
                restore_overhead_s = self._aegis_restore_overhead_s(
                    action=action,
                    token_count=max(pre_failure_total, 0),
                    restored_tokens=restore_tokens,
                )
            else:
                action, restore_tokens, restore_overhead_s = candidates[0]
        elif policy in {"host", "host_only", "hostbackup", "host_backup", "uniform_host"}:
            if host_tokens > 0:
                action = "host_restore"
                restore_tokens = host_tokens
                restore_overhead_s = self._aegis_restore_overhead_s(
                    action=action,
                    token_count=max(pre_failure_total, 0),
                    restored_tokens=restore_tokens,
                )
            else:
                action, restore_tokens, restore_overhead_s = candidates[0]
        elif policy in {
            "gpu",
            "gpu_only",
            "gpu_replica",
            "gpureplica",
            "uniform_gpu",
        }:
            if peer_tokens > 0:
                action = "gpu_replica"
                restore_tokens = peer_tokens
                restore_overhead_s = self._aegis_restore_overhead_s(
                    action=action,
                    token_count=max(pre_failure_total, 0),
                    restored_tokens=restore_tokens,
                )
            else:
                action, restore_tokens, restore_overhead_s = candidates[0]
        elif policy in {"baseline", "none", "no_backup", "nobackup"}:
            action, restore_tokens, restore_overhead_s = candidates[0]
        else:
            action, restore_tokens, restore_overhead_s = min(
                candidates,
                key=lambda item: item[2],
            )
        recompute_tokens = max(pre_failure_total - restore_tokens, 0)
        sim["host_backup_lookup_hit"] = host_tokens > 0
        sim["host_backup_lookup_tokens"] = host_tokens
        sim["peer_gpu_lookup_hit"] = peer_tokens > 0
        sim["peer_gpu_lookup_tokens"] = peer_tokens
        sim["host_backup_source_rid"] = entry.rid
        sim["peer_gpu_source_rid"] = entry.rid if peer_tokens > 0 else ""
        sim["peer_gpu_rank"] = int(entry.peer_dp_rank)
        sim["restore_tokens"] = restore_tokens
        sim["restore_recompute_tokens"] = recompute_tokens
        sim["restore_action"] = action
        sim["restore_data_plane"] = aegis_restore_data_plane(action)
        sim["restore_timing_source"] = aegis_restore_timing_source()
        sim["restore_overhead_s"] = restore_overhead_s

    def _drain_detokenizer(self, max_messages: int = 256) -> int:
        drained = 0
        for _ in range(max_messages):
            try:
                msg = self.tokenizer_manager.recv_from_detokenizer.recv_pyobj(flags=1)
            except Exception:
                break
            drained += 1
            self._apply_detokenizer_message(msg)
        return drained

    def _drain_loop(self) -> None:
        while not self._drain_stop.is_set():
            if self._drain_detokenizer(max_messages=512) == 0:
                self._drain_stop.wait(timeout=Scheduler._IDLE_SLEEP_S)

    def start_background_scheduler(self) -> None:
        if self._drain_thread is not None:
            return
        self._drain_stop.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            name="sglsim-detokenizer-drain",
            daemon=True,
        )
        self._drain_thread.start()

    def stop_background_scheduler(self) -> None:
        self._drain_stop.set()
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=5)
            self._drain_thread = None

    def _check_scheduler_workers_alive(self) -> None:
        if self._dp_controller is not None:
            self._dp_controller.ensure_workers_alive()
            return
        for dp_rank, proc in enumerate(self._scheduler_procs):
            if proc.exitcode is not None:
                raise SchedulerWorkerDiedError(
                    f"DP scheduler worker {dp_rank} exited with code {proc.exitcode}"
                )

    def wait_for_rid(self, rid: str, timeout_s: float = 3600.0) -> RequestStats:
        event = self._completion_event(rid)
        deadline = time.monotonic() + timeout_s
        while not event.is_set():
            self._check_scheduler_workers_alive()
            self._drain_detokenizer()
            if event.wait(timeout=0.05):
                break
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"Timed out waiting for request {rid}")
        stats = REQUEST_STATS.get(rid)
        if stats is None:
            raise RuntimeError(f"Missing stats for completed request {rid}")
        return stats

    def completion_text(self, rid: str) -> str:
        return self._completion_text.get(rid, "")

    def _stamp_created_time(self, req: GenerateReqInput) -> None:
        created = time.time()
        req.created_time = created
        req.simulation["created_time"] = created

    def submit(self, req: GenerateReqInput) -> None:
        validate_generate_request(
            req,
            ConfigManager.get_model_info(self.server_args.model_path),
        )
        self._register_rid(req.rid)
        self._stamp_created_time(req)
        self._apply_aegis_host_backup_lookup(req)
        self.tokenizer_manager.generate_request(req)

    def generate_from_http(self, req: GenerateReqInput) -> tuple[RequestStats, str]:
        validate_generate_request(
            req,
            ConfigManager.get_model_info(self.server_args.model_path),
        )
        self._register_rid(req.rid)
        self._stamp_created_time(req)
        self._apply_aegis_host_backup_lookup(req)
        self.tokenizer_manager.generate_request(req)
        stats = self.wait_for_rid(req.rid)
        return stats, self.completion_text(req.rid)

    def _cache_metrics(self) -> dict:
        comm = ConfigManager.get_communication_config()
        out = StateManager.get_comm_totals()
        out["comm_mode"] = comm.mode
        return out

    def _any_stalled(self) -> bool:
        return False

    def run_until_idle(self, max_steps: int = 100_000) -> EngineResult:
        del max_steps
        deadline = time.monotonic() + 3600.0
        while time.monotonic() < deadline:
            self._drain_detokenizer()
            pending = [
                rid
                for rid, event in self._completion_events.items()
                if not event.is_set()
            ]
            if not pending:
                break
            time.sleep(0.01)

        from sglsim.srt.observability.bench_metrics import calc_bench_metrics

        duration = max(
            (s.last_event_time for s in REQUEST_STATS.values()),
            default=0.0,
        )
        metrics = calc_bench_metrics(list(REQUEST_STATS.values()), duration)
        metrics.update(self._cache_metrics())
        metrics["stalled"] = self._any_stalled()
        return EngineResult(metrics=metrics, request_stats=dict(REQUEST_STATS))

    def shutdown(self) -> None:
        self.stop_background_scheduler()
        for cmd_queue in self._cmd_queues:
            try:
                cmd_queue.put(("stop",))
            except Exception:
                pass
        for proc in self._scheduler_procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        if self._detokenizer_proc.is_alive():
            try:
                registry = self._ipc_registry.get_queue(
                    self.port_args.detokenizer_ipc_name
                )
                if registry is not None:
                    registry.put({"op": "stop"})
            except Exception:
                pass
            self._detokenizer_proc.join(timeout=5)
            if self._detokenizer_proc.is_alive():
                self._detokenizer_proc.terminate()
                self._detokenizer_proc.join(timeout=2)
            if self._detokenizer_proc.is_alive():
                self._detokenizer_proc.kill()
                self._detokenizer_proc.join(timeout=2)
        SimIpcRegistry.install_global(None)
