from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from sglsim.srt.managers.data_parallel_controller import (
    DataParallelController,
    SchedulerWorkerDiedError,
    launch_pipeline_dp_scheduler_processes,
    launch_scheduler_process,
    launch_pipeline_scheduler_processes,
    launch_scheduler_processes,
)
from sglsim.srt.managers.detokenizer_manager import run_detokenizer_process
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.managers.pipeline_parallel import PipelineProcessGroup
from sglsim.srt.managers.tokenizer_manager import TokenizerManager, resolve_dp_size
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.utils.logging import configure_logger

logger = logging.getLogger(__name__)
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.aegis_exact_contract import (
    load_certified_exact_window_plan,
)
from sglsim.srt.simulation.manager.state import StateManager
from sglsim.srt.simulation.recovery_model import (
    aegis_restore_data_plane,
    aegis_restore_overhead_s,
    aegis_restore_timing_source,
)
from sglsim.srt.simulation.types import RequestStats, SimulationMode
from sglsim.srt.utils.network import (
    SimIpcRegistry,
    build_dp_ipc_registry,
    build_dp_pp_ipc_registry,
)
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
    gpu_replica_quota_tokens: int = 0
    gpu_replica_overlap_tokens: int = 0
    gpu_replica_unique_tokens: int = 0
    gpu_replica_reserved_tokens: int = 0
    gpu_replica_evictable: bool = False


@dataclass
class LaunchState:
    server_args: ServerArgs
    port_args: PortArgs
    dp_size: int
    ipc_registry: SimIpcRegistry
    scheduler_procs: list[mp.Process]
    cmd_queues: list[mp.Queue]
    stats_queue: mp.Queue
    detokenizer_proc: mp.Process
    tokenizer_manager: TokenizerManager
    dp_controller: DataParallelController | None
    pipeline_process_group: PipelineProcessGroup | None = None


def _strict_json_telemetry(
    value: object,
    *,
    path: str = "",
    replaced_paths: list[str],
) -> object:
    """Replace non-finite diagnostic scores with explicit JSON nulls."""

    if isinstance(value, dict):
        return {
            str(key): _strict_json_telemetry(
                nested,
                path=f"{path}.{key}" if path else str(key),
                replaced_paths=replaced_paths,
            )
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_telemetry(
                nested,
                path=f"{path}[{index}]",
                replaced_paths=replaced_paths,
            )
            for index, nested in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        replaced_paths.append(path)
        return None
    return value


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


def _configured_exact_window_path(server_args: ServerArgs) -> Path | None:
    raw = ConfigManager._raw or {}
    aegis = raw.get("aegis_l1_l2", {})
    if not isinstance(aegis, dict):
        return None
    value = str(aegis.get("exact_window_plan_path") or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        if not server_args.sim_config_path:
            raise ValueError(
                "relative exact_window_plan_path requires sim_config_path"
            )
        path = Path(server_args.sim_config_path).expanduser().resolve().parent / path
    return path.resolve()


def _wait_for_dp_budget_barrier(
    controller: DataParallelController,
    stats_queue: mp.Queue,
    *,
    timeout_s: float = 60.0,
) -> dict:
    """Drain stage ACKs until every DP.PP worker has one budget version."""

    deadline = time.monotonic() + max(float(timeout_s), 0.1)
    while True:
        status = controller.aegis_budget_acknowledgement_status()
        if status["all_acknowledged"]:
            return status
        controller.ensure_workers_alive()
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                "timed out waiting for the DP x PP budget ACK barrier: "
                f"{status}"
            )
        try:
            event = stats_queue.get(timeout=min(remaining, 0.1))
        except queue.Empty:
            continue
        if isinstance(event, dict):
            controller.apply_aegis_rank_event(event)


def _install_configured_exact_window(
    server_args: ServerArgs,
    controller: DataParallelController | None,
    stats_queue: mp.Queue,
) -> None:
    """Install an offline-certified plan before the HTTP server is ready."""

    path = _configured_exact_window_path(server_args)
    if path is None:
        return
    if controller is None:
        raise ValueError("certified exact windows require dp_size > 1")
    aegis = (ConfigManager._raw or {}).get("aegis_l1_l2", {})
    if not isinstance(aegis, dict) or not bool(aegis.get("enabled", False)):
        raise ValueError("exact_window_plan_path requires aegis_l1_l2.enabled=true")
    plan = load_certified_exact_window_plan(path)
    _wait_for_dp_budget_barrier(controller, stats_queue)
    installed = controller.install_certified_exact_window(plan)
    final_barrier = _wait_for_dp_budget_barrier(controller, stats_queue)
    logger.info(
        "Installed certified exact window %s from %s with budget barrier %s",
        installed,
        path,
        final_barrier,
    )


def launch_subprocesses(server_args: ServerArgs) -> LaunchState:
    configure_logger(server_args)
    if server_args.sim_config_path:
        os.environ["SGLSIM_CONFIG_PATH"] = server_args.sim_config_path
    ConfigManager.load(server_args.sim_config_path)
    ConfigManager.apply_server_args_overrides(server_args)
    resolved_scheduler = ConfigManager.get_scheduler_config()
    server_args.dp_size = int(resolved_scheduler.dp_size)
    server_args.tp_size = int(resolved_scheduler.tp_size)
    configured_pp_size = int(resolved_scheduler.pp_size)
    raw_cfg = getattr(ConfigManager, "_raw", None) or {}
    if bool(raw_cfg.get("logical_pp", False)):
        server_args.logical_pp = True
    logical_pp = bool(getattr(server_args, "logical_pp", False))
    server_args.pp_size = 1 if logical_pp else configured_pp_size
    if server_args.pp_size > 1:
        # Matches ServerArgs.__post_init__ for a PP size supplied by the config
        # rather than directly on the CLI.
        server_args.disable_overlap_schedule = True
    os.environ["SGLSIM_SIMULATION_MODE"] = _resolve_simulation_mode(server_args).value

    port_args = PortArgs.init_new(server_args)
    dp_size = resolve_dp_size(server_args)
    _log_sglang_startup_events(server_args, dp_size)
    pipeline_process_group: PipelineProcessGroup | None = None
    logical_pp = bool(getattr(server_args, "logical_pp", False))
    process_pp_size = 1 if logical_pp else int(server_args.pp_size)
    ipc_registry = (
        build_dp_pp_ipc_registry(port_args, dp_size, process_pp_size)
        if process_pp_size > 1
        else build_dp_ipc_registry(port_args, dp_size)
    )
    SimIpcRegistry.install_global(ipc_registry)

    if process_pp_size > 1:
        pipeline_process_group, stats_queue = launch_pipeline_scheduler_processes(
            server_args,
            port_args,
            ipc_registry,
            dp_size=dp_size,
            pp_size=process_pp_size,
            tp_size=server_args.tp_size,
        )
        # Keep the long-standing DP-indexed compatibility views.  Group-wide
        # health, failure, and shutdown paths use ``pipeline_process_group``.
        scheduler_procs = [
            pipeline_process_group.leader_for_dp(dp_rank).process
            for dp_rank in range(dp_size)
        ]
        cmd_queues = [
            pipeline_process_group.leader_for_dp(dp_rank).command_queue
            for dp_rank in range(dp_size)
        ]
    else:
        if logical_pp and int(server_args.pp_size) > 1:
            # Workers must not expect a physical PP mesh.
            server_args.pp_size = 1
        scheduler_procs, cmd_queues, stats_queue, _ready_events = (
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
            scheduler_cmd_queues=cmd_queues,
            pipeline_process_group=pipeline_process_group,
        )
        if dp_size > 1
        else None
    )
    _install_configured_exact_window(
        server_args,
        dp_controller,
        stats_queue,
    )
    tokenizer_manager = TokenizerManager(
        server_args,
        port_args,
        dp_controller=dp_controller,
    )
    logger.info(
        "launch_subprocesses ready dp_size=%d schedulers=%d",
        dp_size,
        (
            pipeline_process_group.process_count
            if pipeline_process_group is not None
            else len(scheduler_procs)
        ),
    )
    return LaunchState(
        server_args=server_args,
        port_args=port_args,
        dp_size=dp_size,
        ipc_registry=ipc_registry,
        scheduler_procs=scheduler_procs,
        cmd_queues=cmd_queues,
        stats_queue=stats_queue,
        detokenizer_proc=detokenizer_proc,
        tokenizer_manager=tokenizer_manager,
        dp_controller=dp_controller,
        pipeline_process_group=pipeline_process_group,
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
        self._pipeline_process_group = state.pipeline_process_group
        self._scheduler_stats_queue = state.stats_queue
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
        self._scheduler_telemetry_lock = threading.Lock()
        self._scheduler_telemetry: list[dict] = []
        self._calibration_failure_lock = threading.Lock()
        self._calibration_failure_sequence = 0
        self._calibration_failure_events: list[dict] = []
        REQUEST_STATS.clear()

    def is_ready(self) -> bool:
        return self._ready

    def _record_calibration_failure_event(
        self,
        *,
        operation: str,
        failure_mode: str,
        dp_rank: int,
        result: dict,
    ) -> dict:
        """Record an observational failure-control event for causal replay."""

        aegis_raw = (ConfigManager._raw or {}).get("aegis_l1_l2", {})
        if not isinstance(aegis_raw, dict):
            aegis_raw = {}
        raw_reload_horizon = aegis_raw.get(
            "modeled_model_reload_horizon_s"
        )
        if isinstance(raw_reload_horizon, dict):
            raw_reload_horizon = raw_reload_horizon.get(str(failure_mode))
        try:
            reload_horizon_s = (
                max(float(raw_reload_horizon), 0.0)
                if raw_reload_horizon is not None
                else None
            )
        except (TypeError, ValueError):
            reload_horizon_s = None
        with self._calibration_failure_lock:
            self._calibration_failure_sequence += 1
            event = {
                "event_id": (
                    f"online-failure-{self._calibration_failure_sequence:06d}"
                ),
                "event_sequence": self._calibration_failure_sequence,
                "observed_at_s": time.time(),
                "operation": str(operation),
                "failure_mode": str(failure_mode),
                "dp_rank": int(dp_rank),
                "failure_domain": f"dp_rank:{int(dp_rank)}",
                "accepted": bool(
                    result.get(
                        "accepted",
                        result.get("released", result.get("restarted", False)),
                    )
                ),
                "control_result": dict(result),
                "information_scope": "observed_failure_control_event",
                "model_reload_horizon_s": reload_horizon_s,
                "model_reload_horizon_source": (
                    "configured_causal_model"
                    if reload_horizon_s is not None
                    else "unavailable"
                ),
            }
            self._calibration_failure_events.append(event)
            return dict(event)

    def soft_fail_dp_rank(self, dp_rank: int) -> dict:
        if self._dp_controller is None:
            if dp_rank != 0:
                raise ValueError(f"invalid DP rank {dp_rank}")
            result = {
                "accepted": False,
                "requested_dp_rank": int(dp_rank),
                "soft_failed_ranks": [],
                "dp_size": self.dp_size,
            }
            self._record_calibration_failure_event(
                operation="inject",
                failure_mode="soft",
                dp_rank=dp_rank,
                result=result,
            )
            return result
        accepted = self._dp_controller.soft_fail_rank(int(dp_rank))
        result = {
            "accepted": accepted,
            "requested_dp_rank": int(dp_rank),
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "dp_size": self.dp_size,
        }
        self._record_calibration_failure_event(
            operation="inject",
            failure_mode="soft",
            dp_rank=dp_rank,
            result=result,
        )
        return result

    def recover_soft_failed_dp_rank(self, dp_rank: int) -> dict:
        if self._dp_controller is None:
            if dp_rank != 0:
                raise ValueError(f"invalid DP rank {dp_rank}")
            result = {
                "released": False,
                "requested_dp_rank": int(dp_rank),
                "soft_failed_ranks": [],
                "dp_size": self.dp_size,
            }
            self._record_calibration_failure_event(
                operation="recover",
                failure_mode="soft",
                dp_rank=dp_rank,
                result=result,
            )
            return result
        # Keep the rank unavailable until its scheduler reaches an idle SGLang
        # lifecycle boundary and has discarded all volatile GPU-resident KV.
        # Host L2 metadata survives this simulated GPU failure.
        l1_clear = self._dp_controller.clear_rank_l1_before_recovery(
            int(dp_rank),
            timeout_s=1800.0,
        )
        released = self._dp_controller.recover_soft_failed_rank(int(dp_rank))
        result = {
            "released": released,
            "requested_dp_rank": int(dp_rank),
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "dp_size": self.dp_size,
            "l1_clear": l1_clear,
        }
        self._record_calibration_failure_event(
            operation="recover",
            failure_mode="soft",
            dp_rank=dp_rank,
            result=result,
        )
        return result

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
            if self._pipeline_process_group is not None:
                failed_workers = self._pipeline_process_group.for_dp(dp_rank)
            else:
                failed_workers = ()
            processes = (
                tuple(worker.process for worker in failed_workers)
                if failed_workers
                else (self._scheduler_procs[dp_rank],)
            )
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
            exitcode = processes[0].exitcode
            stage_exitcodes = [
                {
                    "pp_rank": worker.spec.stage.pp_rank,
                    "tp_rank": worker.spec.stage.tp_rank,
                    "pid": worker.process.pid,
                    "exitcode": worker.process.exitcode,
                }
                for worker in failed_workers
            ]
        result = {
            "accepted": accepted,
            "requested_dp_rank": dp_rank,
            "hard_failed_ranks": self._dp_controller.hard_failed(),
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "dp_size": self.dp_size,
            "exitcode": exitcode,
            "terminated_process_count": len(processes),
            "stage_exitcodes": stage_exitcodes,
        }
        self._record_calibration_failure_event(
            operation="inject",
            failure_mode="hard",
            dp_rank=dp_rank,
            result=result,
        )
        return result

    def restart_hard_failed_dp_rank(self, dp_rank: int) -> dict:
        if self._dp_controller is None or self.dp_size <= 1:
            raise ValueError("hard DP failure recovery requires dp_size > 1")
        dp_rank = int(dp_rank)
        if dp_rank < 0 or dp_rank >= self.dp_size:
            raise ValueError(f"invalid DP rank {dp_rank}")
        with self._failure_control_lock:
            if self._pipeline_process_group is not None:
                old_group = self._pipeline_process_group
                for worker in old_group.for_dp(dp_rank):
                    proc = worker.process
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=2)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=2)
                self._reset_dp_pipeline_channels(dp_rank)
                restarted_workers = launch_pipeline_dp_scheduler_processes(
                    self.server_args,
                    self.port_args,
                    self._ipc_registry,
                    dp_size=self.dp_size,
                    dp_rank=dp_rank,
                    pp_size=old_group.pp_size,
                    tp_size=old_group.tp_size,
                    stats_queue=self._scheduler_stats_queue,
                )
                replacements = {
                    worker.spec.stage: worker for worker in restarted_workers
                }
                new_group = PipelineProcessGroup(
                    dp_size=old_group.dp_size,
                    pp_size=old_group.pp_size,
                    tp_size=old_group.tp_size,
                    workers=tuple(
                        replacements.get(worker.spec.stage, worker)
                        for worker in old_group.workers
                    ),
                )
                self._pipeline_process_group = new_group
                self._scheduler_procs = [
                    new_group.leader_for_dp(rank).process
                    for rank in range(self.dp_size)
                ]
                self._cmd_queues = [
                    new_group.leader_for_dp(rank).command_queue
                    for rank in range(self.dp_size)
                ]
                self._dp_controller.replace_pipeline_process_group(new_group)
                self._dp_controller.replay_current_dp_budget(dp_rank)
                proc = new_group.leader_for_dp(dp_rank).process
                stage_pids = [
                    {
                        "pp_rank": worker.spec.stage.pp_rank,
                        "tp_rank": worker.spec.stage.tp_rank,
                        "pid": worker.process.pid,
                    }
                    for worker in restarted_workers
                ]
            else:
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
                    stats_queue=self._scheduler_stats_queue,
                )
                if not ready_event.wait(timeout=60):
                    proc.terminate()
                    proc.join(timeout=2)
                    raise RuntimeError(
                        f"DP scheduler subprocess {dp_rank} failed to restart"
                    )
                self._scheduler_procs[dp_rank] = proc
                self._cmd_queues[dp_rank] = cmd_queue
                stage_pids = []
            restarted = self._dp_controller.mark_rank_restarted(dp_rank)
        result = {
            "restarted": restarted,
            "requested_dp_rank": dp_rank,
            "hard_failed_ranks": self._dp_controller.hard_failed(),
            "soft_failed_ranks": self._dp_controller.soft_failed(),
            "dp_size": self.dp_size,
            "pid": proc.pid,
            "restarted_process_count": (
                len(stage_pids) if stage_pids else 1
            ),
            "stage_pids": stage_pids,
        }
        self._record_calibration_failure_event(
            operation="recover",
            failure_mode="hard",
            dp_rank=dp_rank,
            result=result,
        )
        return result

    def _reset_dp_pipeline_channels(self, dp_rank: int) -> None:
        """Start a new causal generation for one failed DP pipeline ring."""

        pp_size = int(self.server_args.pp_size)
        endpoints: set[str] = set()
        for pp_rank in range(pp_size):
            endpoints.update(
                {
                    self.port_args.scheduler_input_ipc_name_for_stage(
                        dp_rank, pp_rank
                    ),
                    self.port_args.pp_proxy_ipc_name_for_stage(dp_rank, pp_rank),
                    self.port_args.pp_output_ipc_name_for_stage(dp_rank, pp_rank),
                }
            )
        for endpoint in endpoints:
            self._ipc_registry.reset(endpoint)

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
        for field_name in (
            "aegis_slack_lcb_ms",
            "aegis_slack_q50_ms",
            "aegis_slack_q90_ms",
            "aegis_slack_target_coverage",
        ):
            if field_name in msg:
                setattr(stats, field_name, float(msg[field_name]))
        for field_name in (
            "aegis_slack_calibration_id",
            "aegis_slack_calibration_group",
            "aegis_slack_source",
            "aegis_slack_kind",
            "aegis_need_state",
            "aegis_slack_role",
        ):
            if field_name in msg:
                setattr(stats, field_name, str(msg[field_name]))
        if "aegis_slack_turns_completed" in msg:
            stats.aegis_slack_turns_completed = int(
                msg["aegis_slack_turns_completed"]
            )
        if "aegis_backup_kv" in msg:
            stats.aegis_backup_kv = bool(msg["aegis_backup_kv"])
        if "aegis_worker_seq" in msg:
            stats.aegis_worker_seq = int(msg["aegis_worker_seq"])
        if "aegis_assigned_dp_rank" in msg:
            stats.aegis_assigned_dp_rank = int(msg["aegis_assigned_dp_rank"])
        if "aegis_route_plan_dp_rank" in msg:
            stats.aegis_route_plan_dp_rank = int(
                msg["aegis_route_plan_dp_rank"]
            )
        if "aegis_route_plan_applied" in msg:
            applied = msg["aegis_route_plan_applied"]
            stats.aegis_route_plan_applied = (
                bool(applied) if isinstance(applied, bool) else None
            )
        if "aegis_exact_admission_state_id" in msg:
            stats.aegis_exact_admission_state_id = str(
                msg["aegis_exact_admission_state_id"]
            )
        if "aegis_exact_admission_information_mode" in msg:
            stats.aegis_exact_admission_information_mode = str(
                msg["aegis_exact_admission_information_mode"]
            )
        if "aegis_exact_observed_unavailable_dp_ranks" in msg:
            raw_unavailable = msg[
                "aegis_exact_observed_unavailable_dp_ranks"
            ]
            stats.aegis_exact_observed_unavailable_dp_ranks = [
                int(rank) for rank in raw_unavailable
            ] if isinstance(raw_unavailable, (list, tuple)) else []
        if "aegis_host_visible_tokens" in msg:
            stats.aegis_host_visible_tokens = int(msg["aegis_host_visible_tokens"])
        if "aegis_gpu_replica_tokens" in msg:
            stats.aegis_gpu_replica_tokens = int(msg["aegis_gpu_replica_tokens"])
        if "aegis_gpu_replica_dp_rank" in msg:
            stats.aegis_gpu_replica_dp_rank = int(msg["aegis_gpu_replica_dp_rank"])
        if "aegis_gpu_replica_quota_tokens" in msg:
            stats.aegis_gpu_replica_quota_tokens = int(
                msg["aegis_gpu_replica_quota_tokens"]
            )
        if "aegis_gpu_replica_overlap_tokens" in msg:
            stats.aegis_gpu_replica_overlap_tokens = int(
                msg["aegis_gpu_replica_overlap_tokens"]
            )
        if "aegis_gpu_replica_unique_tokens" in msg:
            stats.aegis_gpu_replica_unique_tokens = int(
                msg["aegis_gpu_replica_unique_tokens"]
            )
        if "aegis_gpu_replica_reserved_tokens" in msg:
            stats.aegis_gpu_replica_reserved_tokens = int(
                msg["aegis_gpu_replica_reserved_tokens"]
            )
        if "aegis_gpu_replica_evictable" in msg:
            stats.aegis_gpu_replica_evictable = bool(
                msg["aegis_gpu_replica_evictable"]
            )
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
        if "aegis_replay_capacity_limited_tokens" in msg:
            stats.aegis_replay_capacity_limited_tokens = int(
                msg["aegis_replay_capacity_limited_tokens"]
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
        if "aegis_replay_gpu_direct_reuse" in msg:
            stats.aegis_replay_gpu_direct_reuse = bool(
                msg["aegis_replay_gpu_direct_reuse"]
            )
        if "aegis_replay_host_restore_path" in msg:
            stats.aegis_replay_host_restore_path = str(
                msg["aegis_replay_host_restore_path"]
            )
        if "aegis_replay_l3_restore_path" in msg:
            stats.aegis_replay_l3_restore_path = str(
                msg["aegis_replay_l3_restore_path"]
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
        if "aegis_l3_backup_lookup_hit" in msg:
            stats.aegis_l3_backup_lookup_hit = bool(
                msg["aegis_l3_backup_lookup_hit"]
            )
        if "aegis_l3_backup_lookup_tokens" in msg:
            stats.aegis_l3_backup_lookup_tokens = int(
                msg["aegis_l3_backup_lookup_tokens"]
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
            dp_controller = getattr(self, "_dp_controller", None)
            if dp_controller is not None:
                observe = getattr(dp_controller, "observe_completed_request", None)
                if callable(observe):
                    observe(stats)
                released = dp_controller.release_request(rid)
                if released is not None:
                    stats.aegis_gpu_replica_dp_rank = released.backup_dp_rank
                    stats.aegis_gpu_replica_quota_tokens = released.quota_tokens
                    stats.aegis_gpu_replica_overlap_tokens = released.overlap_tokens
                    stats.aegis_gpu_replica_unique_tokens = released.unique_tokens
                    stats.aegis_gpu_replica_reserved_tokens = released.reserved_tokens
                    stats.aegis_gpu_replica_evictable = released.evictable
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
                int(stats.aegis_gpu_replica_dp_rank)
                if int(stats.aegis_gpu_replica_dp_rank) >= 0
                else (
                    (int(stats.aegis_assigned_dp_rank) + 1)
                    % max(int(self.dp_size), 1)
                )
            ),
            host_visible_tokens=int(stats.aegis_host_visible_tokens),
            gpu_replica_tokens=int(stats.aegis_gpu_replica_tokens),
            input_length=int(stats.input_length),
            generated_tokens=int(stats.generated_tokens),
            updated_time=time.time(),
            gpu_replica_quota_tokens=int(stats.aegis_gpu_replica_quota_tokens),
            gpu_replica_overlap_tokens=int(stats.aegis_gpu_replica_overlap_tokens),
            gpu_replica_unique_tokens=int(stats.aegis_gpu_replica_unique_tokens),
            gpu_replica_reserved_tokens=int(stats.aegis_gpu_replica_reserved_tokens),
            gpu_replica_evictable=bool(stats.aegis_gpu_replica_evictable),
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
        no_peer_locality = ablation in {
            "no_peer_locality",
            "disable_peer_locality",
        }

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
        elif no_peer_locality:
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
            detok = self._drain_detokenizer(max_messages=512)
            aegis = self._drain_aegis_rank_events(max_messages=512)
            if detok == 0 and aegis == 0:
                self._drain_stop.wait(timeout=Scheduler._IDLE_SLEEP_S)

    def _drain_aegis_rank_events(self, max_messages: int = 256) -> int:
        import queue as queue_module

        drained = 0
        for _ in range(max_messages):
            try:
                event = self._scheduler_stats_queue.get_nowait()
            except queue_module.Empty:
                break
            if isinstance(event, dict):
                if event.get("type") == "scheduler_telemetry":
                    row = event.get("row")
                    if isinstance(row, dict):
                        with self._scheduler_telemetry_lock:
                            self._scheduler_telemetry.append(dict(row))
                elif self._dp_controller is not None:
                    self._dp_controller.apply_aegis_rank_event(event)
            drained += 1
        return drained

    def calibration_telemetry(self) -> dict:
        """Snapshot optional online telemetry in the common gate schema."""

        if not self.server_args.scheduler_telemetry:
            raise RuntimeError(
                "scheduler telemetry is disabled; launch with "
                "--scheduler-telemetry"
            )
        self._drain_detokenizer(max_messages=4096)
        self._drain_aegis_rank_events(max_messages=1_000_000)
        with self._scheduler_telemetry_lock:
            scheduler_rows = [dict(row) for row in self._scheduler_telemetry]
        request_stats = {
            rid: stats.to_sim_stats_dict()
            for rid, stats in REQUEST_STATS.items()
        }
        aegis = self.aegis_status(after_sequence=0)
        from sglsim.srt.simulation.calibration_metrics import (
            build_common_calibration_metrics,
        )

        payload = {
            "schema_version": 1,
            "simulation_time_mode": "wall",
            "evidence_tier": "sglsim-online-instrumented",
            "identity": {
                "model_path": self.server_args.model_path,
                "dp_size": self.dp_size,
                "tp_size": self.server_args.tp_size,
                "pp_size": self.server_args.pp_size,
                "sim_config_path": self.server_args.sim_config_path,
            },
            "physical_topology": self._physical_scheduler_topology(),
            "calibration_metrics": build_common_calibration_metrics(
                request_stats=request_stats,
                scheduler_telemetry=scheduler_rows,
                failure_events=self._calibration_failure_events,
                aegis_events=aegis.get("events", ()),
            ),
            "request_stats": request_stats,
            "scheduler_telemetry": scheduler_rows,
            "failure_events": [
                dict(event) for event in self._calibration_failure_events
            ],
            "aegis_status": aegis,
        }
        replaced_paths: list[str] = []
        strict_payload = _strict_json_telemetry(
            payload,
            replaced_paths=replaced_paths,
        )
        if not isinstance(strict_payload, dict):
            raise TypeError("calibration telemetry projection must be a dict")
        strict_payload["serialization_audit"] = {
            "strict_json": True,
            "nonfinite_values_replaced_with_null": len(replaced_paths),
            "replacement_paths": replaced_paths,
        }
        return strict_payload

    def _physical_scheduler_topology(self) -> dict:
        """Return the live one-process-per-stage Scheduler topology.

        This is calibration evidence, not a second launch abstraction.  The
        process handles are the exact handles created by ``launch_subprocesses``
        and consumed by the normal online shutdown/failure paths.
        """

        group = self._pipeline_process_group
        if group is not None:
            workers = tuple(group.workers)
            stage_processes = [
                {
                    "stage_id": (
                        f"dp{worker.spec.stage.dp_rank}."
                        f"pp{worker.spec.stage.pp_rank}."
                        f"tp{worker.spec.stage.tp_rank}"
                    ),
                    "dp_rank": int(worker.spec.stage.dp_rank),
                    "pp_rank": int(worker.spec.stage.pp_rank),
                    "tp_rank": int(worker.spec.stage.tp_rank),
                    "pid": worker.process.pid,
                    "is_alive": bool(worker.process.is_alive()),
                    "exitcode": worker.process.exitcode,
                }
                for worker in workers
            ]
        else:
            stage_processes = [
                {
                    "stage_id": f"dp{dp_rank}.pp0.tp0",
                    "dp_rank": int(dp_rank),
                    "pp_rank": 0,
                    "tp_rank": 0,
                    "pid": process.pid,
                    "is_alive": bool(process.is_alive()),
                    "exitcode": process.exitcode,
                }
                for dp_rank, process in enumerate(self._scheduler_procs)
            ]
        expected = (
            int(self.dp_size)
            * int(self.server_args.pp_size)
            * int(self.server_args.tp_size)
        )
        return {
            "topology_contract": "one_scheduler_process_per_dp_pp_tp_stage",
            "dp_size": int(self.dp_size),
            "pp_size": int(self.server_args.pp_size),
            "tp_size": int(self.server_args.tp_size),
            "expected_process_count": expected,
            "process_count": len(stage_processes),
            "alive_process_count": sum(
                int(row["is_alive"]) for row in stage_processes
            ),
            "stage_count": len(stage_processes),
            "stage_ids": [row["stage_id"] for row in stage_processes],
            "alive_stage_ids": [
                row["stage_id"] for row in stage_processes if row["is_alive"]
            ],
            "stage_processes": stage_processes,
        }

    def aegis_status(self, *, after_sequence: int = 0) -> dict:
        self._drain_aegis_rank_events(max_messages=4096)
        if self._dp_controller is None:
            return {
                "policy": "aegiskv_l1_l2",
                "dp_size": self.dp_size,
                "latest_sequence": 0,
                "router_load": [],
                "rank_states": {},
                "recovery_hints": [],
                "events": [],
            }
        return self._dp_controller.aegis_status(after_sequence=after_sequence)

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
        if self._pipeline_process_group is not None:
            dead_stages = [
                worker.spec.stage
                for worker in self._pipeline_process_group.workers
                if worker.process.exitcode is not None
            ]
            if dead_stages:
                raise SchedulerWorkerDiedError(
                    f"PP scheduler worker(s) died (stages={dead_stages})"
                )
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
        if self._dp_controller is not None:
            self._dp_controller.close_event_journal()
        if self._pipeline_process_group is not None:
            self._pipeline_process_group.broadcast_command(("stop",))
            scheduler_procs = self._pipeline_process_group.processes
        else:
            for cmd_queue in self._cmd_queues:
                try:
                    cmd_queue.put(("stop",))
                except Exception:
                    pass
            scheduler_procs = tuple(self._scheduler_procs)
        for proc in scheduler_procs:
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
