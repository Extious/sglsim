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
from sglsim.srt.simulation.types import RequestStats, SimulationMode
from sglsim.srt.utils.network import SimIpcRegistry, build_dp_ipc_registry
from sglsim.srt.utils.request_validation import validate_generate_request


@dataclass
class EngineResult:
    metrics: dict
    request_stats: dict


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
        REQUEST_STATS.clear()

    def is_ready(self) -> bool:
        return self._ready

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
        if "agent_id" in msg:
            stats.agent_id = str(msg["agent_id"])
        if "critical_agent" in msg:
            stats.critical_agent = bool(msg["critical_agent"])
        if "text" in msg:
            self._completion_text[rid] = str(msg["text"])
        if msg.get("finished"):
            if stats.status != "aborted":
                stats.status = "completed"
            with self._completion_lock:
                event = self._completion_events.get(rid)
                if event is not None:
                    event.set()

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
        self.tokenizer_manager.generate_request(req)

    def generate_from_http(self, req: GenerateReqInput) -> tuple[RequestStats, str]:
        validate_generate_request(
            req,
            ConfigManager.get_model_info(self.server_args.model_path),
        )
        self._register_rid(req.rid)
        self._stamp_created_time(req)
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
