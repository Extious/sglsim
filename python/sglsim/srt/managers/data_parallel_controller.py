"""Data-parallel request dispatch aligned with sglang.srt.managers.data_parallel_controller."""

from __future__ import annotations

import logging
import multiprocessing as mp
import zlib
from enum import Enum, auto

from sglsim.srt.managers.io_struct import (
    SimulateGpuFailureReqInput,
    SimulateGpuRecoveryReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.utils.network import SimZmqContext, SimZmqSocket, get_zmq_socket

logger = logging.getLogger(__name__)


def routing_key_to_dp_rank(routing_key: str, dp_size: int) -> int:
    if dp_size <= 1:
        return 0
    key_bytes = routing_key.encode("utf-8")
    return zlib.crc32(key_bytes) % dp_size


class LoadBalanceMethod(Enum):
    ROUND_ROBIN = auto()

    @classmethod
    def from_str(cls, method: str) -> LoadBalanceMethod:
        name = method.upper()
        if name in ("AUTO", "ROUND_ROBIN"):
            return cls.ROUND_ROBIN
        raise ValueError(f"Unsupported load_balance_method: {method}")


class SchedulerWorkerDiedError(RuntimeError):
    """Raised when a DP scheduler subprocess has exited unexpectedly."""


class DataParallelController:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        *,
        scheduler_procs: list[mp.Process] | None = None,
    ) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self._scheduler_procs = scheduler_procs
        sched = ConfigManager.get_scheduler_config()
        self.load_balance_method = LoadBalanceMethod.from_str(
            sched.load_balance_method
        )
        self.enable_routing_key_affinity = sched.enable_routing_key_affinity
        self.round_robin_counter = 0
        self.routing_key_to_rank: dict[str, int] = {}
        ctx = SimZmqContext()
        self.recv_from_tokenizer = get_zmq_socket(
            ctx,
            SimZmqSocket.PULL,
            port_args.scheduler_input_ipc_name,
            bind=False,
        )
        self.workers = [
            get_zmq_socket(
                ctx,
                SimZmqSocket.PUSH,
                port_args.scheduler_input_ipc_name_for_dp(dp_rank),
                bind=True,
            )
            for dp_rank in range(sched.dp_size)
        ]
        self.status = [True] * len(self.workers)

    def refresh_worker_status(self) -> list[int]:
        """Mark workers dead when their scheduler subprocess has exited."""
        if not self._scheduler_procs:
            return []
        dead: list[int] = []
        for dp_rank, proc in enumerate(self._scheduler_procs):
            if proc.exitcode is None:
                continue
            if self.status[dp_rank]:
                logger.error(
                    "DP scheduler worker %d exited with code %s",
                    dp_rank,
                    proc.exitcode,
                )
            self.status[dp_rank] = False
            dead.append(dp_rank)
        return dead

    def ensure_workers_alive(self) -> None:
        dead = self.refresh_worker_status()
        if dead:
            raise SchedulerWorkerDiedError(
                f"DP scheduler worker(s) died (ranks={dead})"
            )

    def _record_assigned_rank(self, req: TokenizedGenerateReqInput, dp_rank: int) -> None:
        sim = req.sampling_params.custom_params.setdefault("simulation", {})
        sim["assigned_dp_rank"] = dp_rank

    def _send_to_worker(self, req: TokenizedGenerateReqInput, dp_rank: int) -> bool:
        if not self.status[dp_rank]:
            return False
        self._record_assigned_rank(req, dp_rank)
        self.workers[dp_rank].send_pyobj(req)
        return True

    def handle_simulate_gpu_failure(self, req: SimulateGpuFailureReqInput) -> None:
        dp_rank = int(req.dp_rank)
        if dp_rank < 0 or dp_rank >= len(self.workers):
            raise ValueError(f"Invalid dp_rank={dp_rank}")
        self.workers[dp_rank].send_pyobj(req)
        self.status[dp_rank] = False
        stale_keys = [
            key for key, rank in self.routing_key_to_rank.items() if rank == dp_rank
        ]
        for key in stale_keys:
            self.routing_key_to_rank.pop(key, None)
        logger.warning("Simulated GPU failure on dp_rank=%d", dp_rank)

    def handle_simulate_gpu_recovery(self, req: SimulateGpuRecoveryReqInput) -> None:
        dp_rank = int(req.dp_rank)
        if dp_rank < 0 or dp_rank >= len(self.workers):
            raise ValueError(f"Invalid dp_rank={dp_rank}")
        self.workers[dp_rank].send_pyobj(req)
        self.status[dp_rank] = True
        logger.warning("Simulated GPU recovery on dp_rank=%d", dp_rank)

    def round_robin_scheduler(self, req: TokenizedGenerateReqInput) -> int | None:
        for _ in range(len(self.workers)):
            if self.status[self.round_robin_counter]:
                dp_rank = self.round_robin_counter
                self._send_to_worker(req, dp_rank)
                self.round_robin_counter = (self.round_robin_counter + 1) % len(
                    self.workers
                )
                return dp_rank
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
        rid = getattr(req, "rid", "?")
        logger.error("No healthy DP worker for request %s", rid)
        self.ensure_workers_alive()
        raise SchedulerWorkerDiedError(
            f"No healthy DP worker for request {rid}"
        )

    def routing_key_scheduler(self, req: TokenizedGenerateReqInput) -> int | None:
        if not req.routing_key:
            return self.round_robin_scheduler(req)

        mapped_rank = self.routing_key_to_rank.get(req.routing_key)
        if mapped_rank is not None and self.status[mapped_rank]:
            self._send_to_worker(req, mapped_rank)
            return mapped_rank

        for _ in range(len(self.workers)):
            rank = self.round_robin_counter
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
            if self.status[rank]:
                self.routing_key_to_rank[req.routing_key] = rank
                self._send_to_worker(req, rank)
                return rank

        rid = getattr(req, "rid", "?")
        logger.error("No healthy DP worker for routing-key request %s", rid)
        self.ensure_workers_alive()
        raise SchedulerWorkerDiedError(
            f"No healthy DP worker for request {rid}"
        )

    def dispatch(self, req: TokenizedGenerateReqInput) -> int | None:
        self.ensure_workers_alive()
        if self.load_balance_method == LoadBalanceMethod.ROUND_ROBIN:
            # Mirror sglang round_robin_scheduler: when routing-key affinity is
            # enabled, sticky round-robin assigns the first-seen key to the next
            # healthy DP rank and keeps subsequent requests on that rank.
            if self.enable_routing_key_affinity:
                return self.routing_key_scheduler(req)
            return self.round_robin_scheduler(req)
        return None


def launch_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    dp_size: int,
) -> tuple[list[mp.Process], list[mp.Queue], mp.Queue, list[mp.Event]]:
    from sglsim.srt.managers.scheduler import run_scheduler_process

    processes: list[mp.Process] = []
    cmd_queues: list[mp.Queue] = []
    ready_events: list[mp.Event] = []
    stats_queue: mp.Queue = mp.Queue()

    for dp_rank in range(dp_size):
        cmd_queue: mp.Queue = mp.Queue()
        ready_event = mp.Event()
        proc = mp.Process(
            target=run_scheduler_process,
            args=(
                server_args,
                port_args,
                dp_rank,
                ipc_registry,
                cmd_queue,
                stats_queue,
                ready_event,
            ),
            daemon=True,
        )
        processes.append(proc)
        cmd_queues.append(cmd_queue)
        ready_events.append(ready_event)
        proc.start()

    for event in ready_events:
        if not event.wait(timeout=60):
            raise RuntimeError("DP scheduler subprocess failed to become ready")

    return processes, cmd_queues, stats_queue, ready_events


def launch_dp_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    dp_size: int,
) -> tuple[list[mp.Process], list[mp.Queue], mp.Queue, list[mp.Event]]:
    return launch_scheduler_processes(
        server_args, port_args, ipc_registry, dp_size
    )
