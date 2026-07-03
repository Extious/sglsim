"""Data-parallel request dispatch aligned with sglang.srt.managers.data_parallel_controller."""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import zlib
from enum import Enum, auto

from sglsim.srt.managers.io_struct import TokenizedGenerateReqInput
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
        self.soft_failed_ranks: set[int] = set()
        self._soft_failure_ref_counts: dict[int, int] = {}
        self._soft_failure_lock = threading.Lock()
        self.hard_failed_ranks: set[int] = set()
        self._hard_failure_lock = threading.Lock()

    def _validate_rank(self, dp_rank: int) -> None:
        if dp_rank < 0 or dp_rank >= len(self.workers):
            raise ValueError(f"invalid DP rank {dp_rank}")

    def soft_fail_rank(self, dp_rank: int) -> bool:
        self._validate_rank(dp_rank)
        with self._hard_failure_lock:
            if dp_rank in self.hard_failed_ranks:
                logger.info(
                    "Ignored soft-fail request for hard-failed DP rank %d",
                    dp_rank,
                )
                return False
        with self._soft_failure_lock:
            active = sorted(self.soft_failed_ranks)
            if active and dp_rank not in self.soft_failed_ranks:
                logger.info(
                    "Ignored soft-fail request for DP rank %d; active soft-failed ranks=%s",
                    dp_rank,
                    active,
                )
                return False
            self._soft_failure_ref_counts[dp_rank] = (
                self._soft_failure_ref_counts.get(dp_rank, 0) + 1
            )
            self.soft_failed_ranks.add(dp_rank)
        logger.warning("Soft-failed DP rank %d for simulated failover", dp_rank)
        return True

    def recover_soft_failed_rank(self, dp_rank: int) -> bool:
        self._validate_rank(dp_rank)
        with self._soft_failure_lock:
            count = self._soft_failure_ref_counts.get(dp_rank, 0)
            if count <= 0:
                logger.info(
                    "Ignored recover request for non-soft-failed DP rank %d",
                    dp_rank,
                )
                return False
            if count == 1:
                self._soft_failure_ref_counts.pop(dp_rank, None)
                self.soft_failed_ranks.discard(dp_rank)
            else:
                self._soft_failure_ref_counts[dp_rank] = count - 1
        logger.info("Recovered soft-failed DP rank %d", dp_rank)
        return True

    def mark_rank_hard_failed(self, dp_rank: int) -> bool:
        self._validate_rank(dp_rank)
        with self._hard_failure_lock:
            already_failed = dp_rank in self.hard_failed_ranks
            self.hard_failed_ranks.add(dp_rank)
            self.status[dp_rank] = False
        with self._soft_failure_lock:
            self.soft_failed_ranks.discard(dp_rank)
            self._soft_failure_ref_counts.pop(dp_rank, None)
        if not already_failed:
            logger.warning("Hard-failed DP rank %d for simulated failover", dp_rank)
        return not already_failed

    def mark_rank_restarted(self, dp_rank: int) -> bool:
        self._validate_rank(dp_rank)
        with self._hard_failure_lock:
            was_failed = dp_rank in self.hard_failed_ranks
            self.hard_failed_ranks.discard(dp_rank)
            self.status[dp_rank] = True
        logger.info("Restarted hard-failed DP rank %d", dp_rank)
        return was_failed

    def soft_failed(self) -> list[int]:
        with self._soft_failure_lock:
            return sorted(self.soft_failed_ranks)

    def hard_failed(self) -> list[int]:
        with self._hard_failure_lock:
            return sorted(self.hard_failed_ranks)

    def _worker_available(self, dp_rank: int) -> bool:
        with self._soft_failure_lock:
            soft_failed = dp_rank in self.soft_failed_ranks
        with self._hard_failure_lock:
            hard_failed = dp_rank in self.hard_failed_ranks
        return self.status[dp_rank] and not soft_failed and not hard_failed

    def refresh_worker_status(self) -> list[int]:
        """Mark workers dead when their scheduler subprocess has exited."""
        if not self._scheduler_procs:
            return []
        dead: list[int] = []
        for dp_rank, proc in enumerate(self._scheduler_procs):
            if proc.exitcode is None:
                continue
            with self._hard_failure_lock:
                expected_dead = dp_rank in self.hard_failed_ranks
            if expected_dead:
                self.status[dp_rank] = False
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
        if not self._worker_available(dp_rank):
            return False
        self._record_assigned_rank(req, dp_rank)
        self.workers[dp_rank].send_pyobj(req)
        return True

    def round_robin_scheduler(self, req: TokenizedGenerateReqInput) -> int | None:
        for _ in range(len(self.workers)):
            if self._worker_available(self.round_robin_counter):
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
        if mapped_rank is not None and self._worker_available(mapped_rank):
            self._send_to_worker(req, mapped_rank)
            return mapped_rank

        for _ in range(len(self.workers)):
            rank = self.round_robin_counter
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )
            if self._worker_available(rank):
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
            sim = req.sampling_params.custom_params.get("simulation", {})
            force_dp_rank = sim.get("force_dp_rank")
            if force_dp_rank is not None:
                try:
                    dp_rank = int(force_dp_rank)
                except (TypeError, ValueError):
                    dp_rank = -1
                if 0 <= dp_rank < len(self.workers) and self._worker_available(dp_rank):
                    self._send_to_worker(req, dp_rank)
                    return dp_rank
                logger.warning(
                    "Ignoring invalid forced DP rank %s for request %s",
                    force_dp_rank,
                    getattr(req, "rid", "?"),
                )
            # Mirror sglang round_robin_scheduler: when routing-key affinity is
            # enabled, sticky round-robin assigns the first-seen key to the next
            # healthy DP rank and keeps subsequent requests on that rank.
            if self.enable_routing_key_affinity:
                return self.routing_key_scheduler(req)
            return self.round_robin_scheduler(req)
        return None


def launch_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    dp_rank: int,
    *,
    stats_queue: mp.Queue | None = None,
) -> tuple[mp.Process, mp.Queue, mp.Event]:
    from sglsim.srt.managers.scheduler import run_scheduler_process

    cmd_queue: mp.Queue = mp.Queue()
    ready_event = mp.Event()
    if stats_queue is None:
        stats_queue = mp.Queue()
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
    proc.start()
    return proc, cmd_queue, ready_event


def launch_scheduler_processes(
    server_args: ServerArgs,
    port_args: PortArgs,
    ipc_registry,
    dp_size: int,
) -> tuple[list[mp.Process], list[mp.Queue], mp.Queue, list[mp.Event]]:
    processes: list[mp.Process] = []
    cmd_queues: list[mp.Queue] = []
    ready_events: list[mp.Event] = []
    stats_queue: mp.Queue = mp.Queue()

    for dp_rank in range(dp_size):
        proc, cmd_queue, ready_event = launch_scheduler_process(
            server_args,
            port_args,
            ipc_registry,
            dp_rank,
            stats_queue=stats_queue,
        )
        processes.append(proc)
        cmd_queues.append(cmd_queue)
        ready_events.append(ready_event)

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
