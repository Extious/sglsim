"""Event-driven scheduler subprocess used by the offline parent DES."""

from __future__ import annotations

import os
import queue
from collections import deque
from dataclasses import replace
from typing import Any

from sglsim.srt.managers.scheduler import REQUEST_STATS, Scheduler
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.state import StateManager
from sglsim.srt.simulation.time_driver import LogicalClockDriver
from sglsim.srt.utils.network import SimIpcRegistry


class _CapturedDetokenizerSender:
    """Synchronous offline-only replacement for the scheduler PUSH socket.

    ``multiprocessing.Queue.put`` returns before its feeder thread publishes
    the object.  A parent DES must observe all messages caused by one rank
    event before it pops the next event, so the worker returns this outbox as
    part of the command response instead of racing a second IPC channel.
    """

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    def send_pyobj(self, message: Any) -> None:
        if not isinstance(message, dict):
            raise TypeError("offline detokenizer message must be a dict")
        self._messages.append(dict(message))

    def drain(self) -> list[dict[str, Any]]:
        messages = self._messages
        self._messages = []
        return messages


class _CapturedRequestReceiver:
    """Synchronous offline-only replacement for the scheduler PULL socket.

    The online controller-to-scheduler path intentionally uses an IPC queue.
    ``multiprocessing.Queue.put`` may return before its feeder publishes the
    request, which is harmless under a polling wall-clock loop but can let a
    parent DES pop an unrelated future event first.  The offline control reply
    therefore carries requests selected by the real controller directly into
    this rank-local inbox; :meth:`Scheduler.recv_requests` still consumes them
    at the pinned SGLang control-flow location.
    """

    def __init__(self) -> None:
        self._messages: deque[Any] = deque()

    def extend(self, messages: list[Any]) -> None:
        self._messages.extend(messages)

    def recv_pyobj(self, *, flags: int = 0) -> Any:
        del flags
        if not self._messages:
            raise queue.Empty
        return self._messages.popleft()


class _CapturedCommandReceiver:
    """Rank-local inbox for parent-ordered runtime control commands."""

    def __init__(self) -> None:
        self._messages: deque[Any] = deque()

    def extend(self, messages: list[Any]) -> None:
        self._messages.extend(messages)

    def get_nowait(self) -> Any:
        if not self._messages:
            raise queue.Empty
        return self._messages.popleft()


class _CapturedEventQueue:
    """Return ACK/control events in the same response as the causal step."""

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    def put(self, message: Any) -> None:
        if not isinstance(message, dict):
            raise TypeError("offline Aegis event must be a dict")
        self._messages.append(dict(message))

    def drain(self) -> list[dict[str, Any]]:
        messages = self._messages
        self._messages = []
        return messages


class _CapturedObjectSender:
    """Synchronous outbox for immutable PP request/proxy/output envelopes."""

    def __init__(self) -> None:
        self._messages: list[Any] = []

    def send_pyobj(self, message: Any) -> None:
        self._messages.append(message)

    def drain(self) -> list[Any]:
        messages = self._messages
        self._messages = []
        return messages


def _drain_runtime_commands(scheduler: Scheduler, cmd_queue) -> bool:
    """Apply shared online runtime commands without wall-clock polling."""

    should_stop = False
    while True:
        try:
            message = cmd_queue.get_nowait()
        except queue.Empty:
            break
        if scheduler._handle_runtime_command(message):
            should_stop = True
            break
    scheduler._maybe_clear_pending_gpu_l1()
    return should_stop


def run_offline_scheduler_worker(
    server_args: ServerArgs,
    port_args: PortArgs,
    dp_rank: int,
    pp_rank: int,
    tp_rank: int,
    ipc_registry: SimIpcRegistry,
    runtime_cmd_queue,
    control_queue,
    result_queue,
    stats_queue,
    ready_event,
    enable_hicache_global_des: bool = False,
    worker_incarnation: int = 0,
) -> None:
    """Own one physical Scheduler stage and one independent logical clock."""

    if server_args.sim_config_path:
        os.environ["SGLSIM_CONFIG_PATH"] = server_args.sim_config_path
    SimIpcRegistry.install_global(ipc_registry)
    ConfigManager.load(server_args.sim_config_path)
    ConfigManager.apply_server_args_overrides(server_args)
    StateManager.reset()
    REQUEST_STATS.clear()
    driver = LogicalClockDriver()
    scheduler = Scheduler(
        server_args,
        port_args,
        dp_rank=dp_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        time_driver=driver,
    )
    if enable_hicache_global_des:
        scheduler.cache.enable_offline_hicache_des(
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            scheduler_tp_rank=tp_rank,
            kv_bytes_per_token_per_tp_rank=(
                ConfigManager.get_kv_cache_bytes_per_token(
                    server_args.model_path
                )
            ),
            worker_incarnation=worker_incarnation,
        )
    detokenizer_outbox = _CapturedDetokenizerSender()
    request_inbox = _CapturedRequestReceiver()
    runtime_command_inbox = _CapturedCommandReceiver()
    aegis_event_outbox = _CapturedEventQueue()
    scheduler.recv_from_tokenizer = request_inbox
    scheduler.send_to_detokenizer = detokenizer_outbox
    scheduler._aegis_event_queue = aegis_event_outbox
    pp_request_outbox = _CapturedObjectSender()
    pp_proxy_outbox = _CapturedObjectSender()
    pp_output_outbox = _CapturedObjectSender()
    pp_proxy_inbox = _CapturedRequestReceiver()
    pp_output_inbox = _CapturedRequestReceiver()
    if int(server_args.pp_size) > 1:
        scheduler.init_pp_loop_state()
        if scheduler._pp_send_request is not None:
            scheduler._pp_send_request = pp_request_outbox
        if scheduler._pp_recv_proxy is not None:
            scheduler._pp_recv_proxy = pp_proxy_inbox
        if scheduler._pp_send_proxy is not None:
            scheduler._pp_send_proxy = pp_proxy_outbox
        scheduler._pp_recv_output = pp_output_inbox
        scheduler._pp_send_output = pp_output_outbox
    pending_step = None
    pending_kv_before_prepare: dict[str, Any] | None = None
    pending_kv_after_prepare: dict[str, Any] | None = None
    ready_event.set()

    def _inject_payload(payload: dict[str, Any], *, allow_work: bool) -> None:
        incoming_requests = list(payload.get("incoming_requests", ()))
        pp_requests = list(payload.get("pp_request_messages", ()))
        pp_proxies = list(payload.get("pp_proxy_messages", ()))
        pp_outputs = list(payload.get("pp_output_messages", ()))
        if incoming_requests and pp_rank != 0:
            raise ValueError("external requests may enter only PP0")
        if pp_requests and pp_rank == 0:
            raise ValueError("PP0 cannot receive a forwarded PP request envelope")
        if not allow_work and (incoming_requests or pp_requests or pp_proxies):
            raise ValueError("work dependencies are valid only for prepare")
        request_inbox.extend(incoming_requests)
        request_inbox.extend(pp_requests)
        pp_proxy_inbox.extend(pp_proxies)
        pp_output_inbox.extend(pp_outputs)
        runtime_command_inbox.extend(
            list(payload.get("runtime_commands", ()))
        )
        hicache_acks = list(payload.get("hicache_acks", ()))
        if hicache_acks:
            if not enable_hicache_global_des:
                raise RuntimeError(
                    "ordinary HiCache ACK received without offline GlobalDES adapter"
                )
            scheduler.cache.apply_offline_hicache_acks(hicache_acks)

    def _response_fields() -> dict[str, Any]:
        detokenizer_messages = detokenizer_outbox.drain()
        terminal_request_stats: dict[str, dict[str, Any]] = {}
        for message in detokenizer_messages:
            rid = str(message.get("rid") or "")
            stats = REQUEST_STATS.get(rid)
            # The Scheduler emits an offline ``finalize`` envelope with
            # ``finished=False`` and DetokenizerManager turns that envelope
            # into the client-visible finished message in the parent.  The
            # process-local stats are already terminal at this boundary.
            if stats is not None and stats.status in {"completed", "aborted"}:
                terminal_request_stats[rid] = stats.to_sim_stats_dict()
        return {
            "dp_rank": dp_rank,
            "pp_rank": pp_rank,
            "tp_rank": tp_rank,
            "aegis_events": aegis_event_outbox.drain(),
            "detokenizer_messages": detokenizer_messages,
            # A hard restart intentionally clears process-local REQUEST_STATS.
            # Return terminal rows in the same causal response as the final
            # detokenizer message so the parent can preserve them across that
            # restart without racing another IPC queue.
            "terminal_request_stats": terminal_request_stats,
            "pp_request_messages": pp_request_outbox.drain(),
            "pp_proxy_messages": pp_proxy_outbox.drain(),
            "pp_output_messages": pp_output_outbox.drain(),
            "hicache_intents": (
                scheduler.cache.drain_offline_hicache_intents()
                if enable_hicache_global_des
                else []
            ),
        }

    while True:
        command = control_queue.get()
        if not isinstance(command, tuple) or len(command) != 3:
            result_queue.put(
                {
                    "dp_rank": dp_rank,
                    "pp_rank": pp_rank,
                    "tp_rank": tp_rank,
                    "request_id": -1,
                    "type": "error",
                    "error": f"invalid offline worker command: {command!r}",
                }
            )
            continue
        request_id, op, payload = command
        payload = payload if isinstance(payload, dict) else {}
        try:
            if op == "stop":
                result_queue.put(
                    {
                        **_response_fields(),
                        "request_id": request_id,
                        "type": "stopped",
                        "kv_resource_snapshot": (
                            scheduler.offline_kv_resource_snapshot()
                        ),
                    }
                )
                return

            if op == "prepare":
                if pending_step is not None:
                    raise RuntimeError("stage already has a prepared step")
                at_s = float(payload.get("at_s", driver.now()))
                if at_s > driver.now():
                    driver.set_time(at_s)
                kv_before_prepare = scheduler.offline_kv_resource_snapshot()
                _inject_payload(payload, allow_work=True)
                if _drain_runtime_commands(scheduler, runtime_command_inbox):
                    raise RuntimeError("runtime command requested worker stop")
                pending_step = (
                    scheduler.prepare_offline_pp_step()
                    if int(server_args.pp_size) > 1
                    else scheduler.prepare_offline_step()
                )
                kv_after_prepare = scheduler.offline_kv_resource_snapshot()
                if pending_step is not None:
                    pending_kv_before_prepare = kv_before_prepare
                    pending_kv_after_prepare = kv_after_prepare
                    telemetry = pending_step.telemetry()
                else:
                    pending_kv_before_prepare = None
                    pending_kv_after_prepare = None
                    telemetry = {
                        "dp_rank": dp_rank,
                        "pp_rank": pp_rank,
                        "tp_rank": tp_rank,
                        "batch_mode": "idle",
                        "waiting_depth": len(scheduler.waiting_queue),
                        "running_depth": len(scheduler.running_batch.reqs),
                    }
                telemetry["kv_resource_before_prepare"] = kv_before_prepare
                telemetry["kv_resource_after_prepare"] = kv_after_prepare
                result_queue.put(
                    {
                        **_response_fields(),
                        "request_id": request_id,
                        "type": "prepared",
                        "logical_now_s": driver.now(),
                        "idle": pending_step is None,
                        "blocked_on_slot": bool(
                            pending_step is None
                            and int(server_args.pp_size) > 1
                            and not scheduler._rank_idle()
                        ),
                        "rank_idle": scheduler._rank_idle(),
                        "duration_s": (
                            pending_step.duration_s if pending_step is not None else 0.0
                        ),
                        "kv_resource_before_prepare": kv_before_prepare,
                        "kv_resource_after_prepare": kv_after_prepare,
                        "kv_resource_snapshot": kv_after_prepare,
                        "telemetry": telemetry,
                    }
                )
                continue

            if op == "complete":
                if pending_step is None:
                    raise RuntimeError("rank has no prepared step")
                completed_at = float(payload["completed_at"])
                dependency_delay_s = float(
                    payload.get("hicache_dependency_delay_s", 0.0)
                )
                if dependency_delay_s < 0.0:
                    raise ValueError(
                        "ordinary HiCache dependency delay must be nonnegative"
                    )
                if dependency_delay_s and pending_step.pipeline_plan is not None:
                    # The plan was constructed at prepare time, before the
                    # parent knew when the all-TP H2D/D2H dependency would ACK.
                    # Shift that immutable plan by exactly the external wait;
                    # its compute duration/accounting remains unchanged.
                    pending_step = replace(
                        pending_step,
                        pipeline_plan=scheduler._shift_offline_plan(
                            pending_step.pipeline_plan,
                            dependency_delay_s,
                        ),
                    )
                if completed_at > driver.now():
                    driver.set_time(completed_at)
                _inject_payload(payload, allow_work=False)
                telemetry = (
                    scheduler.complete_offline_pp_step(pending_step)
                    if int(server_args.pp_size) > 1
                    else scheduler.complete_offline_step(pending_step)
                )
                pending_step = None
                _drain_runtime_commands(scheduler, runtime_command_inbox)
                if int(server_args.pp_size) > 1:
                    scheduler._pp_pump_outputs(
                        cmd_queue=None,
                        stop_event=None,
                    )
                kv_at_completion = scheduler.offline_kv_resource_snapshot()
                telemetry["kv_resource_before_prepare"] = (
                    pending_kv_before_prepare
                )
                telemetry["kv_resource_after_prepare"] = (
                    pending_kv_after_prepare
                )
                telemetry["kv_resource_at_completion"] = kv_at_completion
                pending_kv_before_prepare = None
                pending_kv_after_prepare = None
                result_queue.put(
                    {
                        **_response_fields(),
                        "request_id": request_id,
                        "type": "completed",
                        "logical_now_s": driver.now(),
                        "rank_idle": scheduler._rank_idle(),
                        "all_done": scheduler._all_done(),
                        "kv_resource_at_completion": kv_at_completion,
                        "kv_resource_snapshot": kv_at_completion,
                        "telemetry": telemetry,
                    }
                )
                continue

            if op == "poll":
                if pending_step is not None:
                    raise RuntimeError("cannot poll an in-flight stage step")
                at_s = float(payload.get("at_s", driver.now()))
                if at_s > driver.now():
                    driver.set_time(at_s)
                _inject_payload(payload, allow_work=False)
                if _drain_runtime_commands(scheduler, runtime_command_inbox):
                    raise RuntimeError("runtime command requested worker stop")
                if int(server_args.pp_size) > 1:
                    scheduler._pp_pump_outputs(
                        cmd_queue=None,
                        stop_event=None,
                    )
                    scheduler._maybe_clear_pending_gpu_l1()
                kv_snapshot = scheduler.offline_kv_resource_snapshot()
                result_queue.put(
                    {
                        **_response_fields(),
                        "request_id": request_id,
                        "type": "polled",
                        "logical_now_s": driver.now(),
                        "rank_idle": scheduler._rank_idle(),
                        "kv_resource_snapshot": kv_snapshot,
                    }
                )
                continue

            if op == "snapshot":
                if pending_step is not None:
                    raise RuntimeError("cannot snapshot an in-flight stage step")
                _inject_payload(payload, allow_work=False)
                _drain_runtime_commands(scheduler, runtime_command_inbox)
                if int(server_args.pp_size) > 1:
                    scheduler._pp_pump_outputs(
                        cmd_queue=None,
                        stop_event=None,
                    )
                kv_snapshot = scheduler.offline_kv_resource_snapshot()
                result_queue.put(
                    {
                        **_response_fields(),
                        "request_id": request_id,
                        "type": "snapshot",
                        "logical_now_s": driver.now(),
                        "rank_idle": scheduler._rank_idle(),
                        "kv_resource_snapshot": kv_snapshot,
                        "request_stats": {
                            rid: stats.to_sim_stats_dict()
                            for rid, stats in REQUEST_STATS.items()
                        },
                        "comm_totals": StateManager.get_comm_totals(),
                        "aegis_rank_state": scheduler.aegis_backup_manager.snapshot(),
                        "profile_totals": dict(scheduler._profile_totals),
                        "hicache_intent_audit": (
                            scheduler.cache.offline_hicache_intent_audit()
                            if enable_hicache_global_des
                            else []
                        ),
                        "pp_state": {
                            "step_sequence": int(
                                getattr(scheduler, "_pp_step_sequence", 0)
                            ),
                            "pending_sequences": sorted(
                                int(value)
                                for value in getattr(
                                    scheduler,
                                    "_pp_pending_batches",
                                    {},
                                )
                            ),
                            "last_request_sequence": int(
                                getattr(
                                    scheduler,
                                    "_pp_last_request_sequence",
                                    -1,
                                )
                            ),
                            "last_proxy_sequence": int(
                                getattr(
                                    scheduler,
                                    "_pp_last_proxy_sequence",
                                    -1,
                                )
                            ),
                            "last_output_sequence": int(
                                getattr(
                                    scheduler,
                                    "_pp_last_output_sequence",
                                    -1,
                                )
                            ),
                        },
                    }
                )
                continue

            raise ValueError(f"unknown offline worker operation: {op}")
        except BaseException as exc:
            result_queue.put(
                {
                    "dp_rank": dp_rank,
                    "pp_rank": pp_rank,
                    "tp_rank": tp_rank,
                    "request_id": request_id,
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
