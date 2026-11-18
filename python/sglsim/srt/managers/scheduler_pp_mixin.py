"""Pipeline-parallel Scheduler event loop aligned with pinned SGLang.

The ordering mirrors ``sglang.srt.managers.scheduler_pp_mixin``:

1. forward newly received request metadata to the next PP stage;
2. run the same scheduling decision in every stage;
3. wait for previous-stage proxy readiness, run the local predicted forward,
   and forward proxy readiness;
4. circulate the last-stage result through PP0 and every later stage so each
   stage applies the shared batch-result/KV lifecycle exactly once.

Unlike GPU SGLang, proxy/output payloads contain metadata rather than tensors.
The causal queues and per-microbatch state are still process-local and ordered.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any

from sglsim.srt.managers.pipeline_parallel import (
    PPOutputEnvelope,
    PPProxyEnvelope,
    PPRequestEnvelope,
    PPStageBatchEnvelope,
    PipelineBatchIdentityError,
    StageId,
    assert_batch_identity,
)
from sglsim.srt.utils.network import (
    SimZmqContext,
    SimZmqSocket,
    get_zmq_socket,
)


class _PipelineStop(Exception):
    pass


@dataclass
class _PendingPPBatch:
    slot: int
    batch: Any
    envelope: PPStageBatchEnvelope


class SchedulerPPMixin:
    """Clock-agnostic PP transport plus online wall-clock event loop."""

    _PP_POLL_SLEEP_S = 0.0005

    def init_pp_transport(self) -> None:
        pp_size = int(self.server_args.pp_size)
        if pp_size <= 1:
            return
        if int(self.server_args.tp_size) != 1:
            raise NotImplementedError(
                "physical PP runtime currently requires TP=1; TP process groups "
                "must be implemented before enabling TP>1"
            )
        ctx = SimZmqContext()
        self._pp_send_request = None
        self._pp_recv_proxy = None
        self._pp_send_proxy = None
        if self.pp_rank < pp_size - 1:
            next_pp_rank = self.pp_rank + 1
            self._pp_send_request = get_zmq_socket(
                ctx,
                SimZmqSocket.PUSH,
                self.port_args.scheduler_input_ipc_name_for_stage(
                    self.dp_rank, next_pp_rank
                ),
                bind=False,
            )
            self._pp_send_proxy = get_zmq_socket(
                ctx,
                SimZmqSocket.PUSH,
                self.port_args.pp_proxy_ipc_name_for_stage(
                    self.dp_rank, next_pp_rank
                ),
                bind=False,
            )
        if self.pp_rank > 0:
            self._pp_recv_proxy = get_zmq_socket(
                ctx,
                SimZmqSocket.PULL,
                self.port_args.pp_proxy_ipc_name_for_stage(
                    self.dp_rank, self.pp_rank
                ),
                bind=False,
            )
        output_target = (self.pp_rank + 1) % pp_size
        self._pp_recv_output = get_zmq_socket(
            ctx,
            SimZmqSocket.PULL,
            self.port_args.pp_output_ipc_name_for_stage(
                self.dp_rank, self.pp_rank
            ),
            bind=False,
        )
        self._pp_send_output = get_zmq_socket(
            ctx,
            SimZmqSocket.PUSH,
            self.port_args.pp_output_ipc_name_for_stage(
                self.dp_rank, output_target
            ),
            bind=False,
        )

    def init_pp_loop_state(self) -> None:
        """Initialize the same microbatch state families as pinned SGLang."""

        pp_size = int(self.server_args.pp_size)
        if pp_size <= 1:
            raise RuntimeError("PP loop requires pp_size > 1")
        # SGLang uses pp_size + pp_async_batch_depth.  SGLSim does not expose
        # async depth yet, so the exact supported configuration is depth zero.
        self._pp_loop_size = pp_size
        self._pp_running_mbs = [self._new_running_batch() for _ in range(pp_size)]
        self._pp_last_mbs = [None for _ in range(pp_size)]
        self._pp_pending_batches: dict[int, _PendingPPBatch] = {}
        # Pinned SGLang ``SchedulerPPMixin.event_loop_pp`` does not apply a
        # returned output as soon as it reaches a stage.  After launching
        # microbatch ``q`` it commits the result occupying the *next* slot,
        # i.e. sequence ``q - (pp_size - 1)``.  The output ring reaches later
        # PP stages first, so eager application makes stage-local radix-cache
        # state diverge under restored-prefix retries.  Buffer arrivals and
        # expose them only at the same microbatch frontier as pinned SGLang.
        self._pp_received_outputs: dict[int, PPOutputEnvelope] = {}
        self._pp_last_received_output_sequence = -1
        self._pp_step_sequence = 0
        self._pp_last_request_sequence = -1
        self._pp_last_proxy_sequence = -1
        self._pp_last_output_sequence = -1

    def run_event_loop_pp_until_stop(
        self,
        *,
        cmd_queue=None,
        stop_event=None,
    ) -> None:
        """Run the physical PP event loop in this Scheduler subprocess."""

        self.init_pp_loop_state()
        while True:
            try:
                self._pp_raise_if_stopping(cmd_queue, stop_event)
                self._pp_pump_outputs(cmd_queue, stop_event)
                # A recovery command may arrive while another microbatch slot
                # is active.  Keep it pending until the entire local PP ring
                # reaches an idle SGLang lifecycle boundary, then emit this
                # stage's clear ACK.  The DP controller publishes recovery
                # only after every stage ACKs.
                self._maybe_clear_pending_gpu_l1()
                sequence = self._pp_step_sequence
                slot = sequence % self._pp_loop_size
                self._pp_wait_for_slot(slot, cmd_queue, stop_event)

                sched_start = time.perf_counter()
                received = self._pp_receive_and_forward_requests(
                    sequence=sequence,
                    cmd_queue=cmd_queue,
                    stop_event=stop_event,
                )
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
                        cmd_queue=cmd_queue,
                        stop_event=stop_event,
                    )
                    try:
                        assert_batch_identity(upstream.batch, local_envelope)
                    except PipelineBatchIdentityError as exc:
                        self._pp_record_lifecycle_error(
                            sequence=sequence,
                            phase="proxy_batch_identity",
                            error=exc,
                            expected=upstream.batch,
                            observed=local_envelope,
                        )
                        raise

                if batch is not None:
                    self.run_batch(batch)
                    batch.pipeline_overlap = False
                    self._advance_time_step(
                        batch=batch,
                        sched_elapsed=time.perf_counter() - sched_start,
                        overlap=False,
                    )
                    post_run_ids = tuple(req.rid for req in batch.reqs)
                    if post_run_ids != local_envelope.request_ids:
                        error = PipelineBatchIdentityError(
                            "run_batch changed PP request membership: "
                            f"planned={local_envelope.request_ids}, "
                            f"post_run={post_run_ids}"
                        )
                        self._pp_record_lifecycle_error(
                            sequence=sequence,
                            phase="post_run_membership",
                            error=error,
                            expected=local_envelope.request_ids,
                            observed=post_run_ids,
                        )
                        raise error
                    self._pp_pending_batches[sequence] = _PendingPPBatch(
                        slot=slot,
                        batch=batch,
                        envelope=local_envelope,
                    )

                if self.pp_rank < self.server_args.pp_size - 1:
                    proxy = PPProxyEnvelope(
                        source_stage=self.stage_id,
                        target_stage=StageId(
                            self.dp_rank,
                            self.pp_rank + 1,
                            self.tp_rank,
                        ),
                        step_sequence=sequence,
                        batch=local_envelope,
                    )
                    self._pp_send_proxy.send_pyobj(proxy)
                else:
                    output = PPOutputEnvelope.from_last_stage(
                        last_stage=self.stage_id,
                        pp_size=self.server_args.pp_size,
                        step_sequence=sequence,
                        batch=local_envelope,
                        result_logical_time=self._time_driver.now(),
                    )
                    self._pp_send_output.send_pyobj(output)

                self._pp_step_sequence += 1
                self._pp_pump_outputs(cmd_queue, stop_event)
                if batch is None and not received:
                    self._time_driver.idle_wait(self._IDLE_SLEEP_S)
            except _PipelineStop:
                break

        # A normal Engine shutdown happens after requests complete.  Pump any
        # already-arrived results without waiting on peers that received the
        # same stop command.
        self._pp_pump_outputs(cmd_queue=None, stop_event=None)

    def _pp_receive_and_forward_requests(
        self,
        *,
        sequence: int,
        cmd_queue,
        stop_event,
    ) -> bool:
        if self.pp_rank == 0:
            payloads = tuple(self.recv_requests())
            request_ids = tuple(payload.rid for payload in payloads)
            if self._pp_send_request is not None:
                self._pp_send_request.send_pyobj(
                    PPRequestEnvelope(
                        source_stage=self.stage_id,
                        target_stage=StageId(
                            self.dp_rank,
                            1,
                            self.tp_rank,
                        ),
                        step_sequence=sequence,
                        request_ids=request_ids,
                        request_payloads=payloads,
                    )
                )
            self._pp_last_request_sequence = sequence
            return bool(payloads)

        message = self._pp_wait_for_message(
            socket=self.recv_from_tokenizer,
            expected_type=PPRequestEnvelope,
            sequence=sequence,
            cmd_queue=cmd_queue,
            stop_event=stop_event,
            phase="request",
        )
        if message.target_stage != self.stage_id:
            raise PipelineBatchIdentityError(
                f"request target {message.target_stage} != {self.stage_id}"
            )
        now = self._time_driver.now()
        for payload in message.request_payloads:
            self._register_incoming_request(payload, now=now)
        if self._pp_send_request is not None:
            self._pp_send_request.send_pyobj(message.forward_from(self.stage_id))
        self._pp_last_request_sequence = sequence
        return bool(message.request_payloads)

    def _pp_wait_for_proxy(
        self,
        *,
        sequence: int,
        cmd_queue,
        stop_event,
    ) -> PPProxyEnvelope:
        proxy = self._pp_wait_for_message(
            socket=self._pp_recv_proxy,
            expected_type=PPProxyEnvelope,
            sequence=sequence,
            cmd_queue=cmd_queue,
            stop_event=stop_event,
            phase="proxy",
        )
        if proxy.target_stage != self.stage_id:
            raise PipelineBatchIdentityError(
                f"proxy target {proxy.target_stage} != {self.stage_id}"
            )
        self._pp_last_proxy_sequence = sequence
        return proxy

    def _pp_wait_for_message(
        self,
        *,
        socket,
        expected_type,
        sequence: int,
        cmd_queue,
        stop_event,
        phase: str,
    ):
        while True:
            self._pp_raise_if_stopping(cmd_queue, stop_event)
            self._pp_pump_outputs(cmd_queue, stop_event)
            try:
                message = socket.recv_pyobj(flags=1)
            except Exception:
                time.sleep(self._PP_POLL_SLEEP_S)
                continue
            if not isinstance(message, expected_type):
                error = PipelineBatchIdentityError(
                    f"unexpected PP {phase} message: {type(message).__name__}"
                )
                self._pp_record_lifecycle_error(
                    sequence=sequence,
                    phase=f"{phase}_type",
                    error=error,
                    expected=expected_type.__name__,
                    observed=type(message).__name__,
                )
                raise error
            if message.step_sequence != sequence:
                error = PipelineBatchIdentityError(
                    f"PP {phase} sequence mismatch: "
                    f"expected={sequence}, observed={message.step_sequence}"
                )
                self._pp_record_lifecycle_error(
                    sequence=sequence,
                    phase=f"{phase}_sequence",
                    error=error,
                    expected=sequence,
                    observed=message.step_sequence,
                )
                raise error
            return message

    def _pp_wait_for_slot(self, slot: int, cmd_queue, stop_event) -> None:
        while any(
            pending.slot == slot
            for pending in self._pp_pending_batches.values()
        ):
            self._pp_raise_if_stopping(cmd_queue, stop_event)
            progressed = self._pp_pump_outputs(cmd_queue, stop_event)
            if not progressed:
                time.sleep(self._PP_POLL_SLEEP_S)

    def _pp_pump_outputs(self, cmd_queue, stop_event) -> bool:
        if not hasattr(self, "_pp_recv_output"):
            return False
        progressed = False
        while True:
            if cmd_queue is not None or stop_event is not None:
                self._pp_raise_if_stopping(cmd_queue, stop_event)
            try:
                output = self._pp_recv_output.recv_pyobj(flags=1)
            except Exception:
                break
            if not isinstance(output, PPOutputEnvelope):
                error = PipelineBatchIdentityError(
                    f"unexpected PP output message: {type(output).__name__}"
                )
                self._pp_record_lifecycle_error(
                    sequence=self._pp_step_sequence,
                    phase="output_type",
                    error=error,
                    expected="PPOutputEnvelope",
                    observed=type(output).__name__,
                )
                raise error
            if output.target_stage != self.stage_id:
                raise PipelineBatchIdentityError(
                    f"output target {output.target_stage} != {self.stage_id}"
                )
            if output.step_sequence <= self._pp_last_received_output_sequence:
                raise PipelineBatchIdentityError(
                    "PP output receive sequence is not strictly increasing at stage "
                    f"{self.stage_id}: "
                    f"last={self._pp_last_received_output_sequence}, "
                    f"observed={output.step_sequence}"
                )
            self._pp_received_outputs[output.step_sequence] = output
            self._pp_last_received_output_sequence = output.step_sequence
            progressed = True

        # At the end of sequence q, ``_pp_step_sequence`` has already advanced
        # to q + 1.  Pinned SGLang processes the result in the next microbatch
        # slot, so the inclusive commit frontier is (q + 1) - pp_size.  The
        # same expression at the beginning of a step admits only results that
        # should already have been committed by the previous step.
        apply_frontier = self._pp_step_sequence - self._pp_loop_size
        while self._pp_last_output_sequence < apply_frontier:
            next_sequence = self._pp_last_output_sequence + 1
            output = self._pp_received_outputs.pop(next_sequence, None)
            if output is None:
                break
            pending = self._pp_pending_batches.pop(output.step_sequence, None)
            if output.batch_identity is None:
                if pending is not None:
                    raise PipelineBatchIdentityError(
                        "idle PP output has a local pending batch"
                    )
            else:
                if pending is None:
                    raise PipelineBatchIdentityError(
                        "PP output references a missing local pending batch: "
                        f"sequence={output.step_sequence} stage={self.stage_id}"
                    )
                if pending.envelope.identity_tuple() != output.batch_identity:
                    raise PipelineBatchIdentityError(
                        "PP output batch identity does not match local pending batch"
                    )
                self._pp_apply_batch_result(pending)
            self._pp_last_output_sequence = output.step_sequence
            forwarded = output.consume_and_forward(
                current_stage=self.stage_id,
                pp_size=self.server_args.pp_size,
            )
            if forwarded is not None:
                self._pp_send_output.send_pyobj(forwarded)
            progressed = True
        return progressed

    def _pp_apply_batch_result(self, pending: _PendingPPBatch) -> None:
        saved_running = self.running_batch
        saved_last = self.last_batch
        try:
            self.running_batch = self._pp_running_mbs[pending.slot]
            self.last_batch = self._pp_last_mbs[pending.slot]
            self.process_batch_result(pending.batch)
            self.last_batch = pending.batch
            self._finalize_last_batch()
            self._pp_running_mbs[pending.slot] = self.running_batch
            self._pp_last_mbs[pending.slot] = self.last_batch
        finally:
            self.running_batch = saved_running
            self.last_batch = saved_last

    def _pp_batch_envelope(
        self,
        *,
        batch,
        slot: int,
        sequence: int,
    ) -> PPStageBatchEnvelope | None:
        if batch is None:
            return None
        route_versions: list[int] = []
        for req in batch.reqs:
            sim = req.sampling_params.custom_params.get("simulation", {})
            try:
                route_versions.append(
                    max(
                        int(
                            sim.get(
                                "aegis_route_version",
                                sim.get("route_version", 1),
                            )
                        ),
                        1,
                    )
                )
            except (TypeError, ValueError):
                route_versions.append(1)
        return PPStageBatchEnvelope(
            stage=self.stage_id,
            microbatch_id=slot,
            batch_id=batch.batch_id,
            request_ids=tuple(req.rid for req in batch.reqs),
            forward_mode=batch.mode.value,
            token_frontiers=tuple(
                max(int(req.kv_committed_length), 0) for req in batch.reqs
            ),
            route_versions=tuple(route_versions),
            budget_version=max(int(getattr(self, "_aegis_budget_version", 1)), 1),
            causal_sequence=sequence,
        )

    def _pp_raise_if_stopping(self, cmd_queue, stop_event) -> None:
        if stop_event is not None and stop_event.is_set():
            raise _PipelineStop
        if cmd_queue is None:
            return
        while True:
            try:
                message = cmd_queue.get_nowait()
            except queue.Empty:
                return
            if self._handle_runtime_command(message):
                raise _PipelineStop

    def _pp_record_lifecycle_error(
        self,
        *,
        sequence: int,
        phase: str,
        error: Exception,
        expected: object,
        observed: object,
    ) -> None:
        event = {
            "type": "pp_lifecycle_mismatch",
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            "tp_rank": self.tp_rank,
            "sequence": int(sequence),
            "phase": phase,
            "error": str(error),
            "expected": repr(expected),
            "observed": repr(observed),
            "time": self._time_driver.now(),
        }
        if self._aegis_event_queue is not None:
            self._aegis_event_queue.put(event)
        self.logger.error("PP lifecycle mismatch: %s", event)
