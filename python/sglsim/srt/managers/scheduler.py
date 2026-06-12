from __future__ import annotations

import os
import time
from collections import deque

from sglsim.srt.managers.chunked_prefill import split_prefill_length
from sglsim.srt.managers.io_struct import (
    FailoverBatchReqInput,
    ReqSnapshot,
    SimulateGpuFailureReqInput,
    SimulateGpuRecoveryReqInput,
    TokenizedGenerateReqInput,
)
from sglsim.srt.managers.prefill_adder import AddReqResult, PrefillAdder
from sglsim.srt.managers.schedule_batch import ForwardMode, Req, RequestStatus, ScheduleBatch
from sglsim.srt.managers.schedule_policy import SchedulePolicy
from sglsim.srt.mem_cache.common import extend_alloc_tokens
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.observability.scheduler_metrics_mixin import PrefillStats, SchedulerMetricsMixin
from sglsim.srt.server_args import PortArgs, ServerArgs
from sglsim.srt.simulation.forward_slot import ForwardSlot
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.simulation.manager.env import simulation_mode
from sglsim.srt.simulation.manager.state import StateManager
from sglsim.srt.simulation.pipeline_accounting import (
    PipelineStages,
    compute_pipeline_sleep_s,
)
from sglsim.srt.simulation.types import RequestStats, SimulationMode
from sglsim.srt.time_predictor.base import InferTimePredictor
from sglsim.srt.utils.network import SimIpcRegistry, SimZmqContext, SimZmqSocket, get_zmq_socket
from sglsim.srt.utils.logging import scheduler_logger

REQUEST_STATS: dict[str, RequestStats] = {}


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_terminal_status(status: str) -> bool:
    return status in ("completed", "aborted")


class Scheduler(SchedulerMetricsMixin):
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
    ) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self.dp_rank = dp_rank
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
        self.ft_config = ConfigManager.get_fault_tolerance_config()
        self._gpu_failed = False
        self._failure_injected_time = 0.0
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
        self.init_metrics()
        ctx = SimZmqContext()
        self.recv_from_tokenizer = get_zmq_socket(
            ctx,
            SimZmqSocket.PULL,
            port_args.scheduler_input_ipc_name_for_dp(dp_rank),
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
                if isinstance(obj, SimulateGpuFailureReqInput):
                    self.handle_simulate_gpu_failure(obj)
                    continue
                if isinstance(obj, SimulateGpuRecoveryReqInput):
                    self.handle_simulate_gpu_recovery(obj)
                    continue
                recv_reqs.append(obj)
        except Exception:
            pass

        now = time.time()
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
        if self._gpu_failed:
            self.logger.warning(
                "Dropping request %s on failed dp_rank=%d",
                tok_req.rid,
                self.dp_rank,
            )
            return
        req = tok_req.to_req()
        if req.rid not in REQUEST_STATS:
            REQUEST_STATS[req.rid] = RequestStats(rid=req.rid)
        stats = REQUEST_STATS[req.rid]
        if req.is_failover_retried and req.pre_failover_backed_up_tokens > 0:
            backed = min(req.pre_failover_backed_up_tokens, req.prefill_length)
            req.cached_length = backed
            req.kv_committed_length = backed
            req.cache_protected_len = backed
        stats.input_length = req.prefill_length
        stats.output_length = req.original_max_new_tokens or req.sampling_params.max_new_tokens
        stats.generated_tokens = req.total_generated_tokens
        sim = tok_req.sampling_params.custom_params.get("simulation", {})
        if not req.is_failover_retried or stats.created_time < 0:
            stats.created_time = float(sim.get("server_created_time", now))
            stats.last_event_time = stats.created_time
        stats.queue_start = now
        stats.status = "running"
        stats.is_failover_retried = bool(req.is_failover_retried)
        stats.failed_dp_rank = (
            int(req.failover_source_dp_rank)
            if req.failover_source_dp_rank is not None
            else -1
        )
        stats.pre_failover_output_tokens = int(req.pre_failover_output_tokens)
        stats.pre_failover_backed_up_tokens = int(req.pre_failover_backed_up_tokens)
        stats.recompute_tokens = int(req.failover_recompute_tokens)
        if req.is_failover_retried:
            stats.failover_injected_time = float(
                sim.get("failover_injected_time", stats.failover_injected_time)
            )
            stats.failover_retry_time = float(sim.get("failover_retry_time", now))
        else:
            stats.failover_retry_time = 0.0
        if req.is_failover_retried and stats.failover_injected_time > 0:
            stats.failover_overhead_s = max(now - stats.failover_injected_time, 0.0)
        stats.agent_id = str(sim.get("agent_id", sim.get("worker_id", "")))
        stats.critical_agent = bool(sim.get("critical_agent", False))
        if self.cache.config.l3_capacity_pages > 0:
            self.cache.prefetch_from_storage(req)
        self._req_registry[req.rid] = req
        self._enqueue_req(req, stats.queue_start)

    def _stats_payload(self, req: Req, *, finished: bool, last_event_time: float) -> dict:
        stats = REQUEST_STATS[req.rid]
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
            "is_failover_retried": stats.is_failover_retried,
            "failed_dp_rank": stats.failed_dp_rank,
            "pre_failover_output_tokens": stats.pre_failover_output_tokens,
            "pre_failover_backed_up_tokens": stats.pre_failover_backed_up_tokens,
            "recompute_tokens": stats.recompute_tokens,
            "failover_overhead_s": stats.failover_overhead_s,
            "failover_injected_time": stats.failover_injected_time,
            "failover_retry_time": stats.failover_retry_time,
            "agent_id": stats.agent_id,
            "critical_agent": stats.critical_agent,
        }

    def _send_detokenizer_update(
        self, req: Req, *, finished: bool, last_event_time: float
    ) -> None:
        self.send_to_detokenizer.send_pyobj(
            self._stats_payload(req, finished=finished, last_event_time=last_event_time)
        )

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
        last_event_time = time.time()
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
            last_event_time = time.time()
            stats.last_event_time = last_event_time
            stats.generated_tokens = req.total_generated_tokens
            self._send_detokenizer_update(
                req, finished=True, last_event_time=last_event_time
            )

    def _complete_request(self, req: Req) -> None:
        stats = REQUEST_STATS[req.rid]
        req.status = RequestStatus.COMPLETED
        stats.status = "completed"
        last_event_time = time.time()
        stats.last_event_time = last_event_time
        stats.generated_tokens = req.total_generated_tokens
        self.cache.finish_request(req)
        self._purge_req_from_scheduler_state(req)
        self._req_registry.pop(req.rid, None)
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

    def _gpu_backup_enabled(self) -> bool:
        return (
            self.ft_config.enabled
            and self.ft_config.strategy == "gpu_backup"
            and not self.cache.hierarchical_enabled
        )

    def _req_gpu_backup_enabled(self, req: Req) -> bool:
        if not self._gpu_backup_enabled():
            return False
        if self.ft_config.backup_policy != "critical_agents":
            return True
        sim = req.sampling_params.custom_params.get("simulation", {})
        return bool(sim.get("critical_agent", False) or sim.get("backup_kv", False))

    def _update_peer_backup_after_prefill(self, req: Req) -> None:
        if not self._req_gpu_backup_enabled(req):
            return
        prompt_len = req.original_prompt_len or req.prefill_length
        if self.ft_config.backup_prefill_tokens == "all":
            req.peer_backed_up_tokens = max(req.peer_backed_up_tokens, prompt_len)

    def _update_peer_backup_after_decode(self, req: Req) -> None:
        if not self._req_gpu_backup_enabled(req) or not self.ft_config.backup_decode:
            return
        lag = self.ft_config.peer_backup_lag_tokens
        durable_decode = max(req.total_generated_tokens - lag, 0)
        prompt_len = req.original_prompt_len or (
            req.input_token_count if req.input_token_count is not None else len(req.input_ids)
        )
        req.peer_backed_up_tokens = max(
            req.peer_backed_up_tokens,
            prompt_len + durable_decode,
        )

    def _backed_up_tokens_for_snapshot(self, req: Req) -> int:
        if self._req_gpu_backup_enabled(req):
            return max(int(req.peer_backed_up_tokens), 0)
        return 0

    def _collect_failover_reqs(self) -> list[Req]:
        reqs: list[Req] = []
        seen: set[str] = set()

        def add(req: Req | None) -> None:
            if req is None:
                return
            if req.rid in seen:
                return
            seen.add(req.rid)
            if req.status != RequestStatus.COMPLETED:
                reqs.append(req)

        for req in self.waiting_queue:
            add(req)
        add(self.chunked_req)
        for batch in (self.running_batch, self.last_batch):
            if batch is None:
                continue
            for req in batch.reqs:
                add(req)
        for batch in self._result_queue:
            for req in batch.reqs:
                add(req)
        return reqs

    def _snapshot_req(self, req: Req) -> ReqSnapshot:
        sim = dict(req.sampling_params.custom_params.get("simulation", {}))
        return ReqSnapshot(
            rid=req.rid,
            output_ids=list(req.output_ids),
            backed_up_tokens=self._backed_up_tokens_for_snapshot(req),
            origin_input_ids=list(req.input_ids[: req.original_prompt_len or len(req.input_ids)]),
            origin_input_token_count=req.original_prompt_len or req.input_token_count,
            sampling_params=req.sampling_params,
            original_max_new_tokens=req.original_max_new_tokens
            or req.sampling_params.max_new_tokens,
            simulation=sim,
            routing_key=req.routing_key,
        )

    def _clear_failed_rank_scheduler_state(self) -> None:
        for req in self._collect_failover_reqs():
            self.cache.release_request(req)
        self.waiting_queue.clear()
        self.chunked_req = None
        self.running_batch = self._new_running_batch()
        self.last_batch = None
        self._result_queue.clear()
        self.cache = KVCachePool(ConfigManager.get_cache_config(self.server_args.model_path))
        self.new_token_ratio = self.init_new_token_ratio
        self._waiting_plan_dirty = False

    def handle_simulate_gpu_failure(self, recv_req: SimulateGpuFailureReqInput) -> None:
        if int(recv_req.dp_rank) != int(self.dp_rank):
            return
        now = time.time()
        reqs = self._collect_failover_reqs()
        snapshots = [self._snapshot_req(req) for req in reqs]
        for snap in snapshots:
            snap.simulation["failover_injected_time"] = now
        for snap in snapshots:
            stats = REQUEST_STATS.get(snap.rid)
            if stats is not None:
                stats.failover_injected_time = now
        self._failure_injected_time = now
        self._clear_failed_rank_scheduler_state()
        self._gpu_failed = True
        self.send_to_detokenizer.send_pyobj(
            {
                "op": "failover_batch",
                "failover": FailoverBatchReqInput(
                    failed_dp_rank=int(self.dp_rank),
                    snapshots=snapshots,
                ),
            }
        )
        self.logger.warning(
            "GPU failure simulated on dp_rank=%d snapshots=%d",
            self.dp_rank,
            len(snapshots),
        )

    def handle_simulate_gpu_recovery(self, recv_req: SimulateGpuRecoveryReqInput) -> None:
        if int(recv_req.dp_rank) != int(self.dp_rank):
            return
        self._gpu_failed = False
        self.logger.warning("GPU recovery simulated on dp_rank=%d", self.dp_rank)

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

    def _advance_wall_clock_step(
        self,
        *,
        batch: ScheduleBatch,
        sched_elapsed: float,
        overlap: bool,
    ) -> None:
        forward_s = batch.inference_dur_s
        if forward_s <= 0:
            # Prefill-alloc blocked batches can yield forward_s==0; avoid a
            # tight scheduler spin that burns wall clock without sim progress.
            if sched_elapsed < 1e-3:
                time.sleep(0.001)
            return

        load_s = StateManager.pop_hicache_l2_load_dur()
        backup_pending_s = self.cache.hicache.pending_backup_dur
        self._profile_add("pipeline_forward", forward_s)
        self._profile_add("pipeline_h2d", load_s)
        self._profile_add("pipeline_backup_pending", backup_pending_s)
        result = compute_pipeline_sleep_s(
            PipelineStages(
                h2d_s=load_s,
                forward_s=forward_s,
                backup_pending_s=backup_pending_s,
            ),
            last_infer_s=self._forward_slot.last_infer_s,
            overlap=overlap,
        )

        now = time.time()
        cpu_overflow = (
            self._forward_slot.cpu_overflow_s(
                now=now,
                sched_elapsed_s=sched_elapsed,
            )
            if overlap
            else 0.0
        )
        sleep_s = result.sleep_s + cpu_overflow
        self._profile_add("pipeline_sleep", sleep_s)
        self._profile_add("pipeline_sleep_base", result.sleep_s)
        self._profile_add("pipeline_cpu_overflow", cpu_overflow)
        self._profile_add("pipeline_comm_h2d_visible", result.comm_h2d_visible_s)
        self._profile_add("pipeline_comm_d2h_remainder", result.comm_d2h_remainder_s)
        if sleep_s > 0:
            time.sleep(sleep_s)

        self._forward_slot.record_forward(
            now=time.time(),
            forward_s=forward_s,
            sleep_s=result.sleep_s,
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

        ts = time.time()
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
            predicted = float(self.predictor.predict_infer_time(pred_batch))
            batch.inference_dur_s = predicted
            forward = predicted
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
        return time.time()

    def process_batch_result(self, batch: ScheduleBatch) -> None:
        completed_reqs: list[Req] = []
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
                stats.service_time_s += max(batch.inference_dur_s, 0.0)
                if req.prefill_chunks_remaining:
                    self.stash_chunked_request(req)
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
                self._update_peer_backup_after_prefill(req)
                if req.mark_finished_length():
                    completed_reqs.append(req)
            else:
                if self.enable_overlap and (
                    req.status in (RequestStatus.COMPLETED, RequestStatus.ABORTED)
                    or req.is_retracted
                ):
                    batch.reqs = [r for r in batch.reqs if r is not req]
                    continue
                if req.mark_finished_length():
                    completed_reqs.append(req)
                    continue
                stats.service_time_s += max(batch.inference_dur_s, 0.0)
                req.append_output_token(self._synthetic_output_token(req))
                self._update_peer_backup_after_decode(req)
                if req.mark_finished_length():
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
                    max(batch.inference_dur_s, 0.0)
                    if batch.inference_dur_s > 0
                    else latency
                )
            if latency > 0:
                stats.gen_token_latencies.append(latency)
            stats.last_event_time = response_time
            if id(req) in completed_ids:
                continue
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
        self.logger.warning(
            "Scheduler busy-wait: waiting=%d running=%d chunked=%d last_batch=%s pending=%s",
            len(self.waiting_queue),
            len(self.running_batch.reqs),
            1 if self.chunked_req is not None else 0,
            "set" if self.last_batch is not None else "none",
            pending,
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
        self._advance_wall_clock_step(
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
            self._advance_wall_clock_step(
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
        if self._gpu_failed:
            self.recv_requests()
            if self._gpu_failed:
                self._profile_add("event_loop_step", time.perf_counter() - t0)
                return False
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
                    if msg[0] == "stop":
                        break
                except queue_module.Empty:
                    pass

            if self.event_loop_step():
                continue

            if self._recover_idle_pending_requests():
                continue

            if self._rank_idle():
                if stop_event is not None and stop_event.wait(timeout=self._IDLE_SLEEP_S):
                    break
                if cmd_queue is not None:
                    try:
                        msg = cmd_queue.get(timeout=self._IDLE_SLEEP_S)
                        if msg[0] == "stop":
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
) -> None:
    import logging
    import os

    from sglsim.srt.utils.logging import configure_logger

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
    scheduler = Scheduler(server_args, port_args, dp_rank=dp_rank)
    ready_event.set()
    # Move the large, process-lifetime heap (torch/transformers/aiconfigurator)
    # into the permanent generation so per-step decode allocations don't trigger
    # full gen2 rescans of millions of static objects.
    import gc

    gc.collect()
    gc.freeze()
    try:
        scheduler.run_event_loop_until_stop(cmd_queue=cmd_queue)
    except Exception:
        scheduler.logger.exception("Scheduler subprocess dp_rank=%d crashed", dp_rank)
        raise
