from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from sglsim.srt.simulator.metrics import RequestStats, dump_simulator_outputs
from sglsim.srt.simulator.predictor import build_simulator_predictor
from sglsim.srt.simulator.state import SimulatorState, set_active_state

_INSTALLED = False


class NativeSchedulerState:
    def __init__(self, config, predictor):
        self.config = config
        self.predictor = predictor
        self.clock = SimulatorState()
        self.request_stats: dict[str, RequestStats] = defaultdict(RequestStats)
        self.iteration_stats: list[dict] = []
        self.last_cpu_ts = 0.0


def _past_kv_length(req: Any) -> int:
    return _safe_len(getattr(req, "prefix_indices", None)) + _safe_len(
        getattr(req, "output_ids", None)
    )


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    return int(len(value))


def _prediction_requests(batch) -> list[dict[str, int]]:
    requests: list[dict[str, int]] = []
    forward_mode = getattr(batch, "forward_mode", None)
    if forward_mode is None:
        return requests

    if forward_mode.is_extend():
        for req in batch.reqs:
            extend_length = getattr(req, "extend_input_len", None)
            if extend_length is None:
                extend_length = _safe_len(getattr(req, "input_ids", None))
            requests.append(
                {
                    "extend_length": int(extend_length),
                    "past_kv_length": _past_kv_length(req),
                }
            )
    elif forward_mode.is_decode():
        for req in batch.reqs:
            requests.append(
                {
                    "extend_length": 1,
                    "past_kv_length": _past_kv_length(req),
                }
            )
    return requests


def _is_tokenized_generate_req(req: Any) -> bool:
    return hasattr(req, "rid") and hasattr(req, "input_ids") and hasattr(
        req, "sampling_params"
    )


def _record_arrived_requests(native: NativeSchedulerState, reqs: list[Any]) -> None:
    now = time.time()
    for req in reqs:
        if not _is_tokenized_generate_req(req):
            continue
        stats = native.request_stats[req.rid]
        stats.rid = req.rid
        stats.input_length = len(req.input_ids)
        stats.output_length = int(getattr(req.sampling_params, "max_new_tokens", 0) or 0)
        custom_params = getattr(req.sampling_params, "custom_params", None) or {}
        sim = custom_params.get("simulation", {})
        stats.created_time = float(sim.get("server_created_time", now))
        stats.last_event_time = stats.created_time
        stats.queue_start = now
    if reqs and native.last_cpu_ts == 0:
        native.last_cpu_ts = time.time()


def install(config) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from sglsim.srt.managers import scheduler as scheduler_mod

    Scheduler = scheduler_mod.Scheduler
    original_init = Scheduler.__init__
    original_recv_requests = Scheduler.recv_requests
    original_get_new_batch_prefill = Scheduler.get_new_batch_prefill
    original_run_batch = Scheduler.run_batch
    original_process_batch_result = Scheduler.process_batch_result
    original_profile = Scheduler.profile

    def __init__(self, *args, **kwargs):
        from sglsim.srt.simulator.native.debug import trace

        trace("Scheduler.__init__:start")
        original_init(self, *args, **kwargs)
        trace("Scheduler.__init__:after-original")
        predictor = build_simulator_predictor(self.server_args, self.model_config)
        self.sglsim_native = NativeSchedulerState(config, predictor)
        set_active_state(self.sglsim_native.clock)
        trace("Scheduler.__init__:end")

    def recv_requests(self, *args, **kwargs):
        reqs = original_recv_requests(self, *args, **kwargs)
        _record_arrived_requests(self.sglsim_native, reqs)
        return reqs

    def get_new_batch_prefill(self, *args, **kwargs):
        batch = original_get_new_batch_prefill(self, *args, **kwargs)
        if batch is not None:
            for req in batch.reqs:
                stats = self.sglsim_native.request_stats[req.rid]
                stats.final_reused_tokens = getattr(req, "cached_tokens", 0)
                if stats.queue_end == -1:
                    stats.queue_end = time.time()
        return batch

    def run_batch(self, batch, *args, **kwargs):
        result = original_run_batch(self, batch, *args, **kwargs)
        requests = _prediction_requests(batch)
        if requests:
            native = self.sglsim_native
            native.clock.inc_iteration()
            latency = float(native.predictor.predict_requests(requests))
            if config.mode == "blocking":
                time.sleep(max(0.0, latency))
            native.clock.set_current_inference_dur(latency)
            native.iteration_stats.append(
                {
                    "requests": requests,
                    "forward_latency": latency,
                    "l2_load_latency": native.clock.pop_hicache_l2_load_dur(),
                    "l2_backup_latency": native.clock.pop_hicache_l2_backup_dur(),
                }
            )
        return result

    def process_batch_result(self, batch, *args, **kwargs):
        result = original_process_batch_result(self, batch, *args, **kwargs)
        now = time.time()
        if batch is not None:
            for req in getattr(batch, "reqs", []):
                stats = self.sglsim_native.request_stats[req.rid]
                if stats.rid and getattr(req, "is_chunked", 0) == 0:
                    stats.gen_token_latencies.append(now - stats.last_event_time)
                    stats.last_event_time = now
        return result

    def profile(self, req, *args, **kwargs):
        stats = sorted(
            [item for item in self.sglsim_native.request_stats.values() if item.rid],
            key=lambda item: item.created_time,
        )
        dump_simulator_outputs(
            self.sglsim_native.config.output_dir,
            stats,
            self.sglsim_native.iteration_stats,
            num_warmup=self.sglsim_native.config.num_warmup,
        )
        return original_profile(self, req, *args, **kwargs)

    Scheduler.__init__ = __init__
    Scheduler.recv_requests = recv_requests
    Scheduler.get_new_batch_prefill = get_new_batch_prefill
    Scheduler.run_batch = run_batch
    Scheduler.process_batch_result = process_batch_result
    Scheduler.profile = profile


def is_installed() -> bool:
    return _INSTALLED
