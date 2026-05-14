from __future__ import annotations

import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from sglang.srt.simulator.metrics import RequestStats, dump_simulator_outputs
from sglang.srt.simulator.state import (
    SimulatorState,
    set_active_runtime,
    set_active_state,
)


@dataclass
class PredictionRequest:
    extend_length: int
    past_kv_length: int


class PredictionBatch:
    def __init__(self, requests: List[dict]):
        self.reqs = [
            PredictionRequest(
                extend_length=item["extend_length"],
                past_kv_length=item["past_kv_length"],
            )
            for item in requests
        ]

    def request_info(self) -> List[List[int]]:
        return [[req.extend_length, req.past_kv_length] for req in self.reqs]

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def is_prefill(self) -> bool:
        return not self.is_decode()

    def is_decode(self) -> bool:
        return all(req.extend_length <= 1 for req in self.reqs)


class PredictorAdapter:
    def __init__(self, predictor: Any):
        self.predictor = predictor

    def predict_requests(self, requests: List[dict]) -> float:
        if hasattr(self.predictor, "predict_requests"):
            return float(self.predictor.predict_requests(requests))
        return float(self.predictor.predict_infer_time(PredictionBatch(requests)))


class SchedulerSimulatorRuntime:
    def __init__(
        self,
        *,
        mode: str,
        output_dir: str,
        num_warmup: int,
        overlap_schedule: bool,
        predictor: Any,
    ):
        self.mode = mode
        self.output_dir = output_dir
        self.num_warmup = num_warmup
        self.overlap_schedule = overlap_schedule
        self.predictor = PredictorAdapter(predictor) if predictor is not None else None
        self.state = SimulatorState()
        self.request_stats: Dict[str, RequestStats] = defaultdict(RequestStats)
        self.iteration_stats: List[Dict[str, Any]] = []
        self.future_queue: List[tuple[float, int, Any]] = []
        self.offline_recv_all_request = False
        self.last_cpu_ts = 0.0
        self._last_prediction_requests: List[dict] = []
        set_active_state(self.state)
        set_active_runtime(self)

    @classmethod
    def from_scheduler(cls, scheduler, overlap_schedule: bool):
        settings = scheduler.simulator_settings
        predictor = getattr(scheduler, "simulator_predictor", None)
        return cls(
            mode=settings.mode,
            output_dir=settings.output_dir,
            num_warmup=settings.num_warmup,
            overlap_schedule=overlap_schedule,
            predictor=predictor,
        )

    def reset(self) -> None:
        self.state.reset()
        self.request_stats.clear()
        self.iteration_stats.clear()
        self.future_queue.clear()
        self.offline_recv_all_request = False
        self.last_cpu_ts = 0.0
        self._last_prediction_requests = []

    def _simulation_params(self, req) -> dict | None:
        sampling_params = getattr(req, "sampling_params", None)
        custom_params = getattr(sampling_params, "custom_params", None)
        if custom_params is None:
            return None
        return custom_params.get("simulation")

    def _is_generate_req(self, req) -> bool:
        return hasattr(req, "rid") and hasattr(req, "input_ids") and hasattr(req, "sampling_params")

    def filter_recv_requests(self, recv_reqs: List[Any]) -> List[Any]:
        if self.mode == "blocking":
            self._record_arrived_requests(recv_reqs)
            return recv_reqs

        immediate = []
        for req in recv_reqs:
            sim_params = self._simulation_params(req)
            if not self._is_generate_req(req) or sim_params is None:
                immediate.append(req)
                continue
            enqueue_time = sim_params.get("queue_start") or sim_params["created_time"]
            heapq.heappush(self.future_queue, (enqueue_time, time.time_ns(), req))
            total_request = sim_params["total_request"]
            if len(self.future_queue) == total_request:
                self.offline_recv_all_request = True

        arrived = list(immediate)
        while self.offline_recv_all_request and self.future_queue:
            enqueue_time, _, req = self.future_queue[0]
            if enqueue_time > self.state.global_clock:
                break
            heapq.heappop(self.future_queue)
            arrived.append(req)

        self._record_arrived_requests(arrived)
        return arrived

    def _record_arrived_requests(self, recv_reqs: List[Any]) -> None:
        now = time.time()
        for req in recv_reqs:
            if not self._is_generate_req(req):
                continue
            sim_params = self._simulation_params(req) or {}
            stats = self.request_stats[req.rid]
            stats.rid = req.rid
            stats.input_length = len(req.input_ids)
            stats.output_length = req.sampling_params.max_new_tokens
            if self.mode == "blocking":
                stats.created_time = sim_params.get("server_created_time", now)
                stats.last_event_time = stats.created_time
                stats.queue_start = now
            else:
                stats.created_time = sim_params["created_time"]
                stats.last_event_time = stats.created_time
                queue_start = sim_params.get("queue_start")
                if queue_start is not None:
                    self.state.set_global_clock(queue_start)
                stats.queue_start = self.state.global_clock
        if recv_reqs and self.last_cpu_ts == 0:
            self.last_cpu_ts = time.time()

    def on_new_batch_prefill(self, new_batch, waiting_queue_len: int, running_batch_len: int) -> None:
        if new_batch is not None:
            for req in new_batch.reqs:
                stats = self.request_stats[req.rid]
                stats.final_reused_tokens = getattr(req, "cached_tokens", 0)
                if stats.queue_end == -1:
                    stats.queue_end = time.time() if self.mode == "blocking" else self.state.global_clock
        elif self.mode == "offline" and self.future_queue and running_batch_len == 0:
            next_created_time, _, _ = self.future_queue[0]
            self.state.set_global_clock(next_created_time + 1e-6)
        elif running_batch_len == 0 and waiting_queue_len > 0:
            self.state.step_global_clock(0.005)
            self.state.set_current_inference_dur(0.005)

    def _build_prediction_batch(self, batch):
        requests = []
        for req in batch.reqs:
            if batch.forward_mode.is_extend():
                extend_length = req.extend_input_len
            elif batch.forward_mode.is_decode():
                extend_length = 1
            else:
                continue
            requests.append(
                {
                    "extend_length": extend_length,
                    "past_kv_length": len(req.prefix_indices) + len(req.output_ids),
                }
            )
        return requests

    def on_run_batch(self, batch, result) -> None:
        if result.__class__.__name__ != "GenerationBatchResult":
            return
        requests = self._build_prediction_batch(batch)
        if not requests:
            return
        self.state.inc_iteration()
        predicted_latency = self._predict_latency(requests)
        if self.mode == "blocking":
            time.sleep(abs(predicted_latency))
            now = time.time()
            forward_latency = now - self.last_cpu_ts if self.last_cpu_ts else predicted_latency
            self.last_cpu_ts = now
        else:
            forward_latency = predicted_latency
        self.state.set_current_inference_dur(float(forward_latency))
        self._last_prediction_requests = requests

    def _predict_latency(self, requests: List[dict]) -> float:
        if self.predictor is None:
            total_tokens = sum(item["extend_length"] for item in requests)
            return max(total_tokens, 1) * 0.001
        return float(self.predictor.predict_requests(requests))

    def on_process_batch_result(self, batch) -> None:
        if batch is None or len(batch.reqs) == 0:
            return
        load_dur = self.state.pop_hicache_l2_load_dur()
        backup_dur = self.state.pop_hicache_l2_backup_dur()
        inference_dur = self.state.current_inference_dur
        if self.overlap_schedule:
            self.state.step_global_clock(max(load_dur - self.state.last_inference_dur, 0))
            self.state.step_global_clock(inference_dur)
            response_time = self.state.global_clock + backup_dur
        else:
            self.state.step_global_clock(load_dur + inference_dur + backup_dur)
            response_time = self.state.global_clock
        for req in batch.reqs:
            if getattr(req, "is_chunked", 0) == 0:
                stats = self.request_stats[req.rid]
                stats.gen_token_latencies.append(response_time - stats.last_event_time)
                stats.last_event_time = response_time
        self.iteration_stats.append(
            {
                "requests": self._last_prediction_requests,
                "forward_latency": inference_dur,
                "l2_load_latency": load_dur,
                "l2_backup_latency": backup_dur,
            }
        )
        self.last_cpu_ts = time.time()

    def dump_outputs(self) -> dict:
        stats = sorted(
            [item for item in self.request_stats.values() if item.rid],
            key=lambda item: item.created_time,
        )
        return dump_simulator_outputs(
            self.output_dir,
            stats,
            self.iteration_stats,
            num_warmup=self.num_warmup,
        )
