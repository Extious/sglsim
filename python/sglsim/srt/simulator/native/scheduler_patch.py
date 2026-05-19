from __future__ import annotations

from typing import Any

from sglsim.srt.simulator.predictor import build_simulator_predictor
from sglsim.srt.simulator.scheduler import SchedulerSimulatorRuntime

_INSTALLED = False


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
    original_event_loop_normal = Scheduler.event_loop_normal

    def event_loop_overlap(self, *args, **kwargs):
        return original_event_loop_normal(self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        from sglsim.srt.simulator.native.debug import trace

        trace("Scheduler.__init__:start")
        original_init(self, *args, **kwargs)
        trace("Scheduler.__init__:after-original")
        overlap_schedule = not getattr(self.server_args, "disable_overlap_schedule", False)
        self.server_args.disable_overlap_schedule = True
        predictor = build_simulator_predictor(self.server_args, self.model_config)
        self.sglsim_sim_runtime = SchedulerSimulatorRuntime(
            mode=config.mode,
            output_dir=config.output_dir,
            num_warmup=config.num_warmup,
            overlap_schedule=overlap_schedule,
            predictor=predictor,
        )
        trace("Scheduler.__init__:end")

    def recv_requests(self, *args, **kwargs):
        reqs = original_recv_requests(self, *args, **kwargs)
        return self.sglsim_sim_runtime.filter_recv_requests(reqs)

    def get_new_batch_prefill(self, *args, **kwargs):
        batch = original_get_new_batch_prefill(self, *args, **kwargs)
        waiting_queue_len = len(getattr(self, "waiting_queue", []) or [])
        running_batch_len = len(getattr(getattr(self, "running_batch", None), "reqs", []) or [])
        self.sglsim_sim_runtime.on_new_batch_prefill(
            batch, waiting_queue_len, running_batch_len
        )
        return batch

    def run_batch(self, batch, *args, **kwargs):
        result = original_run_batch(self, batch, *args, **kwargs)
        self.sglsim_sim_runtime.on_run_batch(batch, result)
        return result

    def process_batch_result(self, batch, *args, **kwargs):
        result = original_process_batch_result(self, batch, *args, **kwargs)
        self.sglsim_sim_runtime.on_process_batch_result(batch)
        return result

    def profile(self, req, *args, **kwargs):
        self.sglsim_sim_runtime.dump_outputs_and_reset()
        return original_profile(self, req, *args, **kwargs)

    Scheduler.__init__ = __init__
    Scheduler.recv_requests = recv_requests
    Scheduler.get_new_batch_prefill = get_new_batch_prefill
    Scheduler.run_batch = run_batch
    Scheduler.process_batch_result = process_batch_result
    Scheduler.profile = profile
    Scheduler.event_loop_overlap = event_loop_overlap


def is_installed() -> bool:
    return _INSTALLED
