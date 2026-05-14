from __future__ import annotations

from sglsim.srt.scheduling.batch import ScheduleBatch
from sglsim.srt.scheduling.request import Request
from sglsim.srt.simulator.scheduler import SchedulerSimulatorRuntime


class CPUScheduler:
    def __init__(self, runtime: SchedulerSimulatorRuntime):
        self.runtime = runtime
        self.waiting_queue: list[Request] = []
        self.running_reqs: list[Request] = []
        self.completed_reqs: list[Request] = []

    def add_requests(self, reqs: list[Request]) -> list[Request]:
        arrived = self.runtime.filter_recv_requests(reqs)
        self.waiting_queue.extend(arrived)
        return arrived

    def poll_arrivals(self) -> list[Request]:
        arrived = self.runtime.filter_recv_requests([])
        self.waiting_queue.extend(arrived)
        return arrived

    def step(self) -> bool:
        self.poll_arrivals()
        if self.waiting_queue:
            prefill_reqs = list(self.waiting_queue)
            self.waiting_queue.clear()
            batch = ScheduleBatch.extend(prefill_reqs)
            self.runtime.on_new_batch_prefill(
                batch,
                waiting_queue_len=len(self.waiting_queue),
                running_batch_len=len(self.running_reqs),
            )
            self.runtime.run_requests(prefill_reqs, decode=False)
            for req in prefill_reqs:
                req.output_ids.append(1)
            self.runtime.on_process_batch_result(batch)
            self._split_running(prefill_reqs)
            return True

        decode_reqs = [req for req in self.running_reqs if not req.is_finished()]
        if decode_reqs:
            batch = ScheduleBatch.decode(decode_reqs)
            self.runtime.run_requests(decode_reqs, decode=True)
            for req in decode_reqs:
                req.output_ids.append(1)
            self.runtime.on_process_batch_result(batch)
            self._split_running(self.running_reqs)
            return True

        self.runtime.on_new_batch_prefill(
            None,
            waiting_queue_len=len(self.waiting_queue),
            running_batch_len=len(self.running_reqs),
        )
        return bool(self.waiting_queue or self.running_reqs or self.runtime.future_queue)

    def run(self, max_steps: int | None = None) -> list[Request]:
        steps = 0
        while max_steps is None or steps < max_steps:
            made_progress = self.step()
            if not made_progress:
                break
            steps += 1
        return self.completed_reqs

    def _split_running(self, candidates: list[Request]) -> None:
        still_running = []
        for req in candidates:
            if req.is_finished():
                if req not in self.completed_reqs:
                    self.completed_reqs.append(req)
            else:
                still_running.append(req)
        self.running_reqs = still_running
