from __future__ import annotations

import itertools
from typing import Any

from sglsim.srt.scheduling.request import Request, SamplingParams
from sglsim.srt.scheduling.scheduler import CPUScheduler
from sglsim.srt.simulator.scheduler import SchedulerSimulatorRuntime


class SimulatorApp:
    def __init__(
        self,
        *,
        mode: str,
        output_dir: str,
        predictor: Any,
        num_warmup: int = 0,
        overlap_schedule: bool = False,
    ):
        self.runtime = SchedulerSimulatorRuntime(
            mode=mode,
            output_dir=output_dir,
            num_warmup=num_warmup,
            overlap_schedule=overlap_schedule,
            predictor=predictor,
        )
        self.scheduler = CPUScheduler(self.runtime)
        self._rid_counter = itertools.count(1)

    def generate(
        self,
        *,
        input_ids: list[int],
        max_new_tokens: int,
        rid: str | None = None,
        simulation_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        rid = rid or f"req-{next(self._rid_counter)}"
        simulation_params = simulation_params or {"created_time": 0.0, "total_request": 1}
        request = Request(
            rid=rid,
            input_ids=input_ids,
            sampling_params=SamplingParams(
                max_new_tokens=max_new_tokens,
                custom_params={"simulation": simulation_params},
            ),
        )

        self.scheduler.add_requests([request])
        self.scheduler.run(max_steps=max(max_new_tokens, 1) + 1)
        metrics = self.runtime.dump_outputs()
        return {
            "rid": rid,
            "output_ids": request.output_ids,
            "metrics": metrics,
        }

