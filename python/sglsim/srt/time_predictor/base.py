from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sglsim.srt.spec.accelerator.base import AcceleratorInfo
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig


@dataclass
class ScheduleRequest:
    extend_length: int = 0
    past_kv_length: int = 0


@dataclass
class ScheduleBatch:
    reqs: list[ScheduleRequest] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def is_decode(self) -> bool:
        for req in self.reqs:
            if req.extend_length > 1:
                return False
        return True

    def is_prefill(self) -> bool:
        return not self.is_decode()


class InferTimePredictor(ABC):
    name = "base"

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerSimConfig,
        raw: dict | None = None,
    ) -> None:
        self.model = model
        self.hw = hw
        self.config = config
        self.raw = raw or {}

    @abstractmethod
    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        ...


def physical_pipeline_stage_latency(
    predicted_pipeline_latency_s: float,
    *,
    pp_size: int,
) -> float:
    """Partition one predictor's pipeline latency across physical PP stages.

    SGLSim predictors historically returned the latency for the configured PP
    pipeline as one logical Scheduler step.  Once each PP stage owns an
    independent process and causally ordered forward, charging that value in
    every process multiplies first-microbatch latency by ``pp_size``.  Until a
    calibrated per-stage breakdown is available, use the equal-layer partition
    that preserves the predictor's end-to-end value while exposing real
    pipeline overlap between consecutive microbatches.
    """

    if pp_size <= 0:
        raise ValueError(f"pp_size must be positive, got {pp_size}")
    if predicted_pipeline_latency_s < 0:
        raise ValueError(
            "predicted_pipeline_latency_s must be nonnegative, got "
            f"{predicted_pipeline_latency_s}"
        )
    return float(predicted_pipeline_latency_s) / int(pp_size)
