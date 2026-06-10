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
