from sglsim.srt.managers.schedule_batch import ForwardMode, ScheduleBatch
from sglsim.srt.spec.accelerator.base import AcceleratorInfo
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.time_predictor.base import InferTimePredictor, ScheduleBatch as PredBatch


class ConstantTimePredictor(InferTimePredictor):
    name = "constant"

    def __init__(
        self,
        prefill_s: float = 0.010,
        decode_s: float = 0.001,
        model: ModelInfo | None = None,
        hw: AcceleratorInfo | None = None,
        config: SchedulerSimConfig | None = None,
        raw: dict | None = None,
    ) -> None:
        super().__init__(
            model or ModelInfo(),
            hw or AcceleratorInfo(),
            config or SchedulerSimConfig(),
            raw,
        )
        if prefill_s < 0 or decode_s < 0:
            raise ValueError("latency values must be non-negative")
        self.prefill_s = prefill_s
        self.decode_s = decode_s

    def predict_infer_time(self, batch: PredBatch) -> float:
        if batch.is_decode():
            return self.decode_s
        return self.prefill_s

    def predict_batch_latency(self, batch: ScheduleBatch) -> float:
        if batch.mode == ForwardMode.DECODE:
            return self.decode_s
        return self.prefill_s
