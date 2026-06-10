from sglsim.srt.managers.schedule_batch import ForwardMode, ScheduleBatch
from sglsim.srt.spec.accelerator.base import AcceleratorInfo
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.simulation.manager.scheduler_config import SchedulerSimConfig
from sglsim.srt.time_predictor.base import InferTimePredictor, ScheduleBatch as PredBatch


class TopologyAwareTimePredictor(InferTimePredictor):
    name = "topology"

    def __init__(
        self,
        base_prefill_s: float = 0.010,
        base_decode_s: float = 0.001,
        tp_size: int = 1,
        pp_size: int = 1,
        model: ModelInfo | None = None,
        hw: AcceleratorInfo | None = None,
        config: SchedulerSimConfig | None = None,
        raw: dict | None = None,
    ) -> None:
        cfg = config or SchedulerSimConfig(tp_size=tp_size, pp_size=pp_size)
        super().__init__(model or ModelInfo(), hw or AcceleratorInfo(), cfg, raw)
        if base_prefill_s < 0 or base_decode_s < 0:
            raise ValueError("latency values must be non-negative")
        if tp_size <= 0 or pp_size <= 0:
            raise ValueError("parallel sizes must be positive")
        self.base_prefill_s = base_prefill_s
        self.base_decode_s = base_decode_s
        self._tp_size = tp_size
        self._pp_size = pp_size

    def predict_infer_time(self, batch: PredBatch) -> float:
        base = self.base_decode_s if batch.is_decode() else self.base_prefill_s
        return base * (1 / max(self._tp_size, 1)) * max(self._pp_size, 1)

    def predict_batch_latency(self, batch: ScheduleBatch) -> float:
        base = (
            self.base_prefill_s
            if batch.mode == ForwardMode.PREFILL
            else self.base_decode_s
        )
        return base * (1 / max(self._tp_size, 1)) * max(self._pp_size, 1)
