from __future__ import annotations

from sglsim.srt.simulator.scheduler import PredictionBatch


class ConstantTimePredictor:
    def __init__(self, *, prefill_ms: float = 10.0, decode_ms: float = 1.0):
        self.prefill_ms = prefill_ms
        self.decode_ms = decode_ms

    def predict_requests(self, requests: list[dict]) -> float:
        batch = PredictionBatch(requests)
        if batch.is_decode():
            return len(requests) * (self.decode_ms / 1000.0)
        total_tokens = sum(item["extend_length"] for item in requests)
        return max(total_tokens, 1) * (self.prefill_ms / 1000.0)

    def predict_infer_time(self, batch: PredictionBatch) -> float:
        if batch.is_decode():
            return batch.batch_size * (self.decode_ms / 1000.0)
        return max(batch.num_context_tokens, 1) * (self.prefill_ms / 1000.0)
