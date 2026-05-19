from __future__ import annotations

from sglsim.srt.simulator.predictor_constant import ConstantTimePredictor


def test_constant_predictor_prefill_and_decode():
    predictor = ConstantTimePredictor(prefill_ms=10.0, decode_ms=2.0)
    prefill = predictor.predict_requests([{"extend_length": 4, "past_kv_length": 0}])
    decode = predictor.predict_requests(
        [
            {"extend_length": 1, "past_kv_length": 10},
            {"extend_length": 1, "past_kv_length": 11},
        ]
    )
    assert prefill == 0.04
    assert decode == 0.004
