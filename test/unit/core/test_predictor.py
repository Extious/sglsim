import pytest

from sglsim.srt.managers.schedule_batch import Batch, ForwardMode, Req, SamplingParams
from sglsim.srt.time_predictor.constant import ConstantTimePredictor
from sglsim.srt.time_predictor.topology import TopologyAwareTimePredictor


def test_constant_predictor_uses_prefill_latency_for_prefill_batch():
    predictor = ConstantTimePredictor(prefill_s=0.010, decode_s=0.001)
    req = Req("r0", [1, 2, 3], SamplingParams(max_new_tokens=1))
    batch = Batch(0, 0, ForwardMode.PREFILL, [req])
    assert predictor.predict_batch_latency(batch) == 0.010


def test_constant_predictor_returns_decode_latency_for_decode_batch():
    predictor = ConstantTimePredictor(prefill_s=0.010, decode_s=0.001)
    batch = Batch(
        0,
        0,
        ForwardMode.DECODE,
        [Req("r0", [1], SamplingParams(max_new_tokens=1))],
    )
    assert predictor.predict_batch_latency(batch) == 0.001


@pytest.mark.parametrize(
    ("prefill_s", "decode_s"),
    [(-0.001, 0.001), (0.010, -0.001)],
)
def test_constant_predictor_rejects_negative_latencies(prefill_s, decode_s):
    with pytest.raises(ValueError):
        ConstantTimePredictor(prefill_s=prefill_s, decode_s=decode_s)


def test_topology_aware_predictor_scales_prefill_latency_by_tp_and_pp():
    predictor = TopologyAwareTimePredictor(
        base_prefill_s=0.010, base_decode_s=0.001, tp_size=2, pp_size=4
    )
    batch = Batch(
        0,
        0,
        ForwardMode.PREFILL,
        [Req("r0", [1, 2, 3], SamplingParams(max_new_tokens=1))],
    )
    assert predictor.predict_batch_latency(batch) == 0.020


@pytest.mark.parametrize(
    ("tp_size", "pp_size"),
    [(0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_topology_aware_predictor_rejects_non_positive_parallel_sizes(tp_size, pp_size):
    with pytest.raises(ValueError):
        TopologyAwareTimePredictor(
            base_prefill_s=0.010, base_decode_s=0.001, tp_size=tp_size, pp_size=pp_size
        )
