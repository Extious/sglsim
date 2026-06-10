from __future__ import annotations

from sglsim.srt.utils.tokenizer_runtime import measure_decode, measure_encode


def test_measure_encode_mock_returns_ids():
    ids, dur = measure_encode("mock-model", "hello world")
    assert len(ids) > 0
    assert dur >= 0.0


def test_measure_decode_mock():
    text, dur = measure_decode("mock-model", [1, 2, 3])
    assert text == ""
    assert dur >= 0.0
