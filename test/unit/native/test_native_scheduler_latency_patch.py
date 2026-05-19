from __future__ import annotations

from types import SimpleNamespace

import torch

from sglsim.srt.simulator.native.scheduler_patch import install
from sglsim.srt.simulator.scheduler import build_prediction_requests


class Mode:
    def __init__(self, name):
        self.name = name

    def is_extend(self):
        return self.name == "extend"

    def is_decode(self):
        return self.name == "decode"


def test_build_prediction_requests_for_extend():
    req = SimpleNamespace(extend_input_len=32, prefix_indices=[1, 2], output_ids=[3])
    batch = SimpleNamespace(forward_mode=Mode("extend"), reqs=[req])
    assert build_prediction_requests(batch) == [{"extend_length": 32, "past_kv_length": 3}]


def test_build_prediction_requests_for_decode():
    req = SimpleNamespace(extend_input_len=32, prefix_indices=[1, 2], output_ids=[3, 4])
    batch = SimpleNamespace(forward_mode=Mode("decode"), reqs=[req])
    assert build_prediction_requests(batch) == [{"extend_length": 1, "past_kv_length": 4}]


def test_build_prediction_requests_accept_tensor_prefix_indices():
    req = SimpleNamespace(
        extend_input_len=32,
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        output_ids=[3, 4],
    )
    batch = SimpleNamespace(forward_mode=Mode("extend"), reqs=[req])
    assert build_prediction_requests(batch) == [{"extend_length": 32, "past_kv_length": 2}]


def test_scheduler_patch_installs_once_without_constructing_scheduler():
    install(SimpleNamespace(mode="blocking", output_dir="/tmp/sglsim/native-test", num_warmup=0))
    install(SimpleNamespace(mode="blocking", output_dir="/tmp/sglsim/native-test", num_warmup=0))
