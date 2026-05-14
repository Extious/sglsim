import json
import sys
import types
from pathlib import Path

import pytest

from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulator.predictor import (
    ScheduleBatch,
    ScheduleRequest,
    build_simulator_predictor,
    resolve_model_info,
    resolve_scheduler_config,
)
from sglsim.srt.simulator.types import DataType


def test_schedule_batch_classifies_decode():
    batch = ScheduleBatch([ScheduleRequest(1, 4), ScheduleRequest(1, 8)])

    assert batch.is_decode()
    assert batch.request_info() == [[1, 4], [1, 8]]


def test_resolve_scheduler_config_auto_dtype_uses_model_config():
    args = types.SimpleNamespace(
        dtype="auto",
        kv_cache_dtype="auto",
        mem_fraction_static=0.8,
        max_total_tokens=1024,
        tp_size=1,
        ep_size=1,
        dp_size=1,
        pp_size=1,
    )
    model_config = types.SimpleNamespace(dtype="torch.bfloat16")

    config = resolve_scheduler_config(args, model_config)

    assert config.data_type == DataType.BF16
    assert config.kv_cache_data_type == DataType.BF16


def test_resolve_model_info_mha():
    model_config = types.SimpleNamespace(
        dtype="torch.float16",
        attention_arch="MHA",
        hf_text_config={},
        model_path="sim-model",
        context_len=4096,
        hidden_size=2048,
        head_dim=128,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_key_value_heads=16,
        v_head_dim=128,
        vocab_size=32000,
    )

    model = resolve_model_info(model_config)

    assert model.model_path == "sim-model"
    assert model.attention_arch == "MHA"
    assert not model.is_mla()


def test_build_predictor_uses_aiconfigurator_adapter(tmp_path: Path):
    class DummyPredictor:
        def __init__(self, model, hw, config, **kwargs):
            self.model = model
            self.hw = hw
            self.config = config
            self.kwargs = kwargs

        def predict_requests(self, requests):
            return 1.5 + len(requests)

    fake_module = types.ModuleType("sglsim.srt.simulator.predictor_aiconfigurator")
    fake_module.AIConfiguratorTimePredictor = DummyPredictor
    old_module = sys.modules.get("sglsim.srt.simulator.predictor_aiconfigurator")
    sys.modules["sglsim.srt.simulator.predictor_aiconfigurator"] = fake_module
    try:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "platform": {"accelerator": {"name": "unit", "hbm_capacity_gb": 80}},
            "predictor": {"name": "aiconfigurator", "database_mode": "EMPIRICAL"},
            "scheduler": {"tp_size": 4, "data_type": "BF16", "kv_cache_data_type": "FP8"},
        }))
        args = ServerArgs.from_cli_args(["--sim-config-path", str(config_path)])
        args.dtype = "float16"
        args.kv_cache_dtype = "auto"
        args.mem_fraction_static = 0.8
        args.max_total_tokens = 1024
        args.tp_size = 1
        args.ep_size = 1
        args.dp_size = 1
        args.pp_size = 1
        model_config = types.SimpleNamespace(
            dtype="torch.float16",
            attention_arch="MHA",
            hf_text_config={},
            model_path="sim-model",
            context_len=4096,
            hidden_size=2048,
            head_dim=128,
            num_attention_heads=16,
            num_hidden_layers=24,
            num_key_value_heads=16,
            v_head_dim=128,
            vocab_size=32000,
        )

        predictor = build_simulator_predictor(args, model_config)
    finally:
        if old_module is None:
            sys.modules.pop("sglsim.srt.simulator.predictor_aiconfigurator", None)
        else:
            sys.modules["sglsim.srt.simulator.predictor_aiconfigurator"] = old_module

    assert predictor.config.tp_size == 4
    assert predictor.config.data_type == DataType.BF16
    assert predictor.config.kv_cache_data_type == DataType.FP8
    assert predictor.predict_requests([{"extend_length": 1, "past_kv_length": 0}]) == 2.5


def test_build_predictor_raises_when_aiconfigurator_unavailable(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"predictor": {"name": "aiconfigurator"}}))
    args = ServerArgs.from_cli_args(["--sim-config-path", str(config_path)])
    args.dtype = "float16"
    args.kv_cache_dtype = "auto"
    args.tp_size = args.ep_size = args.dp_size = args.pp_size = 1
    model_config = types.SimpleNamespace(dtype="torch.float16", attention_arch="unsupported")

    with pytest.raises(ValueError):
        build_simulator_predictor(args, model_config)
