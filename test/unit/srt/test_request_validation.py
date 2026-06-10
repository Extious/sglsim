import json

import pytest

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.spec.model.base import ModelInfo
from sglsim.srt.utils.request_validation import validate_generate_request


def test_validate_rejects_input_longer_than_context_len():
    model = ModelInfo.from_model_path("test-model", context_len=128)
    req = GenerateReqInput(
        rid="1",
        input_len=128,
        max_new_tokens=1,
    )
    with pytest.raises(ValueError, match="input \\(128 tokens\\) is longer"):
        validate_generate_request(req, model)


def test_validate_rejects_input_plus_output_exceeding_context_len():
    model = ModelInfo.from_model_path("test-model", context_len=128)
    req = GenerateReqInput(
        rid="1",
        input_len=100,
        max_new_tokens=28,
    )
    with pytest.raises(ValueError, match="Requested token count exceeds"):
        validate_generate_request(req, model)


def test_validate_allows_request_within_context_len():
    model = ModelInfo.from_model_path("test-model", context_len=128)
    req = GenerateReqInput(
        rid="1",
        input_len=100,
        max_new_tokens=27,
    )
    validate_generate_request(req, model)


def test_validate_skips_when_context_len_unset():
    model = ModelInfo.from_model_path("mock-model", context_len=None)
    req = GenerateReqInput(
        rid="1",
        input_len=100000,
        max_new_tokens=100000,
    )
    validate_generate_request(req, model)


def test_engine_submit_rejects_overlong_request(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(
        json.dumps({"context_length": 16}),
        encoding="utf-8",
    )
    engine = Engine(
        ServerArgs(model_path="mock-model", sim_config_path=str(cfg))
    )
    with pytest.raises(ValueError, match="input \\(20 tokens\\) is longer"):
        engine.submit(
            GenerateReqInput(
                rid="1",
                input_len=20,
                max_new_tokens=1,
                simulation={"total_request": 1},
            )
        )


def test_config_manager_uses_context_length_override(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(
        json.dumps({"context_length": 40960, "model_path": "Qwen/Qwen3-8B"}),
        encoding="utf-8",
    )
    ConfigManager.load(str(cfg))
    model = ConfigManager.get_model_info("Qwen/Qwen3-8B")
    assert model.context_len == 40960
