from __future__ import annotations

import pytest

from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.time_predictor.constant import ConstantTimePredictor


@pytest.fixture(autouse=True)
def _stub_aiconfigurator_predictor(monkeypatch):
    def _get_predictor(cls, model_path=None):
        if cls._predictor is None:
            cls._predictor = ConstantTimePredictor(prefill_s=0.0032768, decode_s=0.048608)
        return cls._predictor

    monkeypatch.setattr(
        ConfigManager,
        "get_inference_time_predictor",
        classmethod(_get_predictor),
    )
    yield
    ConfigManager._predictor = None
