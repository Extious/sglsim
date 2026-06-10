from __future__ import annotations

from sglsim.srt.simulation.manager.config import ConfigManager
from sglsim.srt.spec.model.base import ModelInfo


class MockModel:
    def forward(self) -> None:
        return None


class ModelRunner:
    def __init__(self, model_path: str) -> None:
        self.model_config = ConfigManager.get_model_info(model_path)
        self.model = MockModel()
        self.max_total_num_tokens = 65536

    def initialize(self) -> None:
        model: ModelInfo = self.model_config
        layers = max(model.num_hidden_layers, 1)
        hidden = max(model.hidden_size, 1)
        self.max_total_num_tokens = max(4096, layers * hidden // 4)
