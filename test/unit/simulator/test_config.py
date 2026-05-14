import json
from pathlib import Path

import pytest

from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulator.config import SimulatorSettings


def test_settings_load_config(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"predictor": {"name": "aiconfigurator"}}))
    args = ServerArgs.from_cli_args(["--sim-config-path", str(config)])

    settings = SimulatorSettings.from_server_args(args)

    assert settings.enabled is True
    assert settings.mode == "offline"
    assert settings.raw_config["predictor"]["name"] == "aiconfigurator"


def test_settings_reject_non_aiconfigurator(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"predictor": {"name": "dummy"}}))
    args = ServerArgs.from_cli_args(["--sim-config-path", str(config)])

    with pytest.raises(ValueError, match="predictor.name must be aiconfigurator"):
        SimulatorSettings.from_server_args(args)
