from pathlib import Path
from types import SimpleNamespace

import pytest

from sglsim.srt.simulator.native.config import NativeSimulationConfig


def test_native_config_from_server_args(tmp_path: Path):
    cfg = tmp_path / "sim.json"
    cfg.write_text('{"predictor": {"name": "aiconfigurator"}}')
    args = SimpleNamespace(
        sim_config_path=str(cfg),
        simulator_output_dir=str(tmp_path / "out"),
        simulator_mode="blocking",
        simulator_num_warmup=3,
    )
    native = NativeSimulationConfig.from_server_args(args)
    assert native.sim_config_path == str(cfg)
    assert native.output_dir == str(tmp_path / "out")
    assert native.mode == "blocking"
    assert native.num_warmup == 3


def test_native_config_rejects_bad_mode(tmp_path: Path):
    args = SimpleNamespace(
        sim_config_path="",
        simulator_output_dir=str(tmp_path / "out"),
        simulator_mode="fast",
        simulator_num_warmup=0,
    )
    with pytest.raises(ValueError, match="unsupported simulator mode"):
        NativeSimulationConfig.from_server_args(args)
