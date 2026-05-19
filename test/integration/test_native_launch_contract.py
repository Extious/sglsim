from __future__ import annotations

import argparse
from pathlib import Path

from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulator.native.bootstrap import install_native_simulation


def test_native_server_args_accept_simulator_flags(tmp_path: Path):
    cfg = tmp_path / "sim.json"
    cfg.write_text('{"predictor": {"name": "aiconfigurator"}}')

    ap = argparse.ArgumentParser()
    ServerArgs.add_cli_args(ap)
    raw = ap.parse_args(
        [
            "--model-path",
            "Qwen/Qwen3-8B",
            "--host",
            "127.0.0.1",
            "--port",
            "28000",
            "--dp-size",
            "2",
            "--sim-config-path",
            str(cfg),
            "--simulator-output-dir",
            str(tmp_path / "out"),
            "--simulator-mode",
            "blocking",
            "--simulator-num-warmup",
            "0",
        ]
    )
    args = ServerArgs.from_cli_args(raw)
    assert args.sim_config_path == str(cfg)
    assert args.simulator_output_dir == str(tmp_path / "out")
    assert args.simulator_mode == "blocking"
    assert args.simulator_num_warmup == 0


def test_install_native_simulation_installs_patch_modules(tmp_path: Path):
    cfg = tmp_path / "sim.json"
    cfg.write_text('{"predictor": {"name": "aiconfigurator"}}')

    ap = argparse.ArgumentParser()
    ServerArgs.add_cli_args(ap)
    raw = ap.parse_args(
        [
            "--model-path",
            "Qwen/Qwen3-8B",
            "--sim-config-path",
            str(cfg),
        ]
    )
    config = install_native_simulation(ServerArgs.from_cli_args(raw))

    from sglsim.srt.simulator.native import cache_controller_patch
    from sglsim.srt.simulator.native import memory_pool_patch
    from sglsim.srt.simulator.native import model_config_patch
    from sglsim.srt.simulator.native import model_runner_patch
    from sglsim.srt.simulator.native import scheduler_patch

    assert config.sim_config_path == str(cfg)
    assert model_config_patch.is_installed()
    assert model_runner_patch.is_installed()
    assert scheduler_patch.is_installed()
    assert cache_controller_patch.is_installed()
    assert memory_pool_patch.is_installed()
