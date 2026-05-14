from pathlib import Path

from sglsim import __version__
from sglsim.srt.server_args import ServerArgs


def test_package_name_is_sglsim():
    assert __version__ == "0.1.0"


def test_server_args_require_existing_config(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text('{"predictor": {"name": "aiconfigurator"}}')

    args = ServerArgs.from_cli_args(["--sim-config-path", str(config)])

    assert args.sim_config_path == str(config)
    assert args.simulator_mode == "offline"
