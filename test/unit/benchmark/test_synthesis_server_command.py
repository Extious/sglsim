from pathlib import Path

from benchmark.synthesis_requests import run as run_mod
from benchmark.synthesis_requests.server import build_server_command
from benchmark.synthesis_requests.utils import ServerConfig


def _server_config() -> ServerConfig:
    return ServerConfig(
        model_path="mock-model",
        host="127.0.0.1",
        port=30000,
        dp_size=1,
        tp_size=1,
        pp_size=1,
        raw={},
    )


def test_build_server_command_can_disable_overlap_schedule():
    cmd = build_server_command(
        server=_server_config(),
        sim_config_path=Path("/tmp/sim.json"),
        disable_overlap_schedule=True,
    )

    assert "--disable-overlap-schedule" in cmd


def test_build_server_command_default_keeps_overlap_enabled():
    cmd = build_server_command(
        server=_server_config(),
        sim_config_path=Path("/tmp/sim.json"),
    )

    assert "--disable-overlap-schedule" not in cmd


def test_run_main_forwards_disable_overlap_flag(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_experiment(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Path("/tmp/fake-run")

    monkeypatch.setattr(run_mod, "run_experiment", _fake_run_experiment)

    rc = run_mod.main(["--disable-overlap-schedule"])

    assert rc == 0
    assert captured["kwargs"]["disable_overlap_schedule"] is True
