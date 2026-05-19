from __future__ import annotations

from types import SimpleNamespace

from sglsim import launch_server
from sglsim.srt.simulator.native.bootstrap import run_native_scheduler_process


def test_configure_native_env_sets_simulator_and_failover_paths(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("SGLSIM_SIMULATOR_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SGLANG_FAILOVER_EVENTS_FILE", raising=False)

    cfg = tmp_path / "sim.json"
    cfg.write_text("{}", encoding="utf-8")
    out = tmp_path / "out"

    launch_server._configure_native_env(
        SimpleNamespace(
            sim_config_path=str(cfg),
            simulator_output_dir=str(out),
        )
    )

    assert launch_server.os.environ["SGLSIM_SIMULATOR_CONFIG_PATH"] == str(cfg)
    assert launch_server.os.environ["SGLANG_FAILOVER_EVENTS_FILE"] == str(
        out / "failover_events.jsonl"
    )
    assert out.is_dir()


def test_configure_native_env_preserves_explicit_failover_file(monkeypatch, tmp_path):
    explicit = tmp_path / "custom.jsonl"
    monkeypatch.setenv("SGLANG_FAILOVER_EVENTS_FILE", str(explicit))

    launch_server._configure_native_env(
        SimpleNamespace(
            sim_config_path="",
            simulator_output_dir=str(tmp_path / "out"),
        )
    )

    assert launch_server.os.environ["SGLANG_FAILOVER_EVENTS_FILE"] == str(explicit)


def test_main_launches_with_native_scheduler_process(monkeypatch, tmp_path):
    captured = {}

    class FakeServerArgs:
        sim_config_path = ""
        simulator_output_dir = str(tmp_path / "out")
        simulator_mode = "blocking"
        simulator_num_warmup = 0
        model_path = "simulator-model"

        @staticmethod
        def add_cli_args(parser):
            parser.add_argument("--model-path", default="simulator-model")

        @staticmethod
        def from_cli_args(raw_args):
            return FakeServerArgs()

    def fake_launch(server_args, **kwargs):
        captured["server_args"] = server_args
        captured["scheduler"] = kwargs.get("run_scheduler_process_func")

    monkeypatch.setattr(
        "sglsim.srt.server_args.ServerArgs",
        FakeServerArgs,
    )
    monkeypatch.setattr(
        "sglsim.srt.entrypoints.http_server.launch_server",
        fake_launch,
    )
    monkeypatch.setattr(
        "sglsim.srt.simulator.native.bootstrap.install_native_simulation",
        lambda server_args: None,
    )
    monkeypatch.setattr(
        "sglsim.srt.utils.kill_process_tree",
        lambda *args, **kwargs: None,
    )

    launch_server.main(["--model-path", "simulator-model"])

    assert captured["scheduler"] is run_native_scheduler_process
