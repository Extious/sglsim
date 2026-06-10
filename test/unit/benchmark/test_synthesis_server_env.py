from pathlib import Path

from benchmark.synthesis_requests.server import _server_env


def test_server_env_enables_projected_log_milliseconds_by_default(monkeypatch):
    monkeypatch.delenv("SGLSIM_LOG_MS", raising=False)

    env = _server_env(Path("server.log"))

    assert env["SGLSIM_LOG_PROFILE"] == "sglang_projected"
    assert env["SGLSIM_LOG_MS"] == "1"
