from __future__ import annotations

from pathlib import Path


def test_online_deploy_defaults_to_native_runtime():
    config_files = [
        Path("benchmark/online/src/config/remote_backup-fixed-a100/server.json"),
        Path("benchmark/online/src/config/baseline-fixed-a100/server.json"),
    ]
    for path in config_files:
        text = path.read_text(encoding="utf-8")
        assert '"runtime_mode": "native"' in text


def test_runtime_service_not_imported_by_native_launch():
    launch = Path("python/sglsim/launch_server.py").read_text(encoding="utf-8")
    assert "RuntimeService" not in launch
    assert "sglsim.srt.http.server" not in launch
