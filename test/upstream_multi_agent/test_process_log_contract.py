from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "benchmark" / "online" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_process_log_paths_are_run_scoped_under_one_log_root(tmp_path, monkeypatch):
    from process_logs import PROCESS_LOG_NAMES, get_process_log_paths

    monkeypatch.setenv("SGLSIM_RUN_ID", "run-123")

    paths = get_process_log_paths(tmp_path)

    assert set(paths) == set(PROCESS_LOG_NAMES)
    assert {path.parent for path in paths.values()} == {tmp_path}
    assert {path.name for path in paths.values()} == {
        "main_run-123.log",
        "server_run-123.log",
        "backup_server_run-123.log",
        "injector_run-123.log",
        "client_run-123.log",
    }


def test_process_log_path_allows_explicit_env_override(tmp_path, monkeypatch):
    from process_logs import get_process_log_path

    override = tmp_path / "custom-server.log"
    monkeypatch.setenv("SGLSIM_ONLINE_SERVER_LOG", str(override))
    monkeypatch.setenv("SGLSIM_RUN_ID", "run-123")

    assert get_process_log_path(tmp_path, "server") == override
