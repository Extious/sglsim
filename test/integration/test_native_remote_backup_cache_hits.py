from __future__ import annotations

import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "benchmark" / "online" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.deploy_server import build_launch_cmd, build_remote_backup_cmd


def test_native_remote_backup_launches_dedicated_server_and_client_mode():
    cfg = {
        "runtime_mode": "native",
        "model_path": "simulator-model",
        "dp_size": 2,
        "pp_size": 1,
        "tp_size": 1,
        "kv_backup": "remote_backup",
        "enable_cache_report": True,
        "enable_hicache": True,
        "hicache_size_gb": 1,
        "hicache_ratio": 0,
        "hicache_storage_backend": "remote_backup",
        "hicache_storage_prefetch_policy": "wait_complete",
        "remote_backup_port_base": 30123,
        "remote_backup_buffer_size_gb": 4.0,
        "log_dir": "/tmp/sglsim-test",
    }

    remote_cmd = build_remote_backup_cmd(cfg)
    assert remote_cmd[:6] == [
        "uv",
        "run",
        "--extra",
        "native",
        "python",
        "-m",
    ]
    assert (
        "sglsim.srt.mem_cache.storage.remote_backup.launch_remote_backup_server"
        in remote_cmd
    )
    assert remote_cmd[remote_cmd.index("--port") + 1] == "30123"

    cmd = build_launch_cmd(
        cfg,
        model_path="simulator-model",
        node_rank=0,
        total_nodes=1,
        server_port=28000,
        dist_init_addr="",
        effective_hicache_extra_config="",
        py_bin="/ignored/python",
    )
    assert cmd[:8] == [
        "uv",
        "run",
        "--extra",
        "native",
        "--extra",
        "aiconfigurator",
        "python",
        "-m",
    ]
    assert "sglsim.launch_server" in cmd
    extra = json.loads(
        cmd[cmd.index("--hicache-storage-backend-extra-config") + 1]
    )
    assert extra["remote_backup_mode"] == "client"
    assert extra["remote_backup_url"] == "localhost:30123"
    assert extra["remote_backup_port"] == 30123
