from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "benchmark" / "online" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from server.config import ServerConfig


def test_online_fixed_server_config_defaults_to_native_runtime():
    cfg = ServerConfig.from_dict({})
    assert cfg.runtime_mode == "native"


def test_online_fixed_deploy_flags_include_native_runtime():
    cfg = ServerConfig.from_dict(
        {
            "runtime_mode": "native",
            "dp_size": 2,
            "pp_size": 1,
            "tp_size": 1,
            "hicache_storage": {
                "backend": "remote_backup",
                "remote_backup_port_base": 30123,
            },
            "simulator": {"config_path": "sim.json"},
        }
    )

    flags = cfg.build_deploy_flags(
        model_path="simulator-model",
        server_port=28000,
        kv_backup="remote_backup",
        venv_dir=".venv",
    )

    assert flags[flags.index("--runtime-mode") + 1] == "native"
    assert flags[flags.index("--kv-backup") + 1] == "remote_backup"
    assert flags[flags.index("--remote-backup-port-base") + 1] == "30123"
    assert flags[flags.index("--sim-config-path") + 1] == "sim.json"
