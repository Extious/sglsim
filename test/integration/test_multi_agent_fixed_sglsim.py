from __future__ import annotations

import sys
from pathlib import Path


def test_online_deploy_builds_sglsim_launch_command_by_default():
    src_root = Path(__file__).resolve().parents[2] / "benchmark" / "online" / "src"
    sys.path.insert(0, str(src_root))
    from benchmark.online.src.server.deploy_server import build_launch_cmd

    cfg = {
        "dp_size": 2,
        "pp_size": 1,
        "tp_size": 1,
        "kv_backup": "remote_backup",
        "enable_cache_report": True,
        "enable_hicache": True,
        "hicache_size_gb": 4,
        "hicache_ratio": 0.0,
        "hicache_storage_backend": "remote_backup",
        "hicache_storage_prefetch_policy": "wait_complete",
        "remote_backup_port_base": 30000,
        "remote_backup_buffer_size_gb": 8.0,
        "load_balance_method": "round_robin",
        "reasoning_parser": "qwen3",
        "context_length": 4096,
        "json_model_override_args": "{}",
        "retry_queue_same_priority_as_waiting": True,
        "chat_template": "",
        "quantization": "",
        "tool_call_parser": "qwen",
    }

    cmd = build_launch_cmd(
        cfg,
        model_path="Qwen/Qwen3-8B",
        node_rank=0,
        total_nodes=1,
        server_port=28000,
        dist_init_addr="127.0.0.1:28100",
        effective_hicache_extra_config="",
        py_bin="python",
    )

    assert cmd[:3] == ["python", "-m", "sglsim.launch_server"]
    assert "--tp-size" in cmd
    assert "--tp" not in cmd
    assert "--disable-overlap-schedule" not in cmd
    assert "--kv-backup" in cmd
    assert "--remote-backup-port-base" in cmd
