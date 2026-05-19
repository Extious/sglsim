#!/usr/bin/env python3
"""Direct SGLSim server deployer for the fixed online benchmark."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_logs import decode_control_event, encode_control_event, get_process_log_path


DEFAULT_CONFIG: dict[str, Any] = {
    "model_path": "Qwen/Qwen3-8B",
    "dp_size": 2,
    "pp_size": 1,
    "tp_size": 1,
    "nnodes": 1,
    "server_port_base": 28000,
    "tool_call_parser": "qwen",
    "reasoning_parser": "",
    "context_length": 0,
    "json_model_override_args": "{}",
    "retry_queue_same_priority_as_waiting": False,
    "chat_template": "",
    "enable_cache_report": True,
    "enable_hicache": True,
    "hicache_size_gb": 16,
    "hicache_ratio": 2.0,
    "kv_backup": "none",
    "hicache_storage_backend": "",
    "hicache_storage_prefetch_policy": "",
    "remote_backup_port_base": 30000,
    "remote_backup_buffer_size_gb": 32.0,
    "quantization": "",
    "runtime_mode": "native",
    "sim_config_path": "",
    "simulator_mode": "blocking",
    "simulator_num_warmup": 0,
    "wait_timeout": 1800,
    "log_dir": "",
    "venv_dir": ".venv",
    "load_balance_method": "",
}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_json_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = dict(DEFAULT_CONFIG)
    server = raw.get("server", {})
    if isinstance(server, dict):
        cfg.update(server)
    cfg["kv_backup"] = raw.get("kv_backup", cfg["kv_backup"])
    storage = raw.get("hicache_storage", {})
    if isinstance(storage, dict):
        cfg["hicache_storage_backend"] = storage.get("backend", "")
        cfg["hicache_storage_prefetch_policy"] = storage.get("prefetch_policy", "")
        cfg["remote_backup_port_base"] = storage.get(
            "remote_backup_port_base",
            cfg["remote_backup_port_base"],
        )
        cfg["remote_backup_buffer_size_gb"] = storage.get(
            "remote_backup_buffer_size_gb",
            cfg["remote_backup_buffer_size_gb"],
        )
    cfg["name"] = raw.get("name", "")
    return cfg


def merge_cli_overrides(cfg: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cfg)
    for key, value in args.items():
        if value is not None:
            merged[key] = value
    return merged


def load_config(argv: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="SGLSim server deployer")
    parser.add_argument("config", nargs="?", help="Path or config directory name")
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--dp-size", dest="dp_size", type=int, default=None)
    parser.add_argument("--pp-size", dest="pp_size", type=int, default=None)
    parser.add_argument("--tp-size", dest="tp_size", type=int, default=None)
    parser.add_argument("--nnodes", dest="nnodes", type=int, default=None)
    parser.add_argument("--hicache-size-gb", dest="hicache_size_gb", type=int, default=None)
    parser.add_argument("--hicache-ratio", dest="hicache_ratio", type=float, default=None)
    parser.add_argument("--tool-call-parser", dest="tool_call_parser", default=None)
    parser.add_argument("--reasoning-parser", dest="reasoning_parser", default=None)
    parser.add_argument("--context-length", dest="context_length", type=int, default=None)
    parser.add_argument("--json-model-override-args", dest="json_model_override_args", default=None)
    parser.add_argument("--retry-queue-same-priority-as-waiting", action="store_true", default=None)
    parser.add_argument("--chat-template", dest="chat_template", default=None)
    parser.add_argument("--enable-cache-report", dest="enable_cache_report", action="store_true", default=None)
    parser.add_argument("--disable-cache-report", dest="disable_cache_report", action="store_true", default=None)
    parser.add_argument("--enable-hicache", dest="enable_hicache", action="store_true", default=None)
    parser.add_argument("--disable-hicache", dest="disable_hicache", action="store_true", default=None)
    parser.add_argument("--kv-backup", dest="kv_backup", default=None)
    parser.add_argument("--hicache-storage-backend", dest="hicache_storage_backend", default=None)
    parser.add_argument("--hicache-storage-prefetch-policy", dest="hicache_storage_prefetch_policy", default=None)
    parser.add_argument("--remote-backup-port-base", dest="remote_backup_port_base", type=int, default=None)
    parser.add_argument("--remote-backup-buffer-size-gb", dest="remote_backup_buffer_size_gb", type=float, default=None)
    parser.add_argument("--quantization", dest="quantization", default=None)
    parser.add_argument(
        "--runtime-mode",
        dest="runtime_mode",
        choices=["native", "legacy"],
        default=None,
    )
    parser.add_argument("--sim-config-path", dest="sim_config_path", default=None)
    parser.add_argument("--simulator-mode", dest="simulator_mode", default=None)
    parser.add_argument("--simulator-num-warmup", dest="simulator_num_warmup", type=int, default=None)
    parser.add_argument("--load-balance-method", dest="load_balance_method", default=None)
    parser.add_argument("--server-port-base", dest="server_port_base", type=int, default=None)
    parser.add_argument("--wait-timeout", dest="wait_timeout", type=int, default=None)
    parser.add_argument("--log-dir", dest="log_dir", default=None)
    parser.add_argument("--venv-dir", dest="venv_dir", default=None)
    args = parser.parse_args(argv)

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            cfg_path = Path(__file__).resolve().parent.parent / "config" / args.config / "server.json"
        cfg = load_json_config(cfg_path) if cfg_path.exists() else dict(DEFAULT_CONFIG)
    else:
        cfg = dict(DEFAULT_CONFIG)

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "disable_cache_report", "disable_hicache"}
        and value is not None
    }
    cfg = merge_cli_overrides(cfg, overrides)
    if args.disable_cache_report:
        cfg["enable_cache_report"] = False
    if args.disable_hicache:
        cfg["enable_hicache"] = False
    return cfg


def build_stages(
    nnodes: int,
    dp_size: int,
    pp_size: int,
    tp_size: int,
    kv_backup: str,
    node_rank: int,
    node_name: str,
    server_port: int,
    head_node: str,
) -> list[dict[str, Any]]:
    del nnodes, node_rank, head_node
    return [
        {
            "dp_rank": dp_rank,
            "pp_rank": pp_rank,
            "tp_rank": tp_rank,
            "node_rank": 0,
            "node": node_name,
            "node_url": f"http://{node_name}:{server_port}",
            "head_server_url": f"http://{node_name}:{server_port}",
            "kill_pattern": "",
            "kv_backup": kv_backup,
        }
        for dp_rank in range(int(dp_size))
        for pp_rank in range(int(pp_size))
        for tp_rank in range(int(tp_size))
    ]


def build_launch_cmd(
    cfg: dict[str, Any],
    model_path: str,
    node_rank: int,
    total_nodes: int,
    server_port: int,
    dist_init_addr: str,
    effective_hicache_extra_config: str,
    py_bin: str,
) -> list[str]:
    del node_rank, dist_init_addr, effective_hicache_extra_config
    runtime_mode = str(cfg.get("runtime_mode", "native") or "native")
    if runtime_mode == "native":
        cmd = [
            "uv",
            "run",
            "--extra",
            "native",
            "--extra",
            "aiconfigurator",
            "python",
        ]
    elif runtime_mode == "legacy":
        cmd = [py_bin]
    else:
        raise ValueError(f"unsupported runtime_mode={runtime_mode!r}")

    cmd.extend(
        [
        "-m",
        "sglsim.launch_server",
        "--model-path",
        model_path,
        "--dp-size",
        str(cfg["dp_size"]),
        "--pp-size",
        str(cfg["pp_size"]),
        "--tp-size",
        str(cfg["tp_size"]),
        "--host",
        "0.0.0.0",
        "--port",
        str(server_port),
        "--nnodes",
        str(total_nodes),
        "--simulator-output-dir",
        str(cfg.get("log_dir", "/tmp/sglsim/output")),
        ]
    )
    if cfg.get("enable_cache_report"):
        cmd.append("--enable-cache-report")
    if cfg.get("enable_hicache"):
        cmd.append("--enable-hierarchical-cache")
    if int(cfg.get("hicache_size_gb", 0) or 0) > 0:
        cmd.extend(["--hicache-size", str(cfg["hicache_size_gb"])])
    if cfg.get("hicache_ratio"):
        cmd.extend(["--hicache-ratio", str(cfg["hicache_ratio"])])
    if cfg.get("hicache_storage_backend"):
        cmd.extend(["--hicache-storage-backend", str(cfg["hicache_storage_backend"])])
    if cfg.get("hicache_storage_prefetch_policy"):
        cmd.extend(["--hicache-storage-prefetch-policy", str(cfg["hicache_storage_prefetch_policy"])])
    if str(cfg.get("kv_backup", "none")) == "remote_backup":
        remote_port = int(cfg.get("remote_backup_port_base", 30000) or 30000)
        extra_config = {
            "remote_backup_mode": "client",
            "remote_backup_url": f"localhost:{remote_port}",
            "remote_backup_port": remote_port,
            "remote_backup_buffer_size_gb": float(
                cfg.get("remote_backup_buffer_size_gb", 32.0) or 32.0
            ),
        }
        cmd.extend(
            [
                "--hicache-storage-backend-extra-config",
                json.dumps(extra_config, sort_keys=True),
            ]
        )
    if cfg.get("load_balance_method"):
        cmd.extend(["--load-balance-method", str(cfg["load_balance_method"])])
    if cfg.get("reasoning_parser"):
        cmd.extend(["--reasoning-parser", str(cfg["reasoning_parser"])])
    context_length = int(cfg.get("context_length", 0) or 0)
    if context_length > 0:
        cmd.extend(["--context-length", str(context_length)])
    model_override_args = str(cfg.get("json_model_override_args", "") or "").strip()
    if model_override_args and model_override_args != "{}":
        cmd.extend(["--json-model-override-args", model_override_args])
    if cfg.get("retry_queue_same_priority_as_waiting"):
        cmd.append("--retry-queue-same-priority-as-waiting")
    if cfg.get("chat_template"):
        cmd.extend(["--chat-template", str(cfg["chat_template"])])
    if cfg.get("quantization"):
        cmd.extend(["--quantization", str(cfg["quantization"])])
    if cfg.get("tool_call_parser"):
        cmd.extend(["--tool-call-parser", str(cfg["tool_call_parser"])])
    if cfg.get("sim_config_path"):
        cmd.extend(["--sim-config-path", str(cfg["sim_config_path"])])
        if cfg.get("simulator_mode"):
            cmd.extend(["--simulator-mode", str(cfg["simulator_mode"])])
        cmd.extend(["--simulator-num-warmup", str(int(cfg.get("simulator_num_warmup", 0) or 0))])
    return cmd


def build_remote_backup_cmd(cfg: dict[str, Any]) -> list[str]:
    return [
        "uv",
        "run",
        "--extra",
        "native",
        "python",
        "-m",
        "sglsim.srt.mem_cache.storage.remote_backup.launch_remote_backup_server",
        "--port",
        str(int(cfg.get("remote_backup_port_base", 30000) or 30000)),
        "--buffer-size-gb",
        str(float(cfg.get("remote_backup_buffer_size_gb", 32.0) or 32.0)),
    ]


def _python_bin(cfg: dict[str, Any]) -> str:
    venv_dir = Path(str(cfg.get("venv_dir", ".venv")))
    candidate = venv_dir / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _start_log_reader(proc: subprocess.Popen, log_path: Path) -> threading.Thread:
    def reader() -> None:
        assert proc.stdout is not None
        with log_path.open("ab") as fh:
            for raw_line in proc.stdout:
                data = raw_line if isinstance(raw_line, bytes) else raw_line.encode("utf-8", errors="replace")
                fh.write(data)
                fh.flush()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return thread


def _wait_healthy(url: str, proc: subprocess.Popen, timeout_s: int) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with opener.open(url, timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return False


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(sys.argv[1:] if argv is None else argv)
    log_dir = Path(str(cfg.get("log_dir") or "/tmp/sglsim/online-logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    server_port = int(cfg.get("server_port_base", 28000))
    py_bin = _python_bin(cfg)
    model_path = str(cfg.get("model_path") or "simulator-model")
    cmd = build_launch_cmd(
        cfg,
        model_path=model_path,
        node_rank=0,
        total_nodes=int(cfg.get("nnodes", 1)),
        server_port=server_port,
        dist_init_addr="",
        effective_hicache_extra_config="",
        py_bin=py_bin,
    )
    remote_proc: subprocess.Popen | None = None
    if str(cfg.get("kv_backup", "none")) == "remote_backup":
        remote_cmd = build_remote_backup_cmd(cfg)
        remote_log = get_process_log_path(log_dir, "backup_server")
        remote_log.write_bytes(b"")
        log(f"Launching SGLSim remote backup server: {' '.join(remote_cmd)}")
        remote_env = os.environ.copy()
        remote_env["NO_PROXY"] = remote_env.get(
            "NO_PROXY", "localhost,127.0.0.1,0.0.0.0,::1"
        )
        remote_env["no_proxy"] = remote_env.get("no_proxy", remote_env["NO_PROXY"])
        remote_env["PYTHONUNBUFFERED"] = "1"
        remote_proc = subprocess.Popen(
            remote_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=remote_env,
            text=True,
            bufsize=1,
        )
        _start_log_reader(remote_proc, remote_log)
        time.sleep(1.0)
        if remote_proc.poll() is not None:
            log(f"ERROR: remote backup server exited early (rc={remote_proc.returncode})")
            return 1

    log(f"Launching SGLSim server: {' '.join(cmd)}")

    env = os.environ.copy()
    env["NO_PROXY"] = env.get("NO_PROXY", "localhost,127.0.0.1,0.0.0.0,::1")
    env["no_proxy"] = env.get("no_proxy", env["NO_PROXY"])
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    server_log = get_process_log_path(log_dir, "server")
    server_log.write_bytes(b"")
    _start_log_reader(proc, server_log)

    server_url = f"http://127.0.0.1:{server_port}"
    if not _wait_healthy(server_url + "/health", proc, int(cfg.get("wait_timeout", 1800))):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if remote_proc is not None and remote_proc.poll() is None:
            remote_proc.terminate()
        return 1

    manifest = {
        "server_url": server_url,
        "kv_backup": str(cfg.get("kv_backup", "none")),
        "dp_size": int(cfg.get("dp_size", 1)),
        "pp_size": int(cfg.get("pp_size", 1)),
        "tp_size": int(cfg.get("tp_size", 1)),
        "nnodes": int(cfg.get("nnodes", 1)),
        "stages": build_stages(
            int(cfg.get("nnodes", 1)),
            int(cfg.get("dp_size", 1)),
            int(cfg.get("pp_size", 1)),
            int(cfg.get("tp_size", 1)),
            str(cfg.get("kv_backup", "none")),
            0,
            "127.0.0.1",
            server_port,
            "127.0.0.1",
        ),
    }
    print(encode_control_event("server_ready", manifest=manifest), flush=True)

    stop = False

    def _stop(_sig: int, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop and proc.poll() is None:
        time.sleep(0.5)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if remote_proc is not None and remote_proc.poll() is None:
        remote_proc.terminate()
        try:
            remote_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            remote_proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
