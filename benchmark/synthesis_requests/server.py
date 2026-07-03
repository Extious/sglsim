from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .utils import ServerConfig, read_log_tail

logger = logging.getLogger("sglsim.benchmark.synthesis_requests.server")

_STOP_TIMEOUT_S = 5.0
_KILL_TIMEOUT_S = 5.0


def build_server_command(
    *,
    server: ServerConfig,
    sim_config_path: Path,
    streaming_arrival: bool = True,
    log_level: str = "info",
    disable_overlap_schedule: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "sglsim.launch_server",
        "--model-path",
        server.model_path,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--tp-size",
        str(server.tp_size),
        "--pp-size",
        str(server.pp_size),
        "--dp-size",
        str(server.dp_size),
        "--sim-config-path",
        str(sim_config_path),
        "--log-level",
        log_level,
    ]
    if streaming_arrival:
        command.append("--streaming-arrival")
    if disable_overlap_schedule:
        command.append("--disable-overlap-schedule")
    return command


def _server_env(log_path: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["SGLSIM_SERVER_LOG"] = str(log_path)
    env.setdefault("SGLSIM_LOG_PROFILE", "sglang_projected")
    env.setdefault("SGLSIM_LOG_BASE_TIME_EPOCH_S", str(time.time()))
    env.setdefault("SGLSIM_LOG_MS", "1")
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    return env


def _launch_server_pids_for_port(port: int) -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "sglsim.launch_server"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Failed to inspect stale launch_server processes: %s", exc)
        return []
    if result.returncode not in (0, 1):
        return []

    pids: list[int] = []
    for raw_pid in result.stdout.splitlines():
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        try:
            raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (OSError, FileNotFoundError):
            continue
        args = [part.decode("utf-8", errors="replace") for part in raw_cmdline.split(b"\0") if part]
        if "sglsim.launch_server" not in " ".join(args):
            continue
        if "--port" not in args:
            continue
        idx = args.index("--port")
        if idx + 1 >= len(args):
            continue
        try:
            proc_port = int(args[idx + 1])
        except ValueError:
            continue
        if proc_port == int(port):
            pids.append(pid)
    return pids


def cleanup_stale_launch_servers(port: int) -> None:
    """Stop leftover launch_server processes for the same port only."""
    pids = _launch_server_pids_for_port(port)
    if not pids:
        return
    killed = 0
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except ProcessLookupError:
                pass
    if killed:
        logger.info(
            "Terminated %d stale sglsim.launch_server process(es) on port %d",
            killed,
            int(port),
        )
        time.sleep(0.5)


def cleanup_all_stale_launch_servers() -> None:
    """Compatibility helper for explicit whole-system cleanup."""
    try:
        result = subprocess.run(
            ["pkill", "-f", "sglsim.launch_server"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Failed to pkill stale launch_server processes: %s", exc)
        return
    if result.returncode == 0:
        logger.info("Terminated stale sglsim.launch_server processes")
        time.sleep(0.5)


def _signal_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _warn_if_port_in_use(host: str, port: int) -> None:
    url = f"http://{host}:{int(port)}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=2) as resp:
            if resp.status < 500:
                logger.warning(
                    "Port %s:%s already serves /health; stop stale processes first",
                    host,
                    port,
                )
    except Exception:
        return


def start_server(
    command: list[str],
    log_path: Path,
    *,
    stderr_path: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    logger.info("Starting server: %s", " ".join(command))
    host, port = "127.0.0.1", 30000
    for idx, arg in enumerate(command):
        if arg == "--host" and idx + 1 < len(command):
            host = command[idx + 1]
        elif arg == "--port" and idx + 1 < len(command):
            port = int(command[idx + 1])
    cleanup_stale_launch_servers(port)
    _warn_if_port_in_use(host, port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    resolved_stderr_path = stderr_path or (log_path.parent / "server.stderr.log")
    resolved_stderr_path.write_text("", encoding="utf-8")
    stderr_file = resolved_stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
        env=_server_env(log_path, extra_env=extra_env),
        start_new_session=True,
    )
    process._stderr_file = stderr_file  # type: ignore[attr-defined]
    process._sglsim_port = port  # type: ignore[attr-defined]
    return process


def stop_server(
    process: subprocess.Popen,
    log_path: Path | None = None,
    *,
    force: bool = False,
) -> None:
    if process.poll() is not None:
        logger.info("Server exited with code %s", process.returncode)
        cleanup_stale_launch_servers(int(getattr(process, "_sglsim_port", 30000)))
        return
    pgid = process.pid
    if force:
        logger.info("Force-killing server process group (pgid=%s)", pgid)
        _signal_process_group(pgid, signal.SIGKILL)
    else:
        logger.info("Terminating server process group (pgid=%s)", pgid)
        _signal_process_group(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=_KILL_TIMEOUT_S if force else _STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.warning(
            "Server did not exit after %s; sending SIGKILL to pgid=%s",
            "SIGKILL" if force else "SIGTERM",
            pgid,
        )
        _signal_process_group(pgid, signal.SIGKILL)
        try:
            process.wait(timeout=_KILL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.error("Server process group did not exit after SIGKILL")
    cleanup_stale_launch_servers(int(getattr(process, "_sglsim_port", 30000)))
    logger.info("Server stopped with code %s", process.returncode)
    if process.returncode not in (0, -signal.SIGTERM, -signal.SIGKILL):
        tail = read_log_tail(log_path)
        if tail:
            logger.error("Last server log lines:\n%s", tail)
