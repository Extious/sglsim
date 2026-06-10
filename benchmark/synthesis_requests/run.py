from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
import json
import urllib.request
from datetime import datetime
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]
_PYTHON_ROOT = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_ROOT):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from benchmark.synthesis_requests.client import (  # noqa: E402
    build_prepared_requests,
    run_client,
    wait_for_server_ready,
)
from benchmark.synthesis_requests.server import (  # noqa: E402
    build_server_command,
    start_server,
    stop_server,
)
from benchmark.synthesis_requests.utils import (  # noqa: E402
    CLIENT_LOG_PATH,
    DEFAULT_CONFIG_DIR,
    DEFAULT_RESULTS_ROOT,
    SERVER_LOG_PATH,
    load_config_dir,
    new_run_dir,
    setup_file_logger,
    write_metrics_json,
    write_request_details_csv,
    write_simulator_config,
)

_SERVER_READY_TIMEOUT_S = 300.0


def _post_fault_control(host: str, port: int, path: str, dp_rank: int) -> None:
    url = f"http://{host}:{int(port)}{path}"
    body = json.dumps({"dp_rank": int(dp_rank)}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=30) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"{path} failed with status={resp.status}")


def _completed_count(host: str, port: int) -> int:
    url = f"http://{host}:{int(port)}/metrics"
    req = urllib.request.Request(url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    return int(raw.get("completed", 0) or 0)


def _start_fault_trigger_thread(
    *,
    server_cfg,
    runner_log,
    stop_event: threading.Event,
) -> threading.Thread | None:
    raw = server_cfg.raw.get("fault_injection")
    if not isinstance(raw, dict):
        ft = server_cfg.raw.get("fault_tolerance", {})
        raw = ft.get("trigger") if isinstance(ft, dict) else None
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None

    dp_rank = int(raw.get("fault_dp_rank", raw.get("dp_rank", 0)))
    inject_after_completed = raw.get("inject_after_completed")
    inject_after_s = raw.get("inject_after_s")
    recover_after_s = raw.get("recover_after_s")

    def _run() -> None:
        start = time.monotonic()
        injected = False
        try:
            while not stop_event.is_set() and not injected:
                if inject_after_completed is not None:
                    completed = _completed_count(server_cfg.host, server_cfg.port)
                    if completed >= int(inject_after_completed):
                        injected = True
                elif inject_after_s is not None:
                    injected = time.monotonic() - start >= float(inject_after_s)
                else:
                    injected = True
                if not injected:
                    stop_event.wait(timeout=0.25)
            if stop_event.is_set() or not injected:
                return
            runner_log.info("Injecting GPU failure dp_rank=%d", dp_rank)
            _post_fault_control(
                server_cfg.host,
                server_cfg.port,
                "/simulate_gpu_failure",
                dp_rank,
            )
            if recover_after_s is not None:
                if not stop_event.wait(timeout=float(recover_after_s)):
                    runner_log.info("Recovering GPU dp_rank=%d", dp_rank)
                    _post_fault_control(
                        server_cfg.host,
                        server_cfg.port,
                        "/simulate_gpu_recovery",
                        dp_rank,
                    )
        except Exception as exc:
            runner_log.exception("Fault trigger failed: %s", exc)

    thread = threading.Thread(
        target=_run,
        name="synthesis-fault-trigger",
        daemon=True,
    )
    thread.start()
    return thread


def run_experiment(
    config_dir: Path,
    results_root: Path,
    *,
    log_level: str = "info",
    disable_overlap_schedule: bool = False,
) -> Path:
    client_cfg, server_cfg = load_config_dir(config_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = new_run_dir(results_root)
    run_server_log_path = run_dir / "server.log"
    run_server_stderr_log_path = run_dir / "server.stderr.log"

    runner_log = setup_file_logger(
        "sglsim.benchmark.synthesis_requests.runner",
        CLIENT_LOG_PATH,
        log_level,
    )
    setup_file_logger(
        "sglsim.benchmark.synthesis_requests.client",
        CLIENT_LOG_PATH,
        log_level,
    )
    setup_file_logger(
        "sglsim.benchmark.synthesis_requests.server",
        SERVER_LOG_PATH,
        log_level,
    )

    sim_config_path = write_simulator_config(
        run_dir,
        client=client_cfg,
        server=server_cfg,
        timestamp=timestamp,
        disable_overlap_schedule=disable_overlap_schedule or None,
    )
    runner_log.info(
        "Experiment config_dir=%s run_dir=%s",
        config_dir,
        run_dir,
    )

    prepared = build_prepared_requests(client_cfg, server_cfg)
    runner_log.info("Prepared %d requests", len(prepared))

    process_holder: list[subprocess.Popen | None] = [None]
    trigger_stop = threading.Event()
    trigger_thread: threading.Thread | None = None
    sigint_count = 0

    def _sigint_cleanup(signum: int, frame: object) -> None:
        nonlocal sigint_count
        sigint_count += 1
        proc = process_holder[0]
        force = sigint_count >= 2
        if proc is not None and proc.poll() is None:
            runner_log.info(
                "SIGINT received (count=%d); stopping server force=%s",
                sigint_count,
                force,
            )
            stop_server(proc, run_server_log_path, force=force)
        raise KeyboardInterrupt

    old_sigint = signal.signal(signal.SIGINT, _sigint_cleanup)
    process = start_server(
        build_server_command(
            server=server_cfg,
            sim_config_path=sim_config_path,
            log_level=log_level,
            disable_overlap_schedule=disable_overlap_schedule,
        ),
        run_server_log_path,
        stderr_path=run_server_stderr_log_path,
    )
    process_holder[0] = process
    try:
        wait_for_server_ready(
            server_cfg.host,
            server_cfg.port,
            timeout_s=_SERVER_READY_TIMEOUT_S,
            process=process,
            log_path=run_server_log_path,
        )
        trigger_thread = _start_fault_trigger_thread(
            server_cfg=server_cfg,
            runner_log=runner_log,
            stop_event=trigger_stop,
        )
        rows = run_client(
            server_cfg.host,
            server_cfg.port,
            prepared,
            client_cfg,
        )
        csv_path = write_request_details_csv(run_dir, rows)
        write_metrics_json(run_dir, rows)
        runner_log.info("Wrote %s", csv_path)
        runner_log.info("results_dir=%s", run_dir)
        return run_dir
    finally:
        trigger_stop.set()
        if trigger_thread is not None:
            trigger_thread.join(timeout=2)
        proc = process_holder[0]
        if proc is not None and proc.poll() is None:
            stop_server(proc, run_server_log_path, force=True)
        process_holder[0] = None
        signal.signal(signal.SIGINT, old_sigint)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SGLSim synthesis_requests benchmark")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing client.json and server.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help="Disable overlap scheduler in the launched SGLSim server.",
    )
    args = parser.parse_args(argv)

    try:
        run_experiment(
            args.config_dir,
            args.results_root,
            log_level=args.log_level,
            disable_overlap_schedule=bool(args.disable_overlap_schedule),
        )
    except Exception as exc:
        setup_file_logger(
            "sglsim.benchmark.synthesis_requests.runner",
            CLIENT_LOG_PATH,
            args.log_level,
        ).exception("Experiment failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
