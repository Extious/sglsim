"""Fixed online benchmark pipeline: server + fixed client + fault injection."""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import importlib.util
import struct
import zlib
from pathlib import Path

from client.config import FixedClientConfig, FixedRunnerConfig
from failure.config import FaultInjectionConfig, FaultInjectorConfig
from process_logs import (
    copy_process_logs,
    decode_control_event,
    get_process_log_path,
    process_log_env,
    redirect_current_process_to_log,
)
from server.config import ServerConfig
from utils import build_experiment_config


PLOT_SCRIPT_DEPENDENCIES = {
    "backup_cache_hits.py": ("matplotlib",),
}


def _plot_dependencies_available(script_name: str) -> bool:
    return all(
        importlib.util.find_spec(dep) is not None
        for dep in PLOT_SCRIPT_DEPENDENCIES.get(script_name, ())
    )


def _repo_root() -> Path:
    env = os.environ.get("SGLSIM_REPO_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def _ensure_run_id(default: str) -> str:
    run_id = os.environ.get("SGLSIM_RUN_ID", "").strip() or default
    os.environ["SGLSIM_RUN_ID"] = run_id
    return run_id


class ExperimentPipeline:
    def __init__(
        self,
        *,
        server_cfg: ServerConfig,
        fi_cfg: FaultInjectionConfig,
        fixed_cfg: FixedClientConfig,
        kv_backup: str,
        log_dir: Path,
        output_dir: Path,
        timestamp: str,
        venv: Path,
        deploy_config_name: str,
    ) -> None:
        self.server_cfg = server_cfg
        self.fi_cfg = fi_cfg
        self.fixed_cfg = fixed_cfg
        self.kv_backup = kv_backup
        self.log_dir = log_dir
        run_dir_name = Path(deploy_config_name).name if deploy_config_name else timestamp
        self.output_dir = output_dir / run_dir_name
        self.timestamp = timestamp
        self._venv = venv
        self._deploy_config_name = deploy_config_name
        self._src_root = Path(__file__).resolve().parent
        self._server_proc: subprocess.Popen | None = None
        self._client_proc: subprocess.Popen | None = None
        self._injector_proc: subprocess.Popen | None = None
        self._logs_copied = False

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        self._log_path = get_process_log_path(log_dir, "main")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        def _sig_handler(_sig: int, _frame) -> None:
            for proc in (self._client_proc, self._injector_proc, self._server_proc):
                if proc and proc.poll() is None:
                    proc.terminate()
            raise SystemExit(130)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def _start_log_reader(
        self,
        proc: subprocess.Popen,
        log_path: Path,
        events: "queue.Queue[dict] | None" = None,
    ) -> threading.Thread:
        def _reader() -> None:
            assert proc.stdout is not None
            with log_path.open("ab") as fh:
                for raw_line in proc.stdout:
                    data = raw_line if isinstance(raw_line, bytes) else raw_line.encode("utf-8", errors="replace")
                    fh.write(data)
                    fh.flush()
                    line = data.decode("utf-8", errors="replace").rstrip("\n")
                    record = decode_control_event(line)
                    if record is not None and events is not None:
                        events.put(record)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        return thread

    def _wait_for_event(
        self,
        proc: subprocess.Popen,
        events: "queue.Queue[dict]",
        event_name: str,
        timeout_s: int,
    ) -> dict | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                self.log(f"ERROR: subprocess exited early while waiting for {event_name} (rc={proc.returncode})")
                return None
            try:
                record = events.get(timeout=1.0)
            except queue.Empty:
                continue
            if record.get("event") == event_name:
                return record
        self.log(f"ERROR: timed out waiting for {event_name} after {timeout_s}s")
        return None

    def run(self) -> int:
        srv_cfg = self.server_cfg
        fi_cfg = self.fi_cfg
        fixed_cfg = self.fixed_cfg
        out = self.output_dir
        log_dir = self.log_dir
        src_root = self._src_root
        repo = _repo_root()
        py = str(self._venv / "bin" / "python")
        if not Path(py).exists():
            py = sys.executable
        deploy_py = str(repo / "benchmark" / "online" / "src" / "server" / "deploy_server.py")

        self.log(f"{'=' * 60}")
        self.log(f"EXPERIMENT topology={srv_cfg.topology_tag()} kv_backup={self.kv_backup}")
        self.log(f"  result_dir: {out}")
        self.log(f"  log_dir:    {log_dir}")

        client_rc = 1
        try:
            self.log("Step 1: Start SGLSim server subprocess...")
            server_flags = srv_cfg.build_deploy_flags(
                model_path=srv_cfg.model_path,
                server_port=28000,
                kv_backup=self.kv_backup,
                venv_dir=str(self._venv),
            )
            server_cmd = [py, deploy_py, "--log-dir", str(log_dir)]
            if self._deploy_config_name:
                server_cmd.append(self._deploy_config_name)
            server_cmd.extend(server_flags)
            log_env = os.environ.copy()
            log_env.update(process_log_env(log_dir))
            server_log = get_process_log_path(log_dir, "server")
            server_log.write_bytes(b"")
            server_events: queue.Queue[dict] = queue.Queue()
            self._server_proc = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=log_env,
                text=True,
                bufsize=1,
            )
            self._start_log_reader(self._server_proc, server_log, server_events)
            server_ready = self._wait_for_event(
                self._server_proc,
                server_events,
                "server_ready",
                timeout_s=srv_cfg.wait_timeout,
            )
            if server_ready is None:
                return 1
            manifest = server_ready.get("manifest") or {}
            if not isinstance(manifest, dict) or not manifest.get("server_url"):
                self.log("ERROR: malformed server_ready control event")
                return 1
            head_url = str(manifest["server_url"])
            self.log(f"  Server ready at {head_url}")

            self.log("Step 2: Start injector and fixed client subprocesses...")
            injector_cfg = FaultInjectorConfig(
                server_url=head_url,
                stage_manifest=None,
                kv_backup=self.kv_backup,
                output_dir=out,
                inject_after_job=fi_cfg.after_job,
                inject_after_task=fi_cfg.after_task,
                timeline_after_start_s=getattr(fi_cfg, "timeline_after_start_s", ""),
                inject_delay=fi_cfg.delay,
                fault_dp_rank=fi_cfg.dp_rank,
                fault_pp_rank=fi_cfg.pp_rank,
                fault_tp_rank=fi_cfg.tp_rank,
                method="api",
                recovery_delay=fi_cfg.recovery_delay,
                recover_after_job=fi_cfg.recover_after_job,
                recover_after_task=getattr(fi_cfg, "recover_after_task", ""),
                server_topology=srv_cfg.topology_tag(),
                stage_manifest_data=manifest,
            )
            injector_flags = injector_cfg.build_worker_flags()
            inj_log = get_process_log_path(log_dir, "injector")
            inj_log.write_bytes(b"")
            injector_events: queue.Queue[dict] = queue.Queue()
            self._injector_proc = subprocess.Popen(
                [py, "-m", "failure.injector_worker"] + injector_flags
                + ["--control-host", "127.0.0.1", "--control-port", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(src_root),
                env=log_env,
                text=True,
                bufsize=1,
            )
            self._start_log_reader(self._injector_proc, inj_log, injector_events)
            injector_ready = self._wait_for_event(
                self._injector_proc,
                injector_events,
                "injector_ready",
                timeout_s=60,
            )
            if injector_ready is None:
                return 1
            control_url = str(injector_ready.get("control_url", ""))
            if not control_url:
                self.log("ERROR: malformed injector_ready control event")
                return 1

            worker_cfg = FixedRunnerConfig(
                server_url=head_url,
                model_path=srv_cfg.model_path,
                input_len=int(fixed_cfg.input_len),
                output_len=int(fixed_cfg.output_len),
                num_requests=int(fixed_cfg.num_requests),
                app_workers=int(fixed_cfg.app_workers),
                seed=int(fixed_cfg.seed),
                ignore_eos=int(fixed_cfg.ignore_eos),
                output_dir=out.resolve(),
                control_url=control_url,
            )
            cli_log = get_process_log_path(log_dir, "client")
            cli_log.write_bytes(b"")
            self._client_proc = subprocess.Popen(
                [py, "-m", "client.fixed_worker"] + worker_cfg.build_worker_flags(),
                stdout=cli_log.open("ab"),
                stderr=subprocess.STDOUT,
                cwd=str(src_root),
                env=log_env,
            )
            client_rc = self._client_proc.wait()
            self.log(f"  Client exited (rc={client_rc})")
            if self._injector_proc and self._injector_proc.poll() is None:
                self._injector_proc.terminate()
            try:
                self._injector_proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._injector_proc.kill()

            self.log("Step 3: Generate plots...")
            self._generate_plots(out)
            self.log(f"Experiment complete. Results in {out}")
        finally:
            for proc in (self._client_proc, self._injector_proc, self._server_proc):
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            self._client_proc = None
            self._injector_proc = None
            self._server_proc = None
            self._copy_logs_to_result()

        return client_rc

    def _copy_logs_to_result(self) -> None:
        if self._logs_copied:
            return
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        copy_process_logs(self.log_dir, self.output_dir / "logs")
        self._logs_copied = True

    def _generate_plots(self, out: Path) -> None:
        cache_csv = out / "cache_hits.csv"
        if cache_csv.is_file() and cache_csv.stat().st_size > 0:
            self._run_plot(
                "backup_cache_hits.py",
                out / "figures" / "backup_cache_hit_profile.png",
                "--cache-csv",
                str(cache_csv),
            )

    def _run_plot(self, script_name: str, output: Path, *extra_args: str) -> None:
        script = self._src_root / "plot" / script_name
        if not script.is_file():
            return
        if not _plot_dependencies_available(script_name):
            self.log(f"  Plot {script_name}: missing optional plotting dependency; writing fallback PNG")
            self._write_fallback_png(output)
            return
        cmd = [sys.executable, str(script), "--output", str(output), *extra_args]
        with self._log_path.open("a") as log_fh:
            subprocess.run(
                cmd,
                check=False,
                cwd=str(self._src_root),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )

    def _write_fallback_png(self, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        width, height = 640, 240
        rows = []
        for y in range(height):
            row = bytearray([0])
            for x in range(width):
                if y < 40:
                    color = (42, 79, 118)
                elif 90 <= y <= 150 and 80 <= x <= 560:
                    color = (84, 162, 75)
                elif 150 < y <= 175 and 80 <= x <= 560:
                    color = (228, 87, 86)
                else:
                    color = (248, 250, 252)
                row.extend(color)
            rows.append(bytes(row))
        raw = b"".join(rows)

        def chunk(kind: bytes, data: bytes) -> bytes:
            checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )
        output.write_bytes(png)


def main() -> int:
    try:
        cfg = build_experiment_config(sys.argv[1:])
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _ensure_run_id(time.strftime("%Y%m%d_%H%M%S"))
    main_log_handle = redirect_current_process_to_log(
        get_process_log_path(cfg.log_dir, "main")
    )
    try:
        pipeline = ExperimentPipeline(
            server_cfg=cfg.server,
            fi_cfg=cfg.fault_injection,
            fixed_cfg=cfg.fixed,
            kv_backup=cfg.kv_backup,
            log_dir=cfg.log_dir,
            output_dir=cfg.output_dir,
            timestamp=run_id,
            venv=cfg.venv,
            deploy_config_name=cfg.deploy_config_name,
        )
        return pipeline.run()
    finally:
        main_log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
