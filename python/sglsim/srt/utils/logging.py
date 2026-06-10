from __future__ import annotations

import json
import logging
import logging.config
import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglsim.srt.server_args import ServerArgs


class SimClockFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str,
        datefmt: str | None = None,
        *,
        use_projected_time: bool = False,
    ) -> None:
        super().__init__(fmt, datefmt)
        self.use_projected_time = use_projected_time

    @staticmethod
    def _render_epoch(epoch: float, datefmt: str | None) -> str:
        whole_seconds = int(epoch)
        milliseconds = int(round((epoch - whole_seconds) * 1000))
        if milliseconds == 1000:
            whole_seconds += 1
            milliseconds = 0
        time_tuple = time.localtime(whole_seconds)
        rendered = time.strftime(datefmt or "%Y-%m-%d %H:%M:%S", time_tuple)
        if _log_ms_enabled():
            return f"{rendered}.{milliseconds:03d}"
        return rendered

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        if self.use_projected_time:
            sim_time_s = float(getattr(record, "sim_time_s", 0.0))
            base = float(os.getenv("SGLSIM_LOG_BASE_TIME_EPOCH_S", record.created))
            projected = base + sim_time_s
            wall_rendered = self._render_epoch(record.created, datefmt)
            proj_rendered = self._render_epoch(projected, datefmt)
            return f"{wall_rendered} | {proj_rendered}"
        return super().formatTime(record, datefmt)


def _log_ms_enabled() -> bool:
    return os.getenv("SGLSIM_LOG_MS", "").lower() in ("1", "true", "yes")


def _log_formatter(
    fmt: str,
    *,
    use_projected_time: bool = False,
) -> SimClockFormatter:
    return SimClockFormatter(
        fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        use_projected_time=use_projected_time,
    )


def _use_projected_log_time() -> bool:
    return os.getenv("SGLSIM_LOG_PROFILE", "").lower() == "sglang_projected"


class _FlushingFileHandler(logging.FileHandler):
    def __init__(self, filename: str, *, encoding: str = "utf-8") -> None:
        super().__init__(filename, mode="a", encoding=encoding)

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class _FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
            self.flush()
        except ValueError:
            self.stream = sys.stdout
            super().emit(record)
            self.flush()


def _maybe_file_handler(formatter: logging.Formatter) -> logging.Handler | None:
    log_path = os.environ.get("SGLSIM_SERVER_LOG")
    if not log_path:
        return None
    handler = _FlushingFileHandler(log_path)
    handler.setFormatter(formatter)
    return handler


def _stream_handler(formatter: logging.Formatter) -> logging.Handler:
    handler = _FlushingStreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    return handler


def _attach_log_handlers(
    log: logging.Logger,
    formatter: logging.Formatter,
) -> None:
    file_handler = _maybe_file_handler(formatter)
    if file_handler is not None:
        log.addHandler(file_handler)
        return
    log.addHandler(_stream_handler(formatter))


def scheduler_logger(
    name: str,
    server_args: ServerArgs,
    dp_rank: int,
) -> logging.Logger:
    logger_name = f"{name}.dp{dp_rank}"
    log = logging.getLogger(logger_name)
    log.handlers.clear()
    log.filters.clear()

    use_projected_time = _use_projected_log_time()
    maybe_ms = (
        "" if use_projected_time else (".%(msecs)03d" if _log_ms_enabled() else "")
    )
    log.setLevel(getattr(logging, server_args.log_level.upper(), logging.INFO))
    log.propagate = False
    formatter = _log_formatter(
        f"[%(asctime)s{maybe_ms} DP{dp_rank}] %(message)s",
        use_projected_time=use_projected_time,
    )
    _attach_log_handlers(log, formatter)
    return log


def configure_logger(server_args: ServerArgs, prefix: str = "") -> None:
    config_path = os.getenv("SGLSIM_LOGGING_CONFIG_PATH")
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"SGLSIM_LOGGING_CONFIG_PATH={config_path} does not exist"
            )
        with open(config_path, encoding="utf-8") as file:
            logging.config.dictConfig(json.load(file))
        return

    use_projected_time = _use_projected_log_time()
    maybe_ms = (
        "" if use_projected_time else (".%(msecs)03d" if _log_ms_enabled() else "")
    )
    log_format = f"[%(asctime)s{maybe_ms}{prefix}] %(message)s"
    formatter = _log_formatter(
        log_format,
        use_projected_time=use_projected_time,
    )
    root = logging.getLogger()
    root.handlers.clear()
    _attach_log_handlers(root, formatter)
    root.setLevel(getattr(logging, server_args.log_level.upper(), logging.INFO))
    for name in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)
