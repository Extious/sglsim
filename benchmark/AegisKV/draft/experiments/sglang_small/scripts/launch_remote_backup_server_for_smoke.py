#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import signal
import sys
import time
from pathlib import Path


def load_remote_backup_server(sglang_dir: Path):
    module_path = (
        sglang_dir
        / "python"
        / "sglang"
        / "srt"
        / "mem_cache"
        / "storage"
        / "remote_backup"
        / "remote_backup_server.py"
    )
    spec = importlib.util.spec_from_file_location(
        "aegiskv_remote_backup_server", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load remote backup server from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RemoteBackupServer


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remote backup server launcher for AegisKV SGLang smokes"
    )
    parser.add_argument("--sglang-dir", type=Path, default=Path(os.environ.get("SGLANG_DIR", "/home/comp/25482351/sglang")))
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--buffer-size-gb", type=float, default=32.0)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--request-aware", type=parse_bool, default=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("aegiskv_remote_backup_smoke")

    RemoteBackupServer = load_remote_backup_server(args.sglang_dir)
    logger.info(
        "Starting remote backup smoke server: port=%d, buffer_size_gb=%.1f, "
        "page_size=%d, request_aware=%s, sglang_dir=%s",
        args.port,
        args.buffer_size_gb,
        args.page_size,
        args.request_aware,
        args.sglang_dir,
    )
    server = RemoteBackupServer(
        port=args.port,
        max_buffer_size_gb=args.buffer_size_gb,
        page_size=args.page_size,
        request_aware=args.request_aware,
    )
    server.start()

    def shutdown(signum, frame):
        logger.info("Shutting down remote backup smoke server")
        server.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())
