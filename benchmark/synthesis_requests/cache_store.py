from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

from .utils import BENCHMARK_DIR, ClientConfig, ServerConfig, WorkloadRequest

if TYPE_CHECKING:
    from .client import PreparedRequest

CACHE_DIR = BENCHMARK_DIR / "cache"
LEGACY_CACHE_DIR = Path.home() / ".cache" / "sglsim" / "benchmark"
CACHE_SCHEMA_VERSION = "synthesis-requests-cache-v2"


def _trace_jsonl_fingerprint(client: ClientConfig) -> dict[str, Any] | None:
    if not client.trace_jsonl_path:
        return None
    path = Path(client.trace_jsonl_path)
    if not path.is_absolute():
        path = BENCHMARK_DIR.parents[1] / path
    path = path.resolve()
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
        }
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest()[:16],
    }


def _canonical_payload(client: ClientConfig, server: ServerConfig) -> dict[str, Any]:
    workload = asdict(client)
    workload["request_rate"] = (
        "inf" if client.request_rate == float("inf") else client.request_rate
    )
    trace_fingerprint = _trace_jsonl_fingerprint(client)
    if trace_fingerprint is not None:
        workload["trace_jsonl_fingerprint"] = trace_fingerprint
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model_path": server.model_path,
        "workload": workload,
    }


def cache_key(client: ClientConfig, server: ServerConfig) -> str:
    raw = json.dumps(
        _canonical_payload(client, server),
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _workload_path(key: str) -> Path:
    return CACHE_DIR / f"{key}_workload.pkl"


def _prepared_path(key: str) -> Path:
    return CACHE_DIR / f"{key}_prepared.pkl"


def _manifest_path(key: str) -> Path:
    return CACHE_DIR / f"{key}_manifest.json"


def load_workload(
    client: ClientConfig,
    server: ServerConfig,
    *,
    logger: logging.Logger | None = None,
) -> list[WorkloadRequest] | None:
    key = cache_key(client, server)
    path = _workload_path(key)
    if not path.is_file():
        return None
    if logger:
        logger.info("Loading cached GSP workload from %s", path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_workload(
    client: ClientConfig,
    server: ServerConfig,
    workload: list[WorkloadRequest],
    *,
    logger: logging.Logger | None = None,
) -> None:
    key = cache_key(client, server)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _workload_path(key)
    if logger:
        logger.info("Caching GSP workload to %s", path)
    with path.open("wb") as handle:
        pickle.dump(workload, handle)
    _write_manifest(key, client, server, workload_count=len(workload))


def load_prepared(
    client: ClientConfig,
    server: ServerConfig,
    *,
    logger: logging.Logger | None = None,
) -> list[Any] | None:
    key = cache_key(client, server)
    path = _prepared_path(key)
    if not path.is_file():
        return None
    if logger:
        logger.info("Loading cached prepared requests from %s", path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_prepared(
    client: ClientConfig,
    server: ServerConfig,
    prepared: list[Any],
    *,
    logger: logging.Logger | None = None,
) -> None:
    key = cache_key(client, server)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _prepared_path(key)
    if logger:
        logger.info("Caching prepared requests to %s", path)
    with path.open("wb") as handle:
        pickle.dump(prepared, handle)
    _write_manifest(key, client, server, prepared_count=len(prepared))


def _write_manifest(
    key: str,
    client: ClientConfig,
    server: ServerConfig,
    *,
    workload_count: int | None = None,
    prepared_count: int | None = None,
) -> None:
    manifest_path = _manifest_path(key)
    payload: dict[str, Any] = {
        "cache_key": key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _canonical_payload(client, server),
    }
    if workload_count is not None:
        payload["workload_count"] = workload_count
    if prepared_count is not None:
        payload["prepared_count"] = prepared_count
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def warn_legacy_cache(logger: logging.Logger | None) -> None:
    if not LEGACY_CACHE_DIR.is_dir():
        return
    if logger and any(LEGACY_CACHE_DIR.glob("gen_shared_prefix_*.pkl")):
        logger.info(
            "Legacy cache at %s is ignored; use %s instead",
            LEGACY_CACHE_DIR,
            CACHE_DIR,
        )
