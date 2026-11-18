from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


_RESET = "\033[0m"
_COLORS = {
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "bold": "\033[1m",
}


def paint(text: object, color: str) -> str:
    if os.getenv("NO_COLOR") or not os.isatty(1):
        return str(text)
    return f"{_COLORS[color]}{text}{_RESET}"


def request_json(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = None
    method = "GET"
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc


def wait_for_server(base_url: str, *, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if request_json(base_url, "/health", timeout=2.0).get("status") == "ok":
                return
        except Exception as exc:  # server may still be starting
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"SGLSim server not ready at {base_url}: {last_error}")


def wait_for_hint(
    base_url: str,
    backup_key: str,
    *,
    timeout: float = 2.0,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = request_json(base_url, "/aegis/status", timeout=5.0)
        for hint in status.get("recovery_hints", []):
            if str(hint.get("backup_key")) == str(backup_key):
                return hint
        time.sleep(0.03)
    return None
