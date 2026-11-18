"""Append-only machine-readable evidence journals for runtime events."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping


class AppendOnlyJsonlJournal:
    """Flush every JSON record without changing runtime event semantics.

    The caller owns sequence numbers and timestamps.  This class only provides
    a thread-safe, append-only durability boundary.  Reusing an existing path
    appends another run rather than truncating prior evidence; formal manifests
    must therefore bind a unique path to each run.
    """

    def __init__(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved
        self._handle = resolved.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._records_written = 0
        self._closed = False

    @property
    def records_written(self) -> int:
        with self._lock:
            return self._records_written

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot append to a closed event journal")
            self._handle.write(line + "\n")
            self._handle.flush()
            self._records_written += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "path": str(self.path),
                "records_written": self._records_written,
                "closed": self._closed,
                "mode": "append_only_jsonl_flush_each_record",
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._handle.flush()
            self._handle.close()
            self._closed = True

