"""Merge an online WCT artifact with the server's common telemetry snapshot."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def merge_online_calibration(
    wct: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    if telemetry.get("simulation_time_mode") != "wall":
        raise ValueError("calibration telemetry is not an online wall-clock snapshot")
    original_requests = int(wct.get("request_count", -1))
    retry_requests = int(wct.get("retry_summary", {}).get("count", 0))
    expected = original_requests + retry_requests
    observed = int(telemetry.get("calibration_metrics", {}).get("request_count", -2))
    if expected != observed:
        raise ValueError(
            f"WCT/telemetry request count mismatch: expected={expected} observed={observed}"
        )
    merged = dict(wct)
    merged.update(
        {
            "simulation_time_mode": "wall",
            "evidence_tier": telemetry.get(
                "evidence_tier", "sglsim-online-instrumented"
            ),
            "calibration_identity": dict(telemetry.get("identity", {})),
            "physical_topology": dict(telemetry.get("physical_topology", {})),
            "calibration_metrics": dict(telemetry.get("calibration_metrics", {})),
            "request_stats": dict(telemetry.get("request_stats", {})),
            "scheduler_telemetry": list(
                telemetry.get("scheduler_telemetry", ())
            ),
            "online_instrumentation": {
                "scheduler_telemetry_enabled": True,
                "original_request_count": original_requests,
                "retry_request_count": retry_requests,
                "observed_request_count": observed,
                "note": (
                    "Calibration-only queue emission was enabled; the normal "
                    "online server path remains disabled by default."
                ),
            },
            "online_event_history": dict(
                telemetry.get("aegis_status", {}).get("event_history", {})
            ),
        }
    )
    return merged


def _read_url(url: str, *, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attach optional online scheduler telemetry to a WCT result."
    )
    parser.add_argument("--wct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--telemetry-url",
        default="http://127.0.0.1:31000/calibration/telemetry",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    wct = json.loads(args.wct.read_text(encoding="utf-8"))
    telemetry = _read_url(args.telemetry_url, timeout_s=float(args.timeout))
    merged = merge_online_calibration(wct, telemetry)
    _atomic_write(args.output.resolve(), merged)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "requests": merged["request_count"],
                "scheduler_rows": len(merged["scheduler_telemetry"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
