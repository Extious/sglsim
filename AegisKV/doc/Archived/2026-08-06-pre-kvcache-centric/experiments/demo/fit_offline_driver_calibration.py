"""Fit external online-driver overhead from a no-failure paired smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def fit_driver_calibration(
    online: dict[str, Any],
    offline: dict[str, Any],
) -> dict[str, Any]:
    online_routes = {
        str(row["rid"]): dict(row)
        for workflow in online.get("workflows", ())
        for row in workflow.get("request_routes", ())
    }
    offline_routes = {
        str(row["rid"]): dict(row)
        for workflow in offline.get("workflows", ())
        for row in workflow.get("request_routes", ())
    }
    common = sorted(set(online_routes) & set(offline_routes))
    if len(common) < 2:
        raise ValueError("driver calibration requires at least two paired requests")
    samples = [
        (
            float(online_routes[rid]["completion_index"]),
            float(online_routes[rid]["completion_elapsed_s"])
            - float(offline_routes[rid]["completion_elapsed_s"]),
            rid,
        )
        for rid in common
    ]
    x_mean = sum(x for x, _y, _rid in samples) / len(samples)
    y_mean = sum(y for _x, y, _rid in samples) / len(samples)
    denominator = sum((x - x_mean) ** 2 for x, _y, _rid in samples)
    if denominator == 0.0:
        raise ValueError("paired requests have no completion-index variation")
    request_overhead_s = sum(
        (x - x_mean) * (y - y_mean) for x, y, _rid in samples
    ) / denominator
    startup_overhead_s = y_mean - request_overhead_s * x_mean
    startup_overhead_s = max(startup_overhead_s, 0.0)
    request_overhead_s = max(request_overhead_s, 0.0)
    residuals = [
        y - (startup_overhead_s + request_overhead_s * x)
        for x, y, _rid in samples
    ]
    return {
        "schema_version": 1,
        "fit": "completion_delta_s = startup_overhead_s + completion_index * request_overhead_s",
        "sample_count": len(samples),
        "startup_overhead_s": startup_overhead_s,
        "request_overhead_s": request_overhead_s,
        "rmse_s": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        "max_abs_error_s": max(abs(value) for value in residuals),
        "scope": "external workflow driver only; scheduler service time is unchanged",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit offline external-driver delays from a paired smoke."
    )
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit_driver_calibration(
        json.loads(args.online.read_text(encoding="utf-8")),
        json.loads(args.offline.read_text(encoding="utf-8")),
    )
    result["online_artifact"] = str(args.online.resolve())
    result["offline_artifact"] = str(args.offline.resolve())
    result["recommended_cli"] = (
        f"--driver-startup-overhead-s {result['startup_overhead_s']:.9f} "
        f"--driver-request-overhead-s {result['request_overhead_s']:.9f}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
