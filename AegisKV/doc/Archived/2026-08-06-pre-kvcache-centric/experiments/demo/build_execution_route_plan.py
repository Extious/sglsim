"""Build a matched execution-route plan from a measured policy result."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_execution_route_plan(source_result: Path) -> dict[str, Any]:
    source_result = source_result.resolve()
    payload = json.loads(source_result.read_text(encoding="utf-8"))
    identity = dict(payload.get("identity") or {})
    dp_size = int(payload.get("dp_size") or identity.get("dp_size") or 0)
    if dp_size <= 0:
        raise ValueError(f"invalid source dp_size in {source_result}")
    routes: dict[str, dict[str, int]] = {}
    for workflow in payload.get("workflows", []):
        for route in workflow.get("request_routes", []):
            rid = str(route.get("rid") or "")
            rank = int(route.get("execution_dp_rank", -1))
            if not rid:
                raise ValueError(f"source result has an empty request id: {source_result}")
            if rid in routes:
                raise ValueError(f"source result has duplicate route for {rid}")
            if not 0 <= rank < dp_size:
                raise ValueError(f"source result has invalid rank {rank} for {rid}")
            routes[rid] = {"execution_dp_rank": rank}
    expected = int(payload.get("request_count", 0))
    if not routes or len(routes) != expected:
        raise ValueError(
            f"source route count {len(routes)} does not match request_count {expected}"
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_result": str(source_result),
        "source_policy": str(payload.get("policy") or "unknown"),
        "source_run_id": source_result.parent.name,
        "model": str(payload.get("model") or ""),
        "dp_size": dp_size,
        "pp_size": int(
            payload.get("pp_size") or identity.get("pp_size") or 1
        ),
        "route_count": len(routes),
        "routes": dict(sorted(routes.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_execution_route_plan(args.source_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
