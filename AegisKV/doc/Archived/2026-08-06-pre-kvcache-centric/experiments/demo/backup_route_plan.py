"""Build and load calibration-only AegisKV backup-target route plans."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_backup_route_plan(source_result: Path) -> dict[str, Any]:
    source = source_result.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    routes: dict[str, int] = {}
    for event in payload.get("aegis_profile_events", ()):
        if not isinstance(event, dict) or event.get("kind") != "router_top2":
            continue
        rid = str(event.get("rid", ""))
        backup_rank = int(event.get("backup_dp_rank", -1))
        if not rid or backup_rank < 0:
            continue
        existing = routes.get(rid)
        if existing is not None and existing != backup_rank:
            raise ValueError(
                f"conflicting backup target for {rid}: {existing} vs {backup_rank}"
            )
        routes[rid] = backup_rank
    for workflow in payload.get("workflows", ()):
        if not isinstance(workflow, dict):
            continue
        for route in workflow.get("request_routes", ()):
            if not isinstance(route, dict):
                continue
            rid = str(route.get("rid", ""))
            backup_rank = int(
                route.get(
                    "planned_backup_dp_rank",
                    route.get("backup_dp_rank", -1),
                )
            )
            if not rid or backup_rank < 0:
                continue
            existing = routes.get(rid)
            if existing is not None and existing != backup_rank:
                raise ValueError(
                    f"conflicting backup target for {rid}: "
                    f"{existing} vs {backup_rank}"
                )
            routes[rid] = backup_rank
    expected = int(payload.get("request_count", 0))
    if len(routes) != expected:
        raise ValueError(
            f"backup route count {len(routes)} does not match request_count {expected}"
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_only": True,
        "source_result": str(source),
        "source_policy": str(payload.get("policy", "")),
        "model": str(payload.get("model", "")),
        "dp_size": int(payload.get("dp_size", 1)),
        "pp_size": int(payload.get("pp_size", 1)),
        "route_count": len(routes),
        "routes": {
            rid: {"backup_dp_rank": rank} for rid, rank in sorted(routes.items())
        },
    }


def load_backup_route_plan(path: Path, *, dp_size: int) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported backup route plan schema: {path}")
    if not bool(payload.get("calibration_only", False)):
        raise ValueError("backup route plan must be marked calibration_only")
    if int(payload.get("dp_size", -1)) != int(dp_size):
        raise ValueError(
            f"backup route plan dp_size={payload.get('dp_size')}, expected {dp_size}"
        )
    routes = {}
    for rid, raw in payload.get("routes", {}).items():
        rank = int(raw.get("backup_dp_rank", -1) if isinstance(raw, dict) else raw)
        if not 0 <= rank < int(dp_size):
            raise ValueError(f"invalid backup rank {rank} for {rid}")
        routes[str(rid)] = rank
    if not routes:
        raise ValueError(f"backup route plan has no routes: {path}")
    return routes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a calibration-only backup-target plan from an online result."
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_backup_route_plan(args.source_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
