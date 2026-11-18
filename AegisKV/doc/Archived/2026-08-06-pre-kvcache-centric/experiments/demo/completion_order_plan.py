"""Build/load calibration-only workflow response-completion order plans."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_completion_order_plan(source_result: Path) -> dict[str, Any]:
    source = source_result.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    workflows: dict[str, list[dict[str, Any]]] = {}
    request_count = 0
    for workflow in payload.get("workflows", ()):
        if not isinstance(workflow, dict):
            continue
        workflow_id = str(workflow.get("workflow_id", ""))
        rows = [
            {
                "rid": str(row.get("rid", "")),
                "completion_index": int(row.get("completion_index", -1)),
            }
            for row in workflow.get("request_routes", ())
            if isinstance(row, dict)
        ]
        rows.sort(key=lambda row: (row["completion_index"], row["rid"]))
        if not workflow_id or any(not row["rid"] for row in rows):
            raise ValueError("online result contains an empty workflow/request ID")
        expected_indices = list(range(1, len(rows) + 1))
        if [row["completion_index"] for row in rows] != expected_indices:
            raise ValueError(
                f"workflow {workflow_id} completion indices are not contiguous"
            )
        workflows[workflow_id] = rows
        request_count += len(rows)
    if request_count != int(payload.get("request_count", -1)):
        raise ValueError(
            "completion-order request count does not match source result: "
            f"rows={request_count} source={payload.get('request_count')}"
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_only": True,
        "causal_boundary": (
            "freeze external workflow response-delivery order only; routing, "
            "placement, capacity admission, ACK, and scheduler logic remain live"
        ),
        "source_result": str(source),
        "source_policy": str(payload.get("policy", "")),
        "model": str(payload.get("model", "")),
        "dp_size": int(payload.get("dp_size", 1)),
        "pp_size": int(payload.get("pp_size", 1)),
        "workflow_count": len(workflows),
        "request_count": request_count,
        "workflows": workflows,
    }


def load_completion_order_plan(
    path: Path,
    *,
    selected_request_ids: set[str],
) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported completion-order schema: {path}")
    if not bool(payload.get("calibration_only", False)):
        raise ValueError("completion-order plan must be marked calibration_only")
    order: dict[str, int] = {}
    for rows in payload.get("workflows", {}).values():
        if not isinstance(rows, list):
            raise ValueError("completion-order workflow entry must be a list")
        for row in rows:
            rid = str(row.get("rid", "")) if isinstance(row, dict) else ""
            index = (
                int(row.get("completion_index", -1))
                if isinstance(row, dict)
                else -1
            )
            if not rid or index <= 0 or rid in order:
                raise ValueError(f"invalid/duplicate completion-order row: {row!r}")
            order[rid] = index
    missing = sorted(selected_request_ids - set(order))
    if missing:
        raise ValueError(
            f"completion-order plan misses {len(missing)} selected requests; "
            f"first={missing[0]}"
        )
    extra = set(order) - selected_request_ids
    return {rid: order[rid] for rid in sorted(selected_request_ids)} if extra else order


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract calibration-only workflow response completion order."
    )
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_completion_order_plan(args.source_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
