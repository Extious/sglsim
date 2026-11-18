#!/usr/bin/env python3
"""Audit recovery-target concurrency without using invalid timing values."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _verified_file(record: object, *, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is missing")
    path = Path(str(record.get("path") or "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != str(record.get("sha256") or ""):
        raise ValueError(f"{label} sha256 mismatch")
    return path


def _require_markers(path: Path, markers: tuple[str, ...], *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise ValueError(f"{label} semantic markers missing: {missing}")


def audit(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    auditor_path = Path(__file__).resolve()
    config = _object(config_path)
    if config.get("design_semantics") != "source_owned_independent_hbm_dram_v2":
        raise ValueError("dispatch audit does not use the current design")

    csv_path = _verified_file(config.get("failure_rows"), label="failure rows")
    mooncake_path = _verified_file(
        config.get("frozen_mooncake_router"), label="Mooncake Router"
    )
    sglsim_path = _verified_file(
        config.get("sglsim_failure_evaluator"), label="SGLsim evaluator"
    )
    patch_path = _verified_file(
        config.get("staged_runtime_patch"), label="staged runtime patch"
    )
    _require_markers(
        mooncake_path,
        ("for request_id in order:", "await self._replay_one("),
        label="Mooncake global-serial recovery",
    )
    _require_markers(
        sglsim_path,
        (
            "by_target: dict[str, list[ReplayRequest]]",
            "for target, requests in sorted(by_target.items()):",
            "-decisions[protection][row.row.rid].visible_harm_ms",
        ),
        label="SGLsim target-local Slack/harm recovery",
    )
    _require_markers(
        patch_path,
        ("target_parallel_due_time_queues", "await asyncio.gather("),
        label="staged target-parallel patch",
    )

    policy = str(config.get("policy") or "")
    expected = config.get("expected_profile")
    if not policy or not isinstance(expected, dict):
        raise ValueError("policy or expected profile is missing")
    by_event: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"event_id", "policy", "request_id", "activated_target"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("failure rows lack dispatch identity columns")
        for row in reader:
            if row["policy"] != policy:
                continue
            event_id = str(row["event_id"])
            request_id = str(row["request_id"])
            target = str(row["activated_target"])
            if not event_id or not request_id or not target:
                raise ValueError("dispatch identity cannot be empty")
            key = (event_id, request_id)
            if key in seen:
                raise ValueError(f"duplicate event/request dispatch row: {key}")
            seen.add(key)
            by_event[event_id].append((request_id, target))
    observed_rows = sum(len(rows) for rows in by_event.values())
    if observed_rows != int(expected.get("row_count", -1)):
        raise ValueError(f"failure row count mismatch: {observed_rows}")
    if len(by_event) != int(expected.get("event_count", -1)):
        raise ValueError(f"failure event count mismatch: {len(by_event)}")

    points: list[dict[str, int]] = []
    targets: set[str] = set()
    for rows in by_event.values():
        counts = Counter(target for _request, target in rows)
        targets.update(counts)
        points.append(
            {
                "request_count": len(rows),
                "target_queue_count": len(counts),
                "maximum_target_queue_depth": max(counts.values()),
            }
        )
    multi_request = [row for row in points if row["request_count"] > 1]
    opportunities = [
        row
        for row in points
        if row["target_queue_count"] > 1
        and row["request_count"] > row["maximum_target_queue_depth"]
    ]
    mean_unit_parallelism_all = statistics.mean(
        row["request_count"] / row["maximum_target_queue_depth"]
        for row in points
    )
    mean_unit_parallelism_multi = statistics.mean(
        row["request_count"] / row["maximum_target_queue_depth"]
        for row in multi_request
    )

    return {
        "schema_version": 1,
        "evidence_class": "recovery_dispatch_topology_diagnostic_not_performance_result",
        "status": "PASS",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "design_semantics": config["design_semantics"],
        "auditor": {"path": str(auditor_path), "sha256": _sha256(auditor_path)},
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "inputs": {
            "failure_rows": {"path": str(csv_path), "sha256": _sha256(csv_path)},
            "frozen_mooncake_router": {
                "path": str(mooncake_path),
                "sha256": _sha256(mooncake_path),
                "observed_semantics": "global_serial_replay_loop",
            },
            "sglsim_failure_evaluator": {
                "path": str(sglsim_path),
                "sha256": _sha256(sglsim_path),
                "observed_semantics": "independent_target_due_time_queues",
            },
            "staged_runtime_patch": {
                "path": str(patch_path),
                "sha256": _sha256(patch_path),
                "observed_semantics": "target_parallel_due_time_queues",
            },
        },
        "policy": policy,
        "topology": {
            "failure_event_count": len(points),
            "request_dispatch_count": observed_rows,
            "distinct_activated_targets": len(targets),
            "events_with_multiple_requests": len(multi_request),
            "events_with_multiple_targets": sum(
                row["target_queue_count"] > 1 for row in points
            ),
            "events_with_cross_target_hol_opportunity": len(opportunities),
            "maximum_requests_per_event": max(
                row["request_count"] for row in points
            ),
            "maximum_target_queues_per_event": max(
                row["target_queue_count"] for row in points
            ),
            "maximum_target_local_queue_depth": max(
                row["maximum_target_queue_depth"] for row in points
            ),
            "mean_target_queues_per_event": statistics.mean(
                row["target_queue_count"] for row in points
            ),
            "mean_unit_service_parallelism_ceiling_all_events": (
                mean_unit_parallelism_all
            ),
            "mean_unit_service_parallelism_ceiling_multi_request_events": (
                mean_unit_parallelism_multi
            ),
        },
        "runtime_alignment_status": "TARGET_QUEUE_MISMATCH_WITH_STAGED_FIX",
        "performance_results_present": False,
        "formal_ready": False,
        "paper_ready": False,
        "evidence_boundary": (
            "Only request-to-target identities and queue cardinalities are used. "
            "Recovery durations, invalid RTX4090 coefficients, WCT, and speedup "
            "are not consumed or reported. Unit-service parallelism is a "
            "topological ceiling, not a performance prediction."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
