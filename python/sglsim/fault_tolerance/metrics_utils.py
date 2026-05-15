from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_ts(raw) -> datetime | None:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None


def _sort_key(job_id: str) -> tuple[int, str]:
    if job_id.isdigit():
        return (0, f"{int(job_id):010d}")
    return (1, job_id)


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_impacted_jobs(events: list[dict[str, Any]]) -> list[str]:
    fault_ts = None
    recover_ts = None
    for event in events:
        event_type = event.get("event")
        ts = _parse_ts(event.get("timestamp"))
        if event_type == "fault_injected" and ts is not None:
            fault_ts = ts if fault_ts is None else min(fault_ts, ts)
        elif event_type == "gpu_recovered" and ts is not None and fault_ts is not None:
            if ts >= fault_ts:
                recover_ts = ts if recover_ts is None else min(recover_ts, ts)

    if fault_ts is None:
        return []
    impacted = set()
    for event in events:
        if event.get("event") != "job_completed":
            continue
        ts = _parse_ts(event.get("timestamp"))
        if ts is None:
            continue
        if ts >= fault_ts and (recover_ts is None or ts <= recover_ts):
            job_id = str(event.get("job_id", "") or "")
            if job_id:
                impacted.add(job_id)
    return sorted(impacted, key=_sort_key)


def build_internal_failover_metrics_payload(
    *,
    events: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    impacted = [str(job.get("job_id", "")) for job in jobs if job.get("affected")]
    if not impacted:
        impacted = find_impacted_jobs(events)
    impacted = sorted({job_id for job_id in impacted if job_id}, key=_sort_key)
    return {
        "impacted_job_ids": impacted,
        "benchmark_events": events,
        "jobs": [
            {
                "job_id": str(job.get("job_id", "")),
                "recovered_checkpointed_tokens": int(job.get("remote_reuse_tokens", 0) or 0)
                + int(job.get("host_reuse_tokens", 0) or 0),
            }
            for job in sorted(jobs, key=lambda item: _sort_key(str(item.get("job_id", ""))))
        ],
    }


def write_internal_failover_metrics(
    output: Path,
    *,
    events: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = build_internal_failover_metrics_payload(events=events, jobs=jobs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
