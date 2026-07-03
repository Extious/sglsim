#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READINESS_DIR = SGLSIM_LARGE_ROOT / "results" / "strict_final_readiness"
DEFAULT_BRIDGE_ROWS = DEFAULT_READINESS_DIR / "p0_physical_bridge_rows.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_candidates.csv"
DEFAULT_AUDIT_JSON = (
    DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_candidate_generation_audit.json"
)
DEFAULT_AUDIT_MD = (
    DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_candidate_generation_audit.md"
)

EVENT_SCHEMA_VERSION = "p0-live-copy-event-v1"
FINAL_DATA_PLANE = "physical_tensor_copy"
DISALLOWED_TIMING_SOURCES = {
    "",
    "default_coefficients",
    "physical_copy_csv",
    "physical_copy_model_missing_csv",
    "physical_recovery_path_csv",
}
PHYSICAL_ACTION_TO_TIMING_SOURCE = {
    "host_restore_h2d": "live_host_h2d_copy",
    "gpu_replica_restore_p2p": "live_peer_p2p_copy",
}
LIVE_EVENT_KIND_FIELDS = (
    "event_kind",
    "event_type",
    "copy_kind",
    "physical_copy_kind",
)
LIVE_EVENT_KINDS = {
    "live_physical_tensor_copy",
    "physical_tensor_copy",
    "live_host_h2d_copy",
    "host_h2d_copy",
    "live_peer_p2p_copy",
    "peer_p2p_copy",
}
BRIDGE_DIGEST_FIELDS = (
    "suite_label",
    "request_csv",
    "rid",
    "restore_action",
    "restored_tokens",
    "physical_action",
)

CANDIDATE_FIELDS = [
    "suite_label",
    "request_csv",
    "rid",
    "restore_action",
    "restored_tokens",
    "aegis_replay_restore_data_plane",
    "aegis_replay_restore_timing_source",
    "physical_copy_verified",
    "copy_latency_ms",
    "copy_bytes",
    "source_artifact",
    "physical_action",
    "event_kind",
    "event_schema_version",
    "event_source",
    "event_line",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            rows.append(
                {
                    "_event_source": str(path),
                    "_event_line": line_no,
                    "_invalid_json": True,
                }
            )
            continue
        if isinstance(decoded, dict):
            decoded["_event_source"] = str(path)
            decoded["_event_line"] = line_no
            rows.append(decoded)
    return rows


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: object) -> int:
    try:
        if value in ("", None):
            return 0
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> object:
    for key in keys:
        value = row.get(key)
        if value not in ("", None):
            return value
    return ""


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("suite_label") or "").strip(),
        str(row.get("request_csv") or "").strip(),
        str(row.get("rid") or "").strip(),
    )


def bridge_required_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    required: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if not _bool(row.get("bridge_eligible")):
            continue
        key = row_key(row)
        if all(key):
            required[key] = row
    return required


def bridge_required_digest(
    required: dict[tuple[str, str, str], dict[str, str]]
) -> str:
    digest_rows = []
    for key, row in sorted(required.items()):
        digest_row = {
            field: str(row.get(field) or "").strip()
            for field in BRIDGE_DIGEST_FIELDS
        }
        digest_row["key"] = list(key)
        digest_rows.append(digest_row)
    encoded = json.dumps(digest_rows, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_event_rows(
    *,
    event_csvs: list[Path],
    event_jsonls: list[Path],
) -> tuple[list[dict[str, Any]], list[Path], list[Path]]:
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    missing: list[Path] = []
    for path in event_csvs:
        if not path.exists():
            missing.append(path)
            continue
        sources.append(path)
        for line_no, row in enumerate(_read_csv(path), start=2):
            out: dict[str, Any] = dict(row)
            out["_event_source"] = str(path)
            out["_event_line"] = line_no
            rows.append(out)
    for path in event_jsonls:
        if not path.exists():
            missing.append(path)
            continue
        sources.append(path)
        rows.extend(_read_jsonl(path))
    return rows, sources, missing


def validate_live_event(
    event: dict[str, Any],
    bridge: dict[str, str],
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if event.get("_invalid_json"):
        return False, ["invalid_jsonl_row"]
    if row_key(event) != row_key(bridge):
        problems.append("key_mismatch")

    event_action = str(_first(event, ("restore_action", "aegis_replay_restore_action"))).strip()
    if event_action and event_action != str(bridge.get("restore_action") or "").strip():
        problems.append("restore_action_mismatch")

    event_tokens = _int(_first(event, ("restored_tokens", "aegis_replay_restored_tokens")))
    if event_tokens and event_tokens != _int(bridge.get("restored_tokens")):
        problems.append("restored_tokens_mismatch")

    explicit_data_plane = str(
        _first(
            event,
            (
                "aegis_replay_restore_data_plane",
                "restore_data_plane",
                "data_plane",
            ),
        )
    ).strip().lower()
    if explicit_data_plane and explicit_data_plane != FINAL_DATA_PLANE:
        problems.append("data_plane_not_physical_tensor_copy")
    event_kind = str(_first(event, LIVE_EVENT_KIND_FIELDS)).strip().lower()
    has_live_identity = (
        explicit_data_plane == FINAL_DATA_PLANE
        or event_kind in LIVE_EVENT_KINDS
    )
    if not has_live_identity:
        problems.append("live_physical_copy_identity_missing")

    timing_source = str(
        _first(
            event,
            (
                "aegis_replay_restore_timing_source",
                "restore_timing_source",
                "timing_source",
            ),
        )
        or PHYSICAL_ACTION_TO_TIMING_SOURCE.get(
            str(bridge.get("physical_action") or "").strip(),
            "",
        )
    ).strip().lower()
    if timing_source in DISALLOWED_TIMING_SOURCES:
        problems.append("timing_source_is_not_live_copy")

    if not _bool(_first(event, ("physical_copy_verified", "verified", "live_copy_verified"))):
        problems.append("physical_copy_verified_is_false")
    if _float(_first(event, ("copy_latency_ms", "latency_ms", "elapsed_ms", "duration_ms"))) <= 0.0:
        problems.append("copy_latency_ms_missing")
    if _int(_first(event, ("copy_bytes", "bytes", "tensor_bytes", "kv_bytes"))) <= 0:
        problems.append("copy_bytes_missing")
    return not problems, problems


def build_candidate_row(event: dict[str, Any], bridge: dict[str, str]) -> dict[str, Any]:
    physical_action = str(bridge.get("physical_action") or "").strip()
    timing_source = str(
        _first(
            event,
            (
                "aegis_replay_restore_timing_source",
                "restore_timing_source",
                "timing_source",
            ),
        )
        or PHYSICAL_ACTION_TO_TIMING_SOURCE.get(physical_action, "live_physical_tensor_copy")
    ).strip()
    source_artifact = str(_first(event, ("source_artifact", "artifact", "trace_path"))).strip()
    event_source = str(event.get("_event_source") or "")
    event_line = str(event.get("_event_line") or "")
    if not source_artifact:
        source_artifact = f"{event_source}:{event_line}" if event_line else event_source
    return {
        "suite_label": str(bridge.get("suite_label") or ""),
        "request_csv": str(bridge.get("request_csv") or ""),
        "rid": str(bridge.get("rid") or ""),
        "restore_action": str(bridge.get("restore_action") or ""),
        "restored_tokens": _int(bridge.get("restored_tokens")),
        "aegis_replay_restore_data_plane": FINAL_DATA_PLANE,
        "aegis_replay_restore_timing_source": timing_source,
        "physical_copy_verified": "true",
        "copy_latency_ms": _float(
            _first(event, ("copy_latency_ms", "latency_ms", "elapsed_ms", "duration_ms"))
        ),
        "copy_bytes": _int(_first(event, ("copy_bytes", "bytes", "tensor_bytes", "kv_bytes"))),
        "source_artifact": source_artifact,
        "physical_action": physical_action,
        "event_kind": str(_first(event, LIVE_EVENT_KIND_FIELDS)).strip(),
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "event_source": event_source,
        "event_line": event_line,
    }


def build_payload(
    *,
    bridge_rows: Path,
    event_csvs: list[Path],
    event_jsonls: list[Path],
) -> dict[str, Any]:
    required = bridge_required_rows(_read_csv(bridge_rows))
    required_digest = bridge_required_digest(required)
    events, sources, missing_sources = read_event_rows(
        event_csvs=event_csvs,
        event_jsonls=event_jsonls,
    )
    events_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for event in events:
        events_by_key.setdefault(row_key(event), []).append(event)

    candidates: list[dict[str, Any]] = []
    rejected_events: list[dict[str, Any]] = []
    accepted_keys: set[tuple[str, str, str]] = set()
    for key, bridge in sorted(required.items()):
        for event in events_by_key.get(key, []):
            ok, problems = validate_live_event(event, bridge)
            if ok:
                candidates.append(build_candidate_row(event, bridge))
                accepted_keys.add(key)
                break
            rejected_events.append(
                {
                    "event_line": event.get("_event_line", ""),
                    "event_source": event.get("_event_source", ""),
                    "key": list(key),
                    "problems": problems,
                }
            )

    extra_events = sum(1 for key in events_by_key if key not in required)
    missing_keys = sorted(set(required) - accepted_keys)
    ready = bool(required) and not missing_keys and len(candidates) == len(required)
    if missing_sources:
        status = "missing_event_sources"
    elif not sources:
        status = "no_event_sources"
    elif ready:
        status = "candidates_complete"
    else:
        status = "candidates_partial"
    return {
        "accepted_rows": len(candidates),
        "bridge_required_digest": required_digest,
        "bridge_rows": str(bridge_rows),
        "candidate_rows": candidates,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "event_rows": len(events),
        "event_sources": [str(path) for path in sources],
        "extra_events": extra_events,
        "missing_event_sources": [str(path) for path in missing_sources],
        "missing_rows": len(missing_keys),
        "missing_sample": [list(key) for key in missing_keys[:5]],
        "rejected_events": rejected_events[:20],
        "rejected_event_rows": len(rejected_events),
        "required_rows": len(required),
        "source": "build_p0_physical_tensor_copy_candidates",
        "status": status,
        "strict_final_candidate_ready": ready,
    }


def write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown(payload: dict[str, Any], *, output_csv: Path) -> str:
    lines = [
        "# P0 Physical Tensor-Copy Candidate Builder",
        "",
        "This report builds strict-final candidate rows from live physical-copy events.",
        "It never promotes calibration-only or metadata-materialized rows.",
        "",
        f"- status: `{payload['status']}`",
        f"- strict-final candidate ready: `{str(payload['strict_final_candidate_ready']).lower()}`",
        f"- bridge required digest: `{payload['bridge_required_digest']}`",
        f"- event schema version: `{payload['event_schema_version']}`",
        f"- required rows: `{payload['required_rows']}`",
        f"- event rows: `{payload['event_rows']}`",
        f"- accepted rows: `{payload['accepted_rows']}`",
        f"- missing rows: `{payload['missing_rows']}`",
        f"- rejected event rows: `{payload['rejected_event_rows']}`",
        f"- extra events: `{payload['extra_events']}`",
        f"- candidate CSV: `{output_csv}`",
        "",
        "## Accepted Live-Copy Identity",
        "",
        "- Event rows must explicitly set a physical tensor-copy data plane or a live physical-copy event kind.",
        "- `physical_copy_verified`, positive copy latency, and positive copy bytes are necessary but not sufficient.",
        "",
        "## Event Sources",
        "",
    ]
    sources = payload.get("event_sources") or []
    if sources:
        lines.extend(f"- `{source}`" for source in sources)
    else:
        lines.append("- No event source was provided or found.")
    missing_sources = payload.get("missing_event_sources") or []
    if missing_sources:
        lines.extend(["", "## Missing Event Sources", ""])
        lines.extend(f"- `{source}`" for source in missing_sources)
    rejected = payload.get("rejected_events") or []
    if rejected:
        lines.extend(["", "## Rejected Sample", ""])
        for item in rejected[:5]:
            lines.append(
                f"- `{item['event_source']}:{item['event_line']}` key `{item['key']}` problems `{item['problems']}`"
            )
    missing = payload.get("missing_sample") or []
    if missing:
        lines.extend(["", "## Missing Sample", ""])
        for key in missing:
            lines.append(f"- `{key}`")
    lines.append("")
    return "\n".join(lines)


def write_audit(path: Path, payload: dict[str, Any], *, output_csv: Path) -> None:
    serializable = {key: value for key, value in payload.items() if key != "candidate_rows"}
    serializable["output_csv"] = str(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build P0 strict-final candidate CSV from live physical tensor-copy events."
    )
    parser.add_argument("--bridge-rows", type=Path, default=DEFAULT_BRIDGE_ROWS)
    parser.add_argument("--event-csv", type=Path, action="append", default=[])
    parser.add_argument("--event-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        bridge_rows=args.bridge_rows,
        event_csvs=args.event_csv,
        event_jsonls=args.event_jsonl,
    )
    write_candidates(args.output_csv, payload["candidate_rows"])
    write_audit(args.audit_json, payload, output_csv=args.output_csv)
    args.audit_md.parent.mkdir(parents=True, exist_ok=True)
    args.audit_md.write_text(
        render_markdown(payload, output_csv=args.output_csv),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "candidate_rows"}, indent=2, sort_keys=True))
    if args.strict and not payload["strict_final_candidate_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
