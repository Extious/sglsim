#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READINESS_DIR = SGLSIM_LARGE_ROOT / "results" / "strict_final_readiness"
DEFAULT_OUTPUT_CSV = (
    DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.template.csv"
)
DEFAULT_OUTPUT_MD = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_event_schema.md"
DEFAULT_BRIDGE_ROWS = DEFAULT_READINESS_DIR / "p0_physical_bridge_rows.csv"
DEFAULT_REQUIRED_CSV = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.required.csv"
DEFAULT_REQUIRED_JSON = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.required.json"

EVENT_SCHEMA_VERSION = "p0-live-copy-event-v1"
FINAL_DATA_PLANE = "physical_tensor_copy"
ACCEPTED_EVENT_KINDS = (
    "live_physical_tensor_copy",
    "live_host_h2d_copy",
    "live_peer_p2p_copy",
)
ACCEPTED_TIMING_SOURCES = (
    "live_cuda_copy",
    "live_host_h2d_copy",
    "live_peer_p2p_copy",
)
TEMPLATE_FIELDS = [
    "suite_label",
    "request_csv",
    "rid",
    "event_schema_version",
    "event_kind",
    "restore_action",
    "restored_tokens",
    "aegis_replay_restore_data_plane",
    "aegis_replay_restore_timing_source",
    "physical_copy_verified",
    "copy_latency_ms",
    "copy_bytes",
    "source_artifact",
    "physical_action",
    "src_rank",
    "dst_rank",
    "src_device",
    "dst_device",
    "copy_backend",
    "copy_started_at_ns",
    "copy_ended_at_ns",
]
REQUIRED_FIELDS = [
    "suite_label",
    "request_csv",
    "rid",
    "event_schema_version",
    "event_kind",
    "restore_action",
    "restored_tokens",
    "aegis_replay_restore_data_plane",
    "aegis_replay_restore_timing_source",
    "physical_action",
    "expected_measurement",
    "physical_copy_verified",
    "copy_latency_ms",
    "copy_bytes",
    "source_artifact",
    "src_rank",
    "dst_rank",
    "src_device",
    "dst_device",
    "copy_backend",
    "copy_started_at_ns",
    "copy_ended_at_ns",
]
PHYSICAL_ACTION_EVENT_KIND = {
    "host_restore_h2d": "live_host_h2d_copy",
    "gpu_replica_restore_p2p": "live_peer_p2p_copy",
}


def write_csv_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TEMPLATE_FIELDS)
        writer.writeheader()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def required_event_rows(bridge_rows: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in _read_csv(bridge_rows):
        if not _bool(row.get("bridge_eligible")):
            continue
        physical_action = str(row.get("physical_action") or "").strip()
        event_kind = PHYSICAL_ACTION_EVENT_KIND.get(
            physical_action,
            "live_physical_tensor_copy",
        )
        rows.append(
            {
                "suite_label": str(row.get("suite_label") or "").strip(),
                "request_csv": str(row.get("request_csv") or "").strip(),
                "rid": str(row.get("rid") or "").strip(),
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "event_kind": event_kind,
                "restore_action": str(row.get("restore_action") or "").strip(),
                "restored_tokens": str(row.get("restored_tokens") or "").strip(),
                "aegis_replay_restore_data_plane": FINAL_DATA_PLANE,
                "aegis_replay_restore_timing_source": event_kind,
                "physical_action": physical_action,
                "expected_measurement": (
                    "measure this live copy; calibration-only rows are not accepted"
                ),
                "physical_copy_verified": "",
                "copy_latency_ms": "",
                "copy_bytes": "",
                "source_artifact": "",
                "src_rank": "",
                "dst_rank": "",
                "src_device": "",
                "dst_device": "",
                "copy_backend": "",
                "copy_started_at_ns": "",
                "copy_ended_at_ns": "",
            }
        )
    return rows


def write_required_events(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_required_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_action: dict[str, int] = {}
    for row in rows:
        event_kind = str(row.get("event_kind") or "")
        action = str(row.get("restore_action") or "")
        by_kind[event_kind] = by_kind.get(event_kind, 0) + 1
        by_action[action] = by_action.get(action, 0) + 1
    return {
        "required_event_rows": len(rows),
        "required_event_kinds": dict(sorted(by_kind.items())),
        "required_restore_actions": dict(sorted(by_action.items())),
    }


def render_markdown(*, output_csv: Path, required_csv: Path | None = None) -> str:
    example = {
        "suite_label": "<copy from p0_physical_bridge_rows.csv>",
        "request_csv": "<copy from p0_physical_bridge_rows.csv>",
        "rid": "<copy from p0_physical_bridge_rows.csv>",
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "event_kind": "live_peer_p2p_copy",
        "restore_action": "gpu_replica",
        "restored_tokens": 256,
        "aegis_replay_restore_data_plane": FINAL_DATA_PLANE,
        "aegis_replay_restore_timing_source": "live_peer_p2p_copy",
        "physical_copy_verified": True,
        "copy_latency_ms": 0.73,
        "copy_bytes": 16777216,
        "source_artifact": "<absolute path to the live-copy trace or log>",
        "physical_action": "gpu_replica_restore_p2p",
        "src_rank": 0,
        "dst_rank": 1,
        "src_device": "cuda:0",
        "dst_device": "cuda:1",
        "copy_backend": "torch.distributed",
        "copy_started_at_ns": 0,
        "copy_ended_at_ns": 730000,
    }
    lines = [
        "# P0 Physical Tensor-Copy Event Schema",
        "",
        "This template describes the live-copy event input accepted by",
        "`build_p0_physical_tensor_copy_candidates.py`. It is a schema/header",
        "artifact only; it does not provide evidence and does not create strict-final",
        "candidate rows by itself.",
        "",
        f"- event schema version: `{EVENT_SCHEMA_VERSION}`",
        f"- CSV template: `{output_csv}`",
        f"- required data plane: `{FINAL_DATA_PLANE}`",
        f"- accepted event kinds: `{', '.join(ACCEPTED_EVENT_KINDS)}`",
        f"- accepted live timing sources: `{', '.join(ACCEPTED_TIMING_SOURCES)}`",
        *(
            [f"- required-event manifest: `{required_csv}`"]
            if required_csv is not None
            else []
        ),
        "",
        "## Required Identity",
        "",
        "- `suite_label`, `request_csv`, and `rid` must match a row in",
        "  `p0_physical_bridge_rows.csv`.",
        "- `restore_action` and `restored_tokens`, when present, must match the bridge row.",
        "- Each event must explicitly identify a live physical tensor copy by setting",
        "  `aegis_replay_restore_data_plane=physical_tensor_copy` or by using one of",
        "  the accepted `event_kind` values.",
        "",
        "## Required Measurement",
        "",
        "- `physical_copy_verified` must be true.",
        "- `copy_latency_ms` must be positive.",
        "- `copy_bytes` must be positive.",
        "- `source_artifact` must point to the live-copy trace, log, or run artifact.",
        "",
        "## Required-Event Manifest",
        "",
        "The required-event manifest, when generated, is a collection checklist.",
        "Rows in that file intentionally leave `physical_copy_verified`,",
        "`copy_latency_ms`, `copy_bytes`, and `source_artifact` blank. Filling",
        "those fields requires a measured live Host H2D or peer P2P copy trace.",
        "The manifest is not evidence by itself and must not be passed off as a",
        "candidate CSV.",
        "",
        "## JSONL Example",
        "",
        "Do not use this placeholder object as evidence; replace the identity fields",
        "with exact values from the current bridge rows and measured live-copy values.",
        "",
        "```json",
        json.dumps(example, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def build_payload(
    *,
    output_csv: Path,
    output_md: Path,
    required_csv: Path | None = None,
    required_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    required_summary = summarize_required_rows(required_rows or [])
    return {
        "accepted_event_kinds": list(ACCEPTED_EVENT_KINDS),
        "accepted_timing_sources": list(ACCEPTED_TIMING_SOURCES),
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "required_csv": str(required_csv) if required_csv is not None else "",
        **required_summary,
        "source": "write_p0_physical_tensor_copy_event_template",
        "status": "template_written",
        "template_fields": TEMPLATE_FIELDS,
        "required_fields": REQUIRED_FIELDS,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the P0 live physical tensor-copy event schema template."
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--bridge-rows", type=Path, default=DEFAULT_BRIDGE_ROWS)
    parser.add_argument("--required-csv", type=Path, default=DEFAULT_REQUIRED_CSV)
    parser.add_argument("--required-json", type=Path, default=DEFAULT_REQUIRED_JSON)
    args = parser.parse_args(argv)

    write_csv_template(args.output_csv)
    required_rows = required_event_rows(args.bridge_rows)
    write_required_events(args.required_csv, required_rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        render_markdown(output_csv=args.output_csv, required_csv=args.required_csv),
        encoding="utf-8",
    )
    payload = build_payload(
        output_csv=args.output_csv,
        output_md=args.output_md,
        required_csv=args.required_csv,
        required_rows=required_rows,
    )
    args.required_json.parent.mkdir(parents=True, exist_ok=True)
    args.required_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
