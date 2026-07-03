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
DEFAULT_CANDIDATES = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_candidates.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_candidate_audit.json"
DEFAULT_OUTPUT_MD = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_candidate_audit.md"

FINAL_DATA_PLANE = "physical_tensor_copy"
DISALLOWED_TIMING_SOURCES = {
    "",
    "default_coefficients",
    "physical_copy_csv",
    "physical_copy_model_missing_csv",
    "physical_recovery_path_csv",
}
BRIDGE_DIGEST_FIELDS = (
    "suite_label",
    "request_csv",
    "rid",
    "restore_action",
    "restored_tokens",
    "physical_action",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
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


def validate_candidate_row(
    row: dict[str, str],
    required: dict[tuple[str, str, str], dict[str, str]],
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    key = row_key(row)
    bridge = required.get(key)
    if bridge is None:
        problems.append("not_in_p0_required_restore_rows")
        return False, problems

    data_plane = str(
        row.get("aegis_replay_restore_data_plane")
        or row.get("restore_data_plane")
        or row.get("data_plane")
        or ""
    ).strip().lower()
    timing_source = str(
        row.get("aegis_replay_restore_timing_source")
        or row.get("restore_timing_source")
        or row.get("timing_source")
        or ""
    ).strip().lower()
    if data_plane != FINAL_DATA_PLANE:
        problems.append("data_plane_not_physical_tensor_copy")
    if timing_source in DISALLOWED_TIMING_SOURCES:
        problems.append("timing_source_is_not_live_copy")
    if not _bool(row.get("physical_copy_verified")):
        problems.append("physical_copy_verified_is_false")
    if _int(row.get("restored_tokens")) != _int(bridge.get("restored_tokens")):
        problems.append("restored_tokens_mismatch")
    if str(row.get("restore_action") or "").strip() != str(
        bridge.get("restore_action") or ""
    ).strip():
        problems.append("restore_action_mismatch")
    if _float(row.get("copy_latency_ms")) <= 0.0:
        problems.append("copy_latency_ms_missing")
    if _int(row.get("copy_bytes")) <= 0:
        problems.append("copy_bytes_missing")
    if not str(row.get("source_artifact") or "").strip():
        problems.append("source_artifact_missing")
    return not problems, problems


def build_payload(*, bridge_rows: Path, candidates_csv: Path) -> dict[str, Any]:
    required = bridge_required_rows(_read_csv(bridge_rows))
    required_digest = bridge_required_digest(required)
    candidate_rows = _read_csv(candidates_csv)
    accepted_keys: set[tuple[str, str, str]] = set()
    invalid_rows: list[dict[str, Any]] = []
    extra_rows = 0
    for index, row in enumerate(candidate_rows, start=2):
        key = row_key(row)
        ok, problems = validate_candidate_row(row, required)
        if ok:
            accepted_keys.add(key)
        else:
            if "not_in_p0_required_restore_rows" in problems:
                extra_rows += 1
            invalid_rows.append(
                {
                    "csv_line": index,
                    "key": list(key),
                    "problems": problems,
                }
            )

    missing_keys = sorted(set(required) - accepted_keys)
    ready = (
        bool(required)
        and candidates_csv.exists()
        and len(candidate_rows) > 0
        and not invalid_rows
        and not missing_keys
        and len(accepted_keys) == len(required)
    )
    status = "strict_final_candidate_ready" if ready else "not_ready"
    if not candidates_csv.exists():
        status = "missing_candidate_csv"
    return {
        "accepted_rows": len(accepted_keys),
        "bridge_required_digest": required_digest,
        "bridge_rows": str(bridge_rows),
        "candidate_csv": str(candidates_csv),
        "candidate_rows": len(candidate_rows),
        "extra_rows": extra_rows,
        "invalid_rows": invalid_rows,
        "missing_rows": len(missing_keys),
        "missing_sample": [list(key) for key in missing_keys[:5]],
        "required_rows": len(required),
        "source": "validate_p0_physical_tensor_copy_candidates",
        "status": status,
        "strict_final_candidate_ready": ready,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P0 Physical Tensor-Copy Candidate Audit",
        "",
        "This report validates candidate live physical tensor-copy evidence against",
        "the P0 restore rows that are currently covered only by calibration bridges.",
        "It does not make metadata-materialized rows strict-final.",
        "",
        f"- status: `{payload['status']}`",
        f"- strict-final candidate ready: `{str(payload['strict_final_candidate_ready']).lower()}`",
        f"- bridge required digest: `{payload['bridge_required_digest']}`",
        f"- required rows: `{payload['required_rows']}`",
        f"- candidate rows: `{payload['candidate_rows']}`",
        f"- accepted rows: `{payload['accepted_rows']}`",
        f"- missing rows: `{payload['missing_rows']}`",
        f"- invalid rows: `{len(payload['invalid_rows'])}`",
        f"- extra rows: `{payload['extra_rows']}`",
        "",
        "## Required Candidate Fields",
        "",
        "- `suite_label`, `request_csv`, and `rid` must identify a P0 restore row.",
        "- `aegis_replay_restore_data_plane` must be `physical_tensor_copy`.",
        "- `aegis_replay_restore_timing_source` must be live-copy evidence, not a calibration CSV.",
        "- `physical_copy_verified` must be true.",
        "- `restored_tokens`, `restore_action`, `copy_latency_ms`, `copy_bytes`, and `source_artifact` must be present.",
        "",
    ]
    invalid = payload.get("invalid_rows") or []
    if invalid:
        lines.extend(["## Invalid Sample", ""])
        for item in invalid[:5]:
            lines.append(
                f"- line `{item['csv_line']}` key `{item['key']}` problems `{item['problems']}`"
            )
        lines.append("")
    missing = payload.get("missing_sample") or []
    if missing:
        lines.extend(["## Missing Sample", ""])
        for key in missing:
            lines.append(f"- `{key}`")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate candidate P0 live physical tensor-copy evidence."
    )
    parser.add_argument("--bridge-rows", type=Path, default=DEFAULT_BRIDGE_ROWS)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    payload = build_payload(
        bridge_rows=args.bridge_rows,
        candidates_csv=args.candidates_csv,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and not payload["strict_final_candidate_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
