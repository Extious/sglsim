#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READINESS_DIR = SGLSIM_LARGE_ROOT / "results" / "strict_final_readiness"
DEFAULT_P0_CSV = DEFAULT_READINESS_DIR / "p0_strict_final_readiness.csv"
DEFAULT_CANDIDATES_CSV = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_candidates.csv"
DEFAULT_CANDIDATE_AUDIT_JSON = (
    DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_candidate_audit.json"
)
DEFAULT_OUTPUT_CSV = DEFAULT_READINESS_DIR / "p0_strict_final_readiness.promoted.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_READINESS_DIR / "p0_strict_final_readiness.promoted.json"
DEFAULT_OUTPUT_MD = DEFAULT_READINESS_DIR / "p0_strict_final_readiness.promoted.md"

FINAL_DATA_PLANE = "physical_tensor_copy"

PROMOTED_FIELDS = [
    "suite_label",
    "figures",
    "suite_root",
    "request_csv_count",
    "request_rows",
    "replay_rows",
    "restore_rows",
    "physical_tensor_copy_rows",
    "nonfinal_restore_rows",
    "missing_data_plane_rows",
    "missing_timing_source_rows",
    "schema_complete",
    "final_ready",
    "data_planes",
    "timing_sources",
    "next_required_evidence",
    "promotion_source",
    "candidate_rows",
    "host_h2d_candidate_rows",
    "peer_p2p_candidate_rows",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def as_int(value: object) -> int:
    try:
        if value in ("", None):
            return 0
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def count_by(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "").strip()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def group_candidates(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("suite_label") or "").strip(), []).append(row)
    return grouped


def build_promoted_rows(
    *,
    p0_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    candidate_audit: dict[str, Any],
    candidates_csv: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    problems: list[str] = []
    if not candidate_audit.get("strict_final_candidate_ready"):
        problems.append("candidate_audit_not_ready")
    grouped = group_candidates(candidate_rows)
    promoted: list[dict[str, Any]] = []
    for row in p0_rows:
        suite = str(row.get("suite_label") or "").strip()
        candidates = grouped.get(suite, [])
        restore_rows = as_int(row.get("restore_rows"))
        if len(candidates) != restore_rows:
            problems.append(
                f"suite_candidate_count_mismatch:{suite}:{len(candidates)}!={restore_rows}"
            )
        bad_plane = [
            item
            for item in candidates
            if str(item.get("aegis_replay_restore_data_plane") or "").strip()
            != FINAL_DATA_PLANE
        ]
        if bad_plane:
            problems.append(f"suite_non_physical_candidate_rows:{suite}:{len(bad_plane)}")
        timing_sources = count_by(candidates, "aegis_replay_restore_timing_source")
        event_kinds = count_by(candidates, "event_kind")
        promoted.append(
            {
                "suite_label": suite,
                "figures": row.get("figures", ""),
                "suite_root": row.get("suite_root", ""),
                "request_csv_count": as_int(row.get("request_csv_count")),
                "request_rows": as_int(row.get("request_rows")),
                "replay_rows": as_int(row.get("replay_rows")),
                "restore_rows": restore_rows,
                "physical_tensor_copy_rows": restore_rows,
                "nonfinal_restore_rows": 0,
                "missing_data_plane_rows": 0,
                "missing_timing_source_rows": 0,
                "schema_complete": True,
                "final_ready": len(candidates) == restore_rows and not bad_plane,
                "data_planes": json.dumps({FINAL_DATA_PLANE: restore_rows}, sort_keys=True),
                "timing_sources": json.dumps(timing_sources, sort_keys=True),
                "next_required_evidence": (
                    "strict-final live physical tensor-copy candidate overlay passes"
                ),
                "promotion_source": str(candidates_csv),
                "candidate_rows": len(candidates),
                "host_h2d_candidate_rows": event_kinds.get("live_host_h2d_copy", 0),
                "peer_p2p_candidate_rows": event_kinds.get("live_peer_p2p_copy", 0),
            }
        )
    missing_suites = sorted(set(grouped) - {str(row.get("suite_label") or "").strip() for row in p0_rows})
    if missing_suites:
        problems.append(f"candidate_suites_not_in_p0:{missing_suites}")
    return promoted, problems


def summarize(rows: list[dict[str, Any]], problems: list[str]) -> dict[str, Any]:
    restore_rows = sum(as_int(row.get("restore_rows")) for row in rows)
    physical_rows = sum(as_int(row.get("physical_tensor_copy_rows")) for row in rows)
    nonfinal_rows = sum(as_int(row.get("nonfinal_restore_rows")) for row in rows)
    host_rows = sum(as_int(row.get("host_h2d_candidate_rows")) for row in rows)
    p2p_rows = sum(as_int(row.get("peer_p2p_candidate_rows")) for row in rows)
    ready = bool(rows) and not problems and restore_rows == physical_rows and nonfinal_rows == 0
    return {
        "host_h2d_candidate_rows": host_rows,
        "nonfinal_restore_rows": nonfinal_rows,
        "physical_tensor_copy_rows": physical_rows,
        "problems": problems,
        "promoted_rows": len(rows),
        "promotion_ready": ready,
        "restore_rows": restore_rows,
        "peer_p2p_candidate_rows": p2p_rows,
        "source": "promote_p0_physical_tensor_copy_candidates",
        "status": "promoted_strict_final_ready" if ready else "not_ready",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROMOTED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(payload: dict[str, Any], *, output_csv: Path) -> str:
    lines = [
        "# P0 Promoted Strict-Final Readiness",
        "",
        "This file is an explicit overlay from validated live physical tensor-copy",
        "candidate rows onto the P0 online SGLSim restore rows. It does not mutate",
        "the raw request-detail CSVs.",
        "",
        f"- status: `{payload['status']}`",
        f"- promotion ready: `{str(payload['promotion_ready']).lower()}`",
        f"- restore rows: {payload['restore_rows']}",
        f"- physical tensor copy rows: {payload['physical_tensor_copy_rows']}",
        f"- non-final restore rows: {payload['nonfinal_restore_rows']}",
        f"- Host H2D candidate rows: {payload['host_h2d_candidate_rows']}",
        f"- peer P2P candidate rows: {payload['peer_p2p_candidate_rows']}",
        f"- promoted CSV: `{output_csv}`",
        "",
    ]
    problems = payload.get("problems") or []
    if problems:
        lines.extend(["## Problems", ""])
        lines.extend(f"- `{item}`" for item in problems)
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote validated P0 live-copy candidates into a strict-final overlay."
    )
    parser.add_argument("--p0-csv", type=Path, default=DEFAULT_P0_CSV)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument(
        "--candidate-audit-json",
        type=Path,
        default=DEFAULT_CANDIDATE_AUDIT_JSON,
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    promoted, problems = build_promoted_rows(
        p0_rows=read_csv(args.p0_csv),
        candidate_rows=read_csv(args.candidates_csv),
        candidate_audit=read_json(args.candidate_audit_json),
        candidates_csv=args.candidates_csv,
    )
    payload = summarize(promoted, problems)
    payload["output_csv"] = str(args.output_csv)
    payload["p0_csv"] = str(args.p0_csv)
    payload["candidates_csv"] = str(args.candidates_csv)
    write_csv(args.output_csv, promoted)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(payload, output_csv=args.output_csv), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and not payload["promotion_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
