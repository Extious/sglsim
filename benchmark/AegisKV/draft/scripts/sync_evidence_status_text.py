#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DRAFT_ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = DRAFT_ROOT.parent
DEFAULT_STATUS_JSON = (
    DRAFT_ROOT
    / "experiments"
    / "sglang_small"
    / "results"
    / "strict_final_recovery_matrix_audit"
    / "strict_final_bound_slurm_jobs.json"
)
DEFAULT_READINESS_JSON = (
    DRAFT_ROOT / "experiments" / "strict_final_dataplane_readiness.json"
)
DEFAULT_PAPER = DRAFT_ROOT / "paper" / "current_results_overview.md"
DEFAULT_BEAMER = AEGIS_ROOT / "beamer" / "presentation.tex"


def read_status(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Status JSON must contain an object: {path}")
    return value


def read_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def status_counts(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("status_counts")
    if not isinstance(raw, dict):
        return {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    return {
        key: int(raw.get(key, 0) or 0)
        for key in ("pending", "running", "completed", "failed")
    }


def short_status(payload: dict[str, Any]) -> str:
    counts = status_counts(payload)
    total = int(payload.get("total_bound_jobs", 0) or 0)
    if total and counts["pending"] == total:
        return f"{total} jobs pending"
    if total and counts["running"] == total:
        return f"{total} jobs running"
    if total and counts["completed"] == total:
        return f"{total} jobs complete"
    parts = [
        f"{counts[key]} {label}"
        for key, label in (
            ("pending", "pending"),
            ("running", "running"),
            ("completed", "complete"),
            ("failed", "failed"),
        )
        if counts[key]
    ]
    return " / ".join(parts) if parts else "0 jobs active"


def poll_hhmm(checked_at: str) -> str:
    try:
        return datetime.fromisoformat(checked_at).strftime("%H:%M")
    except ValueError:
        return checked_at[:5] if checked_at else "unknown"


def readiness_values(readiness: dict[str, Any]) -> dict[str, int | str]:
    p0 = readiness.get("p0") if isinstance(readiness.get("p0"), dict) else {}
    bridge = (
        readiness.get("bridge")
        if isinstance(readiness.get("bridge"), dict)
        else {}
    )
    matrix = (
        readiness.get("matrix")
        if isinstance(readiness.get("matrix"), dict)
        else {}
    )
    event_template = (
        readiness.get("p0_physical_tensor_copy_event_template")
        if isinstance(readiness.get("p0_physical_tensor_copy_event_template"), dict)
        else {}
    )
    live_copy_events = (
        readiness.get("p0_live_copy_events")
        if isinstance(readiness.get("p0_live_copy_events"), dict)
        else {}
    )
    p2p_plan = (
        readiness.get("p0_live_p2p_collection_plan")
        if isinstance(readiness.get("p0_live_p2p_collection_plan"), dict)
        else {}
    )
    promotion = (
        readiness.get("p0_promotion")
        if isinstance(readiness.get("p0_promotion"), dict)
        else {}
    )
    cases = matrix.get("cases") if isinstance(matrix.get("cases"), dict) else {}
    wait = (
        cases.get("same_write_wait_complete")
        if isinstance(cases.get("same_write_wait_complete"), dict)
        else {}
    )
    def as_int(value: object) -> int:
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    missing_schema = as_int(p0.get("missing_data_plane_rows")) + as_int(
        p0.get("missing_timing_source_rows")
    )
    promotion_ready = bool(promotion.get("promotion_ready"))
    promoted_physical_rows = as_int(promotion.get("physical_tensor_copy_rows"))
    p2p_ready = bool(p2p_plan.get("p2p_ready"))
    measured_p2p_rows = as_int(p2p_plan.get("p2p_live_copy_event_rows"))
    skipped_copy_rows = 0 if p2p_ready else as_int(live_copy_events.get("skipped_rows"))
    measured_h2d_rows = as_int(
        (live_copy_events.get("event_rows_by_kind") or {}).get(
            "live_host_h2d_copy", 0
        )
        if isinstance(live_copy_events.get("event_rows_by_kind"), dict)
        else 0
    )
    return {
        "accepted_pairs": as_int(matrix.get("accepted_pairs")),
        "host_matrix_ready": bool(readiness.get("host_matrix_ready")),
        "matrix_status": str(matrix.get("overall_status") or "unknown"),
        "physical_restore_rows": as_int(matrix.get("physical_restore_rows")),
        "p0_bridge_rows": as_int(bridge.get("bridge_eligible_rows")),
        "p0_bridge_missing_rows": as_int(bridge.get("bridge_missing_rows")),
        "p0_copy_rows": (
            promoted_physical_rows
            if promotion_ready and promoted_physical_rows
            else as_int(p0.get("physical_tensor_copy_rows"))
        ),
        "p0_restore_rows": as_int(p0.get("restore_rows")),
        "required_pairs": as_int(matrix.get("required_pairs")),
        "required_live_copy_events": as_int(event_template.get("required_rows")),
        "required_live_copy_h2d": as_int(
            (event_template.get("required_event_kinds") or {}).get(
                "live_host_h2d_copy", 0
            )
            if isinstance(event_template.get("required_event_kinds"), dict)
            else 0
        ),
        "required_live_copy_p2p": as_int(
            (event_template.get("required_event_kinds") or {}).get(
                "live_peer_p2p_copy", 0
            )
            if isinstance(event_template.get("required_event_kinds"), dict)
            else 0
        ),
        "measured_live_copy_events": measured_h2d_rows + measured_p2p_rows,
        "measured_live_copy_h2d": measured_h2d_rows,
        "measured_live_copy_p2p": measured_p2p_rows,
        "skipped_live_copy_events": skipped_copy_rows,
        "p2p_plan_required_rows": as_int(p2p_plan.get("required_p2p_rows")),
        "p2p_plan_status": str(p2p_plan.get("status") or "missing"),
        "promotion_ready": promotion_ready,
        "schema_missing_rows": missing_schema,
        "strict_final_ready": bool(readiness.get("strict_final_ready")),
        "wait_accepted": as_int(wait.get("accepted_pairs")),
        "wait_required": as_int(wait.get("required_repeats")),
    }


def goal_audit_line(values: dict[str, int | str]) -> str:
    if values.get("strict_final_ready"):
        return "Goal audit: 17/17 pass，0 partial，0 blocked"
    if values.get("host_matrix_ready"):
        return "Goal audit: 16/17 pass，0 partial，1 blocked"
    return "Goal audit: 15/17 pass，1 partial，1 blocked"


def tex_escape(value: object) -> str:
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_")


def sync_beamer(text: str, payload: dict[str, Any], readiness: dict[str, Any]) -> str:
    checked_at = str(payload.get("checked_at") or "")
    values = readiness_values(readiness)
    text, count = re.subn(
        r"\\item Goal audit: \d+/\d+ pass，\d+ partial，\d+ blocked",
        lambda _match: rf"\item {goal_audit_line(values)}",
        text,
        count=1,
    )
    if values["p0_restore_rows"]:
        text, count = re.subn(
            r"\\item P0 restore: \d+ rows，copy=\d+(?:，bridge=\d+)?，schema missing=\d+",
            lambda _match: (
                r"\item P0 restore: "
                f"{values['p0_restore_rows']} rows，copy={values['p0_copy_rows']}，"
                f"bridge={values['p0_bridge_rows']}，"
                f"schema missing={values['schema_missing_rows']}"
            ),
            text,
            count=1,
        )
        if count != 1:
            raise ValueError("Could not find the beamer P0 restore status line")
    if values["required_pairs"]:
        text, count = re.subn(
            r"\\item SGLang matrix: .*?wait-complete \d+/\d+",
            lambda _match: (
                r"\item SGLang matrix: "
                f"accepted {values['accepted_pairs']}/{values['required_pairs']}，"
                f"wait-complete {values['wait_accepted']}/{values['wait_required']}"
            ),
            text,
            count=1,
        )
        if count != 1:
            raise ValueError("Could not find the beamer SGLang matrix status line")
    if values["required_live_copy_events"]:
        p2p_tail = (
            f"P2P {values['measured_live_copy_p2p']}"
            if values["measured_live_copy_p2p"]
            else f"P2P blocked {values['skipped_live_copy_events']}"
        )
        text = re.sub(
            r"\\item P0 live-copy events: required \d+，(?:measured \d+，)?H2D \d+，(?:P2P|P2P blocked) \d+",
            lambda _match: (
                r"\item P0 live-copy events: "
                f"required {values['required_live_copy_events']}，"
                f"measured {values['measured_live_copy_events']}，"
                f"H2D {values['measured_live_copy_h2d']}，"
                f"{p2p_tail}"
            ),
            text,
            count=1,
        )
    if values["p2p_plan_required_rows"]:
        text = re.sub(
            r"\\item P2P remote plan: [^\n]*",
            lambda _match: (
                r"\item P2P remote plan: "
                f"{tex_escape(values['p2p_plan_status'])}，"
                f"rows {values['p2p_plan_required_rows']}"
            ),
            text,
            count=1,
        )
    replacement = (
        r"\item r2/r3 Slurm: "
        f"{short_status(payload)}，poll {poll_hhmm(checked_at)}；"
        r"complete 后 sync/audit"
    )
    updated, count = re.subn(
        r"\\item r2/r3 Slurm: [^\n]*",
        lambda _match: replacement,
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not find the beamer r2/r3 Slurm status line")
    return updated


def sync_paper(text: str, payload: dict[str, Any], readiness: dict[str, Any]) -> str:
    values = readiness_values(readiness)
    if values.get("strict_final_ready"):
        pass_count, partial_count, blocked_count = 17, 0, 0
    elif values.get("host_matrix_ready"):
        pass_count, partial_count, blocked_count = 16, 0, 1
    else:
        pass_count, partial_count, blocked_count = 15, 1, 1
    blocked_phrase = (
        f"{blocked_count} 个为 `blocked`。"
        if blocked_count == 0
        else "唯一 `blocked` 项仍是"
    )
    goal_audit_replacement = (
        f"17 个检查项中 {pass_count} 个为 `pass`，"
        f"{partial_count} 个为 `partial`，"
        f"{blocked_phrase}"
    )
    text, count = re.subn(
        r"17 个检查项中 \d+ 个为 `pass`，\d+ 个为 `partial`，"
        r"(?:(?:唯一 )?`blocked` 项仍是|0 个为 `blocked`。)",
        goal_audit_replacement,
        text,
        count=1,
    )
    if values["p0_restore_rows"]:
        legacy_replacement = (
            f"P0 suites 有 {values['p0_restore_rows']} 条 restore rows；"
            f"physical bridge 覆盖 {values['p0_bridge_rows']} 条，但\n"
            f"`physical_tensor_copy` 行数仍为 {values['p0_copy_rows']}。"
        )
        strict_final_replacement = (
            f"P0 suites 有 {values['p0_restore_rows']} 条 restore rows；"
            "原始 request-detail rows 仍保留\n"
            "`metadata_materialization` 记录，但 strict-final readiness 现在使用显式\n"
            f"promotion overlay：{values['p0_copy_rows']}/"
            f"{values['p0_restore_rows']} 条 restore rows 都有已验证 live\n"
            "`physical_tensor_copy` candidate，因此 effective\n"
            f"`physical_tensor_copy_rows={values['p0_copy_rows']}`、"
            "effective non-final rows 为 0。"
        )
        p0_replacement = (
            strict_final_replacement
            if values.get("strict_final_ready") and values.get("promotion_ready")
            else legacy_replacement
        )
        text, count = re.subn(
            r"P0 suites 有 \d+ 条 restore rows(?:；physical bridge 覆盖 \d+ 条)?，?但\s*"
            r"`physical_tensor_copy` 行数仍为 \d+。",
            p0_replacement,
            text,
            count=1,
        )
        if count != 1 and values.get("strict_final_ready"):
            text, count = re.subn(
                r"P0 suites 有 \d+ 条 restore rows；原始 request-detail rows 仍保留\s*"
                r"`metadata_materialization` 记录，但 strict-final readiness 现在使用显式\s*"
                r"promotion overlay：\d+/\d+ 条 restore rows 都有已验证 live\s*"
                r"`physical_tensor_copy` candidate，因此 effective\s*"
                r"`physical_tensor_copy_rows=\d+`、effective non-final rows 为 0。",
                p0_replacement,
                text,
                count=1,
            )
        if count != 1:
            raise ValueError("Could not find the paper P0 restore status sentence")
    if values["required_pairs"]:
        text, count = re.subn(
            r"same-write-cost\s*best-effort\s*"
            r"恢复收益已有 \d+/\d+ 个 SGLang repeats 通过，"
            r"wait-complete\s*完整物理恢复(?:只有|已有|也已有)\s*"
            r"\d+/\d+ 个 repeats 通过(?:。|，)",
            (
                "same-write-cost best-effort\n"
                "恢复收益已有 "
                f"{values['accepted_pairs'] - values['wait_accepted']}/"
                f"{values['required_pairs'] - values['wait_required']} "
                "个 SGLang repeats 通过，wait-complete 完整物理恢复已有\n"
                f"{values['wait_accepted']}/{values['wait_required']} 个 repeats 通过，"
            ),
            text,
            count=1,
        )
        if count != 1:
            raise ValueError("Could not find the paper recovery matrix count sentence")
    checked_at = str(payload.get("checked_at") or "")
    overall = str(payload.get("overall_status") or "unknown")
    counts = status_counts(payload)
    replacement = (
        f"最新 Slurm poll（`{checked_at}`）显示这 4 个作业的当前状态为\n"
        f"`{overall}`：`pending={counts['pending']}`、"
        f"`running={counts['running']}`、`completed={counts['completed']}`、"
        f"`failed={counts['failed']}`；每个作业请求"
    )
    updated, count = re.subn(
        r"最新 Slurm poll（`[^`]+`）显示这 4 个作业(?:仍处于|的当前状态为)\s*"
        r"`[^`]+`：`pending=\d+`、`running=\d+`、`completed=\d+`、"
        r"`failed=\d+`；每个作业请求",
        replacement,
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not find the paper Slurm poll status paragraph")
    return updated


def sync_files(
    *,
    status_json: Path,
    paper: Path,
    beamer: Path,
    readiness_json: Path | None = DEFAULT_READINESS_JSON,
) -> dict[str, Any]:
    payload = read_status(status_json)
    readiness = read_optional_json(readiness_json)
    paper.write_text(
        sync_paper(paper.read_text(encoding="utf-8"), payload, readiness),
        encoding="utf-8",
    )
    beamer.write_text(
        sync_beamer(beamer.read_text(encoding="utf-8"), payload, readiness),
        encoding="utf-8",
    )
    return {
        "beamer": str(beamer),
        "checked_at": payload.get("checked_at", ""),
        "dataplane_readiness_status": readiness.get("status", ""),
        "paper": str(paper),
        "short_status": short_status(payload),
        "source": str(status_json),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync current Slurm status text into AegisKV paper and beamer files."
    )
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--readiness-json", type=Path, default=DEFAULT_READINESS_JSON)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--beamer", type=Path, default=DEFAULT_BEAMER)
    args = parser.parse_args()

    result = sync_files(
        status_json=args.status_json,
        readiness_json=args.readiness_json,
        paper=args.paper,
        beamer=args.beamer,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
