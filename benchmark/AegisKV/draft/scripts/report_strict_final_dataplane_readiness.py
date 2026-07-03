#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DRAFT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = DRAFT_ROOT / "experiments"
DEFAULT_P0_CSV = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_strict_final_readiness.csv"
)
DEFAULT_MATRIX_JSON = (
    EXPERIMENTS
    / "sglang_small"
    / "results"
    / "strict_final_recovery_matrix_audit"
    / "strict_final_recovery_matrix_audit.json"
)
DEFAULT_SLURM_JSON = (
    EXPERIMENTS
    / "sglang_small"
    / "results"
    / "strict_final_recovery_matrix_audit"
    / "strict_final_bound_slurm_jobs.json"
)
DEFAULT_BRIDGE_CSV = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_physical_bridge_summary.csv"
)
DEFAULT_BRIDGE_ROWS_CSV = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_physical_bridge_rows.csv"
)
DEFAULT_P0_LIVE_AUDITS_JSON = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_live_movement_audits.json"
)
DEFAULT_P0_CANDIDATE_AUDIT_JSON = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_physical_tensor_copy_candidate_audit.json"
)
DEFAULT_P0_CANDIDATE_GENERATION_AUDIT_JSON = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_physical_tensor_copy_candidate_generation_audit.json"
)
DEFAULT_P0_EVENT_TEMPLATE_CSV = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_physical_tensor_copy_events.template.csv"
)
DEFAULT_P0_EVENT_REQUIRED_CSV = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_physical_tensor_copy_events.required.csv"
)
DEFAULT_P0_EVENT_SCHEMA_MD = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_physical_tensor_copy_event_schema.md"
)
DEFAULT_P0_LIVE_COPY_EVENTS_JSON = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_physical_tensor_copy_events.live.json"
)
DEFAULT_P0_LIVE_P2P_PLAN_JSON = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_live_p2p_collection_plan.json"
)
DEFAULT_P0_PROMOTION_JSON = (
    EXPERIMENTS
    / "sglsim_large"
    / "results"
    / "strict_final_readiness"
    / "p0_strict_final_readiness.promoted.json"
)
DEFAULT_PHYSICAL_RESTORE_ROOT = (
    EXPERIMENTS / "sglang_small" / "results" / "physical_restore_evidence"
)
OUT_JSON = EXPERIMENTS / "strict_final_dataplane_readiness.json"
OUT_MD = EXPERIMENTS / "strict_final_dataplane_readiness.md"

BRIDGE_DIGEST_FIELDS = (
    "suite_label",
    "request_csv",
    "rid",
    "restore_action",
    "restored_tokens",
    "physical_action",
)
EVENT_SCHEMA_VERSION = "p0-live-copy-event-v1"


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
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("suite_label") or "").strip(),
        str(row.get("request_csv") or "").strip(),
        str(row.get("rid") or "").strip(),
    )


def parse_counter_map(value: object) -> dict[str, int]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {str(key): as_int(count) for key, count in decoded.items()}


def merge_counter_map(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        for name, count in parse_counter_map(row.get(key)).items():
            out[name] = out.get(name, 0) + count
    return out


def bridge_required_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    required: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if not as_bool(row.get("bridge_eligible")):
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


def summarize_bridge_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    required = bridge_required_rows(rows)
    return {
        "bridge_rows_csv_rows": len(rows),
        "required_digest": bridge_required_digest(required),
        "required_rows": len(required),
    }


def collect_physical_restore_csvs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    paths: list[Path] = []
    primary = root / "physical_restore_runs.csv"
    if primary.exists():
        paths.append(primary)
    paths.extend(
        path
        for path in sorted(root.glob("*/physical_restore_runs.csv"))
        if path != primary
    )
    return paths


def summarize_p0(rows: list[dict[str, str]]) -> dict[str, Any]:
    restore_rows = sum(as_int(row.get("restore_rows")) for row in rows)
    physical_rows = sum(as_int(row.get("physical_tensor_copy_rows")) for row in rows)
    nonfinal_rows = sum(as_int(row.get("nonfinal_restore_rows")) for row in rows)
    missing_data_plane_rows = sum(as_int(row.get("missing_data_plane_rows")) for row in rows)
    missing_timing_source_rows = sum(
        as_int(row.get("missing_timing_source_rows")) for row in rows
    )
    schema_complete = bool(rows) and all(as_bool(row.get("schema_complete")) for row in rows)
    all_final_ready = bool(rows) and all(as_bool(row.get("final_ready")) for row in rows)
    data_planes = merge_counter_map(rows, "data_planes")
    timing_sources = merge_counter_map(rows, "timing_sources")
    final_ready = (
        bool(rows)
        and restore_rows > 0
        and physical_rows == restore_rows
        and missing_data_plane_rows == 0
        and missing_timing_source_rows == 0
        and schema_complete
        and all_final_ready
    )
    return {
        "data_planes": data_planes,
        "final_ready": final_ready,
        "missing_data_plane_rows": missing_data_plane_rows,
        "missing_timing_source_rows": missing_timing_source_rows,
        "nonfinal_restore_rows": nonfinal_rows,
        "physical_tensor_copy_rows": physical_rows,
        "restore_rows": restore_rows,
        "schema_complete": schema_complete,
        "suite_count": len(rows),
        "timing_sources": timing_sources,
    }


def summarize_bridge(rows: list[dict[str, str]]) -> dict[str, Any]:
    restore_rows = sum(as_int(row.get("restore_rows")) for row in rows)
    eligible_rows = sum(as_int(row.get("bridge_eligible_rows")) for row in rows)
    missing_rows = sum(as_int(row.get("bridge_missing_rows")) for row in rows)
    bridge_ready = bool(rows) and restore_rows > 0 and eligible_rows == restore_rows
    action_counts: dict[str, int] = {}
    physical_counts: dict[str, int] = {}
    evidence_tiers: set[str] = set()
    for row in rows:
        for key, value in parse_counter_map(row.get("restore_actions")).items():
            action_counts[key] = action_counts.get(key, 0) + value
        for key, value in parse_counter_map(row.get("physical_actions")).items():
            physical_counts[key] = physical_counts.get(key, 0) + value
        try:
            tiers = json.loads(str(row.get("evidence_tiers") or "[]"))
        except json.JSONDecodeError:
            tiers = []
        if isinstance(tiers, list):
            evidence_tiers.update(str(item) for item in tiers if str(item))
    return {
        "bridge_eligible_rows": eligible_rows,
        "bridge_missing_rows": missing_rows,
        "bridge_ready": bridge_ready,
        "evidence_tiers": sorted(evidence_tiers),
        "physical_actions": dict(sorted(physical_counts.items())),
        "restore_actions": dict(sorted(action_counts.items())),
        "restore_rows": restore_rows,
        "suite_count": len(rows),
    }


def summarize_p0_live_audits(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    suites = payload.get("suites") if isinstance(payload.get("suites"), list) else []
    exists = bool(payload)
    return {
        "exists": exists,
        "final_ready": bool(summary.get("final_ready")) if exists else False,
        "missing_data_plane_rows": as_int(summary.get("missing_data_plane_rows")),
        "missing_timing_source_rows": as_int(summary.get("missing_timing_source_rows")),
        "nonfinal_restore_rows": as_int(summary.get("nonfinal_restore_rows")),
        "physical_tensor_copy_rows": as_int(summary.get("physical_tensor_copy_rows")),
        "restore_rows": as_int(summary.get("restore_rows")),
        "schema_complete": bool(summary.get("schema_complete")) if exists else False,
        "source": str(payload.get("source") or ""),
        "suite_count": as_int(summary.get("suite_count")),
        "suites": [
            {
                "figures": str(item.get("figures") or ""),
                "final_ready": bool(item.get("final_ready")),
                "label": str(item.get("label") or ""),
                "nonfinal_restore_rows": as_int(item.get("nonfinal_restore_rows")),
                "physical_tensor_copy_rows": as_int(
                    item.get("physical_tensor_copy_rows")
                ),
                "restore_rows": as_int(item.get("restore_rows")),
            }
            for item in suites
            if isinstance(item, dict)
        ],
    }


def p0_live_matches_p0(p0_live: dict[str, Any], p0: dict[str, Any]) -> bool:
    if not p0_live.get("exists"):
        return True
    keys = [
        "missing_data_plane_rows",
        "missing_timing_source_rows",
        "nonfinal_restore_rows",
        "physical_tensor_copy_rows",
        "restore_rows",
        "schema_complete",
        "suite_count",
    ]
    return all(p0_live.get(key) == p0.get(key) for key in keys)


def summarize_candidate_audit(payload: dict[str, Any]) -> dict[str, Any]:
    invalid_rows = payload.get("invalid_rows")
    if not isinstance(invalid_rows, list):
        invalid_rows = []
    return {
        "accepted_rows": as_int(payload.get("accepted_rows")),
        "bridge_required_digest": str(payload.get("bridge_required_digest") or ""),
        "bridge_rows": str(payload.get("bridge_rows") or ""),
        "candidate_csv": str(payload.get("candidate_csv") or ""),
        "candidate_rows": as_int(payload.get("candidate_rows")),
        "exists": bool(payload),
        "extra_rows": as_int(payload.get("extra_rows")),
        "invalid_rows": len(invalid_rows),
        "missing_rows": as_int(payload.get("missing_rows")),
        "required_rows": as_int(payload.get("required_rows")),
        "source": str(payload.get("source") or ""),
        "status": str(payload.get("status") or "missing_audit_json"),
        "strict_final_candidate_ready": bool(
            payload.get("strict_final_candidate_ready")
        ),
    }


def summarize_candidate_generation_audit(payload: dict[str, Any]) -> dict[str, Any]:
    missing_sources = payload.get("missing_event_sources")
    if not isinstance(missing_sources, list):
        missing_sources = []
    sources = payload.get("event_sources")
    if not isinstance(sources, list):
        sources = []
    return {
        "accepted_rows": as_int(payload.get("accepted_rows")),
        "bridge_required_digest": str(payload.get("bridge_required_digest") or ""),
        "bridge_rows": str(payload.get("bridge_rows") or ""),
        "event_schema_version": str(payload.get("event_schema_version") or ""),
        "event_rows": as_int(payload.get("event_rows")),
        "event_sources": [str(item) for item in sources],
        "exists": bool(payload),
        "missing_event_sources": [str(item) for item in missing_sources],
        "missing_event_source_count": len(missing_sources),
        "missing_rows": as_int(payload.get("missing_rows")),
        "rejected_event_rows": as_int(payload.get("rejected_event_rows")),
        "required_rows": as_int(payload.get("required_rows")),
        "source": str(payload.get("source") or ""),
        "status": str(payload.get("status") or "missing_generation_audit_json"),
        "strict_final_candidate_ready": bool(
            payload.get("strict_final_candidate_ready")
        ),
    }


def summarize_live_copy_events(payload: dict[str, Any]) -> dict[str, Any]:
    event_rows_by_kind = payload.get("event_rows_by_kind")
    if not isinstance(event_rows_by_kind, dict):
        event_rows_by_kind = {}
    skipped_rows_by_reason = payload.get("skipped_rows_by_reason")
    if not isinstance(skipped_rows_by_reason, dict):
        skipped_rows_by_reason = {}
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    peer_access = environment.get("peer_access")
    if not isinstance(peer_access, dict):
        peer_access = {}
    latency = payload.get("latency_by_kind")
    if not isinstance(latency, dict):
        latency = {}
    return {
        "bytes_per_token": as_int(payload.get("bytes_per_token")),
        "copy_backend": str(payload.get("copy_backend") or ""),
        "cuda_available": bool(environment.get("cuda_available")),
        "device_count": as_int(environment.get("device_count")),
        "event_kind_filter": str(payload.get("event_kind_filter") or ""),
        "event_rows": as_int(payload.get("event_rows")),
        "event_rows_by_kind": {
            str(key): as_int(value) for key, value in event_rows_by_kind.items()
        },
        "exists": bool(payload),
        "latency_by_kind": latency,
        "measurement_repeats": as_int(payload.get("measurement_repeats")),
        "output_csv": str(payload.get("output_csv") or ""),
        "peer_access": {str(key): bool(value) for key, value in peer_access.items()},
        "required_rows": as_int(payload.get("required_rows")),
        "selected_rows": as_int(payload.get("selected_rows")),
        "skipped_rows": as_int(payload.get("skipped_rows")),
        "skipped_rows_by_reason": {
            str(key): as_int(value) for key, value in skipped_rows_by_reason.items()
        },
        "strict_final_candidate_ready": bool(
            payload.get("strict_final_candidate_ready")
        ),
        "warmup_repeats": as_int(payload.get("warmup_repeats")),
    }


def summarize_p2p_collection_plan(payload: dict[str, Any]) -> dict[str, Any]:
    commands = payload.get("commands")
    if not isinstance(commands, dict):
        commands = {}
    return {
        "commands": {str(key): str(value) for key, value in commands.items()},
        "exists": bool(payload),
        "p2p_live_copy_event_rows": as_int(
            payload.get("p2p_live_copy_event_rows")
        ),
        "p2p_live_copy_skipped_rows": as_int(
            payload.get("p2p_live_copy_skipped_rows")
        ),
        "p2p_ready": bool(payload.get("p2p_ready")),
        "remote_host": str(payload.get("remote_host") or ""),
        "remote_workdir": str(payload.get("remote_workdir") or ""),
        "required_p2p_rows": as_int(payload.get("required_p2p_rows")),
        "required_p2p_tokens": as_int(payload.get("required_p2p_tokens")),
        "sbatch": str(payload.get("sbatch") or ""),
        "source": str(payload.get("source") or ""),
        "status": str(payload.get("status") or "missing_p2p_collection_plan"),
    }


def summarize_p0_promotion(payload: dict[str, Any]) -> dict[str, Any]:
    problems = payload.get("problems")
    if not isinstance(problems, list):
        problems = []
    return {
        "exists": bool(payload),
        "host_h2d_candidate_rows": as_int(payload.get("host_h2d_candidate_rows")),
        "nonfinal_restore_rows": as_int(payload.get("nonfinal_restore_rows")),
        "peer_p2p_candidate_rows": as_int(payload.get("peer_p2p_candidate_rows")),
        "physical_tensor_copy_rows": as_int(
            payload.get("physical_tensor_copy_rows")
        ),
        "problems": [str(item) for item in problems],
        "promoted_rows": as_int(payload.get("promoted_rows")),
        "promotion_ready": bool(payload.get("promotion_ready")),
        "restore_rows": as_int(payload.get("restore_rows")),
        "source": str(payload.get("source") or ""),
        "status": str(payload.get("status") or "missing_promotion_json"),
    }


def summarize_event_template(
    template_csv: Path,
    schema_md: Path,
    required_csv: Path | None = None,
) -> dict[str, Any]:
    fields: list[str] = []
    row_count = 0
    if template_csv.exists() and template_csv.stat().st_size > 0:
        with template_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            row_count = sum(1 for _ in reader)
    required_rows = read_csv(required_csv) if required_csv is not None else []
    required_event_kinds: dict[str, int] = {}
    required_restore_actions: dict[str, int] = {}
    for row in required_rows:
        event_kind = str(row.get("event_kind") or "").strip()
        restore_action = str(row.get("restore_action") or "").strip()
        if event_kind:
            required_event_kinds[event_kind] = required_event_kinds.get(event_kind, 0) + 1
        if restore_action:
            required_restore_actions[restore_action] = (
                required_restore_actions.get(restore_action, 0) + 1
            )
    return {
        "csv": str(template_csv),
        "csv_exists": template_csv.exists(),
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "fields": fields,
        "required_csv": str(required_csv) if required_csv is not None else "",
        "required_csv_exists": bool(required_csv and required_csv.exists()),
        "required_event_kinds": dict(sorted(required_event_kinds.items())),
        "required_restore_actions": dict(sorted(required_restore_actions.items())),
        "required_rows": len(required_rows),
        "schema_md": str(schema_md),
        "schema_md_exists": schema_md.exists(),
        "template_rows": row_count,
    }


def summarize_sglang_physical_restore(root: Path) -> dict[str, Any]:
    csvs = collect_physical_restore_csvs(root)
    rows: list[dict[str, str]] = []
    for path in csvs:
        rows.extend(read_csv(path))

    states: dict[str, int] = {}
    for row in rows:
        state = str(row.get("physical_restore_state") or "").strip()
        if state:
            states[state] = states.get(state, 0) + 1

    full_rows = states.get("full_prefetch_materialized", 0)
    partial_rows = states.get("partial_async_materialized", 0)
    materialized_rows = full_rows + partial_rows
    hidden_control_rows = states.get("host_frontier_not_visible_to_retry", 0)
    job_ids = sorted(
        {
            str(row.get("slurm_job_id") or "").strip()
            for row in rows
            if str(row.get("slurm_job_id") or "").strip()
        }
    )
    return {
        "csv_count": len(csvs),
        "full_prefetch_materialized_rows": full_rows,
        "hidden_control_rows": hidden_control_rows,
        "job_ids": job_ids,
        "materialized_rows": materialized_rows,
        "max_data_pages": max((as_int(row.get("max_data_pages")) for row in rows), default=0),
        "partial_async_materialized_rows": partial_rows,
        "prefetch_io_completed_tokens": sum(
            as_int(row.get("prefetch_io_completed_tokens_sum")) for row in rows
        ),
        "ready_for_host_matrix": materialized_rows > 0 and hidden_control_rows > 0,
        "retry_cached_tokens_after_accept": sum(
            as_int(row.get("retry_cached_tokens_after_accept_total")) for row in rows
        ),
        "rows": len(rows),
        "source_root": str(root),
        "states": dict(sorted(states.items())),
    }


def summarize_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    case_rows = matrix.get("case_rows") if isinstance(matrix.get("case_rows"), list) else []
    cases: dict[str, dict[str, Any]] = {}
    accepted_pairs = 0
    required_pairs = 0
    for row in case_rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        required = as_int(row.get("required_repeats"))
        accepted = as_int(row.get("accepted_pairs"))
        required_pairs += required
        accepted_pairs += accepted
        cases[case_id] = {
            "accepted_pairs": accepted,
            "observed_pairs": as_int(row.get("observed_pairs")),
            "required_repeats": required,
            "status": str(row.get("status") or ""),
        }
    return {
        "accepted_pairs": accepted_pairs,
        "cases": cases,
        "matrix_ready": bool(matrix.get("matrix_ready")),
        "overall_status": str(matrix.get("overall_status") or ""),
        "physical_restore_csv_count": as_int(matrix.get("physical_restore_csv_count")),
        "physical_restore_rows": as_int(matrix.get("physical_restore_rows")),
        "required_pairs": required_pairs,
    }


def summarize_slurm(slurm: dict[str, Any]) -> dict[str, Any]:
    counts = slurm.get("status_counts") if isinstance(slurm.get("status_counts"), dict) else {}
    return {
        "checked_at": str(slurm.get("checked_at") or ""),
        "overall_status": str(slurm.get("overall_status") or ""),
        "completed": as_int(counts.get("completed")),
        "failed": as_int(counts.get("failed")),
        "pending": as_int(counts.get("pending")),
        "running": as_int(counts.get("running")),
        "total_bound_jobs": as_int(slurm.get("total_bound_jobs")),
    }


def audit_matches_current_bridge(audit: dict[str, Any], bridge_rows: dict[str, Any]) -> bool:
    if not audit.get("exists"):
        return True
    if as_int(audit.get("required_rows")) != as_int(bridge_rows.get("required_rows")):
        return False
    audit_digest = str(audit.get("bridge_required_digest") or "")
    current_digest = str(bridge_rows.get("required_digest") or "")
    return bool(audit_digest) and audit_digest == current_digest


def build_payload(
    *,
    p0_csv: Path = DEFAULT_P0_CSV,
    p0_live_audits_json: Path = DEFAULT_P0_LIVE_AUDITS_JSON,
    p0_candidate_audit_json: Path = DEFAULT_P0_CANDIDATE_AUDIT_JSON,
    p0_candidate_generation_audit_json: Path = (
        DEFAULT_P0_CANDIDATE_GENERATION_AUDIT_JSON
    ),
    p0_event_template_csv: Path = DEFAULT_P0_EVENT_TEMPLATE_CSV,
    p0_event_required_csv: Path = DEFAULT_P0_EVENT_REQUIRED_CSV,
    p0_event_schema_md: Path = DEFAULT_P0_EVENT_SCHEMA_MD,
    p0_live_copy_events_json: Path = DEFAULT_P0_LIVE_COPY_EVENTS_JSON,
    p0_live_p2p_plan_json: Path = DEFAULT_P0_LIVE_P2P_PLAN_JSON,
    p0_promotion_json: Path = DEFAULT_P0_PROMOTION_JSON,
    matrix_json: Path = DEFAULT_MATRIX_JSON,
    slurm_json: Path = DEFAULT_SLURM_JSON,
    bridge_csv: Path = DEFAULT_BRIDGE_CSV,
    bridge_rows_csv: Path = DEFAULT_BRIDGE_ROWS_CSV,
    physical_restore_root: Path = DEFAULT_PHYSICAL_RESTORE_ROOT,
) -> dict[str, Any]:
    p0 = summarize_p0(read_csv(p0_csv))
    p0_live_audits = summarize_p0_live_audits(read_json(p0_live_audits_json))
    p0_live_consistent = p0_live_matches_p0(p0_live_audits, p0)
    candidate_audit = summarize_candidate_audit(read_json(p0_candidate_audit_json))
    candidate_generation_audit = summarize_candidate_generation_audit(
        read_json(p0_candidate_generation_audit_json)
    )
    event_template = summarize_event_template(
        p0_event_template_csv,
        p0_event_schema_md,
        p0_event_required_csv,
    )
    live_copy_events = summarize_live_copy_events(read_json(p0_live_copy_events_json))
    p2p_collection_plan = summarize_p2p_collection_plan(
        read_json(p0_live_p2p_plan_json)
    )
    p0_promotion = summarize_p0_promotion(read_json(p0_promotion_json))
    bridge = summarize_bridge(read_csv(bridge_csv))
    bridge_rows = summarize_bridge_rows(read_csv(bridge_rows_csv))
    candidate_audit_bridge_consistent = audit_matches_current_bridge(
        candidate_audit,
        bridge_rows,
    )
    candidate_generation_bridge_consistent = audit_matches_current_bridge(
        candidate_generation_audit,
        bridge_rows,
    )
    matrix = summarize_matrix(read_json(matrix_json))
    slurm = summarize_slurm(read_json(slurm_json))
    sglang_physical_restore = summarize_sglang_physical_restore(physical_restore_root)
    host_matrix_ready = (
        matrix["matrix_ready"]
        and matrix["physical_restore_rows"] > 0
        and matrix["accepted_pairs"] >= matrix["required_pairs"] > 0
    )
    promotion_bridge_consistent = (
        not p0_promotion["exists"]
        or p0_promotion["restore_rows"] == bridge_rows["required_rows"]
    )
    promoted_p0_ready = bool(
        p0_promotion["promotion_ready"]
        and promotion_bridge_consistent
        and candidate_audit["strict_final_candidate_ready"]
        and candidate_audit_bridge_consistent
    )
    strict_p0_ready = bool(
        (
            p0["final_ready"]
            and (
                not p0_live_audits["exists"]
                or (p0_live_consistent and p0_live_audits["final_ready"])
            )
        )
        or promoted_p0_ready
    )
    if not p0_live_consistent:
        status = "p0_live_audit_mismatch"
    elif strict_p0_ready:
        status = "strict_final_ready"
    elif not host_matrix_ready and bridge["bridge_ready"]:
        status = "waiting_on_matrix_and_live_p0_physical_copy"
    elif not host_matrix_ready:
        status = "waiting_on_matrix_and_p0_physical_bridge"
    elif bridge["bridge_ready"] and not candidate_audit["strict_final_candidate_ready"]:
        status = "host_matrix_ready_waiting_on_live_p0_physical_copy_candidates"
    elif bridge["bridge_ready"]:
        status = "host_matrix_ready_physical_copy_candidates_ready_p0_rows_not_promoted"
    else:
        status = "host_matrix_ready_p0_physical_bridge_missing"
    actions: list[str] = []
    if not p0_live_consistent:
        actions.append(
            "regenerate P0 live movement audits and P0 strict-final summary until their restore/data-plane counts match"
        )
    if candidate_generation_audit["missing_event_source_count"] > 0:
        actions.append(
            "fix missing live physical-copy event source paths before rebuilding P0 physical_tensor_copy candidates"
        )
    if not candidate_generation_bridge_consistent:
        actions.append(
            "regenerate P0 physical_tensor_copy candidate generation audit against the current bridge rows"
        )
    if not candidate_audit_bridge_consistent:
        actions.append(
            "rerun validate_p0_physical_tensor_copy_candidates.py against the current bridge rows"
        )
    if not host_matrix_ready:
        actions.append(
            "finish and sync the bound real SGLang wait-complete r2/r3 jobs"
        )
    if not p0["final_ready"]:
        if bridge["bridge_ready"]:
            if candidate_audit["strict_final_candidate_ready"]:
                if not p0_promotion["promotion_ready"]:
                    actions.append(
                        "promote the accepted live physical_tensor_copy candidate evidence into P0 strict-final policy rows; current rows remain metadata_materialization"
                    )
            else:
                if live_copy_events["exists"] and live_copy_events["skipped_rows"] > 0:
                    actions.append(
                        "collect the remaining live peer GPUReplica P2P tensor-copy rows on hardware with CUDA peer access; current local run accepted "
                        f"{live_copy_events['event_rows']} live rows and skipped "
                        f"{live_copy_events['skipped_rows']} rows"
                    )
                if (
                    p2p_collection_plan["exists"]
                    and not p2p_collection_plan["p2p_ready"]
                ):
                    actions.append(
                        "submit and sync the prepared P0 live P2P collection plan "
                        f"({p2p_collection_plan['required_p2p_rows']} required rows) "
                        f"from {p2p_collection_plan['remote_host']}:{p2p_collection_plan['remote_workdir']}"
                    )
                actions.append(
                    "provide and validate live physical_tensor_copy candidate rows for every bridged P0 restore row; P0 calibration bridge is complete but current rows remain metadata_materialization"
                )
        else:
            actions.append(
                "replace or bridge SGLSim P0 metadata_materialization restore rows with validated physical_tensor_copy evidence"
            )
    return {
        "bridge": bridge,
        "bridge_rows": bridge_rows,
        "host_matrix_ready": host_matrix_ready,
        "inputs": {
            "bridge_csv": str(bridge_csv),
            "bridge_rows_csv": str(bridge_rows_csv),
            "matrix_json": str(matrix_json),
            "physical_restore_root": str(physical_restore_root),
            "p0_candidate_audit_json": str(p0_candidate_audit_json),
            "p0_candidate_generation_audit_json": str(
                p0_candidate_generation_audit_json
            ),
            "p0_event_schema_md": str(p0_event_schema_md),
            "p0_event_required_csv": str(p0_event_required_csv),
            "p0_event_template_csv": str(p0_event_template_csv),
            "p0_live_copy_events_json": str(p0_live_copy_events_json),
            "p0_live_p2p_plan_json": str(p0_live_p2p_plan_json),
            "p0_promotion_json": str(p0_promotion_json),
            "p0_live_audits_json": str(p0_live_audits_json),
            "p0_csv": str(p0_csv),
            "slurm_json": str(slurm_json),
        },
        "matrix": matrix,
        "next_required_actions": actions,
        "p0": p0,
        "p0_candidate_audit_bridge_consistent": candidate_audit_bridge_consistent,
        "p0_candidate_generation_bridge_consistent": (
            candidate_generation_bridge_consistent
        ),
        "p0_physical_tensor_copy_candidate_generation_audit": (
            candidate_generation_audit
        ),
        "p0_physical_tensor_copy_candidate_audit": candidate_audit,
        "p0_physical_tensor_copy_event_template": event_template,
        "p0_live_copy_events": live_copy_events,
        "p0_live_p2p_collection_plan": p2p_collection_plan,
        "p0_promotion": p0_promotion,
        "p0_promotion_bridge_consistent": promotion_bridge_consistent,
        "p0_live_audits": p0_live_audits,
        "p0_live_audits_consistent": p0_live_consistent,
        "sglang_physical_restore": sglang_physical_restore,
        "slurm": slurm,
        "source": "report_strict_final_dataplane_readiness",
        "status": status,
        "strict_final_ready": strict_p0_ready,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    p0 = payload["p0"]
    p0_live = payload["p0_live_audits"]
    candidate_generation = payload[
        "p0_physical_tensor_copy_candidate_generation_audit"
    ]
    candidate = payload["p0_physical_tensor_copy_candidate_audit"]
    event_template = payload["p0_physical_tensor_copy_event_template"]
    live_copy_events = payload["p0_live_copy_events"]
    p2p_plan = payload["p0_live_p2p_collection_plan"]
    p0_promotion = payload["p0_promotion"]
    bridge = payload["bridge"]
    bridge_rows = payload["bridge_rows"]
    matrix = payload["matrix"]
    physical_restore = payload["sglang_physical_restore"]
    slurm = payload["slurm"]
    lines = [
        "# Strict-Final Data-Plane Readiness",
        "",
        "This report separates final policy-sweep rows from real SGLang recovery-path evidence.",
        "It must not relabel SGLSim metadata materialization as physical tensor KV movement.",
        "",
        f"- Status: `{payload['status']}`",
        f"- Strict-final ready: `{str(payload['strict_final_ready']).lower()}`",
        f"- Host recovery matrix ready: `{str(payload['host_matrix_ready']).lower()}`",
        "",
        "## P0 SGLSim Policy Rows",
        "",
        f"- Restore rows: {p0['restore_rows']}",
        f"- Physical tensor copy rows: {p0['physical_tensor_copy_rows']}",
        f"- Non-final restore rows: {p0['nonfinal_restore_rows']}",
        f"- Data planes: `{p0['data_planes']}`",
        f"- Timing sources: `{p0['timing_sources']}`",
        "",
        "## P0 Per-Suite Live Movement Audits",
        "",
        f"- Audit exists: `{str(p0_live['exists']).lower()}`",
        f"- Consistent with P0 summary: `{str(payload['p0_live_audits_consistent']).lower()}`",
        f"- Restore rows: {p0_live['restore_rows']}",
        f"- Physical tensor copy rows: {p0_live['physical_tensor_copy_rows']}",
        f"- Non-final restore rows: {p0_live['nonfinal_restore_rows']}",
        f"- Missing data-plane rows: {p0_live['missing_data_plane_rows']}",
        f"- Missing timing-source rows: {p0_live['missing_timing_source_rows']}",
        f"- Schema complete: `{str(p0_live['schema_complete']).lower()}`",
        "",
        "## P0 Physical Calibration Bridge",
        "",
        f"- Bridge ready: `{str(bridge['bridge_ready']).lower()}`",
        f"- Bridge eligible rows: {bridge['bridge_eligible_rows']}/{bridge['restore_rows']}",
        f"- Bridge missing rows: {bridge['bridge_missing_rows']}",
        f"- Bridge detail required rows: {bridge_rows['required_rows']}",
        f"- Bridge detail digest: `{bridge_rows['required_digest']}`",
        f"- Restore actions: `{bridge['restore_actions']}`",
        f"- Physical actions: `{bridge['physical_actions']}`",
        f"- Evidence tiers: `{bridge['evidence_tiers']}`",
        "",
        "## P0 Live Physical Tensor-Copy Candidate Audit",
        "",
        f"- Audit exists: `{str(candidate['exists']).lower()}`",
        f"- Status: `{candidate['status']}`",
        f"- Strict-final candidate ready: `{str(candidate['strict_final_candidate_ready']).lower()}`",
        f"- Consistent with current bridge rows: `{str(payload['p0_candidate_audit_bridge_consistent']).lower()}`",
        f"- Candidate bridge digest: `{candidate['bridge_required_digest']}`",
        f"- Required rows: {candidate['required_rows']}",
        f"- Candidate rows: {candidate['candidate_rows']}",
        f"- Accepted rows: {candidate['accepted_rows']}",
        f"- Missing rows: {candidate['missing_rows']}",
        f"- Invalid rows: {candidate['invalid_rows']}",
        f"- Extra rows: {candidate['extra_rows']}",
        "",
        "## P0 Live Physical Tensor-Copy Candidate Generation",
        "",
        f"- Audit exists: `{str(candidate_generation['exists']).lower()}`",
        f"- Status: `{candidate_generation['status']}`",
        f"- Consistent with current bridge rows: `{str(payload['p0_candidate_generation_bridge_consistent']).lower()}`",
        f"- Generation bridge digest: `{candidate_generation['bridge_required_digest']}`",
        f"- Event schema version: `{candidate_generation['event_schema_version']}`",
        f"- Event rows: {candidate_generation['event_rows']}",
        f"- Event sources: {len(candidate_generation['event_sources'])}",
        f"- Missing event sources: {candidate_generation['missing_event_source_count']}",
        f"- Accepted rows: {candidate_generation['accepted_rows']}",
        f"- Missing rows: {candidate_generation['missing_rows']}",
        f"- Rejected event rows: {candidate_generation['rejected_event_rows']}",
        "",
        "## P0 Live Physical Tensor-Copy Event Template",
        "",
        f"- Event schema version: `{event_template['event_schema_version']}`",
        f"- CSV exists: `{str(event_template['csv_exists']).lower()}`",
        f"- Schema doc exists: `{str(event_template['schema_md_exists']).lower()}`",
        f"- CSV: `{event_template['csv']}`",
        f"- Schema doc: `{event_template['schema_md']}`",
        f"- Template data rows: {event_template['template_rows']}",
        f"- Template fields: `{event_template['fields']}`",
        f"- Required-event manifest exists: `{str(event_template['required_csv_exists']).lower()}`",
        f"- Required-event manifest: `{event_template['required_csv']}`",
        f"- Required-event rows: {event_template['required_rows']}",
        f"- Required event kinds: `{event_template['required_event_kinds']}`",
        f"- Required restore actions: `{event_template['required_restore_actions']}`",
        "",
        "## P0 Live Physical Tensor-Copy Collection",
        "",
        f"- Collection exists: `{str(live_copy_events['exists']).lower()}`",
        f"- Output CSV: `{live_copy_events['output_csv']}`",
        f"- Selected rows: {live_copy_events['selected_rows']}/{live_copy_events['required_rows']}",
        f"- Verified event rows: {live_copy_events['event_rows']}",
        f"- Event rows by kind: `{live_copy_events['event_rows_by_kind']}`",
        f"- Skipped rows: {live_copy_events['skipped_rows']}",
        f"- Skipped rows by reason: `{live_copy_events['skipped_rows_by_reason']}`",
        f"- CUDA device count: {live_copy_events['device_count']}",
        f"- CUDA peer access: `{live_copy_events['peer_access']}`",
        f"- Measurement repeats: {live_copy_events['measurement_repeats']}",
        f"- Bytes per token: {live_copy_events['bytes_per_token']}",
        "",
        "## P0 Live P2P Collection Plan",
        "",
        f"- Plan exists: `{str(p2p_plan['exists']).lower()}`",
        f"- Status: `{p2p_plan['status']}`",
        f"- P2P ready: `{str(p2p_plan['p2p_ready']).lower()}`",
        f"- Required P2P rows: {p2p_plan['required_p2p_rows']}",
        f"- Required P2P tokens: {p2p_plan['required_p2p_tokens']}",
        f"- Existing P2P live rows: {p2p_plan['p2p_live_copy_event_rows']}",
        f"- Existing P2P skipped rows: {p2p_plan['p2p_live_copy_skipped_rows']}",
        f"- Remote target: `{p2p_plan['remote_host']}:{p2p_plan['remote_workdir']}`",
        f"- Sbatch: `{p2p_plan['sbatch']}`",
        "",
        "## P0 Promoted Strict-Final Overlay",
        "",
        f"- Promotion exists: `{str(p0_promotion['exists']).lower()}`",
        f"- Status: `{p0_promotion['status']}`",
        f"- Promotion ready: `{str(p0_promotion['promotion_ready']).lower()}`",
        f"- Consistent with current bridge rows: `{str(payload['p0_promotion_bridge_consistent']).lower()}`",
        f"- Restore rows: {p0_promotion['restore_rows']}",
        f"- Physical tensor copy rows: {p0_promotion['physical_tensor_copy_rows']}",
        f"- Non-final restore rows: {p0_promotion['nonfinal_restore_rows']}",
        f"- Host H2D candidate rows: {p0_promotion['host_h2d_candidate_rows']}",
        f"- Peer P2P candidate rows: {p0_promotion['peer_p2p_candidate_rows']}",
        f"- Problems: `{p0_promotion['problems']}`",
        "",
        "## Real SGLang Host Recovery Matrix",
        "",
        f"- Matrix status: `{matrix['overall_status']}`",
        f"- Accepted pairs: {matrix['accepted_pairs']}/{matrix['required_pairs']}",
        f"- Physical restore rows: {matrix['physical_restore_rows']}",
        f"- Physical restore CSVs: {matrix['physical_restore_csv_count']}",
        "",
        "## Real SGLang Host Restore Materialization",
        "",
        f"- CSVs: {physical_restore['csv_count']}",
        f"- Rows: {physical_restore['rows']}",
        f"- Materialized rows: {physical_restore['materialized_rows']}",
        f"- Full prefetch materialized rows: {physical_restore['full_prefetch_materialized_rows']}",
        f"- Partial async materialized rows: {physical_restore['partial_async_materialized_rows']}",
        f"- Hidden-control rows: {physical_restore['hidden_control_rows']}",
        f"- Retry cached tokens after accept: {physical_restore['retry_cached_tokens_after_accept']}",
        f"- Prefetch IO completed tokens: {physical_restore['prefetch_io_completed_tokens']}",
        f"- States: `{physical_restore['states']}`",
        "",
        "## Bound Slurm Jobs",
        "",
        f"- Checked at: `{slurm['checked_at']}`",
        f"- Status: `{slurm['overall_status']}`",
        f"- Pending/running/completed/failed: {slurm['pending']}/{slurm['running']}/{slurm['completed']}/{slurm['failed']}",
        "",
        "## Next Required Actions",
        "",
    ]
    actions = payload.get("next_required_actions")
    if isinstance(actions, list) and actions:
        lines.extend(f"- {item}" for item in actions)
    else:
        lines.append("- No remaining data-plane readiness action.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report strict-final data-plane readiness across SGLSim P0 and real SGLang evidence."
    )
    parser.add_argument("--p0-csv", type=Path, default=DEFAULT_P0_CSV)
    parser.add_argument(
        "--p0-live-audits-json",
        type=Path,
        default=DEFAULT_P0_LIVE_AUDITS_JSON,
    )
    parser.add_argument(
        "--p0-candidate-audit-json",
        type=Path,
        default=DEFAULT_P0_CANDIDATE_AUDIT_JSON,
    )
    parser.add_argument(
        "--p0-candidate-generation-audit-json",
        type=Path,
        default=DEFAULT_P0_CANDIDATE_GENERATION_AUDIT_JSON,
    )
    parser.add_argument(
        "--p0-event-template-csv",
        type=Path,
        default=DEFAULT_P0_EVENT_TEMPLATE_CSV,
    )
    parser.add_argument(
        "--p0-event-required-csv",
        type=Path,
        default=DEFAULT_P0_EVENT_REQUIRED_CSV,
    )
    parser.add_argument(
        "--p0-event-schema-md",
        type=Path,
        default=DEFAULT_P0_EVENT_SCHEMA_MD,
    )
    parser.add_argument(
        "--p0-live-copy-events-json",
        type=Path,
        default=DEFAULT_P0_LIVE_COPY_EVENTS_JSON,
    )
    parser.add_argument(
        "--p0-live-p2p-plan-json",
        type=Path,
        default=DEFAULT_P0_LIVE_P2P_PLAN_JSON,
    )
    parser.add_argument(
        "--p0-promotion-json",
        type=Path,
        default=DEFAULT_P0_PROMOTION_JSON,
    )
    parser.add_argument("--bridge-csv", type=Path, default=DEFAULT_BRIDGE_CSV)
    parser.add_argument("--bridge-rows-csv", type=Path, default=DEFAULT_BRIDGE_ROWS_CSV)
    parser.add_argument("--matrix-json", type=Path, default=DEFAULT_MATRIX_JSON)
    parser.add_argument("--slurm-json", type=Path, default=DEFAULT_SLURM_JSON)
    parser.add_argument(
        "--physical-restore-root", type=Path, default=DEFAULT_PHYSICAL_RESTORE_ROOT
    )
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    payload = build_payload(
        p0_csv=args.p0_csv,
        p0_live_audits_json=args.p0_live_audits_json,
        p0_candidate_audit_json=args.p0_candidate_audit_json,
        p0_candidate_generation_audit_json=args.p0_candidate_generation_audit_json,
        p0_event_template_csv=args.p0_event_template_csv,
        p0_event_required_csv=args.p0_event_required_csv,
        p0_event_schema_md=args.p0_event_schema_md,
        p0_live_copy_events_json=args.p0_live_copy_events_json,
        p0_live_p2p_plan_json=args.p0_live_p2p_plan_json,
        p0_promotion_json=args.p0_promotion_json,
        bridge_csv=args.bridge_csv,
        bridge_rows_csv=args.bridge_rows_csv,
        matrix_json=args.matrix_json,
        slurm_json=args.slurm_json,
        physical_restore_root=args.physical_restore_root,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
