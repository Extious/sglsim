#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_PLAN_PAIRS = (
    EXP_DIR
    / "plans"
    / "strict_final_recovery_matrix"
    / "strict_final_recovery_pairs.csv"
)
DEFAULT_PLAN_JSON = (
    EXP_DIR
    / "plans"
    / "strict_final_recovery_matrix"
    / "strict_final_recovery_matrix.json"
)
DEFAULT_PHYSICAL_RESTORE_CSV = (
    RESULTS_DIR / "physical_restore_evidence" / "physical_restore_runs.csv"
)
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "strict_final_recovery_matrix_audit"

CASE_BEST_EFFORT = "same_write_best_effort"
CASE_WAIT_COMPLETE = "same_write_wait_complete"
MATERIALIZED_STATES = {
    "partial_async_materialized",
    "full_prefetch_materialized",
}
PAIR_SUMMARY_DIR_PREFIXES = (
    "failure_visibility_pair",
    "strict_final_same_write",
)


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


def as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def case_from_pair(row: dict[str, Any]) -> str:
    policy = str(row.get("hicache_storage_prefetch_policy") or "").strip()
    run_id = str(row.get("run_id") or "").lower()
    if policy == "best_effort" or "best_effort" in run_id:
        return CASE_BEST_EFFORT
    if policy == "wait_complete" or "wait_complete" in run_id or "same_write" in run_id:
        return CASE_WAIT_COMPLETE
    return ""


def collect_pair_summaries(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/failure_visibility_pair_summary.json")):
        if not path.parent.name.startswith(PAIR_SUMMARY_DIR_PREFIXES):
            continue
        row = read_json(path)
        if not row:
            continue
        row["summary_json"] = str(path)
        row["pair_dir"] = str(path.parent)
        row["case_id"] = case_from_pair(row)
        if row["case_id"] not in {CASE_BEST_EFFORT, CASE_WAIT_COMPLETE}:
            continue
        rows.append(row)
    return rows


def planned_repeats(plan_pairs: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in read_csv(plan_pairs):
        case_id = str(row.get("case_id") or "")
        if case_id:
            counts[case_id] = counts.get(case_id, 0) + 1
    if not counts:
        counts = {CASE_BEST_EFFORT: 3, CASE_WAIT_COMPLETE: 3}
    return counts


def load_plan(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload if payload else {"runs": [], "pairs": []}


def _planned_pairs_by_case(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pair in plan.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        case_id = str(pair.get("case_id") or "")
        grouped.setdefault(case_id, []).append(pair)
    for pairs in grouped.values():
        pairs.sort(key=lambda row: as_int(row.get("repeat")))
    return grouped


def _runs_by_label(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for run in plan.get("runs") or []:
        if isinstance(run, dict) and str(run.get("run_label") or ""):
            out[str(run["run_label"])] = run
    return out


def select_missing_plan_pairs(
    *,
    case_rows: list[dict[str, Any]],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    planned_by_case = _planned_pairs_by_case(plan)
    runs_by_label = _runs_by_label(plan)
    missing: list[dict[str, Any]] = []
    for case in case_rows:
        case_id = str(case.get("case_id") or "")
        required = as_int(case.get("required_repeats"))
        accepted = as_int(case.get("accepted_pairs"))
        missing_count = max(0, required - accepted)
        if missing_count <= 0:
            continue
        planned = planned_by_case.get(case_id, [])
        for pair in planned[accepted : accepted + missing_count]:
            invisible = runs_by_label.get(str(pair.get("invisible_run_label") or ""), {})
            visible = runs_by_label.get(str(pair.get("visible_run_label") or ""), {})
            missing.append(
                {
                    "case_id": case_id,
                    "pair_id": pair.get("pair_id", ""),
                    "repeat": pair.get("repeat", ""),
                    "reason": (
                        f"{case_id} accepted_pairs={accepted}, "
                        f"required_repeats={required}"
                    ),
                    "invisible_run_label": pair.get("invisible_run_label", ""),
                    "visible_run_label": pair.get("visible_run_label", ""),
                    "invisible_submit_command": invisible.get("submit_command", ""),
                    "visible_submit_command": visible.get("submit_command", ""),
                    "sync_pair_command": pair.get("sync_pair_command", ""),
                    "physical_restore_command": pair.get("physical_restore_command", ""),
                }
            )
    return missing


def collect_physical_restore_csvs(primary: Path) -> list[Path]:
    csvs: list[Path] = []
    if primary.exists():
        csvs.append(primary)
    if primary.parent.exists():
        csvs.extend(
            path
            for path in sorted(primary.parent.glob("*/physical_restore_runs.csv"))
            if path != primary
        )

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in csvs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def physical_rows_by_job(paths: Path | list[Path]) -> dict[str, dict[str, str]]:
    csv_paths = (
        collect_physical_restore_csvs(paths) if isinstance(paths, Path) else paths
    )
    rows: dict[str, dict[str, str]] = {}
    for path in csv_paths:
        for row in read_csv(path):
            job = str(row.get("slurm_job_id") or "").strip()
            if job:
                annotated = dict(row)
                annotated["_physical_restore_source_csv"] = str(path)
                rows[job] = annotated
    return rows


def evaluate_pair(
    pair: dict[str, Any],
    physical_by_job: dict[str, dict[str, str]],
) -> dict[str, Any]:
    case_id = str(pair.get("case_id") or "")
    invisible_job = str(pair.get("invisible_slurm_job_id") or "")
    visible_job = str(pair.get("visible_slurm_job_id") or "")
    invisible_physical = physical_by_job.get(invisible_job, {})
    visible_physical = physical_by_job.get(visible_job, {})

    invisible_success = as_int(pair.get("invisible_success_count"))
    visible_success = as_int(pair.get("visible_success_count"))
    invisible_failover = as_int(pair.get("invisible_failover_response_count"))
    visible_failover = as_int(pair.get("visible_failover_response_count"))
    invisible_backed = as_int(pair.get("invisible_pre_failover_backed_up_tokens"))
    visible_backed = as_int(pair.get("visible_pre_failover_backed_up_tokens"))
    retry_norm = as_float(pair.get("visible_retry_latency_norm_vs_invisible"))
    visible_state = str(visible_physical.get("physical_restore_state") or "")
    invisible_state = str(invisible_physical.get("physical_restore_state") or "")
    same_write_cost = invisible_backed > 0 and visible_backed > 0
    success_ok = (
        invisible_success > 0
        and visible_success > 0
        and invisible_success == invisible_failover
        and visible_success == visible_failover
    )
    visible_has_storage_path = (
        as_int(visible_physical.get("max_data_pages")) > 0
        and as_int(visible_physical.get("prefetch_accepted_count")) > 0
    )
    invisible_hidden_control = (
        invisible_state == "host_frontier_not_visible_to_retry"
        and as_int(invisible_physical.get("max_data_pages")) == 0
        and as_int(invisible_physical.get("prefetch_accepted_count")) == 0
        and as_int(invisible_physical.get("retry_cached_tokens_after_accept_total")) == 0
    )
    visible_materialized = visible_state in MATERIALIZED_STATES
    full_materialized = (
        visible_state == "full_prefetch_materialized"
        and as_float(visible_physical.get("prefetch_io_completion_ratio")) >= 1.0
    )

    checks: dict[str, bool] = {
        "success_ok": success_ok,
        "same_write_cost": same_write_cost,
        "invisible_hidden_control": invisible_hidden_control,
        "visible_has_storage_path": visible_has_storage_path,
        "visible_materialized": visible_materialized,
    }
    if case_id == CASE_BEST_EFFORT:
        checks["latency_improves"] = retry_norm > 0 and retry_norm < 1.0
        accepted = all(checks.values())
    elif case_id == CASE_WAIT_COMPLETE:
        checks["full_materialized"] = full_materialized
        accepted = (
            success_ok
            and same_write_cost
            and invisible_hidden_control
            and visible_has_storage_path
            and full_materialized
        )
    else:
        accepted = False

    return {
        "case_id": case_id,
        "run_id": pair.get("run_id", ""),
        "accepted": str(accepted).lower(),
        "status": "accepted" if accepted else "rejected",
        "invisible_job_id": invisible_job,
        "visible_job_id": visible_job,
        "invisible_success_count": invisible_success,
        "visible_success_count": visible_success,
        "invisible_pre_failover_backed_up_tokens": invisible_backed,
        "visible_pre_failover_backed_up_tokens": visible_backed,
        "same_write_cost": str(same_write_cost).lower(),
        "visible_retry_latency_norm_vs_invisible": retry_norm,
        "visible_latency_delta_pct_vs_invisible": as_float(
            pair.get("visible_latency_delta_pct_vs_invisible")
        ),
        "invisible_physical_restore_state": invisible_state,
        "visible_physical_restore_state": visible_state,
        "invisible_physical_restore_source_csv": invisible_physical.get(
            "_physical_restore_source_csv", ""
        ),
        "visible_physical_restore_source_csv": visible_physical.get(
            "_physical_restore_source_csv", ""
        ),
        "invisible_max_data_pages": as_int(invisible_physical.get("max_data_pages")),
        "invisible_prefetch_accepted_count": as_int(
            invisible_physical.get("prefetch_accepted_count")
        ),
        "invisible_retry_cached_tokens_after_accept_total": as_int(
            invisible_physical.get("retry_cached_tokens_after_accept_total")
        ),
        "visible_max_data_pages": as_int(visible_physical.get("max_data_pages")),
        "visible_prefetch_accepted_count": as_int(
            visible_physical.get("prefetch_accepted_count")
        ),
        "visible_prefetch_io_completion_ratio": as_float(
            visible_physical.get("prefetch_io_completion_ratio")
        ),
        "visible_retry_cached_tokens_after_accept_total": as_int(
            visible_physical.get("retry_cached_tokens_after_accept_total")
        ),
        "checks": json.dumps(checks, sort_keys=True),
        "summary_json": pair.get("summary_json", ""),
        "visible_result_dir": pair.get("visible_result_dir", ""),
        "invisible_result_dir": pair.get("invisible_result_dir", ""),
        "evidence_tier": "sglang-small-strict-final-recovery-matrix-audit",
        "source": "audit_strict_final_recovery_matrix",
    }


def summarize_cases(
    pair_rows: list[dict[str, Any]],
    required: dict[str, int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case_id in (CASE_BEST_EFFORT, CASE_WAIT_COMPLETE):
        rows = [row for row in pair_rows if row["case_id"] == case_id]
        accepted = [row for row in rows if str(row["accepted"]) == "true"]
        improving = [
            row
            for row in rows
            if as_float(row.get("visible_retry_latency_norm_vs_invisible")) < 1.0
        ]
        required_repeats = required.get(case_id, 3)
        if len(accepted) >= required_repeats:
            status = "pass"
        elif rows:
            status = "partial"
        else:
            status = "missing"
        norms = [as_float(row.get("visible_retry_latency_norm_vs_invisible")) for row in rows]
        out.append(
            {
                "case_id": case_id,
                "status": status,
                "required_repeats": required_repeats,
                "observed_pairs": len(rows),
                "accepted_pairs": len(accepted),
                "improving_pairs": len(improving),
                "visible_retry_latency_norm_mean": (
                    sum(norms) / len(norms) if norms else 0.0
                ),
                "invisible_job_ids": ";".join(str(row["invisible_job_id"]) for row in rows),
                "visible_job_ids": ";".join(str(row["visible_job_id"]) for row in rows),
                "accepted_run_ids": ";".join(str(row["run_id"]) for row in accepted),
                "note": (
                    "best-effort requires same write cost, visible Host materialization, "
                    "and recovery p99 lower than lookup-disabled control"
                    if case_id == CASE_BEST_EFFORT
                    else "wait-complete requires same write cost and full Host materialization; "
                    "latency improvement is not required for this correctness case"
                ),
            }
        )
    return out


def overall_status(case_rows: list[dict[str, Any]]) -> str:
    statuses = {str(row["status"]) for row in case_rows}
    if statuses == {"pass"}:
        return "pass"
    if "partial" in statuses or "pass" in statuses:
        return "partial"
    return "missing"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: object, digits: int = 3) -> str:
    return f"{as_float(value):.{digits}f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strict-Final Recovery Matrix Audit",
        "",
        "This report audits real SGLang same-write-cost recovery pairs. It is an",
        "acceptance report for the SGLang recovery-path matrix, not a replacement",
        "for SGLSim P0 `physical_tensor_copy` rows.",
        "",
        "## Summary",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Matrix ready: `{str(payload['matrix_ready']).lower()}`",
        f"- Pair rows: {payload['pair_count']}",
        f"- Physical restore rows: {payload['physical_restore_rows']}",
        f"- Physical restore CSVs: {payload['physical_restore_csv_count']}",
        "",
        "## Cases",
        "",
        "| Case | Status | Required | Observed | Accepted | Improving | Mean visible/control p99 | Jobs |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["case_rows"]:
        lines.append(
            "| {case} | {status} | {required} | {observed} | {accepted} | {improving} | {norm}x | {inv} -> {vis} |".format(
                case=row["case_id"],
                status=row["status"],
                required=row["required_repeats"],
                observed=row["observed_pairs"],
                accepted=row["accepted_pairs"],
                improving=row["improving_pairs"],
                norm=fmt_float(row["visible_retry_latency_norm_mean"]),
                inv=row["invisible_job_ids"],
                vis=row["visible_job_ids"],
            )
        )
    lines.extend(
        [
            "",
            "## Accepted Pairs",
            "",
            "| Case | Run | Invisible job | Invisible state | Invisible data pages | Visible job | Norm | Visible state | Cached tokens |",
            "|---|---|---:|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in payload["pair_rows"]:
        if str(row["accepted"]) != "true":
            continue
        lines.append(
            "| {case} | {run} | {inv} | {inv_state} | {inv_pages} | {vis} | {norm}x | {state} | {cached} |".format(
                case=row["case_id"],
                run=row["run_id"],
                inv=row["invisible_job_id"],
                inv_state=row["invisible_physical_restore_state"],
                inv_pages=row["invisible_max_data_pages"],
                vis=row["visible_job_id"],
                norm=fmt_float(row["visible_retry_latency_norm_vs_invisible"]),
                state=row["visible_physical_restore_state"],
                cached=row["visible_retry_cached_tokens_after_accept_total"],
            )
        )
    lines.extend(
        [
            "",
            "## Missing Planned Runs",
            "",
        ]
    )
    if payload["missing_plan_pairs"]:
        lines.extend(
            [
                "The following commands are selected from the generated matrix plan",
                "based on the current accepted-pair counts.",
                "",
            ]
        )
        for row in payload["missing_plan_pairs"]:
            lines.extend(
                [
                    f"### {row['pair_id']}",
                    "",
                    f"- Reason: {row['reason']}",
                    "",
                    "```bash",
                    row["invisible_submit_command"],
                    row["visible_submit_command"],
                    "",
                    "# After both Slurm job ids are known, replace placeholders:",
                    row["sync_pair_command"],
                    row["physical_restore_command"],
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(["No missing planned runs for the configured matrix.", ""])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "A `pass` here means the real SGLang recovery-path matrix is complete for",
            "the two Host restore cases covered by this harness. It still does not",
            "prove peer GPUReplica P2P recovery or make the SGLSim metadata-only P0",
            "rows strict-final by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    *,
    results_dir: Path = RESULTS_DIR,
    plan_pairs: Path = DEFAULT_PLAN_PAIRS,
    plan_json: Path = DEFAULT_PLAN_JSON,
    physical_restore_csv: Path = DEFAULT_PHYSICAL_RESTORE_CSV,
) -> dict[str, Any]:
    pair_summaries = collect_pair_summaries(results_dir)
    physical_restore_csvs = collect_physical_restore_csvs(physical_restore_csv)
    physical_by_job = physical_rows_by_job(physical_restore_csvs)
    pair_rows = [
        evaluate_pair(pair, physical_by_job)
        for pair in pair_summaries
    ]
    required = planned_repeats(plan_pairs)
    case_rows = summarize_cases(pair_rows, required)
    plan = load_plan(plan_json)
    missing_plan_pairs = select_missing_plan_pairs(case_rows=case_rows, plan=plan)
    status = overall_status(case_rows)
    return {
        "overall_status": status,
        "matrix_ready": status == "pass",
        "required_repeats": required,
        "pair_count": len(pair_rows),
        "physical_restore_rows": len(physical_by_job),
        "physical_restore_csv_count": len(physical_restore_csvs),
        "physical_restore_csvs": [str(path) for path in physical_restore_csvs],
        "case_rows": case_rows,
        "pair_rows": pair_rows,
        "missing_plan_pairs": missing_plan_pairs,
        "inputs": {
            "results_dir": str(results_dir),
            "plan_pairs": str(plan_pairs),
            "plan_json": str(plan_json),
            "physical_restore_csv": str(physical_restore_csv),
            "physical_restore_csvs": [str(path) for path in physical_restore_csvs],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit completed SGLang strict-final recovery matrix pairs."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plan-pairs", type=Path, default=DEFAULT_PLAN_PAIRS)
    parser.add_argument("--plan-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument(
        "--physical-restore-csv", type=Path, default=DEFAULT_PHYSICAL_RESTORE_CSV
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    payload = run_audit(
        results_dir=args.results_dir,
        plan_pairs=args.plan_pairs,
        plan_json=args.plan_json,
        physical_restore_csv=args.physical_restore_csv,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "strict_final_recovery_matrix_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "strict_final_recovery_case_summary.csv", payload["case_rows"])
    write_csv(args.output_dir / "strict_final_recovery_pair_rows.csv", payload["pair_rows"])
    write_csv(
        args.output_dir / "strict_final_recovery_missing_jobs.csv",
        payload["missing_plan_pairs"],
    )
    (args.output_dir / "strict_final_recovery_matrix_audit.md").write_text(
        render_markdown(payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "overall_status": payload["overall_status"],
                "matrix_ready": payload["matrix_ready"],
                "pair_count": payload["pair_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
