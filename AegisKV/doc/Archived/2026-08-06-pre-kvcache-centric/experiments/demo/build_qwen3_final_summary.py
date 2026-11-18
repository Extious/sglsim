"""Fail-closed final gate for the Qwen3-32B DP4/PP4 AegisKV study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from .validate_wct_suite import validate as validate_five_policy_suite


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "presentation"
    / "experiments"
    / "qwen3_32b_dp4_pp4_final_summary.json"
)
PRIMARY_MIN_P90_GAIN_PERCENT = 1.0
PRIMARY_MIN_RESTORED_FRACTION = 0.10
PRIMARY_MIN_WORKFLOW_WIN_FRACTION = 0.50


def _gain_percent(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return 100.0 * (baseline - candidate) / baseline


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_profile(results_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    aegis = _load(results_dir / "aegiskv.json")
    baseline = _load(results_dir / "no_backup.json")
    comparable = (
        "evidence_tier",
        "model",
        "dp_size",
        "pp_size",
        "workflow_filter",
        "workflow_concurrency",
        "failure_manifest",
        "execution_route_plan",
        "replay",
    )
    for field in comparable:
        if aegis.get(field) != baseline.get(field):
            errors.append(f"{results_dir}: pair differs in {field}")
    required_scope = {
        "evidence_tier": "sglsim-online-measured",
        "model": "Qwen/Qwen3-32B",
        "dp_size": 4,
        "pp_size": 4,
    }
    for field, expected in required_scope.items():
        if aegis.get(field) != expected:
            errors.append(
                f"{results_dir}: {field}={aegis.get(field)!r}, expected {expected!r}"
            )
    if int(dict(aegis.get("replay") or {}).get("context_length", 0)) != 32768:
        errors.append(f"{results_dir}: context length is not 32768")

    aegis_rows = {
        str(row["workflow_id"]): float(row["wct_s"])
        for row in aegis.get("workflows", [])
    }
    baseline_rows = {
        str(row["workflow_id"]): float(row["wct_s"])
        for row in baseline.get("workflows", [])
    }
    if not aegis_rows or set(aegis_rows) != set(baseline_rows):
        errors.append(f"{results_dir}: paired workflow IDs differ or are empty")
    deltas = [
        aegis_rows[key] - baseline_rows[key]
        for key in sorted(set(aegis_rows) & set(baseline_rows))
    ]
    wins = sum(delta < 0.0 for delta in deltas)
    win_fraction = wins / max(len(deltas), 1)
    mean_gain = _gain_percent(
        mean(aegis_rows.values()) if aegis_rows else 0.0,
        mean(baseline_rows.values()) if baseline_rows else 0.0,
    )
    aegis_p90 = float(dict(aegis.get("wct_summary_s") or {}).get("p90", 0.0))
    baseline_p90 = float(
        dict(baseline.get("wct_summary_s") or {}).get("p90", 0.0)
    )
    p90_gain = _gain_percent(aegis_p90, baseline_p90)

    aegis_retry = dict(aegis.get("retry_summary") or {})
    baseline_retry = dict(baseline.get("retry_summary") or {})
    restored_fraction = float(aegis_retry.get("restored_fraction", 0.0))
    if int(aegis_retry.get("count", 0)) <= 0:
        errors.append(f"{results_dir}: AegisKV has no retry evidence")
    if int(aegis_retry.get("count", 0)) != int(baseline_retry.get("count", 0)):
        errors.append(f"{results_dir}: retry counts are not paired")
    if int(aegis_retry.get("restored_tokens", 0)) <= 0:
        errors.append(f"{results_dir}: AegisKV restored no KV tokens")
    if int(baseline_retry.get("restored_tokens", 0)) != 0:
        errors.append(f"{results_dir}: no-backup restored KV tokens")
    for policy, payload in (("aegiskv", aegis), ("no_backup", baseline)):
        audit = dict(payload.get("matched_victim_audit") or {})
        if not bool(audit.get("enabled", False)):
            errors.append(f"{results_dir}: {policy} is not a matched-victim run")
        if not bool(audit.get("exact_match", False)):
            errors.append(f"{results_dir}: {policy} matched victims differ")
        if audit.get("route_violations"):
            errors.append(f"{results_dir}: {policy} has route violations")
        if audit.get("retry_route_violations"):
            errors.append(f"{results_dir}: {policy} has retry route violations")
        if audit.get("timing_violations"):
            errors.append(f"{results_dir}: {policy} has retry timing violations")
        route_audit = dict(payload.get("execution_route_plan_audit") or {})
        if not bool(route_audit.get("enabled", False)):
            errors.append(f"{results_dir}: {policy} has no execution route plan")
        if not bool(route_audit.get("exact_match", False)):
            errors.append(f"{results_dir}: {policy} execution routes differ from plan")

    scope = str(
        dict(aegis.get("failure_manifest") or {})
        .get("deployment", {})
        .get("failure_scope", "unknown")
    )
    return (
        {
            "run_id": results_dir.name,
            "results_dir": str(results_dir.resolve()),
            "failure_scope": scope,
            "workflow_concurrency": int(aegis.get("workflow_concurrency", 0)),
            "workflow_count": len(aegis_rows),
            "failure_count": int(aegis.get("failure_count_injected", 0)),
            "retry_count": int(aegis_retry.get("count", 0)),
            "aegiskv_p90_wct_s": aegis_p90,
            "no_backup_p90_wct_s": baseline_p90,
            "p90_wct_gain_percent": round(p90_gain, 6),
            "mean_wct_gain_percent": round(mean_gain, 6),
            "workflow_wins": wins,
            "workflow_win_fraction": round(win_fraction, 6),
            "restored_fraction": restored_fraction,
            "restored_tokens": int(aegis_retry.get("restored_tokens", 0)),
            "recompute_tokens": int(aegis_retry.get("recompute_tokens", 0)),
        },
        errors,
    )


def build_summary(
    *, primary_results_dir: Path, additional_profile_dirs: list[Path]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        validate_five_policy_suite(primary_results_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"primary five-policy validation failed: {exc}")
    primary, pair_errors = paired_profile(primary_results_dir)
    errors.extend(pair_errors)
    if primary["p90_wct_gain_percent"] < PRIMARY_MIN_P90_GAIN_PERCENT:
        errors.append("primary p90 WCT gain is below the fixed 1% gate")
    if primary["mean_wct_gain_percent"] <= 0.0:
        errors.append("primary mean WCT did not improve")
    if primary["workflow_win_fraction"] < PRIMARY_MIN_WORKFLOW_WIN_FRACTION:
        errors.append("primary workflow win fraction is below 0.5")
    if primary["restored_fraction"] < PRIMARY_MIN_RESTORED_FRACTION:
        errors.append("primary restored fraction is below the fixed 10% gate")

    additional: list[dict[str, Any]] = []
    if not additional_profile_dirs:
        errors.append("at least one additional workload/failure profile is required")
    for directory in additional_profile_dirs:
        profile, profile_errors = paired_profile(directory)
        additional.append(profile)
        errors.extend(profile_errors)
        if profile["p90_wct_gain_percent"] <= 0.0:
            errors.append(f"{directory}: p90 WCT did not improve")
        if profile["mean_wct_gain_percent"] <= 0.0:
            errors.append(f"{directory}: mean WCT did not improve")
        if profile["restored_fraction"] <= 0.0:
            errors.append(f"{directory}: retry profile has no KV reuse")

    return (
        {
            "schema_version": 1,
            "gate_pass": not errors,
            "run_id": primary["run_id"],
            "model": "Qwen/Qwen3-32B",
            "dp_size": 4,
            "pp_size": 4,
            "context_length": 32768,
            "workflow_count": primary["workflow_count"],
            "failure_count": primary["failure_count"],
            "wct_p90_gain_percent": primary["p90_wct_gain_percent"],
            "restored_fraction": primary["restored_fraction"],
            "fixed_gates": {
                "primary_min_p90_gain_percent": PRIMARY_MIN_P90_GAIN_PERCENT,
                "primary_min_restored_fraction": PRIMARY_MIN_RESTORED_FRACTION,
                "primary_min_workflow_win_fraction": (
                    PRIMARY_MIN_WORKFLOW_WIN_FRACTION
                ),
                "minimum_additional_profiles": 1,
            },
            "primary_five_policy_profile": primary,
            "additional_profiles": additional,
            "errors": errors,
        },
        errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-results-dir", type=Path, required=True)
    parser.add_argument(
        "--additional-profile-dir", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary, errors = build_summary(
        primary_results_dir=args.primary_results_dir.resolve(),
        additional_profile_dirs=[path.resolve() for path in args.additional_profile_dir],
    )
    if errors:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
