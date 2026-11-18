from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .plot_wct_cdf import POLICY_STYLE, load_policy_results


def validate(results_dir: Path) -> dict[str, Any]:
    results = load_policy_results(results_dir)
    reference = results["aegiskv"]
    workflow_ids = {
        str(row["workflow_id"]) for row in reference.get("workflows", [])
    }
    errors: list[str] = []
    checks: dict[str, Any] = {}

    comparable = (
        "model",
        "dp_size",
        "trace",
        "workflow_filter",
        "workflow_concurrency",
        "failure_manifest",
        "execution_route_plan",
        "replay",
    )
    for policy, payload in results.items():
        if {
            str(row["workflow_id"]) for row in payload.get("workflows", [])
        } != workflow_ids:
            errors.append(f"{policy}: workflow IDs differ")
        for field in comparable:
            if payload.get(field) != reference.get(field):
                errors.append(f"{policy}: field {field} differs")
        if payload.get("failure_count_injected") != payload.get(
            "failure_count_requested"
        ):
            errors.append(f"{policy}: not all failures were injected")
        for event in payload.get("failure_events", []):
            if float(event.get("actual_outage_s", 0.0)) < float(
                event.get("duration_s", 0.0)
            ):
                errors.append(f"{policy}: outage shorter than manifest duration")
            if not isinstance(event.get("l1_clear"), dict):
                errors.append(f"{policy}: missing L1-clear result")
        retry = payload.get("retry_summary", {})
        matched_audit = payload.get("matched_victim_audit", {})
        if bool(matched_audit.get("enabled", False)):
            if not bool(matched_audit.get("exact_match", False)):
                errors.append(f"{policy}: matched failure victims differ")
            if matched_audit.get("route_violations"):
                errors.append(f"{policy}: matched victim route was not frozen")
            if matched_audit.get("retry_route_violations"):
                errors.append(f"{policy}: matched retry route was not frozen")
            if matched_audit.get("timing_violations"):
                errors.append(
                    f"{policy}: matched retry started before its failure event"
                )
            if int(retry.get("count", 0)) != len(
                matched_audit.get("expected_backup_keys", [])
            ):
                errors.append(
                    f"{policy}: retry count differs from matched victim count"
                )
        if int(retry.get("capacity_limited_tokens", 0)) > int(
            retry.get("recompute_tokens", 0)
        ):
            errors.append(f"{policy}: capacity-limited tokens exceed recompute")
        route_audit = dict(payload.get("execution_route_plan_audit") or {})
        if not bool(route_audit.get("enabled", False)):
            errors.append(f"{policy}: execution route plan is not enabled")
        if not bool(route_audit.get("exact_match", False)):
            errors.append(f"{policy}: execution routes differ from the plan")
        if policy == "no_backup":
            if int(retry.get("restored_tokens", 0)) != 0:
                errors.append("no_backup: restored tokens must be zero")
        else:
            backup = payload.get("backup_decision_summary", {})
            if not bool(backup.get("ack_complete", False)):
                errors.append(f"{policy}: asynchronous backup ACKs incomplete")
            replayed = sum(
                int(row.get("replayed_prompt_tokens", 0))
                + int(row.get("replayed_output_tokens", 0))
                for row in payload.get("workflows", [])
            )
            decided = (
                int(backup.get("l1_tokens", 0))
                + int(backup.get("l2_tokens", 0))
                + int(backup.get("l3_tokens", 0))
                + int(backup.get("recompute_tokens", 0))
            )
            if decided != replayed:
                errors.append(
                    f"{policy}: backup token conservation failed "
                    f"({decided} != {replayed})"
                )
        for rank, state in payload.get("final_rank_states", {}).items():
            alpha = float(state.get("alpha", 0.0))
            beta = float(state.get("beta", 0.0))
            if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
                errors.append(f"{policy}: invalid alpha/beta on rank {rank}")

    checks.update(
        {
            "valid": not errors,
            "workflow_count": len(workflow_ids),
            "policies": list(POLICY_STYLE),
            "errors": errors,
        }
    )
    (results_dir / "validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise ValueError("; ".join(errors))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a five-policy WCT suite.")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.results_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
