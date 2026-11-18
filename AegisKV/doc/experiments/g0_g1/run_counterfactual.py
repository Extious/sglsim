#!/usr/bin/env python3
"""Run the pre-registered tiny G0/G1 semantic checks.

Outputs from this script are synthetic semantic checks and schema audits.  They
must not be relabelled as calibrated or real-system performance evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sglsim.srt.simulation.aegis_workflow_harm import (
    PolicyName,
    ProtectionAction,
    RecoveryEffect,
    ShadowJob,
    ShadowScenario,
    TargetOption,
    TargetState,
    WeightedScenario,
    WorkflowHarmPortfolio,
    WorkflowState,
)


POLICIES: tuple[PolicyName, ...] = (
    "load_size",
    "criticality",
    "request_slack",
    "aegis_joint",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
TRACE_FILES = {
    "failure_position": REPO_ROOT
    / "AegisKV/figures/data/real-cc-failure-position-workflow.csv",
    "parallel_gap": REPO_ROOT
    / "AegisKV/figures/data/real-cc-workflow-parallel-gap.csv",
    "causal_slack": REPO_ROOT
    / "AegisKV/figures/data/alg1-dynamic-slack-snapshots.csv",
}


def _action(
    action_id: str,
    *effects: tuple[str, str, float],
    demand: float = 1.0,
) -> ProtectionAction:
    return ProtectionAction(
        action_id=action_id,
        demand=demand,
        effects=tuple(
            RecoveryEffect(
                failure_domain="node0",
                job_id=job_id,
                target_id=target_id,
                protected_recovery_s=protected_s,
            )
            for job_id, target_id, protected_s in effects
        ),
    )


def _affected(
    job_id: str,
    workflow_id: str,
    target_id: str,
    recovery_s: float,
    *,
    slack_s: float = 0.0,
    criticality: float = 1.0,
) -> ShadowJob:
    return ShadowJob(
        job_id=job_id,
        workflow_id=workflow_id,
        source_domain="node0",
        target_options=(
            TargetOption(
                target_id=target_id,
                resume_service_s=1.0,
                unprotected_recovery_s=recovery_s,
            ),
        ),
        slack_lcb_s=slack_s,
        criticality=criticality,
    )


def _portfolio(
    *,
    scenario_id: str,
    workflows: tuple[WorkflowState, ...],
    targets: tuple[TargetState, ...],
    jobs: tuple[ShadowJob, ...],
    actions: tuple[ProtectionAction, ...],
) -> WorkflowHarmPortfolio:
    return WorkflowHarmPortfolio(
        scenarios=(
            WeightedScenario(
                probability=1.0,
                scenario=ShadowScenario(
                    scenario_id=scenario_id,
                    failure_domain="node0",
                    workflows=workflows,
                    targets=targets,
                    jobs=jobs,
                ),
            ),
        ),
        actions=actions,
    )


def correlated_domain() -> WorkflowHarmPortfolio:
    return _portfolio(
        scenario_id="correlated_domain",
        workflows=(
            WorkflowState("workflow-a", healthy_finish_s=1.0),
            WorkflowState("workflow-b", healthy_finish_s=1.0),
        ),
        targets=(TargetState("t0"), TargetState("t1"), TargetState("t2")),
        jobs=(
            _affected(
                "a-fast",
                "workflow-a",
                "t0",
                10.0,
                slack_s=2.0,
                criticality=1.0,
            ),
            _affected("a-slow", "workflow-a", "t1", 12.0),
            _affected(
                "b-only",
                "workflow-b",
                "t2",
                6.0,
                criticality=0.5,
            ),
        ),
        actions=(
            _action("protect-a-fast", ("a-fast", "t0", 0.0)),
            _action("protect-b", ("b-only", "t2", 0.0)),
        ),
    )


def bystander_queue() -> WorkflowHarmPortfolio:
    bystanders = tuple(
        ShadowJob(
            job_id=f"bystander-{index}",
            workflow_id=f"bystander-workflow-{index}",
            source_domain=None,
            target_options=(TargetOption("busy", 100.0),),
            queue_rank=index,
        )
        for index in range(3)
    )
    return _portfolio(
        scenario_id="bystander_queue",
        workflows=(
            WorkflowState("affected-busy", healthy_finish_s=1.0),
            WorkflowState("affected-isolated", healthy_finish_s=1.0),
            WorkflowState("bystander-workflow-0", healthy_finish_s=100.0),
            WorkflowState("bystander-workflow-1", healthy_finish_s=200.0),
            WorkflowState("bystander-workflow-2", healthy_finish_s=300.0),
        ),
        targets=(TargetState("busy"), TargetState("isolated")),
        jobs=(
            _affected(
                "affected-on-busy",
                "affected-busy",
                "busy",
                4.0,
                criticality=0.5,
            ),
            _affected(
                "affected-on-isolated",
                "affected-isolated",
                "isolated",
                5.0,
            ),
            *bystanders,
        ),
        actions=(
            _action("protect-busy", ("affected-on-busy", "busy", 0.0)),
            _action(
                "protect-isolated",
                ("affected-on-isolated", "isolated", 0.0),
            ),
        ),
    )


def shared_copy() -> WorkflowHarmPortfolio:
    return _portfolio(
        scenario_id="shared_copy",
        workflows=tuple(
            WorkflowState(f"workflow-{index}", healthy_finish_s=1.0)
            for index in range(3)
        ),
        targets=tuple(TargetState(f"t{index}") for index in range(3)),
        jobs=(
            _affected("shared-0", "workflow-0", "t0", 4.0, criticality=0.25),
            _affected("shared-1", "workflow-1", "t1", 4.0, criticality=0.25),
            _affected("single", "workflow-2", "t2", 6.0),
        ),
        actions=(
            _action(
                "one-shared-copy",
                ("shared-0", "t0", 0.0),
                ("shared-1", "t1", 0.0),
            ),
            _action("one-critical-copy", ("single", "t2", 0.0)),
        ),
    )


def negative_control() -> WorkflowHarmPortfolio:
    return _portfolio(
        scenario_id="negative_control",
        workflows=(
            WorkflowState("workflow-a", healthy_finish_s=1.0),
            WorkflowState("workflow-b", healthy_finish_s=1.0),
        ),
        targets=(TargetState("t0"), TargetState("t1")),
        jobs=(
            _affected("a", "workflow-a", "t0", 10.0),
            _affected("b", "workflow-b", "t1", 5.0),
        ),
        actions=(
            _action("protect-a", ("a", "t0", 0.0)),
            _action("protect-b", ("b", "t1", 0.0)),
        ),
    )


def _run_case(
    case_name: str, portfolio: WorkflowHarmPortfolio, *, budget: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    no_protection = portfolio.evaluate(frozenset())
    exact = portfolio.solve_exact(budget=budget)
    policy_results = {
        policy: portfolio.greedy(policy, budget=budget) for policy in POLICIES
    }
    rows: list[dict[str, Any]] = []
    for policy, result in policy_results.items():
        harm = result.true_outcome.expected_harm
        rows.append(
            {
                "case": case_name,
                "evidence_class": "synthetic_semantic_check",
                "policy": policy,
                "selected_actions": ";".join(result.selected_actions),
                "used_budget": result.used_budget,
                "true_tardiness_harm_s": harm.tardiness_s,
                "true_wct_harm_s": harm.wct_s,
                "wct_harm_reduction_s": (
                    no_protection.expected_harm.wct_s - harm.wct_s
                ),
                "matches_exact": result.selected_actions == exact.selected_actions,
            }
        )
    summary = {
        "case": case_name,
        "evidence_class": "synthetic_semantic_check",
        "budget": budget,
        "instance_sha256": portfolio.sha256,
        "no_protection_harm": asdict(no_protection.expected_harm),
        "policies": {
            policy: {
                "selected_actions": list(result.selected_actions),
                "used_budget": result.used_budget,
                "true_harm": asdict(result.true_outcome.expected_harm),
                "steps": [asdict(step) for step in result.steps],
                "scenario_schedules": [
                    asdict(schedule) for schedule in result.true_outcome.scenarios
                ],
            }
            for policy, result in policy_results.items()
        },
        "exact": {
            "selected_actions": list(exact.selected_actions),
            "used_budget": exact.used_budget,
            "true_harm": asdict(exact.outcome.expected_harm),
            "certificate": asdict(exact.certificate),
        },
    }
    return summary, rows


def _csv_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": 0, "columns": []}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = sum(1 for _ in reader)
        columns = list(reader.fieldnames or [])
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": True,
        "rows": rows,
        "columns": columns,
    }


def _schema_audit() -> dict[str, Any]:
    inventory = {name: _csv_inventory(path) for name, path in TRACE_FILES.items()}
    all_columns = {
        column
        for item in inventory.values()
        for column in item.get("columns", [])
    }
    capabilities = {
        "workflow_identity": {
            "required_any": ["workflow_id"],
            "status": "SUPPORTED"
            if "workflow_id" in all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
        "fanout_structure_and_branch_gap": {
            "required_all": ["fanout_width", "max_duration_gap_s"],
            "status": "SUPPORTED"
            if {"fanout_width", "max_duration_gap_s"} <= all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
        "causal_slack_snapshot": {
            "required_all": [
                "predicted_slack_lcb_ms",
                "information_scope",
            ],
            "status": "SUPPORTED"
            if {"predicted_slack_lcb_ms", "information_scope"} <= all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
        "failure_position_workflow_delta": {
            "required_all": ["failure_position_fraction", "max_delta_wct_s"],
            "status": "SUPPORTED"
            if {"failure_position_fraction", "max_delta_wct_s"} <= all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
        "source_failure_domain": {
            "required_all": ["source_failure_domain"],
            "status": "SUPPORTED"
            if "source_failure_domain" in all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
        "target_queue_and_batch_state": {
            "required_all": ["target_id", "target_queue_work_s", "batch_state"],
            "status": "SUPPORTED"
            if {"target_id", "target_queue_work_s", "batch_state"} <= all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
        "lost_capacity_and_reload": {
            "required_all": ["lost_capacity", "model_reload_horizon_s"],
            "status": "SUPPORTED"
            if {"lost_capacity", "model_reload_horizon_s"} <= all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
        "kv_object_sharing_and_target_prefix": {
            "required_all": [
                "kv_object_id",
                "target_id",
                "reusable_prefix_tokens",
            ],
            "status": "SUPPORTED"
            if {
                "kv_object_id",
                "target_id",
                "reusable_prefix_tokens",
            }
            <= all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
        "bystander_counterfactual": {
            "required_all": [
                "bystander_workflow_id",
                "bystander_finish_without_redirect_s",
                "bystander_finish_with_redirect_s",
            ],
            "status": "SUPPORTED"
            if {
                "bystander_workflow_id",
                "bystander_finish_without_redirect_s",
                "bystander_finish_with_redirect_s",
            }
            <= all_columns
            else "UNSUPPORTED_BY_SCHEMA",
        },
    }
    g1_required = (
        "source_failure_domain",
        "target_queue_and_batch_state",
        "lost_capacity_and_reload",
        "bystander_counterfactual",
    )
    g1_status = (
        "SCHEMA_CAPABLE_NOT_PROVEN"
        if all(capabilities[name]["status"] == "SUPPORTED" for name in g1_required)
        else "NOT_PROVEN"
    )
    return {
        "evidence_class": "schema_capability_audit",
        "inventory": inventory,
        "capabilities": capabilities,
        "g1_status": g1_status,
        "interpretation": (
            "The checked-in artifacts support workflow structure, failure-position "
            "counterfactuals, and causal Slack snapshots, but not joint domain, "
            "lost-capacity, target-queue, or bystander claims."
        ),
    }


def _semantic_verdict(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {case["case"]: case for case in cases}
    positive_checks: dict[str, bool] = {}
    for name in ("correlated_domain", "bystander_queue"):
        case = by_name[name]
        exact_actions = case["exact"]["selected_actions"]
        aegis = case["policies"]["aegis_joint"]
        baseline_wct = max(
            case["policies"][policy]["true_harm"]["wct_s"]
            for policy in ("load_size", "criticality", "request_slack")
        )
        best_baseline_wct = min(
            case["policies"][policy]["true_harm"]["wct_s"]
            for policy in ("load_size", "criticality", "request_slack")
        )
        positive_checks[name] = (
            aegis["selected_actions"] == exact_actions
            and aegis["true_harm"]["wct_s"] < best_baseline_wct
            and baseline_wct >= best_baseline_wct
        )
    negative = by_name["negative_control"]
    negative_choices = {
        tuple(negative["policies"][policy]["selected_actions"])
        for policy in POLICIES
    }
    exact_all = all(
        case["policies"]["aegis_joint"]["selected_actions"]
        == case["exact"]["selected_actions"]
        for case in cases
    )
    semantic_pass = all(positive_checks.values()) and len(negative_choices) == 1 and exact_all
    return {
        "evidence_class": "synthetic_semantic_check",
        "semantic_nonvacuity": "PASS" if semantic_pass else "FAIL",
        "positive_case_checks": positive_checks,
        "negative_control_all_policies_agree": len(negative_choices) == 1,
        "aegis_matches_exact_all_cases": exact_all,
        "g0_status": "NOT_PROVEN",
        "g1_status": "NOT_PROVEN",
        "reason": (
            "Synthetic cases can reject broken semantics but cannot establish "
            "stable disagreement or prevalence on held-out real workloads."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=float, default=1.0)
    args = parser.parse_args()

    cases = {
        "correlated_domain": correlated_domain(),
        "bystander_queue": bystander_queue(),
        "shared_copy": shared_copy(),
        "negative_control": negative_control(),
    }
    case_summaries: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for case_name, portfolio in cases.items():
        summary, rows = _run_case(case_name, portfolio, budget=args.budget)
        case_summaries.append(summary)
        decision_rows.extend(rows)

    schema = _schema_audit()
    verdict = _semantic_verdict(case_summaries)
    payload = {
        "schema_version": "aegiskv-g0-g1-v1",
        "evidence_class": "synthetic_semantic_check",
        "comparison_contract": "same_actions_budget_and_true_joint_evaluator",
        "cases": case_summaries,
        "semantic_verdict": verdict,
        "trace_schema_audit": schema,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "counterfactual_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = args.output_dir / "policy_decisions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)
    schema_path = args.output_dir / "trace_schema_audit.json"
    schema_path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"semantic_nonvacuity={verdict['semantic_nonvacuity']}")
    print(f"G0={verdict['g0_status']} G1={schema['g1_status']}")
    print(f"summary={summary_path}")
    print(f"decisions={csv_path}")
    print(f"schema_audit={schema_path}")


if __name__ == "__main__":
    main()

