from __future__ import annotations

import json
from dataclasses import replace

import pytest

from AegisKV.experiments.src.build_weka_observed_workload import (
    DEFAULT_OUTPUT_ROOT,
    SOURCE_SHA256,
    BuildSpec,
    WekaObservedBuildError,
    assign_seed_partitions,
    audit_artifact,
    stratify_workflows,
    visible_residual_ticket_fields,
)
from AegisKV.experiments.src.request.weka_causal import (
    CausalSlackTicket,
    WekaAgentBranch,
    WekaRequest,
    WekaStage,
    WekaWorkflow,
)
from sglsim.srt.simulation.aegis_residual_dag import (
    RESIDUAL_DAG_SLACK_SEMANTICS,
)


def _workflow(name: str, requests: int) -> WekaWorkflow:
    rows = tuple(
        WekaRequest(
            rid=f"{name}-r{index}",
            workflow_id=name,
            stage_id=1,
            agent_id=f"{name}-a{index % 2}",
            role="subagent",
            model="unit",
            prompt_tokens=64,
            output_tokens=1,
            arrival_ms=float(index),
            api_time_ms=10.0,
            ttft_ms=None,
            hash_ids=(1,),
        )
        for index in range(requests)
    )
    return WekaWorkflow(
        workflow_id=name,
        models=("unit",),
        stages=(
            WekaStage(
                stage_id=1,
                kind="subagent",
                branches=(
                    WekaAgentBranch(f"{name}-left", "subagent", rows[::2]),
                    WekaAgentBranch(f"{name}-right", "subagent", rows[1::2]),
                ),
            ),
        ),
    )


def _synthetic_spec() -> BuildSpec:
    return replace(
        BuildSpec.from_suite(),
        source_workflow_count=126,
        source_request_count=0,
    )


def test_frozen_boundaries_are_exhaustive_equal_cardinality_tertiles() -> None:
    spec = _synthetic_spec()
    counts = {
        "small": [30 + index for index in range(42)],
        "medium": [163 + index for index in range(42)],
        "long": [475 + index for index in range(42)],
    }
    workflows = tuple(
        _workflow(f"{stratum}-{index}", count)
        for stratum, values in counts.items()
        for index, count in enumerate(values)
    )

    strata = stratify_workflows(workflows, spec)

    assert {key: len(value) for key, value in strata.items()} == {
        "small": 42,
        "medium": 42,
        "long": 42,
    }
    assert max(len(row.requests) for row in strata["small"]) <= 162
    assert min(len(row.requests) for row in strata["medium"]) >= 163
    assert min(len(row.requests) for row in strata["long"]) >= 475


def test_each_seed_gets_one_workflow_per_stratum_without_replacement() -> None:
    spec = _synthetic_spec()
    strata = {
        rule.name: tuple(
            _workflow(
                f"{rule.name}-{index}",
                rule.minimum_requests_inclusive + index,
            )
            for index in range(42)
        )
        for rule in spec.strata
    }

    first = assign_seed_partitions(strata, spec)
    second = assign_seed_partitions(strata, spec)

    assert first == second
    assert len(first) == 30
    selected = [
        workflow.workflow_id
        for workflows in first.values()
        for workflow in workflows
    ]
    assert len(selected) == 30
    assert len(set(selected)) == 30
    assert all(len(workflows) == 1 for workflows in first.values())


def test_insufficient_stratum_fails_before_sampling_with_replacement() -> None:
    spec = _synthetic_spec()
    strata = {
        rule.name: tuple(
            _workflow(
                f"{rule.name}-{index}", rule.minimum_requests_inclusive
            )
            for index in range(9 if rule.name == "long" else 42)
        )
        for rule in spec.strata
    }

    with pytest.raises(WekaObservedBuildError, match="seed-disjoint"):
        assign_seed_partitions(strata, spec)


def test_modeled_ticket_converts_to_current_design_without_actual_label() -> None:
    ticket = CausalSlackTicket(
        request_id="request",
        workflow_id="workflow",
        stage_id=3,
        agent_id="agent",
        published_at_ms=10.0,
        slack_lcb_ms=7.5,
        slack_q50_ms=9.0,
        actual_slack_ms=999.0,
        remaining_q50_ms=20.0,
        remaining_interval_lower_ms=12.0,
        remaining_interval_upper_ms=30.0,
        fold=2,
        predictor_sha256="a" * 64,
    )

    fields = visible_residual_ticket_fields(
        ticket=ticket,
        request_arrival_ms=10.0,
        workflow_start_ms=2.0,
    )

    assert fields["workflow_slack_ms"] == 7.5
    assert fields["aegis_slack_semantics"] == RESIDUAL_DAG_SLACK_SEMANTICS
    assert fields["aegis_slack_uses_future_information"] is False
    assert fields["aegis_slack_usable"] is True
    assert fields["aegis_slack_reason"] == "visible_residual_dag_cpm"
    assert fields["aegis_slack_es_ms"] == 0.0
    assert fields["aegis_slack_ls_ms"] == 7.5
    assert fields["aegis_slack_critical_path_ms"] == 37.5
    assert fields["aegis_slack_observed_at_ms"] == 8.0
    assert "actual" not in json.dumps(fields, sort_keys=True).lower()


@pytest.mark.parametrize("fold,published", [(-1, 10.0), (2, 11.0)])
def test_unmodeled_or_noncausal_ticket_is_explicit_fail_close_zero(
    fold: int, published: float
) -> None:
    ticket = CausalSlackTicket(
        request_id="request",
        workflow_id="workflow",
        stage_id=3,
        agent_id="agent",
        published_at_ms=published,
        slack_lcb_ms=100.0,
        slack_q50_ms=100.0,
        actual_slack_ms=100.0,
        remaining_q50_ms=100.0,
        remaining_interval_lower_ms=90.0,
        remaining_interval_upper_ms=110.0,
        fold=fold,
        predictor_sha256="fail-close-zero" if fold < 0 else "b" * 64,
    )

    fields = visible_residual_ticket_fields(
        ticket=ticket,
        request_arrival_ms=10.0,
        workflow_start_ms=0.0,
    )

    assert fields["workflow_slack_ms"] == 0.0
    assert fields["aegis_slack_usable"] is False
    assert fields["aegis_slack_uses_future_information"] is False
    assert fields["aegis_slack_reason"] == "weka_agentx_fail_close_zero"
    assert fields["weka_ticket_modeled"] is False


def test_actual_compiled_artifact_is_pinned_and_auditable() -> None:
    audit = audit_artifact(DEFAULT_OUTPUT_ROOT)
    manifest = json.loads(
        (DEFAULT_OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["source"]["source_sha256"] == SOURCE_SHA256
    assert manifest["source"]["workflow_count"] == 393
    assert manifest["source"]["request_count"] == 98_827
    assert manifest["eligibility"]["eligible_workflow_count"] == 126
    assert manifest["dag_stratification"]["selected_workflow_count"] == 30
    assert manifest["dag_stratification"]["unselected_eligible_workflow_count"] == 96
    assert audit["partition_count"] == 30
    assert audit["workflow_count"] == 30
    assert audit["request_count"] > 0
    assert audit["modeled_requests"] + audit["unmodeled_fail_close_zero_requests"] == audit[
        "request_count"
    ]
    assert manifest["evidence"]["hardware_calibrated"] is False
    assert manifest["evidence"]["paper_ready"] is False
    first = json.loads(
        (DEFAULT_OUTPUT_ROOT / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert len(first["input_token_identity"]["block_namespace_sha256"]) == 64
