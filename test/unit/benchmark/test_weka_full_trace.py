from __future__ import annotations

from collections import Counter

from benchmark.agentic_workflows.weka_full_trace import (
    WekaAgentBranch,
    WekaRequest,
    WekaStage,
    WekaWorkflow,
    build_causal_samples,
    crossfit_causal_slack,
    parse_weka_workflow,
    workflow_fold,
    zero_slack_tickets_for_unmodeled_requests,
)
from sglsim.srt.simulation.aegis_gbdt import AegisGBDTConfig


def _request(
    workflow_id: str,
    agent_id: str,
    rid: str,
    arrival_ms: float,
    duration_ms: float,
) -> WekaRequest:
    return WekaRequest(
        rid=rid,
        workflow_id=workflow_id,
        stage_id=1,
        agent_id=agent_id,
        role="subagent",
        model="test-model",
        prompt_tokens=128,
        output_tokens=16,
        arrival_ms=arrival_ms,
        api_time_ms=duration_ms,
        ttft_ms=None,
    )


def _parallel_workflow(workflow_id: str, offset: float = 0.0) -> WekaWorkflow:
    fast_id = f"{workflow_id}-fast"
    slow_id = f"{workflow_id}-slow"
    fast = WekaAgentBranch(
        fast_id,
        "subagent",
        (
            _request(workflow_id, fast_id, f"{fast_id}-0", 0.0, 10.0 + offset),
            _request(
                workflow_id,
                fast_id,
                f"{fast_id}-1",
                10.0 + offset,
                10.0,
            ),
        ),
    )
    slow = WekaAgentBranch(
        slow_id,
        "subagent",
        (
            _request(workflow_id, slow_id, f"{slow_id}-0", 5.0, 18.0 + offset),
            _request(
                workflow_id,
                slow_id,
                f"{slow_id}-1",
                23.0 + offset,
                8.0,
            ),
        ),
    )
    return WekaWorkflow(
        workflow_id=workflow_id,
        models=("test-model",),
        stages=(WekaStage(1, "subagent", (fast, slow)),),
    )


def test_parser_preserves_exact_tokens_and_real_overlap() -> None:
    workflow = parse_weka_workflow(
        {
            "id": "workflow-00000001",
            "models": ["m"],
            "block_size": 128,
            "requests": [
                {
                    "model": "m",
                    "in": 0,
                    "out": 0,
                    "t": 0,
                    "api_time": 1,
                    "hash_ids": [11, 12],
                },
                {
                    "type": "subagent",
                    "agent_id": "a",
                    "t": 1,
                    "duration_ms": 2000,
                    "requests": [
                        {
                            "model": "m",
                            "in": 17,
                            "out": 3,
                            "t": 1,
                            "api_time": 2,
                            "hash_ids": [21, 22, 23],
                        }
                    ],
                },
                {
                    "type": "subagent",
                    "agent_id": "b",
                    "t": 2,
                    "duration_ms": 1000,
                    "requests": [
                        {"model": "m", "in": 23, "out": 5, "t": 2, "api_time": 1}
                    ],
                },
            ],
        }
    )
    assert [(row.prompt_tokens, row.output_tokens) for row in workflow.requests] == [
        (0, 0),
        (17, 3),
        (23, 5),
    ]
    assert len(workflow.stages) == 2
    assert len(workflow.stages[1].branches) == 2
    assert workflow.block_size == 128
    assert workflow.requests[0].hash_ids == (11, 12)
    assert workflow.requests[1].hash_ids == (21, 22, 23)


def test_causal_frontier_does_not_reveal_future_branch() -> None:
    samples = build_causal_samples([_parallel_workflow("wf")])
    first_fast = next(
        row
        for row in samples
        if row.event_ms == 0.0 and row.agent_id.endswith("-fast")
    )
    # Feature 8 is the number of agents actually revealed and active now.  The
    # slow branch starts at t=5 and must not leak into the t=0 feature vector.
    assert first_fast.features[8] == 1.0
    assert first_fast.publish_for_request_id.endswith("-fast-0")


def test_ticket_is_not_backdated_into_already_arrived_request() -> None:
    workflow_id = "overlap"
    fast_id = "overlap-fast"
    slow_id = "overlap-slow"
    overlapping = WekaWorkflow(
        workflow_id=workflow_id,
        models=("test-model",),
        stages=(
            WekaStage(
                1,
                "subagent",
                (
                    WekaAgentBranch(
                        fast_id,
                        "subagent",
                        (
                            _request(workflow_id, fast_id, "fast-0", 0.0, 10.0),
                            _request(workflow_id, fast_id, "fast-1", 5.0, 10.0),
                        ),
                    ),
                    WekaAgentBranch(
                        slow_id,
                        "subagent",
                        (_request(workflow_id, slow_id, "slow-0", 0.0, 30.0),),
                    ),
                ),
            ),
        ),
    )
    samples = build_causal_samples([overlapping])
    assert not any(
        row.publish_for_request_id == "fast-1" and row.event_ms > 5.0
        for row in samples
    )


def test_crossfit_is_workflow_disjoint_and_fail_closes_missing_requests() -> None:
    workflows: list[WekaWorkflow] = []
    fold_counts: Counter[int] = Counter()
    candidate = 0
    while min((fold_counts.get(fold, 0) for fold in range(5)), default=0) < 2:
        workflow_id = f"crossfit-{candidate:04d}"
        fold = workflow_fold(workflow_id, 5, salt="unit-split")
        if fold_counts[fold] < 2:
            workflows.append(_parallel_workflow(workflow_id, float(candidate % 7)))
            fold_counts[fold] += 1
        candidate += 1

    samples = build_causal_samples(workflows)
    tickets, metrics = crossfit_causal_slack(
        samples,
        folds=5,
        split_salt="unit-split",
        gbdt_config=AegisGBDTConfig(
            estimators=3,
            learning_rate=0.2,
            max_depth=2,
            min_leaf=1,
            max_bins=4,
        ),
    )
    assert tickets
    assert metrics["forbidden_feature_audit_passed"] is True
    assert metrics["interval_calibration_unit"] == "workflow_max_absolute_residual"
    assert 0.0 <= metrics["workflow_simultaneous_interval_coverage"] <= 1.0
    for fold in metrics["predictors"].values():
        assert fold["fit_workflows"] > 0
        assert fold["calibration_workflows"] > 0
        assert fold["evaluation_workflows"] > 0
        assert fold["calibration_unit"] == "workflow_max_absolute_residual"
        assert fold["calibration_score_count"] == fold["calibration_workflows"]

    singleton_id = "singleton"
    singleton_agent = "singleton-main"
    singleton = WekaWorkflow(
        workflow_id=singleton_id,
        models=("test-model",),
        stages=(
            WekaStage(
                1,
                "main",
                (
                    WekaAgentBranch(
                        singleton_agent,
                        "main",
                        (
                            _request(
                                singleton_id,
                                singleton_agent,
                                "singleton-request",
                                0.0,
                                1.0,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    ticket_workflows = [*workflows, singleton]
    all_tickets = zero_slack_tickets_for_unmodeled_requests(
        ticket_workflows, tickets
    )
    assert len(all_tickets) == sum(
        len(workflow.requests) for workflow in ticket_workflows
    )
    missing = [ticket for ticket in all_tickets.values() if ticket.fold == -1]
    assert missing
    assert all(ticket.slack_lcb_ms == 0.0 for ticket in missing)
    assert all(ticket.predictor_sha256 == "fail-close-zero" for ticket in missing)
