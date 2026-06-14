from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sglsim.fault_tolerance.aegiskv import (
    AegisKVPolicyConfig,
    KVSegment,
    LifecycleState,
    SegmentType,
    apply_backup_policy,
    backed_up_token_count,
    is_critical_agent_id,
    score_segment,
)

FAILURE_SITES = (
    "S1_RepoMapper",
    "S2_ToolBlocked",
    "O0_Diagnosis",
    "S4_Patch",
    "S5_ToolBlocked",
    "O0_Final",
)

FAILURE_SITE_TO_AGENT: dict[str, str] = {
    "S1_RepoMapper": "S1",
    "S2_ToolBlocked": "S2",
    "O0_Diagnosis": "O0",
    "S4_Patch": "S4",
    "S5_ToolBlocked": "S5",
    "O0_Final": "O0",
}

ROUND1_SUBAGENTS = ("S1", "S2", "S3")
ROUND2_SUBAGENTS = ("S4", "S5", "S6")

STRATEGY_FAILOVER_OVERHEAD_S: dict[str, float] = {
    "baseline": 1.2,
    "naive_backup": 0.20,
    "aegiskv": 0.14,
}

STRATEGY_BACKUP_OVERHEAD_PER_TOKEN_S: dict[str, float] = {
    "baseline": 0.0,
    "naive_backup": 0.0000045,
    "aegiskv": 0.0000032,
}

PREFILL_S_PER_TOKEN = 0.00011
DECODE_S_PER_TOKEN = 0.002
RECOMPUTE_S_PER_TOKEN = 0.0012
SUFFIX_REPLAY_S_PER_TOKEN = 0.000055
BASE_OVERHEAD_S = 8.0


@dataclass(frozen=True)
class MultiAgentWorkflowConfig:
    num_workflows: int = 8
    subagents_per_round: int = 3
    workflow_start_stagger_s: float = 0.25
    tool_blocked_duration_ms: int = 10000
    system_prefix_tokens: int = 4096
    tool_schema_tokens: int = 1024
    task_plan_tokens: int = 2048
    selected_context_tokens: int = 4096
    branch_tail_tokens: int = 3072
    tool_blocked_prefix_tokens: int = 6144
    diagnosis_tokens: int = 4096
    verifier_context_tokens: int = 5120
    final_synthesis_tokens: int = 4096
    output_tokens_default: int = 64
    output_tokens_critical: int = 512
    output_tokens_main: int = 256
    output_tokens_per_invocation: int = 512  # legacy alias for output_tokens_critical


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    strategy: str
    failure_site: str
    repeat_idx: int
    seed: int
    backup_budget_ratio: float
    peer_backup_lag_tokens: int
    num_workflows: int


@dataclass(frozen=True)
class RunResult:
    run: dict[str, Any]
    segments: list[dict[str, Any]]
    recovery_paths: list[dict[str, Any]]
    workflows: list[dict[str, Any]]
    agent_timelines: list[dict[str, Any]]


def _output_tokens_for_agent(agent_id: str, cfg: MultiAgentWorkflowConfig) -> int:
    if agent_id == "O0":
        return cfg.output_tokens_main
    if agent_id in {"S2", "S5"}:
        return cfg.output_tokens_critical
    return cfg.output_tokens_default


def _failure_phase_for_site(failure_site: str) -> str | None:
    if failure_site == "O0_Diagnosis":
        return "round1_fanin"
    if failure_site == "O0_Final":
        return "round2_fanin"
    return None


def _is_failure_target(
    *,
    agent_id: str,
    phase_id: str,
    failure_site: str,
    impacted: bool,
) -> bool:
    if not impacted:
        return False
    failure_agent = FAILURE_SITE_TO_AGENT[failure_site]
    if agent_id != failure_agent:
        return False
    expected_phase = _failure_phase_for_site(failure_site)
    if expected_phase is not None:
        return phase_id == expected_phase
    return True


def _seg(
    workflow_id: str,
    local_id: str,
    cfg: MultiAgentWorkflowConfig,
    *,
    phase_id: str,
    agent_id: str,
    parent_agent_id: str,
    role: str,
    segment_type: SegmentType,
    lifecycle_state: LifecycleState,
    tokens: int,
    sink_distance: int,
    fanout_degree: int = 1,
    fanin_degree: int = 1,
    blocked_duration_ms: int = 0,
    summarized: bool = False,
) -> KVSegment:
    return KVSegment(
        segment_id=f"{workflow_id}:{local_id}",
        workflow_id=workflow_id,
        phase_id=phase_id,
        agent_id=agent_id,
        parent_agent_id=parent_agent_id,
        role=role,
        segment_type=segment_type,
        lifecycle_state=lifecycle_state,
        tokens=int(tokens),
        sink_distance=int(sink_distance),
        output_tokens=_output_tokens_for_agent(agent_id, cfg),
        fanout_degree=int(fanout_degree),
        fanin_degree=int(fanin_degree),
        blocked_duration_ms=int(blocked_duration_ms),
        summarized=bool(summarized),
    )


def build_two_round_coding_workflow(
    workflow_id: str,
    cfg: MultiAgentWorkflowConfig,
) -> list[KVSegment]:
    b = cfg.tool_blocked_duration_ms
    return [
        _seg(
            workflow_id,
            "P1",
            cfg,
            phase_id="workflow_creation",
            agent_id="O0",
            parent_agent_id="",
            role="common_prefix",
            segment_type=SegmentType.SYSTEM_TOOL_PREFIX,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.system_prefix_tokens,
            sink_distance=5,
            fanout_degree=7,
        ),
        _seg(
            workflow_id,
            "P2",
            cfg,
            phase_id="workflow_creation",
            agent_id="O0",
            parent_agent_id="",
            role="main_orchestrator",
            segment_type=SegmentType.TOOL_SCHEMA,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.tool_schema_tokens,
            sink_distance=4,
        ),
        _seg(
            workflow_id,
            "P3",
            cfg,
            phase_id="workflow_creation",
            agent_id="O0",
            parent_agent_id="",
            role="main_orchestrator",
            segment_type=SegmentType.SHARED_HISTORY,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.task_plan_tokens,
            sink_distance=4,
            fanout_degree=cfg.subagents_per_round,
        ),
        _seg(
            workflow_id,
            "S10",
            cfg,
            phase_id="round1_fanout",
            agent_id="S1",
            parent_agent_id="O0",
            role="repo_mapper",
            segment_type=SegmentType.SHARED_HISTORY,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.selected_context_tokens,
            sink_distance=3,
            fanout_degree=2,
        ),
        _seg(
            workflow_id,
            "S11",
            cfg,
            phase_id="round1_fanout",
            agent_id="S1",
            parent_agent_id="O0",
            role="repo_mapper",
            segment_type=SegmentType.BRANCH_TAIL,
            lifecycle_state=LifecycleState.SUMMARIZED,
            tokens=cfg.branch_tail_tokens,
            sink_distance=3,
            summarized=True,
        ),
        _seg(
            workflow_id,
            "S12",
            cfg,
            phase_id="round1_fanout",
            agent_id="S3",
            parent_agent_id="O0",
            role="api_history_checker",
            segment_type=SegmentType.BRANCH_TAIL,
            lifecycle_state=LifecycleState.SUMMARIZED,
            tokens=cfg.branch_tail_tokens,
            sink_distance=3,
            summarized=True,
        ),
        _seg(
            workflow_id,
            "S14",
            cfg,
            phase_id="round1_fanout",
            agent_id="S2",
            parent_agent_id="O0",
            role="test_reproducer",
            segment_type=SegmentType.VERIFIER_CONTEXT,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.selected_context_tokens,
            sink_distance=3,
        ),
        _seg(
            workflow_id,
            "S15",
            cfg,
            phase_id="round1_fanout",
            agent_id="S2",
            parent_agent_id="O0",
            role="test_reproducer",
            segment_type=SegmentType.TOOL_BLOCKED_PREFIX,
            lifecycle_state=LifecycleState.TOOL_BLOCKED,
            tokens=cfg.tool_blocked_prefix_tokens,
            sink_distance=3,
            blocked_duration_ms=b,
        ),
        _seg(
            workflow_id,
            "P5",
            cfg,
            phase_id="round1_fanin",
            agent_id="O0",
            parent_agent_id="O0",
            role="main_diagnosis",
            segment_type=SegmentType.SHARED_HISTORY,
            lifecycle_state=LifecycleState.FANIN_READY,
            tokens=cfg.selected_context_tokens,
            sink_distance=2,
            fanin_degree=cfg.subagents_per_round,
        ),
        _seg(
            workflow_id,
            "P6",
            cfg,
            phase_id="round1_fanin",
            agent_id="O0",
            parent_agent_id="O0",
            role="main_diagnosis",
            segment_type=SegmentType.DIAGNOSIS_SUMMARY,
            lifecycle_state=LifecycleState.FANIN_READY,
            tokens=cfg.diagnosis_tokens,
            sink_distance=2,
            fanout_degree=cfg.subagents_per_round,
            fanin_degree=cfg.subagents_per_round,
        ),
        _seg(
            workflow_id,
            "S18",
            cfg,
            phase_id="round2_fanout",
            agent_id="S4",
            parent_agent_id="O0",
            role="patch_implementer",
            segment_type=SegmentType.HOT_DECODE_TAIL,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.branch_tail_tokens,
            sink_distance=1,
        ),
        _seg(
            workflow_id,
            "S20",
            cfg,
            phase_id="round2_fanout",
            agent_id="S5",
            parent_agent_id="O0",
            role="regression_verifier",
            segment_type=SegmentType.VERIFIER_CONTEXT,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.verifier_context_tokens,
            sink_distance=1,
        ),
        _seg(
            workflow_id,
            "S21",
            cfg,
            phase_id="round2_fanout",
            agent_id="S5",
            parent_agent_id="O0",
            role="regression_verifier",
            segment_type=SegmentType.TOOL_BLOCKED_PREFIX,
            lifecycle_state=LifecycleState.TOOL_BLOCKED,
            tokens=cfg.tool_blocked_prefix_tokens,
            sink_distance=1,
            blocked_duration_ms=b,
        ),
        _seg(
            workflow_id,
            "S24",
            cfg,
            phase_id="round2_fanout",
            agent_id="S6",
            parent_agent_id="O0",
            role="review_checker",
            segment_type=SegmentType.BRANCH_TAIL,
            lifecycle_state=LifecycleState.SUMMARIZED,
            tokens=cfg.branch_tail_tokens,
            sink_distance=1,
            summarized=True,
        ),
        _seg(
            workflow_id,
            "P7",
            cfg,
            phase_id="round2_fanin",
            agent_id="O0",
            parent_agent_id="O0",
            role="main_final_resume",
            segment_type=SegmentType.SHARED_HISTORY,
            lifecycle_state=LifecycleState.FANIN_READY,
            tokens=cfg.selected_context_tokens,
            sink_distance=0,
            fanin_degree=cfg.subagents_per_round,
        ),
        _seg(
            workflow_id,
            "P8",
            cfg,
            phase_id="round2_fanin",
            agent_id="O0",
            parent_agent_id="O0",
            role="main_final_resume",
            segment_type=SegmentType.FINAL_SYNTHESIS_PREFIX,
            lifecycle_state=LifecycleState.FINAL_RESUME,
            tokens=cfg.final_synthesis_tokens,
            sink_distance=0,
            fanin_degree=cfg.subagents_per_round,
        ),
    ]


REQUIRED_SEGMENTS_BY_FAILURE_SITE: dict[str, tuple[str, ...]] = {
    "S1_RepoMapper": ("P1", "S10", "S11"),
    "S2_ToolBlocked": ("P1", "S14", "S15"),
    "O0_Diagnosis": ("P1", "P2", "P3", "P5", "P6"),
    "S4_Patch": ("P1", "P6", "S18"),
    "S5_ToolBlocked": ("P1", "P6", "S20", "S21"),
    "O0_Final": ("P1", "P2", "P3", "P5", "P6", "P7", "P8"),
}

# Agent execution order for timeline simulation (O0 appears in three phases).
TIMELINE_ROW_ORDER: tuple[tuple[str, str], ...] = (
    ("O0", "workflow_creation"),
    *((agent_id, "round1_fanout") for agent_id in ROUND1_SUBAGENTS),
    ("O0", "round1_fanin"),
    *((agent_id, "round2_fanout") for agent_id in ROUND2_SUBAGENTS),
    ("O0", "round2_fanin"),
)


def _segment_duration_s(segment: KVSegment) -> float:
    prefill = segment.tokens * PREFILL_S_PER_TOKEN
    decode = segment.output_tokens * DECODE_S_PER_TOKEN
    blocked = segment.blocked_duration_ms / 1000.0
    return prefill + decode + blocked


def _agent_segments(
    segments: list[KVSegment],
    agent_id: str,
) -> list[KVSegment]:
    return [segment for segment in segments if segment.agent_id == agent_id]


def _recovery_extra_s(
    segments: list[KVSegment],
    *,
    required_local: tuple[str, ...],
    peer_backup_lag_tokens: int,
) -> tuple[float, int, int, int]:
    by_local = {segment.segment_id.rsplit(":", 1)[1]: segment for segment in segments}
    recompute_tokens = 0
    suffix_tokens = 0
    for local in required_local:
        segment = by_local[local]
        reused, suffix = backed_up_token_count(
            segment,
            peer_backup_lag_tokens=peer_backup_lag_tokens,
        )
        recompute_tokens += max(segment.tokens - reused - suffix, 0)
        suffix_tokens += suffix
    replay_s = (
        recompute_tokens * RECOMPUTE_S_PER_TOKEN
        + suffix_tokens * SUFFIX_REPLAY_S_PER_TOKEN
    )
    return replay_s, recompute_tokens, suffix_tokens, recompute_tokens + suffix_tokens


def _segments_for_phase(
    segments: list[KVSegment],
    agent_id: str,
    phase_id: str,
) -> list[KVSegment]:
    return [
        segment
        for segment in segments
        if segment.agent_id == agent_id and segment.phase_id == phase_id
    ]


def build_workflow_timeline(
    *,
    workflow_id: str,
    segments: list[KVSegment],
    spec: RunSpec,
    impacted: bool,
    backup_overhead_per_workflow_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_local = REQUIRED_SEGMENTS_BY_FAILURE_SITE[spec.failure_site]
    failover_overhead = STRATEGY_FAILOVER_OVERHEAD_S[spec.strategy] if impacted else 0.0

    recovery_extra_s = 0.0
    recovery_recompute = 0
    if impacted:
        recovery_extra_s, recovery_recompute, _, _ = _recovery_extra_s(
            segments,
            required_local=required_local,
            peer_backup_lag_tokens=spec.peer_backup_lag_tokens,
        )

    timeline_rows: list[dict[str, Any]] = []
    t = BASE_OVERHEAD_S * 0.1

    # O0 workflow creation
    o0_start = t
    for segment in _segments_for_phase(segments, "O0", "workflow_creation"):
        t += _segment_duration_s(segment)
    timeline_rows.append(
        _agent_row(
            spec,
            workflow_id,
            agent_id="O0",
            phase_id="workflow_creation",
            start_s=o0_start,
            end_s=t,
            is_failure_target=_is_failure_target(
                agent_id="O0",
                phase_id="workflow_creation",
                failure_site=spec.failure_site,
                impacted=impacted,
            ),
            recovery_extra_s=0.0,
            impacted=impacted,
        )
    )

    # Round 1 fan-out parallel
    r1_start = t
    r1_durations: dict[str, float] = {}
    for agent_id in ROUND1_SUBAGENTS:
        agent_segs = _agent_segments(segments, agent_id)
        r1_durations[agent_id] = sum(
            _segment_duration_s(segment) for segment in agent_segs
        )
    r1_max = max(r1_durations.values())
    t = r1_start + r1_max
    for agent_id in ROUND1_SUBAGENTS:
        is_target = _is_failure_target(
            agent_id=agent_id,
            phase_id="round1_fanout",
            failure_site=spec.failure_site,
            impacted=impacted,
        )
        timeline_rows.append(
            _agent_row(
                spec,
                workflow_id,
                agent_id=agent_id,
                phase_id="round1_fanout",
                start_s=r1_start,
                end_s=r1_start + r1_durations[agent_id],
                is_failure_target=is_target,
                recovery_extra_s=recovery_extra_s if is_target else 0.0,
                impacted=impacted,
            )
        )

    # O0 round-1 fan-in
    o0_fanin1_start = t
    for segment in _segments_for_phase(segments, "O0", "round1_fanin"):
        t += _segment_duration_s(segment)
    is_o0_diag_target = _is_failure_target(
        agent_id="O0",
        phase_id="round1_fanin",
        failure_site=spec.failure_site,
        impacted=impacted,
    )
    timeline_rows.append(
        _agent_row(
            spec,
            workflow_id,
            agent_id="O0",
            phase_id="round1_fanin",
            start_s=o0_fanin1_start,
            end_s=t,
            is_failure_target=is_o0_diag_target,
            recovery_extra_s=recovery_extra_s if is_o0_diag_target else 0.0,
            impacted=impacted,
        )
    )

    # Round 2 fan-out parallel
    r2_start = t
    r2_durations: dict[str, float] = {}
    for agent_id in ROUND2_SUBAGENTS:
        agent_segs = _agent_segments(segments, agent_id)
        r2_durations[agent_id] = sum(
            _segment_duration_s(segment) for segment in agent_segs
        )
    r2_max = max(r2_durations.values())
    t = r2_start + r2_max
    for agent_id in ROUND2_SUBAGENTS:
        is_target = _is_failure_target(
            agent_id=agent_id,
            phase_id="round2_fanout",
            failure_site=spec.failure_site,
            impacted=impacted,
        )
        timeline_rows.append(
            _agent_row(
                spec,
                workflow_id,
                agent_id=agent_id,
                phase_id="round2_fanout",
                start_s=r2_start,
                end_s=r2_start + r2_durations[agent_id],
                is_failure_target=is_target,
                recovery_extra_s=recovery_extra_s if is_target else 0.0,
                impacted=impacted,
            )
        )

    # O0 round-2 fan-in
    o0_fanin2_start = t
    for segment in _segments_for_phase(segments, "O0", "round2_fanin"):
        t += _segment_duration_s(segment)
    is_o0_final_target = _is_failure_target(
        agent_id="O0",
        phase_id="round2_fanin",
        failure_site=spec.failure_site,
        impacted=impacted,
    )
    timeline_rows.append(
        _agent_row(
            spec,
            workflow_id,
            agent_id="O0",
            phase_id="round2_fanin",
            start_s=o0_fanin2_start,
            end_s=t,
            is_failure_target=is_o0_final_target,
            recovery_extra_s=recovery_extra_s if is_o0_final_target else 0.0,
            impacted=impacted,
        )
    )

    wct_s = (
        t
        + backup_overhead_per_workflow_s
        + failover_overhead
        + recovery_extra_s
    )
    workflow_row = {
        "run_id": _run_id(spec),
        "experiment": spec.experiment,
        "strategy": spec.strategy,
        "failure_site": spec.failure_site,
        "repeat_idx": spec.repeat_idx,
        "seed": spec.seed,
        "num_workflows": spec.num_workflows,
        "workflow_id": workflow_id,
        "wct_s": round(wct_s, 6),
        "impacted_by_failure": int(impacted),
        "recovery_extra_s": round(recovery_extra_s, 6),
        "failover_overhead_s": round(failover_overhead, 6),
        "backup_overhead_s": round(backup_overhead_per_workflow_s, 6),
        "recovery_recompute_tokens": recovery_recompute,
    }
    return timeline_rows, workflow_row


def _agent_row(
    spec: RunSpec,
    workflow_id: str,
    *,
    agent_id: str,
    phase_id: str,
    start_s: float,
    end_s: float,
    is_failure_target: bool,
    recovery_extra_s: float,
    impacted: bool,
) -> dict[str, Any]:
    duration_s = end_s - start_s
    if is_failure_target and recovery_extra_s > 0:
        status = "failed_and_recovered"
        total_duration = duration_s + recovery_extra_s
    else:
        status = "normal"
        total_duration = duration_s
    return {
        "run_id": _run_id(spec),
        "experiment": spec.experiment,
        "strategy": spec.strategy,
        "failure_site": spec.failure_site,
        "repeat_idx": spec.repeat_idx,
        "seed": spec.seed,
        "num_workflows": spec.num_workflows,
        "workflow_id": workflow_id,
        "agent_id": agent_id,
        "is_critical": int(is_critical_agent_id(agent_id)),
        "phase_id": phase_id,
        "start_s": round(start_s, 6),
        "end_s": round(end_s, 6),
        "duration_s": round(duration_s, 6),
        "recovery_extra_s": round(recovery_extra_s, 6),
        "total_duration_s": round(total_duration, 6),
        "is_failure_target": int(is_failure_target),
        "impacted_by_failure": int(impacted),
        "status": status,
        "failure_inject_s": round(end_s, 6) if is_failure_target else "",
    }


def _workflow_impacted_by_failure(workflow_index: int, seed: int) -> bool:
    mixed = (workflow_index * 1103515245 + seed * 12345 + 0x9E3779B9) & 0xFFFFFFFF
    return mixed % 100 < 50


def simulate_run(
    spec: RunSpec,
    workflow_cfg: MultiAgentWorkflowConfig | None = None,
) -> RunResult:
    cfg = workflow_cfg or MultiAgentWorkflowConfig(num_workflows=spec.num_workflows)
    cfg = MultiAgentWorkflowConfig(**{**asdict(cfg), "num_workflows": spec.num_workflows})
    policy_name = {
        "baseline": "baseline",
        "naive_backup": "naive_backup",
        "aegiskv": "aegiskv",
    }[spec.strategy]
    policy = AegisKVPolicyConfig(
        policy=policy_name,
        peer_backup_lag_tokens=spec.peer_backup_lag_tokens,
    )

    workflow_segments: dict[str, list[KVSegment]] = {}
    all_segments: list[KVSegment] = []
    for index in range(spec.num_workflows):
        wf_id = f"wf-{index}"
        wf_segs = build_two_round_coding_workflow(wf_id, cfg)
        workflow_segments[wf_id] = wf_segs
        all_segments.extend(wf_segs)

    protected = apply_backup_policy(all_segments, policy)
    protected_by_wf: dict[str, list[KVSegment]] = {}
    for segment in protected:
        protected_by_wf.setdefault(segment.workflow_id, []).append(segment)

    by_workflow_local: dict[tuple[str, str], KVSegment] = {}
    for segment in protected:
        local = segment.segment_id.rsplit(":", 1)[1]
        by_workflow_local[(segment.workflow_id, local)] = segment

    impacted_workflows = [
        f"wf-{index}"
        for index in range(spec.num_workflows)
        if _workflow_impacted_by_failure(index, spec.seed)
    ]
    if not impacted_workflows:
        impacted_workflows = ["wf-0"]

    required_local = REQUIRED_SEGMENTS_BY_FAILURE_SITE[spec.failure_site]
    segment_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    workflow_rows: list[dict[str, Any]] = []
    agent_timeline_rows: list[dict[str, Any]] = []

    total_required = 0
    total_reused = 0
    total_recompute = 0
    total_suffix = 0
    total_critical_recompute = 0
    total_non_critical_recompute = 0
    backed_up_tokens = sum(segment.tokens for segment in protected if segment.is_backed_up)

    backup_overhead_s = (
        backed_up_tokens * STRATEGY_BACKUP_OVERHEAD_PER_TOKEN_S[spec.strategy]
    )
    backup_overhead_per_wf = backup_overhead_s / max(spec.num_workflows, 1)

    for segment in protected:
        local = segment.segment_id.rsplit(":", 1)[1]
        required = (
            segment.workflow_id in impacted_workflows
            and local in set(required_local)
        )
        reused, suffix = backed_up_token_count(
            segment,
            peer_backup_lag_tokens=spec.peer_backup_lag_tokens,
        )
        reused = reused if required else 0
        suffix = suffix if required and segment.is_backed_up else 0
        recompute = max(segment.tokens - reused - suffix, 0) if required else 0
        if required:
            total_required += segment.tokens
            total_reused += reused
            total_suffix += suffix
            total_recompute += recompute
            if segment.is_critical_agent:
                total_critical_recompute += recompute
            else:
                total_non_critical_recompute += recompute

        segment_rows.append(
            {
                "run_id": _run_id(spec),
                "experiment": spec.experiment,
                "strategy": spec.strategy,
                "failure_site": spec.failure_site,
                "repeat_idx": spec.repeat_idx,
                "seed": spec.seed,
                "peer_backup_lag_tokens": spec.peer_backup_lag_tokens,
                "num_workflows": spec.num_workflows,
                "workflow_id": segment.workflow_id,
                "segment_id": segment.segment_id,
                "local_segment": local,
                "phase_id": segment.phase_id,
                "agent_id": segment.agent_id,
                "is_critical_agent": int(segment.is_critical_agent),
                "parent_agent_id": segment.parent_agent_id,
                "role": segment.role,
                "segment_type": segment.segment_type.value,
                "lifecycle_state": segment.lifecycle_state.value,
                "sink_distance": segment.sink_distance,
                "criticality_score": round(score_segment(segment), 6),
                "output_tokens": segment.output_tokens,
                "backup_action": segment.backup_action.value,
                "backup_location": segment.backup_location.value,
                "tokens": segment.tokens,
                "required_for_recovery": int(required),
                "reused_tokens": reused,
                "suffix_replay_tokens": suffix,
                "recomputed_tokens": recompute,
            }
        )

    for workflow_id in impacted_workflows:
        required_segments = [by_workflow_local[(workflow_id, item)] for item in required_local]
        required_tokens = sum(segment.tokens for segment in required_segments)
        reused_tokens = 0
        suffix_tokens = 0
        for segment in required_segments:
            reused, suffix = backed_up_token_count(
                segment,
                peer_backup_lag_tokens=spec.peer_backup_lag_tokens,
            )
            reused_tokens += reused
            suffix_tokens += suffix if segment.is_backed_up else 0
        recompute_tokens = max(required_tokens - reused_tokens - suffix_tokens, 0)
        if reused_tokens == 0:
            path = "no_valid_checkpoint"
        elif recompute_tokens == 0 and suffix_tokens == 0:
            path = "full_segment_restore"
        else:
            path = "prefix_restore_suffix_replay"
        recovery_rows.append(
            {
                "run_id": _run_id(spec),
                "experiment": spec.experiment,
                "strategy": spec.strategy,
                "failure_site": spec.failure_site,
                "repeat_idx": spec.repeat_idx,
                "seed": spec.seed,
                "workflow_id": workflow_id,
                "required_tokens": required_tokens,
                "gpu_replica_reuse_tokens": reused_tokens,
                "suffix_replay_tokens": suffix_tokens,
                "recomputed_tokens": recompute_tokens,
                "recovery_path": path,
            }
        )

    for index in range(spec.num_workflows):
        wf_id = f"wf-{index}"
        impacted = wf_id in impacted_workflows
        wf_segments = protected_by_wf[wf_id]
        tl_rows, wf_row = build_workflow_timeline(
            workflow_id=wf_id,
            segments=wf_segments,
            spec=spec,
            impacted=impacted,
            backup_overhead_per_workflow_s=backup_overhead_per_wf,
        )
        workflow_rows.append(wf_row)
        agent_timeline_rows.extend(tl_rows)

    wct_values = [float(row["wct_s"]) for row in workflow_rows]
    mean_wct = mean(wct_values) if wct_values else 0.0
    failover_overhead_s = STRATEGY_FAILOVER_OVERHEAD_S[spec.strategy]
    reuse_rate = total_reused / total_required if total_required else 0.0
    critical_survival = (
        1.0 - total_critical_recompute / total_required
        if total_required
        else 0.0
    )

    run = {
        "run_id": _run_id(spec),
        "experiment": spec.experiment,
        "strategy": spec.strategy,
        "failure_site": spec.failure_site,
        "repeat_idx": spec.repeat_idx,
        "seed": spec.seed,
        "peer_backup_lag_tokens": spec.peer_backup_lag_tokens,
        "num_workflows": spec.num_workflows,
        "impacted_workflows": len(impacted_workflows),
        "total_required_tokens": total_required,
        "gpu_replica_reuse_tokens": total_reused,
        "suffix_replay_tokens": total_suffix,
        "total_recomputed_tokens": total_recompute,
        "critical_recomputed_tokens": total_critical_recompute,
        "non_critical_recomputed_tokens": total_non_critical_recompute,
        "backed_up_tokens": backed_up_tokens,
        "kv_reuse_rate": reuse_rate,
        "critical_segment_survival_rate": critical_survival,
        "workflow_completion_time_s": mean_wct,
        "failover_overhead_s": failover_overhead_s,
        "backup_overhead_s": backup_overhead_s,
    }
    return RunResult(
        run=run,
        segments=segment_rows,
        recovery_paths=recovery_rows,
        workflows=workflow_rows,
        agent_timelines=agent_timeline_rows,
    )


def _run_id(spec: RunSpec) -> str:
    return (
        f"{spec.experiment}__{spec.strategy}__{spec.failure_site}"
        f"__lag{spec.peer_backup_lag_tokens}"
        f"__n{spec.num_workflows}__r{spec.repeat_idx}"
    )


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    var = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var) / math.sqrt(len(values))


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in runs:
        key = (
            row["experiment"],
            row["strategy"],
            row["failure_site"],
            row["peer_backup_lag_tokens"],
            row["num_workflows"],
        )
        groups.setdefault(key, []).append(row)
    points: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        metrics = [
            "workflow_completion_time_s",
            "total_recomputed_tokens",
            "critical_recomputed_tokens",
            "non_critical_recomputed_tokens",
            "backed_up_tokens",
            "gpu_replica_reuse_tokens",
            "suffix_replay_tokens",
            "kv_reuse_rate",
            "critical_segment_survival_rate",
            "backup_overhead_s",
        ]
        point = {
            "experiment": key[0],
            "strategy": key[1],
            "failure_site": key[2],
            "peer_backup_lag_tokens": key[3],
            "num_workflows": key[4],
            "repeats": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            point[f"{metric}_mean"] = mean(values)
            point[f"{metric}_stderr"] = stderr(values)
        points.append(point)
    return {"points": points}


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
