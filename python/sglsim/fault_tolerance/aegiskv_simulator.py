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
    BackupLocation,
    KVSegment,
    LifecycleState,
    SegmentType,
    apply_backup_policy,
    backed_up_token_count,
)


FAILURE_SITES = (
    "E1_RepoMapper",
    "E2_ToolBlocked",
    "O0_J1_Diagnosis",
    "F1_Patch",
    "F2_ToolBlocked",
    "O0_J2_Final",
)


@dataclass(frozen=True)
class MultiAgentWorkflowConfig:
    num_workflows: int = 8
    workflow_start_stagger_s: float = 0.25
    tool_blocked_duration_ms: int = 180000
    system_prefix_tokens: int = 4096
    tool_schema_tokens: int = 1024
    task_plan_tokens: int = 2048
    selected_context_tokens: int = 4096
    branch_tail_tokens: int = 3072
    tool_blocked_prefix_tokens: int = 6144
    diagnosis_tokens: int = 4096
    verifier_context_tokens: int = 5120
    final_synthesis_tokens: int = 4096
    output_tokens_per_invocation: int = 512


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


def _seg(
    workflow_id: str,
    local_id: str,
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
            "S1",
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
            "S2",
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
            "S3",
            phase_id="workflow_creation",
            agent_id="O0",
            parent_agent_id="",
            role="main_orchestrator",
            segment_type=SegmentType.SHARED_HISTORY,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.task_plan_tokens,
            sink_distance=4,
            fanout_degree=3,
        ),
        _seg(
            workflow_id,
            "S10",
            phase_id="round1_fanout",
            agent_id="E1",
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
            phase_id="round1_fanout",
            agent_id="E1",
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
            phase_id="round1_fanout",
            agent_id="E3",
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
            phase_id="round1_fanout",
            agent_id="E2",
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
            phase_id="round1_fanout",
            agent_id="E2",
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
            "S5",
            phase_id="round1_fanin",
            agent_id="O0-J1",
            parent_agent_id="O0",
            role="main_diagnosis",
            segment_type=SegmentType.SHARED_HISTORY,
            lifecycle_state=LifecycleState.FANIN_READY,
            tokens=cfg.selected_context_tokens,
            sink_distance=2,
            fanin_degree=3,
        ),
        _seg(
            workflow_id,
            "S6",
            phase_id="round1_fanin",
            agent_id="O0-J1",
            parent_agent_id="O0",
            role="main_diagnosis",
            segment_type=SegmentType.DIAGNOSIS_SUMMARY,
            lifecycle_state=LifecycleState.FANIN_READY,
            tokens=cfg.diagnosis_tokens,
            sink_distance=2,
            fanout_degree=3,
            fanin_degree=3,
        ),
        _seg(
            workflow_id,
            "S18",
            phase_id="round2_fanout",
            agent_id="F1",
            parent_agent_id="O0-J1",
            role="patch_implementer",
            segment_type=SegmentType.HOT_DECODE_TAIL,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.branch_tail_tokens,
            sink_distance=1,
        ),
        _seg(
            workflow_id,
            "S20",
            phase_id="round2_fanout",
            agent_id="F2",
            parent_agent_id="O0-J1",
            role="regression_verifier",
            segment_type=SegmentType.VERIFIER_CONTEXT,
            lifecycle_state=LifecycleState.ACTIVE,
            tokens=cfg.verifier_context_tokens,
            sink_distance=1,
        ),
        _seg(
            workflow_id,
            "S21",
            phase_id="round2_fanout",
            agent_id="F2",
            parent_agent_id="O0-J1",
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
            phase_id="round2_fanout",
            agent_id="F3",
            parent_agent_id="O0-J1",
            role="review_checker",
            segment_type=SegmentType.BRANCH_TAIL,
            lifecycle_state=LifecycleState.SUMMARIZED,
            tokens=cfg.branch_tail_tokens,
            sink_distance=1,
            summarized=True,
        ),
        _seg(
            workflow_id,
            "S7",
            phase_id="round2_fanin",
            agent_id="O0-J2",
            parent_agent_id="O0-J1",
            role="main_final_resume",
            segment_type=SegmentType.SHARED_HISTORY,
            lifecycle_state=LifecycleState.FANIN_READY,
            tokens=cfg.selected_context_tokens,
            sink_distance=0,
            fanin_degree=3,
        ),
        _seg(
            workflow_id,
            "S8",
            phase_id="round2_fanin",
            agent_id="O0-J2",
            parent_agent_id="O0-J1",
            role="main_final_resume",
            segment_type=SegmentType.FINAL_SYNTHESIS_PREFIX,
            lifecycle_state=LifecycleState.FINAL_RESUME,
            tokens=cfg.final_synthesis_tokens,
            sink_distance=0,
            fanin_degree=3,
        ),
    ]


REQUIRED_SEGMENTS_BY_FAILURE_SITE: dict[str, tuple[str, ...]] = {
    "E1_RepoMapper": ("S1", "S10", "S11"),
    "E2_ToolBlocked": ("S1", "S14", "S15"),
    "O0_J1_Diagnosis": ("S1", "S2", "S3", "S5", "S6"),
    "F1_Patch": ("S1", "S6", "S18"),
    "F2_ToolBlocked": ("S1", "S6", "S20", "S21"),
    "O0_J2_Final": ("S1", "S2", "S3", "S5", "S6", "S7", "S8"),
}


def _base_workflow_time_s(cfg: MultiAgentWorkflowConfig) -> float:
    # A simple service model: prefill-heavy long context plus bounded decode.
    tokens = (
        cfg.system_prefix_tokens
        + cfg.task_plan_tokens
        + cfg.selected_context_tokens * 4
        + cfg.branch_tail_tokens * 4
        + cfg.tool_blocked_prefix_tokens * 2
        + cfg.diagnosis_tokens
        + cfg.verifier_context_tokens
        + cfg.final_synthesis_tokens
    )
    return 8.0 + tokens * 0.00011 + cfg.output_tokens_per_invocation * 0.002


def _workflow_impacted_by_failure(workflow_index: int, seed: int) -> bool:
    # Deterministic seed mixing approximates different failure times intersecting
    # slightly different active workflow sets across repeated online runs.
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
        "gpu_backup_agent": "gpu_backup_agent",
        "aegiskv_segment": "aegiskv_segment",
    }[spec.strategy]
    policy = AegisKVPolicyConfig(
        policy=policy_name,
        backup_budget_ratio=spec.backup_budget_ratio,
        peer_backup_lag_tokens=spec.peer_backup_lag_tokens,
    )
    all_segments: list[KVSegment] = []
    for index in range(spec.num_workflows):
        all_segments.extend(build_two_round_coding_workflow(f"wf-{index}", cfg))
    protected = apply_backup_policy(all_segments, policy)
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
    total_required = 0
    total_reused = 0
    total_recompute = 0
    total_suffix = 0
    protected_critical = 0
    total_critical_required = 0

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
        critical = segment.criticality_score >= 100.0 or segment.sink_distance <= 1
        if required and critical:
            total_critical_required += segment.tokens
            protected_critical += reused
        segment_rows.append(
            {
                "run_id": _run_id(spec),
                "experiment": spec.experiment,
                "strategy": spec.strategy,
                "failure_site": spec.failure_site,
                "repeat_idx": spec.repeat_idx,
                "seed": spec.seed,
                "backup_budget_ratio": spec.backup_budget_ratio,
                "peer_backup_lag_tokens": spec.peer_backup_lag_tokens,
                "num_workflows": spec.num_workflows,
                "workflow_id": segment.workflow_id,
                "segment_id": segment.segment_id,
                "local_segment": local,
                "phase_id": segment.phase_id,
                "agent_id": segment.agent_id,
                "parent_agent_id": segment.parent_agent_id,
                "role": segment.role,
                "segment_type": segment.segment_type.value,
                "lifecycle_state": segment.lifecycle_state.value,
                "sink_distance": segment.sink_distance,
                "criticality_score": round(segment.criticality_score, 6),
                "backup_action": segment.backup_action.value,
                "backup_location": segment.backup_location.value,
                "tokens": segment.tokens,
                "required_for_recovery": int(required),
                "reused_tokens": reused,
                "suffix_replay_tokens": suffix,
                "recomputed_tokens": recompute,
            }
        )
        if required:
            total_required += segment.tokens
            total_reused += reused
            total_suffix += suffix
            total_recompute += recompute

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

    base_wct = _base_workflow_time_s(cfg) + (spec.num_workflows - 1) * cfg.workflow_start_stagger_s
    failover_overhead_s = {"baseline": 1.2, "gpu_backup_agent": 0.20, "aegiskv_segment": 0.14}[spec.strategy]
    replay_s = total_recompute * 0.00018 + total_suffix * 0.000055
    backup_overhead_s = (
        sum(segment.tokens for segment in protected if segment.is_backed_up)
        * {"baseline": 0.0, "gpu_backup_agent": 0.0000045, "aegiskv_segment": 0.0000032}[spec.strategy]
    )
    workflow_completion_time_s = base_wct + failover_overhead_s + replay_s + backup_overhead_s
    reuse_rate = total_reused / total_required if total_required else 0.0
    critical_survival = (
        protected_critical / total_critical_required if total_critical_required else 0.0
    )
    run = {
        "run_id": _run_id(spec),
        "experiment": spec.experiment,
        "strategy": spec.strategy,
        "failure_site": spec.failure_site,
        "repeat_idx": spec.repeat_idx,
        "seed": spec.seed,
        "backup_budget_ratio": spec.backup_budget_ratio,
        "peer_backup_lag_tokens": spec.peer_backup_lag_tokens,
        "num_workflows": spec.num_workflows,
        "impacted_workflows": len(impacted_workflows),
        "total_required_tokens": total_required,
        "gpu_replica_reuse_tokens": total_reused,
        "suffix_replay_tokens": total_suffix,
        "total_recomputed_tokens": total_recompute,
        "kv_reuse_rate": reuse_rate,
        "critical_segment_survival_rate": critical_survival,
        "workflow_completion_time_s": workflow_completion_time_s,
        "failover_overhead_s": failover_overhead_s,
        "backup_overhead_s": backup_overhead_s,
    }
    return RunResult(run=run, segments=segment_rows, recovery_paths=recovery_rows)


def _run_id(spec: RunSpec) -> str:
    return (
        f"{spec.experiment}__{spec.strategy}__{spec.failure_site}"
        f"__b{spec.backup_budget_ratio:g}__lag{spec.peer_backup_lag_tokens}"
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
            row["backup_budget_ratio"],
            row["peer_backup_lag_tokens"],
            row["num_workflows"],
        )
        groups.setdefault(key, []).append(row)
    points: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        metrics = [
            "workflow_completion_time_s",
            "total_recomputed_tokens",
            "gpu_replica_reuse_tokens",
            "suffix_replay_tokens",
            "kv_reuse_rate",
            "critical_segment_survival_rate",
        ]
        point = {
            "experiment": key[0],
            "strategy": key[1],
            "failure_site": key[2],
            "backup_budget_ratio": key[3],
            "peer_backup_lag_tokens": key[4],
            "num_workflows": key[5],
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
