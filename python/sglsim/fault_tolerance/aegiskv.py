from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable


class SegmentType(str, Enum):
    SYSTEM_TOOL_PREFIX = "SystemToolPrefix"
    TOOL_SCHEMA = "ToolSchema"
    SHARED_HISTORY = "SharedHistory"
    DIAGNOSIS_SUMMARY = "DiagnosisSummary"
    VERIFIER_CONTEXT = "VerifierContext"
    TOOL_BLOCKED_PREFIX = "ToolBlockedPrefix"
    HOT_DECODE_TAIL = "HotDecodeTail"
    FINAL_SYNTHESIS_PREFIX = "FinalSynthesisPrefix"
    BRANCH_TAIL = "BranchTail"


class LifecycleState(str, Enum):
    ACTIVE = "Active"
    TOOL_BLOCKED = "ToolBlocked"
    FANIN_READY = "FaninReady"
    SUMMARIZED = "Summarized"
    FINAL_RESUME = "FinalResume"
    COMPLETED = "Completed"


class BackupLocation(str, Enum):
    NONE = "None"
    GPU_REPLICA = "GPUReplica"


class BackupAction(str, Enum):
    KEEP_ACTIVE_ONLY = "KeepActiveOnly"
    REPLICATE_TO_PEER_GPU = "ReplicateToPeerGPU"
    RELEASE_BACKUP = "ReleaseBackup"
    RESTORE_BEFORE_REPLAY = "RestoreBeforeReplay"


CRITICAL_AGENT_IDS: frozenset[str] = frozenset({"O0", "S2", "S5"})


def is_critical_agent_id(agent_id: str) -> bool:
    return agent_id in CRITICAL_AGENT_IDS


@dataclass(frozen=True)
class KVSegment:
    segment_id: str
    workflow_id: str
    phase_id: str
    agent_id: str
    parent_agent_id: str
    role: str
    segment_type: SegmentType
    lifecycle_state: LifecycleState
    tokens: int
    sink_distance: int
    output_tokens: int = 64
    fanout_degree: int = 1
    fanin_degree: int = 1
    blocked_duration_ms: int = 0
    summarized: bool = False
    backup_location: BackupLocation = BackupLocation.NONE
    backup_action: BackupAction = BackupAction.KEEP_ACTIVE_ONLY
    criticality_score: float = 0.0

    @property
    def is_backed_up(self) -> bool:
        return self.backup_location == BackupLocation.GPU_REPLICA

    @property
    def is_critical_agent(self) -> bool:
        return is_critical_agent_id(self.agent_id)


@dataclass(frozen=True)
class AegisKVPolicyConfig:
    policy: str = "aegiskv"
    backup_target: str = "GPUReplica"
    peer_backup_lag_tokens: int = 8


TYPE_WEIGHTS: dict[SegmentType, float] = {
    SegmentType.SYSTEM_TOOL_PREFIX: 92.0,
    SegmentType.TOOL_SCHEMA: 45.0,
    SegmentType.SHARED_HISTORY: 68.0,
    SegmentType.DIAGNOSIS_SUMMARY: 118.0,
    SegmentType.VERIFIER_CONTEXT: 82.0,
    SegmentType.TOOL_BLOCKED_PREFIX: 125.0,
    SegmentType.HOT_DECODE_TAIL: 74.0,
    SegmentType.FINAL_SYNTHESIS_PREFIX: 132.0,
    SegmentType.BRANCH_TAIL: 34.0,
}


def score_segment(segment: KVSegment) -> float:
    """Diagnostic score only; not used for backup placement in v2 policies."""
    score = TYPE_WEIGHTS[segment.segment_type]
    if segment.lifecycle_state == LifecycleState.TOOL_BLOCKED:
        score += 34.0
    elif segment.lifecycle_state == LifecycleState.FANIN_READY:
        score += 22.0
    elif segment.lifecycle_state == LifecycleState.FINAL_RESUME:
        score += 38.0
    elif segment.lifecycle_state == LifecycleState.SUMMARIZED:
        score -= 38.0

    if segment.sink_distance <= 0:
        score += 38.0
    elif segment.sink_distance == 1:
        score += 28.0
    elif segment.sink_distance == 2:
        score += 14.0

    score += min(segment.blocked_duration_ms / 60000.0, 4.0) * 5.0
    score += max(segment.fanout_degree - 1, 0) * 3.0
    score += max(segment.fanin_degree - 1, 0) * 2.0
    if segment.summarized:
        score -= 45.0
    return max(score, 0.0)


def annotate_scores(segments: Iterable[KVSegment]) -> list[KVSegment]:
    return [
        replace(segment, criticality_score=score_segment(segment))
        for segment in segments
    ]


def _eligible_for_backup(segment: KVSegment) -> bool:
    if segment.lifecycle_state == LifecycleState.COMPLETED:
        return False
    return True


def apply_backup_policy(
    segments: Iterable[KVSegment],
    config: AegisKVPolicyConfig,
) -> list[KVSegment]:
    scored = annotate_scores(segments)
    policy = config.policy

    if policy in {"baseline", "none"}:
        chosen: set[str] = set()
    elif policy == "naive_backup":
        chosen = {
            segment.segment_id
            for segment in scored
            if _eligible_for_backup(segment)
        }
    elif policy == "aegiskv":
        chosen = {
            segment.segment_id
            for segment in scored
            if _eligible_for_backup(segment) and segment.is_critical_agent
        }
    else:
        raise ValueError(f"unknown AegisKV policy: {policy}")

    out: list[KVSegment] = []
    for segment in scored:
        if segment.segment_id in chosen:
            out.append(
                replace(
                    segment,
                    backup_location=BackupLocation.GPU_REPLICA,
                    backup_action=BackupAction.REPLICATE_TO_PEER_GPU,
                )
            )
        else:
            action = (
                BackupAction.RELEASE_BACKUP
                if segment.summarized
                else BackupAction.KEEP_ACTIVE_ONLY
            )
            out.append(
                replace(
                    segment,
                    backup_location=BackupLocation.NONE,
                    backup_action=action,
                )
            )
    return out


def backed_up_token_count(
    segment: KVSegment,
    *,
    peer_backup_lag_tokens: int,
) -> tuple[int, int]:
    if not segment.is_backed_up:
        return (0, 0)
    if segment.segment_type in {
        SegmentType.TOOL_BLOCKED_PREFIX,
        SegmentType.HOT_DECODE_TAIL,
        SegmentType.FINAL_SYNTHESIS_PREFIX,
    }:
        suffix = min(max(peer_backup_lag_tokens, 0), segment.tokens)
        return (max(segment.tokens - suffix, 0), suffix)
    return (segment.tokens, 0)
