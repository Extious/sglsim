"""Pipeline stage accounting for GPU forward simulation (H2D / Forward / D2H overlap)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStages:
    h2d_s: float
    forward_s: float
    backup_pending_s: float


@dataclass(frozen=True)
class PipelineStepResult:
    visible_gpu_s: float
    backup_remainder_s: float
    forward_s: float
    overlapped_backup_s: float


@dataclass(frozen=True)
class PipelineSleepResult:
    sleep_s: float
    forward_s: float
    diagnostic_visible_gpu_s: float
    comm_h2d_visible_s: float
    comm_d2h_remainder_s: float


def compute_pipeline_sleep_s(
    stages: PipelineStages,
    *,
    last_infer_s: float,
    overlap: bool,
) -> PipelineSleepResult:
    """Wall-clock sleep for one scheduler step (Pipeline v2).

    With overlap enabled, dedicated copy streams hide HiCache transfer behind
    forward work, so the step only sleeps for forward time. With overlap
    disabled, the normal event loop observes the serial path: H2D, forward, and
    pending backup all contribute to the wall clock for the step.
    """
    diagnostic = compute_visible_gpu(stages, last_infer_s=last_infer_s, overlap=overlap)
    forward_s = diagnostic.forward_s
    h2d_s = max(stages.h2d_s, 0.0)
    last_infer_s = max(last_infer_s, 0.0)

    if overlap:
        comm_h2d_visible = max(h2d_s - last_infer_s, 0.0)
    else:
        comm_h2d_visible = h2d_s
    sleep_s = forward_s if overlap else diagnostic.visible_gpu_s

    return PipelineSleepResult(
        sleep_s=sleep_s,
        forward_s=forward_s,
        diagnostic_visible_gpu_s=diagnostic.visible_gpu_s,
        comm_h2d_visible_s=comm_h2d_visible,
        comm_d2h_remainder_s=diagnostic.backup_remainder_s,
    )


def compute_visible_gpu(
    stages: PipelineStages,
    *,
    last_infer_s: float,
    overlap: bool,
) -> PipelineStepResult:
    h2d_s = max(stages.h2d_s, 0.0)
    forward_s = max(stages.forward_s, 0.0)
    backup_pending_s = max(stages.backup_pending_s, 0.0)
    last_infer_s = max(last_infer_s, 0.0)

    if overlap:
        load_visible = max(h2d_s - last_infer_s, 0.0)
        overlapped_backup = min(backup_pending_s, forward_s)
        backup_remainder = backup_pending_s - overlapped_backup
        visible = load_visible + forward_s + backup_remainder
    else:
        load_visible = h2d_s
        overlapped_backup = 0.0
        backup_remainder = backup_pending_s
        visible = h2d_s + forward_s + backup_pending_s

    return PipelineStepResult(
        visible_gpu_s=visible,
        backup_remainder_s=backup_remainder,
        forward_s=forward_s,
        overlapped_backup_s=overlapped_backup,
    )
