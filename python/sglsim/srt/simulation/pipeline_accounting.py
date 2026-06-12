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

    Overlap ON:  H2D hidden behind previous step's forward; D2H hidden
                 behind current forward.  Sleep = forward time only.

    Overlap OFF: H2D is on the critical path (previous batch drained before
                 this one starts), but D2H still overlaps with forward via
                 the GPU copy engine — a hardware fact that exists regardless
                 of scheduler-level overlap.
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

    # D2H runs on the GPU copy engine which operates concurrently with the
    # compute engine (SM).  This hardware-level parallelism exists regardless
    # of whether the *scheduler* overlaps prefill + decode in the event loop.
    # `--disable-overlap-schedule` serialises the scheduler batches but does
    # NOT disable the independent copy engine, so D2H always partially hides
    # behind forward work.
    overlapped_backup = min(backup_pending_s, forward_s)
    backup_remainder = backup_pending_s - overlapped_backup

    if overlap:
        # H2D is hidden behind the previous step's forward (overlap ON).
        load_visible = max(h2d_s - last_infer_s, 0.0)
    else:
        # H2D is on the critical path because the scheduler waits for the
        # previous batch to fully drain before starting the next one.
        load_visible = h2d_s

    visible = load_visible + forward_s + backup_remainder

    return PipelineStepResult(
        visible_gpu_s=visible,
        backup_remainder_s=backup_remainder,
        forward_s=forward_s,
        overlapped_backup_s=overlapped_backup,
    )
