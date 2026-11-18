"""Time advancement drivers shared by online and offline schedulers.

The scheduler owns SGLang control flow and metadata side effects.  A driver
only answers "what time is it?" and advances the Pipeline v2 critical path.
The default wall driver preserves real sleeping; the logical driver never
sleeps and owns one monotonic clock for one DP rank.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

from sglsim.srt.simulation.clock import Clock
from sglsim.srt.simulation.forward_slot import ForwardSlot
from sglsim.srt.simulation.pipeline_accounting import (
    PipelineSleepResult,
    PipelineStages,
    compute_pipeline_sleep_s,
)


@dataclass(frozen=True)
class PipelineAdvanceResult:
    accounting: PipelineSleepResult
    cpu_overflow_s: float
    elapsed_s: float
    completed_at: float


@dataclass(frozen=True)
class LogicalPipelinePlan:
    accounting: PipelineSleepResult
    started_at: float
    completed_at: float


class SchedulerTimeDriver(Protocol):
    """Clock/advance seam; policy and cache code must remain mode-agnostic."""

    mode: str

    def now(self) -> float: ...

    def delay(self, duration_s: float) -> float: ...

    def idle_wait(self, duration_s: float) -> None: ...

    def advance_pipeline(
        self,
        stages: PipelineStages,
        *,
        last_infer_s: float,
        sched_elapsed_s: float,
        overlap: bool,
        forward_slot: ForwardSlot,
    ) -> PipelineAdvanceResult: ...


class WallClockDriver:
    """Existing online wall-clock behavior behind an injectable interface."""

    mode = "wall"

    def __init__(
        self,
        *,
        now_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn

    def now(self) -> float:
        return float(self._now_fn())

    def delay(self, duration_s: float) -> float:
        duration_s = max(float(duration_s), 0.0)
        if duration_s > 0:
            self._sleep_fn(duration_s)
        return self.now()

    def idle_wait(self, duration_s: float) -> None:
        duration_s = max(float(duration_s), 0.0)
        if duration_s > 0:
            self._sleep_fn(duration_s)

    def advance_pipeline(
        self,
        stages: PipelineStages,
        *,
        last_infer_s: float,
        sched_elapsed_s: float,
        overlap: bool,
        forward_slot: ForwardSlot,
    ) -> PipelineAdvanceResult:
        accounting = compute_pipeline_sleep_s(
            stages,
            last_infer_s=last_infer_s,
            overlap=overlap,
        )
        now = self.now()
        cpu_overflow_s = (
            forward_slot.cpu_overflow_s(
                now=now,
                sched_elapsed_s=max(float(sched_elapsed_s), 0.0),
            )
            if overlap
            else 0.0
        )
        elapsed_s = accounting.sleep_s + cpu_overflow_s
        if elapsed_s > 0:
            self._sleep_fn(elapsed_s)
        completed_at = self.now()
        forward_slot.record_forward(
            now=completed_at,
            forward_s=accounting.forward_s,
            sleep_s=accounting.sleep_s,
        )
        return PipelineAdvanceResult(
            accounting=accounting,
            cpu_overflow_s=cpu_overflow_s,
            elapsed_s=elapsed_s,
            completed_at=completed_at,
        )


class LogicalClockDriver:
    """Independent logical clock for exactly one DP scheduler rank."""

    mode = "logical"

    def __init__(self, *, start_s: float = 0.0, clock: Clock | None = None) -> None:
        if clock is not None and start_s != 0.0:
            raise ValueError("pass either clock or start_s, not both")
        self.clock = clock if clock is not None else Clock(float(start_s))

    def now(self) -> float:
        return self.clock.now

    def set_time(self, timestamp_s: float) -> float:
        self.clock.set(float(timestamp_s))
        return self.clock.now

    def delay(self, duration_s: float) -> float:
        self.clock.advance(max(float(duration_s), 0.0))
        return self.clock.now

    def idle_wait(self, duration_s: float) -> None:
        # Wall-only anti-spin wait.  Offline idle jumps are owned by the parent
        # event coordinator, so an idle wait must not invent logical work.
        del duration_s

    def advance_pipeline(
        self,
        stages: PipelineStages,
        *,
        last_infer_s: float,
        sched_elapsed_s: float,
        overlap: bool,
        forward_slot: ForwardSlot,
    ) -> PipelineAdvanceResult:
        plan = self.plan_pipeline(
            stages,
            last_infer_s=last_infer_s,
            sched_elapsed_s=sched_elapsed_s,
            overlap=overlap,
        )
        return self.complete_pipeline(plan, forward_slot=forward_slot)

    def plan_pipeline(
        self,
        stages: PipelineStages,
        *,
        last_infer_s: float,
        sched_elapsed_s: float,
        overlap: bool,
    ) -> LogicalPipelinePlan:
        """Compute a rank completion without advancing the rank clock."""

        del sched_elapsed_s  # OS/Python scheduling jitter is wall-only.
        accounting = compute_pipeline_sleep_s(
            stages,
            last_infer_s=last_infer_s,
            overlap=overlap,
        )
        return LogicalPipelinePlan(
            accounting=accounting,
            started_at=self.clock.now,
            completed_at=self.clock.now + accounting.sleep_s,
        )

    def complete_pipeline(
        self,
        plan: LogicalPipelinePlan,
        *,
        forward_slot: ForwardSlot,
    ) -> PipelineAdvanceResult:
        """Commit one previously planned rank completion."""

        # The parent GlobalDES serializes timestamps as integer nanoseconds;
        # converting that event time back to float seconds can exceed the
        # child's plan by sub-nanosecond rounding noise.  Larger advances are
        # still a causal error.
        completion_tolerance_s = 1e-9
        if self.clock.now > plan.completed_at + completion_tolerance_s:
            raise ValueError(
                "logical rank advanced beyond its pending pipeline completion"
            )
        if self.clock.now < plan.completed_at:
            self.clock.set(plan.completed_at)
        elapsed_s = plan.accounting.sleep_s
        completed_at = self.clock.now
        forward_slot.record_forward(
            now=completed_at,
            forward_s=plan.accounting.forward_s,
            sleep_s=elapsed_s,
        )
        return PipelineAdvanceResult(
            accounting=plan.accounting,
            cpu_overflow_s=0.0,
            elapsed_s=elapsed_s,
            completed_at=completed_at,
        )
