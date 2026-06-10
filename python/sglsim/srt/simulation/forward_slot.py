"""Forward-slot wall-clock model for pipeline simulation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ForwardSlot:
    slot_end_wall: float | None = None
    last_infer_s: float = 0.0

    def cpu_budget_s(self, now: float) -> float:
        if self.slot_end_wall is None:
            return 0.0
        return max(0.0, self.slot_end_wall - now)

    def cpu_overflow_s(self, *, now: float, sched_elapsed_s: float) -> float:
        return max(0.0, sched_elapsed_s - self.cpu_budget_s(now))

    def record_forward(
        self,
        *,
        now: float,
        forward_s: float,
        sleep_s: float,
    ) -> None:
        self.last_infer_s = forward_s
        base = self.slot_end_wall if self.slot_end_wall is not None else now
        self.slot_end_wall = max(now, base) + sleep_s
