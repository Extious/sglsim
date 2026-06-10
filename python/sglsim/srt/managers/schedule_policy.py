from __future__ import annotations

from collections import deque
from typing import TypeVar

from sglsim.srt.managers.schedule_batch import Req, ScheduleBatch

T = TypeVar("T")


def pop_fifo(queue: deque[T], limit: int) -> list[T]:
    if limit <= 0:
        return []
    out: list[T] = []
    while queue and len(out) < limit:
        out.append(queue.popleft())
    return out


class SchedulePolicy:
    def __init__(self, policy: str = "fcfs") -> None:
        self.policy = policy

    def calc_priority(self, waiting_queue: deque[Req], running_batch: ScheduleBatch) -> None:
        if self.policy == "fcfs":
            return
        raise ValueError(f"Unsupported schedule_policy={self.policy!r}")
