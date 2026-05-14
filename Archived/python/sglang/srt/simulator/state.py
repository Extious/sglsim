from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SimulatorState:
    iteration: int = 0
    global_clock: float = 0
    last_inference_dur: float = 0
    current_inference_dur: float = 0
    hicache_l2_load_dur: float = 0
    hicache_l2_backup_dur: float = 0

    def reset(self) -> None:
        self.iteration = 0
        self.global_clock = 0
        self.last_inference_dur = 0
        self.current_inference_dur = 0
        self.hicache_l2_load_dur = 0
        self.hicache_l2_backup_dur = 0

    def inc_iteration(self) -> None:
        self.iteration += 1

    def step_global_clock(self, dur: float) -> None:
        self.global_clock += dur

    def set_global_clock(self, clock: float) -> None:
        self.global_clock = clock

    def set_current_inference_dur(self, dur: float) -> None:
        self.last_inference_dur = self.current_inference_dur
        self.current_inference_dur = dur

    def inc_hicache_l2_load_dur(self, dur: float) -> None:
        self.hicache_l2_load_dur += dur

    def inc_hicache_l2_backup_dur(self, dur: float) -> None:
        self.hicache_l2_backup_dur += dur

    def pop_hicache_l2_load_dur(self) -> float:
        dur = self.hicache_l2_load_dur
        self.hicache_l2_load_dur = 0
        return dur

    def pop_hicache_l2_backup_dur(self) -> float:
        dur = self.hicache_l2_backup_dur
        self.hicache_l2_backup_dur = 0
        return dur


_active_state: Optional[SimulatorState] = None
_active_runtime: Optional[Any] = None


def set_active_state(state: Optional[SimulatorState]) -> None:
    global _active_state
    _active_state = state


def get_active_state() -> Optional[SimulatorState]:
    return _active_state


def set_active_runtime(runtime: Optional[Any]) -> None:
    global _active_runtime
    _active_runtime = runtime


def get_active_runtime() -> Optional[Any]:
    return _active_runtime
