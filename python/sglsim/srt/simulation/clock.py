from dataclasses import dataclass


@dataclass
class Clock:
    now: float = 0.0

    def advance(self, delta: float) -> None:
        if delta < 0:
            raise ValueError("clock cannot advance by a negative duration")
        self.now += delta

    def set(self, timestamp: float) -> None:
        if timestamp < self.now:
            raise ValueError("clock cannot move backwards")
        self.now = timestamp
