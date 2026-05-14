from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sglsim.srt.scheduling.request import Request


class ForwardMode(Enum):
    EXTEND = "extend"
    DECODE = "decode"

    def is_extend(self) -> bool:
        return self is ForwardMode.EXTEND

    def is_decode(self) -> bool:
        return self is ForwardMode.DECODE


@dataclass
class ScheduleBatch:
    reqs: list[Request]
    forward_mode: ForwardMode

    def __len__(self) -> int:
        return len(self.reqs)

    @classmethod
    def extend(cls, reqs: list[Request]) -> "ScheduleBatch":
        return cls(reqs=reqs, forward_mode=ForwardMode.EXTEND)

    @classmethod
    def decode(cls, reqs: list[Request]) -> "ScheduleBatch":
        return cls(reqs=reqs, forward_mode=ForwardMode.DECODE)


@dataclass
class GenerationBatchResult:
    output_token_id: int = 1

