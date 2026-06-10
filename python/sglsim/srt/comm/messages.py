from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    src_rank: int
    dst_rank: int
    message_type: str
    payload: dict[str, Any] = field(default_factory=dict)
