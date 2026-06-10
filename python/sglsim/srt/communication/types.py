from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommPath(str, Enum):
    H2D = "h2d"
    D2H = "d2h"
    DISK_READ = "disk_read"
    DISK_WRITE = "disk_write"


@dataclass(frozen=True)
class CommunicationRequest:
    path: CommPath
    host_page_ids: tuple[int, ...] = ()
    device_page_ids: tuple[int, ...] = ()
    page_ids: tuple[int, ...] = ()
    page_size: int = 16


@dataclass(frozen=True)
class CommunicationPrediction:
    duration_s: float
    source: str
    path: CommPath
    num_bytes: int = 0
