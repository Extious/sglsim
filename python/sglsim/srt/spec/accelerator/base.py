from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AcceleratorInfo:
    name: str = "a100_sxm"
    hbm_capacity_gb: float = 80.0
