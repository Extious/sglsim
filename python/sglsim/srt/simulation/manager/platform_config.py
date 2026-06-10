from __future__ import annotations

from dataclasses import dataclass, field

from sglsim.srt.spec.accelerator.base import AcceleratorInfo


@dataclass
class PlatformConfig:
    accelerator: AcceleratorInfo = field(default_factory=AcceleratorInfo)
    disk_read_bandwidth_gb: float = 8.0
    disk_write_bandwidth_gb: float = 8.0
    memory_read_bandwidth_gb: float = 64.0
    memory_write_bandwidth_gb: float = 64.0
