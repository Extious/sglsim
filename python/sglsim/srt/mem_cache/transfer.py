from dataclasses import dataclass


@dataclass(frozen=True)
class TransferModel:
    fixed_latency_s: float
    bandwidth_bytes_per_s: float

    def __post_init__(self):
        if self.fixed_latency_s < 0:
            raise ValueError("fixed_latency_s must be non-negative")
        if self.bandwidth_bytes_per_s <= 0:
            raise ValueError("bandwidth_bytes_per_s must be positive")

    def duration_s(self, num_bytes: int) -> float:
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        if num_bytes == 0:
            return 0.0
        return self.fixed_latency_s + num_bytes / self.bandwidth_bytes_per_s
