from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    max_new_tokens: int = 1
    custom_params: dict = field(default_factory=dict)


@dataclass
class Request:
    rid: str
    input_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    output_ids: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    is_chunked: int = 0

    @property
    def extend_input_len(self) -> int:
        return len(self.input_ids)

    @property
    def prefix_indices(self) -> list[int]:
        return [0] * self.cached_tokens

    def is_finished(self) -> bool:
        return len(self.output_ids) >= self.sampling_params.max_new_tokens

