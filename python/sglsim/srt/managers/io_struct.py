from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sglsim.srt.managers.schedule_batch import Req, SamplingParams


@dataclass
class GenerateReqInput:
    rid: str
    input_ids: list[int] | None = None
    input_len: int | None = None
    prompt: str = ""
    max_new_tokens: int = 1
    created_time: float = 0.0
    routing_key: str | None = None
    simulation: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizedGenerateReqInput:
    rid: str
    input_ids: list[int]
    sampling_params: SamplingParams
    input_token_count: int | None = None
    routing_key: str | None = None
    original_prompt_len: int | None = None
    original_max_new_tokens: int | None = None

    @classmethod
    def from_generate(cls, req: GenerateReqInput) -> TokenizedGenerateReqInput:
        if req.input_ids is not None:
            input_ids = list(req.input_ids)
            input_token_count = len(input_ids)
        elif req.input_len is not None:
            input_ids = []
            input_token_count = int(req.input_len)
        else:
            raise ValueError("input_ids or input_len required")
        custom = {"simulation": dict(req.simulation)}
        return cls(
            rid=req.rid,
            input_ids=input_ids,
            input_token_count=input_token_count,
            routing_key=req.routing_key,
            sampling_params=SamplingParams(
                max_new_tokens=req.max_new_tokens,
                custom_params=custom,
            ),
        )

    def to_req(self) -> Req:
        sim = self.sampling_params.custom_params.get("simulation", {})
        return Req(
            rid=self.rid,
            input_ids=list(self.input_ids),
            input_token_count=self.input_token_count,
            sampling_params=self.sampling_params,
            created_time=float(sim.get("created_time", 0.0)),
            routing_key=self.routing_key,
            original_prompt_len=(
                self.original_prompt_len
                if self.original_prompt_len is not None
                else self.input_token_count
                if self.input_token_count is not None
                else len(self.input_ids)
            ),
            original_max_new_tokens=(
                self.original_max_new_tokens
                if self.original_max_new_tokens is not None
                else self.sampling_params.max_new_tokens
            ),
        )


@dataclass
class BatchTokenOutput:
    rid: str
    output_ids: list[int]
    finished: bool
