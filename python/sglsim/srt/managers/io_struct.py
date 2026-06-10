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
    failover_prefix_ids: list[int] | None = None
    pre_failover_output_tokens: int = 0
    pre_failover_backed_up_tokens: int = 0
    failed_dp_rank: int | None = None
    is_failover_retry: bool = False
    failover_recompute_tokens: int = 0

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
            failover_prefix_ids=(
                list(self.failover_prefix_ids)
                if self.failover_prefix_ids is not None
                else None
            ),
            pre_failover_output_tokens=int(self.pre_failover_output_tokens),
            pre_failover_backed_up_tokens=int(self.pre_failover_backed_up_tokens),
            failover_source_dp_rank=self.failed_dp_rank,
            is_failover_retried=bool(self.is_failover_retry),
            failover_recompute_tokens=int(self.failover_recompute_tokens),
        )


@dataclass
class BatchTokenOutput:
    rid: str
    output_ids: list[int]
    finished: bool


@dataclass
class SimulateGpuFailureReqInput:
    dp_rank: int = 0


@dataclass
class SimulateGpuRecoveryReqInput:
    dp_rank: int = 0


@dataclass
class ReqSnapshot:
    rid: str
    output_ids: list[int]
    backed_up_tokens: int = 0
    origin_input_ids: list[int] = field(default_factory=list)
    origin_input_token_count: int | None = None
    sampling_params: SamplingParams | None = None
    original_max_new_tokens: int | None = None
    simulation: dict[str, Any] = field(default_factory=dict)
    routing_key: str | None = None


@dataclass
class FailoverBatchReqInput:
    failed_dp_rank: int = 0
    snapshots: list[ReqSnapshot] = field(default_factory=list)
