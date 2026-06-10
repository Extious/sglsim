from __future__ import annotations

from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.spec.model.base import ModelInfo


def input_token_count(req: GenerateReqInput) -> int:
    if req.input_ids is not None:
        return len(req.input_ids)
    if req.input_len is not None:
        return int(req.input_len)
    return 0


def validate_generate_request(req: GenerateReqInput, model: ModelInfo) -> None:
    context_len = model.context_len
    if context_len is None:
        return

    input_tokens = input_token_count(req)
    max_new_tokens = int(req.max_new_tokens)

    if input_tokens >= context_len:
        raise ValueError(
            f"The input ({input_tokens} tokens) is longer than the "
            f"model's context length ({context_len} tokens)."
        )

    if input_tokens + max_new_tokens >= context_len:
        total_tokens = input_tokens + max_new_tokens
        raise ValueError(
            "Requested token count exceeds the model's maximum context length "
            f"of {context_len} tokens. You requested a total of {total_tokens} "
            f"tokens: {input_tokens} tokens from the input and "
            f"{max_new_tokens} output tokens."
        )
