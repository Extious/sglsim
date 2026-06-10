from __future__ import annotations


def resolve_max_num_reqs(
    *,
    token_capacity: int,
    context_len: int,
    dp_size: int = 1,
    user_max_running_requests: int | None = None,
) -> int:
    """Match sglang.srt.model_executor.model_runner_kv_cache_mixin._resolve_max_num_reqs."""
    context_len = max(context_len, 1)
    estimated = int(token_capacity / context_len * 512)
    estimated = max(min(estimated, 4096), 2048)

    if user_max_running_requests is not None:
        max_num_reqs = min(user_max_running_requests // max(dp_size, 1), estimated)
    else:
        max_num_reqs = min(estimated, token_capacity // 2)

    return max(1, max_num_reqs)
