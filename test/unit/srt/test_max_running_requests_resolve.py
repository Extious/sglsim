from sglsim.srt.model_executor.model_runner_kv_cache_mixin import resolve_max_num_reqs


def test_resolve_max_num_reqs_auto():
    cap = resolve_max_num_reqs(
        token_capacity=400_000,
        context_len=40960,
        dp_size=2,
        user_max_running_requests=None,
    )
    assert cap >= 2048


def test_resolve_max_num_reqs_user_cap_scales_by_dp():
    cap = resolve_max_num_reqs(
        token_capacity=400_000,
        context_len=40960,
        dp_size=2,
        user_max_running_requests=100,
    )
    assert cap == 50
