from pathlib import Path

from sglsim.srt.config import CacheConfig
from sglsim.srt.managers.schedule_batch import Req, SamplingParams
from sglsim.srt.mem_cache.kv_cache_pool import KVCachePool
from sglsim.srt.utils.synthesis_prompt import SynthesisPromptBuilder, SynthesisWorkloadConfig


def _extend_full_prefill(controller: KVCachePool, req: Req) -> None:
    controller.prepare_prefill(req)
    controller.extend_prefill(req, len(req.input_ids))


def test_file_prompt_body_token_ids_are_shared(tmp_path):
    prompt_path = tmp_path / "body.txt"
    prompt_path.write_text("shared radix body for synthesis benchmark", encoding="utf-8")
    builder = SynthesisPromptBuilder(
        SynthesisWorkloadConfig(
            model_path="mock-model",
            prompt_path=prompt_path,
            seed=1,
        )
    )
    ids1 = builder.build_token_ids("1", 0)
    ids2 = builder.build_token_ids("2", 1)
    body = builder.body_token_ids()
    assert ids1[-len(body) :] == body
    assert ids2[-len(body) :] == body
    assert ids1[: len(ids1) - len(body)] != ids2[: len(ids2) - len(body)]

    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=256,
            l2_capacity_pages=256,
        )
    )
    req0 = Req("1", ids1, SamplingParams(max_new_tokens=1))
    _extend_full_prefill(controller, req0)
    controller.check_hicache_events()

    assert ids1[: len(ids1) - len(body)] != ids2[: len(ids2) - len(body)]


def test_synthesis_two_reqs_low_prefix_match_and_kv_usage(tmp_path):
    prompt_path = tmp_path / "body.txt"
    prompt_path.write_text("shared radix body for synthesis benchmark", encoding="utf-8")
    builder = SynthesisPromptBuilder(
        SynthesisWorkloadConfig(
            model_path="mock-model",
            prompt_path=prompt_path,
            seed=1,
        )
    )
    ids1 = builder.build_token_ids("1", 0)
    ids2 = builder.build_token_ids("2", 1)
    body = builder.body_token_ids()
    marker_len = len(ids1) - len(body)

    controller = KVCachePool(
        CacheConfig(
            page_size=1,
            l1_capacity_pages=4096,
            l2_capacity_pages=0,
        )
    )
    match_before = controller.radix.match_prefix(ids2)
    assert match_before.matched_length < len(body)

    _extend_full_prefill(controller, Req("1", ids1, SamplingParams(max_new_tokens=1)))
    match_after_first = controller.radix.match_prefix(ids2)
    assert match_after_first.matched_length <= marker_len + 2

    _extend_full_prefill(controller, Req("2", ids2, SamplingParams(max_new_tokens=1)))
    used, total, _, _ = controller.kv_token_usage_details()
    assert total > 0
    assert used >= len(ids1) + len(ids2) - 4
    assert used > len(body)
