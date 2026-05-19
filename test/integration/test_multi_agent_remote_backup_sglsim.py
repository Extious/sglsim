from __future__ import annotations


def test_remote_backup_response_metadata_has_cache_hit_csv_fields():
    from sglsim.srt.entrypoints.openai.serving_completions import (
        completion_response_from_state,
    )
    from sglsim.srt.managers.io_struct import RequestState, SamplingParams

    state = RequestState(
        rid="r1",
        origin_input_text="prompt",
        origin_input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=1),
        return_cached_tokens_details=True,
        output_ids=[4],
        output_text="tok4",
        cached_tokens_details={
            "device": 1,
            "host": 0,
            "storage": 2,
            "storage_query": 3,
            "reused_device": 1,
            "reused_host": 0,
            "reused_storage": 2,
            "storage_backend": "remote_backup",
        },
    )

    body = completion_response_from_state(state, model="simulator-model")
    usage = body["usage"]
    details = usage["cached_tokens_details"]

    cache_row = {
        "l1_match": int(details.get("device", 0)),
        "l2_match": int(details.get("host", 0)),
        "remote_match": int(details.get("storage_query", 0)),
        "remote_prefetch": int(details.get("storage", 0)),
        "reused_device": int(details.get("reused_device", 0)),
        "reused_host": int(details.get("reused_host", 0)),
        "reused_storage": int(details.get("reused_storage", 0)),
    }

    assert details["storage_backend"] == "remote_backup"
    assert cache_row == {
        "l1_match": 1,
        "l2_match": 0,
        "remote_match": 3,
        "remote_prefetch": 2,
        "reused_device": 1,
        "reused_host": 0,
        "reused_storage": 2,
    }
