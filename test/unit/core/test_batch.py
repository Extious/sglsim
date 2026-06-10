from sglsim.srt.managers.schedule_batch import Batch, ForwardMode
from sglsim.srt.managers.schedule_batch import Req, SamplingParams


def test_prefill_batch_builds_prediction_items():
    req = Req(
        rid="r0",
        input_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_new_tokens=1),
        cached_length=2,
        kv_committed_length=2,
        prefill_chunks_remaining=[2],
    )
    batch = Batch(batch_id=0, rank_id=0, mode=ForwardMode.PREFILL, reqs=[req])

    assert batch.batch_size == 1
    assert batch.prediction_items == [{"extend_length": 2, "past_kv_length": 2}]


def test_decode_batch_builds_prediction_items_from_kv_committed_length():
    req = Req(
        rid="r0",
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=2),
        kv_committed_length=4,
    )
    req.append_output_token(10)
    batch = Batch(batch_id=1, rank_id=0, mode=ForwardMode.DECODE, reqs=[req])

    assert batch.prediction_items == [{"extend_length": 1, "past_kv_length": 4}]
