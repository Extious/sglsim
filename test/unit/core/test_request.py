from sglsim.srt.managers.schedule_batch import Req, RequestStatus, SamplingParams


def test_req_tracks_prefill_and_decode_lengths():
    req = Req(
        rid="r0",
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_new_tokens=2),
        created_time=1.5,
    )

    assert req.status == RequestStatus.WAITING
    assert req.prefill_length == 3
    assert req.decode_length == 0

    req.append_output_token(10)

    assert req.decode_length == 1
    assert req.total_length == 4
    assert req.remaining_decode_tokens == 1


def test_request_status_values_are_lower_case_strings():
    assert RequestStatus.WAITING.value == "waiting"
    assert RequestStatus.PREFILL.value == "prefill"
    assert RequestStatus.DECODE.value == "decode"
    assert RequestStatus.COMPLETED.value == "completed"
    assert RequestStatus.ABORTED.value == "aborted"
