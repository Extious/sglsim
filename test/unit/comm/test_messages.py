from sglsim.srt.comm.messages import Message


def test_message_records_src_dst_type_and_payload():
    message = Message(
        src_rank=0,
        dst_rank=1,
        message_type="kv_prefetch",
        payload={"pages": [1, 2]},
    )

    assert message.src_rank == 0
    assert message.dst_rank == 1
    assert message.message_type == "kv_prefetch"
    assert message.payload == {"pages": [1, 2]}


def test_message_payload_defaults_to_distinct_empty_dicts():
    first = Message(src_rank=0, dst_rank=1, message_type="kv_prefetch")
    second = Message(src_rank=1, dst_rank=0, message_type="kv_ack")

    assert first.payload == {}
    assert second.payload == {}
    assert first.payload is not second.payload
