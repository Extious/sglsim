from sglsim.srt.comm import Message, MessageTrace


def test_message_trace_records_timestamp_and_message_metadata_only():
    trace = MessageTrace()
    message = Message(
        src_rank=0,
        dst_rank=1,
        message_type="kv_prefetch",
        payload={"pages": [1, 2]},
    )

    trace.record_send(message=message, timestamp=1.5)

    assert len(trace.events) == 1
    event = trace.events[0]
    assert event.timestamp == 1.5
    assert event.event_type == "send"
    assert event.fields == {
        "src_rank": 0,
        "dst_rank": 1,
        "message_type": "kv_prefetch",
    }


def test_comm_package_exports_public_names():
    import sglsim.srt.comm as comm

    assert comm.__all__ == ["Message", "InprocChannel", "MessageTrace"]
