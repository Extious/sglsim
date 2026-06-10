from sglsim.srt.comm import InprocChannel, Message


def test_inproc_channel_returns_fifo_messages_for_destination():
    channel = InprocChannel()
    first = Message(src_rank=0, dst_rank=1, message_type="kv_prefetch")
    second = Message(src_rank=2, dst_rank=1, message_type="kv_ack")

    channel.send(first)
    channel.send(second)

    assert channel.recv(dst_rank=1) == first
    assert channel.recv(dst_rank=1) == second


def test_inproc_channel_returns_none_when_destination_queue_is_empty():
    channel = InprocChannel()

    assert channel.recv(dst_rank=1) is None


def test_inproc_channel_keeps_destination_queues_separate():
    channel = InprocChannel()
    first = Message(src_rank=0, dst_rank=1, message_type="kv_prefetch")
    second = Message(src_rank=0, dst_rank=2, message_type="kv_prefetch")

    channel.send(first)
    channel.send(second)

    assert channel.recv(dst_rank=2) == second
    assert channel.recv(dst_rank=1) == first
    assert channel.recv(dst_rank=1) is None
    assert channel.recv(dst_rank=2) is None
