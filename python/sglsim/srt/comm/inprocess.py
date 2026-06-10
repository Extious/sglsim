from __future__ import annotations

from collections import defaultdict, deque

from sglsim.srt.comm.messages import Message


class InprocChannel:
    def __init__(self) -> None:
        self._queues: dict[int, deque[Message]] = defaultdict(deque)

    def send(self, message: Message) -> None:
        self._queues[message.dst_rank].append(message)

    def recv(self, dst_rank: int) -> Message | None:
        queue = self._queues[dst_rank]
        if not queue:
            return None
        return queue.popleft()
