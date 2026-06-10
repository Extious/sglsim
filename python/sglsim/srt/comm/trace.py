from __future__ import annotations

from dataclasses import dataclass, field

from sglsim.srt.comm.messages import Message


@dataclass
class TraceEvent:
    timestamp: float
    event_type: str
    fields: dict = field(default_factory=dict)


class MessageTrace:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record_send(self, message: Message, timestamp: float) -> None:
        self.events.append(
            TraceEvent(
                timestamp=timestamp,
                event_type="send",
                fields={
                    "src_rank": message.src_rank,
                    "dst_rank": message.dst_rank,
                    "message_type": message.message_type,
                },
            )
        )

    def record_recv(self, message: Message, timestamp: float) -> None:
        self.events.append(
            TraceEvent(
                timestamp=timestamp,
                event_type="recv",
                fields={
                    "src_rank": message.src_rank,
                    "dst_rank": message.dst_rank,
                    "message_type": message.message_type,
                },
            )
        )
