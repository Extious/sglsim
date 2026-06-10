from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeEvent:
    event_type: str
    timestamp: float
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventLog:
    events: list[RuntimeEvent] = field(default_factory=list)

    def emit(self, event_type: str, timestamp: float, **fields: Any) -> RuntimeEvent:
        event = RuntimeEvent(event_type=event_type, timestamp=timestamp, fields=dict(fields))
        self.events.append(event)
        return event
