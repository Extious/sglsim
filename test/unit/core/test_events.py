from sglsim.srt.simulation.events import EventLog, RuntimeEvent


def test_event_log_records_ordered_events():
    log = EventLog()
    log.emit("request_arrived", timestamp=0.1, rid="r0")
    log.emit("batch_started", timestamp=0.2, batch_id=1)

    assert [event.event_type for event in log.events] == [
        "request_arrived",
        "batch_started",
    ]


def test_event_log_emit_returns_event_with_payload_fields():
    log = EventLog()

    event = log.emit("request_arrived", timestamp=0.1, rid="r0", priority=2)

    assert isinstance(event, RuntimeEvent)
    assert event.fields == {"rid": "r0", "priority": 2}
    assert log.events == [event]
