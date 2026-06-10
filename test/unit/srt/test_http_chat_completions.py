from fastapi.testclient import TestClient

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.entrypoints.http_server import create_app
from sglsim.srt.server_args import ServerArgs


def test_chat_completions_applies_messages():
    engine = Engine(
        ServerArgs(
            streaming_arrival=True,
            model_path="mock-model",
        )
    )
    engine.start_background_scheduler()
    app = create_app(engine)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
            "max_completion_tokens": 1,
            "user": "mt-1",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert "sim_stats" in payload
    engine.stop_background_scheduler()
