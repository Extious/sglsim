from fastapi.testclient import TestClient

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.entrypoints.http_server import create_app
from sglsim.srt.server_args import ServerArgs


def test_completions_prompt_tokenizes():
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
        "/v1/completions",
        json={
            "model": "mock-model",
            "prompt": "hello world",
            "max_tokens": 1,
            "user": "1",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["usage"]["prompt_tokens"] > 0
    engine.stop_background_scheduler()
