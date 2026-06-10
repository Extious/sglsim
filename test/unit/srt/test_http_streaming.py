from fastapi.testclient import TestClient

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.entrypoints.http_server import create_app
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.types import RequestStats


class _FakeHttpEngine:
    def __init__(self) -> None:
        self.server_args = ServerArgs(model_path="mock-model")
        self.submitted_routing_key: str | None = None

    def is_ready(self) -> bool:
        return True

    def generate_from_http(self, req):
        self.submitted_routing_key = req.routing_key
        return RequestStats(rid=req.rid, input_length=2, output_length=1), ""


def test_completions_preserves_routing_key_header():
    engine = _FakeHttpEngine()
    app = create_app(engine)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        headers={"X-SMG-Routing-Key": "group-a"},
        json={
            "model": "mock-model",
            "input_len": 2,
            "max_tokens": 1,
            "user": "routing-header",
        },
    )

    assert response.status_code == 200
    assert engine.submitted_routing_key == "group-a"


def test_completions_blocks_until_rid_done():
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
            "prompt": "hello",
            "max_tokens": 2,
            "user": "1",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "1"
    engine.stop_background_scheduler()


def test_completions_uses_client_input_ids():
    engine = Engine(
        ServerArgs(
            streaming_arrival=True,
            model_path="mock-model",
        )
    )
    engine.start_background_scheduler()
    app = create_app(engine)
    client = TestClient(app)
    input_ids = [10, 11, 12]
    response = client.post(
        "/v1/completions",
        json={
            "model": "mock-model",
            "prompt": "unused",
            "input_ids": input_ids,
            "max_tokens": 1,
            "user": "9",
        },
    )
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == len(input_ids)
    engine.stop_background_scheduler()


def test_completions_emits_synthetic_access_log(capsys):
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
            "prompt": "hello",
            "max_tokens": 1,
            "user": "synthetic-access",
        },
    )

    assert response.status_code == 200
    captured = capsys.readouterr()
    assert "Completed /v1/completions rid=synthetic-access output_tokens=1" in captured.out
    assert 'INFO:     127.0.0.1:30000 - "POST /v1/completions HTTP/1.1" 200 OK' in captured.out
    engine.stop_background_scheduler()


def test_completions_reports_generated_tokens_separately():
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
            "prompt": "hello",
            "max_tokens": 2,
            "user": "generated-token-stats",
        },
    )

    assert response.status_code == 200
    sim_stats = response.json()["sim_stats"]
    assert sim_stats["output_length"] == 2
    assert sim_stats["generated_tokens"] == 2
    engine.stop_background_scheduler()
