from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.synthesis_requests import cache_store
from benchmark.synthesis_requests.cache_store import (
    cache_key,
    load_prepared,
    load_workload,
    save_prepared,
    save_workload,
)
from benchmark.synthesis_requests.client import PreparedRequest
from benchmark.synthesis_requests.utils import ClientConfig, ServerConfig, WorkloadRequest


def _client(**overrides) -> ClientConfig:
    base = dict(
        dataset_name="generated-shared-prefix",
        seed=7,
        request_rate=float("inf"),
        max_concurrency=None,
        gsp_num_groups=2,
        gsp_prompts_per_group=2,
        gsp_system_prompt_len=32,
        gsp_question_len=8,
        gsp_output_len=4,
        gsp_range_ratio=1.0,
        gsp_fast_prepare=True,
        gsp_send_routing_key=False,
        gsp_num_turns=1,
        gsp_ordered=True,
    )
    base.update(overrides)
    return ClientConfig(**base)


def _server() -> ServerConfig:
    return ServerConfig(
        model_path="mock-model",
        host="127.0.0.1",
        port=30000,
        dp_size=1,
        tp_size=1,
        pp_size=1,
        raw={"model_path": "mock-model", "scheduler": {"dp_size": 1}},
    )


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cache_store, "CACHE_DIR", tmp_path / "cache")
    return cache_store.CACHE_DIR


def test_cache_key_changes_with_seed():
    assert cache_key(_client(seed=1), _server()) != cache_key(
        _client(seed=2), _server()
    )


def test_cache_key_changes_with_trace_jsonl_content(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"rid":"a","prompt":"first"}\n', encoding="utf-8")
    client = _client(
        dataset_name="claude-code-trace-jsonl",
        trace_jsonl_path=str(trace_path),
        trace_limit=1,
    )
    first_key = cache_key(client, _server())

    trace_path.write_text('{"rid":"a","prompt":"second"}\n', encoding="utf-8")

    assert cache_key(client, _server()) != first_key


def test_workload_cache_roundtrip(cache_dir):
    client = _client()
    server = _server()
    workload = [
        WorkloadRequest(
            rid="1",
            prompt="hello",
            prompt_len=2,
            output_len=4,
            group_id=0,
        )
    ]
    save_workload(client, server, workload)
    loaded = load_workload(client, server)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].rid == "1"
    assert loaded[0].prompt_len == 2


def test_prepared_cache_roundtrip(cache_dir):
    client = _client()
    server = _server()
    prepared = [
        PreparedRequest(
            rid="1",
            prompt="hello",
            prompt_len=2,
            output_len=4,
            body=b'{"model":"mock-model"}',
        )
    ]
    save_prepared(client, server, prepared)
    loaded = load_prepared(client, server)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].rid == "1"
    assert loaded[0].body == b'{"model":"mock-model"}'


def test_different_seed_cache_miss(cache_dir):
    client_a = _client(seed=1)
    client_b = _client(seed=2)
    server = _server()
    workload = [
        WorkloadRequest(
            rid="1",
            prompt="hello",
            prompt_len=2,
            output_len=4,
            group_id=0,
        )
    ]
    save_workload(client_a, server, workload)
    assert load_workload(client_b, server) is None
    assert load_prepared(client_b, server) is None
