from __future__ import annotations

import json

from benchmark.synthesis_requests.client import (
    PreparedRequest,
    build_prepared_requests_from_workload,
    run_multi_turn_session_sync,
)
from benchmark.synthesis_requests.utils import (
    ClientConfig,
    ServerConfig,
    WorkloadRequest,
    generate_shared_prefix_workload,
)
from test.unit.benchmark.test_gsp_workload import _client, _server


def test_build_prepared_multi_turn_session():
    client = _client(
        gsp_num_groups=1,
        gsp_prompts_per_group=1,
        gsp_num_turns=2,
        gsp_fast_prepare=False,
        gsp_ordered=True,
    )
    workload = generate_shared_prefix_workload(client, _server())
    assert len(workload) == 1
    assert isinstance(workload[0].prompt, list)
    assert len(workload[0].prompt) == 2

    prepared = build_prepared_requests_from_workload(workload, client, _server())
    assert len(prepared) == 1
    assert prepared[0].is_multi_turn
    assert prepared[0].turn_prompts is not None
    assert prepared[0].body == b""


def test_multi_turn_session_posts_chat_per_turn(monkeypatch):
    calls: list[dict] = []

    def fake_post(host, port, *, body, routing_key, timeout_s=3600.0):
        payload = json.loads(body.decode("utf-8"))
        calls.append(payload)
        turn_idx = len(calls) - 1
        return (
            {
                "choices": [{"message": {"content": f"reply-{turn_idx}"}}],
                "sim_stats": {"status": "ok", "output_length": 2},
            },
            0.01,
        )

    monkeypatch.setattr(
        "benchmark.synthesis_requests.client.post_chat_completions_sync",
        fake_post,
    )

    prepared = PreparedRequest(
        rid="7",
        prompt="sys\n\nq0",
        prompt_len=10,
        output_len=2,
        body=b"",
        turn_prompts=("sys\n\nq0", "q1"),
        model_path="mock-model",
        agent_id="7",
        critical_agent=True,
    )
    outputs = run_multi_turn_session_sync("127.0.0.1", 30000, prepared)
    assert len(outputs) == 2
    assert outputs[0][0] == "7-t0"
    assert outputs[1][0] == "7-t1"
    assert len(calls) == 2
    assert calls[0]["messages"] == [{"role": "user", "content": "sys\n\nq0"}]
    assert calls[0]["simulation"] == {
        "parallel_worker_client": True,
        "worker_id": "7",
        "worker_seq": 0,
        "agent_id": "7",
        "critical_agent": True,
        "backup_kv": True,
    }
    assert calls[1]["messages"] == [
        {"role": "user", "content": "sys\n\nq0"},
        {"role": "assistant", "content": "reply-0"},
        {"role": "user", "content": "q1"},
    ]
    assert calls[1]["simulation"] == {
        "parallel_worker_client": True,
        "worker_id": "7",
        "worker_seq": 1,
        "agent_id": "7",
        "critical_agent": True,
        "backup_kv": True,
    }
