from __future__ import annotations

import asyncio
import json

from benchmark.synthesis_requests.client import (
    PreparedRequest,
    _failover_replay_simulation,
    _run_client_async,
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


def test_build_prepared_request_carries_aegis_policy_metadata():
    client = _client(
        gsp_num_groups=1,
        gsp_prompts_per_group=1,
        gsp_aegis_policy="host_only",
        gsp_aegis_ablation="no_lifecycle",
        gsp_aegis_gpu_budget_fraction=0.5,
        gsp_failure_after_output_tokens=3,
        gsp_backup_kv=True,
    )
    workload = [
        WorkloadRequest(
            rid="r0",
            prompt="hello",
            prompt_len=8,
            output_len=2,
            group_id=0,
            agent_id="a0",
            critical_agent=False,
        )
    ]

    prepared = build_prepared_requests_from_workload(workload, client, _server())
    payload = json.loads(prepared[0].body.decode("utf-8"))

    assert prepared[0].aegis_policy == "host_only"
    assert prepared[0].aegis_ablation == "no_lifecycle"
    assert prepared[0].aegis_gpu_budget_fraction == 0.5
    assert prepared[0].failure_after_output_tokens == 3
    assert prepared[0].backup_kv is True
    assert payload["simulation"]["aegis_policy"] == "host_only"
    assert payload["simulation"]["aegis_ablation"] == "no_lifecycle"
    assert payload["simulation"]["aegis_gpu_budget_fraction"] == 0.5
    assert payload["simulation"]["failure_after_output_tokens"] == 3
    assert payload["simulation"]["backup_kv"] is True


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
        "workflow_id": "",
        "stage_id": -1,
        "stage_width": 1,
        "stage_local_agent_index": 0,
        "decode_dominant": False,
        "workflow_shape": "",
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
        "workflow_id": "",
        "stage_id": -1,
        "stage_width": 1,
        "stage_local_agent_index": 0,
        "decode_dominant": False,
        "workflow_shape": "",
        "backup_kv": True,
    }


def test_staged_workflow_client_respects_stage_barriers(monkeypatch):
    events: list[tuple[str, str, int]] = []

    async def fake_bounded_post(
        host,
        port,
        prepared,
        semaphore,
        failover_replay,
        failover_replay_critical_only,
        failover_injection,
        failover_retry_rank_offset,
        failover_retry_policy,
        progress,
    ):
        events.append(("start", prepared.rid, prepared.stage_id))
        await asyncio.sleep(0)
        events.append(("end", prepared.rid, prepared.stage_id))
        return []

    monkeypatch.setattr(
        "benchmark.synthesis_requests.client._bounded_post",
        fake_bounded_post,
    )

    def req(rid: str, stage_id: int, local_idx: int, stage_width: int) -> PreparedRequest:
        return PreparedRequest(
            rid=rid,
            prompt="q",
            prompt_len=8,
            output_len=2,
            body=b"{}",
            model_path="mock-model",
            agent_id=rid,
            workflow_id="wf-0",
            stage_id=stage_id,
            stage_width=stage_width,
            stage_local_agent_index=local_idx,
            workflow_shape="1-3-1",
        )

    asyncio.run(
        _run_client_async(
            "127.0.0.1",
            30000,
            [
                req("s0", 0, 0, 1),
                req("s1a", 1, 0, 3),
                req("s1b", 1, 1, 3),
                req("s1c", 1, 2, 3),
                req("s2", 2, 0, 1),
            ],
            max_concurrency=None,
            failover_replay=False,
            failover_replay_critical_only=True,
            failover_injection="soft",
            failover_retry_rank_offset=1,
            failover_retry_policy="",
        )
    )

    positions = {event: idx for idx, event in enumerate(events)}
    assert positions[("start", "s0", 0)] < positions[("end", "s0", 0)]
    assert all(
        positions[("start", rid, 1)] > positions[("end", "s0", 0)]
        for rid in ("s1a", "s1b", "s1c")
    )
    assert max(
        positions[("end", rid, 1)] for rid in ("s1a", "s1b", "s1c")
    ) < positions[("start", "s2", 2)]
    assert max(
        positions[("start", rid, 1)] for rid in ("s1a", "s1b", "s1c")
    ) < min(positions[("end", rid, 1)] for rid in ("s1a", "s1b", "s1c"))


def test_failover_replay_retry_rank_offset_defaults_to_peer_rank():
    prepared = PreparedRequest(
        rid="7",
        prompt="q",
        prompt_len=10,
        output_len=2,
        body=b"",
        model_path="mock-model",
        agent_id="7",
        critical_agent=True,
        dp_size=3,
    )

    sim = _failover_replay_simulation(
        prepared,
        turn_idx=0,
        sim_stats={
            "aegis_candidate_failed_dp_rank": 1,
            "aegis_candidate_pre_failure_output_tokens": 1,
            "aegis_candidate_pre_failure_total_tokens": 32,
        },
    )

    assert sim["retry_dp_rank"] == 2
    assert sim["force_dp_rank"] == 2


def test_failover_replay_preserves_explicit_policy():
    prepared = PreparedRequest(
        rid="7",
        prompt="q",
        prompt_len=10,
        output_len=2,
        body=b"",
        model_path="mock-model",
        agent_id="7",
        critical_agent=False,
        aegis_policy="gpu_replica",
        aegis_ablation="no_pressure",
        aegis_gpu_budget_fraction=0.25,
        failure_after_output_tokens=5,
        backup_kv=True,
        dp_size=4,
    )

    sim = _failover_replay_simulation(
        prepared,
        turn_idx=0,
        sim_stats={
            "aegis_candidate_failed_dp_rank": 1,
            "aegis_candidate_pre_failure_output_tokens": 1,
            "aegis_candidate_pre_failure_total_tokens": 32,
        },
    )

    assert sim["aegis_policy"] == "gpu_replica"
    assert sim["aegis_ablation"] == "no_pressure"
    assert sim["aegis_gpu_budget_fraction"] == 0.25
    assert sim["failure_after_output_tokens"] == 1
    assert sim["backup_kv"] is True


def test_failover_replay_retry_rank_offset_can_target_non_peer_rank():
    prepared = PreparedRequest(
        rid="7",
        prompt="q",
        prompt_len=10,
        output_len=2,
        body=b"",
        model_path="mock-model",
        agent_id="7",
        critical_agent=True,
        dp_size=3,
    )

    sim = _failover_replay_simulation(
        prepared,
        turn_idx=0,
        sim_stats={
            "aegis_candidate_failed_dp_rank": 1,
            "aegis_candidate_pre_failure_output_tokens": 1,
            "aegis_candidate_pre_failure_total_tokens": 32,
        },
        retry_rank_offset=2,
    )

    assert sim["retry_dp_rank"] == 0
    assert sim["force_dp_rank"] == 0


def test_failover_replay_retry_policy_can_force_non_peer_rank():
    prepared = PreparedRequest(
        rid="7",
        prompt="q",
        prompt_len=10,
        output_len=2,
        body=b"",
        model_path="mock-model",
        agent_id="7",
        critical_agent=True,
        dp_size=4,
    )

    sim = _failover_replay_simulation(
        prepared,
        turn_idx=0,
        sim_stats={
            "aegis_candidate_failed_dp_rank": 1,
            "aegis_candidate_pre_failure_output_tokens": 1,
            "aegis_candidate_pre_failure_total_tokens": 32,
        },
        retry_policy="non_peer",
    )

    assert sim["retry_dp_rank"] == 3
    assert sim["force_dp_rank"] == 3
    assert sim["retry_route_policy"] == "non_peer"


def test_failover_replay_retry_policy_random_healthy_avoids_failed_rank():
    prepared = PreparedRequest(
        rid="7",
        prompt="q",
        prompt_len=10,
        output_len=2,
        body=b"",
        model_path="mock-model",
        agent_id="7",
        critical_agent=True,
        dp_size=4,
    )

    sim = _failover_replay_simulation(
        prepared,
        turn_idx=2,
        sim_stats={
            "aegis_candidate_failed_dp_rank": 1,
            "aegis_candidate_pre_failure_output_tokens": 1,
            "aegis_candidate_pre_failure_total_tokens": 32,
        },
        retry_policy="random_healthy",
    )

    assert sim["retry_dp_rank"] in {0, 2, 3}
    assert sim["force_dp_rank"] == sim["retry_dp_rank"]
    assert sim["retry_route_policy"] == "random_healthy"
