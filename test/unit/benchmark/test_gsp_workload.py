from __future__ import annotations

import json
from pathlib import Path

from benchmark.synthesis_requests.utils import (
    ClientConfig,
    ServerConfig,
    generate_shared_prefix_workload,
    load_client_config,
    load_config_dir,
    parse_request_rate,
    require_request_rate_inf,
)


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


def test_parse_request_rate():
    assert parse_request_rate("inf") == float("inf")
    assert parse_request_rate(None) == float("inf")


def test_require_request_rate_inf():
    require_request_rate_inf(float("inf"))
    try:
        require_request_rate_inf(10.0)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "request_rate must be inf" in str(exc)


def test_load_client_config_rejects_finite_request_rate(tmp_path: Path):
    path = tmp_path / "client.json"
    path.write_text(
        json.dumps(
            {
                "dataset_name": "generated-shared-prefix",
                "gsp_num_groups": 1,
                "gsp_prompts_per_group": 1,
                "gsp_system_prompt_len": 16,
                "gsp_question_len": 4,
                "gsp_output_len": 2,
                "request_rate": 10,
            }
        ),
        encoding="utf-8",
    )
    try:
        load_client_config(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "request_rate must be inf" in str(exc)


def test_generate_shared_prefix_workload_mock():
    client = _client()
    requests = generate_shared_prefix_workload(client, _server())
    assert len(requests) == 4
    by_group: dict[int, set[str]] = {}
    for req in requests:
        prefix = req.first_turn_prompt.split("\n\n", 1)[0]
        by_group.setdefault(req.group_id, set()).add(prefix)
    assert len(by_group) == 2
    assert len(by_group[0]) == 1
    assert all(r.prompt_len > 0 for r in requests)


def test_load_client_config_accepts_aegis_policy_fields(tmp_path: Path):
    path = tmp_path / "client.json"
    path.write_text(
        json.dumps(
            {
                "dataset_name": "generated-shared-prefix",
                "seed": 1,
                "request_rate": "inf",
                "gsp_num_groups": 1,
                "gsp_prompts_per_group": 1,
                "gsp_system_prompt_len": 16,
                "gsp_question_len": 4,
                "gsp_output_len": 2,
                "gsp_failover_retry_policy": "random_healthy",
                "gsp_aegis_policy": "host_only",
                "gsp_aegis_ablation": "no_lifecycle",
                "gsp_aegis_gpu_budget_fraction": 0.5,
                "gsp_failure_after_output_tokens": 4,
                "gsp_backup_kv": True,
            }
        ),
        encoding="utf-8",
    )

    client = load_client_config(path)

    assert client.gsp_aegis_policy == "host_only"
    assert client.gsp_aegis_ablation == "no_lifecycle"
    assert client.gsp_aegis_gpu_budget_fraction == 0.5
    assert client.gsp_failure_after_output_tokens == 4
    assert client.gsp_failover_retry_policy == "random_healthy"
    assert client.gsp_backup_kv is True


def test_gsp_fast_prepare_prompt_len_is_one():
    client = _client(gsp_fast_prepare=True)
    requests = generate_shared_prefix_workload(client, _server())
    assert all(r.prompt_len == 1 for r in requests)


def test_prompt_len_matches_mock_encode():
    from sglsim.srt.utils.tokenizer_runtime import encode_text

    client = _client(gsp_fast_prepare=False)
    requests = generate_shared_prefix_workload(client, _server())
    for req in requests:
        expected = max(len(encode_text("mock-model", req.first_turn_prompt)), 1)
        assert req.prompt_len == expected


def test_num_prompts_truncates():
    client = _client(num_prompts=2)
    requests = generate_shared_prefix_workload(client, _server())
    assert len(requests) == 2


def test_generate_staged_13131_workload_marks_critical_long_decode_agents():
    client = _client(
        gsp_num_groups=1,
        gsp_prompts_per_group=9,
        gsp_workflow_shape=(1, 3, 1, 3, 1),
        gsp_stage_long_decode_agent_index=0,
        gsp_long_decode_output_multiplier=5.0,
        gsp_output_len=10,
        gsp_range_ratio=1.0,
        gsp_fast_prepare=False,
        gsp_ordered=True,
    )

    requests = generate_shared_prefix_workload(client, _server())

    assert len(requests) == 9
    assert [r.stage_width for r in requests] == [1, 3, 3, 3, 1, 3, 3, 3, 1]
    assert [r.stage_id for r in requests] == [0, 1, 1, 1, 2, 3, 3, 3, 4]
    assert [r.critical_agent for r in requests] == [
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
    ]
    assert [r.decode_dominant for r in requests] == [
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert [r.output_len for r in requests] == [10, 50, 10, 10, 10, 50, 10, 10, 10]
    assert {r.workflow_id for r in requests} == {"wf-0"}
    assert {r.workflow_shape for r in requests} == {"1-3-1-3-1"}


def test_generate_staged_workload_uses_stage_context_lens_for_first_turn():
    from sglsim.srt.utils.tokenizer_runtime import encode_text

    client = _client(
        gsp_num_groups=1,
        gsp_prompts_per_group=9,
        gsp_workflow_shape=(1, 3, 1, 3, 1),
        gsp_stage_context_lens=(40, 40, 80, 40, 120),
        gsp_stage_long_decode_agent_index=0,
        gsp_output_len=10,
        gsp_range_ratio=1.0,
        gsp_fast_prepare=False,
        gsp_ordered=True,
        gsp_num_turns=3,
    )

    requests = generate_shared_prefix_workload(client, _server())

    assert len(requests) == 9
    assert [r.stage_id for r in requests] == [0, 1, 1, 1, 2, 3, 3, 3, 4]
    assert [len(encode_text("mock-model", r.first_turn_prompt)) for r in requests] == [
        40,
        40,
        40,
        40,
        80,
        40,
        40,
        40,
        120,
    ]
    assert [r.agent_id for r in requests] == [
        "wf-0-main",
        "wf-0-s1-a0",
        "wf-0-s1-a1",
        "wf-0-s1-a2",
        "wf-0-main",
        "wf-0-s3-a0",
        "wf-0-s3-a1",
        "wf-0-s3-a2",
        "wf-0-main",
    ]
    assert [r.critical_agent for r in requests] == [
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
    ]


def test_generate_staged_workload_rejects_mismatched_stage_context_lens():
    client = _client(
        gsp_num_groups=1,
        gsp_prompts_per_group=9,
        gsp_workflow_shape=(1, 3, 1, 3, 1),
        gsp_stage_context_lens=(40, 80),
        gsp_ordered=True,
    )

    try:
        generate_shared_prefix_workload(client, _server())
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "gsp_stage_context_lens" in str(exc)


def test_load_config_dir(tmp_path: Path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "client.json").write_text(
        json.dumps(
            {
                "dataset_name": "generated-shared-prefix",
                "gsp_num_groups": 1,
                "gsp_prompts_per_group": 1,
                "gsp_system_prompt_len": 16,
                "gsp_question_len": 4,
                "gsp_output_len": 2,
                "request_rate": "inf",
            }
        ),
        encoding="utf-8",
    )
    (cfg / "server.json").write_text(
        json.dumps(
            {
                "model_path": "mock-model",
                "scheduler": {"dp_size": 1, "tp_size": 1, "pp_size": 1},
            }
        ),
        encoding="utf-8",
    )
    client, server = load_config_dir(cfg)
    assert client.num_requests == 1
    assert client.request_rate == float("inf")
    assert server.model_path == "mock-model"
