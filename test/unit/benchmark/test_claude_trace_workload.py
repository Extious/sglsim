from __future__ import annotations

import json

from benchmark.agentic_workflows.claude_trace import build_workload
from benchmark.synthesis_requests.utils import (
    ClientConfig,
    ServerConfig,
    generate_claude_trace_workload,
)


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


def _trace_client(path: str) -> ClientConfig:
    return ClientConfig(
        dataset_name="claude-code-trace-jsonl",
        seed=1,
        request_rate=float("inf"),
        max_concurrency=1,
        gsp_num_groups=1,
        gsp_prompts_per_group=1,
        gsp_system_prompt_len=0,
        gsp_question_len=0,
        gsp_output_len=64,
        gsp_range_ratio=1.0,
        gsp_fast_prepare=False,
        gsp_send_routing_key=False,
        gsp_num_turns=1,
        gsp_ordered=True,
        trace_jsonl_path=path,
        trace_limit=1,
    )


def test_build_workload_from_hf_style_claude_trace_row():
    rows = build_workload(
        [
            {
                "id": "row-1",
                "request_id": "req-1",
                "source_repo": "archit11/claude-code-traces",
                "source_table": "find_hf_claude_code_archit11_claude-code-traces",
                "system_prompt": "You are Claude Code.",
                "messages_json": json.dumps(
                    [
                        {"role": "user", "content": "Inspect the repo."},
                        {"role": "assistant", "content": "I will list files."},
                        {"role": "user", "content": "Go deeper."},
                    ]
                ),
                "tools_json": json.dumps([{"name": "Task"}]),
            }
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["rid"] == "req-1"
    assert row["critical_agent"] is True
    assert row["decode_dominant"] is True
    assert row["workflow_shape"] == "real-claude-code-trace"
    assert row["source_table"] == "find_hf_claude_code_archit11_claude-code-traces"
    assert row["source_dataset"] == "nlile/misc-merged-claude-code-traces-v1"
    assert row["turn_prompts"][0].startswith("You are Claude Code.")
    assert row["turn_prompts"][1] == "Go deeper."


def test_generate_claude_trace_workload_reads_jsonl(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(
            {
                "rid": "req-1",
                "turn_prompts": ["first", "second"],
                "output_len": 32,
                "critical_agent": True,
                "workflow_id": "wf",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    requests = generate_claude_trace_workload(_trace_client(str(path)), _server())

    assert len(requests) == 1
    assert requests[0].rid == "req-1"
    assert requests[0].prompt == ["first", "second"]
    assert requests[0].output_len == 32
    assert requests[0].critical_agent is True


def test_build_workload_can_filter_unserveable_long_prompts():
    rows = build_workload(
        [
            {
                "request_id": "long",
                "user_prompt": " ".join(["token"] * 6),
            },
            {
                "request_id": "short",
                "user_prompt": "short prompt",
            },
        ],
        max_input_words=5,
    )

    assert [row["rid"] for row in rows] == ["short"]
