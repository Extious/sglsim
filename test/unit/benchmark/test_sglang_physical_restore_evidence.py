from __future__ import annotations

import json
from pathlib import Path

from AegisKV.draft.experiments.sglang_small.scripts.extract_physical_restore_evidence import (
    parse_server_log,
    summarize_result_dir,
)


def _write_result(
    tmp_path: Path,
    *,
    name: str,
    log: str,
    backed_up_tokens: int = 20548,
) -> Path:
    result_dir = tmp_path / name
    result_dir.mkdir()
    (result_dir / "server.log").write_text(log, encoding="utf-8")
    (result_dir / "provenance.txt").write_text(
        "\n".join(
            [
                "slurm_job_id=12345",
                "host=hkbugpusrv11",
                "model_path=Qwen/Qwen3-8B",
                "enable_remote_backup=1",
                "disable_failover_remote_lookup=0",
                "hicache_storage_prefetch_policy=wait_complete",
                "prompt_repeat=256",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (result_dir / "failure_summary.json").write_text(
        json.dumps(
            {
                "count": 4,
                "success_count": 4,
                "failover_response_count": 4,
                "captured_request_count": 4,
                "dispatched_request_count": 4,
                "latency_p95_s": 20.0,
                "latency_p99_s": 21.0,
                "response_pre_failover_output_tokens": 748,
                "response_pre_failover_backed_up_tokens": backed_up_tokens,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return result_dir


def test_parse_full_prefetch_materialized(tmp_path: Path) -> None:
    log = """
[x DP1] _storage_hit_query_tokens: req=aegiskv-failure-smoke-0000, prefix_pages=0, data_pages=5137, range=5137, tokens_to_fetch=5312, boundary_token=81917, total_tokens=5312
[x DP1] Prefetch ACCEPTED for request aegiskv-failure-smoke-0000: 5137 pages (5137 tokens) from storage backend
[x DP1] Prefetch IO START for aegiskv-failure-smoke-0000: pages=5137 tokens=5137
[x DP1] Prefetch IO END for aegiskv-failure-smoke-0000: completed=5137 / expected=5137 tokens
[x DP1] Prefill batch, #new-seq: 1, #new-token: 175, #cached-token: 5137, token usage: 0.01
"""
    result = parse_server_log(_write_result(tmp_path, name="full", log=log) / "server.log")
    assert result["max_data_pages"] == 5137
    assert result["prefetch_io_completion_ratio"] == 1.0
    assert result["retry_cached_tokens_after_accept_total"] == 5137

    row = summarize_result_dir(tmp_path / "full")
    assert row["physical_restore_state"] == "full_prefetch_materialized"


def test_parse_partial_async_materialized(tmp_path: Path) -> None:
    log = """
[x DP1] _storage_hit_query_tokens: req=aegiskv-failure-smoke-0002, prefix_pages=0, data_pages=5137, range=5137, tokens_to_fetch=5324, boundary_token=81917, total_tokens=5324
[x DP1] Prefetch ACCEPTED for request aegiskv-failure-smoke-0002: 5137 pages (5137 tokens) from storage backend
[x DP1] Prefetch IO START for aegiskv-failure-smoke-0002: pages=5137 tokens=5137
[x DP1] Prefetch IO END for aegiskv-failure-smoke-0002: completed=2816 / expected=5137 tokens
[x DP1] Prefill batch, #new-seq: 3, #new-token: 8192, #cached-token: 2830, token usage: 0.05
"""
    row = summarize_result_dir(
        _write_result(tmp_path, name="partial", log=log)
    )
    assert row["physical_restore_state"] == "partial_async_materialized"
    assert row["prefetch_io_completed_tokens_sum"] == 2816
    assert row["retry_cached_tokens_after_accept_max"] == 2830


def test_parse_lookup_disabled_control(tmp_path: Path) -> None:
    log = """
[x DP0] Prefill batch, #new-seq: 2, #new-token: 8192, #cached-token: 28, token usage: 0.04
[x DP1] Prefill batch, #new-seq: 2, #new-token: 8192, #cached-token: 0, token usage: 0.02
"""
    row = summarize_result_dir(
        _write_result(
            tmp_path,
            name="control",
            log=log,
            backed_up_tokens=20548,
        )
    )
    assert row["physical_restore_state"] == "host_frontier_not_visible_to_retry"
    assert row["retry_cached_tokens_after_accept_total"] == 0
