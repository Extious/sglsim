from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYNC_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "sync_failure_pair_from_remote.py"
)
MERGE_FIG11B_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "merge_fig11b_sglang_pair.py"
)
AGGREGATE_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglang_small"
    / "scripts"
    / "aggregate_failure_visibility_pairs.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_sync_module():
    return load_module("sglang_sync_failure_pair", SYNC_SCRIPT)


def test_render_markdown_for_synced_failure_pair():
    module = load_sync_module()
    text = module.render_markdown(
        title="Unit Pair",
        summary={
            "invisible_success_count": 4,
            "visible_success_count": 4,
            "invisible_failover_response_count": 4,
            "visible_failover_response_count": 4,
            "invisible_pre_failover_backed_up_tokens": 0,
            "visible_pre_failover_backed_up_tokens": 8192,
            "frontier_token_delta": 8192,
            "visible_backed_up_tokens_per_request": 2048.0,
            "invisible_latency_p99_s": 16.0,
            "visible_latency_p99_s": 12.0,
            "visible_retry_latency_norm_vs_invisible": 0.75,
            "visible_latency_delta_pct_vs_invisible": -25.0,
        },
        invisible_job_id="1",
        visible_job_id="2",
        output_dir=module.EXP_DIR / "results" / "unit_pair",
        note="Unit note.",
        prefetch_policy="wait_complete",
    )

    assert "# Unit Pair" in text
    assert "job `1`" in text
    assert "job `2`" in text
    assert "`wait_complete`" in text
    assert "8,192" in text
    assert "0.750x" in text
    assert "-25.0%" in text


def test_sync_path_passes_custom_ssh_command_to_rsync(monkeypatch, tmp_path):
    module = load_sync_module()
    commands = []

    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/rsync")
    monkeypatch.setattr(module, "run", lambda command: commands.append(command))

    module.sync_path(
        "25482351@172.27.244.19:~/aegiskv_sglang_small/results/x/",
        tmp_path / "x",
        rsync_rsh="ssh -F /dev/null -o ProxyCommand=jump",
    )

    assert commands == [
        [
            "rsync",
            "-a",
            "-e",
            "ssh -F /dev/null -o ProxyCommand=jump",
            "25482351@172.27.244.19:~/aegiskv_sglang_small/results/x/",
            str(tmp_path / "x"),
        ]
    ]


def test_sync_path_falls_back_to_scp_with_custom_ssh(monkeypatch, tmp_path):
    module = load_sync_module()
    commands = []

    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module, "run", lambda command: commands.append(command))

    module.sync_path(
        "25482351@172.27.244.19:~/aegiskv_sglang_small/results/x/",
        tmp_path / "x",
        rsync_rsh="ssh -F /dev/null -o ProxyCommand=jump",
    )

    assert commands == [
        [
            "scp",
            "-F",
            "/dev/null",
            "-o",
            "ProxyCommand=jump",
            "-r",
            "25482351@172.27.244.19:~/aegiskv_sglang_small/results/x",
            str(tmp_path),
        ]
    ]


def test_merge_fig11b_keeps_same_write_cost_overhead_at_zero():
    module = load_module("sglang_merge_fig11b_pair", MERGE_FIG11B_SCRIPT)
    rows = module.convert_pair_rows(
        [
            {
                "policy": "SGLang remote_backup lookup disabled",
                "role": "host_invisible_remote_backup_backend_lookup_disabled",
                "retry_latency_norm": "1.0",
                "request_count": "4",
                "failover_response_count": "4",
                "pre_failover_backed_up_tokens": "20548",
                "evidence_tier": "sglang-small-paired-smoke",
                "source": "sglang_dp2_failure_visibility_pair",
                "run_id": "unit",
                "slurm_job_id": "1",
            },
            {
                "policy": "SGLang remote_backup lookup enabled",
                "role": "host_visible_remote_backup_backend_lookup_enabled",
                "retry_latency_norm": "0.84",
                "request_count": "4",
                "failover_response_count": "4",
                "pre_failover_backed_up_tokens": "20548",
                "evidence_tier": "sglang-small-paired-smoke",
                "source": "sglang_dp2_failure_visibility_pair",
                "run_id": "unit",
                "slurm_job_id": "2",
            },
        ]
    )

    assert [row["policy"] for row in rows] == [
        "SGLang remote_backup lookup disabled",
        "SGLang remote_backup lookup enabled",
    ]
    assert {row["p99_overhead_relative_pct"] for row in rows} == {0.0}
    assert rows[1]["retry_latency_norm"] == 0.84


def test_aggregate_same_write_best_effort_pairs():
    module = load_module("sglang_aggregate_failure_pairs", AGGREGATE_SCRIPT)
    rows = [
        {
            "run_id": "failure_visibility_pair_same_write_best_effort_a",
            "invisible_slurm_job_id": "10",
            "visible_slurm_job_id": "11",
            "invisible_pre_failover_backed_up_tokens": 20548,
            "visible_pre_failover_backed_up_tokens": 20548,
            "visible_retry_latency_norm_vs_invisible": 0.96,
            "visible_latency_delta_pct_vs_invisible": -4.0,
            "frontier_token_delta": 0,
            "visible_latency_p99_s": 24.0,
            "invisible_latency_p99_s": 25.0,
            "hicache_storage_prefetch_policy": "best_effort",
        },
        {
            "run_id": "failure_visibility_pair_same_write_best_effort_b",
            "invisible_slurm_job_id": "12",
            "visible_slurm_job_id": "13",
            "invisible_pre_failover_backed_up_tokens": 20548,
            "visible_pre_failover_backed_up_tokens": 20548,
            "visible_retry_latency_norm_vs_invisible": 0.94,
            "visible_latency_delta_pct_vs_invisible": -6.0,
            "frontier_token_delta": 0,
            "visible_latency_p99_s": 23.5,
            "invisible_latency_p99_s": 25.0,
            "hicache_storage_prefetch_policy": "best_effort",
        },
    ]
    for row in rows:
        row["pair_class"] = module._pair_class(row)

    aggregate = module.aggregate_rows(rows)
    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["pair_class"] == "same_write_best_effort"
    assert row["pair_count"] == 2
    assert row["visible_retry_latency_norm_mean"] == 0.95
    assert row["improving_pairs"] == 2
    assert row["invisible_job_ids"] == "10;12"
    assert row["visible_job_ids"] == "11;13"


def test_merge_fig11b_from_aggregate_rows():
    module = load_module("sglang_merge_fig11b_aggregate", MERGE_FIG11B_SCRIPT)
    rows = module.convert_aggregate_rows(
        [
            {
                "pair_class": "same_write_best_effort",
                "hicache_storage_prefetch_policy": "best_effort",
                "pair_count": "2",
                "repeat_run_ids": "a;b",
                "invisible_job_ids": "10;12",
                "visible_job_ids": "11;13",
                "visible_retry_latency_norm_mean": "0.95",
                "visible_retry_latency_norm_std": "0.014",
                "visible_retry_latency_norm_ci95": "0.02",
                "visible_backed_up_tokens_mean": "20548",
                "evidence_tier": "sglang-small-paired-smoke",
                "source": "sglang_failure_visibility_pair_aggregate",
            }
        ],
        pair_class="same_write_best_effort",
    )

    assert [row["policy"] for row in rows] == [
        "SGLang remote_backup lookup disabled",
        "SGLang remote_backup lookup enabled",
    ]
    assert rows[0]["retry_latency_norm"] == 1.0
    assert rows[1]["retry_latency_norm"] == 0.95
    assert rows[1]["retry_latency_norm_ci95"] == 0.02
    assert rows[0]["slurm_job_id"] == "10;12"
    assert rows[1]["slurm_job_id"] == "11;13"
