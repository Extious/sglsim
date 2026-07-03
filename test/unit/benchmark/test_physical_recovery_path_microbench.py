from __future__ import annotations

from AegisKV.draft.experiments.sglang_small.scripts.aggregate_physical_recovery_path_microbench import (
    aggregate_rows,
    render_markdown as render_aggregate_markdown,
)
from AegisKV.draft.experiments.sglang_small.scripts.physical_recovery_path_microbench import (
    parse_token_counts,
    render_markdown,
    summarize_samples,
)


def test_summarize_recovery_action_reports_restore_metadata() -> None:
    row = summarize_samples(
        action="gpu_replica_restore_p2p",
        recovery_path="GPUReplica",
        phase="restore",
        copy_path="p2p_gpu0_to_gpu1",
        token_count=2048,
        byte_count=2048 * 147456,
        samples_ms=[2.0, 4.0, 6.0],
        available=True,
        source_location="surviving_peer_gpu_replica",
        target_location="retry_rank_gpu_kv",
        source_device=0,
        target_device=1,
        note="unit",
    )

    assert row["action"] == "gpu_replica_restore_p2p"
    assert row["recovery_path"] == "GPUReplica"
    assert row["is_restore_action"] == "true"
    assert row["p50_ms"] == 4.0
    assert row["p95_ms"] > row["p50_ms"]
    assert row["p50_gbps"] > 0


def test_parse_recovery_token_counts_rejects_empty_values() -> None:
    assert parse_token_counts("512, 1024,2048") == [512, 1024, 2048]

    try:
        parse_token_counts("")
    except Exception as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty token count list should fail")


def test_recovery_markdown_states_boundary() -> None:
    row = summarize_samples(
        action="host_restore_h2d",
        recovery_path="HostBackup",
        phase="restore",
        copy_path="h2d_pinned",
        token_count=512,
        byte_count=512 * 147456,
        samples_ms=[1.5],
        available=True,
        source_location="pinned_host_backup",
        target_location="retry_rank_gpu_kv",
    )
    text = render_markdown(
        [row],
        {
            "host": "unit-node",
            "slurm_job_id": "1",
            "cuda_visible_devices": "0,1",
            "torch_version": "unit",
            "torch_cuda_device_count": 2,
            "source_device": 0,
            "target_device": 1,
            "bytes_per_token": 147456,
            "repeats": 1,
            "warmup": 0,
        },
    )

    assert "Physical Recovery Path Microbenchmark" in text
    assert "host_restore_h2d" in text
    assert "not final end-to-end AegisKV policy evidence" in text


def test_aggregate_physical_recovery_rows_groups_by_action_and_tokens() -> None:
    aggregate = aggregate_rows(
        [
            {
                "action": "host_restore_h2d",
                "recovery_path": "HostBackup",
                "phase": "restore",
                "copy_path": "h2d_pinned",
                "tokens": "8192",
                "bytes": "1207959552",
                "mib": "1152",
                "available": "true",
                "p50_ms": "48.0",
                "p95_ms": "50.0",
                "p50_gbps": "25",
                "slurm_job_id": "1",
                "host": "a",
                "bytes_per_token": "147456",
            },
            {
                "action": "host_restore_h2d",
                "recovery_path": "HostBackup",
                "phase": "restore",
                "copy_path": "h2d_pinned",
                "tokens": "8192",
                "bytes": "1207959552",
                "mib": "1152",
                "available": "true",
                "p50_ms": "50.0",
                "p95_ms": "52.0",
                "p50_gbps": "24",
                "slurm_job_id": "2",
                "host": "b",
                "bytes_per_token": "147456",
            },
            {
                "action": "host_restore_h2d",
                "recovery_path": "HostBackup",
                "phase": "restore",
                "copy_path": "h2d_pinned",
                "tokens": "8192",
                "available": "false",
                "p50_ms": "100",
                "slurm_job_id": "3",
            },
        ]
    )

    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["action"] == "host_restore_h2d"
    assert row["repeat_count"] == 2
    assert row["slurm_job_ids"] == "1;2"
    assert row["p50_ms_mean"] == 49.0
    assert row["p50_ms_ci95"] > 0


def test_aggregate_markdown_states_runtime_boundary() -> None:
    text = render_aggregate_markdown(
        [{"slurm_job_id": "1", "host": "a"}],
        [
            {
                "action": "gpu_replica_restore_p2p",
                "tokens": 8192,
                "phase": "restore",
                "repeat_count": 1,
                "p50_ms_mean": 4.0,
                "p50_ms_ci95": 0.0,
                "p50_gbps_mean": 300.0,
                "slurm_job_ids": "1",
            }
        ],
    )

    assert "Physical Recovery Path Microbenchmark Aggregate" in text
    assert "not final end-to-end policy evidence" in text
