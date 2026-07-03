from __future__ import annotations

from AegisKV.draft.experiments.sglang_small.scripts.physical_kv_copy_microbench import (
    parse_token_counts,
    render_markdown,
    summarize_samples,
)
from AegisKV.draft.experiments.sglang_small.scripts.aggregate_physical_kv_copy_microbench import (
    aggregate_rows,
)


def test_summarize_samples_reports_latency_and_bandwidth() -> None:
    row = summarize_samples(
        path_name="p2p_gpu0_to_gpu1",
        token_count=1024,
        byte_count=1024 * 147456,
        samples_ms=[1.0, 2.0, 3.0],
        available=True,
        note="unit",
    )

    assert row["path"] == "p2p_gpu0_to_gpu1"
    assert row["tokens"] == 1024
    assert row["samples"] == 3
    assert row["p50_ms"] == 2.0
    assert row["p95_ms"] > row["p50_ms"]
    assert row["p50_gbps"] > 0
    assert row["available"] == "true"


def test_parse_token_counts_rejects_empty_values() -> None:
    assert parse_token_counts("128, 256,512") == [128, 256, 512]

    try:
        parse_token_counts("")
    except Exception as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty token count list should fail")


def test_render_markdown_includes_calibration_boundary() -> None:
    row = summarize_samples(
        path_name="h2d_pinned",
        token_count=128,
        byte_count=128 * 147456,
        samples_ms=[0.5],
        available=True,
    )
    text = render_markdown(
        [row],
        {
            "host": "unit-node",
            "slurm_job_id": "1",
            "cuda_visible_devices": "0,1",
            "torch_version": "unit",
            "torch_cuda_device_count": 2,
            "bytes_per_token": 147456,
            "repeats": 1,
            "warmup": 0,
        },
    )

    assert "Physical KV Copy Microbenchmark" in text
    assert "h2d_pinned" in text
    assert "not final end-to-end AegisKV policy evidence" in text


def test_aggregate_physical_copy_rows_groups_by_path_and_tokens() -> None:
    aggregate = aggregate_rows(
        [
            {
                "path": "p2p_gpu0_to_gpu1",
                "tokens": "8192",
                "bytes": "1207959552",
                "mib": "1152",
                "available": "true",
                "p50_ms": "4.0",
                "p95_ms": "4.4",
                "p50_gbps": "300",
                "slurm_job_id": "1",
                "host": "a",
                "bytes_per_token": "147456",
            },
            {
                "path": "p2p_gpu0_to_gpu1",
                "tokens": "8192",
                "bytes": "1207959552",
                "mib": "1152",
                "available": "true",
                "p50_ms": "6.0",
                "p95_ms": "6.6",
                "p50_gbps": "200",
                "slurm_job_id": "2",
                "host": "b",
                "bytes_per_token": "147456",
            },
            {
                "path": "p2p_gpu0_to_gpu1",
                "tokens": "8192",
                "available": "false",
                "p50_ms": "100",
                "slurm_job_id": "3",
            },
        ]
    )

    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["path"] == "p2p_gpu0_to_gpu1"
    assert row["repeat_count"] == 2
    assert row["slurm_job_ids"] == "1;2"
    assert row["p50_ms_mean"] == 5.0
    assert row["p50_ms_ci95"] > 0
