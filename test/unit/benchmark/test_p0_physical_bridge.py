from __future__ import annotations

import csv
from pathlib import Path

from AegisKV.draft.experiments.sglsim_large.scripts.build_p0_physical_bridge import (
    build_bridge_rows,
    summarize_bridge,
)


def _write_physical_aggregate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "action",
        "phase",
        "tokens",
        "p50_ms_mean",
        "p95_ms_mean",
        "p50_ms_ci95",
        "evidence_tier",
        "source",
        "slurm_job_ids",
    ]
    rows = [
        {
            "action": "host_restore_h2d",
            "phase": "restore",
            "tokens": "512",
            "p50_ms_mean": "3.0",
            "p95_ms_mean": "3.2",
            "p50_ms_ci95": "0.1",
            "evidence_tier": "physical",
            "source": "microbench",
            "slurm_job_ids": "1;2;3",
        },
        {
            "action": "gpu_replica_restore_p2p",
            "phase": "restore",
            "tokens": "1024",
            "p50_ms_mean": "0.5",
            "p95_ms_mean": "0.6",
            "p50_ms_ci95": "0.01",
            "evidence_tier": "physical",
            "source": "microbench",
            "slurm_job_ids": "1;2;3",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_request_details(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rid",
        "aegis_failover_replay",
        "aegis_replay_restore_action",
        "aegis_replay_restored_tokens",
        "aegis_replay_restore_data_plane",
        "aegis_replay_restore_timing_source",
    ]
    rows = [
        {
            "rid": "host-retry",
            "aegis_failover_replay": "true",
            "aegis_replay_restore_action": "host_restore",
            "aegis_replay_restored_tokens": "300",
            "aegis_replay_restore_data_plane": "metadata_materialization",
            "aegis_replay_restore_timing_source": "physical_recovery_path_csv",
        },
        {
            "rid": "gpu-retry",
            "aegis_failover_replay": "true",
            "aegis_replay_restore_action": "gpu_replica",
            "aegis_replay_restored_tokens": "800",
            "aegis_replay_restore_data_plane": "metadata_materialization",
            "aegis_replay_restore_timing_source": "physical_recovery_path_csv",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_build_p0_bridge_maps_restore_rows_to_physical_token_buckets(tmp_path: Path) -> None:
    physical = tmp_path / "physical.csv"
    suite = tmp_path / "suite"
    _write_physical_aggregate(physical)
    _write_request_details(suite / "run0" / "request_details.csv")

    rows = build_bridge_rows(
        specs=[("p0", "Fig. 8/9/10", suite)],
        physical_aggregate=physical,
    )
    summary = summarize_bridge(rows)

    assert len(rows) == 2
    assert {row["physical_action"] for row in rows} == {
        "gpu_replica_restore_p2p",
        "host_restore_h2d",
    }
    assert {row["measurement_tokens"] for row in rows} == {512, 1024}
    assert all(row["bridge_eligible"] for row in rows)
    assert summary[0]["restore_rows"] == 2
    assert summary[0]["bridge_eligible_rows"] == 2
    assert summary[0]["bridge_missing_rows"] == 0
    assert summary[0]["bridge_ready"] is True
