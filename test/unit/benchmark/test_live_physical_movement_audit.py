from __future__ import annotations

import csv
from pathlib import Path

from AegisKV.draft.experiments.sglsim_large.scripts.audit_live_physical_movement import (
    audit_suite,
)


def _write_request_details(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "rid",
        "aegis_failover_replay",
        "aegis_replay_restore_action",
        "aegis_replay_restore_data_plane",
        "aegis_replay_restore_timing_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_suite_rejects_metadata_materialization(tmp_path: Path) -> None:
    _write_request_details(
        tmp_path / "run0" / "request_details.csv",
        [
            {
                "rid": "r0",
                "aegis_failover_replay": "true",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_data_plane": "metadata_materialization",
                "aegis_replay_restore_timing_source": "physical_recovery_path_csv",
            }
        ],
    )

    _, totals = audit_suite(tmp_path)

    assert totals["restore_rows"] == 1
    assert totals["nonfinal_restore_rows"] == 1
    assert totals["final_ready"] is False
    assert totals["data_planes"] == {"metadata_materialization": 1}
    assert totals["schema_complete"] is True
    assert totals["missing_data_plane_rows"] == 0
    assert totals["missing_timing_source_rows"] == 0


def test_audit_suite_accepts_physical_tensor_copy(tmp_path: Path) -> None:
    _write_request_details(
        tmp_path / "run0" / "request_details.csv",
        [
            {
                "rid": "r0",
                "aegis_failover_replay": "true",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_data_plane": "physical_tensor_copy",
                "aegis_replay_restore_timing_source": "live_cuda_copy",
            }
        ],
    )

    _, totals = audit_suite(tmp_path)

    assert totals["restore_rows"] == 1
    assert totals["nonfinal_restore_rows"] == 0
    assert totals["final_ready"] is True
    assert totals["schema_complete"] is True


def test_audit_suite_rejects_missing_timing_source(tmp_path: Path) -> None:
    _write_request_details(
        tmp_path / "run0" / "request_details.csv",
        [
            {
                "rid": "r0",
                "aegis_failover_replay": "true",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_data_plane": "physical_tensor_copy",
                "aegis_replay_restore_timing_source": "",
            }
        ],
    )

    _, totals = audit_suite(tmp_path)

    assert totals["restore_rows"] == 1
    assert totals["physical_tensor_copy_rows"] == 1
    assert totals["missing_timing_source_rows"] == 1
    assert totals["schema_complete"] is False
    assert totals["final_ready"] is False
