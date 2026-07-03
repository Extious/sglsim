from __future__ import annotations

import csv
from pathlib import Path

from AegisKV.draft.experiments.sglsim_large.scripts.audit_p0_strict_final_readiness import (
    _suite_rows,
)


def _write_request_details(
    path: Path,
    data_plane: str,
    *,
    timing_source: str = "physical_recovery_path_csv",
) -> None:
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
        writer.writerow(
            {
                "rid": "r0-retry",
                "aegis_failover_replay": "true",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_data_plane": data_plane,
                "aegis_replay_restore_timing_source": timing_source,
            }
        )


def test_p0_readiness_rejects_metadata_only_restore(tmp_path: Path) -> None:
    _write_request_details(
        tmp_path / "suite" / "run0" / "request_details.csv",
        "metadata_materialization",
    )

    rows = _suite_rows(label="p0", figures="Fig. 8/9/10", suite_root=tmp_path / "suite")

    assert rows[0]["restore_rows"] == 1
    assert rows[0]["physical_tensor_copy_rows"] == 0
    assert rows[0]["nonfinal_restore_rows"] == 1
    assert rows[0]["schema_complete"] is True
    assert rows[0]["final_ready"] is False


def test_p0_readiness_accepts_physical_tensor_copy(tmp_path: Path) -> None:
    _write_request_details(
        tmp_path / "suite" / "run0" / "request_details.csv",
        "physical_tensor_copy",
    )

    rows = _suite_rows(label="p0", figures="Fig. 8/9/10", suite_root=tmp_path / "suite")

    assert rows[0]["restore_rows"] == 1
    assert rows[0]["physical_tensor_copy_rows"] == 1
    assert rows[0]["nonfinal_restore_rows"] == 0
    assert rows[0]["schema_complete"] is True
    assert rows[0]["final_ready"] is True


def test_p0_readiness_rejects_missing_timing_source(tmp_path: Path) -> None:
    _write_request_details(
        tmp_path / "suite" / "run0" / "request_details.csv",
        "physical_tensor_copy",
        timing_source="",
    )

    rows = _suite_rows(label="p0", figures="Fig. 8/9/10", suite_root=tmp_path / "suite")

    assert rows[0]["restore_rows"] == 1
    assert rows[0]["physical_tensor_copy_rows"] == 1
    assert rows[0]["missing_timing_source_rows"] == 1
    assert rows[0]["schema_complete"] is False
    assert rows[0]["final_ready"] is False
