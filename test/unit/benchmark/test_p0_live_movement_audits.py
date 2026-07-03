from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "write_p0_live_movement_audits.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_write_p0_live_movement_audits",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_request_details(path: Path, rows: list[dict[str, str]]) -> None:
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


def test_write_p0_live_movement_audits_keeps_nonfinal_boundary(
    tmp_path: Path,
) -> None:
    module = load_module()
    suite_a = tmp_path / "suite_a"
    suite_b = tmp_path / "suite_b"
    _write_request_details(
        suite_a / "run0" / "request_details.csv",
        [
            {
                "rid": "a0",
                "aegis_failover_replay": "true",
                "aegis_replay_restore_action": "host_restore",
                "aegis_replay_restore_data_plane": "metadata_materialization",
                "aegis_replay_restore_timing_source": "physical_recovery_path_csv",
            }
        ],
    )
    _write_request_details(
        suite_b / "run0" / "request_details.csv",
        [
            {
                "rid": "b0",
                "aegis_failover_replay": "true",
                "aegis_replay_restore_action": "gpu_replica",
                "aegis_replay_restore_data_plane": "physical_tensor_copy",
                "aegis_replay_restore_timing_source": "live_cuda_copy",
            }
        ],
    )

    payload = module.run(
        output_dir=tmp_path / "out",
        specs=[
            ("suite_a", "Fig. A", suite_a),
            ("suite_b", "Fig. B", suite_b),
        ],
    )

    assert payload["summary"]["restore_rows"] == 2
    assert payload["summary"]["physical_tensor_copy_rows"] == 1
    assert payload["summary"]["nonfinal_restore_rows"] == 1
    assert payload["summary"]["final_ready"] is False
    assert payload["suites"][0]["data_planes"] == {"metadata_materialization": 1}
    assert payload["suites"][1]["data_planes"] == {"physical_tensor_copy": 1}
    assert (suite_a / "live_physical_movement_audit.csv").exists()
    assert (suite_b / "live_physical_movement_audit.md").exists()

    written = json.loads((tmp_path / "out" / "p0_live_movement_audits.json").read_text())
    assert written["summary"]["schema_complete"] is True
    assert "metadata materialization" in (
        tmp_path / "out" / "p0_live_movement_audits.md"
    ).read_text(encoding="utf-8")
