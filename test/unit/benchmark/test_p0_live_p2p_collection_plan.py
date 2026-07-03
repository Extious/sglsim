from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PLAN_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "plan_p0_live_p2p_collection.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_required(path: Path) -> None:
    fields = [
        "suite_label",
        "request_csv",
        "rid",
        "event_kind",
        "restore_action",
        "restored_tokens",
    ]
    rows = [
        {
            "suite_label": "p0",
            "request_csv": "/tmp/request_details.csv",
            "rid": "h2d",
            "event_kind": "live_host_h2d_copy",
            "restore_action": "host_restore",
            "restored_tokens": "128",
        },
        {
            "suite_label": "p0",
            "request_csv": "/tmp/request_details.csv",
            "rid": "p2p-a",
            "event_kind": "live_peer_p2p_copy",
            "restore_action": "gpu_replica",
            "restored_tokens": "256",
        },
        {
            "suite_label": "p0",
            "request_csv": "/tmp/request_details.csv",
            "rid": "p2p-b",
            "event_kind": "live_peer_p2p_copy",
            "restore_action": "gpu_replica",
            "restored_tokens": "512",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_plans_remote_p2p_collection_when_p2p_output_missing(tmp_path: Path) -> None:
    module = load_module(PLAN_SCRIPT, "aegiskv_plan_p0_live_p2p")
    required = tmp_path / "events.required.csv"
    host_live = tmp_path / "events.live.json"
    p2p_live = tmp_path / "missing-p2p.json"
    write_required(required)
    host_live.write_text(
        json.dumps(
            {
                "event_rows": 1,
                "event_rows_by_kind": {"live_host_h2d_copy": 1},
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_payload(
        required_csv=required,
        host_live_json=host_live,
        p2p_live_json=p2p_live,
        remote_host="gpubox",
        remote_workdir="~/p0",
        python_bin="/opt/venv/bin/python",
    )

    assert payload["status"] == "ready_to_submit_p2p_live_copy_job"
    assert payload["required_rows"] == 3
    assert payload["required_p2p_rows"] == 2
    assert payload["required_p2p_tokens"] == 768
    assert payload["host_live_copy_event_rows_by_kind"] == {"live_host_h2d_copy": 1}
    assert payload["p2p_live_copy_event_rows"] == 0
    assert "gpubox" in payload["commands"]["submit"]
    assert "aegiskv_p0_live_p2p_copy_events.sbatch" in payload["commands"]["submit"]


def test_marks_p2p_collection_complete_when_all_peer_rows_exist(tmp_path: Path) -> None:
    module = load_module(PLAN_SCRIPT, "aegiskv_plan_p0_live_p2p_complete")
    required = tmp_path / "events.required.csv"
    host_live = tmp_path / "events.live.json"
    p2p_live = tmp_path / "events.p2p_live.json"
    write_required(required)
    host_live.write_text(json.dumps({"event_rows": 1}), encoding="utf-8")
    p2p_live.write_text(
        json.dumps(
            {
                "event_rows": 2,
                "event_rows_by_kind": {"live_peer_p2p_copy": 2},
                "skipped_rows": 0,
            }
        ),
        encoding="utf-8",
    )

    payload = module.build_payload(
        required_csv=required,
        host_live_json=host_live,
        p2p_live_json=p2p_live,
        remote_host="gpubox",
        remote_workdir="~/p0",
        python_bin="python3",
    )

    assert payload["status"] == "p2p_live_copy_complete"
    assert payload["p2p_ready"] is True
