from __future__ import annotations

import csv
import importlib.util
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
    / "validate_p0_physical_tensor_copy_candidates.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_validate_p0_physical_tensor_copy_candidates",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_bridge_rows(path: Path) -> None:
    fields = [
        "suite_label",
        "request_csv",
        "rid",
        "restore_action",
        "restored_tokens",
        "bridge_eligible",
    ]
    rows = [
        {
            "suite_label": "suite_a",
            "request_csv": "/tmp/a/request_details.csv",
            "rid": "r0",
            "restore_action": "host_restore",
            "restored_tokens": "128",
            "bridge_eligible": "true",
        },
        {
            "suite_label": "suite_a",
            "request_csv": "/tmp/a/request_details.csv",
            "rid": "r1",
            "restore_action": "gpu_replica",
            "restored_tokens": "256",
            "bridge_eligible": "true",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "suite_label",
        "request_csv",
        "rid",
        "restore_action",
        "restored_tokens",
        "aegis_replay_restore_data_plane",
        "aegis_replay_restore_timing_source",
        "physical_copy_verified",
        "copy_latency_ms",
        "copy_bytes",
        "source_artifact",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def valid_row(rid: str, action: str, tokens: int) -> dict[str, object]:
    return {
        "suite_label": "suite_a",
        "request_csv": "/tmp/a/request_details.csv",
        "rid": rid,
        "restore_action": action,
        "restored_tokens": tokens,
        "aegis_replay_restore_data_plane": "physical_tensor_copy",
        "aegis_replay_restore_timing_source": "live_cuda_copy",
        "physical_copy_verified": "true",
        "copy_latency_ms": "1.25",
        "copy_bytes": "4096",
        "source_artifact": "/tmp/live-copy.jsonl",
    }


def test_missing_candidate_csv_is_not_ready(tmp_path: Path) -> None:
    module = load_module()
    bridge = tmp_path / "bridge.csv"
    write_bridge_rows(bridge)

    payload = module.build_payload(
        bridge_rows=bridge,
        candidates_csv=tmp_path / "missing.csv",
    )

    assert payload["status"] == "missing_candidate_csv"
    assert payload["strict_final_candidate_ready"] is False
    assert len(payload["bridge_required_digest"]) == 64
    assert payload["required_rows"] == 2
    assert payload["accepted_rows"] == 0


def test_accepts_complete_live_physical_tensor_copy_candidates(tmp_path: Path) -> None:
    module = load_module()
    bridge = tmp_path / "bridge.csv"
    candidates = tmp_path / "candidates.csv"
    write_bridge_rows(bridge)
    write_candidate_rows(
        candidates,
        [
            valid_row("r0", "host_restore", 128),
            valid_row("r1", "gpu_replica", 256),
        ],
    )

    payload = module.build_payload(bridge_rows=bridge, candidates_csv=candidates)

    assert payload["status"] == "strict_final_candidate_ready"
    assert payload["strict_final_candidate_ready"] is True
    assert len(payload["bridge_required_digest"]) == 64
    assert payload["accepted_rows"] == 2
    assert payload["missing_rows"] == 0
    assert payload["invalid_rows"] == []


def test_rejects_metadata_or_calibration_only_candidates(tmp_path: Path) -> None:
    module = load_module()
    bridge = tmp_path / "bridge.csv"
    candidates = tmp_path / "candidates.csv"
    write_bridge_rows(bridge)
    bad = valid_row("r0", "host_restore", 128)
    bad["aegis_replay_restore_data_plane"] = "metadata_materialization"
    bad["aegis_replay_restore_timing_source"] = "physical_recovery_path_csv"
    write_candidate_rows(candidates, [bad])

    payload = module.build_payload(bridge_rows=bridge, candidates_csv=candidates)

    assert payload["status"] == "not_ready"
    assert payload["strict_final_candidate_ready"] is False
    assert payload["accepted_rows"] == 0
    problems = payload["invalid_rows"][0]["problems"]
    assert "data_plane_not_physical_tensor_copy" in problems
    assert "timing_source_is_not_live_copy" in problems
