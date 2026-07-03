from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PROMOTE_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "promote_p0_physical_tensor_copy_candidates.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_p0(path: Path) -> None:
    rows = [
        {
            "suite_label": "suite-a",
            "figures": "Fig. 8",
            "suite_root": "/tmp/suite-a",
            "request_csv_count": "1",
            "request_rows": "10",
            "replay_rows": "4",
            "restore_rows": "2",
            "physical_tensor_copy_rows": "0",
            "nonfinal_restore_rows": "2",
            "missing_data_plane_rows": "0",
            "missing_timing_source_rows": "0",
            "schema_complete": "True",
            "final_ready": "False",
            "data_planes": json.dumps({"metadata_materialization": 2}),
            "timing_sources": json.dumps({"physical_recovery_path_csv": 2}),
            "next_required_evidence": "replace metadata",
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_row(rid: str, event_kind: str) -> dict[str, str]:
    return {
        "suite_label": "suite-a",
        "request_csv": "/tmp/request_details.csv",
        "rid": rid,
        "restore_action": "host_restore" if "h2d" in event_kind else "gpu_replica",
        "restored_tokens": "128",
        "aegis_replay_restore_data_plane": "physical_tensor_copy",
        "aegis_replay_restore_timing_source": event_kind,
        "physical_copy_verified": "true",
        "copy_latency_ms": "0.5",
        "copy_bytes": "4096",
        "source_artifact": "/tmp/events.csv",
        "physical_action": "host_restore_h2d",
        "event_kind": event_kind,
        "event_schema_version": "p0-live-copy-event-v1",
        "event_source": "/tmp/events.csv",
        "event_line": "2",
    }


def write_candidates(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_promotes_ready_candidates_to_strict_final_overlay(tmp_path: Path) -> None:
    module = load_module(PROMOTE_SCRIPT, "aegiskv_promote_p0_candidates")
    p0 = tmp_path / "p0.csv"
    candidates = tmp_path / "candidates.csv"
    write_p0(p0)
    write_candidates(
        candidates,
        [
            candidate_row("r0", "live_host_h2d_copy"),
            candidate_row("r1", "live_peer_p2p_copy"),
        ],
    )

    promoted, problems = module.build_promoted_rows(
        p0_rows=module.read_csv(p0),
        candidate_rows=module.read_csv(candidates),
        candidate_audit={"strict_final_candidate_ready": True},
        candidates_csv=candidates,
    )
    payload = module.summarize(promoted, problems)

    assert problems == []
    assert payload["promotion_ready"] is True
    assert payload["restore_rows"] == 2
    assert payload["physical_tensor_copy_rows"] == 2
    assert payload["host_h2d_candidate_rows"] == 1
    assert payload["peer_p2p_candidate_rows"] == 1
    assert promoted[0]["final_ready"] is True
    assert promoted[0]["data_planes"] == json.dumps(
        {"physical_tensor_copy": 2}, sort_keys=True
    )


def test_rejects_promotion_when_candidate_count_is_incomplete(tmp_path: Path) -> None:
    module = load_module(PROMOTE_SCRIPT, "aegiskv_promote_p0_candidates_incomplete")
    p0 = tmp_path / "p0.csv"
    candidates = tmp_path / "candidates.csv"
    write_p0(p0)
    write_candidates(candidates, [candidate_row("r0", "live_host_h2d_copy")])

    promoted, problems = module.build_promoted_rows(
        p0_rows=module.read_csv(p0),
        candidate_rows=module.read_csv(candidates),
        candidate_audit={"strict_final_candidate_ready": True},
        candidates_csv=candidates,
    )
    payload = module.summarize(promoted, problems)

    assert payload["promotion_ready"] is False
    assert problems == ["suite_candidate_count_mismatch:suite-a:1!=2"]
