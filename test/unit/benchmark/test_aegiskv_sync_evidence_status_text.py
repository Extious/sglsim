from __future__ import annotations

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
    / "scripts"
    / "sync_evidence_status_text.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_sync_evidence_status_text",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_status(path: Path, *, status: str, pending: int, running: int, completed: int) -> None:
    path.write_text(
        json.dumps(
            {
                "checked_at": "2026-07-02T19:33:07+08:00",
                "overall_status": status,
                "status_counts": {
                    "completed": completed,
                    "failed": 0,
                    "pending": pending,
                    "running": running,
                },
                "total_bound_jobs": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def write_readiness(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "bridge": {
                    "bridge_eligible_rows": 693,
                    "bridge_missing_rows": 0,
                },
                "matrix": {
                    "accepted_pairs": 4,
                    "cases": {
                        "same_write_wait_complete": {
                            "accepted_pairs": 1,
                            "required_repeats": 3,
                        }
                    },
                    "overall_status": "partial",
                    "physical_restore_rows": 11,
                    "required_pairs": 6,
                },
                "p0_physical_tensor_copy_event_template": {
                    "required_event_kinds": {
                        "live_host_h2d_copy": 375,
                        "live_peer_p2p_copy": 318,
                    },
                    "required_rows": 693,
                },
                "p0_live_copy_events": {
                    "event_rows": 375,
                    "event_rows_by_kind": {
                        "live_host_h2d_copy": 375,
                    },
                    "skipped_rows": 318,
                },
                "p0_live_p2p_collection_plan": {
                    "required_p2p_rows": 318,
                    "status": "ready_to_submit_p2p_live_copy_job",
                },
                "p0": {
                    "missing_data_plane_rows": 0,
                    "missing_timing_source_rows": 0,
                    "physical_tensor_copy_rows": 0,
                    "restore_rows": 693,
                },
                "status": "waiting_on_matrix_and_p0_physical_bridge",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_syncs_pending_status_into_paper_and_beamer(tmp_path: Path) -> None:
    module = load_module()
    status_json = tmp_path / "status.json"
    readiness_json = tmp_path / "readiness.json"
    paper = tmp_path / "current_results_overview.md"
    beamer = tmp_path / "presentation.tex"
    write_status(status_json, status="pending", pending=4, running=0, completed=0)
    write_readiness(readiness_json)
    paper.write_text(
        (
            "P0 suites 有 1 条 restore rows，但\n"
            "`physical_tensor_copy` 行数仍为 1。\n"
            "`strict_final_recovery_matrix_audit`：当前 same-write-cost best-effort\n"
            "恢复收益已有 0/1 个 SGLang repeats 通过，wait-complete 完整物理恢复只有\n"
            "0/1 个 repeats 通过。\n"
            "最新 Slurm poll（`2026-07-02T19:27:08+08:00`）显示这 4 个作业仍处于\n"
            "`pending`：`pending=4`、`running=0`、`completed=0`、`failed=0`；每个作业请求\n"
        ),
        encoding="utf-8",
    )
    beamer.write_text(
        (
            r"\item P0 restore: 1 rows，copy=1，schema missing=2"
            "\n"
            r"\item P0 live-copy events: required 0，measured 0，H2D 0，P2P blocked 0"
            "\n"
            r"\item P2P remote plan: missing，rows 0"
            "\n"
            r"\item SGLang matrix: hidden-control 1/1，wait-complete 0/1"
            "\n"
            r"\item r2/r3 Slurm: 4 jobs pending，poll 19:27；wrapper 仅 complete 后 sync/audit"
        ),
        encoding="utf-8",
    )

    result = module.sync_files(
        status_json=status_json,
        readiness_json=readiness_json,
        paper=paper,
        beamer=beamer,
    )

    assert result["short_status"] == "4 jobs pending"
    assert (
        result["dataplane_readiness_status"]
        == "waiting_on_matrix_and_p0_physical_bridge"
    )
    assert "2026-07-02T19:33:07+08:00" in paper.read_text(encoding="utf-8")
    assert "P0 suites 有 693 条 restore rows；physical bridge 覆盖 693 条" in paper.read_text(encoding="utf-8")
    assert "恢复收益已有 3/3 个 SGLang repeats 通过" in paper.read_text(
        encoding="utf-8"
    )
    assert "wait-complete 完整物理恢复已有\n1/3 个 repeats 通过" in paper.read_text(
        encoding="utf-8"
    )
    assert "`pending=4`" in paper.read_text(encoding="utf-8")
    assert "P0 restore: 693 rows，copy=0，bridge=693，schema missing=0" in beamer.read_text(
        encoding="utf-8"
    )
    assert "P0 live-copy events: required 693，measured 375，H2D 375，P2P blocked 318" in beamer.read_text(
        encoding="utf-8"
    )
    assert r"P2P remote plan: ready\_to\_submit\_p2p\_live\_copy\_job，rows 318" in beamer.read_text(
        encoding="utf-8"
    )
    assert "SGLang matrix: accepted 4/6，wait-complete 1/3" in beamer.read_text(
        encoding="utf-8"
    )
    assert "r2/r3 Slurm: 4 jobs pending，poll 19:33" in beamer.read_text(
        encoding="utf-8"
    )
    assert "complete 后 sync/audit" in beamer.read_text(encoding="utf-8")


def test_short_status_describes_mixed_states() -> None:
    module = load_module()
    payload = {
        "status_counts": {"completed": 1, "failed": 0, "pending": 2, "running": 1},
        "total_bound_jobs": 4,
    }

    assert module.short_status(payload) == "2 pending / 1 running / 1 complete"
