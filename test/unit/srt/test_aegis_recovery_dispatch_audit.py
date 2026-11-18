from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from AegisKV.experiments.real_weka_recovery_dispatch.audit_target_parallelism import (
    audit,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> Path:
    mooncake = _write(
        tmp_path / "router.py",
        "for request_id in order:\n    await self._replay_one(context)\n",
    )
    sglsim = _write(
        tmp_path / "run.py",
        "by_target: dict[str, list[ReplayRequest]] = {}\n"
        "for target, requests in sorted(by_target.items()):\n"
        "    key = -decisions[protection][row.row.rid].visible_harm_ms\n",
    )
    patch = _write(
        tmp_path / "fix.patch",
        "+target_parallel_due_time_queues\n+await asyncio.gather(tasks)\n",
    )
    rows = tmp_path / "rows.csv"
    with rows.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["event_id", "policy", "request_id", "activated_target"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"event_id": "e1", "policy": "aegis", "request_id": "a", "activated_target": "t1"},
                {"event_id": "e1", "policy": "aegis", "request_id": "b", "activated_target": "t2"},
                {"event_id": "e1", "policy": "aegis", "request_id": "c", "activated_target": "t1"},
                {"event_id": "e2", "policy": "aegis", "request_id": "d", "activated_target": "t1"},
                {"event_id": "e2", "policy": "other", "request_id": "d", "activated_target": "t1"},
            ]
        )
    records = {
        "design_semantics": "source_owned_independent_hbm_dram_v2",
        "policy": "aegis",
        "failure_rows": {"path": str(rows), "sha256": _sha256(rows)},
        "frozen_mooncake_router": {
            "path": str(mooncake),
            "sha256": _sha256(mooncake),
        },
        "sglsim_failure_evaluator": {
            "path": str(sglsim),
            "sha256": _sha256(sglsim),
        },
        "staged_runtime_patch": {
            "path": str(patch),
            "sha256": _sha256(patch),
        },
        "expected_profile": {"event_count": 2, "row_count": 4},
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(records) + "\n", encoding="utf-8")
    return config


def test_dispatch_audit_counts_only_target_topology(tmp_path: Path) -> None:
    result = audit(_fixture(tmp_path))
    topology = result["topology"]
    assert result["status"] == "PASS"
    assert topology["failure_event_count"] == 2
    assert topology["request_dispatch_count"] == 4
    assert topology["events_with_multiple_targets"] == 1
    assert topology["events_with_cross_target_hol_opportunity"] == 1
    assert topology["maximum_target_local_queue_depth"] == 2
    assert result["runtime_alignment_status"] == (
        "TARGET_QUEUE_MISMATCH_WITH_STAGED_FIX"
    )
    assert result["performance_results_present"] is False


def test_dispatch_audit_rejects_source_drift(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["frozen_mooncake_router"]["sha256"] = "0" * 64
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="Mooncake Router sha256 mismatch"):
        audit(config_path)


def test_dispatch_audit_rejects_duplicate_request_in_event(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    rows = Path(config["failure_rows"]["path"])
    with rows.open("a", encoding="utf-8") as stream:
        stream.write("e1,aegis,a,t1\n")
    config["failure_rows"]["sha256"] = _sha256(rows)
    config["expected_profile"]["row_count"] = 5
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="duplicate event/request"):
        audit(config_path)


def test_dispatch_audit_rejects_missing_slack_harm_tie_break(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    evaluator = Path(config["sglsim_failure_evaluator"]["path"])
    evaluator.write_text(
        "by_target: dict[str, list[ReplayRequest]] = {}\n"
        "for target, requests in sorted(by_target.items()):\n    pass\n",
        encoding="utf-8",
    )
    config["sglsim_failure_evaluator"]["sha256"] = _sha256(evaluator)
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Slack/harm recovery"):
        audit(config_path)
