from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sglsim.srt.simulation.global_des import (
    GlobalDES,
    KvLocation,
    PhysicalGpuId,
)
from sglsim.srt.simulation.global_des_artifacts import (
    ArtifactInvariantError,
    ArtifactSerializationError,
    ArtifactTargetExistsError,
    EVIDENCE_CLASS,
    write_global_des_artifacts,
)


def _build_des() -> tuple[GlobalDES, PhysicalGpuId]:
    des = GlobalDES()
    gpu = PhysicalGpuId(
        node_id="node-0",
        local_gpu_id=0,
        dp_rank=0,
        pp_rank=0,
        tp_rank=0,
    )
    des.register_gpu(gpu, capacity_bytes=1_000_000, weights_bytes=100_000)
    des.register_node_dram("node-0", capacity_bytes=2_000_000)
    des.register_extent(
        extent_id="extent-0",
        request_id="request-0",
        size_bytes=4096,
        token_start=0,
        token_end=1,
        model_revision="model-revision",
        kv_abi="kv-abi-v1",
        source_failure_domain="node-0",
    )
    des.place_extent("extent-0", KvLocation.gpu_primary(gpu))
    # Scheduling in reverse timestamp order exercises the stable global journal
    # ordering without mutating any accounted state in callbacks.
    des.schedule_at(20, kind="later")
    des.schedule_at(10, kind="earlier")
    des.run()
    assert des.audit_invariants(strict_extent_accounting=True) == []
    return des, gpu


def _resolved_config() -> dict[str, object]:
    return {
        "request": {"seed": 2026082900, "clock": "global_virtual_time"},
        "failure": {"enabled": False},
        "deployment": {"dp_size": 1, "tp_size": 1, "pp_size": 1},
    }


def _lineage() -> dict[str, str]:
    return {
        "engine": "sglsim_global_des_test",
        "code_hash": "a" * 64,
        "run_id": "deterministic-test-cell",
    }


def _write(path: Path, des: GlobalDES) -> Path:
    return write_global_des_artifacts(
        path,
        des=des,
        rank_steps=(
            {"timestamp_ns": 20, "step_id": "step-2", "dp_rank": 0},
            {"timestamp_ns": 10, "step_id": "step-1", "dp_rank": 0},
        ),
        requests=({"arrival_ns": 0, "request_id": "request-0"},),
        failures=(),
        resolved_config=_resolved_config(),
        lineage_metadata=_lineage(),
        audit_details={
            "required_invariants": {
                "global_event_order": {
                    "passed": True,
                    "evidence": "unit-test journal",
                }
            }
        },
        recovery_transfers=(
            {
                "recovery_id": "recovery-0",
                "retry_request_id": "request-0-retry",
                "source_tier": "node_dram",
                "ready_ns": 20,
                "start_ns": 20,
                "end_ns": 21,
                "size_bytes": 4096,
            },
        ),
    )


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_writer_is_reproducible_and_manifest_counts_every_raw_row(
    tmp_path: Path,
) -> None:
    first_des, _ = _build_des()
    second_des, _ = _build_des()
    first = _write(tmp_path / "cell-a", first_des)
    second = _write(tmp_path / "cell-b", second_des)

    first_manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert first_manifest == second_manifest
    assert first_manifest["status"] == "success"
    assert first_manifest["evidence_class"] == EVIDENCE_CLASS
    assert first_manifest["simulation"]["real_hardware_measurement"] is False
    assert first_manifest["optional_files"] == [
        "transfer_groups.jsonl",
        "transfer_resources.jsonl",
        "recovery_transfers.jsonl",
    ]

    expected_files = {
        "events.jsonl",
        "rank_steps.jsonl",
        "gpu_hbm_timeseries.jsonl",
        "node_dram_timeseries.jsonl",
        "kv_extents.jsonl",
        "transfers.jsonl",
        "requests.jsonl",
        "failures.jsonl",
        "audit.json",
        "transfer_groups.jsonl",
        "transfer_resources.jsonl",
        "recovery_transfers.jsonl",
    }
    assert set(first_manifest["files"]) == expected_files
    assert {path.name for path in first.iterdir()} == {
        *expected_files,
        "manifest.json",
    }

    for filename, descriptor in first_manifest["files"].items():
        payload = (first / filename).read_bytes()
        assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
        expected_rows = 1 if filename == "audit.json" else len(
            payload.splitlines()
        )
        assert descriptor["row_count"] == expected_rows
        assert descriptor["schema"]
        assert descriptor["evidence_class"] == EVIDENCE_CLASS
        assert descriptor["engine"] == "sglsim_global_des_test"
        assert descriptor["code_hash"] == "a" * 64
        assert descriptor["config_sha256"] == first_manifest["config_sha256"]

    events = _jsonl_rows(first / "events.jsonl")
    audit = json.loads((first / "audit.json").read_text())
    assert audit["details"]["required_invariants"]["global_event_order"][
        "passed"
    ] is True
    assert [row["timestamp_ns"] for row in events] == sorted(
        row["timestamp_ns"] for row in events
    )
    assert all(isinstance(row["before"], dict) for row in events)
    assert all(isinstance(row["after"], dict) for row in events)
    assert [row["step_id"] for row in _jsonl_rows(first / "rank_steps.jsonl")] == [
        "step-1",
        "step-2",
    ]
    assert _jsonl_rows(first / "recovery_transfers.jsonl")[0][
        "recovery_id"
    ] == "recovery-0"

    for filename in expected_files:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_writer_rejects_nan_without_leaving_cell_or_temporary_directory(
    tmp_path: Path,
) -> None:
    des, _ = _build_des()
    target = tmp_path / "nan-cell"
    with pytest.raises(ArtifactSerializationError, match="non-finite"):
        write_global_des_artifacts(
            target,
            des=des,
            rank_steps=({"timestamp_ns": 0, "duration_s": float("nan")},),
            requests=(),
            failures=(),
            resolved_config=_resolved_config(),
            lineage_metadata=_lineage(),
        )
    assert not target.exists()
    assert list(tmp_path.glob(".nan-cell.tmp-*")) == []


def test_writer_fails_closed_when_strict_extent_accounting_fails(
    tmp_path: Path,
) -> None:
    des, gpu = _build_des()
    des.gpu_hbm[gpu].primary_kv += 1
    target = tmp_path / "invalid-cell"

    with pytest.raises(ArtifactInvariantError, match="ledger=.*required"):
        _write(target, des)

    assert not target.exists()
    assert list(tmp_path.glob(".invalid-cell.tmp-*")) == []


def test_writer_never_overwrites_an_existing_cell_directory(tmp_path: Path) -> None:
    des, _ = _build_des()
    target = tmp_path / "existing-cell"
    target.mkdir()
    sentinel = target / "user-data.txt"
    sentinel.write_text("keep me")

    with pytest.raises(ArtifactTargetExistsError, match="will not be overwritten"):
        _write(target, des)

    assert sentinel.read_text() == "keep me"
    assert set(target.iterdir()) == {sentinel}


def test_writer_exports_canonical_recovery_load_groups(tmp_path: Path) -> None:
    des, _ = _build_des()
    des.register_recovery_load_group(
        group_id="recovery-group-0",
        retry_request_id="request-0-retry",
        request_backup_key="request-0",
        route_version=2,
        expected_shard_keys=("pp0/tp0",),
    )

    target = _write(tmp_path / "recovery-group-cell", des)
    rows = _jsonl_rows(target / "transfer_groups.jsonl")

    assert [row["group_id"] for row in rows] == ["recovery-group-0"]
    assert rows[0]["group_class"] == "aegis_recovery_load"
    assert rows[0]["retry_request_id"] == "request-0-retry"


def test_scheduler_aggregate_mode_does_not_require_fabricated_prefix_extents(
    tmp_path: Path,
) -> None:
    des, gpu = _build_des()
    # This extra allocator-owned aggregate deliberately has no request extent.
    des.gpu_hbm[gpu].prefix_kv = 8192
    target = tmp_path / "scheduler-aggregate"

    write_global_des_artifacts(
        target,
        des=des,
        rank_steps=(),
        requests=(),
        failures=(),
        resolved_config=_resolved_config(),
        lineage_metadata=_lineage(),
        extent_accounting_mode="scheduler_aggregate",
        supplemental_audit_errors=(),
    )

    audit = json.loads((target / "audit.json").read_text())
    assert audit["passed"] is True
    assert audit["scheduler_aggregate_accounting"] is True
    assert audit["extent_accounting_mode"] == "scheduler_aggregate"
    assert not any(
        str(row.get("extent_id", "")).startswith("artifact-final-prefix/")
        for row in _jsonl_rows(target / "kv_extents.jsonl")
    )
