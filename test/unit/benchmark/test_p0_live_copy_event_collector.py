from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "collect_p0_live_copy_events.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def required_row(
    *,
    rid: str,
    event_kind: str,
    restored_tokens: int,
    physical_action: str,
    restore_action: str,
) -> dict[str, str]:
    return {
        "suite_label": "p0",
        "request_csv": "/tmp/request_details.csv",
        "rid": rid,
        "event_schema_version": "p0-live-copy-event-v1",
        "event_kind": event_kind,
        "restore_action": restore_action,
        "restored_tokens": str(restored_tokens),
        "aegis_replay_restore_data_plane": "physical_tensor_copy",
        "aegis_replay_restore_timing_source": event_kind,
        "physical_action": physical_action,
    }


def test_selects_required_rows_by_kind_and_limit() -> None:
    collector = load_module(COLLECTOR_SCRIPT, "aegiskv_collect_p0_live_copy_select")
    rows = [
        required_row(
            rid="h2d-1",
            event_kind="live_host_h2d_copy",
            restored_tokens=128,
            physical_action="host_restore_h2d",
            restore_action="host_restore",
        ),
        required_row(
            rid="p2p-1",
            event_kind="live_peer_p2p_copy",
            restored_tokens=256,
            physical_action="gpu_replica_restore_p2p",
            restore_action="gpu_replica",
        ),
        required_row(
            rid="h2d-2",
            event_kind="live_host_h2d_copy",
            restored_tokens=512,
            physical_action="host_restore_h2d",
            restore_action="host_restore",
        ),
    ]

    selected = collector.select_required_rows(
        rows,
        event_kind="live_host_h2d_copy",
        limit=1,
    )

    assert [row["rid"] for row in selected] == ["h2d-1"]


def test_builds_event_rows_accepted_by_candidate_builder_shape(tmp_path: Path) -> None:
    collector = load_module(COLLECTOR_SCRIPT, "aegiskv_collect_p0_live_copy_build")
    row = required_row(
        rid="h2d-1",
        event_kind="live_host_h2d_copy",
        restored_tokens=128,
        physical_action="host_restore_h2d",
        restore_action="host_restore",
    )
    plan = collector.copy_plan(
        row,
        host_target_device=0,
        peer_source_device=0,
        peer_target_device=1,
    )

    event = collector.build_event_row(
        row,
        latency_ms=0.125,
        copy_bytes=collector.copy_bytes_for_row(row, bytes_per_token=147456),
        source_artifact=tmp_path / "events.live.csv",
        started_at_ns=100,
        ended_at_ns=200,
        repeats=3,
        warmup=1,
        bytes_per_token=147456,
        plan=plan,
    )

    assert event["event_schema_version"] == "p0-live-copy-event-v1"
    assert event["event_kind"] == "live_host_h2d_copy"
    assert event["aegis_replay_restore_data_plane"] == "physical_tensor_copy"
    assert event["aegis_replay_restore_timing_source"] == "live_host_h2d_copy"
    assert event["physical_copy_verified"] == "true"
    assert float(event["copy_latency_ms"]) > 0
    assert event["copy_bytes"] == 128 * 147456
    assert event["src_device"] == "cpu:pinned"
    assert event["dst_device"] == "cuda:0"
    assert event["copy_backend"] == "torch.cuda.copy_"
    assert event["source_artifact"].endswith("events.live.csv")


def test_writes_live_event_csv_with_stable_header(tmp_path: Path) -> None:
    collector = load_module(COLLECTOR_SCRIPT, "aegiskv_collect_p0_live_copy_csv")
    output = tmp_path / "events.live.csv"
    row = required_row(
        rid="p2p-1",
        event_kind="live_peer_p2p_copy",
        restored_tokens=256,
        physical_action="gpu_replica_restore_p2p",
        restore_action="gpu_replica",
    )
    plan = collector.copy_plan(
        row,
        host_target_device=0,
        peer_source_device=0,
        peer_target_device=1,
    )
    event = collector.build_event_row(
        row,
        latency_ms=0.25,
        copy_bytes=256 * 147456,
        source_artifact=output,
        started_at_ns=300,
        ended_at_ns=500,
        repeats=1,
        warmup=0,
        bytes_per_token=147456,
        plan=plan,
    )

    collector._write_csv(output, [event])

    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["event_kind"] == "live_peer_p2p_copy"
    assert rows[0]["src_device"] == "cuda:0"
    assert rows[0]["dst_device"] == "cuda:1"
    assert rows[0]["copy_path_available"] == "true"
