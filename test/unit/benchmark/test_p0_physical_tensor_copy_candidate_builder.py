from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BUILDER_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "build_p0_physical_tensor_copy_candidates.py"
)
VALIDATOR_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "validate_p0_physical_tensor_copy_candidates.py"
)
TEMPLATE_SCRIPT = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "draft"
    / "experiments"
    / "sglsim_large"
    / "scripts"
    / "write_p0_physical_tensor_copy_event_template.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
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
        "physical_action",
        "bridge_eligible",
    ]
    rows = [
        {
            "suite_label": "p0",
            "request_csv": "/tmp/a/request_details.csv",
            "rid": "host-retry",
            "restore_action": "host_restore",
            "restored_tokens": "128",
            "physical_action": "host_restore_h2d",
            "bridge_eligible": "true",
        },
        {
            "suite_label": "p0",
            "request_csv": "/tmp/a/request_details.csv",
            "rid": "gpu-retry",
            "restore_action": "gpu_replica",
            "restored_tokens": "256",
            "physical_action": "gpu_replica_restore_p2p",
            "bridge_eligible": "true",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_event_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def event_row(rid: str, action: str, tokens: int) -> dict[str, object]:
    return {
        "suite_label": "p0",
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


def test_builds_validator_ready_candidates_from_csv_events(tmp_path: Path) -> None:
    builder = load_module(BUILDER_SCRIPT, "aegiskv_build_p0_candidates")
    validator = load_module(VALIDATOR_SCRIPT, "aegiskv_validate_p0_candidates_for_builder")
    bridge = tmp_path / "bridge.csv"
    events = tmp_path / "events.csv"
    candidates = tmp_path / "candidates.csv"
    write_bridge_rows(bridge)
    write_event_csv(
        events,
        [
            event_row("host-retry", "host_restore", 128),
            event_row("gpu-retry", "gpu_replica", 256),
        ],
    )

    payload = builder.build_payload(
        bridge_rows=bridge,
        event_csvs=[events],
        event_jsonls=[],
    )
    builder.write_candidates(candidates, payload["candidate_rows"])
    validated = validator.build_payload(bridge_rows=bridge, candidates_csv=candidates)

    assert payload["status"] == "candidates_complete"
    assert payload["event_schema_version"] == "p0-live-copy-event-v1"
    assert len(payload["bridge_required_digest"]) == 64
    assert payload["bridge_required_digest"] == validated["bridge_required_digest"]
    assert payload["accepted_rows"] == 2
    assert payload["missing_rows"] == 0
    assert validated["status"] == "strict_final_candidate_ready"


def test_writes_live_copy_event_template_without_candidate_rows(tmp_path: Path) -> None:
    template = load_module(TEMPLATE_SCRIPT, "aegiskv_write_p0_event_template")
    output_csv = tmp_path / "events.template.csv"
    output_md = tmp_path / "event_schema.md"
    bridge = tmp_path / "bridge.csv"
    required_csv = tmp_path / "events.required.csv"
    write_bridge_rows(bridge)

    template.write_csv_template(output_csv)
    required_rows = template.required_event_rows(bridge)
    template.write_required_events(required_csv, required_rows)
    output_md.write_text(
        template.render_markdown(output_csv=output_csv, required_csv=required_csv),
        encoding="utf-8",
    )
    payload = template.build_payload(
        output_csv=output_csv,
        output_md=output_md,
        required_csv=required_csv,
        required_rows=required_rows,
    )

    with output_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with required_csv.open(newline="", encoding="utf-8") as f:
        required = list(csv.DictReader(f))

    assert rows == []
    assert len(required) == 2
    assert {row["event_kind"] for row in required} == {
        "live_host_h2d_copy",
        "live_peer_p2p_copy",
    }
    assert {row["physical_copy_verified"] for row in required} == {""}
    assert {row["copy_latency_ms"] for row in required} == {""}
    assert {row["copy_bytes"] for row in required} == {""}
    assert payload["status"] == "template_written"
    assert payload["event_schema_version"] == "p0-live-copy-event-v1"
    assert payload["required_event_rows"] == 2
    assert payload["required_event_kinds"] == {
        "live_host_h2d_copy": 1,
        "live_peer_p2p_copy": 1,
    }
    assert "aegis_replay_restore_data_plane" in payload["template_fields"]
    assert "copy_latency_ms" in payload["template_fields"]
    assert "physical_copy_verified" in payload["required_fields"]
    assert "required-event manifest" in output_md.read_text(encoding="utf-8")
    assert "live_peer_p2p_copy" in output_md.read_text(encoding="utf-8")


def test_rejects_metadata_or_calibration_only_events(tmp_path: Path) -> None:
    builder = load_module(BUILDER_SCRIPT, "aegiskv_build_p0_candidates_reject")
    bridge = tmp_path / "bridge.csv"
    events = tmp_path / "events.csv"
    write_bridge_rows(bridge)
    bad = event_row("host-retry", "host_restore", 128)
    bad["aegis_replay_restore_data_plane"] = "metadata_materialization"
    bad["aegis_replay_restore_timing_source"] = "physical_recovery_path_csv"
    write_event_csv(events, [bad])

    payload = builder.build_payload(
        bridge_rows=bridge,
        event_csvs=[events],
        event_jsonls=[],
    )

    assert payload["status"] == "candidates_partial"
    assert payload["accepted_rows"] == 0
    assert payload["missing_rows"] == 2
    problems = payload["rejected_events"][0]["problems"]
    assert "data_plane_not_physical_tensor_copy" in problems
    assert "timing_source_is_not_live_copy" in problems


def test_reports_explicit_missing_event_sources(tmp_path: Path) -> None:
    builder = load_module(BUILDER_SCRIPT, "aegiskv_build_p0_candidates_missing_sources")
    bridge = tmp_path / "bridge.csv"
    missing = tmp_path / "missing-events.csv"
    write_bridge_rows(bridge)

    payload = builder.build_payload(
        bridge_rows=bridge,
        event_csvs=[missing],
        event_jsonls=[],
    )

    assert payload["status"] == "missing_event_sources"
    assert len(payload["bridge_required_digest"]) == 64
    assert payload["event_rows"] == 0
    assert payload["missing_event_sources"] == [str(missing)]
    assert payload["accepted_rows"] == 0
    assert payload["missing_rows"] == 2


def test_defaults_live_timing_source_from_jsonl_physical_action(tmp_path: Path) -> None:
    builder = load_module(BUILDER_SCRIPT, "aegiskv_build_p0_candidates_jsonl")
    bridge = tmp_path / "bridge.csv"
    events = tmp_path / "events.jsonl"
    write_bridge_rows(bridge)
    events.write_text(
        json.dumps(
            {
                "suite_label": "p0",
                "request_csv": "/tmp/a/request_details.csv",
                "rid": "gpu-retry",
                "event_kind": "live_physical_tensor_copy",
                "physical_copy_verified": True,
                "latency_ms": 0.75,
                "bytes": 8192,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = builder.build_payload(
        bridge_rows=bridge,
        event_csvs=[],
        event_jsonls=[events],
    )

    assert payload["status"] == "candidates_partial"
    assert payload["accepted_rows"] == 1
    row = payload["candidate_rows"][0]
    assert row["rid"] == "gpu-retry"
    assert row["event_schema_version"] == "p0-live-copy-event-v1"
    assert row["aegis_replay_restore_data_plane"] == "physical_tensor_copy"
    assert row["aegis_replay_restore_timing_source"] == "live_peer_p2p_copy"


def test_rejects_verified_copy_without_live_copy_identity(tmp_path: Path) -> None:
    builder = load_module(BUILDER_SCRIPT, "aegiskv_build_p0_candidates_no_identity")
    bridge = tmp_path / "bridge.csv"
    events = tmp_path / "events.jsonl"
    write_bridge_rows(bridge)
    events.write_text(
        json.dumps(
            {
                "suite_label": "p0",
                "request_csv": "/tmp/a/request_details.csv",
                "rid": "host-retry",
                "physical_copy_verified": True,
                "latency_ms": 0.5,
                "bytes": 4096,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = builder.build_payload(
        bridge_rows=bridge,
        event_csvs=[],
        event_jsonls=[events],
    )

    assert payload["status"] == "candidates_partial"
    assert payload["accepted_rows"] == 0
    assert "live_physical_copy_identity_missing" in payload["rejected_events"][0][
        "problems"
    ]
