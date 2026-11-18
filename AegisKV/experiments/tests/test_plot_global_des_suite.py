from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from AegisKV.experiments.src.plot_global_des_suite import (
    PlotEvidenceError,
    collect_evidence,
    generate_plots,
    group_readiness,
)


REPO = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO / "AegisKV" / "experiments" / "configs_v3"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _publish_smoke(
    root: Path,
    *,
    cell_id: str = "P4-unit",
    algorithm_variant: str = "full",
) -> Path:
    run = root / f"run-{cell_id}"
    cell = run / "P4" / cell_id
    cell.mkdir(parents=True)
    claims = {
        "modeled_simulation": True,
        "real_hardware_measurement": False,
        "preliminary": True,
        "paper_ready": False,
    }
    tables: dict[str, list[dict]] = {
        "events.jsonl": [{"timestamp_ns": 0, "before": {}, "after": {}}],
        "rank_steps.jsonl": [
            {
                "timestamp_ns": 0,
                "dp_rank": 0,
                "pp_rank": 0,
                "phase": "prepared",
                "running_request_ids": ["r0"],
                "waiting_request_ids": [],
                "physical_primary_bytes_total": 4,
                "physical_prefix_bytes_total": 8,
            },
            {
                "timestamp_ns": 2_000_000_000,
                "dp_rank": 0,
                "pp_rank": 0,
                "phase": "completed",
                "running_request_ids": [],
                "waiting_request_ids": [],
                "physical_primary_bytes_total": 0,
                "physical_prefix_bytes_total": 12,
            },
        ],
        "gpu_hbm_timeseries.jsonl": [
            {
                "timestamp_ns": 0,
                "node_id": "node0",
                "local_gpu_id": 0,
                "dp_rank": 0,
                "pp_rank": 0,
                "tp_rank": 0,
                "weights": 100,
                "runtime": 10,
                "primary_kv": 0,
                "prefix_kv": 0,
                "aegis_pending": 4,
                "aegis_committed": 8,
                "capacity_bytes": 1000,
                "healthy": True,
            },
            {
                "timestamp_ns": 2_000_000_000,
                "node_id": "node0",
                "local_gpu_id": 0,
                "dp_rank": 0,
                "pp_rank": 0,
                "tp_rank": 0,
                "weights": 100,
                "runtime": 10,
                "primary_kv": 0,
                "prefix_kv": 0,
                "aegis_pending": 0,
                "aegis_committed": 12,
                "capacity_bytes": 1000,
                "healthy": True,
            },
        ],
        "node_dram_timeseries.jsonl": [
            {
                "timestamp_ns": 0,
                "node_id": "node0",
                "runtime": 20,
                "hicache_primary": 0,
                "aegis_pending": 3,
                "aegis_committed": 9,
                "capacity_bytes": 2000,
                "healthy": True,
            }
        ],
        "kv_extents.jsonl": [],
        "transfers.jsonl": [
            {
                "timestamp_ns": 0,
                "transfer_id": "t0",
                "start_ns": 0,
                "end_ns": 1_000_000_000,
                "size_bytes": 12,
                "source": {"tier": "gpu_primary"},
                "destination": {"tier": "gpu_aegis"},
                "resources": ["d2d:0"],
                "state": "COMMITTED",
            }
        ],
        "transfer_groups.jsonl": [],
        "transfer_resources.jsonl": [],
        "recovery_transfers.jsonl": [],
        "requests.jsonl": [],
        "failures.jsonl": [
            {
                "event_id": "f0",
                "failure_scope": "worker",
                "injected_offset_s": 1.0,
                "recovered_offset_s": 2.0,
                "affected_dp_ranks": [0],
                "affected_request_ids": [],
                "terminated_stages": [{}],
                "restarted_stages": [{}],
            }
        ],
    }
    files: dict[str, dict] = {}
    for name, rows in tables.items():
        payload = b"".join(_canonical(row) for row in rows)
        (cell / name).write_bytes(payload)
        files[name] = {
            "sha256": _sha(payload),
            "row_count": len(rows),
            "schema": f"test.{name}.v1",
        }
    audit = {
        "schema": "sglsim.global-des.audit.v1",
        "passed": True,
        "details": {
            "real_scheduler_integration": True,
            "claims": claims,
            "metrics": {"failures": {"affected_original_requests": 0}},
        },
    }
    audit_payload = _canonical(audit)
    (cell / "audit.json").write_bytes(audit_payload)
    files["audit.json"] = {
        "sha256": _sha(audit_payload),
        "row_count": 1,
        "schema": "sglsim.global-des.audit.v1",
    }
    manifest = {
        "schema": "sglsim.global-des-cell-manifest.v1",
        "lineage": {
            "suite_id": "aegiskv-paper-global-des-v3",
            "group": "P4",
            "cell_id": cell_id,
            "axes": {
                "algorithm_variant": algorithm_variant,
                "seed": 2026082900,
            },
            "claims": claims,
            "request_universe_hash": "unit-universe",
            "compiled_requests_hash": "unit-requests",
        },
        "files": files,
    }
    manifest["canonical_payload_sha256"] = _sha(_canonical(manifest))
    (cell / "manifest.json").write_bytes(_canonical(manifest))
    suite = {
        "schema": "aegiskv.global-des-scheduler-suite-run.v1",
        "suite_id": "aegiskv-paper-global-des-v3",
        "complete": True,
        "processed_cells": 1,
        "claims": {
            "paper_ready": False,
            "modeled_simulation": True,
            "real_hardware_measurement": False,
        },
        "cells": [
            {
                "group": "P4",
                "cell_id": cell_id,
                "status": "executed",
                "manifest": str((cell / "manifest.json").resolve()),
                "manifest_payload_sha256": manifest["canonical_payload_sha256"],
            }
        ],
    }
    (run / "suite_manifest.json").write_bytes(_canonical(suite))
    return run


def test_partial_smoke_emits_diagnostics_but_no_registered_figure(tmp_path) -> None:
    run = _publish_smoke(tmp_path)
    collection = collect_evidence([run], config_root=CONFIG_ROOT)
    ready, blockers = group_readiness(collection, "P4")
    assert ready is False
    assert any(value.startswith("missing_cells:39/40") for value in blockers)

    output = tmp_path / "plots"
    manifest = generate_plots(collection, output, representative_group="P4")
    assert manifest["claims"]["paper_ready"] is False
    assert manifest["registered_figures_complete"] is False
    assert "P4" in manifest["blockers"]
    assert all(row["figure_kind"] == "representative_mechanism_diagnostic" for row in manifest["figures"])
    assert {row["figure_id"] for row in manifest["figures"]} == {
        "diagnostic_per_dp_rank_state",
        "diagnostic_per_gpu_hbm",
        "diagnostic_per_node_dram",
        "diagnostic_kv_transfer_gantt",
        "diagnostic_failure_timeline",
    }
    for row in manifest["figures"]:
        figure_manifest = json.loads((output / row["manifest"]).read_text())
        assert figure_manifest["claims"]["paper_ready"] is False
        for suffix in ("csv", "png", "pdf"):
            assert (output / f"{row['figure_id']}.{suffix}").stat().st_size > 0


def test_raw_sha_mismatch_is_fail_closed(tmp_path) -> None:
    run = _publish_smoke(tmp_path)
    raw = run / "P4" / "P4-unit" / "failures.jsonl"
    raw.write_text(raw.read_text() + "{}\n")
    with pytest.raises(PlotEvidenceError, match="SHA256 mismatch"):
        collect_evidence([run], config_root=CONFIG_ROOT)


def test_duplicate_axis_identity_across_runs_is_rejected(tmp_path) -> None:
    left = _publish_smoke(tmp_path / "left", cell_id="P4-left")
    right = _publish_smoke(tmp_path / "right", cell_id="P4-right")
    with pytest.raises(PlotEvidenceError, match="duplicate cell identity"):
        collect_evidence([left, right], config_root=CONFIG_ROOT)


def test_treatment_without_baseline_is_reported_and_not_plotted(tmp_path) -> None:
    run = _publish_smoke(
        tmp_path,
        cell_id="P4-treatment-only",
        algorithm_variant="no_algorithm_1",
    )
    collection = collect_evidence([run], config_root=CONFIG_ROOT)
    ready, blockers = group_readiness(collection, "P4")
    assert ready is False
    assert any(value.startswith("missing_baseline:full") for value in blockers)
    manifest = generate_plots(
        collection, tmp_path / "plots", representative_group="P4"
    )
    assert not any(
        row["figure_kind"] == "registered_paper_figure"
        for row in manifest["figures"]
    )


def test_existing_output_is_never_overwritten(tmp_path) -> None:
    collection = collect_evidence(
        [_publish_smoke(tmp_path / "source")], config_root=CONFIG_ROOT
    )
    output = tmp_path / "plots"
    output.mkdir()
    marker = output / "user-data"
    marker.write_text("preserve")
    with pytest.raises(PlotEvidenceError, match="already exists"):
        generate_plots(collection, output, representative_group="P4")
    assert marker.read_text() == "preserve"
