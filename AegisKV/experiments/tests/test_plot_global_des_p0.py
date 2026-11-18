from __future__ import annotations

import csv
import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest

from AegisKV.experiments.src.global_des_config import (
    P0_SCENARIOS,
    REQUIRED_INVARIANTS,
)
from AegisKV.experiments.src.plot_global_des_p0 import (
    CLAIM_SCOPE,
    OUTPUT_SCHEMA,
    P0SourceValidationError,
    main,
    plot_p0,
)
from AegisKV.experiments.src.run_global_des_p0 import run_p0
from sglsim.srt.simulation.global_des_artifacts import EVIDENCE_CLASS


SLUGS = {
    "event_order_trace",
    "gpu_hbm_accounts",
    "node_dram_accounts",
    "kv_transfer_gantt",
    "invariant_matrix",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def rendered_p0(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("plot-global-des-p0")
    run_dir = root / "run"
    assert run_p0(run_dir)["status"] == "PASS"
    p0 = run_dir / "P0"
    manifest = plot_p0(p0)
    assert manifest["status"] == "success"
    return p0, p0 / "figures"


def test_publishes_five_csv_png_pdf_bundles_and_auditable_manifest(
    rendered_p0: tuple[Path, Path],
) -> None:
    p0, figures = rendered_p0
    expected = {"manifest.json"} | {
        f"{slug}.{extension}"
        for slug in SLUGS
        for extension in ("csv", "png", "pdf")
    }
    assert {path.name for path in figures.iterdir()} == expected

    manifest = _json(figures / "manifest.json")
    assert manifest["schema"] == OUTPUT_SCHEMA
    assert manifest["status"] == "success"
    assert manifest["evidence_class"] == EVIDENCE_CLASS
    assert manifest["claim_scope"] == CLAIM_SCOPE
    assert manifest["source"]["status"] == "PASS"
    assert manifest["source"]["contract_cells"] == 12
    assert manifest["source"]["invariants_per_cell"] == 10
    assert manifest["source"]["p0_summary_sha256"] == _sha256(p0 / "summary.json")
    assert manifest["claims"] == {
        "contract_execution_only": True,
        "full_scheduler_integration": False,
        "controller_bridge_target_rotation_closed_loop": True,
        "performance_result": False,
        "real_hardware_measurement": False,
    }
    assert set(manifest["files"]) == expected - {"manifest.json"}
    for name, descriptor in manifest["files"].items():
        path = figures / name
        assert descriptor["sha256"] == _sha256(path)
        assert descriptor["size_bytes"] == path.stat().st_size


def test_png_and_pdf_outputs_are_real_rendered_documents(
    rendered_p0: tuple[Path, Path],
) -> None:
    _, figures = rendered_p0
    for slug in SLUGS:
        png = (figures / f"{slug}.png").read_bytes()
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        assert width >= 900 and height >= 500
        pdf = (figures / f"{slug}.pdf").read_bytes()
        assert pdf.startswith(b"%PDF-")
        assert len(pdf) > 5_000


def test_csvs_preserve_lineage_and_cover_the_required_contract_witnesses(
    rendered_p0: tuple[Path, Path],
) -> None:
    p0, figures = rendered_p0
    summary = _json(p0 / "summary.json")
    summary_sha = _sha256(p0 / "summary.json")
    for slug in SLUGS:
        rows = _csv(figures / f"{slug}.csv")
        assert rows
        assert {row["evidence_class"] for row in rows} == {EVIDENCE_CLASS}
        assert {row["claim_scope"] for row in rows} == {CLAIM_SCOPE}
        assert {row["suite_id"] for row in rows} == {summary["suite_id"]}
        assert {row["source_code_hash"] for row in rows} == {summary["code_hash"]}
        assert {row["source_p0_summary_sha256"] for row in rows} == {summary_sha}
        assert all(len(row["source_manifest_payload_sha256"]) == 64 for row in rows)
        assert all(len(row["source_event_content_sha256"]) == 64 for row in rows)

    event_rows = _csv(figures / "event_order_trace.csv")
    tie = [
        row
        for row in event_rows
        if row["contract_scenario"] == "stable_same_timestamp_tie_hash"
    ]
    assert [row["event_kind"] for row in tie] == [
        "same_timestamp_failure",
        "same_timestamp_control_first",
        "same_timestamp_control_second",
    ]
    assert len({row["timestamp_ns"] for row in tie}) == 1
    assert [row["priority"] for row in tie] == ["0", "5", "5"]

    hbm = _csv(figures / "gpu_hbm_accounts.csv")
    assert {row["contract_scenario"] for row in hbm} == {
        "failure_after_commit_source_first",
        "failure_in_pending_window",
        "target_first_and_node_failure",
    }
    rotated = [
        row
        for row in hbm
        if row["contract_scenario"] == "target_first_and_node_failure"
    ]
    assert {row["dp_rank"] for row in rotated} == {"1", "2"}
    assert any(row["reason"] == "gpu_failed" for row in rotated)
    assert max(float(row["aegis_pending_gib"]) for row in hbm) > 0
    assert max(float(row["aegis_committed_gib"]) for row in hbm) > 0
    assert {row["reason"] for row in hbm} >= {
        "transfer_reserved",
        "transfer_committed",
        "transfer_aborted",
    }

    dram = _csv(figures / "node_dram_accounts.csv")
    assert {row["node_id"] for row in dram} == {
        "node0",
        "node1",
        "node2",
        "node3",
    }
    assert max(float(row["aegis_pending_gib"]) for row in dram) > 0
    assert max(float(row["aegis_committed_gib"]) for row in dram) > 0
    assert any(row["healthy"] == "False" for row in dram)

    transfers = _csv(figures / "kv_transfer_gantt.csv")
    assert len(transfers) == 5
    assert {row["contract_scenario"] for row in transfers} == {
        "same_node_peer_hbm_path",
        "cross_node_peer_hbm_path",
        "source_to_node_dram_path",
        "node_dram_to_recovery_gpu_path",
        "failure_in_pending_window",
    }
    assert all(row["resource_path_exact"] for row in transfers)
    assert {row["final_state"] for row in transfers} == {"COMMITTED", "ABORTED"}

    matrix = _csv(figures / "invariant_matrix.csv")
    assert len(matrix) == 12 * 10
    assert {row["contract_scenario"] for row in matrix} == set(P0_SCENARIOS)
    assert {row["invariant"] for row in matrix} == set(REQUIRED_INVARIANTS)
    assert {row["passed"] for row in matrix} == {"True"}
    assert all(len(row["evidence_sha256"]) == 64 for row in matrix)


def test_refuses_overwrite_without_touching_existing_directory(
    rendered_p0: tuple[Path, Path], tmp_path: Path
) -> None:
    p0, _ = rendered_p0
    existing = tmp_path / "figures"
    existing.mkdir()
    sentinel = existing / "user-data"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        plot_p0(p0, figures_dir=existing)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert main(["--p0-dir", str(p0), "--figures-dir", str(existing)]) == 2
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_same_p0_source_renders_byte_identical_diagnostics(
    rendered_p0: tuple[Path, Path], tmp_path: Path
) -> None:
    p0, first = rendered_p0
    second = tmp_path / "repeat"

    assert plot_p0(p0, figures_dir=second)["status"] == "success"

    assert {path.name for path in first.iterdir()} == {
        path.name for path in second.iterdir()
    }
    for first_file in first.iterdir():
        assert first_file.read_bytes() == (second / first_file.name).read_bytes()


def test_rejects_a_source_cell_whose_manifest_checksum_no_longer_matches(
    rendered_p0: tuple[Path, Path], tmp_path: Path
) -> None:
    p0, _ = rendered_p0
    corrupt = tmp_path / "P0"
    shutil.copytree(p0, corrupt, ignore=shutil.ignore_patterns("figures"))
    events = corrupt / "same_node_peer_hbm_path" / "events.jsonl"
    events.write_bytes(events.read_bytes() + b"\n")

    with pytest.raises(P0SourceValidationError, match="SHA256 mismatch"):
        plot_p0(corrupt, figures_dir=tmp_path / "not-published")

    assert not (tmp_path / "not-published").exists()
