from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = REPO_ROOT / "AegisKV" / "experiments" / "configured_s1_s9_v1"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_s1_s9_three_layer_configs_cover_frozen_matrix() -> None:
    config = _load_module("aegiskv_s1_s9_config", SUITE_ROOT / "config.py")
    suite, groups = config.load_all(SUITE_ROOT / "configs")
    config.validate_no_duplicate_axes(groups)

    assert [group.group for group in groups] == [f"S{i}" for i in range(1, 10)]
    assert sum(group.planned_cells for group in groups) == 843
    assert sum(group.unique_cells for group in groups) == 783
    assert suite["design_semantics"] == "source_owned_independent_hbm_dram_v2"
    assert suite["reference_threads"][0]["thread_id"] == "01a005a8-95b8-7c41-af1b-50d06b711944"
    assert all(group.source_rows > 0 for group in groups)
    assert all(group.deployment["evidence"]["paper_ready"] is False for group in groups)


def test_known_evidence_gaps_are_explicit() -> None:
    config = _load_module("aegiskv_s1_s9_config_gaps", SUITE_ROOT / "config.py")
    _, groups = config.load_all(SUITE_ROOT / "configs")
    by_group = {group.group: config.group_manifest(group) for group in groups}

    assert "analytic" in by_group["S3"]["known_gaps"][0]
    assert "real prefix-identity workload" in by_group["S6"]["known_gaps"][0]
    assert by_group["S8"]["reuse_from"] == "S1"
    assert by_group["S8"]["unique_cells"] == 0
    assert by_group["S9"]["calibration_acceptance"] == "FAIL"


def test_render_all_creates_immutable_manifest_and_all_figures(tmp_path: Path) -> None:
    runner = _load_module("aegiskv_s1_s9_runner", SUITE_ROOT / "run.py")
    output = runner.execute(SUITE_ROOT / "configs", tmp_path)
    assert output is not None

    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["status"] == "completed_with_evidence_gaps"
    assert manifest["fresh_simulation_run"] is False
    assert manifest["paper_ready"] is False
    assert len(manifest["groups"]) == 9
    assert manifest["figure_count"] == 20
    group_figures = [item for group in manifest["groups"] for item in group["figures"]]
    assert sum(item["path"].endswith(".png") for item in group_figures) == 9
    assert sum(item["path"].endswith(".pdf") for item in group_figures) == 9
    assert (output / "figures" / "s1_s9_overview_montage.png").is_file()
    assert (output / "figures" / "s1_s9_overview_montage.pdf").is_file()
    for group in [f"S{i}" for i in range(1, 10)]:
        snapshot = output / "config_snapshot" / group
        assert {path.name for path in snapshot.glob("*.json")} == {
            "request.json",
            "failure.json",
            "deployment.json",
        }
