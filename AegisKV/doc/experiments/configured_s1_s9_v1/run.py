#!/usr/bin/env python3
"""Validate all three-layer configs and regenerate the complete S1--S9 set."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for path in (REPO_ROOT, REPO_ROOT / "python", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from .config import (
        DEFAULT_CONFIG_ROOT,
        GROUPS,
        SOURCE_FILENAMES,
        group_manifest,
        load_all,
        sha256_file,
        validate_no_duplicate_axes,
        write_json,
    )
    from .plot_all import render_group, render_montage
except ImportError:
    from config import (  # type: ignore[no-redef]
        DEFAULT_CONFIG_ROOT,
        GROUPS,
        SOURCE_FILENAMES,
        group_manifest,
        load_all,
        sha256_file,
        validate_no_duplicate_axes,
        write_json,
    )
    from plot_all import render_group, render_montage  # type: ignore[no-redef]


DEFAULT_RESULTS_ROOT = HERE / "results"


def _unique_output_dir(results_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(results_root) / f"{stamp}_aegiskv_s1_s9"
    output.mkdir(parents=True, exist_ok=False)
    return output


def execute(config_root: Path, results_root: Path, *, validate_only: bool = False) -> Path | None:
    suite, groups = load_all(config_root)
    validate_no_duplicate_axes(groups)
    if validate_only:
        return None

    output = _unique_output_dir(results_root)
    snapshot_root = output / "config_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(config_root) / "suite.json", snapshot_root / "suite.json")
    for group in groups:
        target = snapshot_root / group.group
        target.mkdir(parents=True, exist_ok=False)
        for filename in SOURCE_FILENAMES:
            shutil.copy2(group.config_dir / filename, target / filename)

    figures_dir = output / "figures"
    figure_paths: list[Path] = []
    group_records = []
    for group in groups:
        paths = render_group(group, figures_dir)
        figure_paths.extend(paths)
        record = group_manifest(group)
        record["figures"] = [
            {
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
        group_records.append(record)

    montage_paths = render_montage(
        [path for path in figure_paths if path.suffix == ".png"],
        figures_dir,
    )
    figure_paths.extend(montage_paths)
    status = "completed_with_evidence_gaps"
    manifest = {
        "schema_version": "aegiskv.s1-s9-result-manifest.v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "consume_frozen_audited_result_and_render",
        "fresh_simulation_run": False,
        "paper_ready": False,
        "design_semantics": suite["design_semantics"],
        "groups": group_records,
        "nominal_cells_before_reuse": suite["nominal_cells_before_reuse"],
        "minimum_unique_cells": suite["minimum_unique_cells"],
        "figure_count": len(figure_paths),
        "montage": [
            {"path": str(path.relative_to(output)), "sha256": sha256_file(path)}
            for path in montage_paths
        ],
        "evidence_boundary": suite["evidence_boundary"],
    }
    write_json(output / "experiment_manifest.json", manifest)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the S1-S9 three-layer configs and render every figure"
    )
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = execute(
            args.config_root,
            args.results_root,
            validate_only=bool(args.validate_only),
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "validated" if output is None else "completed_with_evidence_gaps",
                "output_dir": None if output is None else str(output),
                "manifest": None if output is None else str(output / "experiment_manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
