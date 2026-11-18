"""Command-line orchestration for the three-layer AegisKV suite."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audit import audit_lineage
from .config import (
    CONFIG_ROOT,
    EXPERIMENT_ROOT,
    PARAMETER_PROVENANCE_PATH,
    SOURCE_REGISTRY_PATH,
    canonical_json_bytes,
    iter_config_files,
    load_suite,
    matrix_rows,
    sha256_bytes,
    sha256_file,
)
from .plot import render_all
from .report import write_analysis_report
from .simulation import run_cell


RESULTS_ROOT = EXPERIMENT_ROOT / "results"
REPO_ROOT = EXPERIMENT_ROOT.parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _write_gzip_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(
        filename=str(path), mode="wb", compresslevel=6, mtime=0
    ) as handle:
        handle.write(canonical_json_bytes(value))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    keys = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _lineage_dir(results_root: Path, profile: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = Path(results_root) / f"{stamp}_{profile}_complete_v2"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = Path(results_root) / f"{stamp}_{profile}_complete_v2_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def fetch_data() -> dict[str, Any]:
    registry = json.loads(SOURCE_REGISTRY_PATH.read_text(encoding="utf-8"))["sources"]
    source = registry["tracelab_release"]
    destination = EXPERIMENT_ROOT / "data" / "cache" / "syfi_coding_trace-v0.0.2.duckdb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == source["sha256"]:
        return {"status": "present", "path": str(destination), "sha256": source["sha256"]}
    temporary = destination.with_suffix(".download")
    if temporary.exists():
        temporary.unlink()
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--output",
                str(temporary),
                str(source["url"]),
            ],
            check=True,
        )
        actual = sha256_file(temporary)
        if actual != source["sha256"]:
            raise RuntimeError(f"download SHA-256 mismatch: {actual}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "downloaded", "path": str(destination), "sha256": source["sha256"]}


def _snapshot_configs(lineage: Path, loaded: Any, selected_groups: list[str]) -> dict[str, str]:
    raw_root = lineage / "config_snapshot"
    hashes: dict[str, str] = {}
    for path in iter_config_files(CONFIG_ROOT):
        relative = path.relative_to(CONFIG_ROOT)
        if relative.parts[0] not in {"suite.json", "common", *selected_groups}:
            continue
        target = raw_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha256_file(path)
    shutil.copy2(SOURCE_REGISTRY_PATH, raw_root / "source_registry.json")
    hashes["source_registry.json"] = sha256_file(SOURCE_REGISTRY_PATH)
    shutil.copy2(
        PARAMETER_PROVENANCE_PATH, raw_root / "parameter_provenance.json"
    )
    hashes["parameter_provenance.json"] = sha256_file(
        PARAMETER_PROVENANCE_PATH
    )
    for name in (
        "experiment_config_rationale.md",
        "reviewer_configuration_audit.md",
        "physical_calibration_status.md",
    ):
        source = EXPERIMENT_ROOT / "research" / name
        shutil.copy2(source, raw_root / name)
        hashes[name] = sha256_file(source)
    design_source = EXPERIMENT_ROOT.parent / "doc" / "design" / "current_design.md"
    shutil.copy2(design_source, raw_root / "current_design.md")
    hashes["current_design.md"] = sha256_file(design_source)
    for group in loaded.groups:
        if group.group not in selected_groups:
            continue
        target = lineage / "resolved_configs" / group.group
        _write_json(target / "request.json", group.request)
        _write_json(target / "failure.json", group.failure)
        _write_json(target / "deployment.json", group.deployment)
    return hashes


def _snapshot_execution_code(lineage: Path) -> dict[str, str]:
    paths = [
        EXPERIMENT_ROOT / "run.py",
        *sorted((EXPERIMENT_ROOT / "src").rglob("*.py")),
        REPO_ROOT / "python/sglsim/srt/simulation/aegis_complete.py",
        REPO_ROOT / "python/sglsim/srt/simulation/aegis_full_design.py",
        REPO_ROOT / "python/sglsim/srt/simulation/aegis_gbdt.py",
        REPO_ROOT / "python/sglsim/srt/simulation/aegis_independent_tiers.py",
        REPO_ROOT / "python/sglsim/srt/simulation/aegis_residual_dag.py",
        REPO_ROOT / "python/sglsim/srt/simulation/aegis_transfer_engine.py",
    ]
    hashes: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(REPO_ROOT)
        target = lineage / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha256_file(path)
    return hashes


def run_suite(
    *,
    profile: str,
    groups: list[str] | None,
    results_root: Path,
) -> Path:
    loaded = load_suite(CONFIG_ROOT)
    selected = groups or list(loaded.suite["groups"])
    unknown = set(selected) - set(loaded.suite["groups"])
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    lineage = _lineage_dir(results_root, profile)
    config_hashes = _snapshot_configs(lineage, loaded, selected)
    code_hashes = _snapshot_execution_code(lineage)
    profile_config = loaded.suite["profiles"][profile]
    summaries: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    timelines: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    shadow_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    shadow_paths: dict[tuple[Any, ...], str] = {}
    expected = 0
    for group in loaded.groups:
        if group.group not in selected:
            continue
        matrix = matrix_rows(group, loaded.suite, profile)
        expected += len(matrix)
        for index, axes in enumerate(matrix, 1):
            print(f"[{group.group} {index}/{len(matrix)}] {json.dumps(axes, sort_keys=True)}", flush=True)
            shadow_key = (
                group.group,
                axes.get("seed"),
                axes.get("model"),
                axes.get("context_tokens", group.request["workload"].get("context_tokens")),
                axes.get("load_multiplier", 1.0),
            )
            cell = run_cell(
                group,
                axes,
                suite_profile=profile_config,
                shadow_result=shadow_cache.get(shadow_key),
                experiment_root=EXPERIMENT_ROOT,
            )
            shadow_cache.setdefault(shadow_key, cell.shadow_result)
            if shadow_key not in shadow_paths:
                shadow_sha = sha256_bytes(canonical_json_bytes(cell.shadow_result))
                shadow_relative = f"raw/shadows/{shadow_sha[:16]}.json.gz"
                _write_gzip_json(lineage / shadow_relative, cell.shadow_result)
                shadow_paths[shadow_key] = shadow_relative
            raw_relative = f"raw/cells/{cell.cell_id}.json.gz"
            _write_gzip_json(
                lineage / raw_relative,
                {
                    "cell_id": cell.cell_id,
                    "axes": axes,
                    "workload_metadata": cell.workload_metadata,
                    "failure_metadata": cell.failure_metadata,
                    "shadow_artifact": shadow_paths[shadow_key],
                    "result": cell.raw_result,
                },
            )
            summaries.append(
                {
                    **cell.summary,
                    "raw_artifact": raw_relative,
                    "shadow_artifact": shadow_paths[shadow_key],
                }
            )
            observations.extend(cell.observations)
            timelines.extend(cell.timeline)
            verifications.extend(cell.verification)

    _write_csv(lineage / "summary.csv", summaries)
    _write_csv(lineage / "observations.csv", observations)
    _write_csv(lineage / "timeline.csv", timelines)
    _write_csv(lineage / "verification.csv", verifications)
    manifest = {
        "schema_version": "aegiskv.experiment-lineage.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "groups": selected,
        "expected_cell_count": expected,
        "completed_cell_count": len(summaries),
        "config_sha256": config_hashes,
        "code_sha256": code_hashes,
        "execution_engine": loaded.suite["execution_engine"],
        "design_semantics": loaded.suite["design_semantics"],
        "fresh_simulation_run": True,
        "consumed_archived_preliminary_csv": False,
        "metadata_only": True,
        "physical_result": False,
        "plots_rendered": False,
        "paper_ready_calibrated": False,
    }
    _write_json(lineage / "manifest.json", manifest)
    return lineage


def plot_lineage(lineage: Path) -> list[Path]:
    figures = render_all(lineage)
    manifest_path = lineage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plots_rendered"] = True
    manifest["figures"] = [
        {"path": str(path.relative_to(lineage)), "sha256": sha256_file(path)} for path in figures
    ]
    _write_json(manifest_path, manifest)
    return figures


def audit_and_write(lineage: Path) -> dict[str, Any]:
    audit = audit_lineage(lineage)
    _write_json(lineage / "audit.json", audit)
    write_analysis_report(lineage, audit)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("fetch-data")
    for command in ("run", "all"):
        run = sub.add_parser(command)
        run.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
        run.add_argument("--groups", nargs="*")
        run.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    for command in ("plot", "audit"):
        action = sub.add_parser(command)
        action.add_argument("--lineage", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            loaded = load_suite(CONFIG_ROOT)
            payload = {
                "status": "PASS",
                "groups": list(loaded.suite["groups"]),
                "smoke_cells": sum(len(matrix_rows(group, loaded.suite, "smoke")) for group in loaded.groups),
                "paper_cells": sum(len(matrix_rows(group, loaded.suite, "paper")) for group in loaded.groups),
            }
        elif args.command == "fetch-data":
            payload = fetch_data()
        elif args.command in {"run", "all"}:
            lineage = run_suite(profile=args.profile, groups=args.groups, results_root=args.results_root)
            if args.command == "all":
                plot_lineage(lineage)
                audit = audit_and_write(lineage)
            else:
                audit = None
            payload = {"status": "completed", "lineage": str(lineage), "audit": audit}
        elif args.command == "plot":
            figures = plot_lineage(args.lineage)
            payload = {"status": "plotted", "lineage": str(args.lineage), "figure_count": len(figures)}
        elif args.command == "audit":
            audit = audit_and_write(args.lineage)
            payload = audit
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.command == "audit" and payload.get("semantic_and_reproducibility_status") != "PASS":
        return 1
    if args.command == "all" and payload.get("audit", {}).get("semantic_and_reproducibility_status") != "PASS":
        return 1
    return 0


__all__ = ["audit_and_write", "fetch_data", "main", "plot_lineage", "run_suite"]
