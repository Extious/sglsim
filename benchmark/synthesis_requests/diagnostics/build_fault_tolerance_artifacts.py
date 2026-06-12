from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PYTHON = REPO / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from benchmark.synthesis_requests.diagnostics.fault_tolerance_paper_tables import (
    write_tables,
)
from benchmark.synthesis_requests.diagnostics.plot_fault_tolerance_sweeps import plot


REQUIRED_OUTPUTS = (
    "fault_tolerance_sweep_hpca.png",
    "fault_tolerance_sweep_hpca.pdf",
    "fault_tolerance_sweep_system.png",
    "fault_tolerance_sweep_system.pdf",
    "fault_tolerance_sweep_summary.json",
    "fault_tolerance_sweep_raw_summary.json",
    "fault_tolerance_paper_tables.json",
    "fault_tolerance_paper_tables.md",
    "fault_tolerance_paper_tables.tex",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    entries = _read_json(manifest_path)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no entries: {manifest_path}")
    missing: list[str] = []
    for entry in entries:
        run_dir = Path(str(entry.get("run_dir", "")))
        for filename in ("metrics.json", "request_details.csv"):
            path = run_dir / filename
            if not path.is_file():
                missing.append(str(path))
    if missing:
        formatted = "\n".join(missing)
        raise FileNotFoundError(f"Missing run artifacts:\n{formatted}")
    return entries


def _axis_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        axis = str(entry.get("axis", ""))
        counts[axis] = counts.get(axis, 0) + 1
    return counts


def _validate_outputs(output_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(str(path))
        else:
            outputs[name] = str(path)
    if missing:
        formatted = "\n".join(missing)
        raise FileNotFoundError(f"Missing generated artifacts:\n{formatted}")
    return outputs


def build_artifacts(manifest_path: Path, output_dir: Path) -> Path:
    entries = _check_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot(manifest_path, output_dir)
    summary_path = output_dir / "fault_tolerance_sweep_summary.json"
    write_tables(summary_path, output_dir)

    summary = _read_json(summary_path)
    paper_tables = _read_json(output_dir / "fault_tolerance_paper_tables.json")
    artifacts = _validate_outputs(output_dir)
    bundle = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "num_runs": len(entries),
        "axis_counts": _axis_counts(entries),
        "num_summary_rows": len(summary),
        "strategy_summary": paper_tables.get("strategy", {}),
        "artifacts": artifacts,
    }
    bundle_path = output_dir / "fault_tolerance_artifacts.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FT paper artifact bundle")
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/synthesis_requests/results/ft_hpca_figures"),
    )
    args = parser.parse_args(argv)
    bundle = build_artifacts(args.manifest, args.output_dir)
    print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
