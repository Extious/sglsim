#!/usr/bin/env python3
"""Bind one immutable baseline suite to a validated V100 full replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AegisKV.experiments.real_weka_baselines.run import (
    V100_HARDWARE_LABEL,
    V100_PROFILE_SEMANTICS,
    preflight_baseline_suite,
)
from AegisKV.experiments.real_weka_full_design.run import (
    preflight_v100_config,
)
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS


ROOT = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def build_suite_binding(
    *,
    template_path: Path,
    base_config_path: Path,
    import_manifest_path: Path,
    anchor_result_dir: Path,
) -> dict[str, Any]:
    template_path = template_path.resolve()
    base_config_path = base_config_path.resolve()
    import_manifest_path = import_manifest_path.resolve()
    anchor_result_dir = anchor_result_dir.resolve()
    template = _load_json(template_path)
    if template.get("status") == "READY":
        raise ValueError("refusing to repurpose a READY baseline suite")
    _config, _runtime, _provenance, _projection, binding = (
        preflight_v100_config(base_config_path, import_manifest_path)
    )
    expected_anchor = (
        "full_393_workflows_v100_"
        f"{binding['physical_profile_sha256'][:12]}"
    )
    if not anchor_result_dir.is_dir() or anchor_result_dir.name != expected_anchor:
        raise ValueError("full replay anchor is not derived from the V100 profile")
    hashes_path = anchor_result_dir / "artifact_hashes.json"
    policy_keys = (
        "fidelity_contract",
        "primary_policies",
        "dedicated_controls",
        "ghostserve",
        "lumen",
        "restore_controls",
        "statistics",
        "audit",
    )
    missing = [key for key in policy_keys if key not in template]
    if missing:
        raise ValueError(f"baseline policy template is incomplete: {missing}")
    return {
        "schema_version": 2,
        "evidence_class": "sglsim_v100_baseline_suite_binding",
        "experiment_id": (
            "real-weka-capacity-matched-baselines-v100-"
            f"{binding['physical_profile_sha256'][:12]}"
        ),
        "status": "READY",
        "design_semantics": DESIGN_SEMANTICS,
        "hardware_label": V100_HARDWARE_LABEL,
        "profile_semantics": V100_PROFILE_SEMANTICS,
        "base_experiment_config": _artifact(base_config_path),
        "base_import_manifest": _artifact(import_manifest_path),
        "anchor_result": {
            "path": str(anchor_result_dir),
            "artifact_hashes": _artifact(hashes_path),
            "config_sha256": binding["config"]["sha256"],
            "import_manifest_sha256": binding["import_manifest"]["sha256"],
            "physical_profile_sha256": binding["physical_profile_sha256"],
            "source_manifest_sha256": binding["source_manifest_sha256"],
            "config_manifest_sha256": binding["config_manifest_sha256"],
            "config_verification_sha256": binding[
                "config_verification_sha256"
            ],
        },
        **{key: template[key] for key in policy_keys},
        "paper_ready": False,
        "formal_ready": False,
        "evidence_boundary": (
            "Hash-bound input for a metadata-only capacity-matched baseline "
            "replay; it does not authorize a physical or paper-ready claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=ROOT / "config.json")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-import-manifest", type=Path, required=True)
    parser.add_argument("--anchor-result", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_config.resolve()
    if output.exists():
        raise FileExistsError(f"baseline suite binding is immutable: {output}")
    suite = build_suite_binding(
        template_path=args.template,
        base_config_path=args.base_config,
        import_manifest_path=args.base_import_manifest,
        anchor_result_dir=args.anchor_result,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(suite, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(serialized)
    try:
        preflight_baseline_suite(temporary)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"status": "PASS", "output_config": str(output)}, indent=2))


if __name__ == "__main__":
    main()
