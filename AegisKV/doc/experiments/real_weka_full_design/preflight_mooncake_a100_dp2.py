#!/usr/bin/env python3
"""Validate and import one A100-DP2 Mooncake bundle without replaying it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    A100DP2InputPaths,
    artifact_record,
    sha256_file,
    validate_a100_dp2_bundle,
)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def preflight_and_write(*, inputs: A100DP2InputPaths, output_root: Path) -> Path:
    """Run the complete fail-closed preflight, then create immutable output.

    No output path is resolved into a directory and no filesystem object is
    created until ``validate_a100_dp2_bundle`` has re-hashed and validated the
    complete producer lineage.  This function never loads a trace, fits a
    model, creates a simulator runtime, or runs a replay.
    """

    result = validate_a100_dp2_bundle(inputs)
    directory_name = f"a100_dp2_preflight_{result.lineage_identity_sha256[:16]}"
    output_root = output_root.resolve()
    output_dir = output_root / directory_name
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable A100-DP2 preflight: {output_dir}"
        )

    manifest = result.manifest()
    manifest["output_directory_name"] = directory_name
    manifest["output_directory_identity_sha256"] = (
        result.lineage_identity_sha256
    )

    # Validation above is the ordering fence: only now may the independent
    # A100 preflight output directory be created.
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    profile_path = output_dir / "a100-dp2-profile.json"
    _write_new(
        profile_path,
        _json_bytes(
            {
                "schema_version": 2,
                "profile_schema_class": "A100DP2PhysicalProfile",
                "consumer_compatibility_identity": manifest[
                    "consumer_compatibility_identity"
                ],
                "hardware_label": manifest["hardware_label"],
                "design_semantics": manifest["design_semantics"],
                "physical_profile_semantics": manifest[
                    "physical_profile_semantics"
                ],
                "runtime_admission_decision_wire_values": manifest[
                    "runtime_admission_decision_wire_values"
                ],
                "model": manifest["model"],
                "dtype": manifest["dtype"],
                "layout": manifest["layout"],
                "model_revision": manifest["model_revision"],
                "transfer_cost_model_schema": manifest[
                    "transfer_cost_model_schema"
                ],
                "runtime_consumption_contract": manifest[
                    "runtime_consumption_contract"
                ],
                "metrics": manifest["profile_metrics"],
                "source_profile": manifest["input_artifacts"][
                    "physical_profile"
                ],
                "authorization": manifest["authorization"],
            }
        ),
    )
    manifest["normalized_profile"] = artifact_record(profile_path)
    manifest_path = output_dir / "preflight-manifest.json"
    _write_new(manifest_path, _json_bytes(manifest))
    if sha256_file(profile_path) != manifest["normalized_profile"]["sha256"]:
        raise RuntimeError("written A100-DP2 profile hash changed")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--model-submission-gate", type=Path, required=True)
    parser.add_argument("--config-manifest", type=Path, required=True)
    parser.add_argument("--config-verification", type=Path, required=True)
    parser.add_argument("--physical-profile", type=Path, required=True)
    parser.add_argument("--physical-profile-audit", type=Path, required=True)
    parser.add_argument("--aimd-stall-calibration", type=Path, required=True)
    parser.add_argument("--mechanism-gate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_dir = preflight_and_write(
        inputs=A100DP2InputPaths(
            source_manifest=args.source_manifest,
            runtime_binding=args.runtime_binding,
            model_submission_gate=args.model_submission_gate,
            config_manifest=args.config_manifest,
            config_verification=args.config_verification,
            physical_profile=args.physical_profile,
            physical_profile_audit=args.physical_profile_audit,
            aimd_stall_calibration=args.aimd_stall_calibration,
            mechanism_gate=args.mechanism_gate,
        ),
        output_root=args.output_root,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
