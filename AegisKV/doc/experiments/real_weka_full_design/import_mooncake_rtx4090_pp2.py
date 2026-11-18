#!/usr/bin/env python3
"""Import the isolated RTX4090 PP2 Mooncake snapshot without replaying it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
for search_root in (REPO_ROOT, REPO_ROOT / "python"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from AegisKV.experiments.real_weka_full_design.rtx4090_pp2_contract import (
    DESIGN_SEMANTICS,
    HARDWARE_LABEL,
    IMPORT_EVIDENCE_CLASS,
    PROFILE_SEMANTICS,
    RUNTIME_CONSUMPTION_CONTRACT,
    build_profile,
    default_input_paths,
    sha256_file,
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def import_snapshot(*, mooncake_root: Path, output_root: Path) -> Path:
    """Validate every producer artifact, then create one immutable import dir."""

    profile = build_profile(default_input_paths(mooncake_root))
    profile_bytes = _json_bytes(profile)
    profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    directory_name = f"rtx4090_pp2_{profile_sha256[:16]}"
    output_root = output_root.resolve()
    output_dir = output_root / directory_name
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable RTX4090 import: {output_dir}"
        )

    # No filesystem output is created before the complete producer snapshot
    # has passed build_profile above.
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    profile_path = output_dir / "profile.json"
    profile_path.write_bytes(profile_bytes)
    if sha256_file(profile_path) != profile_sha256:
        raise RuntimeError("written RTX4090 profile hash changed")

    manifest = {
        "schema_version": 1,
        "evidence_class": IMPORT_EVIDENCE_CLASS,
        "status": "PASS",
        "design_semantics": DESIGN_SEMANTICS,
        "profile_semantics": PROFILE_SEMANTICS,
        "hardware_label": HARDWARE_LABEL,
        "output_directory_name": directory_name,
        "output_directory_identity_sha256": profile_sha256,
        "output_profile": {
            "path": str(profile_path.resolve()),
            "sha256": profile_sha256,
            "size_bytes": profile_path.stat().st_size,
        },
        "lineage_inputs": profile["lineage_inputs"],
        "lineage_input_sha256": profile["lineage_input_sha256"],
        "runtime_consumption_contract": RUNTIME_CONSUMPTION_CONTRACT,
        "transport_contract": dict(profile["transport_observation"]),
        "recovery_transport_contract": dict(
            profile["recovery_transport_observation"]
        ),
        "p2p_topology_diagnostic": dict(profile["p2p_topology_diagnostic"]),
        "evidence_role": "preliminary_targeted_physical_diagnostic",
        "default_replay_authorized": False,
        "formal_replay_authorized": False,
        "figure_generation_authorized": False,
        "paper_performance_claim_authorized": False,
        "mixed_v100_or_a100_lineage": False,
        "formal_ready": False,
        "paper_ready": False,
    }
    manifest_path = output_dir / "import-manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mooncake-root",
        type=Path,
        default=Path("/home/comp/zhanzhao/Mooncake"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_dir = import_snapshot(
        mooncake_root=args.mooncake_root,
        output_root=args.output_root,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
