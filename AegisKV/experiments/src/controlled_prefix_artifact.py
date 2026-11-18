"""Build and audit the independent P6 controlled-prefix input artifact.

The artifact is intentionally synthetic.  It defines deterministic token-stream
identities and an arrival schedule, but it is not an observation from TraceLab
or from a serving deployment.  Keeping it separate prevents the P6 reuse sweep
from acquiring false empirical provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_PREFIX_PATH = (
    EXPERIMENT_ROOT
    / "data"
    / "controlled_prefix"
    / "controlled_prefix_reuse-v1.json"
)
CONTROLLED_PREFIX_DATASET_ID = "aegiskv-controlled-prefix-reuse-v1"
CONTROLLED_PREFIX_SCHEMA = "aegiskv.controlled-prefix-artifact.v1"
CONTROLLED_PREFIX_GENERATOR = "aegiskv.controlled-prefix-generator.v1"
CONTROLLED_PREFIX_SHA256 = (
    "26b01d51a799c9c82b786525234d7c7853a65f9317c88f12292ca237333a5afe"
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _stream_digest(preimage: str) -> str:
    return hashlib.sha256(preimage.encode("ascii")).hexdigest()


def build_controlled_prefix_artifact() -> dict[str, object]:
    """Return the complete compact generator specification for P6.

    Four content families each contain eight generated sessions.  A request
    cell selects two sessions from every family, then replays all sixteen rounds
    from each selected session.  These cardinalities are preregistered simulator
    choices, not measured properties.
    """

    families = []
    for family_index in range(4):
        family_id = f"family-{family_index:02d}"
        preimage = (
            f"{CONTROLLED_PREFIX_DATASET_ID}:family={family_index}:shared-prefix"
        )
        families.append(
            {
                "family_id": family_id,
                "prefix_stream_preimage": preimage,
                "prefix_stream_sha256": _stream_digest(preimage),
            }
        )
    return {
        "schema_version": CONTROLLED_PREFIX_SCHEMA,
        "dataset_id": CONTROLLED_PREFIX_DATASET_ID,
        "generator": {
            "schema_version": CONTROLLED_PREFIX_GENERATOR,
            "implementation": (
                "AegisKV/experiments/src/controlled_prefix_artifact.py"
            ),
            "empirical_measurement": False,
            "provenance": (
                "Preregistered deterministic synthetic workload for isolating "
                "prefix-reuse effects; not TraceLab data and not a measured trace."
            ),
        },
        "token_identity": {
            "algorithm": "sha256_counter_u32_mod_vocab",
            "vocab_size": 151936,
            "content_bytes_materialized": False,
        },
        "workload_shape": {
            "content_family_count": 4,
            "sessions_per_family": 8,
            "selected_sessions_per_family": 2,
            "rounds_per_session": 16,
            "input_tokens_per_request": 15872,
            "controllable_prefix_tokens": 11904,
            "output_tokens_per_request": 512,
            "round_spacing_ns": 250000000,
            "session_stagger_ns": 10000000,
            "parameter_provenance": {
                "cardinalities": (
                    "Preregistered balanced-factorial simulation choice; not measured."
                ),
                "token_counts": (
                    "Preregistered 16K-context controlled point: 15,872 input plus "
                    "512 output tokens; not measured."
                ),
                "arrival_spacing": (
                    "Preregistered deterministic schedule used to isolate reuse; "
                    "not measured."
                ),
            },
        },
        "session_identity_recipe": {
            "session_id_format": "cp-family-{family_index:02d}-session-{session_index:02d}",
            "suffix_stream_preimage_format": (
                "aegiskv-controlled-prefix-reuse-v1:family={family_index}:"
                "session={session_index}:round={round_index}:reuse={reuse_numerator}/4:"
                "private-suffix"
            ),
            "complete_sessions_only": True,
        },
        "content_families": families,
    }


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(values: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON number {item} in {path}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"top-level JSON object required in {path}")
    return value


def audit_controlled_prefix_artifact(
    path: Path = CONTROLLED_PREFIX_PATH,
    *,
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, Any], str]:
    """Verify exact bytes and reproducibility against the checked-in generator."""

    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(
            f"controlled-prefix SHA-256 mismatch: expected {expected_sha256}, got {sha256}"
        )
    parsed = _strict_json(resolved)
    expected = build_controlled_prefix_artifact()
    if parsed != expected:
        raise ValueError(
            "controlled-prefix artifact does not match its deterministic generator"
        )
    if data != canonical_json_bytes(expected):
        raise ValueError("controlled-prefix artifact is not canonical generator output")
    return parsed, sha256


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONTROLLED_PREFIX_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing artifact instead of writing it",
    )
    args = parser.parse_args(argv)
    if args.check:
        _artifact, digest = audit_controlled_prefix_artifact(args.output)
        print(f"PASS {args.output} sha256={digest}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(build_controlled_prefix_artifact()))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line wrapper
    raise SystemExit(_main())


__all__ = [
    "CONTROLLED_PREFIX_DATASET_ID",
    "CONTROLLED_PREFIX_GENERATOR",
    "CONTROLLED_PREFIX_PATH",
    "CONTROLLED_PREFIX_SCHEMA",
    "CONTROLLED_PREFIX_SHA256",
    "audit_controlled_prefix_artifact",
    "build_controlled_prefix_artifact",
    "canonical_json_bytes",
]
