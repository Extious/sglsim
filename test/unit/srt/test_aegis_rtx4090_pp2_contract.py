from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from AegisKV.experiments.real_weka_full_design.consume_mooncake_rtx4090_pp2 import (
    materialize_diagnostic_runtime,
    validate_import,
)
from AegisKV.experiments.real_weka_full_design.import_mooncake_rtx4090_pp2 import (
    import_snapshot,
)
from AegisKV.experiments.real_weka_full_design import rtx4090_pp2_contract as contract
from AegisKV.experiments.real_weka_full_design.rtx4090_pp2_contract import (
    HARDWARE_LABEL,
    PROFILE_SEMANTICS,
    TCP_DIAGNOSTIC_ACK,
    build_profile,
    default_input_paths,
    load_object,
    sha256_file,
)
from AegisKV.experiments.real_weka_full_design.run import (
    V100_PROFILE_SEMANTICS,
    calibrated_runtime_config,
)


MOONCAKE_ROOT = Path("/home/comp/zhanzhao/Mooncake")
PINNED_REPORT = (
    MOONCAKE_ROOT
    / "aegiskv/runs/fallback-4090-pp2-paired-series-seeds-20260822-20260825"
    / "PRELIMINARY_REPORT.md"
)
pytestmark = pytest.mark.skipif(
    not PINNED_REPORT.is_file(),
    reason="pinned local Mooncake RTX4090 PP2 producer snapshot is absent",
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _import(tmp_path: Path) -> tuple[Path, Path, Path]:
    output_dir = import_snapshot(
        mooncake_root=MOONCAKE_ROOT, output_root=tmp_path / "imports"
    )
    return (
        output_dir,
        output_dir / "profile.json",
        output_dir / "import-manifest.json",
    )


def _forge_rehashed_profile(
    output_dir: Path,
    profile_path: Path,
    manifest_path: Path,
    *,
    mutate,
) -> tuple[Path, Path]:
    profile = load_object(profile_path)
    mutate(profile)
    profile_path.write_bytes(_json_bytes(profile))
    profile_sha = sha256_file(profile_path)
    new_name = f"rtx4090_pp2_{profile_sha[:16]}"
    new_dir = output_dir.parent / new_name
    output_dir.rename(new_dir)
    new_profile = new_dir / "profile.json"
    new_manifest = new_dir / "import-manifest.json"
    manifest = load_object(new_manifest)
    manifest["output_directory_name"] = new_name
    manifest["output_directory_identity_sha256"] = profile_sha
    manifest["transport_contract"] = profile["transport_observation"]
    manifest["recovery_transport_contract"] = profile[
        "recovery_transport_observation"
    ]
    manifest["p2p_topology_diagnostic"] = profile["p2p_topology_diagnostic"]
    manifest["output_profile"] = {
        "path": str(new_profile.resolve()),
        "sha256": profile_sha,
        "size_bytes": new_profile.stat().st_size,
    }
    new_manifest.write_bytes(_json_bytes(manifest))
    return new_profile, new_manifest


def test_builds_isolated_physical_registered_hbm_over_tcp_profile() -> None:
    profile = build_profile(default_input_paths(MOONCAKE_ROOT))

    assert profile["status"] == "PASS"
    assert profile["hardware_label"] == HARDWARE_LABEL
    assert profile["profile_semantics"] == PROFILE_SEMANTICS
    assert profile["source_lineage"]["separate_from_v100_campaign"] is True
    assert profile["source_lineage"]["separate_from_a100_campaign"] is True
    assert profile["source_lineage"]["frozen_source_verification"] == {
        "entry_count": 175,
        "entries_sha256": contract.SOURCE_ENTRIES_SHA256,
        "verification_basis": (
            "pinned_manifest_artifacts_and_canonical_entries_digest"
        ),
        "mutable_live_checkout_match_required": False,
    }
    transport = profile["transport_observation"]
    assert transport["sample_count"] == 48
    assert transport["transfer_protocol"] == "tcp"
    assert transport["physical_path"] == (
        "registered_gpu_memory_over_mooncake_tcp"
    )
    assert transport["direct_cuda_p2p_ready"] is False
    assert transport["exact_d2d_p2p_contract_ready"] is False
    assert transport["simulator_rate_default_eligible"] is False
    assert transport["registered_gpu_memory_over_tcp_verified"] is True
    assert profile["recovery_transport_observation"] == {
        "survivor_agent_state_count": 2,
        "event_log_agent_start_count": 4,
        "transfer_protocol": "tcp",
        "source_owned_transfers_only": True,
    }
    topology = profile["p2p_topology_diagnostic"]
    assert topology["slurm_job_id"] == "135738"
    assert topology["state"] == "COMPLETED"
    assert topology["exit_code"] == "0:0"
    assert topology["gpu_pair_p2p_status_by_node"] == {
        "gpu11": "NS",
        "gpu13": "NS",
    }
    assert topology["standalone_producer_artifact_available"] is False


def test_keeps_targeted_positive_series_and_negative_control_separate() -> None:
    profile = build_profile(default_input_paths(MOONCAKE_ROOT))
    series = profile["preliminary_targeted_physical_series"]
    control = profile["decode_heavy_control"]

    assert series["valid_pair_count"] == 4
    assert len(series["pairs"]) == 4
    assert all(
        row["metrics"]["failure_added_time_saved_ms"] > 0
        for row in series["pairs"]
    )
    assert series["actual_prompt_tokens"] == 2933
    assert series["metrics"]["failure_added_time_saved_ms"]["mean"] == pytest.approx(
        139.81402587890625
    )
    assert control["aggregation_with_long_context_prohibited"] is True
    assert control["advantage_stable"] is False
    assert [
        row["metrics"]["failure_added_time_saved_ms"]
        for row in control["pairs"]
    ] == [-43.68701171875, 34.28125]
    assert control["mean_failure_added_time_saved_ms"] == -4.702880859375


def test_accepts_only_authoritative_figure_v3_and_updated_report() -> None:
    paths = default_input_paths(MOONCAKE_ROOT)
    profile = build_profile(paths)
    figure = profile["reporting_figure"]

    assert paths.figure_audit.parent.name == "figure-v3"
    assert figure["invalid_attempt_count"] == 0
    assert figure["invalid_attempt_note"] == contract.INVALID_ATTEMPT_NOTE
    assert profile["preliminary_report"]["sha256"] == (
        "658a58e855841c61794e50d57b90350d764c32ba91e1e390e37c90ca523d6033"
    )

    old_dir = paths.figure_audit.parent.parent / "figure-v2"
    old_paths = replace(
        paths,
        figure_audit=old_dir / "paired-series-figure-audit.json",
        figure_png=old_dir / "paired-series-effect.png",
        figure_pdf=old_dir / "paired-series-effect.pdf",
        figure_csv=old_dir / "paired-series.csv",
    )
    with pytest.raises(ValueError, match="pinned RTX4090 PP2 artifact"):
        build_profile(old_paths)


def test_import_is_sha_derived_reproducible_and_non_overwritable(
    tmp_path: Path,
) -> None:
    output_dir, profile_path, manifest_path = _import(tmp_path)
    profile, manifest = validate_import(
        profile_path=profile_path, manifest_path=manifest_path
    )

    assert output_dir.name == f"rtx4090_pp2_{sha256_file(profile_path)[:16]}"
    assert manifest["default_replay_authorized"] is False
    assert manifest["formal_replay_authorized"] is False
    assert manifest["figure_generation_authorized"] is False
    assert profile["formal_ready"] is False
    assert profile["paper_ready"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        import_snapshot(
            mooncake_root=MOONCAKE_ROOT, output_root=tmp_path / "imports"
        )


def test_import_creates_no_output_before_complete_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative, _digest = contract.EXPECTED_ARTIFACTS["preliminary_report"]
    monkeypatch.setitem(
        contract.EXPECTED_ARTIFACTS,
        "preliminary_report",
        (relative, "0" * 64),
    )
    output_root = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="preliminary_report sha256"):
        import_snapshot(mooncake_root=MOONCAKE_ROOT, output_root=output_root)
    assert not output_root.exists()


def test_consumer_rejects_rehashed_transport_eligibility_forgery(
    tmp_path: Path,
) -> None:
    output_dir, profile_path, manifest_path = _import(tmp_path)
    profile_path, manifest_path = _forge_rehashed_profile(
        output_dir,
        profile_path,
        manifest_path,
        mutate=lambda profile: profile["transport_observation"].__setitem__(
            "simulator_rate_default_eligible", True
        ),
    )

    with pytest.raises(ValueError, match="normalized profile contract"):
        validate_import(profile_path=profile_path, manifest_path=manifest_path)


def test_consumer_rejects_rehashed_metric_forgery(tmp_path: Path) -> None:
    output_dir, profile_path, manifest_path = _import(tmp_path)
    profile_path, manifest_path = _forge_rehashed_profile(
        output_dir,
        profile_path,
        manifest_path,
        mutate=lambda profile: profile["metrics"].__setitem__(
            "peer_hbm_recovery_effective_bytes_per_ms", 1.0
        ),
    )

    with pytest.raises(ValueError, match="not reproducible"):
        validate_import(profile_path=profile_path, manifest_path=manifest_path)


def test_runtime_materialization_requires_ack_and_keeps_rates_separate(
    tmp_path: Path,
) -> None:
    _output_dir, profile_path, manifest_path = _import(tmp_path)
    profile = load_object(profile_path)

    with pytest.raises(ValueError, match="diagnostic-only"):
        materialize_diagnostic_runtime(
            profile_path=profile_path, manifest_path=manifest_path
        )
    runtime, provenance = materialize_diagnostic_runtime(
        profile_path=profile_path,
        manifest_path=manifest_path,
        acknowledgement=TCP_DIAGNOSTIC_ACK,
    )
    metrics = profile["metrics"]
    kv_bytes = profile["runtime_parameters"]["kv_bytes_per_token"]
    assert runtime.hbm_copy.bandwidth_bytes_per_ms == metrics[
        "peer_hbm_protection_effective_bytes_per_ms"
    ]
    assert runtime.dram_copy.bandwidth_bytes_per_ms == metrics[
        "direct_dram_protection_effective_bytes_per_ms"
    ]
    assert runtime.hbm_load.ms_per_token == pytest.approx(
        kv_bytes / metrics["peer_hbm_recovery_effective_bytes_per_ms"]
    )
    assert runtime.dram_load.ms_per_token == pytest.approx(
        kv_bytes / metrics["dram_recovery_effective_bytes_per_ms"]
    )
    assert runtime.hbm_load.ms_per_token != pytest.approx(
        kv_bytes / metrics["peer_hbm_protection_effective_bytes_per_ms"]
    )
    assert provenance["replay_performed"] is False
    assert provenance["replay_authorized"] is False


def test_v100_formal_consumer_remains_closed_to_rtx4090_profile() -> None:
    assert V100_PROFILE_SEMANTICS == "source_owned_four_physical_paths_v1"
    with pytest.raises(ValueError, match="legacy physical calibration bundles"):
        calibrated_runtime_config(
            {
                "calibration": {
                    "status": "READY",
                    "bundle_semantics": PROFILE_SEMANTICS,
                }
            }
        )
