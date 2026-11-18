from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.agentic_workflows.weka_full_trace import (
    WekaAgentBranch,
    WekaRequest,
    WekaStage,
    WekaWorkflow,
)
from AegisKV.experiments.real_weka_baselines import make_figures
from AegisKV.experiments.real_weka_baselines import run as baseline_replay
from AegisKV.experiments.real_weka_baselines.bind_v100_suite import (
    build_suite_binding,
)
from AegisKV.experiments.real_weka_full_design import run as full_replay
from AegisKV.experiments.real_weka_full_design.calibrated_suite_contract import (
    A100_CONSUMER_KIND,
    V100_CONSUMER_KIND,
    CalibratedConsumerInput,
    CalibratedSuiteContractError,
    materialize_suite_binding,
    preflight_suite_runner,
    validate_suite_binding,
)
from AegisKV.experiments.real_weka_full_design.import_mooncake_v100_profile import (
    import_profile,
)
from AegisKV.experiments.real_weka_full_design.run import (
    V100_COMPATIBILITY_IDENTITY,
    V100_HARDWARE_LABEL,
    V100_PROFILE_SEMANTICS,
    V100_RUNTIME_CONSUMPTION_CONTRACT,
    _legacy_calibrated_runtime_config_diagnostic_only,
    _physical_aimd_pressure,
    _pp_direct_dram_protection_bandwidth,
    calibrated_runtime_config,
)
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS


RESULT_LABEL = "RTX4090-2node-PP2-systemDP2"
A100_DP2_HARDWARE_LABEL = "A100-1node-TP1-PP1-systemDP2"
V100_CAMPAIGN_ID = "qwen3-v100-dp4-pilot-campaign-1"
MOONCAKE_FOUR_CONDITION_CHECKS = {
    "single_source_config_physical_profile_lineage",
    "independent_admit_hbm_admit_dram_and_bypass_exercised",
    "hbm_reject_then_independent_dram_admit_exercised",
    "all_four_independent_residency_outcomes_exercised",
    "hbm_and_dram_paths_confirm_independently",
    "no_target_owned_or_hbm_to_dram_runtime_semantics",
    "failed_aegiskv_physical_peer_hbm_recovery",
    "failed_aegiskv_physical_dram_recovery",
    "aimd_pause_reclaim_satisfy_resume_contract",
    "controller_deletes_hbm_only_tail_without_dram_prerequisite",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path, value: Any, *, embedded: bool = False) -> dict[str, Any]:
    _write_json(path, value)
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if embedded:
        record["value"] = value
    return record


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _event_artifact(
    path: Path, event: dict[str, Any], *, replica: int, stage: int
) -> dict[str, Any]:
    path.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "event_count": 1,
        "pipeline_replica": replica,
        "pipeline_stage": stage,
    }


def _gate(evidence_class: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "evidence_class": evidence_class,
        "design_semantics": DESIGN_SEMANTICS,
        "result_label": RESULT_LABEL,
        "paper_ready": False,
        "evidence_boundary": "synthetic unit-test artifact",
        "details": details,
    }


def _upstream(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.name,
        "sha256": _sha256(path),
        "status": "PASS",
    }


def _bundle(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    peer_path = _write_json(
        tmp_path / "peer.json",
        _gate(
            "aegiskv_4090_pp2_peer_hbm_calibration_gate",
            {
                "physical_calibration": {
                    "conservative_logical_bandwidth_bytes_per_ms": 100.0
                }
            },
        ),
    )
    dram_recovery_path = _write_json(
        tmp_path / "dram-recovery.json",
        _gate(
            "aegiskv_4090_pp_mooncake_dram_calibration_gate",
            {
                "mooncake_dram_calibration": {
                    "summary": {"aggregate_effective_load_gb_per_s": 0.00008}
                }
            },
        ),
    )
    recompute_path = _write_json(
        tmp_path / "recompute.json",
        _gate(
            "aegiskv_4090_pp_recompute_calibration_gate",
            {
                "recompute_calibration": {
                    "pooled_fit": {
                        "fixed_overhead_ms": 1.0,
                        "recompute_ms_per_token": 0.5,
                    }
                }
            },
        ),
    )

    direct_files = []
    for stage, duration in ((0, 4.0), (1, 8.0)):
        direct_files.append(
            _event_artifact(
                tmp_path / f"direct-stage{stage}.jsonl",
                {
                    "event": "pp_copy_completion",
                    "chunk_id": "dram-test",
                    "committed": True,
                    "coordinator_status": "committed",
                    "transfer_succeeded": True,
                    "local_bytes": 100,
                    "transferred_bytes": 100,
                    "duration_ms": duration,
                    "put_results": [0],
                    "request_id": "direct-request",
                    "ordinal": 0,
                    "pipeline_replica": 0,
                    "pipeline_stage": stage,
                },
                replica=0,
                stage=stage,
            )
        )
    direct_manifest = _artifact(tmp_path / "direct-manifest.json", {"frozen": True})
    direct_manifest.update(
        {
            "upstream_peer_hbm_gate": _upstream(peer_path),
            "upstream_mooncake_dram_gate": _upstream(dram_recovery_path),
            "upstream_recompute_gate": _upstream(recompute_path),
        }
    )
    direct_gate = _gate(
        "aegiskv_4090_pp2_hbm_integration_gate",
        {
            "source_config_manifest": direct_manifest,
            "deployment": _artifact(
                tmp_path / "direct-deployment.json",
                {"design_semantics": DESIGN_SEMANTICS, "result_label": RESULT_LABEL},
                embedded=True,
            ),
            "stage_aware_peer_hbm": {"files": direct_files},
        },
    )
    direct_path = _write_json(tmp_path / "direct-gate.json", direct_gate)

    hbm_files = []
    for stage, duration in ((0, 2.0), (1, 4.0)):
        hbm_files.append(
            _event_artifact(
                tmp_path / f"hbm-load-stage{stage}.jsonl",
                {
                    "event": "hbm_physical_load",
                    "loaded": True,
                    "bytes": 100,
                    "loaded_bytes": 100,
                    "duration_ms": duration,
                    "load_results": [True],
                    "missing_storage_keys": [],
                    "request_id": "hbm-request",
                },
                replica=0,
                stage=stage,
            )
        )
    hbm_manifest = _artifact(tmp_path / "hbm-manifest.json", {"frozen": True})
    hbm_manifest.update(
        {
            "upstream_peer_hbm_gate": _upstream(peer_path),
            "upstream_mooncake_dram_gate": _upstream(dram_recovery_path),
            "upstream_recompute_gate": _upstream(recompute_path),
            "upstream_hbm_integration_gate": _upstream(direct_path),
        }
    )
    hbm_gate = _gate(
        "aegiskv_4090_pp2_hbm_recovery_gate",
        {
            "source_config_manifest": hbm_manifest,
            "deployment": _artifact(
                tmp_path / "hbm-deployment.json",
                {"design_semantics": DESIGN_SEMANTICS, "result_label": RESULT_LABEL},
                embedded=True,
            ),
            "stage_aware_recovery": {"files": hbm_files},
        },
    )
    hbm_path = _write_json(tmp_path / "hbm-gate.json", hbm_gate)

    config = {
        "experiment_id": "unit-four-path",
        "model": {"kv_bytes_per_token": 10},
        "cluster": {"dp_size": 2, "pp_size": 2, "tp_size": 1},
        "capacity": {"hbm_bytes_per_target": 1000, "dram_bytes_total": 2000},
        "chunking": {"prefix_tokens": 16, "decode_tokens": 4},
        "aimd": {
            "initial_alpha": 1.0,
            "minimum_alpha": 0.25,
            "maximum_alpha": 1.0,
            "decrease_factor": 0.5,
            "increase_step": 0.1,
            "stable_windows": 3,
        },
        "audit": {"event_log_limit": 16},
        "calibration": {
            "status": "READY",
            "bundle_result_label": RESULT_LABEL,
            "bundle_semantics": "four_independent_physical_paths_v1",
            "peer_hbm_protection_gate": str(peer_path),
            "direct_dram_protection_gate": str(direct_path),
            "hbm_recovery_gate": str(hbm_path),
            "dram_recovery_gate": str(dram_recovery_path),
            "recompute_gate": str(recompute_path),
            "peer_hbm_protection_setup_ms": 0.0,
            "direct_dram_protection_setup_ms": 0.0,
            "hbm_recovery_setup_ms": 0.0,
            "dram_recovery_setup_ms": 0.0,
        },
    }
    return config, direct_path, hbm_path


def test_legacy_ready_four_path_bundle_is_rejected_by_formal_consumer(
    tmp_path: Path,
) -> None:
    config, _direct_path, _hbm_path = _bundle(tmp_path)
    with pytest.raises(ValueError, match="legacy physical calibration bundles"):
        calibrated_runtime_config(config)


def test_invalidated_physical_calibration_fails_before_ingestion(
    tmp_path: Path,
) -> None:
    config, _direct_path, _hbm_path = _bundle(tmp_path)
    config["calibration"]["status"] = "INVALIDATED_TCP_TRANSPORT"

    with pytest.raises(ValueError, match="physical calibration bundle is not READY"):
        calibrated_runtime_config(config)


def test_full_replay_rejects_invalidated_calibration_before_dataset_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _direct_path, _hbm_path = _bundle(tmp_path)
    config["design_semantics"] = DESIGN_SEMANTICS
    config["calibration"]["status"] = "INVALIDATED_TCP_TRANSPORT"
    config_path = _write_json(tmp_path / "config.json", config)
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        full_replay,
        "parse_args",
        lambda: argparse.Namespace(
            config=config_path,
            output_dir=output,
            dataset=tmp_path / "must-not-load.jsonl",
            limit_workflows=None,
            skip_dataset_sha=False,
        ),
    )
    monkeypatch.setattr(
        full_replay,
        "load_weka_workflows",
        lambda *_args, **_kwargs: pytest.fail("invalid calibration loaded the trace"),
    )

    with pytest.raises(ValueError, match="physical calibration bundle is not READY"):
        full_replay.main()
    assert not output.exists()


def test_baseline_figure_renderer_rejects_invalidated_suite_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _write_json(
        tmp_path / "suite.json",
        {"status": "INVALIDATED_UPSTREAM_TCP_TRANSPORT"},
    )
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        make_figures,
        "parse_args",
        lambda: argparse.Namespace(
            results_dir=tmp_path / "missing-results",
            output_dir=output,
            suite_config=suite,
        ),
    )

    with pytest.raises(ValueError, match="baseline suite is not READY"):
        make_figures.main()
    assert not output.exists()


def _v100_profile_bundle(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    template = _write_json(
        tmp_path / "template.json",
        {
            "experiment_id": "invalidated-template",
            "design_semantics": DESIGN_SEMANTICS,
            "dataset": {
                "expected_full_profile": {
                    "workflow_count": 1,
                    "request_count": 3,
                    "input_tokens": 300,
                    "output_tokens": 30,
                }
            },
            "workload": {"workflow_mean_interarrival_ms": 1000.0},
            "model": {
                "revision": "Qwen/Qwen3-8B",
                "kv_abi": "qwen3-8b-bf16-invalid-template",
                "kv_bytes_per_token": 10,
            },
            "cluster": {"dp_size": 4, "pp_size": 1, "tp_size": 1},
            "capacity": {
                "hbm_bytes_per_target": 1000,
                "dram_bytes_total": 8000,
            },
            "chunking": {"prefix_tokens": 16, "decode_tokens": 4},
            "aimd": {
                "initial_alpha": 1.0,
                "minimum_alpha": 0.25,
                "maximum_alpha": 1.0,
                "decrease_factor": 0.5,
                "increase_step": 0.1,
                "stable_windows": 3,
                "pressure_mode": "active_source_request_count_proxy",
                "pressure_calibrated": False,
                "pressure_active_source_requests": 32,
            },
            "audit": {"event_log_limit": 16},
            "calibration": {"status": "INVALIDATED_TCP_TRANSPORT"},
        },
    )
    source_payload = tmp_path / "source.py"
    source_payload.write_text("print('pinned')\n", encoding="utf-8")
    entries = [
        {
            "kind": "deployment_source",
            "relative_path": "source.py",
            "sha256": _sha256(source_payload),
            "size_bytes": source_payload.stat().st_size,
        }
    ]
    entries_sha = hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest = _write_json(
        tmp_path / "source-manifest.json",
        {
            "schema_version": 1,
            "evidence_class": "aegiskv_source_and_native_binary_manifest",
            "status": "PINNED",
            "inventory_policy": "exact_selected_file_set",
            "repository_root": str(tmp_path),
            "git_revision": "f" * 40,
            "entry_count": 1,
            "entries_sha256": entries_sha,
            "entries": entries,
        },
    )
    verification = _write_json(
        tmp_path / "source-verification.json",
        {
            "schema_version": 1,
            "evidence_class": "aegiskv_source_manifest_verification",
            "status": "PASS",
            "failures": [],
            "entry_count": 1,
            "checked_entry_count": 1,
            "entries_sha256": entries_sha,
            "inventory_policy": "exact_selected_file_set",
            "repository_root": str(tmp_path),
            "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        },
    )
    model_revision = "b968826d"
    model_config_sha256 = "c" * 64
    runtime_versions = {"vllm": "1.2.2", "torch": "2.8.0", "cuda": "12.8"}
    source_ref = _record(manifest)
    source_ref["entries_sha256"] = entries_sha
    source_verification_ref = _record(verification)
    activation = _write_json(
        tmp_path / "source-activation.json",
        {
            "schema_version": 1,
            "evidence_class": full_replay.V100_SOURCE_ACTIVATION_EVIDENCE,
            "status": "PASS",
            "campaign_id": V100_CAMPAIGN_ID,
            "hardware_label": V100_HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "source_manifest": source_ref,
            "source_verification": source_verification_ref,
            "superseded_source_manifest_sha256": (
                full_replay.V100_SUPERSEDED_SOURCE_MANIFEST_SHA256
            ),
            "checked_entry_count": 1,
            "activated_runtime": {"source": _record(source_payload)},
            "admission_decision_wire_values": (
                full_replay.V100_RUNTIME_ADMISSION_DECISIONS
            ),
            "required_patch_order": ["v19-strict-run-hardware-profile.patch"],
            "checks": {
                name: True
                for name in (
                    "new_campaign_id",
                    "source_freeze_supersedes_queued_v11",
                    "source_manifest_contract",
                    "source_verification_contract",
                    "all_pinned_source_entries_live",
                    "current_design_runtime_markers",
                    "admission_decision_wire_values",
                )
            },
            "failures": [],
            "mutations_performed": [],
            "jobs_submitted": [],
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    model_path = str(tmp_path / model_revision)
    worker_results = [
        {
            "schema_version": 1,
            "evidence_class": (
                full_replay.V100_MODEL_EXECUTION_WORKER_EVIDENCE
            ),
            "status": "PASS",
            "logical_rank": rank,
            "cuda_visible_devices": str(rank),
            "gpu_name": "Tesla V100S-PCIE-32GB",
            "gpu_capability": [7, 0],
            "model_id": "Qwen/Qwen3-8B",
            "model_path": model_path,
            "model_revision": model_revision,
            "model_config_sha256": model_config_sha256,
            "runtime_versions": runtime_versions,
            "model_runner_v2_requested": True,
            "output_token_ids": [rank + 1],
            "elapsed_ms": 10.0 + rank,
        }
        for rank in range(4)
    ]
    model_probe = _write_json(
        tmp_path / "model-probe.json",
        {
            "schema_version": 1,
            "evidence_class": (
                full_replay.V100_MODEL_EXECUTION_PROBE_EVIDENCE
            ),
            "status": "PASS",
            "hardware_label": V100_HARDWARE_LABEL,
            "model_id": "Qwen/Qwen3-8B",
            "model_path": model_path,
            "model_revision": model_revision,
            "model_config_sha256": model_config_sha256,
            "runtime_versions": runtime_versions,
            "tp_size": 1,
            "pp_size": 1,
            "system_dp_size": 4,
            "worker_results": worker_results,
            "failures": [],
            "jobs_submitted": [],
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    configuration = {
        "schema_version": 1,
        "evidence_class": full_replay.V100_CONFIG_INPUT_EVIDENCE,
        "status": "PINNED",
        "campaign_id": V100_CAMPAIGN_ID,
        "design_semantics": DESIGN_SEMANTICS,
        "hardware_label": V100_HARDWARE_LABEL,
        "layout": full_replay.V100_RUNTIME_PROFILE_LAYOUT,
        "hardware_profile": full_replay.V100_CONFIG_HARDWARE_PROFILE,
        "compatibility": full_replay.V100_CONFIG_COMPATIBILITY_IDENTITY,
        "capacity": {
            "peer_hbm_bytes_per_target": 2 * 1024**3,
            "dram_bytes": 16 * 1024**3,
        },
        "aimd": {
            "initial_alpha": 0.5,
            "min_alpha": 0.1,
            "max_alpha": 0.8,
            "decrease_factor": 0.5,
            "increase_step": 0.05,
            "stable_window_threshold": 3,
            "interval_s": 5.0,
            "pressure_free_bytes": 3 * 1024**3,
        },
        "runtime": {
            "vllm_engine": "v1",
            "vllm_model_runner": "v2",
            "model_runner_v2_requested": True,
        },
        "source_owned_paths": {
            **full_replay.V100_CONFIG_PATH_OWNERS,
            "target_owned_or_hbm_to_dram_path_allowed": False,
        },
        "four_condition_gate": [
            {"treatment": treatment, "failure_method": failure}
            for treatment, failure in sorted(full_replay.V100_PILOT_CONDITIONS)
        ],
    }
    configuration_input = _write_json(
        tmp_path / "configuration-input.json", configuration
    )
    config_manifest = _write_json(
        tmp_path / "config-manifest.json",
        {
            "schema_version": 1,
            "evidence_class": full_replay.V100_CONFIG_MANIFEST_EVIDENCE,
            "status": "PINNED",
            "campaign_id": V100_CAMPAIGN_ID,
            "design_semantics": DESIGN_SEMANTICS,
            "hardware_label": V100_HARDWARE_LABEL,
            "model": "Qwen/Qwen3-8B",
            "model_revision": model_revision,
            "model_config_sha256": model_config_sha256,
            "runtime_versions": runtime_versions,
            "layout": full_replay.V100_RUNTIME_PROFILE_LAYOUT,
            "hardware_profile": full_replay.V100_CONFIG_HARDWARE_PROFILE,
            "source_manifest": source_ref,
            "source_verification": source_verification_ref,
            "source_activation_gate": _record(activation),
            "model_execution_probe": _record(model_probe),
            "configuration_input": _record(configuration_input),
            "configuration": configuration,
            "configuration_sha256": full_replay._canonical_json_sha256(
                configuration
            ),
            "jobs_submitted": [],
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    config_verification = _write_json(
        tmp_path / "config-verification.json",
        {
            "schema_version": 1,
            "evidence_class": full_replay.V100_CONFIG_VERIFICATION_EVIDENCE,
            "status": "PASS",
            "campaign_id": V100_CAMPAIGN_ID,
            "hardware_label": V100_HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "failures": [],
            "manifest": {
                "path": str(config_manifest),
                "sha256": _sha256(config_manifest),
            },
            "jobs_submitted": [],
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    profile = _write_json(
        tmp_path / "physical-profile.json",
        {
            "schema_version": 2,
            "evidence_class": "physical_measurement",
            "status": "passed",
            "failures": [],
            "profile_semantics": V100_PROFILE_SEMANTICS,
            "hardware_label": V100_HARDWARE_LABEL,
            "layout": {"tp_size": 1, "pp_size": 1, "dp_size": 4},
            "model": "Qwen/Qwen3-8B",
            "model_revision": model_revision,
            "compatibility_identity": V100_COMPATIBILITY_IDENTITY,
            "runtime_consumption_contract": V100_RUNTIME_CONSUMPTION_CONTRACT,
            "hardware": {
                "name": "Tesla V100S-PCIE-32GB",
                "capability": [7, 0],
            },
            "metrics": {
                "recompute_ms_per_token": 0.5,
                "recompute_fixed_overhead_ms": 1.0,
                "peer_hbm_protection_effective_bytes_per_ms": 100.0,
                "direct_dram_protection_effective_bytes_per_ms": 80.0,
                "peer_hbm_recovery_effective_bytes_per_ms": 50.0,
                "dram_recovery_effective_bytes_per_ms": 40.0,
            },
            "transport_signature": {
                "dram": {
                    "protocol": "tcp",
                    "connection_pool_enabled": True,
                },
                "peer_hbm": {
                    "protocol": "nvlink_intra",
                    "physical_interconnect": "pcie_p2p",
                    "runtime_value": "1",
                    "engine_sha256": "a" * 64,
                    "peer_access": [[True] * 4 for _ in range(4)],
                },
            },
            "source_manifest": {
                "path": str(manifest),
                "sha256": _sha256(manifest),
                "entries_sha256": entries_sha,
            },
            "config_manifest": _record(config_manifest),
            "config_verification": _record(config_verification),
            "source_artifacts": {
                "recompute": {
                    "path": str(source_payload),
                    "sha256": _sha256(source_payload),
                }
            },
        },
    )
    context_audit = _write_json(
        tmp_path / "context-audit.json",
        {
            "status": "PASS",
            "design_semantics": DESIGN_SEMANTICS,
            "trace": {
                "sha256": "b" * 64,
                "request_count": 3,
                "input_tokens": 300,
                "output_tokens": 30,
            },
            "model": {
                "revision": f"Qwen/Qwen3-8B@{model_revision}",
                "maximum_context_tokens": 40960,
                "kv_bytes_per_token_all_layers": 10,
            },
            "scale_points": [
                {
                    "label": "unit-primary",
                    "factor": 0.004,
                    "role": "preregistered_primary_after_physical_profile_gate",
                    "readiness": "ELIGIBLE_AFTER_VALID_PHYSICAL_PROFILE",
                    "performance_results_present": False,
                    "requests_above_model_context": 0,
                    "requests_above_physical_calibration_context": 0,
                    "context_tokens": {"max": 100},
                }
            ],
        },
    )
    source_ref = {
        "path": str(manifest),
        "sha256": _sha256(manifest),
        "size_bytes": manifest.stat().st_size,
    }
    physical_ref = {
        "path": str(profile),
        "sha256": _sha256(profile),
        "size_bytes": profile.stat().st_size,
    }
    config_ref = _record(config_manifest)
    config_verification_ref = _record(config_verification)
    support_ref = {
        "path": str(source_payload),
        "sha256": _sha256(source_payload),
        "size_bytes": source_payload.stat().st_size,
    }
    runtime_profiles = []
    condition_index = 0
    for treatment in ("no-protection", "aegiskv"):
        for failure_method in ("none", "sigkill_process_group"):
            condition_id = f"{treatment}-{failure_method}"
            run_config = _artifact(
                tmp_path / f"runtime-{condition_index}-config.json",
                {
                    "treatment": treatment,
                    "failure": {"method": failure_method},
                },
            )
            profile_payload = {
                "schema_version": 1,
                "evidence_class": full_replay.V100_RUNTIME_PROFILE_EVIDENCE,
                "status": "PASS",
                "failures": [],
                "design_semantics": DESIGN_SEMANTICS,
                "campaign_id": V100_CAMPAIGN_ID,
                "campaign_condition_id": condition_id,
                "condition_id": condition_id,
                "treatment": treatment,
                "failure_method": failure_method,
                "hardware_label": V100_HARDWARE_LABEL,
                "layout": full_replay.V100_RUNTIME_PROFILE_LAYOUT,
                "input_lineage": {
                    "source_manifest": source_ref,
                    "config_manifest": config_ref,
                    "config_verification": config_verification_ref,
                    "physical_profile": physical_ref,
                    "strict_run_audit": support_ref,
                },
                "formal_inputs": {
                    "campaign_condition_id": condition_id,
                    "source_manifest": source_ref,
                    "physical_profile": physical_ref,
                },
                "artifacts": {
                    "deployment": support_ref,
                    "run_config": run_config,
                    "summary": support_ref,
                    "strict_audit": support_ref,
                    "gpu_telemetry": support_ref,
                    "source_manifest": source_ref,
                    "physical_profile": physical_ref,
                    "config_manifest": config_ref,
                    "config_verification": config_verification_ref,
                    "slurm_co_residency": support_ref,
                    "gpu_processes_before": support_ref,
                    "gpu_processes_after": support_ref,
                    "router_events": support_ref,
                    "gate_events": [support_ref],
                },
            }
            runtime_profile_path = _write_json(
                tmp_path / f"runtime-profile-{condition_index}.json",
                profile_payload,
            )
            runtime_profiles.append(
                {
                    "condition_id": condition_id,
                    "run_dir": str(tmp_path / f"run-{condition_index}"),
                    "profile": {
                        "path": str(runtime_profile_path),
                        "sha256": _sha256(runtime_profile_path),
                    },
                }
            )
            condition_index += 1
    pilot_gate = _write_json(
        tmp_path / "pilot-runtime-gate.json",
        {
            "schema_version": 1,
            "evidence_class": full_replay.V100_PILOT_RUNTIME_GATE_EVIDENCE,
            "status": "PASS",
            "failures": [],
            "design_semantics": DESIGN_SEMANTICS,
            "hardware_label": V100_HARDWARE_LABEL,
            "campaign_id": V100_CAMPAIGN_ID,
            "source_manifest": source_ref,
            "config_manifest": config_ref,
            "config_verification": config_verification_ref,
            "physical_profile": physical_ref,
            "auditor": support_ref,
            "profile_builder": support_ref,
            "profiles": runtime_profiles,
            "checks": [
                {"name": name, "passed": True, "detail": "fixture"}
                for name in sorted(full_replay.V100_REQUIRED_RUNTIME_CHECKS)
            ],
            "pilot_performance_claim_authorized": False,
            "formal_scale_submission_authorized": False,
            "paper_ready": False,
        },
    )
    return template, profile, manifest, verification, context_audit, pilot_gate


def _rewired_v100_config_lineage(
    tmp_path: Path,
    *,
    artifact_key: str,
    mutate: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    _template, profile_path, manifest_path, verification_path, _context, _gate = (
        _v100_profile_bundle(tmp_path)
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    config_path = Path(profile["config_manifest"]["path"])
    config_verification_path = Path(profile["config_verification"]["path"])
    config_manifest = json.loads(config_path.read_text(encoding="utf-8"))
    artifact_path = Path(config_manifest[artifact_key]["path"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    mutate(artifact)
    _write_json(artifact_path, artifact)
    config_manifest[artifact_key] = _record(artifact_path)
    if artifact_key == "configuration_input":
        config_manifest["configuration"] = artifact
        config_manifest["configuration_sha256"] = (
            full_replay._canonical_json_sha256(artifact)
        )
    _write_json(config_path, config_manifest)
    config_verification = json.loads(
        config_verification_path.read_text(encoding="utf-8")
    )
    config_verification["manifest"] = _record(config_path)
    _write_json(config_verification_path, config_verification)
    profile["config_manifest"] = _record(config_path)
    profile["config_verification"] = _record(config_verification_path)
    calibration = {
        "config_manifest": _record(config_path),
        "config_verification": _record(config_verification_path),
    }
    return (
        calibration,
        profile,
        _record(manifest_path),
        _record(verification_path),
    )


def test_v100_config_lineage_rejects_zero_as_hbm_admission(
    tmp_path: Path,
) -> None:
    def mutate(activation: dict[str, Any]) -> None:
        activation["admission_decision_wire_values"]["ADMIT_HBM"] = 0

    calibration, profile, source, source_verification = (
        _rewired_v100_config_lineage(
            tmp_path,
            artifact_key="source_activation_gate",
            mutate=mutate,
        )
    )
    source_manifest = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="source activation contract"):
        full_replay._verify_v100_config_lineage(
            calibration,
            profile,
            source_manifest_record=source,
            source_verification_record=source_verification,
            source_manifest=source_manifest,
        )


def test_v100_config_lineage_rejects_nonunique_probe_devices(
    tmp_path: Path,
) -> None:
    def mutate(probe: dict[str, Any]) -> None:
        probe["worker_results"][-1]["cuda_visible_devices"] = "0"

    calibration, profile, source, source_verification = (
        _rewired_v100_config_lineage(
            tmp_path,
            artifact_key="model_execution_probe",
            mutate=mutate,
        )
    )
    source_manifest = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="four-worker model execution probe"):
        full_replay._verify_v100_config_lineage(
            calibration,
            profile,
            source_manifest_record=source,
            source_verification_record=source_verification,
            source_manifest=source_manifest,
        )


def test_v100_config_lineage_rejects_target_owned_config_input(
    tmp_path: Path,
) -> None:
    def mutate(configuration: dict[str, Any]) -> None:
        configuration["source_owned_paths"][
            "target_owned_or_hbm_to_dram_path_allowed"
        ] = True

    calibration, profile, source, source_verification = (
        _rewired_v100_config_lineage(
            tmp_path,
            artifact_key="configuration_input",
            mutate=mutate,
        )
    )
    source_manifest = json.loads(Path(source["path"]).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="current-design config input"):
        full_replay._verify_v100_config_lineage(
            calibration,
            profile,
            source_manifest_record=source,
            source_verification_record=source_verification,
            source_manifest=source_manifest,
        )


def _materialized_v100_import(
    tmp_path: Path,
    *,
    calibrated_pressure: bool = False,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    pressure = (
        _v100_pressure_calibration(
            tmp_path,
            manifest=manifest,
            profile=profile,
            pilot_gate=pilot_gate,
        )
        if calibrated_pressure
        else None
    )
    config, import_manifest = import_profile(
        template_path=template,
        profile_path=profile,
        source_manifest_path=manifest,
        source_verification_path=verification,
        pilot_runtime_gate_path=pilot_gate,
        aimd_pressure_calibration_path=pressure,
        context_audit_path=context_audit,
        selected_scale_label="unit-primary",
        maximum_calibrated_context_tokens=4096,
    )
    config_path = _write_json(tmp_path / "imported-config.json", config)
    import_manifest["output_config"] = _record(config_path)
    import_manifest_path = _write_json(
        tmp_path / "import-manifest.json", import_manifest
    )
    return config_path, import_manifest_path, config, import_manifest


def _v100_calibrated_suite_matrix(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_scales": [
            {
                "model": config["model"]["revision"],
                "scale_label": "Qwen3-8B",
                "model_revision": config["model"]["model_snapshot_revision"],
            }
        ],
        "system_dp_sizes": [4],
        "concurrency_levels": [1, 8, 32],
        "failure_rates": [0.0, 0.02],
        "capacity_points": [
            {
                "label": "full",
                "peer_hbm_fraction": 1.0,
                "store_dram_fraction": 1.0,
            }
        ],
        "seeds": [20260821, 20260822],
    }


def test_v100_schema_v2_import_can_bind_static_suite_without_replay(
    tmp_path: Path,
) -> None:
    config_path, import_manifest_path, config, import_manifest = (
        _materialized_v100_import(tmp_path, calibrated_pressure=True)
    )
    binding_path = materialize_suite_binding(
        source=CalibratedConsumerInput(
            consumer_kind=V100_CONSUMER_KIND,
            consumer_manifest=import_manifest_path,
            consumer_payload=config_path,
        ),
        matrix=_v100_calibrated_suite_matrix(config),
        output_root=tmp_path / "calibrated-suite",
    )
    binding = validate_suite_binding(binding_path)
    assert binding["source_consumer"]["hardware_label"] == V100_HARDWARE_LABEL
    assert binding["source_consumer"]["layout"]["system_dp_size"] == 4
    assert (
        binding["source_consumer"]["profile_sha256"]
        == import_manifest["physical_profile"]["sha256"]
    )
    assert binding["matrix"]["cell_count"] == 12
    runner = preflight_suite_runner(binding_path)
    assert runner["status"] == "PASS_STATIC_ONLY"
    assert runner["cell_count"] == 12
    assert not (binding_path.parent / "replay.json").exists()
    assert not (binding_path.parent / "performance.json").exists()


def test_v100_import_cannot_cross_into_a100_suite_source(tmp_path: Path) -> None:
    config_path, import_manifest_path, config, _manifest = (
        _materialized_v100_import(tmp_path, calibrated_pressure=True)
    )
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(
        CalibratedSuiteContractError,
        match="A100 and V100 consumer lineages cannot cross",
    ):
        materialize_suite_binding(
            source=CalibratedConsumerInput(
                consumer_kind=A100_CONSUMER_KIND,
                consumer_manifest=import_manifest_path,
                consumer_payload=config_path,
            ),
            matrix=_v100_calibrated_suite_matrix(config),
            output_root=output_root,
        )
    assert not output_root.exists()


def test_suite_binding_rejects_missing_v100_physical_profile_before_output(
    tmp_path: Path,
) -> None:
    config_path, import_manifest_path, config, import_manifest = (
        _materialized_v100_import(tmp_path, calibrated_pressure=True)
    )
    Path(import_manifest["physical_profile"]["path"]).unlink()
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(
        CalibratedSuiteContractError,
        match="V100 schema-v2 consumer preflight failed",
    ):
        materialize_suite_binding(
            source=CalibratedConsumerInput(
                consumer_kind=V100_CONSUMER_KIND,
                consumer_manifest=import_manifest_path,
                consumer_payload=config_path,
            ),
            matrix=_v100_calibrated_suite_matrix(config),
            output_root=output_root,
        )
    assert not output_root.exists()


def _materialized_v100_suite(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    config_path, import_manifest_path, config, _import_manifest = (
        _materialized_v100_import(tmp_path, calibrated_pressure=True)
    )
    (
        _config,
        _runtime,
        calibration,
        _projection,
        full_binding,
    ) = full_replay.preflight_v100_config(config_path, import_manifest_path)
    profile_sha = full_binding["physical_profile_sha256"]
    anchor_dir = tmp_path / f"full_393_workflows_v100_{profile_sha[:12]}"
    anchor_dir.mkdir()
    json_payloads = {
        "algorithm1_metrics.json": {"status": "fixture"},
        "audit.json": {"status": "PASS", "paper_ready": False},
        "calibration_provenance.json": calibration,
        "config_resolved.json": {
            **config,
            "input_binding": full_binding,
            "output_dir_resolved": str(anchor_dir),
        },
        "dataset_manifest.json": config["dataset"]["expected_full_profile"],
        "fault_manifest.json": [],
        "runtime_aggregate.json": {},
        "workload_projection.json": {},
    }
    for name, payload in json_payloads.items():
        _write_json(anchor_dir / name, payload)
    (anchor_dir / "failure_requests.csv").write_text(
        "policy,event_id,request_id\n", encoding="utf-8"
    )
    (anchor_dir / "failure_stages.csv").write_text(
        "policy,event_id,workflow_id,stage_id\n", encoding="utf-8"
    )
    hashes = {
        path.name: _sha256(path)
        for path in sorted(anchor_dir.iterdir())
        if path.is_file()
    }
    hashes_path = _write_json(anchor_dir / "artifact_hashes.json", hashes)
    suite = {
        "schema_version": 2,
        "evidence_class": "sglsim_v100_baseline_suite_binding",
        "status": "READY",
        "design_semantics": DESIGN_SEMANTICS,
        "hardware_label": V100_HARDWARE_LABEL,
        "profile_semantics": V100_PROFILE_SEMANTICS,
        "paper_ready": False,
        "formal_ready": False,
        "base_experiment_config": _record(config_path),
        "base_import_manifest": _record(import_manifest_path),
        "anchor_result": {
            "path": str(anchor_dir),
            "artifact_hashes": _record(hashes_path),
            "config_sha256": full_binding["config"]["sha256"],
            "import_manifest_sha256": full_binding["import_manifest"][
                "sha256"
            ],
            "physical_profile_sha256": profile_sha,
            "source_manifest_sha256": full_binding[
                "source_manifest_sha256"
            ],
            "config_manifest_sha256": full_binding[
                "config_manifest_sha256"
            ],
            "config_verification_sha256": full_binding[
                "config_verification_sha256"
            ],
        },
        "ghostserve": {"data_shards": 8, "parity_shards": 2},
        "lumen": {"placement_lambda": 1.0, "dispatch_overload_factor": 1.25},
        "statistics": {"bootstrap_resamples": 10, "bootstrap_seed": 1},
        "audit": {"progress_every_chunks": 100},
    }
    suite_path = _write_json(tmp_path / "baseline-suite.json", suite)
    return suite_path, anchor_dir, full_binding


def _materialized_baseline_result(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    suite_path, anchor_dir, _full_binding = _materialized_v100_suite(tmp_path)
    (
        suite,
        base_config,
        _runtime,
        calibration,
        _projection,
        _anchor,
        input_binding,
    ) = baseline_replay.preflight_baseline_suite(suite_path)
    results = tmp_path / (
        "full_393_workflows_baselines_v100_"
        f"{input_binding['physical_profile_sha256'][:12]}"
    )
    results.mkdir()
    payloads = {
        "audit.json": {"status": "PASS", "paper_ready": False},
        "calibration_provenance.json": calibration,
        "config_resolved.json": {
            "suite": suite,
            "base_config": base_config,
            "runtime_config": {},
            "anchor_result_dir": str(anchor_dir),
            "input_binding": input_binding,
        },
        "dataset_manifest.json": base_config["dataset"]["expected_full_profile"],
        "paired_bootstrap.json": {},
        "restore_control_summary.json": {},
        "runtime_aggregate.json": {},
        "summary.json": {},
    }
    for name, payload in payloads.items():
        _write_json(results / name, payload)
    (results / "failure_requests.csv").write_text(
        "policy,event_id,request_id\n", encoding="utf-8"
    )
    hashes = {
        path.name: _sha256(path)
        for path in sorted(results.iterdir())
        if path.is_file()
    }
    _write_json(results / "artifact_hashes.json", hashes)
    return suite_path, results, suite, input_binding


def _v100_pressure_calibration(
    tmp_path: Path, *, manifest: Path, profile: Path, pilot_gate: Path
) -> Path:
    gate = json.loads(pilot_gate.read_text(encoding="utf-8"))
    pool_models = {
        f"hbm-vllm-dp{rank}": {
            "cuda_total_bytes": 32768,
            "non_shadow_resident_bytes": 16384,
            "pressure_free_bytes": 2048,
            "local_stall_threshold_ms": 5.0,
            "local_stall_probability_per_operation_window": 0.5,
            "sample_count": 4,
            "stable_window_count": 2,
            "pressure_window_count": 2,
            "memory_pressure_window_count": 1,
            "local_stall_window_count": 1,
            "local_operation_window_count": 2,
        }
        for rank in range(4)
    }
    record = lambda path: {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    aegis_profiles = [
        {
            "condition_id": row["condition_id"],
            "profile": {
                **row["profile"],
                "size_bytes": Path(row["profile"]["path"]).stat().st_size,
            },
        }
        for row in gate["profiles"]
        if str(row["condition_id"]).startswith("aegiskv-")
    ]
    return _write_json(
        tmp_path / "aimd-pressure-calibration.json",
        {
            "schema_version": 1,
            "evidence_class": full_replay.V100_AIMD_PRESSURE_CALIBRATION_EVIDENCE,
            "status": "PASS",
            "failures": [],
            "design_semantics": DESIGN_SEMANTICS,
            "hardware_label": V100_HARDWARE_LABEL,
            "campaign_id": gate["campaign_id"],
            "pressure_mode": full_replay.V100_AIMD_PRESSURE_MODE,
            "pilot_runtime_gate": record(pilot_gate),
            "source_manifest": record(manifest),
            "physical_profile": record(profile),
            "runtime_profiles": aegis_profiles,
            "pool_models": pool_models,
            "sglsim_quantitative_aimd_pressure_authorized": True,
            "formal_scale_submission_authorized": False,
            "paper_performance_claim_authorized": False,
            "paper_ready": False,
            "builder": record(manifest),
        },
    )


def test_v100_profile_import_preserves_four_distinct_physical_paths(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    config, import_manifest = import_profile(
        template_path=template,
        profile_path=profile,
        source_manifest_path=manifest,
        source_verification_path=verification,
        pilot_runtime_gate_path=pilot_gate,
        context_audit_path=context_audit,
        selected_scale_label="unit-primary",
        maximum_calibrated_context_tokens=4096,
    )
    runtime, provenance = calibrated_runtime_config(config)

    assert config["calibration"]["status"] == "READY"
    assert config["model"]["kv_abi"] == V100_COMPATIBILITY_IDENTITY["kv_abi"]
    assert config["model"]["serving_dtype"] == "half"
    assert config["workload"]["token_projection"]["factor"] == 0.004
    assert runtime.hbm_copy.bandwidth_bytes_per_ms == 100.0
    assert runtime.dram_copy.bandwidth_bytes_per_ms == 80.0
    assert runtime.hbm_load.ms_per_token == 10.0 / 50.0
    assert runtime.dram_load.ms_per_token == 10.0 / 40.0
    assert runtime.recompute.ms_per_token == 0.5
    assert runtime.recompute.setup_ms == 1.0
    assert provenance["layout_extrapolation_disclosed"] is False
    assert provenance["runtime_consumption_contract"] == (
        V100_RUNTIME_CONSUMPTION_CONTRACT
    )
    assert provenance["runtime_admission_decision_wire_values"] == {
        "BYPASS": 0,
        "ADMIT_DRAM": 1,
        "INVALID": 2,
        "ADMIT_HBM": 3,
    }
    assert import_manifest["status"] == "PASS"
    assert import_manifest["paper_ready"] is False
    assert import_manifest["runtime_consumption_contract"] == (
        V100_RUNTIME_CONSUMPTION_CONTRACT
    )
    assert import_manifest["runtime_config"][
        "recompute_fixed_overhead_ms"
    ] == 1.0
    assert import_manifest["runtime_alignment"] == {
        "four_condition_physical_gate_passed": True,
        "source_owned_hbm_dram_paths_independently_verified": True,
        "hbm_dram_timing_overlap_required": False,
        "config_manifest_and_verification_lineage_verified": True,
        "runtime_admission_decision_wire_values_verified": True,
        "four_gpu_clean_telemetry_verified": True,
        "aimd_alpha_response_and_direct_reclaim_exercised": True,
        "independent_peer_hbm_and_dram_recovery_exercised": True,
        "all_four_independent_residency_outcomes_exercised": True,
        "hbm_only_direct_reclaim_without_dram_prerequisite_exercised": True,
        "pilot_campaign_id": V100_CAMPAIGN_ID,
        "quantitative_aimd_simulation_authorized": False,
        "quantitative_aimd_claim_authorized": False,
        "paper_performance_claim_authorized": False,
    }
    assert import_manifest["aimd_pressure_model"] == {
        "mode": "active_source_request_count_proxy",
        "calibrated": False,
        "artifact": None,
        "active_source_request_threshold": 32,
        "quantitative_aimd_simulation_authorized": False,
        "quantitative_aimd_claim_authorized": False,
    }
    projection = full_replay._verified_context_projection(config)
    assert projection["factor"] == 0.004
    assert projection["selected"]["readiness"] == (
        "ELIGIBLE_AFTER_VALID_PHYSICAL_PROFILE"
    )


def test_v100_runtime_gate_preserves_mooncake_four_condition_check_names() -> None:
    assert MOONCAKE_FOUR_CONDITION_CHECKS.issubset(
        full_replay.V100_REQUIRED_RUNTIME_CHECKS
    )
    assert "source_hbm_dram_intervals_overlap" not in (
        full_replay.V100_REQUIRED_RUNTIME_CHECKS
    )
    assert {
        "clean_four_gpu_workload_telemetry",
        "source_owned_peer_hbm_and_direct_dram_protection_physical",
        "all_lineage_artifacts_exist_and_match_bytes",
        "target_local_stall_contracts_alpha_and_hbm_quota",
    }.issubset(full_replay.V100_REQUIRED_RUNTIME_CHECKS)


def test_v100_import_rejects_a100_dp2_schema_v2_profile(tmp_path: Path) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["hardware_label"] = A100_DP2_HARDWARE_LABEL
    value["layout"] = {"tp_size": 1, "pp_size": 1, "dp_size": 2}
    _write_json(profile, value)

    with pytest.raises(
        ValueError, match="Mooncake profile provenance contract did not pass"
    ):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_formal_preflight_binds_source_config_and_profile_sha(
    tmp_path: Path,
) -> None:
    config_path, manifest_path, _config, _manifest = (
        _materialized_v100_import(tmp_path)
    )
    config, runtime, provenance, projection, binding = (
        full_replay.preflight_v100_config(config_path, manifest_path)
    )

    assert config["cluster"] == {"dp_size": 4, "pp_size": 1, "tp_size": 1}
    assert runtime.hbm_load.ms_per_token == pytest.approx(10.0 / 50.0)
    assert provenance["runtime_consumption_contract"] == (
        V100_RUNTIME_CONSUMPTION_CONTRACT
    )
    assert projection["factor"] == 0.004
    assert binding["config"]["sha256"] == _sha256(config_path)
    assert binding["import_manifest"]["sha256"] == _sha256(manifest_path)
    assert binding["physical_profile_sha256"] == config["calibration"][
        "physical_profile"
    ]["sha256"]
    assert binding["source_manifest_sha256"] == config["calibration"][
        "source_manifest"
    ]["sha256"]
    assert binding["config_manifest_sha256"] == config["calibration"][
        "config_manifest"
    ]["sha256"]
    assert binding["config_verification_sha256"] == config["calibration"][
        "config_verification"
    ]["sha256"]


def test_v100_formal_preflight_rejects_config_tamper_before_trace_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, manifest_path, config, _manifest = _materialized_v100_import(
        tmp_path
    )
    config["experiment_id"] = "tampered-after-import"
    _write_json(config_path, config)
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        full_replay,
        "parse_args",
        lambda: argparse.Namespace(
            config=config_path,
            import_manifest=manifest_path,
            output_dir=output,
            dataset=tmp_path / "must-not-load.jsonl",
            limit_workflows=None,
            skip_dataset_sha=False,
        ),
    )
    monkeypatch.setattr(
        full_replay,
        "load_weka_workflows",
        lambda *_args, **_kwargs: pytest.fail("tampered config loaded trace"),
    )

    with pytest.raises(ValueError, match="imported experiment config.*sha256"):
        full_replay.main()
    assert not output.exists()


def test_v100_formal_preflight_rehashes_live_source_manifest_entries(
    tmp_path: Path,
) -> None:
    config_path, manifest_path, _config, import_manifest = (
        _materialized_v100_import(tmp_path)
    )
    source_root = Path(
        json.loads(
            Path(import_manifest["source_manifest"]["path"]).read_text(
                encoding="utf-8"
            )
        )["repository_root"]
    )
    (source_root / "source.py").write_text(
        "print('tampered')\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source manifest entry .*mismatch"):
        full_replay.preflight_v100_config(config_path, manifest_path)


def test_full_393_replay_requires_physical_aimd_profile_before_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, manifest_path, config, _manifest = _materialized_v100_import(
        tmp_path
    )
    profile_sha = config["calibration"]["physical_profile"]["sha256"]
    output = tmp_path / f"full_393_workflows_v100_{profile_sha[:12]}"
    monkeypatch.setattr(
        full_replay,
        "parse_args",
        lambda: argparse.Namespace(
            config=config_path,
            import_manifest=manifest_path,
            output_dir=output,
            dataset=tmp_path / "must-not-load.jsonl",
            limit_workflows=None,
            skip_dataset_sha=False,
        ),
    )
    monkeypatch.setattr(
        full_replay,
        "load_weka_workflows",
        lambda *_args, **_kwargs: pytest.fail(
            "uncalibrated full replay loaded trace"
        ),
    )

    with pytest.raises(ValueError, match="physical V100 AIMD pressure"):
        full_replay.main()
    assert not output.exists()


def test_v100_import_rejects_runtime_contract_drift_and_negative_recompute(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["runtime_consumption_contract"][
        "protection_rate_is_subtracted_from_admission_gain"
    ] = True
    _write_json(profile, value)
    with pytest.raises(ValueError, match="provenance contract"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )

    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["metrics"]["recompute_fixed_overhead_ms"] = -0.001
    _write_json(profile, value)
    with pytest.raises(ValueError, match="recompute_fixed_overhead_ms"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_final_audit_accepts_only_current_v100_consumption_contract() -> None:
    metrics = {
        "recompute_ms_per_token": 0.5,
        "recompute_fixed_overhead_ms": 0.0,
        "peer_hbm_protection_effective_bytes_per_ms": 100.0,
        "direct_dram_protection_effective_bytes_per_ms": 80.0,
        "peer_hbm_recovery_effective_bytes_per_ms": 50.0,
        "dram_recovery_effective_bytes_per_ms": 40.0,
    }
    current = {
        "calibration_semantics": V100_PROFILE_SEMANTICS,
        "result_label": V100_HARDWARE_LABEL,
        "compatibility_identity": V100_COMPATIBILITY_IDENTITY,
        "runtime_consumption_contract": V100_RUNTIME_CONSUMPTION_CONTRACT,
        "metrics": metrics,
    }
    assert full_replay._valid_v100_audit_calibration(current)
    assert not full_replay._valid_v100_audit_calibration(
        {**current, "calibration_semantics": "four_independent_physical_paths_v1"}
    )
    assert not full_replay._valid_v100_audit_calibration(
        {
            **current,
            "metrics": {**metrics, "recompute_fixed_overhead_ms": -1.0},
        }
    )
    assert not full_replay._valid_v100_audit_calibration(
        {
            **current,
            "runtime_consumption_contract": {
                **V100_RUNTIME_CONSUMPTION_CONTRACT,
                "cross_tier_transfer_cost_is_charged": True,
            },
        }
    )


def test_protection_rates_do_not_change_algorithm2_admission_gain() -> None:
    def runtime(hbm_bw: float, dram_bw: float) -> full_replay.AegisFullDesignRuntime:
        return full_replay.AegisFullDesignRuntime(
            full_replay.AegisFullDesignConfig(
                hbm_capacity_bytes=1 << 20,
                dram_capacity_bytes=1 << 20,
                kv_bytes_per_token=1024,
                hbm_copy=full_replay.CalibratedTransferPath(
                    0.0, hbm_bw, "protection-hbm"
                ),
                dram_copy=full_replay.CalibratedTransferPath(
                    0.0, dram_bw, "protection-dram"
                ),
                recompute=full_replay.RecoveryTiming(0.0, 1.0),
                hbm_load=full_replay.RecoveryTiming(0.0, 0.1),
                dram_load=full_replay.RecoveryTiming(0.0, 0.2),
                prefix_chunk_tokens=16,
                decode_chunk_tokens=4,
            )
        )

    decisions = []
    for index, value in enumerate((runtime(10.0, 5.0), runtime(10_000.0, 5_000.0))):
        value.register_request(
            request_id=f"r{index}",
            source_worker="source",
            source_failure_domain="source-domain",
            hbm_target="target",
            hbm_failure_domain="target-domain",
            route_version=1,
            prompt_tokens=16,
            output_tokens=0,
            slack_ms=0.0,
            model_revision="model",
            kv_abi="kv",
            parallel_layout="dp4-pp1-tp1-fp16",
            time_ms=0.0,
        )
        decisions.append(
            value.seal_chunk(
                f"r{index}",
                full_replay.SealedChunk(0, 0, 16, "prefix"),
                time_ms=1.0,
            )
        )

    slow, fast = decisions
    assert slow.hbm.admitted and slow.dram.admitted
    assert fast.hbm.admitted and fast.dram.admitted
    assert slow.hbm.gain_ms == pytest.approx(fast.hbm.gain_ms)
    assert slow.dram.gain_ms == pytest.approx(fast.dram.gain_ms)


def test_full_output_name_is_profile_hash_derived_and_immutable(
    tmp_path: Path,
) -> None:
    profile_sha = "a" * 64
    expected = tmp_path / f"full_393_workflows_v100_{profile_sha[:12]}"
    assert full_replay._derived_full_output_dir(
        expected,
        profile_sha256=profile_sha,
        limit_workflows=None,
    ) == expected
    with pytest.raises(ValueError, match="derived from the physical profile"):
        full_replay._derived_full_output_dir(
            tmp_path / "full_393_workflows_four_path_v2",
            profile_sha256=profile_sha,
            limit_workflows=None,
        )
    expected.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        full_replay._derived_full_output_dir(
            expected,
            profile_sha256=profile_sha,
            limit_workflows=None,
        )


def test_baseline_suite_binds_same_v100_profile_config_and_anchor(
    tmp_path: Path,
) -> None:
    suite_path, anchor_dir, full_binding = _materialized_v100_suite(tmp_path)
    (
        suite,
        base_config,
        runtime,
        calibration,
        projection,
        observed_anchor,
        binding,
    ) = baseline_replay.preflight_baseline_suite(suite_path)

    assert suite["status"] == "READY"
    assert base_config["cluster"] == {"dp_size": 4, "pp_size": 1, "tp_size": 1}
    assert runtime.hbm_load.ms_per_token == pytest.approx(10.0 / 50.0)
    assert calibration["runtime_consumption_contract"] == (
        V100_RUNTIME_CONSUMPTION_CONTRACT
    )
    assert projection["factor"] == 0.004
    assert observed_anchor == anchor_dir
    assert binding["config"] == full_binding["config"]
    assert binding["physical_profile_sha256"] == full_binding[
        "physical_profile_sha256"
    ]
    assert binding["config_manifest_sha256"] == full_binding[
        "config_manifest_sha256"
    ]
    assert binding["config_verification_sha256"] == full_binding[
        "config_verification_sha256"
    ]
    assert binding["anchor_result"]["path"] == str(anchor_dir)


def test_baseline_suite_rejects_v100_config_lineage_tamper(tmp_path: Path) -> None:
    suite_path, _anchor_dir, _full_binding = _materialized_v100_suite(tmp_path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["anchor_result"]["config_verification_sha256"] = "0" * 64
    _write_json(suite_path, suite)

    with pytest.raises(ValueError, match="anchor identity contract"):
        baseline_replay.preflight_baseline_suite(suite_path)


def test_baseline_suite_builder_emits_v100_hash_binding(tmp_path: Path) -> None:
    suite_path, anchor_dir, full_binding = _materialized_v100_suite(tmp_path)
    existing = json.loads(suite_path.read_text(encoding="utf-8"))
    template = _write_json(
        tmp_path / "baseline-policy-template.json",
        {
            "status": "INVALIDATED_UPSTREAM_TCP_TRANSPORT",
            **{
                key: existing[key]
                for key in (
                    "ghostserve",
                    "lumen",
                    "statistics",
                    "audit",
                )
            },
            "fidelity_contract": "unit",
            "primary_policies": [],
            "dedicated_controls": [],
            "restore_controls": {},
        },
    )
    built = build_suite_binding(
        template_path=template,
        base_config_path=Path(existing["base_experiment_config"]["path"]),
        import_manifest_path=Path(existing["base_import_manifest"]["path"]),
        anchor_result_dir=anchor_dir,
    )

    assert built["status"] == "READY"
    assert built["schema_version"] == 2
    assert built["anchor_result"]["physical_profile_sha256"] == (
        full_binding["physical_profile_sha256"]
    )
    assert built["anchor_result"]["config_manifest_sha256"] == (
        full_binding["config_manifest_sha256"]
    )
    assert built["anchor_result"]["config_verification_sha256"] == (
        full_binding["config_verification_sha256"]
    )
    assert built["paper_ready"] is False
    assert built["formal_ready"] is False


def test_baseline_suite_rejects_anchor_tamper_before_trace_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_path, anchor_dir, full_binding = _materialized_v100_suite(tmp_path)
    (anchor_dir / "runtime_aggregate.json").write_text(
        '{"tampered": true}\n', encoding="utf-8"
    )
    output = tmp_path / (
        "full_393_workflows_baselines_v100_"
        f"{full_binding['physical_profile_sha256'][:12]}"
    )
    monkeypatch.setattr(
        baseline_replay,
        "parse_args",
        lambda: argparse.Namespace(
            config=suite_path,
            dataset=tmp_path / "must-not-load.jsonl",
            output_dir=output,
            limit_workflows=None,
            skip_dataset_sha=False,
        ),
    )
    monkeypatch.setattr(
        baseline_replay,
        "load_weka_workflows",
        lambda *_args, **_kwargs: pytest.fail("tampered anchor loaded trace"),
    )

    with pytest.raises(ValueError, match="anchor artifact sha256 mismatch"):
        baseline_replay.main()
    assert not output.exists()


def test_baseline_and_figure_outputs_are_hash_derived_and_immutable(
    tmp_path: Path,
) -> None:
    profile_sha = "b" * 64
    baseline_output = tmp_path / (
        f"full_393_workflows_baselines_v100_{profile_sha[:12]}"
    )
    assert baseline_replay._derived_baseline_output_dir(
        baseline_output,
        profile_sha256=profile_sha,
        limit_workflows=None,
    ) == baseline_output
    with pytest.raises(ValueError, match="derived from the physical profile"):
        baseline_replay._derived_baseline_output_dir(
            tmp_path / "full_393_workflows_four_path_v2",
            profile_sha256=profile_sha,
            limit_workflows=None,
        )
    figure_output = tmp_path / f"figures_v100_{profile_sha[:12]}"
    assert make_figures._derived_figure_output_dir(
        figure_output, profile_sha256=profile_sha
    ) == figure_output
    with pytest.raises(ValueError, match="derived from the physical profile"):
        make_figures._derived_figure_output_dir(
            tmp_path / "figures_four_path_v2",
            profile_sha256=profile_sha,
        )
    baseline_output.mkdir()
    figure_output.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        baseline_replay._derived_baseline_output_dir(
            baseline_output,
            profile_sha256=profile_sha,
            limit_workflows=None,
        )
    with pytest.raises(FileExistsError, match="immutable"):
        make_figures._derived_figure_output_dir(
            figure_output, profile_sha256=profile_sha
        )


def test_figure_preflight_uses_bound_v100_anchor_and_rejects_result_tamper(
    tmp_path: Path,
) -> None:
    suite_path, results, suite, input_binding = _materialized_baseline_result(
        tmp_path
    )
    resolved = make_figures._verify_baseline_result(
        results,
        suite=suite,
        input_binding=input_binding,
    )
    assert Path(resolved["anchor_result_dir"]) == Path(
        input_binding["anchor_result"]["path"]
    )
    assert "four_path_v2" not in resolved["anchor_result_dir"]

    (results / "summary.json").write_text(
        '{"tampered": true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="result artifact sha256 mismatch"):
        make_figures._verify_baseline_result(
            results,
            suite=json.loads(suite_path.read_text(encoding="utf-8")),
            input_binding=input_binding,
        )


def test_v100_profile_import_enables_only_hash_bound_physical_aimd_pressure(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    pressure = _v100_pressure_calibration(
        tmp_path,
        manifest=manifest,
        profile=profile,
        pilot_gate=pilot_gate,
    )
    config, import_manifest = import_profile(
        template_path=template,
        profile_path=profile,
        source_manifest_path=manifest,
        source_verification_path=verification,
        pilot_runtime_gate_path=pilot_gate,
        aimd_pressure_calibration_path=pressure,
        context_audit_path=context_audit,
        selected_scale_label="unit-primary",
        maximum_calibrated_context_tokens=4096,
    )
    _runtime, provenance = calibrated_runtime_config(config)

    assert config["aimd"]["pressure_mode"] == full_replay.V100_AIMD_PRESSURE_MODE
    assert config["aimd"]["pressure_calibrated"] is True
    assert "pressure_active_source_requests" not in config["aimd"]
    assert provenance["aimd_pressure"][
        "sglsim_quantitative_aimd_pressure_authorized"
    ] is True
    assert import_manifest["runtime_alignment"][
        "quantitative_aimd_simulation_authorized"
    ] is True
    assert import_manifest["runtime_alignment"][
        "quantitative_aimd_claim_authorized"
    ] is False
    assert import_manifest["runtime_alignment"][
        "paper_performance_claim_authorized"
    ] is False
    assert import_manifest["paper_ready"] is False


def test_v100_profile_import_rejects_tampered_pressure_calibration_input(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    pressure = _v100_pressure_calibration(
        tmp_path,
        manifest=manifest,
        profile=profile,
        pilot_gate=pilot_gate,
    )
    value = json.loads(pressure.read_text(encoding="utf-8"))
    value["pilot_runtime_gate"]["sha256"] = "0" * 64
    _write_json(pressure, value)

    with pytest.raises(ValueError, match="pressure calibration contract"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            aimd_pressure_calibration_path=pressure,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rejects_transport_or_manifest_drift(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["transport_signature"]["peer_hbm"]["protocol"] = "tcp"
    _write_json(profile, value)
    with pytest.raises(ValueError, match="physical profile contract"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )

    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance contract"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rejects_fp16_compatibility_drift(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(profile.read_text(encoding="utf-8"))
    value["compatibility_identity"]["kv_abi"] = "qwen3-8b-bf16-invalid"
    _write_json(profile, value)
    with pytest.raises(ValueError, match="provenance contract"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rejects_cross_layout_projection(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(template.read_text(encoding="utf-8"))
    value["cluster"] = {"dp_size": 8, "pp_size": 2, "tp_size": 1}
    _write_json(template, value)

    with pytest.raises(ValueError, match="layout must match the physical profile"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rejects_undisclosed_aimd_pressure_proxy(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(template.read_text(encoding="utf-8"))
    value["aimd"].pop("pressure_mode")
    _write_json(template, value)

    with pytest.raises(ValueError, match="AIMD pressure proxy"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rejects_failed_runtime_gate(tmp_path: Path) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(pilot_gate.read_text(encoding="utf-8"))
    value["checks"][0]["passed"] = False
    _write_json(pilot_gate, value)

    with pytest.raises(ValueError, match="runtime gate did not pass"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rejects_pre_v20_runtime_gate_class(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(pilot_gate.read_text(encoding="utf-8"))
    value["evidence_class"] = "aegiskv_v100_dp4_pilot_runtime_profile_gate"
    _write_json(pilot_gate, value)

    with pytest.raises(ValueError, match="runtime gate did not pass"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rehashes_config_verification(tmp_path: Path) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    physical = json.loads(profile.read_text(encoding="utf-8"))
    config_verification = Path(physical["config_verification"]["path"])
    value = json.loads(config_verification.read_text(encoding="utf-8"))
    value["status"] = "FAIL"
    value["failures"] = ["tampered"]
    _write_json(config_verification, value)

    with pytest.raises(ValueError, match="config verification artifact sha256"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rejects_cross_condition_config_lineage(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    gate = json.loads(pilot_gate.read_text(encoding="utf-8"))
    runtime_record = gate["profiles"][0]["profile"]
    runtime_profile = Path(runtime_record["path"])
    value = json.loads(runtime_profile.read_text(encoding="utf-8"))
    value["input_lineage"]["config_manifest"]["sha256"] = "9" * 64
    _write_json(runtime_profile, value)
    runtime_record["sha256"] = _sha256(runtime_profile)
    runtime_record["size_bytes"] = runtime_profile.stat().st_size
    _write_json(pilot_gate, gate)

    with pytest.raises(ValueError, match="config lineage artifact identity"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_requires_all_four_independent_outcomes(
    tmp_path: Path,
) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    value = json.loads(pilot_gate.read_text(encoding="utf-8"))
    value["checks"] = [
        row
        for row in value["checks"]
        if row["name"] != "all_four_independent_residency_outcomes_exercised"
    ]
    _write_json(pilot_gate, value)

    with pytest.raises(ValueError, match="runtime gate did not pass"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_v100_profile_import_rejects_runtime_profile_tamper(tmp_path: Path) -> None:
    template, profile, manifest, verification, context_audit, pilot_gate = (
        _v100_profile_bundle(tmp_path)
    )
    gate = json.loads(pilot_gate.read_text(encoding="utf-8"))
    runtime_profile = Path(gate["profiles"][0]["profile"]["path"])
    runtime_profile.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime profile.*sha256 mismatch"):
        import_profile(
            template_path=template,
            profile_path=profile,
            source_manifest_path=manifest,
            source_verification_path=verification,
            pilot_runtime_gate_path=pilot_gate,
            context_audit_path=context_audit,
            selected_scale_label="unit-primary",
            maximum_calibrated_context_tokens=4096,
        )


def test_parallel_layout_identity_uses_calibrated_kv_dtype() -> None:
    base = {
        "cluster": {"dp_size": 4, "pp_size": 1, "tp_size": 1},
        "model": {"serving_dtype": "half", "kv_cache_dtype": "fp16"},
    }
    assert full_replay._parallel_layout_id(base) == "dp4-pp1-tp1-fp16"

    bf16 = {**base, "model": {"kv_cache_dtype": "bfloat16"}}
    assert full_replay._parallel_layout_id(bf16) == "dp4-pp1-tp1-bf16"

    missing = {**base, "model": {}}
    with pytest.raises(ValueError, match="KV cache dtype"):
        full_replay._parallel_layout_id(missing)


def test_physical_aimd_signal_uses_headroom_or_local_hbm_operation_stall() -> None:
    runtime = full_replay.AegisFullDesignRuntime(
        full_replay.AegisFullDesignConfig(
            hbm_capacity_bytes=1 << 20,
            dram_capacity_bytes=1 << 20,
            kv_bytes_per_token=1024,
            hbm_copy=full_replay.CalibratedTransferPath(1.0, 1024.0, "unit"),
            dram_copy=full_replay.CalibratedTransferPath(1.0, 1024.0, "unit"),
            recompute=full_replay.RecoveryTiming(0.0, 1.0),
            hbm_load=full_replay.RecoveryTiming(0.0, 0.1),
            dram_load=full_replay.RecoveryTiming(0.0, 0.2),
            prefix_chunk_tokens=4,
            decode_chunk_tokens=2,
        )
    )
    runtime.register_request(
        request_id="r",
        source_worker="source",
        source_failure_domain="source-domain",
        hbm_target="target-0",
        hbm_failure_domain="target-domain",
        route_version=1,
        prompt_tokens=4,
        output_tokens=0,
        slack_ms=0.0,
        model_revision="m",
        kv_abi="kv",
        parallel_layout="dp1-pp1-tp1-fp16",
        time_ms=0.0,
    )
    runtime.seal_chunk(
        "r",
        full_replay.SealedChunk(0, 0, 4, "prefix"),
        time_ms=1.0,
    )
    base = {
        "cuda_total_bytes": 10000,
        "non_shadow_resident_bytes": 4000,
        "local_stall_threshold_ms": 5.0,
        "local_stall_probability_per_operation_window": 0.5,
    }
    memory = _physical_aimd_pressure(
        runtime,
        {**base, "pressure_free_bytes": 2500},
        time_ms=1.0,
        local_stall_draw=0.9,
    )
    assert memory["memory_pressure"] is True
    assert memory["local_stall_pressure"] is False

    runtime.finish_request("r", time_ms=6.0)
    stall = _physical_aimd_pressure(
        runtime,
        {**base, "pressure_free_bytes": 1000},
        time_ms=6.0,
        local_stall_draw=0.1,
    )
    assert stall["memory_pressure"] is False
    assert stall["local_stall_pressure"] is True
    assert stall["local_resource_operations"]["reclaim_count"] == 1
    assert stall["pressure"] is True

def test_token_projection_scales_tokens_without_changing_workflow_timing() -> None:
    request = WekaRequest(
        rid="r0",
        workflow_id="w0",
        stage_id=0,
        agent_id="a0",
        role="root",
        model="source-model",
        prompt_tokens=100,
        output_tokens=10,
        arrival_ms=5.0,
        api_time_ms=7.0,
        ttft_ms=2.0,
    )
    workflow = WekaWorkflow(
        workflow_id="w0",
        models=("source-model",),
        stages=(
            WekaStage(
                stage_id=0,
                kind="sequential",
                branches=(
                    WekaAgentBranch(
                        agent_id="a0", role="root", requests=(request,)
                    ),
                ),
            ),
        ),
    )
    projected = full_replay._scale_workflows([workflow], 0.004)[0]
    projected_request = projected.requests[0]
    assert projected_request.prompt_tokens == 1
    assert projected_request.output_tokens == 1
    assert projected_request.arrival_ms == request.arrival_ms
    assert projected_request.api_time_ms == request.api_time_ms
    assert projected.finish_ms == workflow.finish_ms


def test_slack_due_recovery_breaks_ties_by_larger_current_harm() -> None:
    def replay_request(
        request_id: str, *, agent_id: str, global_start_ms: float
    ) -> full_replay.ReplayRequest:
        row = WekaRequest(
            rid=request_id,
            workflow_id="workflow",
            stage_id=0,
            agent_id=agent_id,
            role="worker",
            model="test-model",
            prompt_tokens=100,
            output_tokens=10,
            arrival_ms=0.0,
            api_time_ms=100.0,
            ttft_ms=10.0,
        )
        return full_replay.ReplayRequest(
            row=row,
            workflow_base_ms=0.0,
            workflow_origin_ms=0.0,
            global_start_ms=global_start_ms,
            global_finish_ms=global_start_ms + 100.0,
            prefill_finish_ms=global_start_ms + 10.0,
            branch_finish_ms=1000.0,
            stage_finish_ms=1000.0,
            source_rank=0,
            target_rank=1,
            prefix_chunks=1,
            decode_chunks=1,
            source_worker="source-0-g0",
            current_target="target-1-g0",
            started=True,
        )

    low_harm = replay_request("a-low-harm", agent_id="low", global_start_ms=0.0)
    high_harm = replay_request(
        "z-high-harm", agent_id="high", global_start_ms=100.0
    )
    decisions = {
        request_id: full_replay.RecoveryDecision(
            request_id=request_id,
            action="RECOMPUTE",
            restored_tokens=0,
            recompute_tokens=100,
            recovery_ms=recovery_ms,
            slack_ms=10.0,
            visible_harm_ms=visible_harm_ms,
            route_version=1,
            activated_target="target-1-g0",
        )
        for request_id, recovery_ms, visible_harm_ms in (
            (low_harm.row.rid, 20.0, 10.0),
            (high_harm.row.rid, 100.0, 90.0),
        )
    }
    failure_rows, _ = full_replay.evaluate_failure_snapshot(
        event_id="failure",
        failure_time_ms=500.0,
        affected=[low_harm, high_harm],
        decisions={name: decisions for name in ("recompute", "zero_slack", "aegiskv")},
        remaining_slack={low_harm.row.rid: 10.0, high_harm.row.rid: 10.0},
        stage_branch_finishes={
            ("workflow", 0): {"low": 1000.0, "high": 1000.0}
        },
    )

    latest = sorted(
        (row for row in failure_rows if row["policy"] == "aegiskv_latest"),
        key=lambda row: row["queue_position"],
    )
    assert [row["request_id"] for row in latest] == [
        high_harm.row.rid,
        low_harm.row.rid,
    ]


def test_direct_dram_protection_requires_every_pp_stage(tmp_path: Path) -> None:
    config, direct_path, _hbm_path = _bundle(tmp_path)
    gate = json.loads(direct_path.read_text(encoding="utf-8"))
    gate["details"]["stage_aware_peer_hbm"]["files"] = gate["details"][
        "stage_aware_peer_hbm"
    ]["files"][:1]

    with pytest.raises(ValueError, match="incomplete direct DRAM PP group"):
        _pp_direct_dram_protection_bandwidth(gate, pp_size=2)


def test_event_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    _config, direct_path, _hbm_path = _bundle(tmp_path)
    gate = json.loads(direct_path.read_text(encoding="utf-8"))
    event_path = Path(gate["details"]["stage_aware_peer_hbm"]["files"][0]["path"])
    event_path.write_text(event_path.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact sha256 mismatch"):
        _pp_direct_dram_protection_bandwidth(gate, pp_size=2)


def test_incoherent_hbm_recovery_upstream_chain_fails_closed(tmp_path: Path) -> None:
    config, _direct_path, hbm_path = _bundle(tmp_path)
    gate = json.loads(hbm_path.read_text(encoding="utf-8"))
    gate["details"]["source_config_manifest"]["upstream_hbm_integration_gate"][
        "sha256"
    ] = "0" * 64
    _write_json(hbm_path, gate)

    with pytest.raises(ValueError, match="upstream gate sha256 mismatch"):
        _legacy_calibrated_runtime_config_diagnostic_only(config)
