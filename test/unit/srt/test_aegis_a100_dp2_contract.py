from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from AegisKV.experiments.real_weka_full_design.a100_dp2_contract import (
    A100DP2ContractError,
    A100DP2InputPaths,
    A100DP2PhysicalProfile,
    A100_CONSUMER_COMPATIBILITY_IDENTITY,
    A100_REQUIRED_MECHANISM_CHECKS,
    AIMD_CONSUMPTION_CONTRACT,
    CAMPAIGN_ID,
    CONFIG_LAYOUT,
    DESIGN_SEMANTICS,
    GPU_TELEMETRY_COLUMNS,
    GPU_TELEMETRY_CONTRACT,
    GPU_TELEMETRY_UNITS,
    HARDWARE_LABEL,
    HARDWARE_PROFILE,
    PHYSICAL_LAYOUT,
    PHYSICAL_PROFILE_SEMANTICS,
    PROFILE_METRICS,
    RUNTIME_CONSUMPTION_CONTRACT,
    TRANSFER_COST_MODEL_SCHEMA,
    TRANSFER_DURATION_FORMULA,
    TRANSFER_AUDIT_COUNT,
    TRANSFER_PATH_EVENTS,
    TRANSFER_PATH_TREATMENTS,
    TRANSFER_REPEATS_PER_TREATMENT,
    WIRE_VALUES,
    artifact_record,
    canonical_sha256,
    sha256_file,
    validate_a100_dp2_bundle,
)
from AegisKV.experiments.real_weka_full_design.calibrated_suite_contract import (
    A100_CONSUMER_KIND,
    V100_CONSUMER_KIND,
    CalibratedConsumerInput,
    CalibratedSuiteContractError,
    materialize_suite_binding,
    preflight_suite_runner,
    validate_suite_binding,
)
from AegisKV.experiments.real_weka_full_design.preflight_mooncake_a100_dp2 import (
    preflight_and_write,
)


MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
AIMD_RUNTIME_CONFIG = {
    "total_hbm_bytes": 2147483648,
    "interval_s": 5.0,
    "pressure_free_bytes": 3221225472,
    "local_stall_ms": 6.0,
    "initial_alpha": 0.5,
    "min_alpha": 0.1,
    "max_alpha": 0.8,
    "decrease_factor": 0.5,
    "increase_step": 0.05,
    "stable_window_threshold": 3,
}


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_gpu_csv(
    path: Path, *, gpu1_sm: float = 31.0, gpu1_fb: float = 2048.0
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [GPU_TELEMETRY_COLUMNS, GPU_TELEMETRY_UNITS]
    for second in range(3):
        for gpu_id in (0, 1):
            sm = 25.0 + second if gpu_id == 0 else gpu1_sm
            fb = 1024.0 + second if gpu_id == 0 else gpu1_fb
            rows.append(
                (
                    "20260824",
                    f"12:00:0{second * 2}",
                    str(gpu_id),
                    "100",
                    "45",
                    "40",
                    str(sm),
                    "20",
                    "0",
                    "0",
                    "0",
                    "0",
                    "1215",
                    "1410",
                    str(fb),
                    "64",
                    "0",
                    "0",
                    "0",
                    "0",
                    "1",
                    "2",
                )
            )
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def _telemetry_summary(path: Path, *, workload: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "artifact": artifact_record(path),
        "status": "PASS",
        "failures": [],
        "expected_device_count": 2,
        "gpu_ids": ["0", "1"],
        "complete_frame_count": 3,
        "peak_gpu_utilization_pct": 33.0,
        "peak_total_fb_memory_mb": 3076.0,
    }
    if workload:
        value["workload_window"] = {
            "status": "PASS",
            "failures": [],
            "complete_frame_count": 3,
            "duration_s": 2.0,
        }
    return value


def _producer_per_gpu_telemetry() -> dict[str, Any]:
    return {
        "status": "PASS",
        "failures": [],
        "complete_frame_count": 3,
        "observed_gpu_ids": ["0", "1"],
        "devices": {
            "0": {
                "peak_gpu_utilization_pct": 27.0,
                "peak_fb_memory_mb": 1026.0,
            },
            "1": {
                "peak_gpu_utilization_pct": 31.0,
                "peak_fb_memory_mb": 2048.0,
            },
        },
    }


def _zero_path() -> dict[str, Any]:
    return {
        "count": 0,
        "bytes": 0,
        "duration_ms": {"minimum": 0.0},
        "physical": False,
        "initiated_by_owner": False,
        "owner": "",
        "protocol_counts": {},
        "physical_interconnect_counts": {},
    }


def _physical_path(
    *, owner: str, protocol: str, interconnect: str
) -> dict[str, Any]:
    return {
        "count": 1,
        "bytes": 4096,
        "duration_ms": {"minimum": 1.0},
        "physical": True,
        "initiated_by_owner": True,
        "owner": owner,
        "protocol_counts": {protocol: 1},
        "physical_interconnect_counts": {interconnect: 1},
    }


def _affine_transfer_samples(
    event: str,
    *,
    calibration_run_id: str,
    calibration_stream_id: str,
    per_transfer_setup_ms: float,
    first_transfer_warmup_ms: float,
    steady_state_bytes_per_ms: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    loaded = event.endswith("_load")
    for ordinal, byte_count in ((0, 100), (1, 100), (2, 200)):
        duration_ms = (
            per_transfer_setup_ms
            + (first_transfer_warmup_ms if ordinal == 0 else 0.0)
            + byte_count / steady_state_bytes_per_ms
        )
        row: dict[str, Any] = {
            "event": event,
            "bytes": byte_count,
            "duration_ms": duration_ms,
            "calibration_run_id": calibration_run_id,
            "calibration_stream_id": calibration_stream_id,
            "calibration_transfer_ordinal": ordinal,
        }
        if loaded:
            row.update({"loaded": True, "loaded_bytes": byte_count})
        else:
            row["transfer_succeeded"] = True
        rows.append(row)
    return rows


def _declared_transfer_cost_model(
    event: str,
    *,
    per_transfer_setup_ms: float,
    first_transfer_warmup_ms: float,
    steady_state_bytes_per_ms: float,
    stream_ids: list[str],
) -> dict[str, Any]:
    stream_count = len(stream_ids)
    return {
        "schema": TRANSFER_COST_MODEL_SCHEMA,
        "duration_formula": TRANSFER_DURATION_FORMULA,
        "event": event,
        "sample_count": stream_count * 3,
        "stream_count": stream_count,
        "stream_ids": sorted(stream_ids),
        "warmup_sample_count": stream_count,
        "steady_state_sample_count": stream_count * 2,
        "distinct_steady_state_payload_bytes": [100, 200],
        "fit_r_squared": 1.0,
        "per_transfer_setup_ms": per_transfer_setup_ms,
        "first_transfer_warmup_ms": first_transfer_warmup_ms,
        "steady_state_bytes_per_ms": steady_state_bytes_per_ms,
    }


def _condition_events(
    treatment: str, failure_method: str
) -> dict[str, Any]:
    names = (
        "peer_hbm_protection",
        "direct_dram_protection",
        "peer_hbm_recovery",
        "dram_recovery",
    )
    if treatment == "no-protection":
        return {
            "admission_decisions": {},
            "independent_tier_outcomes": {},
            "source_owned_paths": {name: _zero_path() for name in names},
            "path_confirmations": {},
            "forbidden_semantics": {},
            "aimd": {"window_count": 0},
        }

    total_hbm_bytes = AIMD_RUNTIME_CONFIG["total_hbm_bytes"]
    cuda_free_bytes = 4294967296
    cuda_total_bytes = 85899345920

    def timeline_event(
        observed_at_ms: float,
        *,
        pressure: bool,
        local_pressure: bool,
        alpha: float,
        stable_windows: int,
        reclaim_count: int = 0,
    ) -> dict[str, Any]:
        quota_bytes = int(total_hbm_bytes * alpha)
        return {
            "observed_at_ms": observed_at_ms,
            "pressure": pressure,
            "memory_pressure": False,
            "local_stall_pressure": local_pressure,
            "local_stall_threshold_ms": 6.0,
            "pressure_reasons": (
                ["peer_hbm_allocation_reclaim_stall"]
                if local_pressure
                else []
            ),
            "alpha": alpha,
            "quota_bytes": quota_bytes,
            "allocated_bytes_before_control": 0.0,
            "allocated_bytes": 0.0,
            "cuda_free_bytes": float(cuda_free_bytes),
            "cuda_total_bytes": float(cuda_total_bytes),
            "stable_windows": stable_windows,
            "quota_satisfied": True,
            "reclaimed_slot_count": reclaim_count,
            "new_hbm_admissions_paused_before_reclaim": pressure,
            "new_hbm_admissions_resumed_after_reclaim": pressure,
        }

    pool_0_timeline = [
        timeline_event(
            1000.0,
            pressure=False,
            local_pressure=False,
            alpha=0.5,
            stable_windows=1,
        ),
        timeline_event(
            2000.0,
            pressure=False,
            local_pressure=False,
            alpha=0.5,
            stable_windows=2,
        ),
        timeline_event(
            3000.0,
            pressure=False,
            local_pressure=False,
            alpha=0.55,
            stable_windows=0,
        ),
    ]
    pool_1_timeline = [
        timeline_event(
            1000.0,
            pressure=False,
            local_pressure=False,
            alpha=0.5,
            stable_windows=1,
        ),
        timeline_event(
            2000.0,
            pressure=True,
            local_pressure=True,
            alpha=0.25,
            stable_windows=0,
            reclaim_count=1,
        ),
        timeline_event(
            3000.0,
            pressure=False,
            local_pressure=False,
            alpha=0.25,
            stable_windows=1,
        ),
    ]

    def pool_profile(timeline: list[dict[str, Any]]) -> dict[str, Any]:
        alphas = [float(event["alpha"]) for event in timeline]
        return {
            "window_count": len(timeline),
            "initial_alpha": alphas[0],
            "final_alpha": alphas[-1],
            "minimum_alpha": min(alphas),
            "maximum_alpha": max(alphas),
            "local_stall_threshold_ms_values": [6.0],
            "timeline": timeline,
        }
    paths = {
        "peer_hbm_protection": _physical_path(
            owner="source_worker_d2d_p2p",
            protocol="nvlink_intra",
            interconnect="runtime_validated_cuda_p2p",
        ),
        "direct_dram_protection": _physical_path(
            owner="source_worker_d2h",
            protocol="tcp",
            interconnect="host_dram_store",
        ),
        "peer_hbm_recovery": _zero_path(),
        "dram_recovery": _zero_path(),
    }
    if failure_method == "sigkill_process_group":
        paths["peer_hbm_recovery"] = _physical_path(
            owner="survivor_worker_hbm_load",
            protocol="nvlink_intra",
            interconnect="runtime_validated_cuda_p2p",
        )
        paths["dram_recovery"] = _physical_path(
            owner="survivor_worker_store_load",
            protocol="tcp",
            interconnect="host_dram_store",
        )
    return {
        "admission_decisions": {
            "BYPASS": 1,
            "ADMIT_DRAM": 1,
            "INVALID": 0,
            "ADMIT_HBM": 1,
            "UNEXPECTED_TIER_DECISION": 0,
            "HBM_REJECT_THEN_ADMIT_DRAM": 1,
        },
        "independent_tier_outcomes": {
            "HBM_ONLY": 1,
            "DRAM_ONLY": 1,
            "HBM_AND_DRAM": 1,
            "BYPASS": 1,
        },
        "source_owned_paths": paths,
        "path_confirmations": {
            "hbm_independent_ack_count": 1,
            "dram_independent_ack_count": 1,
            "hbm_and_dram_independent_ack_count": 1,
        },
        "forbidden_semantics": {
            "target_owned_task_count": 0,
            "hbm_to_dram_transfer_count": 0,
            "background_destage_count": 0,
            "cross_tier_transfer_count": 0,
        },
        "aimd": {
            "window_count": 6,
            "target_local_allocation_or_reclaim_stall_window_count": 1,
            "malformed_window_count": 0,
            "alpha_decrease_count": 1,
            "alpha_increase_count": 1,
            "quota_contraction_count": 1,
            "new_hbm_admissions_paused_before_reclaim": True,
            "new_hbm_admissions_resumed_after_reclaim": True,
            "quota_unsatisfied_window_count": 0,
            "pools": {
                "hbm-pool-0": pool_profile(pool_0_timeline),
                "hbm-pool-1": pool_profile(pool_1_timeline),
            },
            "reclaim_samples": [
                {
                    "residency": "HBM_ONLY",
                    "controller_selected": True,
                    "dram_backing_required": False,
                    "physical_copy_deleted": True,
                    "freed": [True],
                }
            ],
        },
    }


def _strict_aimd_measurements() -> dict[str, Any]:
    profile_aimd = _condition_events("aegiskv", "none")["aimd"]

    def strict_event(
        event: Mapping[str, Any], *, reclaim_bytes: int = 0
    ) -> dict[str, Any]:
        reclaim_count = int(event["reclaimed_slot_count"])
        return {
            "observed_at_ms": event["observed_at_ms"],
            "pressure": event["pressure"],
            "pressure_reasons": event["pressure_reasons"],
            "alpha": event["alpha"],
            "quota_bytes": int(event["quota_bytes"]),
            "allocated_bytes": int(event["allocated_bytes"]),
            "stable_windows": event["stable_windows"],
            "cuda_free_bytes": int(event["cuda_free_bytes"]),
            "reclaim_count": reclaim_count,
            "reclaim_bytes": reclaim_bytes,
            "hbm_only_reclaim_count": reclaim_count,
        }

    pools: dict[str, Any] = {}
    for pool_id, raw_profile in profile_aimd["pools"].items():
        timeline = raw_profile["timeline"]
        trajectory = [
            strict_event(
                event,
                reclaim_bytes=(256 if event["reclaimed_slot_count"] else 0),
            )
            for event in timeline
        ]
        alphas = [float(event["alpha"]) for event in trajectory]
        pressure_count = sum(event["pressure"] is True for event in trajectory)
        local_count = sum(
            "peer_hbm_allocation_reclaim_stall" in event["pressure_reasons"]
            for event in trajectory
        )
        transitions = 0
        previous_alpha = AIMD_RUNTIME_CONFIG["initial_alpha"]
        for event in trajectory:
            transitions += int(float(event["alpha"]) != previous_alpha)
            previous_alpha = float(event["alpha"])
        pools[pool_id] = {
            "window_count": len(trajectory),
            "pressure_window_count": pressure_count,
            "stable_window_count": len(trajectory) - pressure_count,
            "memory_pressure_window_count": 0,
            "local_stall_pressure_window_count": local_count,
            "initial_alpha": AIMD_RUNTIME_CONFIG["initial_alpha"],
            "final_alpha": alphas[-1],
            "observed_min_alpha": min(alphas),
            "observed_max_alpha": max(alphas),
            "alpha_transition_count": transitions,
            "min_cuda_free_bytes": min(
                int(event["cuda_free_bytes"]) for event in trajectory
            ),
            "peak_quota_utilization": 0.0,
            "reclaim_count": sum(event["reclaim_count"] for event in trajectory),
            "reclaim_bytes": sum(event["reclaim_bytes"] for event in trajectory),
            "hbm_only_reclaim_count": sum(
                event["hbm_only_reclaim_count"] for event in trajectory
            ),
            "trajectory": trajectory,
        }
    summaries = list(pools.values())
    window_count = sum(summary["window_count"] for summary in summaries)
    pressure_count = sum(
        summary["pressure_window_count"] for summary in summaries
    )
    all_alphas = [
        float(event["alpha"])
        for summary in summaries
        for event in summary["trajectory"]
    ]
    return {
        "enabled": True,
        "config": dict(AIMD_RUNTIME_CONFIG),
        "pool_count": 2,
        "window_count": window_count,
        "pressure_window_count": pressure_count,
        "stable_window_count": window_count - pressure_count,
        "memory_pressure_window_count": 0,
        "local_stall_pressure_window_count": 1,
        "pressure_window_fraction": pressure_count / window_count,
        "final_alpha_mean": sum(
            summary["final_alpha"] for summary in summaries
        )
        / len(summaries),
        "observed_min_alpha": min(all_alphas),
        "observed_max_alpha": max(all_alphas),
        "alpha_transition_count": sum(
            summary["alpha_transition_count"] for summary in summaries
        ),
        "peak_quota_utilization": 0.0,
        "reclaim_count": sum(summary["reclaim_count"] for summary in summaries),
        "reclaim_bytes": sum(summary["reclaim_bytes"] for summary in summaries),
        "hbm_only_reclaim_count": sum(
            summary["hbm_only_reclaim_count"] for summary in summaries
        ),
        "pools": pools,
    }


def _fixture(
    tmp_path: Path,
    *,
    producer_per_gpu_mode: str = "valid",
    launcher_telemetry_mode: str = "valid",
) -> tuple[A100DP2InputPaths, dict[str, Path]]:
    execution_root = tmp_path / "isolated-execution-root"
    campaign_root = execution_root / "aegiskv/campaigns" / CAMPAIGN_ID
    repository_root = execution_root / "aegiskv"
    source_file = repository_root / "aegiskv/source.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("CURRENT_DESIGN = True\n", encoding="utf-8")
    entries = [
        {
            "relative_path": "aegiskv/source.py",
            "kind": "python_runtime_source",
            "size_bytes": source_file.stat().st_size,
            "sha256": sha256_file(source_file),
        }
    ]
    source = _write_json(
        campaign_root / "source-manifest.json",
        {
            "schema_version": 1,
            "evidence_class": "aegiskv_source_and_native_binary_manifest",
            "status": "PINNED",
            "inventory_policy": "exact_selected_file_set",
            "repository_root": str(repository_root.resolve()),
            "git_revision": "a" * 40,
            "entry_count": len(entries),
            "entries_sha256": canonical_sha256(entries),
            "entries": entries,
        },
    )
    source_verification = _write_json(
        campaign_root / "source-verification.json",
        {
            "schema_version": 1,
            "evidence_class": "aegiskv_source_manifest_verification",
            "status": "PASS",
            "failures": [],
            "entries_sha256": canonical_sha256(entries),
            "manifest": artifact_record(source),
        },
    )

    source_activation = _write_json(
        campaign_root / "source-activation.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_v19_source_activation_gate",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "activation_revision": "v44-affine-six-run-calibration",
            "source_manifest": artifact_record(source),
            "source_verification": artifact_record(source_verification),
            "jobs_submitted": [],
        },
    )

    runtime_files: dict[str, Path] = {}
    for name in (
        "vllm_package",
        "v2_model_runner",
        "torch_package",
        "torch_c_extension",
        "mooncake_engine",
    ):
        path = execution_root / "runtime" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture runtime {name}\n", encoding="utf-8")
        runtime_files[name] = path
    python_target = execution_root / "runtime/python-real"
    python_target.write_text("fixture runtime python\n", encoding="utf-8")
    python_invocation = execution_root / "venv/bin/python"
    python_invocation.parent.mkdir(parents=True, exist_ok=True)
    python_invocation.symlink_to("../../runtime/python-real")
    runtime_files["python_executable"] = python_invocation
    runtime_identity = {
        "execution_root": str(execution_root.resolve()),
        "python_invocation_path": str(python_invocation),
        "python_symlink_target": "../../runtime/python-real",
        "versions": {
            "vllm": "1.2.2+aegiskv.sm89",
            "torch": "2.10.0+cu128",
            "cuda": "12.8",
        },
        "artifacts": {
            name: artifact_record(path) for name, path in runtime_files.items()
        },
    }

    clean_submission_gate = _write_json(
        campaign_root / "clean-preflight-submission-gate-v45.json",
        {"evidence_class": "fixture-clean-submission-gate", "status": "PASS"},
    )
    clean_launcher = campaign_root / "a100_clean_preflight.sh"
    clean_launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    clean_submission = _write_json(
        campaign_root / "clean-preflight-submission-v45.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_clean_preflight_submission_v45",
            "status": "QUEUED_NO_RUNTIME",
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "clean_job_id": 4242,
            "slurm_comment": (
                "aegis-a100-v45-clean-"
                f"{sha256_file(clean_submission_gate)[:12]}"
            ),
            "clean_preflight_submission_authorized": True,
            "model_execution_probe_submission_authorized": False,
            "runtime_or_physical_result": False,
            "lineage": {
                "submission_gate": artifact_record(clean_submission_gate),
                "clean_launcher": artifact_record(clean_launcher),
            },
        },
    )
    clean_detail = _write_json(
        campaign_root / "clean-preflight-detail.json", {"status": "PASS"}
    )
    clean_audit = _write_json(
        campaign_root / "clean-gpu-preflight-audit.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_clean_gpu_preflight_audit",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "slurm_job_id": "4242",
            "gates": {
                name: True
                for name in (
                    "bidirectional_cuda_p2p_and_integrity",
                    "mooncake_cuda_engine_import",
                    "pinned_vllm_v2_runtime",
                    "pre_run_source_runtime_identity",
                    "qwen3_8b_snapshot_available",
                    "slurm_node_exclusive",
                    "two_a100_80gb_sm80_devices",
                    "zero_foreign_gpu_processes",
                )
            },
            "artifacts": {"detail": artifact_record(clean_detail)},
        },
    )
    clean_scheduler = campaign_root / "clean-sacct.txt"
    clean_scheduler.write_text(
        "4242|COMPLETED|0:0|18|gpu26|gres/gpu:a100=2\n",
        encoding="utf-8",
    )
    runtime_snapshot = _write_json(
        campaign_root / "runtime-identity-snapshot-v46.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_runtime_identity_snapshot_v46",
            "status": "PASS",
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "execution_root": str(execution_root.resolve()),
            "runtime_identity": runtime_identity,
            "gpu_or_model_execution_performed": False,
            "runtime_or_physical_result": False,
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    model_launcher = campaign_root / "model_execution_probe.sh"
    launcher_text = """#!/bin/sh
nvidia-smi dmon \\
  --id=0,1 \\
  --select=pucmet \\
  --delay=2 \\
  --options=DT \\
  --format=csv \\
  --filename=\"${run_dir}/gpu-telemetry.csv\"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=INFO
"""
    if launcher_telemetry_mode == "pucvmet":
        launcher_text = launcher_text.replace("--select=pucmet", "--select=pucvmet")
    elif launcher_telemetry_mode == "one_device":
        launcher_text = launcher_text.replace("--id=0,1", "--id=0")
    elif launcher_telemetry_mode == "space_format":
        launcher_text = launcher_text.replace("--format=csv", "--format=space")
    elif launcher_telemetry_mode != "valid":
        raise AssertionError(
            f"unsupported launcher_telemetry_mode: {launcher_telemetry_mode}"
        )
    model_launcher.write_text(launcher_text, encoding="utf-8")
    model_coordinator = campaign_root / "a100_model_execution_probe.py"
    model_coordinator.write_text("# fixture coordinator\n", encoding="utf-8")
    model_gate_builder = campaign_root / "build_model_submission_gate_v46.py"
    model_gate_builder.write_text("# fixture gate builder\n", encoding="utf-8")
    runtime_binding = _write_json(
        campaign_root / "runtime-binding-v46.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_runtime_binding_v46",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "activation_revision": "v44-affine-six-run-calibration",
            "execution_root": str(execution_root.resolve()),
            "clean_job_id": 4242,
            "source_activation": artifact_record(source_activation),
            "clean_preflight_submission": artifact_record(clean_submission),
            "clean_preflight": artifact_record(clean_audit),
            "clean_scheduler_final": artifact_record(clean_scheduler),
            "runtime_identity_snapshot": artifact_record(runtime_snapshot),
            "runtime_identity": runtime_identity,
            "model_probe_overlay": {
                "model_launcher": artifact_record(model_launcher),
                "model_coordinator": artifact_record(model_coordinator),
            },
            "model_execution_probe_submission_authorized": False,
            "physical_calibration_submission_authorized": False,
            "mechanism_submission_authorized": False,
            "runtime_or_physical_result": False,
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    model_submission_gate = _write_json(
        campaign_root / "model-submission-gate-v46.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_model_submission_gate_v46",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "activation_revision": "v44-affine-six-run-calibration",
            "execution_root": str(execution_root.resolve()),
            "clean_job_id": 4242,
            "source_activation": artifact_record(source_activation),
            "clean_preflight_submission": artifact_record(clean_submission),
            "clean_preflight": artifact_record(clean_audit),
            "scheduler_final": {
                "status": "PASS",
                "failures": [],
                "job_id": "4242",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "elapsed_seconds": 18,
                "node_list": "gpu26",
                "alloc_tres": "gres/gpu:a100=2",
                "raw": "4242|COMPLETED|0:0|18|gpu26|gres/gpu:a100=2",
            },
            "scheduler_final_artifact": artifact_record(clean_scheduler),
            "runtime_identity_snapshot": artifact_record(runtime_snapshot),
            "runtime_binding": artifact_record(runtime_binding),
            "gate_builder": artifact_record(model_gate_builder),
            "model_launcher": artifact_record(model_launcher),
            "model_coordinator": artifact_record(model_coordinator),
            "model_execution_probe_submission_authorized": True,
            "maximum_inflight_model_probes": 1,
            "maximum_job_runtime_seconds": 1200,
            "physical_calibration_submission_authorized": False,
            "mechanism_submission_authorized": False,
            "formal_scale_submission_authorized": False,
            "runtime_or_physical_result": False,
            "jobs_submitted": [],
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    workers = [
        {
            "evidence_class": "a100_qwen3_8b_vllm_v2_worker_execution",
            "status": "PASS",
            "logical_rank": rank,
            "cuda_visible_devices": str(rank),
            "gpu_capability": [8, 0],
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "model_id": MODEL,
            "model_path": "/models/qwen3-8b",
            "model_revision": REVISION,
            "model_config_sha256": "c" * 64,
            "runtime_identity": runtime_identity,
            "model_runner_v2_requested": True,
            "model_runner_v2_engine_config_selected": True,
            "model_runner_v2_log_marker_observed": True,
            "model_runner_v2_observed": True,
        }
        for rank in range(2)
    ]
    model_probe = _write_json(
        tmp_path / "model-probe.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_qwen3_8b_vllm_v2_model_execution_probe",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "model_id": MODEL,
            "model_revision": REVISION,
            "model_config_sha256": "c" * 64,
            "runtime_identity": runtime_identity,
            "tp_size": 1,
            "pp_size": 1,
            "system_dp_size": 2,
            "model_runner_v2_observed": True,
            "input_lineage": {
                "runtime_binding": artifact_record(runtime_binding),
            },
            "worker_results": workers,
            "formal_ready": False,
            "paper_ready": False,
        },
    )

    model_gpu_csv = _write_gpu_csv(
        tmp_path / "model/gpu-telemetry.csv",
        gpu1_sm=0.0 if producer_per_gpu_mode == "inactive_raw" else 31.0,
    )
    model_telemetry = _telemetry_summary(model_gpu_csv)
    producer_per_gpu = _producer_per_gpu_telemetry()
    if producer_per_gpu_mode == "failure":
        producer_per_gpu["status"] = "FAIL"
        producer_per_gpu["failures"] = ["per_gpu_runtime_activity"]
        producer_per_gpu["devices"]["1"]["peak_gpu_utilization_pct"] = 0.0
    elif producer_per_gpu_mode == "drift":
        producer_per_gpu["devices"]["1"]["peak_gpu_utilization_pct"] = 99.0
    elif producer_per_gpu_mode == "inactive_raw":
        # Forge aggregate success while one DP worker has no SM activity.
        producer_per_gpu["devices"]["1"]["peak_gpu_utilization_pct"] = 0.0
    elif producer_per_gpu_mode not in {"valid", "missing"}:
        raise AssertionError(
            f"unsupported producer_per_gpu_mode: {producer_per_gpu_mode}"
        )
    model_details: dict[str, Any] = {
        "model_result": artifact_record(model_probe),
        "gpu_telemetry": model_telemetry,
    }
    if producer_per_gpu_mode != "missing":
        model_details["per_gpu_telemetry"] = producer_per_gpu
    model_audit = _write_json(
        tmp_path / "model-execution-attempt-audit-v47.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_model_execution_attempt_audit_v47",
            "auditor_revision": "dynamic-v46-runtime-binding",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "runtime_binding": artifact_record(runtime_binding),
            "model_submission_gate": artifact_record(model_submission_gate),
            "details": model_details,
            "clean_model_probe_passed": True,
            "registered_copy_submission_authorized": True,
            "physical_calibration_submission_authorized": False,
            "mechanism_submission_authorized": False,
            "paired_campaign_submission_authorized": False,
            "formal_ready": False,
            "paper_ready": False,
        },
    )

    copy_config = _write_json(
        tmp_path / "copy/config.json", {"source_manifest": artifact_record(source)}
    )
    copy_result = _write_json(tmp_path / "copy/result.json", {"status": "PASS"})
    copy_runtime = _write_json(
        tmp_path / "copy/runtime.json", {"status": "PASS"}
    )
    copy_gpu_csv = _write_gpu_csv(tmp_path / "copy/gpu-telemetry.csv")
    registered_copy = _write_json(
        tmp_path / "registered-copy.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_registered_gpu_copy_gate",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "payload_bytes": 4096,
            "upstream": {
                "runtime_binding": artifact_record(runtime_binding),
                "model_execution_probe": artifact_record(model_probe),
                "model_execution_audit": artifact_record(model_audit),
            },
            "artifacts": {
                "config": artifact_record(copy_config),
                "result": artifact_record(copy_result),
                "runtime_engine": artifact_record(copy_runtime),
                "gpu_telemetry": artifact_record(copy_gpu_csv),
            },
            "gpu_telemetry": _telemetry_summary(copy_gpu_csv),
        },
    )

    configuration = {
        "layout": CONFIG_LAYOUT,
        "hardware_profile": HARDWARE_PROFILE,
        "capacity": {
            "peer_hbm_bytes_per_target": AIMD_RUNTIME_CONFIG[
                "total_hbm_bytes"
            ],
            "dram_bytes": 17179869184,
        },
        "aimd": {
            name: value
            for name, value in AIMD_RUNTIME_CONFIG.items()
            if name not in {"total_hbm_bytes", "local_stall_ms"}
        },
        "source_owned_paths": {
            "peer_hbm_protection": "source_worker_d2d_p2p",
            "direct_dram_protection": "source_worker_d2h",
            "peer_hbm_recovery": "survivor_worker_hbm_load",
            "dram_recovery": "survivor_worker_store_load",
            "target_owned_or_hbm_to_dram_path_allowed": False,
        },
        "four_condition_gate": [
            {"treatment": treatment, "failure_method": failure}
            for treatment in ("no-protection", "aegiskv")
            for failure in ("none", "sigkill_process_group")
        ],
    }
    config = _write_json(
        tmp_path / "config-manifest.json",
        {
            "schema_version": 1,
            "evidence_class": "aegiskv_physical_campaign_config_manifest",
            "status": "PINNED",
            "campaign_id": CAMPAIGN_ID,
            "design_semantics": DESIGN_SEMANTICS,
            "hardware_label": HARDWARE_LABEL,
            "model": MODEL,
            "model_revision": REVISION,
            "model_config_sha256": "c" * 64,
            "model_runner_v2_observed": True,
            "layout": CONFIG_LAYOUT,
            "hardware_profile": HARDWARE_PROFILE,
            "source_manifest": artifact_record(source),
            "source_verification": artifact_record(source_verification),
            "model_execution_probe": artifact_record(model_probe),
            "registered_gpu_copy_gate": artifact_record(registered_copy),
            "configuration": configuration,
            "configuration_sha256": canonical_sha256(configuration),
            "jobs_submitted": [],
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    config_verification = _write_json(
        tmp_path / "config-verification.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_config_manifest_verification",
            "status": "PASS",
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "manifest": artifact_record(config),
            "failures": [],
            "jobs_submitted": [],
            "formal_ready": False,
            "paper_ready": False,
        },
    )

    recompute_profile = _write_json(
        tmp_path / "recompute/recompute-profile.json",
        {
            "evidence_class": "recompute_latency_physical_measurement",
            "status": "passed",
            "model": MODEL,
            "model_revision": REVISION,
            "fit": {
                "recompute_ms_per_token": 0.125,
                "fixed_overhead_ms": 4.0,
                "r_squared": 0.99,
            },
        },
    )
    _write_json(
        recompute_profile.parent / "source-manifest.json",
        json.loads(source.read_text(encoding="utf-8")),
    )
    recompute_audit = _write_json(
        tmp_path / "recompute-audit.json",
        {
            "evidence_class": "a100_dp2_recompute_calibration_gate",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "layout": PHYSICAL_LAYOUT,
            "calibration_ready": True,
            "profile": artifact_record(recompute_profile),
            "fit": {
                "recompute_ms_per_token": 0.125,
                "fixed_overhead_ms": 4.0,
                "r_squared": 0.99,
            },
        },
    )

    peer_signature = {
        "protocol": "nvlink_intra",
        "physical_interconnect": "runtime_validated_cuda_p2p",
        "engine_sha256": "e" * 64,
    }
    transfer_parameters = {
        "peer_hbm_protection": {
            "per_transfer_setup_ms": 1.0,
            "first_transfer_warmup_ms": 3.0,
            "steady_state_bytes_per_ms": 50.0,
        },
        "direct_dram_protection": {
            "per_transfer_setup_ms": 2.0,
            "first_transfer_warmup_ms": 4.0,
            "steady_state_bytes_per_ms": 25.0,
        },
        "peer_hbm_recovery": {
            "per_transfer_setup_ms": 1.5,
            "first_transfer_warmup_ms": 2.5,
            "steady_state_bytes_per_ms": 40.0,
        },
        "dram_recovery": {
            "per_transfer_setup_ms": 3.0,
            "first_transfer_warmup_ms": 5.0,
            "steady_state_bytes_per_ms": 20.0,
        },
    }
    paths_by_treatment = {
        "aegiskv": ("peer_hbm_protection", "peer_hbm_recovery"),
        "aegiskv-no-hbm": ("direct_dram_protection", "dram_recovery"),
        "aegiskv-hbm-reject": ("direct_dram_protection", "dram_recovery"),
    }
    local_resource_by_treatment = {
        "aegiskv": ("allocation", 4.0),
        "aegiskv-no-hbm": ("reclaim", 6.0),
        "aegiskv-hbm-reject": ("allocation", 8.0),
    }
    transfer_lineage = {
        "source_manifest": artifact_record(source),
        "config_manifest": artifact_record(config),
        "config_verification": artifact_record(config_verification),
        "runtime_binding": artifact_record(runtime_binding),
        "model_retry_gate": artifact_record(model_submission_gate),
        "recompute_profile": artifact_record(recompute_profile),
    }
    transfer_audits: list[Path] = []
    generic_audits: list[Path] = []
    gate_logs: list[Path] = []
    streams_by_treatment: dict[str, list[str]] = {}
    transfer_index = 0
    for treatment in sorted(paths_by_treatment):
        treatment_streams: list[str] = []
        for repeat_index in range(TRANSFER_REPEATS_PER_TREATMENT):
            job_id = 140000 + transfer_index
            run_id = str(job_id)
            stream_id = f"slurm-job:{run_id}"
            run_dir = tmp_path / f"transfer-{treatment}-{job_id}-repeat-{repeat_index}"
            rows: list[dict[str, Any]] = []
            for path_name in paths_by_treatment[treatment]:
                rows.extend(
                    _affine_transfer_samples(
                        TRANSFER_PATH_EVENTS[path_name],
                        calibration_run_id=run_id,
                        calibration_stream_id=stream_id,
                        **transfer_parameters[path_name],
                    )
                )
            operation, local_duration_ms = local_resource_by_treatment[treatment]
            rows.append(
                {
                    "event": "hbm_local_resource_operation",
                    "target_local": True,
                    "success": True,
                    "operation": operation,
                    "duration_ms": local_duration_ms,
                    "key_count": 1,
                    "bytes": 100,
                }
            )
            _write_json(
                run_dir / "source-manifest.json",
                json.loads(source.read_text(encoding="utf-8")),
            )
            gate_log = _write_jsonl(run_dir / "gate-dp0.jsonl", rows)
            measurement = {
                "hardware_profile": HARDWARE_PROFILE,
                "calibration_repeat_index": repeat_index,
                "calibration_run_id": run_id,
                "calibration_stream_id": stream_id,
                "condition_local": True,
                "input_lineage": transfer_lineage,
                "transport_signature": {
                    "peer_hbm": peer_signature,
                    "dram": {"protocol": "tcp"},
                },
                "gate_log_paths": [str(gate_log.resolve())],
            }
            generic = _write_json(
                run_dir / "generic-audit.json",
                {
                    "evidence_class": "physical_transfer_calibration_audit",
                    "status": "PASS",
                    "failures": [],
                    "run_dir": str(run_dir.resolve()),
                    "treatment": treatment,
                    "calibration_repeat_index": repeat_index,
                    "calibration_run_id": run_id,
                    "calibration_stream_id": stream_id,
                    "condition_local": True,
                    "input_lineage": transfer_lineage,
                    "measurement": measurement,
                },
            )
            wrapper = _write_json(
                tmp_path / f"transfer-{transfer_index}-audit.json",
                {
                    "evidence_class": "a100_dp2_transfer_calibration_gate",
                    "status": "PASS",
                    "failures": [],
                    "campaign_id": CAMPAIGN_ID,
                    "hardware_label": HARDWARE_LABEL,
                    "treatment": treatment,
                    "calibration_repeat_index": repeat_index,
                    "calibration_run_id": run_id,
                    "calibration_stream_id": stream_id,
                    "condition_local": True,
                    "input_lineage": transfer_lineage,
                    "calibration_ready": True,
                    "generic_audit": artifact_record(generic),
                    "measurement": measurement,
                },
            )
            transfer_audits.append(wrapper)
            generic_audits.append(generic)
            gate_logs.append(gate_log)
            treatment_streams.append(stream_id)
            transfer_index += 1
        streams_by_treatment[treatment] = treatment_streams
    assert transfer_index == TRANSFER_AUDIT_COUNT

    transfer_metrics: dict[str, float] = {}
    transfer_measurements: dict[str, Any] = {}
    for path_name, event in TRANSFER_PATH_EVENTS.items():
        parameters = transfer_parameters[path_name]
        calibration_treatment = TRANSFER_PATH_TREATMENTS[path_name]
        stream_ids = streams_by_treatment[calibration_treatment]
        transfer_measurements[path_name] = {
            "event": event,
            "calibration_treatment": calibration_treatment,
            "calibration_repeat_count": TRANSFER_REPEATS_PER_TREATMENT,
            "calibration_stream_ids": sorted(stream_ids),
            "cost_model": _declared_transfer_cost_model(
                event, stream_ids=stream_ids, **parameters
            ),
        }
        for component, value in parameters.items():
            transfer_metrics[f"{path_name}_{component}"] = value

    physical = _write_json(
        tmp_path / "physical-profile-schema-v2-v54.json",
        {
            "schema_version": 2,
            "evidence_class": "physical_measurement",
            "status": "passed",
            "profile_semantics": PHYSICAL_PROFILE_SEMANTICS,
            "design_semantics": DESIGN_SEMANTICS,
            "transfer_cost_model_schema": TRANSFER_COST_MODEL_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "hardware_profile": HARDWARE_PROFILE,
            "layout": PHYSICAL_LAYOUT,
            "model": MODEL,
            "model_revision": REVISION,
            "compatibility_identity": {
                "serving_dtype": "half",
                "kv_cache_dtype": "fp16",
                "layout": PHYSICAL_LAYOUT,
            },
            "metrics": {
                "recompute_ms_per_token": 0.125,
                "recompute_fixed_overhead_ms": 4.0,
                **transfer_metrics,
            },
            "runtime_consumption_contract": RUNTIME_CONSUMPTION_CONTRACT,
            "transfer_measurements": transfer_measurements,
            "source_manifest": artifact_record(source),
            "config_manifest": artifact_record(config),
            "config_verification": artifact_record(config_verification),
            "source_artifacts": {
                "recompute_audit": artifact_record(recompute_audit),
                "recompute_profile": artifact_record(recompute_profile),
                "registered_gpu_copy_audit": artifact_record(registered_copy),
                "transfer_audits": [artifact_record(path) for path in transfer_audits],
                "generic_transfer_audits": [
                    artifact_record(path) for path in generic_audits
                ],
                "gate_logs": [artifact_record(path) for path in gate_logs],
                "source_verification": artifact_record(source_verification),
            },
            "failures": [],
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    physical_submission_gate = _write_json(
        tmp_path / "physical-calibration-submission-gate-v50.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_physical_calibration_submission_gate_v50",
            "gate_revision": "dynamic-v46-v49-lineage",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "calibration_node": "gpu25",
            "registered_copy_inner_audit": artifact_record(registered_copy),
            "source_activation": artifact_record(source_activation),
            "runtime_binding": artifact_record(runtime_binding),
            "model_execution_probe": artifact_record(model_probe),
            "source_manifest": artifact_record(source),
            "config_manifest": artifact_record(config),
            "config_verification": artifact_record(config_verification),
            "recompute_calibration_submission_authorized": True,
            "transfer_calibration_submission_authorized": False,
            "mechanism_submission_authorized": False,
            "paired_campaign_submission_authorized": False,
            "formal_scale_submission_authorized": False,
            "jobs_submitted": [],
            "runtime_or_physical_result": False,
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    recompute_attempt = _write_json(
        tmp_path / "recompute-attempt-audit-v51.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_recompute_attempt_audit_v51",
            "auditor_revision": "dynamic-v50-submission-and-gate",
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "calibration_node": "gpu25",
            "physical_calibration_submission_gate": artifact_record(
                physical_submission_gate
            ),
            "recompute_inner_audit": artifact_record(recompute_audit),
            "transfer_calibration_submission_authorized": True,
            "mechanism_submission_authorized": False,
            "paired_campaign_submission_authorized": False,
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    recompute_compatibility = _write_json(
        tmp_path / "recompute-attempt-compatibility-v52.json",
        {
            **json.loads(recompute_attempt.read_text(encoding="utf-8")),
            "evidence_class": "a100_dp2_recompute_attempt_audit_v35",
            "compatibility_view_revision": "v52-for-frozen-transfer-launcher",
            "compatibility_view_of": artifact_record(recompute_attempt),
        },
    )
    transfer_attempts: dict[str, list[Path]] = {
        treatment: [] for treatment in sorted(paths_by_treatment)
    }
    for index, inner_path in enumerate(transfer_audits):
        inner = json.loads(inner_path.read_text(encoding="utf-8"))
        treatment = inner["treatment"]
        transfer_attempts[treatment].append(
            _write_json(
                tmp_path / f"transfer-attempt-{index}-audit-v53.json",
                {
                    "schema_version": 1,
                    "evidence_class": "a100_dp2_transfer_attempt_audit_v53",
                    "auditor_revision": "dynamic-v52-submission-v51-lineage",
                    "status": "PASS",
                    "failures": [],
                    "campaign_id": CAMPAIGN_ID,
                    "hardware_label": HARDWARE_LABEL,
                    "design_semantics": DESIGN_SEMANTICS,
                    "treatment": treatment,
                    "calibration_repeat_index": inner[
                        "calibration_repeat_index"
                    ],
                    "calibration_run_id": inner["calibration_run_id"],
                    "calibration_stream_id": inner["calibration_stream_id"],
                    "condition_local": True,
                    "calibration_node": "gpu25",
                    "recompute_attempt_audit": artifact_record(
                        recompute_compatibility
                    ),
                    "recompute_attempt_audit_v51": artifact_record(
                        recompute_attempt
                    ),
                    "transfer_inner_audit": artifact_record(inner_path),
                    "transfer_calibration_complete": True,
                    "physical_profile_build_authorized": False,
                    "mechanism_submission_authorized": False,
                    "paired_campaign_submission_authorized": False,
                    "formal_ready": False,
                    "paper_ready": False,
                },
            )
        )
    transfer_series = _write_json(
        tmp_path / "transfer-calibration-series-ready-v53.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_transfer_calibration_series_ready_v53",
            "status": "PASS",
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "treatments": sorted(paths_by_treatment),
            "repeats_per_treatment": TRANSFER_REPEATS_PER_TREATMENT,
            "recompute_attempt_audit_v51": artifact_record(recompute_attempt),
            "recompute_compatibility_view_v52": artifact_record(
                recompute_compatibility
            ),
            "transfer_attempt_audits": {
                treatment: [artifact_record(path) for path in paths]
                for treatment, paths in transfer_attempts.items()
            },
            "physical_profile_build_authorized": True,
            "mechanism_submission_authorized": False,
            "runtime_or_physical_result": True,
            "formal_ready": False,
            "paper_ready": False,
        },
    )
    physical_audit = _write_json(
        tmp_path / "physical-profile-build-audit-v54.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_physical_profile_build_audit_v54",
            "builder_revision": "v53-series-and-v51-lineage",
            "status": "PASS",
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "calibration_node": "gpu25",
            "transfer_series_ready": artifact_record(transfer_series),
            "transfer_series_ready_v53": artifact_record(transfer_series),
            "recompute_attempt_audit": artifact_record(
                recompute_compatibility
            ),
            "transfer_attempt_audits": [
                artifact_record(path)
                for paths in transfer_attempts.values()
                for path in paths
            ],
            "source_manifest": artifact_record(source),
            "config_manifest": artifact_record(config),
            "config_verification": artifact_record(config_verification),
            "physical_profile": artifact_record(physical),
            "physical_profile_build_authorized": True,
            "runtime_or_physical_result": True,
            "mechanism_submission_authorized": False,
            "paired_campaign_submission_authorized": False,
            "formal_ready": False,
            "paper_ready": False,
        },
    )

    builder = tmp_path / "build_aimd_stall_calibration_v55.py"
    builder.write_text("# immutable fixture builder\n", encoding="utf-8")
    aimd = _write_json(
        tmp_path / "aimd-local-stall-calibration-v55.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_aimd_local_stall_calibration_v55",
            "calibration_revision": "v54-physical-profile-lineage",
            "status": "PASS",
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "calibration_node": "gpu25",
            "source_manifest": artifact_record(source),
            "config_manifest": artifact_record(config),
            "config_verification": artifact_record(config_verification),
            "builder_and_verifier": artifact_record(builder),
            "physical_profile": artifact_record(physical),
            "physical_profile_audit": artifact_record(physical_audit),
            "gate_logs": [artifact_record(path) for path in gate_logs],
            "threshold": {
                "environment_variable": "AEGIS_AIMD_LOCAL_STALL_MS",
                "local_stall_ms": 6.0,
                "estimator": (
                    "median_successful_target_local_hbm_operation_duration_ms"
                ),
                "sample_count": 6,
                "source_log_count": 6,
                "operation_counts": {"allocation": 4, "reclaim": 2},
                "minimum_duration_ms": 4.0,
                "median_duration_ms": 6.0,
                "maximum_duration_ms": 8.0,
            },
            "observation_contract": {
                "aimd_interval_s": 5.0,
                "stable_window_threshold": 3,
                "post_workload_observation_s": 25.0,
                "tail_formula": "interval_s * (stable_window_threshold + 2)",
            },
            "performance_direction_used_for_selection": False,
            "mechanism_submission_authorized": False,
            "paired_campaign_submission_authorized": False,
            "formal_scale_submission_authorized": False,
            "paper_ready": False,
            "failures": [],
        },
    )

    conditions = [
        ("healthy-no-protection", "no-protection", "none"),
        ("healthy-aegiskv", "aegiskv", "none"),
        ("failed-aegiskv", "aegiskv", "sigkill_process_group"),
        ("failed-no-protection", "no-protection", "sigkill_process_group"),
    ]
    condition_profiles: list[Path] = []
    for index, (condition_id, treatment, failure_method) in enumerate(conditions):
        strict_payload: dict[str, Any] = {
            "evidence_class": "strict_audit_report",
            "status": "PASS",
            "condition_id": condition_id,
            "failures": [],
        }
        if treatment == "aegiskv":
            strict_payload["measurements"] = {
                "aegiskv_data_plane": {
                    "aimd": _strict_aimd_measurements(),
                }
            }
        strict = _write_json(
            tmp_path / f"conditions/{condition_id}/strict-audit.json",
            strict_payload,
        )
        summary = _write_json(
            tmp_path / f"conditions/{condition_id}/summary.json",
            {
                "campaign_id": CAMPAIGN_ID,
                "campaign_condition_id": condition_id,
                "started_at_ms": 1000.0 + index * 100000.0,
                "finished_at_ms": 3000.0 + index * 100000.0,
                "elapsed_ms": 2000.0,
            },
        )
        condition_gpu_csv = _write_gpu_csv(
            tmp_path / f"conditions/{condition_id}/gpu-telemetry.csv"
        )
        start = 1000.0 + index * 100000.0
        finish = 3000.0 + index * 100000.0
        profile = _write_json(
            tmp_path / f"conditions/{condition_id}/mechanism-profile-v41.json",
            {
                "schema_version": 1,
                "evidence_class": (
                    "aegiskv_current_design_condition_mechanism_profile"
                ),
                "status": "PASS",
                "failures": [],
                "design_semantics": DESIGN_SEMANTICS,
                "runtime_admission_decision_wire_values": WIRE_VALUES,
                "campaign_id": CAMPAIGN_ID,
                "condition_id": condition_id,
                "treatment": treatment,
                "failure_method": failure_method,
                "hardware_label": HARDWARE_LABEL,
                "layout": CONFIG_LAYOUT,
                "input_lineage": {
                    "source_manifest": artifact_record(source),
                    "config_manifest": artifact_record(config),
                    "config_verification": artifact_record(config_verification),
                    "physical_profile": artifact_record(physical),
                    "aimd_stall_calibration": artifact_record(aimd),
                    "strict_run_audit": artifact_record(strict),
                },
                "workload": {
                    "workflow_count": 4,
                    "request_count": 16,
                    "workflow_concurrency": 4,
                    "started_at_ms": start,
                    "finished_at_ms": finish,
                },
                "mechanism_observation_window": {
                    "started_at_ms": finish,
                    "finished_at_ms": finish + 25000.0,
                    "required_duration_ms": 25000.0,
                    "complete": True,
                },
                "aimd_local_stall_calibration": {
                    "local_stall_ms": 6.0,
                    "deployed": True,
                    "runtime_windows_match": True,
                },
                "gpu_telemetry": {
                    **_telemetry_summary(condition_gpu_csv, workload=True),
                    "foreign_gpu_process_count": 0,
                    "post_run_gpu_process_cleanup": True,
                },
                "workload_window_events": _condition_events(
                    treatment, failure_method
                ),
                "artifacts": {
                    "strict_audit": artifact_record(strict),
                    "summary": artifact_record(summary),
                    "gpu_telemetry": artifact_record(condition_gpu_csv),
                },
            },
        )
        condition_profiles.append(profile)

    mechanism_gate = _write_json(
        tmp_path / "four-condition-mechanism-gate-v41.json",
        {
            "schema_version": 1,
            "evidence_class": (
                "a100_dp2_current_design_four_condition_mechanism_gate"
            ),
            "status": "PASS",
            "failures": [],
            "campaign_id": CAMPAIGN_ID,
            "hardware_label": HARDWARE_LABEL,
            "design_semantics": DESIGN_SEMANTICS,
            "source_manifest": artifact_record(source),
            "config_manifest": artifact_record(config),
            "config_verification": artifact_record(config_verification),
            "physical_profile": artifact_record(physical),
            "aimd_stall_calibration": artifact_record(aimd),
            "condition_profiles": [
                artifact_record(path) for path in condition_profiles
            ],
            "checks": [
                {"name": name, "passed": True, "detail": {}}
                for name in sorted(A100_REQUIRED_MECHANISM_CHECKS)
            ],
            "pilot_performance_claim_authorized": False,
            "formal_scale_submission_authorized": False,
            "paper_ready": False,
        },
    )
    inputs = A100DP2InputPaths(
        source_manifest=source,
        runtime_binding=runtime_binding,
        model_submission_gate=model_submission_gate,
        config_manifest=config,
        config_verification=config_verification,
        physical_profile=physical,
        physical_profile_audit=physical_audit,
        aimd_stall_calibration=aimd,
        mechanism_gate=mechanism_gate,
    )
    return inputs, {
        "source_file": source_file,
        "model_probe": model_probe,
        "runtime_binding": runtime_binding,
        "model_submission_gate": model_submission_gate,
        "model_launcher": model_launcher,
        "model_gpu_csv": model_gpu_csv,
        "recompute_audit": recompute_audit,
        "registered_copy": registered_copy,
        "copy_gpu_csv": copy_gpu_csv,
        "transfer_attempt": next(iter(transfer_attempts.values()))[0],
        "transfer_attempts": transfer_attempts,
        "transfer_audits": transfer_audits,
        "generic_audits": generic_audits,
        "gate_logs": gate_logs,
        "transfer_series": transfer_series,
        "hbm_gate_log": gate_logs[0],
        "physical_audit": physical_audit,
        "condition_profile": condition_profiles[1],
        "mechanism_gate": mechanism_gate,
    }


def _rewrite_condition_and_gate(
    profile_path: Path, gate_path: Path, mutate: Any
) -> None:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    mutate(profile)
    _write_json(profile_path, profile)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for reference in gate["condition_profiles"]:
        if Path(reference["path"]).resolve() == profile_path.resolve():
            reference.update(artifact_record(profile_path))
    _write_json(gate_path, gate)


def _rewrite_strict_and_rebind(
    profile_path: Path, gate_path: Path, mutate: Any
) -> None:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    strict_path = Path(profile["input_lineage"]["strict_run_audit"]["path"])
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    mutate(strict)
    _write_json(strict_path, strict)
    strict_record = artifact_record(strict_path)
    profile["input_lineage"]["strict_run_audit"] = strict_record
    profile["artifacts"]["strict_audit"] = strict_record
    _write_json(profile_path, profile)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for reference in gate["condition_profiles"]:
        if Path(reference["path"]).resolve() == profile_path.resolve():
            reference.update(artifact_record(profile_path))
    _write_json(gate_path, gate)


def _a100_suite_matrix(model_revision: str = REVISION) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_scales": [
            {
                "model": MODEL,
                "scale_label": "Qwen3-8B",
                "model_revision": model_revision,
            }
        ],
        "system_dp_sizes": [2],
        "concurrency_levels": [1, 4, 16],
        "failure_rates": [0.0, 0.01],
        "capacity_points": [
            {
                "label": "full",
                "peer_hbm_fraction": 1.0,
                "store_dram_fraction": 1.0,
            },
            {
                "label": "half",
                "peer_hbm_fraction": 0.5,
                "store_dram_fraction": 0.5,
            },
        ],
        "seeds": [20260821, 20260822],
    }


def _materialized_a100_consumer(
    tmp_path: Path,
) -> tuple[A100DP2InputPaths, Path, Path]:
    inputs, _paths = _fixture(tmp_path / "producer")
    output_dir = preflight_and_write(
        inputs=inputs, output_root=tmp_path / "consumer"
    )
    return (
        inputs,
        output_dir / "preflight-manifest.json",
        output_dir / "a100-dp2-profile.json",
    )


def _rewrite_all_aegis_conditions(
    gate_path: Path, mutate: Any
) -> None:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    for reference in gate["condition_profiles"]:
        profile_path = Path(reference["path"])
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile["treatment"] != "aegiskv":
            continue
        mutate(profile)
        _write_json(profile_path, profile)
        reference.update(artifact_record(profile_path))
    _write_json(gate_path, gate)


def test_complete_a100_bundle_passes_and_writes_only_preflight(
    tmp_path: Path,
) -> None:
    inputs, _paths = _fixture(tmp_path)
    result = validate_a100_dp2_bundle(inputs)
    assert isinstance(result.profile, A100DP2PhysicalProfile)
    assert result.profile.peer_hbm_recovery_per_transfer_setup_ms == 1.5
    assert result.profile.peer_hbm_recovery_first_transfer_warmup_ms == 2.5
    assert result.profile.peer_hbm_recovery_steady_state_bytes_per_ms == 40.0
    assert result.aimd_local_stall_ms == 6.0
    assert result.aimd_runtime_config == AIMD_RUNTIME_CONFIG
    assert len(result.condition_profile_artifacts) == 4
    assert len(result.strict_audit_artifacts) == 4

    output_dir = preflight_and_write(
        inputs=inputs, output_root=tmp_path / "independent-a100-output"
    )
    assert output_dir.name.startswith("a100_dp2_preflight_")
    manifest = json.loads((output_dir / "preflight-manifest.json").read_text())
    assert manifest["profile_schema_class"] == "A100DP2PhysicalProfile"
    assert (
        manifest["consumer_compatibility_identity"]
        == A100_CONSUMER_COMPATIBILITY_IDENTITY
    )
    assert manifest["transfer_cost_model_schema"] == TRANSFER_COST_MODEL_SCHEMA
    assert manifest["runtime_consumption_contract"] == RUNTIME_CONSUMPTION_CONTRACT
    assert manifest["hardware_label"] == HARDWARE_LABEL
    assert manifest["aimd_runtime_config"] == AIMD_RUNTIME_CONFIG
    assert manifest["aimd_consumption_contract"] == AIMD_CONSUMPTION_CONTRACT
    assert manifest["aimd_consumption_contract"]["expected_pool_count"] == 2
    assert manifest["authorization"] == {
        "suite_binding_authorized": True,
        "trace_loading_authorized": False,
        "model_fitting_authorized": False,
        "replay_authorized": False,
        "figure_generation_authorized": False,
        "performance_claim_authorized": False,
        "formal_ready": False,
        "paper_ready": False,
    }
    assert not (output_dir / "replay.json").exists()
    assert not (output_dir / "performance.json").exists()
    normalized = json.loads((output_dir / "a100-dp2-profile.json").read_text())
    assert normalized["schema_version"] == 2
    assert (
        normalized["consumer_compatibility_identity"]
        == A100_CONSUMER_COMPATIBILITY_IDENTITY
    )
    assert normalized["transfer_cost_model_schema"] == TRANSFER_COST_MODEL_SCHEMA
    assert normalized["runtime_consumption_contract"][
        "physical_calibration_identifiability"
    ] == {
        "repeats_per_treatment": 2,
        "total_transfer_runs": 6,
        "stream_identity": "slurm_job_run",
        "ordinal_scope": "per_run_per_event_actual_issue_order",
        "path_calibration_treatments": TRANSFER_PATH_TREATMENTS,
        "chunk_or_worker_log_is_stream": False,
    }
    assert set(normalized["metrics"]) == PROFILE_METRICS
    assert not any("effective_bytes_per_ms" in name for name in PROFILE_METRICS)
    with pytest.raises(FileExistsError):
        preflight_and_write(
            inputs=inputs, output_root=tmp_path / "independent-a100-output"
        )


def test_a100_consumer_materializes_only_static_calibrated_suite_binding(
    tmp_path: Path,
) -> None:
    _inputs, manifest_path, profile_path = _materialized_a100_consumer(tmp_path)
    binding_path = materialize_suite_binding(
        source=CalibratedConsumerInput(
            consumer_kind=A100_CONSUMER_KIND,
            consumer_manifest=manifest_path,
            consumer_payload=profile_path,
        ),
        matrix=_a100_suite_matrix(),
        output_root=tmp_path / "suite-output",
    )
    binding = validate_suite_binding(binding_path)
    assert binding["status"] == "BOUND_STATIC_ONLY"
    assert binding["source_consumer"]["hardware_label"] == HARDWARE_LABEL
    assert binding["source_consumer"]["layout"]["system_dp_size"] == 2
    assert binding["matrix"]["cell_count"] == 24
    assert binding["runner_contract"]["matrix_axis_semantics"] == {
        "model_scales": "exact_consumer_model_and_revision_only",
        "system_dp_sizes": "exact_physically_validated_system_dp_only",
        "concurrency_levels": "planned_max_inflight_requests",
        "failure_rates": (
            "planned_per_workflow_failure_probability_axis;"
            "schedule_requires_seeded_runner_materialization"
        ),
        "capacity_points": (
            "fractions_of_same_lineage_calibrated_hbm_and_dram_bytes"
        ),
        "seeds": "deterministic_matrix_and_failure_schedule_seed",
    }
    assert binding["result_state"] == {
        "execution_status": "NOT_RUN",
        "performance_results_present": False,
        "result_artifacts": [],
    }
    runner = preflight_suite_runner(binding_path)
    assert runner["status"] == "PASS_STATIC_ONLY"
    assert runner["cell_count"] == 24
    assert all(
        set(cell) == {"cell_id", "axes", "state"}
        and cell["state"] == "PLANNED_NOT_RUN"
        for cell in runner["cells"]
    )
    first_cell = runner["cells"][0]
    assert first_cell["cell_id"] == canonical_sha256(
        {
            "suite_binding_identity_sha256": binding[
                "suite_binding_identity_sha256"
            ],
            "axes": first_cell["axes"],
        }
    )[:20]
    assert runner["authorization"] == {
        "trace_loading_authorized": False,
        "model_fitting_authorized": False,
        "replay_authorized": False,
        "result_output_authorized": False,
        "performance_claim_authorized": False,
    }
    assert not (binding_path.parent / "replay.json").exists()
    assert not (binding_path.parent / "performance.json").exists()
    with pytest.raises(FileExistsError):
        materialize_suite_binding(
            source=CalibratedConsumerInput(
                consumer_kind=A100_CONSUMER_KIND,
                consumer_manifest=manifest_path,
                consumer_payload=profile_path,
            ),
            matrix=_a100_suite_matrix(),
            output_root=tmp_path / "suite-output",
        )


def test_clean_submission_gate_cannot_masquerade_as_consumer_preflight(
    tmp_path: Path,
) -> None:
    clean_gate = _write_json(
        tmp_path / "clean-preflight-submission-gate-v66.json",
        {
            "schema_version": 1,
            "evidence_class": "a100_dp2_clean_preflight_submission_gate_v66",
            "status": "PASS",
            "clean_preflight_submission_authorized": True,
            "model_submission_authorized": False,
        },
    )
    static_payload = _write_json(tmp_path / "static-readiness.json", {})
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(
        CalibratedSuiteContractError,
        match="clean submission/readiness gate is not.*consumer preflight",
    ):
        materialize_suite_binding(
            source=CalibratedConsumerInput(
                consumer_kind=A100_CONSUMER_KIND,
                consumer_manifest=clean_gate,
                consumer_payload=static_payload,
            ),
            matrix=_a100_suite_matrix(),
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("artifact_name", "legacy_class", "pattern"),
    [
        (
            "runtime_binding",
            "a100_dp2_v20r4_runtime_binding_v25",
            "v46 runtime binding",
        ),
        (
            "model_submission_gate",
            "a100_dp2_model_retry_submission_gate_v30",
            "v46 model-submission gate",
        ),
    ],
)
def test_superseded_v20r4_control_lineage_is_rejected(
    tmp_path: Path, artifact_name: str, legacy_class: str, pattern: str
) -> None:
    inputs, paths = _fixture(tmp_path)
    path = paths[artifact_name]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence_class"] = legacy_class
    _write_json(path, payload)
    with pytest.raises(A100DP2ContractError, match=pattern):
        validate_a100_dp2_bundle(inputs)


def test_v46_runtime_python_must_remain_a_relative_root_local_symlink(
    tmp_path: Path,
) -> None:
    inputs, paths = _fixture(tmp_path)
    binding = json.loads(paths["runtime_binding"].read_text(encoding="utf-8"))
    invocation = Path(binding["runtime_identity"]["python_invocation_path"])
    target = invocation.resolve()
    invocation.unlink()
    invocation.symlink_to(target)
    with pytest.raises(A100DP2ContractError, match="execution-root-local"):
        validate_a100_dp2_bundle(inputs)


@pytest.mark.parametrize("tamper", ["scheduler", "builder"])
def test_v46_model_submission_gate_rebuild_evidence_is_fail_closed(
    tmp_path: Path, tamper: str
) -> None:
    inputs, paths = _fixture(tmp_path)
    gate = json.loads(paths["model_submission_gate"].read_text(encoding="utf-8"))
    if tamper == "scheduler":
        gate["scheduler_final"]["elapsed_seconds"] += 1
        _write_json(paths["model_submission_gate"], gate)
        pattern = "scheduler evidence is not reproducible"
    else:
        Path(gate["gate_builder"]["path"]).write_text(
            "# drifted gate builder\n", encoding="utf-8"
        )
        pattern = "gate builder artifact hash mismatch"
    with pytest.raises(A100DP2ContractError, match=pattern):
        validate_a100_dp2_bundle(inputs)


def test_a100_preflight_cannot_cross_into_v100_suite_source(
    tmp_path: Path,
) -> None:
    _inputs, manifest_path, profile_path = _materialized_a100_consumer(tmp_path)
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(
        CalibratedSuiteContractError,
        match="A100 and V100 consumer lineages cannot cross",
    ):
        materialize_suite_binding(
            source=CalibratedConsumerInput(
                consumer_kind=V100_CONSUMER_KIND,
                consumer_manifest=manifest_path,
                consumer_payload=profile_path,
            ),
            matrix=_a100_suite_matrix(),
            output_root=output_root,
        )
    assert not output_root.exists()


def test_suite_runner_rejects_static_binding_authorization_tamper(
    tmp_path: Path,
) -> None:
    _inputs, manifest_path, profile_path = _materialized_a100_consumer(tmp_path)
    binding_path = materialize_suite_binding(
        source=CalibratedConsumerInput(
            consumer_kind=A100_CONSUMER_KIND,
            consumer_manifest=manifest_path,
            consumer_payload=profile_path,
        ),
        matrix=_a100_suite_matrix(),
        output_root=tmp_path / "suite-output",
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["authorization"]["replay_authorized"] = True
    _write_json(binding_path, binding)
    with pytest.raises(CalibratedSuiteContractError, match="not static-only"):
        preflight_suite_runner(binding_path)


def test_suite_runner_rehashes_original_a100_source_artifacts(
    tmp_path: Path,
) -> None:
    _inputs, manifest_path, profile_path = _materialized_a100_consumer(tmp_path)
    binding_path = materialize_suite_binding(
        source=CalibratedConsumerInput(
            consumer_kind=A100_CONSUMER_KIND,
            consumer_manifest=manifest_path,
            consumer_payload=profile_path,
        ),
        matrix=_a100_suite_matrix(),
        output_root=tmp_path / "suite-output",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(
        Path(manifest["input_artifacts"]["source_manifest"]["path"])
        .read_text(encoding="utf-8")
    )
    source_entry = source_manifest["entries"][0]
    source_file = (
        Path(source_manifest["repository_root"]) / source_entry["relative_path"]
    )
    source_file.write_text("CURRENT_DESIGN = False\n", encoding="utf-8")
    with pytest.raises(A100DP2ContractError, match="source entry hash mismatch"):
        preflight_suite_runner(binding_path)


@pytest.mark.parametrize("axis", ["model", "system_dp"])
def test_suite_matrix_cannot_extrapolate_model_or_system_dp(
    tmp_path: Path, axis: str
) -> None:
    _inputs, manifest_path, profile_path = _materialized_a100_consumer(tmp_path)
    matrix = _a100_suite_matrix()
    if axis == "model":
        matrix["model_scales"][0]["model"] = "Qwen/Qwen3-32B"
        pattern = "model scale/revision"
    else:
        matrix["system_dp_sizes"] = [4]
        pattern = "system-DP"
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(CalibratedSuiteContractError, match=pattern):
        materialize_suite_binding(
            source=CalibratedConsumerInput(
                consumer_kind=A100_CONSUMER_KIND,
                consumer_manifest=manifest_path,
                consumer_payload=profile_path,
            ),
            matrix=matrix,
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("tamper", "pattern"),
    [
        ("missing_residency", "four independent residency outcomes"),
        ("target_owned_hbm", "source-owned HBM D2D/DRAM D2H"),
        ("target_owned_dram", "source-owned HBM D2D/DRAM D2H"),
        ("hbm_to_dram", "target-owned or HBM-to-DRAM"),
        ("missing_ack", "independent HBM/DRAM ACKs"),
        ("missing_recovery", "lacks both physical recovery paths"),
        ("dram_backing_reclaim", "directly reclaim an HBM_ONLY tail"),
    ],
)
def test_a100_strong_current_design_semantics_are_mandatory(
    tmp_path: Path, tamper: str, pattern: str
) -> None:
    inputs, paths = _fixture(tmp_path)

    def mutate(profile: dict[str, Any]) -> None:
        events = profile["workload_window_events"]
        if tamper == "missing_residency":
            events["independent_tier_outcomes"]["HBM_ONLY"] = 0
        elif tamper == "target_owned_hbm":
            events["source_owned_paths"]["peer_hbm_protection"][
                "owner"
            ] = "target_worker"
        elif tamper == "target_owned_dram":
            events["source_owned_paths"]["direct_dram_protection"][
                "owner"
            ] = "target_worker_hbm_to_dram"
        elif tamper == "hbm_to_dram":
            events["forbidden_semantics"]["hbm_to_dram_transfer_count"] = 1
        elif tamper == "missing_ack":
            events["path_confirmations"]["dram_independent_ack_count"] = 0
        elif tamper == "missing_recovery":
            events["source_owned_paths"]["peer_hbm_recovery"]["count"] = 0
        else:
            for sample in events["aimd"]["reclaim_samples"]:
                sample["dram_backing_required"] = True

    _rewrite_all_aegis_conditions(paths["mechanism_gate"], mutate)
    with pytest.raises(A100DP2ContractError, match=pattern):
        validate_a100_dp2_bundle(inputs)


def test_v100_profile_cannot_cross_into_a100_consumer(tmp_path: Path) -> None:
    inputs, _paths = _fixture(tmp_path)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    profile["hardware_label"] = "V100S-1node-TP1-PP1-systemDP4"
    profile["hardware_profile"]["gpu_capability"] = [7, 0]
    profile["layout"]["dp_size"] = 4
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(A100DP2ContractError, match="A100 schema-v2 physical"):
        validate_a100_dp2_bundle(inputs)


def test_old_design_semantics_fail_closed(tmp_path: Path) -> None:
    inputs, _paths = _fixture(tmp_path)
    config = json.loads(inputs.config_manifest.read_text(encoding="utf-8"))
    config["design_semantics"] = "target_owned_background_destage_v1"
    _write_json(inputs.config_manifest, config)
    with pytest.raises(A100DP2ContractError, match="config manifest identity"):
        validate_a100_dp2_bundle(inputs)


def test_missing_required_a100_mechanism_check_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, _paths = _fixture(tmp_path)
    gate = json.loads(inputs.mechanism_gate.read_text(encoding="utf-8"))
    gate["checks"] = [
        check
        for check in gate["checks"]
        if check["name"]
        != "physically_calibrated_local_stall_threshold_applied"
    ]
    _write_json(inputs.mechanism_gate, gate)
    with pytest.raises(
        A100DP2ContractError, match="lacks a required passing check"
    ):
        validate_a100_dp2_bundle(inputs)


def test_missing_frozen_aimd_parameter_fails_closed(tmp_path: Path) -> None:
    inputs, _paths = _fixture(tmp_path)
    config = json.loads(inputs.config_manifest.read_text(encoding="utf-8"))
    config["configuration"]["aimd"].pop("decrease_factor")
    config["configuration_sha256"] = canonical_sha256(config["configuration"])
    _write_json(inputs.config_manifest, config)
    with pytest.raises(A100DP2ContractError, match="AIMD observation contract"):
        validate_a100_dp2_bundle(inputs)


def test_aegis_strict_audit_without_r12_aimd_block_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, paths = _fixture(tmp_path)
    _rewrite_strict_and_rebind(
        paths["condition_profile"],
        paths["mechanism_gate"],
        lambda strict: strict.pop("measurements"),
    )
    with pytest.raises(A100DP2ContractError, match="lacks strict-audit AIMD"):
        validate_a100_dp2_bundle(inputs)


@pytest.mark.parametrize("tamper", ["pool_count", "alpha_law"])
def test_strict_aimd_pool_topology_or_controller_law_tamper_fails_closed(
    tmp_path: Path, tamper: str
) -> None:
    inputs, paths = _fixture(tmp_path)

    def mutate(strict: dict[str, Any]) -> None:
        aimd = strict["measurements"]["aegiskv_data_plane"]["aimd"]
        if tamper == "pool_count":
            aimd["pool_count"] = 4
        else:
            aimd["pools"]["hbm-pool-1"]["trajectory"][1]["alpha"] = 0.3

    _rewrite_strict_and_rebind(
        paths["condition_profile"], paths["mechanism_gate"], mutate
    )
    pattern = "DP2 Router topology" if tamper == "pool_count" else "AIMD/quota law"
    with pytest.raises(A100DP2ContractError, match=pattern):
        validate_a100_dp2_bundle(inputs)


def test_no_protection_strict_audit_with_aimd_activity_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, _paths = _fixture(tmp_path)
    gate = json.loads(inputs.mechanism_gate.read_text(encoding="utf-8"))
    baseline_profile = next(
        Path(reference["path"])
        for reference in gate["condition_profiles"]
        if json.loads(Path(reference["path"]).read_text(encoding="utf-8"))[
            "treatment"
        ]
        == "no-protection"
    )

    def mutate(strict: dict[str, Any]) -> None:
        strict["measurements"] = {
            "aegiskv_data_plane": {"aimd": _strict_aimd_measurements()}
        }

    _rewrite_strict_and_rebind(baseline_profile, inputs.mechanism_gate, mutate)
    with pytest.raises(A100DP2ContractError, match="nonzero strict AIMD"):
        validate_a100_dp2_bundle(inputs)


@pytest.mark.parametrize("tamper", ["source", "strict"])
def test_lineage_or_hash_tamper_fails_closed(
    tmp_path: Path, tamper: str
) -> None:
    inputs, paths = _fixture(tmp_path)
    if tamper == "source":
        paths["source_file"].write_text("CURRENT_DESIGN = False\n", encoding="utf-8")
        pattern = "source entry hash mismatch"
    else:
        profile = json.loads(paths["condition_profile"].read_text(encoding="utf-8"))
        strict = Path(profile["input_lineage"]["strict_run_audit"]["path"])
        strict.write_text(strict.read_text(encoding="utf-8") + " ", encoding="utf-8")
        pattern = "strict.*hash mismatch"
    with pytest.raises(A100DP2ContractError, match=pattern):
        validate_a100_dp2_bundle(inputs)


def test_missing_physical_calibration_rejects_before_output(tmp_path: Path) -> None:
    inputs, _paths = _fixture(tmp_path)
    missing = tmp_path / "missing-physical-profile.json"
    broken = A100DP2InputPaths(
        source_manifest=inputs.source_manifest,
        runtime_binding=inputs.runtime_binding,
        model_submission_gate=inputs.model_submission_gate,
        config_manifest=inputs.config_manifest,
        config_verification=inputs.config_verification,
        physical_profile=missing,
        physical_profile_audit=inputs.physical_profile_audit,
        aimd_stall_calibration=inputs.aimd_stall_calibration,
        mechanism_gate=inputs.mechanism_gate,
    )
    output_root = tmp_path / "must-not-exist"
    with pytest.raises(A100DP2ContractError, match="physical_profile"):
        preflight_and_write(inputs=broken, output_root=output_root)
    assert not output_root.exists()


def test_missing_recompute_or_transfer_calibration_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, paths = _fixture(tmp_path)
    paths["recompute_audit"].unlink()
    with pytest.raises(A100DP2ContractError, match="recompute calibration"):
        validate_a100_dp2_bundle(inputs)


def test_legacy_single_effective_bandwidth_profile_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, _paths = _fixture(tmp_path)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    profile["metrics"] = {
        "recompute_ms_per_token": 0.125,
        "recompute_fixed_overhead_ms": 4.0,
        "peer_hbm_protection_effective_bytes_per_ms": 50.0,
        "direct_dram_protection_effective_bytes_per_ms": 25.0,
        "peer_hbm_recovery_effective_bytes_per_ms": 25.0,
        "dram_recovery_effective_bytes_per_ms": 10.0,
    }
    profile.pop("transfer_measurements")
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(A100DP2ContractError, match="physical metric set changed"):
        validate_a100_dp2_bundle(inputs)


def test_missing_declared_transfer_setup_field_fails_closed(tmp_path: Path) -> None:
    inputs, _paths = _fixture(tmp_path)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    profile["transfer_measurements"]["peer_hbm_protection"]["cost_model"].pop(
        "per_transfer_setup_ms"
    )
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(
        A100DP2ContractError, match="affine transfer cost model is not reproducible"
    ):
        validate_a100_dp2_bundle(inputs)


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("peer_hbm_protection_per_transfer_setup_ms", 0.0),
        ("peer_hbm_protection_first_transfer_warmup_ms", 0.0),
        ("peer_hbm_protection_steady_state_bytes_per_ms", 100.0),
    ],
)
def test_setup_warmup_or_steady_state_cannot_be_folded_into_one_rate(
    tmp_path: Path, metric: str, value: float
) -> None:
    inputs, _paths = _fixture(tmp_path)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    profile["metrics"][metric] = value
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(
        A100DP2ContractError, match="affine transfer metric is not reproducible"
    ):
        validate_a100_dp2_bundle(inputs)


def test_missing_explicit_calibration_transfer_ordinal_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, paths = _fixture(tmp_path)
    gate_log = paths["hbm_gate_log"]
    rows = [
        json.loads(line)
        for line in gate_log.read_text(encoding="utf-8").splitlines()
        if line
    ]
    next(
        row for row in rows if row.get("event") == "hbm_physical_copy"
    ).pop("calibration_transfer_ordinal")
    _write_jsonl(gate_log, rows)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    for reference in profile["source_artifacts"]["gate_logs"]:
        if Path(reference["path"]).resolve() == gate_log.resolve():
            reference.update(artifact_record(gate_log))
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(
        A100DP2ContractError, match="invalid affine physical calibration sample"
    ):
        validate_a100_dp2_bundle(inputs)


def test_transfer_calibration_requires_six_run_artifacts(tmp_path: Path) -> None:
    inputs, _paths = _fixture(tmp_path)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    profile["source_artifacts"]["transfer_audits"].pop()
    profile["source_artifacts"]["generic_transfer_audits"].pop()
    profile["source_artifacts"]["gate_logs"].pop()
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(
        A100DP2ContractError, match="exactly two runs per treatment"
    ):
        validate_a100_dp2_bundle(inputs)


def test_duplicate_run_stream_id_fails_closed(tmp_path: Path) -> None:
    inputs, paths = _fixture(tmp_path)
    first_gate, second_gate = paths["gate_logs"][:2]
    first_rows = [
        json.loads(line)
        for line in first_gate.read_text(encoding="utf-8").splitlines()
        if line
    ]
    duplicate_stream = next(
        row["calibration_stream_id"]
        for row in first_rows
        if row.get("event") == "hbm_physical_load"
    )
    second_rows = [
        json.loads(line)
        for line in second_gate.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in second_rows:
        if row.get("event") in set(TRANSFER_PATH_EVENTS.values()):
            row["calibration_stream_id"] = duplicate_stream
    _write_jsonl(second_gate, second_rows)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    for reference in profile["source_artifacts"]["gate_logs"]:
        if Path(reference["path"]).resolve() == second_gate.resolve():
            reference.update(artifact_record(second_gate))
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(A100DP2ContractError, match="not run-derived"):
        validate_a100_dp2_bundle(inputs)


def test_transfer_repeat_runtime_lineage_drift_fails_closed(tmp_path: Path) -> None:
    inputs, paths = _fixture(tmp_path)
    generic_path = paths["generic_audits"][1]
    generic = json.loads(generic_path.read_text(encoding="utf-8"))
    forged_reference = dict(generic["input_lineage"]["runtime_binding"])
    forged_reference["sha256"] = "0" * 64
    generic["input_lineage"]["runtime_binding"] = forged_reference
    generic["measurement"]["input_lineage"]["runtime_binding"] = forged_reference
    _write_json(generic_path, generic)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    for reference in profile["source_artifacts"]["generic_transfer_audits"]:
        if Path(reference["path"]).resolve() == generic_path.resolve():
            reference.update(artifact_record(generic_path))
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(A100DP2ContractError, match="runtime_binding.*hash mismatch"):
        validate_a100_dp2_bundle(inputs)


@pytest.mark.parametrize("fake_stream", ["chunk:0", "worker-log:dp1"])
def test_chunk_or_worker_log_cannot_masquerade_as_stream(
    tmp_path: Path, fake_stream: str
) -> None:
    inputs, paths = _fixture(tmp_path)
    gate_log = paths["hbm_gate_log"]
    rows = [
        json.loads(line)
        for line in gate_log.read_text(encoding="utf-8").splitlines()
        if line
    ]
    next(row for row in rows if row.get("event") == "hbm_physical_load")[
        "calibration_stream_id"
    ] = fake_stream
    _write_jsonl(gate_log, rows)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    for reference in profile["source_artifacts"]["gate_logs"]:
        if Path(reference["path"]).resolve() == gate_log.resolve():
            reference.update(artifact_record(gate_log))
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(A100DP2ContractError, match="not run-derived"):
        validate_a100_dp2_bundle(inputs)


def test_transfer_ordinal_must_follow_run_issue_order(tmp_path: Path) -> None:
    inputs, paths = _fixture(tmp_path)
    gate_log = paths["hbm_gate_log"]
    rows = [
        json.loads(line)
        for line in gate_log.read_text(encoding="utf-8").splitlines()
        if line
    ]
    event_rows = [row for row in rows if row.get("event") == "hbm_physical_load"]
    event_rows[1]["calibration_transfer_ordinal"] = 2
    event_rows[2]["calibration_transfer_ordinal"] = 1
    _write_jsonl(gate_log, rows)
    profile = json.loads(inputs.physical_profile.read_text(encoding="utf-8"))
    for reference in profile["source_artifacts"]["gate_logs"]:
        if Path(reference["path"]).resolve() == gate_log.resolve():
            reference.update(artifact_record(gate_log))
    _write_json(inputs.physical_profile, profile)
    with pytest.raises(A100DP2ContractError, match="issue order"):
        validate_a100_dp2_bundle(inputs)


def test_v53_series_requires_two_attempts_per_treatment(tmp_path: Path) -> None:
    inputs, paths = _fixture(tmp_path)
    series = json.loads(paths["transfer_series"].read_text(encoding="utf-8"))
    series["transfer_attempt_audits"]["aegiskv"].pop()
    _write_json(paths["transfer_series"], series)
    audit = json.loads(paths["physical_audit"].read_text(encoding="utf-8"))
    audit["transfer_series_ready"] = artifact_record(paths["transfer_series"])
    audit["transfer_series_ready_v53"] = artifact_record(
        paths["transfer_series"]
    )
    _write_json(paths["physical_audit"], audit)
    with pytest.raises(A100DP2ContractError, match="v53 transfer-calibration series"):
        validate_a100_dp2_bundle(inputs)


def test_cross_node_transfer_calibration_lineage_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, paths = _fixture(tmp_path)
    attempt_path = paths["transfer_attempt"]
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["calibration_node"] = "gpu26"
    _write_json(attempt_path, attempt)

    series = json.loads(paths["transfer_series"].read_text(encoding="utf-8"))
    treatment = attempt["treatment"]
    series["transfer_attempt_audits"][treatment] = [
        artifact_record(attempt_path)
        if Path(reference["path"]).resolve() == attempt_path.resolve()
        else reference
        for reference in series["transfer_attempt_audits"][treatment]
    ]
    _write_json(paths["transfer_series"], series)

    audit = json.loads(paths["physical_audit"].read_text(encoding="utf-8"))
    audit["transfer_series_ready"] = artifact_record(paths["transfer_series"])
    audit["transfer_series_ready_v53"] = artifact_record(
        paths["transfer_series"]
    )
    audit["transfer_attempt_audits"] = [
        artifact_record(attempt_path)
        if Path(reference["path"]).resolve() == attempt_path.resolve()
        else reference
        for reference in audit["transfer_attempt_audits"]
    ]
    _write_json(paths["physical_audit"], audit)
    with pytest.raises(A100DP2ContractError, match="crossed node"):
        validate_a100_dp2_bundle(inputs)


def test_performance_window_cannot_include_aimd_observation_tail(
    tmp_path: Path,
) -> None:
    inputs, paths = _fixture(tmp_path)

    def count_tail(profile: dict[str, Any]) -> None:
        profile["workload"]["finished_at_ms"] = profile[
            "mechanism_observation_window"
        ]["finished_at_ms"]

    _rewrite_condition_and_gate(
        paths["condition_profile"], paths["mechanism_gate"], count_tail
    )
    with pytest.raises(A100DP2ContractError, match="counts the AIMD tail"):
        validate_a100_dp2_bundle(inputs)


def test_old_or_wrong_wire_enum_fails_closed(tmp_path: Path) -> None:
    inputs, paths = _fixture(tmp_path)

    def old_wire(profile: dict[str, Any]) -> None:
        profile["runtime_admission_decision_wire_values"] = {
            "BYPASS": 2,
            "ADMIT_DRAM": 1,
            "INVALID": 3,
            "ADMIT_HBM": 0,
        }

    _rewrite_condition_and_gate(
        paths["condition_profile"], paths["mechanism_gate"], old_wire
    )
    with pytest.raises(A100DP2ContractError, match="condition profile identity"):
        validate_a100_dp2_bundle(inputs)


def test_v2_worker_device_identity_is_fail_closed(tmp_path: Path) -> None:
    inputs, paths = _fixture(tmp_path)
    model = json.loads(paths["model_probe"].read_text(encoding="utf-8"))
    model["worker_results"][1]["cuda_visible_devices"] = "0"
    model["worker_results"][1]["model_runner_v2_log_marker_observed"] = False
    _write_json(paths["model_probe"], model)
    with pytest.raises(A100DP2ContractError, match="model execution probe.*hash"):
        validate_a100_dp2_bundle(inputs)


@pytest.mark.parametrize("mode", ["pucvmet", "one_device", "space_format"])
def test_v46_exact_gpu_telemetry_contract_is_fail_closed(
    tmp_path: Path, mode: str
) -> None:
    inputs, _paths = _fixture(tmp_path, launcher_telemetry_mode=mode)
    with pytest.raises(A100DP2ContractError, match="two-device CSV telemetry"):
        validate_a100_dp2_bundle(inputs)


def test_legacy_whitespace_dmon_output_is_rejected(tmp_path: Path) -> None:
    inputs, paths = _fixture(tmp_path)
    profile_path = paths["condition_profile"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    raw_path = Path(profile["artifacts"]["gpu_telemetry"]["path"])
    raw_path.write_text(
        "# gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk fb bar1 "
        "ccpm sbecc dbecc pci rxpci txpci pviol tviol\n"
        "0 100 45 40 20 20 0 0 0 0 1215 1410 1024 64 0 0 0 0 1 2 0 0\n",
        encoding="utf-8",
    )

    def bind_legacy_raw(condition: dict[str, Any]) -> None:
        reference = artifact_record(raw_path)
        condition["artifacts"]["gpu_telemetry"] = reference
        condition["gpu_telemetry"]["artifact"] = reference

    _rewrite_condition_and_gate(profile_path, paths["mechanism_gate"], bind_legacy_raw)
    with pytest.raises(A100DP2ContractError, match="not exact --select=pucmet"):
        validate_a100_dp2_bundle(inputs)


def test_aggregate_nonzero_cannot_hide_inactive_second_gpu(tmp_path: Path) -> None:
    inputs, paths = _fixture(tmp_path)
    profile_path = paths["condition_profile"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    raw_path = Path(profile["artifacts"]["gpu_telemetry"]["path"])
    _write_gpu_csv(raw_path, gpu1_sm=0.0, gpu1_fb=0.0)

    def bind_inactive_gpu_raw(condition: dict[str, Any]) -> None:
        # Keep the producer's aggregate nonzero claims unchanged.  The
        # independent consumer must still inspect each raw GPU stream.
        reference = artifact_record(raw_path)
        condition["artifacts"]["gpu_telemetry"] = reference
        condition["gpu_telemetry"]["artifact"] = reference

    _rewrite_condition_and_gate(
        profile_path, paths["mechanism_gate"], bind_inactive_gpu_raw
    )
    with pytest.raises(A100DP2ContractError, match="activity for GPU 1"):
        validate_a100_dp2_bundle(inputs)


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("missing", "producer per_gpu_telemetry check did not pass"),
        ("failure", "producer per_gpu_telemetry check did not pass"),
        ("drift", "producer per_gpu_runtime_activity mismatch for GPU 1"),
        ("inactive_raw", "activity for GPU 1"),
    ],
)
def test_model_audit_per_gpu_fields_and_checks_are_mandatory(
    tmp_path: Path, mode: str, error: str
) -> None:
    inputs, _paths = _fixture(tmp_path, producer_per_gpu_mode=mode)
    with pytest.raises(A100DP2ContractError, match=error):
        validate_a100_dp2_bundle(inputs)
