#!/usr/bin/env python3
"""Full-scale, real-Weka replay of the current AegisKV design.

The trace supplies real workflow topology, token counts, request durations, and
fan-out timing.  SGLSim supplies metadata-only KV placement and failure replay.
Physical coefficients are read from explicit Mooncake calibration gates; the
script never upgrades those calibration gates into end-to-end runtime evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.agentic_workflows.weka_full_trace import (
    CausalSlackTicket,
    WekaRequest,
    WekaWorkflow,
    build_causal_samples,
    crossfit_causal_slack,
    dataset_manifest,
    default_full_trace_path,
    load_weka_workflows,
    sha256_file,
    zero_slack_tickets_for_unmodeled_requests,
)
from sglsim.srt.simulation.aegis_full_design import (
    AegisFullDesignConfig,
    AegisFullDesignRuntime,
    CalibratedTransferPath,
    SealedChunk,
)
from sglsim.srt.simulation.aegis_gbdt import AegisGBDTConfig
from sglsim.srt.simulation.aegis_independent_tiers import (
    DESIGN_SEMANTICS,
    CopyState,
    ProtectionTier,
    RecoveryDecision,
    RecoveryTiming,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parent
POLICY_SPECS = (
    ("recompute_fcfs", "recompute", "fcfs"),
    ("recompute_slack_due", "recompute", "slack_due"),
    ("zero_slack_independent_fcfs", "zero_slack", "fcfs"),
    ("zero_slack_independent_due", "zero_slack", "slack_due"),
    ("aegiskv_independent_fcfs", "aegiskv", "fcfs"),
    ("aegiskv_latest", "aegiskv", "slack_due"),
)
PLACEMENT_POLICIES = ("aegiskv", "zero_slack")
V100_PROFILE_SEMANTICS = "source_owned_four_physical_paths_v1"
V100_HARDWARE_LABEL = "V100S-1node-TP1-PP1-systemDP4"
V100_PILOT_RUNTIME_GATE_EVIDENCE = (
    "v100s_dp4_current_design_four_condition_mechanism_gate"
)
V100_RUNTIME_PROFILE_EVIDENCE = (
    "aegiskv_v100_dp4_condition_runtime_profile"
)
V100_CONFIG_MANIFEST_EVIDENCE = "aegiskv_physical_campaign_config_manifest"
V100_CONFIG_VERIFICATION_EVIDENCE = (
    "v100s_dp4_config_manifest_verification"
)
V100_SOURCE_ACTIVATION_EVIDENCE = "v100s_dp4_source_activation_gate"
V100_MODEL_EXECUTION_PROBE_EVIDENCE = (
    "v100s_dp4_qwen3_8b_vllm_v2_model_execution_probe"
)
V100_MODEL_EXECUTION_WORKER_EVIDENCE = (
    "v100s_qwen3_8b_vllm_v2_worker_execution"
)
V100_CONFIG_INPUT_EVIDENCE = "v100s_dp4_current_design_config_input"
V100_SUPERSEDED_SOURCE_MANIFEST_SHA256 = (
    "aad3549b40ec9bd8543d94367fba6d0a4a498272c400be06885255f5e903e9da"
)
# Mooncake's runtime wire enum is not zero-based admission: zero is BYPASS.
# SGLSim consumes producer-normalized named counters, but independently checks
# this source-activation identity before accepting any future physical profile.
V100_RUNTIME_ADMISSION_DECISIONS = {
    "BYPASS": 0,
    "ADMIT_DRAM": 1,
    "INVALID": 2,
    "ADMIT_HBM": 3,
}
V100_RUNTIME_PROFILE_LAYOUT = {
    "model": "Qwen/Qwen3-8B",
    "serving_dtype": "half",
    "kv_cache_dtype": "fp16",
    "tp_size": 1,
    "pp_size": 1,
    "system_dp_size": 4,
    "worker_count": 4,
}
V100_REQUIRED_RUNTIME_CHECKS = {
    "four_exact_unique_conditions",
    "all_profiles_strict_pass_current_design",
    "exact_v100s_dp4_hardware_and_layout",
    "clean_four_gpu_workload_telemetry",
    "all_lineage_artifacts_exist_and_match_bytes",
    "healthy_aegiskv_has_no_recovery_load",
    "source_owned_peer_hbm_and_direct_dram_protection_physical",
    "target_local_stall_contracts_alpha_and_hbm_quota",
    "v100s_four_pool_physical_pressure_signal_contract_complete",
    # Stable current-design mechanism names emitted by the Mooncake v20
    # producer.  HBM and DRAM may execute concurrently or separately; timing
    # overlap is intentionally not a correctness requirement.
    "single_source_config_physical_profile_lineage",
    "baseline_has_no_aegiskv_mechanism_or_data_path",
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
V100_PILOT_CONDITIONS = {
    ("no-protection", "none"),
    ("no-protection", "sigkill_process_group"),
    ("aegiskv", "none"),
    ("aegiskv", "sigkill_process_group"),
}
V100_RUNTIME_PROFILE_ARTIFACTS = {
    "deployment",
    "run_config",
    "summary",
    "strict_audit",
    "gpu_telemetry",
    "source_manifest",
    "physical_profile",
    "config_manifest",
    "config_verification",
    "slurm_co_residency",
    "gpu_processes_before",
    "gpu_processes_after",
    "router_events",
    "gate_events",
}
V100_PROFILE_METRICS = (
    "recompute_ms_per_token",
    "recompute_fixed_overhead_ms",
    "peer_hbm_protection_effective_bytes_per_ms",
    "direct_dram_protection_effective_bytes_per_ms",
    "peer_hbm_recovery_effective_bytes_per_ms",
    "dram_recovery_effective_bytes_per_ms",
)
V100_RUNTIME_CONSUMPTION_CONTRACT = {
    "tier_recovery_ms": {
        "peer_hbm": "peer_hbm_recovery_effective_bytes_per_ms",
        "dram": "dram_recovery_effective_bytes_per_ms",
    },
    "protection_observability": {
        "peer_hbm": "peer_hbm_protection_effective_bytes_per_ms",
        "dram": "direct_dram_protection_effective_bytes_per_ms",
    },
    "protection_rate_is_subtracted_from_admission_gain": False,
    "cross_tier_transfer_cost_is_charged": False,
}
V100_AIMD_PRESSURE_CALIBRATION_EVIDENCE = (
    "aegiskv_v100_dp4_aimd_pressure_calibration"
)
V100_AIMD_PRESSURE_MODE = (
    "physical_cuda_headroom_or_local_hbm_resource_stall_v1"
)
V100_COMPATIBILITY_IDENTITY = {
    "serving_dtype": "half",
    "kv_cache_dtype": "fp16",
    "kv_abi": "qwen3-8b-fp16-mrv2-block16-tp1-pp1-v1",
    "block_size": 16,
    "max_model_len": 4096,
    "logical_chunk_tokens": {"prefix": 4096, "decode": 256},
    "layout": {"tp_size": 1, "pp_size": 1, "dp_size": 4},
}
V100_CONFIG_HARDWARE_PROFILE = {
    "gpu_count": 4,
    "gpu_capability": [7, 0],
    "gpu_name_substring": "V100S",
    "physical_interconnect": "pcie_p2p",
}
V100_CONFIG_PATH_OWNERS = {
    "peer_hbm_protection": "source_worker_d2d_p2p",
    "direct_dram_protection": "source_worker_d2h",
    "peer_hbm_recovery": "survivor_worker_hbm_load",
    "dram_recovery": "survivor_worker_store_load",
}
V100_CONFIG_COMPATIBILITY_IDENTITY = {
    name: value
    for name, value in V100_COMPATIBILITY_IDENTITY.items()
    if name != "layout"
}


def _parallel_layout_id(config: dict[str, Any]) -> str:
    """Build the request layout identity from the calibrated model profile."""

    model = config.get("model")
    cluster = config.get("cluster")
    if not isinstance(model, dict) or not isinstance(cluster, dict):
        raise ValueError("model and cluster layout metadata are required")
    raw_dtype = str(
        model.get("kv_cache_dtype") or model.get("serving_dtype") or ""
    ).lower()
    dtype_aliases = {
        "half": "fp16",
        "float16": "fp16",
        "fp16": "fp16",
        "bfloat16": "bf16",
        "bf16": "bf16",
    }
    try:
        dtype = dtype_aliases[raw_dtype]
    except KeyError as exc:
        raise ValueError(
            "calibrated model profile must declare a supported KV cache dtype"
        ) from exc
    return (
        f"dp{int(cluster['dp_size'])}-pp{int(cluster['pp_size'])}-"
        f"tp{int(cluster['tp_size'])}-{dtype}"
    )


@dataclass
class ReplayRequest:
    row: WekaRequest
    workflow_base_ms: float
    workflow_origin_ms: float
    global_start_ms: float
    global_finish_ms: float
    prefill_finish_ms: float
    branch_finish_ms: float
    stage_finish_ms: float
    source_rank: int
    target_rank: int
    prefix_chunks: int
    decode_chunks: int
    source_worker: str = ""
    current_target: str = ""
    started: bool = False
    finished: bool = False
    rotated_before_source: bool = False

    @property
    def chunk_count(self) -> int:
        return self.prefix_chunks + self.decode_chunks

    def chunk(self, ordinal: int, *, prefix_q: int, decode_q: int) -> SealedChunk:
        index = int(ordinal)
        if not 0 <= index < self.chunk_count:
            raise IndexError(index)
        if index < self.prefix_chunks:
            start = index * int(prefix_q)
            end = min(start + int(prefix_q), self.row.prompt_tokens)
            return SealedChunk(index, start, end, "prefix")
        decode_index = index - self.prefix_chunks
        start = self.row.prompt_tokens + decode_index * int(decode_q)
        return SealedChunk(index, start, start + int(decode_q), "decode")

    def seal_time_ms(self, ordinal: int, *, prefix_q: int, decode_q: int) -> float:
        chunk = self.chunk(ordinal, prefix_q=prefix_q, decode_q=decode_q)
        if chunk.kind == "prefix":
            fraction = chunk.token_end / max(self.row.prompt_tokens, 1)
            return self.global_start_ms + (
                self.prefill_finish_ms - self.global_start_ms
            ) * fraction
        decode_tokens = chunk.token_end - self.row.prompt_tokens
        fraction = decode_tokens / max(self.row.output_tokens, 1)
        return self.prefill_finish_ms + (
            self.global_finish_ms - self.prefill_finish_ms
        ) * fraction

    def produced_tokens(self, at_ms: float) -> int:
        point = min(max(float(at_ms), self.global_start_ms), self.global_finish_ms)
        if point <= self.prefill_finish_ms + 1e-9:
            duration = max(self.prefill_finish_ms - self.global_start_ms, 1e-9)
            fraction = (point - self.global_start_ms) / duration
            return min(int(self.row.prompt_tokens * max(fraction, 0.0)), self.row.prompt_tokens)
        decode_duration = max(self.global_finish_ms - self.prefill_finish_ms, 1e-9)
        fraction = (point - self.prefill_finish_ms) / decode_duration
        generated = min(int(self.row.output_tokens * max(fraction, 0.0)), self.row.output_tokens)
        return self.row.prompt_tokens + generated


@dataclass
class ReplayLayout:
    requests: list[ReplayRequest]
    by_id: dict[str, ReplayRequest]
    by_source_rank: dict[int, list[ReplayRequest]]
    by_target_rank: dict[int, list[ReplayRequest]]
    stage_branch_finishes: dict[tuple[str, int], dict[str, float]]
    workflow_bases: dict[str, float]
    ttft_imputation_ratio: float
    workload_start_ms: float
    workload_finish_ms: float
    intended_chunk_count: int


@dataclass
class ReplayOutputs:
    placement_counts: dict[str, Counter[str]] = field(
        default_factory=lambda: {name: Counter() for name in PLACEMENT_POLICIES}
    )
    failure_rows: list[dict[str, Any]] = field(default_factory=list)
    stage_rows: list[dict[str, Any]] = field(default_factory=list)
    fault_events: list[dict[str, Any]] = field(default_factory=list)
    probe_candidates: list[tuple[str, float, ReplayRequest]] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scaled_tokens(value: int, factor: float, *, minimum: int) -> int:
    return max(int(math.ceil(int(value) * float(factor))), int(minimum))


def _scale_workflows(
    workflows: Sequence[WekaWorkflow], factor: float
) -> list[WekaWorkflow]:
    if not math.isfinite(float(factor)) or not 0.0 < float(factor) <= 1.0:
        raise ValueError("token projection factor must be in (0, 1]")
    return [
        replace(
            workflow,
            stages=tuple(
                replace(
                    stage,
                    branches=tuple(
                        replace(
                            branch,
                            requests=tuple(
                                replace(
                                    request,
                                    prompt_tokens=_scaled_tokens(
                                        request.prompt_tokens,
                                        factor,
                                        minimum=1,
                                    ),
                                    output_tokens=_scaled_tokens(
                                        request.output_tokens,
                                        factor,
                                        minimum=0,
                                    ),
                                )
                                for request in branch.requests
                            ),
                        )
                        for branch in stage.branches
                    ),
                )
                for stage in workflow.stages
            ),
        )
        for workflow in workflows
    ]


def _stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _stable_rank(text: str, size: int) -> int:
    return _stable_u64(text) % int(size)


def _source_target_ranks(request_id: str, dp_size: int) -> tuple[int, int]:
    source = _stable_rank(f"source:{request_id}", dp_size)
    offset = 1 + _stable_rank(f"target:{request_id}", dp_size - 1)
    return source, (source + offset) % dp_size


def _workflow_bases(
    workflows: Sequence[WekaWorkflow], mean_interarrival_ms: float
) -> dict[str, float]:
    current = 0.0
    output: dict[str, float] = {}
    for index, workflow in enumerate(workflows):
        if index:
            raw = (_stable_u64(f"arrival:{workflow.workflow_id}") + 0.5) / 2**64
            current += -float(mean_interarrival_ms) * math.log(max(raw, 1e-15))
        output[workflow.workflow_id] = current
    return output


def build_layout(
    workflows: Sequence[WekaWorkflow], config: dict[str, Any]
) -> ReplayLayout:
    prefix_q = int(config["chunking"]["prefix_tokens"])
    decode_q = int(config["chunking"]["decode_tokens"])
    dp_size = int(config["cluster"]["dp_size"])
    bases = _workflow_bases(
        workflows, float(config["workload"]["workflow_mean_interarrival_ms"])
    )
    observed_ratios = [
        min(max(row.ttft_ms / row.api_time_ms, 0.0), 1.0)
        for workflow in workflows
        for row in workflow.requests
        if row.ttft_ms is not None and row.api_time_ms > 0.0
    ]
    imputation_ratio = float(np.median(observed_ratios))
    requests: list[ReplayRequest] = []
    by_source: dict[int, list[ReplayRequest]] = defaultdict(list)
    by_target: dict[int, list[ReplayRequest]] = defaultdict(list)
    stage_finishes: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    origins = {workflow.workflow_id: workflow.start_ms for workflow in workflows}
    for workflow in workflows:
        base = bases[workflow.workflow_id]
        origin = origins[workflow.workflow_id]
        for stage in workflow.stages:
            stage_finish = base + stage.finish_ms - origin
            for branch in stage.branches:
                branch_finish = base + branch.finish_ms - origin
                stage_finishes[(workflow.workflow_id, stage.stage_id)][
                    branch.agent_id
                ] = branch_finish
                for row in branch.requests:
                    start = base + row.arrival_ms - origin
                    finish = start + row.api_time_ms
                    if row.ttft_ms is None:
                        prefill_duration = row.api_time_ms * imputation_ratio
                    else:
                        prefill_duration = min(row.ttft_ms, row.api_time_ms)
                    prefill_finish = start + max(prefill_duration, 0.0)
                    source_rank, target_rank = _source_target_ranks(row.rid, dp_size)
                    replay = ReplayRequest(
                        row=row,
                        workflow_base_ms=base,
                        workflow_origin_ms=origin,
                        global_start_ms=start,
                        global_finish_ms=finish,
                        prefill_finish_ms=prefill_finish,
                        branch_finish_ms=branch_finish,
                        stage_finish_ms=stage_finish,
                        source_rank=source_rank,
                        target_rank=target_rank,
                        prefix_chunks=(row.prompt_tokens + prefix_q - 1) // prefix_q,
                        decode_chunks=row.output_tokens // decode_q,
                    )
                    requests.append(replay)
                    by_source[source_rank].append(replay)
                    by_target[target_rank].append(replay)
    if not requests:
        raise ValueError("replay contains no requests")
    return ReplayLayout(
        requests=requests,
        by_id={row.row.rid: row for row in requests},
        by_source_rank=dict(by_source),
        by_target_rank=dict(by_target),
        stage_branch_finishes=dict(stage_finishes),
        workflow_bases=bases,
        ttft_imputation_ratio=imputation_ratio,
        workload_start_ms=min(row.global_start_ms for row in requests),
        workload_finish_ms=max(row.global_finish_ms for row in requests),
        intended_chunk_count=sum(row.chunk_count for row in requests),
    )


def _validate_gate(path: Path, expected_class: str) -> dict[str, Any]:
    gate = _load_json(path)
    if gate.get("status") != "PASS":
        raise ValueError(f"calibration gate did not pass: {path}")
    if gate.get("design_semantics") != DESIGN_SEMANTICS:
        raise ValueError(f"calibration design semantics mismatch: {path}")
    if gate.get("evidence_class") != expected_class:
        raise ValueError(f"unexpected calibration evidence class: {path}")
    return gate


def _verify_artifact(record: dict[str, Any], *, label: str) -> Path:
    try:
        path = Path(str(record["path"])).resolve()
        expected_sha = str(record["sha256"])
    except KeyError as exc:
        raise ValueError(f"{label} artifact record is incomplete") from exc
    if not path.is_file():
        raise ValueError(f"{label} artifact is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"{label} artifact sha256 mismatch: expected {expected_sha}, "
            f"observed {actual_sha}"
        )
    if "size_bytes" in record and path.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"{label} artifact size mismatch: {path}")
    return path


def _verify_artifact_tree(value: Any, *, label: str) -> None:
    """Re-hash every artifact record nested under a runtime profile."""

    if isinstance(value, dict):
        if "path" in value or "sha256" in value:
            _verify_artifact(value, label=label)
            return
        for name, child in value.items():
            _verify_artifact_tree(child, label=f"{label}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _verify_artifact_tree(child, label=f"{label}[{index}]")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_lower_hex(value: Any, *, lengths: set[int]) -> bool:
    text = str(value or "")
    return len(text) in lengths and all(
        character in "0123456789abcdef" for character in text
    )


def _verify_v100_config_lineage(
    calibration: dict[str, Any],
    profile: dict[str, Any],
    *,
    source_manifest_record: dict[str, Any],
    source_verification_record: dict[str, Any],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Independently validate the complete staged V100 v20 config chain."""

    config_record = calibration.get("config_manifest")
    config_verification_record = calibration.get("config_verification")
    profile_config_record = profile.get("config_manifest")
    profile_config_verification_record = profile.get("config_verification")
    if not all(
        isinstance(record, dict)
        for record in (
            config_record,
            config_verification_record,
            profile_config_record,
            profile_config_verification_record,
        )
    ):
        raise ValueError("V100 config manifest lineage is incomplete")
    _same_artifact_identity(
        config_record,
        profile_config_record,
        label="V100 config manifest",
    )
    _same_artifact_identity(
        config_verification_record,
        profile_config_verification_record,
        label="V100 config verification",
    )
    config_path = _verify_artifact(
        config_record, label="V100 campaign config manifest"
    )
    config_verification_path = _verify_artifact(
        config_verification_record,
        label="V100 campaign config verification",
    )
    config_manifest = _load_json(config_path)
    config_verification = _load_json(config_verification_path)
    config_source = config_manifest.get("source_manifest")
    config_source_verification = config_manifest.get("source_verification")
    activation_record = config_manifest.get("source_activation_gate")
    model_probe_record = config_manifest.get("model_execution_probe")
    configuration_record = config_manifest.get("configuration_input")
    configuration = config_manifest.get("configuration")
    verified_manifest = config_verification.get("manifest")
    model_revision = str(config_manifest.get("model_revision") or "")
    model_config_sha256 = str(
        config_manifest.get("model_config_sha256") or ""
    )
    runtime_versions = config_manifest.get("runtime_versions")
    if not (
        config_manifest.get("schema_version") == 1
        and config_manifest.get("evidence_class")
        == V100_CONFIG_MANIFEST_EVIDENCE
        and config_manifest.get("status") == "PINNED"
        and str(config_manifest.get("campaign_id") or "").startswith(
            "qwen3-v100-dp4-"
        )
        and config_manifest.get("design_semantics") == DESIGN_SEMANTICS
        and config_manifest.get("hardware_label") == V100_HARDWARE_LABEL
        and config_manifest.get("model") == "Qwen/Qwen3-8B"
        and _is_lower_hex(model_revision, lengths=set(range(8, 65)))
        and _is_lower_hex(model_config_sha256, lengths={64})
        and isinstance(runtime_versions, dict)
        and all(
            str(runtime_versions.get(name) or "")
            for name in ("vllm", "torch", "cuda")
        )
        and config_manifest.get("layout") == V100_RUNTIME_PROFILE_LAYOUT
        and config_manifest.get("hardware_profile")
        == V100_CONFIG_HARDWARE_PROFILE
        and isinstance(config_source, dict)
        and isinstance(config_source_verification, dict)
        and isinstance(activation_record, dict)
        and isinstance(model_probe_record, dict)
        and isinstance(configuration_record, dict)
        and isinstance(configuration, dict)
        and config_manifest.get("configuration_sha256")
        == _canonical_json_sha256(configuration)
        and config_manifest.get("jobs_submitted") == []
        and config_manifest.get("formal_ready") is False
        and config_manifest.get("paper_ready") is False
        and profile.get("model_revision") == model_revision
    ):
        raise ValueError("V100 campaign config manifest contract did not pass")
    _same_artifact_identity(
        config_source,
        source_manifest_record,
        label="V100 config/source manifest",
    )
    _same_artifact_identity(
        config_source_verification,
        source_verification_record,
        label="V100 config/source verification",
    )
    if not (
        config_verification.get("schema_version") == 1
        and config_verification.get("evidence_class")
        == V100_CONFIG_VERIFICATION_EVIDENCE
        and config_verification.get("status") == "PASS"
        and config_verification.get("failures") == []
        and config_verification.get("campaign_id")
        == config_manifest.get("campaign_id")
        and config_verification.get("hardware_label") == V100_HARDWARE_LABEL
        and config_verification.get("design_semantics") == DESIGN_SEMANTICS
        and isinstance(verified_manifest, dict)
        and config_verification.get("jobs_submitted") == []
        and config_verification.get("formal_ready") is False
        and config_verification.get("paper_ready") is False
    ):
        raise ValueError("V100 campaign config verification did not pass")
    _same_artifact_identity(
        verified_manifest,
        config_record,
        label="V100 verified config manifest",
    )

    activation_path = _verify_artifact(
        activation_record, label="V100 source activation gate"
    )
    activation = _load_json(activation_path)
    activation_source = activation.get("source_manifest")
    activation_source_verification = activation.get("source_verification")
    activation_checks = activation.get("checks")
    required_activation_checks = {
        "new_campaign_id",
        "source_freeze_supersedes_queued_v11",
        "source_manifest_contract",
        "source_verification_contract",
        "all_pinned_source_entries_live",
        "current_design_runtime_markers",
        "admission_decision_wire_values",
    }
    if not (
        activation.get("schema_version") == 1
        and activation.get("evidence_class")
        == V100_SOURCE_ACTIVATION_EVIDENCE
        and activation.get("status") == "PASS"
        and activation.get("failures") == []
        and activation.get("campaign_id")
        == config_manifest.get("campaign_id")
        and activation.get("hardware_label") == V100_HARDWARE_LABEL
        and activation.get("design_semantics") == DESIGN_SEMANTICS
        and isinstance(activation_source, dict)
        and isinstance(activation_source_verification, dict)
        and activation.get("superseded_source_manifest_sha256")
        == V100_SUPERSEDED_SOURCE_MANIFEST_SHA256
        and int(activation.get("checked_entry_count", -1))
        == int(source_manifest.get("entry_count", -2))
        and activation.get("admission_decision_wire_values")
        == V100_RUNTIME_ADMISSION_DECISIONS
        and isinstance(activation_checks, dict)
        and required_activation_checks.issubset(activation_checks)
        and all(
            activation_checks.get(name) is True
            for name in required_activation_checks
        )
        and isinstance(activation.get("activated_runtime"), dict)
        and activation["activated_runtime"]
        and activation.get("mutations_performed") == []
        and activation.get("jobs_submitted") == []
        and activation.get("formal_ready") is False
        and activation.get("paper_ready") is False
    ):
        raise ValueError("V100 source activation contract did not pass")
    _same_artifact_identity(
        activation_source,
        source_manifest_record,
        label="V100 activation/source manifest",
    )
    _same_artifact_identity(
        activation_source_verification,
        source_verification_record,
        label="V100 activation/source verification",
    )
    _verify_artifact_tree(
        activation["activated_runtime"], label="V100 activated runtime"
    )

    model_probe_path = _verify_artifact(
        model_probe_record, label="V100 four-worker model execution probe"
    )
    model_probe = _load_json(model_probe_path)
    workers = model_probe.get("worker_results")
    worker_rows = workers if isinstance(workers, list) else []
    ranks: set[int] = set()
    devices: set[str] = set()
    model_paths: set[str] = set()
    worker_contract_ok = True
    for worker in worker_rows:
        if not isinstance(worker, dict):
            worker_contract_ok = False
            continue
        try:
            rank = int(worker.get("logical_rank", -1))
            elapsed_ms = float(worker.get("elapsed_ms", 0.0))
        except (TypeError, ValueError):
            worker_contract_ok = False
            continue
        ranks.add(rank)
        devices.add(str(worker.get("cuda_visible_devices") or ""))
        model_paths.add(str(worker.get("model_path") or ""))
        output_tokens = worker.get("output_token_ids")
        worker_contract_ok = worker_contract_ok and bool(
            worker.get("schema_version") == 1
            and worker.get("evidence_class")
            == V100_MODEL_EXECUTION_WORKER_EVIDENCE
            and worker.get("status") == "PASS"
            and "V100" in str(worker.get("gpu_name") or "")
            and worker.get("gpu_capability") == [7, 0]
            and worker.get("model_id") == "Qwen/Qwen3-8B"
            and worker.get("model_revision") == model_revision
            and worker.get("model_config_sha256") == model_config_sha256
            and worker.get("runtime_versions") == runtime_versions
            and worker.get("model_runner_v2_requested") is True
            and isinstance(output_tokens, list)
            and output_tokens
            and all(isinstance(token, int) for token in output_tokens)
            and math.isfinite(elapsed_ms)
            and elapsed_ms > 0.0
        )
    if not (
        model_probe.get("schema_version") == 1
        and model_probe.get("evidence_class")
        == V100_MODEL_EXECUTION_PROBE_EVIDENCE
        and model_probe.get("status") == "PASS"
        and model_probe.get("failures") == []
        and model_probe.get("hardware_label") == V100_HARDWARE_LABEL
        and model_probe.get("model_id") == "Qwen/Qwen3-8B"
        and model_probe.get("model_revision") == model_revision
        and model_probe.get("model_config_sha256") == model_config_sha256
        and model_probe.get("runtime_versions") == runtime_versions
        and model_probe.get("tp_size") == 1
        and model_probe.get("pp_size") == 1
        and model_probe.get("system_dp_size") == 4
        and len(worker_rows) == 4
        and ranks == {0, 1, 2, 3}
        and len(devices) == 4
        and "" not in devices
        and len(model_paths) == 1
        and "" not in model_paths
        and model_probe.get("model_path") in model_paths
        and worker_contract_ok
        and model_probe.get("jobs_submitted") == []
        and model_probe.get("formal_ready") is False
        and model_probe.get("paper_ready") is False
    ):
        raise ValueError("V100 four-worker model execution probe did not pass")

    configuration_path = _verify_artifact(
        configuration_record, label="V100 current-design config input"
    )
    configuration_input = _load_json(configuration_path)
    conditions = configuration_input.get("four_condition_gate")
    condition_rows = conditions if isinstance(conditions, list) else []
    condition_pairs = {
        (
            str(row.get("treatment") or ""),
            str(row.get("failure_method") or ""),
        )
        for row in condition_rows
        if isinstance(row, dict)
    }
    capacity = configuration_input.get("capacity")
    aimd = configuration_input.get("aimd")
    paths = configuration_input.get("source_owned_paths")
    runtime = configuration_input.get("runtime")
    try:
        initial_alpha = float(aimd.get("initial_alpha", -1.0))
        minimum_alpha = float(aimd.get("min_alpha", -1.0))
        maximum_alpha = float(aimd.get("max_alpha", -1.0))
        decrease_factor = float(aimd.get("decrease_factor", -1.0))
        increase_step = float(aimd.get("increase_step", -1.0))
    except (AttributeError, TypeError, ValueError):
        initial_alpha = minimum_alpha = maximum_alpha = -1.0
        decrease_factor = increase_step = -1.0
    if not (
        configuration_input == configuration
        and configuration_input.get("schema_version") == 1
        and configuration_input.get("evidence_class")
        == V100_CONFIG_INPUT_EVIDENCE
        and configuration_input.get("status") == "PINNED"
        and configuration_input.get("campaign_id")
        == config_manifest.get("campaign_id")
        and configuration_input.get("design_semantics") == DESIGN_SEMANTICS
        and configuration_input.get("hardware_label") == V100_HARDWARE_LABEL
        and configuration_input.get("layout") == V100_RUNTIME_PROFILE_LAYOUT
        and configuration_input.get("hardware_profile")
        == V100_CONFIG_HARDWARE_PROFILE
        and configuration_input.get("compatibility")
        == V100_CONFIG_COMPATIBILITY_IDENTITY
        and len(condition_rows) == 4
        and condition_pairs == V100_PILOT_CONDITIONS
        and isinstance(capacity, dict)
        and int(capacity.get("peer_hbm_bytes_per_target", 0)) > 0
        and int(capacity.get("dram_bytes", 0)) > 0
        and isinstance(paths, dict)
        and all(
            paths.get(name) == owner
            for name, owner in V100_CONFIG_PATH_OWNERS.items()
        )
        and paths.get("target_owned_or_hbm_to_dram_path_allowed") is False
        and isinstance(runtime, dict)
        and runtime.get("vllm_engine") == "v1"
        and runtime.get("vllm_model_runner") == "v2"
        and runtime.get("model_runner_v2_requested") is True
        and isinstance(aimd, dict)
        and 0.0 < minimum_alpha <= initial_alpha <= maximum_alpha <= 1.0
        and 0.0 < decrease_factor < 1.0
        and increase_step > 0.0
        and int(aimd.get("stable_window_threshold", 0)) > 0
        and float(aimd.get("interval_s", 0.0)) > 0.0
        and int(aimd.get("pressure_free_bytes", 0)) > 0
    ):
        raise ValueError("V100 current-design config input did not pass")
    return {
        "campaign_id": str(config_manifest["campaign_id"]),
        "model_revision": model_revision,
        "model_config_sha256": model_config_sha256,
        "runtime_versions": dict(runtime_versions),
        "runtime_admission_decision_wire_values": dict(
            V100_RUNTIME_ADMISSION_DECISIONS
        ),
        "config_manifest": {
            "path": str(config_path),
            "sha256": str(config_record["sha256"]),
        },
        "config_verification": {
            "path": str(config_verification_path),
            "sha256": str(config_verification_record["sha256"]),
        },
        "source_activation_gate": {
            "path": str(activation_path),
            "sha256": str(activation_record["sha256"]),
        },
        "model_execution_probe": {
            "path": str(model_probe_path),
            "sha256": str(model_probe_record["sha256"]),
        },
        "configuration_input": {
            "path": str(configuration_path),
            "sha256": str(configuration_record["sha256"]),
        },
    }


def _verify_v100_pilot_runtime_gate(
    calibration: dict[str, Any],
    *,
    physical_profile_record: dict[str, Any],
    source_manifest_record: dict[str, Any],
    source_manifest: dict[str, Any],
    config_lineage: dict[str, Any],
) -> dict[str, Any]:
    gate_record = calibration.get("pilot_runtime_gate")
    if not isinstance(gate_record, dict):
        raise ValueError("V100 calibration lacks the four-condition runtime gate")
    gate_path = _verify_artifact(
        gate_record, label="V100 four-condition pilot runtime gate"
    )
    gate = _load_json(gate_path)
    checks = gate.get("checks")
    check_rows = checks if isinstance(checks, list) else []
    check_names = [
        str(row.get("name") or "")
        for row in check_rows
        if isinstance(row, dict)
    ]
    if not (
        gate.get("schema_version") == 1
        and gate.get("evidence_class") == V100_PILOT_RUNTIME_GATE_EVIDENCE
        and gate.get("status") == "PASS"
        and gate.get("failures") == []
        and gate.get("design_semantics") == DESIGN_SEMANTICS
        and gate.get("hardware_label") == V100_HARDWARE_LABEL
        and gate.get("pilot_performance_claim_authorized") is False
        and gate.get("formal_scale_submission_authorized") is False
        and gate.get("paper_ready") is False
        and len(check_names) == len(set(check_names))
        and V100_REQUIRED_RUNTIME_CHECKS.issubset(set(check_names))
        and all(
            isinstance(row, dict) and row.get("passed") is True
            for row in check_rows
        )
    ):
        raise ValueError("V100 four-condition pilot runtime gate did not pass")

    gate_source = gate.get("source_manifest")
    gate_config = gate.get("config_manifest")
    gate_config_verification = gate.get("config_verification")
    gate_profile = gate.get("physical_profile")
    if not (
        isinstance(gate_source, dict)
        and isinstance(gate_config, dict)
        and isinstance(gate_config_verification, dict)
        and isinstance(gate_profile, dict)
        and gate_source.get("sha256") == source_manifest_record.get("sha256")
        and gate_config.get("sha256")
        == config_lineage["config_manifest"]["sha256"]
        and gate_config_verification.get("sha256")
        == config_lineage["config_verification"]["sha256"]
        and gate_profile.get("sha256")
        == physical_profile_record.get("sha256")
    ):
        raise ValueError("V100 pilot gate physical input identity mismatch")
    _verify_artifact(gate_source, label="V100 pilot source manifest")
    _verify_artifact(gate_config, label="V100 pilot config manifest")
    _verify_artifact(
        gate_config_verification, label="V100 pilot config verification"
    )
    _verify_artifact(gate_profile, label="V100 pilot physical profile")
    _same_artifact_identity(
        gate_source,
        source_manifest_record,
        label="V100 pilot source manifest",
    )
    _same_artifact_identity(
        gate_config,
        config_lineage["config_manifest"],
        label="V100 pilot config manifest",
    )
    _same_artifact_identity(
        gate_config_verification,
        config_lineage["config_verification"],
        label="V100 pilot config verification",
    )
    _same_artifact_identity(
        gate_profile,
        physical_profile_record,
        label="V100 pilot physical profile",
    )

    for name in ("auditor", "profile_builder"):
        record = gate.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"V100 pilot gate lacks {name} provenance")
        _verify_artifact(record, label=f"V100 pilot {name}")

    profiles = gate.get("profiles")
    rows = profiles if isinstance(profiles, list) else []
    campaign_id = str(gate.get("campaign_id") or "")
    condition_ids: set[str] = set()
    observed_conditions: set[tuple[str, str]] = set()
    profile_artifacts: list[dict[str, Any]] = []
    if len(rows) != 4 or campaign_id != config_lineage["campaign_id"]:
        raise ValueError("V100 pilot gate does not pin four runtime profiles")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("profile"), dict):
            raise ValueError(f"V100 pilot runtime profile {index} is incomplete")
        condition_id = str(row.get("condition_id") or "")
        if not condition_id or condition_id in condition_ids:
            raise ValueError("V100 pilot runtime condition identity is invalid")
        condition_ids.add(condition_id)
        profile_record = dict(row["profile"])
        profile_path = _verify_artifact(
            profile_record, label=f"V100 pilot runtime profile {condition_id}"
        )
        profile = _load_json(profile_path)
        formal_inputs = profile.get("formal_inputs")
        input_lineage = profile.get("input_lineage")
        artifacts = profile.get("artifacts")
        if not (
            profile.get("schema_version") == 1
            and profile.get("evidence_class") == V100_RUNTIME_PROFILE_EVIDENCE
            and profile.get("status") == "PASS"
            and profile.get("failures") == []
            and profile.get("design_semantics") == DESIGN_SEMANTICS
            and profile.get("hardware_label") == V100_HARDWARE_LABEL
            and profile.get("layout") == V100_RUNTIME_PROFILE_LAYOUT
            and profile.get("campaign_id") == campaign_id
            and profile.get("campaign_condition_id") == condition_id
            and profile.get("condition_id") == condition_id
            and isinstance(input_lineage, dict)
            and isinstance(formal_inputs, dict)
            and formal_inputs.get("campaign_condition_id") == condition_id
            and formal_inputs.get("source_manifest", {}).get("sha256")
            == source_manifest_record.get("sha256")
            and formal_inputs.get("physical_profile", {}).get("sha256")
            == physical_profile_record.get("sha256")
            and isinstance(artifacts, dict)
            and V100_RUNTIME_PROFILE_ARTIFACTS.issubset(set(artifacts))
            and isinstance(artifacts.get("gate_events"), list)
        ):
            raise ValueError(
                f"V100 pilot runtime profile contract failed: {condition_id}"
            )
        lineage_records = {
            name: input_lineage.get(name)
            for name in (
                "source_manifest",
                "config_manifest",
                "config_verification",
                "physical_profile",
                "strict_run_audit",
            )
        }
        if not all(isinstance(record, dict) for record in lineage_records.values()):
            raise ValueError(
                f"V100 pilot runtime lineage is incomplete: {condition_id}"
            )
        _same_artifact_identity(
            lineage_records["source_manifest"],
            source_manifest_record,
            label=f"runtime profile {condition_id} source lineage",
        )
        _same_artifact_identity(
            lineage_records["config_manifest"],
            config_lineage["config_manifest"],
            label=f"runtime profile {condition_id} config lineage",
        )
        _same_artifact_identity(
            lineage_records["config_verification"],
            config_lineage["config_verification"],
            label=f"runtime profile {condition_id} config verification lineage",
        )
        _same_artifact_identity(
            lineage_records["physical_profile"],
            physical_profile_record,
            label=f"runtime profile {condition_id} physical lineage",
        )
        _same_artifact_identity(
            lineage_records["strict_run_audit"],
            artifacts["strict_audit"],
            label=f"runtime profile {condition_id} strict audit lineage",
        )
        _verify_artifact_tree(
            input_lineage,
            label=f"runtime profile {condition_id} input lineage",
        )
        _verify_artifact(
            formal_inputs["source_manifest"],
            label=f"runtime profile {condition_id} source manifest",
        )
        _verify_artifact(
            formal_inputs["physical_profile"],
            label=f"runtime profile {condition_id} physical profile",
        )
        _verify_artifact_tree(artifacts, label=f"runtime profile {condition_id}")
        run_config_path = _verify_artifact(
            artifacts["run_config"],
            label=f"runtime profile {condition_id} run config",
        )
        run_config = _load_json(run_config_path)
        failure = run_config.get("failure")
        failure_method = (
            str(failure.get("method") or "")
            if isinstance(failure, dict)
            else str(run_config.get("failure_method") or "")
        )
        condition = (str(run_config.get("treatment") or ""), failure_method)
        if not (
            condition == (str(profile.get("treatment") or ""), failure_method)
            and profile.get("failure_method") == failure_method
        ):
            raise ValueError(
                f"V100 pilot treatment mismatch: {condition_id}"
            )
        observed_conditions.add(condition)
        profile_artifacts.append(
            {"condition_id": condition_id, "profile": profile_record}
        )
    if observed_conditions != V100_PILOT_CONDITIONS:
        raise ValueError("V100 pilot runtime profiles have the wrong conditions")
    if str(source_manifest.get("entries_sha256") or "") == "":
        raise ValueError("V100 source manifest entries digest is missing")
    return {
        "gate": {"path": str(gate_path), "sha256": str(gate_record["sha256"])},
        "campaign_id": campaign_id,
        "checks": sorted(V100_REQUIRED_RUNTIME_CHECKS),
        "runtime_profiles": profile_artifacts,
        "physical_source_manifest_sha256": str(source_manifest_record["sha256"]),
        "config_manifest_sha256": str(
            config_lineage["config_manifest"]["sha256"]
        ),
        "config_verification_sha256": str(
            config_lineage["config_verification"]["sha256"]
        ),
        "physical_profile_sha256": str(physical_profile_record["sha256"]),
        "formal_scale_submission_authorized": False,
        "paper_ready": False,
    }


def _verify_v100_aimd_pressure_calibration(
    config: dict[str, Any],
    calibration: dict[str, Any],
    *,
    runtime_gate: dict[str, Any],
    physical_profile_record: dict[str, Any],
    source_manifest_record: dict[str, Any],
) -> dict[str, Any]:
    aimd = config.get("aimd")
    if not isinstance(aimd, dict):
        raise ValueError("V100 config lacks an AIMD pressure declaration")
    mode = str(aimd.get("pressure_mode") or "")
    calibrated = aimd.get("pressure_calibrated")
    if mode == "active_source_request_count_proxy" and calibrated is False:
        threshold = int(aimd.get("pressure_active_source_requests", 0))
        if threshold <= 0:
            raise ValueError("uncalibrated AIMD pressure proxy threshold is invalid")
        return {
            "mode": mode,
            "calibrated": False,
            "active_source_request_threshold": threshold,
            "sglsim_quantitative_aimd_pressure_authorized": False,
        }
    if mode != V100_AIMD_PRESSURE_MODE or calibrated is not True:
        raise ValueError("V100 AIMD pressure mode is neither calibrated nor disclosed")
    record = calibration.get("aimd_pressure_calibration")
    if not isinstance(record, dict):
        raise ValueError("V100 calibration lacks its AIMD pressure artifact")
    path = _verify_artifact(record, label="V100 AIMD pressure calibration")
    value = _load_json(path)
    models = value.get("pool_models")
    expected_pools = {f"hbm-vllm-dp{rank}" for rank in range(4)}
    if not (
        value.get("schema_version") == 1
        and value.get("evidence_class")
        == V100_AIMD_PRESSURE_CALIBRATION_EVIDENCE
        and value.get("status") == "PASS"
        and value.get("failures") == []
        and value.get("hardware_label") == V100_HARDWARE_LABEL
        and value.get("design_semantics") == DESIGN_SEMANTICS
        and value.get("campaign_id") == runtime_gate.get("campaign_id")
        and value.get("pressure_mode") == V100_AIMD_PRESSURE_MODE
        and value.get("pilot_runtime_gate", {}).get("sha256")
        == runtime_gate.get("gate", {}).get("sha256")
        and value.get("source_manifest", {}).get("sha256")
        == source_manifest_record.get("sha256")
        and value.get("physical_profile", {}).get("sha256")
        == physical_profile_record.get("sha256")
        and value.get("sglsim_quantitative_aimd_pressure_authorized") is True
        and value.get("formal_scale_submission_authorized") is False
        and value.get("paper_performance_claim_authorized") is False
        and value.get("paper_ready") is False
        and isinstance(models, dict)
        and set(models) == expected_pools
        and aimd.get("physical_pressure_pool_models") == models
        and isinstance(value.get("runtime_profiles"), list)
        and len(value["runtime_profiles"]) == 2
    ):
        raise ValueError("V100 AIMD pressure calibration contract did not pass")
    for name in ("pilot_runtime_gate", "source_manifest", "physical_profile", "builder"):
        artifact = value.get(name)
        if not isinstance(artifact, dict):
            raise ValueError(f"V100 AIMD pressure calibration lacks {name}")
        _verify_artifact(artifact, label=f"V100 AIMD pressure {name}")
    for index, profile in enumerate(value["runtime_profiles"]):
        artifact = profile.get("profile") if isinstance(profile, dict) else None
        if not isinstance(artifact, dict):
            raise ValueError("V100 AIMD pressure runtime profile is incomplete")
        _verify_artifact(
            artifact, label=f"V100 AIMD pressure runtime profile {index}"
        )
    for pool_id, model in models.items():
        try:
            total = int(model["cuda_total_bytes"])
            resident = int(model["non_shadow_resident_bytes"])
            free_threshold = int(model["pressure_free_bytes"])
            stall_threshold = float(model["local_stall_threshold_ms"])
            stall_probability = float(
                model["local_stall_probability_per_operation_window"]
            )
            sample_count = int(model["sample_count"])
            stable_count = int(model["stable_window_count"])
            pressure_count = int(model["pressure_window_count"])
            memory_count = int(model["memory_pressure_window_count"])
            stall_count = int(model["local_stall_window_count"])
            operation_count = int(model["local_operation_window_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"AIMD pool model is invalid: {pool_id}") from exc
        if not (
            total > 0
            and 0 <= resident < total
            and 0 < free_threshold < total
            and math.isfinite(stall_threshold)
            and stall_threshold > 0.0
            and 0.0 <= stall_probability <= 1.0
            and sample_count > 0
            and stable_count > 0
            and pressure_count > 0
            and memory_count > 0
            and stall_count > 0
            and operation_count > stall_count
        ):
            raise ValueError(f"AIMD pool model is incomplete: {pool_id}")
    return {
        "mode": mode,
        "calibrated": True,
        "artifact": {"path": str(path), "sha256": str(record["sha256"])},
        "pool_models": models,
        "sglsim_quantitative_aimd_pressure_authorized": True,
        "paper_performance_claim_authorized": False,
    }


def _verified_context_projection(config: dict[str, Any]) -> dict[str, Any]:
    projection = config.get("workload", {}).get("token_projection")
    if not isinstance(projection, dict) or projection.get("status") != "PASS":
        raise ValueError("a pinned PASS token projection is required")
    audit_record = projection.get("context_audit")
    if not isinstance(audit_record, dict):
        raise ValueError("token projection lacks its context audit")
    audit_path = _verify_artifact(audit_record, label="context projection audit")
    audit = _load_json(audit_path)
    if not (
        audit.get("status") == "PASS"
        and audit.get("design_semantics") == DESIGN_SEMANTICS
        and audit.get("performance_results_present") is not True
        and audit.get("trace", {}).get("sha256")
        == projection.get("source_trace_sha256")
    ):
        raise ValueError("context projection audit contract did not pass")
    selected_label = str(projection.get("selected_scale_label") or "")
    rows = [
        row
        for row in audit.get("scale_points", [])
        if isinstance(row, dict) and row.get("label") == selected_label
    ]
    if len(rows) != 1:
        raise ValueError("context projection scale is not uniquely preregistered")
    selected = rows[0]
    if not (
        selected.get("readiness") == "ELIGIBLE_AFTER_VALID_PHYSICAL_PROFILE"
        and selected.get("role")
        == "preregistered_primary_after_physical_profile_gate"
        and selected.get("performance_results_present") is False
        and math.isclose(
            float(selected["factor"]),
            float(projection["factor"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and int(selected["requests_above_model_context"]) == 0
        and int(selected["requests_above_physical_calibration_context"]) == 0
    ):
        raise ValueError("selected context projection is not primary-gate eligible")
    model = config.get("model")
    audit_model = audit.get("model")
    if not isinstance(model, dict) or not isinstance(audit_model, dict):
        raise ValueError("context projection model identity is incomplete")
    if not (
        model.get("revision") == "Qwen/Qwen3-8B"
        and model.get("model_snapshot_revision")
        == str(audit_model.get("revision") or "").split("@", 1)[-1]
        and int(model.get("maximum_context_tokens", 0))
        == int(audit_model.get("maximum_context_tokens", -1))
        and int(model.get("kv_bytes_per_token", 0))
        == int(audit_model.get("kv_bytes_per_token_all_layers", -1))
    ):
        raise ValueError("context projection model identity mismatch")
    return {
        **projection,
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "selected": selected,
    }


def _verify_embedded_json(record: dict[str, Any], *, label: str) -> dict[str, Any]:
    path = _verify_artifact(record, label=label)
    value = _load_json(path)
    if "value" in record and record["value"] != value:
        raise ValueError(f"{label} embedded JSON does not match its artifact")
    return value


def _verify_upstream_reference(
    record: dict[str, Any], expected_path: Path, *, label: str
) -> None:
    expected_path = expected_path.resolve()
    if record.get("status") != "PASS":
        raise ValueError(f"{label} upstream gate did not pass")
    if str(record.get("sha256", "")) != sha256_file(expected_path):
        raise ValueError(f"{label} upstream gate sha256 mismatch")
    relative_path = str(record.get("relative_path", ""))
    if not relative_path or not expected_path.as_posix().endswith(relative_path):
        raise ValueError(f"{label} upstream gate path mismatch")


def _verified_jsonl_events(
    gate: dict[str, Any], *, detail_key: str, event_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    try:
        records = gate["details"][detail_key]["files"]
    except KeyError as exc:
        raise ValueError(f"missing {detail_key} event artifacts") from exc
    if not records:
        raise ValueError(f"empty {detail_key} event artifact list")
    for index, record in enumerate(records):
        label = f"{detail_key}[{index}]"
        path = _verify_artifact(record, label=label)
        parsed: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"non-object event in {path}:{line_number}")
                parsed.append(row)
        if len(parsed) != int(record.get("event_count", -1)):
            raise ValueError(f"{label} event_count mismatch")
        replica = int(record["pipeline_replica"])
        stage = int(record["pipeline_stage"])
        selected = 0
        for row in parsed:
            if row.get("event") != event_name:
                continue
            if "pipeline_replica" in row and int(row["pipeline_replica"]) != replica:
                raise ValueError(f"{label} event replica mismatch")
            if "pipeline_stage" in row and int(row["pipeline_stage"]) != stage:
                raise ValueError(f"{label} event stage mismatch")
            events.append(
                {
                    **row,
                    "_artifact_path": str(path),
                    "_pipeline_replica": replica,
                    "_pipeline_stage": stage,
                }
            )
            selected += 1
        files.append(
            {
                "path": str(path),
                "sha256": str(record["sha256"]),
                "event_count": len(parsed),
                "selected_event_count": selected,
                "pipeline_replica": replica,
                "pipeline_stage": stage,
            }
        )
    if not events:
        raise ValueError(f"no {event_name} events found in {detail_key}")
    return events, files


def _pp_direct_dram_protection_bandwidth(
    gate: dict[str, Any], *, pp_size: int
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    events, files = _verified_jsonl_events(
        gate, detail_key="stage_aware_peer_hbm", event_name="pp_copy_completion"
    )
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in events:
        if not (
            str(row.get("chunk_id", "")).startswith("dram-")
            and row.get("committed") is True
            and row.get("coordinator_status") == "committed"
            and row.get("transfer_succeeded") is True
        ):
            raise ValueError("direct DRAM protection event is not committed")
        local_bytes = int(row.get("local_bytes", 0))
        if local_bytes <= 0 or local_bytes != int(row.get("transferred_bytes", -1)):
            raise ValueError("direct DRAM protection byte accounting mismatch")
        if float(row.get("duration_ms", 0.0)) <= 0.0:
            raise ValueError("direct DRAM protection duration is not positive")
        if any(int(value) != 0 for value in row.get("put_results", ())):
            raise ValueError("direct DRAM protection contains a failed PUT")
        key = (str(row["request_id"]), int(row["ordinal"]))
        stage = int(row["_pipeline_stage"])
        if stage in grouped[key]:
            raise ValueError(f"duplicate direct DRAM PP stage for {key}")
        grouped[key][stage] = row
    expected_stages = set(range(int(pp_size)))
    samples: list[dict[str, Any]] = []
    for (request_id, ordinal), stages in sorted(grouped.items()):
        if set(stages) != expected_stages:
            raise ValueError(
                f"incomplete direct DRAM PP group {(request_id, ordinal)}: "
                f"{sorted(stages)}"
            )
        replicas = {int(row["_pipeline_replica"]) for row in stages.values()}
        if len(replicas) != 1:
            raise ValueError("direct DRAM PP group crosses pipeline replicas")
        logical_bytes = sum(int(row["local_bytes"]) for row in stages.values())
        concurrent_duration = max(float(row["duration_ms"]) for row in stages.values())
        samples.append(
            {
                "request_id": request_id,
                "ordinal": ordinal,
                "pipeline_replica": next(iter(replicas)),
                "stages": sorted(stages),
                "logical_bytes": logical_bytes,
                "concurrent_duration_ms": concurrent_duration,
                "logical_bandwidth_bytes_per_ms": logical_bytes / concurrent_duration,
            }
        )
    bandwidth = min(row["logical_bandwidth_bytes_per_ms"] for row in samples)
    return float(bandwidth), samples, files


def _pp_hbm_recovery_bandwidth(
    gate: dict[str, Any], *, pp_size: int
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    events, files = _verified_jsonl_events(
        gate, detail_key="stage_aware_recovery", event_name="hbm_physical_load"
    )
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in events:
        if not row.get("loaded"):
            raise ValueError("HBM recovery event is not loaded")
        byte_count = int(row.get("bytes", 0))
        if byte_count <= 0 or byte_count != int(row.get("loaded_bytes", -1)):
            raise ValueError("HBM recovery byte accounting mismatch")
        if float(row.get("duration_ms", 0.0)) <= 0.0:
            raise ValueError("HBM recovery duration is not positive")
        if not row.get("load_results") or not all(row["load_results"]):
            raise ValueError("HBM recovery contains a failed physical load")
        if row.get("missing_storage_keys"):
            raise ValueError("HBM recovery is missing storage keys")
        replica = int(row["_pipeline_replica"])
        key = (str(row["request_id"]), replica)
        stage = int(row["_pipeline_stage"])
        if stage in grouped[key]:
            raise ValueError(f"duplicate HBM recovery PP stage for {key}")
        grouped[key][stage] = row
    expected_stages = set(range(int(pp_size)))
    samples: list[dict[str, Any]] = []
    for (request_id, replica), stages in sorted(grouped.items()):
        if set(stages) != expected_stages:
            raise ValueError(
                f"incomplete HBM recovery PP group {(request_id, replica)}: "
                f"{sorted(stages)}"
            )
        logical_bytes = sum(int(row["bytes"]) for row in stages.values())
        concurrent_duration = max(float(row["duration_ms"]) for row in stages.values())
        samples.append(
            {
                "request_id": request_id,
                "pipeline_replica": replica,
                "stages": sorted(stages),
                "logical_bytes": logical_bytes,
                "concurrent_duration_ms": concurrent_duration,
                "logical_bandwidth_bytes_per_ms": logical_bytes / concurrent_duration,
            }
        )
    bandwidth = min(row["logical_bandwidth_bytes_per_ms"] for row in samples)
    return float(bandwidth), samples, files


def _require_ready_calibration(config: dict[str, Any]) -> dict[str, Any]:
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("physical calibration bundle is missing")
    if calibration.get("status") != "READY":
        raise ValueError(
            "physical calibration bundle is not READY: "
            f"{calibration.get('status', 'missing')}"
        )
    return calibration


def _verify_profile_source_artifacts(profile: dict[str, Any]) -> None:
    source_artifacts = profile.get("source_artifacts")
    if not isinstance(source_artifacts, dict) or not source_artifacts:
        raise ValueError("V100 physical profile lacks source artifacts")
    records: list[tuple[str, dict[str, Any]]] = []
    for name, value in source_artifacts.items():
        if isinstance(value, dict):
            records.append((str(name), value))
        elif isinstance(value, list):
            records.extend(
                (f"{name}[{index}]", record)
                for index, record in enumerate(value)
                if isinstance(record, dict)
            )
        else:
            raise ValueError(f"V100 physical profile source artifact {name} is invalid")
    if not records:
        raise ValueError("V100 physical profile source artifact set is empty")
    for name, record in records:
        _verify_artifact(record, label=f"V100 physical profile {name}")


def _verify_source_manifest_entries(
    manifest: dict[str, Any], *, manifest_path: Path
) -> dict[str, Any]:
    entries = manifest.get("entries")
    repository_root = Path(str(manifest.get("repository_root", ""))).resolve()
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("evidence_class")
        == "aegiskv_source_and_native_binary_manifest"
        and manifest.get("status") == "PINNED"
        and manifest.get("inventory_policy") == "exact_selected_file_set"
        and repository_root.is_dir()
        and isinstance(entries, list)
        and entries
        and all(isinstance(entry, dict) for entry in entries)
        and len(entries) == int(manifest.get("entry_count", -1))
    ):
        raise ValueError("V100 source manifest structure did not pass")
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    entries_sha = hashlib.sha256(canonical).hexdigest()
    if entries_sha != str(manifest.get("entries_sha256", "")):
        raise ValueError("V100 source manifest entries digest mismatch")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        relative = str(entry.get("relative_path") or "")
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen
        ):
            raise ValueError(f"V100 source manifest entry {index} is unsafe")
        seen.add(relative)
        path = (repository_root / relative_path).resolve()
        if repository_root not in path.parents or not path.is_file():
            raise ValueError(f"V100 source manifest entry is missing: {relative}")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ValueError(f"V100 source manifest entry size mismatch: {relative}")
        if sha256_file(path) != str(entry.get("sha256") or ""):
            raise ValueError(f"V100 source manifest entry sha256 mismatch: {relative}")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "repository_root": str(repository_root),
        "entry_count": len(entries),
        "entries_sha256": entries_sha,
    }


def _v100_profile_runtime_config(
    config: dict[str, Any], calibration: dict[str, Any]
) -> tuple[AegisFullDesignConfig, dict[str, Any]]:
    if calibration.get("hardware_label") != V100_HARDWARE_LABEL:
        raise ValueError("V100 physical profile hardware label is not pinned")
    profile_record = calibration.get("physical_profile")
    manifest_record = calibration.get("source_manifest")
    verification_record = calibration.get("source_verification")
    config_manifest_record = calibration.get("config_manifest")
    config_verification_record = calibration.get("config_verification")
    if not all(
        isinstance(record, dict)
        for record in (
            profile_record,
            manifest_record,
            verification_record,
            config_manifest_record,
            config_verification_record,
        )
    ):
        raise ValueError("V100 calibration artifact records are incomplete")
    profile_path = _verify_artifact(profile_record, label="V100 physical profile")
    manifest_path = _verify_artifact(manifest_record, label="V100 source manifest")
    verification_path = _verify_artifact(
        verification_record, label="V100 source verification"
    )
    profile = _load_json(profile_path)
    manifest = _load_json(manifest_path)
    verification = _load_json(verification_path)
    live_source = _verify_source_manifest_entries(
        manifest, manifest_path=manifest_path
    )
    hardware = profile.get("hardware")
    transport = profile.get("transport_signature")
    peer_transport = transport.get("peer_hbm") if isinstance(transport, dict) else None
    dram_transport = transport.get("dram") if isinstance(transport, dict) else None
    compatibility = profile.get("compatibility_identity")
    peer_access = (
        peer_transport.get("peer_access")
        if isinstance(peer_transport, dict)
        else None
    )
    profile_manifest = profile.get("source_manifest")
    if not (
        profile.get("schema_version") == 2
        and profile.get("evidence_class") == "physical_measurement"
        and profile.get("status") == "passed"
        and profile.get("failures") == []
        and profile.get("profile_semantics") == V100_PROFILE_SEMANTICS
        and profile.get("hardware_label") == V100_HARDWARE_LABEL
        and profile.get("layout") == {"tp_size": 1, "pp_size": 1, "dp_size": 4}
        and profile.get("model") == "Qwen/Qwen3-8B"
        and compatibility == V100_COMPATIBILITY_IDENTITY
        and profile.get("runtime_consumption_contract")
        == V100_RUNTIME_CONSUMPTION_CONTRACT
        and isinstance(hardware, dict)
        and list(hardware.get("capability") or []) == [7, 0]
        and "V100" in str(hardware.get("name") or "")
        and isinstance(peer_transport, dict)
        and peer_transport.get("protocol") == "nvlink_intra"
        and peer_transport.get("physical_interconnect") == "pcie_p2p"
        and peer_transport.get("runtime_value") == "1"
        and len(str(peer_transport.get("engine_sha256") or "")) == 64
        and peer_access == [[True] * 4 for _ in range(4)]
        and isinstance(dram_transport, dict)
        and dram_transport.get("protocol") == "tcp"
        and dram_transport.get("connection_pool_enabled") is True
    ):
        raise ValueError("V100 physical profile contract did not pass")
    config_model = config.get("model")
    config_layout = config.get("cluster")
    calibration_compatibility = calibration.get("compatibility_identity")
    if not (
        isinstance(config_model, dict)
        and config_layout == profile.get("layout")
        and calibration.get("source_layout") == profile.get("layout")
        and calibration.get("simulation_layout") == profile.get("layout")
        and calibration.get("layout_extrapolation") is False
        and calibration_compatibility == V100_COMPATIBILITY_IDENTITY
        and config_model.get("revision") == profile.get("model")
        and config_model.get("model_snapshot_revision")
        == profile.get("model_revision")
        and config_model.get("serving_dtype")
        == compatibility["serving_dtype"]
        and config_model.get("kv_cache_dtype")
        == compatibility["kv_cache_dtype"]
        and config_model.get("kv_abi") == compatibility["kv_abi"]
        and int(config_model.get("block_size", 0))
        == int(compatibility["block_size"])
        and 0
        < int(calibration.get("maximum_calibrated_context_tokens", 0))
        <= int(compatibility["max_model_len"])
    ):
        raise ValueError("V100 config/profile compatibility identity mismatch")
    if not (
        manifest.get("status") == "PINNED"
        and int(manifest.get("entry_count", 0)) > 0
        and isinstance(manifest.get("entries"), list)
        and len(manifest["entries"]) == int(manifest["entry_count"])
        and verification.get("status") == "PASS"
        and verification.get("schema_version") == 1
        and verification.get("failures") == []
        and verification.get("evidence_class")
        == "aegiskv_source_manifest_verification"
        and verification.get("manifest", {}).get("sha256")
        == manifest_record["sha256"]
        and verification.get("entries_sha256") == manifest.get("entries_sha256")
        and int(verification.get("entry_count", -1)) == int(manifest["entry_count"])
        and int(verification.get("checked_entry_count", -1))
        == int(manifest["entry_count"])
        and verification.get("inventory_policy")
        == manifest.get("inventory_policy")
        and verification.get("repository_root")
        == manifest.get("repository_root")
        and isinstance(profile_manifest, dict)
        and profile_manifest.get("sha256") == manifest_record["sha256"]
        and profile_manifest.get("entries_sha256") == manifest.get("entries_sha256")
    ):
        raise ValueError("V100 source manifest chain did not pass")
    config_lineage = _verify_v100_config_lineage(
        calibration,
        profile,
        source_manifest_record=manifest_record,
        source_verification_record=verification_record,
        source_manifest=manifest,
    )
    runtime_gate = _verify_v100_pilot_runtime_gate(
        calibration,
        physical_profile_record=profile_record,
        source_manifest_record=manifest_record,
        source_manifest=manifest,
        config_lineage=config_lineage,
    )
    aimd_pressure = _verify_v100_aimd_pressure_calibration(
        config,
        calibration,
        runtime_gate=runtime_gate,
        physical_profile_record=profile_record,
        source_manifest_record=manifest_record,
    )
    _verify_profile_source_artifacts(profile)
    metrics = profile.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(V100_PROFILE_METRICS):
        raise ValueError("V100 physical profile lacks metrics")
    normalized_metrics: dict[str, float] = {}
    for name in V100_PROFILE_METRICS:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V100 physical profile metric {name} is invalid") from exc
        if not math.isfinite(value) or (
            value < 0.0
            if name == "recompute_fixed_overhead_ms"
            else value <= 0.0
        ):
            raise ValueError(f"V100 physical profile metric {name} is invalid")
        normalized_metrics[name] = value
    kv_bytes = int(config["model"]["kv_bytes_per_token"])
    dp_size = int(config["cluster"]["dp_size"])
    setup_values: dict[str, float] = {}
    for name in (
        "peer_hbm_protection_setup_ms",
        "direct_dram_protection_setup_ms",
        "hbm_recovery_setup_ms",
        "dram_recovery_setup_ms",
    ):
        try:
            setup = float(calibration[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V100 calibration setup {name} is invalid") from exc
        if not math.isfinite(setup) or setup < 0.0:
            raise ValueError(f"V100 calibration setup {name} is invalid")
        setup_values[name] = setup
    runtime = AegisFullDesignConfig(
        hbm_capacity_bytes=int(config["capacity"]["hbm_bytes_per_target"]),
        dram_capacity_bytes=int(config["capacity"]["dram_bytes_total"]) // dp_size,
        kv_bytes_per_token=kv_bytes,
        hbm_copy=CalibratedTransferPath(
            setup_ms=setup_values["peer_hbm_protection_setup_ms"],
            bandwidth_bytes_per_ms=normalized_metrics[
                "peer_hbm_protection_effective_bytes_per_ms"
            ],
            provenance=str(profile_path),
        ),
        dram_copy=CalibratedTransferPath(
            setup_ms=setup_values["direct_dram_protection_setup_ms"],
            bandwidth_bytes_per_ms=normalized_metrics[
                "direct_dram_protection_effective_bytes_per_ms"
            ],
            provenance=str(profile_path),
        ),
        recompute=RecoveryTiming(
            setup_ms=normalized_metrics["recompute_fixed_overhead_ms"],
            ms_per_token=normalized_metrics["recompute_ms_per_token"],
        ),
        hbm_load=RecoveryTiming(
            setup_ms=setup_values["hbm_recovery_setup_ms"],
            ms_per_token=kv_bytes
            / normalized_metrics["peer_hbm_recovery_effective_bytes_per_ms"],
        ),
        dram_load=RecoveryTiming(
            setup_ms=setup_values["dram_recovery_setup_ms"],
            ms_per_token=kv_bytes
            / normalized_metrics["dram_recovery_effective_bytes_per_ms"],
        ),
        prefix_chunk_tokens=int(config["chunking"]["prefix_tokens"]),
        decode_chunk_tokens=int(config["chunking"]["decode_tokens"]),
        pp_size=int(config["cluster"]["pp_size"]),
        tp_size=int(config["cluster"]["tp_size"]),
        initial_hbm_alpha=float(config["aimd"]["initial_alpha"]),
        minimum_hbm_alpha=float(config["aimd"]["minimum_alpha"]),
        maximum_hbm_alpha=float(config["aimd"]["maximum_alpha"]),
        aimd_decrease_factor=float(config["aimd"]["decrease_factor"]),
        aimd_increase_step=float(config["aimd"]["increase_step"]),
        aimd_stable_windows=int(config["aimd"]["stable_windows"]),
        random_salt=str(config["experiment_id"]),
        event_log_limit=int(config["audit"]["event_log_limit"]),
    )
    provenance = {
        "calibration_semantics": V100_PROFILE_SEMANTICS,
        "result_label": V100_HARDWARE_LABEL,
        "physical_profile": {
            "path": str(profile_path),
            "sha256": str(profile_record["sha256"]),
        },
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": str(manifest_record["sha256"]),
            "entries_sha256": str(manifest["entries_sha256"]),
            "live_source_verification": live_source,
        },
        "source_verification": {
            "path": str(verification_path),
            "sha256": str(verification_record["sha256"]),
        },
        "config_manifest": dict(config_lineage["config_manifest"]),
        "config_verification": dict(config_lineage["config_verification"]),
        "source_activation_gate": dict(
            config_lineage["source_activation_gate"]
        ),
        "model_execution_probe": dict(
            config_lineage["model_execution_probe"]
        ),
        "configuration_input": dict(config_lineage["configuration_input"]),
        "model_config_sha256": config_lineage["model_config_sha256"],
        "runtime_versions": dict(config_lineage["runtime_versions"]),
        "runtime_admission_decision_wire_values": dict(
            config_lineage["runtime_admission_decision_wire_values"]
        ),
        "pilot_runtime_gate": runtime_gate,
        "aimd_pressure": aimd_pressure,
        "source_layout": profile["layout"],
        "simulation_layout": dict(config["cluster"]),
        "compatibility_identity": dict(compatibility),
        "runtime_consumption_contract": dict(
            profile["runtime_consumption_contract"]
        ),
        "layout_extrapolation_disclosed": False,
        "metrics": normalized_metrics,
    }
    return runtime, provenance


def _legacy_calibrated_runtime_config_diagnostic_only(
    config: dict[str, Any]
) -> tuple[AegisFullDesignConfig, dict[str, Any]]:
    """Retained only so historical invalidated artifacts remain inspectable.

    Formal callers use :func:`calibrated_runtime_config`, which rejects every
    pre-schema-v2 bundle before this diagnostic parser can be reached.
    """

    calibration = _require_ready_calibration(config)
    if calibration.get("bundle_semantics") == V100_PROFILE_SEMANTICS:
        return _v100_profile_runtime_config(config, calibration)
    peer_path = Path(calibration["peer_hbm_protection_gate"]).resolve()
    direct_dram_path = Path(calibration["direct_dram_protection_gate"]).resolve()
    hbm_recovery_path = Path(calibration["hbm_recovery_gate"]).resolve()
    dram_recovery_path = Path(calibration["dram_recovery_gate"]).resolve()
    recompute_path = Path(calibration["recompute_gate"])
    peer = _validate_gate(
        peer_path, "aegiskv_4090_pp2_peer_hbm_calibration_gate"
    )
    direct_dram = _validate_gate(
        direct_dram_path, "aegiskv_4090_pp2_hbm_integration_gate"
    )
    hbm_recovery = _validate_gate(
        hbm_recovery_path, "aegiskv_4090_pp2_hbm_recovery_gate"
    )
    dram_recovery = _validate_gate(
        dram_recovery_path, "aegiskv_4090_pp_mooncake_dram_calibration_gate"
    )
    recompute = _validate_gate(
        recompute_path, "aegiskv_4090_pp_recompute_calibration_gate"
    )
    gates = (peer, direct_dram, hbm_recovery, dram_recovery, recompute)
    if any(not gate.get("result_label") for gate in gates):
        raise ValueError("four-path calibration bundle is missing a result label")
    result_labels = {str(gate["result_label"]) for gate in gates}
    if len(result_labels) != 1:
        raise ValueError("four-path calibration bundle result labels do not match")
    result_label = next(iter(result_labels))
    if result_label != str(calibration["bundle_result_label"]):
        raise ValueError("four-path calibration bundle label is not pinned by config")
    if calibration.get("bundle_semantics") != "four_independent_physical_paths_v1":
        raise ValueError("four-path calibration semantics are not pinned by config")
    pp_size = int(config["cluster"]["pp_size"])
    peer_protection_bw = float(
        peer["details"]["physical_calibration"][
            "conservative_logical_bandwidth_bytes_per_ms"
        ]
    )
    direct_dram_protection_bw, direct_dram_samples, direct_dram_files = (
        _pp_direct_dram_protection_bandwidth(direct_dram, pp_size=pp_size)
    )
    hbm_recovery_bw, hbm_recovery_samples, hbm_recovery_files = (
        _pp_hbm_recovery_bandwidth(hbm_recovery, pp_size=pp_size)
    )
    dram_recovery_gbps = float(
        dram_recovery["details"]["mooncake_dram_calibration"]["summary"][
            "aggregate_effective_load_gb_per_s"
        ]
    )
    dram_recovery_bw = dram_recovery_gbps * 1_000_000.0
    for name, value in (
        ("Peer-HBM protection", peer_protection_bw),
        ("direct DRAM protection", direct_dram_protection_bw),
        ("HBM recovery", hbm_recovery_bw),
        ("DRAM recovery", dram_recovery_bw),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} bandwidth is not positive and finite")

    direct_manifest = direct_dram["details"]["source_config_manifest"]
    hbm_recovery_manifest = hbm_recovery["details"]["source_config_manifest"]
    _verify_embedded_json(direct_manifest, label="direct DRAM source manifest")
    _verify_embedded_json(hbm_recovery_manifest, label="HBM recovery source manifest")
    _verify_embedded_json(
        direct_dram["details"]["deployment"], label="direct DRAM deployment"
    )
    _verify_embedded_json(
        hbm_recovery["details"]["deployment"], label="HBM recovery deployment"
    )
    upstream_paths = {
        "upstream_peer_hbm_gate": peer_path,
        "upstream_mooncake_dram_gate": dram_recovery_path,
        "upstream_recompute_gate": recompute_path.resolve(),
    }
    for key, expected_path in upstream_paths.items():
        _verify_upstream_reference(
            direct_manifest[key], expected_path, label=f"direct DRAM {key}"
        )
        _verify_upstream_reference(
            hbm_recovery_manifest[key], expected_path, label=f"HBM recovery {key}"
        )
    _verify_upstream_reference(
        hbm_recovery_manifest["upstream_hbm_integration_gate"],
        direct_dram_path,
        label="HBM recovery upstream integration",
    )
    recompute_fit = recompute["details"]["recompute_calibration"]["pooled_fit"]
    kv_bytes = int(config["model"]["kv_bytes_per_token"])
    dp_size = int(config["cluster"]["dp_size"])
    runtime = AegisFullDesignConfig(
        hbm_capacity_bytes=int(config["capacity"]["hbm_bytes_per_target"]),
        dram_capacity_bytes=int(config["capacity"]["dram_bytes_total"]) // dp_size,
        kv_bytes_per_token=kv_bytes,
        hbm_copy=CalibratedTransferPath(
            setup_ms=float(calibration["peer_hbm_protection_setup_ms"]),
            bandwidth_bytes_per_ms=peer_protection_bw,
            provenance=str(peer_path),
        ),
        dram_copy=CalibratedTransferPath(
            setup_ms=float(calibration["direct_dram_protection_setup_ms"]),
            bandwidth_bytes_per_ms=direct_dram_protection_bw,
            provenance=str(direct_dram_path),
        ),
        recompute=RecoveryTiming(
            setup_ms=float(recompute_fit["fixed_overhead_ms"]),
            ms_per_token=float(recompute_fit["recompute_ms_per_token"]),
        ),
        hbm_load=RecoveryTiming(
            setup_ms=float(calibration["hbm_recovery_setup_ms"]),
            ms_per_token=kv_bytes / hbm_recovery_bw,
        ),
        dram_load=RecoveryTiming(
            setup_ms=float(calibration["dram_recovery_setup_ms"]),
            ms_per_token=kv_bytes / dram_recovery_bw,
        ),
        prefix_chunk_tokens=int(config["chunking"]["prefix_tokens"]),
        decode_chunk_tokens=int(config["chunking"]["decode_tokens"]),
        pp_size=pp_size,
        tp_size=int(config["cluster"]["tp_size"]),
        initial_hbm_alpha=float(config["aimd"]["initial_alpha"]),
        minimum_hbm_alpha=float(config["aimd"]["minimum_alpha"]),
        maximum_hbm_alpha=float(config["aimd"]["maximum_alpha"]),
        aimd_decrease_factor=float(config["aimd"]["decrease_factor"]),
        aimd_increase_step=float(config["aimd"]["increase_step"]),
        aimd_stable_windows=int(config["aimd"]["stable_windows"]),
        random_salt=str(config["experiment_id"]),
        event_log_limit=int(config["audit"]["event_log_limit"]),
    )
    provenance = {
        "calibration_semantics": "four_independent_physical_paths_v1",
        "result_label": result_label,
        "paths": {
            "peer_hbm_protection": {
                "path": str(peer_path),
                "sha256": sha256_file(peer_path),
                "paper_ready": bool(peer.get("paper_ready")),
                "evidence_boundary": peer["evidence_boundary"],
                "bandwidth_bytes_per_ms": peer_protection_bw,
            },
            "direct_dram_protection": {
                "path": str(direct_dram_path),
                "sha256": sha256_file(direct_dram_path),
                "paper_ready": bool(direct_dram.get("paper_ready")),
                "evidence_boundary": direct_dram["evidence_boundary"],
                "bandwidth_bytes_per_ms": direct_dram_protection_bw,
                "aggregation": "sum PP-stage bytes / max concurrent stage duration",
                "samples": direct_dram_samples,
                "event_artifacts": direct_dram_files,
            },
            "hbm_recovery": {
                "path": str(hbm_recovery_path),
                "sha256": sha256_file(hbm_recovery_path),
                "paper_ready": bool(hbm_recovery.get("paper_ready")),
                "evidence_boundary": hbm_recovery["evidence_boundary"],
                "bandwidth_bytes_per_ms": hbm_recovery_bw,
                "aggregation": "sum PP-stage bytes / max concurrent stage duration",
                "samples": hbm_recovery_samples,
                "event_artifacts": hbm_recovery_files,
            },
            "dram_recovery": {
                "path": str(dram_recovery_path),
                "sha256": sha256_file(dram_recovery_path),
                "paper_ready": bool(dram_recovery.get("paper_ready")),
                "evidence_boundary": dram_recovery["evidence_boundary"],
                "bandwidth_bytes_per_ms": dram_recovery_bw,
            },
        },
        "recompute_gate": {
            "path": str(recompute_path),
            "sha256": sha256_file(recompute_path),
            "paper_ready": bool(recompute.get("paper_ready")),
            "evidence_boundary": recompute["evidence_boundary"],
            "fit": recompute_fit,
        },
        "projection": (
            "Four distinct Qwen3-8B PP2 DP2 physical path coefficients are "
            "projected to a metadata-only DP8 workload; configured setup "
            "intercepts are zero and contexts beyond 4096 tokens are "
            "extrapolations."
        ),
    }
    return runtime, provenance


def calibrated_runtime_config(
    config: dict[str, Any]
) -> tuple[AegisFullDesignConfig, dict[str, Any]]:
    """Consume only the current hash-pinned V100 schema-v2 profile."""

    calibration = _require_ready_calibration(config)
    semantics = calibration.get("bundle_semantics")
    if semantics != V100_PROFILE_SEMANTICS:
        raise ValueError(
            "legacy physical calibration bundles are not accepted by formal "
            f"replay: {semantics!r}"
        )
    return _v100_profile_runtime_config(config, calibration)


def _same_artifact_identity(
    left: dict[str, Any], right: dict[str, Any], *, label: str
) -> None:
    left_path = Path(str(left.get("path", ""))).resolve()
    right_path = Path(str(right.get("path", ""))).resolve()
    if not (
        left_path == right_path
        and str(left.get("sha256", "")) == str(right.get("sha256", ""))
    ):
        raise ValueError(f"{label} artifact identity mismatch")


def preflight_v100_config(
    config_path: Path, import_manifest_path: Path | None
) -> tuple[
    dict[str, Any],
    AegisFullDesignConfig,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate every formal input before trace ingestion or output creation."""

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = _load_json(config_path)
    if config.get("design_semantics") != DESIGN_SEMANTICS:
        raise ValueError("experiment config is not pinned to the current design")
    runtime, provenance = calibrated_runtime_config(config)
    projection = _verified_context_projection(config)
    if import_manifest_path is None:
        raise ValueError("a hash-bound V100 import manifest is required")
    import_manifest_path = import_manifest_path.resolve()
    if not import_manifest_path.is_file():
        raise FileNotFoundError(import_manifest_path)
    import_manifest = _load_json(import_manifest_path)
    if not (
        import_manifest.get("schema_version") == 1
        and import_manifest.get("evidence_class")
        == "sglsim_v100_physical_profile_import"
        and import_manifest.get("status") == "PASS"
        and import_manifest.get("design_semantics") == DESIGN_SEMANTICS
        and import_manifest.get("hardware_label") == V100_HARDWARE_LABEL
        and import_manifest.get("runtime_consumption_contract")
        == V100_RUNTIME_CONSUMPTION_CONTRACT
        and import_manifest.get("paper_ready") is False
        and import_manifest.get("formal_ready") is False
    ):
        raise ValueError("V100 import manifest contract did not pass")
    output_config = import_manifest.get("output_config")
    if not isinstance(output_config, dict):
        raise ValueError("V100 import manifest lacks its output config binding")
    verified_config_path = _verify_artifact(
        output_config, label="V100 imported experiment config"
    )
    if verified_config_path != config_path:
        raise ValueError("V100 import manifest is bound to a different config")
    template_record = import_manifest.get("template")
    if not isinstance(template_record, dict):
        raise ValueError("V100 import manifest lacks its template provenance")
    _verify_artifact(template_record, label="V100 import template")

    calibration = config["calibration"]
    artifact_pairs = (
        ("physical_profile", "physical profile"),
        ("source_manifest", "source manifest"),
        ("source_verification", "source verification"),
        ("config_manifest", "config manifest"),
        ("config_verification", "config verification"),
        ("pilot_runtime_gate", "pilot runtime gate"),
    )
    for key, label in artifact_pairs:
        manifest_record = import_manifest.get(key)
        calibration_record = calibration.get(key)
        if not isinstance(manifest_record, dict) or not isinstance(
            calibration_record, dict
        ):
            raise ValueError(f"V100 {label} binding is incomplete")
        _verify_artifact(manifest_record, label=f"V100 import {label}")
        _same_artifact_identity(
            manifest_record, calibration_record, label=f"V100 {label}"
        )
    context_record = import_manifest.get("context_projection_audit")
    config_context_record = config["workload"]["token_projection"].get(
        "context_audit"
    )
    if not isinstance(context_record, dict) or not isinstance(
        config_context_record, dict
    ):
        raise ValueError("V100 context projection binding is incomplete")
    _verify_artifact(context_record, label="V100 import context projection audit")
    _same_artifact_identity(
        context_record,
        config_context_record,
        label="V100 context projection audit",
    )
    manifest_provenance = import_manifest.get("calibration_provenance")
    if not isinstance(manifest_provenance, dict):
        raise ValueError("V100 import calibration provenance is missing")
    if not (
        manifest_provenance.get("calibration_semantics")
        == V100_PROFILE_SEMANTICS
        and manifest_provenance.get("result_label") == V100_HARDWARE_LABEL
        and manifest_provenance.get("runtime_consumption_contract")
        == V100_RUNTIME_CONSUMPTION_CONTRACT
        and manifest_provenance.get("physical_profile", {}).get("sha256")
        == calibration["physical_profile"]["sha256"]
        and manifest_provenance.get("source_manifest", {}).get("sha256")
        == calibration["source_manifest"]["sha256"]
        and manifest_provenance.get("config_manifest", {}).get("sha256")
        == calibration["config_manifest"]["sha256"]
        and manifest_provenance.get("config_verification", {}).get("sha256")
        == calibration["config_verification"]["sha256"]
        and provenance.get("physical_profile", {}).get("sha256")
        == calibration["physical_profile"]["sha256"]
        and provenance.get("source_manifest", {}).get("sha256")
        == calibration["source_manifest"]["sha256"]
        and provenance.get("config_manifest", {}).get("sha256")
        == calibration["config_manifest"]["sha256"]
        and provenance.get("config_verification", {}).get("sha256")
        == calibration["config_verification"]["sha256"]
    ):
        raise ValueError("V100 import source/config/profile SHA chain mismatch")
    runtime_manifest = import_manifest.get("runtime_config")
    expected_runtime = {
        "hbm_copy_bytes_per_ms": runtime.hbm_copy.bandwidth_bytes_per_ms,
        "dram_copy_bytes_per_ms": runtime.dram_copy.bandwidth_bytes_per_ms,
        "hbm_load_ms_per_token": runtime.hbm_load.ms_per_token,
        "dram_load_ms_per_token": runtime.dram_load.ms_per_token,
        "recompute_ms_per_token": runtime.recompute.ms_per_token,
        "recompute_fixed_overhead_ms": runtime.recompute.setup_ms,
    }
    if runtime_manifest != expected_runtime:
        raise ValueError("V100 import runtime config binding mismatch")
    alignment = import_manifest.get("runtime_alignment")
    aimd_manifest = import_manifest.get("aimd_pressure_model")
    expected_aimd = provenance["aimd_pressure"]
    if not (
        isinstance(alignment, dict)
        and alignment.get("four_condition_physical_gate_passed") is True
        and alignment.get(
            "source_owned_hbm_dram_paths_independently_verified"
        )
        is True
        and alignment.get("hbm_dram_timing_overlap_required") is False
        and alignment.get(
            "config_manifest_and_verification_lineage_verified"
        )
        is True
        and alignment.get(
            "runtime_admission_decision_wire_values_verified"
        )
        is True
        and provenance.get("runtime_admission_decision_wire_values")
        == V100_RUNTIME_ADMISSION_DECISIONS
        and alignment.get("four_gpu_clean_telemetry_verified") is True
        and alignment.get("aimd_alpha_response_and_direct_reclaim_exercised")
        is True
        and alignment.get(
            "independent_peer_hbm_and_dram_recovery_exercised"
        )
        is True
        and alignment.get("all_four_independent_residency_outcomes_exercised")
        is True
        and alignment.get(
            "hbm_only_direct_reclaim_without_dram_prerequisite_exercised"
        )
        is True
        and alignment.get("pilot_campaign_id")
        == provenance["pilot_runtime_gate"]["campaign_id"]
        and alignment.get("quantitative_aimd_simulation_authorized")
        is bool(expected_aimd["sglsim_quantitative_aimd_pressure_authorized"])
        and alignment.get("quantitative_aimd_claim_authorized") is False
        and alignment.get("paper_performance_claim_authorized") is False
        and isinstance(aimd_manifest, dict)
        and aimd_manifest.get("mode") == expected_aimd["mode"]
        and aimd_manifest.get("calibrated") is expected_aimd["calibrated"]
        and aimd_manifest.get("artifact") == expected_aimd.get("artifact")
        and aimd_manifest.get("quantitative_aimd_simulation_authorized")
        is bool(expected_aimd["sglsim_quantitative_aimd_pressure_authorized"])
        and aimd_manifest.get("quantitative_aimd_claim_authorized") is False
        and import_manifest.get("selected_context_scale")
        == projection["selected"]
    ):
        raise ValueError("V100 import runtime consumption evidence mismatch")
    binding = {
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "import_manifest": {
            "path": str(import_manifest_path),
            "sha256": sha256_file(import_manifest_path),
        },
        "physical_profile_sha256": calibration["physical_profile"]["sha256"],
        "source_manifest_sha256": calibration["source_manifest"]["sha256"],
        "config_manifest_sha256": calibration["config_manifest"]["sha256"],
        "config_verification_sha256": calibration["config_verification"][
            "sha256"
        ],
        "hardware_label": V100_HARDWARE_LABEL,
        "layout": dict(config["cluster"]),
    }
    return config, runtime, provenance, projection, binding


def _derived_full_output_dir(
    requested: Path | None, *, profile_sha256: str, limit_workflows: int | None
) -> Path:
    prefix = (
        "full_393_workflows"
        if limit_workflows is None
        else f"smoke_{int(limit_workflows)}_workflows"
    )
    expected_name = f"{prefix}_v100_{profile_sha256[:12]}"
    output = (
        requested.resolve()
        if requested is not None
        else (EXPERIMENT_ROOT / "results" / expected_name).resolve()
    )
    if output.name != expected_name:
        raise ValueError(
            "formal output directory must be derived from the physical profile "
            f"hash and named {expected_name!r}"
        )
    if output.exists():
        raise FileExistsError(f"formal replay output is immutable: {output}")
    return output


def _ticket_value(ticket: CausalSlackTicket, field_name: str) -> float:
    value = getattr(ticket, str(field_name))
    return max(float(value), 0.0)


def remaining_ticket_slack(
    meta: ReplayRequest,
    ticket: CausalSlackTicket,
    at_ms: float,
    *,
    field_name: str,
) -> float:
    published_global = (
        meta.workflow_base_ms
        + ticket.published_at_ms
        - meta.workflow_origin_ms
    )
    return max(
        _ticket_value(ticket, field_name)
        - max(float(at_ms) - published_global, 0.0),
        0.0,
    )


def _fault_times(layout: ReplayLayout, config: dict[str, Any]) -> dict[str, Any]:
    starts = sorted(row.global_start_ms for row in layout.requests)
    quantile = float(config["faults"]["global_cut_request_start_quantile"])
    cut = starts[min(int(round(quantile * (len(starts) - 1))), len(starts) - 1)]
    gap = float(config["faults"]["target_to_source_gap_ms"])
    skew = float(config["faults"]["rank_skew_ms"])
    dp_size = int(config["cluster"]["dp_size"])
    return {
        "cut_ms": cut,
        "target": {rank: cut + rank * skew for rank in range(dp_size)},
        "source": {
            rank: cut + gap + rank * skew for rank in range(dp_size)
        },
    }


def _recompute_decision(
    meta: ReplayRequest,
    produced_tokens: int,
    remaining_slack_ms: float,
    timing: RecoveryTiming,
) -> RecoveryDecision:
    recovery = (
        timing.setup_ms + int(produced_tokens) * timing.ms_per_token
        if produced_tokens > 0
        else 0.0
    )
    return RecoveryDecision(
        request_id=meta.row.rid,
        action="RECOMPUTE",
        restored_tokens=0,
        recompute_tokens=int(produced_tokens),
        recovery_ms=recovery,
        slack_ms=remaining_slack_ms,
        visible_harm_ms=max(recovery - remaining_slack_ms, 0.0),
        route_version=1,
        activated_target=meta.current_target,
    )


def evaluate_failure_snapshot(
    *,
    event_id: str,
    failure_time_ms: float,
    affected: Sequence[ReplayRequest],
    decisions: dict[str, dict[str, RecoveryDecision]],
    remaining_slack: dict[str, float],
    stage_branch_finishes: dict[tuple[str, int], dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failure_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    meta_by_id = {row.row.rid: row for row in affected}
    for policy, protection, ordering in POLICY_SPECS:
        by_target: dict[str, list[ReplayRequest]] = defaultdict(list)
        for meta in affected:
            by_target[meta.current_target].append(meta)
        queue_delay: dict[str, float] = {}
        for target, requests in sorted(by_target.items()):
            if ordering == "slack_due":
                ordered = sorted(
                    requests,
                    key=lambda row: (
                        remaining_slack[row.row.rid],
                        -decisions[protection][row.row.rid].visible_harm_ms,
                        row.row.rid,
                    ),
                )
            else:
                ordered = sorted(
                    requests,
                    key=lambda row: (row.global_start_ms, row.row.rid),
                )
            cumulative = 0.0
            for queue_position, meta in enumerate(ordered):
                decision = decisions[protection][meta.row.rid]
                queue_start = cumulative
                cumulative += decision.recovery_ms
                queue_delay[meta.row.rid] = cumulative
                actual_slack = max(
                    meta.stage_finish_ms - meta.branch_finish_ms, 0.0
                )
                failure_rows.append(
                    {
                        "event_id": event_id,
                        "failure_time_ms": failure_time_ms,
                        "policy": policy,
                        "protection_state": protection,
                        "ordering": ordering,
                        "request_id": meta.row.rid,
                        "workflow_id": meta.row.workflow_id,
                        "stage_id": meta.row.stage_id,
                        "agent_id": meta.row.agent_id,
                        "source_worker": meta.source_worker,
                        "activated_target": target,
                        "queue_position": queue_position,
                        "queue_start_ms": queue_start,
                        "queue_completion_delay_ms": cumulative,
                        "action": decision.action,
                        "produced_tokens": decision.restored_tokens
                        + decision.recompute_tokens,
                        "restored_tokens": decision.restored_tokens,
                        "recompute_tokens": decision.recompute_tokens,
                        "recovery_service_ms": decision.recovery_ms,
                        "predicted_remaining_slack_ms": remaining_slack[
                            meta.row.rid
                        ],
                        "predicted_visible_harm_ms": max(
                            cumulative - remaining_slack[meta.row.rid], 0.0
                        ),
                        "actual_join_slack_ms": actual_slack,
                        "actual_slack_visible_harm_ms": max(
                            cumulative - actual_slack, 0.0
                        ),
                        "target_rotated_before_source": meta.rotated_before_source,
                        "counterfactual_non_destructive": True,
                        "execution_verified": False,
                        "fence_source": True,
                        "output_frozen": True,
                        "silent_replay": True,
                        "all_rank_barrier": True,
                        "prior_route_version": 1,
                        "next_route_version": 2,
                    }
                )

        by_stage: dict[tuple[str, int], list[ReplayRequest]] = defaultdict(list)
        for meta in affected:
            by_stage[(meta.row.workflow_id, meta.row.stage_id)].append(meta)
        for stage_key, rows in sorted(by_stage.items()):
            base_finishes = stage_branch_finishes[stage_key]
            base_stage_finish = max(base_finishes.values())
            delay_by_agent: dict[str, float] = defaultdict(float)
            for meta in rows:
                delay_by_agent[meta.row.agent_id] = max(
                    delay_by_agent[meta.row.agent_id], queue_delay[meta.row.rid]
                )
            new_finish = max(
                finish + delay_by_agent.get(agent_id, 0.0)
                for agent_id, finish in base_finishes.items()
            )
            stage_rows.append(
                {
                    "event_id": event_id,
                    "policy": policy,
                    "workflow_id": stage_key[0],
                    "stage_id": stage_key[1],
                    "affected_requests": len(rows),
                    "affected_agents": len(delay_by_agent),
                    "base_stage_finish_ms": base_stage_finish,
                    "modeled_stage_finish_ms": new_finish,
                    "join_barrier_harm_ms": max(new_finish - base_stage_finish, 0.0),
                }
            )
    return failure_rows, stage_rows


def _runtime_grid(
    runtime_config: AegisFullDesignConfig, dp_size: int
) -> dict[str, dict[int, AegisFullDesignRuntime]]:
    return {
        policy: {
            rank: AegisFullDesignRuntime(
                replace(
                    runtime_config,
                    random_salt=f"{runtime_config.random_salt}:{policy}:{rank}",
                ),
                target_id=f"target-{rank}-g0",
                dram_pool_id=f"store-dram-partition-{rank}",
                dram_failure_domain=f"store-domain-{rank}",
            )
            for rank in range(dp_size)
        }
        for policy in PLACEMENT_POLICIES
    }


def _physical_aimd_pressure(
    runtime: AegisFullDesignRuntime,
    model: Mapping[str, Any],
    *,
    time_ms: float,
    local_stall_draw: float,
) -> dict[str, Any]:
    """Project CUDA headroom OR calibrated local allocation/reclaim stall."""

    local_operations = runtime.sample_hbm_local_resource_operations(time_ms=time_ms)
    snapshot = runtime.snapshot(include_controller=True)
    hbm_used = int(snapshot["controller"]["pools"]["HBM"]["used_bytes"])
    total = int(model["cuda_total_bytes"])
    non_shadow = int(model["non_shadow_resident_bytes"])
    free_threshold = int(model["pressure_free_bytes"])
    stall_threshold = float(model["local_stall_threshold_ms"])
    stall_probability = float(
        model["local_stall_probability_per_operation_window"]
    )
    if not (
        total > 0
        and 0 <= non_shadow < total
        and 0 < free_threshold < total
        and math.isfinite(stall_threshold)
        and stall_threshold > 0.0
        and 0.0 <= stall_probability <= 1.0
        and math.isfinite(float(local_stall_draw))
        and 0.0 <= float(local_stall_draw) < 1.0
    ):
        raise ValueError("physical AIMD pool model is invalid")
    projected_free = max(total - non_shadow - hbm_used, 0)
    memory_pressure = projected_free < free_threshold
    local_stall_pressure = bool(
        int(local_operations["operation_count"]) > 0
        and float(local_stall_draw) < stall_probability
    )
    return {
        "pressure": memory_pressure or local_stall_pressure,
        "memory_pressure": memory_pressure,
        "local_stall_pressure": local_stall_pressure,
        "projected_cuda_free_bytes": projected_free,
        "protected_hbm_used_bytes": hbm_used,
        "local_stall_threshold_ms": stall_threshold,
        "local_stall_probability_per_operation_window": stall_probability,
        "local_resource_operations": local_operations,
    }


def replay_full_trace(
    *,
    layout: ReplayLayout,
    tickets: dict[str, CausalSlackTicket],
    ticket_field: str,
    config: dict[str, Any],
    runtime_config: AegisFullDesignConfig,
) -> tuple[ReplayOutputs, dict[str, dict[int, AegisFullDesignRuntime]], dict[str, Any]]:
    dp_size = int(config["cluster"]["dp_size"])
    prefix_q = int(config["chunking"]["prefix_tokens"])
    decode_q = int(config["chunking"]["decode_tokens"])
    fault_times = _fault_times(layout, config)
    runtimes = _runtime_grid(runtime_config, dp_size)
    outputs = ReplayOutputs()
    # Pressure is target-local but must reflect the target Worker's own primary
    # serving load.  A request's standby target is deliberately a different DP
    # rank and cannot be used as a proxy for that Worker's primary headroom.
    active_primary_by_rank: Counter[int] = Counter()
    aimd_pressure_windows: Counter[str] = Counter()
    aimd_reason_windows: Counter[str] = Counter()
    pressure_mode = str(config["aimd"].get("pressure_mode") or "")
    pressure_models = config["aimd"].get("physical_pressure_pool_models")
    if pressure_mode == V100_AIMD_PRESSURE_MODE:
        if not isinstance(pressure_models, dict) or set(pressure_models) != {
            f"hbm-vllm-dp{rank}" for rank in range(dp_size)
        }:
            raise ValueError("calibrated AIMD replay lacks one model per DP pool")
    elif pressure_mode != "active_source_request_count_proxy":
        raise ValueError("unsupported AIMD pressure mode")
    sequence = 0
    events: list[tuple[float, int, int, str, Any]] = []
    priorities = {
        "target_failure": 0,
        "source_failure": 1,
        "source_snapshot": 1,
        "aimd": 2,
        "start": 3,
        "seal": 4,
        "finish": 5,
    }

    def append_event(at_ms: float, kind: str, payload: Any) -> None:
        nonlocal sequence
        sequence += 1
        events.append((float(at_ms), priorities[kind], sequence, kind, payload))

    for meta in layout.requests:
        append_event(meta.global_start_ms, "start", meta.row.rid)
        append_event(meta.global_finish_ms, "finish", meta.row.rid)
    for rank, at_ms in fault_times["target"].items():
        append_event(at_ms, "target_failure", rank)
    for rank, at_ms in fault_times["source"].items():
        append_event(at_ms, "source_failure", rank)
    snapshot_count = min(
        int(config["faults"]["evaluation_snapshot_count"]),
        len(layout.requests),
    )
    snapshot_fraction = float(
        config["faults"]["snapshot_fraction_within_request"]
    )
    ordered_anchors = sorted(
        layout.requests, key=lambda row: (row.global_start_ms, row.row.rid)
    )
    anchor_indices = np.linspace(
        0, len(ordered_anchors) - 1, snapshot_count, dtype=int
    )
    for snapshot_index, anchor_index in enumerate(anchor_indices):
        anchor = ordered_anchors[int(anchor_index)]
        at_ms = anchor.global_start_ms + anchor.row.api_time_ms * snapshot_fraction
        append_event(
            at_ms,
            "source_snapshot",
            (f"stratified-source-snapshot-{snapshot_index:04d}", anchor.row.rid),
        )
    window = float(config["aimd"]["window_ms"])
    next_window = layout.workload_start_ms + window
    while next_window < layout.workload_finish_ms + 1e-9:
        append_event(next_window, "aimd", None)
        next_window += window
    heapq.heapify(events)

    sealed_events = 0
    started_wall = time.monotonic()
    while events:
        at_ms, _priority, _event_sequence, kind, payload = heapq.heappop(events)
        if kind == "start":
            meta = layout.by_id[payload]
            source_generation = (
                0 if at_ms < fault_times["source"][meta.source_rank] else 1
            )
            target_generation = (
                0 if at_ms < fault_times["target"][meta.target_rank] else 1
            )
            meta.source_worker = f"source-{meta.source_rank}-g{source_generation}"
            meta.current_target = f"target-{meta.target_rank}-g{target_generation}"
            meta.started = True
            active_primary_by_rank[meta.source_rank] += 1
            slack = remaining_ticket_slack(
                meta, tickets[meta.row.rid], at_ms, field_name=ticket_field
            )
            for policy in PLACEMENT_POLICIES:
                policy_slack = slack if policy == "aegiskv" else 0.0
                runtimes[policy][meta.target_rank].register_request(
                    request_id=meta.row.rid,
                    source_worker=meta.source_worker,
                    source_failure_domain=f"source-domain-{meta.source_rank}-g{source_generation}",
                    hbm_target=meta.current_target,
                    hbm_failure_domain=f"target-domain-{meta.target_rank}-g{target_generation}",
                    route_version=1,
                    prompt_tokens=meta.row.prompt_tokens,
                    output_tokens=meta.row.output_tokens,
                    slack_ms=policy_slack,
                    model_revision=str(config["model"]["revision"]),
                    kv_abi=str(config["model"]["kv_abi"]),
                    parallel_layout=_parallel_layout_id(config),
                    time_ms=at_ms,
                )
            if meta.chunk_count:
                sequence += 1
                heapq.heappush(
                    events,
                    (
                        meta.seal_time_ms(0, prefix_q=prefix_q, decode_q=decode_q),
                        priorities["seal"],
                        sequence,
                        "seal",
                        (meta.row.rid, 0),
                    ),
                )

        elif kind == "seal":
            request_id, ordinal = payload
            meta = layout.by_id[request_id]
            chunk = meta.chunk(ordinal, prefix_q=prefix_q, decode_q=decode_q)
            for policy in PLACEMENT_POLICIES:
                decision = runtimes[policy][meta.target_rank].seal_chunk(
                    request_id, chunk, time_ms=at_ms
                )
                outputs.placement_counts[policy][
                    decision.intended_residency.value
                ] += 1
                outputs.placement_counts[policy][
                    f"HBM_REASON:{decision.hbm.reason}"
                ] += 1
                outputs.placement_counts[policy][
                    f"DRAM_REASON:{decision.dram.reason}"
                ] += 1
            sealed_events += 1
            next_ordinal = ordinal + 1
            if next_ordinal < meta.chunk_count:
                sequence += 1
                heapq.heappush(
                    events,
                    (
                        meta.seal_time_ms(
                            next_ordinal, prefix_q=prefix_q, decode_q=decode_q
                        ),
                        priorities["seal"],
                        sequence,
                        "seal",
                        (request_id, next_ordinal),
                    ),
                )
            if sealed_events % int(config["audit"]["progress_every_chunks"]) == 0:
                elapsed = max(time.monotonic() - started_wall, 1e-9)
                print(
                    f"progress sealed={sealed_events}/{layout.intended_chunk_count} "
                    f"rate={sealed_events / elapsed:.0f} chunks/s",
                    flush=True,
                )

        elif kind == "finish":
            meta = layout.by_id[payload]
            for policy in PLACEMENT_POLICIES:
                runtime = runtimes[policy][meta.target_rank]
                if runtime.has_request(meta.row.rid):
                    runtime.finish_request(meta.row.rid, time_ms=at_ms)
            meta.finished = True
            active_primary_by_rank[meta.source_rank] -= 1

        elif kind == "aimd":
            if pressure_mode == "active_source_request_count_proxy":
                threshold = int(
                    config["aimd"]["pressure_active_source_requests"]
                )
                for rank in range(dp_size):
                    pressure = active_primary_by_rank[rank] >= threshold
                    aimd_pressure_windows["rank_windows"] += 1
                    aimd_pressure_windows["pressure_windows"] += int(pressure)
                    for policy in PLACEMENT_POLICIES:
                        runtime = runtimes[policy][rank]
                        runtime.sample_hbm_local_resource_operations(time_ms=at_ms)
                        runtime.control_hbm_quota(
                            pressure=pressure, time_ms=at_ms
                        )
            else:
                assert isinstance(pressure_models, dict)
                for rank in range(dp_size):
                    model = pressure_models[f"hbm-vllm-dp{rank}"]
                    digest = hashlib.sha256(
                        (
                            f"{runtime_config.random_salt}:aimd-local-stall:"
                            f"{rank}:{at_ms:.6f}"
                        ).encode("utf-8")
                    ).digest()
                    local_stall_draw = int.from_bytes(
                        digest[:8], "big"
                    ) / float(2**64)
                    aimd_pressure_windows["rank_windows"] += 1
                    for policy in PLACEMENT_POLICIES:
                        runtime = runtimes[policy][rank]
                        signal = _physical_aimd_pressure(
                            runtime,
                            model,
                            time_ms=at_ms,
                            local_stall_draw=local_stall_draw,
                        )
                        pressure = bool(signal["pressure"])
                        memory_pressure = bool(signal["memory_pressure"])
                        local_stall_pressure = bool(
                            signal["local_stall_pressure"]
                        )
                        aimd_pressure_windows["policy_rank_windows"] += 1
                        aimd_pressure_windows["pressure_windows"] += int(pressure)
                        aimd_reason_windows["memory_pressure"] += int(
                            memory_pressure
                        )
                        aimd_reason_windows["local_stall_pressure"] += int(
                            local_stall_pressure
                        )
                        aimd_reason_windows[f"{policy}:pressure"] += int(pressure)
                        runtime.control_hbm_quota(
                            pressure=pressure, time_ms=at_ms
                        )

        elif kind == "target_failure":
            rank = int(payload)
            failed = f"target-{rank}-g0"
            replacement = f"target-{rank}-g1"
            rotations = Counter()
            for meta in layout.by_target_rank.get(rank, ()):
                if not meta.started or meta.current_target != failed:
                    continue
                rotated = False
                for policy in PLACEMENT_POLICIES:
                    runtime = runtimes[policy][rank]
                    if runtime.has_request(meta.row.rid):
                        result = runtime.rotate_target(
                            meta.row.rid,
                            failed_target=failed,
                            replacement_target=replacement,
                            replacement_failure_domain=f"target-domain-{rank}-g1",
                            time_ms=at_ms,
                        )
                        rotations[policy] += "skipped" not in result
                        rotations[f"{policy}_reprotections"] += int(
                            result["replacement_hbm_reprotections"]
                        )
                        rotated = rotated or "skipped" not in result
                if rotated:
                    meta.current_target = replacement
                    meta.rotated_before_source = True
            outputs.fault_events.append(
                {
                    "event_id": f"target-failure-{rank}",
                    "kind": "target_failure",
                    "time_ms": at_ms,
                    "rank": rank,
                    "failed_target": failed,
                    "replacement_target": replacement,
                    "route_version_changed": False,
                    **dict(rotations),
                }
            )

        elif kind in ("source_failure", "source_snapshot"):
            if kind == "source_failure":
                rank = int(payload)
                worker = f"source-{rank}-g0"
                event_id = f"source-failure-{rank}"
            else:
                event_id, anchor_id = payload
                anchor = layout.by_id[anchor_id]
                rank = anchor.source_rank
                worker = anchor.source_worker
            affected = [
                meta
                for meta in layout.by_source_rank.get(rank, ())
                if meta.started and not meta.finished and meta.source_worker == worker
            ]
            decision_sets: dict[str, dict[str, RecoveryDecision]] = {
                "recompute": {},
                "zero_slack": {},
                "aegiskv": {},
            }
            remaining: dict[str, float] = {}
            for meta in affected:
                produced = meta.produced_tokens(at_ms)
                slack = remaining_ticket_slack(
                    meta, tickets[meta.row.rid], at_ms, field_name=ticket_field
                )
                remaining[meta.row.rid] = slack
                decision_sets["recompute"][meta.row.rid] = _recompute_decision(
                    meta, produced, slack, runtime_config.recompute
                )
                decision_sets["aegiskv"][meta.row.rid] = runtimes[
                    "aegiskv"
                ][meta.target_rank].estimate_source_failure(
                    meta.row.rid,
                    activated_target=meta.current_target,
                    source_token_end=produced,
                    remaining_slack_ms=slack,
                    time_ms=at_ms,
                )
                decision_sets["zero_slack"][meta.row.rid] = runtimes[
                    "zero_slack"
                ][meta.target_rank].estimate_source_failure(
                    meta.row.rid,
                    activated_target=meta.current_target,
                    source_token_end=produced,
                    remaining_slack_ms=0.0,
                    time_ms=at_ms,
                )
            failure_rows, stage_rows = evaluate_failure_snapshot(
                event_id=event_id,
                failure_time_ms=at_ms,
                affected=affected,
                decisions=decision_sets,
                remaining_slack=remaining,
                stage_branch_finishes=layout.stage_branch_finishes,
            )
            outputs.failure_rows.extend(failure_rows)
            outputs.stage_rows.extend(stage_rows)
            rotated = [row for row in affected if row.rotated_before_source]
            candidates = rotated or affected
            if candidates:
                representative = max(
                    candidates,
                    key=lambda row: (row.produced_tokens(at_ms), row.row.rid),
                )
                outputs.probe_candidates.append((event_id, at_ms, representative))
            outputs.fault_events.append(
                {
                    "event_id": event_id,
                    "kind": "source_failure_snapshot",
                    "schedule_kind": kind,
                    "time_ms": at_ms,
                    "rank": rank,
                    "source_worker": worker,
                    "affected_requests": len(affected),
                    "affected_workflows": len(
                        {row.row.workflow_id for row in affected}
                    ),
                    "rotated_target_requests": len(rotated),
                    "counterfactual_non_destructive": True,
                }
            )

    print("flushing independent transfer queues", flush=True)
    for policy in PLACEMENT_POLICIES:
        for runtime in runtimes[policy].values():
            runtime.flush()
            runtime.validate_invariants()
    replay_meta = {
        "sealed_chunk_events": sealed_events,
        "intended_chunk_events": layout.intended_chunk_count,
        "event_loop_wall_seconds": time.monotonic() - started_wall,
        "fault_times": fault_times,
        "aimd_pressure_model": {
            "mode": config["aimd"]["pressure_mode"],
            "calibrated": bool(config["aimd"]["pressure_calibrated"]),
            "active_source_request_threshold": (
                int(config["aimd"]["pressure_active_source_requests"])
                if pressure_mode == "active_source_request_count_proxy"
                else None
            ),
            "rank_windows": int(aimd_pressure_windows["rank_windows"]),
            "policy_rank_windows": int(
                aimd_pressure_windows["policy_rank_windows"]
            ),
            "pressure_windows": int(
                aimd_pressure_windows["pressure_windows"]
            ),
            "pressure_reason_windows": dict(sorted(aimd_reason_windows.items())),
        },
    }
    return outputs, runtimes, replay_meta


def run_fault_execution_probes(
    candidates: Sequence[tuple[str, float, ReplayRequest]],
    *,
    tickets: dict[str, CausalSlackTicket],
    ticket_field: str,
    runtime_config: AegisFullDesignConfig,
    config: dict[str, Any],
    fault_times: dict[str, Any],
) -> list[dict[str, Any]]:
    prefix_q = int(config["chunking"]["prefix_tokens"])
    decode_q = int(config["chunking"]["decode_tokens"])
    probes: list[dict[str, Any]] = []
    best_by_rank: dict[int, tuple[str, float, ReplayRequest]] = {}
    for candidate in candidates:
        event_id, failure_global, meta = candidate
        prior = best_by_rank.get(meta.source_rank)
        score = (
            event_id.startswith("source-failure-"),
            meta.rotated_before_source,
            meta.produced_tokens(failure_global),
        )
        if prior is None:
            best_by_rank[meta.source_rank] = candidate
            continue
        prior_score = (
            prior[0].startswith("source-failure-"),
            prior[2].rotated_before_source,
            prior[2].produced_tokens(prior[1]),
        )
        if score > prior_score:
            best_by_rank[meta.source_rank] = candidate
    for event_id, failure_global, meta in sorted(best_by_rank.values()):
        start = meta.global_start_ms
        failure_at = failure_global - start
        initial_target_generation = (
            0 if start < fault_times["target"][meta.target_rank] else 1
        )
        current_target = f"target-{meta.target_rank}-g{initial_target_generation}"
        runtime = AegisFullDesignRuntime(
            replace(
                runtime_config,
                hbm_capacity_bytes=max(
                    runtime_config.hbm_capacity_bytes,
                    meta.row.prompt_tokens * runtime_config.kv_bytes_per_token,
                ),
                dram_capacity_bytes=max(
                    runtime_config.dram_capacity_bytes,
                    meta.row.prompt_tokens * runtime_config.kv_bytes_per_token,
                ),
                random_salt=f"{runtime_config.random_salt}:probe:{event_id}",
                event_log_limit=512,
            ),
            target_id=current_target,
            dram_pool_id=f"probe-dram-{event_id}",
            dram_failure_domain="probe-store-domain",
        )
        initial_slack = remaining_ticket_slack(
            meta, tickets[meta.row.rid], start, field_name=ticket_field
        )
        runtime.register_request(
            request_id=meta.row.rid,
            source_worker=meta.source_worker,
            source_failure_domain=f"source-domain-{meta.source_worker}",
            hbm_target=current_target,
            hbm_failure_domain=f"target-domain-{meta.target_rank}-g{initial_target_generation}",
            route_version=1,
            prompt_tokens=meta.row.prompt_tokens,
            output_tokens=meta.row.output_tokens,
            slack_ms=initial_slack,
            model_revision=str(config["model"]["revision"]),
            kv_abi=str(config["model"]["kv_abi"]),
            parallel_layout=_parallel_layout_id(config),
            time_ms=0.0,
        )
        local_events: list[tuple[float, str, int]] = []
        for ordinal in range(meta.chunk_count):
            seal = meta.seal_time_ms(
                ordinal, prefix_q=prefix_q, decode_q=decode_q
            )
            if seal <= failure_global + 1e-9:
                local_events.append((seal - start, "seal", ordinal))
        target_failure = fault_times["target"][meta.target_rank]
        if start < target_failure < failure_global:
            local_events.append((target_failure - start, "rotate", -1))
        rotations = 0
        for at_ms, kind, ordinal in sorted(
            local_events, key=lambda row: (row[0], 0 if row[1] == "rotate" else 1)
        ):
            if kind == "rotate":
                replacement = f"target-{meta.target_rank}-g1"
                runtime.rotate_target(
                    meta.row.rid,
                    failed_target=current_target,
                    replacement_target=replacement,
                    replacement_failure_domain=f"target-domain-{meta.target_rank}-g1",
                    time_ms=at_ms,
                )
                current_target = replacement
                rotations += 1
            else:
                runtime.seal_chunk(
                    meta.row.rid,
                    meta.chunk(ordinal, prefix_q=prefix_q, decode_q=decode_q),
                    time_ms=at_ms,
                )
        produced = meta.produced_tokens(failure_global)
        remaining = remaining_ticket_slack(
            meta, tickets[meta.row.rid], failure_global, field_name=ticket_field
        )
        runtime.observe_source_frontier(
            meta.row.rid, source_token_end=produced, time_ms=failure_at
        )
        decision = runtime.recover_source_failure(
            meta.row.rid,
            activated_target=current_target,
            next_route_version=2,
            remaining_slack_ms=remaining,
            time_ms=failure_at,
        )
        completion_at = failure_at + decision.recovery_ms
        completion = runtime.finish_request(meta.row.rid, time_ms=completion_at)
        runtime.flush()
        runtime.validate_invariants()
        snapshot = runtime.snapshot(include_controller=True)
        probes.append(
            {
                "event_id": event_id,
                "request_id": meta.row.rid,
                "produced_tokens": produced,
                "sealed_chunks_before_failure": len(
                    [row for row in local_events if row[1] == "seal"]
                ),
                "target_rotations": rotations,
                "fence_source": True,
                "output_frozen": True,
                "pending_copies_aborted_before_selection": True,
                "action": decision.action,
                "restored_tokens": decision.restored_tokens,
                "recompute_tokens": decision.recompute_tokens,
                "recovery_ms": decision.recovery_ms,
                "prior_route_version": 1,
                "next_route_version": decision.route_version,
                "silent_replay_to_committed_output_position": True,
                "all_rank_barrier_shards": runtime_config.shard_count,
                "completion_request_terminal": bool(completion["completed"]),
                "active_requests_after_probe": snapshot["active_requests"],
                "pending_transfer_jobs_after_flush": snapshot[
                    "pending_transfer_jobs"
                ],
                "controller_event_counts": snapshot["controller_event_counts"],
            }
        )
    return probes


def run_independent_tier_mechanism_probes(
    runtime_config: AegisFullDesignConfig,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Exercise both independent paths even when calibrated DRAM is uneconomic.

    Copy bandwidths remain the calibrated values.  Only the fail-now recompute
    sensitivity is slowed so DRAM has positive admission value.  This probe is
    a state-machine check and is excluded from optimization summaries.
    """

    slow_recompute = RecoveryTiming(
        setup_ms=runtime_config.recompute.setup_ms,
        ms_per_token=max(
            runtime_config.dram_load.ms_per_token * 2.0,
            runtime_config.recompute.ms_per_token,
        ),
    )
    probe_config = replace(
        runtime_config,
        recompute=slow_recompute,
        initial_hbm_alpha=1.0,
        minimum_hbm_alpha=1.0,
        maximum_hbm_alpha=1.0,
        random_salt=f"{runtime_config.random_salt}:independent-tier-probe",
        event_log_limit=256,
    )

    def register(runtime: AegisFullDesignRuntime, request_id: str) -> None:
        runtime.register_request(
            request_id=request_id,
            source_worker="probe-source",
            source_failure_domain="probe-source-domain",
            hbm_target="probe-target",
            hbm_failure_domain="probe-target-domain",
            route_version=1,
            prompt_tokens=4096,
            output_tokens=0,
            slack_ms=0.0,
            model_revision=str(config["model"]["revision"]),
            kv_abi=str(config["model"]["kv_abi"]),
            parallel_layout=_parallel_layout_id(config),
            time_ms=0.0,
        )

    chunk = SealedChunk(0, 0, 4096, "prefix")
    byte_size = 4096 * runtime_config.kv_bytes_per_token

    success = AegisFullDesignRuntime(
        probe_config,
        target_id="probe-target",
        dram_pool_id="probe-dram",
        dram_failure_domain="probe-store-domain",
    )
    register(success, "independent-success")
    admission = success.seal_chunk("independent-success", chunk, time_ms=0.0)
    hbm_finish = runtime_config.hbm_copy.duration_ms(byte_size)
    dram_finish = runtime_config.dram_copy.duration_ms(byte_size)
    success.advance_to(hbm_finish + 1e-6)
    state_after_hbm = {
        "hbm": success.controller.copy_state(
            "independent-success", 0, ProtectionTier.HBM
        ).value,
        "dram": success.controller.copy_state(
            "independent-success", 0, ProtectionTier.DRAM
        ).value,
    }
    success.advance_to(dram_finish + 1e-6)
    state_after_both = {
        "hbm": success.controller.copy_state(
            "independent-success", 0, ProtectionTier.HBM
        ).value,
        "dram": success.controller.copy_state(
            "independent-success", 0, ProtectionTier.DRAM
        ).value,
        "residency": success.controller.residency(
            "independent-success", 0
        ).value,
    }
    success_complete = success.finish_request(
        "independent-success", time_ms=success.now_ms
    )

    hbm_failure = AegisFullDesignRuntime(
        replace(
            probe_config,
            hbm_copy_failure_probability=1.0,
            random_salt=f"{probe_config.random_salt}:hbm-failure",
        ),
        target_id="probe-target",
        dram_pool_id="probe-dram",
        dram_failure_domain="probe-store-domain",
    )
    register(hbm_failure, "independent-hbm-failure")
    failure_admission = hbm_failure.seal_chunk(
        "independent-hbm-failure", chunk, time_ms=0.0
    )
    hbm_failure.flush()
    dram_state = hbm_failure.controller.copy_state(
        "independent-hbm-failure", 0, ProtectionTier.DRAM
    )
    hbm_state = hbm_failure.controller.copy_state(
        "independent-hbm-failure", 0, ProtectionTier.HBM
    )
    recovery = hbm_failure.recover_source_failure(
        "independent-hbm-failure",
        activated_target="probe-target",
        next_route_version=2,
        remaining_slack_ms=0.0,
        time_ms=hbm_failure.now_ms,
    )
    failure_complete = hbm_failure.finish_request(
        "independent-hbm-failure",
        time_ms=hbm_failure.now_ms + recovery.recovery_ms,
    )
    hbm_failure.flush()
    return {
        "evidence_class": "synthetic_state_machine_probe_with_calibrated_copy_paths",
        "included_in_optimization_summary": False,
        "sensitivity_change": {
            "field": "recompute_ms_per_token",
            "production_projection": runtime_config.recompute.ms_per_token,
            "probe_value": slow_recompute.ms_per_token,
            "reason": "make calibrated DRAM load economically admissible",
        },
        "success_case": {
            "both_tiers_admitted": admission.hbm.admitted
            and admission.dram.admitted,
            "state_after_hbm_completion": state_after_hbm,
            "state_after_both_completions": state_after_both,
            "independent_completion_order": hbm_finish < dram_finish,
            "all_shards_per_path": runtime_config.shard_count,
            "request_terminal": bool(success_complete["completed"]),
        },
        "one_path_failure_case": {
            "both_tiers_admitted": failure_admission.hbm.admitted
            and failure_admission.dram.admitted,
            "hbm_record_after_failure": (
                hbm_state.value if isinstance(hbm_state, CopyState) else None
            ),
            "dram_record_after_hbm_failure": (
                dram_state.value if isinstance(dram_state, CopyState) else None
            ),
            "recovery_action": recovery.action,
            "recovery_restored_tokens": recovery.restored_tokens,
            "request_terminal": bool(failure_complete["completed"]),
        },
        "no_hbm_to_dram_transfer": True,
        "copy_kinds": ["source_d2d", "source_d2h"],
    }


def _ticket_manifest(
    tickets: dict[str, CausalSlackTicket], field_name: str
) -> dict[str, Any]:
    values = np.asarray(
        [_ticket_value(ticket, field_name) for ticket in tickets.values()], dtype=float
    )
    digest = hashlib.sha256()
    for request_id in sorted(tickets):
        ticket = tickets[request_id]
        digest.update(
            (
                f"{request_id}\t{ticket.published_at_ms:.9f}\t"
                f"{ticket.slack_q50_ms:.9f}\t{ticket.slack_lcb_ms:.9f}\t"
                f"{ticket.fold}\t{ticket.predictor_sha256}\n"
            ).encode()
        )
    return {
        "ticket_count": len(tickets),
        "primary_slack_field": field_name,
        "nonzero_primary_tickets": int(np.count_nonzero(values > 0.0)),
        "fail_close_zero_tickets": sum(
            ticket.predictor_sha256 == "fail-close-zero"
            for ticket in tickets.values()
        ),
        "primary_slack_p50_ms": float(np.quantile(values, 0.50)),
        "primary_slack_p90_ms": float(np.quantile(values, 0.90)),
        "primary_slack_max_ms": float(values.max()),
        "ticket_manifest_sha256": digest.hexdigest(),
    }


def _summaries(
    failure_rows: Sequence[dict[str, Any]], stage_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy, _protection, _ordering in POLICY_SPECS:
        failures = [row for row in failure_rows if row["policy"] == policy]
        stages = [row for row in stage_rows if row["policy"] == policy]
        service = np.asarray(
            [float(row["recovery_service_ms"]) for row in failures], dtype=float
        )
        queue = np.asarray(
            [float(row["queue_completion_delay_ms"]) for row in failures],
            dtype=float,
        )
        harms = np.asarray(
            [float(row["join_barrier_harm_ms"]) for row in stages], dtype=float
        )
        actions = Counter(str(row["action"]) for row in failures)
        output[policy] = {
            "failure_request_count": len(failures),
            "failure_stage_count": len(stages),
            "action_counts": dict(sorted(actions.items())),
            "mean_recovery_service_ms": float(service.mean()) if len(service) else 0.0,
            "p95_recovery_service_ms": float(np.quantile(service, 0.95)) if len(service) else 0.0,
            "mean_queue_completion_delay_ms": float(queue.mean()) if len(queue) else 0.0,
            "p95_queue_completion_delay_ms": float(np.quantile(queue, 0.95)) if len(queue) else 0.0,
            "total_join_barrier_harm_ms": float(harms.sum()),
            "mean_join_barrier_harm_ms": float(harms.mean()) if len(harms) else 0.0,
            "p95_join_barrier_harm_ms": float(np.quantile(harms, 0.95)) if len(harms) else 0.0,
        }
    return output


def _paired_bootstrap(
    stage_rows: Sequence[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    workflow_ids = sorted({str(row["workflow_id"]) for row in stage_rows})
    if not workflow_ids:
        return {"status": "no_faulted_workflows", "comparisons": {}}
    index = {workflow_id: i for i, workflow_id in enumerate(workflow_ids)}
    values = {
        policy: np.zeros(len(workflow_ids), dtype=float)
        for policy, _protection, _ordering in POLICY_SPECS
    }
    for row in stage_rows:
        values[str(row["policy"])][index[str(row["workflow_id"])]] += float(
            row["join_barrier_harm_ms"]
        )
    treatment = values["aegiskv_latest"]
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    resamples = int(config["statistics"]["bootstrap_resamples"])
    comparisons: dict[str, Any] = {}
    for baseline_name, baseline in values.items():
        if baseline_name == "aegiskv_latest":
            continue
        baseline_mean = float(baseline.mean())
        treatment_mean = float(treatment.mean())
        observed = (
            100.0 * (baseline_mean - treatment_mean) / baseline_mean
            if baseline_mean > 0.0
            else 0.0
        )
        samples = np.empty(resamples, dtype=float)
        for sample in range(resamples):
            picked = rng.integers(0, len(workflow_ids), size=len(workflow_ids))
            base = float(baseline[picked].mean())
            treated = float(treatment[picked].mean())
            samples[sample] = 100.0 * (base - treated) / base if base > 0.0 else 0.0
        comparisons[baseline_name] = {
            "metric": "workflow_aggregated_join_barrier_harm",
            "paired_workflow_count": len(workflow_ids),
            "baseline_mean_ms": baseline_mean,
            "aegiskv_latest_mean_ms": treatment_mean,
            "reduction_percent": observed,
            "bootstrap_95ci_percent": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        }
    return {
        "status": "complete",
        "bootstrap_unit": "faulted_workflow",
        "bootstrap_resamples": resamples,
        "comparisons": comparisons,
    }


def _runtime_snapshots(
    runtimes: dict[str, dict[int, AegisFullDesignRuntime]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy, by_rank in runtimes.items():
        output[policy] = {
            str(rank): runtime.snapshot() for rank, runtime in sorted(by_rank.items())
        }
    return output


def _aggregate_runtime_metrics(snapshots: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy, by_rank in snapshots.items():
        totals: Counter[str] = Counter()
        events: Counter[str] = Counter()
        for snapshot in by_rank.values():
            totals.update(snapshot["metrics"])
            events.update(snapshot["controller_event_counts"])
        output[policy] = {
            "metrics": dict(sorted(totals.items())),
            "controller_event_counts": dict(sorted(events.items())),
        }
    return output


def _valid_v100_audit_calibration(provenance: dict[str, Any]) -> bool:
    metrics = provenance.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(V100_PROFILE_METRICS):
        return False
    try:
        recompute_fixed = float(metrics["recompute_fixed_overhead_ms"])
        rates = [
            float(metrics[name])
            for name in V100_PROFILE_METRICS
            if name != "recompute_fixed_overhead_ms"
        ]
    except (TypeError, ValueError):
        return False
    return bool(
        provenance.get("calibration_semantics") == V100_PROFILE_SEMANTICS
        and provenance.get("result_label") == V100_HARDWARE_LABEL
        and provenance.get("compatibility_identity")
        == V100_COMPATIBILITY_IDENTITY
        and provenance.get("runtime_consumption_contract")
        == V100_RUNTIME_CONSUMPTION_CONTRACT
        and math.isfinite(recompute_fixed)
        and recompute_fixed >= 0.0
        and all(math.isfinite(value) and value > 0.0 for value in rates)
    )


def _audit(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    algorithm1: dict[str, Any],
    tickets: dict[str, CausalSlackTicket],
    layout: ReplayLayout,
    replay_meta: dict[str, Any],
    outputs: ReplayOutputs,
    runtime_snapshots: dict[str, Any],
    probes: Sequence[dict[str, Any]],
    independent_tier_probe: dict[str, Any],
    calibration_provenance: dict[str, Any],
    full_profile: bool,
) -> dict[str, Any]:
    expected = config["dataset"]["expected_full_profile"]
    projection = config["workload"]["token_projection"]
    calibration = config["calibration"]
    dataset_exact = all(
        int(manifest[key]) == int(expected[key])
        for key in ("workflow_count", "request_count", "input_tokens", "output_tokens")
    )
    matching = {
        policy: {
            (row["event_id"], row["request_id"])
            for row in outputs.failure_rows
            if row["policy"] == policy
        }
        for policy, _protection, _ordering in POLICY_SPECS
    }
    first_match = next(iter(matching.values()), set())
    paired = all(keys == first_match for keys in matching.values())
    gates = {
        "design_semantics_current": config["design_semantics"] == DESIGN_SEMANTICS,
        "context_projection_preregistered_and_executable": bool(
            projection.get("status") == "PASS"
            and float(projection.get("factor", 0.0)) == 0.004
            and all(
                row.row.prompt_tokens + row.row.output_tokens
                <= int(config["model"]["maximum_context_tokens"])
                for row in layout.requests
            )
            and all(
                row.row.prompt_tokens + row.row.output_tokens
                <= int(calibration["maximum_calibrated_context_tokens"])
                for row in layout.requests
            )
        ),
        "full_dataset_exact": bool(full_profile and dataset_exact),
        "all_real_requests_ticketed": len(tickets) == manifest["request_count"],
        "algorithm1_causal_feature_audit": bool(
            algorithm1["forbidden_feature_audit_passed"]
        ),
        "algorithm1_workflow_disjoint_crossfit": bool(
            algorithm1["workflow_disjoint_crossfit_passed"]
        ),
        "progressive_seal_count_exact": replay_meta["sealed_chunk_events"]
        == layout.intended_chunk_count,
        "paired_failure_request_keys": paired,
        "target_rotation_exercised": any(
            event["kind"] == "target_failure"
            and int(event.get("aegiskv", 0)) > 0
            for event in outputs.fault_events
        ),
        "source_failure_snapshots_exercised": any(
            event["kind"] == "source_failure_snapshot"
            and int(event["affected_requests"]) > 0
            for event in outputs.fault_events
        ),
        "mutating_failure_path_exercised": bool(probes),
        "failure_fence_route_silent_replay_all_rank_terminal": bool(probes)
        and all(
            probe["fence_source"]
            and probe["output_frozen"]
            and probe["pending_copies_aborted_before_selection"]
            and probe["next_route_version"] == 2
            and probe["silent_replay_to_committed_output_position"]
            and probe["all_rank_barrier_shards"]
            == int(config["cluster"]["pp_size"])
            * int(config["cluster"]["tp_size"])
            and probe["completion_request_terminal"]
            and probe["active_requests_after_probe"] == 0
            and probe["pending_transfer_jobs_after_flush"] == 0
            for probe in probes
        ),
        "all_normal_requests_reclaimed": all(
            snapshot["active_requests"] == 0
            and snapshot["pending_transfer_jobs"] == 0
            for by_rank in runtime_snapshots.values()
            for snapshot in by_rank.values()
        ),
        "independent_no_cross_tier_schema": all(
            snapshot["design_semantics"] == DESIGN_SEMANTICS
            for by_rank in runtime_snapshots.values()
            for snapshot in by_rank.values()
        ),
        "independent_hbm_dram_state_machine_exercised": bool(
            independent_tier_probe["success_case"]["both_tiers_admitted"]
            and independent_tier_probe["success_case"][
                "independent_completion_order"
            ]
            and independent_tier_probe["success_case"][
                "state_after_hbm_completion"
            ]
            == {"hbm": "COMMITTED", "dram": "PENDING"}
            and independent_tier_probe["success_case"][
                "state_after_both_completions"
            ]["residency"]
            == "HBM_AND_DRAM"
            and independent_tier_probe["one_path_failure_case"][
                "dram_record_after_hbm_failure"
            ]
            == "COMMITTED"
            and independent_tier_probe["one_path_failure_case"][
                "recovery_action"
            ]
            == "DRAM"
            and independent_tier_probe["no_hbm_to_dram_transfer"]
        ),
        "independent_physical_path_calibration_bound": bool(
            _valid_v100_audit_calibration(calibration_provenance)
        ),
        "runtime_consumption_contract_enforced": bool(
            calibration_provenance.get("runtime_consumption_contract")
            == V100_RUNTIME_CONSUMPTION_CONTRACT
        ),
        "copy_work_disclosed": True,
        "calibration_extrapolation_disclosed": True,
        "aimd_pressure_contract_disclosed": bool(
            (
                replay_meta["aimd_pressure_model"]["mode"]
                == "active_source_request_count_proxy"
                and replay_meta["aimd_pressure_model"]["calibrated"] is False
                and calibration_provenance.get("aimd_pressure", {}).get(
                    "sglsim_quantitative_aimd_pressure_authorized"
                )
                is False
            )
            or (
                replay_meta["aimd_pressure_model"]["mode"]
                == V100_AIMD_PRESSURE_MODE
                and replay_meta["aimd_pressure_model"]["calibrated"] is True
                and calibration_provenance.get("aimd_pressure", {}).get(
                    "sglsim_quantitative_aimd_pressure_authorized"
                )
                is True
                and int(
                    replay_meta["aimd_pressure_model"]["policy_rank_windows"]
                )
                > 0
            )
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "evidence_class": "full_real_trace_metadata_simulation_with_calibrated_projection",
        "paper_ready": False,
        "physical_end_to_end_verified": False,
        "boundaries": [
            (
                "The Weka workflow topology and timing are real; cross-workflow "
                "interarrival is modeled and token counts use the preregistered "
                "0.004 projection."
            ),
            (
                "Mixed-model Weka requests are mapped to the physical Qwen3-8B "
                "FP16 KV compatibility identity."
            ),
            (
                "Four distinct V100S TP1/PP1/DP4 physical path coefficients are "
                "projected into the separately labeled SGLSim topology."
            ),
            (
                "Failure benefits are paired non-destructive fail-now snapshots; "
                "separate mutating probes validate fencing, abort, route advance, "
                "silent replay, all-rank completion, and reclamation."
            ),
            (
                "SGLSim is metadata-only and communication bookkeeping does not "
                "constitute physical Mooncake recovery evidence."
            ),
            (
                "HBMQuotaAIMD pressure replays the same-layout physical "
                "pre-control CUDA-headroom OR successful Peer-HBM transfer-"
                "duration predicate. This authorizes quantitative pressure "
                "behavior inside metadata simulation, not an end-to-end paper "
                "performance claim."
                if replay_meta["aimd_pressure_model"]["calibrated"] is True
                else "HBMQuotaAIMD pressure is an explicitly uncalibrated count "
                "of active primary requests on each source DP rank; standby-"
                "target load is not substituted for primary load. Quantitative "
                "AIMD claims require a later headroom/stall-calibrated pressure "
                "model."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "config.json")
    parser.add_argument("--import-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--limit-workflows", type=int)
    parser.add_argument("--skip-dataset-sha", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Fail before creating an output directory, loading the 393-workflow trace,
    # or fitting Algorithm 1.  An invalid physical profile must never seed a
    # new replay artifact merely because the expensive ingestion happens first.
    (
        config,
        runtime_config,
        calibration_provenance,
        token_projection,
        input_binding,
    ) = preflight_v100_config(
        args.config,
        getattr(args, "import_manifest", None),
    )
    if (
        args.limit_workflows is None
        and calibration_provenance.get("aimd_pressure", {}).get("calibrated")
        is not True
    ):
        raise ValueError(
            "the formal 393-workflow replay requires a hash-bound physical "
            "V100 AIMD pressure calibration"
        )
    dataset_path = (args.dataset or default_full_trace_path()).resolve()
    output_dir = _derived_full_output_dir(
        args.output_dir,
        profile_sha256=input_binding["physical_profile_sha256"],
        limit_workflows=args.limit_workflows,
    )
    started = time.monotonic()
    print(f"loading Weka trace: {dataset_path}", flush=True)
    source_workflows = load_weka_workflows(dataset_path, limit=args.limit_workflows)
    full_profile = args.limit_workflows is None
    source_sha = (
        "" if args.skip_dataset_sha else sha256_file(dataset_path)
    )
    manifest = dataset_manifest(
        source_workflows, source_path=dataset_path, source_sha256=source_sha
    )
    if full_profile and source_sha != token_projection["source_trace_sha256"]:
        raise ValueError("context projection source trace sha256 mismatch")
    if full_profile:
        expected = config["dataset"]["expected_full_profile"]
        mismatches = {
            key: (int(manifest[key]), int(expected[key]))
            for key in (
                "workflow_count",
                "request_count",
                "input_tokens",
                "output_tokens",
            )
            if int(manifest[key]) != int(expected[key])
        }
        if mismatches:
            raise ValueError(f"full Weka dataset profile mismatch: {mismatches}")
    workflows = _scale_workflows(source_workflows, token_projection["factor"])
    projected_contexts = [
        row.prompt_tokens + row.output_tokens
        for workflow in workflows
        for row in workflow.requests
    ]
    maximum_model_context = int(config["model"]["maximum_context_tokens"])
    if not projected_contexts or max(projected_contexts) > maximum_model_context:
        raise ValueError("projected workload exceeds the model context")
    if (
        full_profile
        and max(projected_contexts)
        != int(token_projection["maximum_projected_context_tokens"])
    ):
        raise ValueError("projected context maximum differs from preregistration")
    output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"loaded workflows={manifest['workflow_count']} requests={manifest['request_count']}",
        flush=True,
    )
    samples = build_causal_samples(workflows)
    gbdt_raw = config["algorithm1"]["gbdt"]
    try:
        tickets, algorithm1 = crossfit_causal_slack(
            samples,
            folds=int(config["algorithm1"]["folds"]),
            split_salt=str(config["algorithm1"]["split_salt"]),
            interval_coverage=float(config["algorithm1"]["interval_coverage"]),
            blend_tau_rounds=float(config["algorithm1"]["blend_tau_rounds"]),
            gbdt_config=AegisGBDTConfig(
                estimators=int(gbdt_raw["estimators"]),
                learning_rate=float(gbdt_raw["learning_rate"]),
                max_depth=int(gbdt_raw["max_depth"]),
                min_leaf=int(gbdt_raw["min_leaf"]),
                max_bins=int(gbdt_raw["max_bins"]),
            ),
        )
    except ValueError:
        if full_profile:
            raise
        # A very small development slice may not populate every workflow fold.
        # Preserve causality by failing every ticket closed to zero; never fit
        # and evaluate on overlapping workflows merely to make a smoke pass.
        tickets = {}
        algorithm1 = {
            "schema": "causal_smoke_fail_close_zero_v1",
            "information_scope": "causal_request_return_prefix_only",
            "forbidden_feature_audit_passed": True,
            "workflow_disjoint_crossfit_passed": False,
            "sample_count": len(samples),
            "ticket_count": 0,
            "workflow_count": len({row.workflow_id for row in samples}),
            "folds": int(config["algorithm1"]["folds"]),
            "predictors": {},
            "smoke_only_reason": "at_least_one_workflow_fold_was_empty",
        }
    tickets = zero_slack_tickets_for_unmodeled_requests(workflows, tickets)
    ticket_field = str(config["algorithm1"]["primary_slack_field"])
    ticket_manifest = _ticket_manifest(tickets, ticket_field)
    del samples
    print(
        f"Algorithm1 samples={algorithm1['sample_count']} "
        f"predicted_tickets={algorithm1['ticket_count']} "
        f"all_tickets={len(tickets)}",
        flush=True,
    )
    layout = build_layout(workflows, config)
    outputs, runtimes, replay_meta = replay_full_trace(
        layout=layout,
        tickets=tickets,
        ticket_field=ticket_field,
        config=config,
        runtime_config=runtime_config,
    )
    probes = run_fault_execution_probes(
        outputs.probe_candidates,
        tickets=tickets,
        ticket_field=ticket_field,
        runtime_config=runtime_config,
        config=config,
        fault_times=replay_meta["fault_times"],
    )
    independent_tier_probe = run_independent_tier_mechanism_probes(
        runtime_config, config
    )
    runtime_snapshots = _runtime_snapshots(runtimes)
    runtime_aggregate = _aggregate_runtime_metrics(runtime_snapshots)
    summaries = _summaries(outputs.failure_rows, outputs.stage_rows)
    bootstrap = _paired_bootstrap(outputs.stage_rows, config)
    extrapolated_requests = sum(
        row.row.prompt_tokens + row.row.output_tokens
        > int(config["calibration"]["maximum_calibrated_context_tokens"])
        for row in layout.requests
    )
    workload = {
        "ttft_observed_requests": sum(
            row.row.ttft_ms is not None for row in layout.requests
        ),
        "ttft_imputed_requests": sum(
            row.row.ttft_ms is None for row in layout.requests
        ),
        "ttft_imputation_ratio": layout.ttft_imputation_ratio,
        "intended_sealed_chunks": layout.intended_chunk_count,
        "workflow_interarrival_semantics": "deterministic_poisson_projection",
        "workflow_mean_interarrival_ms": config["workload"][
            "workflow_mean_interarrival_ms"
        ],
        "workload_span_ms": layout.workload_finish_ms - layout.workload_start_ms,
        "requests_beyond_calibrated_context": extrapolated_requests,
        "fraction_requests_beyond_calibrated_context": extrapolated_requests
        / len(layout.requests),
        "token_projection": {
            "selected_scale_label": token_projection["selected_scale_label"],
            "factor": token_projection["factor"],
            "context_audit": token_projection["context_audit"],
            "source_input_tokens": manifest["input_tokens"],
            "source_output_tokens": manifest["output_tokens"],
            "projected_input_tokens": sum(
                row.prompt_tokens
                for workflow in workflows
                for row in workflow.requests
            ),
            "projected_output_tokens": sum(
                row.output_tokens
                for workflow in workflows
                for row in workflow.requests
            ),
            "maximum_projected_context_tokens": max(projected_contexts),
            "maximum_model_context_tokens": maximum_model_context,
            "post_hoc_primary_selection": False,
        },
    }
    audit = _audit(
        config=config,
        manifest=manifest,
        algorithm1=algorithm1,
        tickets=tickets,
        layout=layout,
        replay_meta=replay_meta,
        outputs=outputs,
        runtime_snapshots=runtime_snapshots,
        probes=probes,
        independent_tier_probe=independent_tier_probe,
        calibration_provenance=calibration_provenance,
        full_profile=full_profile,
    )
    resolved = {
        **config,
        "dataset_resolved": str(dataset_path),
        "output_dir_resolved": str(output_dir),
        "input_binding": input_binding,
        "runtime_config": asdict(runtime_config),
    }
    _dump_json(output_dir / "config_resolved.json", resolved)
    _dump_json(output_dir / "dataset_manifest.json", manifest)
    _dump_json(output_dir / "algorithm1_metrics.json", algorithm1)
    _dump_json(output_dir / "ticket_manifest.json", ticket_manifest)
    _dump_json(output_dir / "calibration_provenance.json", calibration_provenance)
    _dump_json(output_dir / "context_projection.json", token_projection)
    _dump_json(output_dir / "workload_projection.json", workload)
    _dump_json(output_dir / "fault_manifest.json", outputs.fault_events)
    _dump_json(output_dir / "fault_execution_probes.json", probes)
    _dump_json(
        output_dir / "independent_tier_mechanism_probe.json",
        independent_tier_probe,
    )
    _dump_json(
        output_dir / "placement_counts.json",
        {
            policy: dict(sorted(counts.items()))
            for policy, counts in outputs.placement_counts.items()
        },
    )
    _dump_json(output_dir / "runtime_aggregate.json", runtime_aggregate)
    _dump_json(output_dir / "summary.json", summaries)
    _dump_json(output_dir / "paired_bootstrap.json", bootstrap)
    _write_csv(output_dir / "failure_requests.csv", outputs.failure_rows)
    _write_csv(output_dir / "failure_stages.csv", outputs.stage_rows)
    _dump_json(output_dir / "audit.json", audit)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _dump_json(output_dir / "artifact_hashes.json", artifact_hashes)
    elapsed = time.monotonic() - started
    result = {
        "status": audit["status"],
        "output_dir": str(output_dir),
        "wall_seconds": elapsed,
        "dataset": {
            key: manifest[key]
            for key in (
                "workflow_count",
                "request_count",
                "input_tokens",
                "output_tokens",
            )
        },
        "sealed_chunks": replay_meta["sealed_chunk_events"],
        "faulted_requests": len(outputs.failure_rows) // len(POLICY_SPECS),
        "aegiskv_latest": summaries["aegiskv_latest"],
        "failed_gates": audit["failed_gates"],
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
