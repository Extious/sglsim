#!/usr/bin/env python3
"""Validate AegisKV figure artifacts and measured-data readiness.

The figure package deliberately contains expected-trend templates. This
validator prevents those templates from being mistaken for experimental
evidence. A result figure becomes quantitatively claim-ready only when a
matching measured CSV exists under `figures/data/measured/` and satisfies the
per-figure column contract below.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRAFT_ROOT = ROOT.parent
EXPERIMENTS_ROOT = DRAFT_ROOT / "experiments"
MEASURED_DIR = ROOT / "data" / "measured"
STATUS_PATH = ROOT / "evidence_status.json"
HPCA_HYBRID_STATUS_PATH = ROOT / "hpca_hybrid_evidence_status.json"
HPCA_HYBRID_MARKDOWN_PATH = EXPERIMENTS_ROOT / "hpca_hybrid_evidence_status.md"
PHYSICAL_RECOVERY_AGGREGATE = (
    EXPERIMENTS_ROOT
    / "sglang_small"
    / "results"
    / "physical_recovery_path_aggregate"
    / "physical_recovery_path_aggregate.csv"
)
LIVE_MOVEMENT_AUDIT = (
    EXPERIMENTS_ROOT
    / "sglsim_large"
    / "results"
    / "agentic_workflows"
    / "sglsim-current-physical-recovery-smoke-20260702"
    / "live_physical_movement_audit.csv"
)
RECOVERY_ACTION_RUN_ID = "sglsim-current-physical-recovery-action-agentic-r3-20260702"


HYBRID_PHYSICAL_CALIBRATED_STEMS: dict[str, str] = {
    "fig_restore_recompute_boundary": (
        "SGLSim restore/recompute curve with SGLang physical Host/GPUReplica "
        "restore microbench overlay."
    ),
    "fig_physical_recovery_copy_calibration": (
        "Dedicated SGLang/torch CUDA H2D/D2H/P2P copy calibration curve for "
        "the SGLSim recovery timing model."
    ),
    "fig_recovery_model_calibration_error": (
        "SGLSim recovery_model physical_copy mode is checked against "
        "SGLang/torch physical copy anchors."
    ),
    "fig_p0_physical_bridge_coverage": (
        "P0 restore rows are mapped to physical recovery-path calibration "
        "buckets while preserving the live physical-copy boundary."
    ),
    "fig_recovery_path_breakdown": (
        "Online SGLSim recovery-action distribution calibrated by SGLang "
        "physical restore microbench."
    ),
    "fig_dp_scale": (
        "Online SGLSim DP-scale recovery-action suite calibrated by SGLang "
        "physical restore microbench."
    ),
}

HYBRID_AGENTIC_RUN_STEMS = {
    "fig_recovery_path_breakdown",
    "fig_dp_scale",
}

HYBRID_TRACE_DERIVED_STEMS: dict[str, tuple[str, str]] = {
    "fig_motivation_equal_size_loss": (
        "sglsim_agentic_stagebarrier_equal_size_loss_counterfactual",
        "stage-barrier online SGLSim traces support the motivation that equal-size "
        "KV loss has different WCT exposure at Critical vs Non-Critical positions.",
    ),
    "fig_critical_backup_priority": (
        "sglsim_agentic_online_trace_placement_counterfactual",
        "stage-barrier online SGLSim traces support Critical-first GPUReplica "
        "placement as a motivation result.",
    ),
    "fig_critical_agent_profile": (
        "sglsim_agentic_stagebarrier_critical_profile",
        "stage-barrier online SGLSim traces show the long-decode fan-out "
        "agent is the Critical stage bottleneck in the 1-3-1-3-1 workload.",
    ),
    "fig_staged_alpha_risk_tradeoff": (
        "sglsim_agentic_stagebarrier_alpha_risk_counterfactual",
        "stage-barrier online SGLSim traces support the motivation that the "
        "risk-adjusted alpha optimum shifts as failure risk changes.",
    ),
}

HYBRID_MODEL_POLICY_STEMS: dict[str, tuple[str, str]] = {
    "fig_alpha_workload_failure_policy": (
        "aegiskv_alpha_mdp_value_iteration_v1",
        "offline MDP value-iteration policy table supports the online alpha "
        "controller's dependence on Critical pressure and failure risk.",
    ),
}

HYBRID_ONLINE_ALPHA_SWEEP_STEMS: dict[str, tuple[str, str]] = {
    "fig_online_alpha_adaptation": (
        "sglsim_agentic_online_mdp_alpha_sweep",
        "online SGLSim 1-3-1-3-1 serving sweep confirms alpha changes with "
        "workload pressure and failure-rate state.",
    ),
}

HYBRID_SIMULATOR_ACCURACY_STEMS: dict[str, tuple[str, str]] = {
    "fig_sglang_sglsim_accuracy": (
        "synthesis_nohicache_fa3_parity",
        "paired real SGLang fa3 inference and SGLSim wall-clock synthesis run "
        "support base serving-timing accuracy.",
    ),
}

HYBRID_SGLANG_HOST_RESTORE_STEMS: dict[str, tuple[str, str]] = {
    "fig_sglang_host_restore_materialization": (
        "sglang_physical_restore_parser_summary",
        "real SGLang server logs show Host restore materializes retry KV "
        "for visible runs while lookup-disabled controls stay hidden.",
    ),
}

HYBRID_PAIRING_REQUIRED_STEMS: dict[str, str] = {
    "fig_recovery_latency": (
        "needs paired real recovery measurements, not only replay arithmetic"
    ),
    "fig_normal_path_overhead": (
        "needs repeated matched SGLang/SGLSim policy overhead measurements"
    ),
    "fig_hostbackup_visibility": (
        "needs paired Host-visible/invisible recovery measurements"
    ),
    "fig_simulator_calibration": (
        "needs paired SGLang/SGLSim policy ranking measurements"
    ),
}

HYBRID_PAIRED_SOURCE_REQUIREMENTS: dict[str, tuple[tuple[str, str, int], ...]] = {
    "fig_recovery_latency": (
        (
            "fig_recovery_latency",
            "sglsim_agentic_online_recovery_latency_sweep",
            3,
        ),
        ("fig_hostbackup_visibility", "sglang_failure_visibility_pair_aggregate", 3),
    ),
    "fig_normal_path_overhead": (
        (
            "fig_normal_path_overhead",
            "sglsim_agentic_online_normal_path_overhead_sweep",
            3,
        ),
        ("fig_normal_path_overhead", "sglang_dp2_overhead_pair_aggregate", 1),
    ),
    "fig_hostbackup_visibility": (
        ("fig_hostbackup_visibility", "sglsim_replay_retry_counterfactual_plus_policy_overhead_anchor", 3),
        ("fig_hostbackup_visibility", "sglang_failure_visibility_pair_aggregate", 3),
    ),
    "fig_simulator_calibration": (
        ("fig_simulator_calibration", "sglang_sglsim_normal_path_overhead_anchor", 1),
        (
            "fig_recovery_latency",
            "sglsim_agentic_online_recovery_latency_sweep",
            3,
        ),
        ("fig_hostbackup_visibility", "sglang_failure_visibility_pair_aggregate", 3),
        ("fig_normal_path_overhead", "sglang_dp2_overhead_pair_aggregate", 1),
    ),
}

HYBRID_POLICY_SWEEP_REQUIRED_STEMS: dict[str, str] = {
    "fig_static_gpu_replica_budget_gap": (
        "needs implemented online budget sweep rather than replay counterfactual"
    ),
    "fig_overall_wct": (
        "needs final repeated online policy matrix, not replay counterfactual"
    ),
    "fig_ablation": (
        "needs implemented online ablation toggles rather than counterfactual rows"
    ),
    "fig_budget_sensitivity": (
        "needs controlled repeated failure-timing and budget sweeps"
    ),
    "fig_peer_rank_selection": (
        "needs implemented peer-selection policy sweep with route pressure counters"
    ),
}

HYBRID_ONLINE_POLICY_SWEEP_STEMS: dict[str, tuple[str, str]] = {
    "fig_static_gpu_replica_budget_gap": (
        "sglsim_agentic_online_budget_sweep",
        "implemented online SGLSim GPUReplica budget sweep with hard-restart replay",
    ),
    "fig_overall_wct": (
        "sglsim_agentic_online_policy_sweep",
        "implemented online SGLSim policy sweep with explicit gsp_aegis_policy",
    ),
    "fig_peer_rank_selection": (
        "sglsim_agentic_online_peer_route_sweep",
        "implemented online SGLSim peer-route sweep with retry route counters",
    ),
    "fig_ablation": (
        "sglsim_agentic_online_ablation_sweep",
        "implemented online SGLSim ablation sweep with explicit recovery toggles",
    ),
    "fig_budget_sensitivity": (
        "sglsim_agentic_online_budget_timing_sweep",
        "implemented online SGLSim failure-timing and GPUReplica budget sweep",
    ),
}


RESULT_COLUMNS: dict[str, tuple[str, ...]] = {
    "fig_motivation_equal_size_loss": ("segment", "delta_wct_norm"),
    "fig_critical_backup_priority": (
        "backup_budget_fraction",
        "placement_policy",
        "normalized_wct_after_failure",
    ),
    "fig_critical_agent_profile": (
        "stage_slot",
        "stage_id",
        "stage_width",
        "stage_local_agent_index",
        "critical_agent",
        "decode_dominant",
        "mean_output_tokens",
        "mean_latency_ms",
        "stage_critical_path_share",
    ),
    "fig_restore_recompute_boundary": (
        "tokens",
        "recompute_ms",
        "host_restore_ms",
        "gpu_replica_ms",
        "aegis_lower_envelope_ms",
    ),
    "fig_static_gpu_replica_budget_gap": (
        "gpu_replica_budget_fraction",
        "workload",
        "normalized_wct",
    ),
    "fig_alpha_workload_failure_policy": (
        "failure_rate_per_window",
        "critical_pressure",
        "alpha_backup_fraction",
    ),
    "fig_online_alpha_adaptation": (
        "request_class",
        "failure_rate_per_window",
        "critical_pressure",
        "policy_alpha",
        "effective_gpu_replica_alpha",
        "online_observed_alpha",
        "observed_request_rows",
    ),
    "fig_staged_alpha_risk_tradeoff": (
        "failure_rate_per_window",
        "alpha_backup_fraction",
        "normalized_expected_wct",
        "normal_backup_overhead_s",
        "expected_failure_recovery_s",
        "protected_critical_token_fraction",
        "optimal_alpha_for_rate",
    ),
    "fig_overall_wct": ("policy", "normalized_wct"),
    "fig_recovery_latency": (
        "policy",
        "retry_latency_norm",
        "critical_agent_latency_norm",
    ),
    "fig_recovery_path_breakdown": (
        "policy",
        "peer_gpu",
        "host_h2d",
        "ordinary_cache",
        "recompute",
    ),
    "fig_shared_prefix_marginal": (
        "shared_prefix_pct",
        "dup_no_dedup",
        "dup_marginal",
        "coverage_marginal",
    ),
    "fig_normal_path_overhead": (
        "policy",
        "p99_overhead_pct",
        "write_amplification",
    ),
    "fig_hostbackup_visibility": (
        "policy",
        "p99_overhead_relative_pct",
        "retry_latency_norm",
    ),
    "fig_ablation": ("variant", "normalized_wct"),
    "fig_budget_sensitivity": (
        "failure_timing",
        "gpu_replica_budget_fraction",
        "normalized_wct",
    ),
    "fig_dp_scale": ("dp_size", "workflow_count", "normalized_wct"),
    "fig_peer_rank_selection": (
        "policy",
        "normalized_wct",
        "peer_p99_overhead_pct",
    ),
    "fig_simulator_calibration": ("policy", "sglang_norm", "sglsim_norm"),
    "fig_physical_recovery_copy_calibration": (
        "path",
        "tokens",
        "p50_ms",
        "p50_ms_ci95",
    ),
    "fig_recovery_model_calibration_error": (
        "path",
        "tokens",
        "measured_copy_ms",
        "modeled_copy_ms",
        "abs_error_ms",
        "rel_error_pct",
    ),
    "fig_sglang_sglsim_accuracy": (
        "metric",
        "sglang_value",
        "sglsim_value",
        "ratio",
        "rel_error_pct",
        "within_10pct",
    ),
    "fig_p0_physical_bridge_coverage": (
        "suite_label",
        "restore_rows",
        "bridge_eligible_rows",
        "live_physical_tensor_copy_rows",
        "bridge_coverage_fraction",
        "live_copy_fraction",
    ),
    "fig_sglang_host_restore_materialization": (
        "state",
        "rows",
        "retry_cached_tokens_after_accept",
        "prefetch_io_completed_tokens",
        "max_data_pages",
        "materialized",
    ),
}


@dataclass
class FigureEvidence:
    figure: str
    stem: str
    kind: str
    manifest_status: str
    measured_csv: str | None
    measured_status: str
    evidence_tier: str
    row_count: int
    problems: list[str]

    @property
    def claim_ready(self) -> bool:
        if self.kind == "mechanism_diagram":
            return not self.problems
        return (
            self.measured_status == "measured-valid"
            and self.evidence_tier == "final"
            and not self.problems
        )


@dataclass
class HybridSupport:
    physical_recovery_ready: bool
    physical_recovery_summary: dict[str, dict[str, str]]
    live_audit_ready: bool
    live_audit_summary: dict[str, int | bool | dict[str, int]]
    problems: list[str]


@dataclass
class HybridFigureEvidence:
    figure: str
    stem: str
    kind: str
    strict_claim_ready: bool
    hybrid_ready: bool
    hybrid_status: str
    rationale: str
    requirements: list[str]
    caveats: list[str]
    problems: list[str]


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or ()), rows


def _read_rows_if_present(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    return _read_rows(path)


def _split_csv_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _positive_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_counter_map(value: str) -> dict[str, int]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    counters: dict[str, int] = {}
    for key, count in data.items():
        counters[str(key)] = _positive_int(str(count))
    return counters


def _validate_measured_csv(stem: str) -> tuple[str | None, str, str, int, list[str]]:
    expected = RESULT_COLUMNS.get(stem)
    if expected is None:
        return None, "not-required", "not-required", 0, []

    path = MEASURED_DIR / f"{stem}.csv"
    if not path.exists():
        return str(path.relative_to(ROOT)), "missing", "missing", 0, []
    if path.stat().st_size == 0:
        return str(path.relative_to(ROOT)), "empty", "missing", 0, ["measured CSV is empty"]

    fields, rows = _read_rows(path)
    problems: list[str] = []
    missing = [name for name in expected if name not in fields]
    if missing:
        problems.append("missing columns: " + ", ".join(missing))
    if not rows:
        problems.append("measured CSV has no data rows")
    tiers = {
        str(row.get("evidence_tier") or "final").strip().lower()
        for row in rows
    }
    tiers.discard("")
    evidence_tier = "final" if not tiers else ",".join(sorted(tiers))
    preliminary = bool(tiers and tiers != {"final"})
    if preliminary:
        for name in ("evidence_tier", "source", "run_id", "note"):
            if name not in fields:
                problems.append(f"missing provenance column for preliminary data: {name}")
        for idx, row in enumerate(rows, start=2):
            tier = str(row.get("evidence_tier") or "").strip().lower()
            if not tier or tier == "final":
                continue
            for name in ("source", "run_id", "note"):
                if not str(row.get(name) or "").strip():
                    problems.append(f"row {idx}: missing preliminary provenance {name}")
            if tier == "sglang-small-paired":
                for name in ("remote_host", "slurm_job_id", "model_path"):
                    if name not in fields:
                        problems.append(
                            f"missing SGLang provenance column: {name}"
                        )
                    elif not str(row.get(name) or "").strip():
                        problems.append(
                            f"row {idx}: missing SGLang provenance {name}"
                        )

    return (
        str(path.relative_to(ROOT)),
        (
            "preliminary-valid"
            if preliminary and not problems
            else "measured-valid"
            if not problems
            else "invalid"
        ),
        evidence_tier,
        len(rows),
        problems,
    )


def validate() -> tuple[list[FigureEvidence], list[str]]:
    manifest_path = ROOT / "manifest.json"
    if not manifest_path.exists():
        return [], [f"missing manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    global_problems: list[str] = []
    evidence: list[FigureEvidence] = []

    for entry in manifest:
        figure = str(entry["figure"])
        stem = str(entry["stem"])
        kind = str(entry["kind"])
        problems: list[str] = []

        outputs = entry.get("outputs", {})
        if not isinstance(outputs, dict):
            problems.append("manifest outputs is not a dict")
            outputs = {}
        for label, rel in outputs.items():
            path = ROOT / str(rel)
            if not path.exists():
                problems.append(f"missing output {label}: {rel}")
            elif path.stat().st_size == 0:
                problems.append(f"empty output {label}: {rel}")
            elif label == "drawio":
                try:
                    ET.parse(path)
                except ET.ParseError as exc:
                    problems.append(f"invalid drawio XML {rel}: {exc}")

        measured_csv, measured_status, evidence_tier, row_count, measured_problems = (
            _validate_measured_csv(stem)
        )
        problems.extend(measured_problems)
        evidence.append(
            FigureEvidence(
                figure=figure,
                stem=stem,
                kind=kind,
                manifest_status=str(entry.get("evidence_status", "")),
                measured_csv=measured_csv,
                measured_status=measured_status,
                evidence_tier=evidence_tier,
                row_count=row_count,
                problems=problems,
            )
        )

    return evidence, global_problems


def _check_physical_recovery_aggregate() -> tuple[bool, dict[str, dict[str, str]], list[str]]:
    problems: list[str] = []
    summary: dict[str, dict[str, str]] = {}
    required = {
        "host_restore_h2d": "Host restore H2D",
        "gpu_replica_restore_p2p": "GPUReplica restore P2P",
    }
    if not PHYSICAL_RECOVERY_AGGREGATE.exists():
        return False, summary, [f"missing {PHYSICAL_RECOVERY_AGGREGATE}"]
    fields, rows = _read_rows_if_present(PHYSICAL_RECOVERY_AGGREGATE)
    if not rows:
        return False, summary, ["physical recovery aggregate has no rows"]
    for column in (
        "action",
        "tokens",
        "repeat_count",
        "p50_ms_mean",
        "p50_ms_ci95",
        "slurm_job_ids",
        "hosts",
        "evidence_tier",
    ):
        if column not in fields:
            problems.append(f"physical recovery aggregate missing column: {column}")
    for action, label in required.items():
        matches = [
            row
            for row in rows
            if str(row.get("action") or "") == action
            and _positive_int(str(row.get("tokens") or "")) == 8192
        ]
        if not matches:
            problems.append(f"missing 8192-token physical row for {label}")
            continue
        row = matches[0]
        repeat_count = _positive_int(str(row.get("repeat_count") or ""))
        if repeat_count < 3:
            problems.append(f"{label} repeat_count={repeat_count}, expected >=3")
        if float(row.get("p50_ms_mean") or 0.0) <= 0.0:
            problems.append(f"{label} has non-positive p50_ms_mean")
        if not str(row.get("slurm_job_ids") or "").strip():
            problems.append(f"{label} missing slurm_job_ids")
        summary[action] = {
            "p50_ms_mean": str(row.get("p50_ms_mean") or ""),
            "p50_ms_ci95": str(row.get("p50_ms_ci95") or ""),
            "repeat_count": str(row.get("repeat_count") or ""),
            "slurm_job_ids": str(row.get("slurm_job_ids") or ""),
            "hosts": str(row.get("hosts") or ""),
            "evidence_tier": str(row.get("evidence_tier") or ""),
        }
    return not problems, summary, problems


def _check_live_movement_audit() -> tuple[bool, dict[str, int | bool | dict[str, int]], list[str]]:
    problems: list[str] = []
    summary: dict[str, int | bool | dict[str, int]] = {
        "request_rows": 0,
        "replay_rows": 0,
        "restore_rows": 0,
        "physical_tensor_copy_rows": 0,
        "nonfinal_restore_rows": 0,
        "final_ready": False,
        "data_planes": {},
        "timing_sources": {},
    }
    if not LIVE_MOVEMENT_AUDIT.exists():
        return False, summary, [f"missing {LIVE_MOVEMENT_AUDIT}"]
    fields, rows = _read_rows_if_present(LIVE_MOVEMENT_AUDIT)
    if not rows:
        return False, summary, ["live movement audit has no rows"]
    for column in (
        "request_rows",
        "replay_rows",
        "restore_rows",
        "physical_tensor_copy_rows",
        "nonfinal_restore_rows",
        "final_ready",
        "data_planes",
        "timing_sources",
    ):
        if column not in fields:
            problems.append(f"live movement audit missing column: {column}")
    data_planes: dict[str, int] = {}
    timing_sources: dict[str, int] = {}
    for row in rows:
        for key in (
            "request_rows",
            "replay_rows",
            "restore_rows",
            "physical_tensor_copy_rows",
            "nonfinal_restore_rows",
        ):
            summary[key] = int(summary.get(key, 0)) + _positive_int(
                str(row.get(key) or "")
            )
        for key, count in _parse_counter_map(str(row.get("data_planes") or "")).items():
            data_planes[key] = data_planes.get(key, 0) + count
        for key, count in _parse_counter_map(
            str(row.get("timing_sources") or "")
        ).items():
            timing_sources[key] = timing_sources.get(key, 0) + count
    summary["data_planes"] = data_planes
    summary["timing_sources"] = timing_sources
    summary["final_ready"] = bool(summary["restore_rows"]) and bool(
        summary["physical_tensor_copy_rows"] == summary["restore_rows"]
    )
    if not summary["restore_rows"]:
        problems.append("live movement audit has no restore rows")
    if timing_sources.get("physical_recovery_path_csv", 0) <= 0:
        problems.append("live movement audit lacks physical_recovery_path_csv timing")
    if data_planes.get("metadata_materialization", 0) <= 0:
        problems.append("live movement audit does not expose metadata_materialization")
    return not problems, summary, problems


def collect_hybrid_support() -> HybridSupport:
    physical_ready, physical_summary, physical_problems = (
        _check_physical_recovery_aggregate()
    )
    audit_ready, audit_summary, audit_problems = _check_live_movement_audit()
    return HybridSupport(
        physical_recovery_ready=physical_ready,
        physical_recovery_summary=physical_summary,
        live_audit_ready=audit_ready,
        live_audit_summary=audit_summary,
        problems=physical_problems + audit_problems,
    )


def _measured_rows_for_stem(stem: str) -> list[dict[str, str]]:
    _, rows = _read_rows_if_present(MEASURED_DIR / f"{stem}.csv")
    return rows


def _has_agentic_recovery_action_rows(stem: str) -> bool:
    for row in _measured_rows_for_stem(stem):
        if str(row.get("run_id") or "") != RECOVERY_ACTION_RUN_ID:
            continue
        if _positive_int(str(row.get("repeat_count") or "")) >= 3:
            return True
    return False


def _has_measured_source(stem: str, source: str) -> bool:
    for row in _measured_rows_for_stem(stem):
        if str(row.get("source") or "") != source:
            continue
        if _positive_int(str(row.get("repeat_count") or "1")) >= 1:
            return True
    return False


def _all_rows_within_10pct(stem: str) -> bool:
    rows = _measured_rows_for_stem(stem)
    if not rows:
        return False
    for row in rows:
        value = str(row.get("within_10pct") or "").strip().lower()
        if value in {"true", "1", "yes"}:
            continue
        try:
            ratio = float(row.get("ratio") or "nan")
        except ValueError:
            return False
        if abs(ratio - 1.0) > 0.10:
            return False
    return True


def _max_repeat_count_for_source(stem: str, source: str) -> int:
    max_repeat = 0
    for row in _measured_rows_for_stem(stem):
        if str(row.get("source") or "") != source:
            continue
        repeat_count = _positive_int(str(row.get("repeat_count") or "1"))
        max_repeat = max(max_repeat, repeat_count)
    return max_repeat


def _paired_source_problems(stem: str) -> list[str]:
    problems: list[str] = []
    for required_stem, source, min_repeat in HYBRID_PAIRED_SOURCE_REQUIREMENTS[stem]:
        repeat_count = _max_repeat_count_for_source(required_stem, source)
        if repeat_count < min_repeat:
            problems.append(
                f"{required_stem} missing source={source} repeat_count>={min_repeat}"
            )
    return problems


def build_hybrid_evidence(
    evidence: list[FigureEvidence], support: HybridSupport | None = None
) -> tuple[list[HybridFigureEvidence], HybridSupport]:
    if support is None:
        support = collect_hybrid_support()
    hybrid: list[HybridFigureEvidence] = []
    for item in evidence:
        requirements: list[str] = []
        caveats: list[str] = []
        problems = list(item.problems)
        hybrid_ready = item.claim_ready
        status = "strict-final-ready" if item.claim_ready else "not-ready"
        rationale = "strict final evidence already passes"

        if item.kind == "mechanism_diagram":
            hybrid_ready = not item.problems
            status = "design-ready" if hybrid_ready else "invalid-design-artifact"
            rationale = "mechanism diagram; measured CSV is not required"
        elif item.stem in HYBRID_TRACE_DERIVED_STEMS:
            source, source_rationale = HYBRID_TRACE_DERIVED_STEMS[item.stem]
            requirements.extend(
                [
                    "measured CSV must pass schema/provenance validation",
                    f"measured rows must include source={source}",
                    "source traces must use per-workflow stage barriers",
                ]
            )
            if item.measured_status not in {"measured-valid", "preliminary-valid"}:
                problems.append(f"measured status is {item.measured_status}")
            if not _has_measured_source(item.stem, source):
                problems.append(f"missing measured source {source}")
            hybrid_ready = not problems
            status = (
                "hybrid-trace-derived-motivation"
                if hybrid_ready
                else "needs-stagebarrier-trace-evidence"
            )
            rationale = source_rationale
            caveats.append(
                "Trace-derived motivation is not strict-final real-system failure evidence."
            )
        elif item.stem in HYBRID_MODEL_POLICY_STEMS:
            source, source_rationale = HYBRID_MODEL_POLICY_STEMS[item.stem]
            requirements.extend(
                [
                    "measured CSV must pass schema/provenance validation",
                    f"measured rows must include source={source}",
                    "offline policy table must be used by the online alpha controller",
                ]
            )
            if item.measured_status not in {"measured-valid", "preliminary-valid"}:
                problems.append(f"measured status is {item.measured_status}")
            if not _has_measured_source(item.stem, source):
                problems.append(f"missing measured source {source}")
            hybrid_ready = not problems
            status = (
                "hybrid-model-policy"
                if hybrid_ready
                else "needs-model-policy-evidence"
            )
            rationale = source_rationale
            caveats.append(
                "Model-based alpha policy evidence is not a measured serving sweep."
            )
        elif item.stem in HYBRID_ONLINE_ALPHA_SWEEP_STEMS:
            source, source_rationale = HYBRID_ONLINE_ALPHA_SWEEP_STEMS[item.stem]
            requirements.extend(
                [
                    "measured CSV must pass schema/provenance validation",
                    f"measured rows must include source={source}",
                    "online sweep rows must include observed alpha values",
                ]
            )
            if item.measured_status not in {"measured-valid", "preliminary-valid"}:
                problems.append(f"measured status is {item.measured_status}")
            if not _has_measured_source(item.stem, source):
                problems.append(f"missing measured source {source}")
            for row in _measured_rows_for_stem(item.stem):
                if str(row.get("source") or "") != source:
                    continue
                if not str(row.get("online_observed_alpha") or "").strip():
                    problems.append("online alpha sweep row missing observed alpha")
                    break
            hybrid_ready = not problems
            status = (
                "hybrid-online-alpha-sweep"
                if hybrid_ready
                else "needs-online-alpha-sweep"
            )
            rationale = source_rationale
            caveats.append(
                "Online SGLSim alpha sweep is not strict-final physical KV movement evidence."
            )
        elif item.stem in HYBRID_SIMULATOR_ACCURACY_STEMS:
            source, source_rationale = HYBRID_SIMULATOR_ACCURACY_STEMS[item.stem]
            requirements.extend(
                [
                    "measured CSV must pass schema/provenance validation",
                    f"measured rows must include source={source}",
                    "all reported ratios must be within 10% of real SGLang",
                ]
            )
            if item.measured_status not in {"measured-valid", "preliminary-valid"}:
                problems.append(f"measured status is {item.measured_status}")
            if not _has_measured_source(item.stem, source):
                problems.append(f"missing measured source {source}")
            if not _all_rows_within_10pct(item.stem):
                problems.append("at least one SGLSim/SGLang ratio exceeds 10%")
            hybrid_ready = not problems
            status = (
                "hybrid-simulator-accuracy"
                if hybrid_ready
                else "needs-simulator-accuracy-evidence"
            )
            rationale = source_rationale
            caveats.append(
                "This validates the base no-HiCache serving timing path; it is "
                "not end-to-end AegisKV failure-policy evidence."
            )
        elif item.stem in HYBRID_SGLANG_HOST_RESTORE_STEMS:
            source, source_rationale = HYBRID_SGLANG_HOST_RESTORE_STEMS[item.stem]
            rows = _measured_rows_for_stem(item.stem)
            states = {str(row.get("state") or "") for row in rows}
            materialized_rows = sum(
                _positive_int(str(row.get("rows") or ""))
                for row in rows
                if str(row.get("materialized") or "").strip().lower()
                in {"true", "1", "yes"}
            )
            hidden_rows = sum(
                _positive_int(str(row.get("rows") or ""))
                for row in rows
                if str(row.get("state") or "") == "host_frontier_not_visible_to_retry"
            )
            cached_tokens = sum(
                _positive_int(str(row.get("retry_cached_tokens_after_accept") or ""))
                for row in rows
            )
            requirements.extend(
                [
                    "measured CSV must pass schema/provenance validation",
                    f"measured rows must include source={source}",
                    "visible Host restore rows must materialize retry cached tokens",
                    "lookup-disabled control rows must stay hidden",
                ]
            )
            if item.measured_status not in {"measured-valid", "preliminary-valid"}:
                problems.append(f"measured status is {item.measured_status}")
            if not _has_measured_source(item.stem, source):
                problems.append(f"missing measured source {source}")
            if "full_prefetch_materialized" not in states:
                problems.append("missing full_prefetch_materialized state")
            if "partial_async_materialized" not in states:
                problems.append("missing partial_async_materialized state")
            if hidden_rows <= 0:
                problems.append("missing hidden-control rows")
            if materialized_rows <= 0 or cached_tokens <= 0:
                problems.append("Host restore rows do not materialize retry KV")
            hybrid_ready = not problems
            status = (
                "hybrid-sglang-host-restore"
                if hybrid_ready
                else "needs-sglang-host-restore-evidence"
            )
            rationale = source_rationale
            caveats.append(
                "This is real SGLang Host restore materialization evidence, "
                "not P0 large-sweep live physical_tensor_copy."
            )
        elif item.stem in HYBRID_PHYSICAL_CALIBRATED_STEMS:
            requirements.extend(
                [
                    "measured CSV must pass schema/provenance validation",
                    "8192-token Host and GPUReplica physical restore microbench",
                    "live audit must disclose metadata_materialization data plane",
                ]
            )
            if item.measured_status not in {"measured-valid", "preliminary-valid"}:
                problems.append(f"measured status is {item.measured_status}")
            if not support.physical_recovery_ready:
                problems.append("physical recovery aggregate is not ready")
            if not support.live_audit_ready:
                problems.append("live movement audit is not ready")
            if item.stem in HYBRID_AGENTIC_RUN_STEMS and not _has_agentic_recovery_action_rows(
                item.stem
            ):
                problems.append(
                    f"missing {RECOVERY_ACTION_RUN_ID} measured rows"
                )
            hybrid_ready = not problems
            status = (
                "hybrid-physical-calibrated"
                if hybrid_ready
                else "hybrid-calibration-incomplete"
            )
            rationale = HYBRID_PHYSICAL_CALIBRATED_STEMS[item.stem]
            caveats.append(
                "SGLSim remains metadata-only; live recovery rows use "
                "metadata_materialization with physical_recovery_path_csv timing."
            )
            caveats.append(
                "This is not the same as strict-final live physical_tensor_copy."
            )
        elif item.stem in HYBRID_PAIRING_REQUIRED_STEMS:
            requirements.extend(
                [
                    "measured CSV must pass schema/provenance validation",
                    "paired SGLang small-run anchors must be present",
                    "matching SGLSim online/replay policy rows must be present",
                ]
            )
            if item.measured_status not in {"measured-valid", "preliminary-valid"}:
                problems.append(f"measured status is {item.measured_status}")
            problems.extend(_paired_source_problems(item.stem))
            hybrid_ready = not problems
            status = "hybrid-paired-anchor" if hybrid_ready else "needs-paired-measurements"
            rationale = (
                "paired SGLang small-run anchors plus matching SGLSim policy "
                "evidence are present"
                if hybrid_ready
                else HYBRID_PAIRING_REQUIRED_STEMS[item.stem]
            )
            caveats.append(
                "Hybrid paired anchors are not strict-final end-to-end physical "
                "CUDA KV-copy recovery measurements."
            )
        elif item.stem in HYBRID_ONLINE_POLICY_SWEEP_STEMS:
            source, source_rationale = HYBRID_ONLINE_POLICY_SWEEP_STEMS[item.stem]
            requirements.extend(
                [
                    "measured CSV must pass schema/provenance validation",
                    f"measured rows must include source={source}",
                ]
            )
            if item.measured_status not in {"measured-valid", "preliminary-valid"}:
                problems.append(f"measured status is {item.measured_status}")
            if not _has_measured_source(item.stem, source):
                problems.append(f"missing measured source {source}")
            hybrid_ready = not problems
            status = "hybrid-online-policy-sweep" if hybrid_ready else "needs-online-policy-sweep"
            rationale = source_rationale if hybrid_ready else HYBRID_POLICY_SWEEP_REQUIRED_STEMS[item.stem]
        elif item.stem in HYBRID_POLICY_SWEEP_REQUIRED_STEMS:
            hybrid_ready = False
            status = "needs-online-policy-sweep"
            rationale = HYBRID_POLICY_SWEEP_REQUIRED_STEMS[item.stem]

        hybrid.append(
            HybridFigureEvidence(
                figure=item.figure,
                stem=item.stem,
                kind=item.kind,
                strict_claim_ready=item.claim_ready,
                hybrid_ready=hybrid_ready,
                hybrid_status=status,
                rationale=rationale,
                requirements=requirements,
                caveats=caveats,
                problems=problems,
            )
        )
    return hybrid, support


def write_status(evidence: list[FigureEvidence], global_problems: list[str]) -> None:
    payload = {
        "global_problems": global_problems,
        "summary": {
            "figures": len(evidence),
            "claim_ready": sum(1 for item in evidence if item.claim_ready),
            "result_figures": sum(1 for item in evidence if item.kind != "mechanism_diagram"),
            "measured_valid": sum(
                1 for item in evidence if item.measured_status == "measured-valid"
            ),
            "preliminary_valid": sum(
                1 for item in evidence if item.measured_status == "preliminary-valid"
            ),
        },
        "figures": [
            {
                "figure": item.figure,
                "stem": item.stem,
                "kind": item.kind,
                "manifest_status": item.manifest_status,
                "measured_csv": item.measured_csv,
                "measured_status": item.measured_status,
                "evidence_tier": item.evidence_tier,
                "row_count": item.row_count,
                "claim_ready": item.claim_ready,
                "problems": item.problems,
            }
            for item in evidence
        ],
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def write_hybrid_status(
    hybrid: list[HybridFigureEvidence],
    support: HybridSupport,
    global_problems: list[str],
) -> None:
    result_figures = [item for item in hybrid if item.kind != "mechanism_diagram"]
    blocked = [item for item in result_figures if not item.hybrid_ready]
    payload = {
        "summary": {
            "figures": len(hybrid),
            "hybrid_ready": sum(1 for item in hybrid if item.hybrid_ready),
            "strict_final_ready": sum(1 for item in hybrid if item.strict_claim_ready),
            "result_figures": len(result_figures),
            "hybrid_ready_result_figures": sum(
                1 for item in result_figures if item.hybrid_ready
            ),
            "blocked_result_figures": len(blocked),
        },
        "support": {
            "physical_recovery_ready": support.physical_recovery_ready,
            "physical_recovery_aggregate": str(
                PHYSICAL_RECOVERY_AGGREGATE.relative_to(DRAFT_ROOT)
            ),
            "physical_recovery_summary": support.physical_recovery_summary,
            "live_audit_ready": support.live_audit_ready,
            "live_movement_audit": str(LIVE_MOVEMENT_AUDIT.relative_to(DRAFT_ROOT)),
            "live_audit_summary": support.live_audit_summary,
            "problems": support.problems,
        },
        "global_problems": global_problems,
        "figures": [
            {
                "figure": item.figure,
                "stem": item.stem,
                "kind": item.kind,
                "strict_claim_ready": item.strict_claim_ready,
                "hybrid_ready": item.hybrid_ready,
                "hybrid_status": item.hybrid_status,
                "rationale": item.rationale,
                "requirements": item.requirements,
                "caveats": item.caveats,
                "problems": item.problems,
            }
            for item in hybrid
        ],
    }
    HPCA_HYBRID_STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# HPCA Hybrid Evidence Status",
        "",
        "This report is generated by "
        "`figures/scripts/validate_figure_evidence.py --write-hpca-hybrid`.",
        "",
        "The hybrid gate is intentionally weaker than `--strict-final`: it accepts "
        "large-scale online SGLSim evidence only when the data plane boundary is "
        "explicitly disclosed and the recovery timing is anchored by small "
        "physical SGLang/torch CUDA movement measurements.",
        "",
        "## Summary",
        "",
        f"- Figures: {len(hybrid)}",
        f"- Strict-final-ready figures: {payload['summary']['strict_final_ready']}",
        f"- Hybrid-ready figures: {payload['summary']['hybrid_ready']}",
        f"- Quantitative result figures: {len(result_figures)}",
        f"- Hybrid-ready result figures: {payload['summary']['hybrid_ready_result_figures']}",
        f"- Blocked result figures: {len(blocked)}",
        "",
        "## Calibration Support",
        "",
        f"- Physical recovery aggregate ready: {support.physical_recovery_ready}",
        f"- Live movement audit ready: {support.live_audit_ready}",
        "- SGLSim data plane: metadata materialization with calibrated timing; "
        "not live CUDA tensor movement.",
        "",
        "## Per-Figure Hybrid Gate",
        "",
        "| Figure | Stem | Status | Hybrid ready | Rationale |",
        "|---|---|---|---|---|",
    ]
    for item in hybrid:
        lines.append(
            "| {figure} | `{stem}` | `{status}` | {ready} | {rationale} |".format(
                figure=item.figure,
                stem=item.stem,
                status=item.hybrid_status,
                ready="yes" if item.hybrid_ready else "no",
                rationale=item.rationale.replace("|", "/"),
            )
        )
    remaining_text = (
        "The HPCA hybrid gate has no remaining blocked figures. Remaining "
        "paper-strengthening work is strict-final evidence: replace preliminary "
        "SGLSim metadata-materialization rows with final end-to-end measurements "
        "that use physical KV movement or explicitly validated real-system "
        "recovery paths."
        if not blocked
        else (
            "The blocked rows are the figures that still need paired real-system "
            "measurements or implemented online policy sweeps before the draft "
            "can be called HPCA-ready under the hybrid evidence bar."
        )
    )
    lines.extend(["", "## Remaining HPCA Work", "", remaining_text, ""])
    for item in blocked:
        lines.append(f"- {item.figure} `{item.stem}`: {item.rationale}")
    if support.problems or global_problems:
        lines.extend(["", "## Validator Problems", ""])
        for problem in global_problems + support.problems:
            lines.append(f"- {problem}")
    HPCA_HYBRID_MARKDOWN_PATH.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-final",
        action="store_true",
        help="Require every figure, including result figures, to be claim-ready.",
    )
    parser.add_argument(
        "--write-hpca-hybrid",
        action="store_true",
        help="Write the HPCA hybrid evidence status JSON and Markdown reports.",
    )
    parser.add_argument(
        "--strict-hpca-hybrid",
        action="store_true",
        help="Require every figure to pass the HPCA hybrid evidence gate.",
    )
    args = parser.parse_args(argv)

    evidence, global_problems = validate()
    write_status(evidence, global_problems)
    hybrid: list[HybridFigureEvidence] = []
    if args.write_hpca_hybrid or args.strict_hpca_hybrid:
        hybrid, support = build_hybrid_evidence(evidence)
        write_hybrid_status(hybrid, support, global_problems)

    result_missing = [
        item
        for item in evidence
        if item.kind != "mechanism_diagram" and not item.claim_ready
    ]
    broken = [item for item in evidence if item.problems]

    print(f"wrote {STATUS_PATH}")
    print(f"figures={len(evidence)} claim_ready={sum(item.claim_ready for item in evidence)}")
    preliminary = sum(1 for item in evidence if item.measured_status == "preliminary-valid")
    if preliminary:
        print(f"result_figures_with_preliminary_data={preliminary}")
    print(f"result_figures_without_final_measured_data={len(result_missing)}")
    if args.write_hpca_hybrid or args.strict_hpca_hybrid:
        hybrid_missing = [item for item in hybrid if not item.hybrid_ready]
        print(f"wrote {HPCA_HYBRID_STATUS_PATH}")
        print(f"wrote {HPCA_HYBRID_MARKDOWN_PATH}")
        print(
            "hpca_hybrid_ready="
            f"{sum(item.hybrid_ready for item in hybrid)}/{len(hybrid)}"
        )
        print(f"hpca_hybrid_blocked={len(hybrid_missing)}")
    if broken:
        print("invalid artifacts:")
        for item in broken:
            print(f"  {item.figure} {item.stem}: {'; '.join(item.problems)}")
    if global_problems:
        print("global problems:")
        for problem in global_problems:
            print(f"  {problem}")

    if broken or global_problems:
        return 1
    if args.strict_final and result_missing:
        print("strict-final failed: final measured data is still missing for result figures")
        return 2
    if args.strict_hpca_hybrid and any(not item.hybrid_ready for item in hybrid):
        print("strict-hpca-hybrid failed: hybrid evidence is still incomplete")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
