"""Fail-closed cost/effect accounting for current-design AegisKV.

The runtime remains metadata-only.  This module validates already-produced
runtime snapshots and paired effect rows; it neither loads a trace nor runs a
simulation.  Recovery-hit counts are accepted only when bound to an eligible,
same-lineage physical artifact or profile plus a PASS mechanism gate.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS


OUTPUT_CONTRACT = "aegiskv_cost_effect_accounting_v1"
COMPONENT_COST_MODEL_SCHEMA = "affine_setup_first_transfer_warmup_v1"
PATH_CONTRACT = {
    "source_to_peer_hbm": {
        "tier": "HBM",
        "copy_kind": "source_d2d",
        "recovery_hit_key": "peer_hbm",
    },
    "source_to_store_dram": {
        "tier": "DRAM",
        "copy_kind": "source_d2h",
        "recovery_hit_key": "dram",
    },
}
COMPONENT_NAMES = (
    "target_allocation",
    "source_pack",
    "engine_transfer",
    "target_finish",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if parsed < 0 or parsed != value:
        raise ValueError(f"{name} must be a nonnegative integer")
    return parsed


def _finite_nonnegative(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def _finite_positive(name: str, value: Any) -> float:
    parsed = _finite_nonnegative(name, value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _require_sha256(name: str, value: Any) -> str:
    parsed = str(value)
    if _SHA256_RE.fullmatch(parsed) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return parsed


def _sum_count_map(name: str, value: Any) -> Counter[str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    output: Counter[str] = Counter()
    for key, count in value.items():
        if not str(key):
            raise ValueError(f"{name} contains an empty key")
        output[str(key)] += _nonnegative_int(f"{name}.{key}", count)
    return output


def _validate_hit_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_lineage_id: str,
    expected_hardware_label: str,
) -> dict[str, Any]:
    required = {
        "evidence_kind",
        "lineage_id",
        "hardware_label",
        "preflight_sha256",
        "source_manifest_sha256",
        "config_manifest_sha256",
        "config_verification_sha256",
        "physical_profile_sha256",
        "recovery_hit_artifact_sha256",
        "mechanism_gate_sha256",
        "mechanism_gate_status",
        "same_lineage",
        "simulator_rate_default_eligible",
        "component_breakdown_declared",
        "runtime_counter_source",
        "hit_counts",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ValueError(f"recovery-hit evidence is missing {missing}")
    kind = str(evidence["evidence_kind"])
    if kind not in {"physical_observed", "profile_derived"}:
        raise ValueError("recovery-hit evidence kind is not physical/profile derived")
    if evidence["lineage_id"] != expected_lineage_id:
        raise ValueError("recovery-hit evidence crosses the runtime lineage")
    if evidence["hardware_label"] != expected_hardware_label:
        raise ValueError("recovery-hit evidence crosses the hardware lineage")
    if evidence["same_lineage"] is not True:
        raise ValueError("recovery-hit evidence is not same-lineage")
    if evidence["mechanism_gate_status"] != "PASS":
        raise ValueError("recovery-hit evidence lacks a PASS mechanism gate")
    if evidence["simulator_rate_default_eligible"] is not True:
        raise ValueError("recovery-hit evidence is ineligible for matched replay")
    if not isinstance(evidence["component_breakdown_declared"], bool):
        raise ValueError("component_breakdown_declared must be boolean")
    counter_source = evidence["runtime_counter_source"]
    observed_output: dict[str, Any] | None = None
    if kind == "profile_derived":
        if counter_source not in {
            "recovery_estimate_action_counts",
            "recovery_execution_action_counts",
        }:
            raise ValueError(
                "profile-derived hits require an exact runtime counter source"
            )
        if "observed_successful_protection" in evidence:
            raise ValueError(
                "profile-derived hits cannot mix physical copy observations"
            )
    else:
        if counter_source is not None:
            raise ValueError("physical hits cannot claim a simulator counter source")
        observed = evidence.get("observed_successful_protection")
        if not isinstance(observed, Mapping) or set(observed) != set(PATH_CONTRACT):
            raise ValueError(
                "physical hits require the exact physical protection population"
            )
        observed_output = {}
        for path_id, expected in PATH_CONTRACT.items():
            row = observed[path_id]
            if not isinstance(row, Mapping) or set(row) != {
                "tier",
                "copy_kind",
                "successful_copies",
                "successful_bytes",
            }:
                raise ValueError(
                    f"physical protection population {path_id} schema mismatch"
                )
            if (
                row["tier"] != expected["tier"]
                or row["copy_kind"] != expected["copy_kind"]
            ):
                raise ValueError(
                    f"physical protection population {path_id} identity mismatch"
                )
            observed_output[path_id] = {
                "tier": expected["tier"],
                "copy_kind": expected["copy_kind"],
                "successful_copies": _nonnegative_int(
                    f"observed_successful_protection.{path_id}.successful_copies",
                    row["successful_copies"],
                ),
                "successful_bytes": _nonnegative_int(
                    f"observed_successful_protection.{path_id}.successful_bytes",
                    row["successful_bytes"],
                ),
            }
            if bool(observed_output[path_id]["successful_copies"]) != bool(
                observed_output[path_id]["successful_bytes"]
            ):
                raise ValueError(
                    f"physical protection population {path_id} count/bytes mismatch"
                )
    hits = evidence["hit_counts"]
    if not isinstance(hits, Mapping) or set(hits) != {"peer_hbm", "dram"}:
        raise ValueError("recovery hit_counts must contain peer_hbm and dram exactly")
    return {
        "evidence_kind": kind,
        "lineage_id": expected_lineage_id,
        "hardware_label": expected_hardware_label,
        **{
            name: _require_sha256(name, evidence[name])
            for name in (
                "preflight_sha256",
                "source_manifest_sha256",
                "config_manifest_sha256",
                "config_verification_sha256",
                "physical_profile_sha256",
                "recovery_hit_artifact_sha256",
            )
        },
        "mechanism_gate_sha256": _require_sha256(
            "mechanism_gate_sha256", evidence["mechanism_gate_sha256"]
        ),
        "mechanism_gate_status": "PASS",
        "same_lineage": True,
        "simulator_rate_default_eligible": True,
        "component_breakdown_declared": evidence[
            "component_breakdown_declared"
        ],
        "runtime_counter_source": counter_source,
        "hit_counts": {
            key: _nonnegative_int(f"hit_counts.{key}", hits[key])
            for key in ("peer_hbm", "dram")
        },
        "observed_successful_protection": observed_output,
    }


def _aggregate_runtime_snapshots(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Counter[str],
    Counter[str],
]:
    if not snapshots:
        raise ValueError("at least one finalized runtime snapshot is required")
    paths: dict[str, Counter[str]] = {
        path_id: Counter() for path_id in PATH_CONTRACT
    }
    path_service_ms = {path_id: 0.0 for path_id in PATH_CONTRACT}
    pools: dict[str, Counter[str]] = {
        tier: Counter() for tier in ("HBM", "DRAM")
    }
    pool_rejections = {tier: Counter() for tier in ("HBM", "DRAM")}
    pool_evictions = {tier: Counter() for tier in ("HBM", "DRAM")}
    pool_removals = {tier: Counter() for tier in ("HBM", "DRAM")}
    recovery_estimates: Counter[str] = Counter()
    recovery_executions: Counter[str] = Counter()
    count_fields = (
        "attempted_copies",
        "attempted_bytes",
        "successful_copies",
        "successful_bytes",
        "failed_copies",
        "cancelled_copies",
        "pending_copies",
        "first_transfer_warmup_count",
        "healthy_period_attempted_copies",
        "healthy_period_attempted_bytes",
        "healthy_period_first_transfer_warmup_count",
    )
    churn_fields = (
        "admission_attempt_count",
        "admission_success_count",
        "admission_rejection_count",
        "admitted_bytes",
        "eviction_count",
        "evicted_bytes",
        "removal_count",
        "removed_bytes",
        "churn_event_count",
        "churn_bytes",
    )
    for rank, snapshot in sorted(snapshots.items(), key=lambda item: str(item[0])):
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"runtime snapshot {rank} is not an object")
        if snapshot.get("design_semantics") != DESIGN_SEMANTICS:
            raise ValueError(f"runtime snapshot {rank} uses legacy design semantics")
        if _nonnegative_int(
            f"runtime snapshot {rank}.pending_transfer_jobs",
            snapshot.get("pending_transfer_jobs"),
        ) != 0:
            raise ValueError("runtime accounting requires a finalized transfer queue")
        observed_paths = snapshot.get("protection_paths")
        if not isinstance(observed_paths, Mapping) or set(observed_paths) != set(
            PATH_CONTRACT
        ):
            raise ValueError("runtime snapshot lacks the exact protection path set")
        for path_id, expected in PATH_CONTRACT.items():
            row = observed_paths[path_id]
            if not isinstance(row, Mapping):
                raise ValueError(f"protection path {path_id} is not an object")
            if (
                row.get("path_id") != path_id
                or row.get("tier") != expected["tier"]
                or row.get("copy_kind") != expected["copy_kind"]
            ):
                raise ValueError(f"protection path {path_id} identity mismatch")
            parsed = {
                name: _nonnegative_int(f"{path_id}.{name}", row.get(name))
                for name in count_fields
            }
            service_ms = _finite_nonnegative(
                f"{path_id}.healthy_period_transfer_service_ms",
                row.get("healthy_period_transfer_service_ms"),
            )
            terminal = sum(
                parsed[name]
                for name in (
                    "successful_copies",
                    "failed_copies",
                    "cancelled_copies",
                    "pending_copies",
                )
            )
            if parsed["attempted_copies"] != terminal:
                raise ValueError(f"protection path {path_id} copies do not reconcile")
            if parsed["successful_bytes"] > parsed["attempted_bytes"]:
                raise ValueError(f"protection path {path_id} successful bytes overflow")
            if (
                bool(parsed["attempted_copies"]) != bool(parsed["attempted_bytes"])
                or bool(parsed["successful_copies"])
                != bool(parsed["successful_bytes"])
                or bool(parsed["healthy_period_attempted_copies"])
                != bool(parsed["healthy_period_attempted_bytes"])
            ):
                raise ValueError(
                    f"protection path {path_id} copy counts and bytes disagree"
                )
            if (
                parsed["healthy_period_attempted_copies"]
                > parsed["attempted_copies"]
                or parsed["healthy_period_attempted_bytes"]
                > parsed["attempted_bytes"]
            ):
                raise ValueError(f"protection path {path_id} healthy work overflows")
            if (
                parsed["first_transfer_warmup_count"] > 1
                or parsed["first_transfer_warmup_count"]
                > parsed["attempted_copies"]
                or parsed["healthy_period_first_transfer_warmup_count"]
                > parsed["first_transfer_warmup_count"]
                or parsed["healthy_period_first_transfer_warmup_count"]
                > parsed["healthy_period_attempted_copies"]
            ):
                raise ValueError(
                    f"protection path {path_id} warm-up counts do not reconcile"
                )
            paths[path_id].update(parsed)
            path_service_ms[path_id] += service_ms
        observed_pools = snapshot.get("pool_churn")
        if not isinstance(observed_pools, Mapping) or set(observed_pools) != {
            "HBM",
            "DRAM",
        }:
            raise ValueError("runtime snapshot lacks exact per-pool churn")
        for tier in ("HBM", "DRAM"):
            row = observed_pools[tier]
            if not isinstance(row, Mapping):
                raise ValueError(f"pool churn {tier} is not an object")
            parsed = {
                name: _nonnegative_int(f"pool_churn.{tier}.{name}", row.get(name))
                for name in churn_fields
            }
            if parsed["admission_attempt_count"] != (
                parsed["admission_success_count"]
                + parsed["admission_rejection_count"]
            ):
                raise ValueError(f"pool churn {tier} admissions do not reconcile")
            if parsed["churn_event_count"] != (
                parsed["admission_success_count"] + parsed["eviction_count"]
            ):
                raise ValueError(f"pool churn {tier} event count does not reconcile")
            if parsed["churn_bytes"] != (
                parsed["admitted_bytes"] + parsed["evicted_bytes"]
            ):
                raise ValueError(f"pool churn {tier} bytes do not reconcile")
            if (
                parsed["eviction_count"] > parsed["removal_count"]
                or parsed["evicted_bytes"] > parsed["removed_bytes"]
            ):
                raise ValueError(f"pool churn {tier} evictions exceed removals")
            rejection_reasons = _sum_count_map(
                f"pool_churn.{tier}.rejection_reason_counts",
                row.get("rejection_reason_counts"),
            )
            eviction_reasons = _sum_count_map(
                f"pool_churn.{tier}.eviction_reason_counts",
                row.get("eviction_reason_counts"),
            )
            removal_reasons = _sum_count_map(
                f"pool_churn.{tier}.removal_reason_counts",
                row.get("removal_reason_counts"),
            )
            if sum(rejection_reasons.values()) != parsed["admission_rejection_count"]:
                raise ValueError(
                    f"pool churn {tier} rejection reasons do not reconcile"
                )
            if sum(eviction_reasons.values()) != parsed["eviction_count"]:
                raise ValueError(f"pool churn {tier} eviction reasons do not reconcile")
            if sum(removal_reasons.values()) != parsed["removal_count"]:
                raise ValueError(f"pool churn {tier} removal reasons do not reconcile")
            pools[tier].update(parsed)
            pool_rejections[tier].update(rejection_reasons)
            pool_evictions[tier].update(eviction_reasons)
            pool_removals[tier].update(removal_reasons)
        recovery_estimates.update(
            _sum_count_map(
                f"runtime snapshot {rank}.recovery_estimate_action_counts",
                snapshot.get("recovery_estimate_action_counts"),
            )
        )
        recovery_executions.update(
            _sum_count_map(
                f"runtime snapshot {rank}.recovery_execution_action_counts",
                snapshot.get("recovery_execution_action_counts"),
            )
        )
    path_output: dict[str, Any] = {}
    for path_id, totals in paths.items():
        path_output[path_id] = {
            **dict(sorted(totals.items())),
            "tier": PATH_CONTRACT[path_id]["tier"],
            "copy_kind": PATH_CONTRACT[path_id]["copy_kind"],
            "healthy_period_transfer_service_ms": path_service_ms[path_id],
        }
    pool_output = {
        tier: {
            **dict(sorted(pools[tier].items())),
            "rejection_reason_counts": dict(sorted(pool_rejections[tier].items())),
            "eviction_reason_counts": dict(sorted(pool_evictions[tier].items())),
            "removal_reason_counts": dict(sorted(pool_removals[tier].items())),
        }
        for tier in ("HBM", "DRAM")
    }
    return path_output, pool_output, recovery_estimates, recovery_executions


def _component_breakdowns(
    component_profiles: Mapping[str, Mapping[str, Any]] | None,
    *,
    paths: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    supplied = component_profiles or {}
    if not isinstance(supplied, Mapping):
        raise ValueError("component profiles must be an object")
    unknown = sorted(set(supplied) - set(PATH_CONTRACT))
    if unknown:
        raise ValueError(f"component profiles contain unknown paths {unknown}")
    declared = bool(evidence["component_breakdown_declared"])
    if declared and not supplied:
        raise ValueError("declared component breakdown has no available path")
    if not declared and supplied:
        raise ValueError("component breakdown was supplied but not declared")
    output: dict[str, Any] = {}
    missing_fields: list[str] = []
    for path_id in PATH_CONTRACT:
        if path_id not in supplied:
            output[path_id] = {
                "available": False,
                "cost_model_schema": COMPONENT_COST_MODEL_SCHEMA,
                "components": None,
            }
            for component in COMPONENT_NAMES:
                missing_fields.extend(
                    (
                        f"{path_id}.{component}.per_transfer_setup_ms",
                        f"{path_id}.{component}.first_transfer_warmup_ms",
                        f"{path_id}.{component}.steady_state_bytes_per_ms",
                    )
                )
            continue
        profile = supplied[path_id]
        if not isinstance(profile, Mapping):
            raise ValueError(f"component profile {path_id} is not an object")
        if (
            profile.get("lineage_id") != evidence["lineage_id"]
            or profile.get("hardware_label") != evidence["hardware_label"]
            or profile.get("cost_model_schema") != COMPONENT_COST_MODEL_SCHEMA
            or profile.get("profile_sha256")
            != evidence["physical_profile_sha256"]
        ):
            raise ValueError(f"component profile {path_id} crosses profile lineage")
        components = profile.get("components")
        if not isinstance(components, Mapping) or set(components) != set(
            COMPONENT_NAMES
        ):
            raise ValueError(f"component profile {path_id} is partial")
        path = paths[path_id]
        copies = int(path["healthy_period_attempted_copies"])
        byte_count = int(path["healthy_period_attempted_bytes"])
        first_transfers = int(
            path["healthy_period_first_transfer_warmup_count"]
        )
        totals: dict[str, Any] = {}
        modeled_total = 0.0
        for component in COMPONENT_NAMES:
            row = components[component]
            if not isinstance(row, Mapping) or set(row) != {
                "per_transfer_setup_ms",
                "first_transfer_warmup_ms",
                "steady_state_bytes_per_ms",
            }:
                raise ValueError(f"component {path_id}.{component} schema mismatch")
            setup = _finite_nonnegative(
                f"{path_id}.{component}.per_transfer_setup_ms",
                row["per_transfer_setup_ms"],
            )
            warmup = _finite_nonnegative(
                f"{path_id}.{component}.first_transfer_warmup_ms",
                row["first_transfer_warmup_ms"],
            )
            rate_raw = row["steady_state_bytes_per_ms"]
            rate = (
                None
                if rate_raw is None
                else _finite_positive(
                    f"{path_id}.{component}.steady_state_bytes_per_ms",
                    rate_raw,
                )
            )
            total = (
                copies * setup
                + first_transfers * warmup
                + (0.0 if rate is None else byte_count / rate)
            )
            modeled_total += total
            totals[component] = {
                "per_transfer_setup_ms": setup,
                "first_transfer_warmup_ms": warmup,
                "steady_state_bytes_per_ms": rate,
                "healthy_period_total_ms": total,
            }
        observed = float(path["healthy_period_transfer_service_ms"])
        tolerance = max(1e-6, observed * 1e-9)
        if not math.isclose(modeled_total, observed, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"component profile {path_id} does not reconcile with service cost"
            )
        output[path_id] = {
            "available": True,
            "profile_sha256": profile["profile_sha256"],
            "cost_model_schema": COMPONENT_COST_MODEL_SCHEMA,
            "components": totals,
            "healthy_period_total_ms": modeled_total,
        }
    return output, missing_fields


def _paired_effect_contract(
    effect_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not effect_pairs:
        raise ValueError("paired effects must retain at least one complete pair")
    required = {
        "pair_id",
        "baseline_whole_workload_wct_ms",
        "aegis_whole_workload_wct_ms",
        "baseline_failure_local_ms",
        "aegis_failure_local_ms",
        "included",
        "performance_direction_used_for_inclusion",
    }
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(effect_pairs):
        if not isinstance(pair, Mapping):
            raise ValueError(f"effect pair {index} is not an object")
        missing = sorted(required - set(pair))
        if missing:
            raise ValueError(f"effect pair {index} is missing {missing}")
        pair_id = str(pair["pair_id"])
        if not pair_id or pair_id in seen:
            raise ValueError("effect pair IDs must be nonempty and unique")
        seen.add(pair_id)
        if pair["included"] is not True:
            raise ValueError("all structurally valid effect pairs must be retained")
        if pair["performance_direction_used_for_inclusion"] is not False:
            raise ValueError("effect-pair inclusion must be direction blind")
        baseline_whole = _finite_positive(
            f"effect pair {pair_id}.baseline_whole_workload_wct_ms",
            pair["baseline_whole_workload_wct_ms"],
        )
        aegis_whole = _finite_positive(
            f"effect pair {pair_id}.aegis_whole_workload_wct_ms",
            pair["aegis_whole_workload_wct_ms"],
        )
        baseline_failure = _finite_nonnegative(
            f"effect pair {pair_id}.baseline_failure_local_ms",
            pair["baseline_failure_local_ms"],
        )
        aegis_failure = _finite_nonnegative(
            f"effect pair {pair_id}.aegis_failure_local_ms",
            pair["aegis_failure_local_ms"],
        )
        whole_delta = aegis_whole - baseline_whole
        failure_delta = aegis_failure - baseline_failure
        whole_direction = (
            "improvement"
            if whole_delta < -1e-12
            else "regression"
            if whole_delta > 1e-12
            else "null"
        )
        failure_direction = (
            "benefit"
            if failure_delta < -1e-12
            else "harm"
            if failure_delta > 1e-12
            else "null"
        )
        rows.append(
            {
                "pair_id": pair_id,
                "baseline_whole_workload_wct_ms": baseline_whole,
                "aegis_whole_workload_wct_ms": aegis_whole,
                "whole_workload_wct_change_ms": whole_delta,
                "whole_workload_direction": whole_direction,
                "baseline_failure_local_ms": baseline_failure,
                "aegis_failure_local_ms": aegis_failure,
                "failure_local_change_ms": failure_delta,
                "failure_local_direction": failure_direction,
                "included": True,
                "performance_direction_used_for_inclusion": False,
            }
        )
    count = len(rows)
    baseline_whole_total = sum(
        row["baseline_whole_workload_wct_ms"] for row in rows
    )
    aegis_whole_total = sum(row["aegis_whole_workload_wct_ms"] for row in rows)
    baseline_failure_total = sum(row["baseline_failure_local_ms"] for row in rows)
    aegis_failure_total = sum(row["aegis_failure_local_ms"] for row in rows)
    whole_change = aegis_whole_total - baseline_whole_total
    failure_change = aegis_failure_total - baseline_failure_total
    failure_benefit = failure_change < -1e-12
    whole_improvement = whole_change < -1e-12
    whole_regression = whole_change > 1e-12
    direction_counts = {
        "whole_workload": dict(
            sorted(Counter(row["whole_workload_direction"] for row in rows).items())
        ),
        "failure_local": dict(
            sorted(Counter(row["failure_local_direction"] for row in rows).items())
        ),
    }
    if failure_benefit and whole_regression:
        label = "failure_local_benefit_with_whole_workload_regression"
    elif failure_benefit and whole_improvement:
        label = "joint_whole_workload_and_failure_local_improvement"
    elif failure_benefit:
        label = "failure_local_benefit_without_whole_workload_improvement"
    elif whole_improvement:
        label = "whole_workload_improvement_without_failure_local_benefit"
    else:
        label = "no_joint_advantage"
    return {
        "pair_count": count,
        "pairs": rows,
        "aggregate": {
            "baseline_whole_workload_wct_ms": baseline_whole_total,
            "aegis_whole_workload_wct_ms": aegis_whole_total,
            "whole_workload_wct_change_ms": whole_change,
            "baseline_failure_local_ms": baseline_failure_total,
            "aegis_failure_local_ms": aegis_failure_total,
            "failure_local_change_ms": failure_change,
        },
        "direction_counts": direction_counts,
        "claim_gate": {
            "failure_local_benefit": failure_benefit,
            "whole_workload_improvement": whole_improvement,
            "whole_workload_regression": whole_regression,
            "joint_directional_improvement": (
                failure_benefit and whole_improvement
            ),
            "general_advantage_supported": False,
            "performance_claim_authorized": False,
            "general_claim_blockers": [
                "metadata_only_descriptive_accounting_without_formal_inference",
                *(
                    ["whole_workload_wct_regression"]
                    if whole_regression
                    else []
                ),
            ],
            "classification": label,
        },
    }


def _amplification(
    successful_copies: int,
    successful_bytes: int,
    hits: int,
    *,
    protection_population_source: str,
) -> dict[str, Any]:
    if hits == 0:
        return {
            "recovery_hits": 0,
            "successful_protection_copies": successful_copies,
            "successful_protection_bytes": successful_bytes,
            "protection_population_source": protection_population_source,
            "amplification_defined": False,
            "successful_copies_per_hit": None,
            "successful_bytes_per_hit": None,
        }
    return {
        "recovery_hits": hits,
        "successful_protection_copies": successful_copies,
        "successful_protection_bytes": successful_bytes,
        "protection_population_source": protection_population_source,
        "amplification_defined": True,
        "successful_copies_per_hit": successful_copies / hits,
        "successful_bytes_per_hit": successful_bytes / hits,
    }


def build_aegis_output_contract(
    *,
    policy: str,
    runtime_snapshots: Mapping[str, Mapping[str, Any]],
    effect_pairs: Sequence[Mapping[str, Any]],
    recovery_hit_evidence: Mapping[str, Any],
    expected_lineage_id: str,
    expected_hardware_label: str,
    component_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate and aggregate one policy's finalized metadata output."""

    if not str(policy):
        raise ValueError("policy must be nonempty")
    if not str(expected_lineage_id) or not str(expected_hardware_label):
        raise ValueError("expected lineage and hardware labels are required")
    expected_lineage_id = _require_sha256(
        "expected_lineage_id", expected_lineage_id
    )
    paths, pools, profile_estimates, profile_executions = (
        _aggregate_runtime_snapshots(runtime_snapshots)
    )
    evidence = _validate_hit_evidence(
        recovery_hit_evidence,
        expected_lineage_id=expected_lineage_id,
        expected_hardware_label=expected_hardware_label,
    )
    if evidence["evidence_kind"] == "profile_derived":
        counter_source = str(evidence["runtime_counter_source"])
        selected_counter = (
            profile_estimates
            if counter_source == "recovery_estimate_action_counts"
            else profile_executions
        )
        expected_hits = {
            "peer_hbm": int(selected_counter["PEER_HBM"]),
            "dram": int(selected_counter["DRAM"]),
        }
        if evidence["hit_counts"] != expected_hits:
            raise ValueError(
                "profile-derived recovery hits do not match runtime decisions"
            )
    components, missing_component_fields = _component_breakdowns(
        component_profiles,
        paths=paths,
        evidence=evidence,
    )
    effects = _paired_effect_contract(effect_pairs)
    protection_tiers = {
        tier: {
            "path_ids": sorted(
                path_id
                for path_id, row in paths.items()
                if row["tier"] == tier
            ),
            "successful_copies": sum(
                int(row["successful_copies"])
                for row in paths.values()
                if row["tier"] == tier
            ),
            "successful_bytes": sum(
                int(row["successful_bytes"])
                for row in paths.values()
                if row["tier"] == tier
            ),
        }
        for tier in ("HBM", "DRAM")
    }
    amplification: dict[str, Any] = {}
    total_copies = 0
    total_bytes = 0
    total_hits = 0
    if evidence["evidence_kind"] == "profile_derived":
        protection_population = paths
        population_source = "metadata_runtime_successful_protection"
    else:
        protection_population = evidence["observed_successful_protection"]
        population_source = "same_artifact_physical_successful_protection"
    for path_id, contract in PATH_CONTRACT.items():
        successful_copies = int(
            protection_population[path_id]["successful_copies"]
        )
        successful_bytes = int(
            protection_population[path_id]["successful_bytes"]
        )
        hits = int(evidence["hit_counts"][contract["recovery_hit_key"]])
        amplification[path_id] = _amplification(
            successful_copies,
            successful_bytes,
            hits,
            protection_population_source=population_source,
        )
        total_copies += successful_copies
        total_bytes += successful_bytes
        total_hits += hits
    amplification["all_protection_paths"] = _amplification(
        total_copies,
        total_bytes,
        total_hits,
        protection_population_source=population_source,
    )
    healthy_service = {
        path_id: {
            "attempted_copies": int(row["healthy_period_attempted_copies"]),
            "attempted_bytes": int(row["healthy_period_attempted_bytes"]),
            "first_transfer_warmup_count": int(
                row["healthy_period_first_transfer_warmup_count"]
            ),
            "transfer_service_ms": float(
                row["healthy_period_transfer_service_ms"]
            ),
        }
        for path_id, row in paths.items()
    }
    healthy_service["all_protection_paths"] = {
        "attempted_copies": sum(
            row["attempted_copies"] for row in healthy_service.values()
        ),
        "attempted_bytes": sum(
            row["attempted_bytes"] for row in healthy_service.values()
        ),
        "first_transfer_warmup_count": sum(
            row["first_transfer_warmup_count"]
            for row in healthy_service.values()
        ),
        "transfer_service_ms": sum(
            row["transfer_service_ms"] for row in healthy_service.values()
        ),
    }
    return {
        "status": "PASS",
        "schema": OUTPUT_CONTRACT,
        "design_semantics": DESIGN_SEMANTICS,
        "policy": str(policy),
        "lineage_id": expected_lineage_id,
        "hardware_label": expected_hardware_label,
        "metadata_only": True,
        "cuda_or_nccl_executed": False,
        "physical_kv_bytes_moved_by_sglsim": False,
        "authorization": {
            "replay_authorized": False,
            "figure_generation_authorized": False,
            "performance_claim_authorized": False,
            "formal_ready": False,
            "paper_ready": False,
        },
        "evidence_boundary": (
            "Validated metadata accounting only; no CUDA/NCCL or physical KV "
            "movement is performed by this contract."
        ),
        "protection_paths": paths,
        "protection_tiers": protection_tiers,
        "recovery_hit_evidence": evidence,
        "runtime_recovery_action_counts": {
            "estimated": dict(sorted(profile_estimates.items())),
            "executed": dict(sorted(profile_executions.items())),
        },
        "copy_to_hit_amplification": amplification,
        "per_pool_churn": pools,
        "healthy_period_transfer_service_cost": healthy_service,
        "transfer_service_components": components,
        "component_cost_model_schema": COMPONENT_COST_MODEL_SCHEMA,
        "profile_component_breakdown_complete": not missing_component_fields,
        "remaining_profile_dependent_fields": missing_component_fields,
        "joint_effect_contract": effects,
    }


__all__ = [
    "COMPONENT_COST_MODEL_SCHEMA",
    "COMPONENT_NAMES",
    "OUTPUT_CONTRACT",
    "PATH_CONTRACT",
    "build_aegis_output_contract",
]
