#!/usr/bin/env python3
"""Assess only the formal positive-BOUNDED opportunity sub-gate.

This consumes the JSON output of ``score_bounded_opportunity.py``.  It refuses
development rows, schema rejections, mixed collection contracts, and an
unfrozen/mismatched preregistration.  The workflow fraction is inferred at the
workflow unit with a Wilson interval.  The decision-snapshot KV-byte ratio uses
a deterministic workflow-cluster bootstrap, so repeated decisions from one
workflow never become independent samples.

Passing this tool is deliberately *not* a G1 pass.  Action crossing and EDF
dispatch-change prevalence are absent from the v3 bounded-scope schema, and G0
matched-baseline disagreement is a separate study.  The output names those
missing obligations instead of silently upgrading a coverage result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable


PREREGISTRATION_FIELDS = frozenset(
    {
        "schema_version",
        "preregistration_id",
        "frozen_before_test_open",
        "input_schema_version",
        "dataset_id",
        "split_id",
        "sampling_policy_id",
        "confidence_level",
        "bootstrap_replicates",
        "bootstrap_seed",
        "minimum_all_workflows",
        "minimum_eligible_workflows",
        "eligible_workflow_fraction_lower_threshold",
        "eligible_kv_byte_fraction_lower_threshold",
    }
)


class AssessmentRefused(ValueError):
    """The artifact cannot support a formal opportunity assessment."""


def _strict_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AssessmentRefused(f"{name} must be an integer >= {minimum}")
    return value


def _finite_fraction(name: str, value: object, *, open_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssessmentRefused(f"{name} must be a number")
    parsed = float(value)
    lower_ok = parsed > 0.0 if open_zero else parsed >= 0.0
    if not math.isfinite(parsed) or not lower_ok or parsed > 1.0:
        interval = "(0, 1]" if open_zero else "[0, 1]"
        raise AssessmentRefused(f"{name} must be finite and in {interval}")
    return parsed


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssessmentRefused(f"{name} must be a nonempty string")
    return value.strip()


@dataclass(frozen=True)
class OpportunityPreregistration:
    preregistration_id: str
    input_schema_version: int
    dataset_id: str
    split_id: str
    sampling_policy_id: str
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    minimum_all_workflows: int
    minimum_eligible_workflows: int
    eligible_workflow_fraction_lower_threshold: float
    eligible_kv_byte_fraction_lower_threshold: float


def parse_preregistration(raw: dict[str, Any]) -> OpportunityPreregistration:
    missing = sorted(PREREGISTRATION_FIELDS - raw.keys())
    extra = sorted(raw.keys() - PREREGISTRATION_FIELDS)
    if missing or extra:
        raise AssessmentRefused(
            f"preregistration fields mismatch: missing={missing}, extra={extra}"
        )
    if raw["schema_version"] != 1:
        raise AssessmentRefused("preregistration schema_version must equal 1")
    if raw["frozen_before_test_open"] is not True:
        raise AssessmentRefused(
            "preregistration must be frozen before the formal test split opens"
        )
    confidence = _finite_fraction(
        "confidence_level", raw["confidence_level"], open_zero=True
    )
    if confidence >= 1.0:
        raise AssessmentRefused("confidence_level must be less than 1")
    return OpportunityPreregistration(
        preregistration_id=_nonempty(
            "preregistration_id", raw["preregistration_id"]
        ),
        input_schema_version=_strict_int(
            "input_schema_version", raw["input_schema_version"], minimum=1
        ),
        dataset_id=_nonempty("dataset_id", raw["dataset_id"]),
        split_id=_nonempty("split_id", raw["split_id"]),
        sampling_policy_id=_nonempty(
            "sampling_policy_id", raw["sampling_policy_id"]
        ),
        confidence_level=confidence,
        bootstrap_replicates=_strict_int(
            "bootstrap_replicates",
            raw["bootstrap_replicates"],
            minimum=1_000,
        ),
        bootstrap_seed=_strict_int(
            "bootstrap_seed", raw["bootstrap_seed"], minimum=0
        ),
        minimum_all_workflows=_strict_int(
            "minimum_all_workflows",
            raw["minimum_all_workflows"],
            minimum=1,
        ),
        minimum_eligible_workflows=_strict_int(
            "minimum_eligible_workflows",
            raw["minimum_eligible_workflows"],
            minimum=1,
        ),
        eligible_workflow_fraction_lower_threshold=_finite_fraction(
            "eligible_workflow_fraction_lower_threshold",
            raw["eligible_workflow_fraction_lower_threshold"],
        ),
        eligible_kv_byte_fraction_lower_threshold=_finite_fraction(
            "eligible_kv_byte_fraction_lower_threshold",
            raw["eligible_kv_byte_fraction_lower_threshold"],
        ),
    )


def _wilson_interval(
    successes: int, total: int, confidence_level: float
) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise AssessmentRefused("Wilson interval needs 0 <= successes <= total")
    alpha = 1.0 - confidence_level
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise AssessmentRefused("cannot take a quantile of an empty sample")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


def _cluster_ratio_interval(
    clusters: tuple[tuple[int, int], ...],
    *,
    confidence_level: float,
    replicates: int,
    seed: int,
) -> tuple[float, float, float, int]:
    numerator = sum(item[0] for item in clusters)
    denominator = sum(item[1] for item in clusters)
    if denominator <= 0:
        raise AssessmentRefused(
            "eligible formal workflows contain zero active KV-byte denominator"
        )
    point = numerator / denominator
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(replicates):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sample_denominator = sum(item[1] for item in sampled)
        if sample_denominator <= 0:
            continue
        ratios.append(
            sum(item[0] for item in sampled) / sample_denominator
        )
    if len(ratios) < math.ceil(0.99 * replicates):
        raise AssessmentRefused(
            "too many cluster-bootstrap replicates had zero KV denominator"
        )
    alpha = 1.0 - confidence_level
    return (
        point,
        _quantile(ratios, alpha / 2.0),
        _quantile(ratios, 1.0 - alpha / 2.0),
        len(ratios),
    )


def _require_row_bool(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    if type(value) is not bool:
        raise AssessmentRefused(f"formal row {field} must be a boolean")
    return value


def _require_row_bytes(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AssessmentRefused(
            f"formal row {field} must be a nonnegative integer"
        )
    return value


def assess_scored_result(
    scored: dict[str, Any],
    preregistration_raw: dict[str, Any],
) -> dict[str, Any]:
    prereg = parse_preregistration(preregistration_raw)
    if scored.get("schema_version") != prereg.input_schema_version:
        raise AssessmentRefused("scored artifact schema version mismatch")
    if scored.get("gate_assessment") != "NOT_ASSESSED":
        raise AssessmentRefused("input scorer must not pre-claim a gate result")
    if scored.get("unsupported_records") != 0:
        raise AssessmentRefused(
            "formal assessment refuses every schema-rejected input row"
        )
    if scored.get("formal_input_consistency") != "PASS":
        raise AssessmentRefused("formal collection contract is absent or mixed")
    rows_value = scored.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise AssessmentRefused("scored artifact contains no formal rows")
    if scored.get("input_records") != len(rows_value):
        raise AssessmentRefused("scored artifact row count is inconsistent")
    if scored.get("supported_records") != len(rows_value):
        raise AssessmentRefused("not every input row was supported")

    expected_contract = (
        prereg.dataset_id,
        prereg.split_id,
        prereg.sampling_policy_id,
    )
    contract_rows = scored.get("formal_collection_contracts")
    if contract_rows != [
        {
            "dataset_id": expected_contract[0],
            "split_id": expected_contract[1],
            "sampling_policy_id": expected_contract[2],
        }
    ]:
        raise AssessmentRefused(
            "formal artifact does not match the frozen collection contract"
        )

    seen_decisions: set[tuple[str, str, str]] = set()
    workflows: dict[str, dict[str, Any]] = {}
    for row_value in rows_value:
        if not isinstance(row_value, dict):
            raise AssessmentRefused("formal row must be an object")
        row = row_value
        if row.get("evidence_class") != "held_out_formal":
            raise AssessmentRefused(
                "formal assessment refuses development or mixed-evidence rows"
            )
        if row.get("status") not in {"SCORED", "NO_PROOF_SCOPE"}:
            raise AssessmentRefused("formal row lacks a scored zero/positive state")
        if (
            row.get("dataset_id"),
            row.get("split_id"),
            row.get("sampling_policy_id"),
        ) != expected_contract:
            raise AssessmentRefused("formal row collection contract mismatch")
        workflow_id = _nonempty("workflow_id", row.get("workflow_id"))
        decision_id = _nonempty("decision_id", row.get("decision_id"))
        failure_domain = _nonempty(
            "failure_domain", row.get("failure_domain")
        )
        decision_key = (workflow_id, decision_id, failure_domain)
        if decision_key in seen_decisions:
            raise AssessmentRefused("duplicate formal decision identity")
        seen_decisions.add(decision_key)

        eligible = _require_row_bool(
            row, "eligible_declared_parallel_region"
        )
        positive = _require_row_bool(
            row, "has_positive_bounded_deadline"
        )
        active_bytes = _require_row_bytes(
            row, "failure_exposed_active_kv_bytes_total"
        )
        positive_bytes = _require_row_bytes(
            row, "positive_bounded_kv_bytes"
        )
        if positive_bytes > active_bytes:
            raise AssessmentRefused(
                "positive bounded KV bytes exceed the sampling-frame denominator"
            )
        if not eligible and positive:
            raise AssessmentRefused(
                "an ineligible row cannot contribute positive opportunity"
            )

        workflow = workflows.setdefault(
            workflow_id,
            {
                "eligible": False,
                "positive": False,
                "all_active_bytes": 0,
                "all_positive_bytes": 0,
                "eligible_active_bytes": 0,
                "eligible_positive_bytes": 0,
                "workflow_templates": set(),
                "models": set(),
                "layouts": set(),
            },
        )
        workflow["workflow_templates"].add(
            _nonempty("workflow_template", row.get("workflow_template"))
        )
        workflow["models"].add(_nonempty("model", row.get("model")))
        workflow["layouts"].add(
            _nonempty("layout_digest", row.get("layout_digest"))
        )
        workflow["positive"] = workflow["positive"] or positive
        workflow["all_active_bytes"] += active_bytes
        workflow["all_positive_bytes"] += positive_bytes
        if eligible:
            workflow["eligible"] = True
            workflow["eligible_active_bytes"] += active_bytes
            workflow["eligible_positive_bytes"] += positive_bytes

    for workflow_id, workflow in workflows.items():
        for field in ("workflow_templates", "models", "layouts"):
            if len(workflow[field]) != 1:
                raise AssessmentRefused(
                    f"workflow {workflow_id} changes {field} within the frozen split"
                )

    eligible_workflows = [
        item for item in workflows.values() if item["eligible"]
    ]
    positive_workflows = sum(item["positive"] for item in eligible_workflows)
    all_count = len(workflows)
    all_positive_count = sum(item["positive"] for item in workflows.values())
    eligible_count = len(eligible_workflows)
    if all_count < prereg.minimum_all_workflows:
        sample_size_status = "FAIL_MINIMUM_ALL_WORKFLOWS"
    elif eligible_count < prereg.minimum_eligible_workflows:
        sample_size_status = "FAIL_MINIMUM_ELIGIBLE_WORKFLOWS"
    else:
        sample_size_status = "PASS"
    if eligible_count == 0:
        raise AssessmentRefused("formal artifact contains no eligible workflows")

    workflow_point = positive_workflows / eligible_count
    workflow_lower, workflow_upper = _wilson_interval(
        positive_workflows, eligible_count, prereg.confidence_level
    )
    unconditional_workflow_point = all_positive_count / all_count
    unconditional_workflow_lower, unconditional_workflow_upper = (
        _wilson_interval(
            all_positive_count, all_count, prereg.confidence_level
        )
    )
    kv_point, kv_lower, kv_upper, valid_replicates = _cluster_ratio_interval(
        tuple(
            (
                int(item["eligible_positive_bytes"]),
                int(item["eligible_active_bytes"]),
            )
            for item in eligible_workflows
        ),
        confidence_level=prereg.confidence_level,
        replicates=prereg.bootstrap_replicates,
        seed=prereg.bootstrap_seed,
    )
    (
        unconditional_kv_point,
        unconditional_kv_lower,
        unconditional_kv_upper,
        unconditional_valid_replicates,
    ) = _cluster_ratio_interval(
        tuple(
            (
                int(item["all_positive_bytes"]),
                int(item["all_active_bytes"]),
            )
            for item in workflows.values()
        ),
        confidence_level=prereg.confidence_level,
        replicates=prereg.bootstrap_replicates,
        seed=prereg.bootstrap_seed,
    )
    workflow_pass = (
        workflow_lower
        >= prereg.eligible_workflow_fraction_lower_threshold
    )
    kv_pass = kv_lower >= prereg.eligible_kv_byte_fraction_lower_threshold
    subgate_pass = sample_size_status == "PASS" and workflow_pass and kv_pass

    strata = {
        "workflow_templates": sorted(
            {next(iter(item["workflow_templates"])) for item in workflows.values()}
        ),
        "models": sorted(
            {next(iter(item["models"])) for item in workflows.values()}
        ),
        "layouts": sorted(
            {next(iter(item["layouts"])) for item in workflows.values()}
        ),
    }
    return {
        "schema_version": 1,
        "evidence_class": "held_out_formal",
        "assessment_scope": "G1_positive_bounded_opportunity_only",
        "gate_assessment": (
            "OPPORTUNITY_SUBGATE_PASS"
            if subgate_pass
            else "OPPORTUNITY_SUBGATE_FAIL"
        ),
        "overall_g1_status": (
            "NOT_ASSESSED_ACTIONABILITY_AND_EDF_CHANGE_MISSING"
            if subgate_pass
            else "FAIL_OPPORTUNITY_SUBGATE"
        ),
        "not_assessed": [
            "protected_vs_unprotected_action_crossing",
            "edf_dispatch_or_preemption_change",
            "matched_baseline_action_set_disagreement_G0",
            "realized_WCT_or_SLO_benefit_G4",
        ],
        "preregistration_id": prereg.preregistration_id,
        "collection_contract": {
            "dataset_id": prereg.dataset_id,
            "split_id": prereg.split_id,
            "sampling_policy_id": prereg.sampling_policy_id,
        },
        "sample_size_status": sample_size_status,
        "all_workflows": all_count,
        "all_positive_workflows": all_positive_count,
        "eligible_workflows": eligible_count,
        "eligible_positive_workflows": positive_workflows,
        "unconditional_workflow_fraction": {
            "point": unconditional_workflow_point,
            "lower": unconditional_workflow_lower,
            "upper": unconditional_workflow_upper,
            "confidence_level": prereg.confidence_level,
            "method": "workflow_unit_Wilson",
            "role": "mandatory_scope_disclosure_not_the_conditional_gate",
        },
        "unconditional_decision_snapshot_kv_byte_fraction": {
            "point": unconditional_kv_point,
            "lower": unconditional_kv_lower,
            "upper": unconditional_kv_upper,
            "confidence_level": prereg.confidence_level,
            "method": "workflow_cluster_percentile_bootstrap",
            "bootstrap_replicates_requested": prereg.bootstrap_replicates,
            "bootstrap_replicates_valid": unconditional_valid_replicates,
            "bootstrap_seed": prereg.bootstrap_seed,
            "role": "mandatory_scope_disclosure_not_the_conditional_gate",
        },
        "eligible_workflow_fraction": {
            "point": workflow_point,
            "lower": workflow_lower,
            "upper": workflow_upper,
            "confidence_level": prereg.confidence_level,
            "method": "workflow_unit_Wilson",
            "threshold_lower": (
                prereg.eligible_workflow_fraction_lower_threshold
            ),
            "passes": workflow_pass,
        },
        "eligible_decision_snapshot_kv_byte_fraction": {
            "point": kv_point,
            "lower": kv_lower,
            "upper": kv_upper,
            "confidence_level": prereg.confidence_level,
            "method": "workflow_cluster_percentile_bootstrap",
            "bootstrap_replicates_requested": prereg.bootstrap_replicates,
            "bootstrap_replicates_valid": valid_replicates,
            "bootstrap_seed": prereg.bootstrap_seed,
            "threshold_lower": prereg.eligible_kv_byte_fraction_lower_threshold,
            "passes": kv_pass,
        },
        "observed_strata": strata,
        "evidence_boundary": (
            "Coverage sub-gate only. Passing cannot be relabelled as G1, G0, "
            "G4, physical data-plane, or end-to-end evidence."
        ),
    }


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise AssessmentRefused(f"{path} must contain one JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-json", type=Path, required=True)
    parser.add_argument("--preregistration-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    scored, scored_sha256 = _read_json(args.scored_json)
    preregistration, preregistration_sha256 = _read_json(
        args.preregistration_json
    )
    result = assess_scored_result(scored, preregistration)
    result["scored_json"] = str(args.scored_json)
    result["scored_sha256"] = scored_sha256
    result["preregistration_json"] = str(args.preregistration_json)
    result["preregistration_sha256"] = preregistration_sha256
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
