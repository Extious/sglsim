#!/usr/bin/env python3
"""Train the simulation-declared causal Algorithm-1 point predictor.

This is distinct from the raw Weka api/think-time predictor.  Labels here are
remaining service under the frozen SGLSim model, so accuracy measures fidelity
to declared simulator coefficients and must not be generalized to hardware.
Future service is label-only: every target request is predicted from its
currently visible frontier metadata by a model fitted on disjoint workflows.

The current AegisKV design applies CPM to point predictions ``hat(w)``.  A
split-conformal absolute-error interval is retained as an uncertainty
diagnostic, but its lower Slack bound is deliberately not a policy input: the
global absolute margin is vacuous for this heterogeneous trace and collapses
the central mechanism to all-zero Slack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.agentic_workflows.weka_full_trace import workflow_fold
from sglsim.srt.simulation.aegis_gbdt import AegisGBDTConfig, AegisGBDTRegressor


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocol.json"
COMPACT = ROOT / "cache" / "weka_compact_v1"
OUTPUT = ROOT / "cache" / "modeled_algorithm1_v1"
PRODUCER = Path(__file__).resolve()

# These are mechanism-validity thresholds, fixed before any performance run.
# They reject an accidentally all-zero Algorithm-1 input; they are not an
# accuracy or performance claim and do not select a favorable predictor.
MIN_NONZERO_FRACTION_ALL_REQUESTS = 0.05
MIN_NONZERO_FRACTION_ELIGIBLE_FRONTIERS = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _branch_and_round(request_id: str) -> tuple[int, int]:
    tail = request_id.rsplit("-p", 1)[1]
    if "-m" in tail:
        return 0, int(tail.rsplit("-m", 1)[1])
    branch_part, round_part = tail.rsplit("-s", 1)[1].split("-t", 1)
    return int(branch_part) + 1, int(round_part)


def _regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    positive = actual > 1e-9
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "count": int(len(actual)),
        "mae_ms": float(np.mean(np.abs(error))),
        "mape": (
            float(np.mean(np.abs(error[positive]) / actual[positive]))
            if np.any(positive)
            else 0.0
        ),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else 0.0,
    }


def _positive_prediction_diagnostics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    actual_positive = actual > 1e-9
    predicted_positive = predicted > 1e-9
    true_positive = int(np.sum(actual_positive & predicted_positive))
    false_positive = int(np.sum(~actual_positive & predicted_positive))
    false_negative = int(np.sum(actual_positive & ~predicted_positive))
    true_negative = int(np.sum(~actual_positive & ~predicted_positive))
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "positive_precision": true_positive
        / max(true_positive + false_positive, 1),
        "positive_recall": true_positive
        / max(true_positive + false_negative, 1),
        "signed_bias_ms": float(np.mean(predicted - actual)),
        "underprediction_fraction": float(np.mean(predicted < actual - 1e-9)),
        "overprediction_fraction": float(np.mean(predicted > actual + 1e-9)),
    }


def _upper_quantile(values: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = math.ceil((len(ordered) + 1) * coverage)
    return float(ordered[min(max(rank - 1, 0), len(ordered) - 1)])


def prepare(
    protocol_path: Path,
    compact_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    algorithm1_protocol = protocol["algorithm1_sensitivity"]
    protocol_gate = algorithm1_protocol["nondegenerate_mechanism_gate"]
    if algorithm1_protocol.get("central_policy_input_array") != "predicted_slack_ms":
        raise ValueError("protocol must select predicted_slack_ms for central m1")
    if float(protocol_gate["minimum_nonzero_fraction_all_requests"]) != (
        MIN_NONZERO_FRACTION_ALL_REQUESTS
    ):
        raise ValueError("protocol/core all-request nonzero threshold mismatch")
    if float(protocol_gate["minimum_nonzero_fraction_eligible_frontiers"]) != (
        MIN_NONZERO_FRACTION_ELIGIBLE_FRONTIERS
    ):
        raise ValueError("protocol/core eligible-frontier threshold mismatch")
    compact_manifest_path = compact_dir / "manifest.json"
    compact_manifest = json.loads(compact_manifest_path.read_text(encoding="utf-8"))
    if output_dir.exists():
        if not force:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    def load(name: str) -> np.ndarray:
        return np.load(compact_dir / compact_manifest["arrays"][name]["path"], mmap_mode="r")

    request_id = load("request_id")
    workflow_index = load("workflow_index")
    stage_id = load("stage_id")
    role = load("role")
    arrival = load("arrival_ms")
    prompt_raw = load("prompt_tokens_raw")
    model = protocol["models"]["qwen3_8b"]
    scale = float(protocol["workload"]["central_prompt_scale"])
    output_cap = int(protocol["workload"]["output_cap_tokens"])
    output_raw = load("output_tokens_raw")
    reuse_raw = load("input_reuse_tokens_raw")
    prompt = np.maximum(np.ceil(prompt_raw.astype(float) * scale), 1.0)
    reuse = np.minimum(
        np.rint(reuse_raw.astype(float) * scale), prompt
    )
    uncached_prompt = np.maximum(prompt - reuse, 0.0)
    output = np.minimum(output_raw.astype(float), output_cap)
    profile = model["simulated_compute_profile"]
    service = (
        float(profile["prefill_setup_ms"])
        + uncached_prompt * float(profile["prefill_ms_per_token"])
        + output * float(profile["decode_ms_per_token"])
    )

    count = len(request_id)
    branch = np.empty(count, dtype=np.int16)
    round_index = np.empty(count, dtype=np.int16)
    for index, rid in enumerate(request_id):
        branch[index], round_index[index] = _branch_and_round(str(rid))

    branch_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    round_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index in range(count):
        key = (int(workflow_index[index]), int(stage_id[index]), int(branch[index]))
        branch_groups[key].append(index)
        round_groups[
            (
                int(workflow_index[index]),
                int(stage_id[index]),
                int(round_index[index]),
            )
        ].append(index)

    # Reconstruct the causal active-frontier visibility incrementally.  At
    # round r, only branches observed at or before r are part of the visible
    # residual DAG.  If a previously visible branch has no current node, the
    # frontier is incomplete and Algorithm 1 must publish zero Slack.  This is
    # equivalent to an online cumulative update; no future branch identity is
    # consulted by an earlier snapshot.
    rounds_by_stage: dict[tuple[int, int], list[tuple[int, int, int]]] = (
        defaultdict(list)
    )
    for key in round_groups:
        rounds_by_stage[key[:2]].append(key)
    active_frontier_width = np.zeros(count, dtype=float)
    complete_visible_frontier = np.zeros(count, dtype=bool)
    for keys in rounds_by_stage.values():
        visible_branches: set[int] = set()
        for key in sorted(keys, key=lambda value: value[2]):
            indices = round_groups[key]
            current_branches = [int(branch[index]) for index in indices]
            current_set = set(current_branches)
            visible_branches.update(current_set)
            active_frontier_width[indices] = float(len(current_set))
            complete = (
                len(current_set) > 1
                and len(current_branches) == len(current_set)
                and current_set == visible_branches
            )
            if complete:
                complete_visible_frontier[indices] = True

    actual_remaining = np.zeros(count, dtype=float)
    elapsed = np.zeros(count, dtype=float)
    max_revealed_prompt = np.zeros(count, dtype=float)
    sum_revealed_prompt = np.zeros(count, dtype=float)
    for indices in branch_groups.values():
        indices.sort(key=lambda idx: (int(round_index[idx]), float(arrival[idx]), str(request_id[idx])))
        suffix = 0.0
        for idx in reversed(indices):
            suffix += float(service[idx])
            actual_remaining[idx] = suffix
        used = 0.0
        prompt_sum = 0.0
        prompt_max = 0.0
        for idx in indices:
            elapsed[idx] = used
            prompt_sum += float(prompt[idx])
            prompt_max = max(prompt_max, float(prompt[idx]))
            sum_revealed_prompt[idx] = prompt_sum
            max_revealed_prompt[idx] = prompt_max
            used += float(service[idx])

    requested_max_output = np.full(count, output_cap, dtype=float)
    features = np.column_stack(
        (
            np.log1p(elapsed),
            round_index.astype(float),
            round_index.astype(float) ** 2,
            np.log1p(prompt),
            np.log1p(uncached_prompt),
            reuse / np.maximum(prompt, 1.0),
            np.log1p(requested_max_output),
            np.log1p(max_revealed_prompt),
            np.log1p(sum_revealed_prompt),
            active_frontier_width,
            stage_id.astype(float),
            role.astype(float),
            role.astype(float) * np.log1p(prompt),
            round_index.astype(float) * np.log1p(elapsed),
        )
    )
    workflow_ids = np.asarray(
        [f"workflow-{int(value):04d}" for value in workflow_index], dtype="U32"
    )
    folds = np.asarray(
        [
            workflow_fold(str(value), 5, salt="aegiskv-modeled-service-v2")
            for value in workflow_ids
        ],
        dtype=np.int8,
    )
    predicted = np.zeros(count, dtype=float)
    lower = np.zeros(count, dtype=float)
    upper = np.zeros(count, dtype=float)
    predictor_manifests: dict[str, Any] = {}
    config = AegisGBDTConfig(
        estimators=64,
        learning_rate=0.06,
        max_depth=4,
        min_leaf=48,
        max_bins=48,
    )
    for target_fold in range(5):
        calibration_fold = (target_fold + 1) % 5
        fit_mask = (folds != target_fold) & (folds != calibration_fold)
        calibration_mask = folds == calibration_fold
        target_mask = folds == target_fold
        regressor = AegisGBDTRegressor(config).fit(features[fit_mask], actual_remaining[fit_mask])
        calibration_prediction = regressor.predict(features[calibration_mask])
        margin = _upper_quantile(
            np.abs(calibration_prediction - actual_remaining[calibration_mask]), 0.90
        )
        fold_prediction = np.maximum(regressor.predict(features[target_mask]), 0.0)
        predicted[target_mask] = fold_prediction
        lower[target_mask] = np.maximum(fold_prediction - margin, 0.0)
        upper[target_mask] = fold_prediction + margin
        predictor_manifests[str(target_fold)] = {
            **regressor.manifest(),
            "calibrated_absolute_error_ms": margin,
            "fit_request_count": int(np.sum(fit_mask)),
            "calibration_request_count": int(np.sum(calibration_mask)),
            "target_request_count": int(np.sum(target_mask)),
            "fit_workflow_count": int(len(set(workflow_index[fit_mask]))),
            "calibration_workflow_count": int(
                len(set(workflow_index[calibration_mask]))
            ),
            "target_workflow_count": int(len(set(workflow_index[target_mask]))),
        }

    actual_slack = np.zeros(count, dtype=float)
    predicted_slack = np.zeros(count, dtype=float)
    slack_lcb = np.zeros(count, dtype=float)
    for indices in round_groups.values():
        if not complete_visible_frontier[indices[0]]:
            continue
        for index in indices:
            others = [other for other in indices if other != index]
            actual_slack[index] = max(
                max((actual_remaining[other] for other in others), default=0.0)
                - actual_remaining[index],
                0.0,
            )
            predicted_slack[index] = max(
                max((predicted[other] for other in others), default=0.0)
                - predicted[index],
                0.0,
            )
            slack_lcb[index] = max(
                max((lower[other] for other in others), default=0.0)
                - upper[index],
                0.0,
            )

    arrays = {
        "modeled_service_ms": service.astype(np.float32),
        "actual_remaining_ms": actual_remaining.astype(np.float32),
        "predicted_remaining_ms": predicted.astype(np.float32),
        "remaining_lower_ms": lower.astype(np.float32),
        "remaining_upper_ms": upper.astype(np.float32),
        "actual_slack_ms": actual_slack.astype(np.float32),
        "predicted_slack_ms": predicted_slack.astype(np.float32),
        "slack_lcb_ms": slack_lcb.astype(np.float32),
        "complete_visible_frontier": complete_visible_frontier.astype(np.int8),
    }
    array_manifest: dict[str, Any] = {}
    for name, array in arrays.items():
        path = output_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        array_manifest[name] = {
            "path": path.name,
            "sha256": _sha256(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }

    remaining_metrics = _regression(actual_remaining, predicted)
    remaining_metrics["interval_coverage"] = float(
        np.mean((lower <= actual_remaining) & (actual_remaining <= upper))
    )
    eligible_actual_slack = actual_slack[complete_visible_frontier]
    eligible_predicted_slack = predicted_slack[complete_visible_frontier]
    eligible_lcb = slack_lcb[complete_visible_frontier]
    slack_metrics = _regression(eligible_actual_slack, eligible_predicted_slack)
    slack_metrics.update(
        _positive_prediction_diagnostics(
            eligible_actual_slack, eligible_predicted_slack
        )
    )
    slack_metrics["lcb_coverage"] = float(
        np.mean(eligible_lcb <= eligible_actual_slack + 1e-9)
    )
    eligible_count = int(np.sum(complete_visible_frontier))
    predicted_nonzero_count = int(np.sum(predicted_slack > 1e-9))
    predicted_nonzero_eligible = int(np.sum(eligible_predicted_slack > 1e-9))
    actual_nonzero_count = int(np.sum(actual_slack > 1e-9))
    lcb_nonzero_count = int(np.sum(slack_lcb > 1e-9))
    nonzero_fraction_all = predicted_nonzero_count / max(count, 1)
    nonzero_fraction_eligible = predicted_nonzero_eligible / max(eligible_count, 1)
    mechanism_gate = {
        "gate_name": "central_m1_causal_point_slack_is_non_degenerate",
        "threshold_selected_before_performance_campaign": True,
        "minimum_nonzero_fraction_all_requests": (
            MIN_NONZERO_FRACTION_ALL_REQUESTS
        ),
        "minimum_nonzero_fraction_eligible_frontiers": (
            MIN_NONZERO_FRACTION_ELIGIBLE_FRONTIERS
        ),
        "observed_nonzero_fraction_all_requests": nonzero_fraction_all,
        "observed_nonzero_fraction_eligible_frontiers": (
            nonzero_fraction_eligible
        ),
        "passed": (
            nonzero_fraction_all >= MIN_NONZERO_FRACTION_ALL_REQUESTS
            and nonzero_fraction_eligible
            >= MIN_NONZERO_FRACTION_ELIGIBLE_FRONTIERS
        ),
        "performance_result_consulted": False,
    }
    if not mechanism_gate["passed"]:
        raise AssertionError(
            "central Algorithm-1 point Slack is degenerate: "
            f"all={nonzero_fraction_all:.6f}, "
            f"eligible={nonzero_fraction_eligible:.6f}"
        )
    manifest = {
        "schema_version": 2,
        "evidence_class": (
            "workflow_disjoint_causal_visible_sglsim_point_predictor_v2"
        ),
        "evidence_boundary": (
            "fidelity to declared SGLSim coefficients only; not physical "
            "inference accuracy; round-synchronous visible-frontier CPM "
            "abstraction rather than full online residual-DAG execution"
        ),
        "campaign_id": protocol["campaign_id"],
        "producer": {
            "path": str(PRODUCER),
            "sha256": _sha256(PRODUCER),
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "algorithm1_contract": algorithm1_protocol,
        },
        "compact_workload": {
            "path": str(compact_manifest_path),
            "sha256": _sha256(compact_manifest_path),
            "canonical_sha256": compact_manifest["canonical_sha256"],
        },
        "model": "qwen3_8b",
        "prompt_scale": scale,
        "output_cap_tokens": output_cap,
        "feature_names": [
            "log_elapsed_modeled_service_ms",
            "completed_request_count",
            "completed_request_count_squared",
            "log_current_prompt_tokens",
            "log_current_uncached_prompt_tokens",
            "input_reuse_fraction",
            "log_requested_max_output_tokens",
            "log_max_revealed_prompt_tokens",
            "log_sum_revealed_prompt_tokens",
            "active_frontier_width",
            "stage_id",
            "role_is_subagent",
            "role_x_log_current_prompt",
            "completed_x_log_elapsed",
        ],
        "workflow_disjoint_crossfit": True,
        "folds": 5,
        "crossfit_contract": {
            "split_unit": "workflow_id",
            "fit_calibration_target_workflows_disjoint_per_fold": True,
            "target_prediction_uses_target_labels": False,
            "target_prediction_uses_target_or_future_truth_features": False,
            "future_service_used_as_training_label_only": True,
            "fold_salt": "aegiskv-modeled-service-v2",
        },
        "causal_feature_contract": {
            "observation": (
                "current request admission after prior request/tool return"
            ),
            "uses_current_prompt_and_reuse": True,
            "uses_completed_prior_modeled_service": True,
            "uses_requested_output_cap_not_realized_output": True,
            "uses_future_output_or_completion_as_feature": False,
            "visible_branch_set_built_incrementally": True,
            "missing_or_incomplete_frontier_publishes_zero_slack": True,
            "feature_names_forbidden_future_truth": True,
            "logical_round_frontier_is_visibility_snapshot": True,
            "within_round_wall_clock_release_skew_replayed": False,
        },
        "cpm_contract": {
            "semantics": "single_level_visible_frontier_exact_cpm",
            "formula": "max(max_other_predicted_remaining-self,0)",
            "eligible_frontier_count": eligible_count,
            "ineligible_or_singleton_count": count - eligible_count,
            "policy_slack_for_ineligible_or_missing_frontier_ms": 0.0,
        },
        "policy_input": {
            "array": "predicted_slack_ms",
            "estimate": "workflow_disjoint_crossfit_point_estimate",
            "uses_future_truth": False,
            "uses_actual_slack": False,
            "uses_conformal_lcb": False,
            "current_design_hat_w_point_prediction": True,
            "nonzero_count": predicted_nonzero_count,
            "nonzero_fraction_all_requests": nonzero_fraction_all,
            "nonzero_count_eligible_frontiers": predicted_nonzero_eligible,
            "nonzero_fraction_eligible_frontiers": nonzero_fraction_eligible,
        },
        "ground_truth": {
            "array": "actual_slack_ms",
            "policy_input": False,
            "label_or_endpoint_only": True,
            "nonzero_count": actual_nonzero_count,
            "nonzero_fraction_all_requests": actual_nonzero_count
            / max(count, 1),
        },
        "uncertainty_diagnostic": {
            "array": "slack_lcb_ms",
            "policy_input": False,
            "method": "workflow_disjoint_fold_absolute_error_margin",
            "nonzero_count": lcb_nonzero_count,
            "nonzero_fraction_all_requests": lcb_nonzero_count / max(count, 1),
            "zero_fraction_all_requests": 1.0
            - lcb_nonzero_count / max(count, 1),
            "vacuous_for_central_policy": (
                lcb_nonzero_count / max(count, 1)
                < MIN_NONZERO_FRACTION_ALL_REQUESTS
            ),
            "retained_to_report_uncertainty_failure_not_hide_it": True,
        },
        "mechanism_gate": mechanism_gate,
        "sensitivity_contract": {
            "m0": "oracle_actual_slack_sensitivity_not_primary",
            "m_between_0_and_1": "partial_prediction_error_interpolation",
            "m1": "central_workflow_disjoint_crossfit_point_prediction",
            "m_greater_than_1": (
                "linear_prediction_error_extrapolation_stress_not_accuracy_claim"
            ),
        },
        "predictors": predictor_manifests,
        "remaining": remaining_metrics,
        "slack": slack_metrics,
        "arrays": array_manifest,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--compact-dir", type=Path, default=COMPACT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = prepare(
        args.protocol, args.compact_dir, args.output_dir, force=args.force
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output_dir),
                "remaining": result["remaining"],
                "slack": result["slack"],
                "mechanism_gate": result["mechanism_gate"],
                "uncertainty_diagnostic": result[
                    "uncertainty_diagnostic"
                ],
                "canonical_sha256": result["canonical_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
