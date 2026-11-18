"""Read-only compact Weka workload projection for one simulation scenario."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sglsim.srt.simulation.aegis_large_scale import MODEL_PROFILES, RequestInput


ROOT = Path(__file__).resolve().parent
COMPACT = ROOT / "cache" / "weka_compact_v1"
MODELED_ALG1 = ROOT / "cache" / "modeled_algorithm1_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _branch_and_round(request_id: str) -> tuple[int, int]:
    """Decode only the branch/round identity visible in a request ID."""

    tail = request_id.rsplit("-p", 1)[1]
    if "-m" in tail:
        return 0, int(tail.rsplit("-m", 1)[1])
    branch_part, round_part = tail.rsplit("-s", 1)[1].split("-t", 1)
    return int(branch_part) + 1, int(round_part)


def _array_sha256(name: str, array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(name.encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _regression_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | int]:
    actual64 = np.asarray(actual, dtype=float)
    predicted64 = np.asarray(predicted, dtype=float)
    error = predicted64 - actual64
    positive = actual64 > 1e-9
    denominator = float(np.sum((actual64 - np.mean(actual64)) ** 2))
    return {
        "count": int(len(actual64)),
        "mae_ms": float(np.mean(np.abs(error))),
        "mape": (
            float(np.mean(np.abs(error[positive]) / actual64[positive]))
            if np.any(positive)
            else 0.0
        ),
        "r2": (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0.0
            else 0.0
        ),
        "signed_bias_ms": float(np.mean(error)),
    }


def _scenario_specific_cpm(
    *,
    workflow_index: np.ndarray,
    stage_id: np.ndarray,
    branch: np.ndarray,
    round_index: np.ndarray,
    actual_scenario_service_ms: np.ndarray,
    causal_visible_scenario_service_proxy_ms: np.ndarray,
    central_causal_visible_service_proxy_ms: np.ndarray,
    central_crossfit_predicted_remaining_ms: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rebuild actual and predicted Slack for one declared scenario.

    Actual remaining work is an exact suffix sum of the realized scenario
    service vector.  The policy prediction never reads that suffix truth or
    realized output: it scales a workflow-disjoint central cross-fit remaining
    prediction by only the current request's submission-visible scenario /
    central service-proxy ratio.  CPM is then applied independently to actual
    and predicted remaining work at each complete currently visible frontier.
    """

    arrays = tuple(
        np.asarray(value)
        for value in (
            workflow_index,
            stage_id,
            branch,
            round_index,
            actual_scenario_service_ms,
            causal_visible_scenario_service_proxy_ms,
            central_causal_visible_service_proxy_ms,
            central_crossfit_predicted_remaining_ms,
        )
    )
    count = len(arrays[0])
    if any(len(value) != count for value in arrays):
        raise ValueError("scenario CPM inputs must have identical lengths")
    actual_service = np.asarray(actual_scenario_service_ms, dtype=float)
    visible_service = np.asarray(
        causal_visible_scenario_service_proxy_ms, dtype=float
    )
    central_visible_service = np.asarray(
        central_causal_visible_service_proxy_ms, dtype=float
    )
    central_prediction = np.asarray(
        central_crossfit_predicted_remaining_ms, dtype=float
    )
    if (
        not np.all(np.isfinite(actual_service))
        or np.any(actual_service <= 0.0)
        or not np.all(np.isfinite(visible_service))
        or np.any(visible_service <= 0.0)
        or not np.all(np.isfinite(central_visible_service))
        or np.any(central_visible_service <= 0.0)
        or not np.all(np.isfinite(central_prediction))
        or np.any(central_prediction < 0.0)
    ):
        raise ValueError("scenario CPM service/prediction inputs are invalid")

    branch_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    round_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index in range(count):
        branch_groups[
            (
                int(workflow_index[index]),
                int(stage_id[index]),
                int(branch[index]),
            )
        ].append(index)
        round_groups[
            (
                int(workflow_index[index]),
                int(stage_id[index]),
                int(round_index[index]),
            )
        ].append(index)

    actual_remaining = np.zeros(count, dtype=np.float64)
    for indices in branch_groups.values():
        indices.sort(key=lambda index: (int(round_index[index]), index))
        suffix = 0.0
        for index in reversed(indices):
            suffix += float(actual_service[index])
            actual_remaining[index] = suffix

    # This is the only predictor-side scenario transform.  Both numerator and
    # denominator are submission-time-visible proxies: projected prompt/reuse,
    # the declared model coefficients, and the effective requested output cap.
    # Realized output and all actual suffix labels stay endpoint-only.
    predicted_remaining = np.maximum(central_prediction, 0.0) * (
        visible_service / central_visible_service
    )

    rounds_by_stage: dict[tuple[int, int], list[tuple[int, int, int]]] = (
        defaultdict(list)
    )
    for key in round_groups:
        rounds_by_stage[key[:2]].append(key)
    complete_visible_frontier = np.zeros(count, dtype=bool)
    actual_slack = np.zeros(count, dtype=np.float64)
    predicted_slack = np.zeros(count, dtype=np.float64)
    for keys in rounds_by_stage.values():
        visible_branches: set[int] = set()
        for key in sorted(keys, key=lambda value: value[2]):
            indices = round_groups[key]
            current_branches = [int(branch[index]) for index in indices]
            current_set = set(current_branches)
            visible_branches.update(current_set)
            complete = (
                len(current_set) > 1
                and len(current_branches) == len(current_set)
                and current_set == visible_branches
            )
            if not complete:
                continue
            complete_visible_frontier[indices] = True
            for index in indices:
                others = [other for other in indices if other != index]
                actual_slack[index] = max(
                    max(
                        (actual_remaining[other] for other in others),
                        default=0.0,
                    )
                    - actual_remaining[index],
                    0.0,
                )
                predicted_slack[index] = max(
                    max(
                        (predicted_remaining[other] for other in others),
                        default=0.0,
                    )
                    - predicted_remaining[index],
                    0.0,
                )

    return {
        "actual_scenario_service_ms": actual_service,
        "causal_visible_scenario_service_proxy_ms": visible_service,
        "central_causal_visible_service_proxy_ms": central_visible_service,
        "actual_remaining_ms": actual_remaining,
        "causally_scaled_crossfit_predicted_remaining_ms": predicted_remaining,
        "actual_slack_ms": actual_slack,
        "predicted_slack_ms": predicted_slack,
        "complete_visible_frontier": complete_visible_frontier,
    }


def _policy_slack_for_multiplier(
    *,
    actual_slack_ms: np.ndarray,
    predicted_slack_ms: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Apply the preregistered counterfactual prediction-error sensitivity.

    The central ``m=1`` branch deliberately does not read actual Slack.  Other
    values are explicit oracle/error-stress sensitivities, not predictor
    accuracy claims.
    """

    predicted = np.asarray(predicted_slack_ms, dtype=float)
    if not math.isfinite(float(multiplier)) or float(multiplier) < 0.0:
        raise ValueError("Algorithm-1 prediction-error multiplier is invalid")
    if math.isclose(float(multiplier), 1.0, rel_tol=0.0, abs_tol=1e-12):
        return np.maximum(predicted, 0.0).copy()
    actual = np.asarray(actual_slack_ms, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted Slack arrays must align")
    return np.maximum(actual + float(multiplier) * (predicted - actual), 0.0)


@dataclass(frozen=True)
class ProjectedWorkload:
    rows: tuple[RequestInput, ...]
    metadata: dict[str, Any]


class CompactWeka:
    """Memory-mapped compact arrays shared by all policy runs in a worker."""

    def __init__(
        self,
        compact_dir: Path = COMPACT,
        modeled_algorithm1_dir: Path = MODELED_ALG1,
    ) -> None:
        self.compact_dir = compact_dir
        self.alg1_dir = modeled_algorithm1_dir
        self.manifest_path = compact_dir / "manifest.json"
        self.alg1_manifest_path = modeled_algorithm1_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.alg1_manifest = json.loads(
            self.alg1_manifest_path.read_text(encoding="utf-8")
        )
        if self.alg1_manifest.get("protocol", {}).get("sha256") != _sha256(
            ROOT / "protocol.json"
        ):
            raise ValueError("Algorithm-1 cache is not bound to current protocol")
        if self.alg1_manifest.get("producer", {}).get("sha256") != _sha256(
            ROOT / "prepare_modeled_algorithm1.py"
        ):
            raise ValueError("Algorithm-1 cache is not bound to current producer")
        if self.alg1_manifest.get("schema_version") != 2:
            raise ValueError("Algorithm-1 cache must use causal point schema v2")
        policy_input = dict(self.alg1_manifest.get("policy_input") or {})
        mechanism_gate = dict(self.alg1_manifest.get("mechanism_gate") or {})
        crossfit = dict(self.alg1_manifest.get("crossfit_contract") or {})
        causal = dict(self.alg1_manifest.get("causal_feature_contract") or {})
        if policy_input.get("array") != "predicted_slack_ms":
            raise ValueError("Algorithm-1 policy input must be predicted_slack_ms")
        if policy_input.get("uses_future_truth") is not False:
            raise ValueError("Algorithm-1 policy input must reject future truth")
        if policy_input.get("uses_actual_slack") is not False:
            raise ValueError("actual Slack cannot be an Algorithm-1 policy input")
        if policy_input.get("uses_conformal_lcb") is not False:
            raise ValueError("vacuous conformal LCB cannot drive central Algorithm 1")
        if mechanism_gate.get("passed") is not True:
            raise ValueError("Algorithm-1 non-degenerate mechanism gate has not passed")
        if crossfit.get("fit_calibration_target_workflows_disjoint_per_fold") is not True:
            raise ValueError("Algorithm-1 predictor must be workflow-disjoint")
        if crossfit.get("target_prediction_uses_target_labels") is not False:
            raise ValueError("Algorithm-1 target prediction consumed target labels")
        if causal.get("uses_future_output_or_completion_as_feature") is not False:
            raise ValueError("Algorithm-1 feature contract admits future truth")
        if causal.get("missing_or_incomplete_frontier_publishes_zero_slack") is not True:
            raise ValueError("Algorithm-1 cache does not fail closed on missing DAG state")
        if (
            self.alg1_manifest["compact_workload"]["canonical_sha256"]
            != self.manifest["canonical_sha256"]
        ):
            raise ValueError("Algorithm-1 cache is not bound to compact Weka input")

        def load(base: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
            record = manifest["arrays"][name]
            path = base / record["path"]
            if _sha256(path) != record["sha256"]:
                raise ValueError(f"compact array hash mismatch: {name}")
            return np.load(path, mmap_mode="r", allow_pickle=False)

        self.request_id = load(compact_dir, self.manifest, "request_id")
        self.workflow_index = load(compact_dir, self.manifest, "workflow_index")
        self.stage_id = load(compact_dir, self.manifest, "stage_id")
        self.arrival_ms = load(compact_dir, self.manifest, "arrival_ms")
        self.api_time_ms = load(compact_dir, self.manifest, "api_time_ms")
        self.prompt_raw = load(compact_dir, self.manifest, "prompt_tokens_raw")
        self.output_raw = load(compact_dir, self.manifest, "output_tokens_raw")
        self.reuse_raw = load(compact_dir, self.manifest, "input_reuse_tokens_raw")
        central_model = MODEL_PROFILES["qwen3-8b"]
        central_prompt_scale = float(self.alg1_manifest["prompt_scale"])
        central_output_cap = int(self.alg1_manifest["output_cap_tokens"])
        central_prompt = np.maximum(
            np.ceil(self.prompt_raw.astype(float) * central_prompt_scale), 1
        )
        central_reuse = np.minimum(
            np.rint(self.reuse_raw.astype(float) * central_prompt_scale),
            central_prompt,
        )
        central_effective_requested_output_cap = np.maximum(
            np.minimum(
                central_output_cap,
                central_model.max_context_tokens - central_prompt,
            ),
            0,
        )
        self.central_causal_visible_service_proxy = (
            central_model.prefill_setup_ms
            + (central_prompt - central_reuse)
            * central_model.prefill_ms_per_token
            + central_effective_requested_output_cap
            * central_model.decode_ms_per_token
        )
        self.modeled_service_central = load(
            modeled_algorithm1_dir,
            self.alg1_manifest,
            "modeled_service_ms",
        )
        self.actual_slack_central = load(
            modeled_algorithm1_dir,
            self.alg1_manifest,
            "actual_slack_ms",
        )
        self.predicted_slack_central = load(
            modeled_algorithm1_dir,
            self.alg1_manifest,
            "predicted_slack_ms",
        )
        self.predicted_remaining_central = load(
            modeled_algorithm1_dir,
            self.alg1_manifest,
            "predicted_remaining_ms",
        )
        self.complete_visible_frontier = load(
            modeled_algorithm1_dir,
            self.alg1_manifest,
            "complete_visible_frontier",
        )
        predicted_nonzero_count = int(
            np.sum(self.predicted_slack_central > 1e-9)
        )
        eligible = self.complete_visible_frontier.astype(bool)
        predicted_nonzero_eligible = int(
            np.sum(self.predicted_slack_central[eligible] > 1e-9)
        )
        if not np.all(np.isfinite(self.predicted_slack_central)) or np.any(
            self.predicted_slack_central < 0.0
        ):
            raise ValueError("Algorithm-1 point Slack must be finite and nonnegative")
        if predicted_nonzero_count != int(policy_input.get("nonzero_count", -1)):
            raise ValueError("Algorithm-1 point Slack count does not match manifest")
        if predicted_nonzero_eligible != int(
            policy_input.get("nonzero_count_eligible_frontiers", -1)
        ):
            raise ValueError(
                "Algorithm-1 eligible point Slack count does not match manifest"
            )
        self.algorithm1_policy_input = policy_input
        self.algorithm1_mechanism_gate = mechanism_gate
        self.branch = np.empty(len(self.request_id), dtype=np.int16)
        self.round_index = np.empty(len(self.request_id), dtype=np.int16)
        for index, request_id in enumerate(self.request_id):
            self.branch[index], self.round_index[index] = _branch_and_round(
                str(request_id)
            )
        workflows = json.loads(
            (compact_dir / self.manifest["workflow_reuse"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        self.workflow_ids = tuple(str(row["workflow_id"]) for row in workflows)
        if len(self.workflow_ids) != self.manifest["source"]["workflow_count"]:
            raise AssertionError("workflow identity table is incomplete")
        self._origins = np.full(len(self.workflow_ids), np.inf, dtype=float)
        self._finishes = np.full(len(self.workflow_ids), -np.inf, dtype=float)
        np.minimum.at(self._origins, self.workflow_index, self.arrival_ms)
        np.maximum.at(
            self._finishes,
            self.workflow_index,
            self.arrival_ms + self.api_time_ms,
        )

    def project(
        self,
        scenario: dict[str, Any],
        protocol: dict[str, Any],
        *,
        seed: int,
    ) -> ProjectedWorkload:
        model_key = str(scenario["model_id"]).replace("_", "-")
        model_key = {"qwen3-8b": "qwen3-8b", "qwen3-32b": "qwen3-32b"}[model_key]
        model = MODEL_PROFILES[model_key]
        prompt_scale = float(scenario["prompt_scale"])
        output_cap = int(protocol["workload"]["output_cap_tokens"])
        prompt = np.maximum(np.ceil(self.prompt_raw.astype(float) * prompt_scale), 1).astype(np.int32)
        output = np.minimum(self.output_raw, output_cap).astype(np.int32)
        overflow = prompt.astype(np.int64) + output.astype(np.int64) - int(model.max_context_tokens)
        if np.any(overflow > 0):
            output = np.maximum(output.astype(np.int64) - np.maximum(overflow, 0), 0).astype(np.int32)
        reuse = np.minimum(
            np.rint(self.reuse_raw.astype(float) * prompt_scale).astype(np.int32),
            prompt,
        )
        service = (
            model.prefill_setup_ms
            + (prompt - reuse).astype(float) * model.prefill_ms_per_token
            + output.astype(float) * model.decode_ms_per_token
        )
        effective_requested_output_cap = np.maximum(
            np.minimum(
                output_cap,
                int(model.max_context_tokens) - prompt.astype(np.int64),
            ),
            0,
        )
        causal_visible_service_proxy = (
            model.prefill_setup_ms
            + (prompt - reuse).astype(float) * model.prefill_ms_per_token
            + effective_requested_output_cap.astype(float)
            * model.decode_ms_per_token
        )
        raw_api_median = float(np.median(np.maximum(self.api_time_ms, 1e-6)))
        service_median = float(np.median(service))
        time_scale = service_median / raw_api_median
        relative_arrival = (
            self.arrival_ms - self._origins[self.workflow_index]
        ) * time_scale
        workflow_span = (
            self._finishes - self._origins
        ) * time_scale
        workflow_duration = np.maximum(workflow_span, service_median)
        mean_span = float(np.mean(workflow_duration))
        concurrency = int(scenario["active_workflow_concurrency"])
        rng = random.Random(int(seed) ^ 0xAE615CA5)
        workflow_base = np.zeros(len(self.workflow_ids), dtype=float)
        workflow_finish = np.zeros(len(self.workflow_ids), dtype=float)
        launch_order = list(range(len(self.workflow_ids)))
        rng.shuffle(launch_order)
        running: list[tuple[float, int]] = []
        initial = min(concurrency, len(launch_order))
        for workflow in launch_order[:initial]:
            workflow_base[workflow] = 0.0
            workflow_finish[workflow] = float(workflow_duration[workflow])
            heapq.heappush(
                running, (workflow_finish[workflow], workflow)
            )
        for workflow in launch_order[initial:]:
            released_at, _completed_workflow = heapq.heappop(running)
            workflow_base[workflow] = float(released_at)
            workflow_finish[workflow] = (
                float(released_at) + float(workflow_duration[workflow])
            )
            heapq.heappush(
                running, (workflow_finish[workflow], workflow)
            )
        global_arrival = workflow_base[self.workflow_index] + relative_arrival

        concurrency_events = sorted(
            [
                *((float(workflow_base[index]), 1) for index in range(len(workflow_base))),
                *((float(workflow_finish[index]), -1) for index in range(len(workflow_finish))),
            ],
            key=lambda row: (row[0], row[1]),
        )
        active = 0
        previous = concurrency_events[0][0]
        weighted_levels: list[tuple[int, float]] = []
        active_intervals: list[tuple[float, float, int]] = []
        peak_active_workflows = 0
        for at_ms, delta in concurrency_events:
            duration = max(at_ms - previous, 0.0)
            if duration:
                weighted_levels.append((active, duration))
                active_intervals.append((previous, at_ms, active))
            active += delta
            peak_active_workflows = max(peak_active_workflows, active)
            previous = at_ms
        total_duration = sum(duration for _level, duration in weighted_levels)
        mean_active_workflows = (
            sum(level * duration for level, duration in weighted_levels)
            / max(total_duration, 1e-9)
        )

        def weighted_quantile(q: float) -> float:
            threshold = total_duration * q
            cumulative = 0.0
            for level, duration in sorted(weighted_levels):
                cumulative += duration
                if cumulative >= threshold:
                    return float(level)
            return float(weighted_levels[-1][0]) if weighted_levels else 0.0

        stabilization_count = min(
            int(protocol["workload"]["controller_stabilization_completions"]),
            len(workflow_finish) - 1,
        )
        stabilization_end_ms = float(
            np.sort(workflow_finish)[stabilization_count - 1]
        )
        last_launch_ms = float(np.max(workflow_base))
        if last_launch_ms <= stabilization_end_ms:
            raise AssertionError(
                "finite workload lacks a steady measurement interval after "
                "the controller stabilization prefix"
            )
        steady_levels: list[tuple[int, float]] = []
        for left, right, level in active_intervals:
            overlap = max(
                min(float(right), last_launch_ms)
                - max(float(left), stabilization_end_ms),
                0.0,
            )
            if overlap:
                steady_levels.append((level, overlap))
        steady_duration = sum(duration for _level, duration in steady_levels)
        steady_mean = sum(
            level * duration for level, duration in steady_levels
        ) / max(steady_duration, 1e-9)

        def steady_quantile(q: float) -> float:
            threshold = steady_duration * q
            cumulative = 0.0
            for level, duration in sorted(steady_levels):
                cumulative += duration
                if cumulative >= threshold:
                    return float(level)
            return float(steady_levels[-1][0]) if steady_levels else 0.0

        error_multiplier = float(
            scenario["algorithm1_prediction_error_multiplier"]
        )
        scenario_cpm = _scenario_specific_cpm(
            workflow_index=self.workflow_index,
            stage_id=self.stage_id,
            branch=self.branch,
            round_index=self.round_index,
            actual_scenario_service_ms=service,
            causal_visible_scenario_service_proxy_ms=(
                causal_visible_service_proxy
            ),
            central_causal_visible_service_proxy_ms=(
                self.central_causal_visible_service_proxy
            ),
            central_crossfit_predicted_remaining_ms=(
                self.predicted_remaining_central
            ),
        )
        actual_remaining = scenario_cpm["actual_remaining_ms"]
        predicted_remaining = scenario_cpm[
            "causally_scaled_crossfit_predicted_remaining_ms"
        ]
        actual_slack = scenario_cpm["actual_slack_ms"]
        predicted_slack = scenario_cpm["predicted_slack_ms"]
        complete_visible_frontier = scenario_cpm[
            "complete_visible_frontier"
        ]
        policy_slack = _policy_slack_for_multiplier(
            actual_slack_ms=actual_slack,
            predicted_slack_ms=predicted_slack,
            multiplier=error_multiplier,
        )

        scenario_algorithm1_arrays = {
            "actual_scenario_service_ms": scenario_cpm[
                "actual_scenario_service_ms"
            ],
            "causal_visible_scenario_service_proxy_ms": scenario_cpm[
                "causal_visible_scenario_service_proxy_ms"
            ],
            "central_causal_visible_service_proxy_ms": scenario_cpm[
                "central_causal_visible_service_proxy_ms"
            ],
            "actual_remaining_ms": actual_remaining,
            "causally_scaled_crossfit_predicted_remaining_ms": (
                predicted_remaining
            ),
            "actual_slack_ms": actual_slack,
            "predicted_slack_ms": predicted_slack,
            "effective_policy_slack_ms": policy_slack,
            "complete_visible_frontier": complete_visible_frontier.astype(
                np.int8
            ),
        }
        scenario_algorithm1_array_hashes = {
            name: _array_sha256(name, array)
            for name, array in scenario_algorithm1_arrays.items()
        }
        scenario_algorithm1_digest = hashlib.sha256()
        for name in sorted(scenario_algorithm1_array_hashes):
            scenario_algorithm1_digest.update(name.encode("ascii"))
            scenario_algorithm1_digest.update(
                scenario_algorithm1_array_hashes[name].encode("ascii")
            )
        eligible = complete_visible_frontier.astype(bool)
        scenario_remaining_metrics = _regression_metrics(
            actual_remaining, predicted_remaining
        )
        scenario_slack_metrics = _regression_metrics(
            actual_slack[eligible], predicted_slack[eligible]
        )
        is_central_service_projection = (
            model_key == "qwen3-8b"
            and math.isclose(
                prompt_scale,
                float(self.alg1_manifest["prompt_scale"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and output_cap == int(self.alg1_manifest["output_cap_tokens"])
        )
        central_regeneration_alignment = {
            "applies": is_central_service_projection,
            "visible_service_proxy_ratio_bit_exact_one": bool(
                np.array_equal(
                    causal_visible_service_proxy,
                    self.central_causal_visible_service_proxy,
                )
            ),
            "modeled_service_max_abs_error_ms": float(
                np.max(
                    np.abs(
                        service.astype(float)
                        - self.modeled_service_central.astype(float)
                    )
                )
            ),
            "actual_slack_max_abs_error_ms": float(
                np.max(
                    np.abs(
                        actual_slack
                        - self.actual_slack_central.astype(float)
                    )
                )
            ),
            "predicted_slack_max_abs_error_ms": float(
                np.max(
                    np.abs(
                        predicted_slack
                        - self.predicted_slack_central.astype(float)
                    )
                )
            ),
            "complete_visible_frontier_exact_match": bool(
                np.array_equal(
                    complete_visible_frontier.astype(np.int8),
                    self.complete_visible_frontier.astype(np.int8),
                )
            ),
        }
        central_regeneration_alignment["passed"] = bool(
            not is_central_service_projection
            or (
                central_regeneration_alignment[
                    "visible_service_proxy_ratio_bit_exact_one"
                ]
                and central_regeneration_alignment[
                    "complete_visible_frontier_exact_match"
                ]
                and np.allclose(
                    service,
                    self.modeled_service_central,
                    rtol=1e-6,
                    atol=1e-3,
                )
                and np.allclose(
                    actual_slack,
                    self.actual_slack_central,
                    rtol=1e-6,
                    atol=1e-3,
                )
                and np.allclose(
                    predicted_slack,
                    self.predicted_slack_central,
                    rtol=1e-6,
                    atol=1e-3,
                )
            )
        )
        if not central_regeneration_alignment["passed"]:
            raise ValueError(
                "scenario-specific CPM failed central-cache regeneration gate"
            )

        # Bind the complete policy-independent projected workload without
        # serializing 98,827 request dictionaries. Source/cache manifests bind
        # request identity and index order; these arrays bind every field that
        # varies by scenario or paired seed.
        workload_digest = hashlib.sha256()
        workload_digest.update(
            self.manifest["canonical_sha256"].encode("ascii")
        )
        workload_digest.update(
            self.alg1_manifest["canonical_sha256"].encode("ascii")
        )
        for name, array in (
            ("prompt_tokens", prompt),
            ("output_tokens", output),
            ("input_reuse_tokens", reuse),
            ("arrival_ms", global_arrival),
            ("actual_scenario_service_ms", service),
            (
                "causal_visible_scenario_service_proxy_ms",
                causal_visible_service_proxy,
            ),
            (
                "central_causal_visible_service_proxy_ms",
                self.central_causal_visible_service_proxy,
            ),
            ("actual_remaining_ms", actual_remaining),
            ("predicted_remaining_ms", predicted_remaining),
            ("predicted_slack_ms", predicted_slack),
            ("actual_slack_ms", actual_slack),
            ("effective_policy_slack_ms", policy_slack),
            (
                "complete_visible_frontier",
                complete_visible_frontier.astype(np.int8),
            ),
        ):
            contiguous = np.ascontiguousarray(array)
            workload_digest.update(name.encode("ascii"))
            workload_digest.update(contiguous.dtype.str.encode("ascii"))
            workload_digest.update(
                json.dumps(contiguous.shape, separators=(",", ":")).encode()
            )
            workload_digest.update(contiguous.tobytes(order="C"))

        rows = tuple(
            RequestInput(
                request_id=str(self.request_id[index]),
                workflow_id=self.workflow_ids[int(self.workflow_index[index])],
                prompt_tokens=int(prompt[index]),
                output_tokens=int(output[index]),
                # ``slack_ms`` is the scenario's effective policy input.  The
                # base prediction remains separate so run_cell can audit the
                # sensitivity without applying it a second time.
                slack_ms=float(policy_slack[index]),
                actual_slack_ms=float(actual_slack[index]),
                predicted_slack_ms=float(predicted_slack[index]),
                policy_slack_multiplier_applied=error_multiplier,
                # Native hash IDs are workflow-local.  The exact reusable token
                # count is supplied separately, while this key keeps requests
                # in the reuse domain routable to the same logical worker.
                prefix_group=f"workflow-{int(self.workflow_index[index]):04d}",
                arrival_ms=float(global_arrival[index]),
                input_reuse_tokens=int(reuse[index]),
            )
            for index in np.argsort(global_arrival, kind="stable")
        )
        metadata = {
            "schema_version": 2,
            "source_compact_manifest_sha256": _sha256(self.manifest_path),
            "source_compact_canonical_sha256": self.manifest["canonical_sha256"],
            "modeled_algorithm1_manifest_sha256": _sha256(
                self.alg1_manifest_path
            ),
            "modeled_algorithm1_canonical_sha256": self.alg1_manifest[
                "canonical_sha256"
            ],
            "policy_independent_projected_workload_sha256": (
                workload_digest.hexdigest()
            ),
            "projected_workload_hash_contract": (
                "compact request identity/order plus prompt, output, reuse, "
                "arrival, actual service, causal visible service proxies, "
                "actual/predicted remaining, "
                "actual/base-predicted/effective-policy slack, and visible-"
                "frontier arrays"
            ),
            "scenario_specific_algorithm1_arrays_sha256": (
                scenario_algorithm1_array_hashes
            ),
            "scenario_specific_algorithm1_arrays_canonical_sha256": (
                scenario_algorithm1_digest.hexdigest()
            ),
            "scenario_specific_cpm_central_regeneration_alignment": (
                central_regeneration_alignment
            ),
            "request_count": len(rows),
            "workflow_count": len(self.workflow_ids),
            "prompt_scale": prompt_scale,
            "output_cap_tokens": output_cap,
            "simulated_model": model_key,
            "time_projection_scale": time_scale,
            "target_offered_trace_workflow_concurrency": concurrency,
            "mean_projected_workflow_span_ms": mean_span,
            "mean_workflow_launch_gap_ms": float(
                np.mean(np.diff(np.sort(np.unique(workflow_base))))
            ),
            "arrival_calibration": (
                "scale every workflow's native relative request offsets by "
                "median_simulated_service_ms/median_raw_api_time_ms; then use "
                "a seeded policy-independent offered schedule: launch min(C,N) "
                "at time zero and launch one pending workflow at each projected "
                "trace-lifespan completion"
            ),
            "native_relative_order_preserved_per_workflow": True,
            "achieved_offered_trace_active_workflows_time_mean": (
                mean_active_workflows
            ),
            "achieved_offered_trace_active_workflows_time_p50": (
                weighted_quantile(0.50)
            ),
            "achieved_offered_trace_active_workflows_time_p95": (
                weighted_quantile(0.95)
            ),
            "achieved_offered_trace_active_workflows_peak": (
                peak_active_workflows
            ),
            "controller_stabilization_completion_count": stabilization_count,
            "steady_measurement_start_ms": stabilization_end_ms,
            "steady_measurement_end_ms": last_launch_ms,
            "steady_measurement_duration_ms": steady_duration,
            "achieved_offered_trace_steady_active_workflows_time_mean": (
                steady_mean
            ),
            "achieved_offered_trace_steady_active_workflows_time_p50": (
                steady_quantile(0.50)
            ),
            "achieved_offered_trace_steady_active_workflows_time_p95": (
                steady_quantile(0.95)
            ),
            "achieved_offered_trace_steady_active_workflows_peak": max(
                (level for level, _duration in steady_levels), default=0
            ),
            "achieved_offered_trace_steady_concurrency_relative_error": abs(
                steady_mean - concurrency
            ) / max(concurrency, 1),
            "offered_schedule_completion_semantics": (
                "projected trace-lifespan completion, policy-independent; "
                "not simulator FCFS or failure-adjusted completion"
            ),
            "drain_end_ms": float(np.max(workflow_finish)),
            "drain_duration_ms": float(np.max(workflow_finish)) - last_launch_ms,
            "projected_horizon_ms": float(np.max(workflow_finish)),
            "fault_horizon_start_ms": 0.0,
            "fault_horizon_end_ms": float(np.max(workflow_finish)),
            "all_393_workflows_in_wct": True,
            "stabilization_prefix_excluded_from_wct": False,
            "input_prompt_tokens": int(np.sum(prompt, dtype=np.int64)),
            "input_reuse_tokens": int(np.sum(reuse, dtype=np.int64)),
            "input_reuse_fraction": float(
                np.sum(reuse, dtype=np.int64)
                / max(int(np.sum(prompt, dtype=np.int64)), 1)
            ),
            "algorithm1_slack_source": (
                "scenario_specific_exact_cpm_over_causally_scaled_crossfit_"
                "remaining_prediction"
            ),
            "algorithm1_base_prediction_array": "predicted_slack_ms",
            "algorithm1_policy_input_array": "effective_policy_slack_ms",
            "algorithm1_m1_policy_input_array": "predicted_slack_ms",
            "algorithm1_policy_input_uses_future_truth": False,
            "algorithm1_predicted_remaining_uses_future_truth": False,
            "algorithm1_predicted_remaining_scenario_scaling_inputs": (
                "current_request_projected_prompt_reuse_effective_requested_"
                "output_cap_and_declared_model_coefficients_only"
            ),
            "algorithm1_realized_output_is_prediction_input": False,
            "algorithm1_visible_output_proxy": (
                "min(protocol_output_cap,max_context_tokens-projected_prompt_"
                "tokens)"
            ),
            "algorithm1_m1_policy_path_reads_actual_slack": False,
            "algorithm1_actual_slack_is_ground_truth_only": True,
            "algorithm1_conformal_lcb_is_diagnostic_only": True,
            "predicted_policy_slack_nonzero_count": int(
                np.sum(predicted_slack > 1e-9)
            ),
            "predicted_policy_slack_nonzero_fraction": float(
                np.mean(predicted_slack > 1e-9)
            ),
            "predicted_policy_slack_nonzero_count_eligible_frontiers": int(
                np.sum(predicted_slack[eligible] > 1e-9)
            ),
            "predicted_policy_slack_nonzero_fraction_eligible_frontiers": float(
                np.mean(predicted_slack[eligible] > 1e-9)
            ),
            "effective_policy_slack_nonzero_count": int(
                np.sum(policy_slack > 1e-9)
            ),
            "effective_policy_slack_nonzero_fraction": float(
                np.mean(policy_slack > 1e-9)
            ),
            "algorithm1_scenario_base_prediction_nonzero_gate_passed": bool(
                np.mean(predicted_slack > 1e-9)
                >= float(
                    self.algorithm1_mechanism_gate[
                        "minimum_nonzero_fraction_all_requests"
                    ]
                )
                and np.mean(predicted_slack[eligible] > 1e-9)
                >= float(
                    self.algorithm1_mechanism_gate[
                        "minimum_nonzero_fraction_eligible_frontiers"
                    ]
                )
            ),
            "algorithm1_central_m1_nonzero_slack_gate_passed": bool(
                self.algorithm1_mechanism_gate["passed"]
            ),
            "algorithm1_prediction_diagnostics": {
                "source": (
                    "scenario_specific_base_prediction_before_error_"
                    "multiplier"
                ),
                "remaining": scenario_remaining_metrics,
                "slack": scenario_slack_metrics,
                "uncertainty": self.alg1_manifest[
                    "uncertainty_diagnostic"
                ],
                "central_crossfit_training_cache": {
                    "remaining": self.alg1_manifest["remaining"],
                    "slack": self.alg1_manifest["slack"],
                },
            },
            "algorithm1_prediction_error_multiplier": error_multiplier,
            "algorithm1_oracle": error_multiplier == 0.0,
            "algorithm1_multiplier_semantics": {
                "m0": "oracle_actual_slack_sensitivity_not_primary",
                "m_between_0_and_1": "partial_prediction_error_interpolation",
                "m1": "central_crossfit_point_prediction",
                "m_greater_than_1": (
                    "linear_prediction_error_extrapolation_stress_not_accuracy_claim"
                ),
            },
            "algorithm1_multiplier_applied_after_scenario_exact_cpm": True,
            "algorithm1_multiplier_is_counterfactual_endpoint_interpolation": (
                error_multiplier != 1.0
            ),
            "algorithm1_physical_accuracy_claim": False,
            "input_reuse_deduplicates_protection_copy": False,
        }
        return ProjectedWorkload(rows=rows, metadata=metadata)


__all__ = ["CompactWeka", "ProjectedWorkload"]
