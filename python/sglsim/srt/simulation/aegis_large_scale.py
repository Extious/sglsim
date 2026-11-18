"""Fast cluster-level AegisKV simulation cells.

This module is an isolated, metadata-only experiment driver.  It composes the
current AegisKV controller and the capacity-matched baseline runtimes; it does
not add a logical-clock branch to the online SGLSim server or change pinned
SGLang scheduling behavior.

The driver is intentionally explicit about its fidelity boundary.  Rank-local
FCFS service is a fast surrogate for large matrix sweeps.  A result emitted by
this module is therefore ``simulated_sglsim_only`` evidence, not a physical
measurement and not an online/offline scheduler-parity result.

The important contract is that experiment knobs are operational rather than
labels:

* DP changes request routing, per-worker resources, and cluster fault exposure;
* PP/TP/model profiles change shard counts, KV bytes, compute, and recovery;
* concurrency changes rank queues and the HBM-pressure signal;
* transfer bandwidth changes asynchronous copy and recovery durations;
* failure rate generates a per-worker Poisson process;
* a separate, policy-independent active+sealed exposure probe estimates
  fault-conditioned recovery efficacy without changing primary state;
* Algorithm 1, 2, and 3 can each be disabled independently; and
* Algorithm 3 receives only target-local HBM pressure.  Failure rate is never
  a direct AIMD input.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from sglsim.srt.simulation.aegis_baselines import (
    BaselinePolicy,
    BaselineRequestProfile,
    ProtectionBaselineConfig,
    ProtectionBaselineRuntime,
)
from sglsim.srt.simulation.aegis_full_design import (
    AegisFullDesignConfig,
    AegisFullDesignRuntime,
    CalibratedTransferPath,
    SealedChunk,
    sealed_chunks,
)
from sglsim.srt.simulation.aegis_independent_tiers import (
    DESIGN_SEMANTICS,
    IndependentTierConfig,
    ProtectionTier,
    RecoveryDecision,
    RecoveryTiming,
)
from sglsim.srt.simulation.aegis_residual_dag import (
    RESIDUAL_DAG_SLACK_SEMANTICS,
)


CELL_SCHEMA_VERSION = "aegiskv_sglsim_large_scale_cell_v2"
EVIDENCE_CLASS = "simulated_sglsim_only"
ALPHA_DRIVER_CONTRACT = "target_local_hbm_pressure_only"
ACTIVE_SOURCE_PROBE_EVIDENCE_CLASS = (
    "active_source_fault_conditioned_recovery_counterfactual_v1"
)
REUSE_BIN_CONTRACT = {
    "none": "input_reuse_fraction == 0",
    "partial": "0 < input_reuse_fraction < 0.90",
    "high": "0.90 <= input_reuse_fraction < 0.99",
    "near_complete": "0.99 <= input_reuse_fraction <= 1",
}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    max_context_tokens: int
    kv_bytes_per_token: int
    tp_size: int
    pp_size: int
    prefill_setup_ms: float
    prefill_ms_per_token: float
    decode_ms_per_token: float
    recompute_setup_ms: float
    recompute_ms_per_token: float

    @property
    def shard_count(self) -> int:
        return int(self.tp_size) * int(self.pp_size)


MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen3-8b": ModelProfile(
        name="Qwen3-8B-FP16",
        max_context_tokens=40_960,
        kv_bytes_per_token=147_456,
        tp_size=1,
        pp_size=1,
        prefill_setup_ms=4.0,
        prefill_ms_per_token=0.060,
        decode_ms_per_token=0.34,
        recompute_setup_ms=4.0,
        recompute_ms_per_token=0.060,
    ),
    "qwen3-32b": ModelProfile(
        name="Qwen3-32B-BF16",
        max_context_tokens=32_768,
        kv_bytes_per_token=262_144,
        tp_size=1,
        pp_size=4,
        prefill_setup_ms=9.0,
        prefill_ms_per_token=0.160,
        decode_ms_per_token=0.92,
        recompute_setup_ms=9.0,
        recompute_ms_per_token=0.160,
    ),
}

MODEL_ALIASES = {
    "qwen3_8b": "qwen3-8b",
    "qwen/qwen3-8b": "qwen3-8b",
    "qwen3_32b": "qwen3-32b",
    "qwen/qwen3-32b": "qwen3-32b",
}


BASELINE_POLICIES = tuple(policy.value for policy in BaselinePolicy)
SUPPORTED_POLICIES = (
    "aegiskv",
    "static_independent_fcfs",
    "recompute_only",
    *BASELINE_POLICIES,
)


@dataclass(frozen=True)
class AlgorithmToggles:
    algorithm1: bool = True
    algorithm2: bool = True
    algorithm3: bool = True

    @classmethod
    def from_value(
        cls, value: "AlgorithmToggles | Mapping[str, Any] | str | None"
    ) -> "AlgorithmToggles":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            label = value.strip().lower().replace("_", "-")
            aliases = {
                "full": cls(),
                "none": cls(False, False, False),
                "all-off": cls(False, False, False),
                "no-alg1": cls(algorithm1=False),
                "no-alg2": cls(algorithm2=False),
                "no-alg3": cls(algorithm3=False),
                "alg1-off": cls(algorithm1=False),
                "alg2-off": cls(algorithm2=False),
                "alg3-off": cls(algorithm3=False),
            }
            if label not in aliases:
                raise ValueError(f"unknown AegisKV ablation {value!r}")
            return aliases[label]
        if not isinstance(value, Mapping):
            raise TypeError("ablation must be a mapping, string, or AlgorithmToggles")
        return cls(
            algorithm1=bool(value.get("algorithm1", True)),
            algorithm2=bool(value.get("algorithm2", True)),
            algorithm3=bool(value.get("algorithm3", True)),
        )

    @property
    def label(self) -> str:
        disabled = [
            name
            for name, enabled in (
                ("alg1", self.algorithm1),
                ("alg2", self.algorithm2),
                ("alg3", self.algorithm3),
            )
            if not enabled
        ]
        return "full" if not disabled else "no-" + "-".join(disabled)


@dataclass(frozen=True)
class LargeScaleCellConfig:
    """One deterministic large-scale simulation cell.

    The legacy ``*_gib_per_target`` capacity field names denote capacity per
    physical model shard (HBM GPU or DRAM stage partition).  A logical DP
    target represents its compatible
    TP/PP shard set, so the controller receives that per-shard capacity times
    ``tp_size * pp_size``.  The model KV footprint and stage-lane bandwidth are
    aggregated over the same logical target.
    """

    model: str = "qwen3-8b"
    dp_size: int = 64
    gpus_per_node: int = 4
    concurrency: int = 128
    request_count: int = 2_048
    workflow_count: int = 393
    prompt_scale: float = 0.016
    output_cap_tokens: int = 4_096
    reuse_probability: float = 0.35
    algorithm1_prediction_error_multiplier: float = 1.0
    failure_rate_per_1000_worker_hours: float = 5.952
    repair_time_ms: float = 60_000.0
    control_window_ms: float = 30_000.0
    # Optional steady-state telemetry window.  It never clips the physical
    # worker Poisson process or removes workflows from WCT aggregation.
    measurement_start_ms: float = 0.0
    measurement_end_ms: float | None = None
    hbm_capacity_gib_per_target: float = 4.0
    dram_capacity_gib_per_target: float = 64.0
    hbm_copy_bandwidth_gbps: float = 25.0
    dram_copy_bandwidth_gbps: float = 11.0
    remote_hbm_load_bandwidth_gbps: float = 10.0
    dram_load_bandwidth_gbps: float = 10.0
    network_effective_bandwidth_gbps: float | None = None
    hbm_copy_setup_ms: float = 0.05
    dram_copy_setup_ms: float = 0.10
    hbm_load_setup_ms: float = 0.15
    dram_load_setup_ms: float = 0.15
    initial_hbm_alpha: float = 0.5
    minimum_hbm_alpha: float = 0.0
    maximum_hbm_alpha: float = 1.0
    aimd_decrease_factor: float = 0.5
    aimd_increase_step: float = 0.05
    aimd_stable_windows: int = 3
    pressure_queue_depth: int = 2
    hbm_quota_pressure_fraction: float = 0.95
    prefix_chunk_tokens: int = 4_096
    decode_chunk_tokens: int = 256
    hbm_copy_failure_probability: float = 0.0
    dram_copy_failure_probability: float = 0.0
    # Source-failure cells model one fail-stop event at a time.  Target-first
    # sensitivity requires a separate event schedule with a nonzero interval.
    target_failure_probability: float = 0.0
    minimum_faults_per_cell: int = 0
    # A second, fault-conditioned estimand.  Probe tuples are drawn from the
    # policy-independent base active+sealed GPU-shard exposure and evaluated
    # without mutating the simulated controller.  The primary campaign sets
    # this to the preregistered K=64; zero keeps the estimand disabled.
    active_source_probe_count: int = 0
    event_log_limit: int = 0

    def __post_init__(self) -> None:
        if self.model_key not in MODEL_PROFILES:
            raise ValueError(f"unsupported model profile {self.model!r}")
        for name in (
            "dp_size",
            "gpus_per_node",
            "concurrency",
            "request_count",
            "workflow_count",
            "output_cap_tokens",
            "pressure_queue_depth",
            "prefix_chunk_tokens",
            "decode_chunk_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.minimum_faults_per_cell, bool) or int(
            self.minimum_faults_per_cell
        ) < 0:
            raise ValueError("minimum_faults_per_cell must be nonnegative")
        if isinstance(self.active_source_probe_count, bool) or int(
            self.active_source_probe_count
        ) < 0:
            raise ValueError("active_source_probe_count must be nonnegative")
        for name in (
            "prompt_scale",
            "control_window_ms",
            "hbm_capacity_gib_per_target",
            "dram_capacity_gib_per_target",
            "hbm_copy_bandwidth_gbps",
            "dram_copy_bandwidth_gbps",
            "remote_hbm_load_bandwidth_gbps",
            "dram_load_bandwidth_gbps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.network_effective_bandwidth_gbps is not None:
            network = float(self.network_effective_bandwidth_gbps)
            if not math.isfinite(network) or network <= 0.0:
                raise ValueError(
                    "network_effective_bandwidth_gbps must be finite and positive"
                )
        for name in (
            "failure_rate_per_1000_worker_hours",
            "repair_time_ms",
            "measurement_start_ms",
            "algorithm1_prediction_error_multiplier",
            "hbm_copy_setup_ms",
            "dram_copy_setup_ms",
            "hbm_load_setup_ms",
            "dram_load_setup_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.measurement_end_ms is not None:
            end = float(self.measurement_end_ms)
            if not math.isfinite(end) or end <= float(self.measurement_start_ms):
                raise ValueError(
                    "measurement_end_ms must be finite and exceed measurement_start_ms"
                )
        for name in (
            "reuse_probability",
            "hbm_copy_failure_probability",
            "dram_copy_failure_probability",
            "target_failure_probability",
            "initial_hbm_alpha",
            "minimum_hbm_alpha",
            "maximum_hbm_alpha",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if float(self.target_failure_probability) != 0.0:
            raise ValueError(
                "source-failure cells require target_failure_probability=0; "
                "target-first sensitivity needs a separate spaced event schedule"
            )
        quota_pressure = float(self.hbm_quota_pressure_fraction)
        if not math.isfinite(quota_pressure) or not 0.0 < quota_pressure <= 1.0:
            raise ValueError("hbm_quota_pressure_fraction must lie in (0, 1]")
        if not (
            self.minimum_hbm_alpha
            <= self.initial_hbm_alpha
            <= self.maximum_hbm_alpha
        ):
            raise ValueError("initial HBM alpha must lie inside configured bounds")
        if not 0.0 < float(self.aimd_decrease_factor) < 1.0:
            raise ValueError("aimd_decrease_factor must lie in (0, 1)")
        if int(self.event_log_limit) < 0:
            raise ValueError("event_log_limit must be nonnegative")

    @property
    def model_key(self) -> str:
        value = str(self.model).strip().lower()
        return MODEL_ALIASES.get(value, value)

    @property
    def model_profile(self) -> ModelProfile:
        return MODEL_PROFILES[self.model_key]

    @property
    def total_gpus(self) -> int:
        return (
            int(self.dp_size)
            * int(self.model_profile.pp_size)
            * int(self.model_profile.tp_size)
        )

    @property
    def node_count(self) -> int:
        return math.ceil(self.total_gpus / int(self.gpus_per_node))

    @classmethod
    def from_value(
        cls, value: "LargeScaleCellConfig | Mapping[str, Any]"
    ) -> "LargeScaleCellConfig":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("config must be a LargeScaleCellConfig or mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown large-scale cell fields: {unknown}")
        return cls(**dict(value))


@dataclass(frozen=True)
class RequestInput:
    request_id: str
    workflow_id: str
    prompt_tokens: int
    output_tokens: int
    slack_ms: float
    prefix_group: str
    arrival_ms: float | None = None
    input_reuse_tokens: int | None = None
    actual_slack_ms: float | None = None
    # The large-scale Weka projector keeps the scenario-specific base
    # prediction separate from the sensitivity-adjusted ``slack_ms`` policy
    # input.  Generic callers leave both fields unset and retain the historical
    # in-core interpolation path.
    predicted_slack_ms: float | None = None
    policy_slack_multiplier_applied: float | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.workflow_id or not self.prefix_group:
            raise ValueError("request/workflow/prefix identities are required")
        if int(self.prompt_tokens) <= 0 or int(self.output_tokens) < 0:
            raise ValueError("request token counts are invalid")
        if float(self.slack_ms) < 0.0:
            raise ValueError("request slack must be nonnegative")
        if self.arrival_ms is not None and float(self.arrival_ms) < 0.0:
            raise ValueError("arrival_ms must be nonnegative")
        if self.input_reuse_tokens is not None and not (
            0 <= int(self.input_reuse_tokens) <= int(self.prompt_tokens)
        ):
            raise ValueError("input_reuse_tokens must lie within the prompt")
        if self.actual_slack_ms is not None and float(self.actual_slack_ms) < 0.0:
            raise ValueError("actual_slack_ms must be nonnegative")
        if self.predicted_slack_ms is not None and (
            not math.isfinite(float(self.predicted_slack_ms))
            or float(self.predicted_slack_ms) < 0.0
        ):
            raise ValueError("predicted_slack_ms must be finite and nonnegative")
        if self.policy_slack_multiplier_applied is not None:
            multiplier = float(self.policy_slack_multiplier_applied)
            if not math.isfinite(multiplier) or multiplier < 0.0:
                raise ValueError(
                    "policy_slack_multiplier_applied must be finite and nonnegative"
                )
            if self.predicted_slack_ms is None:
                raise ValueError(
                    "pre-applied policy Slack requires its base prediction"
                )


@dataclass
class _ScheduledRequest:
    request: RequestInput
    source_rank: int
    target_rank: int
    arrival_ms: float
    start_ms: float
    base_finish_ms: float
    queue_ms: float
    service_ms: float
    prefill_ms: float
    decode_ms: float
    reused_tokens: int
    chunks: tuple[SealedChunk, ...]
    policy_slack_ms: float
    failure_count: int = 0
    recovery_ms: float = 0.0
    policy_visible_harm_ms: float = 0.0
    actual_visible_harm_ms: float = 0.0
    policy_remaining_slack_at_failure_ms: float = 0.0
    actual_remaining_slack_at_failure_ms: float = 0.0
    restored_tokens: int = 0
    recompute_tokens: int = 0
    recovery_action: str = "NONE"

    @property
    def total_tokens(self) -> int:
        return self.request.prompt_tokens + self.request.output_tokens

    @property
    def adjusted_finish_ms(self) -> float:
        return self.base_finish_ms + self.recovery_ms


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bytes_per_ms(gigabytes_per_second: float) -> float:
    return float(gigabytes_per_second) * 1_000_000.0


def _reuse_bin(reused_tokens: int, prompt_tokens: int) -> str:
    prompt = int(prompt_tokens)
    if prompt <= 0:
        return "NA"
    fraction = min(max(int(reused_tokens) / prompt, 0.0), 1.0)
    if fraction <= 0.0:
        return "none"
    if fraction < 0.90:
        return "partial"
    if fraction < 0.99:
        return "high"
    return "near_complete"


def _gib(value: float) -> int:
    return int(round(float(value) * (1024**3)))


def _sample_weka_like_request(
    rng: random.Random,
    config: LargeScaleCellConfig,
    index: int,
    prefix_groups: list[str],
) -> RequestInput:
    # Quantile-shaped samples derived from the checked Weka context audit.
    raw_prompt = rng.choices(
        (32_000, 142_016, 549_504, 863_471, 989_824),
        weights=(0.15, 0.35, 0.30, 0.15, 0.05),
        k=1,
    )[0]
    prompt = max(int(round(raw_prompt * float(config.prompt_scale))), 1)
    output = min(
        int(
            rng.choices(
                (128, 444, 1_024, 2_659, 4_096),
                weights=(0.15, 0.35, 0.25, 0.15, 0.10),
                k=1,
            )[0]
        ),
        int(config.output_cap_tokens),
    )
    model = config.model_profile
    if prompt + output > model.max_context_tokens:
        output = max(model.max_context_tokens - prompt, 0)
    workflow_id = f"workflow-{index % int(config.workflow_count):04d}"
    if prefix_groups and rng.random() < float(config.reuse_probability):
        prefix_group = rng.choice(prefix_groups)
    else:
        prefix_group = f"prefix-{index:06d}"
        prefix_groups.append(prefix_group)
    slack_class = rng.random()
    if slack_class < 0.38:
        slack = rng.uniform(0.0, 50.0)
    elif slack_class < 0.76:
        slack = rng.uniform(50.0, 250.0)
    else:
        slack = rng.uniform(250.0, 1_200.0)
    return RequestInput(
        request_id=f"request-{index:06d}",
        workflow_id=workflow_id,
        prompt_tokens=prompt,
        output_tokens=output,
        slack_ms=slack,
        prefix_group=prefix_group,
    )


def generate_workload(
    config: LargeScaleCellConfig | Mapping[str, Any], seed: int
) -> tuple[RequestInput, ...]:
    """Generate a deterministic Weka-shaped workload when no trace is supplied."""

    resolved = LargeScaleCellConfig.from_value(config)
    rng = random.Random(int(seed) ^ 0xAE615)
    groups: list[str] = []
    requests = [
        _sample_weka_like_request(rng, resolved, index, groups)
        for index in range(int(resolved.request_count))
    ]
    # Derive an arrival stream whose infinite-server expected concurrency is
    # the requested value.  Rank-local queueing is applied later.
    model = resolved.model_profile
    mean_service = statistics.fmean(
        model.prefill_setup_ms
        + request.prompt_tokens * model.prefill_ms_per_token
        + request.output_tokens * model.decode_ms_per_token
        for request in requests
    )
    mean_interarrival = mean_service / max(int(resolved.concurrency), 1)
    arrival = 0.0
    stamped: list[RequestInput] = []
    for request in requests:
        arrival += rng.expovariate(1.0 / max(mean_interarrival, 1e-9))
        stamped.append(replace(request, arrival_ms=arrival))
    return tuple(stamped)


def _normalize_workload(
    config: LargeScaleCellConfig,
    seed: int,
    workload: Sequence[RequestInput | Mapping[str, Any]] | None,
) -> tuple[RequestInput, ...]:
    if workload is None:
        return generate_workload(config, seed)
    rows: list[RequestInput] = []
    for item in workload:
        row = item if isinstance(item, RequestInput) else RequestInput(**dict(item))
        if row.prompt_tokens + row.output_tokens > config.model_profile.max_context_tokens:
            raise ValueError(
                f"request {row.request_id!r} exceeds model context capacity"
            )
        rows.append(row)
    if not rows:
        raise ValueError("workload cannot be empty")
    if any(row.arrival_ms is None for row in rows):
        raise ValueError("caller-supplied workload must provide every arrival_ms")
    identities = [row.request_id for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("workload request IDs must be unique")
    return tuple(sorted(rows, key=lambda row: (float(row.arrival_ms or 0.0), row.request_id)))


def _schedule_requests(
    config: LargeScaleCellConfig,
    workload: Sequence[RequestInput],
) -> list[_ScheduledRequest]:
    model = config.model_profile
    rank_available = [0.0] * int(config.dp_size)
    prefix_owner: dict[str, tuple[int, int]] = {}
    scheduled: list[_ScheduledRequest] = []
    for index, request in enumerate(workload):
        arrival = float(request.arrival_ms or 0.0)
        reused_tokens = 0
        if request.input_reuse_tokens is not None:
            reused_tokens = min(
                int(request.prompt_tokens), int(request.input_reuse_tokens)
            )
            if request.prefix_group in prefix_owner:
                source_rank = prefix_owner[request.prefix_group][0]
            else:
                source_rank = index % int(config.dp_size)
                prefix_owner[request.prefix_group] = (
                    source_rank,
                    int(request.prompt_tokens),
                )
        elif request.prefix_group in prefix_owner:
            preferred_rank, stored_prompt = prefix_owner[request.prefix_group]
            source_rank = preferred_rank
            reused_tokens = min(int(request.prompt_tokens), int(stored_prompt))
        else:
            # Round robin is the pinned controller's default DP routing shape.
            source_rank = index % int(config.dp_size)
            prefix_owner[request.prefix_group] = (
                source_rank,
                int(request.prompt_tokens),
            )
        target_rank = _peer_target_rank(config, source_rank)
        uncached_prompt = max(int(request.prompt_tokens) - reused_tokens, 0)
        prefill = model.prefill_setup_ms + uncached_prompt * model.prefill_ms_per_token
        decode = int(request.output_tokens) * model.decode_ms_per_token
        service = max(prefill + decode, 1e-6)
        start = max(arrival, rank_available[source_rank])
        finish = start + service
        rank_available[source_rank] = finish
        chunks = sealed_chunks(
            request.prompt_tokens,
            request.output_tokens,
            prefix_chunk_tokens=config.prefix_chunk_tokens,
            decode_chunk_tokens=config.decode_chunk_tokens,
        )
        actual_slack = (
            float(request.actual_slack_ms)
            if request.actual_slack_ms is not None
            else float(request.slack_ms)
        )
        if request.policy_slack_multiplier_applied is not None:
            applied_multiplier = float(request.policy_slack_multiplier_applied)
            if not math.isclose(
                applied_multiplier,
                float(config.algorithm1_prediction_error_multiplier),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "caller-precomputed policy Slack multiplier does not match "
                    "the cell configuration"
                )
            policy_slack = max(float(request.slack_ms), 0.0)
        else:
            policy_slack = max(
                0.0,
                actual_slack
                + float(config.algorithm1_prediction_error_multiplier)
                * (float(request.slack_ms) - actual_slack),
            )
        scheduled.append(
            _ScheduledRequest(
                request=request,
                source_rank=source_rank,
                target_rank=target_rank,
                arrival_ms=arrival,
                start_ms=start,
                base_finish_ms=finish,
                queue_ms=start - arrival,
                service_ms=service,
                prefill_ms=prefill,
                decode_ms=decode,
                reused_tokens=reused_tokens,
                chunks=chunks,
                policy_slack_ms=policy_slack,
            )
        )
    return scheduled


def _slack_evidence(
    request: RequestInput, *, enabled: bool, policy_slack_ms: float | None = None
) -> dict[str, Any]:
    usable = bool(enabled)
    value = (
        float(request.slack_ms)
        if policy_slack_ms is None
        else max(float(policy_slack_ms), 0.0)
    )
    return {
        "workflow_slack_ms": value if usable else 0.0,
        "aegis_slack_semantics": RESIDUAL_DAG_SLACK_SEMANTICS,
        "aegis_slack_uses_future_information": False,
        "aegis_slack_residual_dag_visible": True,
        "aegis_slack_metadata_only": True,
        "aegis_slack_usable": usable,
        "aegis_slack_reason": (
            "synthetic_visible_residual_dag" if usable else "algorithm1_disabled"
        ),
        "aegis_slack_graph_version": f"{request.workflow_id}:visible-v1",
        "aegis_slack_node_id": request.request_id,
        "aegis_slack_agent_id": request.request_id,
        "aegis_slack_input_provenance_sha256": _sha256_text(
            f"{request.workflow_id}:{request.request_id}:{value}"
        ),
    }


def _recovery_timing_from_bandwidth(
    *, setup_ms: float, kv_bytes_per_token: int, bandwidth_gbps: float, lanes: int
) -> RecoveryTiming:
    aggregate = _bytes_per_ms(bandwidth_gbps) * max(int(lanes), 1)
    return RecoveryTiming(
        setup_ms=float(setup_ms),
        ms_per_token=int(kv_bytes_per_token) / aggregate,
    )


class _CellAdapter:
    def register(self, request: _ScheduledRequest, *, slack_enabled: bool) -> None:
        raise NotImplementedError

    def seal(
        self, request: _ScheduledRequest, chunk: SealedChunk, time_ms: float
    ) -> str:
        raise NotImplementedError

    def finish(self, request: _ScheduledRequest, time_ms: float) -> None:
        raise NotImplementedError

    def rotate_target(self, request: _ScheduledRequest, time_ms: float) -> str:
        return f"target-{request.target_rank}"

    def fail_worker(self, worker_rank: int, time_ms: float) -> dict[str, Any]:
        """Invalidate protection physically resident on a failed worker."""

        del worker_rank, time_ms
        return {"invalidated_hbm_requests": 0, "invalidated_hbm_bytes": 0}

    def repair_worker(self, worker_rank: int, time_ms: float) -> dict[str, Any]:
        """Re-enable future HBM admission without resurrecting old copies."""

        del worker_rank, time_ms
        return {"future_hbm_admission_reenabled": False}

    def estimate_failure(
        self,
        request: _ScheduledRequest,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        time_ms: float,
        activated_target: str,
        slack_enabled: bool,
    ) -> RecoveryDecision:
        raise NotImplementedError

    def estimate_recovery_at(
        self,
        request: _ScheduledRequest,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        time_ms: float,
        activated_target: str,
        slack_enabled: bool,
    ) -> RecoveryDecision:
        """Pure shadow estimate after the adapter's normal time advance.

        Baseline/static/recompute adapters already implement their fail-now
        estimator without policy-state mutations.  AegisKV overrides this
        method because its primary uptime-failure path intentionally records
        the observed source frontier and slack metadata.
        """

        return self.estimate_failure(
            request,
            produced_tokens=produced_tokens,
            remaining_slack_ms=remaining_slack_ms,
            time_ms=time_ms,
            activated_target=activated_target,
            slack_enabled=slack_enabled,
        )

    def control_window(
        self,
        *,
        time_ms: float,
        pressure_by_target: Mapping[int, bool],
        algorithm3: bool,
        apply_aimd_update: bool,
        active_requests: int,
        queued_requests: int,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError


def _network_effective_bandwidth(config: LargeScaleCellConfig) -> float:
    if config.network_effective_bandwidth_gbps is not None:
        return float(config.network_effective_bandwidth_gbps)
    return min(
        float(config.dram_copy_bandwidth_gbps),
        float(config.remote_hbm_load_bandwidth_gbps),
        float(config.dram_load_bandwidth_gbps),
    )


def _dp_rank_node(config: LargeScaleCellConfig, rank: int) -> int:
    return (
        int(rank) * int(config.model_profile.shard_count)
    ) // int(config.gpus_per_node)


def _logical_dp_ranks_per_node(config: LargeScaleCellConfig) -> int:
    return max(
        int(config.gpus_per_node) // int(config.model_profile.shard_count),
        1,
    )


def _peer_target_rank(config: LargeScaleCellConfig, source_rank: int) -> int:
    """Bind a compatible peer, preferring a node-local DP replica."""

    size = int(config.dp_size)
    source = int(source_rank) % size
    per_node = _logical_dp_ranks_per_node(config)
    group_start = (source // per_node) * per_node
    group_end = min(group_start + per_node, size)
    if group_end - group_start > 1:
        return group_start + (source - group_start + 1) % (group_end - group_start)
    return (source + 1) % size


def _peer_source_rank(config: LargeScaleCellConfig, target_rank: int) -> int:
    """Inverse of the deterministic node-local peer binding."""

    size = int(config.dp_size)
    target = int(target_rank) % size
    per_node = _logical_dp_ranks_per_node(config)
    group_start = (target // per_node) * per_node
    group_end = min(group_start + per_node, size)
    if group_end - group_start > 1:
        return group_start + (target - group_start - 1) % (group_end - group_start)
    return (target - 1) % size


def _target_hbm_copy_bandwidth(
    config: LargeScaleCellConfig, target_rank: int | None
) -> float:
    cap = float(config.hbm_copy_bandwidth_gbps)
    if target_rank is None:
        return cap
    source_rank = _peer_source_rank(config, int(target_rank))
    if _dp_rank_node(config, source_rank) == _dp_rank_node(config, target_rank):
        return cap
    return min(cap, _network_effective_bandwidth(config))


def _full_design_config(
    config: LargeScaleCellConfig, *, target_rank: int | None = None
) -> AegisFullDesignConfig:
    model = config.model_profile
    lanes = model.shard_count
    hbm_capacity = _gib(config.hbm_capacity_gib_per_target) * lanes
    dram_capacity = _gib(config.dram_capacity_gib_per_target) * lanes
    return AegisFullDesignConfig(
        hbm_capacity_bytes=hbm_capacity,
        dram_capacity_bytes=dram_capacity,
        kv_bytes_per_token=model.kv_bytes_per_token,
        hbm_copy=CalibratedTransferPath(
            setup_ms=config.hbm_copy_setup_ms,
            bandwidth_bytes_per_ms=(
                _bytes_per_ms(
                    _target_hbm_copy_bandwidth(config, target_rank)
                )
                * lanes
            ),
            provenance="synthetic-large-scale-cell:hbm-copy",
        ),
        dram_copy=CalibratedTransferPath(
            setup_ms=config.dram_copy_setup_ms,
            bandwidth_bytes_per_ms=(
                _bytes_per_ms(
                    min(
                        float(config.dram_copy_bandwidth_gbps),
                        _network_effective_bandwidth(config),
                    )
                )
                * lanes
            ),
            provenance="synthetic-large-scale-cell:source-d2h",
        ),
        recompute=RecoveryTiming(
            model.recompute_setup_ms, model.recompute_ms_per_token
        ),
        hbm_load=_recovery_timing_from_bandwidth(
            setup_ms=config.hbm_load_setup_ms,
            kv_bytes_per_token=model.kv_bytes_per_token,
            bandwidth_gbps=min(
                float(config.remote_hbm_load_bandwidth_gbps),
                _network_effective_bandwidth(config),
            ),
            lanes=lanes,
        ),
        dram_load=_recovery_timing_from_bandwidth(
            setup_ms=config.dram_load_setup_ms,
            kv_bytes_per_token=model.kv_bytes_per_token,
            bandwidth_gbps=min(
                float(config.dram_load_bandwidth_gbps),
                _network_effective_bandwidth(config),
            ),
            lanes=lanes,
        ),
        prefix_chunk_tokens=config.prefix_chunk_tokens,
        decode_chunk_tokens=config.decode_chunk_tokens,
        pp_size=model.pp_size,
        tp_size=model.tp_size,
        initial_hbm_alpha=config.initial_hbm_alpha,
        minimum_hbm_alpha=config.minimum_hbm_alpha,
        maximum_hbm_alpha=config.maximum_hbm_alpha,
        aimd_decrease_factor=config.aimd_decrease_factor,
        aimd_increase_step=config.aimd_increase_step,
        aimd_stable_windows=config.aimd_stable_windows,
        hbm_copy_failure_probability=config.hbm_copy_failure_probability,
        dram_copy_failure_probability=config.dram_copy_failure_probability,
        random_salt="aegiskv-large-scale-cell-v1",
        event_log_limit=config.event_log_limit,
    )


class _AegisAdapter(_CellAdapter):
    def __init__(
        self, config: LargeScaleCellConfig, toggles: AlgorithmToggles
    ) -> None:
        self.config = config
        self.toggles = toggles
        self.runtimes = [
            AegisFullDesignRuntime(
                _full_design_config(config, target_rank=rank),
                target_id=f"target-{rank}",
                dram_pool_id=f"store-dram-partition-{rank}",
                dram_failure_domain="store-cluster",
            )
            for rank in range(int(config.dp_size))
        ]
        self._request_target: dict[str, int] = {}
        self._failed_hbm_targets: set[int] = set()
        self._admission_outcomes: Counter[str] = Counter()
        self._protection_copy_attempted_bytes = 0
        self._minimum_hbm_allocation_bytes = (
            min(config.prefix_chunk_tokens, config.decode_chunk_tokens)
            * config.model_profile.kv_bytes_per_token
        )

    def _runtime(self, request: _ScheduledRequest) -> AegisFullDesignRuntime:
        return self.runtimes[self._request_target[request.request.request_id]]

    def register(self, request: _ScheduledRequest, *, slack_enabled: bool) -> None:
        target = int(request.target_rank)
        self._request_target[request.request.request_id] = target
        self.runtimes[target].register_request(
            request_id=request.request.request_id,
            source_worker=f"dp-{request.source_rank}",
            source_failure_domain=(
                f"node-{_dp_rank_node(self.config, request.source_rank)}"
            ),
            hbm_target=f"target-{target}",
            hbm_failure_domain=(
                f"node-{_dp_rank_node(self.config, target)}"
            ),
            route_version=1,
            prompt_tokens=request.request.prompt_tokens,
            output_tokens=request.request.output_tokens,
            slack_ms=request.policy_slack_ms if slack_enabled else 0.0,
            model_revision=self.config.model_profile.name,
            kv_abi="sglsim-metadata-kv-v1",
            parallel_layout=(
                f"dp{self.config.dp_size}-tp{self.config.model_profile.tp_size}"
                f"-pp{self.config.model_profile.pp_size}"
            ),
            time_ms=request.arrival_ms,
            slack_evidence=_slack_evidence(
                request.request,
                enabled=slack_enabled,
                policy_slack_ms=request.policy_slack_ms,
            ),
        )

    def seal(
        self, request: _ScheduledRequest, chunk: SealedChunk, time_ms: float
    ) -> str:
        allowed: Iterable[ProtectionTier]
        if self.toggles.algorithm2:
            allowed = (
                (ProtectionTier.DRAM,)
                if int(request.target_rank) in self._failed_hbm_targets
                else (ProtectionTier.HBM, ProtectionTier.DRAM)
            )
        else:
            # ``-Alg2`` is implemented by the separate static adapter selected
            # in ``run_cell``; reaching this path is a fail-closed error.
            raise AssertionError("Algorithm 2 disabled on the SlackLandlord adapter")
        decision = self._runtime(request).seal_chunk(
            request.request.request_id,
            chunk,
            time_ms=time_ms,
            allowed_tiers=allowed,
        )
        self._admission_outcomes[decision.intended_residency.value] += 1
        self._protection_copy_attempted_bytes += (
            int(chunk.token_count)
            * int(self.config.model_profile.kv_bytes_per_token)
            * (int(decision.hbm.admitted) + int(decision.dram.admitted))
        )
        return decision.intended_residency.value

    def fail_worker(self, worker_rank: int, time_ms: float) -> dict[str, Any]:
        rank = int(worker_rank)
        self._failed_hbm_targets.add(rank)
        runtime = self.runtimes[rank]
        runtime.advance_to(time_ms)
        invalidated_requests = 0
        invalidated_bytes = 0
        for request_id, target in tuple(self._request_target.items()):
            if target != rank or not runtime.has_request(request_id):
                continue
            result = runtime.controller.invalidate_tier(
                request_id,
                ProtectionTier.HBM,
                reason="peer_hbm_worker_fail_stop",
            )
            released = int(result["released_bytes"])
            if released:
                invalidated_requests += 1
                invalidated_bytes += released
        return {
            "invalidated_hbm_requests": invalidated_requests,
            "invalidated_hbm_bytes": invalidated_bytes,
        }

    def repair_worker(self, worker_rank: int, time_ms: float) -> dict[str, Any]:
        del time_ms
        rank = int(worker_rank)
        was_failed = rank in self._failed_hbm_targets
        self._failed_hbm_targets.discard(rank)
        return {"future_hbm_admission_reenabled": was_failed}

    def finish(self, request: _ScheduledRequest, time_ms: float) -> None:
        runtime = self._runtime(request)
        if runtime.has_request(request.request.request_id):
            runtime.finish_request(request.request.request_id, time_ms=time_ms)

    def rotate_target(self, request: _ScheduledRequest, time_ms: float) -> str:
        runtime = self._runtime(request)
        replacement = (request.target_rank + 1) % int(self.config.dp_size)
        if runtime.has_request(request.request.request_id):
            runtime.rotate_target(
                request.request.request_id,
                failed_target=f"target-{request.target_rank}",
                replacement_target=f"target-{replacement}",
                replacement_failure_domain=(
                    f"node-{_dp_rank_node(self.config, replacement)}"
                ),
                time_ms=time_ms,
            )
        return f"target-{replacement}"

    def estimate_failure(
        self,
        request: _ScheduledRequest,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        time_ms: float,
        activated_target: str,
        slack_enabled: bool,
    ) -> RecoveryDecision:
        runtime = self._runtime(request)
        return runtime.estimate_source_failure(
            request.request.request_id,
            activated_target=activated_target,
            source_token_end=produced_tokens,
            remaining_slack_ms=(remaining_slack_ms if slack_enabled else 0.0),
            time_ms=time_ms,
            slack_evidence=_slack_evidence(
                request.request,
                enabled=slack_enabled,
                policy_slack_ms=request.policy_slack_ms,
            ),
        )

    def estimate_recovery_at(
        self,
        request: _ScheduledRequest,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        time_ms: float,
        activated_target: str,
        slack_enabled: bool,
    ) -> RecoveryDecision:
        runtime = self._runtime(request)
        return runtime.estimate_recovery_at(
            request.request.request_id,
            activated_target=activated_target,
            source_token_end=produced_tokens,
            slack_ms=(remaining_slack_ms if slack_enabled else 0.0),
            time_ms=time_ms,
        )

    def _aggregate(self, *, include_runtime: bool = False) -> dict[str, Any]:
        controllers = [
            runtime.controller.telemetry_snapshot() for runtime in self.runtimes
        ]
        hbm = [row["pools"]["HBM"] for row in controllers]
        dram = [row["pools"]["DRAM"] for row in controllers]
        raw_alpha = [float(row["hbm_alpha"]) for row in controllers]
        stable_windows = [int(row["stable_windows"]) for row in controllers]
        effective_alpha = [
            (
                0.0
                if rank in self._failed_hbm_targets
                or int(pool["limit_bytes"]) < self._minimum_hbm_allocation_bytes
                else alpha
            )
            for rank, (pool, alpha) in enumerate(
                zip(hbm, raw_alpha, strict=True)
            )
        ]
        payload = {
            "hbm_used_bytes": sum(int(pool["used_bytes"]) for pool in hbm),
            "hbm_limit_bytes": sum(
                0 if rank in self._failed_hbm_targets else int(pool["limit_bytes"])
                for rank, pool in enumerate(hbm)
            ),
            "hbm_capacity_bytes": sum(int(pool["capacity_bytes"]) for pool in hbm),
            "hbm_available_capacity_bytes": sum(
                0
                if rank in self._failed_hbm_targets
                else int(pool["capacity_bytes"])
                for rank, pool in enumerate(hbm)
            ),
            "failed_hbm_target_count": len(self._failed_hbm_targets),
            "dram_used_bytes": sum(int(pool["used_bytes"]) for pool in dram),
            "dram_limit_bytes": sum(int(pool["limit_bytes"]) for pool in dram),
            "dram_capacity_bytes": sum(int(pool["capacity_bytes"]) for pool in dram),
            "controller_alpha_mean": statistics.fmean(raw_alpha),
            "controller_alpha_min": min(raw_alpha),
            "controller_alpha_max": max(raw_alpha),
            "controller_stable_windows_mean": statistics.fmean(
                stable_windows
            ),
            "controller_stable_windows_min": min(stable_windows),
            "controller_stable_windows_max": max(stable_windows),
            "effective_alpha_mean": statistics.fmean(effective_alpha),
            "effective_alpha_min": min(effective_alpha),
            "effective_alpha_max": max(effective_alpha),
            "effective_alpha_zero_targets": sum(
                value == 0.0 for value in effective_alpha
            ),
            "admission_outcomes": dict(sorted(self._admission_outcomes.items())),
            "protection_copy_attempted_bytes": (
                self._protection_copy_attempted_bytes
            ),
        }
        if include_runtime:
            runtime_snapshots = [runtime.snapshot() for runtime in self.runtimes]
            audited_copy_attempted_bytes = sum(
                int(path["attempted_bytes"])
                for snapshot in runtime_snapshots
                for path in snapshot["protection_paths"].values()
            )
            if audited_copy_attempted_bytes != self._protection_copy_attempted_bytes:
                raise AssertionError(
                    "Aegis protection-copy byte accounting did not reconcile"
                )
            payload["target_snapshots"] = [
                {
                    **snapshot,
                    "controller_telemetry": controllers[index],
                }
                for index, snapshot in enumerate(runtime_snapshots)
            ]
        return payload

    def control_window(
        self,
        *,
        time_ms: float,
        pressure_by_target: Mapping[int, bool],
        algorithm3: bool,
        apply_aimd_update: bool,
        active_requests: int,
        queued_requests: int,
    ) -> dict[str, Any]:
        pressure_count = 0
        for rank, runtime in enumerate(self.runtimes):
            runtime.advance_to(time_ms)
            pressure = bool(pressure_by_target.get(rank, False))
            controller = runtime.controller.telemetry_snapshot()
            hbm_pool = controller["pools"]["HBM"]
            quota_pressure = (
                int(hbm_pool["limit_bytes"]) > 0
                and int(hbm_pool["used_bytes"])
                >= float(self.config.hbm_quota_pressure_fraction)
                * int(hbm_pool["limit_bytes"])
            )
            pressure = pressure or quota_pressure
            pressure_count += int(pressure)
            if algorithm3 and apply_aimd_update:
                runtime.control_hbm_quota(pressure=pressure, time_ms=time_ms)
        aggregate = self._aggregate()
        aggregate.update(
            {
                "time_ms": time_ms,
                "active_requests": int(active_requests),
                "queued_requests": int(queued_requests),
                "pressure_targets": pressure_count,
                "pressure_fraction": pressure_count / max(len(self.runtimes), 1),
                "algorithm3_enabled": bool(algorithm3),
            }
        )
        aggregate.pop("target_snapshots", None)
        return aggregate

    def flush(self) -> None:
        for runtime in self.runtimes:
            runtime.flush()
            runtime.validate_invariants()

    def snapshot(self) -> dict[str, Any]:
        return self._aggregate(include_runtime=True)


def _baseline_config(
    config: LargeScaleCellConfig,
    policy: BaselinePolicy,
    *,
    target_rank: int | None = None,
) -> ProtectionBaselineConfig:
    full = _full_design_config(config, target_rank=target_rank)
    return ProtectionBaselineConfig(
        policy=policy,
        hbm_capacity_bytes=full.hbm_capacity_bytes,
        dram_capacity_bytes=full.dram_capacity_bytes,
        kv_bytes_per_token=full.kv_bytes_per_token,
        hbm_copy=full.hbm_copy,
        dram_copy=full.dram_copy,
        recompute=full.recompute,
        hbm_load=full.hbm_load,
        dram_load=full.dram_load,
        pp_size=full.pp_size,
        tp_size=full.tp_size,
        parity_data_shards=max(full.shard_count, 8),
        parity_shards=2,
        parity_encode_ms_per_token=full.recompute.ms_per_token / 16.0,
        parity_decode_ms_per_token=full.hbm_load.ms_per_token / 8.0,
        event_log_limit=config.event_log_limit,
    )


class _BaselineAdapter(_CellAdapter):
    def __init__(self, config: LargeScaleCellConfig, policy: BaselinePolicy) -> None:
        self.config = config
        self.policy = policy
        self.runtimes = [
            ProtectionBaselineRuntime(
                _baseline_config(config, policy, target_rank=rank),
                target_id=f"target-{rank}",
            )
            for rank in range(int(config.dp_size))
        ]
        self._request_target: dict[str, int] = {}
        self._failed_hbm_targets: set[int] = set()
        self._admitted = 0
        self._rejected = 0

    def _runtime(self, request: _ScheduledRequest) -> ProtectionBaselineRuntime:
        return self.runtimes[self._request_target[request.request.request_id]]

    def register(self, request: _ScheduledRequest, *, slack_enabled: bool) -> None:
        del slack_enabled
        target = int(request.target_rank)
        self._request_target[request.request.request_id] = target
        self.runtimes[target].register_request(
            BaselineRequestProfile(
                request_id=request.request.request_id,
                source_worker=f"dp-{request.source_rank}",
                source_failure_domain=(
                    f"node-{_dp_rank_node(self.config, request.source_rank)}"
                ),
                target_worker=f"target-{target}",
                target_failure_domain=(
                    f"node-{_dp_rank_node(self.config, target)}"
                ),
                route_version=1,
                total_tokens=request.total_tokens,
                slack_ms=request.policy_slack_ms,
            ),
            time_ms=request.arrival_ms,
        )

    def seal(
        self, request: _ScheduledRequest, chunk: SealedChunk, time_ms: float
    ) -> str:
        if (
            self.policy
            in (BaselinePolicy.DEJAVU_NEIGHBOR, BaselinePolicy.KEVLARFLOW_RING)
            and int(request.target_rank) in self._failed_hbm_targets
        ):
            # Keep the per-request sealed-chunk ordinal progressive even while
            # the fixed HBM target is unavailable.  The first call blocks and
            # invalidates the request replica; later calls are idempotent.  We
            # still pass the chunk through ``seal_chunk`` so a genuine replay
            # or out-of-order event remains a hard error and the missing
            # prefix can never be silently resumed after repair.
            self._runtime(request).invalidate_target_failure(
                request.request.request_id, time_ms=time_ms
            )
        admission = self._runtime(request).seal_chunk(
            request.request.request_id, chunk, time_ms=time_ms
        )
        if admission.admitted:
            self._admitted += 1
            return (
                "HBM_ONLY"
                if self._runtime(request).config.tier.value == "HBM"
                else "DRAM_ONLY"
            )
        else:
            self._rejected += 1
            return "BYPASS"

    def fail_worker(self, worker_rank: int, time_ms: float) -> dict[str, Any]:
        rank = int(worker_rank)
        self._failed_hbm_targets.add(rank)
        runtime = self.runtimes[rank]
        invalidated_requests = 0
        invalidated_bytes = 0
        for request_id, target in tuple(self._request_target.items()):
            if target != rank:
                continue
            released = runtime.invalidate_target_failure(
                request_id, time_ms=time_ms
            )
            if released:
                invalidated_requests += 1
                invalidated_bytes += released
        return {
            "invalidated_hbm_requests": invalidated_requests,
            "invalidated_hbm_bytes": invalidated_bytes,
        }

    def repair_worker(self, worker_rank: int, time_ms: float) -> dict[str, Any]:
        del time_ms
        rank = int(worker_rank)
        was_failed = rank in self._failed_hbm_targets
        self._failed_hbm_targets.discard(rank)
        return {"future_hbm_admission_reenabled": was_failed}

    def finish(self, request: _ScheduledRequest, time_ms: float) -> None:
        runtime = self._runtime(request)
        if runtime.has_request(request.request.request_id):
            runtime.finish_request(request.request.request_id, time_ms=time_ms)

    def rotate_target(self, request: _ScheduledRequest, time_ms: float) -> str:
        runtime = self._runtime(request)
        replacement = (request.target_rank + 1) % int(self.config.dp_size)
        runtime.rotate_target(
            request.request.request_id,
            replacement_target=f"target-{replacement}",
            replacement_failure_domain=(
                f"node-{_dp_rank_node(self.config, replacement)}"
            ),
            time_ms=time_ms,
        )
        return f"target-{replacement}"

    def estimate_failure(
        self,
        request: _ScheduledRequest,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        time_ms: float,
        activated_target: str,
        slack_enabled: bool,
    ) -> RecoveryDecision:
        return self._runtime(request).estimate_source_failure(
            request.request.request_id,
            produced_tokens=produced_tokens,
            remaining_slack_ms=(remaining_slack_ms if slack_enabled else 0.0),
            activated_target=activated_target,
            time_ms=time_ms,
        )

    def _aggregate(self) -> dict[str, Any]:
        snapshots = [runtime.snapshot() for runtime in self.runtimes]
        is_hbm = self.policy in (
            BaselinePolicy.DEJAVU_NEIGHBOR,
            BaselinePolicy.KEVLARFLOW_RING,
        )
        used = sum(int(row["used_bytes"]) for row in snapshots)
        capacity = sum(int(row["capacity_bytes"]) for row in snapshots)
        available_capacity = (
            sum(
                int(row["capacity_bytes"])
                for rank, row in enumerate(snapshots)
                if rank not in self._failed_hbm_targets
            )
            if is_hbm
            else capacity
        )
        copy_attempted = sum(
            int(row["metrics"]["copy_stored_bytes"]) for row in snapshots
        )
        return {
            "hbm_used_bytes": used if is_hbm else 0,
            "hbm_limit_bytes": available_capacity if is_hbm else 0,
            "hbm_capacity_bytes": capacity if is_hbm else 0,
            "hbm_available_capacity_bytes": (
                available_capacity if is_hbm else 0
            ),
            "failed_hbm_target_count": (
                len(self._failed_hbm_targets) if is_hbm else 0
            ),
            "dram_used_bytes": used if not is_hbm else 0,
            "dram_limit_bytes": capacity if not is_hbm else 0,
            "dram_capacity_bytes": capacity if not is_hbm else 0,
            "controller_alpha_mean": 1.0 if is_hbm else 0.0,
            "controller_alpha_min": 1.0 if is_hbm else 0.0,
            "controller_alpha_max": 1.0 if is_hbm else 0.0,
            "controller_stable_windows_mean": 0.0,
            "controller_stable_windows_min": 0,
            "controller_stable_windows_max": 0,
            "effective_alpha_mean": 1.0 if is_hbm else 0.0,
            "effective_alpha_min": 1.0 if is_hbm else 0.0,
            "effective_alpha_max": 1.0 if is_hbm else 0.0,
            "effective_alpha_zero_targets": 0 if is_hbm else len(self.runtimes),
            "admission_outcomes": {
                "ADMITTED": self._admitted,
                "REJECTED": self._rejected,
            },
            "protection_copy_attempted_bytes": copy_attempted,
            "target_snapshots": snapshots,
        }

    def control_window(
        self,
        *,
        time_ms: float,
        pressure_by_target: Mapping[int, bool],
        algorithm3: bool,
        apply_aimd_update: bool,
        active_requests: int,
        queued_requests: int,
    ) -> dict[str, Any]:
        del algorithm3, apply_aimd_update
        for runtime in self.runtimes:
            runtime.advance_to(time_ms)
        aggregate = self._aggregate()
        aggregate.update(
            {
                "time_ms": time_ms,
                "active_requests": int(active_requests),
                "queued_requests": int(queued_requests),
                "pressure_targets": sum(bool(value) for value in pressure_by_target.values()),
                "pressure_fraction": (
                    sum(bool(value) for value in pressure_by_target.values())
                    / max(len(self.runtimes), 1)
                ),
                "algorithm3_enabled": False,
            }
        )
        aggregate.pop("target_snapshots", None)
        return aggregate

    def flush(self) -> None:
        for runtime in self.runtimes:
            runtime.flush()
            runtime.validate_invariants()

    def snapshot(self) -> dict[str, Any]:
        return self._aggregate()


class _RecomputeAdapter(_CellAdapter):
    def __init__(self, config: LargeScaleCellConfig) -> None:
        self.config = config
        self._active: set[str] = set()

    def register(self, request: _ScheduledRequest, *, slack_enabled: bool) -> None:
        del slack_enabled
        self._active.add(request.request.request_id)

    def seal(
        self, request: _ScheduledRequest, chunk: SealedChunk, time_ms: float
    ) -> str:
        del request, chunk, time_ms
        return "BYPASS"

    def finish(self, request: _ScheduledRequest, time_ms: float) -> None:
        del time_ms
        self._active.discard(request.request.request_id)

    def estimate_failure(
        self,
        request: _ScheduledRequest,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        time_ms: float,
        activated_target: str,
        slack_enabled: bool,
    ) -> RecoveryDecision:
        del time_ms
        model = self.config.model_profile
        recovery = (
            model.recompute_setup_ms
            + int(produced_tokens) * model.recompute_ms_per_token
            if produced_tokens
            else 0.0
        )
        slack = remaining_slack_ms if slack_enabled else 0.0
        return RecoveryDecision(
            request_id=request.request.request_id,
            action="RECOMPUTE",
            restored_tokens=0,
            recompute_tokens=int(produced_tokens),
            recovery_ms=recovery,
            slack_ms=slack,
            visible_harm_ms=max(recovery - slack, 0.0),
            route_version=1,
            activated_target=activated_target,
        )

    def control_window(
        self,
        *,
        time_ms: float,
        pressure_by_target: Mapping[int, bool],
        algorithm3: bool,
        apply_aimd_update: bool,
        active_requests: int,
        queued_requests: int,
    ) -> dict[str, Any]:
        del algorithm3, apply_aimd_update
        pressure = sum(bool(value) for value in pressure_by_target.values())
        return {
            "time_ms": time_ms,
            "active_requests": int(active_requests),
            "queued_requests": int(queued_requests),
            "pressure_targets": pressure,
            "pressure_fraction": pressure / max(int(self.config.dp_size), 1),
            "algorithm3_enabled": False,
            "hbm_used_bytes": 0,
            "hbm_limit_bytes": 0,
            "hbm_capacity_bytes": 0,
            "hbm_available_capacity_bytes": 0,
            "failed_hbm_target_count": 0,
            "dram_used_bytes": 0,
            "dram_limit_bytes": 0,
            "dram_capacity_bytes": 0,
            "controller_alpha_mean": 0.0,
            "controller_alpha_min": 0.0,
            "controller_alpha_max": 0.0,
            "controller_stable_windows_mean": 0.0,
            "controller_stable_windows_min": 0,
            "controller_stable_windows_max": 0,
            "effective_alpha_mean": 0.0,
            "effective_alpha_min": 0.0,
            "effective_alpha_max": 0.0,
            "effective_alpha_zero_targets": int(self.config.dp_size),
            "admission_outcomes": {},
            "protection_copy_attempted_bytes": 0,
        }

    def flush(self) -> None:
        return

    def snapshot(self) -> dict[str, Any]:
        return {
            "hbm_used_bytes": 0,
            "hbm_limit_bytes": 0,
            "hbm_capacity_bytes": 0,
            "hbm_available_capacity_bytes": 0,
            "failed_hbm_target_count": 0,
            "dram_used_bytes": 0,
            "dram_limit_bytes": 0,
            "dram_capacity_bytes": 0,
            "controller_alpha_mean": 0.0,
            "controller_alpha_min": 0.0,
            "controller_alpha_max": 0.0,
            "controller_stable_windows_mean": 0.0,
            "controller_stable_windows_min": 0,
            "controller_stable_windows_max": 0,
            "effective_alpha_mean": 0.0,
            "effective_alpha_min": 0.0,
            "effective_alpha_max": 0.0,
            "effective_alpha_zero_targets": int(self.config.dp_size),
            "admission_outcomes": {},
            "protection_copy_attempted_bytes": 0,
            "target_snapshots": [],
        }


@dataclass
class _StaticRecord:
    chunk: SealedChunk
    byte_size: int
    sequence: int
    state: str = "PENDING"


@dataclass(order=True)
class _StaticJob:
    completed_at_ms: float
    sequence: int
    request_id: str = field(compare=False)
    ordinal: int = field(compare=False)


class _StaticTierPool:
    """FCFS independent-tier ablation with asynchronous copy completion."""

    def __init__(
        self,
        *,
        capacity_bytes: int,
        limit_bytes: int,
        kv_bytes_per_token: int,
        copy_path: CalibratedTransferPath,
    ) -> None:
        self.capacity_bytes = int(capacity_bytes)
        self.limit_bytes = min(max(int(limit_bytes), 0), self.capacity_bytes)
        self.kv_bytes_per_token = int(kv_bytes_per_token)
        self.copy_path = copy_path
        self.used_bytes = 0
        self.now_ms = 0.0
        self.lane_available_ms = 0.0
        self.sequence = 0
        self.records: dict[str, list[_StaticRecord]] = defaultdict(list)
        self.next_ordinal: dict[str, int] = defaultdict(int)
        self.blocked: set[str] = set()
        self.jobs: list[_StaticJob] = []
        self.admitted = 0
        self.admitted_bytes = 0
        self.rejected = 0
        self.evicted = 0

    def register(self, request_id: str, time_ms: float) -> None:
        self.advance_to(time_ms)
        self.records[str(request_id)] = []
        self.next_ordinal[str(request_id)] = 0
        self.blocked.discard(str(request_id))

    def advance_to(self, time_ms: float) -> None:
        target = float(time_ms)
        if target < self.now_ms - 1e-9:
            raise ValueError("static independent tier time regressed")
        while self.jobs and self.jobs[0].completed_at_ms <= target + 1e-9:
            job = heapq.heappop(self.jobs)
            self.now_ms = job.completed_at_ms
            records = self.records.get(job.request_id, ())
            if not 0 <= job.ordinal < len(records):
                continue
            record = records[job.ordinal]
            if record.state != "PENDING":
                continue
            if any(prior.state != "COMMITTED" for prior in records[: job.ordinal]):
                continue
            record.state = "COMMITTED"
        self.now_ms = target

    def seal(self, request_id: str, chunk: SealedChunk, time_ms: float) -> bool:
        self.advance_to(time_ms)
        request_id = str(request_id)
        expected = self.next_ordinal[request_id]
        if int(chunk.ordinal) != expected:
            raise ValueError("static independent chunk ordinal is not progressive")
        self.next_ordinal[request_id] = expected + 1
        byte_size = int(chunk.token_count) * self.kv_bytes_per_token
        records = self.records[request_id]
        if request_id in self.blocked or self.used_bytes + byte_size > self.limit_bytes:
            self.blocked.add(request_id)
            self.rejected += 1
            return False
        if int(chunk.ordinal) != len(records):
            self.blocked.add(request_id)
            self.rejected += 1
            return False
        self.sequence += 1
        records.append(
            _StaticRecord(
                chunk=chunk,
                byte_size=byte_size,
                sequence=self.sequence,
            )
        )
        self.used_bytes += byte_size
        service = self.copy_path.duration_ms(byte_size)
        self.lane_available_ms = max(self.lane_available_ms, self.now_ms) + service
        heapq.heappush(
            self.jobs,
            _StaticJob(
                completed_at_ms=self.lane_available_ms,
                sequence=self.sequence,
                request_id=request_id,
                ordinal=chunk.ordinal,
            ),
        )
        self.admitted += 1
        self.admitted_bytes += byte_size
        return True

    def _drop_tail(self, request_id: str) -> int:
        records = self.records.get(request_id, [])
        if not records:
            return 0
        record = records.pop()
        self.used_bytes -= record.byte_size
        self.blocked.add(request_id)
        self.evicted += 1
        return record.byte_size

    def set_limit(self, limit_bytes: int, time_ms: float) -> None:
        self.advance_to(time_ms)
        self.limit_bytes = min(max(int(limit_bytes), 0), self.capacity_bytes)
        while self.used_bytes > self.limit_bytes:
            candidates = [
                (records[-1].sequence, request_id)
                for request_id, records in self.records.items()
                if records
            ]
            if not candidates:
                raise RuntimeError("static tier lost contraction victims")
            _sequence, victim = max(candidates)
            self._drop_tail(victim)

    def clear_request(self, request_id: str) -> int:
        records = self.records.get(str(request_id), [])
        released = sum(record.byte_size for record in records)
        self.used_bytes -= released
        records.clear()
        self.blocked.add(str(request_id))
        return released

    def reject_sealed(self, request_id: str, chunk: SealedChunk, time_ms: float) -> None:
        """Advance a failed-tier ordinal without reserving any bytes."""

        self.advance_to(time_ms)
        request_id = str(request_id)
        expected = self.next_ordinal[request_id]
        if int(chunk.ordinal) != expected:
            raise ValueError("static rejected chunk ordinal is not progressive")
        self.next_ordinal[request_id] = expected + 1
        self.blocked.add(request_id)
        self.rejected += 1

    def finish(self, request_id: str, time_ms: float) -> None:
        self.advance_to(time_ms)
        self.clear_request(request_id)
        self.records.pop(str(request_id), None)
        self.next_ordinal.pop(str(request_id), None)
        self.blocked.discard(str(request_id))

    def committed_prefix(self, request_id: str) -> int:
        frontier = 0
        for record in self.records.get(str(request_id), ()):
            if record.state != "COMMITTED" or record.chunk.token_start != frontier:
                break
            frontier = record.chunk.token_end
        return frontier

    def snapshot(self) -> dict[str, Any]:
        if not 0 <= self.used_bytes <= self.limit_bytes <= self.capacity_bytes:
            raise AssertionError("static independent capacity invariant failed")
        return {
            "capacity_bytes": self.capacity_bytes,
            "limit_bytes": self.limit_bytes,
            "used_bytes": self.used_bytes,
            "admitted_chunks": self.admitted,
            "admitted_bytes": self.admitted_bytes,
            "rejected_chunks": self.rejected,
            "evicted_chunks": self.evicted,
            "active_requests": len(self.records),
        }


class _StaticIndependentAdapter(_CellAdapter):
    """Algorithm-2 ablation: independent HBM/DRAM FCFS admission."""

    def __init__(self, config: LargeScaleCellConfig) -> None:
        self.config = config
        full = _full_design_config(config)
        initial_hbm_limit = int(
            full.hbm_capacity_bytes * float(config.initial_hbm_alpha)
        )
        self.hbm = [
            _StaticTierPool(
                capacity_bytes=full.hbm_capacity_bytes,
                limit_bytes=initial_hbm_limit,
                kv_bytes_per_token=full.kv_bytes_per_token,
                copy_path=_full_design_config(
                    config, target_rank=rank
                ).hbm_copy,
            )
            for rank in range(int(config.dp_size))
        ]
        self.dram = [
            _StaticTierPool(
                capacity_bytes=full.dram_capacity_bytes,
                limit_bytes=full.dram_capacity_bytes,
                kv_bytes_per_token=full.kv_bytes_per_token,
                copy_path=full.dram_copy,
            )
            for _ in range(int(config.dp_size))
        ]
        controller_config = IndependentTierConfig(
            hbm_capacity_bytes=full.hbm_capacity_bytes,
            dram_capacity_bytes=full.dram_capacity_bytes,
            initial_hbm_alpha=config.initial_hbm_alpha,
            minimum_hbm_alpha=config.minimum_hbm_alpha,
            maximum_hbm_alpha=config.maximum_hbm_alpha,
            aimd_decrease_factor=config.aimd_decrease_factor,
            aimd_increase_step=config.aimd_increase_step,
            aimd_stable_windows=config.aimd_stable_windows,
            recompute=full.recompute,
            hbm_load=full.hbm_load,
            dram_load=full.dram_load,
            retain_event_log=False,
            retain_record_history=False,
        )
        from sglsim.srt.simulation.aegis_independent_tiers import (
            AegisIndependentTierController,
        )

        self.alpha_controllers = [
            AegisIndependentTierController(controller_config)
            for _ in range(int(config.dp_size))
        ]
        self._request_target: dict[str, int] = {}
        self._failed_hbm_targets: set[int] = set()
        self._minimum_hbm_allocation_bytes = (
            min(config.prefix_chunk_tokens, config.decode_chunk_tokens)
            * config.model_profile.kv_bytes_per_token
        )

    def register(self, request: _ScheduledRequest, *, slack_enabled: bool) -> None:
        del slack_enabled
        target = int(request.target_rank)
        request_id = request.request.request_id
        self._request_target[request_id] = target
        self.hbm[target].register(request_id, request.arrival_ms)
        self.dram[target].register(request_id, request.arrival_ms)

    def seal(
        self, request: _ScheduledRequest, chunk: SealedChunk, time_ms: float
    ) -> str:
        target = self._request_target[request.request.request_id]
        if target not in self._failed_hbm_targets:
            hbm_admitted = self.hbm[target].seal(
                request.request.request_id, chunk, time_ms
            )
        else:
            self.hbm[target].reject_sealed(
                request.request.request_id, chunk, time_ms
            )
            hbm_admitted = False
        dram_admitted = self.dram[target].seal(
            request.request.request_id, chunk, time_ms
        )
        if hbm_admitted and dram_admitted:
            return "HBM_AND_DRAM"
        if hbm_admitted:
            return "HBM_ONLY"
        if dram_admitted:
            return "DRAM_ONLY"
        return "BYPASS"

    def fail_worker(self, worker_rank: int, time_ms: float) -> dict[str, Any]:
        rank = int(worker_rank)
        self._failed_hbm_targets.add(rank)
        pool = self.hbm[rank]
        pool.advance_to(time_ms)
        invalidated_requests = 0
        invalidated_bytes = 0
        for request_id, target in tuple(self._request_target.items()):
            if target != rank:
                continue
            released = pool.clear_request(request_id)
            if released:
                invalidated_requests += 1
                invalidated_bytes += released
        return {
            "invalidated_hbm_requests": invalidated_requests,
            "invalidated_hbm_bytes": invalidated_bytes,
        }

    def repair_worker(self, worker_rank: int, time_ms: float) -> dict[str, Any]:
        del time_ms
        rank = int(worker_rank)
        was_failed = rank in self._failed_hbm_targets
        self._failed_hbm_targets.discard(rank)
        return {"future_hbm_admission_reenabled": was_failed}

    def finish(self, request: _ScheduledRequest, time_ms: float) -> None:
        request_id = request.request.request_id
        target = self._request_target[request_id]
        self.hbm[target].finish(request_id, time_ms)
        self.dram[target].finish(request_id, time_ms)

    def rotate_target(self, request: _ScheduledRequest, time_ms: float) -> str:
        request_id = request.request.request_id
        target = self._request_target[request_id]
        self.hbm[target].advance_to(time_ms)
        self.hbm[target].clear_request(request_id)
        replacement = (request.target_rank + 1) % int(self.config.dp_size)
        return f"target-{replacement}"

    def estimate_failure(
        self,
        request: _ScheduledRequest,
        *,
        produced_tokens: int,
        remaining_slack_ms: float,
        time_ms: float,
        activated_target: str,
        slack_enabled: bool,
    ) -> RecoveryDecision:
        request_id = request.request.request_id
        target = self._request_target[request_id]
        self.hbm[target].advance_to(time_ms)
        self.dram[target].advance_to(time_ms)
        total = max(int(produced_tokens), 0)
        hbm_prefix = min(self.hbm[target].committed_prefix(request_id), total)
        dram_prefix = min(self.dram[target].committed_prefix(request_id), total)
        full = _full_design_config(self.config)

        def value(prefix: int, timing: RecoveryTiming, action: str) -> tuple[float, str, int]:
            suffix = total - prefix
            recovery = 0.0
            if prefix:
                recovery += timing.setup_ms + prefix * timing.ms_per_token
            if suffix:
                recovery += (
                    full.recompute.setup_ms
                    + suffix * full.recompute.ms_per_token
                )
            return recovery, action if prefix else "RECOMPUTE", prefix

        candidates = [value(0, full.recompute, "RECOMPUTE")]
        if hbm_prefix:
            candidates.append(value(hbm_prefix, full.hbm_load, "PEER_HBM"))
        if dram_prefix:
            candidates.append(value(dram_prefix, full.dram_load, "DRAM"))
        recovery, action, restored = min(
            candidates, key=lambda row: (row[0], -row[2], row[1])
        )
        slack = remaining_slack_ms if slack_enabled else 0.0
        return RecoveryDecision(
            request_id=request_id,
            action=action,
            restored_tokens=restored,
            recompute_tokens=total - restored,
            recovery_ms=recovery,
            slack_ms=slack,
            visible_harm_ms=max(recovery - slack, 0.0),
            route_version=1,
            activated_target=activated_target,
        )

    def _aggregate(self) -> dict[str, Any]:
        hbm = [pool.snapshot() for pool in self.hbm]
        dram = [pool.snapshot() for pool in self.dram]
        controller_telemetry = [
            controller.telemetry_snapshot()
            for controller in self.alpha_controllers
        ]
        raw = [float(row["hbm_alpha"]) for row in controller_telemetry]
        stable_windows = [
            int(row["stable_windows"]) for row in controller_telemetry
        ]
        effective = [
            0.0
            if rank in self._failed_hbm_targets
            or row["limit_bytes"] < self._minimum_hbm_allocation_bytes
            else alpha
            for rank, (row, alpha) in enumerate(zip(hbm, raw, strict=True))
        ]
        return {
            "hbm_used_bytes": sum(row["used_bytes"] for row in hbm),
            "hbm_limit_bytes": sum(
                0 if rank in self._failed_hbm_targets else row["limit_bytes"]
                for rank, row in enumerate(hbm)
            ),
            "hbm_capacity_bytes": sum(row["capacity_bytes"] for row in hbm),
            "hbm_available_capacity_bytes": sum(
                0 if rank in self._failed_hbm_targets else row["capacity_bytes"]
                for rank, row in enumerate(hbm)
            ),
            "failed_hbm_target_count": len(self._failed_hbm_targets),
            "dram_used_bytes": sum(row["used_bytes"] for row in dram),
            "dram_limit_bytes": sum(row["limit_bytes"] for row in dram),
            "dram_capacity_bytes": sum(row["capacity_bytes"] for row in dram),
            "controller_alpha_mean": statistics.fmean(raw),
            "controller_alpha_min": min(raw),
            "controller_alpha_max": max(raw),
            "controller_stable_windows_mean": statistics.fmean(
                stable_windows
            ),
            "controller_stable_windows_min": min(stable_windows),
            "controller_stable_windows_max": max(stable_windows),
            "effective_alpha_mean": statistics.fmean(effective),
            "effective_alpha_min": min(effective),
            "effective_alpha_max": max(effective),
            "effective_alpha_zero_targets": sum(value == 0.0 for value in effective),
            "admission_outcomes": {
                "HBM_ADMITTED": sum(row["admitted_chunks"] for row in hbm),
                "HBM_REJECTED": sum(row["rejected_chunks"] for row in hbm),
                "DRAM_ADMITTED": sum(row["admitted_chunks"] for row in dram),
                "DRAM_REJECTED": sum(row["rejected_chunks"] for row in dram),
            },
            "protection_copy_attempted_bytes": sum(
                row["admitted_bytes"] for row in (*hbm, *dram)
            ),
            "target_snapshots": [
                {"target": index, "HBM": hbm[index], "DRAM": dram[index]}
                for index in range(len(hbm))
            ],
        }

    def control_window(
        self,
        *,
        time_ms: float,
        pressure_by_target: Mapping[int, bool],
        algorithm3: bool,
        apply_aimd_update: bool,
        active_requests: int,
        queued_requests: int,
    ) -> dict[str, Any]:
        pressure_count = 0
        for rank, controller in enumerate(self.alpha_controllers):
            self.hbm[rank].advance_to(time_ms)
            self.dram[rank].advance_to(time_ms)
            pressure = bool(pressure_by_target.get(rank, False))
            pressure = pressure or (
                self.hbm[rank].limit_bytes > 0
                and self.hbm[rank].used_bytes
                >= float(self.config.hbm_quota_pressure_fraction)
                * self.hbm[rank].limit_bytes
            )
            pressure_count += int(pressure)
            if algorithm3 and apply_aimd_update:
                result = controller.control_hbm_quota(pressure=pressure)
                self.hbm[rank].set_limit(result["quota_bytes"], time_ms)
        aggregate = self._aggregate()
        aggregate.update(
            {
                "time_ms": time_ms,
                "active_requests": int(active_requests),
                "queued_requests": int(queued_requests),
                "pressure_targets": pressure_count,
                "pressure_fraction": pressure_count / max(int(self.config.dp_size), 1),
                "algorithm3_enabled": bool(algorithm3),
            }
        )
        aggregate.pop("target_snapshots", None)
        return aggregate

    def flush(self) -> None:
        for pool in (*self.hbm, *self.dram):
            pool.advance_to(max(pool.now_ms, pool.lane_available_ms))
            pool.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return self._aggregate()


def _resolve_policy(
    policy: str,
    ablation: AlgorithmToggles | Mapping[str, Any] | str | None,
) -> tuple[str, AlgorithmToggles, str]:
    label = str(policy).strip().lower()
    if label == "aegiskv":
        toggles = AlgorithmToggles.from_value(ablation)
        return label, toggles, "aegiskv"
    if label.startswith("aegiskv_alg"):
        suffix = label.removeprefix("aegiskv_alg")
        bits = suffix[:3]
        if len(bits) != 3 or any(value not in "01" for value in bits):
            raise ValueError(f"invalid compositional AegisKV policy {policy!r}")
        toggles = AlgorithmToggles(*(value == "1" for value in bits))
        explicit = AlgorithmToggles.from_value(ablation)
        if ablation is not None and explicit != toggles:
            raise ValueError("policy bits and explicit ablation disagree")
        return label, toggles, "aegiskv"
    if label == "static_independent_fcfs":
        return label, AlgorithmToggles(False, False, True), "static"
    if label == "recompute_only":
        return label, AlgorithmToggles(False, False, False), "recompute"
    if label in BASELINE_POLICIES:
        return label, AlgorithmToggles(False, False, False), "baseline"
    raise ValueError(
        f"unsupported policy {policy!r}; expected AegisKV or {SUPPORTED_POLICIES}"
    )


def _make_adapter(
    config: LargeScaleCellConfig,
    *,
    policy_label: str,
    toggles: AlgorithmToggles,
    adapter_kind: str,
) -> _CellAdapter:
    if adapter_kind == "recompute":
        return _RecomputeAdapter(config)
    if adapter_kind == "baseline":
        return _BaselineAdapter(config, BaselinePolicy(policy_label))
    if adapter_kind == "static" or not toggles.algorithm2:
        return _StaticIndependentAdapter(config)
    return _AegisAdapter(config, toggles)


def _chunk_seal_time(request: _ScheduledRequest, chunk: SealedChunk) -> float:
    prompt = int(request.request.prompt_tokens)
    if chunk.kind == "prefix":
        if request.prefill_ms <= 0.0:
            return request.start_ms
        progress = chunk.token_end / max(prompt, 1)
        return min(
            request.start_ms + request.prefill_ms * progress,
            request.base_finish_ms,
        )
    decoded = max(chunk.token_end - prompt, 0)
    decode_fraction = decoded / max(int(request.request.output_tokens), 1)
    return min(
        request.start_ms + request.prefill_ms + request.decode_ms * decode_fraction,
        request.base_finish_ms,
    )


@dataclass(frozen=True)
class _FaultEvent:
    time_ms: float
    source_rank: int
    shard_rank: int
    forced_floor_event: bool = False


@dataclass(frozen=True)
class _ActiveSourceProbe:
    probe_id: str
    stratum_index: int
    time_ms: float
    request_id: str
    workflow_id: str
    source_rank: int
    target_rank: int
    shard_rank: int
    eligible_interval_start_ms: float
    eligible_interval_end_ms: float
    produced_tokens: int
    sealed_frontier_tokens: int

    def tuple_fields(self) -> dict[str, Any]:
        """Policy-independent identity used for cross-policy pairing."""

        return {
            "probe_id": self.probe_id,
            "stratum_index": self.stratum_index,
            "request_id": self.request_id,
            "workflow_id": self.workflow_id,
            "time_ms": self.time_ms,
            "source_rank": self.source_rank,
            "target_rank": self.target_rank,
            "shard_rank": self.shard_rank,
            "produced_tokens": self.produced_tokens,
            "sealed_frontier_tokens": self.sealed_frontier_tokens,
        }


def _generate_faults(
    config: LargeScaleCellConfig,
    *,
    horizon_ms: float,
    seed: int,
) -> tuple[_FaultEvent, ...]:
    rate = float(config.failure_rate_per_1000_worker_hours)
    if rate <= 0.0 or horizon_ms <= 0.0:
        faults: list[_FaultEvent] = []
    else:
        # rate / 1000 worker-hours -> rate per millisecond per physical worker.
        rate_per_ms = rate / (1000.0 * 3600.0 * 1000.0)
        rng = random.Random(int(seed) ^ 0xFA117)
        faults = []
        for source_rank in range(int(config.dp_size)):
            for shard_rank in range(config.model_profile.shard_count):
                now = rng.expovariate(rate_per_ms)
                while now < horizon_ms:
                    faults.append(
                        _FaultEvent(
                            time_ms=now,
                            source_rank=source_rank,
                            shard_rank=shard_rank,
                        )
                    )
                    now += rng.expovariate(rate_per_ms)
    minimum = int(config.minimum_faults_per_cell)
    if len(faults) < minimum and horizon_ms > 0.0:
        for index in range(minimum - len(faults)):
            faults.append(
                _FaultEvent(
                    time_ms=horizon_ms * (index + 1) / (minimum + 1),
                    source_rank=(index * 17 + int(seed)) % int(config.dp_size),
                    shard_rank=index % config.model_profile.shard_count,
                    forced_floor_event=True,
                )
            )
    return tuple(
        sorted(
            faults,
            key=lambda row: (row.time_ms, row.source_rank, row.shard_rank),
        )
    )


def _produced_tokens(
    request: _ScheduledRequest,
    *,
    time_ms: float,
    sealed_frontier: int,
) -> int:
    elapsed = max(float(time_ms) - request.start_ms, 0.0)
    prompt = int(request.request.prompt_tokens)
    output = int(request.request.output_tokens)
    if elapsed <= request.prefill_ms:
        fraction = elapsed / max(request.prefill_ms, 1e-9)
        produced = int(round(prompt * fraction))
    else:
        decode_elapsed = elapsed - request.prefill_ms
        decoded = int(
            round(output * min(decode_elapsed / max(request.decode_ms, 1e-9), 1.0))
        )
        produced = prompt + decoded
    return min(max(produced, int(sealed_frontier)), request.total_tokens)


def _active_source_probe_schedule(
    config: LargeScaleCellConfig,
    requests: Sequence[_ScheduledRequest],
    *,
    seed: int,
) -> tuple[tuple[_ActiveSourceProbe, ...], dict[str, Any]]:
    """Stratify a fixed number of points over active+sealed shard exposure.

    The sampling frame is derived exclusively from the policy-independent base
    FCFS schedule.  Each logical request contributes one interval per model
    shard, beginning when its first complete chunk seals and ending at its
    no-failure completion.  Admission, copy completion, tier residency,
    recovery action, and uptime-fault outcomes are deliberately absent from
    both eligibility and sampling.
    """

    requested = int(config.active_source_probe_count)
    # Canonical order defines cumulative exposure before stratification.
    segments: list[tuple[float, float, _ScheduledRequest]] = []
    eligible_request_ids: set[str] = set()
    logical_request_ms = 0.0
    for request in requests:
        if not request.chunks:
            continue
        eligible_start = min(
            _chunk_seal_time(request, chunk) for chunk in request.chunks
        )
        eligible_end = float(request.base_finish_ms)
        duration = max(eligible_end - eligible_start, 0.0)
        if duration <= 0.0:
            continue
        eligible_request_ids.add(request.request.request_id)
        logical_request_ms += duration
        segments.append((eligible_start, eligible_end, request))
    segments.sort(
        key=lambda row: (
            row[0],
            row[2].source_rank,
            row[2].request.request_id,
        )
    )
    shard_count = config.model_profile.shard_count
    total_exposure_ms = sum(
        (end - start) * shard_count for start, end, _request in segments
    )
    universe_hasher = hashlib.sha256()
    universe_header = {
        "schema": "active_source_eligible_exposure_universe_v1",
        "model_key": config.model_key,
        "model_name": config.model_profile.name,
        "tp_size": config.model_profile.tp_size,
        "pp_size": config.model_profile.pp_size,
        "system_dp_size": config.dp_size,
        "exposure_unit": "physical_gpu_shard_millisecond",
    }
    universe_hasher.update(
        (
            json.dumps(
                universe_header,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    for start, end, request in segments:
        for shard_rank in range(shard_count):
            universe_hasher.update(
                (
                    json.dumps(
                        {
                            "request_id": request.request.request_id,
                            "workflow_id": request.request.workflow_id,
                            "source_rank": request.source_rank,
                            "target_rank": request.target_rank,
                            "shard_rank": shard_rank,
                            "eligible_start_ms": start,
                            "eligible_end_ms": end,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
    empty_hash = _sha256_json([])
    metadata: dict[str, Any] = {
        "evidence_class": ACTIVE_SOURCE_PROBE_EVIDENCE_CLASS,
        "status": "disabled" if requested == 0 else "pending",
        "requested_probe_count": requested,
        "realized_probe_count": 0,
        "eligible_request_count": len(eligible_request_ids),
        "eligible_interval_count": len(segments) * shard_count,
        "eligible_logical_request_service_ms": logical_request_ms,
        "eligible_exposure_gpu_shard_ms": total_exposure_ms,
        "eligible_exposure_gpu_shard_hours": total_exposure_ms / 3_600_000.0,
        "sampling_frame": (
            "policy_independent_base_schedule_active_source_with_at_least_one_"
            "complete_sealed_chunk_gpu_shard_time"
        ),
        "sampling_method": (
            "equal_cumulative_exposure_strata_with_one_seeded_uniform_offset_"
            "per_stratum"
        ),
        "selection_reads_policy_state": False,
        "selection_filters_admission_or_recovery_outcome": False,
        "probe_tuple_fields": [
            "probe_id",
            "stratum_index",
            "request_id",
            "workflow_id",
            "time_ms",
            "source_rank",
            "target_rank",
            "shard_rank",
            "produced_tokens",
            "sealed_frontier_tokens",
        ],
        "probe_tuple_sha256": empty_hash,
        "eligible_exposure_universe_sha256": universe_hasher.hexdigest(),
        "eligible_exposure_universe_hash_contract": (
            "sha256_canonical_jsonl_header_then_segments_sorted_by_"
            "eligible_start_source_request_and_expanded_by_shard_rank"
        ),
        "eligible_exposure_universe_header": universe_header,
    }
    if requested == 0:
        return (), metadata
    if total_exposure_ms <= 0.0 or not segments:
        raise ValueError(
            "active_source_probe_count is positive but the base schedule has "
            "no active+sealed GPU-shard exposure"
        )

    prng_key_payload = {
        "sampler": "equal_exposure_strata_uniform_offset_v1",
        "seed": int(seed),
        "probe_count": requested,
        "eligible_exposure_universe_sha256": universe_hasher.hexdigest(),
    }
    prng_key_sha256 = _sha256_json(prng_key_payload)
    metadata.update(
        {
            "sampling_prng_key_sha256": prng_key_sha256,
            "sampling_prng_key_contract": (
                "seed+probe_count+eligible_universe; excludes_policy_failure_"
                "rate_admission_residency_and_recovery_outcome"
            ),
        }
    )
    rng = random.Random(int(prng_key_sha256[:16], 16))
    stratum_width = total_exposure_ms / requested
    probes: list[_ActiveSourceProbe] = []
    segment_index = 0
    cumulative_before = 0.0
    for index in range(requested):
        exposure_point = (index + rng.random()) * stratum_width
        while segment_index < len(segments) - 1:
            start, end, _request = segments[segment_index]
            duration = (end - start) * shard_count
            if exposure_point < cumulative_before + duration:
                break
            cumulative_before += duration
            segment_index += 1
        start, end, request = segments[segment_index]
        interval_duration = end - start
        local_exposure = min(
            max(exposure_point - cumulative_before, 0.0),
            interval_duration * shard_count,
        )
        shard_rank = min(int(local_exposure / interval_duration), shard_count - 1)
        offset = local_exposure - shard_rank * interval_duration
        offset = min(max(offset, 0.0), interval_duration)
        # random() is strictly below one, so every probe remains in the active
        # half-open interval and cannot race the base completion event.
        time_ms = min(start + offset, math.nextafter(end, start))
        sealed = max(
            (
                chunk.token_end
                for chunk in request.chunks
                if _chunk_seal_time(request, chunk) <= time_ms + 1e-12
            ),
            default=0,
        )
        if sealed <= 0:
            raise AssertionError("active-source probe lost its sealed frontier")
        produced = _produced_tokens(
            request,
            time_ms=time_ms,
            sealed_frontier=sealed,
        )
        probes.append(
            _ActiveSourceProbe(
                probe_id=f"active-source-probe-{index:04d}",
                stratum_index=index,
                time_ms=time_ms,
                request_id=request.request.request_id,
                workflow_id=request.request.workflow_id,
                source_rank=request.source_rank,
                target_rank=request.target_rank,
                shard_rank=shard_rank,
                eligible_interval_start_ms=start,
                eligible_interval_end_ms=end,
                produced_tokens=produced,
                sealed_frontier_tokens=sealed,
            )
        )
    tuple_rows = [probe.tuple_fields() for probe in probes]
    metadata.update(
        {
            "status": "ready",
            "realized_probe_count": len(probes),
            "probe_tuple_sha256": _sha256_json(tuple_rows),
            "stratum_width_gpu_shard_ms": stratum_width,
        }
    )
    return tuple(probes), metadata


def _workflow_rows(requests: Sequence[_ScheduledRequest]) -> list[dict[str, Any]]:
    grouped: dict[str, list[_ScheduledRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.request.workflow_id].append(request)
    rows = []
    for workflow_id, members in sorted(grouped.items()):
        arrival = min(row.arrival_ms for row in members)
        base_finish = max(row.base_finish_ms for row in members)
        actual_failure_harm = max(
            (row.actual_visible_harm_ms for row in members), default=0.0
        )
        policy_failure_harm = max(
            (row.policy_visible_harm_ms for row in members), default=0.0
        )
        surrogate_finish = base_finish + actual_failure_harm
        rows.append(
            {
                "workflow_id": workflow_id,
                "request_count": len(members),
                "failed_request_count": sum(row.failure_count > 0 for row in members),
                "arrival_ms": arrival,
                "base_join_completion_ms": base_finish,
                "simulated_join_completion_ms": surrogate_finish,
                "base_workflow_wct_ms": base_finish - arrival,
                "simulated_workflow_wct_ms": surrogate_finish - arrival,
                "failure_added_workflow_wct_ms": actual_failure_harm,
                "actual_failure_added_workflow_harm_surrogate_ms": (
                    actual_failure_harm
                ),
                "policy_predicted_failure_added_workflow_harm_surrogate_ms": (
                    policy_failure_harm
                ),
                "workflow_failure_metric_semantics": (
                    "max_per_workflow_actual_causal_slack_harm_surrogate; "
                    "not_end_to_end_dag_execution_replay"
                ),
            }
        )
    return rows


def _active_interval_stats(
    rows: Sequence[Mapping[str, Any]], *, finish_key: str
) -> dict[str, float | int]:
    events = sorted(
        [
            *((float(row["arrival_ms"]), 1) for row in rows),
            *((float(row[finish_key]), -1) for row in rows),
        ],
        key=lambda item: (item[0], item[1]),
    )
    if not events:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "peak": 0, "duration_ms": 0.0}
    active = 0
    peak = 0
    previous = float(events[0][0])
    weighted: list[tuple[int, float]] = []
    for time_ms, delta in events:
        duration = max(float(time_ms) - previous, 0.0)
        if duration:
            weighted.append((active, duration))
        active += int(delta)
        if active < 0:
            raise AssertionError("workflow active interval accounting underflow")
        peak = max(peak, active)
        previous = float(time_ms)
    if active != 0:
        raise AssertionError("workflow active interval accounting did not drain")
    total = sum(duration for _level, duration in weighted)

    def quantile(q: float) -> float:
        if not weighted or total <= 0.0:
            return 0.0
        threshold = total * float(q)
        cumulative = 0.0
        for level, duration in sorted(weighted, key=lambda item: item[0]):
            cumulative += duration
            if cumulative >= threshold:
                return float(level)
        return float(weighted[-1][0])

    return {
        "mean": (
            sum(level * duration for level, duration in weighted) / total
            if total > 0.0
            else 0.0
        ),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "peak": peak,
        "duration_ms": total,
    }


def _request_rows(
    requests: Sequence[_ScheduledRequest], *, toggles: AlgorithmToggles
) -> list[dict[str, Any]]:
    return [
        {
            "request_id": row.request.request_id,
            "workflow_id": row.request.workflow_id,
            "source_rank": row.source_rank,
            "target_rank": row.target_rank,
            "arrival_ms": row.arrival_ms,
            "start_ms": row.start_ms,
            "base_finish_ms": row.base_finish_ms,
            "simulated_finish_ms": row.adjusted_finish_ms,
            "queue_ms": row.queue_ms,
            "prefill_ms": row.prefill_ms,
            "decode_ms": row.decode_ms,
            "base_response_ms": row.base_finish_ms - row.arrival_ms,
            "simulated_response_ms": row.adjusted_finish_ms - row.arrival_ms,
            "request_completion_metric_semantics": (
                "base_completion_plus_fail_now_recovery_delay_surrogate; "
                "not_rescheduled_execution"
            ),
            "prompt_tokens": row.request.prompt_tokens,
            "output_tokens": row.request.output_tokens,
            "input_reuse_tokens": row.reused_tokens,
            "input_reuse_fraction": row.reused_tokens
            / max(row.request.prompt_tokens, 1),
            "prefix_group": row.request.prefix_group,
            "protection_copy_deduplicated_by_reuse": False,
            "slack_ms": row.request.slack_ms,
            "actual_slack_ms": (
                row.request.actual_slack_ms
                if row.request.actual_slack_ms is not None
                else row.request.slack_ms
            ),
            "predicted_policy_slack_ms": (
                row.request.predicted_slack_ms
                if row.request.predicted_slack_ms is not None
                else row.request.slack_ms
            ),
            "caller_supplied_policy_slack_ms": row.request.slack_ms,
            "policy_slack_multiplier_preapplied": (
                row.request.policy_slack_multiplier_applied is not None
            ),
            "policy_slack_multiplier_applied": (
                row.request.policy_slack_multiplier_applied
            ),
            "policy_slack_ms": row.policy_slack_ms,
            "effective_policy_slack_ms": (
                row.policy_slack_ms if toggles.algorithm1 else 0.0
            ),
            "algorithm1_enabled": toggles.algorithm1,
            "failure_count": row.failure_count,
            "recovery_action": row.recovery_action,
            "recovery_ms": row.recovery_ms,
            "fail_now_recovery_delay_surrogate_ms": row.recovery_ms,
            "policy_remaining_slack_at_failure_ms": (
                row.policy_remaining_slack_at_failure_ms
            ),
            "actual_remaining_slack_at_failure_ms": (
                row.actual_remaining_slack_at_failure_ms
            ),
            "policy_modeled_visible_harm_ms": row.policy_visible_harm_ms,
            "actual_modeled_visible_harm_ms": row.actual_visible_harm_ms,
            # Compatibility field is explicitly ground-truth actual-slack
            # harm, never the admission policy's predicted slack.
            "modeled_visible_harm_ms": row.actual_visible_harm_ms,
            "restored_tokens": row.restored_tokens,
            "recompute_tokens": row.recompute_tokens,
        }
        for row in requests
    ]


def _failure_history_snapshot(
    *,
    snapshot_time_ms: float,
    snapshot_scope: str,
    currently_unavailable_physical_gpu_shards: set[tuple[int, int]],
    physical_gpu_shards_ever_failed: set[tuple[int, int]],
    currently_unavailable_logical_replica_targets: set[int],
    logical_replica_targets_ever_failed: set[int],
    old_hbm_copy_invalidation_event_count: int,
    old_hbm_copy_invalidated_requests_cumulative: int,
    old_hbm_copy_invalidated_bytes_cumulative: int,
) -> dict[str, Any]:
    """Freeze failure history at one explicitly named simulation boundary."""

    return {
        "snapshot_time_ms": float(snapshot_time_ms),
        "snapshot_scope": snapshot_scope,
        "current_physical_gpu_shard_outage_count": len(
            currently_unavailable_physical_gpu_shards
        ),
        "currently_unavailable_physical_gpu_shards": [
            {"source_rank": rank, "shard_rank": shard}
            for rank, shard in sorted(
                currently_unavailable_physical_gpu_shards
            )
        ],
        "physical_gpu_shards_ever_failed_count": len(
            physical_gpu_shards_ever_failed
        ),
        "physical_gpu_shards_ever_failed": [
            {"source_rank": rank, "shard_rank": shard}
            for rank, shard in sorted(physical_gpu_shards_ever_failed)
        ],
        "current_logical_replica_target_outage_count": len(
            currently_unavailable_logical_replica_targets
        ),
        "currently_unavailable_logical_replica_targets": sorted(
            currently_unavailable_logical_replica_targets
        ),
        "logical_replica_targets_ever_failed_count": len(
            logical_replica_targets_ever_failed
        ),
        "logical_replica_targets_ever_failed": sorted(
            logical_replica_targets_ever_failed
        ),
        "old_hbm_copy_invalidation_event_count_cumulative": int(
            old_hbm_copy_invalidation_event_count
        ),
        "old_hbm_copy_invalidated_requests_cumulative": int(
            old_hbm_copy_invalidated_requests_cumulative
        ),
        "old_hbm_copy_invalidated_bytes_cumulative": int(
            old_hbm_copy_invalidated_bytes_cumulative
        ),
        "repair_clears_current_outage_not_ever_failed_history": True,
        "logical_replica_outage_is_conservative_aggregate_of_stage_shard_faults": (
            True
        ),
    }


def _summary(
    *,
    config: LargeScaleCellConfig,
    policy: str,
    toggles: AlgorithmToggles,
    requests: Sequence[_ScheduledRequest],
    workflow_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    repair_rows: Sequence[Mapping[str, Any]],
    failure_state_at_horizon: Mapping[str, Any],
    final_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    response = [row.adjusted_finish_ms - row.arrival_ms for row in requests]
    base_response = [row.base_finish_ms - row.arrival_ms for row in requests]
    workflow_wct = [float(row["simulated_workflow_wct_ms"]) for row in workflow_rows]
    failure_added_wct = [
        float(row["failure_added_workflow_wct_ms"]) for row in workflow_rows
    ]
    actual_visible_harm = [
        row.actual_visible_harm_ms for row in requests if row.failure_count
    ]
    policy_visible_harm = [
        row.policy_visible_harm_ms for row in requests if row.failure_count
    ]
    controller_alpha = [
        float(row["controller_alpha_mean"]) for row in window_rows
    ]
    effective_alpha = [float(row["effective_alpha_mean"]) for row in window_rows]
    hbm_used = [int(row["hbm_used_bytes"]) for row in window_rows]
    dram_used = [int(row["dram_used_bytes"]) for row in window_rows]
    measured_windows = [
        row for row in window_rows if bool(row.get("in_measurement_window"))
    ]
    measured_controller_alpha = [
        float(row["controller_alpha_mean"]) for row in measured_windows
    ]
    measured_effective_alpha = [
        float(row["effective_alpha_mean"]) for row in measured_windows
    ]
    measured_hbm_used = [int(row["hbm_used_bytes"]) for row in measured_windows]
    measured_dram_used = [int(row["dram_used_bytes"]) for row in measured_windows]
    prompt_tokens = sum(row.request.prompt_tokens for row in requests)
    reuse_tokens = sum(row.reused_tokens for row in requests)
    workload_reuse_bin_request_counts = Counter(
        _reuse_bin(row.reused_tokens, row.request.prompt_tokens)
        for row in requests
    )
    workload_reuse_bin_prompt_tokens: Counter[str] = Counter()
    workload_reuse_bin_input_reuse_tokens: Counter[str] = Counter()
    for row in requests:
        reuse_bin = _reuse_bin(row.reused_tokens, row.request.prompt_tokens)
        workload_reuse_bin_prompt_tokens[reuse_bin] += int(
            row.request.prompt_tokens
        )
        workload_reuse_bin_input_reuse_tokens[reuse_bin] += int(
            row.reused_tokens
        )
    actions = Counter(row.recovery_action for row in requests if row.failure_count)
    effective_faults = sum(bool(row.get("effective")) for row in failure_rows)
    base_workflow_concurrency = _active_interval_stats(
        workflow_rows, finish_key="base_join_completion_ms"
    )
    simulated_workflow_concurrency = _active_interval_stats(
        workflow_rows, finish_key="simulated_join_completion_ms"
    )
    logical_kv_bytes = sum(
        row.total_tokens * config.model_profile.kv_bytes_per_token
        for row in requests
    )
    protection_copy_attempted_bytes = int(
        final_snapshot.get("protection_copy_attempted_bytes", 0)
    )
    failure_history_at_horizon = dict(failure_state_at_horizon)
    final_failure_history = dict(final_snapshot.get("failure_history", {}))
    predicted_policy_slack = [
        max(
            float(
                row.request.predicted_slack_ms
                if row.request.predicted_slack_ms is not None
                else row.request.slack_ms
            ),
            0.0,
        )
        for row in requests
    ]
    caller_supplied_policy_slack = [
        max(float(row.request.slack_ms), 0.0) for row in requests
    ]
    resolved_policy_slack = [max(float(row.policy_slack_ms), 0.0) for row in requests]
    effective_policy_slack = (
        resolved_policy_slack if toggles.algorithm1 else [0.0] * len(requests)
    )
    predicted_policy_slack_nonzero_count = sum(
        value > 1e-9 for value in predicted_policy_slack
    )
    caller_supplied_policy_slack_nonzero_count = sum(
        value > 1e-9 for value in caller_supplied_policy_slack
    )
    resolved_policy_slack_nonzero_count = sum(
        value > 1e-9 for value in resolved_policy_slack
    )
    effective_policy_slack_nonzero_count = sum(
        value > 1e-9 for value in effective_policy_slack
    )
    algorithm1_mechanism_state = (
        "disabled_zero_effective_slack"
        if not toggles.algorithm1
        else (
            "enabled_nonzero_effective_slack"
            if effective_policy_slack_nonzero_count
            else "enabled_degenerate_zero_effective_slack"
        )
    )
    run_horizon_ms = max((row.base_finish_ms for row in requests), default=0.0)
    return {
        "policy": policy,
        "ablation": toggles.label,
        "primary_workflow_metric_semantics": (
            "base_workflow_wct_plus_max_actual_causal_slack_harm_surrogate; "
            "not_end_to_end_dag_execution_replay"
        ),
        "request_count": len(requests),
        "workflow_count": len(workflow_rows),
        "explicit_actual_slack_request_count": sum(
            row.request.actual_slack_ms is not None for row in requests
        ),
        "actual_slack_fallback_request_count": sum(
            row.request.actual_slack_ms is None for row in requests
        ),
        "algorithm1_mechanism_state": algorithm1_mechanism_state,
        "algorithm1_enabled": toggles.algorithm1,
        "algorithm1_prediction_error_multiplier": float(
            config.algorithm1_prediction_error_multiplier
        ),
        "algorithm1_prediction_error_multiplier_semantics": (
            "0=oracle_actual_slack_sensitivity; 0<m<1=partial_error; "
            "1=caller_supplied_causal_prediction; m>1=linear_error_stress"
        ),
        "predicted_policy_slack_nonzero_count": (
            predicted_policy_slack_nonzero_count
        ),
        "predicted_policy_slack_nonzero_fraction": (
            predicted_policy_slack_nonzero_count / max(len(requests), 1)
        ),
        "caller_supplied_policy_slack_nonzero_count": (
            caller_supplied_policy_slack_nonzero_count
        ),
        "caller_supplied_policy_slack_nonzero_fraction": (
            caller_supplied_policy_slack_nonzero_count
            / max(len(requests), 1)
        ),
        "requests_with_preapplied_policy_slack_multiplier": sum(
            row.request.policy_slack_multiplier_applied is not None
            for row in requests
        ),
        "resolved_policy_slack_nonzero_count": (
            resolved_policy_slack_nonzero_count
        ),
        "resolved_policy_slack_nonzero_fraction": (
            resolved_policy_slack_nonzero_count / max(len(requests), 1)
        ),
        "effective_policy_slack_nonzero_count": (
            effective_policy_slack_nonzero_count
        ),
        "effective_policy_slack_nonzero_fraction": (
            effective_policy_slack_nonzero_count / max(len(requests), 1)
        ),
        "algorithm1_nonzero_slack_input_observed": (
            predicted_policy_slack_nonzero_count > 0
        ),
        "realized_base_active_workflow_concurrency": base_workflow_concurrency,
        "realized_simulated_active_workflow_concurrency": (
            simulated_workflow_concurrency
        ),
        "simulated_request_response_mean_ms": statistics.fmean(response),
        "simulated_request_response_p50_ms": _quantile(response, 0.50),
        "simulated_request_response_p95_ms": _quantile(response, 0.95),
        "simulated_request_response_p99_ms": _quantile(response, 0.99),
        "base_request_response_mean_ms": statistics.fmean(base_response),
        "simulated_workflow_wct_mean_ms": statistics.fmean(workflow_wct),
        "simulated_workflow_wct_p50_ms": _quantile(workflow_wct, 0.50),
        "simulated_workflow_wct_p95_ms": _quantile(workflow_wct, 0.95),
        "simulated_workflow_wct_p99_ms": _quantile(workflow_wct, 0.99),
        "failure_added_workflow_wct_mean_ms": statistics.fmean(failure_added_wct),
        "failure_added_workflow_wct_p95_ms": _quantile(failure_added_wct, 0.95),
        "actual_failure_added_workflow_harm_surrogate_mean_ms": (
            statistics.fmean(failure_added_wct)
        ),
        "actual_failure_added_workflow_harm_surrogate_p50_ms": _quantile(
            failure_added_wct, 0.50
        ),
        "actual_failure_added_workflow_harm_surrogate_p95_ms": _quantile(
            failure_added_wct, 0.95
        ),
        "actual_failure_added_workflow_harm_surrogate_p99_ms": _quantile(
            failure_added_wct, 0.99
        ),
        "legacy_metric_aliases": {
            "failure_added_workflow_wct_*": (
                "actual_failure_added_workflow_harm_surrogate_*"
            ),
            "simulated_workflow_wct_*": (
                "base_workflow_wct_plus_actual_harm_surrogate_*"
            ),
        },
        "legacy_metric_aliases_forbidden_for_formal_claims": True,
        "modeled_visible_harm_mean_ms": (
            statistics.fmean(actual_visible_harm) if actual_visible_harm else 0.0
        ),
        "modeled_visible_harm_p95_ms": _quantile(actual_visible_harm, 0.95),
        "actual_modeled_visible_harm_mean_ms": (
            statistics.fmean(actual_visible_harm) if actual_visible_harm else 0.0
        ),
        "actual_modeled_visible_harm_p95_ms": _quantile(
            actual_visible_harm, 0.95
        ),
        "policy_modeled_visible_harm_mean_ms": (
            statistics.fmean(policy_visible_harm) if policy_visible_harm else 0.0
        ),
        "policy_modeled_visible_harm_p95_ms": _quantile(
            policy_visible_harm, 0.95
        ),
        "failure_events_injected": len(failure_rows),
        "worker_repair_events_scheduled": len(repair_rows),
        "worker_repair_events_effective": sum(
            bool(row.get("future_hbm_admission_reenabled"))
            for row in repair_rows
        ),
        "worker_repair_events_superseded": sum(
            bool(row.get("superseded_by_later_failure")) for row in repair_rows
        ),
        "failure_rate_exposure_unit": "total_gpu_shard_hours",
        "modeled_failure_exposure_gpu_hours": (
            config.total_gpus * run_horizon_ms / 3_600_000.0
        ),
        "failure_events_effective": effective_faults,
        "failure_events_ineffective": len(failure_rows) - effective_faults,
        "physical_gpu_shards_ever_failed_count": int(
            failure_history_at_horizon.get(
                "physical_gpu_shards_ever_failed_count", 0
            )
        ),
        "current_physical_gpu_shard_outage_count_at_horizon": int(
            failure_history_at_horizon.get(
                "current_physical_gpu_shard_outage_count", 0
            )
        ),
        "current_physical_gpu_shard_outage_count_post_all_events": int(
            final_failure_history.get(
                "current_physical_gpu_shard_outage_count", 0
            )
        ),
        "logical_replica_targets_ever_failed_count": int(
            failure_history_at_horizon.get(
                "logical_replica_targets_ever_failed_count", 0
            )
        ),
        "current_logical_replica_target_outage_count_at_horizon": int(
            failure_history_at_horizon.get(
                "current_logical_replica_target_outage_count", 0
            )
        ),
        "current_logical_replica_target_outage_count_post_all_events": int(
            final_failure_history.get(
                "current_logical_replica_target_outage_count", 0
            )
        ),
        "failure_events_on_previously_failed_physical_gpu_shard": sum(
            bool(row.get("physical_gpu_shard_ever_failed_before_event"))
            for row in failure_rows
        ),
        "failure_events_while_logical_replica_target_already_unavailable": sum(
            bool(
                row.get(
                    "logical_replica_target_currently_unavailable_before_event"
                )
            )
            for row in failure_rows
        ),
        "effective_failures_with_bound_peer_hbm_target_prior_unavailable": sum(
            bool(row.get("effective"))
            and bool(
                row.get(
                    "bound_peer_hbm_target_currently_unavailable_before_source_event"
                )
            )
            for row in failure_rows
        ),
        "old_hbm_copy_invalidation_event_count_cumulative": int(
            failure_history_at_horizon.get(
                "old_hbm_copy_invalidation_event_count_cumulative", 0
            )
        ),
        "old_hbm_copy_invalidated_requests_cumulative": int(
            failure_history_at_horizon.get(
                "old_hbm_copy_invalidated_requests_cumulative", 0
            )
        ),
        "old_hbm_copy_invalidated_bytes_cumulative": int(
            failure_history_at_horizon.get(
                "old_hbm_copy_invalidated_bytes_cumulative", 0
            )
        ),
        "forced_failure_floor_events": sum(
            bool(row.get("forced_floor_event")) for row in failure_rows
        ),
        "recovery_action_counts": dict(sorted(actions.items())),
        "restored_tokens": sum(row.restored_tokens for row in requests),
        "recomputed_tokens": sum(row.recompute_tokens for row in requests),
        "input_prompt_tokens": prompt_tokens,
        "input_reuse_tokens": reuse_tokens,
        "input_reuse_fraction": reuse_tokens / max(prompt_tokens, 1),
        "requests_with_input_reuse": sum(row.reused_tokens > 0 for row in requests),
        "workload_reuse_bin_contract": dict(REUSE_BIN_CONTRACT),
        "workload_reuse_bin_request_counts": {
            name: int(workload_reuse_bin_request_counts[name])
            for name in REUSE_BIN_CONTRACT
        },
        "workload_reuse_bin_prompt_tokens": {
            name: int(workload_reuse_bin_prompt_tokens[name])
            for name in REUSE_BIN_CONTRACT
        },
        "workload_reuse_bin_input_reuse_tokens": {
            name: int(workload_reuse_bin_input_reuse_tokens[name])
            for name in REUSE_BIN_CONTRACT
        },
        "workload_reuse_bins_use_all_requests_not_probe_sample": True,
        "protection_copy_deduplicated_by_input_reuse": False,
        "protection_copy_attempted_bytes": protection_copy_attempted_bytes,
        "logical_request_kv_bytes": logical_kv_bytes,
        "copy_amplification": (
            protection_copy_attempted_bytes / max(logical_kv_bytes, 1)
        ),
        "controller_alpha_mean": (
            statistics.fmean(controller_alpha)
            if controller_alpha
            else float(final_snapshot["controller_alpha_mean"])
        ),
        "controller_alpha_min": (
            min(controller_alpha)
            if controller_alpha
            else float(final_snapshot["controller_alpha_min"])
        ),
        "controller_alpha_max": (
            max(controller_alpha)
            if controller_alpha
            else float(final_snapshot["controller_alpha_max"])
        ),
        "initial_snapshot_sample_count": sum(
            bool(row.get("initial_sample")) for row in window_rows
        ),
        "aimd_periodic_updates_applied": sum(
            bool(row.get("aimd_update_applied")) for row in window_rows
        ),
        "first_aimd_periodic_update_ms": next(
            (
                float(row["time_ms"])
                for row in window_rows
                if bool(row.get("aimd_update_applied"))
            ),
            None,
        ),
        "effective_alpha_mean": (
            statistics.fmean(effective_alpha)
            if effective_alpha
            else float(final_snapshot["effective_alpha_mean"])
        ),
        "effective_alpha_min": (
            min(effective_alpha)
            if effective_alpha
            else float(final_snapshot["effective_alpha_min"])
        ),
        "effective_alpha_max": (
            max(effective_alpha)
            if effective_alpha
            else float(final_snapshot["effective_alpha_max"])
        ),
        "effective_alpha_zero_window_fraction": (
            sum(value == 0.0 for value in effective_alpha)
            / max(len(effective_alpha), 1)
        ),
        "hbm_used_bytes_time_mean": statistics.fmean(hbm_used) if hbm_used else 0.0,
        "hbm_used_bytes_peak": max(hbm_used, default=0),
        "dram_used_bytes_time_mean": statistics.fmean(dram_used) if dram_used else 0.0,
        "dram_used_bytes_peak": max(dram_used, default=0),
        "measurement_window_sample_count": len(measured_windows),
        "measurement_controller_alpha_mean": (
            statistics.fmean(measured_controller_alpha)
            if measured_controller_alpha
            else 0.0
        ),
        "measurement_controller_alpha_min": min(
            measured_controller_alpha, default=0.0
        ),
        "measurement_controller_alpha_max": max(
            measured_controller_alpha, default=0.0
        ),
        "measurement_effective_alpha_mean": (
            statistics.fmean(measured_effective_alpha)
            if measured_effective_alpha
            else 0.0
        ),
        "measurement_effective_alpha_min": min(
            measured_effective_alpha, default=0.0
        ),
        "measurement_effective_alpha_max": max(
            measured_effective_alpha, default=0.0
        ),
        "measurement_pressure_fraction_mean": (
            statistics.fmean(
                float(row["pressure_fraction"]) for row in measured_windows
            )
            if measured_windows
            else 0.0
        ),
        "measurement_hbm_used_bytes_time_mean": (
            statistics.fmean(measured_hbm_used) if measured_hbm_used else 0.0
        ),
        "measurement_hbm_used_bytes_peak": max(measured_hbm_used, default=0),
        "measurement_dram_used_bytes_time_mean": (
            statistics.fmean(measured_dram_used) if measured_dram_used else 0.0
        ),
        "measurement_dram_used_bytes_peak": max(measured_dram_used, default=0),
        "hbm_capacity_bytes": int(final_snapshot["hbm_capacity_bytes"]),
        "dram_capacity_bytes": int(final_snapshot["dram_capacity_bytes"]),
        "admission_outcomes": dict(final_snapshot.get("admission_outcomes", {})),
    }


def run_cell(
    config: LargeScaleCellConfig | Mapping[str, Any],
    seed: int,
    policy: str = "aegiskv",
    ablation: AlgorithmToggles | Mapping[str, Any] | str | None = None,
    workload: Sequence[RequestInput | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one deterministic policy/configuration/seed cell.

    The return value is JSON serializable and contains aggregate ``summary``
    plus request-, workflow-, control-window-, and failure-level rows.
    """

    resolved = LargeScaleCellConfig.from_value(config)
    policy_label, toggles, adapter_kind = _resolve_policy(policy, ablation)
    inputs = _normalize_workload(resolved, int(seed), workload)
    scheduled = _schedule_requests(resolved, inputs)
    active_source_probes, active_source_probe_metadata = (
        _active_source_probe_schedule(resolved, scheduled, seed=int(seed))
    )
    adapter = _make_adapter(
        resolved,
        policy_label=policy_label,
        toggles=toggles,
        adapter_kind=adapter_kind,
    )
    horizon = max(row.base_finish_ms for row in scheduled)
    measurement_start = float(resolved.measurement_start_ms)
    measurement_end = (
        horizon
        if resolved.measurement_end_ms is None
        else float(resolved.measurement_end_ms)
    )
    faults = _generate_faults(resolved, horizon_ms=horizon, seed=int(seed))

    # Stable tie order: repair, registration, execution start, chunk seal,
    # fault/shadow query, completion, then the control-window sample.  Shadow
    # probes are added after the uptime faults so a mathematically exact tie
    # retains the pre-existing primary-event order.
    events: list[tuple[float, int, int, str, Any]] = []
    sequence = 0

    def push(time_ms: float, priority: int, kind: str, payload: Any) -> None:
        nonlocal sequence
        sequence += 1
        heapq.heappush(events, (float(time_ms), priority, sequence, kind, payload))

    for request in scheduled:
        push(request.arrival_ms, 0, "arrival", request)
        push(request.start_ms, 1, "start", request)
        for chunk in request.chunks:
            push(_chunk_seal_time(request, chunk), 2, "seal", (request, chunk))
        push(request.base_finish_ms, 4, "finish", request)
    for fault in faults:
        push(fault.time_ms, 3, "failure", fault)
    for probe in active_source_probes:
        push(probe.time_ms, 3, "active_source_probe", probe)
    push(0.0, 5, "window", None)
    window = float(resolved.control_window_ms)
    sample_time = window
    while sample_time < horizon + 1e-9:
        push(sample_time, 5, "window", None)
        sample_time += window
    if not math.isclose((sample_time - window), horizon, abs_tol=1e-9):
        push(horizon, 5, "window", None)

    scheduled_by_id = {row.request.request_id: row for row in scheduled}
    outstanding_by_rank = [0] * int(resolved.dp_size)
    running_by_rank: dict[int, str] = {}
    recovery_pressure_until = [0.0] * int(resolved.dp_size)
    worker_failure_generation = [0] * int(resolved.dp_size)
    physical_gpu_shard_failure_generation: dict[tuple[int, int], int] = (
        defaultdict(int)
    )
    currently_unavailable_physical_gpu_shards: set[tuple[int, int]] = set()
    physical_gpu_shards_ever_failed: set[tuple[int, int]] = set()
    # A physical stage-shard fault conservatively invalidates and blocks the
    # entire logical replica's Peer-HBM target until the latest rank-level
    # repair.  Keep this aggregate state distinct from physical hazard history.
    currently_unavailable_logical_replica_targets: set[int] = set()
    logical_replica_targets_ever_failed: set[int] = set()
    old_hbm_copy_invalidation_event_count = 0
    old_hbm_copy_invalidated_requests_cumulative = 0
    old_hbm_copy_invalidated_bytes_cumulative = 0
    sealed_frontier: dict[str, int] = defaultdict(int)
    failure_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    active_source_probe_rows: list[dict[str, Any]] = []
    reuse_bin_intended_residency_counts = {
        name: Counter() for name in REUSE_BIN_CONTRACT
    }
    failure_state_at_horizon: dict[str, Any] | None = None
    last_processed_event_time_ms = 0.0
    post_horizon_event_count = 0

    while events:
        time_ms, _priority, _sequence, kind, payload = heapq.heappop(events)
        if (
            failure_state_at_horizon is None
            and time_ms > horizon + 1e-9
        ):
            # Repairs and recovery-extended completion events are intentionally
            # drained after the workload's base horizon.  Freeze the hazard
            # state before processing any such cleanup event so fields named
            # ``*_at_horizon`` cannot accidentally report the post-repair
            # state.
            failure_state_at_horizon = _failure_history_snapshot(
                snapshot_time_ms=horizon,
                snapshot_scope=(
                    "base_run_horizon_after_all_events_at_or_before_horizon"
                ),
                currently_unavailable_physical_gpu_shards=(
                    currently_unavailable_physical_gpu_shards
                ),
                physical_gpu_shards_ever_failed=(
                    physical_gpu_shards_ever_failed
                ),
                currently_unavailable_logical_replica_targets=(
                    currently_unavailable_logical_replica_targets
                ),
                logical_replica_targets_ever_failed=(
                    logical_replica_targets_ever_failed
                ),
                old_hbm_copy_invalidation_event_count=(
                    old_hbm_copy_invalidation_event_count
                ),
                old_hbm_copy_invalidated_requests_cumulative=(
                    old_hbm_copy_invalidated_requests_cumulative
                ),
                old_hbm_copy_invalidated_bytes_cumulative=(
                    old_hbm_copy_invalidated_bytes_cumulative
                ),
            )
        if time_ms > horizon + 1e-9:
            post_horizon_event_count += 1
        last_processed_event_time_ms = max(
            last_processed_event_time_ms, float(time_ms)
        )
        if kind == "arrival":
            request = payload
            adapter.register(request, slack_enabled=toggles.algorithm1)
            outstanding_by_rank[request.source_rank] += 1
        elif kind == "start":
            request = payload
            running_by_rank[request.source_rank] = request.request.request_id
        elif kind == "seal":
            request, chunk = payload
            intended_residency = adapter.seal(request, chunk, time_ms)
            reuse_bin_intended_residency_counts[
                _reuse_bin(
                    request.reused_tokens,
                    request.request.prompt_tokens,
                )
            ][intended_residency] += 1
            sealed_frontier[request.request.request_id] = max(
                sealed_frontier[request.request.request_id], chunk.token_end
            )
        elif kind == "failure":
            fault: _FaultEvent = payload
            request_id = running_by_rank.get(fault.source_rank)
            physical_shard = (fault.source_rank, fault.shard_rank)
            physical_unavailable_before = (
                physical_shard in currently_unavailable_physical_gpu_shards
            )
            physical_ever_failed_before = (
                physical_shard in physical_gpu_shards_ever_failed
            )
            logical_target_unavailable_before = (
                fault.source_rank
                in currently_unavailable_logical_replica_targets
            )
            logical_target_ever_failed_before = (
                fault.source_rank in logical_replica_targets_ever_failed
            )
            bound_target_rank = (
                scheduled_by_id[request_id].target_rank
                if request_id is not None
                else None
            )
            bound_target_unavailable_before = (
                bound_target_rank
                in currently_unavailable_logical_replica_targets
                if bound_target_rank is not None
                else False
            )
            bound_target_ever_failed_before = (
                bound_target_rank in logical_replica_targets_ever_failed
                if bound_target_rank is not None
                else False
            )
            currently_unavailable_physical_gpu_shards.add(physical_shard)
            physical_gpu_shards_ever_failed.add(physical_shard)
            currently_unavailable_logical_replica_targets.add(
                fault.source_rank
            )
            logical_replica_targets_ever_failed.add(fault.source_rank)
            worker_failure_generation[fault.source_rank] += 1
            failure_generation = worker_failure_generation[fault.source_rank]
            physical_gpu_shard_failure_generation[physical_shard] += 1
            physical_failure_generation = (
                physical_gpu_shard_failure_generation[physical_shard]
            )
            repair_at_ms = time_ms + float(resolved.repair_time_ms)
            push(
                repair_at_ms,
                -1,
                "repair",
                (
                    fault.source_rank,
                    fault.shard_rank,
                    failure_generation,
                    physical_failure_generation,
                ),
            )
            recovery_pressure_until[fault.source_rank] = max(
                recovery_pressure_until[fault.source_rank], repair_at_ms
            )
            worker_failure = adapter.fail_worker(fault.source_rank, time_ms)
            invalidated_requests_this_event = int(
                worker_failure.get("invalidated_hbm_requests", 0)
            )
            invalidated_bytes_this_event = int(
                worker_failure.get("invalidated_hbm_bytes", 0)
            )
            if invalidated_requests_this_event or invalidated_bytes_this_event:
                old_hbm_copy_invalidation_event_count += 1
            old_hbm_copy_invalidated_requests_cumulative += (
                invalidated_requests_this_event
            )
            old_hbm_copy_invalidated_bytes_cumulative += (
                invalidated_bytes_this_event
            )
            base = {
                "time_ms": time_ms,
                "source_rank": fault.source_rank,
                "shard_rank": fault.shard_rank,
                "failure_rate_per_1000_worker_hours": (
                    resolved.failure_rate_per_1000_worker_hours
                ),
                "forced_floor_event": fault.forced_floor_event,
                "effective": False,
                "request_id": request_id or "",
                "reason": "idle_worker" if request_id is None else "",
                "in_measurement_window": (
                    measurement_start <= time_ms <= measurement_end
                ),
                "fault_semantics": (
                    "single_worker_shard_failure_counterfactual_with_aggregate_"
                    "peer_hbm_invalidation"
                ),
                "failure_rate_exposure_unit": "total_gpu_shard_hours",
                # Compatibility row: this event did not fabricate a target
                # failure immediately before the source fault.  Natural prior
                # target outage/failure history is recorded separately below.
                "compound_target_first_failure_injected": False,
                "forced_synthetic_target_first_injection": False,
                "physical_gpu_shard_currently_unavailable_before_event": (
                    physical_unavailable_before
                ),
                "physical_gpu_shard_ever_failed_before_event": (
                    physical_ever_failed_before
                ),
                "physical_gpu_shard_currently_unavailable_after_event": True,
                "physical_gpu_shard_ever_failed_after_event": True,
                "logical_replica_target_currently_unavailable_before_event": (
                    logical_target_unavailable_before
                ),
                "logical_replica_target_ever_failed_before_event": (
                    logical_target_ever_failed_before
                ),
                "logical_replica_target_currently_unavailable_after_event": True,
                "logical_replica_target_ever_failed_after_event": True,
                "bound_peer_hbm_target_rank": bound_target_rank,
                "bound_peer_hbm_target_currently_unavailable_before_source_event": (
                    bound_target_unavailable_before
                ),
                "bound_peer_hbm_target_ever_failed_before_source_event": (
                    bound_target_ever_failed_before
                ),
                "current_physical_gpu_shard_outage_count_after_event": len(
                    currently_unavailable_physical_gpu_shards
                ),
                "physical_gpu_shards_ever_failed_count_after_event": len(
                    physical_gpu_shards_ever_failed
                ),
                "current_logical_replica_target_outage_count_after_event": len(
                    currently_unavailable_logical_replica_targets
                ),
                "logical_replica_targets_ever_failed_count_after_event": len(
                    logical_replica_targets_ever_failed
                ),
                "old_hbm_copy_invalidation_event_count_cumulative": (
                    old_hbm_copy_invalidation_event_count
                ),
                "old_hbm_copy_invalidated_requests_cumulative": (
                    old_hbm_copy_invalidated_requests_cumulative
                ),
                "old_hbm_copy_invalidated_bytes_cumulative": (
                    old_hbm_copy_invalidated_bytes_cumulative
                ),
                "worker_failure_generation": failure_generation,
                "physical_gpu_shard_failure_generation": (
                    physical_failure_generation
                ),
                "repair_scheduled_at_ms": repair_at_ms,
                **worker_failure,
            }
            if request_id is not None:
                request = scheduled_by_id[request_id]
                if request.failure_count:
                    base["reason"] = "request_already_failed"
                else:
                    produced = _produced_tokens(
                        request,
                        time_ms=time_ms,
                        sealed_frontier=sealed_frontier[request_id],
                    )
                    elapsed = max(time_ms - request.arrival_ms, 0.0)
                    predicted_remaining_slack = max(
                        request.policy_slack_ms
                        - elapsed,
                        0.0,
                    )
                    policy_remaining_slack = (
                        predicted_remaining_slack if toggles.algorithm1 else 0.0
                    )
                    actual_slack_budget = (
                        float(request.request.actual_slack_ms)
                        if request.request.actual_slack_ms is not None
                        else float(request.request.slack_ms)
                    )
                    actual_remaining_slack = max(
                        actual_slack_budget - elapsed, 0.0
                    )
                    activated_target = f"target-{request.target_rank}"
                    decision = adapter.estimate_failure(
                        request,
                        produced_tokens=produced,
                        remaining_slack_ms=policy_remaining_slack,
                        time_ms=time_ms,
                        activated_target=activated_target,
                        slack_enabled=toggles.algorithm1,
                    )
                    actual_visible_harm = max(
                        decision.recovery_ms - actual_remaining_slack, 0.0
                    )
                    request.failure_count += 1
                    request.recovery_ms += decision.recovery_ms
                    request.policy_visible_harm_ms += decision.visible_harm_ms
                    request.actual_visible_harm_ms += actual_visible_harm
                    request.policy_remaining_slack_at_failure_ms = (
                        policy_remaining_slack
                    )
                    request.actual_remaining_slack_at_failure_ms = (
                        actual_remaining_slack
                    )
                    request.restored_tokens += decision.restored_tokens
                    request.recompute_tokens += decision.recompute_tokens
                    request.recovery_action = decision.action
                    base.update(
                        {
                            "effective": True,
                            "reason": "active_request_failed",
                            "produced_tokens": produced,
                            "remaining_slack_ms": actual_remaining_slack,
                            "policy_remaining_slack_ms": policy_remaining_slack,
                            "actual_remaining_slack_ms": actual_remaining_slack,
                            "actual_slack_source": (
                                "policy_independent_actual_duration_boundary_"
                                "cpm_slack_with_elapsed_due_time_decrement"
                                if request.request.actual_slack_ms is not None
                                else "input_slack_exact_fallback"
                            ),
                            "recovery_action": decision.action,
                            "restored_tokens": decision.restored_tokens,
                            "recompute_tokens": decision.recompute_tokens,
                            "recovery_ms": decision.recovery_ms,
                            "fail_now_recovery_delay_surrogate_ms": (
                                decision.recovery_ms
                            ),
                            "policy_modeled_visible_harm_ms": (
                                decision.visible_harm_ms
                            ),
                            "actual_modeled_visible_harm_ms": actual_visible_harm,
                            "modeled_visible_harm_ms": actual_visible_harm,
                        }
                    )
            failure_rows.append(base)
        elif kind == "active_source_probe":
            probe: _ActiveSourceProbe = payload
            request = scheduled_by_id[probe.request_id]
            elapsed = max(time_ms - request.arrival_ms, 0.0)
            policy_remaining_slack = (
                max(request.policy_slack_ms - elapsed, 0.0)
                if toggles.algorithm1
                else 0.0
            )
            actual_slack_budget = (
                float(request.request.actual_slack_ms)
                if request.request.actual_slack_ms is not None
                else float(request.request.slack_ms)
            )
            actual_remaining_slack = max(actual_slack_budget - elapsed, 0.0)
            decision = adapter.estimate_recovery_at(
                request,
                produced_tokens=probe.produced_tokens,
                remaining_slack_ms=policy_remaining_slack,
                time_ms=time_ms,
                activated_target=f"target-{request.target_rank}",
                slack_enabled=toggles.algorithm1,
            )
            actual_harm = max(
                decision.recovery_ms - actual_remaining_slack, 0.0
            )
            action = str(decision.action)
            if action.startswith("PEER_HBM"):
                source_class = "PEER_HBM"
            elif action.startswith("DRAM") or action.startswith("HOST_DRAM"):
                source_class = "DRAM"
            elif action.startswith("RECOMPUTE"):
                source_class = "RECOMPUTE"
            else:
                source_class = "MIXED_OR_BASELINE_SPECIFIC"
            preceding_uptime_faults = sum(
                fault.time_ms <= time_ms + 1e-12 for fault in faults
            )
            active_source_probe_rows.append(
                {
                    **probe.tuple_fields(),
                    "stratum_index": probe.stratum_index,
                    "workflow_id": probe.workflow_id,
                    "target_rank": probe.target_rank,
                    "eligible_active_source": True,
                    "eligible_complete_sealed_frontier": True,
                    "eligible_interval_start_ms": (
                        probe.eligible_interval_start_ms
                    ),
                    "eligible_interval_end_ms": probe.eligible_interval_end_ms,
                    "eligible_exposure_universe_sha256": (
                        active_source_probe_metadata[
                            "eligible_exposure_universe_sha256"
                        ]
                    ),
                    "eligibility_uses_policy_state": False,
                    "eligibility_filters_admission_or_recovery_outcome": False,
                    "eligibility_universe": (
                        "policy_independent_base_active_service_with_complete_"
                        "sealed_frontier_gpu_shard_time"
                    ),
                    "prompt_tokens": request.request.prompt_tokens,
                    "input_reuse_tokens": request.reused_tokens,
                    "input_reuse_fraction": (
                        min(
                            max(
                                request.reused_tokens
                                / request.request.prompt_tokens,
                                0.0,
                            ),
                            1.0,
                        )
                    ),
                    "input_reuse_bin": _reuse_bin(
                        request.reused_tokens,
                        request.request.prompt_tokens,
                    ),
                    "input_reuse_enters_base_service_schedule": True,
                    "reuse_outcome_does_not_filter_probe_selection": True,
                    "fault_conditioning_semantics": (
                        "fail_now_active_source_recovery_counterfactual; "
                        "non_mutating_shadow_query"
                    ),
                    "policy_remaining_slack_ms": policy_remaining_slack,
                    "actual_remaining_slack_ms": actual_remaining_slack,
                    "actual_slack_source": (
                        "policy_independent_actual_duration_boundary_cpm_slack_"
                        "with_elapsed_due_time_decrement"
                        if request.request.actual_slack_ms is not None
                        else "input_slack_exact_fallback"
                    ),
                    "recovery_action": action,
                    "recovery_source_class": source_class,
                    "recovery_ms": decision.recovery_ms,
                    "policy_modeled_visible_harm_ms": decision.visible_harm_ms,
                    "actual_modeled_visible_harm_ms": actual_harm,
                    "actual_visible_harm_ms": actual_harm,
                    "restored_tokens": decision.restored_tokens,
                    "peer_hbm_restored_tokens": (
                        decision.restored_tokens
                        if source_class == "PEER_HBM"
                        else 0
                    ),
                    "dram_restored_tokens": (
                        decision.restored_tokens
                        if source_class == "DRAM"
                        else 0
                    ),
                    "mixed_or_baseline_specific_restored_tokens": (
                        decision.restored_tokens
                        if source_class == "MIXED_OR_BASELINE_SPECIFIC"
                        else 0
                    ),
                    "recompute_tokens": decision.recompute_tokens,
                    "uptime_poisson_faults_at_or_before_probe": (
                        preceding_uptime_faults
                    ),
                    "probe_state_includes_uptime_poisson_history": bool(
                        preceding_uptime_faults
                    ),
                    "primary_counters_updated": False,
                }
            )
        elif kind == "repair":
            (
                worker_rank,
                shard_rank,
                generation,
                physical_generation,
            ) = payload
            physical_shard = (int(worker_rank), int(shard_rank))
            current_generation = worker_failure_generation[int(worker_rank)]
            superseded = int(generation) != current_generation
            current_physical_generation = (
                physical_gpu_shard_failure_generation[physical_shard]
            )
            physical_repair_superseded = (
                int(physical_generation) != current_physical_generation
            )
            physical_unavailable_before_repair = (
                physical_shard in currently_unavailable_physical_gpu_shards
            )
            logical_unavailable_before_repair = (
                int(worker_rank)
                in currently_unavailable_logical_replica_targets
            )
            repair_result = (
                {"future_hbm_admission_reenabled": False}
                if superseded
                else adapter.repair_worker(int(worker_rank), time_ms)
            )
            if not physical_repair_superseded:
                currently_unavailable_physical_gpu_shards.discard(
                    physical_shard
                )
            if not superseded:
                currently_unavailable_logical_replica_targets.discard(
                    int(worker_rank)
                )
            repair_rows.append(
                {
                    "time_ms": time_ms,
                    "after_base_run_horizon": time_ms > horizon + 1e-9,
                    "worker_rank": int(worker_rank),
                    "worker_failure_generation": int(generation),
                    "shard_rank": int(shard_rank),
                    "physical_gpu_shard_failure_generation": int(
                        physical_generation
                    ),
                    "superseded_by_later_failure": superseded,
                    "physical_repair_superseded_by_later_same_shard_failure": (
                        physical_repair_superseded
                    ),
                    "old_hbm_copies_restored": False,
                    "post_repair_hbm_reprotection_scheduled": False,
                    "physical_gpu_shard_currently_unavailable_before_repair": (
                        physical_unavailable_before_repair
                    ),
                    "physical_gpu_shard_currently_unavailable_after_repair": (
                        physical_shard
                        in currently_unavailable_physical_gpu_shards
                    ),
                    "physical_gpu_shard_ever_failed": (
                        physical_shard in physical_gpu_shards_ever_failed
                    ),
                    "logical_replica_target_currently_unavailable_before_repair": (
                        logical_unavailable_before_repair
                    ),
                    "logical_replica_target_currently_unavailable_after_repair": (
                        int(worker_rank)
                        in currently_unavailable_logical_replica_targets
                    ),
                    "logical_replica_target_ever_failed": (
                        int(worker_rank) in logical_replica_targets_ever_failed
                    ),
                    "current_physical_gpu_shard_outage_count_after_repair": len(
                        currently_unavailable_physical_gpu_shards
                    ),
                    "physical_gpu_shards_ever_failed_count_after_repair": len(
                        physical_gpu_shards_ever_failed
                    ),
                    "current_logical_replica_target_outage_count_after_repair": len(
                        currently_unavailable_logical_replica_targets
                    ),
                    "logical_replica_targets_ever_failed_count_after_repair": len(
                        logical_replica_targets_ever_failed
                    ),
                    **repair_result,
                }
            )
        elif kind == "finish":
            request = payload
            if time_ms < request.adjusted_finish_ms - 1e-9:
                # Recovery extends this request's retention lifetime.  The
                # fast FCFS surrogate intentionally does not push later rank
                # service, but it must not reclaim protected state at the old
                # no-failure completion timestamp.
                push(request.adjusted_finish_ms, 4, "finish", request)
                continue
            adapter.finish(request, time_ms)
            if running_by_rank.get(request.source_rank) == request.request.request_id:
                running_by_rank.pop(request.source_rank, None)
            outstanding_by_rank[request.source_rank] -= 1
            if outstanding_by_rank[request.source_rank] < 0:
                raise AssertionError("rank outstanding request count underflow")
        elif kind == "window":
            initial_sample = math.isclose(
                float(time_ms), 0.0, rel_tol=0.0, abs_tol=1e-12
            )
            aimd_update_applied = bool(
                toggles.algorithm3 and not initial_sample
            )
            pressure_by_target = {}
            for target in range(int(resolved.dp_size)):
                source = _peer_source_rank(resolved, target)
                pressure_by_target[target] = (
                    outstanding_by_rank[source] >= int(resolved.pressure_queue_depth)
                    or time_ms < recovery_pressure_until[source]
                )
            active = len(running_by_rank)
            outstanding = sum(outstanding_by_rank)
            row = adapter.control_window(
                time_ms=time_ms,
                pressure_by_target=pressure_by_target,
                algorithm3=toggles.algorithm3,
                apply_aimd_update=aimd_update_applied,
                active_requests=active,
                queued_requests=max(outstanding - active, 0),
            )
            row.update(
                {
                    "window_index": len(window_rows),
                    "failure_rate_per_1000_worker_hours_observed_metadata": (
                        resolved.failure_rate_per_1000_worker_hours
                    ),
                    "failure_rate_is_direct_alpha_input": False,
                    "alpha_driver_contract": ALPHA_DRIVER_CONTRACT,
                    "initial_sample": initial_sample,
                    "aimd_update_applied": aimd_update_applied,
                    "current_physical_gpu_shard_outage_count": len(
                        currently_unavailable_physical_gpu_shards
                    ),
                    "physical_gpu_shards_ever_failed_count": len(
                        physical_gpu_shards_ever_failed
                    ),
                    "current_logical_replica_target_outage_count": len(
                        currently_unavailable_logical_replica_targets
                    ),
                    "logical_replica_targets_ever_failed_count": len(
                        logical_replica_targets_ever_failed
                    ),
                    "old_hbm_copy_invalidation_event_count_cumulative": (
                        old_hbm_copy_invalidation_event_count
                    ),
                    "old_hbm_copy_invalidated_requests_cumulative": (
                        old_hbm_copy_invalidated_requests_cumulative
                    ),
                    "old_hbm_copy_invalidated_bytes_cumulative": (
                        old_hbm_copy_invalidated_bytes_cumulative
                    ),
                    "in_measurement_window": (
                        measurement_start <= time_ms <= measurement_end
                    ),
                }
            )
            window_rows.append(row)
        else:
            raise AssertionError(f"unknown cell event {kind!r}")

    if failure_state_at_horizon is None:
        failure_state_at_horizon = _failure_history_snapshot(
            snapshot_time_ms=horizon,
            snapshot_scope=(
                "base_run_horizon_after_all_events_at_or_before_horizon"
            ),
            currently_unavailable_physical_gpu_shards=(
                currently_unavailable_physical_gpu_shards
            ),
            physical_gpu_shards_ever_failed=physical_gpu_shards_ever_failed,
            currently_unavailable_logical_replica_targets=(
                currently_unavailable_logical_replica_targets
            ),
            logical_replica_targets_ever_failed=(
                logical_replica_targets_ever_failed
            ),
            old_hbm_copy_invalidation_event_count=(
                old_hbm_copy_invalidation_event_count
            ),
            old_hbm_copy_invalidated_requests_cumulative=(
                old_hbm_copy_invalidated_requests_cumulative
            ),
            old_hbm_copy_invalidated_bytes_cumulative=(
                old_hbm_copy_invalidated_bytes_cumulative
            ),
        )

    adapter.flush()
    # Event execution is chronological, whereas the paired statistical design
    # is indexed by exposure stratum.  Canonicalize raw evidence by stratum so
    # directly hashing ``probe_tuple_fields`` reproduces the schedule hash.
    active_source_probe_rows.sort(key=lambda row: int(row["stratum_index"]))
    final_snapshot = adapter.snapshot()
    final_snapshot["failure_history"] = _failure_history_snapshot(
        snapshot_time_ms=last_processed_event_time_ms,
        snapshot_scope=(
            "post_all_scheduled_events_including_repairs_after_horizon"
        ),
        currently_unavailable_physical_gpu_shards=(
            currently_unavailable_physical_gpu_shards
        ),
        physical_gpu_shards_ever_failed=physical_gpu_shards_ever_failed,
        currently_unavailable_logical_replica_targets=(
            currently_unavailable_logical_replica_targets
        ),
        logical_replica_targets_ever_failed=logical_replica_targets_ever_failed,
        old_hbm_copy_invalidation_event_count=(
            old_hbm_copy_invalidation_event_count
        ),
        old_hbm_copy_invalidated_requests_cumulative=(
            old_hbm_copy_invalidated_requests_cumulative
        ),
        old_hbm_copy_invalidated_bytes_cumulative=(
            old_hbm_copy_invalidated_bytes_cumulative
        ),
    )
    workflows = _workflow_rows(scheduled)
    requests = _request_rows(scheduled, toggles=toggles)
    for row in requests:
        row["arrival_in_measurement_window"] = (
            measurement_start <= float(row["arrival_ms"]) <= measurement_end
        )
        row["completion_in_measurement_window"] = (
            measurement_start
            <= float(row["simulated_finish_ms"])
            <= measurement_end
        )
    for row in workflows:
        row["arrival_in_measurement_window"] = (
            measurement_start <= float(row["arrival_ms"]) <= measurement_end
        )
        row["join_completion_in_measurement_window"] = (
            measurement_start
            <= float(row["simulated_join_completion_ms"])
            <= measurement_end
        )
    summary = _summary(
        config=resolved,
        policy=policy_label,
        toggles=toggles,
        requests=scheduled,
        workflow_rows=workflows,
        window_rows=window_rows,
        failure_rows=failure_rows,
        repair_rows=repair_rows,
        failure_state_at_horizon=failure_state_at_horizon,
        final_snapshot=final_snapshot,
    )
    summary.update(
        {
            "base_run_horizon_ms": float(horizon),
            "post_all_events_last_event_time_ms": float(
                last_processed_event_time_ms
            ),
            "post_horizon_event_count": int(post_horizon_event_count),
            "repair_events_after_horizon": sum(
                bool(row["after_base_run_horizon"]) for row in repair_rows
            ),
        }
    )
    probe_recovery = [
        float(row["recovery_ms"]) for row in active_source_probe_rows
    ]
    probe_actual_harm = [
        float(row["actual_visible_harm_ms"])
        for row in active_source_probe_rows
    ]
    probe_actions = Counter(
        str(row["recovery_action"]) for row in active_source_probe_rows
    )
    probe_rows_by_reuse_bin = {
        name: [
            row
            for row in active_source_probe_rows
            if row["input_reuse_bin"] == name
        ]
        for name in REUSE_BIN_CONTRACT
    }
    summary.update(
        {
            "active_source_probe_status": active_source_probe_metadata["status"],
            "active_source_probe_requested_count": int(
                active_source_probe_metadata["requested_probe_count"]
            ),
            "active_source_probe_realized_count": len(
                active_source_probe_rows
            ),
            "active_source_probe_tuple_sha256": (
                active_source_probe_metadata["probe_tuple_sha256"]
            ),
            "active_source_probe_eligible_request_count": int(
                active_source_probe_metadata["eligible_request_count"]
            ),
            "active_source_probe_eligible_exposure_gpu_shard_hours": float(
                active_source_probe_metadata[
                    "eligible_exposure_gpu_shard_hours"
                ]
            ),
            "active_source_probe_recovery_mean_ms": (
                statistics.fmean(probe_recovery) if probe_recovery else 0.0
            ),
            "active_source_probe_recovery_p50_ms": _quantile(
                probe_recovery, 0.50
            ),
            "active_source_probe_recovery_p95_ms": _quantile(
                probe_recovery, 0.95
            ),
            "active_source_probe_actual_visible_harm_mean_ms": (
                statistics.fmean(probe_actual_harm)
                if probe_actual_harm
                else 0.0
            ),
            "active_source_probe_actual_visible_harm_p50_ms": _quantile(
                probe_actual_harm, 0.50
            ),
            "active_source_probe_actual_visible_harm_p95_ms": _quantile(
                probe_actual_harm, 0.95
            ),
            "active_source_probe_recovery_action_counts": dict(
                sorted(probe_actions.items())
            ),
            "active_source_probe_restored_tokens": sum(
                int(row["restored_tokens"])
                for row in active_source_probe_rows
            ),
            "active_source_probe_peer_hbm_restored_tokens": sum(
                int(row["peer_hbm_restored_tokens"])
                for row in active_source_probe_rows
            ),
            "active_source_probe_dram_restored_tokens": sum(
                int(row["dram_restored_tokens"])
                for row in active_source_probe_rows
            ),
            "active_source_probe_mixed_or_baseline_specific_restored_tokens": sum(
                int(row["mixed_or_baseline_specific_restored_tokens"])
                for row in active_source_probe_rows
            ),
            "active_source_probe_recomputed_tokens": sum(
                int(row["recompute_tokens"])
                for row in active_source_probe_rows
            ),
            "active_source_probe_rows_with_prior_uptime_fault_history": sum(
                bool(row["probe_state_includes_uptime_poisson_history"])
                for row in active_source_probe_rows
            ),
            "active_source_probe_reuse_bin_contract": dict(
                REUSE_BIN_CONTRACT
            ),
            "active_source_probe_reuse_bin_counts": {
                name: len(rows)
                for name, rows in probe_rows_by_reuse_bin.items()
            },
            "active_source_probe_recovery_mean_ms_by_reuse_bin": {
                name: (
                    statistics.fmean(
                        float(row["recovery_ms"]) for row in rows
                    )
                    if rows
                    else None
                )
                for name, rows in probe_rows_by_reuse_bin.items()
            },
            "active_source_probe_actual_visible_harm_mean_ms_by_reuse_bin": {
                name: (
                    statistics.fmean(
                        float(row["actual_visible_harm_ms"]) for row in rows
                    )
                    if rows
                    else None
                )
                for name, rows in probe_rows_by_reuse_bin.items()
            },
            "active_source_probe_reuse_bin_statistics_are_descriptive": True,
            "workload_reuse_bin_intended_residency_chunk_counts": {
                name: {
                    residency: int(counts[residency])
                    for residency in (
                        "HBM_ONLY",
                        "DRAM_ONLY",
                        "HBM_AND_DRAM",
                        "BYPASS",
                    )
                }
                for name, counts in reuse_bin_intended_residency_counts.items()
            },
            "workload_reuse_bin_tier_admission_chunk_counts": {
                name: {
                    "HBM": int(counts["HBM_ONLY"] + counts["HBM_AND_DRAM"]),
                    "DRAM": int(counts["DRAM_ONLY"] + counts["HBM_AND_DRAM"]),
                    "BYPASS": int(counts["BYPASS"]),
                }
                for name, counts in reuse_bin_intended_residency_counts.items()
            },
            "workload_reuse_bin_admission_semantics": (
                "all_sealed_chunks_grouped_by_source_request_reuse; intended_"
                "independent_tier_residency_at_admission_not_sampled_and_not_"
                "final_async_commit_state"
            ),
        }
    )
    model = resolved.model_profile
    peer_placements = [
        (
            source,
            _peer_target_rank(resolved, source),
            _dp_rank_node(resolved, source)
            == _dp_rank_node(
                resolved, _peer_target_rank(resolved, source)
            ),
        )
        for source in range(int(resolved.dp_size))
    ]
    resolved_config = {
        **asdict(resolved),
        "model_key": resolved.model_key,
        "model_profile": asdict(model),
        "system_dp_size": resolved.dp_size,
        "total_gpu_count": resolved.total_gpus,
        "node_count": resolved.node_count,
        "physical_worker_failure_domains": resolved.total_gpus,
        "resolved_capacity_accounting": {
            "legacy_capacity_field_suffix": "gib_per_target",
            "actual_input_unit": (
                "GiB_per_model_shard; HBM_per_GPU_and_DRAM_per_stage_partition"
            ),
            "logical_dp_target_shard_count": model.shard_count,
            "aggregate_hbm_capacity_bytes_per_logical_dp_target": (
                _gib(resolved.hbm_capacity_gib_per_target)
                * model.shard_count
            ),
            "aggregate_dram_capacity_bytes_per_logical_dp_target": (
                _gib(resolved.dram_capacity_gib_per_target)
                * model.shard_count
            ),
        },
        "failure_rate_exposure": {
            "unit": "total_gpu_shard_hours",
            "poisson_process_count": resolved.total_gpus,
            "logical_dp_rank_count": resolved.dp_size,
            "shards_per_logical_dp_rank": model.shard_count,
            "repair_time_effect": (
                "logical repair event clears admission block and pressure; no queue "
                "outage, executed replacement, or rescheduling"
            ),
        },
        "peer_hbm_binding": {
            "policy": "compatible_node_local_ring_else_next_dp_replica",
            "same_node_source_count": sum(row[2] for row in peer_placements),
            "cross_node_source_count": sum(not row[2] for row in peer_placements),
            "source_target_ranks": [
                {"source_rank": source, "target_rank": target, "same_node": local}
                for source, target, local in peer_placements
            ],
        },
        "resolved_compute_coefficients": {
            "prefill_setup_ms": model.prefill_setup_ms,
            "prefill_ms_per_token": model.prefill_ms_per_token,
            "decode_ms_per_token": model.decode_ms_per_token,
            "recompute_setup_ms": model.recompute_setup_ms,
            "recompute_ms_per_token": model.recompute_ms_per_token,
            "provenance": "synthetic preregistered coefficient; not hardware evidence",
        },
        "resolved_transfer_rates_gbps": {
            "unit": "decimal_GB_per_s",
            "network_effective": _network_effective_bandwidth(resolved),
            "same_node_source_d2d": min(
                float(resolved.hbm_copy_bandwidth_gbps),
                float(resolved.hbm_copy_bandwidth_gbps),
            ),
            "cross_node_source_d2d": min(
                float(resolved.hbm_copy_bandwidth_gbps),
                _network_effective_bandwidth(resolved),
            ),
            "source_d2h_store": min(
                float(resolved.dram_copy_bandwidth_gbps),
                _network_effective_bandwidth(resolved),
            ),
            "peer_hbm_load": min(
                float(resolved.remote_hbm_load_bandwidth_gbps),
                _network_effective_bandwidth(resolved),
            ),
            "dram_load": min(
                float(resolved.dram_load_bandwidth_gbps),
                _network_effective_bandwidth(resolved),
            ),
            "per_target_independent_effective_lanes": True,
            "pp_tp_stage_lane_aggregation": (
                "independent_per_stage_effective_rate_times_shard_count"
            ),
            "shared_nic_contention_modeled": False,
        },
        "resolved_pressure_signals": {
            "queue_depth_threshold": int(resolved.pressure_queue_depth),
            "hbm_quota_occupancy_fraction_threshold": float(
                resolved.hbm_quota_pressure_fraction
            ),
            "failure_rate_is_direct_input": False,
        },
        "resolved_measurement_window": {
            "start_ms": measurement_start,
            "end_ms": measurement_end,
            "fault_process_clipped_to_window": False,
            "workflow_wct_filtered_to_window": False,
        },
    }
    return {
        "schema_version": CELL_SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "design_semantics": DESIGN_SEMANTICS,
        "seed": int(seed),
        "policy": policy_label,
        "algorithms": asdict(toggles),
        "algorithm_effects": {
            "algorithm1_slack": (
                "caller_supplied_causal_visible_residual_dag_point_cpm"
                if toggles.algorithm1
                else "zero_slack_fail_closed"
            ),
            "algorithm2_admission": (
                "slack_landlord_independent_hbm_dram"
                if toggles.algorithm2
                else "static_independent_hbm_dram_fcfs"
            ),
            "algorithm3_alpha": (
                "target_local_pressure_only_aimd"
                if toggles.algorithm3
                else "fixed_configured_alpha"
            ),
        },
        "ablation": toggles.label,
        "config": resolved_config,
        "summary": summary,
        "formal_metric_contract": {
            "primary_failure_endpoint_prefix": (
                "actual_failure_added_workflow_harm_surrogate_"
            ),
            "dual_failure_estimands": {
                "uptime_hazard_incidence": (
                    "full_horizon_wall_clock_poisson_including_idle_events"
                ),
                "fault_conditioned_recovery_efficacy": (
                    ACTIVE_SOURCE_PROBE_EVIDENCE_CLASS
                ),
            },
            "fault_conditioned_endpoint_prefix": "active_source_probe_",
            "legacy_aliases_forbidden_for_claims": [
                "failure_added_workflow_wct_*",
                "simulated_workflow_wct_*",
            ],
            "end_to_end_dag_execution_replay": False,
            "failure_time_actual_slack_semantics": (
                "policy_independent_actual_duration_boundary_cpm_slack_with_"
                "elapsed_due_time_decrement"
            ),
            "failure_history_contract": {
                "physical_hazard_identity": "source_rank_plus_shard_rank",
                "logical_replica_target_outage": (
                    "conservative_rank_aggregate_until_latest_60s_repair"
                ),
                "horizon_state_source": "failure_state_at_horizon",
                "post_all_events_state_source": (
                    "final_state.failure_history"
                ),
                "summary_at_horizon_fields_source": (
                    "failure_state_at_horizon"
                ),
                "events_after_horizon_are_cleanup_not_horizon_state": True,
                "repair_preserves_ever_failed_and_old_copy_invalidation": True,
                "forced_synthetic_target_first_injection": False,
                "natural_prior_target_and_repeated_shard_history_retained": True,
                "standby_rotation_modeled": False,
                "complete_current_design_target_failover_evaluated": False,
                "target_failure_history_is_no_standby_rotation_conservative_sensitivity": (
                    True
                ),
            },
        },
        "request_rows": requests,
        "workflow_rows": workflows,
        "window_rows": window_rows,
        "failure_rows": failure_rows,
        "repair_rows": repair_rows,
        "failure_state_at_horizon": failure_state_at_horizon,
        "active_source_probe_metadata": {
            **active_source_probe_metadata,
            "realized_probe_count": len(active_source_probe_rows),
        },
        "active_source_probe_rows": active_source_probe_rows,
        "active_source_fault_conditioned_recovery": {
            **active_source_probe_metadata,
            "realized_probe_count": len(active_source_probe_rows),
            "raw_row_location": (
                "active_source_fault_conditioned_recovery.probe_rows"
            ),
            "statistical_unit": (
                "one_equal_exposure_stratified_mean_per_scenario_seed; "
                "paired_seed_outer_bootstrap_only"
            ),
            "formal_ci_unit": (
                f"paired_seed_mean_over_{int(resolved.active_source_probe_count)}_"
                "strata"
            ),
            "probe_inner_bootstrap_allowed": False,
            "probe_distribution_quantiles": (
                "descriptive_nonindependent_no_ci"
            ),
            "reuse_analysis_contract": {
                "bins": dict(REUSE_BIN_CONTRACT),
                "fraction_definition": (
                    "clip(input_reuse_tokens/prompt_tokens,[0,1]); "
                    "prompt_tokens=0=>NA (zero prompts are rejected at input)"
                ),
                "workload_distribution_source": (
                    "all_request_rows_not_probe_sample"
                ),
                "recovery_endpoint_source": (
                    "64_equal_exposure_stratified_probe_rows_descriptive"
                ),
                "reuse_does_not_deduplicate_protection_copy": True,
            },
            "endpoint_contract": {
                "recovery_delay": "recovery_ms",
                "ground_truth_harm": "actual_visible_harm_ms",
                "policy_harm": "policy_modeled_visible_harm_ms",
                "protected_prefix": "restored_tokens",
                "peer_hbm_prefix": "peer_hbm_restored_tokens",
                "dram_prefix": "dram_restored_tokens",
                "recompute_suffix": "recompute_tokens",
            },
            "uptime_hazard_estimand_is_separate": True,
            "reference_state_is_uptime_fault_free": not any(
                row["probe_state_includes_uptime_poisson_history"]
                for row in active_source_probe_rows
            ),
            "probe_rows": active_source_probe_rows,
        },
        "final_state": final_snapshot,
        "gates": {
            "simulated_only_label_present": True,
            "full_pinned_sglang_scheduler_replay": False,
            "rank_local_fcfs_surrogate_declared": True,
            "dp_changes_routing_capacity_and_failure_exposure": True,
            "model_changes_kv_shards_compute_and_recovery": True,
            "pp_tp_capacity_aggregation_explicit": True,
            "bandwidth_enters_copy_and_load_duration": True,
            "network_axis_enters_cross_node_and_store_paths": True,
            "peer_hbm_target_placement_is_operational": True,
            "per_node_shared_nic_contention_modeled": False,
            "concurrency_enters_queue_and_pressure": True,
            "failure_rate_generates_physical_worker_poisson_events": True,
            "single_worker_shard_failure_counterfactual": True,
            "executed_physical_fail_stop": False,
            "failure_queue_outage_replacement_rescheduling_modeled": False,
            "forced_synthetic_target_first_injection_disabled": True,
            "natural_uptime_target_failure_history_preserved": True,
            "physical_gpu_shard_and_logical_replica_history_separated": True,
            "current_outage_and_ever_failed_counters_separated": True,
            "horizon_failure_state_frozen_before_post_horizon_events": True,
            "summary_at_horizon_fields_derive_from_horizon_failure_state": (
                int(
                    summary[
                        "current_physical_gpu_shard_outage_count_at_horizon"
                    ]
                )
                == int(
                    failure_state_at_horizon[
                        "current_physical_gpu_shard_outage_count"
                    ]
                )
                and int(
                    summary[
                        "current_logical_replica_target_outage_count_at_horizon"
                    ]
                )
                == int(
                    failure_state_at_horizon[
                        "current_logical_replica_target_outage_count"
                    ]
                )
            ),
            "final_state_is_post_all_events_not_horizon_state": (
                final_snapshot["failure_history"]["snapshot_scope"]
                == "post_all_scheduled_events_including_repairs_after_horizon"
            ),
            "old_hbm_copy_invalidation_counters_cumulative": True,
            "standby_rotation_modeled": False,
            "complete_current_design_target_failover_evaluated": False,
            "target_failure_history_is_no_standby_rotation_conservative_sensitivity": (
                True
            ),
            "failed_worker_peer_hbm_invalidated_for_all_policies": True,
            "explicit_worker_repair_event_modeled": True,
            "repair_reenables_future_hbm_admission_only": True,
            "repair_does_not_restore_old_hbm_copies": True,
            "repair_does_not_schedule_hbm_reprotection": True,
            "fault_process_covers_full_run_horizon": True,
            "uptime_hazard_idle_events_retained": True,
            "uptime_hazard_has_no_forced_effective_fault_floor": (
                int(resolved.minimum_faults_per_cell) == 0
            ),
            "active_source_probe_schedule_policy_independent": True,
            "active_source_probe_schedule_failure_rate_independent": True,
            "active_source_probe_uses_physical_gpu_shard_time": True,
            "active_source_probe_requires_active_and_sealed": all(
                bool(row["eligible_active_source"])
                and bool(row["eligible_complete_sealed_frontier"])
                and float(row["eligible_interval_start_ms"])
                <= float(row["time_ms"])
                < float(row["eligible_interval_end_ms"])
                and int(row["sealed_frontier_tokens"]) > 0
                for row in active_source_probe_rows
            ),
            "active_source_probe_does_not_filter_policy_outcomes": all(
                not bool(row["eligibility_filters_admission_or_recovery_outcome"])
                for row in active_source_probe_rows
            ),
            "active_source_probe_count_realized_or_disabled": (
                len(active_source_probe_rows)
                == int(resolved.active_source_probe_count)
            ),
            "active_source_probe_one_per_equal_exposure_stratum": (
                sorted(
                    int(row["stratum_index"])
                    for row in active_source_probe_rows
                )
                == list(range(int(resolved.active_source_probe_count)))
            ),
            "active_source_probe_tuple_hash_present": bool(
                active_source_probe_metadata["probe_tuple_sha256"]
            ),
            "active_source_probe_query_non_mutating": all(
                not bool(row["primary_counters_updated"])
                for row in active_source_probe_rows
            ),
            "active_source_probe_excluded_from_primary_counters": True,
            "active_source_probe_ground_truth_harm_uses_actual_slack": True,
            "active_source_probe_ci_uses_paired_seed_outer_bootstrap_only": True,
            "active_source_probe_inner_bootstrap_forbidden": True,
            "measurement_window_only_filters_steady_telemetry": True,
            "t0_control_window_is_snapshot_only": (
                len(window_rows) > 0
                and math.isclose(
                    float(window_rows[0]["time_ms"]),
                    0.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and bool(window_rows[0]["initial_sample"])
                and not bool(window_rows[0]["aimd_update_applied"])
            ),
            "aimd_updates_only_at_positive_periodic_windows": all(
                bool(row["aimd_update_applied"])
                == bool(toggles.algorithm3 and float(row["time_ms"]) > 0.0)
                for row in window_rows
            ),
            "initial_sample_does_not_advance_stable_counter": (
                not toggles.algorithm3
                or int(window_rows[0]["controller_stable_windows_max"]) == 0
            ),
            "alpha_pressure_only": True,
            "failure_rate_direct_alpha_dependency_absent": True,
            "effective_alpha_operational_zero_supported": True,
            "input_reuse_does_not_deduplicate_protection_copy": True,
            "workflow_join_wct_aggregated": False,
            "workflow_actual_harm_surrogate_aggregated": True,
            "end_to_end_dag_failure_execution_replay": False,
            "failure_is_fail_now_counterfactual_recovery_delay": True,
            "ground_truth_harm_uses_actual_slack": True,
            "failure_time_actual_slack_is_boundary_cpm_with_elapsed_decrement": (
                True
            ),
            "policy_predicted_slack_not_used_as_ground_truth": True,
            "algorithm1_policy_input_is_request_slack_when_multiplier_one": (
                float(resolved.algorithm1_prediction_error_multiplier) != 1.0
                or all(
                    math.isclose(
                        row.policy_slack_ms,
                        max(
                            float(
                                row.request.predicted_slack_ms
                                if row.request.predicted_slack_ms is not None
                                else row.request.slack_ms
                            ),
                            0.0,
                        ),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    for row in scheduled
                )
            ),
            "algorithm1_preapplied_multiplier_matches_cell_config": all(
                row.request.policy_slack_multiplier_applied is None
                or math.isclose(
                    float(row.request.policy_slack_multiplier_applied),
                    float(resolved.algorithm1_prediction_error_multiplier),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for row in scheduled
            ),
            "algorithm1_base_prediction_and_effective_policy_input_separated": all(
                row.request.policy_slack_multiplier_applied is None
                or row.request.predicted_slack_ms is not None
                for row in scheduled
            ),
            "algorithm1_nonzero_slack_input_observed": bool(
                summary["algorithm1_nonzero_slack_input_observed"]
            ),
            "algorithm1_enabled_has_nonzero_effective_slack_or_is_diagnostic": (
                not toggles.algorithm1
                or bool(summary["effective_policy_slack_nonzero_count"])
            ),
            "algorithm1_disabled_zeros_effective_slack": (
                toggles.algorithm1
                or int(summary["effective_policy_slack_nonzero_count"]) == 0
            ),
            "formal_analysis_uses_explicit_actual_harm_surrogate_fields": True,
            "legacy_wct_aliases_forbidden_for_formal_claims": True,
            "all_requests_have_explicit_actual_slack": all(
                row.request.actual_slack_ms is not None for row in scheduled
            ),
            "failure_local_metric_named_modeled_visible_harm": True,
            "minimum_fault_floor_preserves_rate_semantics": (
                int(resolved.minimum_faults_per_cell) == 0
            ),
        },
        "approximation_boundary": {
            "scheduler": "rank-local FCFS service surrogate",
            "workflow_join": (
                "base join WCT plus max per-workflow actual causal-slack harm; "
                "not end-to-end DAG failure execution replay"
            ),
            "failure_queue_feedback": (
                "fail-now recovery-delay counterfactual extends retention and HBM "
                "pressure but does not reschedule later rank service"
            ),
            "measurement_window": (
                "tags steady alpha/occupancy/pressure only; faults and all workflow "
                "WCT remain full-horizon"
            ),
            "active_source_fault_conditioned_recovery": (
                "policy-independent active+sealed GPU-shard exposure is sampled "
                "before policy execution; each timestamp advances ordinary copy "
                "time and then performs a pure fail-now query. If uptime Poisson "
                "faults coexist, later probe rows explicitly report that history; "
                "the clean recovery-efficacy reference uses failure_rate=0."
            ),
            "protection": "event-level current controller or declared baseline adapter",
            "algorithm1": (
                "RequestInput.predicted_slack_ms is the scenario causal base "
                "prediction when provided; RequestInput.slack_ms is the effective "
                "sensitivity-adjusted policy input; RequestInput.actual_slack_ms "
                "is isolated boundary-CPM endpoint truth. The large-scale Weka "
                "projector binds scenario-specific cross-fit point-estimate and "
                "multiplier provenance; generic callers use the audited in-core "
                "interpolation path."
            ),
            "physical_data_movement": False,
            "network_contention": (
                "path rates are effective end-to-end caps; per-node shared-NIC "
                "queue contention is not simulated"
            ),
            "worker_failure": (
                "single-worker-shard counterfactual: fail-now estimate for the active "
                "source request plus conservative invalidation of aggregate Peer-HBM "
                "resident on that worker; DRAM survives; an explicit logical repair "
                "event re-enables only future HBM admission; no queue outage, executed "
                "replacement, rescheduling, old-copy restoration, or post-failure "
                "HBM re-protection. Physical hazard history is keyed by "
                "(source_rank,shard_rank), while HBM invalidation/outage is an "
                "explicit conservative logical-replica aggregate."
            ),
            "target_failover": (
                "The authoritative design and source implementation require atomic "
                "healthy standby rotation and subsequent-chunk admission to the "
                "replacement target. This large-scale DES cannot migrate its "
                "per-target runtime/controller/independent-DRAM state, so it does "
                "not model standby rotation. Natural prior-target invalidation and "
                "repair history is retained only as a conservative no-standby-"
                "rotation sensitivity and is not complete current-design failover."
            ),
            "failure_time_actual_slack": (
                "policy-independent actual-duration boundary CPM Slack minus elapsed "
                "time, floored at zero; the failure event does not dynamically replay "
                "or recompute the residual DAG"
            ),
        },
    }


__all__ = [
    "ACTIVE_SOURCE_PROBE_EVIDENCE_CLASS",
    "ALPHA_DRIVER_CONTRACT",
    "AlgorithmToggles",
    "BASELINE_POLICIES",
    "CELL_SCHEMA_VERSION",
    "EVIDENCE_CLASS",
    "LargeScaleCellConfig",
    "MODEL_PROFILES",
    "ModelProfile",
    "RequestInput",
    "REUSE_BIN_CONTRACT",
    "generate_workload",
    "run_cell",
]
