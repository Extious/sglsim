"""Deterministic TraceLab request compiler for the global-DES experiments.

This module deliberately separates two things which are easy to conflate:

* a *request universe* is the source provider-session/round identity selected by
  ``(group, seed, request_universe)``; and
* a compiled replay applies the selected context and load axes to that same
  universe.

Consequently a policy, failure, deployment, context, or load change cannot
silently resample requests.  The compiler also refuses to invent missing paper
configuration.  The current v3 files can be inspected with
``request_layer_readiness``; compilation raises ``WorkloadNotReadyError`` until
the sampling, projection, and timestamp contracts are explicit.

TraceLab does not contain prompt text.  Each output record therefore carries a
deterministically generatable token identity instead of materialising hundreds
of thousands of synthetic token IDs.  A provider-session owns one prefix
stream, while every round owns a distinct suffix stream.  This preserves the
trace's declared within-session prefix identity without pretending that the
public artifact contains cross-session content hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from AegisKV.experiments.src.controlled_prefix_artifact import (
    CONTROLLED_PREFIX_DATASET_ID,
    CONTROLLED_PREFIX_SCHEMA,
    CONTROLLED_PREFIX_SHA256,
    audit_controlled_prefix_artifact,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACELAB_PATH = (
    EXPERIMENT_ROOT / "data" / "cache" / "syfi_coding_trace-v0.0.2.duckdb"
)
TRACELAB_DATASET_ID = "syfi-coding-trace-v0.0.2"
TRACELAB_SHA256 = (
    "a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744"
)
COMPILER_SCHEMA = "aegiskv.global-des-request-universe.v1"
TOKEN_IDENTITY_SCHEMA = "aegiskv.deterministic-token-identity.v1"
SUPPORTED_GROUPS = frozenset(f"P{index}" for index in range(1, 11))


_EXPECTED_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "rounds": (
        ("round_pk", "BIGINT"),
        ("ingest_seq", "BIGINT"),
        ("provider", "VARCHAR"),
        ("project", "VARCHAR"),
        ("session_id", "VARCHAR"),
        ("round_index", "BIGINT"),
        ("round_id", "VARCHAR"),
        ("model", "VARCHAR"),
        ("input_tokens_total", "BIGINT"),
        ("prefix_tokens", "BIGINT"),
        ("newly_append_tokens", "BIGINT"),
        ("claude_uncached_input_tokens", "BIGINT"),
        ("claude_cache_creation_input_tokens", "BIGINT"),
        ("claude_cache_read_input_tokens", "BIGINT"),
        ("output_tokens", "BIGINT"),
        ("current_input_event_count", "BIGINT"),
        ("current_user_message_count", "BIGINT"),
        ("current_tool_result_count", "BIGINT"),
        ("current_user_message_chars", "BIGINT"),
        ("current_tool_result_chars", "BIGINT"),
        ("current_input_chars", "BIGINT"),
        ("first_input_event_type", "VARCHAR"),
        ("user", "VARCHAR"),
        ("store", "VARCHAR"),
        ("trace_key", "VARCHAR"),
        ("turn_id", "UUID"),
        ("reasoning_output_tokens", "BIGINT"),
    ),
    "timing_events": (
        ("round_pk", "BIGINT"),
        ("event_index", "BIGINT"),
        ("event_type", "VARCHAR"),
        ("source", "VARCHAR"),
        ("timestamp", "TIMESTAMP"),
        ("tool_call_id", "VARCHAR"),
        ("tool_index", "BIGINT"),
        ("tool_name", "VARCHAR"),
        ("is_error", "BOOLEAN"),
        ("result_chars", "BIGINT"),
        ("content_chars", "BIGINT"),
    ),
    "tool_calls": (
        ("round_pk", "BIGINT"),
        ("tool_index", "BIGINT"),
        ("tool_name", "VARCHAR"),
        ("tool_call_id", "VARCHAR"),
        ("emitted_at", "TIMESTAMP"),
        ("result_at", "TIMESTAMP"),
        ("tool_wall_latency_ms", "BIGINT"),
        ("tool_internal_latency_ms", "BIGINT"),
        ("is_error", "BOOLEAN"),
        ("input_chars", "BIGINT"),
        ("result_chars", "BIGINT"),
        ("continuation_of_tool_call_id", "VARCHAR"),
        ("command_status", "VARCHAR"),
        ("command_exit_code", "HUGEINT"),
        ("executables", "VARCHAR[]"),
        ("executable_parse_status", "VARCHAR"),
        ("executable_parse_reason", "VARCHAR"),
        ("command_skeleton", "VARCHAR"),
    ),
    "trace_source": (("path", "VARCHAR"),),
}


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _json_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReadinessError:
    """One structured, machine-readable reason compilation cannot start."""

    code: str
    config_path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "config_path": self.config_path,
            "message": self.message,
        }


@dataclass(frozen=True)
class WorkloadReadiness:
    errors: tuple[ReadinessError, ...]

    @property
    def ready(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "errors": [error.as_dict() for error in self.errors],
        }


class WorkloadNotReadyError(RuntimeError):
    """Raised before any request is emitted when the contract is incomplete."""

    def __init__(self, errors: Sequence[ReadinessError]):
        self.errors = tuple(errors)
        super().__init__(
            "request workload is not ready: "
            + "; ".join(
                f"{error.code}@{error.config_path}: {error.message}"
                for error in self.errors
            )
        )

    def as_dict(self) -> dict[str, object]:
        return WorkloadReadiness(self.errors).as_dict()


class TraceLabDataError(RuntimeError):
    """Raised for an artifact, schema, or selected-row invariant violation."""

    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(message)


@dataclass(frozen=True)
class TraceLabAudit:
    path: str
    sha256: str
    schema_sha256: str
    round_count: int
    provider_session_count: int
    tool_call_count: int
    timing_event_count: int
    provider_counts: tuple[tuple[str, int, int], ...]
    nonmonotonic_event1_transitions: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
            "round_count": self.round_count,
            "provider_session_count": self.provider_session_count,
            "tool_call_count": self.tool_call_count,
            "timing_event_count": self.timing_event_count,
            "provider_counts": [
                {"provider": provider, "rounds": rounds, "sessions": sessions}
                for provider, rounds, sessions in self.provider_counts
            ],
            "nonmonotonic_event1_transitions": self.nonmonotonic_event1_transitions,
        }


@dataclass(frozen=True)
class ControlledPrefixAudit:
    """Lineage summary for the deterministic, explicitly non-empirical P6 input."""

    path: str
    sha256: str
    schema_version: str
    content_family_count: int
    generated_session_count: int
    selected_sessions_per_family: int
    rounds_per_session: int
    empirical_measurement: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "content_family_count": self.content_family_count,
            "generated_session_count": self.generated_session_count,
            "selected_sessions_per_family": self.selected_sessions_per_family,
            "rounds_per_session": self.rounds_per_session,
            "empirical_measurement": self.empirical_measurement,
        }


@dataclass(frozen=True)
class TokenIdentity:
    """Compact recipe for a deterministic synthetic input-token sequence."""

    prefix_stream_sha256: str
    prefix_tokens: int
    suffix_stream_sha256: str
    suffix_tokens: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": TOKEN_IDENTITY_SCHEMA,
            "prefix_stream_sha256": self.prefix_stream_sha256,
            "prefix_tokens": self.prefix_tokens,
            "suffix_stream_sha256": self.suffix_stream_sha256,
            "suffix_tokens": self.suffix_tokens,
            "generator": "sha256_counter_u32_mod_vocab",
        }


@dataclass(frozen=True)
class RequestRecord:
    """Immutable, JSON-compatible request-layer replay record."""

    rid: str
    workflow: str
    session: str
    provider: str
    round_index: int
    arrival_ns: int
    input_tokens: int
    output_tokens: int
    cached_prefix_tokens: int
    prefix_key: str
    input_token_identity: TokenIdentity
    source_round_pk: int
    source_round_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "rid": self.rid,
            "workflow": self.workflow,
            "session": self.session,
            "provider": self.provider,
            "round_index": self.round_index,
            "arrival_ns": self.arrival_ns,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "prefix_key": self.prefix_key,
            "input_token_identity": self.input_token_identity.as_dict(),
            "source_round_pk": self.source_round_pk,
            "source_round_id": self.source_round_id,
        }


@dataclass(frozen=True)
class CompiledWorkload:
    records: tuple[RequestRecord, ...]
    request_universe_hash: str
    compiled_requests_hash: str
    dataset_audit: TraceLabAudit | ControlledPrefixAudit
    metadata: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": COMPILER_SCHEMA,
            "request_universe_hash": self.request_universe_hash,
            "compiled_requests_hash": self.compiled_requests_hash,
            "records": [record.as_dict() for record in self.records],
            "dataset_audit": self.dataset_audit.as_dict(),
            "metadata": dict(self.metadata),
        }

    def iter_tokenized_requests(self, *, vocab_size: int = 151936) -> Iterable[Any]:
        """Yield real SGLSim request objects without an implicit record cap.

        Token IDs are materialised lazily per request so callers can schedule an
        arrival and release their temporary list before constructing the next
        request.  The method never shortens a frozen request universe.
        """

        from sglsim.srt.managers.io_struct import TokenizedGenerateReqInput
        from sglsim.srt.managers.schedule_batch import SamplingParams

        for record in self.records:
            simulation = {
                "workflow_id": record.workflow,
                "request_session": record.session,
                "request_provider": record.provider,
                "round_index": record.round_index,
                "arrival_ns": record.arrival_ns,
                "cached_prefix_tokens": record.cached_prefix_tokens,
                "prefix_key": record.prefix_key,
                "source_round_pk": record.source_round_pk,
                "source_round_id": record.source_round_id,
                "request_universe_hash": self.request_universe_hash,
                "compiled_requests_hash": self.compiled_requests_hash,
            }
            yield TokenizedGenerateReqInput(
                rid=record.rid,
                input_ids=list(
                    deterministic_token_ids(
                        record.input_token_identity,
                        vocab_size=vocab_size,
                    )
                ),
                input_token_count=record.input_tokens,
                sampling_params=SamplingParams(
                    max_new_tokens=record.output_tokens,
                    custom_params={"simulation": simulation},
                ),
                original_prompt_len=record.input_tokens,
                original_max_new_tokens=record.output_tokens,
            )

    def to_tokenized_requests(
        self, *, vocab_size: int = 151936
    ) -> tuple[Any, ...]:
        """Materialise the complete frozen request universe as SGLSim inputs."""

        return tuple(self.iter_tokenized_requests(vocab_size=vocab_size))


def _issue(
    errors: list[ReadinessError], code: str, path: str, message: str
) -> None:
    errors.append(ReadinessError(code=code, config_path=path, message=message))


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _finite_positive(value: object) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def request_layer_readiness(
    resolved_request: Mapping[str, Any],
    axes: Mapping[str, Any],
) -> WorkloadReadiness:
    """Return all known config gaps without guessing replacement values.

    ``resolved_request`` is the resolved v3 request-layer envelope, not merely
    its ``fixed`` member.  ``axes`` may be a complete experiment cell; only
    axes declared by the request layer participate in request compilation.
    """

    errors: list[ReadinessError] = []
    if resolved_request.get("schema_version") != "aegiskv.global-des-layer.v3":
        _issue(errors, "schema_mismatch", "#/schema_version", "resolved v3 request layer required")
    if resolved_request.get("layer") != "request":
        _issue(errors, "layer_mismatch", "#/layer", "request layer required")
    group = resolved_request.get("group")
    if group not in SUPPORTED_GROUPS:
        _issue(errors, "unsupported_group", "#/group", "compiler scope is P1-P10")
    fixed = resolved_request.get("fixed")
    declared_axes = resolved_request.get("axes")
    if not isinstance(fixed, Mapping):
        _issue(errors, "missing_fixed", "#/fixed", "resolved fixed object required")
        return WorkloadReadiness(tuple(errors))
    if not isinstance(declared_axes, Mapping):
        _issue(errors, "missing_axes", "#/axes", "resolved axes object required")
        declared_axes = {}

    for axis_name, allowed in declared_axes.items():
        path = f"#/axes/{axis_name}"
        if axis_name not in axes:
            _issue(errors, "axis_value_missing", path, "selected request-axis value required")
            continue
        if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes)):
            _issue(errors, "axis_domain_invalid", path, "axis domain must be a non-empty sequence")
        elif axes[axis_name] not in allowed:
            _issue(errors, "axis_value_out_of_domain", path, f"selected value {axes[axis_name]!r} is not declared")

    workload = fixed.get("workload")
    arrival = fixed.get("arrival")
    context = fixed.get("context")
    seed_contract = fixed.get("seed")
    lineage = fixed.get("artifact_lineage")
    for name, value in (
        ("workload", workload),
        ("arrival", arrival),
        ("context", context),
        ("seed", seed_contract),
        ("artifact_lineage", lineage),
    ):
        if not isinstance(value, Mapping):
            _issue(errors, "request_section_missing", f"#/fixed/{name}", "object required")
    if not all(
        isinstance(value, Mapping)
        for value in (workload, arrival, context, seed_contract, lineage)
    ):
        return WorkloadReadiness(tuple(errors))

    kind = workload.get("kind")
    trace_kind = kind == "tracelab_projected_session"
    controlled_kind = kind == "controlled_prefix_reuse"
    if not (trace_kind or controlled_kind):
        _issue(
            errors,
            "unsupported_workload_kind",
            "#/fixed/workload/kind",
            f"unsupported kind {kind!r}",
        )

    if arrival.get("clock") != "global_virtual_time" or arrival.get("timestamp_unit") != "ns":
        _issue(errors, "arrival_clock_mismatch", "#/fixed/arrival", "global virtual nanosecond clock required")
    if arrival.get("preserve_request_order") is not True:
        _issue(errors, "request_order_not_preserved", "#/fixed/arrival/preserve_request_order", "must be true")
    if arrival.get("zero_origin") != "minimum_selected_timestamp":
        _issue(errors, "arrival_origin_missing", "#/fixed/arrival/zero_origin", "minimum selected timestamp required")

    if trace_kind:
        if workload.get("dataset_id") != TRACELAB_DATASET_ID:
            _issue(errors, "dataset_id_mismatch", "#/fixed/workload/dataset_id", f"expected pinned dataset {TRACELAB_DATASET_ID!r}")
        configured_sha256 = workload.get("sha256")
        if configured_sha256 is None:
            _issue(errors, "input_sha256_value_missing", "#/fixed/workload/sha256", "pinned SHA-256 value must be recorded")
        elif configured_sha256 != TRACELAB_SHA256:
            _issue(errors, "input_sha256_mismatch", "#/fixed/workload/sha256", f"expected {TRACELAB_SHA256}")
        if workload.get("request_universe") != "provider_sessions":
            _issue(errors, "sampling_unit_mismatch", "#/fixed/workload/request_universe", "provider_sessions required")
        if workload.get("selection") != "deterministic_by_seed":
            _issue(errors, "selection_contract_mismatch", "#/fixed/workload/selection", "deterministic_by_seed required")

        sampling = workload.get("sampling")
        if not isinstance(sampling, Mapping):
            _issue(errors, "sampling_spec_missing", "#/fixed/workload/sampling", "explicit provider-session sample count, strata, bins, and complete-session policy required")
        else:
            if sampling.get("unit") != "provider_session":
                _issue(errors, "sampling_unit_mismatch", "#/fixed/workload/sampling/unit", "provider_session required")
            if list(sampling.get("strata", [])) != ["provider", "session_round_count_bin"]:
                _issue(errors, "sampling_strata_underspecified", "#/fixed/workload/sampling/strata", "provider and session_round_count_bin required")
            bins = sampling.get("session_round_count_bins")
            if not (isinstance(bins, list) and bins and all(_positive_int(value) for value in bins) and bins == sorted(set(bins))):
                _issue(errors, "sampling_bins_invalid", "#/fixed/workload/sampling/session_round_count_bins", "strictly increasing positive lower bounds required")
            if not _positive_int(sampling.get("sessions_per_stratum")):
                _issue(errors, "sample_count_missing", "#/fixed/workload/sampling/sessions_per_stratum", "positive exact sample count required")
            if not _positive_int(sampling.get("expected_selected_sessions")):
                _issue(errors, "sample_count_missing", "#/fixed/workload/sampling/expected_selected_sessions", "positive exact total session count required")
            if sampling.get("complete_sessions_only") is not True:
                _issue(errors, "silent_truncation_forbidden", "#/fixed/workload/sampling/complete_sessions_only", "must be true; round windows/truncation are forbidden")

        projection = workload.get("context_projection")
        if not isinstance(projection, Mapping):
            _issue(errors, "projection_spec_missing", "#/fixed/workload/context_projection", "rank-preserving scale, source anchor, output cap, and clipping audit must be explicit")
        else:
            if projection.get("algorithm") != "rank_preserving_scale_to_target_median_then_clip":
                _issue(errors, "projection_algorithm_mismatch", "#/fixed/workload/context_projection/algorithm", "rank_preserving_scale_to_target_median_then_clip required")
            if not _positive_int(projection.get("source_median_input_tokens")):
                _issue(errors, "projection_anchor_missing", "#/fixed/workload/context_projection/source_median_input_tokens", "positive source median required")
            if not _finite_positive(projection.get("target_median_fraction")) or float(projection.get("target_median_fraction", 0)) > 1.0:
                _issue(errors, "projection_target_invalid", "#/fixed/workload/context_projection/target_median_fraction", "finite fraction in (0, 1] required")
            if not _positive_int(projection.get("output_cap_tokens")):
                _issue(errors, "output_cap_missing", "#/fixed/workload/context_projection/output_cap_tokens", "positive output cap required")
            if projection.get("preserve_prefix_fraction") is not True:
                _issue(errors, "prefix_projection_invalid", "#/fixed/workload/context_projection/preserve_prefix_fraction", "must be true")
            if projection.get("clipping_audit_required") is not True:
                _issue(errors, "silent_projection_forbidden", "#/fixed/workload/context_projection/clipping_audit_required", "must be true")

        exclusions = workload.get("anomaly_exclusions")
        if not isinstance(exclusions, Mapping):
            _issue(errors, "anomaly_exclusion_contract_missing", "#/fixed/workload/anomaly_exclusions", "the complete observed anomaly set must be frozen before sampling")
        else:
            if exclusions.get("policy") != "exclude_before_seeded_sampling":
                _issue(errors, "anomaly_exclusion_policy_invalid", "#/fixed/workload/anomaly_exclusions/policy", "exclude_before_seeded_sampling required")
            if exclusions.get("criterion") != "event_index_1_timestamp_decrease_under_round_index_then_round_pk":
                _issue(errors, "anomaly_exclusion_criterion_invalid", "#/fixed/workload/anomaly_exclusions/criterion", "exact event-1 monotonicity criterion required")
            rows = exclusions.get("excluded_provider_sessions")
            normalized: list[tuple[str, str, int]] = []
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, Mapping) and isinstance(row.get("provider"), str) and isinstance(row.get("session_id"), str) and _positive_int(row.get("nonmonotonic_transitions")):
                        normalized.append((row["provider"], row["session_id"], int(row["nonmonotonic_transitions"])))
            if not isinstance(rows, list) or len(normalized) != len(rows) or normalized != sorted(set(normalized)):
                _issue(errors, "anomaly_exclusion_set_invalid", "#/fixed/workload/anomaly_exclusions/excluded_provider_sessions", "sorted unique provider/session/transition rows required")
            if exclusions.get("observed_affected_sessions") != len(normalized) or exclusions.get("observed_nonmonotonic_transitions") != sum(row[2] for row in normalized):
                _issue(errors, "anomaly_exclusion_count_mismatch", "#/fixed/workload/anomaly_exclusions", "declared session and transition counts must match the explicit set")

        provenance = workload.get("parameter_provenance")
        required_provenance = {"source_median_input_tokens", "session_round_count_bins", "sessions_per_stratum", "target_median_fraction", "output_cap_tokens"}
        if not isinstance(provenance, Mapping) or not required_provenance.issubset(provenance) or not all(isinstance(provenance[key], str) and provenance[key] for key in required_provenance):
            _issue(errors, "parameter_provenance_missing", "#/fixed/workload/parameter_provenance", "observed anchors and modeled choices must have explicit provenance")

        if arrival.get("timestamp_source") != "timing_events.event_index_1.timestamp":
            _issue(errors, "arrival_timestamp_source_missing", "#/fixed/arrival/timestamp_source", "observable round-boundary event must be explicit")
        if arrival.get("session_order") != "round_index_then_round_pk":
            _issue(errors, "session_order_missing", "#/fixed/arrival/session_order", "round_index_then_round_pk required")
        if arrival.get("nonmonotonic_timestamp_policy") != "fail_selected_session":
            _issue(errors, "arrival_anomaly_policy_missing", "#/fixed/arrival/nonmonotonic_timestamp_policy", "fail_selected_session required; silent clamping is forbidden")
        if arrival.get("load_scaling") != "divide_relative_arrival_ns":
            _issue(errors, "load_scaling_missing", "#/fixed/arrival/load_scaling", "load may only divide relative arrival times")
        mode = arrival.get("mode")
        if mode not in {"trace_replay", "preregistered_low_high_recovery_trace"}:
            _issue(errors, "arrival_mode_unsupported", "#/fixed/arrival/mode", f"unsupported mode {mode!r}")
        if mode == "preregistered_low_high_recovery_trace":
            phase = axes.get("load_phase")
            multipliers = arrival.get("phase_load_multipliers")
            order = arrival.get("phase_order")
            if not isinstance(multipliers, Mapping) or phase not in multipliers or not isinstance(order, list) or phase not in order:
                _issue(errors, "load_phase_undefined", "#/fixed/arrival", "selected phase and multiplier required")
            if arrival.get("phase_replay_scope") != "same_complete_request_universe_per_phase":
                _issue(errors, "phase_replay_scope_missing", "#/fixed/arrival/phase_replay_scope", "same complete request universe per phase required")
            if arrival.get("phase_boundary_policy") != "no_request_truncation_allow_spill":
                _issue(errors, "phase_boundary_policy_missing", "#/fixed/arrival/phase_boundary_policy", "phase duration must not silently drop requests")
            if arrival.get("phase_duration_semantics") != "nominal_control_interval_not_request_cutoff" or arrival.get("phase_origin_policy") != "independent_zero_origin_per_phase_cell":
                _issue(errors, "phase_semantics_missing", "#/fixed/arrival", "phase duration and origin semantics must be explicit")
            effective_load = multipliers.get(phase) if isinstance(multipliers, Mapping) else None
        else:
            effective_load = axes.get("load_multiplier", arrival.get("central_load_multiplier"))
        if not _finite_positive(effective_load):
            _issue(errors, "load_multiplier_invalid", "#/fixed/arrival", "selected load multiplier must be finite and positive")

    elif controlled_kind:
        if workload.get("dataset_id") != CONTROLLED_PREFIX_DATASET_ID:
            _issue(errors, "dataset_id_mismatch", "#/fixed/workload/dataset_id", f"expected {CONTROLLED_PREFIX_DATASET_ID!r}")
        if workload.get("sha256") != CONTROLLED_PREFIX_SHA256:
            _issue(errors, "input_sha256_mismatch", "#/fixed/workload/sha256", f"expected {CONTROLLED_PREFIX_SHA256}")
        if workload.get("artifact_schema") != CONTROLLED_PREFIX_SCHEMA:
            _issue(errors, "artifact_schema_mismatch", "#/fixed/workload/artifact_schema", f"expected {CONTROLLED_PREFIX_SCHEMA}")
        if workload.get("request_universe") != "synthetic_content_family_sessions" or workload.get("selection") != "deterministic_stratified_by_seed":
            _issue(errors, "controlled_sampling_contract_mismatch", "#/fixed/workload", "synthetic family-stratified selection required")
        sampling = workload.get("sampling")
        if not isinstance(sampling, Mapping) or sampling.get("unit") != "synthetic_session" or list(sampling.get("strata", [])) != ["content_family"] or not _positive_int(sampling.get("sessions_per_family")) or not _positive_int(sampling.get("expected_selected_sessions")) or sampling.get("complete_sessions_only") is not True:
            _issue(errors, "controlled_sampling_spec_invalid", "#/fixed/workload/sampling", "complete exact per-family synthetic sampling required")
        if workload.get("content_identity_audited") is not True or workload.get("reference_count_audited") is not True:
            _issue(errors, "controlled_identity_audit_missing", "#/fixed/workload", "content identity and reference count audits are mandatory")
        if workload.get("empirical_measurement") is not False:
            _issue(errors, "controlled_empirical_label_invalid", "#/fixed/workload/empirical_measurement", "controlled artifact must be labeled non-empirical")
        provenance = workload.get("controlled_parameter_provenance")
        if not isinstance(provenance, Mapping) or not all(isinstance(value, str) and value for value in provenance.values()):
            _issue(errors, "parameter_provenance_missing", "#/fixed/workload/controlled_parameter_provenance", "synthetic choices must have explicit non-measurement provenance")
        reuse = axes.get("reuse_level")
        if reuse not in (0.0, 0.25, 0.5, 0.75, 1.0):
            _issue(errors, "reuse_level_invalid", "#/axes/reuse_level", "quarter-step reuse level required")
        if arrival.get("mode") != "controlled_artifact_schedule" or arrival.get("timestamp_source") != "controlled_prefix_artifact.generated_relative_arrival_ns":
            _issue(errors, "controlled_arrival_contract_invalid", "#/fixed/arrival", "pinned controlled artifact schedule required")
        if arrival.get("session_order") != "round_index_then_session_id" or arrival.get("nonmonotonic_timestamp_policy") != "artifact_monotonicity_required" or arrival.get("load_scaling") != "none":
            _issue(errors, "controlled_arrival_contract_invalid", "#/fixed/arrival", "monotonic unscaled controlled schedule required")

    seed = axes.get("seed")
    if type(seed) is not int:
        _issue(errors, "explicit_seed_missing", "#/axes/seed", "integer seed is mandatory")
    if seed_contract.get("rng_algorithm") != "python_random_mt19937":
        _issue(errors, "rng_algorithm_mismatch", "#/fixed/seed/rng_algorithm", "python_random_mt19937 required")
    if not isinstance(seed_contract.get("namespace"), str) or not seed_contract.get("namespace"):
        _issue(errors, "seed_namespace_missing", "#/fixed/seed/namespace", "non-empty namespace required")
    if list(seed_contract.get("pairing_key", [])) != ["group", "seed", "request_universe"]:
        _issue(errors, "pairing_key_mismatch", "#/fixed/seed/pairing_key", "group/seed/request_universe required")

    selected_context = axes.get("context_tokens", context.get("central_tokens"))
    maximum_context = context.get("maximum_tokens")
    if not (
        type(selected_context) is int
        and type(maximum_context) is int
        and 2 <= selected_context <= maximum_context
    ):
        _issue(errors, "context_axis_invalid", "#/fixed/context", "positive selected context at or below maximum required")
    if context.get("truncate_over_limit") is not False:
        _issue(errors, "silent_truncation_forbidden", "#/fixed/context/truncate_over_limit", "must be false; only audited projection clipping is allowed")
    if context.get("output_tokens_mode") != "trace_or_preregistered":
        _issue(errors, "output_mode_mismatch", "#/fixed/context/output_tokens_mode", "trace_or_preregistered required")

    arrival_provenance = arrival.get("parameter_provenance")
    if (
        not isinstance(arrival_provenance, Mapping)
        or not isinstance(arrival_provenance.get("timestamp_source"), str)
        or not arrival_provenance.get("timestamp_source")
        or not isinstance(arrival_provenance.get("load_scaling"), str)
        or not arrival_provenance.get("load_scaling")
    ):
        _issue(
            errors,
            "parameter_provenance_missing",
            "#/fixed/arrival/parameter_provenance",
            "arrival source and modeled load transform provenance are required",
        )

    artifacts = lineage.get("input_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1 or not isinstance(artifacts[0], str):
        _issue(errors, "input_artifact_ambiguous", "#/fixed/artifact_lineage/input_artifacts", "exactly one request-source artifact path required")
    return WorkloadReadiness(tuple(errors))


def _schema_rows(connection: Any) -> dict[str, tuple[tuple[str, str], ...]]:
    rows = connection.execute(
        """
        select table_name, ordinal_position, column_name, data_type
        from information_schema.columns
        where table_schema = 'main'
          and table_name in ('rounds', 'timing_events', 'tool_calls', 'trace_source')
        order by table_name, ordinal_position
        """
    ).fetchall()
    result: dict[str, list[tuple[str, str]]] = {}
    for table, _ordinal, column, data_type in rows:
        result.setdefault(str(table), []).append((str(column), str(data_type)))
    return {table: tuple(columns) for table, columns in result.items()}


@lru_cache(maxsize=8)
def _audit_cached(
    path_text: str,
    expected_sha256: str,
    size_bytes: int,
    mtime_ns: int,
) -> TraceLabAudit:
    del size_bytes, mtime_ns
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise TraceLabDataError(
            "duckdb_unavailable",
            "TraceLab compilation requires the benchmark extra (duckdb)",
        ) from exc

    path = Path(path_text)
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise TraceLabDataError(
            "artifact_sha256_mismatch",
            f"TraceLab SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}",
        )
    connection = duckdb.connect(str(path), read_only=True)
    try:
        actual_schema = _schema_rows(connection)
        if actual_schema != _EXPECTED_SCHEMA:
            raise TraceLabDataError(
                "artifact_schema_mismatch",
                f"TraceLab schema drift: expected {_EXPECTED_SCHEMA!r}, got {actual_schema!r}",
            )
        counts = connection.execute(
            """
            select
              (select count(*) from rounds),
              (select count(*) from (select distinct provider, session_id from rounds)),
              (select count(*) from tool_calls),
              (select count(*) from timing_events)
            """
        ).fetchone()
        provider_counts = tuple(
            (str(provider), int(rounds), int(sessions))
            for provider, rounds, sessions in connection.execute(
                """
                select provider, count(*) rounds, count(distinct session_id) sessions
                from rounds group by provider order by provider
                """
            ).fetchall()
        )
        invariant_counts = connection.execute(
            """
            select
              sum(case when round_pk is null or provider is null or session_id is null
                        or round_index is null or round_id is null
                        or input_tokens_total is null or prefix_tokens is null
                        or newly_append_tokens is null or output_tokens is null
                       then 1 else 0 end),
              sum(case when round_pk < 0 or round_index < 0 or input_tokens_total < 0
                        or prefix_tokens < 0 or newly_append_tokens < 0
                        or output_tokens < 0 then 1 else 0 end),
              sum(case when prefix_tokens + newly_append_tokens <> input_tokens_total
                       then 1 else 0 end),
              (select count(*) from (
                 select round_pk from rounds group by round_pk having count(*) <> 1
               )),
              (select count(*) from (
                 select provider, session_id, round_index from rounds
                 group by all having count(*) <> 1
               )),
              (select count(*) from (
                 select r.round_pk
                 from rounds r left join timing_events t using (round_pk)
                 group by r.round_pk
                 having count(t.timestamp) filter (where t.event_index = 1) <> 1
               ))
            from rounds
            """
        ).fetchone()
        if any(int(value or 0) for value in invariant_counts):
            raise TraceLabDataError(
                "artifact_row_invariant_failed",
                f"TraceLab required-row invariant counts are {tuple(invariant_counts)!r}",
            )
        nonmonotonic = int(
            connection.execute(
                """
                with arrivals as (
                  select r.round_pk, r.provider, r.session_id, r.round_index,
                         min(t.timestamp) filter (where t.event_index = 1) as ts
                  from rounds r left join timing_events t using (round_pk)
                  group by all
                ), ordered as (
                  select *, lag(ts) over (
                    partition by provider, session_id order by round_index, round_pk
                  ) as previous_ts
                  from arrivals
                )
                select count(*) from ordered where ts < previous_ts
                """
            ).fetchone()[0]
        )
    finally:
        connection.close()
    schema_sha256 = _json_hash(
        {table: list(columns) for table, columns in sorted(_EXPECTED_SCHEMA.items())}
    )
    return TraceLabAudit(
        path=str(path),
        sha256=actual_sha256,
        schema_sha256=schema_sha256,
        round_count=int(counts[0]),
        provider_session_count=int(counts[1]),
        tool_call_count=int(counts[2]),
        timing_event_count=int(counts[3]),
        provider_counts=provider_counts,
        nonmonotonic_event1_transitions=nonmonotonic,
    )


def audit_tracelab(
    path: Path = DEFAULT_TRACELAB_PATH,
    *,
    expected_sha256: str = TRACELAB_SHA256,
) -> TraceLabAudit:
    """Verify the pinned bytes, exact schema, and request-critical row invariants."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise TraceLabDataError("artifact_missing", f"TraceLab artifact is missing: {resolved}")
    stat = resolved.stat()
    return _audit_cached(
        str(resolved), str(expected_sha256), int(stat.st_size), int(stat.st_mtime_ns)
    )


@lru_cache(maxsize=8)
def _audit_anomaly_sessions_cached(
    path_text: str,
    expected_sha256: str,
    size_bytes: int,
    mtime_ns: int,
) -> tuple[tuple[str, str, int], ...]:
    del size_bytes, mtime_ns
    # The byte hash is checked before this helper is called; keeping it in the
    # cache key prevents reuse across a differently pinned artifact.
    del expected_sha256
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise TraceLabDataError("duckdb_unavailable", "duckdb is required") from exc
    connection = duckdb.connect(path_text, read_only=True)
    try:
        rows = connection.execute(
            """
            with arrivals as (
              select r.round_pk, r.provider, r.session_id, r.round_index,
                     min(t.timestamp) filter (where t.event_index = 1) as ts
              from rounds r left join timing_events t using (round_pk)
              group by all
            ), ordered as (
              select *, lag(ts) over (
                partition by provider, session_id order by round_index, round_pk
              ) as previous_ts
              from arrivals
            )
            select provider, session_id, count(*) as transitions
            from ordered where ts < previous_ts
            group by provider, session_id
            order by provider, session_id
            """
        ).fetchall()
    finally:
        connection.close()
    return tuple((str(row[0]), str(row[1]), int(row[2])) for row in rows)


def audit_tracelab_anomaly_sessions(
    path: Path = DEFAULT_TRACELAB_PATH,
    *,
    expected_sha256: str = TRACELAB_SHA256,
) -> tuple[tuple[str, str, int], ...]:
    """Return the exact sessions excluded by the frozen monotonicity rule."""

    resolved = Path(path).resolve()
    audit_tracelab(resolved, expected_sha256=expected_sha256)
    stat = resolved.stat()
    return _audit_anomaly_sessions_cached(
        str(resolved), str(expected_sha256), int(stat.st_size), int(stat.st_mtime_ns)
    )


@dataclass(frozen=True)
class _RawRound:
    provider: str
    session_id: str
    round_index: int
    round_pk: int
    round_id: str
    input_tokens: int
    prefix_tokens: int
    append_tokens: int
    output_tokens: int
    timestamp: datetime


def _session_bin(round_count: int, lower_bounds: Sequence[int]) -> str | None:
    if round_count < lower_bounds[0]:
        return None
    for lower, upper in zip(lower_bounds, lower_bounds[1:]):
        if lower <= round_count < upper:
            return f"{lower}-{upper - 1}"
    return f"{lower_bounds[-1]}+"


def _select_sessions(
    connection: Any,
    *,
    sampling: Mapping[str, Any],
    namespace: str,
    group: str,
    seed: int,
    excluded_provider_sessions: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[tuple[str, str, int, str], ...]:
    lower_bounds = tuple(int(value) for value in sampling["session_round_count_bins"])
    per_stratum = int(sampling["sessions_per_stratum"])
    strata: dict[tuple[str, str], list[tuple[str, str, int, str]]] = {}
    for provider, session_id, count in connection.execute(
        "select provider, session_id, count(*) n from rounds group by 1, 2"
    ).fetchall():
        if (str(provider), str(session_id)) in excluded_provider_sessions:
            continue
        bin_name = _session_bin(int(count), lower_bounds)
        if bin_name is None:
            continue
        row = (str(provider), str(session_id), int(count), bin_name)
        strata.setdefault((str(provider), bin_name), []).append(row)
    rng = random.Random(f"{namespace}:{group}:{seed}:provider_sessions")
    selected: list[tuple[str, str, int, str]] = []
    for key in sorted(strata):
        candidates = sorted(strata[key])
        if len(candidates) < per_stratum:
            raise TraceLabDataError(
                "insufficient_sampling_stratum",
                f"TraceLab stratum {key!r} has {len(candidates)} sessions, needs {per_stratum}",
            )
        selected.extend(rng.sample(candidates, per_stratum))
    selected.sort(key=lambda row: (row[0], row[1]))
    expected = int(sampling["expected_selected_sessions"])
    if len(selected) != expected:
        raise TraceLabDataError(
            "selected_session_count_mismatch",
            f"selected {len(selected)} complete sessions, config requires exactly {expected}",
        )
    return tuple(selected)


def _fetch_rounds(
    connection: Any,
    sessions: Sequence[tuple[str, str, int, str]],
) -> tuple[_RawRound, ...]:
    clauses: list[str] = []
    parameters: list[str] = []
    for provider, session_id, _count, _bin_name in sessions:
        clauses.append("(r.provider = ? and r.session_id = ?)")
        parameters.extend((provider, session_id))
    if not clauses:
        raise TraceLabDataError("empty_request_universe", "no provider sessions selected")
    rows = connection.execute(
        f"""
        select r.provider, r.session_id, r.round_index, r.round_pk, r.round_id,
               r.input_tokens_total, r.prefix_tokens, r.newly_append_tokens,
               r.output_tokens,
               min(t.timestamp) filter (where t.event_index = 1) as arrival_at
        from rounds r left join timing_events t using (round_pk)
        where {' or '.join(clauses)}
        group by all
        order by r.provider, r.session_id, r.round_index, r.round_pk
        """,
        parameters,
    ).fetchall()
    result = tuple(
        _RawRound(
            provider=str(row[0]),
            session_id=str(row[1]),
            round_index=int(row[2]),
            round_pk=int(row[3]),
            round_id=str(row[4]),
            input_tokens=int(row[5]),
            prefix_tokens=int(row[6]),
            append_tokens=int(row[7]),
            output_tokens=int(row[8]),
            timestamp=row[9],
        )
        for row in rows
    )
    expected_rounds = sum(row[2] for row in sessions)
    if len(result) != expected_rounds:
        raise TraceLabDataError(
            "complete_session_violation",
            f"selected sessions declare {expected_rounds} rounds but query returned {len(result)}",
        )
    previous: dict[tuple[str, str], _RawRound] = {}
    for row in result:
        key = (row.provider, row.session_id)
        prior = previous.get(key)
        if prior is not None and row.timestamp < prior.timestamp:
            raise TraceLabDataError(
                "selected_session_timestamp_nonmonotonic",
                (
                    f"{key!r} round {row.round_index} timestamp {row.timestamp!s} "
                    f"precedes round {prior.round_index} timestamp {prior.timestamp!s}; "
                    "fail_selected_session forbids silent reordering or clamping"
                ),
            )
        previous[key] = row
    return result


def _round_half_up(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("round-half-up requires a non-negative numerator and positive denominator")
    return (2 * numerator + denominator) // (2 * denominator)


def _project_round(
    row: _RawRound,
    *,
    context_tokens: int,
    projection: Mapping[str, Any],
) -> tuple[int, int, int, bool, bool]:
    source_median = int(projection["source_median_input_tokens"])
    target_fraction = Fraction(str(projection["target_median_fraction"]))
    scale_numerator = context_tokens * target_fraction.numerator
    scale_denominator = source_median * target_fraction.denominator
    scaled_prompt = max(
        _round_half_up(row.input_tokens * scale_numerator, scale_denominator), 1
    )
    prompt = min(scaled_prompt, context_tokens - 1)
    if row.input_tokens > 0:
        prefix = _round_half_up(row.prefix_tokens * prompt, row.input_tokens)
    else:
        prefix = 0
    prefix = min(max(prefix, 0), prompt)
    output_cap = int(projection["output_cap_tokens"])
    output = min(max(row.output_tokens, 0), output_cap, context_tokens - prompt)
    return (
        prompt,
        prefix,
        output,
        prompt < scaled_prompt,
        output < row.output_tokens,
    )


def _artifact_path(
    fixed: Mapping[str, Any], experiment_root: Path
) -> Path:
    raw = fixed["artifact_lineage"]["input_artifacts"][0]
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        experiment_root = Path(experiment_root).resolve()
        if candidate.parts[:2] == ("AegisKV", "experiments"):
            candidate = experiment_root.parents[1] / candidate
        else:
            candidate = experiment_root / candidate
    return candidate.resolve()


def _load_multiplier(
    arrival: Mapping[str, Any], axes: Mapping[str, Any]
) -> tuple[float, str | None]:
    if arrival["mode"] == "preregistered_low_high_recovery_trace":
        phase = str(axes["load_phase"])
        multiplier = float(arrival["phase_load_multipliers"][phase])
        return multiplier, phase
    return float(axes.get("load_multiplier", arrival["central_load_multiplier"])), None


def compile_tracelab_workload(
    resolved_request: Mapping[str, Any],
    axes: Mapping[str, Any],
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
) -> CompiledWorkload:
    """Compile one request cell, or fail before emitting a partial universe."""

    readiness = request_layer_readiness(resolved_request, axes)
    if not readiness.ready:
        raise WorkloadNotReadyError(readiness.errors)
    fixed = resolved_request["fixed"]
    workload = fixed["workload"]
    if workload["kind"] != "tracelab_projected_session":
        raise TraceLabDataError(
            "wrong_workload_compiler",
            "compile_tracelab_workload only accepts tracelab_projected_session; "
            "use compile_workload for kind-based dispatch",
        )
    arrival = fixed["arrival"]
    context = fixed["context"]
    seed_contract = fixed["seed"]
    group = str(resolved_request["group"])
    seed = int(axes["seed"])
    context_tokens = int(axes.get("context_tokens", context["central_tokens"]))
    load_multiplier, load_phase = _load_multiplier(arrival, axes)
    if not math.isfinite(load_multiplier) or load_multiplier <= 0.0:
        raise WorkloadNotReadyError(
            (
                ReadinessError(
                    "load_multiplier_invalid",
                    "#/axes/load_multiplier",
                    "load multiplier must be finite and positive",
                ),
            )
        )

    artifact = _artifact_path(fixed, Path(experiment_root))
    audit = audit_tracelab(
        artifact, expected_sha256=str(workload["sha256"])
    )
    configured_exclusions = tuple(
        (
            str(row["provider"]),
            str(row["session_id"]),
            int(row["nonmonotonic_transitions"]),
        )
        for row in workload["anomaly_exclusions"]["excluded_provider_sessions"]
    )
    observed_exclusions = audit_tracelab_anomaly_sessions(
        artifact, expected_sha256=str(workload["sha256"])
    )
    if configured_exclusions != observed_exclusions:
        raise TraceLabDataError(
            "anomaly_exclusion_set_mismatch",
            "frozen anomaly exclusions do not exactly match the pinned TraceLab artifact: "
            f"configured={configured_exclusions!r}, observed={observed_exclusions!r}",
        )
    excluded_pairs = frozenset((row[0], row[1]) for row in observed_exclusions)
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - checked by audit first
        raise TraceLabDataError("duckdb_unavailable", "duckdb is required") from exc
    connection = duckdb.connect(str(artifact), read_only=True)
    try:
        sessions = _select_sessions(
            connection,
            sampling=workload["sampling"],
            namespace=str(seed_contract["namespace"]),
            group=group,
            seed=seed,
            excluded_provider_sessions=excluded_pairs,
        )
        raw_rounds = _fetch_rounds(connection, sessions)
    finally:
        connection.close()

    pairing_values = {
        "group": group,
        "seed": seed,
        "request_universe": workload["request_universe"],
    }
    universe_payload = {
        "schema_version": COMPILER_SCHEMA,
        "dataset_sha256": audit.sha256,
        "pairing_key": [
            [name, pairing_values[name]] for name in seed_contract["pairing_key"]
        ],
        "sessions": [
            {
                "provider": provider,
                "session_id": session_id,
                "round_count": count,
                "round_count_bin": bin_name,
            }
            for provider, session_id, count, bin_name in sessions
        ],
        "rounds": [
            [
                row.provider,
                row.session_id,
                row.round_index,
                row.round_pk,
                row.round_id,
            ]
            for row in raw_rounds
        ],
    }
    universe_hash = _json_hash(universe_payload)

    origin = min(row.timestamp for row in raw_rounds)
    projected: list[tuple[_RawRound, int, int, int, int, bool, bool]] = []
    for row in raw_rounds:
        raw_relative_ns = (
            (row.timestamp - origin).days * 86_400_000_000_000
            + (row.timestamp - origin).seconds * 1_000_000_000
            + (row.timestamp - origin).microseconds * 1_000
        )
        load_fraction = Fraction(str(load_multiplier))
        arrival_ns = _round_half_up(
            raw_relative_ns * load_fraction.denominator,
            load_fraction.numerator,
        )
        prompt, prefix, output, input_clipped, output_clipped = _project_round(
            row,
            context_tokens=context_tokens,
            projection=workload["context_projection"],
        )
        projected.append(
            (
                row,
                arrival_ns,
                prompt,
                prefix,
                output,
                input_clipped,
                output_clipped,
            )
        )

    records: list[RequestRecord] = []
    for row, arrival_ns, prompt, prefix, output, _input_clipped, _output_clipped in projected:
        workflow_identity = f"{audit.sha256}:{row.provider}:{row.session_id}"
        workflow_digest = hashlib.sha256(workflow_identity.encode("utf-8")).hexdigest()
        suffix_identity = f"{workflow_identity}:{row.round_pk}:{row.round_id}:suffix"
        suffix_digest = hashlib.sha256(suffix_identity.encode("utf-8")).hexdigest()
        rid = "tl-" + hashlib.sha256(
            f"{workflow_identity}:{row.round_pk}:{row.round_id}".encode("utf-8")
        ).hexdigest()[:24]
        prefix_key = f"tl-prefix-{workflow_digest}"
        records.append(
            RequestRecord(
                rid=rid,
                workflow=f"tl-workflow-{workflow_digest[:24]}",
                session=row.session_id,
                provider=row.provider,
                round_index=row.round_index,
                arrival_ns=arrival_ns,
                input_tokens=prompt,
                output_tokens=output,
                cached_prefix_tokens=prefix,
                prefix_key=prefix_key,
                input_token_identity=TokenIdentity(
                    prefix_stream_sha256=workflow_digest,
                    prefix_tokens=prefix,
                    suffix_stream_sha256=suffix_digest,
                    suffix_tokens=prompt - prefix,
                ),
                source_round_pk=row.round_pk,
                source_round_id=row.round_id,
            )
        )
    records.sort(
        key=lambda record: (
            record.arrival_ns,
            record.workflow,
            record.round_index,
            record.source_round_pk,
        )
    )
    record_dicts = [record.as_dict() for record in records]
    metadata: dict[str, object] = {
        "group": group,
        "seed": seed,
        "pairing_key": pairing_values,
        "selected_sessions": len(sessions),
        "selected_requests": len(records),
        "complete_sessions_only": True,
        "context_tokens": context_tokens,
        "load_multiplier": load_multiplier,
        "load_phase": load_phase,
        "arrival_transform": "observed_event1_relative_ns_divided_by_load_multiplier",
        "output_token_visibility": "preregistered_driver_only_not_policy_input",
        "projection_algorithm": workload["context_projection"]["algorithm"],
        "input_projection_clipped_records": sum(item[5] for item in projected),
        "output_projection_clipped_records": sum(item[6] for item in projected),
        "rounds_silently_truncated": 0,
        "sessions_silently_truncated": 0,
        "prefix_identity_scope": "within_provider_session_only",
        "anomaly_policy": "complete_observed_set_excluded_before_seeded_sampling",
        "excluded_anomalous_sessions": len(observed_exclusions),
        "excluded_nonmonotonic_transitions": sum(row[2] for row in observed_exclusions),
    }
    return CompiledWorkload(
        records=tuple(records),
        request_universe_hash=universe_hash,
        compiled_requests_hash=_json_hash(record_dicts),
        dataset_audit=audit,
        metadata=MappingProxyType(metadata),
    )


def _compile_controlled_prefix_workload(
    resolved_request: Mapping[str, Any],
    axes: Mapping[str, Any],
    *,
    experiment_root: Path,
) -> CompiledWorkload:
    fixed = resolved_request["fixed"]
    workload = fixed["workload"]
    seed_contract = fixed["seed"]
    group = str(resolved_request["group"])
    seed = int(axes["seed"])
    reuse_level = Fraction(str(axes["reuse_level"]))
    reuse_numerator = reuse_level.numerator * (4 // reuse_level.denominator)
    if reuse_level.denominator not in (1, 2, 4) or not 0 <= reuse_numerator <= 4:
        raise TraceLabDataError(
            "reuse_level_invalid", "controlled reuse must be an exact quarter step"
        )

    artifact_path = _artifact_path(fixed, Path(experiment_root))
    try:
        artifact, artifact_sha256 = audit_controlled_prefix_artifact(
            artifact_path,
            expected_sha256=str(workload["sha256"]),
        )
    except (OSError, ValueError) as exc:
        raise TraceLabDataError("controlled_artifact_invalid", str(exc)) from exc

    shape = artifact["workload_shape"]
    families = artifact["content_families"]
    family_count = int(shape["content_family_count"])
    sessions_per_family = int(shape["sessions_per_family"])
    selected_per_family = int(shape["selected_sessions_per_family"])
    configured_sampling = workload["sampling"]
    if (
        len(families) != family_count
        or int(configured_sampling["sessions_per_family"]) != selected_per_family
        or int(configured_sampling["expected_selected_sessions"])
        != family_count * selected_per_family
    ):
        raise TraceLabDataError(
            "controlled_artifact_sampling_mismatch",
            "request config sample cardinality does not match the pinned artifact",
        )

    rng = random.Random(
        f"{seed_contract['namespace']}:{group}:{seed}:"
        f"{workload['request_universe']}"
    )
    selected_sessions: list[tuple[int, int, str, str]] = []
    for family_index, family in enumerate(families):
        candidates = list(range(sessions_per_family))
        for session_index in sorted(rng.sample(candidates, selected_per_family)):
            selected_sessions.append(
                (
                    family_index,
                    session_index,
                    str(family["family_id"]),
                    str(family["prefix_stream_sha256"]),
                )
            )

    rounds_per_session = int(shape["rounds_per_session"])
    input_tokens = int(shape["input_tokens_per_request"])
    controllable_prefix = int(shape["controllable_prefix_tokens"])
    output_tokens = int(shape["output_tokens_per_request"])
    context_tokens = int(fixed["context"]["central_tokens"])
    if input_tokens + output_tokens > context_tokens:
        raise TraceLabDataError(
            "controlled_context_overflow",
            "controlled request exceeds the configured context window",
        )
    shared_prefix_tokens = _round_half_up(
        controllable_prefix * reuse_level.numerator,
        reuse_level.denominator,
    )

    universe_payload = {
        "schema_version": COMPILER_SCHEMA,
        "dataset_sha256": artifact_sha256,
        "pairing_key": [
            [name, {
                "group": group,
                "seed": seed,
                "request_universe": workload["request_universe"],
            }[name]]
            for name in seed_contract["pairing_key"]
        ],
        "sessions": [
            {
                "family_index": family_index,
                "family_id": family_id,
                "session_index": session_index,
                "round_count": rounds_per_session,
            }
            for family_index, session_index, family_id, _digest in selected_sessions
        ],
    }
    universe_hash = _json_hash(universe_payload)
    raw_records: list[RequestRecord] = []
    session_stagger_ns = int(shape["session_stagger_ns"])
    round_spacing_ns = int(shape["round_spacing_ns"])
    raw_arrivals: list[int] = []
    recipe = artifact["session_identity_recipe"]
    for family_index, session_index, family_id, prefix_digest in selected_sessions:
        session_id = str(recipe["session_id_format"]).format(
            family_index=family_index,
            session_index=session_index,
        )
        workflow_digest = hashlib.sha256(
            f"{artifact_sha256}:{session_id}".encode("ascii")
        ).hexdigest()
        for round_index in range(rounds_per_session):
            raw_arrival = (
                round_index * round_spacing_ns
                + (family_index * sessions_per_family + session_index)
                * session_stagger_ns
            )
            raw_arrivals.append(raw_arrival)
            suffix_preimage = str(recipe["suffix_stream_preimage_format"]).format(
                family_index=family_index,
                session_index=session_index,
                round_index=round_index,
                reuse_numerator=reuse_numerator,
            )
            suffix_digest = hashlib.sha256(
                suffix_preimage.encode("ascii")
            ).hexdigest()
            source_round_pk = (
                (family_index * sessions_per_family + session_index)
                * rounds_per_session
                + round_index
            )
            rid = "cp-" + hashlib.sha256(
                f"{artifact_sha256}:{session_id}:{round_index}".encode("ascii")
            ).hexdigest()[:24]
            raw_records.append(
                RequestRecord(
                    rid=rid,
                    workflow=f"cp-workflow-{workflow_digest[:24]}",
                    session=session_id,
                    provider="synthetic-controlled",
                    round_index=round_index,
                    arrival_ns=raw_arrival,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_prefix_tokens=shared_prefix_tokens,
                    prefix_key=(
                        f"cp-prefix-{family_id}-{prefix_digest}"
                        if shared_prefix_tokens
                        else f"cp-no-shared-prefix-{rid}"
                    ),
                    input_token_identity=TokenIdentity(
                        prefix_stream_sha256=prefix_digest,
                        prefix_tokens=shared_prefix_tokens,
                        suffix_stream_sha256=suffix_digest,
                        suffix_tokens=input_tokens - shared_prefix_tokens,
                    ),
                    source_round_pk=source_round_pk,
                    source_round_id=f"{session_id}-round-{round_index:02d}",
                )
            )
    origin = min(raw_arrivals)
    records = tuple(
        sorted(
            (
                RequestRecord(
                    **{
                        **record.__dict__,
                        "arrival_ns": record.arrival_ns - origin,
                    }
                )
                for record in raw_records
            ),
            key=lambda record: (
                record.arrival_ns,
                record.workflow,
                record.round_index,
                record.source_round_pk,
            ),
        )
    )

    if shared_prefix_tokens:
        reference_counts: dict[str, int] = {}
        for record in records:
            reference_counts[record.prefix_key] = (
                reference_counts.get(record.prefix_key, 0) + 1
            )
        expected_references = selected_per_family * rounds_per_session
        if set(reference_counts.values()) != {expected_references}:
            raise TraceLabDataError(
                "controlled_reference_count_mismatch",
                f"expected {expected_references} references per shared family prefix",
            )
    else:
        expected_references = 0

    audit = ControlledPrefixAudit(
        path=str(artifact_path),
        sha256=artifact_sha256,
        schema_version=str(artifact["schema_version"]),
        content_family_count=family_count,
        generated_session_count=family_count * sessions_per_family,
        selected_sessions_per_family=selected_per_family,
        rounds_per_session=rounds_per_session,
        empirical_measurement=bool(artifact["generator"]["empirical_measurement"]),
    )
    metadata: dict[str, object] = {
        "group": group,
        "seed": seed,
        "pairing_key": {
            "group": group,
            "seed": seed,
            "request_universe": workload["request_universe"],
        },
        "selected_sessions": len(selected_sessions),
        "selected_requests": len(records),
        "complete_sessions_only": True,
        "context_tokens": context_tokens,
        "reuse_level": float(reuse_level),
        "shared_prefix_tokens": shared_prefix_tokens,
        "references_per_shared_family_prefix": expected_references,
        "dataset_kind": "deterministic_synthetic_controlled",
        "empirical_measurement": False,
        "trace_lab_source": False,
        "arrival_transform": "pinned_controlled_artifact_schedule",
        "rounds_silently_truncated": 0,
        "sessions_silently_truncated": 0,
        "prefix_identity_scope": "cross_session_within_content_family",
    }
    return CompiledWorkload(
        records=records,
        request_universe_hash=universe_hash,
        compiled_requests_hash=_json_hash(
            [record.as_dict() for record in records]
        ),
        dataset_audit=audit,
        metadata=MappingProxyType(metadata),
    )


def compile_workload(
    resolved_request: Mapping[str, Any],
    axes: Mapping[str, Any],
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
) -> CompiledWorkload:
    """Compile any executable P1-P10 request kind with fail-closed dispatch."""

    readiness = request_layer_readiness(resolved_request, axes)
    if not readiness.ready:
        raise WorkloadNotReadyError(readiness.errors)
    kind = resolved_request["fixed"]["workload"]["kind"]
    if kind == "tracelab_projected_session":
        return compile_tracelab_workload(
            resolved_request,
            axes,
            experiment_root=experiment_root,
        )
    if kind == "controlled_prefix_reuse":
        return _compile_controlled_prefix_workload(
            resolved_request,
            axes,
            experiment_root=Path(experiment_root),
        )
    raise TraceLabDataError("unsupported_workload_kind", f"unsupported kind {kind!r}")


def deterministic_token_ids(
    identity: TokenIdentity,
    *,
    vocab_size: int,
) -> Iterable[int]:
    """Materialise the identity recipe without an implicit length cap.

    Callers that need an array must make that memory cost explicit themselves;
    this iterator never drops tokens.  Prefix streams are shared by session and
    suffix streams are round-specific.
    """

    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    for stream, count in (
        (identity.prefix_stream_sha256, identity.prefix_tokens),
        (identity.suffix_stream_sha256, identity.suffix_tokens),
    ):
        for index in range(count):
            digest = hashlib.sha256(f"{stream}:{index}".encode("ascii")).digest()
            yield int.from_bytes(digest[:4], "big") % vocab_size


__all__ = [
    "COMPILER_SCHEMA",
    "CONTROLLED_PREFIX_DATASET_ID",
    "CONTROLLED_PREFIX_SCHEMA",
    "CONTROLLED_PREFIX_SHA256",
    "DEFAULT_TRACELAB_PATH",
    "TRACELAB_DATASET_ID",
    "TRACELAB_SHA256",
    "CompiledWorkload",
    "ControlledPrefixAudit",
    "ReadinessError",
    "RequestRecord",
    "TokenIdentity",
    "TraceLabAudit",
    "TraceLabDataError",
    "WorkloadNotReadyError",
    "WorkloadReadiness",
    "audit_tracelab",
    "audit_tracelab_anomaly_sessions",
    "compile_workload",
    "compile_tracelab_workload",
    "deterministic_token_ids",
    "request_layer_readiness",
]
