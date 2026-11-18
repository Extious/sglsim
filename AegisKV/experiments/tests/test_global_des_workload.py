from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from itertools import product

import pytest

duckdb = pytest.importorskip("duckdb")

from AegisKV.experiments.src.global_des_config import load_suite
from AegisKV.experiments.src.global_des_workload import (
    TRACELAB_SHA256,
    TraceLabDataError,
    WorkloadNotReadyError,
    audit_tracelab,
    audit_tracelab_anomaly_sessions,
    compile_workload,
    compile_tracelab_workload,
    deterministic_token_ids,
    request_layer_readiness,
)
from AegisKV.experiments.src.controlled_prefix_artifact import (
    CONTROLLED_PREFIX_SHA256,
    audit_controlled_prefix_artifact,
)


def _ready_request_layer(group: str) -> dict[str, object]:
    suite = load_suite()
    return deepcopy(suite.group_map[group].request.resolved)


@pytest.fixture(scope="module")
def p1_compilation():
    layer = _ready_request_layer("P1")
    return compile_tracelab_workload(
        layer,
        {
            "seed": 2026082900,
            # Complete cells contain non-request axes.  They must be inert here.
            "policy": "aegiskv",
            "failure_scenario": "worker_1",
            "model": "qwen3-8b",
        },
    )


def test_pinned_artifact_hash_schema_and_observed_cardinality() -> None:
    audit = audit_tracelab()

    assert audit.sha256 == TRACELAB_SHA256
    assert audit.round_count == 665_453
    assert audit.provider_session_count == 8_058
    assert audit.tool_call_count == 743_819
    assert audit.timing_event_count == 2_688_829
    assert audit.provider_counts == (
        ("claude", 305_445, 5_319),
        ("codex", 360_008, 2_739),
    )
    # This is why the selected-session anomaly policy must be explicit.
    assert audit.nonmonotonic_event1_transitions == 23


def test_frozen_v3_request_files_are_ready_for_every_p1_p10_request_cell() -> None:
    suite = load_suite()
    for group_id in (f"P{index}" for index in range(1, 11)):
        group = suite.group_map[group_id]
        names = tuple(group.request.axes)
        for values in product(*(group.request.axes[name] for name in names)):
            axes = dict(zip(names, values))
            assert request_layer_readiness(group.request.resolved, axes).as_dict() == {
                "ready": True,
                "errors": [],
            }


def test_missing_sampling_still_fails_closed_with_structured_error() -> None:
    layer = _ready_request_layer("P1")
    del layer["fixed"]["workload"]["sampling"]
    readiness = request_layer_readiness(layer, {"seed": 2026082900})

    assert readiness.ready is False
    assert any(error.code == "sampling_spec_missing" for error in readiness.errors)
    with pytest.raises(WorkloadNotReadyError) as caught:
        compile_workload(layer, {"seed": 2026082900})
    assert caught.value.as_dict() == readiness.as_dict()


def test_compilation_emits_immutable_json_compatible_complete_session_records(
    p1_compilation,
) -> None:
    result = p1_compilation

    assert result.metadata["selected_sessions"] == 8
    assert result.metadata["selected_requests"] == len(result.records)
    assert result.metadata["rounds_silently_truncated"] == 0
    assert result.metadata["sessions_silently_truncated"] == 0
    assert result.metadata["output_token_visibility"] == (
        "preregistered_driver_only_not_policy_input"
    )
    assert len(result.request_universe_hash) == 64
    assert len(result.compiled_requests_hash) == 64
    json.dumps(result.as_dict(), allow_nan=False, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        result.records[0].rid = "mutated"
    with pytest.raises(TypeError):
        result.metadata["selected_sessions"] = 0

    by_workflow: dict[str, list[object]] = {}
    for record in result.records:
        by_workflow.setdefault(record.workflow, []).append(record)
        assert 0 <= record.cached_prefix_tokens <= record.input_tokens
        assert record.input_token_identity.prefix_tokens == record.cached_prefix_tokens
        assert (
            record.input_token_identity.prefix_tokens
            + record.input_token_identity.suffix_tokens
            == record.input_tokens
        )
    assert len(by_workflow) == 8
    for rows in by_workflow.values():
        ordered = sorted(rows, key=lambda row: (row.round_index, row.source_round_pk))
        assert [row.arrival_ns for row in ordered] == sorted(
            row.arrival_ns for row in ordered
        )
        assert len({row.prefix_key for row in rows}) == 1


def test_same_pairing_key_is_reproducible_and_cell_non_request_axes_are_inert(
    p1_compilation,
) -> None:
    layer = _ready_request_layer("P1")
    replay = compile_tracelab_workload(
        layer,
        {
            "seed": 2026082900,
            "policy": "recompute",
            "failure_scenario": "node_1",
            "model": "qwen3-32b",
            "node_count": 8,
        },
    )

    assert replay.request_universe_hash == p1_compilation.request_universe_hash
    assert replay.compiled_requests_hash == p1_compilation.compiled_requests_hash
    assert replay.records == p1_compilation.records


def test_load_multiplier_only_scales_arrivals_and_never_resamples() -> None:
    layer = _ready_request_layer("P2")
    low = compile_tracelab_workload(
        layer,
        {"seed": 2026082900, "load_multiplier": 0.5, "policy": "aegiskv"},
    )
    high = compile_tracelab_workload(
        layer,
        {"seed": 2026082900, "load_multiplier": 2.0, "policy": "recompute"},
    )

    assert low.request_universe_hash == high.request_universe_hash
    assert [row.rid for row in low.records] == [row.rid for row in high.records]
    assert [row.input_tokens for row in low.records] == [
        row.input_tokens for row in high.records
    ]
    low_by_rid = {row.rid: row for row in low.records}
    high_by_rid = {row.rid: row for row in high.records}
    positive = next(rid for rid, row in low_by_rid.items() if row.arrival_ns > 0)
    assert low_by_rid[positive].arrival_ns == 4 * high_by_rid[positive].arrival_ns


def test_context_projection_changes_sizes_but_not_request_universe() -> None:
    layer = _ready_request_layer("P3")
    short = compile_tracelab_workload(
        layer, {"seed": 2026082900, "context_tokens": 8192}
    )
    long = compile_tracelab_workload(
        layer, {"seed": 2026082900, "context_tokens": 40960}
    )

    assert short.request_universe_hash == long.request_universe_hash
    assert [row.rid for row in short.records] == [row.rid for row in long.records]
    assert all(row.input_tokens + row.output_tokens <= 8192 for row in short.records)
    assert all(row.input_tokens + row.output_tokens <= 40960 for row in long.records)
    assert any(
        lhs.input_tokens != rhs.input_tokens
        for lhs, rhs in zip(short.records, long.records)
    )
    assert short.metadata["input_projection_clipped_records"] > 0


def test_token_identity_reproduces_shared_session_prefix(p1_compilation) -> None:
    candidates = [
        record for record in p1_compilation.records if record.cached_prefix_tokens >= 8
    ]
    first = candidates[0]
    second = next(
        record
        for record in candidates[1:]
        if record.workflow == first.workflow
    )

    assert first.prefix_key == second.prefix_key
    first_ids = list(deterministic_token_ids(first.input_token_identity, vocab_size=151936))
    second_ids = list(
        deterministic_token_ids(second.input_token_identity, vocab_size=151936)
    )
    common = min(first.cached_prefix_tokens, second.cached_prefix_tokens)
    assert first_ids[:common] == second_ids[:common]
    assert len(first_ids) == first.input_tokens
    assert len(second_ids) == second.input_tokens


def test_compiled_workload_exports_real_sglsim_tokenized_requests(
    p1_compilation,
) -> None:
    first_record = p1_compilation.records[0]
    request = next(p1_compilation.iter_tokenized_requests())
    simulation = request.sampling_params.custom_params["simulation"]

    assert request.rid == first_record.rid
    assert len(request.input_ids) == first_record.input_tokens
    assert request.input_token_count == first_record.input_tokens
    assert request.sampling_params.max_new_tokens == first_record.output_tokens
    assert request.original_prompt_len == first_record.input_tokens
    assert simulation["arrival_ns"] == first_record.arrival_ns
    assert simulation["prefix_key"] == first_record.prefix_key
    assert simulation["request_universe_hash"] == p1_compilation.request_universe_hash


def test_missing_axis_value_is_a_structured_readiness_error() -> None:
    layer = _ready_request_layer("P2")
    readiness = request_layer_readiness(layer, {"seed": 2026082900})

    assert readiness.ready is False
    assert any(
        error.code == "axis_value_missing"
        and error.config_path == "#/axes/load_multiplier"
        for error in readiness.errors
    )


def test_p5_phase_window_cannot_silently_truncate_the_universe() -> None:
    layer = _ready_request_layer("P5")
    compiled = {
        phase: compile_workload(
            layer, {"seed": 2026082900, "load_phase": phase}
        )
        for phase in ("low", "high", "recovery")
    }

    assert len({item.request_universe_hash for item in compiled.values()}) == 1
    assert len({tuple(row.rid for row in item.records) for item in compiled.values()}) == 1
    assert all(item.metadata["rounds_silently_truncated"] == 0 for item in compiled.values())
    assert all(item.metadata["sessions_silently_truncated"] == 0 for item in compiled.values())
    by_rid = {
        phase: {row.rid: row for row in item.records}
        for phase, item in compiled.items()
    }
    positive = next(rid for rid, row in by_rid["low"].items() if row.arrival_ns > 0)
    # Arrival time is inversely proportional to the frozen 0.5/2.0 multipliers.
    assert by_rid["low"][positive].arrival_ns == 4 * by_rid["high"][positive].arrival_ns


def test_complete_observed_anomaly_set_is_excluded_before_sampling() -> None:
    layer = _ready_request_layer("P7")
    observed = audit_tracelab_anomaly_sessions()
    configured = tuple(
        (row["provider"], row["session_id"], row["nonmonotonic_transitions"])
        for row in layer["fixed"]["workload"]["anomaly_exclusions"][
            "excluded_provider_sessions"
        ]
    )

    assert configured == observed
    assert len(observed) == 21
    assert sum(row[2] for row in observed) == 23
    excluded = {(provider, session) for provider, session, _count in observed}
    # This frozen seed selected an anomalous session before the exclusion contract.
    compiled = compile_workload(layer, {"seed": 2026082909})
    assert not {(row.provider, row.session) for row in compiled.records} & excluded
    assert compiled.metadata["excluded_anomalous_sessions"] == 21
    assert compiled.metadata["excluded_nonmonotonic_transitions"] == 23


def test_anomaly_exclusion_set_cannot_be_silently_incomplete() -> None:
    layer = _ready_request_layer("P8")
    exclusions = layer["fixed"]["workload"]["anomaly_exclusions"]
    exclusions["excluded_provider_sessions"].pop()
    exclusions["observed_affected_sessions"] = 20
    exclusions["observed_nonmonotonic_transitions"] = 22

    assert request_layer_readiness(layer, {"seed": 2026082901}).ready is True
    with pytest.raises(TraceLabDataError) as caught:
        compile_workload(layer, {"seed": 2026082901})
    assert caught.value.code == "anomaly_exclusion_set_mismatch"


def test_p6_controlled_prefix_artifact_is_independent_reproducible_and_nonempirical() -> None:
    artifact, digest = audit_controlled_prefix_artifact()
    assert digest == CONTROLLED_PREFIX_SHA256
    assert artifact["generator"]["empirical_measurement"] is False
    assert "not TraceLab" in artifact["generator"]["provenance"]
    assert "not a measured trace" in artifact["generator"]["provenance"]

    layer = _ready_request_layer("P6")
    zero = compile_workload(layer, {"seed": 2026082900, "reuse_level": 0.0})
    full = compile_workload(layer, {"seed": 2026082900, "reuse_level": 1.0})

    assert zero.request_universe_hash == full.request_universe_hash
    assert [row.rid for row in zero.records] == [row.rid for row in full.records]
    assert [row.arrival_ns for row in zero.records] == [row.arrival_ns for row in full.records]
    assert len(full.records) == 8 * 16
    assert {row.cached_prefix_tokens for row in zero.records} == {0}
    assert {row.cached_prefix_tokens for row in full.records} == {11904}
    assert full.metadata["references_per_shared_family_prefix"] == 32
    assert full.metadata["empirical_measurement"] is False
    assert full.metadata["trace_lab_source"] is False
    assert full.dataset_audit.empirical_measurement is False

    by_prefix: dict[str, set[str]] = {}
    for row in full.records:
        by_prefix.setdefault(row.prefix_key, set()).add(row.session)
    assert len(by_prefix) == 4
    assert {len(sessions) for sessions in by_prefix.values()} == {2}


def test_every_frozen_seed_compiles_without_timestamp_clamping() -> None:
    suite = load_suite()
    for group_id in (f"P{index}" for index in range(1, 11)):
        group = suite.group_map[group_id]
        for seed in group.request.axes["seed"]:
            axes = {
                name: (seed if name == "seed" else values[0])
                for name, values in group.request.axes.items()
            }
            compiled = compile_workload(group.request.resolved, axes)
            assert compiled.records
            assert [row.arrival_ns for row in compiled.records] == sorted(
                row.arrival_ns for row in compiled.records
            )
