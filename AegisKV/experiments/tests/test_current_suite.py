from __future__ import annotations

from AegisKV.experiments.src.config import (
    CONFIG_ROOT,
    expected_cell_count,
    load_suite,
)
from AegisKV.experiments.src.failure import build_failure_schedule
from AegisKV.experiments.src.simulation.cell import (
    build_inputs,
    build_shadow_result,
)


def test_current_suite_has_three_layers_and_expected_paper_matrix() -> None:
    loaded = load_suite(CONFIG_ROOT)

    assert [group.group for group in loaded.groups] == [
        *[f"S{i}" for i in range(1, 10)],
        "E1",
    ]
    assert expected_cell_count(loaded, "smoke") == 76
    assert expected_cell_count(loaded, "paper") == 613
    minimum = loaded.suite["profiles"]["paper"]["minimum_distinct_seeds"]
    for group in loaded.groups:
        assert group.source_paths[0].name == "request.json"
        assert group.source_paths[1].name == "failure.json"
        assert group.source_paths[2].name == "deployment.json"
        assert len(set(group.axes["seed"])) >= int(minimum[group.group])


def test_parameter_provenance_is_machine_checked_and_exposes_blockers() -> None:
    loaded = load_suite(CONFIG_ROOT)
    provenance = loaded.parameter_provenance
    parameters = provenance["parameters"]

    assert provenance["paper_readiness_blockers"] == [
        "failure.failover_delay",
        "deployment.service_time",
    ]
    assert all(
        parameters[name]["evidence_class"] == "modeled_uncalibrated"
        for name in provenance["paper_readiness_blockers"]
    )
    assert {
        "request.trace_artifact",
        "request.agentx_external_validity",
        "request.causal_slack_prediction",
        "failure.field_hazard",
        "deployment.network_line_rate",
        "deployment.effective_transfer_bandwidth",
        "deployment.service_time",
    }.issubset(parameters)


def test_failure_hazard_and_bandwidth_units_are_not_conflated() -> None:
    loaded = load_suite(CONFIG_ROOT)
    common = loaded.groups[0]
    cluster = common.deployment["cluster"]
    hazard = common.failure["hazard_weighting"]

    assert hazard["central_mttf_gpu_hours"] == 400 * 24
    assert hazard["cross_check_gpu_equivalent_hours"] == 1024 * 7.9
    assert hazard["evidence_scope"] == (
        "training_cluster_proxy_not_inference_field_measurement"
    )
    assert cluster["network_line_rate_Gbit_per_s"] == 200
    assert cluster["effective_cross_node_bandwidth_GB_per_s"] == 20
    assert cluster["effective_cross_node_bandwidth_GB_per_s"] <= (
        cluster["network_line_rate_Gbit_per_s"] / 8
    )


def test_chunking_matches_current_design_with_labeled_s9_override() -> None:
    loaded = load_suite(CONFIG_ROOT)

    for group in loaded.groups:
        protection = group.deployment["protection"]
        expected_prefix = 1024 if group.group == "S9" else 4096
        assert protection["prefix_chunk_tokens"] == expected_prefix
        assert protection["decode_chunk_tokens"] == 256


def test_s4_fifo_adversary_has_distinct_slack_for_every_paper_seed() -> None:
    loaded = load_suite(CONFIG_ROOT)
    group = next(group for group in loaded.groups if group.group == "S4")
    profile = loaded.suite["profiles"]["paper"]

    for seed in group.axes["seed"]:
        axes = {
            "failure_scenario": "worker_3",
            "model": "qwen3-8b",
            "policy": "aegiskv_no_slack",
            "seed": seed,
        }
        deployment, workload, requests = build_inputs(
            group, axes, suite_profile=profile
        )
        shadow = build_shadow_result(deployment, workload, requests)
        schedule = build_failure_schedule(
            shadow, group.failure, axes, deployment.workers
        )
        ordered_slack = list(schedule.metadata["failed_worker_slack_ms"].values())

        assert len(schedule.failures) == 3
        assert len(schedule.slack_refresh_request_ids) == 3
        assert ordered_slack == sorted(ordered_slack, reverse=True)
        assert ordered_slack[0] > ordered_slack[-1]
