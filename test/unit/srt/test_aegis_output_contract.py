from __future__ import annotations

from copy import deepcopy

import pytest

from sglsim.srt.simulation.aegis_full_design import (
    AegisFullDesignConfig,
    AegisFullDesignRuntime,
    CalibratedTransferPath,
    SealedChunk,
)
from sglsim.srt.simulation.aegis_independent_tiers import (
    AegisIndependentTierController,
    ChunkSpec,
    IndependentTierConfig,
    ProtectionTier,
    RecoveryTiming,
    RequestRecoveryProfile,
)
from sglsim.srt.simulation.aegis_output_contract import (
    COMPONENT_COST_MODEL_SCHEMA,
    COMPONENT_NAMES,
    build_aegis_output_contract,
)


LINEAGE = "c" * 64
HARDWARE = "A100-1node-TP1-PP1-systemDP2"
PROFILE_SHA = "a" * 64
GATE_SHA = "b" * 64
PREFLIGHT_SHA = "d" * 64
SOURCE_SHA = "e" * 64
CONFIG_SHA = "f" * 64
CONFIG_VERIFICATION_SHA = "1" * 64
HIT_ARTIFACT_SHA = "2" * 64


def _runtime(
    *, hbm_copy_failure_probability: float = 0.0
) -> AegisFullDesignRuntime:
    config = AegisFullDesignConfig(
        hbm_capacity_bytes=1_000,
        dram_capacity_bytes=1_000,
        kv_bytes_per_token=1,
        hbm_copy=CalibratedTransferPath(
            2.0, 100.0, "unit-hbm-profile", first_transfer_warmup_ms=0.5
        ),
        dram_copy=CalibratedTransferPath(
            3.0, 50.0, "unit-dram-profile", first_transfer_warmup_ms=1.0
        ),
        recompute=RecoveryTiming(10.0, 0.10),
        hbm_load=RecoveryTiming(1.0, 0.01),
        dram_load=RecoveryTiming(2.0, 0.02),
        prefix_chunk_tokens=100,
        decode_chunk_tokens=100,
        event_log_limit=2,
        hbm_copy_failure_probability=hbm_copy_failure_probability,
    )
    runtime = AegisFullDesignRuntime(config)
    runtime.register_request(
        request_id="request-0",
        source_worker="source-0",
        source_failure_domain="source-domain-0",
        hbm_target="target-0",
        hbm_failure_domain="target-domain-0",
        route_version=1,
        prompt_tokens=100,
        output_tokens=0,
        slack_ms=0.0,
        model_revision="model-revision",
        kv_abi="kv-abi",
        parallel_layout="tp1-pp1-dp2",
        time_ms=0.0,
    )
    decision = runtime.seal_chunk(
        "request-0",
        SealedChunk(0, 0, 100, "prefix"),
        time_ms=0.0,
    )
    assert decision.hbm.admitted and decision.dram.admitted
    runtime.flush()
    recovery = runtime.estimate_source_failure(
        "request-0",
        activated_target="target-0",
        source_token_end=100,
        remaining_slack_ms=0.0,
        time_ms=6.0,
    )
    assert recovery.action == (
        "DRAM" if hbm_copy_failure_probability == 1.0 else "PEER_HBM"
    )
    assert runtime.finish_request("request-0", time_ms=10.0)["completed"] is True
    runtime.validate_invariants()
    return runtime


def _hit_evidence(
    *,
    kind: str = "profile_derived",
    peer_hbm: int = 1,
    dram: int = 0,
    hardware: str = HARDWARE,
    eligible: bool = True,
    components: bool = True,
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "evidence_kind": kind,
        "lineage_id": LINEAGE,
        "hardware_label": hardware,
        "preflight_sha256": PREFLIGHT_SHA,
        "source_manifest_sha256": SOURCE_SHA,
        "config_manifest_sha256": CONFIG_SHA,
        "config_verification_sha256": CONFIG_VERIFICATION_SHA,
        "physical_profile_sha256": PROFILE_SHA,
        "recovery_hit_artifact_sha256": HIT_ARTIFACT_SHA,
        "mechanism_gate_sha256": GATE_SHA,
        "mechanism_gate_status": "PASS",
        "same_lineage": True,
        "simulator_rate_default_eligible": eligible,
        "component_breakdown_declared": components,
        "runtime_counter_source": (
            "recovery_estimate_action_counts"
            if kind == "profile_derived"
            else None
        ),
        "hit_counts": {"peer_hbm": peer_hbm, "dram": dram},
    }
    if kind == "physical_observed":
        evidence["observed_successful_protection"] = {
            "source_to_peer_hbm": {
                "tier": "HBM",
                "copy_kind": "source_d2d",
                "successful_copies": 7,
                "successful_bytes": 700,
            },
            "source_to_store_dram": {
                "tier": "DRAM",
                "copy_kind": "source_d2h",
                "successful_copies": 3,
                "successful_bytes": 300,
            },
        }
    return evidence


def _component_profiles() -> dict[str, object]:
    def profile(
        components: dict[str, tuple[float, float, float | None]],
    ) -> dict[str, object]:
        assert set(components) == set(COMPONENT_NAMES)
        return {
            "lineage_id": LINEAGE,
            "hardware_label": HARDWARE,
            "profile_sha256": PROFILE_SHA,
            "cost_model_schema": COMPONENT_COST_MODEL_SCHEMA,
            "components": {
                name: {
                    "per_transfer_setup_ms": setup,
                    "first_transfer_warmup_ms": warmup,
                    "steady_state_bytes_per_ms": rate,
                }
                for name, (setup, warmup, rate) in components.items()
            },
        }

    return {
        "source_to_peer_hbm": profile(
            {
                "target_allocation": (0.5, 0.0, None),
                "source_pack": (0.5, 0.0, None),
                "engine_transfer": (1.0, 0.5, 100.0),
                "target_finish": (0.0, 0.0, None),
            }
        ),
        "source_to_store_dram": profile(
            {
                "target_allocation": (1.0, 0.0, None),
                "source_pack": (1.0, 0.0, None),
                "engine_transfer": (1.0, 1.0, 50.0),
                "target_finish": (0.0, 0.0, None),
            }
        ),
    }


def _effect_pairs() -> list[dict[str, object]]:
    return [
        {
            "pair_id": "pair-0",
            "baseline_whole_workload_wct_ms": 100.0,
            "aegis_whole_workload_wct_ms": 110.0,
            "baseline_failure_local_ms": 50.0,
            "aegis_failure_local_ms": 40.0,
            "included": True,
            "performance_direction_used_for_inclusion": False,
        }
    ]


def _kwargs() -> dict[str, object]:
    return {
        "policy": "aegiskv",
        "runtime_snapshots": {"0": _runtime().snapshot()},
        "effect_pairs": _effect_pairs(),
        "recovery_hit_evidence": _hit_evidence(),
        "expected_lineage_id": LINEAGE,
        "expected_hardware_label": HARDWARE,
        "component_profiles": _component_profiles(),
    }


def test_runtime_snapshot_separates_successful_protection_and_pool_churn() -> None:
    snapshot = _runtime().snapshot()

    hbm = snapshot["protection_paths"]["source_to_peer_hbm"]
    dram = snapshot["protection_paths"]["source_to_store_dram"]
    assert (hbm["successful_copies"], hbm["successful_bytes"]) == (1, 100)
    assert (dram["successful_copies"], dram["successful_bytes"]) == (1, 100)
    assert hbm["healthy_period_first_transfer_warmup_count"] == 1
    assert dram["healthy_period_first_transfer_warmup_count"] == 1
    assert hbm["healthy_period_transfer_service_ms"] == pytest.approx(3.5)
    assert dram["healthy_period_transfer_service_ms"] == pytest.approx(6.0)
    assert snapshot["recovery_estimate_action_counts"] == {"PEER_HBM": 1}
    for tier in ("HBM", "DRAM"):
        churn = snapshot["pool_churn"][tier]
        assert churn["admission_attempt_count"] == 1
        assert churn["admission_success_count"] == 1
        assert churn["eviction_count"] == 0
        assert churn["removal_reason_counts"] == {"request_completed": 1}


def test_failed_transport_is_cost_but_not_successful_protection() -> None:
    snapshot = _runtime(hbm_copy_failure_probability=1.0).snapshot()

    hbm = snapshot["protection_paths"]["source_to_peer_hbm"]
    assert hbm["attempted_copies"] == 1
    assert hbm["attempted_bytes"] == 100
    assert hbm["failed_copies"] == 1
    assert hbm["successful_copies"] == 0
    assert hbm["successful_bytes"] == 0
    assert hbm["healthy_period_transfer_service_ms"] == pytest.approx(3.5)
    churn = snapshot["pool_churn"]["HBM"]
    assert churn["eviction_count"] == 0
    assert churn["removal_reason_counts"] == {
        "copy_completion_validation_failed": 1
    }


def test_first_transfer_warmup_is_charged_once_per_runtime_path() -> None:
    runtime = _runtime()
    runtime.register_request(
        request_id="request-1",
        source_worker="source-1",
        source_failure_domain="source-domain-1",
        hbm_target="target-0",
        hbm_failure_domain="target-domain-0",
        route_version=1,
        prompt_tokens=100,
        output_tokens=0,
        slack_ms=0.0,
        model_revision="model-revision",
        kv_abi="kv-abi",
        parallel_layout="tp1-pp1-dp2",
        time_ms=10.0,
    )
    runtime.seal_chunk(
        "request-1", SealedChunk(0, 0, 100, "prefix"), time_ms=10.0
    )
    runtime.flush()
    runtime.finish_request("request-1", time_ms=17.0)

    snapshot = runtime.snapshot()
    hbm = snapshot["protection_paths"]["source_to_peer_hbm"]
    dram = snapshot["protection_paths"]["source_to_store_dram"]
    assert hbm["healthy_period_attempted_copies"] == 2
    assert dram["healthy_period_attempted_copies"] == 2
    assert hbm["healthy_period_first_transfer_warmup_count"] == 1
    assert dram["healthy_period_first_transfer_warmup_count"] == 1
    assert hbm["healthy_period_transfer_service_ms"] == pytest.approx(6.5)
    assert dram["healthy_period_transfer_service_ms"] == pytest.approx(11.0)


def test_pool_churn_counts_policy_eviction_separately_from_teardown() -> None:
    controller = AegisIndependentTierController(
        IndependentTierConfig(
            hbm_capacity_bytes=100,
            dram_capacity_bytes=100,
            recompute=RecoveryTiming(10.0, 0.10),
            hbm_load=RecoveryTiming(1.0, 0.01),
            dram_load=RecoveryTiming(2.0, 0.02),
        )
    )
    # Both candidates have positive recovery gain.  The first retains less
    # visible harm after causal Slack, so the second is the safe higher-value
    # tail that should evict it from each independently managed tier.
    for request_id, slack_ms in (("low", 10.0), ("high", 0.0)):
        controller.register_request(
            RequestRecoveryProfile(
                request_id=request_id,
                source_worker="source-0",
                source_failure_domain="source-domain-0",
                hbm_target="target-0",
                hbm_failure_domain="target-domain-0",
                route_version=1,
                source_token_end=100,
                slack_ms=slack_ms,
                model_revision="model",
                kv_abi="abi",
                parallel_layout="tp1-pp1-dp2",
            )
        )
        spec = ChunkSpec(
            request_id=request_id,
            ordinal=0,
            token_start=0,
            token_end=100,
            byte_size=100,
            route_version=1,
            source_worker="source-0",
            hbm_target="target-0",
            model_revision="model",
            kv_abi="abi",
            parallel_layout="tp1-pp1-dp2",
        )
        decision = controller.admit_chunk(spec)
        assert decision.hbm.admitted and decision.dram.admitted
        for tier in (ProtectionTier.HBM, ProtectionTier.DRAM):
            controller.complete_copy(
                tier,
                request_id,
                0,
                controller.completion_proof(spec, tier),
            )

    controller.complete_request("high")
    snapshot = controller.snapshot()
    for tier in ("HBM", "DRAM"):
        churn = snapshot["pools"][tier]["churn"]
        assert churn["admission_attempt_count"] == 2
        assert churn["admission_success_count"] == 2
        assert churn["eviction_count"] == 1
        assert churn["evicted_bytes"] == 100
        assert churn["eviction_reason_counts"] == {"slack_landlord_victim": 1}
        assert churn["removal_count"] == 2
        assert churn["removal_reason_counts"] == {
            "request_completed": 1,
            "slack_landlord_victim": 1,
        }
    controller.validate_invariants()


def test_joint_effect_contract_blocks_general_claim_when_total_wct_regresses() -> None:
    report = build_aegis_output_contract(**_kwargs())

    assert report["status"] == "PASS"
    assert report["protection_tiers"] == {
        "HBM": {
            "path_ids": ["source_to_peer_hbm"],
            "successful_copies": 1,
            "successful_bytes": 100,
        },
        "DRAM": {
            "path_ids": ["source_to_store_dram"],
            "successful_copies": 1,
            "successful_bytes": 100,
        },
    }
    assert report["copy_to_hit_amplification"]["source_to_peer_hbm"] == {
        "recovery_hits": 1,
        "successful_protection_copies": 1,
        "successful_protection_bytes": 100,
        "protection_population_source": (
            "metadata_runtime_successful_protection"
        ),
        "amplification_defined": True,
        "successful_copies_per_hit": 1.0,
        "successful_bytes_per_hit": 100.0,
    }
    gate = report["joint_effect_contract"]["claim_gate"]
    assert gate["failure_local_benefit"] is True
    assert gate["whole_workload_regression"] is True
    assert gate["general_advantage_supported"] is False
    assert gate["performance_claim_authorized"] is False
    assert gate["classification"] == (
        "failure_local_benefit_with_whole_workload_regression"
    )
    assert report["joint_effect_contract"]["direction_counts"] == {
        "failure_local": {"benefit": 1},
        "whole_workload": {"regression": 1},
    }
    assert report["profile_component_breakdown_complete"] is True
    assert report["remaining_profile_dependent_fields"] == []
    assert report["metadata_only"] is True
    assert report["cuda_or_nccl_executed"] is False
    assert report["physical_kv_bytes_moved_by_sglsim"] is False
    assert report["authorization"] == {
        "replay_authorized": False,
        "figure_generation_authorized": False,
        "performance_claim_authorized": False,
        "formal_ready": False,
        "paper_ready": False,
    }


def test_missing_profile_components_are_explicit_not_silently_zero() -> None:
    kwargs = _kwargs()
    kwargs["recovery_hit_evidence"] = _hit_evidence(components=False)
    kwargs["component_profiles"] = None

    report = build_aegis_output_contract(**kwargs)

    assert report["profile_component_breakdown_complete"] is False
    assert len(report["remaining_profile_dependent_fields"]) == 24
    assert all(
        row["available"] is False
        for row in report["transfer_service_components"].values()
    )


def test_zero_physical_hits_leave_amplification_undefined() -> None:
    kwargs = _kwargs()
    kwargs["recovery_hit_evidence"] = _hit_evidence(
        kind="physical_observed", peer_hbm=0, dram=0
    )

    report = build_aegis_output_contract(**kwargs)

    hbm = report["copy_to_hit_amplification"]["source_to_peer_hbm"]
    assert hbm["successful_protection_copies"] == 7
    assert hbm["protection_population_source"] == (
        "same_artifact_physical_successful_protection"
    )
    assert hbm["amplification_defined"] is False
    assert hbm["successful_copies_per_hit"] is None


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda kwargs: kwargs["effect_pairs"][0].pop(
                "baseline_whole_workload_wct_ms"
            ),
            "missing",
        ),
        (
            lambda kwargs: kwargs["effect_pairs"][0].__setitem__(
                "performance_direction_used_for_inclusion", True
            ),
            "direction blind",
        ),
        (
            lambda kwargs: kwargs["runtime_snapshots"]["0"][
                "protection_paths"
            ]["source_to_peer_hbm"].__setitem__("successful_copies", 2),
            "copies do not reconcile",
        ),
        (
            lambda kwargs: kwargs["runtime_snapshots"]["0"][
                "protection_paths"
            ]["source_to_peer_hbm"].__setitem__("successful_bytes", 0),
            "copy counts and bytes disagree",
        ),
        (
            lambda kwargs: kwargs["runtime_snapshots"]["0"][
                "protection_paths"
            ]["source_to_peer_hbm"].__setitem__(
                "healthy_period_first_transfer_warmup_count", 2
            ),
            "warm-up counts do not reconcile",
        ),
        (
            lambda kwargs: kwargs["runtime_snapshots"]["0"]["pool_churn"][
                "HBM"
            ].__setitem__("admission_attempt_count", 2),
            "admissions do not reconcile",
        ),
        (
            lambda kwargs: kwargs["recovery_hit_evidence"].__setitem__(
                "hit_counts", {"peer_hbm": 2, "dram": 0}
            ),
            "do not match runtime decisions",
        ),
        (
            lambda kwargs: kwargs["recovery_hit_evidence"].__setitem__(
                "runtime_counter_source", "recovery_execution_action_counts"
            ),
            "do not match runtime decisions",
        ),
        (
            lambda kwargs: kwargs["component_profiles"][
                "source_to_peer_hbm"
            ]["components"].pop("target_finish"),
            "partial",
        ),
        (
            lambda kwargs: kwargs["component_profiles"][
                "source_to_peer_hbm"
            ]["components"]["engine_transfer"].__setitem__(
                "steady_state_bytes_per_ms", 50.0
            ),
            "does not reconcile",
        ),
    ],
)
def test_output_contract_fails_closed_on_incomplete_or_tampered_metrics(
    mutate, match: str
) -> None:
    kwargs = deepcopy(_kwargs())
    mutate(kwargs)

    with pytest.raises(ValueError, match=match):
        build_aegis_output_contract(**kwargs)


def test_rtx4090_diagnostic_cannot_supply_a100_or_v100_hit_evidence() -> None:
    kwargs = _kwargs()
    kwargs["recovery_hit_evidence"] = _hit_evidence(
        hardware="RTX4090-2node-PP2-systemDP2",
        eligible=False,
    )

    with pytest.raises(ValueError, match="hardware lineage"):
        build_aegis_output_contract(**kwargs)


def test_physical_hits_require_same_artifact_copy_population() -> None:
    kwargs = _kwargs()
    evidence = _hit_evidence(kind="physical_observed")
    evidence.pop("observed_successful_protection")
    kwargs["recovery_hit_evidence"] = evidence

    with pytest.raises(ValueError, match="physical protection population"):
        build_aegis_output_contract(**kwargs)
