from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from AegisKV.experiments.src.global_des_config import config_hash, load_suite
from AegisKV.experiments.src.global_des_suite_metrics import evidence_claims
from AegisKV.experiments.src.global_des_suite_runner import (
    CellSpec,
    DEFAULT_P0_SUMMARY,
    ExecutionOptions,
    PreparedWorkload,
    ResumeValidationError,
    SuiteReadinessError,
    SuiteExecutionError,
    execute_prepared_cell,
    prepare_workload,
    select_cells,
    validate_p0_summary,
    validate_prepared_workload_readiness,
    validate_resumable_cell,
)
from AegisKV.experiments.src.global_des_suite_runner import (
    _build_active_exposure_plan,
    _clone_failover_replay,
    _materialize_requests,
    _record_token_ids,
    _validate_active_failure_exposure,
    _validate_physical_recovery_admission_gate,
    _validate_terminal_request_closure,
)
from sglsim.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS
from sglsim.srt.simulation.aegis_residual_dag import (
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
)
from sglsim.srt.simulation.global_des import PhysicalGpuId
from sglsim.srt.simulation.offline_resource_bridge import (
    NodeDramSpec,
    OfflinePhysicalTopology,
    PhysicalGpuSpec,
)


def _topology() -> OfflinePhysicalTopology:
    return OfflinePhysicalTopology(
        gpus=tuple(
            PhysicalGpuSpec(
                PhysicalGpuId(
                    node_id=f"node{rank}",
                    local_gpu_id=0,
                    dp_rank=rank,
                    pp_rank=0,
                    tp_rank=0,
                ),
                hbm_capacity_bytes=1 << 30,
                weights_bytes=1 << 20,
                runtime_bytes=1 << 20,
            )
            for rank in range(2)
        ),
        nodes=(
            NodeDramSpec("node0", 1 << 30, runtime_bytes=1 << 20),
            NodeDramSpec("node1", 1 << 30, runtime_bytes=1 << 20),
        ),
    )


def _runtime(*, aegis: bool) -> dict:
    return {
        "model_path": "mock-model",
        "scheduler": {
            "dp_size": 2,
            "tp_size": 1,
            "pp_size": 1,
            "load_balance_method": "round_robin",
            "max_prefill_tokens": 256,
            "chunked_prefill_size": 256,
            "max_running_requests": 8,
        },
        "cache": {
            "page_size": 16,
            "l1_capacity_pages": 4096,
            "aegis_total_l1_capacity_pages": 4096,
            "l2_capacity_pages": 4096,
        },
        "communication": {"kv_bytes_per_token": 128},
        "aegis_l1_l2": {
            "enabled": aegis,
            "committed_copy_protocol": aegis,
            "design_semantics": DESIGN_SEMANTICS,
            "chunk_tokens": 16,
            "prefix_chunk_tokens": 16,
            "decode_chunk_tokens": 1,
            "host_alignment_tokens": 1,
            "initial_alpha": 1.0,
            "initial_beta": 1.0,
            "topology": {
                "dp_rank_to_node": [0, 1],
                "pp_stages_per_node": 1,
            },
        },
    }


def _workload(cell: CellSpec) -> PreparedWorkload:
    rid = f"unit-{cell.group}"
    record = {
        "rid": rid,
        "workflow": f"workflow-{cell.group}",
        "session": "unit",
        "provider": "unit",
        "round_index": 0,
        "arrival_ns": 0,
        "input_tokens": 16,
        "output_tokens": 0 if cell.group == "P12" else 1,
        "cached_prefix_tokens": 0,
        "prefix_key": f"prefix-{cell.group}",
        "source_round_pk": 0,
        "source_round_id": "unit",
    }
    if cell.group == "P4":
        graph = VisibleResidualDag(
            workflow_id=str(record["workflow"]),
            graph_version="unit-visible-v1",
            observed_at_ms=0.0,
            nodes=(
                ResidualDagNode(rid, "agent", 0.0, active=True, agent_id="unit"),
                ResidualDagNode(
                    f"{rid}-peer", "agent", 1.0, active=True, agent_id="peer"
                ),
            ),
            edges=(),
            next_join_subgraph_complete=True,
        )
        record["simulation"] = build_residual_dag_slack_tickets(graph)[
            rid
        ].to_simulation_fields()
    rows = (record,)
    return PreparedWorkload(
        records=rows,
        request_universe_hash=f"universe-{cell.group}",
        compiled_requests_hash=f"requests-{cell.group}",
        metadata={"unit_contract": True},
    )


def _options() -> ExecutionOptions:
    return ExecutionOptions(
        model_paths={"qwen3-8b": "mock-model", "qwen3-32b": "mock-model"},
        model_revisions={"qwen3-8b": "unit-revision", "qwen3-32b": "unit-revision"},
        smoke=True,
        max_requests_per_cell=1,
        max_events=500,
    )


def test_smoke_selection_is_one_stable_cell_for_p1_p4_p11_p12() -> None:
    suite = load_suite()
    first = select_cells(suite, smoke=True)
    second = select_cells(suite, smoke=True)
    assert [(cell.group, cell.ordinal) for cell in first] == [
        ("P1", 0),
        ("P4", 0),
        ("P11", 0),
        ("P12", 0),
    ]
    assert [cell.cell_id for cell in first] == [cell.cell_id for cell in second]


def test_policy_pair_uses_identical_pre_execution_failure_plan_and_hash() -> None:
    cells = select_cells(load_suite(), groups=["P2"])
    pair = [
        cell
        for cell in cells
        if cell.axes["seed"] == 2026082900
        and cell.axes["load_multiplier"] == 0.5
        and cell.axes["failure_scenario"] == "worker_1"
    ]
    assert {cell.axes["policy"] for cell in pair} == {"recompute", "aegiskv"}
    workload = PreparedWorkload(
        records=(),
        request_universe_hash="a" * 64,
        compiled_requests_hash="b" * 64,
        metadata={},
    )
    rows = [
        {
            "rid": "anchor",
            "arrival_ns": 0,
            "input_tokens": 4096,
            "output_tokens": 64,
            "planned_dp_rank": 1,
        },
        {
            "rid": "later",
            "arrival_ns": 50_000_000_000,
            "input_tokens": 4096,
            "output_tokens": 64,
            "planned_dp_rank": 0,
        },
    ]
    plans = [
        _build_active_exposure_plan(cell, workload, rows, _topology())
        for cell in pair
    ]

    assert plans[0] == plans[1]
    assert plans[0]["anchor_request_id"] == "anchor"
    assert plans[0]["planned_targets"] == [
        {"failure_scope": "worker", "dp_rank": 1}
    ]
    assert plans[0]["shadow_plan_sha256"] == config_hash(
        {
            key: value
            for key, value in plans[0].items()
            if key != "shadow_plan_sha256"
        }
    )


def test_active_exposure_audit_rejects_failure_that_hits_no_original() -> None:
    plan = {
        "anchor_request_id": "anchor",
        "anchor_source_dp_rank": 1,
        "anchor_source_node_id": "node1",
        "active_offset_ns": 1_000_000,
        "planned_targets": [{"failure_scope": "worker", "dp_rank": 1}],
        "shadow_plan_sha256": "c" * 64,
    }
    with pytest.raises(SuiteExecutionError, match="no active original"):
        _validate_active_failure_exposure(
            plan=plan,
            anchor_active_at_ns=0,
            failures=[
                {
                    "failure_scope": "worker",
                    "dp_rank": 1,
                    "injected_offset_s": 0.001,
                    "shadow_plan_sha256": "c" * 64,
                    "affected_request_ids": [],
                }
            ],
            request_rows=[{"rid": "anchor", "failover_replay": False}],
        )


def test_p0_gate_is_contract_only_and_fail_closed(tmp_path) -> None:
    good = tmp_path / "p0.json"
    good.write_text(
        json.dumps(
            {
                "schema": "aegiskv.global-des-p0-summary.v1",
                "status": "PASS",
                "claims": {
                    "configured_p0_contract_gate_passed": True,
                    "contract_execution_only": True,
                    "full_scheduler_integration": False,
                },
                "cells": [{"status": "PASS"} for _ in range(12)],
            }
        ),
        encoding="utf-8",
    )
    assert validate_p0_summary(good)["contract_only"] is True
    bad = tmp_path / "bad.json"
    bad.write_text(good.read_text(encoding="utf-8").replace('"PASS"', '"FAIL"', 1))
    with pytest.raises(SuiteReadinessError, match="p0_gate_failed"):
        validate_p0_summary(bad)


def test_default_p0_gate_binds_fresh_target_rotation_lineage() -> None:
    assert (
        DEFAULT_P0_SUMMARY.parent.parent.name
        == "20260829T121349Z_global_des_p0_contract_v5"
    )
    summary = json.loads(DEFAULT_P0_SUMMARY.read_text(encoding="utf-8"))
    assert summary["claims"]["controller_bridge_target_rotation_closed_loop"] is True
    validated = validate_p0_summary(DEFAULT_P0_SUMMARY)
    assert validated["status"] == "PASS"
    assert validated["contract_only"] is True
    assert validated["source_full_scheduler_integration"] is False


def test_current_p11_selects_one_seed_stratum_partition_and_accepts_explicit_zero() -> None:
    # P11 ordinal 1 is seed 2026082900 / small / AegisKV.
    cell = select_cells(load_suite(), selectors=["P11:1"])[0]
    workload = prepare_workload(
        cell,
        experiment_root=Path("AegisKV/experiments").resolve(),
    )

    assert workload.metadata["seed"] == 2026082900
    assert workload.metadata["dag_stratum"] == "small"
    assert workload.metadata["atomic_workflow_replay"] is True
    assert workload.metadata["smoke_request_truncation_permitted"] is False
    assert len(workload.records) == 57
    assert {row["seed"] for row in workload.records} == {2026082900}
    assert {row["dag_stratum"] for row in workload.records} == {"small"}
    assert len({row["workflow"] for row in workload.records}) == 1
    assert any(
        row["simulation"]["aegis_slack_usable"] is False
        for row in workload.records
    )
    # Complete, explicit Algorithm-1 fail-close-zero tickets are accepted;
    # protection itself remains enabled in materialization.
    validate_prepared_workload_readiness(cell, workload)
    requests, rows = _materialize_requests(
        cell, workload, dp_size=2, smoke_limit=1
    )
    assert len(requests) == len(rows) == 57
    assert all(row["simulation"]["backup_kv"] is True for row in rows)


def test_weka_block_tokens_preserve_intra_workflow_and_isolate_workflows() -> None:
    common = {
        "schema": "aegiskv.weka-hash-block-token-identity.v1",
        "block_size": 2,
        "block_hash_ids": [7, 8],
        "token_count": 4,
        "fallback_stream_sha256": "f" * 64,
    }
    left = {
        "rid": "left",
        "input_tokens": 4,
        "input_token_identity": {
            **common,
            "block_namespace_sha256": "a" * 64,
        },
    }
    same_workflow = {
        "rid": "same",
        "input_tokens": 4,
        "input_token_identity": {
            **common,
            "block_namespace_sha256": "a" * 64,
        },
    }
    other_workflow = {
        "rid": "other",
        "input_tokens": 4,
        "input_token_identity": {
            **common,
            "block_namespace_sha256": "b" * 64,
        },
    }

    assert _record_token_ids(left) == _record_token_ids(same_workflow)
    assert _record_token_ids(left) != _record_token_ids(other_workflow)


def test_atomic_p11_smoke_claim_is_not_labeled_as_request_prefix(
    tmp_path,
) -> None:
    cell = select_cells(load_suite(), groups=["P11"], smoke=True)[0]
    workload = replace(
        _workload(cell),
        metadata={
            "atomic_workflow_replay": True,
            "smoke_request_truncation_permitted": False,
        },
    )
    execution = execute_prepared_cell(
        cell,
        workload,
        tmp_path / cell.cell_id,
        options=_options(),
        p0_lineage={"status": "PASS", "contract_only": True},
        runtime_config_override=_runtime(aegis=False),
        physical_topology_override=_topology(),
    )
    audit = json.loads((execution.path / "audit.json").read_text(encoding="utf-8"))
    claims = audit["details"]["claims"]

    assert claims["smoke_request_subset"] is False
    assert claims["smoke_atomic_workflow_subset"] is True
    assert claims["paper_ready"] is False
    assert execution.manifest["lineage"]["execution_scope"] == (
        "smoke_complete_atomic_workflow"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("aegis_slack_uses_future_information", True),
        ("workflow_slack_ms", 1.0),
        ("aegis_slack_input_provenance_sha256", ""),
    ],
)
def test_p11_fail_close_exception_rejects_future_nonzero_or_incomplete_ticket(
    field: str, value: object
) -> None:
    cell = select_cells(load_suite(), selectors=["P11:1"])[0]
    workload = prepare_workload(
        cell,
        experiment_root=Path("AegisKV/experiments").resolve(),
    )
    index = next(
        index
        for index, row in enumerate(workload.records)
        if row["simulation"]["aegis_slack_usable"] is False
    )
    rows = list(workload.records)
    corrupt = copy.deepcopy(rows[index])
    corrupt["simulation"][field] = value
    rows[index] = corrupt

    with pytest.raises(SuiteReadinessError, match="causal_slack_input_missing"):
        validate_prepared_workload_readiness(
            cell, replace(workload, records=tuple(rows))
        )


@pytest.mark.parametrize("group", ["P1", "P4", "P11", "P12"])
def test_representative_smoke_cell_uses_real_scheduler_and_writes_full_contract(
    tmp_path, group: str
) -> None:
    cell = select_cells(load_suite(), groups=[group], smoke=True)[0]
    execution = execute_prepared_cell(
        cell,
        _workload(cell),
        tmp_path / group / cell.cell_id,
        options=_options(),
        p0_lineage={"status": "PASS", "contract_only": True},
        runtime_config_override=_runtime(
            aegis=group in {"P4", "P12"}
        ),
        physical_topology_override=_topology(),
    )
    required = {
        "events.jsonl",
        "rank_steps.jsonl",
        "gpu_hbm_timeseries.jsonl",
        "node_dram_timeseries.jsonl",
        "kv_extents.jsonl",
        "transfers.jsonl",
        "transfer_groups.jsonl",
        "transfer_resources.jsonl",
        "recovery_transfers.jsonl",
        "requests.jsonl",
        "failures.jsonl",
        "audit.json",
        "manifest.json",
    }
    assert required <= {path.name for path in execution.path.iterdir()}
    audit = json.loads((execution.path / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["details"]["real_scheduler_integration"] is True
    assert audit["details"]["claims"]["paper_ready"] is False
    assert execution.metrics["resources"]["gpu_hbm_peak_used_bytes"]
    assert execution.metrics["resources"]["node_dram_peak_used_bytes"]
    ordinary = execution.metrics["physical_migrations"]["ordinary_hicache"]
    if group != "P12":
        assert ordinary["count"] > 0
        assert ordinary["bytes"] > 0
        assert ordinary["states"] == {"COMMITTED": ordinary["count"]}
    else:
        # The analytical-only P12 smoke request has no HiCache eviction/reload.
        assert ordinary["count"] == 0
        assert ordinary["bytes"] == 0
        assert ordinary["states"] == {}
    assert audit["details"]["ordinary_hicache"]["accounting"][
        "aegis_accounts_used"
    ] is False
    exposure = audit["details"]["active_failure_exposure"]
    if group in {"P1", "P4", "P11"}:
        assert exposure["status"] == "PASS"
        assert exposure["affected_original_request_ids"] == [
            exposure["anchor_request_id"]
        ]
        plan = audit["details"]["policy_independent_failure_plan"]
        assert plan["shadow_plan_sha256"] == exposure["shadow_plan_sha256"]
        assert plan["anchor_source_dp_rank"] == exposure[
            "anchor_source_dp_rank"
        ]
    else:
        assert exposure["status"] == "NOT_APPLICABLE"
    transfer_rows = [
        json.loads(line)
        for line in (execution.path / "transfers.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        row.get("transfer_class") == "ordinary_hicache"
        for row in transfer_rows
    ) is (group != "P12")
    if group == "P12":
        assert execution.metrics["analytical_vs_des"]["all_exact"] is True
        assert execution.metrics["analytical_vs_des"]["evidence"] == (
            "modeled_preliminary_not_hardware_calibration"
        )


def test_protected_cell_without_causal_slack_fails_before_worker_launch(
    tmp_path,
) -> None:
    cell = select_cells(load_suite(), groups=["P4"], smoke=True)[0]
    workload = _workload(cell)
    record = dict(workload.records[0])
    record.pop("simulation")
    with pytest.raises(SuiteReadinessError, match="causal_slack_input_missing"):
        execute_prepared_cell(
            cell,
            replace(workload, records=(record,)),
            tmp_path / "must-not-exist",
            options=_options(),
            p0_lineage={"status": "PASS", "contract_only": True},
            runtime_config_override=_runtime(aegis=True),
            physical_topology_override=_topology(),
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_resume_validates_every_file_and_rejects_corruption(tmp_path) -> None:
    cell = select_cells(load_suite(), groups=["P12"], smoke=True)[0]
    execution = execute_prepared_cell(
        cell,
        _workload(cell),
        tmp_path / cell.cell_id,
        options=_options(),
        p0_lineage={"status": "PASS", "contract_only": True},
        runtime_config_override=_runtime(aegis=True),
        physical_topology_override=_topology(),
    )
    manifest = execution.manifest
    assert validate_resumable_cell(
        execution.path,
        expected_config_sha256=config_hash(cell.resolved_config),
        expected_code_hash=str(manifest["code_hash"]),
    )["canonical_payload_sha256"] == manifest["canonical_payload_sha256"]
    with (execution.path / "events.jsonl").open("ab") as handle:
        handle.write(b"corrupt\n")
    with pytest.raises(ResumeValidationError, match="corrupt"):
        validate_resumable_cell(
            execution.path,
            expected_config_sha256=config_hash(cell.resolved_config),
            expected_code_hash=str(manifest["code_hash"]),
        )


def test_p11_p12_without_calibration_are_explicitly_modeled_preliminary() -> None:
    for group in ("P11", "P12"):
        claims = evidence_claims(group, smoke=False, calibration_lineage={})
        assert claims["modeled_simulation"] is True
        assert claims["real_hardware_measurement"] is False
        assert claims["real_hardware_calibration_present"] is False
        assert claims["preliminary"] is True
        assert claims["paper_ready"] is False


def test_hard_failure_replays_with_new_rid_and_never_completes_killed_original(
    tmp_path,
) -> None:
    base = select_cells(load_suite(), groups=["P1"], smoke=True)[0]
    resolved = copy.deepcopy(dict(base.resolved_config))
    resolved["failure"]["fixed"]["trigger"]["central_time_ns"] = 1
    resolved["failure"]["fixed"]["repair"]["central_horizon_s"] = 0.01
    resolved["failure"]["fixed"]["repair"]["routing_outage_ms"] = 1
    cell = replace(
        base,
        resolved_config=resolved,
        cell_id=base.cell_id + "-hard-replay",
    )
    workload = _workload(cell)
    hard_record = dict(workload.records[0])
    hard_record["rid"] = "hard-rank1"
    workload = replace(workload, records=(hard_record,))
    execution = execute_prepared_cell(
        cell,
        workload,
        tmp_path / cell.cell_id,
        options=_options(),
        p0_lineage={"status": "PASS", "contract_only": True},
        runtime_config_override=_runtime(aegis=False),
        physical_topology_override=_topology(),
    )
    requests = [
        json.loads(line)
        for line in (execution.path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    originals = [row for row in requests if not row["failover_replay"]]
    replays = [row for row in requests if row["failover_replay"]]
    assert len(originals) == 1 and originals[0]["status"] == "failed_superseded_by_replay"
    assert len(replays) == 1 and replays[0]["status"] == "completed"
    assert replays[0]["replay_of"] == originals[0]["rid"]
    assert replays[0]["route_version"] == 2
    assert replays[0]["router_output"]["strict_replay_contract"] is True
    assert replays[0]["router_output"]["base_output_sequence"] == 0
    assert replays[0]["router_output"]["ready_ranks"] == replays[0][
        "router_output"
    ]["expected_ready_ranks"]
    audit = json.loads(
        (execution.path / "audit.json").read_text(encoding="utf-8")
    )
    output_audit = audit["details"]["router_output_conservation"]
    assert output_audit["status"] == "PASS"
    assert output_audit["route_chains"][originals[0]["rid"]] == [1, 2]
    assert output_audit["sequence_gap_count"] == 0
    assert output_audit["sequence_duplicate_count"] == 0
    assert output_audit["real_logits_or_rng_continuation_claimed"] is False
    assert execution.replay_ids == (replays[0]["rid"],)


def test_recovery_admission_audit_rejects_one_ungated_replay() -> None:
    def execution(execution_id: str, *, gated: bool) -> dict[str, object]:
        gate = {
            "request_id": execution_id,
            "group_id": f"group-{execution_id}",
            "state": "ADOPTED",
        }
        return {
            "execution_id": execution_id,
            "ready_ack_evidence": {
                "physical_recovery_job_gate_in_scope": gated,
                "physical_recovery_gate": gate if gated else None,
            },
        }

    router = {
        "physical_recovery_job_activation_gate_in_scope": True,
        "logical_requests": [
            {
                "executions": [
                    execution("retry-a", gated=True),
                    execution("retry-b", gated=False),
                ]
            }
        ],
    }
    physical = {
        "recovery_admission_gate": {
            "admissions": [
                {
                    "request_id": request_id,
                    "group_id": f"group-{request_id}",
                    "state": "ADOPTED",
                    "execution_route_version": index + 2,
                }
                for index, request_id in enumerate(("retry-a", "retry-b"))
            ],
            "pending_holds": [],
        }
    }

    with pytest.raises(SuiteExecutionError, match="retry-b"):
        _validate_physical_recovery_admission_gate(router, physical)


def test_recovery_admission_audit_exports_full_immutable_gate() -> None:
    gate = {
        "request_id": "retry-a",
        "group_id": "group-retry-a",
        "state": "FALLBACK_RECOMPUTE",
        "fallback_reason": "capacity_shortage",
    }
    router = {
        "physical_recovery_job_activation_gate_in_scope": True,
        "logical_requests": [
            {
                "executions": [
                    {
                        "execution_id": "retry-a",
                        "ready_ack_evidence": {
                            "physical_recovery_job_gate_in_scope": True,
                            "physical_recovery_gate": gate,
                        },
                    }
                ]
            }
        ],
    }
    physical = {
        "recovery_admission_gate": {
            "schema": "sglsim.physical_recovery_admission.v1",
            "admissions": [
                {
                    **gate,
                    "execution_route_version": 2,
                    "lost_adopted_shards": [],
                }
            ],
            "events": [{"event": "fallback_released"}],
            "canonical_groups": [],
            "canonical_transfers": [],
            "pending_holds": [],
        }
    }

    audit = _validate_physical_recovery_admission_gate(router, physical)

    assert audit["status"] == "PASS"
    assert audit["terminal_admission_count"] == 1
    assert audit["checked_executions"][0]["state"] == "FALLBACK_RECOMPUTE"
    assert audit["admission_export"]["events"] == [
        {"event": "fallback_released"}
    ]


def test_replay_clone_uses_exact_router_boundary_not_original_full_budget() -> None:
    source = TokenizedGenerateReqInput.from_generate(
        GenerateReqInput(
            rid="clone-source",
            input_ids=[1, 2, 3],
            max_new_tokens=5,
            simulation={"synthetic_output_token_id": 7, "force_dp_rank": 0},
        )
    )
    replay = _clone_failover_replay(
        source,
        event_id="clone-failure",
        failed_dp_rank=0,
        replay_at_s=1.5,
        replay_spec={
            "logical_request_id": "clone-source",
            "failed_route_version": 1,
            "next_route_version": 2,
            "committed_output_position": 2,
            "remaining_max_new_tokens": 3,
            "pre_failure_total_tokens": 5,
            "continuation_state_digest": "continuation-digest",
            "continuation_compatibility_hash": "compatibility-hash",
            "continuation_prefix_hash": "prefix-hash",
            "replay_input_ids": [1, 2, 3, 7, 7],
            "synthetic_output_token_id": 7,
            "routing_key": None,
        },
    )
    sim = replay.sampling_params.custom_params["simulation"]
    assert replay.input_ids == [1, 2, 3, 7, 7]
    assert replay.sampling_params.max_new_tokens == 3
    assert sim["pre_failure_total_tokens"] == 5
    assert sim["aegis_committed_output_position"] == 2
    assert sim["aegis_route_version"] == 2
    assert "force_dp_rank" not in sim


def test_future_failure_preserves_completed_row_across_target_restart(
    tmp_path,
) -> None:
    """A later fail-stop may not erase an already terminal original row."""

    base = select_cells(load_suite(), groups=["P1"], smoke=True)[0]
    resolved = copy.deepcopy(dict(base.resolved_config))
    # This is an explicit legacy absolute-time contract test.  Formal P1-P8,
    # P10, and P11 cells use the policy-independent active-exposure mode.
    resolved["failure"]["fixed"]["trigger"]["mode"] = (
        "deterministic_virtual_time"
    )
    resolved["failure"]["fixed"]["trigger"]["central_time_ns"] = 10_000_000_000
    resolved["failure"]["fixed"]["repair"]["central_horizon_s"] = 0.01
    cell = replace(
        base,
        resolved_config=resolved,
        cell_id=base.cell_id + "-future-failure",
    )
    workload = _workload(cell)
    record = dict(workload.records[0])
    record["rid"] = "hard-rank1"  # Stable hash routes to failed DP rank 1.
    workload = replace(workload, records=(record,))

    execution = execute_prepared_cell(
        cell,
        workload,
        tmp_path / cell.cell_id,
        options=_options(),
        p0_lineage={"status": "PASS", "contract_only": True},
        runtime_config_override=_runtime(aegis=False),
        physical_topology_override=_topology(),
    )

    requests = [
        json.loads(line)
        for line in (execution.path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failures = [
        json.loads(line)
        for line in (execution.path / "failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    audit = json.loads((execution.path / "audit.json").read_text(encoding="utf-8"))
    assert [(row["rid"], row["status"]) for row in requests] == [
        ("hard-rank1", "completed")
    ]
    assert failures[0]["affected_request_ids"] == []
    closure = audit["details"]["request_terminal_closure"]
    assert closure["status"] == "PASS"
    assert closure["original_count"] == closure["completed_leaf_count"] == 1
    assert audit["details"]["metrics"]["scheduler"]["request_count"] == 1


def test_request_terminal_closure_rejects_scheduled_original() -> None:
    with pytest.raises(SuiteExecutionError, match="nonterminal"):
        _validate_terminal_request_closure(
            [
                {
                    "rid": "lost-original",
                    "status": "scheduled",
                    "failover_replay": False,
                    "replay_of": None,
                }
            ]
        )


def test_request_terminal_closure_requires_replay_chain_to_complete() -> None:
    with pytest.raises(SuiteExecutionError, match="no replay successor"):
        _validate_terminal_request_closure(
            [
                {
                    "rid": "killed-original",
                    "status": "failed_superseded_by_replay",
                    "failure_event_id": "failure-0",
                    "failover_replay": False,
                    "replay_of": None,
                }
            ]
        )

    audit = _validate_terminal_request_closure(
        [
            {
                "rid": "killed-original",
                "status": "failed_superseded_by_replay",
                "failure_event_id": "failure-0",
                "failover_replay": False,
                "replay_of": None,
            },
            {
                "rid": "replay-0",
                "status": "completed",
                "failure_event_id": "failure-0",
                "failover_replay": True,
                "replay_of": "killed-original",
            },
        ]
    )
    assert audit["status"] == "PASS"
    assert audit["completed_leaf_count"] == 1
    assert audit["nonterminal_count"] == 0
