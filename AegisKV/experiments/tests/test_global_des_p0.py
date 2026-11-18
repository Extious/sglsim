from __future__ import annotations

import json
from pathlib import Path

import pytest

from AegisKV.experiments.src.global_des_config import (
    P0_SCENARIOS,
    REQUIRED_INVARIANTS,
)
from AegisKV.experiments.src.run_global_des_p0 import main, run_p0
from sglsim.srt.simulation.global_des_artifacts import EVIDENCE_CLASS


REQUIRED_CELL_FILES = {
    "events.jsonl",
    "rank_steps.jsonl",
    "gpu_hbm_timeseries.jsonl",
    "node_dram_timeseries.jsonl",
    "kv_extents.jsonl",
    "transfers.jsonl",
    "requests.jsonl",
    "failures.jsonl",
    "audit.json",
    "transfer_groups.jsonl",
    "transfer_resources.jsonl",
    "recovery_transfers.jsonl",
    "manifest.json",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def two_p0_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("global-des-p0")
    first = root / "first"
    second = root / "second"
    assert run_p0(first)["status"] == "PASS"
    assert run_p0(second)["status"] == "PASS"
    return first, second


def test_p0_mechanically_executes_exactly_twelve_full_contract_cells(
    two_p0_runs: tuple[Path, Path],
) -> None:
    first, _ = two_p0_runs
    p0 = first / "P0"
    summary = _json(p0 / "summary.json")
    cell_directories = tuple(
        path.name for path in p0.iterdir() if path.is_dir()
    )

    assert summary["status"] == "PASS"
    assert summary["configured_suite_status"] == "planned_unexecuted"
    assert summary["executed_contract_cells"] == 12
    assert summary["required_contract_cells"] == 12
    assert tuple(row["contract_scenario"] for row in summary["cells"]) == P0_SCENARIOS
    assert set(cell_directories) == set(P0_SCENARIOS)
    assert summary["evidence_class"] == EVIDENCE_CLASS
    assert summary["claims"]["full_scheduler_integration"] is False
    assert summary["claims"][
        "controller_bridge_target_rotation_closed_loop"
    ] is True
    assert summary["claims"]["performance_result"] is False
    assert summary["claims"]["real_hardware_measurement"] is False
    assert summary["claims"]["configured_p0_contract_gate_passed"] is True
    assert summary["claims"]["p1_p12_execution_ready"] is False
    assert "p1_p12_unblocked" not in summary["claims"]

    for scenario in P0_SCENARIOS:
        cell = p0 / scenario
        assert {path.name for path in cell.iterdir()} == REQUIRED_CELL_FILES
        manifest = _json(cell / "manifest.json")
        audit = _json(cell / "audit.json")
        assert manifest["evidence_class"] == EVIDENCE_CLASS
        assert manifest["simulation"]["clock"] == "global_virtual_time"
        assert manifest["simulation"]["real_hardware_measurement"] is False
        assert manifest["lineage"]["execution_scope"] == "executed_contract_cell_only"
        details = audit["details"]
        assert details["status"] == "PASS"
        assert set(details["required_invariants"]) == set(REQUIRED_INVARIANTS)
        assert all(
            set(record) == {"passed", "evidence"} and record["passed"] is True
            for record in details["required_invariants"].values()
        )
        assert details["claims"]["full_scheduler_integration"] is False
        assert details["claims"][
            "controller_bridge_target_rotation_closed_loop"
        ] is (scenario == "target_first_and_node_failure")
        assert details["claims"]["performance_result"] is False
        assert details["claims"]["real_hardware_measurement"] is False
        assert "paper_ready" not in cell.joinpath("manifest.json").read_text()


def test_tp4_is_four_explicit_physical_gpus_per_replica(
    two_p0_runs: tuple[Path, Path],
) -> None:
    first, _ = two_p0_runs
    cell = first / "P0" / "qwen3_32b_tp4_shard"
    rank_steps = _jsonl(cell / "rank_steps.jsonl")
    source = [
        row
        for row in rank_steps
        if row.get("phase") == "physical_gpu_registered"
        and row.get("dp_rank") == 0
    ]
    targets = [
        row
        for row in rank_steps
        if row.get("phase") == "physical_gpu_registered"
        and row.get("dp_rank") == 1
    ]

    assert len(source) == len(targets) == 4
    assert sorted(row["tp_rank"] for row in source) == [0, 1, 2, 3]
    assert sorted(row["tp_rank"] for row in targets) == [0, 1, 2, 3]
    assert len({(row["node_id"], row["local_gpu_id"]) for row in source + targets}) == 8

    audit = _json(cell / "audit.json")["details"]
    assertion = audit["scenario_assertions"]["qwen3-32b_physical_tp_shards"]
    assert assertion["passed"] is True
    assert assertion["evidence"]["acked_shards"] == ["tp0", "tp1", "tp2", "tp3"]


def test_pending_failure_aborts_transfer_and_releases_target_reservation(
    two_p0_runs: tuple[Path, Path],
) -> None:
    first, _ = two_p0_runs
    cell = first / "P0" / "failure_in_pending_window"
    audit = _json(cell / "audit.json")["details"]
    evidence = audit["scenario_assertions"][
        "pending_failure_aborts_and_releases"
    ]["evidence"]
    transfer_states = [row["state"] for row in _jsonl(cell / "transfers.jsonl")]
    pending_samples = [
        row["aegis_pending"]
        for row in _jsonl(cell / "gpu_hbm_timeseries.jsonl")
        if row["dp_rank"] == 1
    ]

    assert evidence["transfer_start_ns"] < evidence["failure_ns"] < evidence["transfer_end_ns"]
    assert evidence["transfer_state"] == "ABORTED"
    assert evidence["group_state"] == "ABORTED"
    assert evidence["destination_state"] == "ABORTED"
    assert evidence["target_pending_bytes"] == 0
    assert evidence["target_committed_bytes"] == 0
    assert "IN_FLIGHT" in transfer_states and "ABORTED" in transfer_states
    assert max(pending_samples) > 0 and pending_samples[-1] == 0


def test_target_first_rotation_is_same_route_hbm_only_and_fails_closed(
    two_p0_runs: tuple[Path, Path],
) -> None:
    first, _ = two_p0_runs
    cell = first / "P0" / "target_first_and_node_failure"
    details = _json(cell / "audit.json")["details"]
    assertions = details["scenario_assertions"]
    expected = {
        "target_failure_rotates_same_route_without_owner_relabelling",
        "replacement_is_hbm_only_and_source_owned_dram_is_unchanged",
        "source_node_failure_recovers_only_from_healthy_replacement_hbm",
        "unhealthy_or_absent_tiers_fail_closed_to_recompute",
    }
    assert set(assertions) == expected
    assert all(assertions[name]["passed"] is True for name in expected)

    evidence = assertions[
        "target_failure_rotates_same_route_without_owner_relabelling"
    ]["evidence"]
    rotation = evidence["rotation"]
    assert rotation["version"] == evidence["version"] == 1
    assert rotation["route_version"] == evidence["route_version"] == 7
    assert rotation["route_version_changed"] is False
    assert rotation["previous_target_dp_rank"] == 1
    assert rotation["replacement_target_dp_rank"] == 2
    assert rotation["dram_preserved"] is True
    assert evidence["old_hbm_owner"]["target_dp_rank"] == 1
    assert evidence["old_hbm_owner"]["released"] is True
    assert evidence["replacement_hbm_owner"]["target_dp_rank"] == 2
    assert evidence["old_hbm_owner"]["group_id"] != evidence[
        "replacement_hbm_owner"
    ]["group_id"]
    assert evidence["old_hbm_destination_dp_rank"] == 1
    assert evidence["replacement_hbm_destination_dp_rank"] == 2

    assert evidence["rotated_plan_commands"] == [
        {
            "operation": "aegis_l1_l2_backup_plan",
            "target_dp_rank": 2,
            "allowed_tiers": ["l1"],
            "route_version": 7,
            "version": 1,
            "target_rotation_hbm_only": True,
        }
    ]
    assert evidence["replacement_hbm_group_state"] == "COMMITTED"
    assert evidence["dram_group_state"] == "COMMITTED"
    assert evidence["dram_job_count"] == 1
    assert evidence["dram_committed_bytes_before_rotation"] == evidence[
        "dram_committed_bytes_after_rotation"
    ]

    assert evidence["source_failed_hint"]["l1_tokens"] > 0
    assert evidence["source_failed_hint"]["l2_tokens"] == 0
    assert evidence["healthy_recovery"]["selected_tier"] == "gpu_aegis"
    assert set(evidence["healthy_recovery"]["outcomes"].values()) == {
        "COMPLETED"
    }
    assert all(
        row["tier"] == "gpu_aegis"
        and row["gpu"]["dp_rank"] == 2
        for row in evidence["healthy_recovery"]["job_sources"]
    )
    assert all(
        row["tier"] == "gpu_primary"
        and row["gpu"]["dp_rank"] == 3
        for row in evidence["healthy_recovery"]["job_destinations"]
    )
    assert evidence["terminal_hint"]["safe_frontier_tokens"] == 0
    assert evidence["stale_recovery_error"]
    assert evidence["fail_closed_plan"]["selected_tier"] == "recompute"
    assert evidence["fail_closed_plan"]["jobs"] == []
    assert evidence["replacement_rotations_after_source_loss"] == []
    assert evidence["bridge_audit_errors"] == []

    failures = _jsonl(cell / "failures.jsonl")
    assert [row["failure_id"] for row in failures] == [
        "target-gpu-first",
        "source-node-second",
        "replacement-node-third",
    ]
    assert [row["failure_ns"] for row in failures] == sorted(
        row["failure_ns"] for row in failures
    )
    phases = {
        row.get("phase") for row in _jsonl(cell / "rank_steps.jsonl")
    }
    assert {
        "standby_target_rotated",
        "replacement_hbm_committed",
        "healthy_compatible_tier_recovery",
        "unhealthy_tier_fail_closed",
    } <= phases


def test_no_cell_contains_a_peer_hbm_to_dram_destage_edge(
    two_p0_runs: tuple[Path, Path],
) -> None:
    first, _ = two_p0_runs
    for scenario in P0_SCENARIOS:
        transfers = _jsonl(first / "P0" / scenario / "transfers.jsonl")
        assert not any(
            row["source"]["tier"] == "gpu_aegis"
            and row["destination"]["tier"] == "node_dram"
            for row in transfers
        )


def test_two_fresh_runs_have_identical_modeled_event_content(
    two_p0_runs: tuple[Path, Path],
) -> None:
    first, second = two_p0_runs
    first_summary = _json(first / "P0" / "summary.json")
    second_summary = _json(second / "P0" / "summary.json")
    assert first_summary == second_summary

    for scenario in P0_SCENARIOS:
        first_cell = first / "P0" / scenario
        second_cell = second / "P0" / scenario
        assert (first_cell / "events.jsonl").read_bytes() == (
            second_cell / "events.jsonl"
        ).read_bytes()
        first_audit = _json(first_cell / "audit.json")
        second_audit = _json(second_cell / "audit.json")
        assert first_audit["details"]["event_content_sha256"] == second_audit[
            "details"
        ]["event_content_sha256"]
        assert first_audit["details"]["required_invariants"][
            "deterministic_event_hash"
        ]["passed"] is True


def test_runner_refuses_to_overwrite_any_existing_run_directory(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "user-data"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_p0(existing)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert main(["--run-dir", str(existing)]) != 0
    assert sentinel.read_text(encoding="utf-8") == "preserve"
