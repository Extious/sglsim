from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

import run_mechanism_pilot_v2 as pilot
import run_preliminary as core


def task(
    request_id: str, *, arrival: float, start: float, finish: float,
    service: float = 100.0, lane: int = 0,
) -> core.Task:
    return core.Task(
        request_id=request_id,
        workflow_id=request_id,
        workflow_arrival_ms=arrival,
        arrival_ms=arrival,
        source_rank=0,
        lane=lane,
        start_ms=start,
        finish_ms=finish,
        service_ms=service,
        prompt_tokens=4096,
        output_tokens=256,
        predicted_slack_ms=10.0,
    )


def test_empty_fault_replay_exactly_preserves_protection_aware_completion() -> None:
    tasks = [
        task("one", arrival=0.0, start=0.0, finish=110.0),
        task("two", arrival=50.0, start=110.0, finish=225.0),
    ]
    replayed = core.replay_one_fault(tasks, {})
    assert replayed == {"one": 110.0, "two": 225.0}


def test_fault_replay_never_reduces_request_or_workflow_completion() -> None:
    tasks = [
        task("one", arrival=0.0, start=0.0, finish=110.0),
        task("two", arrival=50.0, start=110.0, finish=225.0),
    ]
    replayed = core.replay_one_fault(tasks, {"one": 500.0})
    assert replayed["one"] >= tasks[0].finish_ms
    assert replayed["two"] >= tasks[1].finish_ms
    _ids, base = core.workflow_wct(tasks)
    _ids, fault = core.workflow_wct(tasks, replayed)
    assert np.all(np.asarray(fault) >= np.asarray(base))


def test_policy_specific_failover_contract_is_fail_closed() -> None:
    protocol, _spec = pilot.load_composite_protocol()
    assert core.policy_failover_delay_ms(
        protocol, "Native Mooncake", "source_worker_gpu"
    ) == (60000.0, "replacement_wait")
    assert core.policy_failover_delay_ms(
        protocol, "AegisKV", "source_worker_gpu"
    ) == (500.0, "prebound_fence_activation")
    assert core.policy_failover_delay_ms(
        protocol, "AegisKV", "source_node"
    ) == (60000.0, "replacement_wait")


def test_copy_and_load_bandwidth_are_separate_runtime_inputs() -> None:
    protocol, _spec = pilot.load_composite_protocol()
    protocol["hardware"]["copy_bandwidth_gb_s"]["peer_hbm_same_node"] = 1.25
    protocol["hardware"]["load_bandwidth_gb_s"]["peer_hbm_same_node"] = 7.5
    assert core.copy_bandwidth(protocol, 0)[0] == 1.25
    assert core.load_bandwidth(protocol, 0)[0] == 7.5


def test_own_transfer_burden_does_not_recharge_copy_queue_wait() -> None:
    protocol, _spec = pilot.load_composite_protocol()
    replay = core.ProtectionReplay(protocol, "Native Mooncake")
    first = task("one", arrival=0.0, start=0.0, finish=1000.0, service=1000.0)
    second = replace(first, request_id="two", workflow_id="two")
    replay.start(first, 0.0)
    replay.start(second, 0.0)
    assert replay.copy_queue_wait_ms["two"] > 0.0
    assert replay.copy_service_ms["one"] == replay.copy_service_ms["two"]
    assert replay.health_burden_ms["one"] == replay.health_burden_ms["two"]


def test_small_cell_fixed_point_and_fault_monotonicity(tmp_path: Path) -> None:
    protocol, _spec = pilot.load_composite_protocol()
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 1
    protocol["workload"]["center_context_tokens"] = 4096
    protocol["aegiskv"]["window_ms"] = 300.0
    protocol["failure_model"]["probe_type_schedule"] = ["source_worker_gpu"]
    result = core.run_cell(
        protocol,
        policy="AegisKV",
        seed=2026082600,
        workflows=None,
        active_per_gpu=2,
        probe_count=2,
        output=tmp_path / "small.npz",
        arrival_cutoff_ms=900.0,
    )
    solver = result["mechanism"]["normal_path_schedule_solver"]
    assert solver["converged"] is True
    assert solver["final_burden_residual_finish_ms"] <= 1e-7
    assert result["mechanism"]["empty_fault_replay_max_completion_delta_ms"] <= 1e-7
    with np.load(result["raw_npz"], allow_pickle=False) as raw:
        assert np.all(raw["fault_wct_ms"] >= raw["base_wct_ms"][None, :] - 1e-7)


def main() -> int:
    tests = [
        test_empty_fault_replay_exactly_preserves_protection_aware_completion,
        test_fault_replay_never_reduces_request_or_workflow_completion,
        test_policy_specific_failover_contract_is_fail_closed,
        test_copy_and_load_bandwidth_are_separate_runtime_inputs,
        test_own_transfer_burden_does_not_recharge_copy_queue_wait,
    ]
    passed = 0
    for check in tests:
        check()
        passed += 1
        print(f"PASS {check.__name__}", flush=True)
    with tempfile.TemporaryDirectory(prefix="sglsim-mechanism-test-") as tmp:
        test_small_cell_fixed_point_and_fault_monotonicity(Path(tmp))
    passed += 1
    print("PASS test_small_cell_fixed_point_and_fault_monotonicity", flush=True)
    print(f"RESULT {passed}/6 PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
