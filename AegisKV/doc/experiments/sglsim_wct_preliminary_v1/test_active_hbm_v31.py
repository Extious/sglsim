#!/usr/bin/env python3
"""Executable regression harness for the v3.1 semantic corrections."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

import active_hbm_v3 as v3
import active_hbm_v31 as v31
import run_mechanism_pilot_v2 as pilot
import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def inputs() -> tuple[dict, dict]:
    protocol, _spec = pilot.load_composite_protocol()
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 1
    protocol["workload"]["center_context_tokens"] = 40960
    protocol["workload"]["center_active_requests_per_gpu"] = 16
    protocol["aegiskv"]["window_ms"] = 300.0
    protocol["failure_model"]["probe_type_schedule"] = ["source_worker_gpu"]
    capacity = json.loads((ROOT / "active_hbm_v3_protocol.json").read_text())
    return protocol, capacity


def main() -> int:
    protocol, capacity = inputs()
    nominal, _meta = core.build_workload(
        protocol, seed=2026082600, workflows=None, active_per_gpu=16,
        arrival_cutoff_ms=900.0,
    )
    first = v31.apply_nonoracle_modeled_slack(nominal, protocol, seed=2026082600)
    perturbed = [replace(row, output_tokens=row.output_tokens + 777, service_ms=row.service_ms + 999.0) for row in nominal]
    second = v31.apply_nonoracle_modeled_slack(perturbed, protocol, seed=2026082600)
    assert [row.predicted_slack_ms for row in first] == [row.predicted_slack_ms for row in second]
    print("PASS slack_prediction_ignores_realized_output_and_service", flush=True)

    small_protocol, _spec = pilot.load_composite_protocol()
    small_protocol["models"]["qwen3_8b"]["center_system_dp"] = 1
    replay = v3.ActiveHBMCoupledReplay(small_protocol, "AegisKV", capacity)
    task = core.Task(
        request_id="parallel-copy", workflow_id="parallel-copy",
        workflow_arrival_ms=0.0, arrival_ms=0.0, source_rank=0, lane=0,
        start_ms=0.0, finish_ms=1000.0, service_ms=1000.0,
        prompt_tokens=4096, output_tokens=256, predicted_slack_ms=0.0,
    )
    replay.start(task, 0.0)
    hbm_done = replay.aegis_lane[(0, core.ProtectionTier.HBM)]
    dram_done = replay.aegis_lane[(0, core.ProtectionTier.DRAM)]
    expected_bytes = 4096 * int(small_protocol["models"]["qwen3_8b"]["kv_bytes_per_token"])
    pools = replay.aegis[0].telemetry_snapshot()["pools"]
    assert pools["HBM"]["churn"]["admitted_bytes"] == expected_bytes
    assert pools["DRAM"]["churn"]["admitted_bytes"] == expected_bytes
    assert replay.copy_service_ms[task.request_id] == max(hbm_done, dram_done)
    assert replay.copy_service_ms[task.request_id] < hbm_done + dram_done
    assert replay.health_burden_ms[task.request_id] == replay.visible_copy_burden(task, max(hbm_done, dram_done))
    print("PASS source_owned_paths_parallel_max_not_sum_and_exact_bytes", flush=True)

    domain_protocol, _spec = pilot.load_composite_protocol()
    domain_protocol["models"]["qwen3_8b"]["center_system_dp"] = 16
    domain_tasks = [
        replace(
            task,
            request_id=f"domain-rank-{rank}",
            workflow_id=f"domain-rank-{rank}",
            source_rank=rank,
            service_ms=1000.0,
            finish_ms=1000.0,
        )
        for rank in range(8)
    ]
    node_replay = v31.ExactAdmissionProjectionReplay(
        domain_protocol, "Native Mooncake", capacity
    )
    _tasks, node_delays, _mechanism, _trace = node_replay.run_coupled(
        domain_tasks,
        [core.FaultProbe("node-probe", 10.0, 3, "source_node")],
        control_horizon_ms=100.0,
    )
    worker_replay = v31.ExactAdmissionProjectionReplay(
        domain_protocol, "Native Mooncake", capacity
    )
    _tasks, worker_delays, _mechanism, _trace = worker_replay.run_coupled(
        domain_tasks,
        [core.FaultProbe("worker-probe", 10.0, 3, "source_worker_gpu")],
        control_horizon_ms=100.0,
    )
    assert set(node_delays[0]) == {f"domain-rank-{rank}" for rank in range(8)}
    assert set(worker_delays[0]) == {"domain-rank-3"}
    assert v31.failure_domain_ranks(
        core.FaultProbe("node-domain", 10.0, 11, "source_node"),
        gpus_per_node=8, dp=16,
    ) == tuple(range(8, 16))
    assert v31.failure_domain_ranks(
        core.FaultProbe("worker-domain", 10.0, 11, "source_worker_gpu"),
        gpus_per_node=8, dp=16,
    ) == (11,)
    print("PASS source_node_hits_exact_eight_ranks_and_worker_hits_one", flush=True)

    with tempfile.TemporaryDirectory(prefix="sglsim-v31-test-") as tmp:
        tmp_path = Path(tmp)
        cells = []
        for policy in ("Native Mooncake", "AegisKV", "AegisKV no Algorithm 2"):
            cells.append(v31.run_v31_cell(
                protocol, capacity, policy=policy, seed=2026082600,
                active_per_gpu=16, probe_count=2,
                output=tmp_path / f"{policy.replace(' ', '_')}.npz",
                arrival_cutoff_ms=900.0,
            ))
        assert all(
            cell["mechanism"]["active_hbm_v31"]["projection_prediction_mismatch_count"] == 0
            for cell in cells
        )
        assert cells[2]["mechanism"]["active_hbm_v31"]["old_projection_false_block_unique_request_count"] > 0
        print("PASS exact_admission_projection_matches_real_admission_and_removes_false_blocks", flush=True)

        assert len({cell["workflow_universe_sha256"] for cell in cells}) == 1
        assert len({cell["fault_probe_tuple_sha256"] for cell in cells}) == 1
        for cell in cells:
            failure = cell["failure_conditioned_changed_by_type"][cell["failure_type"]]
            assert failure["all_workflow_observations"] == cell["workflow_count"] * cell["fault_probe_count"]
        print("PASS common_workflow_probe_universe_and_denominator", flush=True)

        for cell in cells:
            with np.load(cell["raw_npz"], allow_pickle=False) as raw:
                assert np.all(raw["fault_wct_ms"] >= raw["base_wct_ms"][None, :] - 1e-7)
                assert np.all(raw["capacity_shared_used_bytes"] <= capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"])
                assert np.all(raw["probe_failure_domain_rank_ids"][:, 0] >= 0)
                assert np.all(raw["probe_failure_domain_rank_ids"][:, 1:] == -1)
            assert cell["mechanism"]["active_hbm_v3"]["capacity_violation_count"] == 0
        print("PASS fault_monotonicity_and_shared_capacity", flush=True)

        for cell in cells:
            failure = cell["failure_conditioned_changed_by_type"][cell["failure_type"]]
            assert failure["same_probe_failure_conditioned_wct_mean_ms"] == failure["all_workflow_wct_mean_ms"]
            assert abs(
                failure["all_workflow_wct_mean_ms"]
                - cell["base_wct_mean_ms"]
                - failure["all_workflow_added_wct_mean_ms"]
            ) < 1e-7
        print("PASS healthy_plus_added_equals_all_workflow_total", flush=True)
    print("RESULT 7/7 PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
