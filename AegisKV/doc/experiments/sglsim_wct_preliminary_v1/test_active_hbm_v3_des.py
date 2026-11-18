from __future__ import annotations

import json
import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

import active_hbm_v3 as v3
import run_mechanism_pilot_v2 as pilot
import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def configured_inputs() -> tuple[dict, dict]:
    protocol, _spec = pilot.load_composite_protocol()
    protocol["models"]["qwen3_8b"]["center_system_dp"] = 1
    protocol["workload"]["center_context_tokens"] = 40960
    protocol["aegiskv"]["window_ms"] = 300.0
    protocol["failure_model"]["probe_type_schedule"] = ["source_worker_gpu"]
    capacity = json.loads((ROOT / "active_hbm_v3_protocol.json").read_text(encoding="utf-8"))
    return protocol, capacity


def run_small(policy: str, output: Path) -> dict:
    protocol, capacity = configured_inputs()
    return v3.run_v3_cell(
        protocol,
        capacity,
        policy=policy,
        seed=2026082600,
        active_per_gpu=16,
        probe_count=2,
        output=output,
        arrival_cutoff_ms=900.0,
    )


def main() -> int:
    protocol, capacity = configured_inputs()
    replay = v3.ActiveHBMCoupledReplay(protocol, "Native Mooncake", capacity)
    nominal = core.Task(
        request_id="oracle-a", workflow_id="oracle-a",
        workflow_arrival_ms=0.0, arrival_ms=0.0, source_rank=0, lane=0,
        start_ms=0.0, finish_ms=1.0, service_ms=1.0,
        prompt_tokens=40960, output_tokens=256, predicted_slack_ms=0.0,
    )
    assert replay.serving_reservation_bytes(nominal) == replay.serving_reservation_bytes(
        replace(nominal, request_id="oracle-b", output_tokens=4096)
    ) == (40960 + 4096) * 147456
    print("PASS dispatch_reservation_ignores_realized_output", flush=True)

    gated = [
        replace(nominal, request_id="a", workflow_id="a", start_ms=10.0, finish_ms=110.0, service_ms=90.0),
        replace(nominal, request_id="b", workflow_id="b", arrival_ms=0.0, start_ms=250.0, finish_ms=350.0, service_ms=90.0),
    ]
    replayed = core.replay_one_fault(gated, {})
    assert replayed == {"a": 110.0, "b": 350.0}
    print("PASS empty_replay_preserves_memory_admission_gaps", flush=True)

    with tempfile.TemporaryDirectory(prefix="sglsim-v3-test-") as tmp:
        tmp_path = Path(tmp)
        native = run_small("Native Mooncake", tmp_path / "native.npz")
        full = run_small("AegisKV", tmp_path / "full.npz")
        no3 = run_small("AegisKV no Algorithm 3", tmp_path / "no3.npz")
        for result in (native, full, no3):
            mechanism = result["mechanism"]["active_hbm_v3"]
            assert mechanism["capacity_violation_count"] == 0
            assert mechanism["capacity_peak_total_hbm_bytes"] <= mechanism["total_hbm_bytes"]
            assert mechanism["memory_blocked_unique_request_count"] > 0
            assert result["mechanism"]["empty_fault_replay_max_completion_delta_ms"] <= 1e-7
            failure = result["failure_conditioned_changed_by_type"][result["failure_type"]]
            assert failure["all_workflow_observations"] == (
                result["workflow_count"] * result["fault_probe_count"]
            )
            assert failure["all_workflow_added_wct_mean_ms"] >= 0.0
            assert len(result["workflow_universe_sha256"]) == 64
            assert len(result["fault_probe_tuple_sha256"]) == 64
        assert len({row["workflow_universe_sha256"] for row in (native, full, no3)}) == 1
        assert len({row["fault_probe_tuple_sha256"] for row in (native, full, no3)}) == 1
        print("PASS shared_capacity_invariant_and_memory_queue", flush=True)

        assert full["mechanism"]["control_window_count"] == 3
        assert full["mechanism"]["active_hbm_v3"]["backup_bytes_released_by_control"] > 0
        assert full["mechanism"]["active_hbm_v3"]["dispatch_at_control_count"] > 0
        assert no3["mechanism"]["control_window_count"] == 0
        assert no3["mechanism"]["active_hbm_v3"]["backup_bytes_released_by_control"] == 0
        assert no3["mechanism"]["active_hbm_v3"]["dispatch_at_control_count"] == 0
        print("PASS algorithm3_releases_backup_and_restores_dispatch_headroom", flush=True)

        for result in (native, full, no3):
            with np.load(result["raw_npz"], allow_pickle=False) as raw:
                assert np.all(raw["fault_wct_ms"] >= raw["base_wct_ms"][None, :] - 1e-7)
                assert np.all(raw["capacity_shared_used_bytes"] <= capacity["shared_capacity"]["serving_plus_peer_backup_budget_bytes"])
                workflow_digest = hashlib.sha256(json.dumps(
                    raw["workflow_id"].tolist(), separators=(",", ":")
                ).encode("utf-8")).hexdigest()
                probe_payload = [
                    {
                        "probe_id": str(probe_id),
                        "time_ms": float(time_ms),
                        "source_rank": int(source_rank),
                        "failure_type": str(failure_type),
                    }
                    for probe_id, time_ms, source_rank, failure_type in zip(
                        raw["probe_id"], raw["probe_time_ms"],
                        raw["probe_source_rank"], raw["probe_failure_type"],
                    )
                ]
                probe_digest = hashlib.sha256(json.dumps(
                    probe_payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")).hexdigest()
                assert workflow_digest == result["workflow_universe_sha256"]
                assert probe_digest == result["fault_probe_tuple_sha256"]
        print("PASS fault_wct_monotone_and_raw_capacity_trace", flush=True)
    print("RESULT 5/5 PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
