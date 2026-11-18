from __future__ import annotations

import inspect
import json
from pathlib import Path

import run_active_hbm_v3_capacity_sanity as sanity


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PROTOCOL_PATH = ROOT / "active_hbm_v3_protocol.json"


def load() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_frozen_model_and_scheduler_sources_match() -> None:
    protocol = load()
    model = protocol["frozen_model"]
    runtime = protocol["runtime_reserve"]
    shared = protocol["shared_capacity"]
    assert sanity.sha256(Path(model["config_path"])) == model["config_sha256"]
    assert sanity.sha256(Path(model["safetensors_index_path"])) == model["safetensors_index_sha256"]
    assert sanity.sha256(REPO / runtime["source_path"]) == runtime["source_sha256"]
    assert sanity.sha256(ROOT / shared["request_max_output_tokens_source_path"]) == shared["request_max_output_tokens_source_sha256"]


def test_weight_kv_and_shared_budget_equations_are_exact() -> None:
    protocol = load()
    model = protocol["frozen_model"]
    runtime = protocol["runtime_reserve"]
    total = protocol["hardware"]["total_hbm_bytes"]
    static = int(total * runtime["mem_fraction_static"])
    kv = 2 * model["layers"] * model["kv_heads"] * model["head_dim"] * model["kv_dtype_bytes"]
    assert model["weight_bytes_per_tp1_pp1_rank"] == 16381470720
    assert kv == model["kv_bytes_per_token"] == 147456
    assert static == runtime["static_hbm_bytes"] == 37795712204
    assert total - static == runtime["runtime_reserve_bytes"] == 5153960756
    assert static - model["weight_bytes_per_tp1_pp1_rank"] == protocol["shared_capacity"]["serving_plus_peer_backup_budget_bytes"] == 21414241484


def test_max_output_is_frozen_from_workload() -> None:
    protocol = load()
    shared = protocol["shared_capacity"]
    workload = json.loads((ROOT / shared["request_max_output_tokens_source_path"]).read_text(encoding="utf-8"))
    assert shared["request_max_output_tokens"] == max(workload["workload"]["output_tokens"]) == 4096


def test_dispatch_reservation_api_has_no_realized_output_oracle() -> None:
    protocol = load()
    assert tuple(inspect.signature(sanity.serving_reservation_bytes).parameters) == (
        "protocol", "prompt_tokens"
    )
    expected = (40960 + 4096) * 147456
    # All possible final lengths share the same dispatch reservation because
    # none can be passed into the dispatch API.
    for _realized_output in (256, 1024, 4096):
        assert sanity.serving_reservation_bytes(protocol, 40960) == expected


def test_capacity_limits_are_conservative_at_max_output() -> None:
    protocol = load()
    budget = protocol["shared_capacity"]["serving_plus_peer_backup_budget_bytes"]
    initial_backup = protocol["backup_projection"]["aegiskv_initial_peer_backup_limit_bytes"]
    for context in protocol["sanity_matrix"]["context_tokens"]:
        serving = sanity.serving_reservation_bytes(protocol, context)
        prompt_backup = context * protocol["frozen_model"]["kv_bytes_per_token"]
        assert serving <= budget
        feasible = [
            lanes for lanes in range(17)
            if lanes * serving + min(initial_backup, lanes * prompt_backup) <= budget
        ]
        lanes = max(feasible)
        assert lanes * serving + min(initial_backup, lanes * prompt_backup) <= budget


def main() -> int:
    tests = [
        test_frozen_model_and_scheduler_sources_match,
        test_weight_kv_and_shared_budget_equations_are_exact,
        test_max_output_is_frozen_from_workload,
        test_dispatch_reservation_api_has_no_realized_output_oracle,
        test_capacity_limits_are_conservative_at_max_output,
    ]
    for check in tests:
        check()
        print(f"PASS {check.__name__}", flush=True)
    print(f"RESULT {len(tests)}/{len(tests)} PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
