from __future__ import annotations

import json

import pytest

from AegisKV.demo.completion_order_plan import (
    build_completion_order_plan,
    load_completion_order_plan,
)


def test_completion_order_plan_round_trips_online_causal_order(tmp_path):
    source = tmp_path / "online.json"
    source.write_text(
        json.dumps(
            {
                "policy": "aegiskv",
                "model": "Qwen/Qwen3-32B",
                "dp_size": 4,
                "pp_size": 4,
                "request_count": 3,
                "workflows": [
                    {
                        "workflow_id": "w0",
                        "request_routes": [
                            {"rid": "r2", "completion_index": 2},
                            {"rid": "r1", "completion_index": 1},
                            {"rid": "r3", "completion_index": 3},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = build_completion_order_plan(source)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload), encoding="utf-8")

    assert load_completion_order_plan(
        plan, selected_request_ids={"r1", "r2", "r3"}
    ) == {"r1": 1, "r2": 2, "r3": 3}
    assert payload["calibration_only"] is True
    assert "placement" in payload["causal_boundary"]


def test_completion_order_plan_rejects_missing_selected_request(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibration_only": True,
                "workflows": {
                    "w0": [{"rid": "r1", "completion_index": 1}]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="misses 1 selected requests"):
        load_completion_order_plan(
            plan, selected_request_ids={"r1", "r2"}
        )
