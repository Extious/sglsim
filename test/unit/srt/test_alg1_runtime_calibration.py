import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "AegisKV"
    / "presentation"
    / "figures"
    / "scripts"
    / "calibrate_alg1_slack.py"
)
SPEC = importlib.util.spec_from_file_location("calibrate_alg1_slack", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _workflow_ids(split, count):
    selected = []
    index = 0
    while len(selected) < count:
        workflow_id = f"wf-{index:04d}"
        if MODULE._split(workflow_id) == split:
            selected.append(workflow_id)
        index += 1
    return selected


def _runtime_rows():
    rows = []
    coefficients = {
        "request_count": 12.0,
        "new_prefill_tokens": 0.4,
        "output_tokens": 3.0,
    }
    workflow_ids = (
        _workflow_ids("fit", 6)
        + _workflow_ids("calibration", 3)
        + _workflow_ids("evaluation", 3)
    )
    for index, workflow_id in enumerate(workflow_ids):
        agents = [
            ("short", 1, 100 + index * 3, 8 + index),
            ("long", 2, 320 + index * 7, 24 + index * 2),
        ]
        durations = {
            agent: (
                request_count * coefficients["request_count"]
                + prefill * coefficients["new_prefill_tokens"]
                + output * coefficients["output_tokens"]
            )
            for agent, request_count, prefill, output in agents
        }
        barrier = max(durations.values())
        for agent, request_count, prefill, output in agents:
            rows.append(
                MODULE.RuntimeRow(
                    source="formal.json",
                    policy="aegiskv",
                    workflow_id=workflow_id,
                    stage_id=2,
                    agent_id=agent,
                    stage_width=2,
                    request_count=request_count,
                    new_prefill_tokens=prefill,
                    output_tokens=output,
                    actual_duration_ms=durations[agent],
                    actual_slack_ms=barrier - durations[agent],
                    evidence_tier="sglsim-online-measured",
                    model="Qwen/Qwen3-32B",
                    dp_size=4,
                    pp_size=4,
                    context_length=32768,
                )
            )
    return rows


def test_runtime_calibrator_uses_workflow_disjoint_gate():
    artifact, audit = MODULE.build_artifact(
        _runtime_rows(),
        target_coverage=0.90,
        critical_threshold_ms=5.0,
        minimum_split_workflows=(6, 3, 3),
        expected_model="Qwen/Qwen3-32B",
        expected_dp_size=4,
        expected_pp_size=4,
        expected_context_length=32768,
    )
    assert artifact["semantics"] == "fixed_dag_cpm_slack_v2"
    assert artifact["evaluation_gate"]["passed"]
    assert artifact["evaluation"]["critical_recall"] == 1.0
    assert artifact["evaluation"]["observed_lower_coverage"] == 1.0
    assert {row["split"] for row in audit} == {
        "fit",
        "calibration",
        "evaluation",
    }


def test_runtime_calibrator_replays_only_causal_progress_prefixes():
    model = {
        "request_setup_ms": 1.0,
        "prefill_ms_per_token": 0.0,
        "decode_ms_per_token": 0.0,
        "lower_scale": 1.0,
        "upper_scale": 1.0,
    }
    common = {
        "source": "formal.json",
        "policy": "no_backup",
        "workflow_id": "wf-progress",
        "stage_id": 2,
        "stage_width": 2,
        "request_count": 4,
        "new_prefill_tokens": 0,
        "output_tokens": 0,
        "actual_slack_ms": 0.0,
        "evidence_tier": "sglsim-online-measured",
        "model": "Qwen/Qwen3-32B",
        "dp_size": 4,
        "pp_size": 4,
        "context_length": 32768,
    }
    fast = MODULE.RuntimeRow(
        **common,
        agent_id="fast",
        actual_duration_ms=8.0,
        progress_observations=tuple(
            MODULE.RuntimeProgressObservation(
                request_id=f"a{index}",
                completion_elapsed_ms=elapsed,
                new_prefill_tokens=0,
                output_tokens=0,
            )
            for index, elapsed in enumerate((2.0, 4.0, 6.0, 8.0))
        ),
    )
    lagging = MODULE.RuntimeRow(
        **common,
        agent_id="lagging",
        actual_duration_ms=40.0,
        progress_observations=tuple(
            MODULE.RuntimeProgressObservation(
                request_id=f"b{index}",
                completion_elapsed_ms=elapsed,
                new_prefill_tokens=0,
                output_tokens=0,
            )
            for index, elapsed in enumerate((10.0, 20.0, 30.0, 40.0))
        ),
    )
    critical, trigger_ms, slack_ms = MODULE._online_progress_critical_agents(
        [fast, lagging], model=model
    )
    assert critical == {"lagging"}
    assert trigger_ms == {"lagging": 30.0}
    assert slack_ms == {"fast": 32.0, "lagging": 0.0}
