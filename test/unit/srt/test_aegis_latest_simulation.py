from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "AegisKV/experiments/latest_design/run_simulation.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "aegiskv_latest_simulation", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latest_simulation_is_deterministic_matched_and_tier_safe() -> None:
    module = _module()
    config_path = SCRIPT.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "seeds": 3,
            "requests_per_seed": 24,
            "bootstrap_resamples": 100,
        }
    )

    first_rows, first_stats = module.run(config)
    second_rows, second_stats = module.run(config)
    audit = module._audit(first_rows, config)
    summary = module._summarize(first_rows, first_stats, config)

    assert first_rows == second_rows
    assert first_stats == second_stats
    assert audit["passed"] is True
    assert audit["matched_request_failure_keys"] is True
    assert audit["restored_prefix_partition_violations"] == []
    assert len(summary["policies"]) == 5
    latest = next(
        row for row in summary["policies"] if row["policy"] == "aegiskv_latest"
    )
    assert latest["copy_token_amplification"] > 0.0
    assert latest["mean_restored_fraction"] > 0.0


def test_latest_simulation_artifact_hash_is_the_emitted_csv(
    tmp_path: Path,
) -> None:
    module = _module()
    rows = [
        {
            "seed": 0,
            "request_id": "request-0",
            "policy": "recompute_only",
            "source_tokens": 8,
            "restored_tokens": 0,
            "recompute_tokens": 8,
            "l1_tokens": 0,
            "l2_tokens": 0,
        }
    ]
    output = tmp_path / "raw_failures.csv"
    module._write_csv(output, rows)

    assert module._file_sha256(output) == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


def test_latest_synthetic_slack_is_built_from_visible_residual_dag() -> None:
    module = _module()
    request = module.ScenarioRequest(
        request_id="request",
        total_tokens=128,
        slack_budget_ms=12.5,
        chunks=(),
    )

    ticket = module._request_slack_ticket(request)
    fields = ticket.to_simulation_fields()

    assert ticket.slack_ms == 12.5
    assert fields["aegis_slack_semantics"] == (
        module.RESIDUAL_DAG_SLACK_SEMANTICS
    )
    assert fields["aegis_slack_residual_dag_visible"] is True
    assert fields["aegis_slack_metadata_only"] is True
