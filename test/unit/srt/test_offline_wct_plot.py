from __future__ import annotations

import csv
import json

from AegisKV.demo.plot_offline_wct_cdf import render


def test_plot_offline_wct_cdf_writes_figure_and_paired_samples(tmp_path):
    identity = {
        "model": "Qwen/Qwen3-32B",
        "dp_size": 4,
        "pp_size": 4,
        "context_length": 32768,
        "base_server_config": "/config/server.json",
        "trace": "/trace/agentic.jsonl",
        "route_plan": "/config/routes.json",
        "failure_manifest": "/config/failures.json",
    }
    specs = []
    for policy, values in (
        ("aegiskv", (9.0, 19.0)),
        ("l1_only", (9.5, 19.5)),
        ("no_backup", (10.0, 20.0)),
    ):
        path = tmp_path / f"{policy}.json"
        path.write_text(
            json.dumps(
                {
                    "policy": policy,
                    "identity": identity,
                    "failure_count_injected": 1,
                    "evidence_tier": "sglsim-offline-unvalidated",
                    "workflows": [
                        {"workflow_id": "wf-0", "wct_s": values[0]},
                        {"workflow_id": "wf-1", "wct_s": values[1]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        specs.append(f"{policy}={path}")

    output = render(specs, tmp_path / "wct-cdf.png")

    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
    samples = output.with_name("wct-cdf-samples.csv")
    assert samples.is_file()
    with samples.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["policy"] for row in rows} == {
        "aegiskv",
        "l1_only",
        "no_backup",
    }
