#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = SGLSIM_LARGE_ROOT / "results" / "agentic_workflows"
OUTPUT_ROOT = RESULTS_ROOT / "sglsim-staged-13131-alpha-online-sweep-20260702"
OUTPUT_MANIFEST = OUTPUT_ROOT / "agentic_workflow_manifest.json"

SCENARIOS = (
    ("online_alpha_dp8_baseload_lowrisk", "base load", 0.003),
    ("online_alpha_dp8_baseload_highrisk", "base load", 0.03),
    ("online_alpha_dp8_highload_lowrisk", "high load", 0.003),
    ("online_alpha_dp8_highload_highrisk", "high load", 0.03),
)


def _run_id(scenario: str) -> str:
    return scenario.replace("_", "-") + "-r1"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _latest_run_dir(run_id: str) -> Path:
    run_root = RESULTS_ROOT / run_id / "runs"
    candidates = sorted(
        path
        for path in run_root.glob("trial_*/*")
        if path.is_dir() and (path / "request_details.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"no request_details.csv under {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_manifest() -> dict:
    plan = []
    for scenario, load_level, failure_rate in SCENARIOS:
        run_id = _run_id(scenario)
        source_manifest = _read_json(RESULTS_ROOT / run_id / "agentic_workflow_manifest.json")
        source_plan = source_manifest.get("plan") if isinstance(source_manifest.get("plan"), list) else []
        source_row = source_plan[0] if source_plan and isinstance(source_plan[0], dict) else {}
        run_dir = _latest_run_dir(run_id)
        plan.append(
            {
                "scenario": scenario,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "load_level": load_level,
                "sweep_point": scenario.removeprefix("online_alpha_dp8_"),
                "failure_rate_per_window": failure_rate,
                "alpha_policy_path": source_row.get("alpha_policy_path", ""),
                "aegis_critical_pressure": source_row.get("aegis_critical_pressure", ""),
                "workflow_count": source_row.get("workflow_count", ""),
                "max_concurrency": source_row.get("max_concurrency", ""),
                "dp_size": source_row.get("dp_size", ""),
                "stage_context_lens": source_row.get("stage_context_lens", []),
            }
        )
    return {
        "run_id": OUTPUT_ROOT.name,
        "scenario": "online_alpha_dp8_mdp_sweep",
        "dry_run": False,
        "runs": len(plan),
        "output_root": str(OUTPUT_ROOT),
        "plan": plan,
        "evidence_tier": "sglsim-agentic-online-alpha-sweep",
    }


def main() -> int:
    manifest = build_manifest()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(OUTPUT_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
