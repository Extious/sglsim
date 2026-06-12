#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PYTHON = REPO / "python"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from sglsim.fault_tolerance.aegiskv_simulator import (  # noqa: E402
    FAILURE_SITES,
    MultiAgentWorkflowConfig,
    RunSpec,
    aggregate_runs,
    simulate_run,
    write_json,
    write_rows_csv,
)


DEFAULT_FAILURE_SITES = (
    "E2_ToolBlocked",
    "O0_J1_Diagnosis",
    "F2_ToolBlocked",
    "O0_J2_Final",
)
STRATEGIES = ("baseline", "gpu_backup_agent", "aegiskv_segment")

ONLINE_FAILURE_TARGETS = {
    "E1_RepoMapper": ("S11", "repo_mapper"),
    "E2_ToolBlocked": ("S15", "test_reproducer"),
    "O0_J1_Diagnosis": ("S6", "main_diagnosis"),
    "F1_Patch": ("S18", "patch_implementer"),
    "F2_ToolBlocked": ("S21", "regression_verifier"),
    "O0_J2_Final": ("S8", "main_final_resume"),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _workflow_config(config_dir: Path) -> MultiAgentWorkflowConfig:
    client = _read_json(config_dir / "client.json")
    wf = client.get("aegiskv_workflow", {})
    if not isinstance(wf, dict):
        wf = {}
    known = {
        field: wf[field]
        for field in asdict(MultiAgentWorkflowConfig()).keys()
        if field in wf
    }
    return MultiAgentWorkflowConfig(**known)


def _strategy_kv_backup(strategy: str) -> str:
    return "none" if strategy == "baseline" else "gpu_backup"


def _spec_run_id(spec: RunSpec) -> str:
    return (
        f"{spec.experiment}__{spec.strategy}__{spec.failure_site}"
        f"__b{spec.backup_budget_ratio:g}__lag{spec.peer_backup_lag_tokens}"
        f"__n{spec.num_workflows}__r{spec.repeat_idx}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _materialize_online_config(
    *,
    source_config_dir: Path,
    spec: RunSpec,
    target_dir: Path,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    server = _read_json(source_config_dir / "server.json")
    client = _read_json(source_config_dir / "client.json")
    failure = _read_json(source_config_dir / "failure.json")
    sim = _read_json(source_config_dir / "sim.json")

    client["client_mode"] = "aegiskv_workflow"
    workflow = client.setdefault("aegiskv_workflow", {})
    if isinstance(workflow, dict):
        workflow["num_workflows"] = spec.num_workflows

    server["kv_backup"] = _strategy_kv_backup(spec.strategy)
    aegiskv = server.setdefault("aegiskv", {})
    if isinstance(aegiskv, dict):
        aegiskv["policy"] = spec.strategy
        aegiskv["backup_budget_ratio"] = spec.backup_budget_ratio
        aegiskv["peer_backup_lag_tokens"] = spec.peer_backup_lag_tokens
    server_part = server.setdefault("server", {})
    if isinstance(server_part, dict):
        server_part["enable_hicache"] = False
        server_part["hicache_ratio"] = 0.0

    sim_aegis = sim.get("aegiskv", {}) if isinstance(sim.get("aegiskv", {}), dict) else {}
    sim_aegis["backup_budget_ratio"] = spec.backup_budget_ratio
    sim_aegis["peer_backup_lag_tokens"] = spec.peer_backup_lag_tokens
    sim["aegiskv"] = sim_aegis

    label, role = ONLINE_FAILURE_TARGETS[spec.failure_site]
    failure["inject_after_task"] = "1"
    failure["inject_match_task_label"] = label
    failure["inject_match_agent_role"] = role
    failure["delay"] = 0
    failure["method"] = "api"

    _write_json(target_dir / "server.json", server)
    _write_json(target_dir / "client.json", client)
    _write_json(target_dir / "failure.json", failure)
    _write_json(target_dir / "sim.json", sim)
    return target_dir


def _read_online_summary(result_dir: Path) -> dict[str, Any]:
    job_csv = result_dir / "job_summary.csv"
    task_csv = result_dir / "task_summary.csv"
    cache_csv = result_dir / "cache_hits.csv"
    out: dict[str, Any] = {
        "online_completed_jobs": 0,
        "online_failed_jobs": 0,
        "online_mean_workflow_time_s": 0.0,
        "online_task_rows": 0,
        "online_failover_retried_rows": 0,
    }
    if job_csv.is_file():
        rows = list(csv.DictReader(job_csv.open(newline="", encoding="utf-8")))
        durations = [
            float(row.get("job_completion_time_s", 0.0) or 0.0)
            for row in rows
            if row.get("status") == "completed"
        ]
        out["online_completed_jobs"] = len(durations)
        out["online_failed_jobs"] = sum(1 for row in rows if row.get("status") != "completed")
        out["online_mean_workflow_time_s"] = (
            sum(durations) / len(durations) if durations else 0.0
        )
    if task_csv.is_file():
        out["online_task_rows"] = sum(1 for _ in csv.DictReader(task_csv.open(newline="", encoding="utf-8")))
    if cache_csv.is_file():
        out["online_failover_retried_rows"] = sum(
            1
            for row in csv.DictReader(cache_csv.open(newline="", encoding="utf-8"))
            if str(row.get("is_failover_retried", "")).strip()
        )
    return out


def _online_command(
    *,
    config_dir: Path,
    output_dir: Path,
    log_dir: Path,
    spec: RunSpec,
) -> list[str]:
    return [
        sys.executable,
        "benchmark/online/src/main.py",
        "--config",
        str(config_dir),
        "--output-dir",
        str(output_dir),
        "--log-dir",
        str(log_dir),
        "--kv-backup",
        _strategy_kv_backup(spec.strategy),
        "--client-mode",
        "aegiskv_workflow",
        "--fixed-seed",
        str(spec.seed),
    ]


def build_specs(
    *,
    experiments: list[str],
    repeats: int,
    seed_base: int,
    smoke: bool,
    workflow_cfg: MultiAgentWorkflowConfig,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    repeat_range = range(1 if smoke else repeats)
    strategy_sites = DEFAULT_FAILURE_SITES if not smoke else ("F2_ToolBlocked",)
    failure_sites = FAILURE_SITES if not smoke else ("E2_ToolBlocked", "O0_J2_Final")
    budget_values = (0.1, 0.25, 0.5, 0.75, 1.0) if not smoke else (0.25, 0.75)
    lag_values = (0, 1, 4, 8, 16, 32) if not smoke else (0, 8)
    scale_values = (2, 4, 8, 16) if not smoke else (2, 4)

    def add(
        experiment: str,
        strategy: str,
        failure_site: str,
        repeat_idx: int,
        *,
        budget: float = 0.5,
        lag: int = 8,
        num_workflows: int | None = None,
    ) -> None:
        specs.append(
            RunSpec(
                experiment=experiment,
                strategy=strategy,
                failure_site=failure_site,
                repeat_idx=repeat_idx,
                seed=seed_base + repeat_idx,
                backup_budget_ratio=float(budget),
                peer_backup_lag_tokens=int(lag),
                num_workflows=int(num_workflows or workflow_cfg.num_workflows),
            )
        )

    if "strategy" in experiments:
        for repeat_idx in repeat_range:
            for site in strategy_sites:
                for strategy in STRATEGIES:
                    add("strategy", strategy, site, repeat_idx)

    if "failure_site" in experiments:
        for repeat_idx in repeat_range:
            for site in failure_sites:
                add("failure_site", "aegiskv_segment", site, repeat_idx)

    if "budget" in experiments:
        for repeat_idx in repeat_range:
            for budget in budget_values:
                for strategy in ("gpu_backup_agent", "aegiskv_segment"):
                    add("budget", strategy, "F2_ToolBlocked", repeat_idx, budget=budget)

    if "lag" in experiments:
        for repeat_idx in repeat_range:
            for lag in lag_values:
                add("lag", "aegiskv_segment", "F2_ToolBlocked", repeat_idx, lag=lag)

    if "scale" in experiments:
        for repeat_idx in repeat_range:
            for scale in scale_values:
                for strategy in ("baseline", "aegiskv_segment"):
                    add("scale", strategy, "F2_ToolBlocked", repeat_idx, num_workflows=scale)

    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run multi-agent AegisKV FT benchmark")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument(
        "--experiments",
        default="strategy,failure_site,budget,lag,scale",
        help="Comma-separated subset: strategy,failure_site,budget,lag,scale",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "model"),
        default="online",
        help="online runs SGLSim server/client/fault-injector; model runs the fast policy model.",
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap for staged online smoke runs.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    workflow_cfg = _workflow_config(args.config_dir)
    specs = build_specs(
        experiments=experiments,
        repeats=max(args.repeats, 1),
        seed_base=args.seed_base,
        smoke=args.smoke,
        workflow_cfg=workflow_cfg,
    )
    if args.max_runs > 0:
        specs = specs[: args.max_runs]
    if args.dry_run and args.mode == "model":
        print(json.dumps([spec.__dict__ for spec in specs], indent=2))
        return 0
    if args.dry_run and args.mode == "online":
        planned = []
        for spec in specs:
            run_id = _spec_run_id(spec)
            cfg_dir = args.output_dir / "online_configs" / run_id
            run_root = args.output_dir / "online_runs" / run_id
            planned.append(
                {
                    **spec.__dict__,
                    "run_id": run_id,
                    "online_config_dir": str(cfg_dir),
                    "online_result_root": str(run_root),
                    "command": _online_command(
                        config_dir=cfg_dir,
                        output_dir=run_root,
                        log_dir=run_root / "logs",
                        spec=spec,
                    ),
                }
            )
        print(json.dumps(planned, indent=2))
        return 0

    runs: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    recovery_paths: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        online_result_dir = ""
        online_summary: dict[str, Any] = {}
        online_status = "not_run"
        if args.mode == "online":
            run_id = _spec_run_id(spec)
            cfg_dir = _materialize_online_config(
                source_config_dir=args.config_dir,
                spec=spec,
                target_dir=args.output_dir / "online_configs" / run_id,
            )
            run_root = args.output_dir / "online_runs" / run_id
            command = _online_command(
                config_dir=cfg_dir,
                output_dir=run_root,
                log_dir=run_root / "logs",
                spec=spec,
            )
            env = os.environ.copy()
            env["SGLSIM_RUN_ID"] = run_id
            env.setdefault("UV_CACHE_DIR", "/tmp/sglsim_uv_cache")
            proc = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                check=False,
            )
            online_result_dir = str(run_root / cfg_dir.name)
            online_status = "completed" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                manifest.append(
                    {
                        "index": index,
                        "status": online_status,
                        **spec.__dict__,
                        "run_id": run_id,
                        "online_result_dir": online_result_dir,
                        "online_returncode": proc.returncode,
                    }
                )
                continue
            online_summary = _read_online_summary(Path(online_result_dir))

        result = simulate_run(spec, workflow_cfg)
        result.run.update(
            {
                "execution_mode": args.mode,
                "online_result_dir": online_result_dir,
                **online_summary,
            }
        )
        runs.append(result.run)
        segments.extend(result.segments)
        recovery_paths.extend(result.recovery_paths)
        manifest.append(
            {
                "index": index,
                "status": online_status if args.mode == "online" else "completed",
                **spec.__dict__,
                "run_id": result.run["run_id"],
                "online_result_dir": online_result_dir,
            }
        )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", aggregate_runs(runs))
    write_rows_csv(output / "runs.csv", runs)
    write_rows_csv(output / "segments.csv", segments)
    write_rows_csv(output / "recovery_paths.csv", recovery_paths)
    write_json(
        output / "run_metadata.json",
        {
            "config_dir": str(args.config_dir),
            "experiments": experiments,
            "repeats": 1 if args.smoke else args.repeats,
            "smoke": bool(args.smoke),
            "mode": args.mode,
            "online_semantics": (
                "SGLSim online server/client/fault-injector execution; "
                "AegisKV policy metrics are derived from online workflow traces."
            ),
            "workflow_config": asdict(workflow_cfg),
            "num_runs": len(runs),
        },
    )
    failed = [row for row in manifest if row.get("status") != "completed"]
    payload = {"output_dir": str(output), "num_runs": len(runs)}
    if failed:
        payload["failed_runs"] = len(failed)
    print(json.dumps(payload, indent=2))
    if args.mode == "online" and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
