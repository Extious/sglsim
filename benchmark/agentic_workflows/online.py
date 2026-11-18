from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_BASE_CONFIG_DIR,
    AgenticWorkflowScenario,
    get_scenario,
    materialize_synthesis_config,
    scenario_plan_row,
)
from .failure import FailureInjectionPlan, failure_plan_from_scenario, validate_failure_plan
from .workload import WorkflowShapeSummary, summarize_workflow_shape


@dataclass(frozen=True)
class OnlineExperimentPlan:
    scenario: str
    trial: int
    config_dir: str
    dp_size: int
    port: int
    max_concurrency: int
    workload: WorkflowShapeSummary
    failure: FailureInjectionPlan
    synthesis_entrypoint: str = "benchmark.synthesis_requests.run.run_experiment"
    result_artifacts: tuple[str, ...] = (
        "simulator_config.json",
        "request_details.csv",
        "metrics.json",
        "server.log",
        "server.stderr.log",
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["workload"] = self.workload.to_row()
        payload["failure"] = self.failure.to_row()
        return payload


def build_online_experiment_plan(
    scenario: AgenticWorkflowScenario,
    *,
    output_dir: Path,
    base_config_dir: Path = DEFAULT_BASE_CONFIG_DIR,
    trial: int = 0,
) -> OnlineExperimentPlan:
    config_dir = materialize_synthesis_config(
        scenario,
        base_config_dir=base_config_dir,
        output_dir=output_dir,
        trial=trial,
    )
    row = scenario_plan_row(scenario, config_dir=config_dir, trial=trial)
    failure = failure_plan_from_scenario(scenario)
    validate_failure_plan(failure)
    return OnlineExperimentPlan(
        scenario=scenario.name,
        trial=int(trial),
        config_dir=str(config_dir),
        dp_size=int(row["dp_size"]),
        port=int(row["port"]),
        max_concurrency=int(row["max_concurrency"]),
        workload=summarize_workflow_shape(scenario),
        failure=failure,
    )


def write_online_experiment_plan(path: Path, plan: OnlineExperimentPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def build_named_online_plan(
    scenario_name: str,
    *,
    output_dir: Path,
    base_config_dir: Path = DEFAULT_BASE_CONFIG_DIR,
    trial: int = 0,
) -> OnlineExperimentPlan:
    return build_online_experiment_plan(
        get_scenario(scenario_name),
        output_dir=output_dir,
        base_config_dir=base_config_dir,
        trial=trial,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize an agentic-workflow online experiment plan."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config-dir", type=Path, default=DEFAULT_BASE_CONFIG_DIR)
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args(argv)

    plan = build_named_online_plan(
        args.scenario,
        output_dir=args.output_dir,
        base_config_dir=args.base_config_dir,
        trial=args.trial,
    )
    plan_path = Path(args.output_dir) / "online_experiment_plan.json"
    write_online_experiment_plan(plan_path, plan)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
