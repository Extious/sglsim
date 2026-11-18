from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_BASE_CONFIG_DIR, DEFAULT_RESULTS_ROOT, REPO_ROOT, SCENARIOS
from .run import execute_scenario
from .summary import write_summary_json


SUITE_CONFIG = Path(__file__).resolve().parent / "config" / "aegiskv_rebuild_suite.json"
BEAMER_DIR = REPO_ROOT / "benchmark" / "AegisKV" / "beamer"


@dataclass(frozen=True)
class RebuildStep:
    stage: str
    scenario: str
    run_id: str
    start_port: int


def _load_suite_config(path: Path = SUITE_CONFIG) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _steps_from_config(config: dict[str, Any]) -> list[RebuildStep]:
    steps: list[RebuildStep] = []
    generated_stages = {"motivation", "alpha_dp8", "policy_100k"}
    for row in config.get("stages", []):
        stage = str(row["stage"])
        scenario = str(row["scenario"])
        if stage in generated_stages or scenario not in SCENARIOS:
            continue
        steps.append(
            RebuildStep(
                stage=stage,
                scenario=scenario,
                run_id=str(row["run_id"]),
                start_port=int(row["start_port"]),
            )
        )

    generated_specs = [
        (
            "motivation",
            [
                name
                for name in sorted(SCENARIOS)
                if name.startswith("motivation_main_subagent_dp8_")
            ],
        ),
        (
            "alpha_dp8",
            [
                name
                for name in sorted(SCENARIOS)
                if name.startswith("alpha_tradeoff_dp8_")
            ],
        ),
        (
            "policy_100k",
            [
                name
                for name in sorted(SCENARIOS)
                if name.startswith("staged_13131_100k_context_")
            ],
        ),
    ]
    for stage, scenario_names in generated_specs:
        for name in scenario_names:
            scenario = SCENARIOS[name]
            steps.append(
                RebuildStep(
                    stage=stage,
                    scenario=name,
                    run_id=name.replace("_", "-") + "-r1",
                    start_port=int(scenario.start_port),
                )
            )
    return steps


def _figure_scripts_from_config(config: dict[str, Any]) -> list[Path]:
    return [REPO_ROOT / str(path) for path in config.get("figure_scripts", [])]


def _select_steps(steps: list[RebuildStep], stages: set[str]) -> list[RebuildStep]:
    if not stages:
        return steps
    return [step for step in steps if step.stage in stages]


def _normalize_stages(stages: list[str] | None) -> set[str]:
    aliases = {"alpha": "alpha_dp8", "policy": "policy_100k"}
    return {aliases.get(stage, stage) for stage in stages or []}


def _clean_run_dirs(output_root: Path, steps: list[RebuildStep]) -> list[str]:
    removed: list[str] = []
    for step in steps:
        run_root = output_root / step.run_id
        if run_root.exists():
            shutil.rmtree(run_root)
            removed.append(str(run_root))
    return removed


def _run_figure_scripts(scripts: list[Path]) -> list[str]:
    rendered: list[str] = []
    for script in scripts:
        subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT, check=True)
        rendered.append(str(script))
    return rendered


def _compile_beamer() -> None:
    subprocess.run(["make", "-B", "LATEX=lualatex"], cwd=BEAMER_DIR, check=True)


def execute_rebuild(
    *,
    stages: set[str],
    output_root: Path = DEFAULT_RESULTS_ROOT,
    base_config_dir: Path = DEFAULT_BASE_CONFIG_DIR,
    runs: int = 1,
    clean: bool = False,
    dry_run: bool = False,
    render_figures: bool = True,
    compile_beamer: bool = False,
    log_level: str = "warning",
) -> dict[str, Any]:
    config = _load_suite_config()
    steps = _select_steps(_steps_from_config(config), stages)
    output_root = Path(output_root).resolve()
    removed = _clean_run_dirs(output_root, steps) if clean else []
    manifests: list[dict[str, Any]] = []

    for step in steps:
        manifests.append(
            execute_scenario(
                scenario_name=step.scenario,
                output_root=output_root,
                run_id=step.run_id,
                base_config_dir=base_config_dir,
                runs=runs,
                start_port=step.start_port,
                log_level=log_level,
                dry_run=dry_run,
            )
        )

    rendered: list[str] = []
    if render_figures and not dry_run:
        rendered = _run_figure_scripts(_figure_scripts_from_config(config))
    if compile_beamer and not dry_run:
        _compile_beamer()

    summary = {
        "rebuild_id": "aegiskv-rebuild-" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "suite_config": str(SUITE_CONFIG),
        "output_root": str(output_root),
        "stages": sorted(stages) if stages else sorted({step.stage for step in steps}),
        "runs": int(runs),
        "clean": bool(clean),
        "dry_run": bool(dry_run),
        "removed": removed,
        "steps": [asdict(step) for step in steps],
        "manifests": manifests,
        "rendered_scripts": rendered,
        "beamer_pdf": str(BEAMER_DIR / "presentation.pdf") if compile_beamer else "",
    }
    write_summary_json(output_root / "aegiskv_rebuild_manifest.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the AegisKV beamer online evidence for requirements 2a-2d."
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=["motivation", "alpha_dp8", "alpha", "policy_100k", "policy"],
        help="Run one stage only; repeat to run multiple stages. Default: all.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--base-config-dir", type=Path, default=DEFAULT_BASE_CONFIG_DIR)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-render-figures", action="store_true")
    parser.add_argument("--compile-beamer", action="store_true")
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args(argv)

    summary = execute_rebuild(
        stages=_normalize_stages(args.stage),
        output_root=args.results_root,
        base_config_dir=args.base_config_dir,
        runs=args.runs,
        clean=bool(args.clean),
        dry_run=bool(args.dry_run),
        render_figures=not bool(args.no_render_figures),
        compile_beamer=bool(args.compile_beamer),
        log_level=args.log_level,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
