#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[6]
EXP_DIR = ROOT / "benchmark" / "AegisKV" / "draft" / "experiments" / "sglsim_large"
DEFAULT_CONFIG_DIR = EXP_DIR / "config_failover_replay"
DEFAULT_RESULTS_ROOT = EXP_DIR / "results" / "failover_replay_matrix"

for path in (ROOT, ROOT / "python"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from benchmark.synthesis_requests.run import run_experiment  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_optional_int(payload: dict[str, Any], key: str, value: int | None) -> None:
    if value is not None:
        payload[key] = value


def trial_config(
    *,
    base_config_dir: Path,
    output_dir: Path,
    trial: int,
    seed_base: int,
    start_port: int,
    groups: int | None,
    prompts_per_group: int | None,
    turns: int | None,
    max_concurrency: int | None,
    system_prompt_len: int | None,
    question_len: int | None,
    output_len: int | None,
    dp_size: int | None,
    failover_retry_rank_offset: int | None,
) -> Path:
    client = load_json(base_config_dir / "client.json")
    server = load_json(base_config_dir / "server.json")

    client["seed"] = seed_base + trial
    apply_optional_int(client, "gsp_num_groups", groups)
    apply_optional_int(client, "gsp_prompts_per_group", prompts_per_group)
    apply_optional_int(client, "gsp_num_turns", turns)
    apply_optional_int(client, "max_concurrency", max_concurrency)
    apply_optional_int(client, "gsp_system_prompt_len", system_prompt_len)
    apply_optional_int(client, "gsp_question_len", question_len)
    apply_optional_int(client, "gsp_output_len", output_len)
    apply_optional_int(
        client,
        "gsp_failover_retry_rank_offset",
        failover_retry_rank_offset,
    )

    server["port"] = start_port + trial
    if dp_size is not None:
        scheduler = dict(server.get("scheduler") or {})
        scheduler["dp_size"] = dp_size
        server["scheduler"] = scheduler
    server["experiment_name"] = (
        f"{server.get('experiment_name', 'aegiskv-sglsim-failover-replay')}"
        f"-matrix-t{trial:03d}"
    )

    config_dir = output_dir / "configs" / f"trial_{trial:03d}"
    write_json(config_dir / "client.json", client)
    write_json(config_dir / "server.json", server)
    return config_dir


def collect_metrics(run_dir: Path, output_csv: Path, label: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(EXP_DIR / "scripts" / "collect_synthesis_metrics.py"),
            "--run-dir",
            str(run_dir),
            "--output-csv",
            str(output_csv),
            "--label",
            label,
            "--note",
            (
                "failover replay matrix trial; retry rows are real HTTP requests "
                "after scheduler subprocess kill, with runtime HostBackup and "
                "peer GPUReplica directory lookup"
            ),
        ],
        cwd=ROOT,
        check=True,
    )


def derive_candidates(run_dirs: list[Path], output_dir: Path, run_id: str) -> Path:
    command = [
        sys.executable,
        str(EXP_DIR / "scripts" / "derive_policy_candidates_from_synthesis.py"),
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
    ]
    for run_dir in run_dirs:
        command.extend(["--run-dir", str(run_dir)])
    subprocess.run(command, cwd=ROOT, check=True)
    return output_dir / "candidate_measured"


def write_run_index(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated SGLSim hard-failover replay trials and derive candidate figure CSVs."
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--start-port", type=int, default=30200)
    parser.add_argument("--groups", type=int, default=None)
    parser.add_argument("--prompts-per-group", type=int, default=None)
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--system-prompt-len", type=int, default=None)
    parser.add_argument("--question-len", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--dp-size", type=int, default=None)
    parser.add_argument("--failover-retry-rank-offset", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--skip-derive", action="store_true")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")

    run_id = args.run_id or "sglsim-failover-replay-matrix-" + datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    output_dir = args.results_root.resolve() / run_id
    run_rows: list[dict[str, object]] = []
    run_dirs: list[Path] = []

    for trial in range(args.runs):
        config_dir = trial_config(
            base_config_dir=args.config_dir.resolve(),
            output_dir=output_dir,
            trial=trial,
            seed_base=args.seed_base,
            start_port=args.start_port,
            groups=args.groups,
            prompts_per_group=args.prompts_per_group,
            turns=args.turns,
            max_concurrency=args.max_concurrency,
            system_prompt_len=args.system_prompt_len,
            question_len=args.question_len,
            output_len=args.output_len,
            dp_size=args.dp_size,
            failover_retry_rank_offset=args.failover_retry_rank_offset,
        )
        trial_results_root = output_dir / "runs" / f"trial_{trial:03d}"
        run_dir = run_experiment(
            config_dir,
            trial_results_root,
            log_level=args.log_level,
        )
        run_dirs.append(run_dir)
        metrics_csv = run_dir / "sglsim_failover_replay_summary.csv"
        collect_metrics(
            run_dir,
            metrics_csv,
            label=f"sglsim_failover_replay_matrix_t{trial:03d}",
        )
        run_rows.append(
            {
                "trial": trial,
                "seed": args.seed_base + trial,
                "port": args.start_port + trial,
                "config_dir": str(config_dir),
                "run_dir": str(run_dir),
                "metrics_csv": str(metrics_csv),
            }
        )

    candidate_csv_dir = ""
    if not args.skip_derive:
        candidate_csv_dir = str(
            derive_candidates(
                run_dirs,
                output_dir / "derived_policy_candidates",
                run_id,
            )
        )

    write_run_index(output_dir / "matrix_runs.csv", run_rows)
    summary = {
        "run_id": run_id,
        "runs": args.runs,
        "run_dirs": [str(path) for path in run_dirs],
        "candidate_csv_dir": candidate_csv_dir,
        "evidence_tier": "sglsim-synthesis-derived",
        "note": (
            "Repeated hard-failover replay matrix. Candidate CSVs remain "
            "preliminary until physical KV movement and final policy sweeps exist."
        ),
    }
    write_json(output_dir / "matrix_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
