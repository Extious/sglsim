#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_OUTPUT_DIR = EXP_DIR / "plans" / "strict_final_recovery_matrix"
SMOKE_SBATCH = "slurm/aegiskv_dp2_failure_smoke.sbatch"


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    prefetch_policy: str
    materialization_target: str
    latency_target: str
    note: str


@dataclass(frozen=True)
class RunSpec:
    pair_id: str
    run_label: str
    repeat: int
    role: str
    prefetch_policy: str
    port: int
    remote_backup_port: int
    env: dict[str, str]
    submit_command: str
    expected_result_dir: str


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    case_id: str
    repeat: int
    prefetch_policy: str
    invisible_run_label: str
    visible_run_label: str
    sync_pair_command: str
    physical_restore_command: str
    acceptance_criteria: list[str]


CASES = [
    MatrixCase(
        case_id="same_write_best_effort",
        prefetch_policy="best_effort",
        materialization_target="partial_async_or_full_materialization",
        latency_target="visible p99 / invisible p99 < 1.0 in most repeats",
        note=(
            "Both runs pay remote_backup write cost; only the visible run can "
            "look up Host pages during failover. This is the recovery-benefit "
            "candidate."
        ),
    ),
    MatrixCase(
        case_id="same_write_wait_complete",
        prefetch_policy="wait_complete",
        materialization_target="full_prefetch_materialized",
        latency_target="materialization correctness; latency may be slower",
        note=(
            "Both runs pay remote_backup write cost; wait_complete makes the "
            "full Host restore path auditable even if it is slower."
        ),
    ),
]


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def remote_submit_command(
    *,
    remote_host: str,
    remote_workdir: str,
    env: dict[str, str],
) -> str:
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    remote = f"cd {remote_workdir} && {env_prefix} sbatch {SMOKE_SBATCH}"
    return shell_join(["ssh", remote_host, remote])


def build_run_specs(
    *,
    repeats: int = 3,
    base_port: int = 32820,
    base_remote_backup_port: int = 33820,
    remote_host: str = "hkbugpumanode",
    remote_workdir: str = "~/aegiskv_sglang_small",
    num_requests: int = 4,
    max_tokens: int = 512,
    prompt_repeat: int = 256,
    failure_delay_s: int = 12,
    ready_timeout_s: int = 2400,
    request_timeout_s: int = 900,
    remote_backup_buffer_gb: int = 16,
    unique_request_prefix: int = 0,
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    job_index = 0
    for case in CASES:
        for repeat in range(1, repeats + 1):
            pair_id = f"strict_final_{case.case_id}_r{repeat}"
            for role, lookup_disabled in (
                ("host_invisible_lookup_disabled", "1"),
                ("host_visible_lookup_enabled", "0"),
            ):
                run_label = f"{pair_id}_{role}"
                env = {
                    "RUN_LABEL": run_label,
                    "PORT": str(base_port + job_index),
                    "REMOTE_BACKUP_PORT": str(base_remote_backup_port + job_index),
                    "ENABLE_REMOTE_BACKUP": "1",
                    "DISABLE_FAILOVER_REMOTE_LOOKUP": lookup_disabled,
                    "HICACHE_STORAGE_PREFETCH_POLICY": case.prefetch_policy,
                    "REMOTE_BACKUP_REQUEST_AWARE": "1",
                    "REMOTE_BACKUP_BUFFER_GB": str(remote_backup_buffer_gb),
                    "NUM_REQUESTS": str(num_requests),
                    "MAX_TOKENS": str(max_tokens),
                    "PROMPT_REPEAT": str(prompt_repeat),
                    "UNIQUE_REQUEST_PREFIX": str(unique_request_prefix),
                    "FAILURE_DELAY_S": str(failure_delay_s),
                    "READY_TIMEOUT_S": str(ready_timeout_s),
                    "REQUEST_TIMEOUT_S": str(request_timeout_s),
                }
                runs.append(
                    RunSpec(
                        pair_id=pair_id,
                        run_label=run_label,
                        repeat=repeat,
                        role=role,
                        prefetch_policy=case.prefetch_policy,
                        port=int(env["PORT"]),
                        remote_backup_port=int(env["REMOTE_BACKUP_PORT"]),
                        env=env,
                        submit_command=remote_submit_command(
                            remote_host=remote_host,
                            remote_workdir=remote_workdir,
                            env=env,
                        ),
                        expected_result_dir="results/dp2_failure_smoke_<slurm_job_id>",
                    )
                )
                job_index += 1
    return runs


def build_pair_specs(runs: list[RunSpec]) -> list[PairSpec]:
    by_pair: dict[str, list[RunSpec]] = {}
    for run in runs:
        by_pair.setdefault(run.pair_id, []).append(run)

    pairs: list[PairSpec] = []
    for pair_id in sorted(by_pair):
        pair_runs = sorted(by_pair[pair_id], key=lambda item: item.role)
        invisible = next(
            run for run in pair_runs if run.role == "host_invisible_lookup_disabled"
        )
        visible = next(run for run in pair_runs if run.role == "host_visible_lookup_enabled")
        case = next(case for case in CASES if case.prefetch_policy == visible.prefetch_policy and case.case_id in pair_id)
        invisible_job = f"<{invisible.run_label}_job_id>"
        visible_job = f"<{visible.run_label}_job_id>"
        output_dir = RESULTS_DIR / pair_id
        sync_command = shell_join(
            [
                "python",
                str(EXP_DIR / "scripts" / "sync_failure_pair_from_remote.py"),
                "--invisible-job-id",
                invisible_job,
                "--visible-job-id",
                visible_job,
                "--run-id",
                pair_id,
                "--same-write-cost-control",
                "--prefetch-policy",
                visible.prefetch_policy,
                "--title",
                f"Strict-final recovery pair {pair_id}",
                "--note",
                case.note,
            ]
        )
        restore_output = RESULTS_DIR / "physical_restore_evidence" / pair_id
        physical_command = shell_join(
            [
                "python",
                str(EXP_DIR / "scripts" / "extract_physical_restore_evidence.py"),
                "--result-dir",
                f"invisible={RESULTS_DIR / ('dp2_failure_smoke_' + invisible_job)}",
                "--result-dir",
                f"visible={RESULTS_DIR / ('dp2_failure_smoke_' + visible_job)}",
                "--output-csv",
                str(restore_output / "physical_restore_runs.csv"),
                "--output-json",
                str(restore_output / "physical_restore_runs.json"),
                "--output-md",
                str(restore_output / "physical_restore_runs.md"),
            ]
        )
        if visible.prefetch_policy == "wait_complete":
            criteria = [
                "both runs have success_count == request_count",
                "both runs have pre_failover_backed_up_tokens > 0",
                "visible run reaches physical_restore_state == full_prefetch_materialized",
                "visible run has prefetch_io_completion_ratio == 1.0",
            ]
        else:
            criteria = [
                "both runs have success_count == request_count",
                "both runs have pre_failover_backed_up_tokens > 0",
                "visible run has max_data_pages > 0 and prefetch_accepted_count > 0",
                "visible run reaches partial_async_materialized or full_prefetch_materialized",
                "visible_retry_latency_norm_vs_invisible < 1.0 for most repeats",
            ]
        pairs.append(
            PairSpec(
                pair_id=pair_id,
                case_id=case.case_id,
                repeat=invisible.repeat,
                prefetch_policy=visible.prefetch_policy,
                invisible_run_label=invisible.run_label,
                visible_run_label=visible.run_label,
                sync_pair_command=sync_command,
                physical_restore_command=physical_command,
                acceptance_criteria=criteria,
            )
        )
    return pairs


def build_plan(**kwargs: Any) -> dict[str, Any]:
    runs = build_run_specs(**kwargs)
    pairs = build_pair_specs(runs)
    aggregate_command = shell_join(
        [
            "python",
            str(EXP_DIR / "scripts" / "aggregate_failure_visibility_pairs.py"),
            "--results-dir",
            str(RESULTS_DIR),
            "--output-dir",
            str(RESULTS_DIR / "failure_visibility_pair_aggregate"),
        ]
    )
    return {
        "plan_id": "strict_final_recovery_matrix",
        "purpose": (
            "Collect real SGLang Host recovery data-plane evidence that can "
            "upgrade metadata-only SGLSim policy rows, without changing the "
            "metadata-only SGLSim runtime."
        ),
        "cases": [asdict(case) for case in CASES],
        "runs": [asdict(run) for run in runs],
        "pairs": [asdict(pair) for pair in pairs],
        "postprocess_commands": {
            "aggregate_failure_visibility_pairs": aggregate_command,
        },
        "strict_final_boundary": (
            "These runs are a real-system recovery-path matrix. They do not by "
            "themselves turn SGLSim restore rows into physical_tensor_copy rows; "
            "the P0 audit should only pass after real recovery evidence is "
            "connected to the final policy claims."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_run_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in plan["runs"]:
        row = {
            key: value
            for key, value in run.items()
            if key not in {"env"}
        }
        env = run.get("env") or {}
        for key, value in env.items():
            row[f"env_{key.lower()}"] = value
        rows.append(row)
    return rows


def flatten_pair_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in plan["pairs"]:
        row = dict(pair)
        row["acceptance_criteria"] = "; ".join(pair.get("acceptance_criteria") or [])
        rows.append(row)
    return rows


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Strict-Final Real-System Recovery Matrix",
        "",
        plan["purpose"],
        "",
        "This plan intentionally keeps SGLSim metadata-only rows separate from real",
        "SGLang recovery-path evidence. Passing this matrix supplies real-system",
        "Host restore evidence; it must not be represented as SGLSim physical tensor",
        "copy rows unless the final audit is explicitly updated to consume that",
        "evidence.",
        "",
        "## Submit Commands",
        "",
    ]
    for run in plan["runs"]:
        lines.extend(
            [
                f"### {run['run_label']}",
                "",
                "```bash",
                run["submit_command"],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Pair Sync And Parsing",
            "",
            "Replace each `<..._job_id>` placeholder with the Slurm job id returned by",
            "`sbatch`.",
            "",
        ]
    )
    for pair in plan["pairs"]:
        lines.extend(
            [
                f"### {pair['pair_id']}",
                "",
                "Acceptance criteria:",
                "",
            ]
        )
        for item in pair["acceptance_criteria"]:
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "```bash",
                pair["sync_pair_command"],
                pair["physical_restore_command"],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Aggregate",
            "",
            "```bash",
            plan["postprocess_commands"]["aggregate_failure_visibility_pairs"],
            "```",
            "",
            "## Boundary",
            "",
            plan["strict_final_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the strict-final SGLang recovery-path experiment matrix."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=32820)
    parser.add_argument("--base-remote-backup-port", type=int, default=33820)
    parser.add_argument("--remote-host", default="hkbugpumanode")
    parser.add_argument("--remote-workdir", default="~/aegiskv_sglang_small")
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-repeat", type=int, default=256)
    parser.add_argument("--failure-delay-s", type=int, default=12)
    parser.add_argument("--remote-backup-buffer-gb", type=int, default=16)
    parser.add_argument("--unique-request-prefix", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    plan = build_plan(
        repeats=args.repeats,
        base_port=args.base_port,
        base_remote_backup_port=args.base_remote_backup_port,
        remote_host=args.remote_host,
        remote_workdir=args.remote_workdir,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        prompt_repeat=args.prompt_repeat,
        failure_delay_s=args.failure_delay_s,
        remote_backup_buffer_gb=args.remote_backup_buffer_gb,
        unique_request_prefix=args.unique_request_prefix,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "strict_final_recovery_matrix.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "strict_final_recovery_runs.csv", flatten_run_rows(plan))
    write_csv(args.output_dir / "strict_final_recovery_pairs.csv", flatten_pair_rows(plan))
    (args.output_dir / "strict_final_recovery_matrix.md").write_text(
        render_markdown(plan),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "runs": len(plan["runs"]),
                "pairs": len(plan["pairs"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
