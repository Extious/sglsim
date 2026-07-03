#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[6]
EXP_DIR = ROOT / "benchmark" / "AegisKV" / "draft" / "experiments" / "sglang_small"
RESULTS_DIR = EXP_DIR / "results"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def scp_options_from_ssh_command(ssh_command: str) -> list[str]:
    if not ssh_command:
        return []
    parts = shlex.split(ssh_command)
    if not parts:
        return []
    if Path(parts[0]).name != "ssh":
        raise RuntimeError(
            "scp fallback expects --rsync-rsh to be an ssh command when rsync is unavailable"
        )
    return parts[1:]


def sync_path(remote: str, local: Path, *, rsync_rsh: str = "") -> None:
    if shutil.which("rsync"):
        local.parent.mkdir(parents=True, exist_ok=True)
        command = ["rsync", "-a"]
        if rsync_rsh:
            command.extend(["-e", rsync_rsh])
        command.extend([remote, str(local)])
        run(command)
        return

    scp_options = scp_options_from_ssh_command(rsync_rsh)
    if remote.endswith("/"):
        local.parent.mkdir(parents=True, exist_ok=True)
        run(["scp", *scp_options, "-r", remote.rstrip("/"), str(local.parent)])
    else:
        local.parent.mkdir(parents=True, exist_ok=True)
        run(["scp", *scp_options, remote, str(local)])


def sync_failure_job(
    *,
    remote_host: str,
    remote_results_dir: str,
    job_id: str,
    local_results_dir: Path,
    rsync_rsh: str = "",
) -> Path:
    local_dir = local_results_dir / f"dp2_failure_smoke_{job_id}"
    sync_path(
        f"{remote_host}:{remote_results_dir}/dp2_failure_smoke_{job_id}/",
        local_dir,
        rsync_rsh=rsync_rsh,
    )
    for suffix in ("out", "err"):
        sync_path(
            f"{remote_host}:{remote_results_dir}/slurm-aegiskv-dp2-failure-smoke-{job_id}.{suffix}",
            local_results_dir / f"slurm-aegiskv-dp2-failure-smoke-{job_id}.{suffix}",
            rsync_rsh=rsync_rsh,
        )
    return local_dir


def summarize_pair(
    *,
    invisible_dir: Path,
    visible_dir: Path,
    output_dir: Path,
    run_id: str,
    invisible_job_id: str,
    visible_job_id: str,
    same_write_cost_control: bool,
    prefetch_policy: str,
) -> Path:
    command = [
        sys.executable,
        str(EXP_DIR / "scripts" / "summarize_failure_pair.py"),
        "--invisible-dir",
        str(invisible_dir),
        "--visible-dir",
        str(visible_dir),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--invisible-job-id",
        invisible_job_id,
        "--visible-job-id",
        visible_job_id,
    ]
    if same_write_cost_control:
        command.append("--same-write-cost-control")
    if prefetch_policy:
        command.extend(["--prefetch-policy", prefetch_policy])
    run(command)
    return output_dir / "failure_visibility_pair_summary.json"


def _num(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def _int(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def render_markdown(
    *,
    title: str,
    summary: dict[str, Any],
    invisible_job_id: str,
    visible_job_id: str,
    output_dir: Path,
    note: str,
    prefetch_policy: str,
) -> str:
    control_success = (
        f"{_int(summary.get('invisible_success_count'))}/"
        f"{_int(summary.get('invisible_failover_response_count'))}"
    )
    visible_success = (
        f"{_int(summary.get('visible_success_count'))}/"
        f"{_int(summary.get('visible_failover_response_count'))}"
    )
    return "\n".join(
        [
            f"# {title}",
            "",
            "Run pair:",
            "",
            f"- Host-invisible control: job `{invisible_job_id}`",
            f"- Host-visible remote_backup run: job `{visible_job_id}`",
            f"- HiCache prefetch policy: `{prefetch_policy or 'not recorded'}`",
            f"- Result directory: `{output_dir.relative_to(EXP_DIR)}/`",
            "",
            "Summary:",
            "",
            f"- Control success/failover responses: {control_success}",
            f"- Visible success/failover responses: {visible_success}",
            f"- Control backed-up tokens: {_int(summary.get('invisible_pre_failover_backed_up_tokens'))}",
            f"- Visible backed-up tokens: {_int(summary.get('visible_pre_failover_backed_up_tokens'))}",
            f"- Frontier delta: {_int(summary.get('frontier_token_delta'))} tokens",
            f"- Visible backed-up tokens/request: {_num(summary.get('visible_backed_up_tokens_per_request'), 1)}",
            f"- Control p99 latency: {_num(summary.get('invisible_latency_p99_s'))} s",
            f"- Visible p99 latency: {_num(summary.get('visible_latency_p99_s'))} s",
            f"- Visible p99 normalized to control: {_num(summary.get('visible_retry_latency_norm_vs_invisible'), 3)}x",
            f"- Visible p99 delta: {_num(summary.get('visible_latency_delta_pct_vs_invisible'), 1)}%",
            "",
            "Interpretation:",
            "",
            note,
            "",
            "Evidence status:",
            "",
            "This is synchronized SGLang small-scale paired evidence. It remains a",
            "calibration/correctness artifact unless the paired run demonstrates a",
            "representative recovery-benefit result under matched normal-path cost.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync two SGLang failure jobs from hkbugpumanode and summarize the pair."
    )
    parser.add_argument("--remote-host", default="hkbugpumanode")
    parser.add_argument(
        "--remote-results-dir",
        default="~/aegiskv_sglang_small/results",
    )
    parser.add_argument("--invisible-job-id", required=True)
    parser.add_argument("--visible-job-id", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--note", default="Paired Host-invisible versus Host-visible failure probe.")
    parser.add_argument(
        "--same-write-cost-control",
        action="store_true",
        help=(
            "Mark the invisible job as remote_backup write-enabled but hidden "
            "from failover lookup."
        ),
    )
    parser.add_argument("--prefetch-policy", default="")
    parser.add_argument(
        "--rsync-rsh",
        default=os.environ.get("AEGISKV_RSYNC_RSH", ""),
        help=(
            "Optional SSH command passed to rsync -e. This is useful when the "
            "cluster must be reached through an explicit jump host or when "
            "system ssh_config is broken."
        ),
    )
    args = parser.parse_args()

    run_id = args.run_id or (
        f"failure_visibility_pair_{args.invisible_job_id}_{args.visible_job_id}"
    )
    output_dir = RESULTS_DIR / run_id
    invisible_dir = sync_failure_job(
        remote_host=args.remote_host,
        remote_results_dir=args.remote_results_dir,
        job_id=args.invisible_job_id,
        local_results_dir=RESULTS_DIR,
        rsync_rsh=args.rsync_rsh,
    )
    visible_dir = sync_failure_job(
        remote_host=args.remote_host,
        remote_results_dir=args.remote_results_dir,
        job_id=args.visible_job_id,
        local_results_dir=RESULTS_DIR,
        rsync_rsh=args.rsync_rsh,
    )
    summary_path = summarize_pair(
        invisible_dir=invisible_dir,
        visible_dir=visible_dir,
        output_dir=output_dir,
        run_id=run_id,
        invisible_job_id=args.invisible_job_id,
        visible_job_id=args.visible_job_id,
        same_write_cost_control=args.same_write_cost_control,
        prefetch_policy=args.prefetch_policy,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    title = args.title or (
        f"DP=2 Failure Visibility Pair {args.invisible_job_id}/{args.visible_job_id}"
    )
    md_path = EXP_DIR / f"dp2_failure_visibility_pair_{args.invisible_job_id}_{args.visible_job_id}_summary.md"
    md_path.write_text(
        render_markdown(
            title=title,
            summary=summary,
            invisible_job_id=args.invisible_job_id,
            visible_job_id=args.visible_job_id,
            output_dir=output_dir,
            note=args.note,
            prefetch_policy=args.prefetch_policy,
        ),
        encoding="utf-8",
    )
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
