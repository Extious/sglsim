#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = EXP_DIR / "results" / "strict_final_recovery_matrix_audit"
DEFAULT_MISSING_CSV = DEFAULT_AUDIT_DIR / "strict_final_recovery_missing_jobs.csv"
DEFAULT_JUMP_HOST = "faith.comp.hkbu.edu.hk"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def remote_payload_from_ssh(command: str) -> str:
    parts = shlex.split(command)
    if len(parts) < 3 or parts[0] != "ssh":
        raise ValueError(f"Expected an ssh command, got: {command}")
    return parts[-1]


def alias_ssh_command(remote_payload: str, *, alias: str) -> str:
    return shlex.join(["ssh", alias, remote_payload])


def direct_proxy_ssh_command(
    remote_payload: str,
    *,
    remote_user: str,
    remote_host: str,
    jump_user: str,
    jump_host: str,
    identity_file: str,
    timeout_s: int,
) -> str:
    proxy = shlex.join(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-i",
            identity_file,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout_s}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "KexAlgorithms=+diffie-hellman-group-exchange-sha1,diffie-hellman-group14-sha1,diffie-hellman-group1-sha1",
            "-o",
            "HostKeyAlgorithms=+ssh-rsa",
            "-W",
            "%h:%p",
            f"{jump_user}@{jump_host}",
        ]
    )
    return shlex.join(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout_s}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ProxyCommand={proxy}",
            f"{remote_user}@{remote_host}",
            remote_payload,
        ]
    )


def direct_proxy_rsync_rsh(
    *,
    jump_user: str,
    jump_host: str,
    identity_file: str,
    timeout_s: int,
) -> str:
    proxy = shlex.join(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-i",
            identity_file,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout_s}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "KexAlgorithms=+diffie-hellman-group-exchange-sha1,diffie-hellman-group14-sha1,diffie-hellman-group1-sha1",
            "-o",
            "HostKeyAlgorithms=+ssh-rsa",
            "-W",
            "%h:%p",
            f"{jump_user}@{jump_host}",
        ]
    )
    return shlex.join(
        [
            "ssh",
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout_s}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ProxyCommand={proxy}",
        ]
    )


def sync_direct_proxy_command(
    sync_command: str,
    *,
    remote_user: str,
    remote_host: str,
    rsync_rsh: str,
) -> str:
    parts = shlex.split(sync_command)
    parts.extend(["--remote-host", f"{remote_user}@{remote_host}"])
    parts.extend(["--rsync-rsh", rsync_rsh])
    return shlex.join(parts)


def build_manifest(
    rows: list[dict[str, str]],
    *,
    alias: str,
    remote_user: str,
    remote_host: str,
    jump_user: str,
    jump_host: str,
    identity_file: str,
    timeout_s: int,
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    rsync_rsh = direct_proxy_rsync_rsh(
        jump_user=jump_user,
        jump_host=jump_host,
        identity_file=identity_file,
        timeout_s=timeout_s,
    )
    for row in rows:
        invisible_payload = remote_payload_from_ssh(row["invisible_submit_command"])
        visible_payload = remote_payload_from_ssh(row["visible_submit_command"])
        jobs.append(
            {
                "case_id": row.get("case_id", ""),
                "pair_id": row.get("pair_id", ""),
                "repeat": row.get("repeat", ""),
                "reason": row.get("reason", ""),
                "invisible_run_label": row.get("invisible_run_label", ""),
                "visible_run_label": row.get("visible_run_label", ""),
                "invisible_submit_alias": alias_ssh_command(
                    invisible_payload,
                    alias=alias,
                ),
                "visible_submit_alias": alias_ssh_command(
                    visible_payload,
                    alias=alias,
                ),
                "invisible_submit_direct_proxy": direct_proxy_ssh_command(
                    invisible_payload,
                    remote_user=remote_user,
                    remote_host=remote_host,
                    jump_user=jump_user,
                    jump_host=jump_host,
                    identity_file=identity_file,
                    timeout_s=timeout_s,
                ),
                "visible_submit_direct_proxy": direct_proxy_ssh_command(
                    visible_payload,
                    remote_user=remote_user,
                    remote_host=remote_host,
                    jump_user=jump_user,
                    jump_host=jump_host,
                    identity_file=identity_file,
                    timeout_s=timeout_s,
                ),
                "sync_pair_command": row.get("sync_pair_command", ""),
                "sync_pair_direct_proxy_command": sync_direct_proxy_command(
                    row.get("sync_pair_command", ""),
                    remote_user=remote_user,
                    remote_host=remote_host,
                    rsync_rsh=rsync_rsh,
                ),
                "physical_restore_command": row.get("physical_restore_command", ""),
            }
        )
    return {
        "job_count": len(jobs),
        "jobs": jobs,
        "ssh": {
            "alias": alias,
            "remote_user": remote_user,
            "remote_host": remote_host,
            "jump_user": jump_user,
            "jump_host": jump_host,
            "identity_file": identity_file,
            "timeout_s": timeout_s,
            "rsync_rsh": rsync_rsh,
        },
        "source": "prepare_strict_final_missing_jobs",
        "note": (
            "Submission helpers for missing strict-final SGLang recovery jobs. "
            "Generated commands do not change any evidence status until Slurm "
            "results are synced and audited."
        ),
    }


def render_submit_script(
    manifest: dict[str, Any],
    *,
    command_key: str,
    title: str,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"echo {shlex.quote(title)}",
        "",
    ]
    for job in manifest["jobs"]:
        lines.extend(
            [
                f"echo {shlex.quote('submit ' + job['pair_id'] + ' invisible')}",
                job[f"invisible_submit_{command_key}"],
                f"echo {shlex.quote('submit ' + job['pair_id'] + ' visible')}",
                job[f"visible_submit_{command_key}"],
                "",
            ]
        )
    return "\n".join(lines)


def render_sync_template(manifest: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Replace each <..._job_id> placeholder with the Slurm job id returned by sbatch.",
        "",
    ]
    for job in manifest["jobs"]:
        lines.extend(
            [
                f"# {job['pair_id']}",
                "# Use this direct-proxy command if the hkbugpumanode SSH alias is unavailable:",
                job["sync_pair_direct_proxy_command"],
                job["physical_restore_command"],
                "",
            ]
        )
    return "\n".join(lines)


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Strict-Final Missing Job Submission Manifest",
        "",
        "This manifest is generated from `strict_final_recovery_missing_jobs.csv`.",
        "It prepares the next Slurm submissions but does not claim new evidence.",
        "",
        f"- Missing pair count: {manifest['job_count']}",
        f"- Remote: `{manifest['ssh']['remote_user']}@{manifest['ssh']['remote_host']}`",
        f"- Jump: `{manifest['ssh']['jump_user']}@{manifest['ssh']['jump_host']}`",
        "",
        "Generated helpers:",
        "",
        "- `submit_missing_alias.sh`: uses the SSH alias from `~/.ssh/config`.",
        "- `submit_missing_direct_proxy.sh`: bypasses system SSH config with `-F /dev/null` and an explicit jump host.",
        "- `sync_missing_template.sh`: fill in Slurm job ids after submission.",
        "- `submit_and_bind_strict_final_missing_jobs.py`: safe dry-run wrapper; add `--execute` to run reachability preflight, submit, capture the Slurm log, and bind returned job ids.",
        "",
        "## Missing Pairs",
        "",
        "| Pair | Case | Repeat | Reason |",
        "|---|---|---:|---|",
    ]
    for job in manifest["jobs"]:
        lines.append(
            f"| {job['pair_id']} | {job['case_id']} | {job['repeat']} | {job['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These helpers are operational artifacts only. The recovery matrix remains",
            "`partial` until the resulting SGLang jobs are synced, parsed, and accepted",
            "by `audit_strict_final_recovery_matrix.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare submit/sync helpers for missing strict-final recovery jobs."
    )
    parser.add_argument("--missing-csv", type=Path, default=DEFAULT_MISSING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--alias", default="hkbugpumanode")
    parser.add_argument("--remote-user", default="25482351")
    parser.add_argument("--remote-host", default="172.27.244.19")
    parser.add_argument("--jump-user", default="zhanzhao")
    parser.add_argument("--jump-host", default=DEFAULT_JUMP_HOST)
    parser.add_argument("--identity-file", default="~/.ssh/id_rsa")
    parser.add_argument("--timeout-s", type=int, default=20)
    args = parser.parse_args()

    rows = read_csv(args.missing_csv)
    manifest = build_manifest(
        rows,
        alias=args.alias,
        remote_user=args.remote_user,
        remote_host=args.remote_host,
        jump_user=args.jump_user,
        jump_host=args.jump_host,
        identity_file=args.identity_file,
        timeout_s=args.timeout_s,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "strict_final_recovery_submission_manifest.json", manifest)
    (args.output_dir / "strict_final_recovery_submission_manifest.md").write_text(
        render_markdown(manifest),
        encoding="utf-8",
    )
    write_text_executable(
        args.output_dir / "submit_missing_alias.sh",
        render_submit_script(
            manifest,
            command_key="alias",
            title="Submitting strict-final missing jobs via SSH alias",
        ),
    )
    write_text_executable(
        args.output_dir / "submit_missing_direct_proxy.sh",
        render_submit_script(
            manifest,
            command_key="direct_proxy",
            title="Submitting strict-final missing jobs via direct proxy SSH",
        ),
    )
    write_text_executable(
        args.output_dir / "sync_missing_template.sh",
        render_sync_template(manifest),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "job_count": manifest["job_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
