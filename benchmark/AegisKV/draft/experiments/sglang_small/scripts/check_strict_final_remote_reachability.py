#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ipaddress
import importlib.util
import json
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


EXP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIT_DIR = EXP_DIR / "results" / "strict_final_recovery_matrix_audit"
DEFAULT_MANIFEST = DEFAULT_AUDIT_DIR / "strict_final_recovery_submission_manifest.json"
DEFAULT_OUTPUT_JSON = DEFAULT_AUDIT_DIR / "strict_final_remote_reachability.json"
DEFAULT_OUTPUT_MD = DEFAULT_AUDIT_DIR / "strict_final_remote_reachability.md"
RESOLV_CONF = Path("/etc/resolv.conf")


def load_prepare_module():
    path = SCRIPT_DIR / "prepare_strict_final_missing_jobs.py"
    spec = importlib.util.spec_from_file_location("prepare_strict_final_missing_jobs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def classify_ssh_status(exit_code: int, output: str) -> str:
    if exit_code == 0:
        return "ok"
    text = output.lower()
    if "bad owner or permissions" in text:
        return "ssh_config_error"
    if "network is unreachable" in text:
        return "network_unreachable"
    if "could not resolve hostname" in text or "name or service not known" in text:
        return "dns_unresolved"
    if "permission denied" in text:
        return "auth_failed"
    if "connection timed out" in text or "operation timed out" in text:
        return "timeout"
    if "connection refused" in text:
        return "connection_refused"
    return "failed"


def classify_getent_status(exit_code: int, output: str) -> str:
    if exit_code == 0 and output.strip():
        return "ok"
    if not output.strip():
        return "no_records"
    return "failed"


def getent_command_for_host(host: str) -> list[str]:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return ["getent", "hosts", host]
    return ["getent", "ahosts", host]


def read_resolver_config(path: Path = RESOLV_CONF) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "nameservers": [], "search": []}
    nameservers: list[str] = []
    search: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "nameserver":
            nameservers.extend(parts[1:])
        elif parts[0] == "search":
            search.extend(parts[1:])
    return {
        "exists": True,
        "nameservers": nameservers,
        "path": str(path),
        "search": search,
    }


def next_step_commands(manifest: dict[str, Any]) -> list[str]:
    ssh = manifest.get("ssh") if isinstance(manifest.get("ssh"), dict) else {}
    remote_user = str(ssh.get("remote_user") or "25482351")
    remote_host = str(ssh.get("remote_host") or "172.27.244.19")
    jump_user = str(ssh.get("jump_user") or "zhanzhao")
    identity_file = str(ssh.get("identity_file") or "~/.ssh/id_rsa")
    timeout_s = int(ssh.get("timeout_s") or 20)
    prepare_cmd = [
        "python",
        str(SCRIPT_DIR / "prepare_strict_final_missing_jobs.py"),
        "--jump-host",
        "<reachable_jump_host_or_ip>",
        "--remote-user",
        remote_user,
        "--remote-host",
        remote_host,
        "--jump-user",
        jump_user,
        "--identity-file",
        identity_file,
        "--timeout-s",
        str(timeout_s),
    ]
    preflight_cmd = [
        "python",
        str(SCRIPT_DIR / "check_strict_final_remote_reachability.py"),
    ]
    submit_cmd = [
        "python",
        str(SCRIPT_DIR / "submit_and_bind_strict_final_missing_jobs.py"),
        "--method",
        "direct-proxy",
        "--execute",
    ]
    return [shlex.join(prepare_cmd), shlex.join(preflight_cmd), shlex.join(submit_cmd)]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def command_check(
    check_id: str,
    command: list[str],
    *,
    classifier: Callable[[int, str], str],
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_command,
) -> dict[str, Any]:
    proc = runner(command)
    output = proc.stdout or ""
    return {
        "command": shlex.join(command),
        "exit_code": proc.returncode,
        "id": check_id,
        "status": classifier(proc.returncode, output),
        "stdout": output.strip(),
    }


def submission_blockers(checks: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for item in checks:
        check_id = str(item.get("id") or "")
        status = str(item.get("status") or "")
        if check_id.startswith("direct_proxy_") and status != "ok":
            blockers.append(f"{check_id}={status}")
    return blockers


def diagnostic_failures(checks: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for item in checks:
        check_id = str(item.get("id") or "")
        status = str(item.get("status") or "")
        if check_id == "alias_default" and status != "ok":
            failures.append(f"{check_id}={status}")
    return failures


def build_checks(manifest: dict[str, Any]) -> list[tuple[str, list[str], Callable[[int, str], str]]]:
    prepare = load_prepare_module()
    ssh = manifest.get("ssh") if isinstance(manifest.get("ssh"), dict) else {}
    alias = str(ssh.get("alias") or "hkbugpumanode")
    remote_user = str(ssh.get("remote_user") or "25482351")
    remote_host = str(ssh.get("remote_host") or "172.27.244.19")
    jump_user = str(ssh.get("jump_user") or "zhanzhao")
    jump_host = str(ssh.get("jump_host") or getattr(prepare, "DEFAULT_JUMP_HOST"))
    identity_file = str(ssh.get("identity_file") or "~/.ssh/id_rsa")
    timeout_s = int(ssh.get("timeout_s") or 20)
    direct_payload = "echo direct-proxy-ok"
    direct_command = shlex.split(
        prepare.direct_proxy_ssh_command(
            direct_payload,
            remote_user=remote_user,
            remote_host=remote_host,
            jump_user=jump_user,
            jump_host=jump_host,
            identity_file=identity_file,
            timeout_s=timeout_s,
        )
    )
    return [
        (
            "alias_default",
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={timeout_s}",
                alias,
                "echo alias-ok",
            ],
            classify_ssh_status,
        ),
        (
            f"direct_proxy_{sanitize_id(jump_host)}",
            direct_command,
            classify_ssh_status,
        ),
        (
            f"getent_jump_{sanitize_id(jump_host)}",
            ["getent", "hosts", jump_host],
            classify_getent_status,
        ),
        (
            f"getent_remote_{sanitize_id(remote_host)}",
            getent_command_for_host(remote_host),
            classify_getent_status,
        ),
    ]


def run_reachability_checks(
    manifest: dict[str, Any],
    *,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = run_command,
) -> dict[str, Any]:
    checks = [
        command_check(check_id, command, classifier=classifier, runner=runner)
        for check_id, command, classifier in build_checks(manifest)
    ]
    ssh_ok = any(
        item["status"] == "ok"
        and (item["id"] == "alias_default" or item["id"].startswith("direct_proxy_"))
        for item in checks
    )
    ssh = manifest.get("ssh") if isinstance(manifest.get("ssh"), dict) else {}
    direct_proxy_blockers = submission_blockers(checks)
    nonblocking_diagnostics = diagnostic_failures(checks)
    remote_ip_status = next(
        (
            str(item.get("status") or "")
            for item in checks
            if str(item.get("id") or "").startswith("getent_remote_")
        ),
        "",
    )
    return {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checks": checks,
        "current_generated_direct_proxy_jump_host": str(ssh.get("jump_host") or ""),
        "diagnostic_failures": nonblocking_diagnostics,
        "impact": (
            "Missing strict-final wait-complete Slurm pairs can be submitted "
            "from this environment."
            if ssh_ok
            else (
                "The generated direct-proxy submission path is blocked by "
                f"{direct_proxy_blockers}. Alias-only diagnostics: "
                f"{nonblocking_diagnostics}. Remote target IP diagnostic: "
                f"{remote_ip_status}."
            )
        ),
        "next_step_commands": next_step_commands(manifest),
        "overall_status": "reachable" if ssh_ok else "unreachable",
        "primary_submission_route": "direct-proxy",
        "resolver": read_resolver_config(),
        "source": "check_strict_final_remote_reachability",
        "submission_blockers": direct_proxy_blockers,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Strict-Final Remote Reachability",
        "",
        "This note records a local reachability check for the missing",
        "`same_write_wait_complete` recovery pairs. It does not change the recovery",
        "matrix evidence level.",
        "",
        f"- Checked at: `{payload['checked_at']}`",
        f"- Overall status: `{payload['overall_status']}`",
        "- Current generated direct-proxy jump host:",
        f"  `{payload.get('current_generated_direct_proxy_jump_host', '')}`",
        f"- Primary submission route: `{payload.get('primary_submission_route', '')}`",
        f"- Submission blockers: `{payload.get('submission_blockers', [])}`",
        f"- Diagnostic failures: `{payload.get('diagnostic_failures', [])}`",
        "- Resolver nameservers:",
        f"  `{', '.join(payload.get('resolver', {}).get('nameservers', []))}`",
        "- Missing pairs remain:",
        "  `strict_final_same_write_wait_complete_r2`,",
        "  `strict_final_same_write_wait_complete_r3`",
        "",
        "| Check | Status | Result |",
        "|---|---|---|",
    ]
    for item in payload["checks"]:
        result = str(item.get("stdout") or "").replace("\n", " ")[:180]
        lines.append(
            f"| `{item['id']}` | `{item['status']}` | `{result}` |"
        )
    lines.extend(
        [
            "",
            "## Impact",
            "",
            str(payload["impact"]),
            "",
            "## Next Commands",
            "",
            "After obtaining a reachable jump host or IP, run:",
            "",
        ]
    )
    for command in payload.get("next_step_commands", []):
        lines.append(f"- `{command}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check read-only SSH reachability before strict-final submissions."
    )
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    payload = run_reachability_checks(read_json(args.manifest_json))
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["overall_status"] == "reachable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
