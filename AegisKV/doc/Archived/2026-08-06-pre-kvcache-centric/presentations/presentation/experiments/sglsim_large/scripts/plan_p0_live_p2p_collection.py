#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SGLSIM_LARGE_ROOT.parents[4]
DEFAULT_READINESS_DIR = SGLSIM_LARGE_ROOT / "results" / "strict_final_readiness"
DEFAULT_REQUIRED_CSV = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.required.csv"
DEFAULT_HOST_LIVE_JSON = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.live.json"
DEFAULT_P2P_LIVE_JSON = DEFAULT_READINESS_DIR / "p0_physical_tensor_copy_events.p2p_live.json"
DEFAULT_PLAN_JSON = DEFAULT_READINESS_DIR / "p0_live_p2p_collection_plan.json"
DEFAULT_PLAN_MD = DEFAULT_READINESS_DIR / "p0_live_p2p_collection_plan.md"
DEFAULT_REMOTE_HOST = "hkbugpumanode"
DEFAULT_REMOTE_WORKDIR = "~/aegiskv_sglsim_large"
DEFAULT_REMOTE_PYTHON = "/home/comp/25482351/sglang/.venv/bin/python3"
SBATCH_PATH = SGLSIM_LARGE_ROOT / "slurm" / "aegiskv_p0_live_p2p_copy_events.sbatch"
COLLECTOR_PATH = SGLSIM_LARGE_ROOT / "scripts" / "collect_p0_live_copy_events.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def as_int(value: object) -> int:
    try:
        if value in ("", None):
            return 0
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def count_required_p2p(rows: list[dict[str, str]]) -> tuple[int, int]:
    p2p_rows = [
        row
        for row in rows
        if str(row.get("event_kind") or "") == "live_peer_p2p_copy"
    ]
    return len(p2p_rows), sum(as_int(row.get("restored_tokens")) for row in p2p_rows)


def event_rows_by_kind(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("event_rows_by_kind")
    if not isinstance(raw, dict):
        return {}
    return {str(key): as_int(value) for key, value in raw.items()}


def shell(command: list[str]) -> str:
    return shlex.join(command)


def remote_commands(
    *,
    required_csv: Path,
    remote_host: str,
    remote_workdir: str,
    python_bin: str,
) -> dict[str, str]:
    readiness_rel = "results/strict_final_readiness"
    commands = {
        "create_remote_dirs": shell(
            [
                "ssh",
                remote_host,
                f"mkdir -p {remote_workdir}/scripts {remote_workdir}/slurm {remote_workdir}/{readiness_rel}",
            ]
        ),
        "copy_collector": shell(
            [
                "rsync",
                "-a",
                rel(COLLECTOR_PATH),
                f"{remote_host}:{remote_workdir}/scripts/",
            ]
        ),
        "copy_sbatch": shell(
            [
                "rsync",
                "-a",
                rel(SBATCH_PATH),
                f"{remote_host}:{remote_workdir}/slurm/",
            ]
        ),
        "copy_required_manifest": shell(
            [
                "rsync",
                "-a",
                rel(required_csv),
                f"{remote_host}:{remote_workdir}/{readiness_rel}/",
            ]
        ),
        "submit": shell(
            [
                "ssh",
                remote_host,
                (
                    f"cd {remote_workdir} && "
                    f"PYTHON_BIN={shlex.quote(python_bin)} "
                    "sbatch slurm/aegiskv_p0_live_p2p_copy_events.sbatch"
                ),
            ]
        ),
        "sync_outputs": shell(
            [
                "rsync",
                "-a",
                f"{remote_host}:{remote_workdir}/{readiness_rel}/p0_physical_tensor_copy_events.p2p_live.*",
                f"{rel(DEFAULT_READINESS_DIR)}/",
            ]
        ),
    }
    return commands


def build_payload(
    *,
    required_csv: Path,
    host_live_json: Path,
    p2p_live_json: Path,
    remote_host: str,
    remote_workdir: str,
    python_bin: str,
) -> dict[str, Any]:
    required_rows = read_csv(required_csv)
    required_p2p_rows, required_p2p_tokens = count_required_p2p(required_rows)
    host_live = read_json(host_live_json)
    p2p_live = read_json(p2p_live_json)
    host_kinds = event_rows_by_kind(host_live)
    p2p_kinds = event_rows_by_kind(p2p_live)
    p2p_live_rows = p2p_kinds.get("live_peer_p2p_copy", 0)
    p2p_skipped_rows = as_int(p2p_live.get("skipped_rows"))
    p2p_ready = bool(required_p2p_rows and p2p_live_rows == required_p2p_rows and p2p_skipped_rows == 0)
    if p2p_ready:
        status = "p2p_live_copy_complete"
    elif p2p_live:
        status = "p2p_live_copy_partial_or_blocked"
    elif required_p2p_rows and SBATCH_PATH.exists() and COLLECTOR_PATH.exists():
        status = "ready_to_submit_p2p_live_copy_job"
    else:
        status = "missing_p2p_collection_inputs"
    return {
        "collector": rel(COLLECTOR_PATH),
        "commands": remote_commands(
            required_csv=required_csv,
            remote_host=remote_host,
            remote_workdir=remote_workdir,
            python_bin=python_bin,
        ),
        "host_live_copy_event_rows": as_int(host_live.get("event_rows")),
        "host_live_copy_event_rows_by_kind": host_kinds,
        "p2p_live_copy_event_rows": as_int(p2p_live.get("event_rows")),
        "p2p_live_copy_event_rows_by_kind": p2p_kinds,
        "p2p_live_copy_json": str(p2p_live_json),
        "p2p_live_copy_skipped_rows": p2p_skipped_rows,
        "p2p_live_copy_skipped_rows_by_reason": p2p_live.get("skipped_rows_by_reason", {}),
        "p2p_ready": p2p_ready,
        "python_bin": python_bin,
        "remote_host": remote_host,
        "remote_workdir": remote_workdir,
        "required_csv": str(required_csv),
        "required_p2p_rows": required_p2p_rows,
        "required_p2p_tokens": required_p2p_tokens,
        "required_rows": len(required_rows),
        "sbatch": rel(SBATCH_PATH),
        "source": "plan_p0_live_p2p_collection",
        "status": status,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P0 Live P2P Collection Plan",
        "",
        "This plan packages the remaining GPUReplica P2P live-copy evidence",
        "collection for hardware that supports CUDA peer access. It does not",
        "treat host-staged cross-GPU movement as peer P2P evidence.",
        "",
        f"- status: `{payload['status']}`",
        f"- required P2P rows: {payload['required_p2p_rows']}",
        f"- required P2P tokens: {payload['required_p2p_tokens']}",
        f"- existing Host live rows: {payload['host_live_copy_event_rows']}",
        f"- existing P2P live rows: {payload['p2p_live_copy_event_rows']}",
        f"- existing P2P skipped rows: {payload['p2p_live_copy_skipped_rows']}",
        f"- remote host: `{payload['remote_host']}`",
        f"- remote workdir: `{payload['remote_workdir']}`",
        f"- sbatch: `{payload['sbatch']}`",
        f"- collector: `{payload['collector']}`",
        "",
        "## Commands",
        "",
    ]
    for name, command in payload["commands"].items():
        lines.append(f"- `{name}`: `{command}`")
    lines.extend(
        [
            "",
            "## Completion Rule",
            "",
            "The P2P side is complete only when `p0_physical_tensor_copy_events.p2p_live.json`",
            "reports `event_rows_by_kind.live_peer_p2p_copy == required_p2p_rows`",
            "and `skipped_rows == 0`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan remote collection of P0 GPUReplica P2P live-copy evidence."
    )
    parser.add_argument("--required-csv", type=Path, default=DEFAULT_REQUIRED_CSV)
    parser.add_argument("--host-live-json", type=Path, default=DEFAULT_HOST_LIVE_JSON)
    parser.add_argument("--p2p-live-json", type=Path, default=DEFAULT_P2P_LIVE_JSON)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-workdir", default=DEFAULT_REMOTE_WORKDIR)
    parser.add_argument("--python-bin", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_PLAN_MD)
    args = parser.parse_args(argv)

    payload = build_payload(
        required_csv=args.required_csv,
        host_live_json=args.host_live_json,
        p2p_live_json=args.p2p_live_json,
        remote_host=args.remote_host,
        remote_workdir=args.remote_workdir,
        python_bin=args.python_bin,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
