from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sglsim.srt.simulation.aegis_exact_contract import (
    load_certified_exact_window_plan,
)

from .common import wait_for_server
from .plot_wct_cdf import render
from .wct_cdf_client import (
    WORKFLOW_LAUNCH_POLICY,
    WORKFLOW_LAUNCH_STAGGER_S,
)
from .workflow_client import DEFAULT_SLACK_CALIBRATION, DEFAULT_TRACE, SLACK_SOURCES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "server.json"
DEFAULT_QUICK_FAILURE_MANIFEST = (
    Path(__file__).resolve().parent / "config" / "failure_manifest.json"
)
DEFAULT_FORMAL_FAILURE_MANIFEST = (
    Path(__file__).resolve().parent
    / "config"
    / "failure_manifest_formal_20k.json"
)
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "wct_cdf"
POLICIES = ("aegiskv", "l1_only", "l2_only", "l3_only", "no_backup")
BASELINE_RECOMPUTE_MS_PER_TOKEN = 0.22


def acquire_run_lock(run_dir: Path, *, lock_name: str = ".suite.lock"):
    """Hold an exclusive lock so two suites cannot mutate one result directory."""

    lock_path = run_dir / lock_name
    lock_handle = lock_path.open("a+", encoding="utf-8")
    lock_handle.seek(0)
    # PID values can be reused and can differ across process namespaces. Only
    # the kernel flock below proves that another live suite owns the directory;
    # the recorded PID is diagnostic text, not an ownership check.
    recorded_owner = lock_handle.read().strip()
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.seek(0)
        owner = lock_handle.read().strip() or "unknown"
        lock_handle.close()
        raise RuntimeError(
            f"result directory is already used by suite PID {owner}: {run_dir}"
        ) from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    return lock_handle


def find_free_port(start: int = 31200) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free TCP port in [{start}, {start + 200})")


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def policy_result_is_complete(
    path: Path,
    *,
    policy: str,
    expected_workflows: int | None,
) -> bool:
    """Return whether ``path`` is a complete, reusable policy result."""

    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    workflows = payload.get("workflows")
    if payload.get("policy") != policy or not isinstance(workflows, list):
        return False
    workflow_count = int(payload.get("workflow_count", -1))
    if workflow_count != len(workflows):
        return False
    if expected_workflows:
        eligible_workflows = int(payload.get("eligible_workflow_count", -1))
        expected_selected = (
            min(int(expected_workflows), eligible_workflows)
            if eligible_workflows >= 0
            else int(expected_workflows)
        )
        if workflow_count != expected_selected:
            return False
    return True


def archive_interrupted_artifacts(run_dir: Path, policy: str) -> list[Path]:
    """Preserve partial evidence before restarting a policy from a clean server."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archived: list[Path] = []
    for path in (
        run_dir / f"{policy}.partial.json",
        run_dir / f"{policy}.json",
        run_dir / f"server-{policy}.log",
    ):
        if not path.exists():
            continue
        destination = path.with_name(
            f"{path.stem}.interrupted-{timestamp}{path.suffix}"
        )
        path.replace(destination)
        archived.append(destination)
    return archived


def snapshot_suite_configuration(
    args: argparse.Namespace,
    run_dir: Path,
) -> Path:
    """Freeze the selected server config and record the experiment profile."""

    source = args.server_config.resolve()
    server_config = json.loads(source.read_text(encoding="utf-8"))
    snapshot = run_dir / "server_config.json"
    exact_plan_snapshot: Path | None = None
    exact_plan_source: str | None = None
    if args.exact_window_plan is not None:
        exact_plan_source_path = args.exact_window_plan.resolve()
        exact_plan = load_certified_exact_window_plan(exact_plan_source_path)
        exact_plan_snapshot = run_dir / "certified_exact_window.json"
        exact_plan_snapshot.write_text(
            exact_plan_source_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        exact_plan_source = str(exact_plan_source_path)
        scheduler = server_config.get("scheduler", {})
        expected_dp = int(args.dp_size or scheduler.get("dp_size", 1))
        expected_pp = int(args.pp_size or scheduler.get("pp_size", 1))
        if (exact_plan.dp_size, exact_plan.pp_size) != (
            expected_dp,
            expected_pp,
        ):
            raise ValueError(
                "certified exact plan topology differs from suite topology: "
                f"plan={exact_plan.dp_size}x{exact_plan.pp_size}, "
                f"suite={expected_dp}x{expected_pp}"
            )
    snapshot.write_text(
        json.dumps(server_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    route_plan_snapshot: Path | None = None
    route_plan_source: str | None = None
    if args.route_plan is not None:
        route_plan_source_path = args.route_plan.resolve()
        route_plan = json.loads(
            route_plan_source_path.read_text(encoding="utf-8")
        )
        route_plan_snapshot = run_dir / "execution_route_plan.json"
        route_plan_snapshot.write_text(
            json.dumps(route_plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        route_plan_source = str(route_plan_source_path)

    aegis = server_config.get("aegis_l1_l2", {})
    scheduler = server_config.get("scheduler", {})
    deployment_model = str(args.model or server_config.get("model_path") or "")
    deployment_dp_size = int(args.dp_size or scheduler.get("dp_size", 1))
    deployment_pp_size = int(args.pp_size or scheduler.get("pp_size", 1))
    recompute_ms_per_token = float(
        aegis.get("recompute_ms_per_token", BASELINE_RECOMPUTE_MS_PER_TOKEN)
    )
    suite_config = {
        "schema_version": 1,
        "profile": (
            "quick-1k"
            if args.quick
            else f"formal-{max(int(args.context_length) // 1024, 1)}k"
        ),
        "server_config_source": str(source),
        "server_config_snapshot": snapshot.name,
        "model": deployment_model,
        "dp_size": deployment_dp_size,
        "pp_size": deployment_pp_size,
        # ``--policies`` slices collection into bounded jobs; the experiment
        # design remains the full five-policy comparison.
        "policies": list(POLICIES),
        "trace": str(args.trace.resolve()),
        "slack_source": args.slack_source,
        "trace_oracle_upper_bound": args.slack_source == "trace_oracle",
        "collect_calibration_telemetry": bool(
            args.collect_calibration_telemetry
        ),
        "slack_calibration": str(args.slack_calibration.resolve()),
        "failure_manifest": str(args.failure_manifest.resolve()),
        "execution_route_plan_source": route_plan_source,
        "execution_route_plan_snapshot": (
            route_plan_snapshot.name if route_plan_snapshot is not None else None
        ),
        "exact_window_plan_source": exact_plan_source,
        "exact_window_plan_snapshot": (
            exact_plan_snapshot.name if exact_plan_snapshot is not None else None
        ),
        "exact_window_applies_only_to_policy": (
            "aegiskv" if exact_plan_snapshot is not None else None
        ),
        "failure_count": args.failure_count,
        "workflow_concurrency": args.workflow_concurrency,
        "workflow_launch_policy": WORKFLOW_LAUNCH_POLICY,
        "workflow_launch_stagger_s": WORKFLOW_LAUNCH_STAGGER_S,
        "max_workflows": args.max_workflows,
        "max_requests_per_workflow": args.max_requests_per_workflow,
        "input_scale": args.input_scale,
        "context_length": args.context_length,
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "recompute_setup_ms": float(aegis.get("recompute_setup_ms", 0.0)),
        "recompute_ms_per_token": recompute_ms_per_token,
        "baseline_recompute_ms_per_token": BASELINE_RECOMPUTE_MS_PER_TOKEN,
        "recompute_penalty_multiplier": (
            recompute_ms_per_token / BASELINE_RECOMPUTE_MS_PER_TOKEN
        ),
    }
    (run_dir / "suite_config.json").write_text(
        json.dumps(suite_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot


def run_suite(args: argparse.Namespace) -> Path:
    # Only one online WCT suite may use this host at a time. Different result
    # directories still share CPU, ports, and wall-clock scheduling, so a
    # per-directory lock alone is insufficient for paper-grade measurements.
    # v2 separates current jobs from a legacy lock that can remain held by an
    # orphaned process in an inaccessible PID namespace.
    global_run_lock = acquire_run_lock(
        DEFAULT_RESULTS_ROOT.parent,
        lock_name=".suite-v2.lock",
    )
    if args.resume_run_dir is None:
        run_dir = args.results_root / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")

    # Keep the handle alive for the full suite lifetime. The kernel releases
    # this advisory lock automatically if the runner exits unexpectedly.
    run_lock = acquire_run_lock(run_dir)

    server_config_snapshot = run_dir / "server_config.json"
    if not server_config_snapshot.exists():
        server_config_snapshot = snapshot_suite_configuration(args, run_dir)
    deployment_config = json.loads(
        server_config_snapshot.read_text(encoding="utf-8")
    )
    scheduler_config = deployment_config.get("scheduler", {})
    deployment_model = str(
        args.model or deployment_config.get("model_path") or "Qwen/Qwen3-8B"
    )
    deployment_dp_size = int(
        args.dp_size or scheduler_config.get("dp_size", 1)
    )
    deployment_pp_size = int(
        args.pp_size or scheduler_config.get("pp_size", 1)
    )
    route_plan_snapshot = run_dir / "execution_route_plan.json"
    exact_plan_snapshot = run_dir / "certified_exact_window.json"
    port = find_free_port(args.start_port)
    base_url = f"http://127.0.0.1:{port}"

    for policy in args.policies:
        output_path = run_dir / f"{policy}.json"
        if policy_result_is_complete(
            output_path,
            policy=policy,
            expected_workflows=args.max_workflows,
        ):
            print(f"\n=== {policy}: reusing complete result ===", flush=True)
            continue

        archived = archive_interrupted_artifacts(run_dir, policy)
        if archived:
            names = ", ".join(path.name for path in archived)
            print(f"\n=== {policy}: archived interrupted artifacts: {names} ===", flush=True)
        print(
            f"\n=== {policy}: starting clean DP={deployment_dp_size} "
            f"PP={deployment_pp_size} SGLSim server ===",
            flush=True,
        )
        log_path = run_dir / f"server-{policy}.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            policy_server_config = server_config_snapshot
            if policy == "aegiskv" and exact_plan_snapshot.is_file():
                policy_config = json.loads(
                    server_config_snapshot.read_text(encoding="utf-8")
                )
                policy_aegis = dict(policy_config.get("aegis_l1_l2", {}))
                policy_aegis["exact_window_plan_path"] = exact_plan_snapshot.name
                policy_config["aegis_l1_l2"] = policy_aegis
                policy_server_config = run_dir / "server_config-aegiskv.json"
                policy_server_config.write_text(
                    json.dumps(policy_config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            command = [
                sys.executable,
                "-m",
                "sglsim.launch_server",
                "--model-path",
                deployment_model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--dp-size",
                str(deployment_dp_size),
                "--pp-size",
                str(deployment_pp_size),
                "--load-balance-method",
                "cache_aware",
                "--sim-config-path",
                str(policy_server_config),
                "--streaming-arrival",
            ]
            event_journal_path = run_dir / f"server-{policy}.events.jsonl"
            if args.collect_calibration_telemetry:
                command.append("--scheduler-telemetry")
                if event_journal_path.exists() and event_journal_path.stat().st_size:
                    raise FileExistsError(
                        "refusing to append calibration evidence to an existing "
                        f"journal: {event_journal_path}"
                    )
            server = subprocess.Popen(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    **(
                        {
                            "SGLSIM_AEGIS_EVENT_JOURNAL_PATH": str(
                                event_journal_path.resolve()
                            )
                        }
                        if args.collect_calibration_telemetry
                        else {}
                    ),
                },
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                wait_for_server(base_url, timeout=90.0)
                client_command = [
                    sys.executable,
                    "-m",
                    "AegisKV.demo.wct_cdf_client",
                    "--base-url",
                    base_url,
                    "--policy",
                    policy,
                    "--model",
                    deployment_model,
                    "--dp-size",
                    str(deployment_dp_size),
                    "--pp-size",
                    str(deployment_pp_size),
                    "--trace",
                    str(args.trace),
                    "--output",
                    str(output_path),
                    "--workflow-concurrency",
                    str(args.workflow_concurrency),
                    "--failure-count",
                    str(args.failure_count),
                    "--failure-manifest",
                    str(args.failure_manifest),
                    "--input-scale",
                    str(args.input_scale),
                    "--context-length",
                    str(args.context_length),
                    "--arrival-speedup",
                    str(args.arrival_speedup),
                    "--request-timeout",
                    str(args.request_timeout),
                    "--slack-calibration",
                    str(args.slack_calibration),
                    "--slack-source",
                    args.slack_source,
                ]
                if args.max_input_tokens is not None:
                    client_command.extend(
                        ["--max-input-tokens", str(args.max_input_tokens)]
                    )
                if route_plan_snapshot.is_file():
                    client_command.extend(
                        ["--route-plan", str(route_plan_snapshot)]
                    )
                if args.max_output_tokens is not None:
                    client_command.extend(
                        ["--max-output-tokens", str(args.max_output_tokens)]
                    )
                if args.max_workflows:
                    client_command.extend(
                        ["--max-workflows", str(args.max_workflows)]
                    )
                if args.max_requests_per_workflow:
                    client_command.extend(
                        [
                            "--max-requests-per-workflow",
                            str(args.max_requests_per_workflow),
                        ]
                    )
                subprocess.run(client_command, cwd=ROOT, check=True)
                if args.collect_calibration_telemetry:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "AegisKV.demo.collect_online_calibration",
                            "--wct",
                            str(output_path),
                            "--output",
                            str(run_dir / f"{policy}.calibration.json"),
                            "--telemetry-url",
                            f"{base_url}/calibration/telemetry",
                            "--timeout",
                            "120",
                        ],
                        cwd=ROOT,
                        check=True,
                    )
            finally:
                stop_server(server)

    suite_complete = all(
        policy_result_is_complete(
            run_dir / f"{policy}.json",
            policy=policy,
            expected_workflows=args.max_workflows,
        )
        for policy in POLICIES
    )
    if suite_complete:
        render(run_dir, run_dir / "wct_cdf.png")
        print(f"\nWCT CDF artifacts: {run_dir}", flush=True)
    else:
        print(
            f"\nCompleted selected policies {args.policies}; "
            f"suite remains resumable at {run_dir}",
            flush=True,
        )
    run_lock.close()
    global_run_lock.close()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run clean online SGLSim servers for five WCT CDF policies."
    )
    parser.add_argument("--server-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dp-size", type=int, default=None)
    parser.add_argument("--pp-size", type=int, default=None)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
        help="Run only the selected policies; useful for bounded resumable jobs.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help=(
            "Resume an existing suite directory: reuse complete policy JSON files "
            "and restart each incomplete policy on a clean server."
        ),
    )
    parser.add_argument("--start-port", type=int, default=31200)
    parser.add_argument("--workflow-concurrency", type=int, default=None)
    parser.add_argument("--max-workflows", type=int, default=None)
    parser.add_argument("--max-requests-per-workflow", type=int, default=None)
    parser.add_argument("--failure-count", type=int, default=None)
    parser.add_argument(
        "--failure-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--route-plan",
        type=Path,
        default=None,
        help=(
            "Optional measured execution route plan, snapshotted into the "
            "suite and reused by every policy."
        ),
    )
    parser.add_argument(
        "--exact-window-plan",
        type=Path,
        default=None,
        help=(
            "Certified finite-window solver artifact. It is snapshotted and "
            "installed before readiness only for policy=aegiskv; baselines "
            "retain the same execution route plan without exact placement."
        ),
    )
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--arrival-speedup", type=float, default=0.0)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=7200.0,
        help="Per-request HTTP timeout; does not alter simulated latency.",
    )
    parser.add_argument(
        "--slack-calibration",
        type=Path,
        default=DEFAULT_SLACK_CALIBRATION,
    )
    parser.add_argument(
        "--slack-source",
        choices=SLACK_SOURCES,
        default="algorithm1",
    )
    parser.add_argument(
        "--collect-calibration-telemetry",
        action="store_true",
        help=(
            "Enable physical-stage Scheduler telemetry, merge the live server "
            "snapshot into <policy>.calibration.json, and persist the full "
            "Aegis event stream as append-only JSONL."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short smoke/video profile; its output is not the full-scale result.",
    )
    return parser


def resolve_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve quick/formal defaults while preserving explicit CLI overrides."""

    if args.quick:
        if args.failure_manifest is None:
            args.failure_manifest = DEFAULT_QUICK_FAILURE_MANIFEST
        if args.failure_count is None:
            args.failure_count = 2
        if args.workflow_concurrency is None:
            args.workflow_concurrency = 20
        if args.max_workflows is None:
            args.max_workflows = 20
        if args.max_requests_per_workflow is None:
            args.max_requests_per_workflow = 0
        args.input_scale = 0.008
        args.context_length = 1024
        args.max_input_tokens = 1008
        args.max_output_tokens = 16
        args.arrival_speedup = 0.0
    else:
        if args.failure_manifest is None:
            args.failure_manifest = DEFAULT_FORMAL_FAILURE_MANIFEST
        if args.failure_count is None:
            args.failure_count = 4
        if args.workflow_concurrency is None:
            args.workflow_concurrency = 12
        if args.max_workflows is None:
            args.max_workflows = 12
        if args.max_requests_per_workflow is None:
            args.max_requests_per_workflow = 128
        if args.context_length is None:
            server_config = json.loads(
                args.server_config.read_text(encoding="utf-8")
            )
            server_context = int(server_config.get("context_length", 20480))
            # Preserve the historical bounded formal profile for generic
            # 128K configs, while allowing an experiment-specific <=32K server
            # config (such as Qwen3-32B DP4/PP4) to define its own boundary.
            args.context_length = (
                server_context if server_context <= 32768 else 20480
            )
    return args


def _terminate_cleanly(signum: int, _frame) -> None:
    """Turn SIGTERM into an exception so policy cleanup ``finally`` blocks run."""

    raise SystemExit(128 + signum)


def main() -> int:
    args = resolve_profile(build_parser().parse_args())
    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_cleanly)
    try:
        run_suite(args)
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
