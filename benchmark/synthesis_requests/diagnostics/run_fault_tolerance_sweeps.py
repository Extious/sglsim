from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PYTHON = REPO / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from benchmark.synthesis_requests.run import run_experiment  # noqa: E402
from benchmark.synthesis_requests.utils import DEFAULT_RESULTS_ROOT  # noqa: E402


DEFAULT_BASE = REPO / "benchmark/synthesis_requests/config/ma_ft_gpu_backup_critical"
DEFAULT_BASELINE = REPO / "benchmark/synthesis_requests/config/ma_ft_baseline"


@dataclass(frozen=True)
class SweepCase:
    name: str
    axis: str
    value: str
    config_dir: Path
    client_patch: dict[str, Any]
    server_patch: dict[str, Any]


@dataclass(frozen=True)
class SweepRunSpec:
    case: SweepCase
    repeat_idx: int
    seed: int
    port: int
    run_idx: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _write_manifest(path: Path, manifest: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _spec_key(spec: SweepRunSpec) -> tuple[str, str, str, int, int, int]:
    case = spec.case
    return (
        case.name,
        case.axis,
        case.value,
        int(spec.repeat_idx),
        int(spec.seed),
        int(spec.port),
    )


def _entry_key(entry: dict[str, Any]) -> tuple[str, str, str, int, int, int]:
    return (
        str(entry.get("name", "")),
        str(entry.get("axis", "")),
        str(entry.get("value", "")),
        int(entry.get("repeat_idx", -1)),
        int(entry.get("seed", -1)),
        int(entry.get("port", -1)),
    )


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return [dict(entry) for entry in raw]


def _completed_keys(manifest: list[dict[str, Any]]) -> set[tuple[str, str, str, int, int, int]]:
    return {
        _entry_key(entry)
        for entry in manifest
        if entry.get("status", "completed") == "completed"
        and entry.get("run_dir")
    }


def _upsert_manifest(
    manifest: list[dict[str, Any]],
    entry: dict[str, Any],
) -> list[dict[str, Any]]:
    key = _entry_key(entry)
    return [old for old in manifest if _entry_key(old) != key] + [entry]


@contextmanager
def _run_timeout(timeout_s: float | None):
    if timeout_s is None or timeout_s <= 0:
        yield
        return
    old_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(f"run exceeded timeout_s={timeout_s}")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _case_config(
    spec: SweepRunSpec,
    root: Path,
) -> Path:
    case = spec.case
    src_client = _read_json(case.config_dir / "client.json")
    src_server = _read_json(case.config_dir / "server.json")
    client = _deep_update(src_client, case.client_patch)
    client["seed"] = int(spec.seed)
    server = _deep_update(src_server, case.server_patch)
    server["port"] = int(spec.port)
    cfg = root / f"{case.name}_r{spec.repeat_idx}"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "client.json").write_text(json.dumps(client, indent=2), encoding="utf-8")
    (cfg / "server.json").write_text(json.dumps(server, indent=2), encoding="utf-8")
    return cfg


def _client_workload_summary(client: dict[str, Any]) -> dict[str, Any]:
    groups = int(client.get("gsp_num_groups", 0) or 0)
    prompts_per_group = int(client.get("gsp_prompts_per_group", 0) or 0)
    turns = int(client.get("gsp_num_turns", 1) or 1)
    agents = groups * prompts_per_group
    num_prompts = client.get("num_prompts")
    if num_prompts is not None:
        agents = min(agents, int(num_prompts))
    return {
        "agents": agents,
        "turns_per_agent": turns,
        "http_requests": agents * turns,
        "max_concurrency": client.get("max_concurrency"),
        "system_prompt_len": client.get("gsp_system_prompt_len"),
        "question_len": client.get("gsp_question_len"),
        "output_len": client.get("gsp_output_len"),
        "critical_agent_ids": client.get("gsp_critical_agent_ids", []),
        "critical_agent_ratio": client.get("gsp_critical_agent_ratio", 0.0),
    }


def _server_summary(server: dict[str, Any]) -> dict[str, Any]:
    scheduler = server.get("scheduler", {})
    ft = server.get("fault_tolerance", {})
    trigger = ft.get("trigger", {}) if isinstance(ft, dict) else {}
    return {
        "model_path": server.get("model_path"),
        "port": server.get("port"),
        "dp_size": scheduler.get("dp_size"),
        "enable_hierarchical_cache": (server.get("cache") or {}).get(
            "enable_hierarchical_cache"
        ),
        "predictor": (server.get("predictor") or {}).get("name"),
        "ft_strategy": ft.get("strategy") if isinstance(ft, dict) else None,
        "backup_policy": ft.get("backup_policy") if isinstance(ft, dict) else None,
        "peer_backup_lag_tokens": (
            ft.get("peer_backup_lag_tokens") if isinstance(ft, dict) else None
        ),
        "inject_after_s": trigger.get("inject_after_s"),
        "recover_after_s": trigger.get("recover_after_s"),
    }


def build_run_specs(
    cases: list[SweepCase],
    *,
    repeats: int,
    seed_base: int,
    port_base: int,
) -> list[SweepRunSpec]:
    specs: list[SweepRunSpec] = []
    run_idx = 0
    for repeat_idx in range(int(repeats)):
        seed = int(seed_base) + repeat_idx
        for case in cases:
            specs.append(
                SweepRunSpec(
                    case=case,
                    repeat_idx=repeat_idx,
                    seed=seed,
                    port=int(port_base) + run_idx,
                    run_idx=run_idx,
                )
            )
            run_idx += 1
    return specs


def _split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def filter_cases(
    cases: list[SweepCase],
    *,
    names: str | None = None,
    axes: str | None = None,
) -> list[SweepCase]:
    name_filter = _split_csv(names)
    axis_filter = _split_csv(axes)
    filtered = [
        case
        for case in cases
        if (not name_filter or case.name in name_filter)
        and (not axis_filter or case.axis in axis_filter)
    ]
    if not filtered:
        raise ValueError(
            f"No sweep cases matched names={sorted(name_filter)} axes={sorted(axis_filter)}"
        )
    return filtered


def build_preview(specs: list[SweepRunSpec]) -> dict[str, Any]:
    axis_values: dict[str, set[str]] = {}
    entries: list[dict[str, Any]] = []
    for spec in specs:
        case = spec.case
        src_client = _read_json(case.config_dir / "client.json")
        src_server = _read_json(case.config_dir / "server.json")
        client = _deep_update(src_client, case.client_patch)
        client["seed"] = int(spec.seed)
        server = _deep_update(src_server, case.server_patch)
        server["port"] = int(spec.port)
        axis_values.setdefault(case.axis, set()).add(case.value)
        entries.append(
            {
                "name": case.name,
                "axis": case.axis,
                "value": case.value,
                "repeat_idx": spec.repeat_idx,
                "seed": spec.seed,
                "port": spec.port,
                "config_dir": str(case.config_dir),
                "client": _client_workload_summary(client),
                "server": _server_summary(server),
            }
        )
    return {
        "total_runs": len(specs),
        "axis_value_counts": {
            axis: len(values) for axis, values in sorted(axis_values.items())
        },
        "seeds": sorted({spec.seed for spec in specs}),
        "ports": [spec.port for spec in specs],
        "runs": entries,
    }


def build_status(
    specs: list[SweepRunSpec],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {_entry_key(entry): entry for entry in manifest}
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for spec in specs:
        case = spec.case
        base = {
            "name": case.name,
            "axis": case.axis,
            "value": case.value,
            "repeat_idx": spec.repeat_idx,
            "seed": spec.seed,
            "port": spec.port,
        }
        entry = by_key.get(_spec_key(spec))
        if entry is None:
            pending.append(base)
        elif entry.get("status", "completed") == "completed" and entry.get("run_dir"):
            completed.append({**base, "run_dir": entry.get("run_dir")})
        elif entry.get("status") == "failed":
            failed.append({**base, "error": entry.get("error", "")})
        else:
            pending.append({**base, "status": entry.get("status", "unknown")})
    axis_completed: dict[str, int] = {}
    for row in completed:
        axis = str(row["axis"])
        axis_completed[axis] = axis_completed.get(axis, 0) + 1
    return {
        "total_runs": len(specs),
        "completed": len(completed),
        "failed": len(failed),
        "pending": len(pending),
        "axis_completed": axis_completed,
        "next_pending": pending[:5],
        "failed_entries": failed,
    }


def build_cases(
    *,
    base_config: Path,
    baseline_config: Path,
    smoke: bool,
) -> list[SweepCase]:
    if smoke:
        common_client = {
            "model_path": None,
        }
        del common_client
        mock_client = {
            "gsp_system_prompt_len": 16,
            "gsp_question_len": 16,
            "gsp_output_len": 40,
            "gsp_num_turns": 2,
            "gsp_num_groups": 4,
            "max_concurrency": 4,
        }
        mock_server = {
            "model_path": "mock-model",
            "predictor": {"name": "constant", "prefill_s": 0.02, "decode_s": 0.02},
            "fault_tolerance": {"trigger": {"inject_after_s": 0.08, "recover_after_s": 0.5}},
        }
    else:
        mock_client = {}
        mock_server = {}
    default_server = (
        mock_server
        if smoke
        else _deep_update(
            mock_server,
            {"fault_tolerance": {"trigger": {"inject_after_s": 40}}},
        )
    )

    cases: list[SweepCase] = [
        SweepCase(
            name="baseline",
            axis="strategy",
            value="baseline",
            config_dir=baseline_config,
            client_patch=mock_client,
            server_patch=default_server,
        ),
        SweepCase(
            name="gpu_critical",
            axis="strategy",
            value="gpu_critical",
            config_dir=base_config,
            client_patch=mock_client,
            server_patch=default_server,
        ),
    ]

    for lag in ([1, 4] if smoke else [0, 1, 4, 8, 16, 32]):
        cases.append(
            SweepCase(
                name=f"lag_{lag}",
                axis="backup_lag",
                value=str(lag),
                config_dir=base_config,
                client_patch=mock_client,
                server_patch=_deep_update(
                    default_server,
                    {"fault_tolerance": {"peer_backup_lag_tokens": lag}},
                ),
            )
        )

    for ratio in ([0.25, 0.5] if smoke else [0.0, 0.25, 0.5, 0.75, 1.0]):
        client_patch = _deep_update(
            mock_client,
            {
                "gsp_critical_agent_ids": [],
                "gsp_critical_agent_ratio": ratio,
            },
        )
        cases.append(
            SweepCase(
                name=f"critical_ratio_{ratio:g}",
                axis="critical_ratio",
                value=str(ratio),
                config_dir=base_config,
                client_patch=client_patch,
                server_patch=default_server,
            )
        )

    for inject_after_s in ([0.04, 0.12] if smoke else [30, 36, 40, 44]):
        cases.append(
            SweepCase(
                name=f"failure_t_{inject_after_s:g}",
                axis="failure_timing_s",
                value=str(inject_after_s),
                config_dir=base_config,
                client_patch=mock_client,
                server_patch=_deep_update(
                    mock_server,
                    {"fault_tolerance": {"trigger": {"inject_after_s": inject_after_s}}},
                ),
            )
        )

    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HPCA-style FT sweep matrix")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independent seeds to run for each sweep point.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1,
        help="First workload seed; repeat r uses seed-base + r.",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=None,
        help="Override the first server port for the generated matrix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and print a sweep preview without starting servers.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print progress against the expected sweep matrix without starting servers.",
    )
    parser.add_argument(
        "--print-runs",
        action="store_true",
        help="With --dry-run, print the full run list instead of only the summary.",
    )
    parser.add_argument(
        "--case-names",
        type=str,
        default=None,
        help="Comma-separated case names to run, e.g. baseline,gpu_critical.",
    )
    parser.add_argument(
        "--axes",
        type=str,
        default=None,
        help="Comma-separated sweep axes to run, e.g. strategy,backup_lag.",
    )
    parser.add_argument(
        "--run-timeout-s",
        type=float,
        default=None,
        help="Optional wall-clock timeout for each individual run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed entries already present in fault_tolerance_sweep_latest.json.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of new runs to execute in this invocation.",
    )
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")

    cases = build_cases(
        base_config=args.base_config,
        baseline_config=args.baseline_config,
        smoke=bool(args.smoke),
    )
    cases = filter_cases(cases, names=args.case_names, axes=args.axes)
    default_port = _read_json(cases[0].config_dir / "server.json").get("port", 30000)
    port_base = int(args.port_base if args.port_base is not None else default_port)
    specs = build_run_specs(
        cases,
        repeats=int(args.repeats),
        seed_base=int(args.seed_base),
        port_base=port_base,
    )
    manifest_path = args.results_root / "fault_tolerance_sweep_latest.json"
    if args.status:
        manifest = _load_manifest(manifest_path)
        print(json.dumps(build_status(specs, manifest), indent=2))
        return 0

    if args.dry_run:
        args.results_root.mkdir(parents=True, exist_ok=True)
        preview = build_preview(specs)
        preview_path = args.results_root / "fault_tolerance_sweep_preview.json"
        preview_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
        printed = preview if args.print_runs else {k: v for k, v in preview.items() if k != "runs"}
        printed["preview_path"] = str(preview_path)
        print(json.dumps(printed, indent=2))
        return 0

    manifest: list[dict[str, Any]] = _load_manifest(manifest_path) if args.resume else []
    completed = _completed_keys(manifest)
    new_runs = 0
    with tempfile.TemporaryDirectory(prefix="sglsim-ft-sweep-") as td:
        cfg_root = Path(td)
        for spec in specs:
            case = spec.case
            if args.resume and _spec_key(spec) in completed:
                continue
            if args.max_runs is not None and new_runs >= args.max_runs:
                break
            cfg = _case_config(spec, cfg_root)
            entry = {
                "name": case.name,
                "axis": case.axis,
                "value": case.value,
                "repeat_idx": spec.repeat_idx,
                "seed": spec.seed,
                "port": spec.port,
                "status": "running",
            }
            try:
                with _run_timeout(args.run_timeout_s):
                    run_dir = run_experiment(
                        cfg,
                        args.results_root,
                        log_level=args.log_level,
                    )
                entry.update({"status": "completed", "run_dir": str(run_dir)})
                manifest = _upsert_manifest(manifest, entry)
                _write_manifest(manifest_path, manifest)
                new_runs += 1
            except Exception as exc:
                entry.update({"status": "failed", "error": repr(exc)})
                manifest = _upsert_manifest(manifest, entry)
                _write_manifest(manifest_path, manifest)
                raise

    _write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
