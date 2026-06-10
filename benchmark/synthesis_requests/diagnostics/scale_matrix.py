"""Run a small predictor matrix and summarize gate signals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[2]
_PYTHON_ROOT = _REPO_ROOT / "python"
for _path in (_REPO_ROOT, _PYTHON_ROOT):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from benchmark.synthesis_requests.diagnostics.latency_budget import analyze_run  # noqa: E402
from benchmark.synthesis_requests.utils import (  # noqa: E402
    CLIENT_LOG_PATH,
    DEFAULT_CONFIG_DIR,
    DEFAULT_RESULTS_ROOT,
    SERVER_LOG_PATH,
    SERVER_STDERR_LOG_PATH,
)


REPO = _REPO_ROOT
_DIAG = REPO / "benchmark/synthesis_requests/diagnostics"
sys.path.insert(0, str(_DIAG))
from parity_refs import REF_NO_OVERLAP  # noqa: E402

REF_DIR = REF_NO_OVERLAP
DEFAULT_MATRIX_CONFIG_DIR = DEFAULT_CONFIG_DIR / "parity_full_no_overlap"
FAIL_COMPLETE = "COMPLETE"
FAIL_TIMEOUT = "MATRIX_TIMEOUT"
FAIL_RUNPY_NONZERO = "RUNPY_RC_NONZERO"
FAIL_NO_RUN_DIR = "NO_RUN_DIR"


@dataclass(frozen=True)
class Variant:
    name: str
    forward_scale: float
    prefill_scale_factor: float
    decode_scale_hicache: float
    init_new_token_ratio: float


@dataclass(frozen=True)
class RunAttempt:
    run_dir: Path | None
    fail_reason: str
    elapsed_s: float
    return_code: int | None


DEFAULT_VARIANTS = [
    Variant(
        name="s1.00",
        forward_scale=1.00,
        prefill_scale_factor=1.05,
        decode_scale_hicache=0.64,
        init_new_token_ratio=0.70,
    ),
    Variant(
        name="s1.10",
        forward_scale=1.10,
        prefill_scale_factor=1.05,
        decode_scale_hicache=0.64,
        init_new_token_ratio=0.70,
    ),
    Variant(
        name="s1.15",
        forward_scale=1.15,
        prefill_scale_factor=1.05,
        decode_scale_hicache=0.64,
        init_new_token_ratio=0.70,
    ),
]


def _ratio(sim: float, ref: float) -> float:
    if ref <= 0:
        return float("inf")
    return sim / ref


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _new_run_dirs(results_root: Path, before: set[str]) -> list[Path]:
    return [p for p in results_root.iterdir() if p.is_dir() and p.name not in before]


def _latest_run_dir(run_dirs: list[Path]) -> Path | None:
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _cleanup_launch_server_for_runs(run_dirs: list[Path]) -> None:
    for run_dir in run_dirs:
        sim_cfg = run_dir / "simulator_config.json"
        if not sim_cfg.is_file():
            continue
        # run.py may leave detached launch_server descendants on timeout.
        subprocess.run(
            ["pkill", "-f", str(sim_cfg)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _archive_logs_to_run_dir(run_dir: Path) -> None:
    for src, name in (
        (SERVER_LOG_PATH, "server.log"),
        (SERVER_STDERR_LOG_PATH, "server.stderr.log"),
        (CLIENT_LOG_PATH, "client.log"),
    ):
        if src.is_file():
            shutil.copy2(src, run_dir / name)


def _artifact_flags(run_dir: Path | None) -> dict[str, str]:
    if run_dir is None:
        return {
            "has_sim_config": "0",
            "has_request_details": "0",
            "has_metrics": "0",
            "has_server_log": "0",
            "has_server_stderr": "0",
            "has_client_log": "0",
        }
    return {
        "has_sim_config": "1" if (run_dir / "simulator_config.json").is_file() else "0",
        "has_request_details": "1" if (run_dir / "request_details.csv").is_file() else "0",
        "has_metrics": "1" if (run_dir / "metrics.json").is_file() else "0",
        "has_server_log": "1" if (run_dir / "server.log").is_file() else "0",
        "has_server_stderr": "1" if (run_dir / "server.stderr.log").is_file() else "0",
        "has_client_log": "1" if (run_dir / "client.log").is_file() else "0",
    }


def _artifact_flags_compact(flags: dict[str, str]) -> str:
    return "|".join(f"{k}={v}" for k, v in flags.items())


def _write_run_health(
    run_dir: Path,
    *,
    variant: Variant,
    fail_reason: str,
    elapsed_s: float,
    return_code: int | None,
    timeout_s: float,
) -> None:
    payload = {
        "variant": variant.name,
        "fail_reason": fail_reason,
        "elapsed_s": round(elapsed_s, 6),
        "timeout_s": float(timeout_s),
        "return_code": return_code,
        "artifact_flags": _artifact_flags(run_dir),
    }
    (run_dir / "run_health.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_variant(
    *,
    base_config_dir: Path,
    results_root: Path,
    variant: Variant,
    log_level: str,
    timeout_s: float,
    disable_overlap_schedule: bool = False,
) -> RunAttempt:
    client = _load_json(base_config_dir / "client.json")
    server = _load_json(base_config_dir / "server.json")
    server.setdefault("predictor", {})
    server.setdefault("scheduler", {})
    server["predictor"]["forward_scale"] = variant.forward_scale
    server["predictor"]["prefill_scale_factor"] = variant.prefill_scale_factor
    server["predictor"]["decode_scale_hicache"] = variant.decode_scale_hicache
    server["scheduler"]["init_new_token_ratio"] = variant.init_new_token_ratio
    if disable_overlap_schedule:
        server["scheduler"]["disable_overlap_schedule"] = True

    with tempfile.TemporaryDirectory(prefix=f"scale-matrix-{variant.name}-") as td:
        cfg_dir = Path(td)
        _write_json(cfg_dir / "client.json", client)
        _write_json(cfg_dir / "server.json", server)
        before = {p.name for p in results_root.iterdir() if p.is_dir()}
        cmd = [
            sys.executable,
            "benchmark/synthesis_requests/run.py",
            "--config-dir",
            str(cfg_dir),
            "--results-root",
            str(results_root),
            "--log-level",
            log_level,
        ]
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Ensure launch_server children in the same process group are torn down.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5.0)
            elapsed = time.monotonic() - start
            run_dirs = _new_run_dirs(results_root, before)
            for run_dir in run_dirs:
                _archive_logs_to_run_dir(run_dir)
            _cleanup_launch_server_for_runs(run_dirs)
            latest = _latest_run_dir(run_dirs)
            if latest is not None:
                _write_run_health(
                    latest,
                    variant=variant,
                    fail_reason=FAIL_TIMEOUT,
                    elapsed_s=elapsed,
                    return_code=None,
                    timeout_s=timeout_s,
                )
            return RunAttempt(
                run_dir=latest,
                fail_reason=FAIL_TIMEOUT,
                elapsed_s=elapsed,
                return_code=None,
            )
        elapsed = time.monotonic() - start
        if rc != 0:
            run_dirs = _new_run_dirs(results_root, before)
            for run_dir in run_dirs:
                _archive_logs_to_run_dir(run_dir)
            _cleanup_launch_server_for_runs(run_dirs)
            latest = _latest_run_dir(run_dirs)
            if latest is not None:
                _write_run_health(
                    latest,
                    variant=variant,
                    fail_reason=FAIL_RUNPY_NONZERO,
                    elapsed_s=elapsed,
                    return_code=rc,
                    timeout_s=timeout_s,
                )
            return RunAttempt(
                run_dir=latest,
                fail_reason=FAIL_RUNPY_NONZERO,
                elapsed_s=elapsed,
                return_code=rc,
            )
        after = _new_run_dirs(results_root, before)
        latest = _latest_run_dir(after)
        if latest is None:
            return RunAttempt(
                run_dir=None,
                fail_reason=FAIL_NO_RUN_DIR,
                elapsed_s=elapsed,
                return_code=rc,
            )
        _write_run_health(
            latest,
            variant=variant,
            fail_reason=FAIL_COMPLETE,
            elapsed_s=elapsed,
            return_code=rc,
            timeout_s=timeout_s,
        )
        return RunAttempt(
            run_dir=latest,
            fail_reason=FAIL_COMPLETE,
            elapsed_s=elapsed,
            return_code=rc,
        )


def _summarize(run_dir: Path, ref_dir: Path, variant: Variant) -> dict[str, str]:
    sim = analyze_run(run_dir)
    ref = analyze_run(ref_dir)
    ref_e2e = ref.get("e2e", {})
    infer = sim.get("inference", {})
    duration = float(sim.get("duration_s", {}).get("mean", 0.0))
    ref_duration = float(ref.get("duration_s", {}).get("mean", 0.0))
    wait = float(sim.get("waiting", {}).get("mean", 0.0))
    svc = float(sim.get("logical_service", {}).get("mean", 0.0))
    infer_mean = float(infer.get("mean", 0.0))
    infer_p50 = float(infer.get("p50", 0.0))
    infer_p90 = float(infer.get("p90", 0.0))
    ref_mean = float(ref_e2e.get("mean", 0.0))
    ref_p50 = float(ref_e2e.get("p50", 0.0))
    ref_p90 = float(ref_e2e.get("p90", 0.0))
    infer_svc = infer_mean - svc
    return {
        "variant": variant.name,
        "run_id": run_dir.name,
        "forward_scale": f"{variant.forward_scale:.3f}",
        "prefill_scale_factor": f"{variant.prefill_scale_factor:.3f}",
        "decode_scale_hicache": f"{variant.decode_scale_hicache:.3f}",
        "init_new_token_ratio": f"{variant.init_new_token_ratio:.3f}",
        "forward_mean_ratio": f"{_ratio(infer_mean, ref_mean):.3f}",
        "forward_p50_ratio": f"{_ratio(infer_p50, ref_p50):.3f}",
        "forward_p90_ratio": f"{_ratio(infer_p90, ref_p90):.3f}",
        "makespan_ratio": f"{_ratio(duration, ref_duration):.3f}",
        "waiting_mean_s": f"{wait:.3f}",
        "infer_minus_svc_mean_s": f"{infer_svc:.3f}",
    }


def _base_row(variant: Variant, attempt: RunAttempt) -> dict[str, str]:
    flags = _artifact_flags(attempt.run_dir)
    row = {
        "variant": variant.name,
        "run_id": attempt.run_dir.name if attempt.run_dir is not None else "-",
        "fail_reason": attempt.fail_reason,
        "elapsed_s": f"{attempt.elapsed_s:.3f}",
        "return_code": (
            str(attempt.return_code) if attempt.return_code is not None else "-"
        ),
        "forward_scale": f"{variant.forward_scale:.3f}",
        "prefill_scale_factor": f"{variant.prefill_scale_factor:.3f}",
        "decode_scale_hicache": f"{variant.decode_scale_hicache:.3f}",
        "init_new_token_ratio": f"{variant.init_new_token_ratio:.3f}",
        "artifact_flags": _artifact_flags_compact(flags),
        **flags,
        "forward_mean_ratio": "-",
        "forward_p50_ratio": "-",
        "forward_p90_ratio": "-",
        "makespan_ratio": "-",
        "waiting_mean_s": "-",
        "infer_minus_svc_mean_s": "-",
    }
    return row


def _parse_variants(raw: list[str] | None) -> list[Variant]:
    if not raw:
        return list(DEFAULT_VARIANTS)
    out: list[Variant] = []
    for item in raw:
        # name,bf,prefill,decode_hicache,init_ratio
        parts = item.split(",")
        if len(parts) != 5:
            raise ValueError(
                "variant format must be name,bf,prefill,decode_hicache,init_ratio"
            )
        out.append(
            Variant(
                name=parts[0],
                forward_scale=float(parts[1]),
                prefill_scale_factor=float(parts[2]),
                decode_scale_hicache=float(parts[3]),
                init_new_token_ratio=float(parts[4]),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictor scale matrix runner.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_MATRIX_CONFIG_DIR,
        help="Benchmark config directory. Defaults to no_overlap profile.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--ref-dir", type=Path, default=REF_DIR)
    parser.add_argument("--log-level", default="warning")
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help="Force scheduler.disable_overlap_schedule=true for each matrix variant.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=900.0,
        help="Per-variant benchmark timeout in seconds (parity_full ~300s makespan).",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="name,bf,prefill,decode_hicache,init_ratio (can be repeated)",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional output CSV path for the matrix summary.",
    )
    args = parser.parse_args()

    variants = _parse_variants(args.variant)
    rows: list[dict[str, str]] = []
    for variant in variants:
        print(f"[matrix] running {variant.name} ...")
        attempt = _run_variant(
            base_config_dir=args.config_dir,
            results_root=args.results_root,
            variant=variant,
            log_level=args.log_level,
            timeout_s=args.timeout_s,
            disable_overlap_schedule=bool(args.disable_overlap_schedule),
        )
        row = _base_row(variant, attempt)
        if attempt.fail_reason != FAIL_COMPLETE or attempt.run_dir is None:
            rows.append(row)
            print(
                f"[matrix] {variant.name} fail reason={attempt.fail_reason} "
                f"elapsed={attempt.elapsed_s:.1f}s run={row['run_id']}"
            )
            continue
        row.update(_summarize(attempt.run_dir, args.ref_dir, variant))
        rows.append(row)
        print(
            "[matrix] done",
            row["variant"],
            "run=",
            row["run_id"],
            "forward(mean/p50/p90)=",
            row["forward_mean_ratio"],
            row["forward_p50_ratio"],
            row["forward_p90_ratio"],
            "makespan=",
            row["makespan_ratio"],
        )

    if not rows:
        return

    headers = list(rows[0].keys())
    print(",".join(headers))
    for row in rows:
        print(",".join(row[h] for h in headers))

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[matrix] wrote {args.csv_out}")


if __name__ == "__main__":
    main()
