"""Baseline quality report for synthesis runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


DIAG = Path(__file__).resolve().parent
REPO = DIAG.parents[2]
RESULTS_ROOT = REPO / "benchmark/synthesis_requests/results"
sys.path.insert(0, str(DIAG))
from parity_refs import REF_NO_OVERLAP_NAME, REF_OVERLAP_NAME  # noqa: E402
from run_integrity import check_run_integrity  # noqa: E402

DEFAULT_RUNS: list[str] = []


def _mean(path: Path, col: str) -> float | None:
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    if not vals:
        return None
    return statistics.mean(vals)


def _metric_duration(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("duration")
    if value is None:
        return None
    return float(value)


def _load_run_health(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@dataclass
class RunQuality:
    log_exists: bool
    log_lines: int
    has_decode_signal: bool
    has_busy_wait: bool
    has_retract: bool

    @property
    def ok(self) -> bool:
        # Keep this conservative: both normal decode signal and reasonable line count.
        return self.log_exists and self.log_lines >= 120 and self.has_decode_signal


def _quality(log_path: Path) -> RunQuality:
    if not log_path.is_file():
        return RunQuality(
            log_exists=False,
            log_lines=0,
            has_decode_signal=False,
            has_busy_wait=False,
            has_retract=False,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    return RunQuality(
        log_exists=True,
        log_lines=len(lines),
        has_decode_signal=("Decode forward start" in text or "Decode batch" in text),
        has_busy_wait=("Scheduler busy-wait" in text),
        has_retract=("Retract requests" in text),
    )


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}"


def _run_row(
    run_dir: Path,
    ref_duration: float | None,
    *,
    ref_dir: Path | None = None,
) -> dict[str, str]:
    csv_path = run_dir / "request_details.csv"
    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "server.log"
    stderr_path = run_dir / "server.stderr.log"
    health = _load_run_health(run_dir / "run_health.json")
    fail_reason = str(health.get("fail_reason", "UNKNOWN"))
    quality = _quality(log_path)
    duration = _metric_duration(metrics_path)
    waiting = _mean(csv_path, "waiting_time_s")
    infer = _mean(csv_path, "inference_time_s")
    svc = _mean(csv_path, "logical_service_time_s")
    total = _mean(csv_path, "total_latency_s")
    ratio = None
    if duration is not None and ref_duration is not None and ref_duration > 0:
        ratio = duration / ref_duration
    infer_svc = None
    if infer is not None and svc is not None:
        infer_svc = infer - svc
    has_csv = csv_path.is_file()
    has_metrics = metrics_path.is_file()
    has_log = log_path.is_file()
    has_stderr = stderr_path.is_file()
    integrity = check_run_integrity(run_dir, ref_dir=ref_dir)
    partial_state = "COMPLETE"
    if not (has_csv and has_metrics):
        if fail_reason == "MATRIX_TIMEOUT":
            partial_state = "PARTIAL_TIMEOUT"
        elif fail_reason == "RUNPY_RC_NONZERO":
            partial_state = "PARTIAL_CRASH"
        elif fail_reason == "NO_RUN_DIR":
            partial_state = "NO_RUN_DIR"
        else:
            partial_state = "PARTIAL_UNKNOWN"
    return {
        "run_id": run_dir.name,
        "partial_state": partial_state,
        "fail_reason": fail_reason,
        "has_request_details": str(has_csv),
        "has_metrics": str(has_metrics),
        "has_server_log": str(has_log),
        "has_server_stderr": str(has_stderr),
        "quality": "OK" if quality.ok else "WARN",
        "integrity": "OK" if integrity.ok else "WARN",
        "integrity_failures": ";".join(integrity.failures) or "-",
        "request_rows": str(integrity.row_count),
        "completed_rows": str(integrity.completed_rows),
        "log_lines": str(quality.log_lines),
        "decode_signal": str(quality.has_decode_signal),
        "busy_wait": str(quality.has_busy_wait),
        "retract": str(quality.has_retract),
        "makespan_s": _fmt(duration),
        "makespan_ratio": _fmt(ratio),
        "waiting_mean_s": _fmt(waiting),
        "inference_mean_s": _fmt(infer),
        "logical_service_mean_s": _fmt(svc),
        "infer_minus_svc_mean_s": _fmt(infer_svc),
        "total_mean_s": _fmt(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline quality report for synthesis runs.")
    parser.add_argument(
        "runs",
        nargs="*",
        default=DEFAULT_RUNS,
        help="Run ids under benchmark/synthesis_requests/results/",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
    )
    parser.add_argument(
        "--ref-run",
        type=str,
        default=REF_NO_OVERLAP_NAME,
        help="Reference run id for makespan_ratio (default: no-overlap SGLang ref).",
    )
    args = parser.parse_args()

    root = args.results_root
    if not args.runs:
        latest = sorted(
            (
                p.name
                for p in root.iterdir()
                if p.is_dir()
                and p.name not in {REF_NO_OVERLAP_NAME, REF_OVERLAP_NAME}
                and (p / "metrics.json").is_file()
            ),
            reverse=True,
        )
        args.runs = latest[:3]
    ref_duration = _metric_duration(root / args.ref_run / "metrics.json")
    ref_dir = root / args.ref_run
    rows: list[dict[str, str]] = []
    for run_id in args.runs:
        rows.append(_run_row(root / run_id, ref_duration, ref_dir=ref_dir))

    headers = [
        "run_id",
        "partial_state",
        "fail_reason",
        "has_request_details",
        "has_metrics",
        "has_server_log",
        "has_server_stderr",
        "quality",
        "integrity",
        "integrity_failures",
        "request_rows",
        "completed_rows",
        "log_lines",
        "decode_signal",
        "busy_wait",
        "retract",
        "makespan_s",
        "makespan_ratio",
        "waiting_mean_s",
        "inference_mean_s",
        "logical_service_mean_s",
        "infer_minus_svc_mean_s",
        "total_mean_s",
    ]
    print(",".join(headers))
    for row in rows:
        print(",".join(row[h] for h in headers))


if __name__ == "__main__":
    main()
