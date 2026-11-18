#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
PYTHON_ROOT = REPO_ROOT / "python"
for path in (REPO_ROOT, PYTHON_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
DRAFT_EXPERIMENTS = SGLSIM_LARGE_ROOT.parent
RESULTS_ROOT = SGLSIM_LARGE_ROOT / "results" / "agentic_workflows"
DEFAULT_OUTPUT_DIR = SGLSIM_LARGE_ROOT / "results" / "strict_final_readiness"
DEFAULT_PHYSICAL_AGGREGATE = (
    DRAFT_EXPERIMENTS
    / "sglang_small"
    / "results"
    / "physical_recovery_path_aggregate"
    / "physical_recovery_path_aggregate.csv"
)

DEFAULT_SUITES: tuple[tuple[str, str, str], ...] = (
    (
        "fig8_9_10_policy_sweep",
        "Fig. 8/9/10",
        "sglsim-staged-13131-policy-stress-v2-r3-20260702",
    ),
    (
        "fig10_14_recovery_action_suite",
        "Fig. 10/14",
        "sglsim-current-physical-recovery-action-agentic-r3-20260702",
    ),
)

RESTORE_ACTION_TO_PHYSICAL_ACTION = {
    "gpu_replica": "gpu_replica_restore_p2p",
    "host_restore": "host_restore_h2d",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: object) -> int:
    try:
        if value in ("", None):
            return 0
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_measurements(path: Path) -> dict[str, list[dict[str, str]]]:
    measurements: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(path):
        action = str(row.get("action") or "").strip()
        phase = str(row.get("phase") or "").strip().lower()
        if not action or phase != "restore":
            continue
        measurements.setdefault(action, []).append(row)
    for rows in measurements.values():
        rows.sort(key=lambda item: _int(item.get("tokens")))
    return measurements


def select_measurement(
    measurements: dict[str, list[dict[str, str]]],
    *,
    physical_action: str,
    restored_tokens: int,
) -> dict[str, str] | None:
    rows = measurements.get(physical_action, [])
    if not rows:
        return None
    for row in rows:
        if _int(row.get("tokens")) >= restored_tokens:
            return row
    return None


def _iter_restore_rows(
    *,
    suite_label: str,
    figures: str,
    suite_root: Path,
    measurements: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request_csv in sorted(suite_root.glob("**/request_details.csv")):
        for row in _read_csv(request_csv):
            if not _bool(row.get("aegis_failover_replay")):
                continue
            restore_action = str(row.get("aegis_replay_restore_action") or "").strip()
            physical_action = RESTORE_ACTION_TO_PHYSICAL_ACTION.get(restore_action)
            if physical_action is None:
                continue
            restored_tokens = _int(row.get("aegis_replay_restored_tokens"))
            measurement = select_measurement(
                measurements,
                physical_action=physical_action,
                restored_tokens=restored_tokens,
            )
            bridge_eligible = measurement is not None and restored_tokens > 0
            rows.append(
                {
                    "suite_label": suite_label,
                    "figures": figures,
                    "run_dir": str(request_csv.parent),
                    "request_csv": str(request_csv),
                    "rid": str(row.get("rid") or ""),
                    "restore_action": restore_action,
                    "restored_tokens": restored_tokens,
                    "sim_data_plane": str(
                        row.get("aegis_replay_restore_data_plane") or ""
                    ).strip(),
                    "sim_timing_source": str(
                        row.get("aegis_replay_restore_timing_source") or ""
                    ).strip(),
                    "physical_action": physical_action,
                    "measurement_tokens": _int(measurement.get("tokens")) if measurement else 0,
                    "measurement_p50_ms": _float(measurement.get("p50_ms_mean")) if measurement else 0.0,
                    "measurement_p95_ms": _float(measurement.get("p95_ms_mean")) if measurement else 0.0,
                    "measurement_ci95_ms": _float(measurement.get("p50_ms_ci95")) if measurement else 0.0,
                    "measurement_evidence_tier": str(
                        measurement.get("evidence_tier") if measurement else ""
                    ),
                    "measurement_source": str(
                        measurement.get("source") if measurement else ""
                    ),
                    "measurement_jobs": str(
                        measurement.get("slurm_job_ids") if measurement else ""
                    ),
                    "bridge_eligible": bridge_eligible,
                    "bridge_boundary": (
                        "physical calibration coverage only; runtime row remains "
                        "non-final unless sim_data_plane is physical_tensor_copy"
                    ),
                }
            )
    return rows


def build_bridge_rows(
    *,
    specs: list[tuple[str, str, Path]],
    physical_aggregate: Path,
) -> list[dict[str, Any]]:
    measurements = load_measurements(physical_aggregate)
    rows: list[dict[str, Any]] = []
    for label, figures, suite_root in specs:
        if not suite_root.exists():
            raise FileNotFoundError(f"missing suite root: {suite_root}")
        rows.extend(
            _iter_restore_rows(
                suite_label=label,
                figures=figures,
                suite_root=suite_root,
                measurements=measurements,
            )
        )
    return rows


def summarize_bridge(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_suite: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_suite.setdefault((str(row["suite_label"]), str(row["figures"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (suite_label, figures), suite_rows in sorted(by_suite.items()):
        action_counts = Counter(str(row["restore_action"]) for row in suite_rows)
        physical_counts = Counter(str(row["physical_action"]) for row in suite_rows)
        eligible = sum(1 for row in suite_rows if bool(row["bridge_eligible"]))
        summary.append(
            {
                "suite_label": suite_label,
                "figures": figures,
                "restore_rows": len(suite_rows),
                "bridge_eligible_rows": eligible,
                "bridge_missing_rows": len(suite_rows) - eligible,
                "bridge_ready": len(suite_rows) > 0 and eligible == len(suite_rows),
                "max_restored_tokens": max(
                    (_int(row["restored_tokens"]) for row in suite_rows),
                    default=0,
                ),
                "max_measurement_tokens": max(
                    (_int(row["measurement_tokens"]) for row in suite_rows),
                    default=0,
                ),
                "restore_actions": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
                "physical_actions": json.dumps(dict(sorted(physical_counts.items())), sort_keys=True),
                "evidence_tiers": json.dumps(
                    sorted(
                        {
                            str(row["measurement_evidence_tier"])
                            for row in suite_rows
                            if str(row["measurement_evidence_tier"])
                        }
                    )
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: list[dict[str, Any]], rows_csv: Path) -> str:
    total_restore = sum(_int(row.get("restore_rows")) for row in summary)
    total_eligible = sum(_int(row.get("bridge_eligible_rows")) for row in summary)
    total_missing = sum(_int(row.get("bridge_missing_rows")) for row in summary)
    bridge_ready = total_restore > 0 and total_eligible == total_restore
    lines = [
        "# P0 Physical Calibration Bridge",
        "",
        "This report maps every P0 SGLSim restore row to a real CUDA recovery-path",
        "microbenchmark row with the same recovery action and a token bucket that covers",
        "the restored-token count. It is bridge evidence for calibrated latency, not",
        "end-to-end live physical KV movement.",
        "",
        "## Summary",
        "",
        f"- restore rows: `{total_restore}`",
        f"- bridge eligible rows: `{total_eligible}`",
        f"- bridge missing rows: `{total_missing}`",
        f"- bridge ready: `{str(bridge_ready).lower()}`",
        f"- row detail: `{rows_csv}`",
        "",
        "## Suites",
        "",
        "| Suite | Figures | Restore rows | Bridge rows | Missing rows | Max restored tokens | Max measurement tokens | Restore actions | Physical actions |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary:
        lines.append(
            "| {suite} | {figures} | {restore} | {eligible} | {missing} | {max_restore} | {max_measurement} | `{actions}` | `{physical}` |".format(
                suite=row["suite_label"],
                figures=row["figures"],
                restore=row["restore_rows"],
                eligible=row["bridge_eligible_rows"],
                missing=row["bridge_missing_rows"],
                max_restore=row["max_restored_tokens"],
                max_measurement=row["max_measurement_tokens"],
                actions=row["restore_actions"],
                physical=row["physical_actions"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Rows counted here still retain their original SGLSim data plane. The strict-final",
            "gate remains closed until live retry rows report",
            "`aegis_replay_restore_data_plane=physical_tensor_copy` or the real SGLang",
            "recovery matrix provides equivalent end-to-end evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Map P0 SGLSim restore rows to physical recovery-path calibration rows."
    )
    parser.add_argument("--physical-aggregate", type=Path, default=DEFAULT_PHYSICAL_AGGREGATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--suite",
        action="append",
        nargs=3,
        metavar=("LABEL", "FIGURES", "SUITE_ROOT"),
        help="Override or append suite as label figures suite_root.",
    )
    args = parser.parse_args(argv)

    if args.suite:
        specs = [(label, figures, Path(root)) for label, figures, root in args.suite]
    else:
        specs = [
            (label, figures, RESULTS_ROOT / root)
            for label, figures, root in DEFAULT_SUITES
        ]

    rows = build_bridge_rows(specs=specs, physical_aggregate=args.physical_aggregate)
    summary = summarize_bridge(rows)
    output_dir = Path(args.output_dir)
    rows_csv = output_dir / "p0_physical_bridge_rows.csv"
    summary_csv = output_dir / "p0_physical_bridge_summary.csv"
    md_path = output_dir / "p0_physical_bridge.md"
    write_csv(rows_csv, rows)
    write_csv(summary_csv, summary)
    md_path.write_text(render_markdown(summary, rows_csv), encoding="utf-8")
    total_restore = sum(_int(row.get("restore_rows")) for row in summary)
    total_eligible = sum(_int(row.get("bridge_eligible_rows")) for row in summary)
    print(
        json.dumps(
            {
                "bridge_ready": total_restore > 0 and total_eligible == total_restore,
                "bridge_eligible_rows": total_eligible,
                "bridge_missing_rows": total_restore - total_eligible,
                "output_md": str(md_path),
                "rows_csv": str(rows_csv),
                "summary_csv": str(summary_csv),
                "restore_rows": total_restore,
            },
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
