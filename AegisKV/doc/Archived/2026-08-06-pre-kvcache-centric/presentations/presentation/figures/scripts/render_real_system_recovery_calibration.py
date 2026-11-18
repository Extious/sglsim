#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[5]
for import_path in (ROOT, ROOT / "python"):
    value = str(import_path)
    if value not in sys.path:
        sys.path.insert(0, value)

from sglsim.srt.simulation.recovery_model import (  # noqa: E402
    DEFAULT_GPU_SETUP_MS,
    DEFAULT_HOST_SETUP_MS,
    DEFAULT_RECOMPUTE_MS_PER_TOKEN,
    DEFAULT_RECOMPUTE_SETUP_MS,
    ENV_PHYSICAL_COPY_CSV,
    ENV_RECOVERY_MODEL,
    aegis_restore_overhead_ms,
)


DRAFT_ROOT = ROOT / "benchmark" / "AegisKV" / "paper_outline"
FIGURES = DRAFT_ROOT / "figures"
PHYSICAL_RECOVERY_CSV = (
    DRAFT_ROOT
    / "experiments"
    / "sglang_small"
    / "results"
    / "physical_recovery_path_aggregate"
    / "physical_recovery_path_aggregate.csv"
)
STRICT_CASE_CSV = (
    DRAFT_ROOT
    / "experiments"
    / "sglang_small"
    / "results"
    / "strict_final_recovery_matrix_audit"
    / "strict_final_recovery_case_summary.csv"
)
STRICT_PAIR_CSV = (
    DRAFT_ROOT
    / "experiments"
    / "sglang_small"
    / "results"
    / "strict_final_recovery_matrix_audit"
    / "strict_final_recovery_pair_rows.csv"
)
PHYSICAL_SMOKE_SUMMARY = (
    DRAFT_ROOT
    / "experiments"
    / "sglsim_large"
    / "results"
    / "agentic_workflows"
    / "sglsim-current-physical-recovery-smoke-20260704-codex2"
    / "current_physical_recovery_smoke_summary.json"
)
OUT_CSV = FIGURES / "data" / "measured" / "fig_real_system_recovery_calibration.csv"
OUT_PNG = DRAFT_ROOT / "Figures" / "fig_real_system_recovery_calibration.png"
OUT_BEAMER = OUT_PNG

FIELDNAMES = [
    "row_type",
    "path_label",
    "action",
    "tokens",
    "mib",
    "measured_copy_ms",
    "model_setup_ms",
    "physical_anchor_restore_ms",
    "sglsim_prediction_ms",
    "copy_component_error_pct",
    "repeat_count",
    "case_id",
    "status",
    "accepted_pairs",
    "improving_pairs",
    "visible_retry_latency_norm_mean",
    "visible_prefetch_io_completion_ratio_mean",
    "visible_latency_delta_pct_mean",
    "run_id",
    "request_rows",
    "replay_rows",
    "restore_rows",
    "physical_tensor_copy_rows",
    "nonfinal_restore_rows",
    "mean_restore_sleep_ms",
    "timing_sources",
    "data_planes",
    "final_ready",
    "evidence_tier",
    "source",
    "note",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    return int(round(_float(value, float(default))))


def _blank_row(**values: object) -> dict[str, str]:
    row = {field: "" for field in FIELDNAMES}
    for key, value in values.items():
        if key in row:
            row[key] = str(value)
    return row


def _with_physical_copy_env() -> tuple[str | None, str | None]:
    old_model = os.environ.get(ENV_RECOVERY_MODEL)
    old_csv = os.environ.get(ENV_PHYSICAL_COPY_CSV)
    os.environ[ENV_RECOVERY_MODEL] = "physical_copy"
    os.environ[ENV_PHYSICAL_COPY_CSV] = str(PHYSICAL_RECOVERY_CSV)
    return old_model, old_csv


def _restore_env(old_model: str | None, old_csv: str | None) -> None:
    if old_model is None:
        os.environ.pop(ENV_RECOVERY_MODEL, None)
    else:
        os.environ[ENV_RECOVERY_MODEL] = old_model
    if old_csv is None:
        os.environ.pop(ENV_PHYSICAL_COPY_CSV, None)
    else:
        os.environ[ENV_PHYSICAL_COPY_CSV] = old_csv


def _path_anchor_rows() -> list[dict[str, str]]:
    physical_rows = [
        row
        for row in _read_csv(PHYSICAL_RECOVERY_CSV)
        if row.get("action") in {"host_restore_h2d", "gpu_replica_restore_p2p"}
    ]
    old_model, old_csv = _with_physical_copy_env()
    try:
        rows: list[dict[str, str]] = []
        for row in physical_rows:
            action = str(row.get("action") or "")
            model_action = "gpu_replica" if action == "gpu_replica_restore_p2p" else "host_restore"
            path_label = "Peer GPU restore" if model_action == "gpu_replica" else "Host restore"
            tokens = _int(row.get("tokens"))
            measured_copy_ms = _float(row.get("p50_ms_mean"))
            setup_ms = (
                DEFAULT_GPU_SETUP_MS
                if model_action == "gpu_replica"
                else DEFAULT_HOST_SETUP_MS
            )
            prediction_ms = aegis_restore_overhead_ms(
                action=model_action,
                token_count=tokens,
                restored_tokens=tokens,
            )
            copy_component_ms = max(prediction_ms - setup_ms, 0.0)
            error_pct = (
                abs(copy_component_ms - measured_copy_ms)
                / max(measured_copy_ms, 1e-9)
                * 100.0
            )
            rows.append(
                _blank_row(
                    row_type="path_latency_anchor",
                    path_label=path_label,
                    action=action,
                    tokens=tokens,
                    mib=f"{_float(row.get('mib')):.3f}",
                    measured_copy_ms=f"{measured_copy_ms:.6f}",
                    model_setup_ms=f"{setup_ms:.6f}",
                    physical_anchor_restore_ms=f"{measured_copy_ms + setup_ms:.6f}",
                    sglsim_prediction_ms=f"{prediction_ms:.6f}",
                    copy_component_error_pct=f"{error_pct:.9f}",
                    repeat_count=_int(row.get("repeat_count")),
                    evidence_tier=row.get("evidence_tier", ""),
                    source=row.get("source", ""),
                    note=(
                        "Physical CUDA copy microbenchmark used as the Host/GPUReplica "
                        "restore timing anchor; this is a latency anchor, not a full "
                        "end-to-end policy run."
                    ),
                )
            )
        return rows
    finally:
        _restore_env(old_model, old_csv)


def _recompute_rows(tokens_list: list[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tokens in tokens_list:
        prediction_ms = DEFAULT_RECOMPUTE_SETUP_MS + tokens * DEFAULT_RECOMPUTE_MS_PER_TOKEN
        rows.append(
            _blank_row(
                row_type="sglsim_recompute_prediction",
                path_label="Recompute",
                action="recompute",
                tokens=tokens,
                measured_copy_ms="",
                model_setup_ms=f"{DEFAULT_RECOMPUTE_SETUP_MS:.6f}",
                physical_anchor_restore_ms="",
                sglsim_prediction_ms=f"{prediction_ms:.6f}",
                evidence_tier="sglsim-recovery-model-prediction",
                source="sglsim_default_recompute_coefficients",
                note=(
                    "Recompute uses the current SGLSim coefficient model; no real "
                    "recompute microbenchmark is claimed in this row."
                ),
            )
        )
    return rows


def _strict_matrix_rows() -> list[dict[str, str]]:
    pair_rows = [row for row in _read_csv(STRICT_PAIR_CSV) if row.get("accepted") == "true"]
    ratios_by_case: dict[str, list[float]] = {}
    deltas_by_case: dict[str, list[float]] = {}
    for row in pair_rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        ratios_by_case.setdefault(case_id, []).append(
            _float(row.get("visible_prefetch_io_completion_ratio"))
        )
        deltas_by_case.setdefault(case_id, []).append(
            _float(row.get("visible_latency_delta_pct_vs_invisible"))
        )

    rows: list[dict[str, str]] = []
    for row in _read_csv(STRICT_CASE_CSV):
        case_id = str(row.get("case_id") or "")
        ratios = ratios_by_case.get(case_id, [])
        deltas = deltas_by_case.get(case_id, [])
        ratio_mean = sum(ratios) / len(ratios) if ratios else 0.0
        delta_mean = sum(deltas) / len(deltas) if deltas else 0.0
        rows.append(
            _blank_row(
                row_type="strict_recovery_matrix_case",
                path_label="Strict recovery matrix",
                case_id=case_id,
                status=row.get("status", ""),
                accepted_pairs=_int(row.get("accepted_pairs")),
                improving_pairs=_int(row.get("improving_pairs")),
                visible_retry_latency_norm_mean=(
                    f"{_float(row.get('visible_retry_latency_norm_mean')):.9f}"
                ),
                visible_prefetch_io_completion_ratio_mean=f"{ratio_mean:.9f}",
                visible_latency_delta_pct_mean=f"{delta_mean:.6f}",
                evidence_tier="sglang-small-strict-final-recovery-matrix-audit",
                source="audit_strict_final_recovery_matrix",
                note=row.get("note", ""),
            )
        )
    return rows


def _smoke_row() -> dict[str, str]:
    summary = _read_json(PHYSICAL_SMOKE_SUMMARY)
    totals = summary.get("audit_totals") if isinstance(summary.get("audit_totals"), dict) else {}
    agentic = (
        summary.get("agentic_summary")
        if isinstance(summary.get("agentic_summary"), dict)
        else {}
    )
    return _blank_row(
        row_type="online_physical_copy_smoke",
        path_label="Online SGLSim physical-copy timing smoke",
        run_id=summary.get("run_id", ""),
        request_rows=_int(totals.get("request_rows")),
        replay_rows=_int(totals.get("replay_rows")),
        restore_rows=_int(totals.get("restore_rows")),
        physical_tensor_copy_rows=_int(totals.get("physical_tensor_copy_rows")),
        nonfinal_restore_rows=_int(totals.get("nonfinal_restore_rows")),
        mean_restore_sleep_ms=f"{_float(agentic.get('mean_restore_sleep_ms')):.6f}",
        timing_sources=json.dumps(totals.get("timing_sources", {}), sort_keys=True),
        data_planes=json.dumps(totals.get("data_planes", {}), sort_keys=True),
        final_ready=str(bool(totals.get("final_ready"))).lower(),
        evidence_tier=summary.get("evidence_tier", ""),
        source=str(PHYSICAL_SMOKE_SUMMARY),
        note=(
            "SGLSim online serving uses the physical_recovery_path_csv timing source; "
            "current data plane remains metadata_materialization, so this is a "
            "non-final calibration smoke."
        ),
    )


def build_rows() -> list[dict[str, str]]:
    path_rows = _path_anchor_rows()
    tokens = sorted({_int(row.get("tokens")) for row in path_rows if row.get("tokens")})
    rows = []
    rows.extend(path_rows)
    rows.extend(_recompute_rows(tokens or [512, 1024, 2048, 4096, 8192]))
    rows.extend(_strict_matrix_rows())
    rows.append(_smoke_row())
    return rows


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _rows_for_path(rows: list[dict[str, str]], label: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("row_type") in {"path_latency_anchor", "sglsim_recompute_prediction"}
        and row.get("path_label") == label
    ]


def render(rows: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), gridspec_kw={"width_ratios": [1.22, 1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    colors = {
        "Host restore": "#2F6C8E",
        "Peer GPU restore": "#35A76B",
        "Recompute": "#B25D4C",
    }
    markers = {"Host restore": "o", "Peer GPU restore": "s", "Recompute": "^"}
    for label in ("Host restore", "Peer GPU restore", "Recompute"):
        series = sorted(_rows_for_path(rows, label), key=lambda row: _float(row.get("tokens")))
        xs = [_float(row.get("tokens")) for row in series]
        ys = [_float(row.get("sglsim_prediction_ms")) for row in series]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker=markers[label],
            linewidth=2.2,
            markersize=5.5,
            color=colors[label],
            label=label,
        )
        if label != "Recompute":
            anchor_ys = [_float(row.get("physical_anchor_restore_ms")) for row in series]
            ax.scatter(
                xs,
                anchor_ys,
                s=54,
                facecolors="white",
                edgecolors=colors[label],
                linewidths=1.4,
                zorder=3,
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.get_xaxis().set_major_formatter(lambda value, _: f"{int(value)}")
    ax.set_xlabel("restored / recomputed KV tokens")
    ax.set_ylabel("restore-path latency (ms, log scale)")
    ax.set_title("Physical copy anchors -> SGLSim restore model", fontsize=12.5, weight="bold")
    ax.grid(True, which="both", axis="y", color="#D6DBDF", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    ax2 = axes[1]
    strict_rows = [
        row for row in rows if row.get("row_type") == "strict_recovery_matrix_case"
    ]
    labels = [
        "best-effort\nvisible restore",
        "wait-complete\nvisible restore",
    ]
    by_case = {row.get("case_id"): row for row in strict_rows}
    ordered_cases = ["same_write_best_effort", "same_write_wait_complete"]
    values = [
        _float(by_case.get(case, {}).get("visible_retry_latency_norm_mean"))
        for case in ordered_cases
    ]
    bar_colors = ["#35A76B", "#D89C24"]
    bars = ax2.bar(labels, values, color=bar_colors, width=0.58)
    ax2.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    ax2.set_ylim(0.0, max(values + [1.0]) * 1.32)
    ax2.set_ylabel("retry latency normalized to invisible control")
    ax2.set_title("Strict recovery matrix + online smoke boundary", fontsize=12.5, weight="bold")
    ax2.grid(True, axis="y", color="#E2E6EA", linewidth=0.8)
    for bar, value, case in zip(bars, values, ordered_cases):
        row = by_case.get(case, {})
        accepted = _int(row.get("accepted_pairs"))
        improving = _int(row.get("improving_pairs"))
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.2f}x\n{accepted}/3 accepted\n{improving}/3 improving",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#111111",
        )
    smoke = next(
        (row for row in rows if row.get("row_type") == "online_physical_copy_smoke"),
        {},
    )
    smoke_lines = [
        "Online SGLSim smoke:",
        f"restore rows: {_int(smoke.get('restore_rows'))}",
        f"timing: {smoke.get('timing_sources')}",
        f"data plane: {smoke.get('data_planes')}",
        f"final-ready: {smoke.get('final_ready')}",
    ]
    ax2.text(
        0.02,
        0.04,
        "\n".join(textwrap.wrap(smoke_lines[0], width=42))
        + "\n"
        + "\n".join(smoke_lines[1:]),
        transform=ax2.transAxes,
        fontsize=8.7,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F6F7F8", "edgecolor": "#C8D0D8"},
    )

    fig.suptitle(
        "Real-system recovery calibration: physical anchors and current SGLSim boundary",
        fontsize=15,
        weight="bold",
        x=0.50,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    rows = build_rows()
    _write_csv(rows, OUT_CSV)
    render(rows, OUT_PNG)
    OUT_BEAMER.parent.mkdir(parents=True, exist_ok=True)
    OUT_BEAMER.write_bytes(OUT_PNG.read_bytes())
    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv": str(OUT_CSV),
                "png": str(OUT_PNG),
                "beamer_png": str(OUT_BEAMER),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
