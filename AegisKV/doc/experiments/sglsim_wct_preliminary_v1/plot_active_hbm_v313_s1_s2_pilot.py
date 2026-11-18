#!/usr/bin/env python3
"""Paper-style, audit-bound S1/S2 previews from frozen v3.1.3 pilots."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader

import run_preliminary as core


ROOT = Path(__file__).resolve().parent
S1 = ROOT / "results/active_hbm_v313_s1_cold_steady_pilot"
S2 = ROOT / "results/active_hbm_v313_batched_s2_pilot"
RESULT_NAME = "active_hbm_v313_paper_preview_s1_s2_pilot"
NAMES = [
    "Native Mooncake", "Static", "DejaVu", "KevlarFlow",
    "GhostServe", "AegisKV",
]
COLORS = {
    "Native Mooncake": "#222222",
    "Static": "#8C564B",
    "DejaVu": "#1F77B4",
    "KevlarFlow": "#9467BD",
    "GhostServe": "#2CA02C",
    "AegisKV": "#D62728",
}
MARKERS = {
    "Native Mooncake": "o", "Static": "s", "DejaVu": "^",
    "KevlarFlow": "D", "GhostServe": "v", "AegisKV": "P",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_s1(stage: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = read_csv(S1 / "metrics.csv")
    cold = {row["display_policy"]: row for row in metrics if row["regime"] == "cold_start_inclusive"}
    if list(name for name in NAMES if name in cold) != NAMES:
        raise AssertionError("S1 exact six cold policies missing")
    source_rows = []
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for name in NAMES:
        raw = S1 / "raw" / Path(next(
            cell["raw_npz"] for cell in json.loads((S1 / "summary.json").read_text())["cells"]
            if cell["regime"] == "cold_start_inclusive" and cell["display_policy"] == name
        )).name
        with np.load(raw, allow_pickle=False) as data:
            workflow_ids = data["workflow_ids"].astype(str)
            for domain, key in (
                ("Worker/GPU", "worker_fault_wct_ms"),
                ("Source node", "node_fault_wct_ms"),
            ):
                matrix = np.asarray(data[key], dtype=float)
                arrays[(name, domain)] = matrix.reshape(-1) / 1000.0
                for probe_index in range(matrix.shape[0]):
                    for workflow_index, workflow_id in enumerate(workflow_ids):
                        source_rows.append({
                            "display_policy": name,
                            "failure_domain": domain,
                            "seed": int(cold[name]["seed"]),
                            "probe_index": probe_index,
                            "workflow_id": workflow_id,
                            "wct_ms": float(matrix[probe_index, workflow_index]),
                            "raw_npz_sha256": cold[name]["raw_npz_sha256"],
                        })
    write_csv(stage / "s1_cold_fault_ecdf_source.csv", source_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True)
    for ax, domain in zip(axes, ("Worker/GPU", "Source node")):
        for name in NAMES:
            x, y = ecdf(arrays[(name, domain)])
            ax.plot(x, y, label=name, color=COLORS[name])
        ax.set_title(domain)
        ax.set_xlabel("Workflow completion time (s)")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.set_ylim(0.0, 1.01)
    axes[0].set_ylabel("Empirical CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, 1.015), columnspacing=1.0, handlelength=1.8,
    )
    fig.suptitle(
        "Cold-start pilot: 1 seed, 72 workflows x 4 probes per failure domain",
        fontsize=8.0, y=0.925,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.79, wspace=0.18)
    for suffix in ("pdf", "png"):
        fig.savefig(
            stage / f"s1_cold_fault_ecdf.{suffix}", dpi=300,
            bbox_inches="tight", pad_inches=0.03,
        )
    plt.close(fig)
    return source_rows, {
        "seed_count": 1,
        "workflow_count": 72,
        "probes_per_domain": 4,
        "samples_per_policy_domain": 288,
    }


def plot_s2(stage: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = read_csv(S2 / "window_samples.csv")
    typed = []
    for row in samples:
        typed.append({
            **row,
            "rate": float(row["rate_per_1000_gpu_shard_hour"]),
            "wct": float(row["wct_mean_ms"]),
            "candidate": int(row["candidate_event_count"]),
            "accepted": int(row["accepted_event_count"]),
            "noop": int(row["noop_event_count"]),
        })
    rates = sorted({row["rate"] for row in typed})
    if rates != [0.0, 5.952, 20.833]:
        raise AssertionError("S2 exact rates changed")
    summary_rows = []
    per_policy: dict[tuple[str, float], tuple[float, float]] = {}
    event_counts = {}
    for policy in ("Native Mooncake", "AegisKV"):
        for rate in rates:
            rows = [row for row in typed if row["policy"] == policy and row["rate"] == rate]
            values = [row["wct"] for row in rows]
            mean = statistics.fmean(values)
            se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
            half = 1.96 * se
            per_policy[(policy, rate)] = (mean, half)
            event_counts[rate] = (
                sum(row["candidate"] for row in rows),
                sum(row["accepted"] for row in rows),
                sum(row["noop"] for row in rows),
            )
            summary_rows.append({
                "series": policy,
                "rate_per_1000_gpu_shard_hour": rate,
                "window_count": len(rows),
                "wct_mean_ms": mean,
                "ci95_half_width_ms": half,
                "paired_gain_mean_ms": "",
                "paired_gain_ci95_half_width_ms": "",
                "candidate_event_count": event_counts[rate][0],
                "accepted_event_count": event_counts[rate][1],
                "noop_event_count": event_counts[rate][2],
                "active_requests_per_gpu": 16,
                "unique_workload_seed_count": 1,
            })
    paired = {}
    for rate in rates:
        native = {
            row["window_id"]: row["wct"] for row in typed
            if row["policy"] == "Native Mooncake" and row["rate"] == rate
        }
        aegis = {
            row["window_id"]: row["wct"] for row in typed
            if row["policy"] == "AegisKV" and row["rate"] == rate
        }
        if native.keys() != aegis.keys():
            raise AssertionError("S2 paired window denominator changed")
        differences = [native[key] - aegis[key] for key in sorted(native)]
        mean = statistics.fmean(differences)
        se = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0.0
        half = 1.96 * se
        paired[rate] = (mean, half)
        summary_rows.append({
            "series": "Paired gain: Native - AegisKV",
            "rate_per_1000_gpu_shard_hour": rate,
            "window_count": len(differences),
            "wct_mean_ms": "",
            "ci95_half_width_ms": "",
            "paired_gain_mean_ms": mean,
            "paired_gain_ci95_half_width_ms": half,
            "candidate_event_count": event_counts[rate][0],
            "accepted_event_count": event_counts[rate][1],
            "noop_event_count": event_counts[rate][2],
            "active_requests_per_gpu": 16,
            "unique_workload_seed_count": 1,
        })
    write_csv(stage / "s2_fixed_c16_rate_source.csv", summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    for policy in ("Native Mooncake", "AegisKV"):
        means = np.asarray([per_policy[(policy, rate)][0] / 1000.0 for rate in rates])
        half = np.asarray([per_policy[(policy, rate)][1] / 1000.0 for rate in rates])
        axes[0].errorbar(
            rates, means, yerr=half, label=policy, color=COLORS[policy],
            marker=MARKERS[policy], markersize=4.5, capsize=2.5,
        )
    axes[0].set_title("All-window WCT")
    axes[0].set_ylabel("Mean workflow completion time (s)")
    axes[0].legend(frameon=False, loc="best")
    gains = np.asarray([paired[rate][0] for rate in rates])
    gain_half = np.asarray([paired[rate][1] for rate in rates])
    axes[1].errorbar(
        rates, gains, yerr=gain_half, color=COLORS["AegisKV"],
        marker="o", markersize=4.5, capsize=2.5,
    )
    axes[1].axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[1].set_title("Paired WCT difference")
    axes[1].set_ylabel("Native - AegisKV (ms)")
    tick_labels = [
        f"{rate:g}\nE={event_counts[rate][0]}"
        for rate in rates
    ]
    for ax in axes:
        ax.set_xticks(rates, tick_labels)
        ax.set_xlabel("Failure rate per 1K GPU-shard-hour\nE = realized candidate events")
        ax.grid(True, alpha=0.22, linewidth=0.6)
    fig.suptitle(
        "Fixed-load pilot: c=16, 2,800 independent 90 s windows/rate, 1 workload seed",
        fontsize=8.0, y=0.985,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.27, top=0.82, wspace=0.26)
    for suffix in ("pdf", "png"):
        fig.savefig(
            stage / f"s2_fixed_c16_rate_paired.{suffix}", dpi=300,
            bbox_inches="tight", pad_inches=0.03,
        )
    plt.close(fig)
    return summary_rows, {
        "window_count_per_rate_policy": 2800,
        "active_requests_per_gpu": 16,
        "unique_workload_seed_count": 1,
        "event_counts": {format(rate, ".3f"): event_counts[rate] for rate in rates},
    }


def main() -> int:
    for result in (S1, S2):
        if not result.exists():
            raise FileNotFoundError(result)
        check = os.system(f"cd '{result}' && sha256sum -c SHA256SUMS >/dev/null")
        if check != 0:
            raise RuntimeError(f"input checksum failed: {result}")
    target = ROOT / "results" / RESULT_NAME
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    stage = target.parent / f".{RESULT_NAME}.stage-{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=False)
    style()
    _s1_rows, s1_stats = plot_s1(stage)
    _s2_rows, s2_stats = plot_s2(stage)
    checks = {
        "s1_exact_six_architectures": s1_stats["samples_per_policy_domain"] == 288,
        "s1_cold_only_and_four_probes": s1_stats["probes_per_domain"] == 4,
        "s2_fixed_c16_not_concurrency_curve": s2_stats["active_requests_per_gpu"] == 16,
        "s2_all_2800_windows_per_rate_policy": s2_stats["window_count_per_rate_policy"] == 2800,
        "s2_single_workload_seed_explicit": s2_stats["unique_workload_seed_count"] == 1,
        "png_width_at_least_1800": all(
            plt.imread(stage / filename).shape[1] >= 1800
            for filename in ("s1_cold_fault_ecdf.png", "s2_fixed_c16_rate_paired.png")
        ),
        "vector_pdfs_single_page": all(
            len(PdfReader(stage / filename).pages) == 1
            for filename in ("s1_cold_fault_ecdf.pdf", "s2_fixed_c16_rate_paired.pdf")
        ),
        "not_paper_ready": True,
    }
    manifest = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "paper_ready": False,
        "input_results": {
            "s1": str(S1), "s1_audit_sha256": core.sha256(S1 / "audit.json"),
            "s1_sha256sums_sha256": core.sha256(S1 / "SHA256SUMS"),
            "s2": str(S2), "s2_audit_sha256": core.sha256(S2 / "audit.json"),
            "s2_sha256sums_sha256": core.sha256(S2 / "SHA256SUMS"),
        },
        "figure_semantics": {
            "S1": (
                "Cold-start one-seed fault-conditioned ECDF; 72 workflows x 4 "
                "common probes/domain. The separate post-warmup c16 cohort is not "
                "shown because its queue is nonstationary."
            ),
            "S2": (
                "Fixed c16 one-workload-seed rate pilot. Error bars are normal 95% "
                "window-sample intervals over 2800 independent fault windows; they "
                "do not include workload-seed uncertainty."
            ),
        },
        "source_stats": {"S1": s1_stats, "S2": s2_stats},
        "runner_sha256": core.sha256(Path(__file__)),
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    audit = {
        "schema_version": 1,
        "status": manifest["status"],
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "paper_ready": False,
        "manifest_sha256": core.sha256(stage / "manifest.json"),
    }
    (stage / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if audit["status"] != "PASS":
        raise RuntimeError(audit["failed_checks"])
    artifacts = {
        path.name: core.sha256(path)
        for path in sorted(stage.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (stage / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(artifacts.items()))
    )
    os.replace(stage, target)
    print(target)
    print(f"audit_sha256={core.sha256(target / 'audit.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
