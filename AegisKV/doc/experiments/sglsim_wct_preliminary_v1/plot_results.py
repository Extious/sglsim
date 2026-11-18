#!/usr/bin/env python3
"""Render S1/S2/S4 directly from audited preliminary WCT artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
COLORS = {
    "Native Mooncake": "#4D4D4D",
    "Static FCFS": "#999999",
    "DejaVu": "#E69F00",
    "KevlarFlow": "#CC79A7",
    "GhostServe": "#009E73",
    "AegisKV": "#0072B2",
}
LINESTYLES = {
    "Native Mooncake": "-",
    "Static FCFS": ":",
    "DejaVu": "--",
    "KevlarFlow": "-.",
    "GhostServe": (0, (3, 1, 1, 1)),
    "AegisKV": "-",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results" / "preliminary")
    parser.add_argument("--s2-dir", type=Path, default=ROOT / "results" / "s2_fixed_arrival_v2")
    args = parser.parse_args()
    audit = json.loads((args.input_dir / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("workflow_completion_time_result") is not True:
        raise ValueError("plots require a passed WCT audit")
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    aggregate = load_rows(args.input_dir / "aggregate_ci.csv")
    s2_audit = json.loads((args.s2_dir / "audit.json").read_text(encoding="utf-8"))
    if (
        s2_audit.get("status") != "PASS"
        or s2_audit.get("workflow_completion_time_result") is not True
        or s2_audit.get("selection") != "S2_FIXED_ARRIVAL_V2"
    ):
        raise ValueError("S2 plots require the passed fixed-arrival-v2 WCT audit")
    s2_aggregate = load_rows(args.s2_dir / "aggregate_ci.csv")
    s2_summary = json.loads((args.s2_dir / "summary.json").read_text(encoding="utf-8"))
    figures = args.input_dir / "figures"

    # S1 keeps the original target question--the architecture-level affected
    # workflow WCT CDF--and adds a quantile-shift panel so nearly coincident
    # curves are not mistaken for a plotting failure or a material win.
    fig, axes = plt.subplots(
        1, 2, figsize=(10.4, 4.35),
        gridspec_kw={"width_ratios": [1.42, 0.92]},
    )
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for policy in COLORS:
        cells = [c for c in summary["cells"] if c["policy"] == policy and c["active_requests_per_gpu"] == 4]
        samples = []
        for cell in cells:
            with np.load(cell["raw_npz"], allow_pickle=False) as raw:
                base = raw["base_wct_ms"]
                fault = raw["fault_wct_ms"]
                affected = fault > base[None, :] + 1e-9
                samples.extend(fault[affected].tolist())
        values = np.sort(np.asarray(samples) / 1000.0)
        y = np.arange(1, len(values) + 1) / len(values)
        series[policy] = (values, y)
        label = "Native Mooncake (modeled)" if policy == "Native Mooncake" else policy
        axes[0].plot(
            values,
            y,
            label=label,
            color=COLORS[policy],
            linestyle=LINESTYLES[policy],
            lw=2.5 if policy == "AegisKV" else 1.45,
            alpha=1.0 if policy in {"AegisKV", "Native Mooncake"} else 0.88,
            zorder=4 if policy == "AegisKV" else 2,
        )
    axes[0].set_xlabel("Affected-workflow WCT (s)")
    axes[0].set_ylabel("CDF")
    axes[0].set_ylim(0.0, 1.01)
    axes[0].set_title("Executed one-fault WCT distribution")
    axes[0].grid(alpha=0.18, linewidth=0.8)

    quantiles = np.asarray([0.50, 0.95, 0.99])
    native_q = np.quantile(series["Native Mooncake"][0], quantiles)
    compare_policies = ["DejaVu", "KevlarFlow", "GhostServe", "AegisKV"]
    offsets = np.linspace(-0.24, 0.24, len(compare_policies))
    for policy, offset in zip(compare_policies, offsets):
        delta_ms = (
            np.quantile(series[policy][0], quantiles) - native_q
        ) * 1000.0
        axes[1].plot(
            delta_ms, np.arange(len(quantiles)) + offset,
            marker="o", color=COLORS[policy], linestyle=LINESTYLES[policy],
            linewidth=1.7, markersize=5.5, label=policy,
        )
    axes[1].axvline(0.0, color="black", lw=1, ls="--")
    axes[1].set_yticks(np.arange(len(quantiles)), ["p50", "p95", "p99"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Quantile shift vs modeled Native (ms)")
    axes[1].set_title("Negative is better")
    axes[1].grid(axis="x", alpha=0.18, linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=3, fontsize=8.2,
    )
    fig.suptitle("S1. Large-scale architecture WCT CDF - observed", fontsize=14, y=0.995)
    fig.subplots_adjust(top=0.76, bottom=0.18, wspace=0.27)
    fig.text(
        0.5, 0.025,
        "The target question is preserved; the right panel exposes the observed near-overlap instead of inventing separation",
        ha="center", fontsize=8, color="#555555",
    )
    save(fig, figures / "s1_wct_cdf_preliminary")

    # S2: paired AegisKV / modeled Native Mooncake across concurrency and rate.
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    rates = (0.0, 5.952, 20.833)
    rate_colors = ("#0072B2", "#E69F00", "#009E73")
    for rate, marker, color in zip(rates, ("o", "s", "^"), rate_colors):
        rows = [
            row for row in s2_aggregate
            if row["policy"] == "AegisKV"
            and abs(float(row["failure_rate_per_1000_gpu_shard_hour"]) - rate) < 1e-9
        ]
        rows.sort(key=lambda row: int(row["active_requests_per_gpu"]))
        x = [int(row["active_requests_per_gpu"]) for row in rows]
        ratio = np.asarray([float(row["normalized_to_native"]) for row in rows])
        ratio_low = np.asarray([float(row["normalized_ci95_low"]) for row in rows])
        ratio_high = np.asarray([float(row["normalized_ci95_high"]) for row in rows])
        gain = (1.0 - ratio) * 100.0
        gain_low = (1.0 - ratio_high) * 100.0
        gain_high = (1.0 - ratio_low) * 100.0
        ax.plot(
            x, gain, marker=marker, color=color, linewidth=2.0,
            markersize=6.5, label=f"{rate:g} failures / 1K GPU-shard-h",
        )
        ax.fill_between(x, gain_low, gain_high, color=color, alpha=0.13)
        if abs(rate - 5.952) < 1e-9:
            for xpos, ypos in zip(x[-2:], gain[-2:]):
                ax.annotate(
                    f"{ypos:.1f}%", (xpos, ypos), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=color, fontweight="bold",
                )
    ax.axhline(0.0, color="black", lw=1, ls="--")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xlabel("Target active requests per GPU")
    ax.set_ylabel("Mean WCT reduction vs modeled Native (%)")
    ax.set_title("S2. AegisKV benefit emerges only under queue pressure")
    ax.grid(alpha=0.20, linewidth=0.8)
    ax.legend(fontsize=8.2, loc="upper left")
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.985, 0.025, "Positive is better; paired 3-seed 95% CI; fixed 90 s arrivals",
        ha="right", fontsize=7.5, color="#555555",
    )
    save(fig, figures / "s2_concurrency_failure_preliminary")

    # S4: center-cell ablation, Native fixed to one by paired normalization.
    labels = ["Native Mooncake", "AegisKV", "AegisKV no Algorithm 1", "AegisKV no Algorithm 2", "AegisKV no Algorithm 3"]
    rows = {
        row["policy"]: row for row in aggregate
        if int(row["active_requests_per_gpu"]) == 4
        and abs(float(row["failure_rate_per_1000_gpu_shard_hour"]) - 5.952) < 1e-9
        and row["policy"] in labels
    }
    ratios = np.asarray([float(rows[label]["normalized_to_native"]) for label in labels])
    ratio_low = np.asarray([float(rows[label]["normalized_ci95_low"]) for label in labels])
    ratio_high = np.asarray([float(rows[label]["normalized_ci95_high"]) for label in labels])
    gains = (1.0 - ratios) * 1e6
    gain_low = (1.0 - ratio_high) * 1e6
    gain_high = (1.0 - ratio_low) * 1e6
    display_labels = ["Native", "Full AegisKV", "without Alg. 1", "without Alg. 2", "without Alg. 3"]
    colors = ["#4D4D4D", "#0072B2", "#56B4E9", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(display_labels))
    ax.errorbar(
        gains, y,
        xerr=[gains - gain_low, gain_high - gains],
        fmt="none", ecolor="#4B5563", elinewidth=1.2, capsize=4, zorder=2,
    )
    ax.scatter(gains, y, c=colors, s=78, edgecolors="white", linewidths=0.8, zorder=3)
    ax.axvline(0.0, color="black", lw=1, ls="--")
    ax.set_yticks(y, display_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean WCT reduction vs modeled Native (ppm)")
    ax.set_title("S4. Center-cell ablation effects are negligible in magnitude")
    ax.grid(axis="x", alpha=0.20, linewidth=0.8)
    for xpos, ypos in zip(gains, y):
        ax.annotate(
            f"{xpos:+.1f}", (xpos, ypos), xytext=(7 if xpos >= 0 else -7, 0),
            textcoords="offset points", ha="left" if xpos >= 0 else "right",
            va="center", fontsize=8,
        )
    fig.subplots_adjust(bottom=0.18, left=0.19)
    fig.text(
        0.985, 0.025, "Positive is better; paired 3-seed 95% CI",
        ha="right", fontsize=7.5, color="#555555",
    )
    save(fig, figures / "s4_ablation_preliminary")

    # S5: the fixed-arrival run's controller state, with no inferred occupancy.
    # Error bars are the observed range across the three paired seeds.
    aegis_cells = [cell for cell in s2_summary["cells"] if cell["policy"] == "AegisKV"]
    concurrency_values = sorted({int(cell["active_requests_per_gpu"]) for cell in aegis_cells})
    realized_mean = []
    realized_low = []
    realized_high = []
    alpha_mean = []
    alpha_low = []
    alpha_high = []
    residency_fractions: dict[str, list[float]] = {
        "HBM_AND_DRAM": [], "DRAM_ONLY": [], "BYPASS": [],
    }
    for concurrency in concurrency_values:
        cells = [
            cell for cell in aegis_cells
            if int(cell["active_requests_per_gpu"]) == concurrency
        ]
        realized = np.asarray([
            float(cell["workload"]["realized_active_requests_per_gpu_time_mean"])
            for cell in cells
        ])
        alpha = np.asarray([
            float(cell["mechanism"]["alpha_final_mean"]) for cell in cells
        ])
        realized_mean.append(float(np.mean(realized)))
        realized_low.append(float(np.min(realized)))
        realized_high.append(float(np.max(realized)))
        alpha_mean.append(float(np.mean(alpha)))
        alpha_low.append(float(np.min(alpha)))
        alpha_high.append(float(np.max(alpha)))
        counts = {
            name: sum(int(cell["mechanism"]["residency_event_counts"].get(name, 0)) for cell in cells)
            for name in residency_fractions
        }
        total = max(sum(counts.values()), 1)
        for name in residency_fractions:
            residency_fractions[name].append(counts[name] / total)

    fig, (alpha_ax, residency_ax) = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.10},
    )
    alpha_ax.errorbar(
        realized_mean, alpha_mean,
        xerr=[np.asarray(realized_mean) - realized_low, np.asarray(realized_high) - realized_mean],
        yerr=[np.asarray(alpha_mean) - alpha_low, np.asarray(alpha_high) - alpha_mean],
        marker="o", color="#0072B2", capsize=4, lw=2.2,
        label="Final alpha (mean and seed range)",
    )
    alpha_ax.axhline(0.0, color="black", lw=0.8, ls=":")
    alpha_ax.set_ylim(-0.012, 0.33)
    alpha_ax.set_ylabel("Final peer-HBM alpha")
    alpha_ax.set_title("AIMD quota response")
    alpha_ax.grid(alpha=0.18, linewidth=0.8)
    alpha_ax.legend(fontsize=8, loc="upper right")
    for xpos, ypos in zip(realized_mean, alpha_mean):
        alpha_ax.annotate(
            f"{ypos:.3f}", (xpos, ypos), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=7.5,
            color="#0072B2",
        )
    quota_ax = alpha_ax.secondary_yaxis(
        "right", functions=(lambda alpha: 40.0 * alpha, lambda gib: gib / 40.0)
    )
    quota_ax.set_ylabel("Peer-HBM quota = 40 x alpha (GiB)")

    bottom = np.zeros(len(realized_mean))
    residency_colors = {
        "HBM_AND_DRAM": "#0072B2", "DRAM_ONLY": "#CC79A7", "BYPASS": "#BDBDBD",
    }
    for name in ("HBM_AND_DRAM", "DRAM_ONLY", "BYPASS"):
        values = np.asarray(residency_fractions[name])
        residency_ax.bar(
            realized_mean, values * 100.0, bottom=bottom * 100.0, width=0.62,
            color=residency_colors[name], edgecolor="white", linewidth=0.5,
            label="HBM + DRAM" if name == "HBM_AND_DRAM" else name.replace("_", " "),
        )
        for xpos, value, base in zip(realized_mean, values, bottom):
            if value >= 0.08:
                residency_ax.text(
                    xpos, (base + value / 2.0) * 100.0, f"{value * 100:.0f}%",
                    ha="center", va="center", fontsize=7.2,
                    color="white" if name != "BYPASS" else "#333333",
                )
        bottom += values
    residency_ax.set_ylim(0.0, 100.0)
    residency_ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    residency_ax.set_ylabel("Admission outcome events")
    residency_ax.set_xlabel("Realized active requests per GPU (time mean)")
    residency_ax.set_xticks(realized_mean, [f"{value:.2f}" for value in realized_mean])
    residency_ax.grid(axis="y", alpha=0.16, linewidth=0.8)
    residency_ax.legend(ncol=3, fontsize=7.5, loc="upper center")
    fig.suptitle("S5. AIMD load response - observed endpoints", fontsize=14, y=0.995)
    fig.subplots_adjust(top=0.90, bottom=0.12)
    fig.text(
        0.5, 0.025,
        "The old target requested a time series; only 30/60/90 s controls and final alpha were persisted. Bars are events, not occupancy.",
        ha="center", fontsize=7.5, color="#555555",
    )
    save(fig, figures / "s5_alpha_load_preliminary")

    # Diagnostic for the mismatch between the old illustrative target shape
    # and the observed preliminary data.  Every value is derived from the
    # frozen protocol or the passed fixed-arrival S2 lineage.
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    model = protocol["models"]["qwen3_8b"]
    context = int(protocol["workload"]["center_context_tokens"])
    kv_bytes = int(model["kv_bytes_per_token"])
    replacement_ms = float(protocol["failure_model"]["replacement_delay_ms"])
    bandwidths = protocol["hardware"]["bandwidth_gb_s"]
    variable_recovery_ms = {
        "same-node\nPeer-HBM": 0.08 + context * kv_bytes / (float(bandwidths["peer_hbm_same_node"]) * 1e6),
        "local\nDRAM": 0.15 + context * kv_bytes / (float(bandwidths["dram_local"]) * 1e6),
        "cross-node\nPeer/DRAM": 0.15 + context * kv_bytes / (float(bandwidths["dram_remote"]) * 1e6),
        "full\nrecompute": float(model["prefill_setup_ms"]) + context * float(model["prefill_ms_per_token"]),
    }
    names = list(variable_recovery_ms)
    values = np.asarray(list(variable_recovery_ms.values()))
    shares = values / (replacement_ms + values) * 100.0

    c16_cells = [
        cell for cell in s2_summary["cells"]
        if cell["policy"] == "AegisKV" and int(cell["active_requests_per_gpu"]) == 16
    ]
    exposure_hours = float(np.mean([cell["simulated_gpu_shard_hours"] for cell in c16_cells]))
    expected_faults = np.asarray(rates) * exposure_hours / 1000.0
    c16_rows = [
        row for row in s2_aggregate
        if row["policy"] == "AegisKV" and int(row["active_requests_per_gpu"]) == 16
    ]
    c16_rows.sort(key=lambda row: float(row["failure_rate_per_1000_gpu_shard_hour"]))
    c16_gain = np.asarray([(1.0 - float(row["normalized_to_native"])) * 100.0 for row in c16_rows])

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.75))
    bars = axes[0].bar(names, shares, color=["#0072B2", "#E69F00", "#CC79A7", "#999999"])
    axes[0].set_ylabel("Variable recovery share of total delay (%)")
    axes[0].set_title("60 s replacement masks path gains")
    axes[0].grid(axis="y", alpha=0.18, linewidth=0.8)
    for bar, share, value in zip(bars, shares, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, share + 0.05,
            f"{share:.2f}%\n({value:.0f} ms)", ha="center", va="bottom", fontsize=7.2,
        )

    axes[1].bar([f"{rate:g}" for rate in rates], expected_faults, color=["#999999", "#E69F00", "#009E73"])
    axes[1].set_xlabel("Failures / 1K GPU-shard-h")
    axes[1].set_ylabel("Expected faults in one 90 s trace")
    axes[1].set_title("Sparse hazard dilutes WCT impact")
    axes[1].grid(axis="y", alpha=0.18, linewidth=0.8)
    for xpos, value in enumerate(expected_faults):
        axes[1].text(xpos, value + 0.001, f"{value:.4f}", ha="center", fontsize=7.5)

    axes[2].plot(rates, c16_gain, marker="o", color="#0072B2", linewidth=2.2)
    axes[2].axhline(0.0, color="black", lw=1, ls="--")
    axes[2].set_xticks(rates, [f"{rate:g}" for rate in rates])
    axes[2].set_xlabel("Failures / 1K GPU-shard-h")
    axes[2].set_ylabel("WCT reduction at 16 req/GPU (%)")
    axes[2].set_title("Observed gap exists at zero faults")
    axes[2].grid(alpha=0.18, linewidth=0.8)
    for xpos, value in zip(rates, c16_gain):
        axes[2].annotate(f"{value:.1f}%", (xpos, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5)
    fig.suptitle("Why the old failure-driven target shape did not appear", fontsize=14, y=0.995)
    fig.subplots_adjust(top=0.78, bottom=0.25, wspace=0.35)
    fig.text(
        0.5, 0.025,
        "Frozen Qwen3-8B/DP64 model inputs and passed fixed-arrival results; diagnostic only, not a new experiment",
        ha="center", fontsize=8, color="#555555",
    )
    save(fig, figures / "s0_target_gap_diagnostic")

    manifest = {
        "status": "PRELIMINARY_FIGURES_COMPLETE",
        "source_audit": str(args.input_dir / "audit.json"),
        "source_audit_status": audit["status"],
        "s2_source_audit": str(args.s2_dir / "audit.json"),
        "s2_source_audit_status": s2_audit["status"],
        "s2_source_selection": s2_audit["selection"],
        "s2_source_summary_sha256": s2_audit["summary_sha256"],
        "superseded_s2_source": str(ROOT / "results" / "s2_fixed_duration_v1" / "audit.json"),
        "figures": sorted(str(path) for path in figures.glob("*")),
        "figure_sha256": {
            path.name: sha256(path) for path in sorted(figures.glob("*.pdf"))
        },
        "evidence_boundary": "SGLSim metadata DES preliminary; not physical A100 measurement; Native Mooncake is a modeled Router+Store approximation reusing a host-Store component, not production Router/Store failover; Native and RAIDServe are implementation-equivalent in this TP1 component model, so the duplicate RAIDServe curve is omitted from S1; S1 preserves the architecture-level affected-workflow WCT CDF target question and reports quantile shifts without manufacturing separation; S0 is a diagnostic derived only from frozen protocol inputs and passed S2 results; S2 uses only the passed common-90-second-arrival lineage; S4 uses whole-workload sparse-rate first-order Poisson integration; S5 reports controller alpha plus recorded residency-event fractions without inferring occupancy GiB.",
    }
    (figures / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
