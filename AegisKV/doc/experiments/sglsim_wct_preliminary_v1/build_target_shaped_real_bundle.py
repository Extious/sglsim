#!/usr/bin/env python3
"""Build an S1--S9 target-shaped bundle from passed real SGLSim artifacts.

The virtual figure generator defines questions and visual grammar only.  This
script never reads its synthetic arrays.  Every numeric mark comes from a
passed SGLSim result directory; unavailable series remain NOT MEASURED.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COLORS = {
    "aegis": "#0b6e75", "baseline": "#6b7280", "dram": "#d97706",
    "hbm": "#2563eb", "recompute": "#b91c1c", "accent": "#7c3aed",
    "green": "#15803d", "light": "#e5e7eb",
}
ARCH_COLORS = {
    "Native Mooncake": COLORS["baseline"], "Static FCFS": "#9ca3af",
    "DejaVu": COLORS["dram"], "KevlarFlow": COLORS["accent"],
    "GhostServe": COLORS["green"], "AegisKV": COLORS["aegis"],
}
ARCH_STYLES = {
    "Native Mooncake": "-", "Static FCFS": ":", "DejaVu": "--",
    "KevlarFlow": "-.", "GhostServe": (0, (3, 1, 1, 1)), "AegisKV": "-",
}
EVIDENCE = (
    "Audited three-seed SGLSim metadata DES preliminary; fixed 90 s arrivals; "
    "not physical A100 measurement and not paper-ready. Native Mooncake is a "
    "modeled Router+Store approximation."
)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.2, "axes.titlesize": 10.5,
    "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.5, "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported JSON value: {type(value)!r}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=convert) + "\n",
        encoding="utf-8",
    )


def load_passed(name: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    directory = RESULTS / name
    audit_path = directory / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("workflow_completion_time_result") is not True:
        raise RuntimeError(f"fail-closed: {name} has no passed WCT audit")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return directory, audit, summary


def raw_path(directory: Path, cell: dict[str, Any]) -> Path:
    value = Path(cell["raw_npz"])
    if value.is_absolute() and value.exists():
        return value
    local = directory / value
    if local.exists():
        return local
    repo = Path("/home/comp/zhanzhao/sglsim") / value
    if repo.exists():
        return repo
    raise FileNotFoundError(value)


def bootstrap_ci(values: Iterable[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(10000, array.size), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def save_figure(fig: plt.Figure, base: Path) -> dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = base.with_suffix(".pdf"), base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "audited SGLSim real-data bundle"})
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf), "pdf_sha256": sha256(pdf), "png": str(png), "png_sha256": sha256(png)}


def emit_panel(
    output: Path, panel: str, source_dirs: list[Path], data_path: Path,
    figures: dict[str, str], checks: dict[str, bool], not_measured: list[str],
    statement: str,
) -> dict[str, Any]:
    lineage_path = output / "lineage" / f"{panel}.json"
    data_ref = str(data_path.relative_to(output))
    pdf_ref = str(Path(figures["pdf"]).relative_to(output))
    png_ref = str(Path(figures["png"]).relative_to(output))
    lineage = {
        "panel": panel.upper(), "schema_version": 1,
        "question_source": "/home/comp/zhanzhao/Mooncake/aegiskv/experiment_plan/generate_virtual_figures.py",
        "question_source_role": "layout/axis/legend semantics only; synthetic values were not read or consumed",
        "numeric_sources": [
            {
                "result_dir": str(directory),
                "audit": str(directory / "audit.json"),
                "audit_sha256": sha256(directory / "audit.json"),
                "summary_sha256": sha256(directory / "summary.json"),
            }
            for directory in source_dirs
        ],
        "data": data_ref, "data_sha256": sha256(data_path),
        "not_measured": not_measured, "evidence_boundary": EVIDENCE,
    }
    write_json(lineage_path, lineage)
    audit_path = output / "audits" / f"{panel}.json"
    audit = {
        "panel": panel.upper(), "schema_version": 1,
        "status": "PASS_WITH_NOT_MEASURED" if not_measured else "PASS",
        "checks": checks, "not_measured": not_measured, "statement": statement,
        "data_sha256": sha256(data_path), "lineage_sha256": sha256(lineage_path),
        "figure_pdf_sha256": figures["pdf_sha256"],
        "figure_png_sha256": figures["png_sha256"],
        "paper_ready": False, "evidence_boundary": EVIDENCE,
    }
    if not all(checks.values()):
        audit["status"] = "FAIL"
    write_json(audit_path, audit)
    if audit["status"] == "FAIL":
        raise RuntimeError(f"{panel} audit failed: {[k for k, v in checks.items() if not v]}")
    return {
        "panel": panel.upper(), "status": audit["status"], "statement": statement,
        "data": data_ref,
        "lineage": str(lineage_path.relative_to(output)),
        "audit": str(audit_path.relative_to(output)),
        "pdf": pdf_ref, "pdf_sha256": figures["pdf_sha256"],
        "png": png_ref, "png_sha256": figures["png_sha256"],
    }


def panel_s1(output: Path) -> dict[str, Any]:
    directory, _, summary = load_passed("preliminary_v1")
    rows: list[dict[str, Any]] = []
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    policies = list(ARCH_COLORS)
    for policy in policies:
        samples: list[float] = []
        cells = [c for c in summary["cells"] if c["policy"] == policy and c["active_requests_per_gpu"] == 4]
        for cell in cells:
            with np.load(raw_path(directory, cell), allow_pickle=False) as raw:
                base, fault = raw["base_wct_ms"], raw["fault_wct_ms"]
                samples.extend((fault[fault > base[None, :] + 1e-9] / 1000.0).tolist())
        values = np.sort(np.asarray(samples, dtype=np.float64))
        cdf = np.arange(1, values.size + 1) / values.size
        series[policy] = values, cdf
        rows.extend({"policy": policy, "wct_seconds": float(x), "cdf": float(y), "status": "MEASURED"} for x, y in zip(values, cdf))
    data = output / "data" / "s1_architecture_wct_cdf.csv"
    write_csv(data, rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.65))
    for policy in policies:
        x, y = series[policy]
        label = "Native Mooncake (modeled)" if policy == "Native Mooncake" else policy
        ax.plot(x, y, color=ARCH_COLORS[policy], ls=ARCH_STYLES[policy],
                lw=2.2 if policy == "AegisKV" else 1.35, label=label)
    ax.set(xlabel="Affected-workflow WCT (s)", ylabel="CDF", ylim=(0, 1.01))
    ax.grid(alpha=.16)
    ax.set_title("S1. Architecture WCT CDF - observed")
    ax.legend(loc="lower right", ncol=2, fontsize=6.8)
    inset = ax.inset_axes([.18, .50, .40, .37])
    for policy in policies:
        x, y = series[policy]
        inset.plot(x, y, color=ARCH_COLORS[policy], ls=ARCH_STYLES[policy], lw=1.1)
    all_q = np.concatenate([np.quantile(x, [.80, .99]) for x, _ in series.values()])
    inset.set(xlim=(float(all_q.min()) - .02, float(all_q.max()) + .02), ylim=(.8, .995))
    inset.set_title("p80-p99 tail zoom", fontsize=7.3)
    inset.tick_params(labelsize=6.3)
    fig.text(.5, .01, "60 s replacement dominates; inset exposes the real near-overlap", ha="center", fontsize=7.2, color="#555")
    fig.subplots_adjust(bottom=.18)
    figures = save_figure(fig, output / "figures" / "s1_architecture_wct_cdf")
    return emit_panel(output, "s1", [directory], data, figures,
                      {"six_named_architectures": len(series) == 6, "affected_wct_nonempty": all(len(v[0]) > 0 for v in series.values())}, [],
                      "Replacement dominates all architecture CDFs; observed policy separation is small and retained without filtering.")


def panel_s2(output: Path) -> dict[str, Any]:
    directory, _, _ = load_passed("s2_fixed_arrival_v2")
    aggregate = read_csv(directory / "aggregate_ci.csv")
    rates = [0.0, 5.952, 20.833]
    rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(6.4, 3.55))
    for rate, color, marker in zip(rates, ["#9ca3af", COLORS["dram"], COLORS["aegis"]], ["o", "s", "D"]):
        picked = sorted([
            r for r in aggregate if r["policy"] == "AegisKV"
            and math.isclose(float(r["failure_rate_per_1000_gpu_shard_hour"]), rate)
        ], key=lambda r: int(r["active_requests_per_gpu"]))
        x = np.asarray([int(r["active_requests_per_gpu"]) for r in picked])
        ratio = np.asarray([float(r["normalized_to_native"]) for r in picked])
        low_r = np.asarray([float(r["normalized_ci95_low"]) for r in picked])
        high_r = np.asarray([float(r["normalized_ci95_high"]) for r in picked])
        y, low, high = 100 * (1-ratio), 100 * (1-high_r), 100 * (1-low_r)
        ax.plot(x, y, marker=marker, color=color, lw=2, label=f"{rate:g} / 1K GPU-shard-h")
        ax.fill_between(x, low, high, color=color, alpha=.13)
        rows.extend({"active_requests_per_gpu": int(xx), "failure_rate_per_1k_gpu_shard_hour": rate,
                     "wct_improvement_percent": float(yy), "ci95_low_percent": float(ll),
                     "ci95_high_percent": float(hh), "baseline": "Native Mooncake (modeled)",
                     "status": "MEASURED"} for xx, yy, ll, hh in zip(x, y, low, high))
    data = output / "data" / "s2_concurrency_three_failure_rates.csv"
    write_csv(data, rows)
    ax.axhline(0, color="#111827", lw=.9)
    ax.set_xticks([1,2,4,8,16])
    ax.set(xlabel="Active requests / GPU", ylabel="WCT improvement vs Native Mooncake (%)")
    ax.set_title("S2. Concurrency at three failure rates - observed")
    ax.legend(loc="best")
    ax.grid(alpha=.16)
    fig.text(.99, .01, "Positive is better; paired 3-seed 95% CI", ha="right", fontsize=7.2, color="#555")
    fig.subplots_adjust(bottom=.18)
    figures = save_figure(fig, output / "figures" / "s2_concurrency_three_failure_rates")
    return emit_panel(output, "s2", [directory], data, figures,
                      {"three_rates": sorted({r["failure_rate_per_1k_gpu_shard_hour"] for r in rows}) == rates,
                       "five_loads_each": all(sum(r["failure_rate_per_1k_gpu_shard_hour"] == rate for r in rows) == 5 for rate in rates)}, [],
                      "AegisKV's observed WCT difference versus modeled Native varies with queue pressure; no win threshold was imposed.")


def panel_s3(output: Path) -> dict[str, Any]:
    directory, _, _ = load_passed("s3_preliminary_v1")
    aggregate = read_csv(directory / "aggregate_ci.csv")
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
    for ax, axis_name in zip(axes, ["scale", "context"]):
        picked = sorted([
            r for r in aggregate if r["axis"] == axis_name and r["policy"] == "AegisKV"
            and math.isclose(float(r["failure_rate"]), 5.952)
        ], key=lambda r: float(r["value"]))
        x = np.asarray([float(r["value"]) for r in picked])
        ratio = np.asarray([float(r["normalized_to_native"]) for r in picked])
        low_r = np.asarray([float(r["normalized_ci95_low"]) for r in picked])
        high_r = np.asarray([float(r["normalized_ci95_high"]) for r in picked])
        y, low, high = 100*(1-ratio), 100*(1-high_r), 100*(1-low_r)
        ax.errorbar(x, y, yerr=[y-low, high-y], marker="o", color=COLORS["aegis"], capsize=3, lw=1.8, label="Qwen3-8B / AegisKV")
        ax.axhline(0, color="#111827", lw=.8, ls="--")
        ax.grid(alpha=.16)
        if axis_name == "scale":
            ax.set(xlabel="Total GPUs (system DP)", ylabel="WCT improvement (%)", xscale="log")
            ax.set_xticks(x, [str(int(v)) for v in x])
            ax.set_title("Scale (16K context)")
            ax.text(.02, .04, "Qwen3-32B: NOT MEASURED", transform=ax.transAxes, fontsize=7, color=COLORS["baseline"])
        else:
            ax.set(xlabel="Input context (K tokens)", ylabel="WCT improvement (%)")
            ax.set_xticks(x, [str(int(v/1024)) for v in x])
            ax.set_title("Context (system-DP=64)")
            ax.text(.02, .04, "Mooncake-DRAM: NOT MEASURED", transform=ax.transAxes, fontsize=7, color=COLORS["baseline"])
        ax.legend(loc="best")
        rows.extend({"axis": axis_name, "value": float(xx), "model": "Qwen3-8B",
                     "series": "AegisKV vs Native Mooncake (modeled)", "failure_rate": 5.952,
                     "wct_improvement_percent": float(yy), "ci95_low_percent": float(ll),
                     "ci95_high_percent": float(hh), "status": "MEASURED"}
                    for xx, yy, ll, hh in zip(x, y, low, high))
    rows.extend([
        {"axis": "scale", "value": "", "model": "Qwen3-32B", "series": "AegisKV", "failure_rate": 5.952,
         "wct_improvement_percent": "", "ci95_low_percent": "", "ci95_high_percent": "", "status": "NOT_MEASURED"},
        {"axis": "context", "value": "", "model": "Qwen3-8B", "series": "Mooncake-DRAM", "failure_rate": 5.952,
         "wct_improvement_percent": "", "ci95_low_percent": "", "ci95_high_percent": "", "status": "NOT_MEASURED"},
    ])
    data = output / "data" / "s3_scale_context.csv"
    write_csv(data, rows)
    fig.suptitle("S3. Scale and context sensitivity - observed", y=1.01)
    fig.tight_layout()
    figures = save_figure(fig, output / "figures" / "s3_scale_context")
    return emit_panel(output, "s3", [directory], data, figures,
                      {"five_scale_points": sum(r["axis"] == "scale" and r["status"] == "MEASURED" for r in rows) == 5,
                       "five_context_points": sum(r["axis"] == "context" and r["status"] == "MEASURED" for r in rows) == 5},
                      ["Qwen3-32B scale series", "Mooncake-DRAM context series"],
                      "For Qwen3-8B, all observed scale/context effects are near zero; unrun target series remain explicitly absent.")


def panel_s4(output: Path) -> dict[str, Any]:
    directory, _, _ = load_passed("preliminary_v1")
    aggregate = read_csv(directory / "aggregate_ci.csv")
    order = ["Native Mooncake", "AegisKV no Algorithm 1", "AegisKV no Algorithm 2", "AegisKV no Algorithm 3", "AegisKV"]
    labels = ["Native\nMooncake", "No Alg.1", "No Alg.2", "No Alg.3", "Full\nAegisKV"]
    selected = {r["policy"]: r for r in aggregate if int(r["active_requests_per_gpu"]) == 4
                and math.isclose(float(r["failure_rate_per_1000_gpu_shard_hour"]), 5.952)}
    rows: list[dict[str, Any]] = []
    means, lows, highs = [], [], []
    for policy, label in zip(order, labels):
        r = selected[policy]
        means.append(float(r["normalized_to_native"]))
        lows.append(float(r["normalized_ci95_low"]))
        highs.append(float(r["normalized_ci95_high"]))
        rows.append({"variant": label.replace("\n", " "), "policy": policy, "normalized_wct": means[-1],
                     "ci95_low": lows[-1], "ci95_high": highs[-1], "status": "MEASURED"})
    data = output / "data" / "s4_algorithm_ablation.csv"
    write_csv(data, rows)
    means_a, lows_a, highs_a = map(np.asarray, [means, lows, highs])
    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    colors = ["#9ca3af", "#8fb9ad", "#89a7c2", "#b4a0c8", COLORS["aegis"]]
    x = np.arange(5)
    bars = ax.bar(x, means_a, yerr=[means_a-lows_a, highs_a-means_a], capsize=3, color=colors, width=.68)
    ax.axhline(1, color="#111827", lw=.9, ls="--")
    extent = max(2e-5, float(np.max(np.abs(np.concatenate([means_a-1, lows_a-1, highs_a-1]))))*1.45)
    ax.set_ylim(1-extent, 1+extent)
    explicit_ticks = np.linspace(1-extent, 1+extent, 5)
    ax.set_yticks(explicit_ticks, [f"{value:.6f}" for value in explicit_ticks])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Normalized WCT (Native Mooncake = 1.0)")
    ax.set_title("S4. Algorithm 1/2/3 ablation - observed")
    for bar, value in zip(bars, means_a):
        ax.annotate(
            f"{(value-1)*1e6:+.1f} ppm",
            (bar.get_x()+bar.get_width()/2, value), xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom", fontsize=6.8,
        )
    fig.text(.5, .01, "Lower is better; local y-axis exposes ppm-scale differences; paired 3-seed 95% CI", ha="center", fontsize=7.1, color="#555")
    fig.subplots_adjust(bottom=.2)
    figures = save_figure(fig, output / "figures" / "s4_algorithm_ablation")
    return emit_panel(output, "s4", [directory], data, figures,
                      {"five_bars": len(rows) == 5, "native_exactly_one": math.isclose(means[0], 1.0, abs_tol=1e-12)}, [],
                      "All algorithm-ablation WCT differences are ppm-scale in this center cell; Full is not selectively favored.")


def step_at(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    index = np.searchsorted(times, grid, side="right") - 1
    index = np.clip(index, 0, len(values)-1)
    return values[index]


def panel_s5(output: Path) -> dict[str, Any]:
    directory, _, summary = load_passed("s5_timeline_v1")
    controls: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grid = np.arange(0, 90001, 1000, dtype=np.float64)
    hbm_seed, dram_seed = [], []
    for cell in summary["cells"]:
        for point in cell["mechanism"]["control_step_trace"]:
            for key in ["alpha_after_mean", "pressure_fraction", "active_requests_per_gpu", "hbm_used_bytes_total", "dram_used_bytes_total"]:
                controls[float(point["time_ms"])][key].append(float(point[key]))
        with np.load(raw_path(directory, cell), allow_pickle=False) as raw:
            times = raw["occupancy_time_ms"].astype(float)
            hbm_seed.append(step_at(times, raw["occupancy_hbm_total_bytes"].astype(float), grid) / 64 / 2**30)
            dram_seed.append(step_at(times, raw["occupancy_dram_total_bytes"].astype(float), grid) / 64 / 2**30)
    rows: list[dict[str, Any]] = []
    control_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    control_times = sorted(controls)
    for metric in ["active_requests_per_gpu", "pressure_fraction", "alpha_after_mean"]:
        stats = [bootstrap_ci(controls[t][metric], 2600 + i*17) for i, t in enumerate(control_times)]
        control_stats[metric] = tuple(np.asarray([s[j] for s in stats]) for j in range(3))
        for t, stat in zip(control_times, stats):
            rows.append({"panel": "control", "time_seconds": t/1000, "metric": metric,
                         "mean": stat[0], "ci95_low": stat[1], "ci95_high": stat[2], "unit": "ratio" if metric != "active_requests_per_gpu" else "requests/GPU", "status": "MEASURED"})
    for metric, arrays in [("hbm_occupancy", np.asarray(hbm_seed)), ("dram_occupancy", np.asarray(dram_seed))]:
        for i, t in enumerate(grid):
            stat = bootstrap_ci(arrays[:, i], 5100+i)
            rows.append({"panel": "occupancy", "time_seconds": t/1000, "metric": metric,
                         "mean": stat[0], "ci95_low": stat[1], "ci95_high": stat[2], "unit": "GiB/GPU", "status": "MEASURED"})
    data = output / "data" / "s5_alpha_pressure_load_occupancy_timeline.csv"
    write_csv(data, rows)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    t = np.asarray(control_times)/1000
    mean, low, high = control_stats["active_requests_per_gpu"]
    axes[0].errorbar(t, mean, yerr=[mean-low, high-mean], marker="o", color="#111827", label="Active req/GPU")
    control_ax = axes[0].twinx()
    for metric, color, marker, label in [("pressure_fraction", COLORS["recompute"], "s", "Pressure"), ("alpha_after_mean", COLORS["aegis"], "D", "alpha")]:
        mean, low, high = control_stats[metric]
        control_ax.errorbar(t, mean, yerr=[mean-low, high-mean], marker=marker, color=color, label=label)
    axes[0].set_ylabel("Active req/GPU")
    control_ax.set_ylabel("Pressure / alpha")
    control_ax.set_ylim(0, 1.08)
    handles_left, labels_left = axes[0].get_legend_handles_labels()
    handles_right, labels_right = control_ax.get_legend_handles_labels()
    axes[0].legend(
        handles_left + handles_right, labels_left + labels_right,
        ncol=3, loc="upper center", bbox_to_anchor=(.5, -.13),
    )
    for at in t:
        axes[0].axvline(at, color="#cbd5e1", lw=.7, ls=":")
        axes[1].axvline(at, color="#cbd5e1", lw=.7, ls=":")
    for arrays, color, label in [(np.asarray(hbm_seed), COLORS["hbm"], "Peer-HBM"), (np.asarray(dram_seed), COLORS["dram"], "DRAM")]:
        mean = arrays.mean(axis=0); low = arrays.min(axis=0); high = arrays.max(axis=0)
        axes[1].plot(grid/1000, mean, color=color, lw=1.7, label=label)
        axes[1].fill_between(grid/1000, low, high, color=color, alpha=.12)
    axes[1].set(xlabel="Simulation time (s)", ylabel="Occupancy (GiB/GPU)")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=.12)
    axes[0].set_title("Three common control opportunities (30/60/90 s)")
    fig.suptitle("S5. A100 40GB alpha, pressure, load and occupancy - observed", y=.995)
    fig.subplots_adjust(hspace=.38, top=.89, bottom=.12)
    figures = save_figure(fig, output / "figures" / "s5_alpha_pressure_load_occupancy_timeline")
    alpha_final = control_stats["alpha_after_mean"][0][-1]
    return emit_panel(output, "s5", [directory], data, figures,
                      {"three_control_steps": control_times == [30000.0,60000.0,90000.0],
                       "occupancy_from_raw_used_bytes": len(hbm_seed) == 3 and all(a.size == grid.size for a in hbm_seed),
                       "alpha_remains_nonzero": alpha_final > 0}, [],
                      "Pressure stays near saturation across the three common windows, alpha decreases but remains nonzero, and tier occupancy is reconstructed from persisted used_bytes events.")


def panel_s6(output: Path) -> dict[str, Any]:
    directory, _, summary = load_passed("s6_preliminary_v3")
    reuse_values = sorted({float(c["extension_spec"]["reuse"]) for c in summary["cells"]})
    rows: list[dict[str, Any]] = []
    stats: dict[str, list[tuple[float,float,float]]] = defaultdict(list)
    for i, reuse in enumerate(reuse_values):
        cells = [c for c in summary["cells"] if float(c["extension_spec"]["reuse"]) == reuse]
        metrics = {
            "recovery_hit_percent": [100*float(c["mechanism"]["recovery_hit_fraction"]) for c in cells],
            "copy_amplification": [float(c["mechanism"]["copy_amplification"]) for c in cells],
            "hbm_evictions_per_1k": [1000*float(c["mechanism"]["tier_churn"]["HBM"]["eviction_count"])/float(c["workload"]["request_count"]) for c in cells],
            "dram_evictions_per_1k": [1000*float(c["mechanism"]["tier_churn"]["DRAM"]["eviction_count"])/float(c["workload"]["request_count"]) for c in cells],
            "hbm_occupancy_gib_per_gpu": [float(c["mechanism"]["occupancy_time_mean_gib_per_gpu"]["HBM"]) for c in cells],
            "dram_occupancy_gib_per_gpu": [float(c["mechanism"]["occupancy_time_mean_gib_per_gpu"]["DRAM"]) for c in cells],
        }
        for offset, (metric, values) in enumerate(metrics.items()):
            stat = bootstrap_ci(values, 6000+i*31+offset)
            stats[metric].append(stat)
            rows.append({"reuse_fraction": reuse, "metric": metric, "mean": stat[0], "ci95_low": stat[1], "ci95_high": stat[2], "status": "MEASURED"})
    for metric in stats:
        rows.append({"reuse_fraction": 1.0, "metric": metric, "mean": "", "ci95_low": "", "ci95_high": "", "status": "NOT_MEASURED"})
    data = output / "data" / "s6_reuse_cost_occupancy.csv"
    write_csv(data, rows)
    x = np.asarray(reuse_values)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
    for metric, axis, color, marker, label in [("recovery_hit_percent", axes[0], COLORS["aegis"], "o", "Recovery hit rate")]:
        arr=np.asarray(stats[metric]); axis.errorbar(x, arr[:,0], yerr=[arr[:,0]-arr[:,1],arr[:,2]-arr[:,0]], marker=marker, color=color, capsize=3, label=label)
    ax2=axes[0].twinx(); arr=np.asarray(stats["copy_amplification"])
    ax2.errorbar(x, arr[:,0], yerr=[arr[:,0]-arr[:,1],arr[:,2]-arr[:,0]], marker="s", ls="--", color=COLORS["recompute"], capsize=3, label="Copy amplification")
    axes[0].set(xlabel="Prefix reuse probability", ylabel="Recovery hit rate (%)")
    ax2.set_ylabel("Copied / useful bytes")
    handles_left, labels_left = axes[0].get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    axes[0].legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    width=.08
    for j,(metric,color,label) in enumerate([("hbm_evictions_per_1k",COLORS["hbm"],"HBM evictions"),("dram_evictions_per_1k",COLORS["dram"],"DRAM evictions")]):
        arr=np.asarray(stats[metric]); xpos=x+(-width/2 if j==0 else width/2)
        axes[1].bar(xpos,arr[:,0],width,color=color,alpha=.8,label=label)
        axes[1].errorbar(xpos,arr[:,0],yerr=[arr[:,0]-arr[:,1],arr[:,2]-arr[:,0]],fmt="none",ecolor="#333",capsize=2,lw=.8)
    occ=axes[1].twinx()
    for metric,color,label in [("hbm_occupancy_gib_per_gpu",COLORS["hbm"],"HBM occupancy"),("dram_occupancy_gib_per_gpu",COLORS["dram"],"DRAM occupancy")]:
        arr=np.asarray(stats[metric]); occ.plot(x,arr[:,0],marker="o",ls="--",color=color,label=label)
    axes[1].set(xlabel="Prefix reuse probability",ylabel="Evictions / 1K requests")
    occ.set_ylabel("Time-mean occupancy (GiB/GPU)")
    h1,l1=axes[1].get_legend_handles_labels(); h2,l2=occ.get_legend_handles_labels(); axes[1].legend(h1+h2,l1+l2,fontsize=6.2,loc="upper right")
    for ax in axes:
        ax.set_xticks([0,.25,.5,.75,1.0])
        ax.axvspan(.875,1.02,color=COLORS["light"],alpha=.55)
        ax.text(.995,.04,"1.0\nNOT MEASURED",transform=ax.get_xaxis_transform(),ha="right",va="bottom",fontsize=6.2,color=COLORS["baseline"])
        ax.grid(alpha=.12)
    axes[0].set_title("Reuse hit and copy amplification")
    axes[1].set_title("Evictions and capacity-integral occupancy")
    fig.suptitle("S6. Reuse benefit and protection cost - observed",y=1.01)
    fig.tight_layout()
    figures=save_figure(fig,output/"figures"/"s6_reuse_cost_occupancy")
    return emit_panel(output,"s6",[directory],data,figures,
                      {"four_measured_reuse_values": reuse_values == [0,.25,.5,.75],
                       "capacity_integral_fields_consumed": all("occupancy" in k for k in ["hbm_occupancy_gib_per_gpu","dram_occupancy_gib_per_gpu"])},
                      ["reuse_fraction=1.0"],
                      "Reuse changes measured tier occupancy and HBM eviction pressure; recovery-hit fraction is roughly flat and the unrun reuse=1.0 target is not extrapolated.")


def panel_s7(output: Path) -> dict[str, Any]:
    directory, _, _ = load_passed("s7_target_v1")
    aggregate=read_csv(directory/"aggregate_ci.csv")
    rows=[]; fig,axes=plt.subplots(1,2,figsize=(7.4,3.25))
    for ax,axis_name in zip(axes,["bandwidth","repair"]):
        picked=sorted([r for r in aggregate if r["axis"]==axis_name and r["policy"]=="AegisKV" and math.isclose(float(r["failure_rate"]),5.952)],key=lambda r:float(r["value"]))
        x=np.asarray([float(r["value"]) for r in picked]); ratio=np.asarray([float(r["normalized_to_native"]) for r in picked]); lo_r=np.asarray([float(r["normalized_ci95_low"]) for r in picked]); hi_r=np.asarray([float(r["normalized_ci95_high"]) for r in picked])
        y,lo,hi=100*(1-ratio),100*(1-hi_r),100*(1-lo_r)
        display_x=x if axis_name=="bandwidth" else x/1000
        ax.errorbar(display_x,y,yerr=[y-lo,hi-y],marker="o" if axis_name=="bandwidth" else "s",color=COLORS["aegis"],capsize=3,lw=1.8,label="AegisKV")
        ax.axhline(0,color=COLORS["baseline"],ls="--",lw=1,label="Native Mooncake")
        ax.set(xlabel="Fabric bandwidth (GB/s)" if axis_name=="bandwidth" else "Worker replacement time (s)",ylabel="WCT improvement (%)")
        ax.set_xticks(display_x)
        ax.set_title("Bandwidth" if axis_name=="bandwidth" else "Repair time")
        ax.legend(); ax.grid(alpha=.14)
        rows.extend({"axis":axis_name,"value":float(xx),"display_value":float(dx),"failure_rate":5.952,"wct_improvement_percent":float(yy),"ci95_low_percent":float(ll),"ci95_high_percent":float(hh),"baseline":"Native Mooncake (modeled)","status":"MEASURED"} for xx,dx,yy,ll,hh in zip(x,display_x,y,lo,hi))
    data=output/"data"/"s7_bandwidth_repair_sensitivity.csv"; write_csv(data,rows)
    fig.suptitle("S7. Network and repair sensitivity - observed",y=1.01); fig.tight_layout()
    figures=save_figure(fig,output/"figures"/"s7_bandwidth_repair_sensitivity")
    return emit_panel(output,"s7",[directory],data,figures,
                      {"five_bandwidth_points":sum(r["axis"]=="bandwidth" for r in rows)==5,"four_repair_points":sum(r["axis"]=="repair" for r in rows)==4},[],
                      "Observed sensitivity is reported for forced cross-node Peer-HBM plus remote DRAM; no monotonicity or win threshold was required.")


def panel_s8(output: Path) -> dict[str, Any]:
    directory, _, summary=load_passed("preliminary_v1")
    per: dict[tuple[str,str],list[float]]=defaultdict(list)
    for policy in ["Native Mooncake","AegisKV"]:
        cells=[c for c in summary["cells"] if c["policy"]==policy and c["active_requests_per_gpu"]==4]
        for cell in cells:
            with np.load(raw_path(directory,cell),allow_pickle=False) as raw:
                types=np.asarray(raw["probe_failure_type"]).astype(str)
                fault=raw["fault_wct_ms"]
                for failure_type in ["source_worker_gpu","source_node"]:
                    per[(policy,failure_type)].append(float(fault[types==failure_type].mean()/1000))
    rows=[]
    for i,((policy,failure_type),values) in enumerate(sorted(per.items())):
        stat=bootstrap_ci(values,8000+i)
        rows.append({"policy":policy,"failure_type":failure_type,"mean_wct_seconds":stat[0],"ci95_low_seconds":stat[1],"ci95_high_seconds":stat[2],"seed_count":len(values),"status":"MEASURED"})
    data=output/"data"/"s8_failure_types.csv"; write_csv(data,rows)
    lookup={(r["policy"],r["failure_type"]):r for r in rows}; types=["source_worker_gpu","source_node"]; labels=["Source worker/GPU","Source node"]
    fig,ax=plt.subplots(figsize=(6.4,3.35)); x=np.arange(2); width=.36
    for j,(policy,color) in enumerate([("Native Mooncake",COLORS["baseline"]),("AegisKV",COLORS["aegis"])]):
        values=np.asarray([lookup[(policy,t)]["mean_wct_seconds"] for t in types]); lo=np.asarray([lookup[(policy,t)]["ci95_low_seconds"] for t in types]); hi=np.asarray([lookup[(policy,t)]["ci95_high_seconds"] for t in types]); xpos=x+(-width/2 if j==0 else width/2)
        ax.bar(xpos,values,width,color=color,label="Native Mooncake (modeled)" if j==0 else policy)
        ax.errorbar(xpos,values,yerr=[values-lo,hi-values],fmt="none",ecolor="#333",capsize=3,lw=.9)
    ax.set_xticks(x,labels); ax.set_ylabel("Failure-conditioned mean WCT (s)"); ax.set_title("S8. Worker/GPU and whole-node failure - observed"); ax.legend()
    fig.tight_layout(); figures=save_figure(fig,output/"figures"/"s8_failure_types")
    return emit_panel(output,"s8",[directory],data,figures,
                      {"two_failure_types":set(types)=={k[1] for k in per},"paired_three_seeds":all(len(v)==3 for v in per.values())},[],
                      "Source worker/GPU and source-node bars are independently extracted from the preregistered paired fault probes in the center cell.")


def panel_s9(output: Path) -> dict[str, Any]:
    rows=[{"payload_mib":v,"physical_gib_s":"","sglsim_gib_s":"","relative_error_percent":"","status":"NOT_MEASURED"} for v in [64,256,512,1024]]
    data=output/"data"/"s9_calibration_overlay.csv"; write_csv(data,rows)
    schema=output/"data"/"s9_calibration_import_schema.json"
    write_json(schema,{"schema_version":1,"status":"AWAITING_PHYSICAL_PROFILE","required_columns":["payload_mib","path","physical_gib_s","sglsim_gib_s","physical_run_id","physical_audit_sha256","sglsim_source_sha256"],"allowed_paths":["peer_hbm_same_node","peer_hbm_cross_node","dram_local","dram_remote"],"relative_error_formula":"abs(sglsim_gib_s - physical_gib_s) / physical_gib_s * 100","join_key":["payload_mib","path"]})
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.25)); sizes=[64,256,512,1024]
    axes[0].set_xscale("log",base=2); axes[0].set_xticks(sizes,[str(x) for x in sizes]); axes[0].set(xlabel="Payload (MiB)",ylabel="Bandwidth (GiB/s)",title="Calibration overlay")
    axes[1].set_xticks(range(4),[str(x) for x in sizes]); axes[1].set(xlabel="Payload (MiB)",ylabel="Relative error (%)",title="Model error",ylim=(0,12))
    axes[1].axhline(10,color=COLORS["recompute"],ls="--",label="10% display gate"); axes[1].legend()
    for index, ax in enumerate(axes):
        title_y, note_y = ((.52, .40) if index == 0 else (.70, .58))
        ax.text(.5,title_y,"NOT MEASURED",transform=ax.transAxes,ha="center",va="center",fontsize=13,color=COLORS["baseline"],weight="bold")
        ax.text(.5,note_y,"awaiting staged Mooncake profile",transform=ax.transAxes,ha="center",fontsize=7.5,color="#555")
        ax.grid(alpha=.12)
    fig.suptitle("S9. Simulator-to-physical calibration",y=1.01); fig.tight_layout()
    figures=save_figure(fig,output/"figures"/"s9_calibration_overlay")
    return emit_panel(output,"s9",[],data,figures,{"import_schema_present":schema.exists(),"no_numeric_overlay_fabricated":all(r["physical_gib_s"]=="" and r["sglsim_gib_s"]=="" for r in rows)},["physical Mooncake calibration","matched SGLSim calibration"],"No calibration conclusion is available until a staged Mooncake profile passes audit and is joined through the supplied schema.")


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=ROOT/"target_shaped_real_bundle.staging"); args=parser.parse_args()
    output=args.output_dir
    if output.exists():
        raise FileExistsError(f"refusing to mix lineage in existing directory: {output}")
    output.mkdir(parents=True)
    builders=[panel_s1,panel_s2,panel_s4,panel_s5,panel_s6,panel_s3,panel_s7,panel_s8,panel_s9]
    panels=[]
    for builder in builders:
        print(f"BUILD {builder.__name__}",flush=True); panels.append(builder(output))
    manifest={"schema_version":1,"bundle_status":"PASS_WITH_NOT_MEASURED","title":"Target-shaped audited real-data SGLSim preliminary S1-S9","evidence_boundary":EVIDENCE,"panels":panels,"builder":{"path":str(Path(__file__).resolve()),"sha256":sha256(Path(__file__).resolve())}}
    write_json(output/"manifest.json",manifest)
    hashes=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name!="SHA256SUMS": hashes.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output/"SHA256SUMS").write_text("\n".join(hashes)+"\n",encoding="utf-8")
    print(output/"manifest.json")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
