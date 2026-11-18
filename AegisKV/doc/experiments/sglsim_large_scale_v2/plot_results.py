#!/usr/bin/env python3
"""Render paper-grade, explicitly simulated-only AegisKV figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "aegiskv_alg111": "#111111",
    "recompute_only": "#D9D9D9",
    "static_independent_fcfs": "#4D4D4D",
    "dejavu_neighbor": "#737373",
    "raidserve_host": "#969696",
    "kevlarflow_ring": "#BDBDBD",
    "ghostserve_parity": "#636363",
}
LABEL = {
    "aegiskv_alg111": "AegisKV",
    "recompute_only": "Recompute",
    "static_independent_fcfs": "Static tiers",
    "dejavu_neighbor": "DejaVu-style",
    "raidserve_host": "RAIDServe-style",
    "kevlarflow_ring": "KevlarFlow-style",
    "ghostserve_parity": "GhostServe-style",
}
PREFIGURE_REQUIRED_CHECKS = frozenset(
    {
        "protocol_sha",
        "matrix_sha",
        "matrix_canonical_sha",
        "producer_source_hashes",
        "cache_manifest_internal_refs_and_generator_sha",
        "raw_contract",
        "canonical_campaign_index_is_zero_failure_complete",
        "t0_aimd_snapshot_and_periodic_contract",
        "failure_history_horizon_and_post_event_drain_accounting_exact",
        "complete_raw_retention",
        "failure_tuple_schedule_identical_across_policies",
        "active_source_probe_raw_contract",
        "active_source_probe_tuple_identical_across_policy_and_rate",
        "selection_provenance_matches_raw_tuples",
        "exact_3110",
        "exact_matrix_run_ids",
        "exact_ten_paired_seeds_per_policy_scenario",
        "analysis_status_exact_for_campaign_mode",
        "analysis_files_match",
        "analysis_contract_pass",
        "analysis_rationality_matches_mode",
        "analysis_algorithm1_identifiability_matches_mode",
        "analysis_paired_comparison_coverage_matches_mode",
        "analysis_paired_probe_comparison_coverage_matches_mode",
        "analysis_scenario_diagnostics_count_exact",
        "analysis_paper_eligibility_exact",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_analysis(analysis_dir: Path, *, allow_partial: bool) -> dict[str, Any]:
    path = analysis_dir / "analysis_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    status = str(manifest["status"])
    if status != "COMPLETE":
        raise ValueError(
            "figures require COMPLETE full-matrix analysis; pilot/custom/failed "
            "analyses remain table-only diagnostics"
        )
    if manifest.get("selection_mode") != "FULL_FROZEN_MATRIX" or manifest.get(
        "paper_eligible"
    ) is not True:
        raise ValueError("COMPLETE analysis lacks full-matrix paper eligibility")
    for name, record in manifest["files"].items():
        candidate = analysis_dir / name
        if sha256(candidate) != record["sha256"]:
            raise ValueError(f"analysis input hash mismatch: {candidate}")
    for name, record in manifest["analysis_sources"].items():
        source = Path(record["path"])
        if sha256(source) != record["sha256"]:
            raise ValueError(f"analysis/render source hash mismatch: {name}")
    return manifest


def verify_prefigure_audit(
    path: Path,
    *,
    analysis_dir: Path,
    analysis_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require the independent raw+analysis full audit before formal figures."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical_payload = dict(payload)
    expected_canonical = str(canonical_payload.pop("canonical_sha256", ""))
    actual_canonical = hashlib.sha256(
        json.dumps(
            canonical_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    auditor_path = Path(str(payload.get("auditor", {}).get("path", "")))
    analysis_sha = sha256(analysis_dir / "analysis_manifest.json")
    checks = {
        "canonical": bool(expected_canonical)
        and expected_canonical == actual_canonical,
        "simulated_only": payload.get("evidence_class")
        == "simulated_sglsim_only_campaign_audit"
        and payload.get("physical_result") is False,
        "status": payload.get("status") == "PASS"
        and payload.get("failed_checks") == []
        and bool(payload.get("checks"))
        and all(bool(value) for value in payload.get("checks", {}).values()),
        "required_full_checks": PREFIGURE_REQUIRED_CHECKS.issubset(
            payload.get("checks", {})
        ),
        "full_prefigure_phase": payload.get("require_full") is True
        and payload.get("figures") is None,
        "analysis_binding": payload.get("analysis", {}).get("sha256")
        == analysis_sha,
        "campaign_binding": payload.get("campaign", {}).get(
            "campaign_index_sha256"
        )
        == analysis_manifest.get("campaign_index", {}).get("sha256"),
        "auditor_binding": auditor_path.is_file()
        and payload.get("auditor", {}).get("sha256") == sha256(auditor_path),
    }
    if not all(checks.values()):
        raise ValueError(f"prefigure full audit is not consumable: {checks}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "canonical_sha256": expected_canonical,
        "auditor_sha256": payload["auditor"]["sha256"],
    }


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def central(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["model_id"] == "qwen3_8b")
        & (frame["active_workflow_concurrency"] == 128)
        & np.isclose(frame["failure_rate_per_1000_worker_hours"], 5.952)
        & (frame["total_gpu_count"] == 64)
        & np.isclose(frame["prompt_scale"], 0.016)
        & np.isclose(frame["network_effective_gb_s"], 10.0)
        & np.isclose(frame["hbm_quota_gib_per_target"], 4.0)
        & np.isclose(frame["dram_quota_gib_per_target"], 64.0)
        & np.isclose(frame["algorithm1_prediction_error_multiplier"], 1.0)
    ]


def pooled_restore_by(data: pd.DataFrame, key: str) -> pd.Series:
    values: dict[Any, float] = {}
    for group_key, group in data.groupby(key):
        restored = float(group["summary.restored_tokens"].sum())
        total = restored + float(group["summary.recomputed_tokens"].sum())
        values[group_key] = restored / total if total > 0.0 else np.nan
    return pd.Series(values, dtype=float).sort_index()


def pooled_scenario_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, group in data.groupby("scenario_id"):
        restored = float(group["summary.restored_tokens"].sum())
        recovered = restored + float(group["summary.recomputed_tokens"].sum())
        prompt = float(group["summary.input_prompt_tokens"].sum())
        logical = float(group["summary.logical_request_kv_bytes"].sum())
        rows.append(
            {
                "scenario_id": scenario_id,
                "input_reuse_fraction": (
                    float(group["summary.input_reuse_tokens"].sum()) / prompt
                    if prompt > 0
                    else np.nan
                ),
                "restore_hit_fraction": restored / recovered if recovered > 0 else np.nan,
                "copy_amplification": (
                    float(group["summary.protection_copy_attempted_bytes"].sum())
                    / logical
                    if logical > 0
                    else np.nan
                ),
                "failure_rate": group[
                    "failure_rate_per_1000_worker_hours"
                ].iloc[0],
                "concurrency": group["active_workflow_concurrency"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    fig.text(
        0.995,
        0.005,
        "SGLSim metadata-DES; no physical inference",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_baselines(
    frame: pd.DataFrame,
    probes: pd.DataFrame,
    paired_probes: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    data = central(frame)
    policies = [policy for policy in PALETTE if policy in set(data["policy"])]
    scenario_id = str(data["scenario_id"].iloc[0])
    comparisons = paired_probes[
        (paired_probes["scenario_id"] == scenario_id)
        & paired_probes["reference"].isin(policies)
        & (paired_probes["metric"] == "normal_state_burden_ms_per_workflow_epoch")
    ].set_index("reference")
    compared_policies = [
        policy
        for policy in policies
        if policy != "aegiskv_alg111" and policy in comparisons.index
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x_compare = np.arange(len(compared_policies))
    improvement = -comparisons.loc[compared_policies, "mean_delta"]
    improvement_low = (
        -comparisons.loc[compared_policies, "delta_ci95_high"]
    )
    improvement_high = (
        -comparisons.loc[compared_policies, "delta_ci95_low"]
    )
    axes[0].bar(
        x_compare,
        improvement,
        yerr=np.vstack(
            [improvement - improvement_low, improvement_high - improvement]
        ),
        color=[PALETTE[p] for p in compared_policies],
        capsize=2,
    )
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_xticks(
        x_compare,
        [LABEL[p] for p in compared_policies],
        rotation=25,
        ha="right",
    )
    axes[0].set_ylabel("AegisKV burden reduction\n(ms/workflow/epoch)")
    axes[0].set_title("Normal-state K64; outer-seed CI")
    central_row = data.iloc[0]
    probe_data = probes[
        (probes["model_id"] == central_row["model_id"])
        & np.isclose(probes["prompt_scale"], central_row["prompt_scale"])
        & (
            probes["active_workflow_concurrency"]
            == central_row["active_workflow_concurrency"]
        )
        & (probes["total_gpu_count"] == central_row["total_gpu_count"])
        & (probes["tp_size"] == central_row["tp_size"])
        & (probes["pp_size"] == central_row["pp_size"])
        & (probes["system_dp_size"] == central_row["system_dp_size"])
        & np.isclose(
            probes["network_effective_gb_s"],
            central_row["network_effective_gb_s"],
        )
        & np.isclose(
            probes["hbm_quota_gib_per_target"],
            central_row["hbm_quota_gib_per_target"],
        )
        & np.isclose(
            probes["dram_quota_gib_per_target"],
            central_row["dram_quota_gib_per_target"],
        )
        & np.isclose(
            probes["algorithm1_prediction_error_multiplier"],
            central_row["algorithm1_prediction_error_multiplier"],
        )
        & np.isclose(probes["failure_rate_per_1000_worker_hours"], 0.0)
    ]
    totals = probe_data.groupby("policy")[["restored_tokens", "recomputed_tokens"]].sum()
    denominator = totals["restored_tokens"] + totals["recomputed_tokens"]
    restore = totals["restored_tokens"].div(denominator.where(denominator > 0))
    recompute = totals["recomputed_tokens"].div(
        denominator.where(denominator > 0)
    )
    valid = denominator > 0
    if not np.allclose(
        (restore[valid] + recompute[valid]).to_numpy(), 1.0
    ):
        raise ValueError("pooled restore/recompute stack does not sum to one")
    x = np.arange(len(policies))
    axes[1].bar(x, restore.loc[policies], color=[PALETTE[p] for p in policies], label="Restore")
    axes[1].bar(
        x,
        recompute.loc[policies],
        bottom=restore.loc[policies],
        color="#D9D9D9",
        label="Recompute",
    )
    axes[1].set_xticks(x, [LABEL[p] for p in policies], rotation=25, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Failed-token fraction")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_title("Rate-zero K64 recovery outcome")
    if (~valid.reindex(policies).fillna(False)).any():
        axes[1].text(
            0.02,
            0.02,
            "NA: zero recovery-token denominator",
            transform=axes[1].transAxes,
            fontsize=10,
        )
    return save(fig, output_dir, "fig01_baselines")


def plot_concurrency_failure(
    frame: pd.DataFrame, burdens: pd.DataFrame, output_dir: Path
) -> list[Path]:
    data = frame[
        (frame["policy"] == "aegiskv_alg111")
        & frame["campaign_axis"].str.contains("concurrency_failure", regex=False)
        & np.isclose(frame["algorithm1_prediction_error_multiplier"], 1.0)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    burden_data = burdens[
        (burdens["policy"] == "aegiskv_alg111")
        & burdens["campaign_axis"].str.contains("concurrency_failure", regex=False)
        & np.isclose(burdens["algorithm1_prediction_error_multiplier"], 1.0)
    ]
    specs = [
        (
            burden_data,
            "normal_state_burden_ms_per_workflow_epoch",
            "Normal-state burden\n(ms/workflow/epoch)",
        ),
        (data, "steady.effective_alpha_mean", "Mean effective alpha"),
        (data, "steady.pressure_fraction_mean", "Pressure fraction"),
    ]
    for ax, (source, metric, title) in zip(axes, specs, strict=True):
        pivot = source.pivot_table(
            index="active_workflow_concurrency",
            columns="failure_rate_per_1000_worker_hours",
            values=metric,
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        image = ax.imshow(
            pivot.to_numpy(), aspect="auto", origin="lower", cmap="Greys"
        )
        ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
        ax.set_xlabel("Failures / 1,000 GPU-shard-h")
        ax.set_ylabel("Offered trace workflows")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    axes[0].text(
        0.02,
        0.98,
        "rate-zero K64 × hazard",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
    )
    return save(fig, output_dir, "fig02_concurrency_failure")


def plot_resources(probes: pd.DataFrame, output_dir: Path) -> list[Path]:
    full = probes[probes["policy"] == "aegiskv_alg111"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    fig.suptitle(
        "Rate-local live-state K64; may include prior uptime history",
        fontsize=10,
    )
    specs = [
        ("cluster_scale", "total_gpu_count", "Total GPUs"),
        ("network_bandwidth", "network_effective_gb_s", "Effective network (GB/s)"),
        ("hbm_capacity", "hbm_quota_gib_per_target", "Peer-HBM quota / GPU shard (GiB)"),
        ("dram_capacity", "dram_quota_gib_per_target", "DRAM quota / GPU shard (GiB)"),
    ]
    for ax, (axis, xkey, xlabel) in zip(axes.flat, specs, strict=True):
        data = full[full["campaign_axis"].str.contains(axis, regex=False)]
        grouped = data.groupby(xkey)
        x = np.asarray(sorted(grouped.groups), dtype=float)
        recovery = grouped["conditional_recovery_mean_ms"].mean().loc[x]
        totals = grouped[["restored_tokens", "recomputed_tokens"]].sum()
        denominator = totals["restored_tokens"] + totals["recomputed_tokens"]
        hit = totals["restored_tokens"].div(denominator.where(denominator > 0)).loc[x]
        line1 = ax.plot(
            x,
            recovery,
            marker="o",
            color="#111111",
            label="K64 recovery",
        )[0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel("K64 recovery (ms)", color=line1.get_color())
        twin = ax.twinx()
        line2 = twin.plot(
            x,
            hit,
            marker="s",
            linestyle="--",
            color="#737373",
            label="Restore hit",
        )[0]
        twin.set_ylim(0, 1)
        twin.set_ylabel("Restore hit", color=line2.get_color())
        ax.grid(axis="y", alpha=0.2)
    return save(fig, output_dir, "fig03_resource_scalability")


def plot_ablation(
    probes: pd.DataFrame, factorial: pd.DataFrame, output_dir: Path
) -> list[Path]:
    data = probes[
        (probes["model_id"] == "qwen3_8b")
        & (probes["active_workflow_concurrency"] == 128)
        & np.isclose(probes["failure_rate_per_1000_worker_hours"], 0.0)
        & (probes["total_gpu_count"] == 64)
        & np.isclose(probes["prompt_scale"], 0.016)
        & np.isclose(probes["network_effective_gb_s"], 10.0)
        & np.isclose(probes["hbm_quota_gib_per_target"], 4.0)
        & np.isclose(probes["dram_quota_gib_per_target"], 64.0)
        & np.isclose(probes["algorithm1_prediction_error_multiplier"], 1.0)
    ].copy()
    algorithm = data[data["policy"].str.startswith("aegiskv_alg")].copy()
    algorithm["bits"] = algorithm["policy"].str.extract(r"aegiskv_alg([01]{3})")[0]
    algorithm["central_rate_burden_ms_per_workflow_epoch"] = (
        5.952
        / 1000.0
        * algorithm["eligible_exposure_gpu_shard_hours"]
        * algorithm["conditional_actual_harm_mean_ms"]
        / 393.0
    )
    means = algorithm.groupby("bits")[
        "central_rate_burden_ms_per_workflow_epoch"
    ].mean().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    axes[0].bar(range(len(means)), means, color="#737373")
    axes[0].set_xticks(range(len(means)), means.index, rotation=45)
    axes[0].set_xlabel("Algorithms 1/2/3 enabled")
    axes[0].set_ylabel("Normal-state burden\n(ms/workflow/epoch)")
    axes[0].set_title("Rate-zero K64 structural 2³ closure")
    effects = factorial[
        factorial["metric"]
        == "single_fault_normal_state_burden_ms_per_workflow_epoch_at_central_rate"
    ]
    axes[1].errorbar(
        effects["effect_mean"],
        np.arange(len(effects)),
        xerr=np.vstack(
            [
                effects["effect_mean"] - effects["effect_ci95_low"],
                effects["effect_ci95_high"] - effects["effect_mean"],
            ]
        ),
        fmt="o",
        color="#111111",
        capsize=2,
    )
    axes[1].axvline(0, color="#777777", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(effects)), effects["term"])
    axes[1].set_xlabel("Factorial effect (ms/workflow/epoch; lower is better)")
    axes[1].set_title("Hierarchical outer-seed contrasts")
    return save(fig, output_dir, "fig04_ablation_factorial")


def plot_timeline(timeline: pd.DataFrame, output_dir: Path) -> list[Path]:
    data = timeline[
        (timeline["policy"] == "aegiskv_alg111")
        & (timeline["seed"] == timeline["seed"].min())
    ].sort_values("time_ms")
    time = (data["time_ms"] - data["time_ms"].min()) / 60_000
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
    axes[0].plot(
        time,
        data["effective_alpha_mean"],
        label="Effective alpha",
        color="#111111",
        linestyle="-",
    )
    axes[0].plot(
        time,
        data["pressure_fraction"],
        label="Pressure",
        color="#737373",
        linestyle="--",
    )
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Fraction")
    axes[0].legend(frameon=False, ncol=2)
    hbm = data["hbm_used_bytes"] / data["hbm_capacity_bytes"]
    dram = data["dram_used_bytes"] / data["dram_capacity_bytes"]
    axes[1].plot(time, hbm, label="Peer-HBM", color="#111111", linestyle="-")
    axes[1].plot(time, dram, label="DRAM", color="#969696", linestyle=":")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Quota occupancy")
    axes[1].set_xlabel("Offered steady interval (min)")
    axes[1].legend(frameon=False, ncol=2)
    return save(fig, output_dir, "fig05_alpha_occupancy_timeline")


def plot_reuse(
    frame: pd.DataFrame, reuse: pd.DataFrame, output_dir: Path
) -> list[Path]:
    zero = reuse[
        (reuse["policy"] == "aegiskv_alg111")
        & reuse["scenario_id"].isin(
            frame[
                (frame["policy"] == "aegiskv_alg111")
                & (frame["model_id"] == "qwen3_8b")
                & (frame["active_workflow_concurrency"] == 128)
                & np.isclose(frame["failure_rate_per_1000_worker_hours"], 0.0)
                & (frame["total_gpu_count"] == 64)
                & np.isclose(frame["prompt_scale"], 0.016)
                & np.isclose(frame["network_effective_gb_s"], 10.0)
                & np.isclose(frame["hbm_quota_gib_per_target"], 4.0)
                & np.isclose(frame["dram_quota_gib_per_target"], 64.0)
                & np.isclose(frame["algorithm1_prediction_error_multiplier"], 1.0)
            ]["scenario_id"]
        )
    ]
    bins = ["none", "partial", "high", "near_complete"]
    pooled = zero.groupby("input_reuse_bin").sum(numeric_only=True).reindex(bins)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    request = pooled["request_count_full_workload"]
    prompt = pooled["prompt_tokens_full_workload"]
    axes[0].bar(np.arange(4) - 0.17, request / request.sum(), 0.34, color="#4D4D4D", label="Requests")
    axes[0].bar(np.arange(4) + 0.17, prompt / prompt.sum(), 0.34, color="#BDBDBD", label="Prompt tokens")
    axes[0].set_xticks(np.arange(4), bins, rotation=25, ha="right")
    axes[0].set_ylabel("Workload fraction")
    axes[0].set_title("All-request Weka reuse")
    axes[0].legend(frameon=False)
    residency_columns = [
        "intended_residency_hbm_only_chunks",
        "intended_residency_dram_only_chunks",
        "intended_residency_hbm_and_dram_chunks",
        "intended_residency_bypass_chunks",
    ]
    residency_denominator = pooled[residency_columns].sum(axis=1)
    residency_na = residency_denominator <= 0
    residency = pooled[residency_columns].div(
        residency_denominator.replace(0, np.nan), axis=0
    )
    bottom = np.zeros(4)
    for column, label, color in zip(
        residency_columns,
        ("HBM", "DRAM", "Both", "Bypass"),
        ("#111111", "#737373", "#BDBDBD", "#EEEEEE"),
        strict=True,
    ):
        values = residency[column].fillna(0).to_numpy()
        axes[1].bar(np.arange(4), values, bottom=bottom, color=color, label=label)
        bottom += values
    for index in np.flatnonzero(residency_na.to_numpy()):
        axes[1].text(index, 0.5, "NA", ha="center", va="center", rotation=90)
    axes[1].set_xticks(np.arange(4), bins, rotation=25, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Intended chunk residency")
    axes[1].set_title("Admission by reuse bin")
    axes[1].legend(frameon=False, ncol=2)
    restored = pooled["probe_restored_tokens"]
    denominator = restored + pooled["probe_recomputed_tokens"]
    restore = restored.div(denominator.where(denominator > 0))
    axes[2].bar(np.arange(4), restore, color="#737373")
    for index in np.flatnonzero(restore.isna().to_numpy()):
        axes[2].text(index, 0.5, "NA", ha="center", va="center", rotation=90)
    axes[2].set_xticks(np.arange(4), bins, rotation=25, ha="right")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("K64 restored-token fraction")
    axes[2].set_title("Descriptive; strata not IID")
    fig.text(
        0.5,
        -0.04,
        "NA denotes no intended chunks (middle) or no probe recovery tokens (right).",
        ha="center",
        va="top",
    )
    return save(fig, output_dir, "fig06_reuse_and_recovery")


def plot_model_alg1(probes: pd.DataFrame, output_dir: Path) -> list[Path]:
    full = probes[probes["policy"] == "aegiskv_alg111"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7))
    fig.suptitle(
        "Rate-local live-state K64; may include prior uptime history",
        fontsize=10,
    )
    context = full[full["campaign_axis"].str.contains("context_scale", regex=False)]
    grouped = context.groupby("prompt_scale")
    x = np.asarray(sorted(grouped.groups), dtype=float)
    axes[0].plot(
        x,
        grouped["conditional_actual_harm_mean_ms"].mean().loc[x],
        marker="o",
        label="Failure harm",
        color="#111111",
    )
    axes[0].set_xlabel("Prompt scale vs raw Weka")
    axes[0].set_ylabel("K64 conditional harm (ms)")
    axes[0].set_title("Context robustness (conditioned)")
    model = full[
        full["campaign_axis"].str.contains("model_scale", regex=False)
        | ((full["model_id"] == "qwen3_8b") & (full["scenario_id"].isin(central(full)["scenario_id"])))
    ]
    model_means = model.groupby("model_id")[
        "conditional_actual_harm_mean_ms"
    ].mean()
    axes[1].bar(
        range(len(model_means)),
        model_means,
        color=["#111111", "#969696"][: len(model_means)],
    )
    axes[1].set_xticks(range(len(model_means)), [name.replace("qwen3_", "Qwen3-") for name in model_means.index])
    axes[1].set_ylabel("K64 conditional harm (ms)")
    axes[1].set_title("Model / PP robustness")
    alg1 = full[
        full["campaign_axis"].str.contains("algorithm1_error", regex=False)
        | full.index.isin(central(full).index)
    ]
    grouped = alg1.groupby("algorithm1_prediction_error_multiplier")
    x = np.asarray(sorted(grouped.groups), dtype=float)
    axes[2].plot(
        x,
        grouped["conditional_actual_harm_mean_ms"].mean().loc[x],
        marker="o",
        color="#4D4D4D",
    )
    axes[2].set_xlabel(
        "Alg. 1 residual multiplier\n(0=oracle, 1=cross-fit point prediction)"
    )
    axes[2].set_ylabel("K64 conditional harm (ms)")
    axes[2].set_title("Prediction error")
    return save(fig, output_dir, "fig07_context_alg1")


def render_all(
    *,
    frame: pd.DataFrame,
    probes: pd.DataFrame,
    burdens: pd.DataFrame,
    paired_probes: pd.DataFrame,
    reuse: pd.DataFrame,
    factorial: pd.DataFrame,
    timeline: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Single typed wiring point covered by the renderer contract test."""

    renderers: list[Callable[[], list[Path]]] = [
        lambda: plot_baselines(frame, probes, paired_probes, output_dir),
        lambda: plot_concurrency_failure(frame, burdens, output_dir),
        lambda: plot_resources(probes, output_dir),
        lambda: plot_ablation(probes, factorial, output_dir),
        lambda: plot_timeline(timeline, output_dir),
        lambda: plot_reuse(frame, reuse, output_dir),
        lambda: plot_model_alg1(probes, output_dir),
    ]
    return [path for renderer in renderers for path in renderer()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefigure-audit", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    manifest = verify_analysis(analysis_dir, allow_partial=args.allow_partial)
    prefigure_audit = verify_prefigure_audit(
        args.prefigure_audit.resolve(),
        analysis_dir=analysis_dir,
        analysis_manifest=manifest,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else analysis_dir / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    style()
    frame = pd.read_csv(analysis_dir / "summary_rows.csv")
    probes = pd.read_csv(analysis_dir / "active_source_probe_seed_rows.csv")
    burdens = pd.read_csv(analysis_dir / "normal_state_hazard_burden.csv")
    paired_probes = pd.read_csv(
        analysis_dir / "paired_probe_outer_seed_bootstrap.csv"
    )
    reuse = pd.read_csv(analysis_dir / "active_source_probe_reuse_bins.csv")
    factorial = pd.read_csv(analysis_dir / "factorial_effects.csv")
    timeline = pd.read_csv(analysis_dir / "central_steady_timelines.csv")
    paths = render_all(
        frame=frame,
        probes=probes,
        burdens=burdens,
        paired_probes=paired_probes,
        reuse=reuse,
        factorial=factorial,
        timeline=timeline,
        output_dir=output_dir,
    )
    figure_manifest = {
        "schema_version": 1,
        "evidence_class": "simulated_sglsim_only_figures",
        "physical_result": False,
        "analysis_status": manifest["status"],
        "selection_mode": manifest["selection_mode"],
        "paper_eligible": bool(manifest["paper_eligible"]),
        "analysis_manifest": {
            "path": str(analysis_dir / "analysis_manifest.json"),
            "sha256": sha256(analysis_dir / "analysis_manifest.json"),
            "canonical_sha256": manifest["canonical_sha256"],
        },
        "prefigure_full_audit": prefigure_audit,
        "renderer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "files": {
            path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(paths)
        },
    }
    canonical = json.dumps(figure_manifest, sort_keys=True, separators=(",", ":")).encode()
    figure_manifest["canonical_sha256"] = hashlib.sha256(canonical).hexdigest()
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(figure_manifest, indent=2))


if __name__ == "__main__":
    main()
