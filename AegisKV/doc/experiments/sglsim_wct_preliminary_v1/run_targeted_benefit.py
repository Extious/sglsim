#!/usr/bin/env python3
"""Run the locked targeted warm-failover AegisKV follow-up campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "targeted_benefit_protocol.json"
CORE_PATH = ROOT / "run_preliminary.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core() -> Any:
    spec = importlib.util.spec_from_file_location("aegis_targeted_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load preliminary DES core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scenario_id(bandwidth: float, repair_ms: float, concurrency: int) -> str:
    return f"dram{bandwidth:g}-repair{repair_ms:g}-c{concurrency}"


def policy_label(policy: str) -> str:
    return policy.lower().replace(" ", "_").replace("/", "_")


def runtime_protocol(parent: dict[str, Any], contract: dict[str, Any], *, bandwidth: float, repair_ms: float) -> dict[str, Any]:
    protocol = json.loads(json.dumps(parent))
    protocol["workload"]["center_context_tokens"] = int(contract["model"]["context_tokens"])
    protocol["failure_model"]["replacement_delay_ms"] = float(repair_ms)
    protocol["failure_model"]["probe_type_schedule"] = list(contract["failure_model"]["types"])
    protocol["hardware"]["bandwidth_gb_s"]["peer_hbm_same_node"] = float(
        contract["hardware_model"]["peer_hbm_bandwidth_gb_s"]
    )
    protocol["hardware"]["bandwidth_gb_s"]["dram_remote"] = float(bandwidth)
    protocol["runtime_overrides"] = {
        "topology": {"peer_hbm": "same_node", "dram": "remote"}
    }
    protocol["normal_path_copy_burden"]["visible_fraction_of_copy_critical_path"] = float(
        contract["controller"]["visible_copy_burden_fraction"]
    )
    protocol["normal_path_copy_burden"]["maximum_fraction_of_request_service"] = float(
        contract["controller"]["visible_copy_burden_cap_fraction_of_service"]
    )
    protocol["aegiskv"]["initial_alpha"] = float(contract["controller"]["initial_alpha"])
    protocol["aegiskv"]["maximum_alpha"] = float(contract["controller"]["maximum_alpha"])
    protocol["aegiskv"]["minimum_alpha"] = float(contract["controller"]["minimum_alpha"])
    return protocol


def paired_seed_metrics(native_path: Path, policy_path: Path) -> dict[str, float | int]:
    with np.load(native_path, allow_pickle=False) as native, np.load(policy_path, allow_pickle=False) as policy:
        if not np.array_equal(native["workflow_id"], policy["workflow_id"]):
            raise ValueError("paired workflow identities differ")
        for key in ("probe_time_ms", "probe_source_rank", "probe_failure_type"):
            if not np.array_equal(native[key], policy[key]):
                raise ValueError(f"paired probe field differs: {key}")
        native_base = native["base_wct_ms"]
        policy_base = policy["base_wct_ms"]
        native_fault = native["fault_wct_ms"]
        policy_fault = policy["fault_wct_ms"]
        affected = (
            (native_fault > native_base[None, :] + 1e-9)
            | (policy_fault > policy_base[None, :] + 1e-9)
        )
        native_values = native_fault[affected]
        policy_values = policy_fault[affected]
        if native_values.size == 0:
            raise ValueError("targeted cell has no affected workflows")
        native_p95 = float(np.quantile(native_values, 0.95))
        policy_p95 = float(np.quantile(policy_values, 0.95))
        return {
            "affected_pair_count": int(native_values.size),
            "native_affected_mean_wct_ms": float(np.mean(native_values)),
            "policy_affected_mean_wct_ms": float(np.mean(policy_values)),
            "mean_wct_reduction_pct": float((1.0 - np.mean(policy_values) / np.mean(native_values)) * 100.0),
            "p95_wct_reduction_pct": float((1.0 - policy_p95 / native_p95) * 100.0),
            "pairwise_win_fraction": float(np.mean(policy_values < native_values)),
            "healthy_mean_wct_overhead_pct": float((np.mean(policy_base) / np.mean(native_base) - 1.0) * 100.0),
        }


def aggregate_metrics(core: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario_id"], row["policy"])].append(row)
    aggregates = []
    for (scenario, policy), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["seed"]))
        output: dict[str, Any] = {
            "scenario_id": scenario,
            "policy": policy,
            "seed_count": len(values),
            "dram_bandwidth_gb_s": float(values[0]["dram_bandwidth_gb_s"]),
            "replacement_delay_ms": float(values[0]["replacement_delay_ms"]),
            "active_requests_per_gpu": int(values[0]["active_requests_per_gpu"]),
            "affected_pair_count": sum(int(row["affected_pair_count"]) for row in values),
        }
        for offset, field in enumerate((
            "mean_wct_reduction_pct",
            "p95_wct_reduction_pct",
            "pairwise_win_fraction",
            "healthy_mean_wct_overhead_pct",
        )):
            estimate, low, high = core.bootstrap_ci(
                [float(row[field]) for row in values],
                seed=2026082600 + offset * 997 + int(values[0]["active_requests_per_gpu"]),
            )
            output[field] = estimate
            output[f"{field}_ci95_low"] = low
            output[f"{field}_ci95_high"] = high
        aggregates.append(output)
    return aggregates


def plot_result(output_dir: Path, aggregate: list[dict[str, Any]], primary: dict[str, Any]) -> Path:
    plt.rcParams.update({
        "font.size": 9.2,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.75))
    colors = {5.0: "#0072B2", 10.0: "#E69F00", 16.0: "#009E73"}
    for metric, ax, title in (
        ("mean_wct_reduction_pct", axes[0], "Affected mean WCT"),
        ("p95_wct_reduction_pct", axes[1], "Affected p95 WCT"),
    ):
        for bandwidth in (5.0, 10.0, 16.0):
            selected = [
                row for row in aggregate
                if row["policy"] == "AegisKV"
                and float(row["replacement_delay_ms"]) == 500.0
                and float(row["dram_bandwidth_gb_s"]) == bandwidth
            ]
            selected.sort(key=lambda row: int(row["active_requests_per_gpu"]))
            x = np.asarray([int(row["active_requests_per_gpu"]) for row in selected])
            y = np.asarray([float(row[metric]) for row in selected])
            low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in selected])
            high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in selected])
            ax.errorbar(
                x, y, yerr=[y - low, high - y], marker="o", capsize=3,
                color=colors[bandwidth], linewidth=2.0, label=f"Remote DRAM {bandwidth:g} GB/s",
            )
        ax.axhline(0.0, color="black", lw=0.9, ls="--")
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("Active requests / GPU")
        ax.set_ylabel("WCT reduction vs modeled Native (%)")
        ax.set_title(title)
        ax.grid(alpha=0.18, linewidth=0.7)
    axes[0].legend(fontsize=7.3, loc="upper right")

    primary_rows = [
        row for row in aggregate
        if float(row["dram_bandwidth_gb_s"]) == float(primary["dram_bandwidth_gb_s"])
        and float(row["replacement_delay_ms"]) == float(primary["replacement_delay_ms"])
        and int(row["active_requests_per_gpu"]) == int(primary["active_requests_per_gpu"])
    ]
    order = ["Static FCFS", "DejaVu", "KevlarFlow", "GhostServe", "AegisKV"]
    lookup = {row["policy"]: row for row in primary_rows}
    selected = [lookup[policy] for policy in order]
    x = np.arange(len(order))
    width = 0.36
    axes[2].bar(
        x - width / 2, [row["mean_wct_reduction_pct"] for row in selected], width,
        color="#0072B2", label="Mean",
    )
    axes[2].bar(
        x + width / 2, [row["p95_wct_reduction_pct"] for row in selected], width,
        color="#CC79A7", label="p95",
    )
    axes[2].axhline(0.0, color="black", lw=0.9, ls="--")
    axes[2].set_xticks(x, ["Static", "DejaVu", "KevlarFlow", "GhostServe", "AegisKV"], rotation=24, ha="right")
    axes[2].set_ylabel("WCT reduction vs modeled Native (%)")
    axes[2].set_title("Primary cell architecture comparison")
    axes[2].legend(fontsize=7.5)
    axes[2].grid(axis="y", alpha=0.18, linewidth=0.7)
    fig.suptitle("Targeted warm-failover regime: AegisKV reduces affected-workflow WCT", fontsize=13.5, y=0.995)
    fig.subplots_adjust(top=0.80, bottom=0.25, wspace=0.34)
    fig.text(
        0.5, 0.025,
        "40K context, A100-40GB model, same-node Peer-HBM 50 GB/s, remote DRAM, 500 ms common takeover; paired 5-seed 95% CI",
        ha="center", fontsize=7.6, color="#555555",
    )
    base = output_dir / "figures" / "targeted_benefit_preliminary"
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "targeted_benefit_v1")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / contract["parent_protocol"]).read_text(encoding="utf-8"))
    core = load_core()

    scenarios = [
        (float(bandwidth), float(repair), int(concurrency))
        for bandwidth in contract["hardware_model"]["dram_bandwidth_gb_s"]
        for repair in contract["failure_model"]["replacement_delay_ms"]
        for concurrency in contract["workload"]["active_requests_per_gpu"]
    ]
    primary = contract["primary_cell"]
    cells: list[tuple[float, float, int, str, int]] = []
    for bandwidth, repair, concurrency in scenarios:
        for policy in contract["rectangular_matrix_policies"]:
            for seed in contract["paired_seeds"]:
                cells.append((bandwidth, repair, concurrency, str(policy), int(seed)))
    for policy in contract["primary_cell_additional_baselines"]:
        for seed in contract["paired_seeds"]:
            cells.append((
                float(primary["dram_bandwidth_gb_s"]),
                float(primary["replacement_delay_ms"]),
                int(primary["active_requests_per_gpu"]),
                str(policy),
                int(seed),
            ))

    summaries = []
    for index, (bandwidth, repair, concurrency, policy, seed) in enumerate(cells, start=1):
        scenario = scenario_id(bandwidth, repair, concurrency)
        raw = output_dir / "raw" / scenario / f"{policy_label(policy)}__s{seed}.npz"
        print(f"RUN {index}/{len(cells)} scenario={scenario} policy={policy!r} seed={seed}", flush=True)
        protocol = runtime_protocol(parent, contract, bandwidth=bandwidth, repair_ms=repair)
        cell = core.run_cell(
            protocol,
            policy=policy,
            seed=seed,
            workflows=int(contract["workload"]["workflows_per_cell"]),
            active_per_gpu=concurrency,
            probe_count=int(contract["failure_model"]["probe_count_per_cell"]),
            output=raw,
        )
        cell["scenario_id"] = scenario
        cell["dram_bandwidth_gb_s"] = bandwidth
        cell["replacement_delay_ms"] = repair
        summaries.append(cell)

    summary_by_key = {
        (row["scenario_id"], row["policy"], int(row["seed"])): row
        for row in summaries
    }
    metric_rows = []
    for row in summaries:
        if row["policy"] == "Native Mooncake":
            continue
        native = summary_by_key[(row["scenario_id"], "Native Mooncake", int(row["seed"]))]
        metric_rows.append({
            "scenario_id": row["scenario_id"],
            "policy": row["policy"],
            "seed": int(row["seed"]),
            "dram_bandwidth_gb_s": float(row["dram_bandwidth_gb_s"]),
            "replacement_delay_ms": float(row["replacement_delay_ms"]),
            "active_requests_per_gpu": int(row["active_requests_per_gpu"]),
            **paired_seed_metrics(Path(native["raw_npz"]), Path(row["raw_npz"])),
        })
    aggregate = aggregate_metrics(core, metric_rows)

    metric_path = output_dir / "paired_seed_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    aggregate_path = output_dir / "aggregate_ci.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)

    expected_scenarios = {scenario_id(*values) for values in scenarios}
    actual_scenarios = {str(row["scenario_id"]) for row in summaries}
    pair_signatures: dict[tuple[str, int], tuple[Any, ...]] = {}
    all_gpu = True
    for row in summaries:
        with np.load(row["raw_npz"], allow_pickle=False) as raw:
            signature = (
                tuple(raw["workflow_id"].tolist()),
                raw["probe_time_ms"].tobytes(),
                raw["probe_source_rank"].tobytes(),
                tuple(raw["probe_failure_type"].tolist()),
            )
            key = (str(row["scenario_id"]), int(row["seed"]))
            if key in pair_signatures and pair_signatures[key] != signature:
                pair_signatures[key] = ("MISMATCH",)
            else:
                pair_signatures[key] = signature
            all_gpu &= set(raw["probe_failure_type"].tolist()) == {"source_worker_gpu"}
    path_counts = Counter()
    for row in summaries:
        if row["policy"] == "AegisKV":
            path_counts.update(row["mechanism"]["path_event_counts"])
    checks = {
        "contract_locked_followup_disclosure": contract["status"] == "LOCKED_CONFIRMATORY_FOLLOWUP_AFTER_EXPLORATORY_DIAGNOSTIC",
        "design_semantics_exact": parent["design_semantics"] == contract["parent_design_semantics"],
        "complete_rectangular_matrix": actual_scenarios == expected_scenarios,
        "exact_cell_count": len(summaries) == len(cells) == 200,
        "unique_cells": len(summary_by_key) == len(summaries),
        "five_paired_seeds": all(
            {int(row["seed"]) for row in summaries if row["scenario_id"] == scenario and row["policy"] in contract["rectangular_matrix_policies"]}
            == set(contract["paired_seeds"])
            for scenario in expected_scenarios
        ),
        "raw_hashes_match": all(sha256(Path(row["raw_npz"])) == row["raw_npz_sha256"] for row in summaries),
        "paired_workflow_and_probe_tuple": all(signature != ("MISMATCH",) for signature in pair_signatures.values()),
        "all_probes_source_worker_gpu": all_gpu,
        "source_owned_hbm_and_dram_paths_exercised": path_counts["copy_peer_hbm_same_node"] > 0 and path_counts["copy_dram_remote"] > 0,
        "zero_target_owned_or_hbm_to_dram": all(
            row["mechanism"]["target_owned_copy_events"] == 0
            and row["mechanism"]["hbm_to_dram_copy_events"] == 0
            for row in summaries
        ),
        "no_performance_threshold_for_audit_pass": contract["integrity_gates"]["no_performance_threshold_for_audit_pass"] is True,
    }
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selection": "TARGETED_WARM_FAILOVER_BENEFIT_V1",
        "paper_ready": False,
        "targeted_preliminary": True,
        "physical_result": False,
        "post_exploratory_selection_disclosed": True,
        "checks": checks,
        "cell_count": len(summaries),
        "scenario_count": len(expected_scenarios),
        "contract_sha256": sha256(PROTOCOL_PATH),
        "parent_protocol_sha256": sha256(ROOT / contract["parent_protocol"]),
        "runner_sha256": sha256(Path(__file__)),
        "core_runner_sha256": sha256(CORE_PATH),
        "paired_seed_metrics_sha256": sha256(metric_path),
        "aggregate_ci_sha256": sha256(aggregate_path),
        "path_event_counts": dict(sorted(path_counts.items())),
        "evidence_boundary": contract["evidence_boundary"],
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "status": audit["status"],
        "selection": audit["selection"],
        "contract": str(PROTOCOL_PATH),
        "cells": summaries,
        "paired_seed_metrics": metric_rows,
        "aggregate": aggregate,
        "evidence_boundary": contract["evidence_boundary"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(f"targeted audit failed: {[name for name, value in checks.items() if not value]}")

    figure = plot_result(output_dir, aggregate, primary)
    manifest = {
        "status": "PASS",
        "audit": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "figure_pdf": str(figure.with_suffix(".pdf")),
        "figure_pdf_sha256": sha256(figure.with_suffix(".pdf")),
        "figure_png": str(figure.with_suffix(".png")),
        "figure_png_sha256": sha256(figure.with_suffix(".png")),
        "evidence_boundary": contract["evidence_boundary"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
