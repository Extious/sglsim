#!/usr/bin/env python3
"""Run auditable S3/S6/S7 fixed-arrival extension cells."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import run_preliminary as core


ROOT = Path(__file__).resolve().parent


def raw_path(output: Path, cell: dict[str, Any]) -> Path:
    value = Path(cell["raw_npz"])
    return value if value.is_absolute() else output / value


def cell_protocol(
    parent: dict[str, Any], *, dp: int, context: int, reuse: float = 0.0,
    topology: dict[str, str] | None = None, repair_ms: float | None = None,
    occupancy: bool = False,
) -> dict[str, Any]:
    protocol = copy.deepcopy(parent)
    protocol["models"]["qwen3_8b"]["center_system_dp"] = int(dp)
    protocol["workload"]["center_context_tokens"] = int(context)
    protocol["workload"]["reuse_fraction"] = float(reuse)
    protocol["runtime_overrides"] = {"record_occupancy_trace": bool(occupancy)}
    if topology is not None:
        protocol["runtime_overrides"]["topology"] = {
            "peer_hbm": topology["peer_hbm"], "dram": topology["dram"],
        }
    if repair_ms is not None:
        protocol["failure_model"]["replacement_delay_ms"] = float(repair_ms)
    return protocol


def run_specs(selection: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if selection == "s3":
        s3 = contract["s3"]
        unique: dict[tuple[int, int, str], dict[str, Any]] = {}
        for dp in s3["scale_system_dp"]:
            for policy in s3["policies"]:
                key = (int(dp), int(s3["scale_context_tokens"]), policy)
                unique[key] = {
                    "axes": [{"axis": "scale", "value": int(dp)}],
                    "dp": int(dp), "context": int(s3["scale_context_tokens"]),
                    "policy": policy,
                }
        for context in s3["context_tokens"]:
            for policy in s3["policies"]:
                key = (int(s3["context_system_dp"]), int(context), policy)
                if key in unique:
                    unique[key]["axes"].append({"axis": "context", "value": int(context)})
                else:
                    unique[key] = {
                        "axes": [{"axis": "context", "value": int(context)}],
                        "dp": int(s3["context_system_dp"]), "context": int(context),
                        "policy": policy,
                    }
        specs.extend(unique.values())
    elif selection == "s5":
        s5 = contract["s5"]
        specs.append({
            "axes": [{"axis": "timeline", "value": "center"}],
            "dp": int(s5["system_dp"]), "context": int(s5["context_tokens"]),
            "reuse": float(s5["reuse_fraction"]), "policy": s5["policy"],
            "occupancy": True,
        })
    elif selection == "s6":
        s6 = contract["s6"]
        for reuse in s6["reuse_fractions"]:
            specs.append({
                "axes": [{"axis": "reuse", "value": float(reuse)}],
                "dp": int(s6["system_dp"]), "context": int(s6["context_tokens"]),
                "reuse": float(reuse), "policy": s6["policy"], "occupancy": True,
            })
    elif selection == "s7":
        s7 = contract["s7"]
        for topology in s7["topologies"]:
            for policy in s7["policies"]:
                specs.append({
                    "axes": [{"axis": "topology", "value": topology["name"]}],
                    "dp": int(s7["system_dp"]), "context": int(s7["context_tokens"]),
                    "topology": topology, "repair_ms": 60000.0, "policy": policy,
                })
    elif selection == "s7_target":
        s7 = contract["s7_target"]
        for bandwidth in s7["fabric_bandwidth_gb_s"]:
            for policy in s7["policies"]:
                specs.append({
                    "axes": [{"axis": "bandwidth", "value": float(bandwidth)}],
                    "dp": int(s7["system_dp"]), "context": int(s7["context_tokens"]),
                    "topology": s7["topology"], "repair_ms": float(s7["reference_repair_delay_ms"]),
                    "bandwidth_gb_s": float(bandwidth), "policy": policy,
                })
        for repair in s7["repair_delay_ms"]:
            for policy in s7["policies"]:
                specs.append({
                    "axes": [{"axis": "repair", "value": float(repair)}],
                    "dp": int(s7["system_dp"]), "context": int(s7["context_tokens"]),
                    "topology": s7["topology"], "repair_ms": float(repair),
                    "bandwidth_gb_s": float(s7["reference_bandwidth_gb_s"]), "policy": policy,
                })
    else:
        raise ValueError(selection)
    return specs


def aggregate_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["axis"], str(row["value"]), row["policy"], float(row["failure_rate"]))].append(row)
    output = []
    for (axis, value, policy, rate), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        native_key = (axis, value, "Native Mooncake", rate)
        native = {int(row["seed"]): float(row["mean_wct_ms"]) for row in grouped.get(native_key, [])}
        wct = [float(row["mean_wct_ms"]) for row in rows]
        ratios = [float(row["mean_wct_ms"]) / native[int(row["seed"])] for row in rows] if native else [1.0] * len(rows)
        seed_key = sum(ord(char) for char in f"{axis}:{value}:{policy}:{rate}")
        mean, low, high = core.bootstrap_ci(wct, seed=20260826 ^ seed_key)
        rmean, rlow, rhigh = core.bootstrap_ci(ratios, seed=20260827 ^ seed_key)
        output.append({
            "axis": axis, "value": value, "policy": policy, "failure_rate": rate,
            "seed_count": len(rows), "mean_wct_ms": mean, "ci95_low_ms": low,
            "ci95_high_ms": high, "normalized_to_native": rmean,
            "normalized_ci95_low": rlow, "normalized_ci95_high": rhigh,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", choices=("s3", "s5", "s6", "s7", "s7_target"), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    contract_path = ROOT / "extension_protocol.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    parent = core.load_protocol()
    output = args.output_dir or ROOT / "results" / f"{args.selection}_preliminary_v1"
    output.mkdir(parents=True, exist_ok=True)
    specs = run_specs(args.selection, contract)
    cells = []
    for index, spec in enumerate(specs):
        for seed in contract["paired_seeds"]:
            topology = spec.get("topology")
            protocol = cell_protocol(
                parent, dp=spec["dp"], context=spec["context"],
                reuse=float(spec.get("reuse", 0.0)), topology=topology,
                repair_ms=spec.get("repair_ms"), occupancy=bool(spec.get("occupancy")),
            )
            if "bandwidth_gb_s" in spec:
                value = float(spec["bandwidth_gb_s"])
                protocol["hardware"]["bandwidth_gb_s"] = {
                    "peer_hbm_same_node": value,
                    "peer_hbm_cross_node": value,
                    "dram_local": value,
                    "dram_remote": value,
                }
            slug = f"cell{index:02d}__{spec['policy'].lower().replace(' ', '_')}__s{seed}"
            print(
                f"RUN-{args.selection.upper()} {index + 1}/{len(specs)} policy={spec['policy']!r} "
                f"dp={spec['dp']} ctx={spec['context']} seed={seed}", flush=True,
            )
            raw = output / "raw" / f"{slug}.npz"
            cell = core.run_cell(
                protocol, policy=spec["policy"], seed=int(seed), workflows=None,
                active_per_gpu=int(contract["active_requests_per_gpu"]),
                probe_count=int(contract["fault_probes_per_seed"]),
                output=raw,
                arrival_cutoff_ms=float(contract["arrival_cutoff_ms"]),
            )
            cell["raw_npz"] = str(Path("raw") / raw.name)
            cell["extension_spec"] = spec
            cells.append(cell)

    payload = {
        "schema_version": 1, "status": "PRELIMINARY_COMPLETE",
        "selection": args.selection.upper(), "workflow_completion_time_result": True,
        "contract": {"path": str(contract_path), "sha256": core.sha256(contract_path)},
        "parent_protocol": {"path": str(ROOT / "protocol.json"), "sha256": core.sha256(ROOT / "protocol.json")},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": core.sha256(Path(__file__).resolve())},
        "core_runner": {"path": str(ROOT / "run_preliminary.py"), "sha256": core.sha256(ROOT / "run_preliminary.py")},
        "cells": cells,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    seed_rows: list[dict[str, Any]] = []
    for cell in cells:
        spec = cell["extension_spec"]
        for axis in spec["axes"]:
            for rate in cell["rate_rows"]:
                seed_rows.append({
                    "axis": axis["axis"], "value": axis["value"], "policy": cell["policy"],
                    "seed": cell["seed"], "failure_rate": rate["failure_rate_per_1000_gpu_shard_hour"],
                    "mean_wct_ms": rate["hazard_integrated_mean_wct_ms"],
                    "workflow_count": cell["workflow_count"], "system_dp": spec["dp"],
                    "context_tokens": spec["context"], "reuse_fraction": spec.get("reuse", 0.0),
                    "arrival_horizon_ms": cell["workload"]["arrival_horizon_ms"],
                    "drain_horizon_ms": cell["workload"]["drain_horizon_ms"],
                    "raw_npz": cell["raw_npz"],
                })
    seed_path = output / "summary.csv"
    with seed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0]))
        writer.writeheader(); writer.writerows(seed_rows)
    aggregate = aggregate_rows(seed_rows)
    aggregate_path = output / "aggregate_ci.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)

    expected_control = [float(value) for value in contract["control_opportunity_times_ms"]]
    checks: dict[str, bool] = {
        "three_paired_seeds": all(
            len([cell for cell in cells if cell["extension_spec"] == spec]) == 3
            for spec in specs
        ),
        "fixed_arrival_horizon_exact": all(
            float(cell["workload"]["arrival_horizon_ms"]) == float(contract["arrival_cutoff_ms"])
            for cell in cells
        ),
        "exact_common_control_opportunities": all(
            cell["mechanism"]["control_opportunity_times_ms"] == expected_control
            for cell in cells if cell["policy"] == "AegisKV"
        ),
        "raw_hashes_and_wct": True,
        "paired_fault_tuples": True,
        "design_semantics_exact": parent["design_semantics"] == core.DESIGN_SEMANTICS,
        "a100_40gib_exact": float(parent["hardware"]["gpu_hbm_gib"]) == 40.0,
        "failure_rates_exact": parent["failure_model"]["rates_per_1000_gpu_shard_hour"] == [0.0, 5.952, 20.833],
        "no_target_owned_or_hbm_to_dram": all(
            cell["mechanism"]["target_owned_copy_events"] == 0
            and cell["mechanism"]["hbm_to_dram_copy_events"] == 0 for cell in cells
        ),
    }
    signatures: dict[tuple[int, int], tuple[bytes, bytes, tuple[str, ...]]] = {}
    for cell in cells:
        resolved_raw = raw_path(output, cell)
        if core.sha256(resolved_raw) != cell["raw_npz_sha256"]:
            checks["raw_hashes_and_wct"] = False
        with np.load(resolved_raw, allow_pickle=False) as raw:
            if np.any(raw["base_wct_ms"] <= 0) or np.any(raw["fault_wct_ms"] <= 0):
                checks["raw_hashes_and_wct"] = False
            signature = (
                raw["probe_time_ms"].tobytes(), raw["probe_source_rank"].tobytes(),
                tuple(raw["probe_failure_type"].tolist()),
            )
            key = (int(cell["extension_spec"]["dp"]), int(cell["seed"]))
            prior = signatures.get(key)
            if prior is not None and prior != signature:
                checks["paired_fault_tuples"] = False
            signatures[key] = signature
    if args.selection == "s3":
        checks["s3_scale_values_exact"] = sorted(contract["s3"]["scale_system_dp"]) == [16, 32, 64, 128, 256]
        checks["s3_context_values_exact"] = sorted(contract["s3"]["context_tokens"]) == [4096, 8192, 16384, 32768, 40960]
    elif args.selection == "s5":
        checks["control_step_timeline_persisted"] = all(
            cell["mechanism"]["control_opportunity_times_ms"] == expected_control
            and len(cell["mechanism"]["control_step_trace"]) == 3
            for cell in cells
        )
        checks["timeline_raw_arrays_saved"] = all(
            all(
                field in np.load(raw_path(output, cell), allow_pickle=False).files
                for field in (
                    "control_time_ms", "control_alpha_after_mean",
                    "control_pressure_fraction", "control_active_requests_per_gpu",
                    "control_hbm_used_bytes_total", "control_dram_used_bytes_total",
                    "occupancy_time_ms", "occupancy_hbm_total_bytes", "occupancy_dram_total_bytes",
                )
            )
            for cell in cells
        )
    elif args.selection == "s6":
        checks["capacity_integrals_saved_not_inferred"] = all(
            cell["mechanism"]["capacity_integral_byte_ms"]["HBM"] >= 0
            and cell["mechanism"]["capacity_integral_byte_ms"]["DRAM"] >= 0
            and cell["mechanism"]["occupancy_observation_horizon_ms"] == float(contract["arrival_cutoff_ms"])
            for cell in cells
        )
        checks["occupancy_trace_saved"] = True
        checks["capacity_integral_matches_raw_trace"] = True
        for cell in cells:
            with np.load(raw_path(output, cell), allow_pickle=False) as raw:
                times = raw["occupancy_time_ms"]
                checks["occupancy_trace_saved"] &= (
                    times.size > 1
                    and bool(np.all(np.diff(times) >= 0.0))
                    and math.isclose(float(times[-1]), float(contract["arrival_cutoff_ms"]), abs_tol=1e-9)
                )
                for tier, field in (
                    ("HBM", "occupancy_hbm_total_bytes"),
                    ("DRAM", "occupancy_dram_total_bytes"),
                ):
                    reconstructed = float(np.sum(raw[field][:-1] * np.diff(times)))
                    recorded = float(cell["mechanism"]["capacity_integral_byte_ms"][tier])
                    checks["capacity_integral_matches_raw_trace"] &= math.isclose(
                        reconstructed, recorded, rel_tol=1e-10, abs_tol=1.0
                    )
        checks["reuse_values_exact"] = sorted(contract["s6"]["reuse_fractions"]) == [0.0, 0.25, 0.5, 0.75]
        checks["copy_eviction_recovery_metrics_present"] = all(
            cell["mechanism"]["copy_amplification"] >= 0.0
            and cell["mechanism"]["eviction_count_total"] >= 0
            and cell["mechanism"]["recovery_hit_fraction"] is not None
            for cell in cells
        )
    elif args.selection == "s7":
        checks["topology_and_repair_values_exact"] = (
            {row["name"] for row in contract["s7"]["topologies"]}
            == {"same+local", "same+remote", "cross+local", "cross+remote"}
            and contract["s7"]["repair_delay_ms"] == [30000.0, 60000.0]
        )
        bandwidths = parent["hardware"]["bandwidth_gb_s"]
        checks["independent_topology_bandwidth_parameters"] = (
            set(bandwidths) == {
                "peer_hbm_same_node", "peer_hbm_cross_node",
                "dram_local", "dram_remote",
            }
            and float(bandwidths["peer_hbm_same_node"]) != float(bandwidths["peer_hbm_cross_node"])
            and float(bandwidths["dram_local"]) != float(bandwidths["dram_remote"])
        )
        checks["forced_topology_paths_exercised"] = True
        for cell in cells:
            spec = cell["extension_spec"]
            if not any(axis["axis"] == "topology" for axis in spec["axes"]):
                continue
            counts = cell["mechanism"]["path_event_counts"]
            dram_key = f"copy_dram_{spec['topology']['dram']}"
            checks["forced_topology_paths_exercised"] &= int(counts.get(dram_key, 0)) > 0
            if cell["policy"] == "AegisKV":
                peer_key = f"copy_peer_hbm_{spec['topology']['peer_hbm']}"
                checks["forced_topology_paths_exercised"] &= int(counts.get(peer_key, 0)) > 0
        repair_cells = [
            cell for cell in cells
            if any(axis["axis"] == "repair" for axis in cell["extension_spec"]["axes"])
        ]
        checks["repair_delay_changes_executed_fault_wct"] = all(
            all(
                set(rows := {
                    float(cell["extension_spec"]["repair_ms"]): float(cell["fault_conditioned_wct_mean_ms"])
                    for cell in repair_cells
                    if cell["policy"] == policy and int(cell["seed"]) == int(seed)
                }) == {30000.0, 60000.0}
                and rows[60000.0] > rows[30000.0]
                for seed in contract["paired_seeds"]
            )
            for policy in contract["s7"]["policies"]
        )
    elif args.selection == "s7_target":
        target = contract["s7_target"]
        checks["target_bandwidth_values_exact"] = target["fabric_bandwidth_gb_s"] == [1.0, 5.0, 10.0, 25.0, 50.0]
        checks["target_repair_values_exact"] = target["repair_delay_ms"] == [10000.0, 30000.0, 60000.0, 120000.0]
        checks["target_axes_paired"] = all(
            len([
                cell for cell in cells
                if cell["policy"] == policy and any(
                    axis["axis"] == axis_name and float(axis["value"]) == float(value)
                    for axis in cell["extension_spec"]["axes"]
                )
            ]) == 3
            for axis_name, values in (
                ("bandwidth", target["fabric_bandwidth_gb_s"]),
                ("repair", target["repair_delay_ms"]),
            )
            for value in values for policy in target["policies"]
        )
    audit = {
        "schema_version": 1, "selection": args.selection.upper(),
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "workflow_completion_time_result": True, "paper_ready": False,
        "evidence_boundary": contract["evidence_boundary"],
        "summary_sha256": core.sha256(summary_path), "seed_summary_sha256": core.sha256(seed_path),
        "aggregate_ci_sha256": core.sha256(aggregate_path),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError([name for name, passed in checks.items() if not passed])
    print(output / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
