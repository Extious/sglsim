"""Fail-closed lineage audit with evidence-class gates."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_lineage(lineage_dir: Path) -> dict[str, Any]:
    lineage_dir = Path(lineage_dir)
    manifest = json.loads((lineage_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = _read_csv(lineage_dir / "summary.csv")
    timeline = _read_csv(lineage_dir / "timeline.csv")
    failures: list[str] = []
    warnings: list[str] = []
    mechanism_checks: dict[str, Any] = {}
    expected = int(manifest["expected_cell_count"])
    snapshot_root = lineage_dir / "config_snapshot"
    config_hashes = manifest.get("config_sha256") or {}
    for required in (
        "suite.json",
        "source_registry.json",
        "parameter_provenance.json",
        "experiment_config_rationale.md",
        "reviewer_configuration_audit.md",
        "physical_calibration_status.md",
        "current_design.md",
    ):
        if required not in config_hashes:
            failures.append(f"missing_config_hash:{required}")
    for relative, expected_sha256 in sorted(config_hashes.items()):
        snapshot = snapshot_root / relative
        if not snapshot.is_file():
            failures.append(f"missing_config_snapshot:{relative}")
        elif _sha256_file(snapshot) != expected_sha256:
            failures.append(f"config_snapshot_sha256:{relative}")
    code_hashes = manifest.get("code_sha256") or {}
    if not code_hashes:
        failures.append("missing_code_snapshot_hashes")
    for relative, expected_sha256 in sorted(code_hashes.items()):
        snapshot = lineage_dir / "code_snapshot" / relative
        if not snapshot.is_file():
            failures.append(f"missing_code_snapshot:{relative}")
        elif _sha256_file(snapshot) != expected_sha256:
            failures.append(f"code_snapshot_sha256:{relative}")
    if len(summary) != expected:
        failures.append(f"cell_count:{len(summary)}!={expected}")
    if len(list((lineage_dir / "raw" / "cells").glob("*.json.gz"))) != expected:
        failures.append("raw_cell_artifact_count_mismatch")

    default_deployment = json.loads(
        (snapshot_root / "common" / "deployment.json").read_text(
            encoding="utf-8"
        )
    )
    default_protection = default_deployment.get("protection") or {}
    if int(default_protection.get("prefix_chunk_tokens") or 0) != 4096:
        failures.append("current_design_default_prefix_chunk_not_4096")
    if int(default_protection.get("decode_chunk_tokens") or 0) != 256:
        failures.append("current_design_default_decode_chunk_not_256")
    chunking_by_group: dict[str, dict[str, int]] = {}
    for group in manifest["groups"]:
        resolved = json.loads(
            (lineage_dir / "resolved_configs" / group / "deployment.json").read_text(
                encoding="utf-8"
            )
        )
        protection = resolved.get("protection") or {}
        prefix_tokens = int(protection.get("prefix_chunk_tokens") or 0)
        decode_tokens = int(protection.get("decode_chunk_tokens") or 0)
        expected_prefix = 1024 if group == "S9" else 4096
        chunking_by_group[group] = {
            "prefix_chunk_tokens": prefix_tokens,
            "decode_chunk_tokens": decode_tokens,
        }
        if prefix_tokens != expected_prefix or decode_tokens != 256:
            failures.append(
                f"current_design_chunking:{group}:{prefix_tokens}/{decode_tokens}"
            )
    mechanism_checks["current_design_chunking"] = chunking_by_group

    raw_by_cell: dict[str, dict[str, Any]] = {}
    failure_pairing: dict[tuple[tuple[str, str], ...], set[str]] = defaultdict(set)
    for row in summary:
        cell_id = row.get("cell_id", "")
        raw_relative = row.get("raw_artifact", "")
        raw_path = lineage_dir / raw_relative
        if not raw_relative or not raw_path.is_file():
            failures.append(f"missing_raw_cell:{cell_id}")
            continue
        raw = _read_gzip_json(raw_path)
        raw_by_cell[cell_id] = raw
        if raw.get("cell_id") != cell_id:
            failures.append(f"raw_cell_identity:{cell_id}")
        result = raw.get("result") or {}
        result_requests = result.get("requests") or {}
        incomplete = sorted(
            request_id
            for request_id, request in result_requests.items()
            if request.get("status") != "COMPLETED" or request.get("output_fenced")
        )
        if incomplete:
            failures.append(f"incomplete_or_fenced_requests:{cell_id}:{len(incomplete)}")
        workload_metadata = raw.get("workload_metadata") or {}
        context_cap = int(workload_metadata.get("context_cap_tokens") or 0)
        maximum_total = int(
            workload_metadata.get("maximum_projected_total_tokens") or 0
        )
        if context_cap <= 0 or maximum_total > context_cap:
            failures.append(
                f"projected_context_limit:{cell_id}:{maximum_total}>{context_cap}"
            )
        failure_metadata = raw.get("failure_metadata") or {}
        failed_workers = failure_metadata.get("failed_workers") or []
        worker_failed_events = int((result.get("event_counts") or {}).get("worker_failed", 0))
        if len(failed_workers) != worker_failed_events:
            failures.append(
                f"scheduled_failure_count:{cell_id}:{len(failed_workers)}!={worker_failed_events}"
            )
        axes = raw.get("axes") or {}
        failure_pair_key = tuple(
            sorted(
                [("group", str(row.get("group", "")))]
                + [
                (str(name), str(value))
                for name, value in axes.items()
                if name
                not in {
                    "policy",
                    "effective_bandwidth_GB_per_s",
                    "host_dram_gib_per_node",
                }
                ]
            )
        )
        failure_pairing[failure_pair_key].add(
            json.dumps(failure_metadata, sort_keys=True, separators=(",", ":"))
        )

        if row.get("metadata_only") != "True" or row.get("physical_result") != "False":
            failures.append(f"evidence_label:{cell_id}")
        scenario = row.get("failure_scenario")
        if scenario and scenario != "no_failure" and row.get("complete_failure_path_exercised") != "True":
            failures.append(f"complete_path_not_exercised:{cell_id}")
        if float(row.get("analytical_recovery_max_abs_error_ms") or 0.0) > 1e-5:
            failures.append(f"analytical_verification:{cell_id}")
        if row.get("group") == "E1":
            predictor = workload_metadata.get("slack_predictor") or {}
            if workload_metadata.get("profile") != "weka_agentx_causal":
                failures.append(f"e1_workload_profile:{cell_id}")
            if workload_metadata.get("dataset_sha256") != (
                "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276"
            ):
                failures.append(f"e1_dataset_sha256:{cell_id}")
            if workload_metadata.get("arrival_model") != (
                "recorded_intra_workflow_fixed_schedule_plus_fixed_workflow_starts"
            ):
                failures.append(f"e1_fixed_schedule:{cell_id}")
            if predictor.get("forbidden_feature_audit_passed") is not True:
                failures.append(f"e1_forbidden_feature_audit:{cell_id}")
            if predictor.get("workflow_disjoint_crossfit_passed") is not True:
                failures.append(f"e1_crossfit_disjoint:{cell_id}")
            if predictor.get("interval_calibration_unit") != (
                "workflow_max_absolute_residual"
            ):
                failures.append(f"e1_calibration_unit:{cell_id}")
            if float(predictor.get("remaining_interval_coverage") or 0.0) < 0.95:
                failures.append(f"e1_request_interval_coverage:{cell_id}")
            if float(
                predictor.get("workflow_simultaneous_interval_coverage") or 0.0
            ) < 0.90:
                failures.append(f"e1_workflow_interval_coverage:{cell_id}")
            if float(predictor.get("slack_lcb_coverage") or 0.0) < 0.95:
                failures.append(f"e1_slack_lcb_coverage:{cell_id}")

    for key, schedules in failure_pairing.items():
        if len(schedules) != 1:
            failures.append(f"unpaired_failure_schedule:{key}")

    request_axes = ("group", "seed", "model", "context_tokens", "load_multiplier")
    paired: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in summary:
        key = tuple(row.get(name, "") for name in request_axes)
        paired[key].add(row.get("workload_sha256", ""))
    for key, hashes in paired.items():
        if len(hashes) != 1 or "" in hashes:
            failures.append(f"unpaired_request_universe:{key}")

    seed_counts = {
        group: len({row.get("seed", "") for row in summary if row.get("group") == group})
        for group in manifest["groups"]
    }
    mechanism_checks["distinct_seed_counts"] = seed_counts
    if manifest.get("profile") == "paper":
        suite_snapshot = json.loads(
            (lineage_dir / "config_snapshot" / "suite.json").read_text(
                encoding="utf-8"
            )
        )
        minimum_seeds = suite_snapshot["profiles"]["paper"].get(
            "minimum_distinct_seeds", {}
        )
        for group in manifest["groups"]:
            minimum = int(minimum_seeds.get(group, 1))
            if seed_counts[group] < minimum:
                failures.append(
                    f"insufficient_paper_seeds:{group}:{seed_counts[group]}<{minimum}"
                )

    s1_allowed_and_required = {
        "recompute": ({"RECOMPUTE"}, {"RECOMPUTE"}),
        "host_checkpoint": ({"DRAM", "RECOMPUTE"}, {"DRAM"}),
        "neighbor_replica": ({"PEER_HBM", "RECOMPUTE"}, {"PEER_HBM"}),
        "aegiskv": (
            {"PEER_HBM", "DRAM", "RECOMPUTE"},
            {"PEER_HBM", "DRAM"},
        ),
    }
    s1_actions: dict[str, set[str]] = defaultdict(set)
    for row in summary:
        if row.get("group") != "S1":
            continue
        raw = raw_by_cell.get(row.get("cell_id", ""), {})
        s1_actions[row.get("policy", "")].update(
            str(event["action"])
            for event in (raw.get("result") or {}).get("events") or []
            if event.get("kind") == "recovery_selected_and_started"
        )
    for policy, (allowed_actions, required_actions) in s1_allowed_and_required.items():
        if policy in s1_actions and (
            not s1_actions[policy].issubset(allowed_actions)
            or not s1_actions[policy].intersection(required_actions)
        ):
            failures.append(
                f"s1_mechanism_adapter:{policy}:{sorted(s1_actions[policy])}"
            )
    mechanism_checks["S1_recovery_actions"] = {
        policy: sorted(actions) for policy, actions in sorted(s1_actions.items())
    }

    s4_ordering: dict[str, int] = defaultdict(int)
    for row in summary:
        if row.get("group") != "S4" or row.get("policy") not in {
            "aegiskv",
            "aegiskv_no_slack",
        }:
            continue
        raw = raw_by_cell.get(row.get("cell_id", ""), {})
        selected = [
            event
            for event in (raw.get("result") or {}).get("events") or []
            if event.get("kind") == "recovery_selected_and_started"
        ]
        due = [float(event["remaining_slack_due_at_ms"]) for event in selected]
        targets = {str(event["activated_target"]) for event in selected}
        policy = row.get("policy", "")
        if len(due) < 2 or len(targets) != 1:
            failures.append(f"s4_missing_one_target_fanin:{row.get('cell_id')}")
        elif policy == "aegiskv" and due != sorted(due):
            failures.append(f"s4_slack_priority_order:{row.get('cell_id')}")
        elif policy == "aegiskv_no_slack" and due == sorted(due):
            failures.append(f"s4_fifo_adversary_not_exercised:{row.get('cell_id')}")
        else:
            s4_ordering[policy] += 1
    mechanism_checks["S4_verified_ordering_cells"] = dict(sorted(s4_ordering.items()))

    timeline_by_cell: dict[str, list[float]] = defaultdict(list)
    for row in timeline:
        value = row.get("mean_alpha", "")
        if value:
            timeline_by_cell[row.get("cell_id", "")].append(float(value))
    s5_ranges: list[float] = []
    s4_no_aimd_ranges: list[float] = []
    for row in summary:
        values = timeline_by_cell.get(row.get("cell_id", ""), [])
        if row.get("group") == "S5":
            dynamic_range = max(values) - min(values) if values else 0.0
            s5_ranges.append(dynamic_range)
            if dynamic_range <= 1e-9:
                failures.append(f"s5_aimd_not_exercised:{row.get('cell_id')}")
        if row.get("group") == "S4" and row.get("policy") == "aegiskv_no_aimd":
            dynamic_range = max(values) - min(values) if values else float("inf")
            s4_no_aimd_ranges.append(dynamic_range)
            if not values or dynamic_range > 1e-12 or any(
                abs(value - 0.5) > 1e-12 for value in values
            ):
                failures.append(f"s4_no_aimd_not_fixed:{row.get('cell_id')}")
    mechanism_checks["S5_AIMD_alpha_range"] = {
        "minimum": min(s5_ranges, default=0.0),
        "maximum": max(s5_ranges, default=0.0),
    }
    mechanism_checks["S4_no_AIMD_alpha_range_max"] = max(
        s4_no_aimd_ranges, default=0.0
    )

    s8_counts: dict[str, set[int]] = defaultdict(set)
    for row in summary:
        if row.get("group") != "S8":
            continue
        scenario = row.get("failure_scenario", "")
        count = int(row.get("failed_worker_count") or 0)
        s8_counts[scenario].add(count)
        if scenario.startswith("worker_") and count != int(scenario.split("_", 1)[1]):
            failures.append(f"s8_failure_scale:{row.get('cell_id')}")
        if scenario == "node_1" and (
            row.get("fault_domain") != "node" or count <= 1
        ):
            failures.append(f"s8_node_domain:{row.get('cell_id')}")
    mechanism_checks["S8_failed_worker_counts"] = {
        scenario: sorted(counts) for scenario, counts in sorted(s8_counts.items())
    }

    s9_transfer_cells = 0
    s9_actions: set[str] = set()
    for row in summary:
        if row.get("group") != "S9":
            continue
        raw = raw_by_cell.get(row.get("cell_id", ""), {})
        selected = [
            event
            for event in (raw.get("result") or {}).get("events") or []
            if event.get("kind") == "recovery_selected_and_started"
        ]
        transfer = [
            event
            for event in selected
            if event.get("action") in {"PEER_HBM", "DRAM"}
            and int(event.get("restored_tokens") or 0) > 0
        ]
        s9_actions.update(str(event.get("action")) for event in selected)
        if len(transfer) != len(selected) or not transfer:
            failures.append(f"s9_transfer_path_not_exercised:{row.get('cell_id')}")
        else:
            s9_transfer_cells += 1
    mechanism_checks["S9_transfer_verification"] = {
        "verified_cells": s9_transfer_cells,
        "actions": sorted(s9_actions),
    }

    e1_rows = [row for row in summary if row.get("group") == "E1"]
    if e1_rows:
        first_raw = raw_by_cell.get(e1_rows[0].get("cell_id", ""), {})
        metadata = first_raw.get("workload_metadata") or {}
        mechanism_checks["E1_AgentX_causal_replay"] = {
            "source_workflows": int(
                (metadata.get("dataset_manifest") or {}).get("workflow_count", 0)
            ),
            "source_requests": int(
                (metadata.get("dataset_manifest") or {}).get("request_count", 0)
            ),
            "eligible_workflows": int(metadata.get("eligible_workflow_count", 0)),
            "selected_workflows_per_seed": int(metadata.get("selected_workflows", 0)),
            "causal_ticket_count": int(metadata.get("causal_ticket_count", 0)),
            "nonzero_slack_ticket_count": int(
                metadata.get("nonzero_slack_ticket_count", 0)
            ),
            "workflow_simultaneous_interval_coverage": float(
                (metadata.get("slack_predictor") or {}).get(
                    "workflow_simultaneous_interval_coverage", 0.0
                )
            ),
        }
        selection_by_seed: dict[str, set[str]] = {}
        actions: set[str] = set()
        for row in e1_rows:
            raw = raw_by_cell.get(row.get("cell_id", ""), {})
            metadata = raw.get("workload_metadata") or {}
            seed = row.get("seed", "")
            selected_ids = set(map(str, metadata.get("selected_workflow_ids") or []))
            prior = selection_by_seed.setdefault(seed, selected_ids)
            if prior != selected_ids:
                failures.append(f"e1_policy_request_pairing:{seed}")
            actions.update(
                str(event.get("action"))
                for event in (raw.get("result") or {}).get("events") or []
                if event.get("kind") == "recovery_selected_and_started"
            )
        seed_items = sorted(selection_by_seed.items())
        for index, (seed, selected_ids) in enumerate(seed_items):
            for other_seed, other_ids in seed_items[index + 1 :]:
                overlap = selected_ids.intersection(other_ids)
                if overlap:
                    failures.append(
                        f"e1_seed_workflow_overlap:{seed}:{other_seed}:{len(overlap)}"
                    )
        mechanism_checks["E1_AgentX_causal_replay"].update(
            {
                "distinct_workflows_across_seeds": len(
                    set().union(*selection_by_seed.values())
                    if selection_by_seed
                    else set()
                ),
                "recovery_actions": sorted(actions),
                "seed_partitions_disjoint": not any(
                    item.startswith("e1_seed_workflow_overlap:")
                    for item in failures
                ),
            }
        )

    s7 = [row for row in summary if row.get("group") == "S7"]
    if s7:
        values = [float(row["failure_added_wct_p95_ms"]) for row in s7]
        bandwidth_means: dict[str, list[float]] = defaultdict(list)
        repair_means: dict[str, list[float]] = defaultdict(list)
        for row in s7:
            bandwidth_means[row["effective_bandwidth_GB_per_s"]].append(
                float(row["failure_added_wct_p95_ms"])
            )
            repair_means[row["repair_time_ms"]].append(
                float(row["failure_added_wct_p95_ms"])
            )
        bw_averages = [sum(values) / len(values) for values in bandwidth_means.values()]
        repair_averages = [sum(values) / len(values) for values in repair_means.values()]
        mechanism_checks["S7_observed_ranges_ms"] = {
            "cell": max(values) - min(values),
            "bandwidth_mean": max(bw_averages) - min(bw_averages),
            "repair_mean": max(repair_averages) - min(repair_averages),
        }
        if len(bandwidth_means) > 1 and max(bw_averages) - min(bw_averages) <= 1e-9:
            warnings.append("s7_bandwidth_axis_has_no_observed_effect")
        if len(repair_means) > 1 and max(repair_averages) - min(repair_averages) <= 1e-9:
            warnings.append("s7_repair_axis_has_no_observed_effect")

    figures = sorted((lineage_dir / "figures").glob("*.png"))
    pdfs = sorted((lineage_dir / "figures").glob("*.pdf"))
    if manifest.get("plots_rendered"):
        expected_figures = len(manifest["groups"])
        if len(figures) != expected_figures or len(pdfs) != expected_figures:
            failures.append(f"figure_count:png={len(figures)},pdf={len(pdfs)}")

    warnings.append("service_time_coefficients_are_modeled_not_same_topology_a100_measurements")
    warnings.append("failover_delay_is_modeled_not_same_topology_a100_measurement")
    warnings.append("field_mttf_is_a_training_cluster_proxy_not_inference_field_measurement")
    if any(group in manifest["groups"] for group in ("S1", "S2", "S3", "S6", "S7", "S8")):
        warnings.append("tracelab_arrivals_are_load_controlled_not_fixed_schedule_replay")
        warnings.append("tracelab_context_is_projected_to_qwen3_limit")
    if any(group in manifest["groups"] for group in ("S4", "S5", "S9")):
        warnings.append("controlled_fork_join_is_topology_stress_not_observed_multi_agent_trace")
    if "E1" in manifest["groups"]:
        warnings.append("agentx_workflow_starts_are_controlled_fixed_interleave")
        warnings.append("agentx_tokens_are_projected_to_qwen3_context_limit")
        warnings.append("agentx_source_api_durations_are_not_qwen_service_times")
    semantic_status = "PASS" if not failures else "FAIL"
    paper_ready_calibrated = False
    status = "PASS_WITH_PHYSICAL_CALIBRATION_BLOCKER" if not failures else "FAIL"
    return {
        "schema_version": "aegiskv.experiment-lineage-audit.v2",
        "status": status,
        "semantic_and_reproducibility_status": semantic_status,
        "paper_ready_calibrated": paper_ready_calibrated,
        "physical_result": False,
        "failure_count": len(failures),
        "failed_checks": failures,
        "warnings": warnings,
        "mechanism_checks": mechanism_checks,
        "cell_count": len(summary),
        "expected_cell_count": expected,
        "png_figure_count": len(figures),
        "pdf_figure_count": len(pdfs),
        "evidence_boundary": "complete metadata DES; physical A100 calibration remains required",
    }


__all__ = ["audit_lineage"]
