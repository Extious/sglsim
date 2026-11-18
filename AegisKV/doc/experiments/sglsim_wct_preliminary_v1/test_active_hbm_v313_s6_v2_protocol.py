#!/usr/bin/env python3
"""Executable structural gates for the frozen S6 v2 preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "active_hbm_v313_s6_v2_protocol.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    spec = json.loads(SPEC.read_text())
    sweep = spec["sweep"]
    semantics = spec["shared_prefix_semantics"]
    checks = [
        (
            "schema_status_and_v1_invalidation_are_explicit",
            spec["schema_version"] == 2
            and spec["status"] == "PREREGISTERED_BEFORE_ANY_S6_V2_RAW_CELL"
            and spec["paper_ready"] is False
            and "per-tier physical residency" in spec["invalidated_v1_reason"],
        ),
        (
            "full_cell_count_is_schema_derived_60",
            len(sweep["reuse_fraction"])
            * len(sweep["topologies"])
            * len(sweep["policies"])
            * len(sweep["paired_seeds"])
            == sweep["registered_cell_count"] == 60,
        ),
        (
            "gate_cell_count_is_schema_derived_4",
            2 * 1 * len(sweep["policies"]) * 1 == 4
            and "= 4" in sweep["gate_cells"],
        ),
        (
            "policy_reuse_topology_and_seeds_are_exact",
            sweep["policies"] == ["Native Mooncake", "AegisKV"]
            and sweep["reuse_fraction"] == [0.0, 0.25, 0.5, 0.75, 1.0]
            and sweep["topologies"] == ["same_node", "cross_node"]
            and sweep["paired_seeds"]
            == [2026082600, 2026082601, 2026082602],
        ),
        (
            "a100_40gib_capacity_and_no_oracle_are_frozen",
            spec["hardware"]["total_hbm_bytes_per_gpu"] == 40 * 1024**3
            and spec["model"]["kv_bytes_per_token"] == 147456
            and spec["model"]["request_max_output_tokens"] == 4096
            and "future cohort" in spec["capacity_equation"]["no_oracle"],
        ),
        (
            "stable_content_and_per_tier_epoch_semantics_are_explicit",
            "(group_id, chunk_ordinal)" in semantics["content_identity"]
            and "tier, epoch" in semantics["physical_epoch_identity"]
            and "PENDING or COMMITTED" in semantics["same_tier_duplicate_rule"]
            and "independently" in semantics["independent_tier_rule"],
        ),
        (
            "resident_active_overlap_and_rematerialization_are_distinct",
            "refcount zero" in semantics["resident_cache_rule"]
            and "active-overlap" in semantics["resident_cache_rule"]
            and "finalized before" in semantics["eviction_rule"]
            and "only a continuous chain" in semantics["recovery"]
            and "minimum predicted residual-DAG slack"
            in semantics["aggregate_priority_rule"]
            and "metadata" in semantics["aggregate_priority_rule"],
        ),
        (
            "rejection_and_legacy_materialized_set_are_fail_closed",
            "Rejected attempts are immediately finalized"
            in semantics["tier_state_machine"]
            and "never controls copy or recovery"
            in semantics["legacy_observed_content_set"],
        ),
        (
            "fixed_arrival_and_paired_probe_denominator_are_frozen",
            spec["workload"]["fixed_arrival_cutoff_ms"] == 90000.0
            and spec["workload"]["control_opportunity_times_ms"]
            == [30000.0, 60000.0, 90000.0]
            and spec["workload"]["fault_probes_per_seed"] == 12
            and "exact request IDs" in spec["workload"]["primary_denominator"],
        ),
        (
            "no_performance_direction_gate_is_registered",
            any("must improve" in value for value in spec["statistics"]["forbidden_gates"])
            and any("monotone" in value for value in spec["statistics"]["forbidden_gates"])
            and "regardless of WCT direction" in spec["selection_policy"],
        ),
        (
            "paper_figure_grammar_is_frozen",
            spec["figure_grammar"]["filename_stem"]
            == "s6_v313_shared_prefix_reuse_v2"
            and spec["figure_grammar"]["minimum_font_pt"] >= 7.0
            and spec["figure_grammar"]["png_minimum_width_px"] >= 1800,
        ),
        (
            "all_source_paths_and_hashes_are_frozen_and_match",
            all(
                Path(row["path"]).is_file()
                and len(row["sha256"]) == 64
                and "TO_BE_FROZEN" not in row["sha256"]
                and sha256(Path(row["path"])) == row["sha256"]
                for row in spec["sources"].values()
            ),
        ),
    ]
    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
