#!/usr/bin/env python3
"""Executable structural gates for the preregistered S6 protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "active_hbm_v313_s6_shared_prefix_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    spec = json.loads(SPEC.read_text())
    checks: list[tuple[str, bool]] = []
    sweep = spec["sweep"]
    checks.append((
        "exact_registered_60_cell_factorial",
        sweep["reuse_fraction"] == [0.0, 0.25, 0.5, 0.75, 1.0]
        and sweep["topologies"] == ["same_node", "cross_node"]
        and sweep["policies"] == ["Native Mooncake", "AegisKV"]
        and len(sweep["paired_seeds"]) == 3
        and 5 * 2 * 2 * 3 == sweep["registered_cell_count"] == 60,
    ))
    checks.append((
        "frozen_a100_40gib_capacity_equation",
        spec["hardware"]["total_hbm_bytes_per_gpu"] == 40 * 1024**3
        and spec["model"]["weight_bytes_per_rank"] == 16381470720
        and spec["hardware"]["runtime_reserve_bytes_per_gpu"] == 5153960756
        and spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"]
        == 21414241484
        and spec["capacity_equation"]["reservation_bytes_per_request"]
        == (16384 + 4096) * 147456,
    ))
    checks.append((
        "pressure_cell_is_analytically_derived_not_result_selected",
        spec["workload"]["offered_active_requests_per_gpu"] == 4
        and spec["capacity_equation"]["initial_quota_excess_over_headroom_bytes"]
        == (
            spec["capacity_equation"]["four_request_serving_reservation_bytes"]
            + spec["capacity_equation"]["initial_alpha_backup_quota_bytes"]
            - spec["hardware"]["serving_plus_peer_backup_budget_bytes_per_gpu"]
        )
        and spec["capacity_equation"]["initial_quota_excess_over_headroom_bytes"] > 0,
    ))
    checks.append((
        "topology_and_directional_bandwidth_are_explicit",
        spec["hardware"]["gpus_per_node"] == 8
        and spec["hardware"]["system_dp"] == 16
        and spec["hardware"]["node_count"] == 2
        and spec["hardware"]["copy_bandwidth_gb_s"] == {
            "peer_hbm_same_node": 50.0,
            "peer_hbm_cross_node": 10.0,
            "dram_local": 16.0,
            "dram_remote": 10.0,
        }
        and spec["hardware"]["recovery_bandwidth_gb_s"]
        == spec["hardware"]["copy_bandwidth_gb_s"],
    ))
    semantics = spec["shared_prefix_semantics"]
    checks.append((
        "shared_identity_refcount_and_committed_recovery_are_frozen",
        semantics["group_size"] == 4
        and semantics["prefix_chunk_tokens"] == 4096
        and "first causal producer" in semantics["physical_copy"]
        and "refcount does not pin" in semantics["cache_lifetime"]
        and "COMMITTED" in semantics["recovery"]
        and "future group membership is not read" in semantics["no_future_admission_value"],
    ))
    checks.append((
        "fixed_arrival_and_three_control_opportunities",
        spec["workload"]["fixed_arrival_cutoff_ms"] == 90000.0
        and spec["workload"]["control_opportunity_times_ms"]
        == [30000.0, 60000.0, 90000.0]
        and spec["workload"]["fault_probes_per_seed"] == 12
        and spec["workload"]["failure_type"] == "source_worker_gpu",
    ))
    forbidden = " ".join(spec["statistics"]["forbidden_gates"])
    checks.append((
        "no_performance_or_monotonicity_success_threshold",
        "must improve" in forbidden
        and "monotone" in forbidden
        and "removed" in forbidden
        and "must win" not in " ".join(spec["hard_gates"]).lower(),
    ))
    grammar = spec["figure_grammar"]
    checks.append((
        "paper_figure_grammar_has_no_evidence_prose",
        grammar["size_inches"] == [7.0, 5.1]
        and grammar["minimum_font_pt"] >= 7.0
        and grammar["png_minimum_width_px"] >= 1800
        and "no pilot" in grammar["panel_text_boundary"],
    ))
    checks.append((
        "all_frozen_sources_match_hash",
        all(
            Path(row["path"]).is_file()
            and sha256(Path(row["path"])) == row["sha256"]
            for row in spec["sources"].values()
        ),
    ))
    checks.append((
        "evidence_boundary_is_fail_closed",
        spec["paper_ready"] is False
        and "modeled/unmeasured" in spec["evidence_boundary"]
        and "not physical A100" in spec["evidence_boundary"]
        and "not a verified production" in spec["evidence_boundary"],
    ))

    failed = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(f"RESULT {len(checks) - len(failed)}/{len(checks)} PASS", flush=True)
    if failed:
        raise SystemExit("failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()

