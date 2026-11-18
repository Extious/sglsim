#!/usr/bin/env python3
"""Executable mechanism gate for S7-v2 failure-target denominators."""

from __future__ import annotations

import json
from pathlib import Path

import active_hbm_v313 as v313
import run_active_hbm_v313_s7_directional as s7v1
import run_active_hbm_v313_s7_failure_target_v2 as v2
import run_preliminary as core


ROOT = Path(__file__).resolve().parent
PARENT = json.loads((ROOT / "active_hbm_v313_s7_directional_protocol.json").read_text())


def universe(seed: int = 2026082600, failure_type: str = "source_worker_gpu"):
    protocol, _capacity = s7v1.configured_protocol(
        PARENT, axis="bandwidth", direction="peer_hbm_recovery",
        bandwidth_gb_s=10.0, repair_ms=None,
    )
    return v2.build_universe_v2(
        PARENT, protocol, seed=seed, failure_type=failure_type
    )


def test_nonempty_targets_and_active_source() -> None:
    nominal, _workload, cohort, probes = universe()
    targets = v2.targets_for_universe(PARENT, nominal, cohort, probes, seed=2026082600)
    reference = [row for row in nominal if row.workflow_id in cohort]
    assert len(probes) == len(targets) == 12
    for probe, ids in zip(probes, targets):
        assert ids
        active = [row for row in reference if row.start_ms <= probe.time_ms < row.finish_ms]
        assert any(row.source_rank == probe.source_rank for row in active)


def test_worker_target_formula_exact() -> None:
    nominal, _workload, cohort, probes = universe(failure_type="source_worker_gpu")
    targets = v2.targets_for_universe(PARENT, nominal, cohort, probes, seed=2026082600)
    reference = [row for row in nominal if row.workflow_id in cohort]
    for probe, ids in zip(probes, targets):
        expected = sorted({
            row.workflow_id for row in reference
            if row.source_rank == probe.source_rank
            and row.start_ms <= probe.time_ms < row.finish_ms
        })
        assert ids == expected


def test_node_target_formula_and_eight_ranks_exact() -> None:
    nominal, _workload, cohort, probes = universe(failure_type="source_node")
    targets = v2.targets_for_universe(PARENT, nominal, cohort, probes, seed=2026082600)
    reference = [row for row in nominal if row.workflow_id in cohort]
    for probe, ids in zip(probes, targets):
        failed = v313.v312.v31.failure_domain_ranks(
            probe, gpus_per_node=8, dp=64
        )
        assert len(failed) == 8
        expected = sorted({
            row.workflow_id for row in reference
            if row.source_rank in failed
            and row.start_ms <= probe.time_ms < row.finish_ms
        })
        assert ids == expected


def test_probe_tuple_independent_of_treatment() -> None:
    hashes = set()
    for direction, bandwidth in (
        ("source_to_peer_hbm_copy", 1.0),
        ("source_to_dram_copy", 50.0),
        ("peer_hbm_recovery", 5.0),
        ("dram_recovery", 25.0),
    ):
        protocol, _capacity = s7v1.configured_protocol(
            PARENT, axis="bandwidth", direction=direction,
            bandwidth_gb_s=bandwidth, repair_ms=None,
        )
        _nominal, _workload, _cohort, probes = v2.build_universe_v2(
            PARENT, protocol, seed=2026082600, failure_type="source_worker_gpu"
        )
        hashes.add(s7v1.probe_hash(probes, include_domain=True))
    assert len(hashes) == 1


def test_target_identity_independent_of_treatment() -> None:
    bundles = set()
    for direction, bandwidth in (
        ("source_to_peer_hbm_copy", 1.0),
        ("dram_recovery", 50.0),
    ):
        protocol, _capacity = s7v1.configured_protocol(
            PARENT, axis="bandwidth", direction=direction,
            bandwidth_gb_s=bandwidth, repair_ms=None,
        )
        nominal, _workload, cohort, probes = v2.build_universe_v2(
            PARENT, protocol, seed=2026082600, failure_type="source_worker_gpu"
        )
        targets = v2.targets_for_universe(PARENT, nominal, cohort, probes, seed=2026082600)
        bundles.add(v2.digest_json([
            v2.digest_json({
                "seed": 2026082600, "failure_type": probe.failure_type,
                "probe_id": probe.probe_id, "timestamp_ms": probe.time_ms,
                "source_rank": probe.source_rank,
                "sorted_workflow_ids": ids,
            }) for probe, ids in zip(probes, targets)
        ]))
    assert len(bundles) == 1


def test_empty_target_fails_closed_without_resampling() -> None:
    try:
        v2.reference_probes_and_targets(
            [], set(), seed=1, failure_type="source_worker_gpu", count=1,
            left_ms=90.0, right_ms=180.0, gpus_per_node=8, dp=64,
        )
    except v2.EmptyFailureTargetError:
        return
    raise AssertionError("empty target did not fail closed")


def test_target_wct_exact_fixed_ids() -> None:
    ids = ["w0", "w1", "w2"]
    values = __import__("numpy").asarray([10.0, 20.0, 40.0])
    assert v2._mean_target_wct(["w0", "w2"], ids, values) == 25.0


def test_missing_target_id_fails_closed() -> None:
    values = __import__("numpy").asarray([10.0])
    try:
        v2._mean_target_wct(["missing"], ["w0"], values)
    except AssertionError:
        return
    raise AssertionError("missing target ID did not fail closed")


def test_worker_and_node_share_base_probe_tuple() -> None:
    _n1, _w1, _c1, worker = universe(failure_type="source_worker_gpu")
    _n2, _w2, _c2, node = universe(failure_type="source_node")
    assert s7v1.probe_hash(worker, include_domain=False) == s7v1.probe_hash(
        node, include_domain=False
    )


def test_fault_domain_does_not_change_reference_trace() -> None:
    worker_nominal, _w1, worker_cohort, _p1 = universe(failure_type="source_worker_gpu")
    node_nominal, _w2, node_cohort, _p2 = universe(failure_type="source_node")
    assert s7v1.workflow_hash(worker_nominal) == s7v1.workflow_hash(node_nominal)
    assert s7v1.digest_json(sorted(worker_cohort)) == s7v1.digest_json(sorted(node_cohort))


TESTS = (
    test_nonempty_targets_and_active_source,
    test_worker_target_formula_exact,
    test_node_target_formula_and_eight_ranks_exact,
    test_probe_tuple_independent_of_treatment,
    test_target_identity_independent_of_treatment,
    test_empty_target_fails_closed_without_resampling,
    test_target_wct_exact_fixed_ids,
    test_missing_target_id_fails_closed,
    test_worker_and_node_share_base_probe_tuple,
    test_fault_domain_does_not_change_reference_trace,
)


def main() -> None:
    passed = 0
    for test in TESTS:
        try:
            test()
        except Exception as error:
            print(f"FAIL {test.__name__}: {type(error).__name__}: {error}")
            raise
        else:
            passed += 1
            print(f"PASS {test.__name__}")
    print(f"RESULT {passed}/{len(TESTS)} PASS")


if __name__ == "__main__":
    main()
