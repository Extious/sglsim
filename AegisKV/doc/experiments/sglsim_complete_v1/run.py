#!/usr/bin/env python3
"""Run the current-design-complete AegisKV metadata DES scenario."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sglsim.srt.simulation.aegis_complete import (
    AegisCompleteSimulator,
    CompleteFailure,
    CompleteRequest,
    CompleteSimulationConfig,
    CompleteSlackUpdate,
    CompleteWorker,
    audit_complete_simulation,
)
from sglsim.srt.simulation.aegis_full_design import (
    AegisFullDesignConfig,
    CalibratedTransferPath,
)
from sglsim.srt.simulation.aegis_independent_tiers import RecoveryTiming
from sglsim.srt.simulation.aegis_residual_dag import (
    ResidualDagNode,
    VisibleResidualDag,
    build_residual_dag_slack_tickets,
)


def _ticket(request_id: str, slack_ms: float, *, version: str, time_ms: float) -> dict:
    graph = VisibleResidualDag(
        workflow_id="complete-demo-workflow",
        graph_version=version,
        observed_at_ms=float(time_ms),
        nodes=(
            ResidualDagNode(
                request_id,
                "agent",
                0.0,
                active=True,
                agent_id=request_id,
            ),
            ResidualDagNode(
                f"critical-{version}",
                "agent",
                float(slack_ms),
                active=True,
                agent_id=f"critical-{version}",
            ),
        ),
        edges=(),
        next_join_subgraph_complete=True,
    )
    return build_residual_dag_slack_tickets(graph)[request_id].to_simulation_fields()


def _config() -> CompleteSimulationConfig:
    design = AegisFullDesignConfig(
        hbm_capacity_bytes=64 * 1024 * 1024,
        dram_capacity_bytes=512 * 1024 * 1024,
        kv_bytes_per_token=1024,
        hbm_copy=CalibratedTransferPath(
            0.10, 10_240.0, "complete-v1-synthetic-hbm-copy"
        ),
        dram_copy=CalibratedTransferPath(
            0.20, 5_120.0, "complete-v1-synthetic-dram-copy"
        ),
        recompute=RecoveryTiming(1.0, 0.10),
        hbm_load=RecoveryTiming(0.5, 0.01),
        dram_load=RecoveryTiming(1.0, 0.02),
        prefix_chunk_tokens=4,
        decode_chunk_tokens=2,
        pp_size=2,
        tp_size=2,
        event_log_limit=20_000,
    )
    return CompleteSimulationConfig(
        workers=tuple(
            CompleteWorker(f"worker-{index}", f"node-{index}")
            for index in range(4)
        ),
        design=design,
        replacement_delay_ms=2.0,
        queue_outage_ms=3.0,
        repair_time_ms=20.0,
        silent_replay_setup_ms=0.25,
        silent_replay_ms_per_token=0.0005,
        control_window_ms=5.0,
        parallel_layout="dp4-tp2-pp2",
    )


def run_demo() -> tuple[dict, dict]:
    requests = (
        CompleteRequest(
            "request-active",
            "complete-demo-workflow",
            0.0,
            12,
            4,
            80.0,
            20.0,
            _ticket("request-active", 20.0, version="visible-v1", time_ms=0.0),
            "worker-0",
        ),
        CompleteRequest(
            "request-queued",
            "complete-demo-workflow",
            1.0,
            8,
            2,
            40.0,
            10.0,
            _ticket("request-queued", 10.0, version="visible-v1", time_ms=1.0),
            "worker-0",
        ),
    )
    failures = (
        # worker-1 is request-active's initial bound HBM target.  This exercises
        # atomic target-first rotation without changing the route version.
        CompleteFailure(22.0, "worker-1"),
        # worker-0 is both the active request's source and the queued request's
        # assigned source.  This exercises recovery plus queue rescheduling.
        CompleteFailure(45.0, "worker-0"),
    )
    slack_updates = (
        CompleteSlackUpdate(
            30.0,
            "request-active",
            12.0,
            _ticket(
                "request-active", 12.0, version="visible-v2", time_ms=30.0
            ),
        ),
    )
    result = AegisCompleteSimulator(_config()).run(
        requests,
        failures=failures,
        slack_updates=slack_updates,
        max_time_ms=500.0,
    )
    audit = audit_complete_simulation(result, require_exercised_failure_path=True)
    return result, audit


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_immutable(path: Path, value: object) -> None:
    content = _canonical(value)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite different artifact: {path}")
        return
    path.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "complete-demo-v1",
    )
    args = parser.parse_args()

    result, audit = run_demo()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_sha = hashlib.sha256(_canonical(result)).hexdigest()
    audit = {**audit, "result_sha256": result_sha}
    summary = {
        "status": audit["status"],
        "schema_version": result["schema_version"],
        "design_semantics": result["design_semantics"],
        "evidence_class": result["evidence_class"],
        "physical_result": False,
        "result_sha256": result_sha,
        "request_count": len(result["requests"]),
        "completed_request_count": sum(
            row["status"] == "COMPLETED" for row in result["requests"].values()
        ),
        "target_rotation_count": audit["target_rotation_count"],
        "source_failure_fence_count": audit["source_failure_fence_count"],
        "queue_reschedule_count": audit["queue_reschedule_count"],
        "recovery_activation_count": audit["recovery_activation_count"],
        "failed_checks": audit["failed_checks"],
    }
    _write_immutable(args.output_dir / "result.json", result)
    _write_immutable(args.output_dir / "audit.json", audit)
    _write_immutable(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
