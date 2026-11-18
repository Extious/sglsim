#!/usr/bin/env python3
"""Generate one bounded two-DP causal snapshot instrumentation smoke."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.aegis_recovery_snapshot import (
    audit_snapshot_capabilities,
    build_causal_recovery_snapshot,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "server.json"


def _active_dp_ranks(artifact: dict[str, Any]) -> set[int]:
    ranks: set[int] = set()
    for row in artifact.get("scheduler_telemetry", ()):
        causal = row.get("causal_state")
        if not isinstance(causal, dict) or not causal.get("active_requests"):
            continue
        ranks.add(int(causal.get("dp_rank", -1)))
    return {rank for rank in ranks if rank >= 0}


def _has_victim_group_commit(artifact: dict[str, Any]) -> bool:
    status = artifact.get("aegis_status")
    if not isinstance(status, dict):
        return False
    return any(
        event.get("kind") == "backup_ack"
        and event.get("rid") == "snapshot-r0"
        and int(event.get("safe_frontier_tokens", 0)) > 0
        and event.get("materialization_scope") == "committed_prefix"
        and event.get("group_commit_verified") is True
        and int(event.get("source_committed_token_end", -1))
        >= int(event.get("safe_frontier_tokens", 0))
        for event in status.get("events", ())
        if isinstance(event, dict)
    )


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--poll-timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = Engine(
        ServerArgs(
            streaming_arrival=True,
            model_path="mock-model",
            dp_size=2,
            sim_config_path=str(args.config.resolve()),
            scheduler_telemetry=True,
            log_level="error",
        )
    )
    engine.start_background_scheduler()
    failure_result: dict[str, Any] | None = None
    recovery_result: dict[str, Any] | None = None
    try:
        for rank in range(2):
            rid = f"snapshot-r{rank}"
            engine.submit(
                GenerateReqInput(
                    rid=rid,
                    input_ids=list(range(64)),
                    max_new_tokens=max(int(args.max_new_tokens), 8),
                    simulation={
                        "workflow_id": f"workflow-{rank}",
                        "agent_id": f"agent-{rank}",
                        "stage_id": 1,
                        "stage_width": 2,
                        "aegis_cache_key": f"object-{rank}",
                        "aegis_backup_key": f"object-{rank}",
                        "aegis_policy": "aegiskv_l1_l2",
                        "backup_kv": True,
                        "critical_agent": True,
                        "aegis_critical_probability": 1.0,
                        "workflow_slack_ms": float(rank * 5),
                        "slack_lcb_ms": float(rank * 5),
                    },
                )
            )

        deadline = time.monotonic() + max(float(args.poll_timeout_s), 1.0)
        pre_failure: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            candidate = engine.calibration_telemetry()
            if (
                _active_dp_ranks(candidate) >= {0, 1}
                and _has_victim_group_commit(candidate)
            ):
                pre_failure = candidate
                break
            time.sleep(0.01)
        if pre_failure is None:
            raise TimeoutError("did not observe concurrent active requests on DP0/DP1")

        failure_result = engine.soft_fail_dp_rank(0)
        if not failure_result.get("accepted"):
            raise RuntimeError(f"soft failure was rejected: {failure_result}")
        at_failure = engine.calibration_telemetry()
        failure_event = at_failure["failure_events"][-1]
        snapshot = build_causal_recovery_snapshot(
            at_failure,
            failure_event_id=str(failure_event["event_id"]),
        )
        audit = audit_snapshot_capabilities(snapshot)

        for rank in range(2):
            engine.wait_for_rid(f"snapshot-r{rank}", timeout_s=30.0)
        recovery_result = engine.recover_soft_failed_dp_rank(0)
        posthoc = engine.calibration_telemetry()

        _json_write(output_dir / "causal_recovery_snapshot.json", snapshot)
        _json_write(output_dir / "snapshot_capability_audit.json", audit)
        # Keep the full source artifact so the cutoff selection can be replayed.
        _json_write(output_dir / "calibration_artifact.json", at_failure)
        summary = {
            "schema_version": "aegiskv-g1-observability-smoke-v2",
            "evidence_class": "sglsim_online_instrumentation_smoke",
            "paper_ready": False,
            "failure_result": failure_result,
            "recovery_result": {
                "released": bool(recovery_result.get("released", False)),
                "requested_dp_rank": int(
                    recovery_result.get("requested_dp_rank", -1)
                ),
                "soft_failed_ranks": list(
                    recovery_result.get("soft_failed_ranks", ())
                ),
                "l1_clear_observed": isinstance(
                    recovery_result.get("l1_clear"), dict
                ),
            },
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "committed_prefixes_at_cutoff": [
                {
                    "rid": str(row.get("rid") or ""),
                    "backup_key": str(row.get("backup_key") or ""),
                    "target_dp_rank": int(row.get("target_dp_rank", -1)),
                    "continuous_recoverable_prefix_tokens": int(
                        row.get("continuous_recoverable_prefix_tokens", 0)
                    ),
                    "source_committed_token_end": int(
                        row.get("source_committed_token_end", 0)
                    ),
                    "copy_completion_evidence": str(
                        row.get("copy_completion_evidence") or ""
                    ),
                }
                for row in snapshot.get("recoverable_prefixes_at_cutoff", ())
                if isinstance(row, dict)
            ],
            "snapshot_capabilities": audit,
            "source_serialization_audit": at_failure.get(
                "serialization_audit", {}
            ),
            "posthoc_completed_requests": sorted(
                rid
                for rid, stats in posthoc["request_stats"].items()
                if stats.get("status") == "completed"
            ),
            "g1_status": "NOT_PROVEN",
            "g2_status": "NOT_PROVEN",
            "limitations": [
                "soft failure is a simulator control event, not a GPU crash",
                "KV restore is not exercised",
                "GroupCommitAck is modeled metadata; physical KV bytes are not transferred in SGLSim",
                "model reload horizon is configured/modelled, not observed",
                "no paired bystander counterfactual is measured",
                "mock-model timings are not physical calibration",
            ],
        }
        _json_write(output_dir / "smoke_summary.json", summary)
        print("evidence_class=sglsim_online_instrumentation_smoke")
        print(f"snapshot_sha256={snapshot['snapshot_sha256']}")
        print(f"interrupted={audit['interrupted_request_count']}")
        print(f"bystanders={audit['bystander_request_count']}")
        print(f"positive_prefixes={audit['positive_target_prefix_count']}")
        print(f"metadata_reservations={audit['metadata_reservation_count']}")
        print(f"g1_ready={audit['g1_ready']}")
        print(f"output_dir={output_dir}")
    finally:
        if failure_result and failure_result.get("accepted") and not recovery_result:
            try:
                engine.recover_soft_failed_dp_rank(0)
            except Exception:
                pass
        engine.shutdown()


if __name__ == "__main__":
    main()
