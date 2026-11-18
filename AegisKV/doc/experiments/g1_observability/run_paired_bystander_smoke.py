#!/usr/bin/env python3
"""Run one bounded, within-run bystander counterfactual instrumentation smoke.

This is not an identified performance experiment.  Two otherwise identical
bystanders run on independent SGLSim DP processes.  One target receives a
retry that can consume a committed prefix; the other receives a retry forced
to recompute.  The artifact proves that paired outcome fields and retry work
can be joined, while retaining explicit rank/process confounding.
"""

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
    build_causal_recovery_snapshot,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "server_paired.json"
DEFAULT_OUTPUT = HERE / "paired_results"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _latest_causal_request(
    artifact: dict[str, Any], rid: str
) -> dict[str, Any] | None:
    latest: tuple[float, dict[str, Any]] | None = None
    for row in artifact.get("scheduler_telemetry", ()):
        causal = row.get("causal_state")
        if not isinstance(causal, dict):
            continue
        for request in causal.get("active_requests", ()):
            if not isinstance(request, dict) or request.get("rid") != rid:
                continue
            candidate = (float(row.get("completed_at", 0.0)), dict(request))
            if latest is None or candidate[0] > latest[0]:
                latest = candidate
    return latest[1] if latest is not None else None


def _committed_prefix(
    artifact: dict[str, Any], *, rid: str, minimum: int
) -> int:
    status = artifact.get("aegis_status")
    if not isinstance(status, dict):
        return 0
    return max(
        (
            int(event.get("safe_frontier_tokens", 0))
            for event in status.get("events", ())
            if isinstance(event, dict)
            and event.get("kind") == "backup_ack"
            and event.get("rid") == rid
            and event.get("materialization_scope") == "committed_prefix"
            and event.get("group_commit_verified") is True
            and int(event.get("source_committed_token_end", -1)) >= minimum
        ),
        default=0,
    )


def _latency(row: dict[str, Any]) -> float:
    return max(
        float(row.get("last_event_time", 0.0))
        - float(row.get("created_time", 0.0)),
        0.0,
    )


def _request_stats_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rid": str(row.get("rid") or ""),
        "status": str(row.get("status") or ""),
        "assigned_dp_rank": int(row.get("aegis_assigned_dp_rank", -1)),
        "latency_s": _latency(row),
        "queue_delay_s": max(
            float(row.get("queue_end", 0.0))
            - float(row.get("queue_start", 0.0)),
            0.0,
        ),
        "service_time_s": float(row.get("service_time_s", 0.0)),
        "generated_tokens": int(row.get("generated_tokens", 0)),
        "restored_tokens": int(row.get("aegis_replay_restored_tokens", 0)),
        "restored_prefill_tokens": int(
            row.get("aegis_replay_restored_prefill_tokens", 0)
        ),
        "recompute_tokens": int(
            row.get("aegis_replay_recompute_tokens", 0)
        ),
        "restore_action": str(
            row.get("aegis_replay_restore_action") or ""
        ),
        "restore_timing_source": str(
            row.get("aegis_replay_restore_timing_source") or ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--bystander-output-tokens", type=int, default=160)
    parser.add_argument("--poll-timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_tokens = max(int(args.prompt_tokens), 64)
    deadline_s = max(float(args.poll_timeout_s), 1.0)
    engine = Engine(
        ServerArgs(
            streaming_arrival=True,
            model_path="mock-model",
            dp_size=3,
            sim_config_path=str(args.config.resolve()),
            scheduler_telemetry=True,
            log_level="error",
        )
    )
    engine.start_background_scheduler()
    failure_result: dict[str, Any] | None = None
    recovered = False
    try:
        engine.submit(
            GenerateReqInput(
                rid="paired-victim",
                input_ids=list(range(prompt_tokens)),
                max_new_tokens=256,
                simulation={
                    "force_dp_rank": 0,
                    "aegis_backup_dp_rank": 1,
                    "workflow_id": "paired-victim-workflow",
                    "agent_id": "victim-agent",
                    "aegis_cache_key": "paired-victim-object",
                    "aegis_backup_key": "paired-victim-object",
                    "aegis_policy": "aegiskv_l1_l2",
                    "backup_kv": True,
                    "critical_agent": True,
                    "aegis_critical_probability": 1.0,
                    "workflow_slack_ms": 0.0,
                },
            )
        )

        deadline = time.monotonic() + deadline_s
        committed = 0
        while time.monotonic() < deadline:
            candidate = engine.calibration_telemetry()
            committed = _committed_prefix(
                candidate,
                rid="paired-victim",
                minimum=prompt_tokens,
            )
            if committed >= prompt_tokens:
                break
            time.sleep(0.01)
        if committed < prompt_tokens:
            raise TimeoutError("victim did not publish the prompt GroupCommit")

        for suffix, rank in (("reuse", 1), ("recompute", 2)):
            engine.submit(
                GenerateReqInput(
                    rid=f"paired-bystander-{suffix}",
                    input_ids=list(range(128)),
                    max_new_tokens=max(int(args.bystander_output_tokens), 32),
                    simulation={
                        "force_dp_rank": rank,
                        "workflow_id": f"paired-bystander-{suffix}-workflow",
                        "agent_id": f"bystander-{suffix}",
                        "paired_role": "healthy_bystander",
                    },
                )
            )

        deadline = time.monotonic() + deadline_s
        at_failure: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            candidate = engine.calibration_telemetry()
            if all(
                _latest_causal_request(candidate, rid) is not None
                for rid in (
                    "paired-bystander-reuse",
                    "paired-bystander-recompute",
                )
            ):
                at_failure = candidate
                break
            time.sleep(0.005)
        if at_failure is None:
            raise TimeoutError("paired bystanders were not concurrently active")

        failure_result = engine.soft_fail_dp_rank(0)
        if not failure_result.get("accepted"):
            raise RuntimeError(f"soft failure rejected: {failure_result}")
        at_failure = engine.calibration_telemetry()
        failure_event = at_failure["failure_events"][-1]
        snapshot = build_causal_recovery_snapshot(
            at_failure,
            failure_event_id=str(failure_event["event_id"]),
        )

        common_retry = {
            "aegis_policy": "aegiskv_l1_l2",
            "failover_replay": True,
            "failed_dp_rank": 0,
            "pre_failure_total_tokens": prompt_tokens,
            "workflow_id": "paired-victim-workflow",
            "agent_id": "victim-agent",
            "critical_agent": True,
        }
        engine.submit(
            GenerateReqInput(
                rid="paired-retry-reuse",
                input_ids=list(range(prompt_tokens)),
                max_new_tokens=1,
                simulation={
                    **common_retry,
                    "aegis_backup_key": "paired-victim-object",
                    "aegis_cache_key": "paired-victim-object",
                    "aegis_route_plan_retry_dp_rank": 1,
                },
            )
        )
        engine.submit(
            GenerateReqInput(
                rid="paired-retry-recompute",
                input_ids=list(range(prompt_tokens)),
                max_new_tokens=1,
                simulation={
                    **common_retry,
                    # Deliberately miss the committed object while preserving
                    # the same retry action space and target scheduler path.
                    "aegis_backup_key": "paired-missing-object",
                    "aegis_cache_key": "paired-missing-object",
                    "aegis_route_plan_retry_dp_rank": 2,
                },
            )
        )

        for rid in (
            "paired-retry-reuse",
            "paired-retry-recompute",
            "paired-bystander-reuse",
            "paired-bystander-recompute",
            "paired-victim",
        ):
            engine.wait_for_rid(rid, timeout_s=60.0)
        recovery = engine.recover_soft_failed_dp_rank(0)
        recovered = bool(recovery.get("released", False))
        posthoc = engine.calibration_telemetry()

        stats = posthoc["request_stats"]
        reuse_retry = _request_stats_projection(stats["paired-retry-reuse"])
        recompute_retry = _request_stats_projection(
            stats["paired-retry-recompute"]
        )
        reuse_bystander = _request_stats_projection(
            stats["paired-bystander-reuse"]
        )
        recompute_bystander = _request_stats_projection(
            stats["paired-bystander-recompute"]
        )
        if reuse_retry["restored_prefill_tokens"] <= 0:
            raise RuntimeError("reuse arm did not materialize a restored prefix")
        if recompute_retry["restored_prefill_tokens"] != 0:
            raise RuntimeError("recompute arm unexpectedly restored a prefix")

        pre_failure_progress = {
            rid: _latest_causal_request(at_failure, rid)
            for rid in (
                "paired-bystander-reuse",
                "paired-bystander-recompute",
            )
        }
        summary = {
            "schema_version": "aegiskv-paired-bystander-smoke-v1",
            "evidence_class": "sglsim_within_run_rank_paired_smoke",
            "paper_ready": False,
            "counterfactual_identification": "NOT_IDENTIFIED",
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "failure_event_id": str(failure_event["event_id"]),
            "failure_result": failure_result,
            "source_committed_prefix_tokens": committed,
            "pre_failure_bystander_progress": pre_failure_progress,
            "reuse_retry": reuse_retry,
            "recompute_retry": recompute_retry,
            "reuse_target_bystander": reuse_bystander,
            "recompute_target_bystander": recompute_bystander,
            "observed_bystander_latency_delta_s": (
                recompute_bystander["latency_s"]
                - reuse_bystander["latency_s"]
            ),
            "observed_retry_latency_delta_s": (
                recompute_retry["latency_s"] - reuse_retry["latency_s"]
            ),
            "recovered_soft_failure": recovered,
            "limitations": [
                "the two arms use different DP processes and are rank-confounded",
                "one within-run pair has no confidence interval",
                "SGLSim GroupCommit and restore are metadata models, not KV byte movement",
                "heuristic predictor timing is not hardware-calibrated evidence",
                "the recompute arm intentionally hides the object; it is a mechanism control, not a faithful external baseline",
            ],
            "g1_status": "NOT_PROVEN",
            "g4_status": "NOT_PROVEN",
        }
        _write_json(output_dir / "causal_recovery_snapshot.json", snapshot)
        _write_json(output_dir / "posthoc_artifact.json", posthoc)
        _write_json(output_dir / "paired_bystander_summary.json", summary)
        print("evidence_class=sglsim_within_run_rank_paired_smoke")
        print(f"snapshot_sha256={snapshot['snapshot_sha256']}")
        print(
            "reuse_restored_prefill_tokens="
            f"{reuse_retry['restored_prefill_tokens']}"
        )
        print(
            "recompute_restored_prefill_tokens="
            f"{recompute_retry['restored_prefill_tokens']}"
        )
        print(
            "observed_bystander_latency_delta_s="
            f"{summary['observed_bystander_latency_delta_s']:.6f}"
        )
        print(f"output_dir={output_dir}")
    finally:
        if failure_result and failure_result.get("accepted") and not recovered:
            try:
                engine.wait_for_rid("paired-victim", timeout_s=60.0)
                engine.recover_soft_failed_dp_rank(0)
            except Exception:
                pass
        engine.shutdown()


if __name__ == "__main__":
    main()
