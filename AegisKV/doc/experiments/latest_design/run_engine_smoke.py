#!/usr/bin/env python3
"""Run the latest AegisKV path through SGLSim's real Engine topology."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sglsim.srt.entrypoints.engine import Engine
from sglsim.srt.managers.io_struct import GenerateReqInput
from sglsim.srt.server_args import ServerArgs
from sglsim.srt.simulation.aegis_independent_tiers import DESIGN_SEMANTICS


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "server_smoke.json")
    parser.add_argument(
        "--output", type=Path, default=root / "results" / "engine_smoke.json"
    )
    args = parser.parse_args()
    engine = Engine(
        ServerArgs(
            sim_config_path=str(args.config),
            dp_size=2,
            pp_size=1,
        )
    )
    try:
        assert engine._dp_controller is not None
        engine.start_background_scheduler()
        for index in range(4):
            engine.submit(
                GenerateReqInput(
                    rid=f"latest-engine-smoke-{index}",
                    # Use concrete token metadata here. ``input_len`` is a
                    # predictor-only shortcut and does not populate the radix
                    # cache's token sequence, so it cannot validate a copied
                    # KV prefix.
                    input_ids=[
                        index * 5000 + token + 1 for token in range(4096)
                    ],
                    max_new_tokens=8,
                    created_time=0.0,
                    simulation={
                        "created_time": 0.0,
                        "aegis_policy": "aegiskv_l1_l2",
                        "backup_kv": True,
                        "aegis_critical_probability": 1.0,
                        "workflow_slack_ms": 0.0,
                        "workflow_id": "latest-engine-smoke-workflow",
                        "agent_id": f"agent-{index}",
                        "worker_seq": index,
                        "aegis_failure_scope": "node",
                    },
                )
            )
        result = engine.run_until_idle()
        status = engine._dp_controller.aegis_status()
        events = status.get("events", [])
        latest_events = [
            event
            for event in events
            if event.get("design_semantics") == DESIGN_SEMANTICS
            or any(
                isinstance(update, dict)
                and update.get("design_semantics") == DESIGN_SEMANTICS
                for update in event.get("hint_updates", [])
            )
        ]
        group_commits = [
            event
            for event in events
            if event.get("kind") == "backup_ack"
            and event.get("materialization_scope") == "committed_prefix"
            and event.get("group_commit_verified") is True
            and int(event.get("safe_frontier_tokens", 0)) > 0
        ]
        group_commit_summaries = []
        for event in group_commits:
            updates = [
                update
                for update in event.get("hint_updates", [])
                if isinstance(update, dict)
            ]
            latest_update = updates[-1] if updates else {}
            group_commit_summaries.append(
                {
                    "rid": str(event.get("rid") or ""),
                    "source_dp_rank": int(event.get("execution_dp_rank", -1)),
                    "target_dp_rank": int(event.get("backup_dp_rank", -1)),
                    "action": str(event.get("action") or ""),
                    "hbm_committed_prefix_tokens": int(
                        event.get("l1_tokens", 0)
                    ),
                    "dram_committed_prefix_tokens": int(
                        event.get("l2_tokens", 0)
                    ),
                    "safe_frontier_tokens": int(
                        event.get("safe_frontier_tokens", 0)
                    ),
                    "source_committed_token_end": int(
                        event.get("source_committed_token_end", 0)
                    ),
                    "design_semantics": str(
                        latest_update.get("design_semantics") or ""
                    ),
                    "group_commit_verified": bool(
                        event.get("group_commit_verified", False)
                    ),
                    "copy_completion_evidence": str(
                        event.get("copy_completion_evidence") or ""
                    ),
                }
            )
        output = {
            "design_semantics": DESIGN_SEMANTICS,
            "evidence_class": "sglsim_engine_metadata_smoke",
            "request_count": len(result.request_stats),
            "all_requests_completed": all(
                item.status == "completed"
                for item in result.request_stats.values()
            ),
            "event_count": len(events),
            "latest_design_event_count": len(latest_events),
            "group_commit_count": len(group_commits),
            "group_commits": group_commit_summaries,
            "physical_runtime_claim": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(output, indent=2))
        if not (
            output["all_requests_completed"]
            and output["latest_design_event_count"] >= 1
            and output["group_commit_count"] >= 1
        ):
            raise SystemExit(2)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
