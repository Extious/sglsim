"""Freeze request-level failure victims from one Cache-Aware calibration run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .wct_cdf_client import load_failure_manifest, normalize_failure_manifest
from .workflow_client import (
    RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION,
    RETRY_ROUTE_MODE_ROUTER_COMMITTED,
    RETRY_ROUTE_MODES,
)


def build_matched_manifest(
    *,
    calibration_result: Path,
    base_manifest: Path,
    dp_size: int | None = None,
    pp_size: int | None = None,
    completion_guard_s: float = 5.0,
    retry_route_mode: str = RETRY_ROUTE_MODE_ROUTER_COMMITTED,
) -> dict[str, Any]:
    if retry_route_mode not in RETRY_ROUTE_MODES:
        raise ValueError(f"unsupported retry_route_mode={retry_route_mode!r}")
    calibration = json.loads(calibration_result.read_text(encoding="utf-8"))
    base_raw = json.loads(base_manifest.read_text(encoding="utf-8"))
    base_deployment = dict(base_raw.get("deployment") or {})
    resolved_dp_size = int(
        dp_size
        or calibration.get("dp_size")
        or base_deployment.get("dp_size")
        or 8
    )
    resolved_pp_size = int(
        pp_size
        or calibration.get("pp_size")
        or base_deployment.get("pp_size")
        or 1
    )
    if resolved_dp_size <= 0 or resolved_pp_size <= 0:
        raise ValueError("resolved DP/PP sizes must be positive")
    for name, resolved, declared in (
        ("dp_size", resolved_dp_size, base_deployment.get("dp_size")),
        ("pp_size", resolved_pp_size, base_deployment.get("pp_size")),
    ):
        if declared is not None and int(declared) != resolved:
            raise ValueError(
                f"base manifest {name}={declared} does not match resolved {resolved}"
            )
    # A legacy base manifest defaults to calibration-fixed retry routing.  The
    # derived formal manifest must instead keep failure/victim selection
    # independent of the policy's immutable Router b(r), so normalize the
    # base timing/rank events first and then explicitly rewrite the contract.
    base_for_load = copy.deepcopy(base_raw)
    if retry_route_mode == RETRY_ROUTE_MODE_ROUTER_COMMITTED:
        base_for_load["schema_version"] = 2
        base_for_load["retry_route_mode"] = retry_route_mode
        for event in base_for_load.get("events", []):
            if isinstance(event, dict):
                for victim in event.get("victims", []):
                    if isinstance(victim, dict):
                        victim.pop("backup_dp_rank", None)
    if base_for_load != base_raw:
        manifest = normalize_failure_manifest(
            base_for_load,
            dp_size=resolved_dp_size,
            pp_size=resolved_pp_size,
            failure_count=None,
            source=str(base_manifest),
        )
    else:
        manifest = load_failure_manifest(
            base_manifest,
            dp_size=resolved_dp_size,
            pp_size=resolved_pp_size,
            failure_count=None,
        )
    ranks = [int(event["dp_rank"]) for event in manifest["events"]]
    if len(set(ranks)) != len(ranks):
        raise ValueError(
            "calibration retry events do not carry event IDs; matched-manifest "
            "derivation currently requires each failure event to use a distinct rank"
        )

    routes: dict[str, dict[str, Any]] = {}
    retry_keys_by_rank: dict[int, set[str]] = {rank: set() for rank in ranks}
    for workflow in calibration.get("workflows", []):
        for route in workflow.get("request_routes", []):
            rid = str(route.get("rid") or "")
            if rid:
                routes[rid] = dict(route)
        for retry in workflow.get("retry_events", []):
            failed_rank = int(retry.get("failed_dp_rank", -1))
            backup_key = str(retry.get("backup_key") or "")
            if failed_rank in retry_keys_by_rank and backup_key:
                retry_keys_by_rank[failed_rank].add(backup_key)

    total_victims = 0
    excluded_near_boundary = 0
    for event in manifest["events"]:
        failed_rank = int(event["dp_rank"])
        stable_completion_deadline = float(event["offset_s"]) - max(
            float(completion_guard_s), 0.0
        )
        victims: list[dict[str, Any]] = []
        for backup_key in sorted(retry_keys_by_rank[failed_rank]):
            route = routes.get(backup_key)
            if route is None:
                raise ValueError(
                    f"calibration result has no request route for victim {backup_key}"
                )
            execution_rank = int(route.get("execution_dp_rank", -1))
            if execution_rank != failed_rank:
                raise ValueError(
                    f"victim {backup_key} route rank {execution_rank} does not "
                    f"match failed rank {failed_rank}"
                )
            backup_rank = int(route.get("backup_dp_rank", -1))
            if (
                retry_route_mode
                == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
                and (
                    not 0 <= backup_rank < resolved_dp_size
                    or backup_rank == failed_rank
                )
            ):
                raise ValueError(
                    f"victim {backup_key} has invalid backup rank {backup_rank}"
                )
            completion_elapsed_s = float(
                route.get("completion_elapsed_s", 0.0)
            )
            if completion_elapsed_s > stable_completion_deadline:
                excluded_near_boundary += 1
                continue
            victim = {
                "backup_key": backup_key,
                "execution_dp_rank": execution_rank,
                "retry_route_mode": retry_route_mode,
            }
            if (
                retry_route_mode
                == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
            ):
                victim["backup_dp_rank"] = backup_rank
            victims.append(victim)
        event["victims"] = victims
        total_victims += len(victims)

    if total_victims == 0:
        raise ValueError("calibration run produced no failure victims")
    nonempty_events = [event for event in manifest["events"] if event["victims"]]
    dropped_empty_events = len(manifest["events"]) - len(nonempty_events)
    return {
        **manifest,
        "schema_version": (
            1
            if retry_route_mode
            == RETRY_ROUTE_MODE_MANIFEST_FIXED_CALIBRATION
            else 2
        ),
        "retry_route_mode": retry_route_mode,
        "events": nonempty_events,
        "calibration_result": str(calibration_result.resolve()),
        "victim_selection": (
            "latest_completed_state_per_agent_on_failed_rank_"
            "with_pre_failure_completion_guard"
        ),
        "completion_guard_s": max(float(completion_guard_s), 0.0),
        "resolved_dp_size": resolved_dp_size,
        "resolved_pp_size": resolved_pp_size,
        "excluded_near_failure_boundary": excluded_near_boundary,
        "dropped_events_without_stable_victims": dropped_empty_events,
        "victim_count": total_victims,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a fixed-time/rank plus matched-victim failure manifest."
    )
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp-size", type=int, default=None)
    parser.add_argument("--pp-size", type=int, default=None)
    parser.add_argument("--completion-guard-s", type=float, default=5.0)
    parser.add_argument(
        "--retry-route-mode",
        choices=RETRY_ROUTE_MODES,
        default=RETRY_ROUTE_MODE_ROUTER_COMMITTED,
        help=(
            "formal default keeps backup/retry routing owned by Router; "
            "manifest_fixed_calibration preserves legacy calibration behavior"
        ),
    )
    args = parser.parse_args()
    payload = build_matched_manifest(
        calibration_result=args.calibration_result,
        base_manifest=args.base_manifest,
        dp_size=args.dp_size,
        pp_size=args.pp_size,
        completion_guard_s=args.completion_guard_s,
        retry_route_mode=args.retry_route_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} with {payload['victim_count']} matched victims",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
