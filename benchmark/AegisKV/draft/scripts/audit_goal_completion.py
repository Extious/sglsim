#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DRAFT_ROOT = Path(__file__).resolve().parents[1]
AEGIS_ROOT = DRAFT_ROOT.parent
REPO_ROOT = DRAFT_ROOT.parents[2]
FIGURES = DRAFT_ROOT / "figures"
EXPERIMENTS = DRAFT_ROOT / "experiments"
BEAMER = AEGIS_ROOT / "beamer"
OUT_MD = EXPERIMENTS / "goal_completion_audit.md"
OUT_JSON = EXPERIMENTS / "goal_completion_audit.json"

PASS = "pass"
PARTIAL = "partial"
BLOCKED = "blocked"
MISSING = "missing"
STATUS_ORDER = {PASS: 0, PARTIAL: 1, BLOCKED: 2, MISSING: 3}


@dataclass(frozen=True)
class RequirementCheck:
    check_id: str
    status: str
    requirement: str
    evidence: str
    next_step: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _normalize_artifact_path(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("policy_table:"):
        text = text.split(":", 1)[1]
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def _png_dimensions(path: Path) -> tuple[int, int]:
    if not path.exists() or path.stat().st_size < 24:
        return (0, 0)
    with path.open("rb") as f:
        header = f.read(24)
    if not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR":
        return (0, 0)
    return (int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big"))


def _status_counts(checks: list[RequirementCheck]) -> dict[str, int]:
    return {
        status: sum(1 for item in checks if item.status == status)
        for status in (PASS, PARTIAL, BLOCKED, MISSING)
    }


def _bound_slurm_short_status(payload: dict[str, Any]) -> str:
    counts_value = payload.get("status_counts")
    if not isinstance(counts_value, dict):
        return ""
    counts = {
        key: int(_float(counts_value.get(key)))
        for key in ("pending", "running", "completed", "failed")
    }
    total = int(_float(payload.get("total_bound_jobs")))
    if total and counts["pending"] == total:
        return f"{total} jobs pending"
    if total and counts["running"] == total:
        return f"{total} jobs running"
    if total and counts["completed"] == total:
        return f"{total} jobs complete"
    if total:
        parts = [
            f"{counts[key]} {label}"
            for key, label in (
                ("pending", "pending"),
                ("running", "running"),
                ("completed", "complete"),
                ("failed", "failed"),
            )
            if counts[key]
        ]
        return " / ".join(parts) if parts else "0 jobs active"
    return ""


def _strict_final_readiness_terms(payload: dict[str, Any]) -> list[str]:
    if not payload:
        return [
            "P0 restore:",
            "SGLang matrix:",
            "wait-complete",
        ]
    p0 = payload.get("p0") if isinstance(payload.get("p0"), dict) else {}
    bridge = (
        payload.get("bridge")
        if isinstance(payload.get("bridge"), dict)
        else {}
    )
    matrix = payload.get("matrix") if isinstance(payload.get("matrix"), dict) else {}
    event_template = (
        payload.get("p0_physical_tensor_copy_event_template")
        if isinstance(payload.get("p0_physical_tensor_copy_event_template"), dict)
        else {}
    )
    live_copy_events = (
        payload.get("p0_live_copy_events")
        if isinstance(payload.get("p0_live_copy_events"), dict)
        else {}
    )
    p2p_plan = (
        payload.get("p0_live_p2p_collection_plan")
        if isinstance(payload.get("p0_live_p2p_collection_plan"), dict)
        else {}
    )
    promotion = (
        payload.get("p0_promotion")
        if isinstance(payload.get("p0_promotion"), dict)
        else {}
    )
    cases = matrix.get("cases") if isinstance(matrix.get("cases"), dict) else {}
    wait = (
        cases.get("same_write_wait_complete")
        if isinstance(cases.get("same_write_wait_complete"), dict)
        else {}
    )
    schema_missing = int(_float(p0.get("missing_data_plane_rows"))) + int(
        _float(p0.get("missing_timing_source_rows"))
    )
    required_event_kinds = (
        event_template.get("required_event_kinds")
        if isinstance(event_template.get("required_event_kinds"), dict)
        else {}
    )
    copy_rows = int(_float(p0.get("physical_tensor_copy_rows")))
    if _bool_text(promotion.get("promotion_ready")):
        promoted_copy_rows = int(_float(promotion.get("physical_tensor_copy_rows")))
        if promoted_copy_rows:
            copy_rows = promoted_copy_rows
    live_event_kinds = (
        live_copy_events.get("event_rows_by_kind")
        if isinstance(live_copy_events.get("event_rows_by_kind"), dict)
        else {}
    )
    measured_h2d = int(_float(live_event_kinds.get("live_host_h2d_copy")))
    measured_p2p = int(_float(p2p_plan.get("p2p_live_copy_event_rows")))
    measured_total = measured_h2d + measured_p2p
    p2p_tail = (
        f"P2P {measured_p2p}"
        if measured_p2p
        else f"P2P blocked {int(_float(live_copy_events.get('skipped_rows')))}"
    )
    return [
        (
            f"P0 restore: {int(_float(p0.get('restore_rows')))} rows，"
            f"copy={copy_rows}，"
            f"bridge={int(_float(bridge.get('bridge_eligible_rows')))}，"
            f"schema missing={schema_missing}"
        ),
        (
            "P0 live-copy events: "
            f"required {int(_float(event_template.get('required_rows')))}，"
            f"measured {measured_total}，"
            f"H2D {measured_h2d}，"
            f"{p2p_tail}"
        ),
        (
            f"SGLang matrix: accepted {int(_float(matrix.get('accepted_pairs')))}/"
            f"{int(_float(matrix.get('required_pairs')))}"
        ),
        (
            f"wait-complete {int(_float(wait.get('accepted_pairs')))}/"
            f"{int(_float(wait.get('required_repeats')))}"
        ),
    ]


def _goal_audit_status_term(payload: dict[str, Any]) -> str:
    if not payload:
        return "Goal audit:"
    if _bool_text(payload.get("strict_final_ready")):
        return "Goal audit: 17/17 pass，0 partial，0 blocked"
    if _bool_text(payload.get("host_matrix_ready")):
        return "Goal audit: 16/17 pass，0 partial，1 blocked"
    return "Goal audit: 15/17 pass，1 partial，1 blocked"


def _overall_status(checks: list[RequirementCheck]) -> str:
    if not checks:
        return MISSING
    worst = max(checks, key=lambda item: STATUS_ORDER.get(item.status, 99))
    return worst.status


def check_workload_definition() -> list[RequirementCheck]:
    for import_root in (REPO_ROOT / "python", REPO_ROOT):
        import_root_key = str(import_root)
        if import_root_key not in sys.path:
            sys.path.insert(0, import_root_key)
    try:
        from benchmark.agentic_workflows.config import SCENARIOS
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return [
            RequirementCheck(
                "workflow_13131_config",
                MISSING,
                "1-3-1-3-1 staged workflow scenarios are registered.",
                f"Could not import benchmark.agentic_workflows.config: {exc}",
                "Fix the benchmark package import before rerunning the audit.",
            )
        ]

    stress_names = [
        "staged_13131_policy_stress_no_backup",
        "staged_13131_policy_stress_host_only",
        "staged_13131_policy_stress_gpu_replica",
        "staged_13131_policy_stress_aegiskv",
    ]
    missing = [name for name in stress_names if name not in SCENARIOS]
    scenarios = [SCENARIOS[name] for name in stress_names if name in SCENARIOS]
    comparable = bool(scenarios) and all(
        scenario.workflow_shape == (1, 3, 1, 3, 1)
        and scenario.stage_long_decode_agent_index == 0
        and scenario.long_decode_output_multiplier >= 8.0
        and scenario.output_len >= 128
        and scenario.critical_agent_ratio == 0.0
        and scenario.replay_critical_only
        and scenario.failover_injection == "soft"
        and scenario.failure_after_output_tokens == 128
        for scenario in scenarios
    )
    policy_set = {scenario.aegis_policy for scenario in scenarios}
    status = PASS if not missing and comparable and len(policy_set) == 4 else MISSING
    checks = [
        RequirementCheck(
            "workflow_13131_config",
            status,
            (
                "Policy experiments use a comparable 1-3-1-3-1 workflow with "
                "singleton stages Critical and one long-decode Critical agent "
                "inside each width-3 stage."
            ),
            (
                f"Registered stress scenarios={len(scenarios)}/4; "
                f"policies={sorted(policy_set)}; missing={missing}; "
                "shape=(1,3,1,3,1), long_decode_multiplier>=8, "
                "soft failure after 128 output tokens."
            ),
            (
                "Register the four staged_13131_policy_stress_* scenarios with "
                "matching shape, seeds, failure timing, and policy labels."
                if status != PASS
                else "Keep this scenario family as the canonical policy sweep input."
            ),
        )
    ]

    utils_path = REPO_ROOT / "benchmark" / "synthesis_requests" / "utils.py"
    text = utils_path.read_text(encoding="utf-8") if utils_path.exists() else ""
    has_stage_rule = "stage_width == 1 or decode_dominant" in text
    has_long_decode_rule = (
        "stage_local_idx == client.gsp_stage_long_decode_agent_index" in text
        and "gsp_long_decode_output_multiplier" in text
    )
    checks.append(
        RequirementCheck(
            "workflow_critical_agent_generation",
            PASS if has_stage_rule and has_long_decode_rule else MISSING,
            (
                "Generated agentic requests mark singleton stages and the "
                "long-decode fan-out agent as Critical."
            ),
            (
                "benchmark/synthesis_requests/utils.py contains staged "
                "decode_dominant and stage_critical generation rules."
                if has_stage_rule and has_long_decode_rule
                else "Could not find the staged Critical-agent generation rules."
            ),
            (
                "Preserve the stage_critical rule when changing the generated "
                "shared-prefix workload."
                if has_stage_rule and has_long_decode_rule
                else "Add explicit staged Critical-agent assignment before using the workload."
            ),
        )
    )
    try:
        from benchmark.synthesis_requests.utils import (
            ClientConfig,
            ServerConfig,
            generate_shared_prefix_workload,
        )

        generated_scenario = SCENARIOS["staged_13131_policy_stress_aegiskv"]
        generated_client = ClientConfig(
            dataset_name="generated-shared-prefix",
            seed=generated_scenario.seed,
            request_rate=float("inf"),
            max_concurrency=generated_scenario.max_concurrency,
            gsp_num_groups=generated_scenario.groups,
            gsp_prompts_per_group=generated_scenario.prompts_per_group,
            gsp_system_prompt_len=generated_scenario.system_prompt_len,
            gsp_question_len=generated_scenario.question_len,
            gsp_output_len=generated_scenario.output_len,
            gsp_range_ratio=generated_scenario.range_ratio,
            gsp_fast_prepare=True,
            gsp_send_routing_key=generated_scenario.send_routing_key,
            gsp_num_turns=generated_scenario.turns,
            gsp_ordered=True,
            gsp_workflow_shape=generated_scenario.workflow_shape,
            gsp_stage_long_decode_agent_index=(
                generated_scenario.stage_long_decode_agent_index
            ),
            gsp_long_decode_output_multiplier=(
                generated_scenario.long_decode_output_multiplier
            ),
            gsp_critical_agent_ratio=generated_scenario.critical_agent_ratio,
            gsp_failover_replay=generated_scenario.failover_replay,
            gsp_failover_replay_critical_only=(
                generated_scenario.replay_critical_only
            ),
            gsp_failover_injection=generated_scenario.failover_injection,
            gsp_failover_retry_rank_offset=(
                generated_scenario.failover_retry_rank_offset
            ),
            gsp_failover_retry_policy=generated_scenario.failover_retry_policy,
            gsp_aegis_policy=generated_scenario.aegis_policy,
            gsp_alpha_policy_path=str(generated_scenario.alpha_policy_path or ""),
            gsp_failure_rate_per_window=generated_scenario.failure_rate_per_window,
            gsp_failure_after_output_tokens=(
                generated_scenario.failure_after_output_tokens
            ),
            gsp_backup_kv=generated_scenario.backup_kv,
        )
        generated_server = ServerConfig(
            model_path="mock-model",
            host="127.0.0.1",
            port=generated_scenario.start_port,
            dp_size=generated_scenario.dp_size,
            tp_size=1,
            pp_size=1,
            raw={"model_path": "mock-model"},
            experiment_name="audit-workload-generation",
        )
        generated_requests = generate_shared_prefix_workload(
            generated_client,
            generated_server,
            logger=None,
        )
        generated_slots = {
            (req.stage_id, req.stage_width, req.stage_local_agent_index)
            for req in generated_requests
        }
        expected_slots = {
            (0, 1, 0): (True, False),
            (1, 3, 0): (True, True),
            (1, 3, 1): (False, False),
            (1, 3, 2): (False, False),
            (2, 1, 0): (True, False),
            (3, 3, 0): (True, True),
            (3, 3, 1): (False, False),
            (3, 3, 2): (False, False),
            (4, 1, 0): (True, False),
        }
        generated_violations: list[str] = []
        generated_workflows = {req.workflow_id for req in generated_requests}
        generated_shape_labels = {req.workflow_shape for req in generated_requests}
        long_decode_outputs: list[int] = []
        fanout_noncritical_outputs: list[int] = []
        for req in generated_requests:
            slot = (req.stage_id, req.stage_width, req.stage_local_agent_index)
            expected = expected_slots.get(slot)
            if expected is None:
                generated_violations.append(f"unexpected:{slot}")
                continue
            if req.critical_agent != expected[0] or req.decode_dominant != expected[1]:
                generated_violations.append(
                    f"{req.rid}:{slot}:critical={req.critical_agent}:"
                    f"decode={req.decode_dominant}"
                )
            if req.stage_width == 3 and req.decode_dominant:
                long_decode_outputs.append(int(req.output_len))
            elif req.stage_width == 3 and not req.critical_agent:
                fanout_noncritical_outputs.append(int(req.output_len))
        critical_rows = sum(1 for req in generated_requests if req.critical_agent)
        noncritical_rows = len(generated_requests) - critical_rows
        long_decode_rows = sum(1 for req in generated_requests if req.decode_dominant)
        expected_request_count = (
            generated_scenario.groups * sum(generated_scenario.workflow_shape)
        )
        max_noncritical_output = max(fanout_noncritical_outputs or [0])
        min_long_decode_output = min(long_decode_outputs or [0])
        generated_ready = (
            len(generated_requests) == expected_request_count
            and len(generated_workflows) == generated_scenario.groups
            and generated_slots == set(expected_slots)
            and not generated_violations
            and critical_rows == generated_scenario.groups * 5
            and noncritical_rows == generated_scenario.groups * 4
            and long_decode_rows == generated_scenario.groups * 2
            and generated_shape_labels == {"1-3-1-3-1"}
            and min_long_decode_output >= 5 * max(max_noncritical_output, 1)
        )
        checks.append(
            RequirementCheck(
                "workflow_generated_request_shape",
                PASS if generated_ready else PARTIAL,
                (
                    "The canonical staged workload generator produces the "
                    "intended 1-3-1-3-1 Critical/Non-Critical request set."
                ),
                (
                    f"scenario={generated_scenario.name}; generated_rows="
                    f"{len(generated_requests)}; expected_rows="
                    f"{expected_request_count}; workflows="
                    f"{len(generated_workflows)}; slots="
                    f"{len(generated_slots)}; critical_rows="
                    f"{critical_rows}; noncritical_rows="
                    f"{noncritical_rows}; long_decode_rows="
                    f"{long_decode_rows}; shape_labels="
                    f"{sorted(generated_shape_labels)}; "
                    f"min_long_decode_output={min_long_decode_output}; "
                    f"max_fanout_noncritical_output="
                    f"{max_noncritical_output}; violations="
                    f"{generated_violations[:3]}."
                ),
                (
                    "Use this generated request set as the canonical "
                    "AegisKV policy workload input."
                    if generated_ready
                    else "Fix the staged workload generator before using policy results."
                ),
            )
        )
    except Exception as exc:  # pragma: no cover - defensive reporting path
        checks.append(
            RequirementCheck(
                "workflow_generated_request_shape",
                MISSING,
                (
                    "The canonical staged workload generator produces the "
                    "intended 1-3-1-3-1 Critical/Non-Critical request set."
                ),
                f"Could not generate staged workload requests: {exc}",
                "Fix synthesis_requests workload generation before rerunning the audit.",
            )
        )
    return checks


def check_motivation_figures() -> list[RequirementCheck]:
    checks: list[RequirementCheck] = []
    profile_rows = _read_csv(FIGURES / "data" / "measured" / "fig_critical_agent_profile.csv")
    if not profile_rows:
        checks.append(
            RequirementCheck(
                "motivation_critical_profile",
                MISSING,
                "Critical-agent profile figure is backed by measured trace data.",
                "Missing fig_critical_agent_profile.csv.",
                "Regenerate the staged 1-3-1-3-1 profile CSV.",
            )
        )
    else:
        singleton_critical = [
            row
            for row in profile_rows
            if int(_float(row.get("stage_width"))) == 1
            and _bool_text(row.get("critical_agent"))
        ]
        fanout_critical = [
            row
            for row in profile_rows
            if int(_float(row.get("stage_width"))) == 3
            and _bool_text(row.get("critical_agent"))
            and _bool_text(row.get("decode_dominant"))
        ]
        fanout_noncritical = [
            row
            for row in profile_rows
            if int(_float(row.get("stage_width"))) == 3
            and not _bool_text(row.get("critical_agent"))
        ]
        critical_tokens = max(
            [_float(row.get("mean_output_tokens")) for row in fanout_critical] or [0.0]
        )
        noncritical_tokens = max(
            [_float(row.get("mean_output_tokens")) for row in fanout_noncritical] or [0.0]
        )
        critical_latency = max(
            [_float(row.get("mean_latency_ms")) for row in fanout_critical] or [0.0]
        )
        noncritical_latency = max(
            [_float(row.get("mean_latency_ms")) for row in fanout_noncritical]
            or [0.0]
        )
        shape_by_stage = {
            int(_float(row.get("stage_id"))): int(_float(row.get("stage_width")))
            for row in profile_rows
        }
        trace_shape = tuple(width for _, width in sorted(shape_by_stage.items()))
        by_slot = {
            (
                int(_float(row.get("stage_id"))),
                int(_float(row.get("stage_width"))),
                int(_float(row.get("stage_local_agent_index"))),
            ): row
            for row in profile_rows
        }
        expected_slots = {
            (0, 1, 0): (True, False),
            (1, 3, 0): (True, True),
            (1, 3, 1): (False, False),
            (1, 3, 2): (False, False),
            (2, 1, 0): (True, False),
            (3, 3, 0): (True, True),
            (3, 3, 1): (False, False),
            (3, 3, 2): (False, False),
            (4, 1, 0): (True, False),
        }
        exact_criticality = len(by_slot) == len(expected_slots) and all(
            slot in by_slot
            and _bool_text(by_slot[slot].get("critical_agent")) == expected[0]
            and _bool_text(by_slot[slot].get("decode_dominant")) == expected[1]
            for slot, expected in expected_slots.items()
        )
        trace_run_dirs = {
            item
            for row in profile_rows
            for item in str(row.get("repeat_run_ids") or "").split(";")
            if item
        }
        trace_files_exist = bool(trace_run_dirs) and all(
            (Path(item) / "request_details.csv").exists() for item in trace_run_dirs
        )
        raw_trace_rows = 0
        raw_trace_workflows: set[str] = set()
        raw_trace_slots: set[tuple[int, int, int]] = set()
        raw_trace_violations: list[str] = []
        if trace_files_exist:
            for item in trace_run_dirs:
                details = Path(item) / "request_details.csv"
                for row in _read_csv(details):
                    if str(row.get("status") or "") != "completed":
                        continue
                    if _bool_text(row.get("aegis_cross_rank_retry")):
                        continue
                    if row.get("stage_id") in ("", None):
                        continue
                    stage_id = int(_float(row.get("stage_id")))
                    stage_width = int(_float(row.get("stage_width")))
                    stage_local_idx = int(_float(row.get("stage_local_agent_index")))
                    slot = (stage_id, stage_width, stage_local_idx)
                    raw_trace_rows += 1
                    raw_trace_slots.add(slot)
                    workflow_id = str(row.get("workflow_id") or "")
                    if workflow_id:
                        raw_trace_workflows.add(workflow_id)
                    expected = expected_slots.get(slot)
                    if expected is None:
                        raw_trace_violations.append(f"unexpected:{slot}")
                        continue
                    if (
                        _bool_text(row.get("critical_agent")) != expected[0]
                        or _bool_text(row.get("decode_dominant")) != expected[1]
                    ):
                        raw_trace_violations.append(
                            f"{row.get('rid', '')}:{slot}:"
                            f"critical={row.get('critical_agent')}:"
                            f"decode={row.get('decode_dominant')}"
                        )
        raw_trace_pattern_matches = (
            trace_files_exist
            and raw_trace_rows > 0
            and not raw_trace_violations
            and raw_trace_slots == set(expected_slots)
        )
        workflow_groups = max(
            [int(_float(row.get("runtime_workflow_groups"))) for row in profile_rows]
            or [0]
        )
        status = (
            PASS
            if trace_shape == (1, 3, 1, 3, 1)
            and len(singleton_critical) == 3
            and len(fanout_critical) == 2
            and len(fanout_noncritical) == 4
            and exact_criticality
            and trace_files_exist
            and raw_trace_pattern_matches
            and len(raw_trace_workflows) >= workflow_groups
            and workflow_groups >= 8
            and critical_tokens >= 2.0 * max(noncritical_tokens, 1.0)
            and critical_latency >= 2.0 * max(noncritical_latency, 1.0)
            else PARTIAL
        )
        checks.append(
            RequirementCheck(
                "motivation_critical_profile",
                status,
                (
                    "Motivation data shows Critical agents matter: singleton "
                    "stages are Critical, and the width-3 long-decode agent is "
                    "decode-dominant."
                ),
                (
                    f"rows={len(profile_rows)}; singleton_critical="
                    f"{len(singleton_critical)}; fanout_long_critical="
                    f"{len(fanout_critical)}; fanout_noncritical="
                    f"{len(fanout_noncritical)}; trace_shape="
                    f"{'-'.join(str(item) for item in trace_shape)}; "
                    f"exact_criticality={exact_criticality}; "
                    f"trace_run_dirs={len(trace_run_dirs)}; "
                    f"trace_files_exist={trace_files_exist}; "
                    f"raw_trace_rows={raw_trace_rows}; "
                    f"raw_trace_slots={len(raw_trace_slots)}; "
                    f"raw_trace_workflows={len(raw_trace_workflows)}; "
                    f"raw_trace_pattern_matches="
                    f"{raw_trace_pattern_matches}; "
                    f"raw_trace_violations={raw_trace_violations[:3]}; "
                    f"runtime_workflow_groups={workflow_groups}; "
                    f"max long Critical output={critical_tokens:.1f}; "
                    f"max fanout Non-Critical output="
                    f"{noncritical_tokens:.1f}; max long Critical latency="
                    f"{critical_latency:.1f} ms; max fanout Non-Critical latency="
                    f"{noncritical_latency:.1f} ms."
                ),
                (
                    "Use this as motivation evidence, then add real-system "
                    "calibration for strict-final."
                    if status == PASS
                    else "Regenerate the profile so the long-decode Critical agent is clear."
                ),
            )
        )

    priority_rows = _read_csv(
        FIGURES / "data" / "measured" / "fig_critical_backup_priority.csv"
    )
    if not priority_rows:
        checks.append(
            RequirementCheck(
                "motivation_backup_priority",
                MISSING,
                "Backup-priority figure compares Critical-first against alternatives.",
                "Missing fig_critical_backup_priority.csv.",
                "Regenerate the backup-priority counterfactual CSV.",
            )
        )
    else:
        by_budget: dict[float, dict[str, float]] = {}
        for row in priority_rows:
            budget = _float(row.get("backup_budget_fraction"))
            policy = str(row.get("placement_policy") or "")
            by_budget.setdefault(budget, {})[policy] = _float(
                row.get("normalized_wct_after_failure")
            )
        expected_policies = {
            "critical-first",
            "size-first",
            "random",
            "non-critical-first",
        }
        observed_policies = {
            str(row.get("placement_policy") or "") for row in priority_rows
        }
        budgets = sorted(by_budget)
        complete_grid = bool(budgets) and all(
            set(values) == expected_policies for values in by_budget.values()
        )
        mid_budgets = [budget for budget in budgets if 0.0 < budget < 1.0]
        comparable_budgets = [
            budget
            for budget, values in by_budget.items()
            if 0.0 < budget < 1.0
            and "critical-first" in values
            and "non-critical-first" in values
            and values["critical-first"] < values["non-critical-first"]
        ]
        critical_best_budgets = [
            budget
            for budget, values in by_budget.items()
            if 0.0 < budget < 1.0
            and values
            and min(values, key=values.get) == "critical-first"
        ]
        improvements = []
        for budget in mid_budgets:
            values = by_budget.get(budget, {})
            critical = values.get("critical-first", 0.0)
            noncritical = values.get("non-critical-first", 0.0)
            if noncritical > 0:
                improvements.append(max((noncritical - critical) / noncritical, 0.0))
        mean_improvement_pct = (
            100.0 * sum(improvements) / len(improvements) if improvements else 0.0
        )
        max_improvement_pct = 100.0 * max(improvements) if improvements else 0.0
        trace_run_dirs = {
            item
            for row in priority_rows
            for item in str(row.get("repeat_run_ids") or "").split(";")
            if item
        }
        trace_files_exist = bool(trace_run_dirs) and all(
            (Path(item) / "request_details.csv").exists() for item in trace_run_dirs
        )
        workflow_traces = max(
            [int(_float(row.get("runtime_workflow_traces"))) for row in priority_rows]
            or [0]
        )
        sources = {str(row.get("source") or "") for row in priority_rows}
        tiers = {str(row.get("evidence_tier") or "") for row in priority_rows}
        status = (
            PASS
            if observed_policies == expected_policies
            and len(budgets) == 11
            and complete_grid
            and len(critical_best_budgets) == len(mid_budgets) == 9
            and len(comparable_budgets) == len(mid_budgets)
            and mean_improvement_pct >= 10.0
            and trace_files_exist
            and workflow_traces >= 8
            and sources == {"sglsim_agentic_online_trace_placement_counterfactual"}
            and tiers == {"sglsim-trace-derived"}
            else PARTIAL
        )
        checks.append(
            RequirementCheck(
                "motivation_backup_priority",
                status,
                (
                    "Motivation data supports prioritizing Critical agents "
                    "under limited GPUReplica backup budget."
                ),
                (
                    f"rows={len(priority_rows)}; policies={sorted(observed_policies)}; "
                    f"budget_points={len(budgets)}; complete_grid={complete_grid}; "
                    f"critical_best_mid_budgets={len(critical_best_budgets)}/"
                    f"{len(mid_budgets)}; beats_noncritical_mid_budgets="
                    f"{len(comparable_budgets)}/{len(mid_budgets)}; "
                    f"mean_improvement_vs_noncritical={mean_improvement_pct:.1f}%; "
                    f"max_improvement_vs_noncritical={max_improvement_pct:.1f}%; "
                    f"runtime_workflow_traces={workflow_traces}; "
                    f"trace_run_dirs={len(trace_run_dirs)}; "
                    f"trace_files_exist={trace_files_exist}; "
                    f"sources={sorted(sources)}; evidence_tiers={sorted(tiers)}."
                ),
                (
                    "Keep this figure in the motivation section."
                    if status == PASS
                    else "Rerun the placement sweep with enough nonzero budget points."
                ),
            )
        )
    return checks


def check_alpha_policy() -> list[RequirementCheck]:
    checks: list[RequirementCheck] = []
    policy_path = (
        EXPERIMENTS / "sglsim_large" / "policies" / "aegiskv_alpha_mdp_policy.json"
    )
    policy_path_key = _normalize_artifact_path(policy_path)
    policy = _read_json(policy_path)
    policy_source = str(policy.get("source") or "")
    states = policy.get("states") if isinstance(policy.get("states"), list) else []
    training = policy.get("training") if isinstance(policy.get("training"), dict) else {}
    gamma = _float(training.get("gamma")) if policy else 0.0
    iterations = int(_float(training.get("iterations"))) if policy else 0
    by_pressure: dict[str, list[tuple[float, float]]] = {}
    for row in states:
        if not isinstance(row, dict):
            continue
        by_pressure.setdefault(str(row.get("critical_pressure")), []).append(
            (_float(row.get("failure_rate_per_window")), _float(row.get("alpha")))
        )
    monotonic = True
    for items in by_pressure.values():
        ordered = [alpha for _, alpha in sorted(items)]
        monotonic = monotonic and all(
            left <= right + 1e-9 for left, right in zip(ordered, ordered[1:])
        )
    fig3b_rows = _read_csv(
        FIGURES / "data" / "measured" / "fig_alpha_workload_failure_policy.csv"
    )
    fig3b_paths = {
        _normalize_artifact_path(row.get("policy_path"))
        for row in fig3b_rows
        if row.get("policy_path")
    }
    fig3b_sources = {
        str(row.get("source") or "")
        for row in fig3b_rows
        if str(row.get("source") or "")
    }
    fig3b_policy_path_match = bool(fig3b_rows) and fig3b_paths == {policy_path_key}
    fig3b_source_match = (
        bool(fig3b_rows) and bool(policy_source) and fig3b_sources == {policy_source}
    )
    runtime_selector_checks = 0
    runtime_selector_matches_policy = False
    runtime_selector_source_match = False
    choose_alpha = None
    try:
        for import_root in (REPO_ROOT / "python", REPO_ROOT):
            import_root_key = str(import_root)
            if import_root_key not in sys.path:
                sys.path.insert(0, import_root_key)
        from sglsim.srt.simulation.alpha_policy import choose_alpha as imported_choose_alpha

        choose_alpha = imported_choose_alpha
        runtime_selector_matches = bool(states)
        runtime_selector_sources: set[str] = set()
        for row in states:
            if not isinstance(row, dict):
                runtime_selector_matches = False
                continue
            decision = choose_alpha(
                critical_agent=True,
                decode_dominant=False,
                stage_width=1,
                critical_pressure=row.get("critical_pressure_value"),
                failure_rate_per_window=row.get("failure_rate_per_window"),
                policy_path=str(policy_path),
            )
            runtime_selector_checks += 1
            runtime_selector_sources.add(_normalize_artifact_path(decision.source))
            if abs(decision.alpha - _float(row.get("alpha"))) > 1e-9:
                runtime_selector_matches = False
        runtime_selector_matches_policy = (
            runtime_selector_checks == len(states) and runtime_selector_matches
        )
        runtime_selector_source_match = runtime_selector_sources == {policy_path_key}
    except Exception:
        runtime_selector_checks = 0
        runtime_selector_matches_policy = False
        runtime_selector_source_match = False
    policy_ready = (
        len(states) >= 15
        and gamma > 0
        and iterations > 0
        and monotonic
        and fig3b_policy_path_match
        and fig3b_source_match
        and runtime_selector_matches_policy
        and runtime_selector_source_match
    )
    checks.append(
        RequirementCheck(
            "mdp_alpha_policy_table",
            PASS if policy_ready else MISSING,
            (
                "Offline MDP/value-iteration policy maps workload pressure and "
                "failure rate to GPUReplica alpha."
            ),
            (
                f"states={len(states)}; pressure_bins={sorted(by_pressure)}; "
                f"gamma={gamma}; iterations={iterations}; "
                f"alpha_nondecreasing_with_failure={monotonic}; "
                f"fig3b_rows={len(fig3b_rows)}; "
                f"fig3b_policy_path_match={fig3b_policy_path_match}; "
                f"fig3b_source_match={fig3b_source_match}; "
                f"runtime_selector_checks={runtime_selector_checks}; "
                f"runtime_selector_matches_policy="
                f"{runtime_selector_matches_policy}; "
                f"runtime_selector_source_match={runtime_selector_source_match}."
            ),
            (
                "Use this policy table as the online alpha controller input."
                if policy_ready
                else "Regenerate the MDP policy table and Fig. 3b from the same policy JSON before claiming adaptive alpha."
            ),
        )
    )

    online_rows = _read_csv(
        FIGURES / "data" / "measured" / "fig_online_alpha_adaptation.csv"
    )
    critical = [
        row
        for row in online_rows
        if str(row.get("request_class") or "").lower().startswith("critical")
    ]
    noncritical = [
        row
        for row in online_rows
        if str(row.get("request_class") or "").lower().startswith("non-critical")
    ]
    critical_positive = any(
        _float(row.get("effective_gpu_replica_alpha")) > 0 for row in critical
    )
    noncritical_zero = bool(noncritical) and all(
        abs(_float(row.get("effective_gpu_replica_alpha"))) < 1e-9
        for row in noncritical
    )
    load_levels = {str(row.get("load_level") or "") for row in online_rows}
    failure_rates = {_float(row.get("failure_rate_per_window")) for row in online_rows}
    online_policy_sources = {
        _normalize_artifact_path(row.get("policy_source"))
        for row in online_rows
        if row.get("policy_source")
    }
    online_policy_source_match = bool(online_rows) and online_policy_sources == {
        policy_path_key
    }
    expected_alpha_by_state = {
        (
            str(row.get("load_level") or ""),
            round(_float(row.get("failure_rate_per_window")), 12),
        ): _float(row.get("policy_alpha"))
        for row in online_rows
    }
    alpha_sweep_manifest = (
        EXPERIMENTS
        / "sglsim_large"
        / "results"
        / "agentic_workflows"
        / "sglsim-staged-13131-alpha-online-sweep-20260702"
        / "agentic_workflow_manifest.json"
    )
    raw_manifest = _read_json(alpha_sweep_manifest)
    raw_trace_files = 0
    raw_critical_rows = 0
    raw_noncritical_rows = 0
    raw_critical_gpu_tokens = 0
    raw_noncritical_gpu_tokens = 0
    raw_policy_sources: set[str] = set()
    raw_alpha_matches_fig3c = bool(raw_manifest)
    runtime_trace_checks = 0
    runtime_trace_alpha_matches_raw = bool(raw_manifest) and choose_alpha is not None
    for item in raw_manifest.get("plan", []):
        if not isinstance(item, dict):
            continue
        run_dir = Path(str(item.get("run_dir") or ""))
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        details = run_dir / "request_details.csv"
        if not details.exists():
            raw_alpha_matches_fig3c = False
            continue
        raw_trace_files += 1
        expected_alpha = expected_alpha_by_state.get(
            (
                str(item.get("load_level") or ""),
                round(_float(item.get("failure_rate_per_window")), 12),
            )
        )
        raw_rows = [
            row
            for row in _read_csv(details)
            if str(row.get("status") or "") == "completed"
            and not _bool_text(row.get("aegis_cross_rank_retry"))
        ]
        raw_alphas = {
            round(_float(row.get("aegis_gpu_budget_fraction")), 12)
            for row in raw_rows
        }
        if expected_alpha is None or raw_alphas != {round(expected_alpha, 12)}:
            raw_alpha_matches_fig3c = False
        for row in raw_rows:
            raw_policy_sources.add(
                _normalize_artifact_path(row.get("aegis_alpha_policy_source"))
            )
            if choose_alpha is None:
                runtime_trace_alpha_matches_raw = False
            else:
                decision = choose_alpha(
                    critical_agent=_bool_text(row.get("critical_agent")),
                    decode_dominant=_bool_text(row.get("decode_dominant")),
                    stage_width=int(_float(row.get("stage_width"), 1.0)),
                    critical_pressure=row.get("aegis_critical_pressure"),
                    failure_rate_per_window=row.get("aegis_failure_rate_per_window"),
                    policy_path=str(policy_path),
                )
                runtime_trace_checks += 1
                if (
                    abs(
                        decision.alpha
                        - _float(row.get("aegis_gpu_budget_fraction"))
                    )
                    > 1e-9
                    or _normalize_artifact_path(decision.source) != policy_path_key
                ):
                    runtime_trace_alpha_matches_raw = False
            if _bool_text(row.get("critical_agent")):
                raw_critical_rows += 1
                raw_critical_gpu_tokens += int(
                    _float(row.get("aegis_gpu_replica_tokens"))
                )
            else:
                raw_noncritical_rows += 1
                raw_noncritical_gpu_tokens += int(
                    _float(row.get("aegis_gpu_replica_tokens"))
                )
    raw_policy_source_match = raw_trace_files > 0 and raw_policy_sources == {
        policy_path_key
    }
    raw_critical_gpu_tokens_positive = raw_critical_gpu_tokens > 0
    raw_noncritical_gpu_tokens_zero = (
        raw_noncritical_rows > 0 and raw_noncritical_gpu_tokens == 0
    )
    critical_policy_alpha_matches_online = bool(critical) and all(
        abs(_float(row.get("policy_alpha")) - _float(row.get("online_observed_alpha")))
        < 1e-9
        and abs(
            _float(row.get("policy_alpha"))
            - _float(row.get("effective_gpu_replica_alpha"))
        )
        < 1e-9
        for row in critical
    )
    online_ready = (
        critical_positive
        and noncritical_zero
        and len(load_levels) >= 2
        and len(failure_rates) >= 2
        and online_policy_source_match
        and critical_policy_alpha_matches_online
        and raw_trace_files >= 4
        and raw_policy_source_match
        and raw_alpha_matches_fig3c
        and runtime_trace_alpha_matches_raw
        and raw_critical_gpu_tokens_positive
        and raw_noncritical_gpu_tokens_zero
    )
    checks.append(
        RequirementCheck(
            "online_alpha_adaptation",
            PASS if online_ready else PARTIAL,
            (
                "Online serving sweep applies alpha only to Critical requests "
                "and changes alpha with workload/failure state."
            ),
            (
                f"rows={len(online_rows)}; critical_rows={len(critical)}; "
                f"noncritical_rows={len(noncritical)}; load_levels="
                f"{sorted(load_levels)}; failure_rates={sorted(failure_rates)}; "
                f"online_policy_source_match={online_policy_source_match}; "
                f"critical_policy_alpha_matches_online="
                f"{critical_policy_alpha_matches_online}; "
                f"noncritical_effective_alpha_zero={noncritical_zero}; "
                f"raw_trace_files={raw_trace_files}; raw_critical_rows="
                f"{raw_critical_rows}; raw_noncritical_rows={raw_noncritical_rows}; "
                f"raw_policy_source_match={raw_policy_source_match}; "
                f"raw_alpha_matches_fig3c={raw_alpha_matches_fig3c}; "
                f"runtime_trace_checks={runtime_trace_checks}; "
                f"runtime_trace_alpha_matches_raw="
                f"{runtime_trace_alpha_matches_raw}; "
                f"raw_critical_gpu_tokens={raw_critical_gpu_tokens}; "
                f"raw_noncritical_gpu_tokens={raw_noncritical_gpu_tokens}."
            ),
            (
                "Keep this figure as the online alpha evidence."
                if online_ready
                else "Rerun the online alpha sweep from the current MDP policy and preserve Critical/Non-Critical rows."
            ),
        )
    )
    return checks


def check_failure_replay_semantics() -> list[RequirementCheck]:
    suite_dir = (
        EXPERIMENTS
        / "sglsim_large"
        / "results"
        / "agentic_workflows"
        / "sglsim-staged-13131-policy-stress-v2-r3-20260702"
    )
    if not suite_dir.exists():
        return [
            RequirementCheck(
                "failure_replay_semantics",
                MISSING,
                (
                    "The current policy suite must inject soft failures at a "
                    "controlled token boundary and replay only Critical agents."
                ),
                f"Missing policy stress suite directory: {suite_dir}",
                "Run the staged_13131_policy_stress_v2 suite and rerun the audit.",
            )
        ]

    config_paths = sorted(suite_dir.glob("*/configs/*/trial_*/client.json"))
    request_paths = sorted(suite_dir.glob("*/runs/trial_*/*/request_details.csv"))
    config_violations: list[str] = []
    for path in config_paths:
        cfg = _read_json(path)
        if cfg.get("gsp_workflow_shape") != [1, 3, 1, 3, 1]:
            config_violations.append(f"{path.name}:shape={cfg.get('gsp_workflow_shape')}")
        if cfg.get("gsp_failover_replay") is not True:
            config_violations.append(f"{path.name}:replay={cfg.get('gsp_failover_replay')}")
        if cfg.get("gsp_failover_replay_critical_only") is not True:
            config_violations.append(
                f"{path.name}:critical_only={cfg.get('gsp_failover_replay_critical_only')}"
            )
        if cfg.get("gsp_failover_injection") != "soft":
            config_violations.append(
                f"{path.name}:injection={cfg.get('gsp_failover_injection')}"
            )
        if cfg.get("gsp_failover_retry_policy") != "peer":
            config_violations.append(
                f"{path.name}:retry_policy={cfg.get('gsp_failover_retry_policy')}"
            )
        if int(_float(cfg.get("gsp_failure_after_output_tokens"))) != 128:
            config_violations.append(
                f"{path.name}:failure_after={cfg.get('gsp_failure_after_output_tokens')}"
            )
        if int(_float(cfg.get("gsp_stage_long_decode_agent_index"))) != 0:
            config_violations.append(
                f"{path.name}:long_decode_idx={cfg.get('gsp_stage_long_decode_agent_index')}"
            )
        if _float(cfg.get("gsp_long_decode_output_multiplier")) < 8.0:
            config_violations.append(
                f"{path.name}:long_decode_mult={cfg.get('gsp_long_decode_output_multiplier')}"
            )

    normal_rows = 0
    retry_rows = 0
    critical_normal_rows = 0
    noncritical_normal_rows = 0
    critical_retry_rows = 0
    noncritical_retry_rows = 0
    soft_retry_rows = 0
    hard_retry_rows = 0
    failover_retry_rows = 0
    restore_rows = 0
    recompute_retry_rows = 0
    long_decode_retry_rows = 0
    retry_slots: set[tuple[int, int, int]] = set()
    replay_actions: set[str] = set()
    replay_data_planes: set[str] = set()
    request_violations: list[str] = []
    pre_failure_boundary_matches = True

    for path in request_paths:
        rows = _read_csv(path)
        for row in rows:
            if str(row.get("status") or "") != "completed":
                continue
            is_retry = _bool_text(row.get("aegis_cross_rank_retry"))
            is_critical = _bool_text(row.get("critical_agent"))
            if is_retry:
                retry_rows += 1
                retry_slots.add(
                    (
                        int(_float(row.get("stage_id"))),
                        int(_float(row.get("stage_width"))),
                        int(_float(row.get("stage_local_agent_index"))),
                    )
                )
                if is_critical:
                    critical_retry_rows += 1
                else:
                    noncritical_retry_rows += 1
                    request_violations.append(f"noncritical_retry:{row.get('rid', '')}")
                if _bool_text(row.get("aegis_soft_failure_injected")):
                    soft_retry_rows += 1
                else:
                    request_violations.append(f"soft_missing:{row.get('rid', '')}")
                if _bool_text(row.get("aegis_hard_failure_injected")):
                    hard_retry_rows += 1
                    request_violations.append(f"hard_retry:{row.get('rid', '')}")
                if _bool_text(row.get("aegis_failover_replay")):
                    failover_retry_rows += 1
                else:
                    request_violations.append(f"failover_missing:{row.get('rid', '')}")
                if _bool_text(row.get("decode_dominant")):
                    long_decode_retry_rows += 1
                action = str(row.get("aegis_replay_restore_action") or "none")
                data_plane = str(row.get("aegis_replay_restore_data_plane") or "none")
                replay_actions.add(action)
                replay_data_planes.add(data_plane)
                if action == "recompute":
                    recompute_retry_rows += 1
                elif action != "none":
                    restore_rows += 1
                generated = int(_float(row.get("generated_tokens")))
                pre_failure = int(
                    _float(row.get("aegis_candidate_pre_failure_output_tokens"))
                )
                if pre_failure != min(generated, 128):
                    pre_failure_boundary_matches = False
                    request_violations.append(
                        f"pre_boundary:{row.get('rid', '')}:{pre_failure}:{generated}"
                    )
            else:
                normal_rows += 1
                if is_critical:
                    critical_normal_rows += 1
                else:
                    noncritical_normal_rows += 1
                if _bool_text(row.get("aegis_failover_replay")):
                    request_violations.append(f"normal_failover:{row.get('rid', '')}")
                if _bool_text(row.get("aegis_soft_failure_injected")):
                    request_violations.append(f"normal_soft:{row.get('rid', '')}")
                if _bool_text(row.get("aegis_hard_failure_injected")):
                    request_violations.append(f"normal_hard:{row.get('rid', '')}")

    expected_retry_slots = {
        (0, 1, 0),
        (1, 3, 0),
        (2, 1, 0),
        (3, 3, 0),
        (4, 1, 0),
    }
    config_ready = len(config_paths) == 12 and not config_violations
    request_ready = (
        len(request_paths) == 12
        and normal_rows == 972
        and retry_rows == 540
        and critical_normal_rows == 540
        and noncritical_normal_rows == 432
        and critical_retry_rows == 540
        and noncritical_retry_rows == 0
        and soft_retry_rows == 540
        and hard_retry_rows == 0
        and failover_retry_rows == 540
        and long_decode_retry_rows == 216
        and retry_slots == expected_retry_slots
        and pre_failure_boundary_matches
        and not request_violations
        and restore_rows == 405
        and recompute_retry_rows == 135
    )
    ready = config_ready and request_ready
    return [
        RequirementCheck(
            "failure_replay_semantics",
            PASS if ready else PARTIAL,
            (
                "Failure replay uses a controlled soft-failure boundary and "
                "only replays Critical agents in the 1-3-1-3-1 policy suite."
            ),
            (
                f"config_files={len(config_paths)}; config_ready={config_ready}; "
                f"request_files={len(request_paths)}; normal_rows={normal_rows}; "
                f"retry_rows={retry_rows}; critical_normal_rows={critical_normal_rows}; "
                f"noncritical_normal_rows={noncritical_normal_rows}; "
                f"critical_retry_rows={critical_retry_rows}; "
                f"noncritical_retry_rows={noncritical_retry_rows}; "
                f"soft_retry_rows={soft_retry_rows}; hard_retry_rows={hard_retry_rows}; "
                f"failover_retry_rows={failover_retry_rows}; "
                f"long_decode_retry_rows={long_decode_retry_rows}; "
                f"replay_slots={len(retry_slots)}; "
                f"pre_failure_boundary_matches={pre_failure_boundary_matches}; "
                f"restore_rows={restore_rows}; recompute_retry_rows={recompute_retry_rows}; "
                f"actions={sorted(replay_actions)}; data_planes={sorted(replay_data_planes)}; "
                f"config_violations={config_violations[:3]}; "
                f"request_violations={request_violations[:3]}."
            ),
            (
                "Use this as the raw trace evidence for failure timing and Critical-only replay."
                if ready
                else "Regenerate the stress-v2 policy suite with soft, Critical-only replay after 128 output tokens."
            ),
        )
    ]


def check_policy_and_physical_dataplane() -> list[RequirementCheck]:
    checks: list[RequirementCheck] = []
    wct_rows = _read_csv(FIGURES / "data" / "measured" / "fig_overall_wct.csv")
    recovery_rows = _read_csv(
        FIGURES / "data" / "measured" / "fig_recovery_latency.csv"
    )
    policies = {str(row.get("policy") or row.get("label") or "") for row in wct_rows}
    recovery_policies = {
        str(row.get("policy") or row.get("label") or "") for row in recovery_rows
    }
    expected = {"no backup", "uniform Host", "uniform GPU", "AegisKV"}
    run_ids = {str(row.get("run_id") or "") for row in wct_rows + recovery_rows}
    has_stress_v2 = any("sglsim-staged-13131-policy-stress-v2-r3-20260702" in run_id for run_id in run_ids)
    checks.append(
        RequirementCheck(
            "online_policy_sweep_results",
            PASS if expected <= policies and expected <= recovery_policies and has_stress_v2 else PARTIAL,
            (
                "SGLSim implements and measures the current policy sweep on "
                "the 1-3-1-3-1 workload."
            ),
            (
                f"Fig.8 policies={sorted(policies)}; Fig.9 policies="
                f"{sorted(recovery_policies)}; stress_v2_run={has_stress_v2}."
            ),
            (
                "Use these rows for hybrid-ready policy evidence."
                if expected <= policies and expected <= recovery_policies and has_stress_v2
                else "Regenerate Fig. 8/9 from the stress-v2 policy suite."
            ),
        )
    )

    p0_rows = _read_csv(
        EXPERIMENTS
        / "sglsim_large"
        / "results"
        / "strict_final_readiness"
        / "p0_strict_final_readiness.csv"
    )
    p0_bridge_rows = _read_csv(
        EXPERIMENTS
        / "sglsim_large"
        / "results"
        / "strict_final_readiness"
        / "p0_physical_bridge_summary.csv"
    )
    restore_rows = sum(int(_float(row.get("restore_rows"))) for row in p0_rows)
    physical_rows = sum(
        int(_float(row.get("physical_tensor_copy_rows"))) for row in p0_rows
    )
    metadata_rows = sum(
        int(_float(row.get("nonfinal_restore_rows"))) for row in p0_rows
    )
    missing_data_plane_rows = sum(
        int(_float(row.get("missing_data_plane_rows"))) for row in p0_rows
    )
    missing_timing_source_rows = sum(
        int(_float(row.get("missing_timing_source_rows"))) for row in p0_rows
    )
    bridge_eligible_rows = sum(
        int(_float(row.get("bridge_eligible_rows"))) for row in p0_bridge_rows
    )
    bridge_missing_rows = sum(
        int(_float(row.get("bridge_missing_rows"))) for row in p0_bridge_rows
    )
    bridge_ready = (
        bool(p0_bridge_rows)
        and restore_rows > 0
        and bridge_eligible_rows == restore_rows
        and bridge_missing_rows == 0
    )
    final_ready = bool(p0_rows) and restore_rows > 0 and physical_rows == restore_rows
    effective_final_ready = final_ready
    effective_physical_rows = physical_rows
    effective_nonfinal_rows = metadata_rows
    dataplane_readiness = _read_json(
        EXPERIMENTS / "strict_final_dataplane_readiness.json"
    )
    latest_bound_slurm = _read_json(
        EXPERIMENTS
        / "sglang_small"
        / "results"
        / "strict_final_recovery_matrix_audit"
        / "strict_final_bound_slurm_jobs.json"
    )
    dataplane_evidence = ""
    if dataplane_readiness:
        matrix_report = (
            dataplane_readiness.get("matrix")
            if isinstance(dataplane_readiness.get("matrix"), dict)
            else {}
        )
        readiness_slurm_report = (
            dataplane_readiness.get("slurm")
            if isinstance(dataplane_readiness.get("slurm"), dict)
            else {}
        )
        slurm_report = latest_bound_slurm or readiness_slurm_report
        physical_restore_report = (
            dataplane_readiness.get("sglang_physical_restore")
            if isinstance(dataplane_readiness.get("sglang_physical_restore"), dict)
            else {}
        )
        p0_live_report = (
            dataplane_readiness.get("p0_live_audits")
            if isinstance(dataplane_readiness.get("p0_live_audits"), dict)
            else {}
        )
        candidate_report = (
            dataplane_readiness.get("p0_physical_tensor_copy_candidate_audit")
            if isinstance(
                dataplane_readiness.get("p0_physical_tensor_copy_candidate_audit"),
                dict,
            )
            else {}
        )
        candidate_generation_report = (
            dataplane_readiness.get(
                "p0_physical_tensor_copy_candidate_generation_audit"
            )
            if isinstance(
                dataplane_readiness.get(
                    "p0_physical_tensor_copy_candidate_generation_audit"
                ),
                dict,
            )
            else {}
        )
        event_template_report = (
            dataplane_readiness.get("p0_physical_tensor_copy_event_template")
            if isinstance(
                dataplane_readiness.get("p0_physical_tensor_copy_event_template"),
                dict,
            )
            else {}
        )
        live_copy_report = (
            dataplane_readiness.get("p0_live_copy_events")
            if isinstance(dataplane_readiness.get("p0_live_copy_events"), dict)
            else {}
        )
        p2p_plan_report = (
            dataplane_readiness.get("p0_live_p2p_collection_plan")
            if isinstance(dataplane_readiness.get("p0_live_p2p_collection_plan"), dict)
            else {}
        )
        promotion_report = (
            dataplane_readiness.get("p0_promotion")
            if isinstance(dataplane_readiness.get("p0_promotion"), dict)
            else {}
        )
        if (
            _bool_text(dataplane_readiness.get("strict_final_ready"))
            and _bool_text(promotion_report.get("promotion_ready"))
        ):
            effective_final_ready = True
            effective_physical_rows = int(
                _float(promotion_report.get("physical_tensor_copy_rows"))
            )
            effective_nonfinal_rows = int(
                _float(promotion_report.get("nonfinal_restore_rows"))
            )
        dataplane_evidence = (
            f"; dataplane_readiness_status={dataplane_readiness.get('status')}; "
            f"host_matrix_ready={dataplane_readiness.get('host_matrix_ready')}; "
            f"p0_bridge_ready={dataplane_readiness.get('bridge', {}).get('bridge_ready', bridge_ready)}; "
            f"p0_live_audits_consistent={dataplane_readiness.get('p0_live_audits_consistent')}; "
            f"p0_live_restore_rows={p0_live_report.get('restore_rows', 0)}; "
            f"p0_live_physical_tensor_copy_rows={p0_live_report.get('physical_tensor_copy_rows', 0)}; "
            f"p0_live_nonfinal_restore_rows={p0_live_report.get('nonfinal_restore_rows', 0)}; "
            f"p0_live_schema_complete={p0_live_report.get('schema_complete')}; "
            f"p0_candidate_status={candidate_report.get('status', '')}; "
            f"p0_candidate_ready={candidate_report.get('strict_final_candidate_ready')}; "
            f"p0_candidate_required_rows={candidate_report.get('required_rows', 0)}; "
            f"p0_candidate_accepted_rows={candidate_report.get('accepted_rows', 0)}; "
            f"p0_candidate_missing_rows={candidate_report.get('missing_rows', 0)}; "
            f"p0_candidate_invalid_rows={candidate_report.get('invalid_rows', 0)}; "
            f"p0_candidate_bridge_consistent={dataplane_readiness.get('p0_candidate_audit_bridge_consistent')}; "
            f"p0_candidate_generation_status={candidate_generation_report.get('status', '')}; "
            f"p0_candidate_generation_schema_version={candidate_generation_report.get('event_schema_version', '')}; "
            f"p0_candidate_generation_event_rows={candidate_generation_report.get('event_rows', 0)}; "
            f"p0_candidate_generation_missing_sources={candidate_generation_report.get('missing_event_source_count', 0)}; "
            f"p0_candidate_generation_bridge_consistent={dataplane_readiness.get('p0_candidate_generation_bridge_consistent')}; "
            f"p0_required_live_copy_events={event_template_report.get('required_rows', 0)}; "
            f"p0_required_event_kinds={event_template_report.get('required_event_kinds', {})}; "
            f"p0_measured_live_copy_events={live_copy_report.get('event_rows', 0)}; "
            f"p0_live_copy_event_kinds={live_copy_report.get('event_rows_by_kind', {})}; "
            f"p0_live_copy_skipped_rows={live_copy_report.get('skipped_rows', 0)}; "
            f"p0_live_copy_skipped_reasons={live_copy_report.get('skipped_rows_by_reason', {})}; "
            f"p0_live_p2p_plan_status={p2p_plan_report.get('status', '')}; "
            f"p0_live_p2p_plan_required_rows={p2p_plan_report.get('required_p2p_rows', 0)}; "
            f"p0_live_p2p_plan_ready={p2p_plan_report.get('p2p_ready')}; "
            f"p0_live_p2p_plan_remote={p2p_plan_report.get('remote_host', '')}:{p2p_plan_report.get('remote_workdir', '')}; "
            f"p0_promotion_status={promotion_report.get('status', '')}; "
            f"p0_promotion_ready={promotion_report.get('promotion_ready')}; "
            f"p0_promotion_physical_tensor_copy_rows={promotion_report.get('physical_tensor_copy_rows', 0)}; "
            f"p0_promotion_nonfinal_restore_rows={promotion_report.get('nonfinal_restore_rows', 0)}; "
            f"p0_promotion_bridge_consistent={dataplane_readiness.get('p0_promotion_bridge_consistent')}; "
            f"sglang_physical_restore_rows={matrix_report.get('physical_restore_rows', 0)}; "
            f"sglang_physical_materialized_rows={physical_restore_report.get('materialized_rows', 0)}; "
            f"sglang_physical_full_rows={physical_restore_report.get('full_prefetch_materialized_rows', 0)}; "
            f"sglang_physical_partial_rows={physical_restore_report.get('partial_async_materialized_rows', 0)}; "
            f"sglang_physical_hidden_control_rows={physical_restore_report.get('hidden_control_rows', 0)}; "
            f"sglang_matrix_pairs={matrix_report.get('accepted_pairs', 0)}/"
            f"{matrix_report.get('required_pairs', 0)}; "
            f"bound_slurm_status={slurm_report.get('overall_status', '')}"
        )
    checks.append(
        RequirementCheck(
            "strict_final_physical_dataplane",
            PASS if effective_final_ready else BLOCKED,
            (
                "P0 policy/recovery rows must use live physical tensor KV copy "
                "rather than metadata materialization before strict-final."
            ),
            (
                f"P0 restore_rows={restore_rows}; physical_tensor_copy_rows="
                f"{physical_rows}; nonfinal_restore_rows={metadata_rows}; "
                f"effective_physical_tensor_copy_rows={effective_physical_rows}; "
                f"effective_nonfinal_restore_rows={effective_nonfinal_rows}; "
                f"physical_bridge_rows={bridge_eligible_rows}; "
                f"physical_bridge_missing_rows={bridge_missing_rows}; "
                f"missing_data_plane_rows={missing_data_plane_rows}; "
                f"missing_timing_source_rows={missing_timing_source_rows}"
                f"{dataplane_evidence}."
            ),
            (
                "Strict-final is satisfied for P0 suites."
                if effective_final_ready
                else "Connect live retry recovery to physical Host H2D / peer P2P KV movement or collect an equivalent real-system recovery path, then rerun the P0 audit."
            ),
        )
    )

    matrix_audit = _read_json(
        EXPERIMENTS
        / "sglang_small"
        / "results"
        / "strict_final_recovery_matrix_audit"
        / "strict_final_recovery_matrix_audit.json"
    )
    matrix_audit_dir = (
        EXPERIMENTS / "sglang_small" / "results" / "strict_final_recovery_matrix_audit"
    )
    missing_job_rows = _read_csv(matrix_audit_dir / "strict_final_recovery_missing_jobs.csv")
    job_binding_rows = _read_csv(matrix_audit_dir / "strict_final_recovery_job_bindings.csv")
    bound_job_bindings = [
        row
        for row in job_binding_rows
        if _bool_text(row.get("bound")) and str(row.get("slurm_job_id") or "")
    ]
    sync_template_exists = (matrix_audit_dir / "sync_bound_jobs.sh").exists()
    sync_template_text = (
        (matrix_audit_dir / "sync_bound_jobs.sh").read_text(encoding="utf-8")
        if sync_template_exists
        else ""
    )
    sync_guard_complete_only = (
        "check_bound_slurm_jobs.py" in sync_template_text
        and "wait for completion before syncing" in sync_template_text
    )
    submit_wrapper_exists = (
        EXPERIMENTS
        / "sglang_small"
        / "scripts"
        / "submit_and_bind_strict_final_missing_jobs.py"
    ).exists()
    wait_sync_script_exists = (
        EXPERIMENTS
        / "sglang_small"
        / "scripts"
        / "wait_and_sync_bound_jobs.py"
    ).exists()
    refresh_status_script_exists = (
        DRAFT_ROOT / "scripts" / "refresh_strict_final_status.py"
    ).exists()
    remote_reachability = _read_json(
        matrix_audit_dir / "strict_final_remote_reachability.json"
    )
    bound_slurm_jobs = _read_json(
        matrix_audit_dir / "strict_final_bound_slurm_jobs.json"
    )
    wait_sync_summary = _read_json(
        matrix_audit_dir / "strict_final_wait_and_sync_summary.json"
    )
    failed_job_retries = _read_json(
        matrix_audit_dir / "strict_final_failed_job_retries.json"
    )
    refresh_status_summary = _read_json(
        EXPERIMENTS / "strict_final_status_refresh.json"
    )
    submission_manifest = _read_json(
        matrix_audit_dir / "strict_final_recovery_submission_manifest.json"
    )
    submission_jump_host = ""
    manifest_ssh = submission_manifest.get("ssh")
    if isinstance(manifest_ssh, dict):
        submission_jump_host = str(manifest_ssh.get("jump_host") or "")
    remote_reachability_evidence = ""
    if remote_reachability:
        remote_checks = remote_reachability.get("checks")
        remote_statuses = []
        if isinstance(remote_checks, list):
            for item in remote_checks:
                if not isinstance(item, dict):
                    continue
                check_id = str(item.get("id") or "")
                status = str(item.get("status") or "")
                if check_id and status:
                    remote_statuses.append(f"{check_id}={status}")
        resolver = remote_reachability.get("resolver")
        resolver_nameservers = []
        if isinstance(resolver, dict) and isinstance(resolver.get("nameservers"), list):
            resolver_nameservers = [str(item) for item in resolver["nameservers"]]
        submission_blockers = remote_reachability.get("submission_blockers")
        if not isinstance(submission_blockers, list):
            submission_blockers = []
        submission_blockers = [str(item) for item in submission_blockers]
        remote_reachability_evidence = (
            f"remote_reachability={remote_reachability.get('overall_status')}; "
            f"remote_checked_at={remote_reachability.get('checked_at')}; "
            f"remote_checks={remote_statuses}; "
            f"submission_blockers={submission_blockers}; "
            f"resolver_nameservers={resolver_nameservers}"
        )
    bound_slurm_evidence = ""
    if bound_slurm_jobs:
        slurm_counts = (
            bound_slurm_jobs.get("status_counts")
            if isinstance(bound_slurm_jobs.get("status_counts"), dict)
            else {}
        )
        slurm_rows = (
            bound_slurm_jobs.get("rows")
            if isinstance(bound_slurm_jobs.get("rows"), list)
            else []
        )
        slurm_start_times = sorted(
            {
                str(row.get("scontrol_StartTime") or "")
                for row in slurm_rows
                if isinstance(row, dict) and str(row.get("scontrol_StartTime") or "")
            }
        )
        slurm_req_tres = sorted(
            {
                str(row.get("scontrol_ReqTRES") or "")
                for row in slurm_rows
                if isinstance(row, dict) and str(row.get("scontrol_ReqTRES") or "")
            }
        )
        bound_slurm_evidence = (
            f"slurm_job_overall={bound_slurm_jobs.get('overall_status')}; "
            f"slurm_job_checked_at={bound_slurm_jobs.get('checked_at')}; "
            f"slurm_bound_jobs={bound_slurm_jobs.get('total_bound_jobs')}; "
            f"slurm_pending={slurm_counts.get('pending', 0)}; "
            f"slurm_running={slurm_counts.get('running', 0)}; "
            f"slurm_completed={slurm_counts.get('completed', 0)}; "
            f"slurm_failed={slurm_counts.get('failed', 0)}; "
            f"slurm_start_times={slurm_start_times}; "
            f"slurm_req_tres={slurm_req_tres}"
        )
    wait_sync_evidence = ""
    if wait_sync_summary:
        polls = (
            wait_sync_summary.get("polls")
            if isinstance(wait_sync_summary.get("polls"), list)
            else []
        )
        latest_poll = polls[-1] if polls and isinstance(polls[-1], dict) else {}
        slurm_counts = (
            bound_slurm_jobs.get("status_counts")
            if isinstance(bound_slurm_jobs.get("status_counts"), dict)
            else {}
        )
        wait_sync_matches_slurm_status = bool(
            latest_poll
            and bound_slurm_jobs
            and str(latest_poll.get("overall_status") or "")
            == str(bound_slurm_jobs.get("overall_status") or "")
        )
        wait_sync_matches_slurm_counts = bool(
            latest_poll
            and bound_slurm_jobs
            and all(
                int(_float(latest_poll.get(key)))
                == int(_float(slurm_counts.get(key)))
                for key in ("pending", "running", "completed", "failed")
            )
            and int(_float(latest_poll.get("total_bound_jobs")))
            == int(_float(bound_slurm_jobs.get("total_bound_jobs")))
        )
        wait_sync_evidence = (
            f"wait_sync_status={wait_sync_summary.get('status')}; "
            f"wait_sync_execute={wait_sync_summary.get('execute')}; "
            f"wait_sync_polls={len(polls)}; "
            f"wait_sync_checked_at={wait_sync_summary.get('checked_at')}; "
            f"wait_sync_latest_poll_status={latest_poll.get('overall_status', '')}; "
            f"wait_sync_latest_poll_at={latest_poll.get('checked_at', '')}; "
            f"wait_sync_matches_slurm_status={wait_sync_matches_slurm_status}; "
            f"wait_sync_matches_slurm_counts={wait_sync_matches_slurm_counts}"
        )
    retry_evidence = ""
    if failed_job_retries:
        retry_entries = (
            failed_job_retries.get("entries")
            if isinstance(failed_job_retries.get("entries"), list)
            else []
        )
        retry_pairs = sorted(
            {
                str(row.get("pair_id") or "")
                for row in retry_entries
                if isinstance(row, dict) and str(row.get("pair_id") or "")
            }
        )
        retry_replacements = sorted(
            {
                f"{row.get('failed_job_id')}->{row.get('new_job_id')}"
                for row in retry_entries
                if isinstance(row, dict)
                and str(row.get("failed_job_id") or "")
                and str(row.get("new_job_id") or "")
            }
        )
        retry_evidence = (
            f"failed_job_retries={len(retry_entries)}; "
            f"retry_pairs={retry_pairs}; "
            f"retry_replacements={retry_replacements}"
        )
    refresh_status_evidence = ""
    if refresh_status_summary:
        refresh_commands = (
            refresh_status_summary.get("commands")
            if isinstance(refresh_status_summary.get("commands"), list)
            else []
        )
        refresh_command_names = [
            str(item.get("name") or "")
            for item in refresh_commands
            if isinstance(item, dict) and str(item.get("name") or "")
        ]
        refresh_commands_ok = bool(refresh_commands) and all(
            isinstance(item, dict) and int(_float(item.get("returncode"), 1)) == 0
            for item in refresh_commands
        )
        refresh_status_evidence = (
            f"refresh_status={refresh_status_summary.get('status')}; "
            f"refresh_checked_at={refresh_status_summary.get('checked_at')}; "
            f"refresh_execute={refresh_status_summary.get('execute')}; "
            f"refresh_build_beamer={refresh_status_summary.get('build_beamer')}; "
            f"refresh_validate_draft={refresh_status_summary.get('validate_draft')}; "
            f"refresh_commands={refresh_command_names}; "
            f"refresh_commands_ok={refresh_commands_ok}"
        )
    case_rows = (
        matrix_audit.get("case_rows")
        if isinstance(matrix_audit.get("case_rows"), list)
        else []
    )
    pair_rows = (
        matrix_audit.get("pair_rows")
        if isinstance(matrix_audit.get("pair_rows"), list)
        else []
    )
    accepted_pair_rows = [
        row
        for row in pair_rows
        if isinstance(row, dict) and _bool_text(row.get("accepted"))
    ]
    hidden_control_pairs = 0
    for row in accepted_pair_rows:
        parsed_checks: dict[str, Any] = {}
        checks_value = row.get("checks")
        if isinstance(checks_value, str):
            try:
                decoded = json.loads(checks_value)
                parsed_checks = decoded if isinstance(decoded, dict) else {}
            except json.JSONDecodeError:
                parsed_checks = {}
        elif isinstance(checks_value, dict):
            parsed_checks = checks_value
        if _bool_text(parsed_checks.get("invisible_hidden_control")):
            hidden_control_pairs += 1
    case_evidence = []
    for row in case_rows:
        if not isinstance(row, dict):
            continue
        case_evidence.append(
            "{case}={status}:{accepted}/{required}".format(
                case=row.get("case_id", ""),
                status=row.get("status", ""),
                accepted=int(_float(row.get("accepted_pairs"))),
                required=int(_float(row.get("required_repeats"))),
            )
        )
    missing_pairs = [
        str(row.get("pair_id") or "")
        for row in matrix_audit.get("missing_plan_pairs", [])
        if isinstance(row, dict) and str(row.get("pair_id") or "")
    ]
    matrix_ready = bool(matrix_audit.get("matrix_ready"))
    if matrix_audit:
        matrix_status = PASS if matrix_ready else PARTIAL
        matrix_evidence = (
            f"overall_status={matrix_audit.get('overall_status')}; "
            f"matrix_ready={matrix_ready}; cases={'; '.join(case_evidence)}."
        )
        if missing_pairs:
            matrix_evidence = (
                f"{matrix_evidence[:-1]}; missing_pairs={missing_pairs}; "
                f"missing_job_rows={len(missing_job_rows)}; "
                f"hidden_control_pairs={hidden_control_pairs}/"
                f"{len(accepted_pair_rows)}; "
                f"job_binding_rows={len(job_binding_rows)}; "
                f"bound_job_bindings={len(bound_job_bindings)}/"
                f"{len(job_binding_rows)}; "
                f"sync_template_exists={sync_template_exists}; "
                f"sync_guard_complete_only={sync_guard_complete_only}; "
                f"submit_wrapper_exists={submit_wrapper_exists}; "
                f"wait_sync_script_exists={wait_sync_script_exists}."
            )
            matrix_evidence = (
                f"{matrix_evidence[:-1]}; "
                f"refresh_status_script_exists={refresh_status_script_exists}."
            )
            if submission_jump_host:
                matrix_evidence = (
                    f"{matrix_evidence[:-1]}; "
                    f"submission_jump_host={submission_jump_host}."
                )
            if remote_reachability_evidence:
                matrix_evidence = (
                    f"{matrix_evidence[:-1]}; {remote_reachability_evidence}."
                )
            if bound_slurm_evidence:
                matrix_evidence = (
                    f"{matrix_evidence[:-1]}; {bound_slurm_evidence}."
                )
            if wait_sync_evidence:
                matrix_evidence = (
                    f"{matrix_evidence[:-1]}; {wait_sync_evidence}."
                )
            if retry_evidence:
                matrix_evidence = (
                    f"{matrix_evidence[:-1]}; {retry_evidence}."
                )
            if refresh_status_evidence:
                matrix_evidence = (
                    f"{matrix_evidence[:-1]}; {refresh_status_evidence}."
                )
        if matrix_ready:
            matrix_next = (
                "Use the completed Host recovery matrix as real-system recovery-path evidence."
            )
        elif (
            missing_pairs
            and job_binding_rows
            and len(bound_job_bindings) == len(job_binding_rows)
        ):
            slurm_status = bound_slurm_jobs.get("overall_status") if bound_slurm_jobs else ""
            slurm_counts = (
                bound_slurm_jobs.get("status_counts")
                if isinstance(bound_slurm_jobs.get("status_counts"), dict)
                else {}
            )
            matrix_next = (
                "Bound Slurm jobs exist for the missing wait-complete pairs "
                f"{missing_pairs}; current_slurm_status={slurm_status or 'unknown'} "
                f"(pending={slurm_counts.get('pending', 0)}, "
                f"running={slurm_counts.get('running', 0)}, "
                f"completed={slurm_counts.get('completed', 0)}, "
                f"failed={slurm_counts.get('failed', 0)}); "
                "wait for them to finish, run wait_and_sync_bound_jobs.py --execute, "
                "extract physical restore evidence, then rerun "
                "audit_strict_final_recovery_matrix.py."
            )
        else:
            matrix_next = (
                "Run prepare_strict_final_missing_jobs.py, submit the generated "
                f"missing wait-complete jobs {missing_pairs}, bind returned Slurm ids with "
                "bind_strict_final_job_ids.py, sync them, then rerun "
                "audit_strict_final_recovery_matrix.py."
            )
    else:
        matrix_status = MISSING
        matrix_evidence = "Missing strict_final_recovery_matrix_audit.json."
        matrix_next = (
            "Run benchmark/AegisKV/draft/experiments/sglang_small/scripts/"
            "audit_strict_final_recovery_matrix.py after syncing SGLang pair results."
        )
    checks.append(
        RequirementCheck(
            "strict_final_recovery_matrix_audit",
            matrix_status,
            (
                "Real SGLang same-write-cost recovery matrix is audited "
                "separately from SGLSim metadata-only P0 rows."
            ),
            matrix_evidence,
            matrix_next,
        )
    )
    return checks


def check_sglang_accuracy() -> list[RequirementCheck]:
    rows = _read_csv(
        FIGURES / "data" / "measured" / "fig_sglang_sglsim_accuracy.csv"
    )
    if not rows:
        return [
            RequirementCheck(
                "sglang_sglsim_accuracy",
                MISSING,
                "Real SGLang inference validates base SGLSim timing.",
                "Missing fig_sglang_sglsim_accuracy.csv.",
                "Run the SGLang/SGLSim paired synthesis accuracy derivation.",
            )
        ]
    within = [row for row in rows if _bool_text(row.get("within_10pct"))]
    max_error = max((_float(row.get("rel_error_pct")) for row in rows), default=0.0)
    expected_metrics = {
        "prefill wall",
        "decode wall",
        "first decode throughput",
        "e2e mean",
        "e2e p50",
        "e2e p90",
        "e2e p99",
        "cache hit tokens",
    }
    observed_metrics = {str(row.get("metric") or "") for row in rows}
    expected_metrics_present = expected_metrics <= observed_metrics
    sources = {str(row.get("source") or "") for row in rows}
    evidence_tiers = {str(row.get("evidence_tier") or "") for row in rows}
    sglang_dirs = {str(row.get("sglang_run") or "") for row in rows}
    sglsim_dirs = {str(row.get("sglsim_run") or "") for row in rows}
    source_dirs_exist = bool(sglang_dirs) and bool(sglsim_dirs)
    for raw_dir in sglang_dirs | sglsim_dirs:
        path = Path(raw_dir)
        if not path.is_absolute():
            path = REPO_ROOT / path
        source_dirs_exist = (
            source_dirs_exist
            and (path / "request_details.csv").exists()
            and (path / "server.log").exists()
        )
        if raw_dir in sglang_dirs:
            source_dirs_exist = source_dirs_exist and (path / "metrics.json").exists()
    completed_pairs = {
        (
            int(_float(row.get("sglang_completed"))),
            int(_float(row.get("sglsim_completed"))),
        )
        for row in rows
    }
    status = (
        PASS
        if expected_metrics_present
        and len(rows) == len(expected_metrics)
        and len(within) == len(rows)
        and max_error <= 10.0
        and completed_pairs == {(60, 60)}
        and sources == {"synthesis_nohicache_fa3_parity"}
        and evidence_tiers == {"sglang-sglsim-synthesis-parity"}
        and source_dirs_exist
        else PARTIAL
    )
    return [
        RequirementCheck(
            "sglang_sglsim_accuracy",
            status,
            "Real SGLang inference validates base SGLSim timing within 10%.",
            (
                f"metrics={len(rows)}; within_10pct={len(within)}/{len(rows)}; "
                f"expected_metrics_present={expected_metrics_present}; "
                f"max_rel_error_pct={max_error:.2f}; completed_pairs="
                f"{sorted(completed_pairs)}; source_dirs_exist={source_dirs_exist}; "
                f"sources={sorted(sources)}; evidence_tiers={sorted(evidence_tiers)}."
            ),
            (
                "Use this as base-serving accuracy evidence; still collect failure-policy accuracy for strict-final."
                if status == PASS
                else "Rerun paired SGLang/SGLSim accuracy until all reported metrics are within 10%."
            ),
        )
    ]


def _included_graphics(tex_path: Path) -> tuple[list[str], list[str]]:
    if not tex_path.exists():
        return [], ["presentation.tex missing"]
    text = tex_path.read_text(encoding="utf-8")
    refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    search_dirs = [tex_path.parent, tex_path.parent / "Figures"]
    missing: list[str] = []
    for ref in refs:
        raw = Path(ref)
        candidates = [raw]
        if raw.suffix == "":
            candidates.extend(Path(str(raw) + ext) for ext in (".png", ".pdf", ".jpg", ".jpeg"))
        if not any((base / candidate).exists() for base in search_dirs for candidate in candidates):
            missing.append(ref)
    return refs, missing


def check_beamer() -> list[RequirementCheck]:
    tex_path = BEAMER / "presentation.tex"
    refs, missing = _included_graphics(tex_path)
    pdf = BEAMER / "presentation.pdf"
    text = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    bound_slurm_jobs = _read_json(
        EXPERIMENTS
        / "sglang_small"
        / "results"
        / "strict_final_recovery_matrix_audit"
        / "strict_final_bound_slurm_jobs.json"
    )
    dataplane_readiness = _read_json(
        EXPERIMENTS / "strict_final_dataplane_readiness.json"
    )
    slurm_short_status = _bound_slurm_short_status(bound_slurm_jobs)
    evidence_status_terms = [
        _goal_audit_status_term(dataplane_readiness),
        f"r2/r3 Slurm: {slurm_short_status}" if slurm_short_status else "r2/r3 Slurm:",
    ]
    evidence_status_terms.extend(_strict_final_readiness_terms(dataplane_readiness))
    missing_evidence_status_terms = [
        term for term in evidence_status_terms if term not in text
    ]
    evidence_status_sync = not missing_evidence_status_terms
    beamer_core_ready = (
        pdf.exists() and pdf.stat().st_size > 0 and bool(refs) and not missing
    )
    beamer_status = (
        PASS if beamer_core_ready and evidence_status_sync
        else PARTIAL if beamer_core_ready
        else MISSING
    )
    forbidden = [
        "fig_backup_data_path",
        "fig_decision_loop",
        "fig_failure_timeline",
        "fig_hostbackup_visibility_pipeline",
        "fig_radix_lock_ack_eviction",
        "fig_system_overview.png",
    ]
    forbidden_hits = [name for name in forbidden if name in text]
    generated_overview_asset = "fig_system_overview_generated_v2_tight.png"
    generated_overview = generated_overview_asset in text
    ref_basenames = {Path(ref).name for ref in refs}
    manifest_rows = _read_json_list(FIGURES / "manifest.json")
    result_pngs = sorted(
        Path(str((row.get("outputs") or {}).get("png") or "")).name
        for row in manifest_rows
        if row.get("kind") == "result_template"
        and str((row.get("outputs") or {}).get("png") or "")
    )
    missing_result_pngs = [
        name for name in result_pngs if name and name not in ref_basenames
    ]
    design_refs = sorted(
        Path(str((row.get("outputs") or {}).get("png") or "")).name
        for row in manifest_rows
        if row.get("kind") == "mechanism_diagram"
        and Path(str((row.get("outputs") or {}).get("png") or "")).name
        in ref_basenames
        and Path(str((row.get("outputs") or {}).get("png") or "")).name
        != generated_overview_asset
    )
    checks = [
        RequirementCheck(
            "beamer_compiles_and_references_figures",
            beamer_status,
            (
                "Beamer deck compiles, all referenced figures exist, and the "
                "current evidence-status slide matches the strict-final audit boundary."
            ),
            (
                f"pdf_exists={pdf.exists()}; pdf_bytes="
                f"{pdf.stat().st_size if pdf.exists() else 0}; "
                f"includegraphics={len(refs)}; missing={missing}; "
                f"evidence_status_sync={evidence_status_sync}; "
                f"missing_evidence_status_terms={missing_evidence_status_terms}."
            ),
            (
                "Recompile after any figure or slide edits."
                if pdf.exists() and not missing and evidence_status_sync
                else "Fix missing beamer figure references or update the evidence-status slide, then rerun make."
            ),
        ),
        RequirementCheck(
            "beamer_no_design_flowcharts",
            PASS if not forbidden_hits and generated_overview else PARTIAL,
            (
                "Beamer excludes detailed design flowcharts and keeps only the "
                "generated system overview plus measured motivation/result figures."
            ),
            (
                f"forbidden_design_refs={forbidden_hits}; "
                f"generated_system_overview_ref={generated_overview}."
            ),
            (
                "Keep design flowcharts out of presentation.tex."
                if not forbidden_hits and generated_overview
                else "Remove design flowchart references and use the generated overview asset."
            ),
        ),
        RequirementCheck(
            "beamer_result_figure_coverage",
            PASS if result_pngs and not missing_result_pngs and not design_refs else PARTIAL,
            (
                "Beamer includes all manifest result/motivation/calibration "
                "figures and excludes mechanism-diagram flowcharts."
            ),
            (
                f"result_figures_in_manifest={len(result_pngs)}; "
                f"referenced_result_figures={len(result_pngs) - len(missing_result_pngs)}; "
                f"missing_result_figures={missing_result_pngs}; "
                f"design_refs={design_refs}."
            ),
            (
                "Keep every result_template PNG in the deck while excluding detailed mechanism diagrams."
                if result_pngs and not missing_result_pngs and not design_refs
                else "Add missing result_template PNGs to presentation.tex or remove mechanism-diagram refs."
            ),
        ),
    ]
    prompt = BEAMER / "Figures" / "fig_system_overview_generated_v2.prompt.md"
    image = BEAMER / "Figures" / generated_overview_asset
    prompt_text = prompt.read_text(encoding="utf-8") if prompt.exists() else ""
    prompt_requirements = [
        "publication-ready system overview diagram",
        "academic systems-paper system overview figure",
        "pure white background",
        "landscape 16:9",
        "readable at slide size",
        "Use only these labels",
        '"Workflow"',
        '"SGLang Serving"',
        '"Radix Metadata"',
        '"AegisKV Controller"',
        '"HostBackup"',
        '"GPUReplica"',
        '"Safe Frontier"',
        '"Failure Retry"',
        '"Cache"',
        '"Recompute"',
        "no logos",
        "no watermark",
        "no extra text",
        "no overlapping arrows",
    ]
    prompt_missing = [
        requirement
        for requirement in prompt_requirements
        if requirement not in prompt_text
    ]
    prompt_refs_tight_asset = generated_overview_asset in prompt_text
    image_width, image_height = _png_dimensions(image)
    image_aspect = image_width / image_height if image_height else 0.0
    image_usable = (
        image.exists()
        and image.stat().st_size > 0
        and image_width >= 1200
        and image_height >= 500
        and 2.0 <= image_aspect <= 2.6
    )
    checks.append(
        RequirementCheck(
            "system_overview_generated_asset",
            (
                PASS
                if prompt.exists()
                and not prompt_missing
                and prompt_refs_tight_asset
                and image_usable
                and generated_overview
                else MISSING
            ),
            "System overview figure has an image-generation prompt and usable PNG.",
            (
                f"prompt_exists={prompt.exists()}; prompt_missing={prompt_missing}; "
                f"prompt_refs_tight_asset={prompt_refs_tight_asset}; "
                f"image_exists={image.exists()}; image_bytes="
                f"{image.stat().st_size if image.exists() else 0}; "
                f"image_width={image_width}; image_height={image_height}; "
                f"image_aspect={image_aspect:.3f}; image_usable={image_usable}; "
                f"beamer_ref={generated_overview}."
            ),
            (
                "Use the generated image in beamer and paper if its visual quality remains acceptable."
                if prompt.exists() and not prompt_missing and image_usable
                else "Restore the generated overview prompt requirements and PNG dimensions."
            ),
        )
    )
    return checks


def check_figure_evidence_gate() -> list[RequirementCheck]:
    status = _read_json(FIGURES / "hpca_hybrid_evidence_status.json")
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    hybrid_ready = int(_float(summary.get("hybrid_ready")))
    figures = int(_float(summary.get("figures")))
    strict_ready = int(_float(summary.get("strict_final_ready")))
    final_gate = _read_json(EXPERIMENTS / "final_gate_blockers.json")
    evidence_status = _read_json(FIGURES / "evidence_status.json")
    evidence_summary = (
        evidence_status.get("summary")
        if isinstance(evidence_status.get("summary"), dict)
        else {}
    )
    final_summary = (
        final_gate.get("summary") if isinstance(final_gate.get("summary"), dict) else {}
    )
    final_hybrid_summary = (
        final_gate.get("hybrid_summary")
        if isinstance(final_gate.get("hybrid_summary"), dict)
        else {}
    )
    final_gate_summary_match = bool(final_summary) and all(
        int(_float(final_summary.get(key))) == int(_float(evidence_summary.get(key)))
        for key in (
            "figures",
            "claim_ready",
            "result_figures",
            "measured_valid",
            "preliminary_valid",
        )
    )
    final_gate_hybrid_summary_match = bool(final_hybrid_summary) and all(
        int(_float(final_hybrid_summary.get(key))) == int(_float(summary.get(key)))
        for key in (
            "figures",
            "hybrid_ready",
            "strict_final_ready",
            "result_figures",
            "hybrid_ready_result_figures",
            "blocked_result_figures",
        )
    )
    final_rows = (
        final_gate.get("result_figures")
        if isinstance(final_gate.get("result_figures"), list)
        else []
    )
    final_by_stem = {
        str(row.get("stem") or ""): row
        for row in final_rows
        if isinstance(row, dict)
    }
    fig1 = final_by_stem.get("fig_motivation_equal_size_loss", {})
    fig3b = final_by_stem.get("fig_alpha_workload_failure_policy", {})
    fig10b = final_by_stem.get("fig_shared_prefix_marginal", {})
    fig1_gate = str(fig1.get("final_gate") or "")
    fig3b_gate = str(fig3b.get("final_gate") or "")
    fig1_trace_motivation = (
        fig1.get("current_tier") == "sglsim-trace-derived-stagebarrier"
        and not bool(fig1.get("claim_ready"))
        and "strict-final still needs" in fig1_gate
        and "satisfied by final CSV rows" not in fig1_gate
    )
    fig3b_runtime_replay_boundary = (
        fig3b.get("current_tier") == "sglsim-model-policy-table"
        and not bool(fig3b.get("claim_ready"))
        and "Fig. 3c and runtime replay verify online use" in fig3b_gate
        and "final paper needs online adaptive-alpha serving sweep" not in fig3b_gate
    )
    strict_ready_stems = sorted(
        str(row.get("stem") or "")
        for row in final_rows
        if bool(row.get("claim_ready"))
    )
    fig10b_only_strict_ready = (
        strict_ready_stems == ["fig_shared_prefix_marginal"]
        and fig10b.get("current_tier") == "final"
    )
    gate_ready = (
        figures > 0
        and hybrid_ready == figures
        and final_gate_summary_match
        and final_gate_hybrid_summary_match
        and fig1_trace_motivation
        and fig3b_runtime_replay_boundary
        and fig10b_only_strict_ready
    )
    return [
        RequirementCheck(
            "hpca_hybrid_figure_gate",
            PASS if gate_ready else PARTIAL,
            "All figures pass the current HPCA hybrid evidence gate and final-gate boundary report is current.",
            (
                f"hpca_hybrid_ready={hybrid_ready}/{figures}; "
                f"strict_final_ready={strict_ready}/{figures}; "
                f"final_gate_summary_match={final_gate_summary_match}; "
                f"final_gate_hybrid_summary_match="
                f"{final_gate_hybrid_summary_match}; "
                f"fig1_trace_motivation={fig1_trace_motivation}; "
                f"fig3b_runtime_replay_boundary="
                f"{fig3b_runtime_replay_boundary}; "
                f"strict_ready_stems={strict_ready_stems}; "
                f"fig10b_only_strict_ready={fig10b_only_strict_ready}."
            ),
            (
                "Hybrid-ready is enough for internal review; strict-final still needs physical recovery rows."
                if gate_ready
                else "Run validate_figure_evidence.py --write-hpca-hybrid --strict-hpca-hybrid and report_final_gate_blockers.py."
            ),
        )
    ]


def run_audit() -> dict[str, Any]:
    checks: list[RequirementCheck] = []
    checks.extend(check_workload_definition())
    checks.extend(check_motivation_figures())
    checks.extend(check_alpha_policy())
    checks.extend(check_failure_replay_semantics())
    checks.extend(check_policy_and_physical_dataplane())
    checks.extend(check_sglang_accuracy())
    checks.extend(check_beamer())
    checks.extend(check_figure_evidence_gate())

    counts = _status_counts(checks)
    overall = _overall_status(checks)
    complete = bool(checks) and counts[BLOCKED] == 0 and counts[MISSING] == 0 and counts[PARTIAL] == 0
    return {
        "summary": {
            "overall_status": overall,
            "complete": complete,
            "checks": len(checks),
            **counts,
        },
        "checks": [asdict(item) for item in checks],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# AegisKV Goal Completion Audit",
        "",
        "This report checks the active AegisKV experiment objective against current artifacts.",
        "It is intentionally stricter than the HPCA hybrid figure gate: calibrated SGLSim metadata rows are not treated as live physical tensor recovery.",
        "",
        "## Summary",
        "",
        f"- overall_status: `{summary['overall_status']}`",
        f"- complete: `{str(summary['complete']).lower()}`",
        f"- checks: `{summary['checks']}`",
        f"- pass: `{summary[PASS]}`",
        f"- partial: `{summary[PARTIAL]}`",
        f"- blocked: `{summary[BLOCKED]}`",
        f"- missing: `{summary[MISSING]}`",
        "",
        "## Checks",
        "",
        "| ID | Status | Requirement | Evidence | Next step |",
        "|---|---|---|---|---|",
    ]
    for item in payload["checks"]:
        lines.append(
            "| `{check_id}` | `{status}` | {requirement} | {evidence} | {next_step} |".format(
                check_id=item["check_id"],
                status=item["status"],
                requirement=str(item["requirement"]).replace("|", "/"),
                evidence=str(item["evidence"]).replace("|", "/"),
                next_step=str(item["next_step"]).replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `pass` means the current artifact proves that sub-requirement at the stated evidence level.",
            "- `partial` means the implementation or evidence exists but still needs stronger coverage.",
            "- `blocked` means current artifacts directly show the final requirement is not met.",
            "- `missing` means the audit could not find the required artifact or invariant.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md", type=Path, default=OUT_MD)
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    payload = run_audit()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
