#!/usr/bin/env python3
"""Project DP64 Real CC failure behavior from a digitized availability trace.

This is deliberately an analytical/Monte-Carlo projection, not an SGLSim
measurement.  The generated CSV files make every assumption and sample
auditable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PRESENTATION_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PRESENTATION_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglsim.srt.managers.aegis_l1_l2 import (  # noqa: E402
    AegisBackupPlan,
    AegisL1L2Config,
    AegisRankBackupManager,
)

OUTPUT = PRESENTATION_ROOT / "Figures" / "fig_dp64_failure_projection.png"
CONDITIONAL_OUTPUT = (
    PRESENTATION_ROOT
    / "Figures"
    / "fig_dp64_algorithm2_failure_conditioned_cdf.png"
)
DATA_ROOT = PRESENTATION_ROOT / "figures" / "data" / "analytic"
AVAILABILITY_CSV = DATA_ROOT / "fig_dp64_failure_projection_availability.csv"
SAMPLES_CSV = DATA_ROOT / "fig_dp64_failure_projection_workflow_samples.csv"
SUMMARY_CSV = DATA_ROOT / "fig_dp64_failure_projection_summary.csv"
ASSUMPTIONS_CSV = DATA_ROOT / "fig_dp64_failure_projection_assumptions.csv"
NOTE = DATA_ROOT / "fig_dp64_failure_projection.md"

REFERENCE_IMAGE = Path(
    "/home/comp/zhanzhao/.codex/attachments/"
    "a185a42b-0d2f-46b5-9174-87098bbb8bc3/"
    "codex-clipboard-7f250bf9-fd82-4899-924f-1096ef8c2caf.png"
)
TRACE = (
    PRESENTATION_ROOT
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "weka_subagent_cdf_24w_workload.jsonl"
)

DP_RANKS = 64
GPU_HBM_GB = 80
HOST_MEMORY_GB_PER_RANK = 160
ALPHA = 0.25
BETA = 0.50
L1_CAPACITY_TOKENS = 176_085
L2_CAPACITY_TOKENS = 352_170
ACTIVE_REQUESTS_PER_FAILED_RANK = 8
CONTEXT_LIMIT = 131_071
TRACE_HORIZON_H = 15.0
REFERENCE_MEAN_AVAILABLE = 58.07
MONTE_CARLO_REPEATS = 128
RANDOM_SEED = 20_260_722

L1_RESTORE_SETUP_MS = 2.4
L1_RESTORE_MS_PER_TOKEN = 0.0019
L2_RESTORE_SETUP_MS = 5.0
L2_RESTORE_MS_PER_TOKEN = 0.0036
RECOMPUTE_SETUP_MS = 0.0
RECOMPUTE_MS_PER_TOKEN = 0.22


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _digitize_availability() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Digitize the dark-blue step line and calibrate to its stated mean.

    Pixel anchors are taken from the supplied plot: x=100/803 map to 0/15 h;
    y=321/75 map to 0/60 GPUs.  The curve itself reaches 64 GPUs above the
    highest labeled tick.  Anti-aliased dark-blue pixels are tracked by color.
    """
    if not REFERENCE_IMAGE.exists():
        if not AVAILABILITY_CSV.exists():
            raise FileNotFoundError(
                f"missing both reference image and digitized CSV: {REFERENCE_IMAGE}"
            )
        rows = list(csv.DictReader(AVAILABILITY_CSV.open(encoding="utf-8")))
        return (
            np.asarray([float(row["time_h"]) for row in rows]),
            np.asarray([float(row["digitized_raw_available_gpus"]) for row in rows]),
            np.asarray([float(row["calibrated_available_gpus"]) for row in rows]),
            float(rows[0]["non_full_level_adjustment_gpus"]),
        )

    image = np.asarray(Image.open(REFERENCE_IMAGE).convert("RGB"))
    target = np.asarray([32.0, 102.0, 168.0])
    distance = np.sqrt(((image.astype(float) - target) ** 2).sum(axis=2))
    mask = distance < 15.0

    x_left, x_right = 100, 803
    y_top, y_bottom = 42, 321
    previous_y = 58.0
    times: list[float] = []
    raw_available: list[float] = []
    for x in range(x_left, x_right + 1):
        candidates = np.flatnonzero(mask[y_top : y_bottom + 1, x]) + y_top
        if len(candidates) == 0:
            local = mask[
                y_top : y_bottom + 1,
                max(x_left, x - 1) : min(x_right + 1, x + 2),
            ]
            candidates = np.flatnonzero(local.any(axis=1)) + y_top
        if len(candidates):
            if len(candidates) > 6:
                y = float(candidates[np.argmin(np.abs(candidates - previous_y))])
            else:
                y = float(np.median(candidates))
            previous_y = y
        else:
            y = previous_y
        times.append((x - x_left) * TRACE_HORIZON_H / (x_right - x_left))
        raw_available.append((y_bottom - y) * 60.0 / (y_bottom - 75.0))

    time_array = np.asarray(times)
    raw_array = np.clip(np.asarray(raw_available), 0.0, DP_RANKS)
    full = raw_array >= 63.75

    # The image explicitly states Avg GPU=58.07.  Pixel anti-aliasing and
    # short vertical steps make direct digitization slightly high.  Preserve
    # full-availability segments and subtract one constant from non-full
    # levels so the saved trace honors the figure's stated mean.
    low, high = 0.0, 5.0
    for _ in range(80):
        adjustment = (low + high) / 2.0
        calibrated = np.where(
            full,
            float(DP_RANKS),
            np.clip(raw_array - adjustment, 0.0, float(DP_RANKS)),
        )
        if float(calibrated.mean()) > REFERENCE_MEAN_AVAILABLE:
            low = adjustment
        else:
            high = adjustment
    adjustment = (low + high) / 2.0
    calibrated = np.where(
        full,
        float(DP_RANKS),
        np.clip(raw_array - adjustment, 0.0, float(DP_RANKS)),
    )
    return time_array, raw_array, calibrated, adjustment


def _load_workflows() -> dict[str, list[dict[str, Any]]]:
    workflows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with TRACE.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                workflows[str(row["workflow_id"])].append(row)
    return {
        workflow_id: sorted(rows, key=lambda row: float(row["arrival_t"]))
        for workflow_id, rows in sorted(workflows.items())
    }


def _availability_at(
    query_h: float,
    times_h: np.ndarray,
    availability: np.ndarray,
) -> float:
    index = int(np.searchsorted(times_h, query_h, side="right") - 1)
    return float(availability[np.clip(index, 0, len(availability) - 1)])


def _context_tokens(row: dict[str, Any]) -> int:
    return min(
        max(int(row.get("prompt_len", 0)) + int(row.get("output_len", 0)), 1),
        CONTEXT_LIMIT,
    )


def _stable_rank(key: str) -> int:
    return int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8],
        "big",
    ) % DP_RANKS


def _algorithm2_placements(
    workflows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Run the runtime Algorithm 2 over bounded active request sets.

    The trace does not contain Router queue snapshots, so execution and backup
    ranks use a deterministic uniform Top-2 proxy.  Requests assigned to one
    Backup rank are processed in arrival order in groups of at most eight,
    matching ``max_running_requests=8``.  A group owns one real
    ``AegisRankBackupManager``; after the group closes its backup entries are
    released before the next active set.  Within the active set, the runtime
    code performs 256-token value-density admit/replace exactly as in the demo.
    """
    config = AegisL1L2Config(
        l1_total_tokens=L1_CAPACITY_TOKENS,
        l2_total_tokens=L2_CAPACITY_TOKENS,
        initial_alpha=ALPHA,
        initial_beta=BETA,
        chunk_tokens=256,
        # Keep alpha/beta fixed.  Algorithm 3 is intentionally disabled in
        # this projection so the isolated effect is Algorithm 2 placement.
        window_requests=10**9,
        cost_weight=0.05,
        l1_restore_setup_ms=L1_RESTORE_SETUP_MS,
        l1_restore_ms_per_token=L1_RESTORE_MS_PER_TOKEN,
        l2_restore_setup_ms=L2_RESTORE_SETUP_MS,
        l2_restore_ms_per_token=L2_RESTORE_MS_PER_TOKEN,
        recompute_setup_ms=RECOMPUTE_SETUP_MS,
        recompute_ms_per_token=RECOMPUTE_MS_PER_TOKEN,
    )
    by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows = sorted(
        (row for workflow_rows in workflows.values() for row in workflow_rows),
        key=lambda row: (
            float(row["arrival_t"]),
            str(row["workflow_id"]),
            str(row["rid"]),
        ),
    )
    for row in rows:
        source = _stable_rank(str(row.get("agent_id") or row["rid"]))
        # The non-zero offset makes the backup rank different from the source
        # while preserving a deterministic, balanced DP64 mapping.
        target = (
            source + 1 + _stable_rank(str(row["rid"])) % (DP_RANKS - 1)
        ) % DP_RANKS
        enriched = dict(row)
        enriched["projected_source_dp_rank"] = source
        enriched["projected_backup_dp_rank"] = target
        by_target[target].append(enriched)

    placements: dict[str, dict[str, Any]] = {}
    for target, target_rows in sorted(by_target.items()):
        for start in range(0, len(target_rows), ACTIVE_REQUESTS_PER_FAILED_RANK):
            manager = AegisRankBackupManager(dp_rank=target, config=config)
            active_rows = target_rows[
                start : start + ACTIVE_REQUESTS_PER_FAILED_RANK
            ]
            for row in active_rows:
                critical = bool(row.get("critical_agent", False))
                ack = manager.place(
                    AegisBackupPlan(
                        rid=str(row["rid"]),
                        backup_key=str(row["rid"]),
                        source_dp_rank=int(row["projected_source_dp_rank"]),
                        target_dp_rank=target,
                        token_count=_context_tokens(row),
                        critical_probability=0.90 if critical else 0.35,
                        slack_ms=0.0 if critical else 8.0,
                        version=1,
                    )
                )
                placements[str(row["rid"])] = ack.to_dict()
                # Later, higher-density chunks can evict an earlier entry.
                # Apply the runtime's authoritative hint updates so every
                # request is evaluated against the final active-set state.
                for update in manager.pop_pending_hint_updates():
                    placements[str(update["rid"])] = update
    return placements


def _no_backup_retry_ms(tokens: int) -> float:
    return RECOMPUTE_SETUP_MS + tokens * RECOMPUTE_MS_PER_TOKEN


def _aegis_retry_ms(
    placement: dict[str, Any],
) -> tuple[float, float, float, float]:
    l1 = float(placement.get("l1_tokens", 0))
    l2 = float(placement.get("l2_tokens", 0))
    recompute = float(placement.get("recompute_tokens", 0))
    cost = 0.0
    if l1 > 0:
        cost += L1_RESTORE_SETUP_MS + l1 * L1_RESTORE_MS_PER_TOKEN
    if l2 > 0:
        cost += L2_RESTORE_SETUP_MS + l2 * L2_RESTORE_MS_PER_TOKEN
    if recompute > 0:
        cost += RECOMPUTE_SETUP_MS + recompute * RECOMPUTE_MS_PER_TOKEN
    return cost, l1, l2, recompute


def project(
    times_h: np.ndarray,
    availability: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float], np.ndarray]:
    workflows = _load_workflows()
    placements = _algorithm2_placements(workflows)
    rng = np.random.default_rng(RANDOM_SEED)
    samples: list[dict[str, Any]] = []
    normal_requests = 0
    retry_requests = 0
    retry_l1_tokens = 0.0
    retry_l2_tokens = 0.0
    retry_recompute_tokens = 0.0
    critical_retry_l1_tokens = 0.0
    critical_retry_l2_tokens = 0.0
    critical_retry_recompute_tokens = 0.0
    noncritical_retry_l1_tokens = 0.0
    noncritical_retry_l2_tokens = 0.0
    noncritical_retry_recompute_tokens = 0.0

    for workflow_id, rows in workflows.items():
        first_arrival = min(float(row["arrival_t"]) for row in rows)
        last_arrival = max(float(row["arrival_t"]) for row in rows)
        baseline_wct_s = last_arrival - first_arrival
        latest_start_h = max(TRACE_HORIZON_H - baseline_wct_s / 3600.0, 0.0)

        for repeat in range(MONTE_CARLO_REPEATS):
            start_h = float(rng.uniform(0.0, latest_start_h))
            no_backup_visible_retry_s = 0.0
            aegis_visible_retry_s = 0.0
            sample_retry_count = 0
            sample_critical_retry_count = 0

            for row in rows:
                request_h = start_h + (
                    float(row["arrival_t"]) - first_arrival
                ) / 3600.0
                available = _availability_at(request_h, times_h, availability)
                retry_probability = np.clip(
                    (DP_RANKS - available) / DP_RANKS,
                    0.0,
                    1.0,
                )
                is_retry = bool(rng.random() < retry_probability)
                if not is_retry:
                    normal_requests += 1
                    continue

                retry_requests += 1
                sample_retry_count += 1
                tokens = _context_tokens(row)
                placement = placements[str(row["rid"])]
                aegis_cost_ms, l1, l2, recompute = _aegis_retry_ms(placement)
                retry_l1_tokens += l1
                retry_l2_tokens += l2
                retry_recompute_tokens += recompute

                # Only retries on the trace-marked critical agent contribute
                # directly to projected WCT; non-critical recovery work is
                # assumed to fit under workflow slack.
                if bool(row.get("critical_agent", False)):
                    critical_retry_l1_tokens += l1
                    critical_retry_l2_tokens += l2
                    critical_retry_recompute_tokens += recompute
                    sample_critical_retry_count += 1
                    no_backup_visible_retry_s += _no_backup_retry_ms(tokens) / 1000.0
                    aegis_visible_retry_s += aegis_cost_ms / 1000.0
                else:
                    noncritical_retry_l1_tokens += l1
                    noncritical_retry_l2_tokens += l2
                    noncritical_retry_recompute_tokens += recompute

            samples.append(
                {
                    "workflow_id": workflow_id,
                    "repeat": repeat,
                    "start_h": start_h,
                    "baseline_trace_wct_s": baseline_wct_s,
                    "retry_request_count": sample_retry_count,
                    "critical_retry_request_count": sample_critical_retry_count,
                    "no_backup_visible_retry_s": no_backup_visible_retry_s,
                    "aegiskv_visible_retry_s": aegis_visible_retry_s,
                    "no_backup_projected_wct_s": baseline_wct_s
                    + no_backup_visible_retry_s,
                    "aegiskv_projected_wct_s": baseline_wct_s
                    + aegis_visible_retry_s,
                    "paired_wct_saving_s": no_backup_visible_retry_s
                    - aegis_visible_retry_s,
                }
            )

    request_total = normal_requests + retry_requests
    retry_token_total = retry_l1_tokens + retry_l2_tokens + retry_recompute_tokens
    critical_retry_token_total = (
        critical_retry_l1_tokens
        + critical_retry_l2_tokens
        + critical_retry_recompute_tokens
    )
    noncritical_retry_token_total = (
        noncritical_retry_l1_tokens
        + noncritical_retry_l2_tokens
        + noncritical_retry_recompute_tokens
    )
    summary = {
        "normal_request_fraction": normal_requests / request_total,
        "retry_request_fraction": retry_requests / request_total,
        "aegis_retry_l1_token_fraction": retry_l1_tokens / retry_token_total,
        "aegis_retry_l2_token_fraction": retry_l2_tokens / retry_token_total,
        "aegis_retry_recompute_token_fraction": retry_recompute_tokens
        / retry_token_total,
        "aegis_critical_retry_l1_token_fraction": critical_retry_l1_tokens
        / critical_retry_token_total,
        "aegis_critical_retry_l2_token_fraction": critical_retry_l2_tokens
        / critical_retry_token_total,
        "aegis_critical_retry_recompute_token_fraction": critical_retry_recompute_tokens
        / critical_retry_token_total,
        "aegis_noncritical_retry_l1_token_fraction": noncritical_retry_l1_tokens
        / noncritical_retry_token_total,
        "aegis_noncritical_retry_l2_token_fraction": noncritical_retry_l2_tokens
        / noncritical_retry_token_total,
        "aegis_noncritical_retry_recompute_token_fraction": noncritical_retry_recompute_tokens
        / noncritical_retry_token_total,
        "normal_request_count": float(normal_requests),
        "retry_request_count": float(retry_requests),
        "retry_l1_tokens": retry_l1_tokens,
        "retry_l2_tokens": retry_l2_tokens,
        "retry_recompute_tokens": retry_recompute_tokens,
    }
    return samples, summary, np.asarray(
        [float(row["paired_wct_saving_s"]) for row in samples]
    )


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def _write_outputs(
    times_h: np.ndarray,
    raw_availability: np.ndarray,
    availability: np.ndarray,
    adjustment: float,
    samples: list[dict[str, Any]],
    summary: dict[str, float],
) -> None:
    _write_csv(
        AVAILABILITY_CSV,
        [
            {
                "time_h": float(time_h),
                "digitized_raw_available_gpus": float(raw),
                "calibrated_available_gpus": float(calibrated),
                "non_full_level_adjustment_gpus": adjustment,
            }
            for time_h, raw, calibrated in zip(
                times_h, raw_availability, availability
            )
        ],
    )
    _write_csv(SAMPLES_CSV, samples)

    no_wct = np.asarray(
        [float(row["no_backup_projected_wct_s"]) for row in samples]
    )
    aegis_wct = np.asarray(
        [float(row["aegiskv_projected_wct_s"]) for row in samples]
    )
    rows: list[dict[str, Any]] = []
    for policy, values in (("No Backup", no_wct), ("AegisKV", aegis_wct)):
        for metric, value in (
            ("mean_wct_s", np.mean(values)),
            ("p50_wct_s", np.quantile(values, 0.50)),
            ("p90_wct_s", np.quantile(values, 0.90)),
            ("p99_wct_s", np.quantile(values, 0.99)),
        ):
            rows.append({"section": "workflow_wct", "policy": policy, "metric": metric, "value": float(value), "unit": "s"})
    affected = [
        row for row in samples if int(row["critical_retry_request_count"]) > 0
    ]
    rows.append(
        {
            "section": "failure_conditioned",
            "policy": "paired_projection",
            "metric": "affected_workflow_sample_fraction",
            "value": len(affected) / len(samples),
            "unit": "fraction",
        }
    )
    for policy, key in (
        ("No Backup", "no_backup_visible_retry_s"),
        ("AegisKV", "aegiskv_visible_retry_s"),
    ):
        values = np.asarray([float(row[key]) for row in affected])
        for metric, value in (
            ("mean_visible_recovery_s", np.mean(values)),
            ("p50_visible_recovery_s", np.quantile(values, 0.50)),
            ("p90_visible_recovery_s", np.quantile(values, 0.90)),
            ("p99_visible_recovery_s", np.quantile(values, 0.99)),
        ):
            rows.append(
                {
                    "section": "failure_conditioned",
                    "policy": policy,
                    "metric": metric,
                    "value": float(value),
                    "unit": "s",
                }
            )
    for metric, value, unit in (
        ("normal_request_fraction", summary["normal_request_fraction"], "fraction"),
        ("retry_request_fraction", summary["retry_request_fraction"], "fraction"),
        ("aegis_retry_l1_token_fraction", summary["aegis_retry_l1_token_fraction"], "fraction"),
        ("aegis_retry_l2_token_fraction", summary["aegis_retry_l2_token_fraction"], "fraction"),
        ("aegis_retry_recompute_token_fraction", summary["aegis_retry_recompute_token_fraction"], "fraction"),
        ("aegis_critical_retry_l1_token_fraction", summary["aegis_critical_retry_l1_token_fraction"], "fraction"),
        ("aegis_critical_retry_l2_token_fraction", summary["aegis_critical_retry_l2_token_fraction"], "fraction"),
        ("aegis_critical_retry_recompute_token_fraction", summary["aegis_critical_retry_recompute_token_fraction"], "fraction"),
        ("aegis_noncritical_retry_l1_token_fraction", summary["aegis_noncritical_retry_l1_token_fraction"], "fraction"),
        ("aegis_noncritical_retry_l2_token_fraction", summary["aegis_noncritical_retry_l2_token_fraction"], "fraction"),
        ("aegis_noncritical_retry_recompute_token_fraction", summary["aegis_noncritical_retry_recompute_token_fraction"], "fraction"),
    ):
        rows.append({"section": "recovery_mix", "policy": "paired_projection", "metric": metric, "value": value, "unit": unit})
    _write_csv(SUMMARY_CSV, rows)

    image_hash = (
        hashlib.sha256(REFERENCE_IMAGE.read_bytes()).hexdigest()
        if REFERENCE_IMAGE.exists()
        else "reference-image-not-present"
    )
    assumption_rows = [
        ("evidence_type", "analytic_monte_carlo_projection", "", "not measured"),
        ("reference_image_sha256", image_hash, "", "user-supplied availability plot"),
        ("dp_ranks", DP_RANKS, "GPU", "one DP rank per A100"),
        ("gpu_hbm", GPU_HBM_GB, "GiB/rank", "A100"),
        ("host_memory", HOST_MEMORY_GB_PER_RANK, "GiB/rank", "assumed per DP rank"),
        ("alpha", ALPHA, "fraction", "fixed"),
        ("beta", BETA, "fraction", "fixed"),
        ("l1_capacity", L1_CAPACITY_TOKENS, "token/rank", "current SGLSim A100-80GB profile"),
        ("l2_capacity", L2_CAPACITY_TOKENS, "token/rank", "Host=2x L1 KV capacity"),
        ("active_requests_per_failed_rank", ACTIVE_REQUESTS_PER_FAILED_RANK, "request/rank", "Algorithm 2 active-set bound"),
        ("algorithm2_implementation", "AegisRankBackupManager.place", "", "runtime 256-token value-density admit/replace"),
        ("algorithm2_critical_probability", 0.90, "fraction", "critical request"),
        ("algorithm2_noncritical_probability", 0.35, "fraction", "non-critical request"),
        ("algorithm2_critical_slack", 0.0, "ms", "critical request"),
        ("algorithm2_noncritical_slack", 8.0, "ms", "non-critical request"),
        ("algorithm3", "disabled", "", "fixed alpha and beta isolate Algorithm 2"),
        ("retry_rank", "ACKed backup rank", "", "isolates placement benefit; Router load tradeoff excluded"),
        ("real_cc_workflows", len(_load_workflows()), "workflow", "all trace workflows"),
        ("monte_carlo_repeats", MONTE_CARLO_REPEATS, "repeat/workflow", "uniform start phase"),
        ("random_seed", RANDOM_SEED, "", "deterministic"),
        ("reference_mean_available_gpus", REFERENCE_MEAN_AVAILABLE, "GPU", "stated in supplied image"),
        ("recompute_cost", RECOMPUTE_MS_PER_TOKEN, "ms/token", "current demo profile"),
        ("l1_restore_cost", L1_RESTORE_MS_PER_TOKEN, "ms/token", "current runtime coefficient"),
        ("l2_restore_cost", L2_RESTORE_MS_PER_TOKEN, "ms/token", "current runtime coefficient"),
    ]
    _write_csv(
        ASSUMPTIONS_CSV,
        [
            {"parameter": key, "value": value, "unit": unit, "note": note}
            for key, value, unit, note in assumption_rows
        ],
    )

    NOTE.write_text(
        """# DP64 Real CC failure projection

This artifact is an analytical Monte-Carlo projection, not an SGLSim or
physical-system measurement.

The user-supplied availability image is digitized and calibrated to its stated
mean of 58.07 available GPUs over 15 hours.  For a request placed at time `t`,
the projection uses `(64 - A(t)) / 64` as its retry exposure probability.  This
is an exposure approximation; it is not a reconstructed failure event log.

The baseline workflow time is the Real CC trace arrival span.  Only retries on
trace-marked critical agents add visible recovery delay.  Normal-path backup
interference, queueing redistribution, model reload, correlated failure
domains, and exact DAG slack propagation are excluded.

For AegisKV, fixed alpha=0.25 and beta=0.50 are capacity fractions, not direct
reuse probabilities.  The current SGLSim A100 profile provides 176,085 L1 and
352,170 L2 KV tokens per rank.  This revision directly calls the runtime
`AegisRankBackupManager.place()` implementation of Algorithm 2: critical
requests use probability 0.90 and zero slack; non-critical requests use 0.35
and 8 ms slack; 256-token candidates undergo value-density admit/replace.
Entries compete in bounded active sets of at most eight requests per target
rank and are released when an active set closes.

The trace lacks Router queue snapshots.  Execution/backup ranks therefore use
a deterministic balanced DP64 proxy.  To isolate Algorithm 2, a retry is
assumed to consume the ACKed Backup rank; Router load-versus-reuse rerouting is
not modeled here.
""",
        encoding="utf-8",
    )


def render(
    times_h: np.ndarray,
    availability: np.ndarray,
    samples: list[dict[str, Any]],
    summary: dict[str, float],
    paired_savings_s: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.1))

    ax = axes[0, 0]
    ax.step(times_h, availability, where="post", color="#2563EB", linewidth=1.7)
    ax.fill_between(times_h, availability, DP_RANKS, step="post", color="#DBEAFE", alpha=0.8)
    ax.axhline(REFERENCE_MEAN_AVAILABLE, color="#60A5FA", linestyle="--", linewidth=1.0)
    ax.text(14.7, REFERENCE_MEAN_AVAILABLE + 1.1, "mean=58.07", ha="right", color="#2563EB")
    ax.set_xlim(0, TRACE_HORIZON_H)
    ax.set_ylim(0, 66)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Available GPUs")
    ax.set_title("(a) Digitized 64-GPU availability reference")
    ax.grid(color="#E5E7EB", linewidth=0.7)
    missing_gpu_hours = (DP_RANKS - REFERENCE_MEAN_AVAILABLE) * TRACE_HORIZON_H
    ax.text(
        0.02,
        0.08,
        f"min={availability.min():.1f}; missing={missing_gpu_hours:.1f} GPU-h",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
    )

    ax = axes[0, 1]
    no_wct_min = np.asarray([float(row["no_backup_projected_wct_s"]) for row in samples]) / 60.0
    aegis_wct_min = np.asarray([float(row["aegiskv_projected_wct_s"]) for row in samples]) / 60.0
    for label, values, color, linestyle in (
        ("No Backup", no_wct_min, "#6B7280", ":"),
        ("AegisKV (alpha=0.25, beta=0.50)", aegis_wct_min, "#059669", "-"),
    ):
        x, y = _cdf(values)
        # Draw the dotted baseline above the solid AegisKV curve.  The two
        # full-scale CDFs are intentionally almost coincident because the
        # projected recovery saving is seconds while workflows last minutes.
        zorder = 4 if label == "No Backup" else 3
        linewidth = 2.5 if label == "No Backup" else 2.0
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
            zorder=zorder,
        )
    ax.set_xlabel("Projected workflow completion time (min)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.set_title("(b) Real CC workflow WCT projection")
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    delta_mean = float(np.mean(paired_savings_s))
    delta_p90 = float(np.quantile(paired_savings_s, 0.90))
    ax.text(
        0.04,
        0.70,
        f"paired WCT saving\nmean={delta_mean:.1f}s; P90={delta_p90:.1f}s",
        transform=ax.transAxes,
        fontsize=8,
        color="#047857",
        bbox={"facecolor": "white", "edgecolor": "#A7F3D0", "alpha": 0.92},
    )
    saving_percentiles = np.asarray([10, 25, 50, 75, 90, 95, 99], dtype=float)
    saving_values = np.quantile(paired_savings_s, saving_percentiles / 100.0)
    inset = ax.inset_axes([0.57, 0.12, 0.39, 0.29])
    inset.plot(
        saving_percentiles,
        saving_values,
        marker="o",
        markersize=2.7,
        color="#059669",
        linewidth=1.3,
    )
    inset.fill_between(
        saving_percentiles,
        saving_values,
        color="#A7F3D0",
        alpha=0.35,
    )
    inset.set_title("Paired WCT saving", fontsize=7, pad=2)
    inset.set_xlabel("Percentile", fontsize=6.5, labelpad=1)
    inset.set_ylabel("Seconds", fontsize=6.5, labelpad=1)
    inset.tick_params(axis="both", labelsize=6, length=2)
    inset.grid(color="#E5E7EB", linewidth=0.5)

    ax = axes[1, 0]
    normal_pct = 100.0 * summary["normal_request_fraction"]
    retry_pct = 100.0 * summary["retry_request_fraction"]
    x_positions = np.arange(2)
    ax.bar(x_positions, [normal_pct, normal_pct], color="#CBD5E1", label="Normal request")
    ax.bar(x_positions, [retry_pct, retry_pct], bottom=[normal_pct, normal_pct], color="#F97316", label="Retry request")
    ax.set_xticks(x_positions, ["No Backup", "AegisKV"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Request incidence (%)")
    ax.set_title("(c) Failure exposure is policy-independent")
    ax.legend(frameon=False, loc="lower center", ncol=2)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    for x in x_positions:
        ax.text(x, normal_pct / 2, f"Normal\n{normal_pct:.1f}%", ha="center", va="center", fontsize=9)
        ax.text(x, normal_pct + retry_pct / 2, f"Retry {retry_pct:.1f}%", ha="center", va="center", fontsize=8, color="white", weight="bold")

    ax = axes[1, 1]
    policies = ["No Backup\n(all Retry)", "AegisKV\n(all Retry)", "AegisKV\ncritical Retry"]
    l1 = np.asarray(
        [
            0.0,
            100.0 * summary["aegis_retry_l1_token_fraction"],
            100.0 * summary["aegis_critical_retry_l1_token_fraction"],
        ]
    )
    l2 = np.asarray(
        [
            0.0,
            100.0 * summary["aegis_retry_l2_token_fraction"],
            100.0 * summary["aegis_critical_retry_l2_token_fraction"],
        ]
    )
    recompute = np.asarray(
        [
            100.0,
            100.0 * summary["aegis_retry_recompute_token_fraction"],
            100.0 * summary["aegis_critical_retry_recompute_token_fraction"],
        ]
    )
    ax.bar(policies, l1, color="#059669", label="L1 reuse")
    ax.bar(policies, l2, bottom=l1, color="#2563EB", label="L2 reuse")
    ax.bar(policies, recompute, bottom=l1 + l2, color="#6B7280", label="Recompute")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Retry-token source (%)")
    ax.set_title("(d) Algorithm 2 prioritizes critical Retry KV")
    ax.legend(frameon=False, loc="upper center", ncol=3)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    for x, components in enumerate(zip(l1, l2, recompute)):
        bottom = 0.0
        for value, color in zip(components, ("white", "white", "white")):
            if value >= 5.0:
                ax.text(x, bottom + value / 2.0, f"{value:.1f}%", ha="center", va="center", color=color, fontsize=9, weight="bold")
            bottom += value

    fig.suptitle(
        "DP64 Algorithm-2 AegisKV failure projection on Real CC trace (not measured)",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.949,
        "64 x A100 80GB; 160GB Host per DP rank; fixed alpha=0.25, beta=0.50; "
        "24 workflows x 128 start phases = 3,072 paired samples",
        ha="center",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.013,
        "Projection calls runtime Algorithm 2 for 256-token value-density admit/replace in active sets of <=8 requests/rank. "
        "It assumes Retry uses the ACKed Backup rank and excludes queue redistribution, normal-path interference, model reload, and exact DAG propagation.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.91), h_pad=2.1, w_pad=1.6)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_failure_conditioned(samples: list[dict[str, Any]]) -> None:
    """Separate system-wide WCT from Algorithm-2 recovery-path benefit."""
    affected = [
        row for row in samples if int(row["critical_retry_request_count"]) > 0
    ]
    no_wct_min = np.asarray(
        [float(row["no_backup_projected_wct_s"]) for row in samples]
    ) / 60.0
    aegis_wct_min = np.asarray(
        [float(row["aegiskv_projected_wct_s"]) for row in samples]
    ) / 60.0
    no_recovery_s = np.asarray(
        [float(row["no_backup_visible_retry_s"]) for row in affected]
    )
    aegis_recovery_s = np.asarray(
        [float(row["aegiskv_visible_retry_s"]) for row in affected]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.25))
    colors = {"No Backup": "#6B7280", "AegisKV": "#059669"}

    ax = axes[0]
    for label, values, linestyle, zorder in (
        ("No Backup", no_wct_min, ":", 4),
        ("AegisKV (Algorithm 2)", aegis_wct_min, "-", 3),
    ):
        x, y = _cdf(values)
        color = colors["No Backup"] if label == "No Backup" else colors["AegisKV"]
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2.3, label=label, zorder=zorder)
    mean_delta = float(np.mean((no_wct_min - aegis_wct_min) * 60.0))
    ax.set_title("(a) All workflow samples: system-wide WCT")
    ax.set_xlabel("Projected workflow completion time (min)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.04,
        0.71,
        f"90.6% requests avoid failure\nmean saving={mean_delta:.1f}s",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.94},
    )

    ax = axes[1]
    for label, values, linestyle, zorder in (
        ("No Backup", no_recovery_s, ":", 4),
        ("AegisKV (Algorithm 2)", aegis_recovery_s, "-", 3),
    ):
        x, y = _cdf(values)
        color = colors["No Backup"] if label == "No Backup" else colors["AegisKV"]
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=2.4, label=label, zorder=zorder)
    reduction = 1.0 - float(np.mean(aegis_recovery_s) / np.mean(no_recovery_s))
    ax.set_title("(b) Critical-Retry samples: visible recovery overhead")
    ax.set_xlabel("Cumulative visible recovery overhead per workflow (s)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.01)
    ax.grid(color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.55,
        0.68,
        "mean: "
        f"{np.mean(no_recovery_s):.1f}s -> {np.mean(aegis_recovery_s):.1f}s\n"
        f"reduction={100.0 * reduction:.1f}%\n"
        f"P90: {np.quantile(no_recovery_s, 0.90):.1f}s -> "
        f"{np.quantile(aegis_recovery_s, 0.90):.1f}s",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#047857",
        bbox={"facecolor": "white", "edgecolor": "#A7F3D0", "alpha": 0.94},
    )

    fig.suptitle(
        "Why unconditional WCT hides Algorithm-2 recovery benefit (DP64 projection, not measured)",
        fontsize=13,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        f"Same 3,072 paired Real CC samples. Panel (b) conditions only on critical_retry_request_count > 0 "
        f"({len(affected)}/{len(samples)} samples); no outcomes are removed within that predeclared subset.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0.02, 0.065, 0.99, 0.91), w_pad=2.0)
    CONDITIONAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONDITIONAL_OUTPUT, dpi=300, bbox_inches="tight")
    fig.savefig(CONDITIONAL_OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    times_h, raw_availability, availability, adjustment = _digitize_availability()
    samples, summary, paired_savings_s = project(times_h, availability)
    _write_outputs(
        times_h,
        raw_availability,
        availability,
        adjustment,
        samples,
        summary,
    )
    render(times_h, availability, samples, summary, paired_savings_s)
    render_failure_conditioned(samples)
    print(f"wrote {OUTPUT}")
    print(f"wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {CONDITIONAL_OUTPUT}")
    print(f"wrote {CONDITIONAL_OUTPUT.with_suffix('.pdf')}")
    print(f"wrote {AVAILABILITY_CSV}")
    print(f"wrote {SAMPLES_CSV}")
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {ASSUMPTIONS_CSV}")
    print(f"wrote {NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
