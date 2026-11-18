#!/usr/bin/env python3
"""Render two AegisKV motivation figures directly from the real CC trace.

The trace exposes ordered, trace-local KV block identifiers (``hash_ids``).
Figure 1 attributes each request's reusable prefix to:

1. same-agent history,
2. cross-agent-only incremental reuse beyond same-agent history, or
3. new KV.

Only source requests that completed before the target request starts are
eligible.  This is a conservative trace characterization; it does not model
runtime routing, eviction, or physical cache residency.

Figure 2 directly plots each subagent's mean request context length versus
total execution time and reports their agent-level Spearman correlation.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
TRACE_PATH = (
    ROOT
    / "benchmark"
    / "AegisKV"
    / "paper_outline"
    / "experiments"
    / "sglsim_large"
    / "traces"
    / "cc-traces-weka-062126"
    / "traces.jsonl"
)
MEASURED_DIR = (
    ROOT / "benchmark" / "AegisKV" / "paper_outline" / "figures" / "data" / "measured"
)
DRAFT_PNG_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "figures" / "png"
BEAMER_FIG_DIR = ROOT / "benchmark" / "AegisKV" / "paper_outline" / "Figures"

REUSE_CSV = MEASURED_DIR / "fig_real_trace_kv_reuse_scope.csv"
VALUE_CSV = MEASURED_DIR / "fig_real_trace_context_value.csv"
REUSE_PNG = DRAFT_PNG_DIR / "fig_real_trace_kv_reuse_scope.png"
VALUE_PNG = DRAFT_PNG_DIR / "fig_real_trace_context_value.png"
REUSE_BEAMER = BEAMER_FIG_DIR / REUSE_PNG.name
VALUE_BEAMER = BEAMER_FIG_DIR / VALUE_PNG.name

SOURCE_NAME = "cc-traces-weka-062126/traces.jsonl"
REUSE_EVIDENCE_TIER = "real-trace-derived-completed-source-prefix"
VALUE_EVIDENCE_TIER = "real-trace-derived-agent-lifetime"
LCP_CHUNK = 8192


@dataclass
class Request:
    trace_id: str
    role: str
    agent_id: str
    start_s: float
    ready_s: float
    input_tokens: int
    output_tokens: int
    api_time_s: float
    think_time_s: float
    hash_ids: list[int]
    serial: int


@dataclass
class ReuseAggregate:
    requests: int = 0
    input_tokens: int = 0
    same_tokens: int = 0
    cross_only_tokens: int = 0
    new_tokens: int = 0
    raw_cross_tokens: int = 0
    cross_positive_requests: int = 0

    def add(
        self,
        *,
        input_tokens: int,
        same_tokens: int,
        cross_tokens: int,
    ) -> None:
        best_tokens = max(same_tokens, cross_tokens)
        self.requests += 1
        self.input_tokens += input_tokens
        self.same_tokens += same_tokens
        self.cross_only_tokens += max(0, cross_tokens - same_tokens)
        self.new_tokens += max(0, input_tokens - best_tokens)
        self.raw_cross_tokens += cross_tokens
        self.cross_positive_requests += int(cross_tokens > 0)


class PrefixFrontier:
    """A compact set of maximal prefix branches for exact LCP queries."""

    def __init__(self) -> None:
        self.branches: list[np.ndarray] = []

    @staticmethod
    def lcp(left: np.ndarray, right: np.ndarray) -> int:
        limit = min(len(left), len(right))
        for start in range(0, limit, LCP_CHUNK):
            end = min(limit, start + LCP_CHUNK)
            mismatch = np.flatnonzero(left[start:end] != right[start:end])
            if mismatch.size:
                return start + int(mismatch[0])
        return limit

    def query(self, sequence: np.ndarray) -> int:
        best = 0
        for branch in self.branches:
            best = max(best, self.lcp(sequence, branch))
            if best == len(sequence):
                break
        return best

    def insert(self, sequence: np.ndarray) -> None:
        remove: list[int] = []
        for idx, branch in enumerate(self.branches):
            common = self.lcp(sequence, branch)
            if common == len(sequence):
                return
            if common == len(branch):
                remove.append(idx)
        for idx in reversed(remove):
            del self.branches[idx]
        self.branches.append(sequence)


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if len(left_array) < 2:
        return float("nan")
    left_rank = _rankdata(left_array)
    right_rank = _rankdata(right_array)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _percent(value: int, total: int) -> float:
    return 100.0 * value / total if total else 0.0


def _flatten_requests(
    trace: dict[str, object],
) -> tuple[list[Request], list[dict[str, object]]]:
    trace_id = str(trace.get("id") or "")
    requests: list[Request] = []
    agent_groups: list[dict[str, object]] = []
    serial = 0

    def append_request(raw: dict[str, object], role: str, agent_id: str) -> None:
        nonlocal serial
        start_s = _float(raw.get("t"))
        api_time_s = max(0.0, _float(raw.get("api_time")))
        requests.append(
            Request(
                trace_id=trace_id,
                role=role,
                agent_id=agent_id,
                start_s=start_s,
                ready_s=start_s + api_time_s,
                input_tokens=max(0, _int(raw.get("in"))),
                output_tokens=max(0, _int(raw.get("out"))),
                api_time_s=api_time_s,
                think_time_s=max(0.0, _float(raw.get("think_time"))),
                hash_ids=list(raw.get("hash_ids") or []),
                serial=serial,
            )
        )
        serial += 1

    for raw_value in trace.get("requests") or []:
        if not isinstance(raw_value, dict):
            continue
        if raw_value.get("type") != "subagent":
            append_request(raw_value, "main", "main")
            continue

        raw_requests = [
            item
            for item in raw_value.get("requests") or []
            if isinstance(item, dict)
        ]
        source_agent_id = str(raw_value.get("agent_id") or f"subagent_{serial}")
        agent_id = f"sub:{source_agent_id}"
        for item in raw_requests:
            append_request(item, "subagent", agent_id)

        if not raw_requests:
            continue
        ordered = sorted(
            raw_requests,
            key=lambda item: (_float(item.get("t")), str(item.get("model") or "")),
        )
        first_start = _float(ordered[0].get("t"))
        derived_end = max(
            _float(item.get("t")) + max(0.0, _float(item.get("api_time")))
            for item in ordered
        )
        duration_ms = _float(raw_value.get("duration_ms"), -1.0)
        duration_s = duration_ms / 1000.0 if duration_ms >= 0.0 else derived_end - first_start
        agent_groups.append(
            {
                "trace_id": trace_id,
                "agent_id": source_agent_id,
                "model": str(ordered[0].get("model") or ""),
                "status": str(raw_value.get("status") or ""),
                "initial_context_tokens": max(0, _int(ordered[0].get("in"))),
                "final_context_tokens": max(0, _int(ordered[-1].get("in"))),
                "max_context_tokens": max(max(0, _int(item.get("in"))) for item in ordered),
                "mean_context_tokens": float(
                    np.mean([max(0, _int(item.get("in"))) for item in ordered])
                ),
                "cumulative_input_tokens": sum(
                    max(0, _int(item.get("in"))) for item in ordered
                ),
                "output_tokens": sum(max(0, _int(item.get("out"))) for item in ordered),
                "iterations": len(ordered),
                "duration_s": max(0.0, duration_s),
                "api_time_s": sum(
                    max(0.0, _float(item.get("api_time"))) for item in ordered
                ),
                "think_time_s": sum(
                    max(0.0, _float(item.get("think_time"))) for item in ordered
                ),
                "tool_use_count": raw_value.get("tool_use_count"),
                "duration_source": (
                    "subagent.duration_ms"
                    if duration_ms >= 0.0
                    else "derived_from_request_timestamps"
                ),
            }
        )
    return requests, agent_groups


def _analyze_reuse(
    requests: list[Request],
    block_size: int,
) -> dict[str, ReuseAggregate]:
    by_start = sorted(requests, key=lambda request: (request.start_s, request.serial))
    by_ready = sorted(requests, key=lambda request: (request.ready_s, request.serial))
    frontiers: dict[str, PrefixFrontier] = {}
    aggregates: dict[str, ReuseAggregate] = defaultdict(ReuseAggregate)
    seen_by_agent: dict[str, int] = defaultdict(int)
    ready_idx = 0

    for request in by_start:
        while (
            ready_idx < len(by_ready)
            and by_ready[ready_idx].ready_s < request.start_s
        ):
            source = by_ready[ready_idx]
            source_sequence = np.asarray(source.hash_ids, dtype=np.int32)
            frontiers.setdefault(source.agent_id, PrefixFrontier()).insert(
                source_sequence
            )
            ready_idx += 1

        sequence = np.asarray(request.hash_ids, dtype=np.int32)
        same_frontier = frontiers.get(request.agent_id)
        same_blocks = same_frontier.query(sequence) if same_frontier else 0
        cross_blocks = 0
        for agent_id, frontier in frontiers.items():
            if agent_id != request.agent_id:
                cross_blocks = max(cross_blocks, frontier.query(sequence))
                if cross_blocks == len(sequence):
                    break

        same_tokens = min(request.input_tokens, same_blocks * block_size)
        cross_tokens = min(request.input_tokens, cross_blocks * block_size)
        if request.role == "main":
            request_class = (
                "main_first"
                if seen_by_agent[request.agent_id] == 0
                else "main_later"
            )
        else:
            request_class = (
                "subagent_first"
                if seen_by_agent[request.agent_id] == 0
                else "subagent_later"
            )
        for key in ("all", request_class):
            aggregates[key].add(
                input_tokens=request.input_tokens,
                same_tokens=same_tokens,
                cross_tokens=cross_tokens,
            )
        seen_by_agent[request.agent_id] += 1
    return aggregates


def _reuse_rows(
    aggregates: dict[str, ReuseAggregate],
    *,
    workflows: int,
    agents: int,
    block_size: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for request_class in (
        "all",
        "main_first",
        "main_later",
        "subagent_first",
        "subagent_later",
    ):
        item = aggregates[request_class]
        rows.append(
            {
                "row_type": "request_class_summary",
                "request_class": request_class,
                "requests": str(item.requests),
                "input_tokens": str(item.input_tokens),
                "same_agent_tokens": str(item.same_tokens),
                "cross_agent_only_tokens": str(item.cross_only_tokens),
                "new_tokens": str(item.new_tokens),
                "raw_cross_overlap_tokens": str(item.raw_cross_tokens),
                "same_agent_share_pct": f"{_percent(item.same_tokens, item.input_tokens):.6f}",
                "cross_agent_only_share_pct": f"{_percent(item.cross_only_tokens, item.input_tokens):.6f}",
                "new_share_pct": f"{_percent(item.new_tokens, item.input_tokens):.6f}",
                "raw_cross_overlap_share_pct": f"{_percent(item.raw_cross_tokens, item.input_tokens):.6f}",
                "cross_positive_requests": str(item.cross_positive_requests),
                "cross_positive_request_share_pct": f"{_percent(item.cross_positive_requests, item.requests):.6f}",
                "workflows": str(workflows),
                "agents": str(agents),
                "block_size_tokens": str(block_size),
                "source": SOURCE_NAME,
                "evidence_tier": REUSE_EVIDENCE_TIER,
                "note": (
                    "Exact ordered-prefix attribution; eligible sources completed "
                    "strictly before target start. Cross-agent-only is marginal "
                    "reuse beyond the best same-agent prefix."
                ),
            }
        )
    return rows


def _value_rows(
    groups: list[dict[str, object]],
) -> tuple[list[dict[str, str]], dict[str, float]]:
    durations = np.asarray([_float(row["duration_s"]) for row in groups], dtype=float)
    mean_context = np.asarray(
        [_float(row["mean_context_tokens"]) for row in groups], dtype=float
    )
    q25_mean_context = float(np.quantile(mean_context, 0.25))
    q90_duration = float(np.quantile(durations, 0.90))

    rows: list[dict[str, str]] = []
    highlighted_iterations: list[float] = []
    for group in groups:
        is_short_long = (
            _float(group["mean_context_tokens"]) <= q25_mean_context
            and _float(group["duration_s"]) >= q90_duration
        )
        if is_short_long:
            highlighted_iterations.append(_float(group["iterations"]))
        rows.append(
            {
                "row_type": "subagent",
                "trace_id": str(group["trace_id"]),
                "agent_id": str(group["agent_id"]),
                "model": str(group["model"]),
                "status": str(group["status"]),
                "initial_context_tokens": str(_int(group["initial_context_tokens"])),
                "final_context_tokens": str(_int(group["final_context_tokens"])),
                "max_context_tokens": str(_int(group["max_context_tokens"])),
                "mean_context_tokens": f"{_float(group['mean_context_tokens']):.6f}",
                "cumulative_input_tokens": str(_int(group["cumulative_input_tokens"])),
                "output_tokens": str(_int(group["output_tokens"])),
                "iterations": str(_int(group["iterations"])),
                "duration_s": f"{_float(group['duration_s']):.6f}",
                "api_time_s": f"{_float(group['api_time_s']):.6f}",
                "think_time_s": f"{_float(group['think_time_s']):.6f}",
                "tool_use_count": (
                    ""
                    if group.get("tool_use_count") is None
                    else str(group["tool_use_count"])
                ),
                "short_context_long_runtime": str(is_short_long).lower(),
                "duration_source": str(group["duration_source"]),
                "source": SOURCE_NAME,
                "evidence_tier": VALUE_EVIDENCE_TIER,
                "note": (
                    "Short-long means mean request context <= P25 and duration >= P90; "
                    "it is a backup-opportunity proxy, not critical-path ground truth."
                ),
            }
        )

    metrics = (
        ("Initial context", "initial_context_tokens"),
        ("Mean context", "mean_context_tokens"),
        ("Max context", "max_context_tokens"),
        ("Iterations", "iterations"),
        ("Output tokens", "output_tokens"),
        ("Between-call time", "think_time_s"),
    )
    correlations: dict[str, float] = {}
    for label, field in metrics:
        value = _spearman(
            [_float(group[field]) for group in groups],
            [_float(group["duration_s"]) for group in groups],
        )
        correlations[label] = value
        rows.append(
            {
                "row_type": "spearman_summary",
                "metric": label,
                "spearman_rho_with_duration": f"{value:.6f}",
                "agents": str(len(groups)),
                "source": SOURCE_NAME,
                "evidence_tier": VALUE_EVIDENCE_TIER,
                "note": "Agent-level Spearman correlation with subagent.duration_ms.",
            }
        )

    summary = {
        "q25_mean_context": q25_mean_context,
        "q90_duration": q90_duration,
        "highlight_count": float(len(highlighted_iterations)),
        "highlight_median_iterations": (
            float(np.median(highlighted_iterations))
            if highlighted_iterations
            else 0.0
        ),
        "overall_median_iterations": float(
            np.median([_float(group["iterations"]) for group in groups])
        ),
    }
    rows.append(
        {
            "row_type": "threshold_summary",
            "mean_context_p25_tokens": f"{q25_mean_context:.6f}",
            "duration_p90_s": f"{q90_duration:.6f}",
            "short_context_long_runtime_agents": str(int(summary["highlight_count"])),
            "short_long_median_iterations": f"{summary['highlight_median_iterations']:.6f}",
            "overall_median_iterations": f"{summary['overall_median_iterations']:.6f}",
            "agents": str(len(groups)),
            "source": SOURCE_NAME,
            "evidence_tier": VALUE_EVIDENCE_TIER,
            "note": (
                "Quantile thresholds are descriptive and fixed from this trace "
                "population; no causal criticality claim is implied."
            ),
        }
    )
    summary.update(correlations)
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8, zorder=0)


def _render_reuse(rows: list[dict[str, str]]) -> None:
    summaries = {
        row["request_class"]: row
        for row in rows
        if row.get("row_type") == "request_class_summary"
    }
    same_color = "#2563EB"
    cross_color = "#F59E0B"
    new_color = "#CBD5E1"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 4.75),
        gridspec_kw={"width_ratios": [0.82, 1.45], "wspace": 0.36},
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.13)
    overall = summaries["all"]
    overall_values = [
        _float(overall["same_agent_share_pct"]),
        _float(overall["cross_agent_only_share_pct"]),
        _float(overall["new_share_pct"]),
    ]
    left = 0.0
    for value, color in zip(
        overall_values, (same_color, cross_color, new_color)
    ):
        axes[0].barh(
            [0],
            [value],
            left=[left],
            height=0.34,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        left += value
    axes[0].text(
        overall_values[0] / 2.0,
        0,
        f"{overall_values[0]:.1f}%\nsame agent",
        ha="center",
        va="center",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].annotate(
        f"Cross-agent-only\n{overall_values[1]:.2f}%",
        xy=(overall_values[0] + overall_values[1] / 2.0, 0.17),
        xytext=(74, 0.68),
        textcoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "-", "color": cross_color, "linewidth": 1.4},
        color="#92400E",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(-0.55, 0.80)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Share of input KV tokens")
    axes[0].set_title("(a) Overall exclusive attribution", loc="left", fontweight="bold")
    _style_axis(axes[0])

    classes = ("main_later", "subagent_first", "subagent_later")
    labels = ("Main later", "Subagent first", "Subagent later")
    y = np.arange(len(classes))
    lefts = np.zeros(len(classes), dtype=float)
    for field, color, label in (
        ("same_agent_share_pct", same_color, "Same-agent"),
        ("cross_agent_only_share_pct", cross_color, "Cross-agent-only"),
        ("new_share_pct", new_color, "New KV"),
    ):
        values = np.asarray([_float(summaries[key][field]) for key in classes])
        axes[1].barh(
            y,
            values,
            left=lefts,
            height=0.54,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for idx, value in enumerate(values):
            if value >= 4.5:
                text_color = "white" if color != new_color else "#374151"
                axes[1].text(
                    lefts[idx] + value / 2.0,
                    idx,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )
        lefts += values
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Share of input KV tokens")
    axes[1].set_title(
        "(b) First-turn inheritance is the exception",
        loc="left",
        fontweight="bold",
    )
    _style_axis(axes[1])

    fig.suptitle(
        "Real CC Trace: Useful KV Reuse Is Overwhelmingly Agent-Local",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.925),
        ncol=3,
        frameon=False,
    )
    for path in (REUSE_PNG, REUSE_BEAMER):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(REUSE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(REUSE_PNG, REUSE_BEAMER)


def _render_value(
    rows: list[dict[str, str]],
    summary: dict[str, float],
) -> None:
    agents = [row for row in rows if row.get("row_type") == "subagent"]
    mean_context_tokens = np.asarray(
        [_float(row["mean_context_tokens"]) for row in agents], dtype=float
    )
    duration_minutes = np.asarray(
        [_float(row["duration_s"]) / 60.0 for row in agents],
        dtype=float,
    )
    mean_context_k_tokens = mean_context_tokens / 1000.0

    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.13)
    ax.scatter(
        mean_context_k_tokens,
        duration_minutes,
        color="#2563EB",
        s=24,
        alpha=0.32,
        linewidths=0,
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean request context length (K tokens)", fontsize=12)
    ax.set_ylabel("Subagent execution time (minutes)", fontsize=12)
    ax.grid(color="#E5E7EB", linewidth=0.7, which="both", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.035,
        0.94,
        f"Spearman $\\rho$ = {summary['Mean context']:+.2f}\n"
        f"$n$ = {len(agents):,} completed subagents",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="#1F2937",
        bbox={
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
            "boxstyle": "round,pad=0.35",
            "alpha": 0.95,
        },
    )

    fig.suptitle(
        "Real CC Trace: Mean Request Context vs. Subagent Execution Time",
        x=0.10,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    for path in (VALUE_PNG, VALUE_BEAMER):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(VALUE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(VALUE_PNG, VALUE_BEAMER)


def main() -> None:
    if not TRACE_PATH.exists():
        raise FileNotFoundError(f"Missing raw CC trace: {TRACE_PATH}")

    aggregate_reuse: dict[str, ReuseAggregate] = defaultdict(ReuseAggregate)
    all_groups: list[dict[str, object]] = []
    workflow_count = 0
    agent_keys: set[tuple[str, str]] = set()
    block_sizes: set[int] = set()

    with TRACE_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            trace = json.loads(line)
            workflow_count += 1
            block_size = _int(trace.get("block_size"), 64)
            block_sizes.add(block_size)
            requests, groups = _flatten_requests(trace)
            trace_reuse = _analyze_reuse(requests, block_size)
            for key, source in trace_reuse.items():
                target = aggregate_reuse[key]
                target.requests += source.requests
                target.input_tokens += source.input_tokens
                target.same_tokens += source.same_tokens
                target.cross_only_tokens += source.cross_only_tokens
                target.new_tokens += source.new_tokens
                target.raw_cross_tokens += source.raw_cross_tokens
                target.cross_positive_requests += source.cross_positive_requests
            all_groups.extend(groups)
            trace_id = str(trace.get("id") or "")
            agent_keys.update((trace_id, request.agent_id) for request in requests)
            if line_number % 25 == 0:
                print(
                    f"parsed {line_number} workflows, "
                    f"{aggregate_reuse['all'].requests:,} requests",
                    flush=True,
                )

    if len(block_sizes) != 1:
        raise ValueError(f"Expected one trace block size, found {sorted(block_sizes)}")
    block_size = next(iter(block_sizes))
    reuse_rows = _reuse_rows(
        aggregate_reuse,
        workflows=workflow_count,
        agents=len(agent_keys),
        block_size=block_size,
    )
    value_rows, value_summary = _value_rows(all_groups)
    _write_csv(REUSE_CSV, reuse_rows)
    _write_csv(VALUE_CSV, value_rows)
    _render_reuse(reuse_rows)
    _render_value(value_rows, value_summary)

    overall = aggregate_reuse["all"]
    print(
        "reuse: "
        f"same={_percent(overall.same_tokens, overall.input_tokens):.3f}%, "
        f"cross-only={_percent(overall.cross_only_tokens, overall.input_tokens):.3f}%, "
        f"new={_percent(overall.new_tokens, overall.input_tokens):.3f}%, "
        f"raw-cross={_percent(overall.raw_cross_tokens, overall.input_tokens):.3f}%",
        flush=True,
    )
    print(
        "context/value: "
        f"agents={len(all_groups):,}, "
        f"short-long={int(value_summary['highlight_count'])}, "
        f"rho(mean-context,duration)={value_summary['Mean context']:.3f}, "
        f"rho(iterations,duration)={value_summary['Iterations']:.3f}",
        flush=True,
    )
    print(f"wrote {REUSE_CSV}")
    print(f"wrote {VALUE_CSV}")
    print(f"wrote {REUSE_PNG}")
    print(f"wrote {VALUE_PNG}")


if __name__ == "__main__":
    main()
