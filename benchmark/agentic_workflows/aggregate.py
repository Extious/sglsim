from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .config import DEFAULT_RESULTS_ROOT


DEFAULT_INPUT = DEFAULT_RESULTS_ROOT / "agentic_workflow_paper_candidates.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "agentic_workflow_paper_aggregate.csv"

GROUP_FIELDS = (
    "scenario",
    "dp_size",
    "workflow_count",
    "turns",
    "groups",
    "prompts_per_group",
    "system_prompt_len",
    "question_len",
    "output_len",
    "range_ratio",
    "max_concurrency",
    "critical_agent_ratio",
    "failover_replay",
    "failover_injection",
    "failover_retry_rank_offset",
    "failover_retry_policy",
    "aegis_policy",
    "aegis_ablation",
    "aegis_gpu_budget_fraction",
    "failure_after_output_tokens",
    "backup_kv",
)
SUM_FIELDS = (
    "completed",
    "failed",
    "replay_rows",
    "host_lookup_hits",
    "peer_lookup_hits",
    "replay_action_host_restore",
    "replay_action_peer_gpu_restore",
    "replay_action_recompute",
    "replay_action_none",
    "replay_action_other",
    "replay_tokens_host_restore",
    "replay_tokens_peer_gpu_restore",
    "replay_tokens_recompute",
    "replay_tokens_none",
    "replay_tokens_other",
    "replay_data_plane_metadata_materialization",
    "replay_data_plane_physical_tensor_copy",
    "replay_data_plane_recompute",
    "replay_data_plane_none",
    "replay_data_plane_other",
)
MEAN_FIELDS = (
    "restored_tokens",
    "recompute_tokens",
    "mean_e2e_ms",
    "p90_e2e_ms",
    "mean_restore_sleep_ms",
)


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else 0.0


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in GROUP_FIELDS)
        groups[key].append(row)

    out: list[dict[str, object]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        row: dict[str, object] = dict(zip(GROUP_FIELDS, key, strict=True))
        row["trials"] = len(group_rows)
        for field in SUM_FIELDS:
            row[f"{field}_total"] = int(sum(_float(item, field) for item in group_rows))
        for field in MEAN_FIELDS:
            values = [_float(item, field) for item in group_rows]
            row[f"{field}_mean"] = statistics.mean(values)
            row[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{field}_ci95"] = _ci95(values)
        tiers = sorted({item.get("evidence_tier", "") for item in group_rows})
        row["evidence_tier"] = ",".join(tier for tier in tiers if tier)
        row["source_run_dirs"] = ";".join(
            item.get("source_run_dir", "") for item in group_rows
        )
        out.append(row)
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> int:
    if not rows:
        return 0
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def aggregate_csv(input_csv: Path, output_csv: Path) -> int:
    return write_csv(output_csv, aggregate_rows(read_csv(input_csv)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate agentic workflow paper candidate rows by scenario."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    count = aggregate_csv(args.input_csv, args.output_csv)
    print(args.output_csv if count else "no rows written")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
