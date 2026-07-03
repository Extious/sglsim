from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import DEFAULT_RESULTS_ROOT


DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "agentic_workflow_paper_candidates.csv"


PAPER_FIELDS = [
    "scenario",
    "seed",
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
    "completed",
    "failed",
    "replay_rows",
    "host_lookup_hits",
    "peer_lookup_hits",
    "restored_tokens",
    "recompute_tokens",
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
    "mean_e2e_ms",
    "p90_e2e_ms",
    "mean_restore_sleep_ms",
    "evidence_tier",
    "source_run_dir",
]


def paper_rows_from_summary(summary_csv: Path) -> list[dict[str, str]]:
    with Path(summary_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "scenario": row.get("scenario", ""),
                "seed": row.get("seed", ""),
                "dp_size": row.get("dp_size", ""),
                "workflow_count": row.get("workflow_count", ""),
                "turns": row.get("turns", ""),
                "groups": row.get("groups", ""),
                "prompts_per_group": row.get("prompts_per_group", ""),
                "system_prompt_len": row.get("system_prompt_len", ""),
                "question_len": row.get("question_len", ""),
                "output_len": row.get("output_len", ""),
                "range_ratio": row.get("range_ratio", ""),
                "max_concurrency": row.get("max_concurrency", ""),
                "critical_agent_ratio": row.get("critical_agent_ratio", ""),
                "failover_replay": row.get("failover_replay", ""),
                "failover_injection": row.get("failover_injection", ""),
                "failover_retry_rank_offset": row.get(
                    "failover_retry_rank_offset", ""
                ),
                "failover_retry_policy": row.get("failover_retry_policy", ""),
                "aegis_policy": row.get("aegis_policy", ""),
                "aegis_ablation": row.get("aegis_ablation", ""),
                "aegis_gpu_budget_fraction": row.get(
                    "aegis_gpu_budget_fraction", ""
                ),
                "failure_after_output_tokens": row.get(
                    "failure_after_output_tokens", ""
                ),
                "backup_kv": row.get("backup_kv", ""),
                "completed": row.get("completed", ""),
                "failed": row.get("failed", ""),
                "replay_rows": row.get("replay_rows", ""),
                "host_lookup_hits": row.get("host_lookup_hits", ""),
                "peer_lookup_hits": row.get("peer_lookup_hits", ""),
                "restored_tokens": row.get("restored_tokens", ""),
                "recompute_tokens": row.get("recompute_tokens", ""),
                "replay_action_host_restore": row.get(
                    "replay_action_host_restore", ""
                ),
                "replay_action_peer_gpu_restore": row.get(
                    "replay_action_peer_gpu_restore", ""
                ),
                "replay_action_recompute": row.get("replay_action_recompute", ""),
                "replay_action_none": row.get("replay_action_none", ""),
                "replay_action_other": row.get("replay_action_other", ""),
                "replay_tokens_host_restore": row.get("replay_tokens_host_restore", ""),
                "replay_tokens_peer_gpu_restore": row.get(
                    "replay_tokens_peer_gpu_restore", ""
                ),
                "replay_tokens_recompute": row.get("replay_tokens_recompute", ""),
                "replay_tokens_none": row.get("replay_tokens_none", ""),
                "replay_tokens_other": row.get("replay_tokens_other", ""),
                "replay_data_plane_metadata_materialization": row.get(
                    "replay_data_plane_metadata_materialization", ""
                ),
                "replay_data_plane_physical_tensor_copy": row.get(
                    "replay_data_plane_physical_tensor_copy", ""
                ),
                "replay_data_plane_recompute": row.get(
                    "replay_data_plane_recompute", ""
                ),
                "replay_data_plane_none": row.get("replay_data_plane_none", ""),
                "replay_data_plane_other": row.get("replay_data_plane_other", ""),
                "mean_e2e_ms": row.get("mean_e2e_ms", ""),
                "p90_e2e_ms": row.get("p90_e2e_ms", ""),
                "mean_restore_sleep_ms": row.get("mean_restore_sleep_ms", ""),
                "evidence_tier": row.get(
                    "evidence_tier", "sglsim-agentic-online-preliminary"
                ),
                "source_run_dir": row.get("run_dir", ""),
            }
        )
    return out


def write_paper_candidates(summary_csv: Path, output_csv: Path) -> int:
    rows = paper_rows_from_summary(summary_csv)
    if not rows:
        return 0
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project agentic workflow summaries into paper-facing candidate rows."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "agentic_workflow_online_summary.csv",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    count = write_paper_candidates(args.summary_csv, args.output_csv)
    print(args.output_csv if count else "no rows written")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
