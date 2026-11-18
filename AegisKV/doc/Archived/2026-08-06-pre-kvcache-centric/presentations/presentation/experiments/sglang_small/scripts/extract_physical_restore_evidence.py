#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


STORAGE_QUERY_RE = re.compile(
    r"_storage_hit_query_tokens: req=(?P<req>[^,]+), "
    r"prefix_pages=(?P<prefix_pages>\d+), "
    r"data_pages=(?P<data_pages>\d+), "
    r"range=(?P<range_tokens>\d+), "
    r"tokens_to_fetch=(?P<tokens_to_fetch>\d+), "
    r"boundary_token=(?P<boundary_token>\d+), "
    r"total_tokens=(?P<total_tokens>\d+)"
)
PREFETCH_ACCEPTED_RE = re.compile(
    r"Prefetch ACCEPTED for request (?P<req>[^:]+): "
    r"(?P<pages>\d+) pages \((?P<tokens>\d+) tokens\)"
)
PREFETCH_REVOKED_RE = re.compile(
    r"Prefetch REVOKED for request (?P<req>[^:]+): "
    r"only (?P<hits>\d+) storage hits"
)
PREFETCH_IO_START_RE = re.compile(
    r"Prefetch IO START for (?P<req>[^:]+): "
    r"pages=(?P<pages>\d+) tokens=(?P<tokens>\d+)"
)
PREFETCH_IO_END_RE = re.compile(
    r"Prefetch IO END for (?P<req>[^:]+): "
    r"completed=(?P<completed>\d+) / expected=(?P<expected>\d+) tokens"
)
PREFILL_RE = re.compile(r"Prefill batch, .*#cached-token: (?P<cached>\d+)")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_provenance(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _as_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _state(
    *,
    backed_up_tokens: int,
    max_data_pages: int,
    accepted_tokens: int,
    io_expected_tokens: int,
    io_completed_tokens: int,
    cached_after_accept_tokens: int,
) -> str:
    if backed_up_tokens <= 0:
        return "no_host_frontier"
    if max_data_pages <= 0:
        return "host_frontier_not_visible_to_retry"
    if accepted_tokens <= 0:
        return "host_frontier_visible_not_prefetched"
    if io_expected_tokens > 0 and io_completed_tokens >= io_expected_tokens:
        if cached_after_accept_tokens > 0:
            return "full_prefetch_materialized"
        return "full_prefetch_completed_not_admitted"
    if cached_after_accept_tokens > 0:
        return "partial_async_materialized"
    if io_completed_tokens > 0:
        return "partial_prefetch_completed_not_admitted"
    return "prefetch_accepted_no_materialization"


def parse_server_log(path: Path) -> dict[str, Any]:
    accepted_seen = False
    storage_queries = 0
    positive_storage_queries = 0
    revoked_prefetches = 0
    accepted_prefetches = 0
    io_starts = 0
    io_ends = 0
    data_pages_sum = 0
    max_data_pages = 0
    accepted_pages_sum = 0
    accepted_tokens_sum = 0
    io_start_pages_sum = 0
    io_start_tokens_sum = 0
    io_expected_tokens_sum = 0
    io_completed_tokens_sum = 0
    cached_after_accept_total = 0
    cached_after_accept_max = 0
    cached_after_accept_batches = 0
    positive_cached_after_accept_batches = 0
    request_ids: set[str] = set()
    accepted_request_ids: set[str] = set()
    io_completed_request_ids: set[str] = set()

    if not path.exists():
        return {
            "server_log_exists": False,
            "storage_query_count": 0,
            "storage_query_positive_count": 0,
            "max_data_pages": 0,
            "storage_data_pages_sum": 0,
            "prefetch_accepted_count": 0,
            "prefetch_revoked_count": 0,
            "prefetch_io_start_count": 0,
            "prefetch_io_end_count": 0,
            "prefetch_accepted_pages_sum": 0,
            "prefetch_accepted_tokens_sum": 0,
            "prefetch_io_start_pages_sum": 0,
            "prefetch_io_start_tokens_sum": 0,
            "prefetch_io_expected_tokens_sum": 0,
            "prefetch_io_completed_tokens_sum": 0,
            "prefetch_io_completion_ratio": 0.0,
            "retry_cached_tokens_after_accept_total": 0,
            "retry_cached_tokens_after_accept_max": 0,
            "retry_cached_batches_after_accept": 0,
            "retry_positive_cached_batches_after_accept": 0,
            "storage_request_count": 0,
            "accepted_request_count": 0,
            "io_completed_request_count": 0,
        }

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := STORAGE_QUERY_RE.search(line):
            storage_queries += 1
            req = match.group("req")
            request_ids.add(req)
            data_pages = _as_int(match.group("data_pages"))
            data_pages_sum += data_pages
            max_data_pages = max(max_data_pages, data_pages)
            if data_pages > 0:
                positive_storage_queries += 1
            continue
        if match := PREFETCH_ACCEPTED_RE.search(line):
            accepted_seen = True
            accepted_prefetches += 1
            req = match.group("req")
            request_ids.add(req)
            accepted_request_ids.add(req)
            accepted_pages_sum += _as_int(match.group("pages"))
            accepted_tokens_sum += _as_int(match.group("tokens"))
            continue
        if match := PREFETCH_REVOKED_RE.search(line):
            revoked_prefetches += 1
            request_ids.add(match.group("req"))
            continue
        if match := PREFETCH_IO_START_RE.search(line):
            io_starts += 1
            req = match.group("req")
            request_ids.add(req)
            io_start_pages_sum += _as_int(match.group("pages"))
            io_start_tokens_sum += _as_int(match.group("tokens"))
            continue
        if match := PREFETCH_IO_END_RE.search(line):
            io_ends += 1
            req = match.group("req")
            request_ids.add(req)
            io_completed_request_ids.add(req)
            io_completed_tokens_sum += _as_int(match.group("completed"))
            io_expected_tokens_sum += _as_int(match.group("expected"))
            continue
        if accepted_seen and (match := PREFILL_RE.search(line)):
            cached = _as_int(match.group("cached"))
            cached_after_accept_batches += 1
            cached_after_accept_total += cached
            cached_after_accept_max = max(cached_after_accept_max, cached)
            if cached > 0:
                positive_cached_after_accept_batches += 1

    completion_ratio = (
        io_completed_tokens_sum / io_expected_tokens_sum
        if io_expected_tokens_sum > 0
        else 0.0
    )
    return {
        "server_log_exists": True,
        "storage_query_count": storage_queries,
        "storage_query_positive_count": positive_storage_queries,
        "max_data_pages": max_data_pages,
        "storage_data_pages_sum": data_pages_sum,
        "prefetch_accepted_count": accepted_prefetches,
        "prefetch_revoked_count": revoked_prefetches,
        "prefetch_io_start_count": io_starts,
        "prefetch_io_end_count": io_ends,
        "prefetch_accepted_pages_sum": accepted_pages_sum,
        "prefetch_accepted_tokens_sum": accepted_tokens_sum,
        "prefetch_io_start_pages_sum": io_start_pages_sum,
        "prefetch_io_start_tokens_sum": io_start_tokens_sum,
        "prefetch_io_expected_tokens_sum": io_expected_tokens_sum,
        "prefetch_io_completed_tokens_sum": io_completed_tokens_sum,
        "prefetch_io_completion_ratio": completion_ratio,
        "retry_cached_tokens_after_accept_total": cached_after_accept_total,
        "retry_cached_tokens_after_accept_max": cached_after_accept_max,
        "retry_cached_batches_after_accept": cached_after_accept_batches,
        "retry_positive_cached_batches_after_accept": positive_cached_after_accept_batches,
        "storage_request_count": len(request_ids),
        "accepted_request_count": len(accepted_request_ids),
        "io_completed_request_count": len(io_completed_request_ids),
    }


def summarize_result_dir(result_dir: Path, label: str = "") -> dict[str, Any]:
    summary = _read_json(result_dir / "failure_summary_checked.json")
    if not summary:
        summary = _read_json(result_dir / "failure_summary.json")
    provenance = _read_provenance(result_dir / "provenance.txt")
    restore = parse_server_log(result_dir / "server.log")
    backed_up_tokens = _as_int(summary.get("response_pre_failover_backed_up_tokens"))
    row: dict[str, Any] = {
        "label": label,
        "result_dir": str(result_dir),
        "run_id": result_dir.name,
        "slurm_job_id": provenance.get("slurm_job_id", result_dir.name.rsplit("_", 1)[-1]),
        "remote_host": provenance.get("host", "hkbugpumanode"),
        "model_path": provenance.get("model_path", "Qwen/Qwen3-8B"),
        "enable_remote_backup": provenance.get("enable_remote_backup", ""),
        "disable_failover_remote_lookup": provenance.get(
            "disable_failover_remote_lookup", ""
        ),
        "hicache_storage_prefetch_policy": provenance.get(
            "hicache_storage_prefetch_policy", ""
        ),
        "prompt_repeat": provenance.get("prompt_repeat", ""),
        "unique_request_prefix": provenance.get("unique_request_prefix", ""),
        "failure_delay_s": provenance.get("failure_delay_s", ""),
        "request_count": _as_int(summary.get("count")),
        "success_count": _as_int(summary.get("success_count")),
        "failover_response_count": _as_int(summary.get("failover_response_count")),
        "captured_request_count": _as_int(summary.get("captured_request_count")),
        "dispatched_request_count": _as_int(summary.get("dispatched_request_count")),
        "latency_p95_s": _as_float(summary.get("latency_p95_s")),
        "latency_p99_s": _as_float(summary.get("latency_p99_s")),
        "pre_failover_output_tokens": _as_int(
            summary.get("response_pre_failover_output_tokens")
        ),
        "pre_failover_backed_up_tokens": backed_up_tokens,
        **restore,
    }
    row["physical_restore_state"] = _state(
        backed_up_tokens=backed_up_tokens,
        max_data_pages=_as_int(row.get("max_data_pages")),
        accepted_tokens=_as_int(row.get("prefetch_accepted_tokens_sum")),
        io_expected_tokens=_as_int(row.get("prefetch_io_expected_tokens_sum")),
        io_completed_tokens=_as_int(row.get("prefetch_io_completed_tokens_sum")),
        cached_after_accept_tokens=_as_int(
            row.get("retry_cached_tokens_after_accept_total")
        ),
    )
    row["evidence_tier"] = "sglang-small-physical-restore-smoke"
    row["source"] = "sglang_server_log_physical_restore_parser"
    row["note"] = (
        "parsed from real SGLang server.log and failure_summary; "
        "small-scale physical Host restore evidence, not final policy evidence"
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_int(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def _fmt_float(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def render_markdown(rows: list[dict[str, Any]], *, csv_path: Path, json_path: Path | None) -> str:
    table = [
        "| Label | Job | Prefetch policy | Same write cost? | Host frontier tokens | Max data pages/request | Prefetch completed tokens | Retry cached tokens | State |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        same_write = (
            "yes, lookup disabled"
            if str(row.get("disable_failover_remote_lookup")) == "1"
            else "yes, lookup enabled"
            if str(row.get("enable_remote_backup")) == "1"
            and int(float(row.get("pre_failover_backed_up_tokens") or 0)) > 0
            and str(row.get("prompt_repeat")) == "256"
            else "no"
        )
        table.append(
            "| {label} | {job} | {policy} | {same_write} | {frontier} | {data_pages} | {completed} | {cached} | {state} |".format(
                label=row.get("label", ""),
                job=row.get("slurm_job_id", ""),
                policy=row.get("hicache_storage_prefetch_policy", ""),
                same_write=same_write,
                frontier=_fmt_int(row.get("pre_failover_backed_up_tokens")),
                data_pages=_fmt_int(row.get("max_data_pages")),
                completed=_fmt_int(row.get("prefetch_io_completed_tokens_sum")),
                cached=_fmt_int(row.get("retry_cached_tokens_after_accept_total")),
                state=row.get("physical_restore_state", ""),
            )
        )

    states = sorted({str(row.get("physical_restore_state", "")) for row in rows})
    return "\n".join(
        [
            "# SGLang Physical Host-Restore Evidence Summary",
            "",
            "This artifact makes the current physical Host-restore evidence auditable. "
            "It is parsed from real SGLang `server.log` files and failure summaries, "
            "not from SGLSim metadata rows.",
            "",
            "Generated data:",
            "",
            f"- CSV: `{csv_path}`",
            f"- JSON: `{json_path}`" if json_path is not None else "- JSON: not emitted",
            "- Parser: `scripts/extract_physical_restore_evidence.py`",
            f"- Rows: {len(rows)}",
            f"- States: `{', '.join(states)}`",
            "",
            "## Current Rows",
            "",
            *table,
            "",
            "## Interpretation",
            "",
            "The current SGLang evidence separates three cases that were previously "
            "mixed in prose:",
            "",
            "1. Host frontier exists but is hidden from the retry rank: lookup-disabled "
            "controls write remote-backup tokens, but the retry path sees no `data_pages` "
            "and admits no storage-restored cached tokens.",
            "2. Full physical Host restore materializes into retry cached tokens: "
            "wait-complete rows show positive `data_pages`, completed prefetch IO, and "
            "retry `#cached-token` equal to the restored page count.",
            "3. Best-effort asynchronous restore can partially materialize before "
            "admission: best-effort rows may accept prefetch, complete a prefix subset, "
            "and admit only the pages ready at retry prefill.",
            "",
            "This is stronger than metadata-only SGLSim evidence for Host restore "
            "materialization. It still is not final HPCA policy evidence because it is "
            "small-scale, Host-only, and does not validate peer GPUReplica P2P physical "
            "copy or a repeated full AegisKV policy matrix.",
            "",
        ]
    )


def parse_labelled_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract physical Host restore/materialization evidence from "
            "SGLang failure-smoke result directories."
        )
    )
    parser.add_argument(
        "--result-dir",
        action="append",
        required=True,
        help="Result directory, or label=/path/to/result_dir.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        summarize_result_dir(path, label=label)
        for label, path in (parse_labelled_dir(value) for value in args.result_dir)
    ]
    write_csv(args.output_csv, rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            render_markdown(
                rows,
                csv_path=args.output_csv,
                json_path=args.output_json,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output_csv": str(args.output_csv),
                "output_json": str(args.output_json or ""),
                "output_md": str(args.output_md or ""),
                "rows": len(rows),
                "states": sorted({str(row["physical_restore_state"]) for row in rows}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
