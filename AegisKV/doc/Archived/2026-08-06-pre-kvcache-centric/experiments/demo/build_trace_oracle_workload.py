"""Build a compact Real-CC workload carrying measured trace-oracle Slack."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from benchmark.agentic_workflows.weka_trace import (
    WEKA_DATASET,
    trace_to_workload,
)

from .wct_cdf_client import has_fan_out_and_main_fan_in, load_workflows
from .workflow_client import DEFAULT_TRACE, DEFAULT_TRACE_ORACLE


DEFAULT_SOURCE = (
    "https://huggingface.co/datasets/semianalysisai/"
    "cc-traces-weka-062126/resolve/main/traces.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_TRACE_ORACLE


def _open_source(source: str) -> BinaryIO:
    path = Path(source)
    if path.is_file():
        return path.open("rb")
    request = urllib.request.Request(
        source,
        headers={"User-Agent": "SGLSim-AegisKV-trace-oracle/1"},
    )
    # This host installs its outbound proxy CA in the system trust bundle,
    # while the uv Python's bundled certifi store does not contain it.
    context = ssl.create_default_context(
        cafile="/etc/pki/tls/certs/ca-bundle.crt"
    )
    return urllib.request.urlopen(request, timeout=120, context=context)


def _iter_jsonl(handle: BinaryIO) -> Iterator[tuple[bytes, dict[str, Any]]]:
    for raw_line in handle:
        if not raw_line.strip():
            continue
        yield raw_line, json.loads(raw_line)


def selected_workflow_ids(base_trace: Path, *, max_workflows: int) -> list[str]:
    selected = [
        workflow_id
        for workflow_id, rows in load_workflows(base_trace).items()
        if has_fan_out_and_main_fan_in(rows)
    ]
    return selected[: max(int(max_workflows), 1)]


def build_trace_oracle_workload(
    *,
    source: str,
    base_trace: Path,
    output: Path,
    max_workflows: int,
) -> dict[str, Any]:
    workflow_ids = selected_workflow_ids(
        base_trace.resolve(), max_workflows=max_workflows
    )
    wanted = set(workflow_ids)
    found: dict[str, list[dict[str, object]]] = {}
    source_digest = hashlib.sha256()
    bytes_scanned = 0
    rows_scanned = 0
    with closing(_open_source(source)) as handle:
        for raw_line, raw in _iter_jsonl(handle):
            bytes_scanned += len(raw_line)
            rows_scanned += 1
            source_digest.update(raw_line)
            workflow_id = str(raw.get("id", ""))
            if workflow_id in wanted:
                found[workflow_id] = trace_to_workload(raw)
                if set(found) == wanted:
                    break

    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"source trace is missing workflows: {missing}")

    base = load_workflows(base_trace.resolve())
    output_rows: list[dict[str, object]] = []
    for workflow_id in workflow_ids:
        enriched = found[workflow_id]
        expected = {
            str(row["rid"]): (
                int(row["prompt_len"]),
                int(row["output_len"]),
                int(row["stage_id"]),
                str(row["agent_id"]),
            )
            for row in base[workflow_id]
        }
        observed = {
            str(row["rid"]): (
                int(row["prompt_len"]),
                int(row["output_len"]),
                int(row["stage_id"]),
                str(row["agent_id"]),
            )
            for row in enriched
        }
        if observed != expected:
            raise RuntimeError(
                f"oracle rebuild does not match normalized base trace for {workflow_id}"
            )
        output_rows.extend(enriched)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(output)

    manifest = {
        "schema_version": 1,
        "semantics": "weka_api_completion_cpm_v1",
        "evidence_role": "trace-oracle upper bound; Algorithm 1 bypassed",
        "uses_future_information": True,
        "source_dataset": WEKA_DATASET,
        "source": source,
        "source_prefix_sha256": source_digest.hexdigest(),
        "source_bytes_scanned": bytes_scanned,
        "source_rows_scanned": rows_scanned,
        "base_trace": str(base_trace.resolve()),
        "workflow_ids": workflow_ids,
        "workflow_count": len(workflow_ids),
        "request_count": len(output_rows),
        "output": str(output.resolve()),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive measured future-completion Slack from the Weka trace."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--base-trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workflows", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_trace_oracle_workload(
        source=args.source,
        base_trace=args.base_trace,
        output=args.output,
        max_workflows=args.max_workflows,
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
