#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "physical_kv_copy_aggregate"


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _ci95(values: list[float]) -> float:
    return 1.96 * _stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(
        Path(results_dir).glob("physical_kv_copy_*/physical_kv_copy_microbench.csv")
    ):
        payload = _read_json(csv_path.with_suffix(".json"))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        for row in _read_csv(csv_path):
            out: dict[str, Any] = dict(row)
            out["result_dir"] = str(csv_path.parent)
            out["summary_json"] = str(csv_path.with_suffix(".json"))
            out["slurm_job_id"] = row.get("slurm_job_id") or metadata.get("slurm_job_id", "")
            out["host"] = row.get("host") or metadata.get("host", "")
            out["cuda_visible_devices"] = metadata.get("cuda_visible_devices", "")
            out["torch_version"] = metadata.get("torch_version", "")
            out["device_count"] = metadata.get("torch_cuda_device_count", "")
            out["bytes_per_token"] = row.get("bytes_per_token") or metadata.get("bytes_per_token", "")
            rows.append(out)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("available", "")).lower() != "true":
            continue
        key = (str(row.get("path") or ""), int(_float(row.get("tokens"))))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (path, tokens), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        p50_ms = [_float(row.get("p50_ms")) for row in group]
        p95_ms = [_float(row.get("p95_ms")) for row in group]
        p50_gbps = [_float(row.get("p50_gbps")) for row in group]
        bytes_values = [_float(row.get("bytes")) for row in group]
        out.append(
            {
                "path": path,
                "tokens": tokens,
                "bytes": int(round(_mean(bytes_values))),
                "mib": _mean([_float(row.get("mib")) for row in group]),
                "repeat_count": len(group),
                "slurm_job_ids": ";".join(str(row.get("slurm_job_id") or "") for row in group),
                "hosts": ";".join(sorted({str(row.get("host") or "") for row in group})),
                "p50_ms_mean": _mean(p50_ms),
                "p50_ms_std": _stdev(p50_ms),
                "p50_ms_ci95": _ci95(p50_ms),
                "p95_ms_mean": _mean(p95_ms),
                "p95_ms_std": _stdev(p95_ms),
                "p95_ms_ci95": _ci95(p95_ms),
                "p50_gbps_mean": _mean(p50_gbps),
                "p50_gbps_std": _stdev(p50_gbps),
                "p50_gbps_ci95": _ci95(p50_gbps),
                "bytes_per_token": str(group[0].get("bytes_per_token") or ""),
                "evidence_tier": "sglang-small-physical-copy-smoke",
                "source": "torch_cuda_physical_kv_copy_microbench_aggregate",
                "note": (
                    "aggregate of small-scale CUDA KV-sized copy microbench runs; "
                    "not end-to-end AegisKV policy evidence"
                ),
            }
        )
    return out


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


def render_markdown(rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> str:
    job_ids = sorted({str(row.get("slurm_job_id") or "") for row in rows if row.get("slurm_job_id")})
    hosts = sorted({str(row.get("host") or "") for row in rows if row.get("host")})
    lines = [
        "# Physical KV Copy Microbenchmark Aggregate",
        "",
        "This report aggregates small-scale CUDA copy calibration runs for AegisKV.",
        "It is physical movement calibration, not final end-to-end policy evidence.",
        "",
        f"- jobs: `{';'.join(job_ids)}`",
        f"- hosts: `{';'.join(hosts)}`",
        "",
        "## Aggregate",
        "",
        "| Path | Tokens | Repeats | p50 ms | 95% CI | p50 GB/s | Jobs |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate:
        lines.append(
            "| {path} | {tokens} | {count} | {p50:.3f} | {ci:.3f} | {gbps:.2f} | {jobs} |".format(
                path=row.get("path", ""),
                tokens=int(row.get("tokens") or 0),
                count=int(row.get("repeat_count") or 0),
                p50=_float(row.get("p50_ms_mean")),
                ci=_float(row.get("p50_ms_ci95")),
                gbps=_float(row.get("p50_gbps_mean")),
                jobs=row.get("slurm_job_ids", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "These rows calibrate physical H2D/D2H/P2P bandwidth and latency for KV-sized tensors.",
            "They do not prove end-to-end AegisKV failure recovery until the recovery path performs the same physical movement under policy control.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate physical KV copy microbench result directories."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = collect_rows(args.results_dir)
    aggregate = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "physical_kv_copy_rows.csv", rows)
    write_csv(args.output_dir / "physical_kv_copy_aggregate.csv", aggregate)
    (args.output_dir / "physical_kv_copy_aggregate.md").write_text(
        render_markdown(rows, aggregate),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "aggregate_rows": len(aggregate),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
