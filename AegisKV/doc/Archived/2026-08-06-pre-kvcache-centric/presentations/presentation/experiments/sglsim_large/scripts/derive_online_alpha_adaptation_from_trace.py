#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from sglsim.srt.simulation.alpha_policy import choose_alpha, fine_tune_alpha


ROOT = Path(__file__).resolve().parents[3]
SGLSIM_LARGE_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DATA_DIR = ROOT / "figures" / "data" / "measured"
DEFAULT_RUN_ROOT = (
    SGLSIM_LARGE_ROOT
    / "results"
    / "agentic_workflows"
    / "sglsim-staged-13131-alpha-online-20260702"
)
DEFAULT_SWEEP_ROOT = (
    SGLSIM_LARGE_ROOT
    / "results"
    / "agentic_workflows"
    / "sglsim-staged-13131-alpha-online-sweep-20260702"
)
DEFAULT_POLICY_PATH = (
    SGLSIM_LARGE_ROOT / "policies" / "aegiskv_alpha_mdp_policy.json"
)
FAILURE_RATES = (0.001, 0.003, 0.01, 0.03, 0.10)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def _classify(row: dict[str, str]) -> str:
    critical = _bool(row.get("critical_agent", ""))
    decode = _bool(row.get("decode_dominant", ""))
    stage_width = int(float(row.get("stage_width") or 1))
    if not critical:
        return "Non-Critical fan-out"
    if decode:
        return "Long-decode Critical"
    if stage_width == 1:
        return "Singleton Critical"
    return "Other Critical"


def _find_request_details(run_root: Path) -> Path:
    direct = run_root / "request_details.csv"
    if direct.exists():
        return direct
    matches = sorted(run_root.glob("runs/trial_*/**/request_details.csv"))
    if not matches:
        raise FileNotFoundError(f"no request_details.csv under {run_root}")
    return matches[-1]


def _read_manifest(run_root: Path) -> dict:
    path = run_root / "agentic_workflow_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _policy_path_from_manifest(manifest: dict) -> Path:
    for row in manifest.get("plan", []):
        value = row.get("alpha_policy_path")
        if value:
            return Path(value)
    return DEFAULT_POLICY_PATH


def _iter_sweep_request_details(
    run_root: Path,
) -> list[tuple[Path, str, str, str]]:
    manifest = _read_manifest(run_root)
    run_id = str(manifest.get("run_id") or run_root.name)
    out: list[tuple[Path, str, str, str]] = []
    for row in manifest.get("plan", []):
        run_dir = str(row.get("run_dir") or "")
        if not run_dir:
            continue
        details = Path(run_dir) / "request_details.csv"
        if not details.exists():
            continue
        load_level = str(row.get("load_level") or "")
        sweep_point = str(row.get("sweep_point") or Path(run_dir).parent.name)
        out.append((details, run_id, load_level, sweep_point))
    return out


def _series_label(request_class: str, load_level: str) -> str:
    if not load_level or request_class == "Non-Critical fan-out":
        return request_class
    return f"{load_level} / {request_class}"


def _attach_online_tuning(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baselines: dict[str, float] = {}
    for row in rows:
        key = str(row.get("request_class") or row.get("series_label") or "")
        latency = _float(row, "observed_mean_latency_ms", 0.0)  # type: ignore[arg-type]
        if latency <= 0:
            continue
        baselines[key] = min(latency, baselines.get(key, latency))

    for row in rows:
        key = str(row.get("request_class") or row.get("series_label") or "")
        latency = _float(row, "observed_mean_latency_ms", 0.0)  # type: ignore[arg-type]
        baseline = baselines.get(key, latency if latency > 0 else 1.0)
        wct_ratio = latency / max(baseline, 1e-9)
        update = fine_tune_alpha(
            base_alpha=row.get("policy_alpha", 0.0),
            observed_wct_ratio=wct_ratio,
            failure_rate_per_window=row.get("failure_rate_per_window", 0.01),
            hbm_pressure=row.get("critical_pressure", 0.5),
        )
        critical = not str(row.get("request_class") or "").lower().startswith(
            "non-critical"
        )
        row["observed_wct_ratio"] = round(wct_ratio, 3)
        row["online_tuned_alpha"] = update.alpha
        row["online_alpha_delta"] = update.delta
        row["online_effective_gpu_replica_alpha"] = update.alpha if critical else 0.0
        row["online_adapter_source"] = update.source
    return rows


def _derive_sweep_rows(run_root: Path) -> list[dict[str, object]]:
    inputs = _iter_sweep_request_details(run_root)
    if not inputs:
        return []

    groups: dict[tuple[str, str, str, float], list[dict[str, str]]] = defaultdict(list)
    run_ids: dict[tuple[str, str, str, float], set[str]] = defaultdict(set)
    for request_details, run_id, load_level, sweep_point in inputs:
        with request_details.open(newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("status") or "") != "completed":
                    continue
                if _bool(row.get("aegis_cross_rank_retry", "")):
                    continue
                critical = _bool(row.get("critical_agent", ""))
                request_class = (
                    "Critical agents" if critical else "Non-Critical fan-out"
                )
                failure_rate = _float(
                    row,
                    "aegis_failure_rate_per_window",
                    0.01,
                )
                key = (
                    _series_label(request_class, load_level),
                    request_class,
                    load_level,
                    failure_rate,
                )
                groups[key].append(row)
                run_ids[key].add(f"{run_id}:{sweep_point}")

    rows: list[dict[str, object]] = []
    for (
        series_label,
        request_class,
        load_level,
        failure_rate,
    ), samples in sorted(groups.items(), key=lambda item: (item[0][0], item[0][3])):
        critical = _bool(samples[0].get("critical_agent", ""))
        observed_alpha = mean(
            _float(row, "aegis_gpu_budget_fraction", -1.0)
            for row in samples
            if _float(row, "aegis_gpu_budget_fraction", -1.0) >= 0.0
        )
        observed_effective_alpha = observed_alpha if critical else 0.0
        pressure = mean(_float(row, "aegis_critical_pressure", 0.0) for row in samples)
        output_tokens = mean(_float(row, "generated_tokens", 0.0) for row in samples)
        latency_ms = mean(_float(row, "total_latency_s", 0.0) * 1000.0 for row in samples)
        rows.append(
            {
                "request_class": request_class,
                "series_label": series_label,
                "load_level": load_level,
                "failure_rate_per_window": failure_rate,
                "critical_pressure": round(pressure, 3),
                "policy_alpha": round(observed_alpha, 3),
                "effective_gpu_replica_alpha": round(observed_effective_alpha, 3),
                "online_observed_alpha": round(observed_effective_alpha, 3),
                "observed_request_rows": len(samples),
                "observed_mean_output_tokens": round(output_tokens, 3),
                "observed_mean_latency_ms": round(latency_ms, 3),
                "policy_source": str(samples[0].get("aegis_alpha_policy_source") or ""),
                "source": "sglsim_agentic_online_mdp_alpha_sweep",
                "run_id": ";".join(sorted(run_ids[(series_label, request_class, load_level, failure_rate)])),
                "evidence_tier": "sglsim-agentic-online-alpha-sweep",
                "note": (
                    "Each point is aggregated from an online 1-3-1-3-1 SGLSim "
                    "serving run with request-supplied workload pressure and "
                    "failure-rate state."
                ),
            }
        )
    return _attach_online_tuning(rows)


def _derive_policy_replay_rows(run_root: Path) -> list[dict[str, object]]:
    manifest = _read_manifest(run_root)
    run_id = str(manifest.get("run_id") or run_root.name)
    policy_path = _policy_path_from_manifest(manifest)
    request_details = _find_request_details(run_root)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with request_details.open(newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("status") or "") != "completed":
                continue
            if _bool(row.get("aegis_cross_rank_retry", "")):
                continue
            groups[_classify(row)].append(row)

    class_order = (
        "Non-Critical fan-out",
        "Singleton Critical",
        "Long-decode Critical",
    )
    rows: list[dict[str, object]] = []
    for request_class in class_order:
        samples = groups.get(request_class, [])
        if not samples:
            continue
        representative = samples[0]
        critical = _bool(representative.get("critical_agent", ""))
        decode = _bool(representative.get("decode_dominant", ""))
        stage_width = int(float(representative.get("stage_width") or 1))
        observed_failure_rate = _float(
            representative,
            "aegis_failure_rate_per_window",
            0.01,
        )
        observed_alpha = mean(
            _float(row, "aegis_gpu_budget_fraction", -1.0)
            for row in samples
            if _float(row, "aegis_gpu_budget_fraction", -1.0) >= 0.0
        )
        observed_effective_alpha = observed_alpha if critical else 0.0
        pressure = mean(_float(row, "aegis_critical_pressure", 0.0) for row in samples)
        output_tokens = mean(_float(row, "generated_tokens", 0.0) for row in samples)
        latency_ms = mean(_float(row, "total_latency_s", 0.0) * 1000.0 for row in samples)

        for failure_rate in FAILURE_RATES:
            decision = choose_alpha(
                critical_agent=critical,
                decode_dominant=decode,
                stage_width=stage_width,
                failure_rate_per_window=failure_rate,
                policy_path=str(policy_path),
            )
            policy_alpha = decision.alpha
            rows.append(
                {
                    "request_class": request_class,
                    "failure_rate_per_window": failure_rate,
                    "critical_pressure": round(pressure, 3),
                    "policy_alpha": policy_alpha,
                    "effective_gpu_replica_alpha": (
                        policy_alpha if critical else 0.0
                    ),
                    "online_observed_alpha": (
                        round(observed_effective_alpha, 3)
                        if abs(failure_rate - observed_failure_rate) < 1e-12
                        else ""
                    ),
                    "observed_request_rows": len(samples),
                    "observed_mean_output_tokens": round(output_tokens, 3),
                    "observed_mean_latency_ms": round(latency_ms, 3),
                    "policy_source": decision.source,
                    "source": "sglsim_agentic_online_mdp_alpha_trace",
                    "run_id": run_id,
                    "evidence_tier": "sglsim-agentic-online-alpha-trace",
                    "note": (
                        "Real-trace online request rows provide workload states; "
                        "MDP policy table is replayed over failure-rate states."
                    ),
                }
            )
    return _attach_online_tuning(rows)


def derive_rows(run_root: Path | None = None) -> list[dict[str, object]]:
    if run_root is None:
        run_root = DEFAULT_SWEEP_ROOT if DEFAULT_SWEEP_ROOT.exists() else DEFAULT_RUN_ROOT
    sweep_rows = _derive_sweep_rows(run_root)
    if sweep_rows:
        return sweep_rows
    return _derive_policy_replay_rows(run_root)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURE_DATA_DIR / "fig_online_alpha_adaptation.csv",
    )
    args = parser.parse_args()

    rows = derive_rows(args.run_root)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
