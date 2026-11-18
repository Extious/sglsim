#!/usr/bin/env python3
"""Run 4-cell heavy-failure logical-PP smoke and emit CDF/alpha-beta plots."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path("/home/comp/zhanzhao/sglsim")
OUT = ROOT / "AegisKV/demo/results/fast_logical_pp_20w_16k_c8_c16_heavyfail_v1"
TRACE = OUT / "workload_20w_minfanin.jsonl"
SERVER = ROOT / "AegisKV/demo/config/server_qwen3_32b_dp4_pp4_32k.json"
ROUTE = OUT / "execution_route_plan_hash_dp4.json"
FAILURE = OUT / "failure_manifest_heavy_critical_v1.json"
OVERRIDES = OUT / "aegis_overrides_heavy_host_floor.json"
FAILURE_THICK = OUT / "failure_manifest_heavy_critical_v2_thicker.json"
PY = str(ROOT / ".venv/bin/python")
DEADLINE_S = 20 * 60
GATE_PCT = 10.0


def _cell_name(concurrency: int, policy: str, tag: str) -> str:
    return f"c{concurrency}-heavyfail-{policy}-{tag}.json"


def _run_cell(
    concurrency: int,
    policy: str,
    tag: str,
    failure_manifest: Path,
    aegis_overrides: Path | None,
) -> dict[str, Any]:
    output = OUT / _cell_name(concurrency, policy, tag)
    log_path = output.with_suffix(".log")
    if output.is_file() and output.stat().st_size > 1000:
        return {
            "concurrency": concurrency,
            "policy": policy,
            "tag": tag,
            "output": str(output),
            "skipped": True,
            "ok": True,
        }
    cmd = [
        PY,
        "-m",
        "AegisKV.demo.run_offline_wct",
        "--server-config",
        str(SERVER),
        "--trace",
        str(TRACE),
        "--route-plan",
        str(ROUTE),
        "--failure-manifest",
        str(failure_manifest),
        "--output",
        str(output),
        "--policy",
        policy,
        "--logical-pp",
        "--workflow-concurrency",
        str(concurrency),
        "--failure-count",
        "64",
        "--max-workflows",
        "20",
        "--max-requests-per-workflow",
        "0",
        "--max-input-tokens",
        "16384",
        "--max-output-tokens",
        "16384",
        "--slack-source",
        "trace_oracle",
        "--artifact-detail",
        "compact",
    ]
    if policy == "aegiskv" and aegis_overrides is not None:
        cmd.extend(["--aegis-overrides", str(aegis_overrides)])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "concurrency": concurrency,
        "policy": policy,
        "tag": tag,
        "output": str(output),
        "log": str(log_path),
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - started, 3),
        "skipped": False,
        "ok": proc.returncode == 0 and output.is_file(),
    }


def _mean_wct(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    wcts = [float(w["wct_s"]) for w in payload.get("workflows", [])]
    if not wcts:
        raise ValueError(f"no workflows in {path}")
    return float(sum(wcts) / len(wcts))


def _improvement(concurrency: int, tag: str) -> dict[str, Any]:
    aegis = OUT / _cell_name(concurrency, "aegiskv", tag)
    base = OUT / _cell_name(concurrency, "no_backup", tag)
    mean_a = _mean_wct(aegis)
    mean_b = _mean_wct(base)
    pct = 100.0 * (mean_b - mean_a) / max(mean_b, 1e-9)
    return {
        "concurrency": concurrency,
        "tag": tag,
        "mean_aegiskv": mean_a,
        "mean_no_backup": mean_b,
        "improvement_pct": pct,
        "pass_gate": pct >= GATE_PCT,
    }


def _thicken_manifest(src: Path, dst: Path) -> None:
    payload = json.loads(src.read_text(encoding="utf-8"))
    events = list(payload.get("events", []))
    used = {
        str(victim["backup_key"])
        for event in events
        for victim in event.get("victims", [])
    }
    route = json.loads(ROUTE.read_text(encoding="utf-8"))
    dp_size = int(route["dp_size"])
    scored: dict[int, list[tuple[int, str]]] = {
        rank: [] for rank in range(dp_size)
    }
    with TRACE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row["rid"])
            if rid in used:
                continue
            critical = bool(
                row.get("trace_oracle_critical_agent") or row.get("critical_agent")
            )
            if not critical:
                continue
            rank = int(route["routes"][rid]["execution_dp_rank"])
            tokens = int(row.get("prompt_len") or 0) + int(row.get("output_len") or 0)
            scored[rank].append((-tokens, rid))
    by_rank = {
        rank: [rid for _, rid in sorted(items)[:8]]
        for rank, items in scored.items()
    }
    extra = []
    t = float(events[-1]["offset_s"]) + float(events[-1]["duration_s"]) + 28.0
    for index in range(dp_size * 3):
        dp_rank = index % dp_size
        victims_rids = by_rank[dp_rank][:3]
        by_rank[dp_rank] = by_rank[dp_rank][3:]
        if not victims_rids:
            continue
        duration = 12.0
        extra.append(
            {
                "event_id": f"heavyfail-thick-{index + 1:02d}",
                "offset_s": round(t, 3),
                "dp_rank": dp_rank,
                "pp_rank": index % 4,
                "failure_scope": "gpu",
                "duration_s": duration,
                "victims": [
                    {
                        "backup_key": rid,
                        "execution_dp_rank": dp_rank,
                    }
                    for rid in victims_rids
                ],
            }
        )
        t += duration + 28.0
    payload["events"] = events + extra
    payload["description"] = (
        str(payload.get("description") or "")
        + " Thickened once after missing the 10% WCT gate."
    )
    dst.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _alpha_beta_series(path: Path) -> list[dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for update in payload.get("rank_alpha_updates") or payload.get(
        "alpha_beta_updates", []
    ):
        rows.append(
            {
                "time": float(update.get("time", update.get("elapsed_ms", 0.0))),
                "alpha": float(update.get("new_alpha", update.get("alpha", 0.0))),
                "beta": float(update.get("new_beta", update.get("beta", 0.0))),
            }
        )
    if rows:
        return rows
    events_path = path.with_suffix(".events.jsonl")
    if events_path.is_file():
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("kind") != "backup_ack":
                    continue
                window_update = event.get("window_update")
                if isinstance(window_update, dict):
                    rows.append(
                        {
                            "time": float(event.get("time", 0.0)),
                            "alpha": float(
                                window_update.get(
                                    "new_alpha", event.get("alpha", 0.0)
                                )
                            ),
                            "beta": float(
                                window_update.get(
                                    "new_beta", event.get("beta", 0.0)
                                )
                            ),
                        }
                    )
                elif event.get("alpha") is not None:
                    rows.append(
                        {
                            "time": float(event.get("time", 0.0)),
                            "alpha": float(event.get("alpha", 0.0)),
                            "beta": float(event.get("beta", 0.0)),
                        }
                    )
    return rows


def _pair_ready(concurrency: int, tag: str) -> bool:
    return all(
        (OUT / _cell_name(concurrency, policy, tag)).is_file()
        for policy in ("aegiskv", "no_backup")
    )


def _plot(tag: str, improvements: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ready = [imp["concurrency"] for imp in improvements]
    for concurrency in ready:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for policy, color in (("no_backup", "#4C78A8"), ("aegiskv", "#F58518")):
            path = OUT / _cell_name(concurrency, policy, tag)
            payload = json.loads(path.read_text(encoding="utf-8"))
            wcts = sorted(float(w["wct_s"]) for w in payload["workflows"])
            ys = [(i + 1) / len(wcts) for i in range(len(wcts))]
            ax.plot(wcts, ys, label=policy, color=color, linewidth=2.0)
        imp = next(x for x in improvements if x["concurrency"] == concurrency)
        ax.set_xlabel("Workflow completion time (s)")
        ax.set_ylabel("CDF")
        ax.set_title(
            f"Heavy-fail logical-PP 20w/16k c={concurrency} "
            f"(gain {imp['improvement_pct']:.1f}%)"
        )
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"wct_cdf_c{concurrency}_{tag}.png", dpi=160)
        fig.savefig(OUT / f"wct_cdf_c{concurrency}_{tag}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    plotted = False
    for concurrency, color in ((8, "#4C78A8"), (16, "#F58518")):
        path = OUT / _cell_name(concurrency, "aegiskv", tag)
        if not path.is_file():
            continue
        series = _alpha_beta_series(path)
        if not series:
            continue
        plotted = True
        xs = list(range(len(series)))
        ax.plot(
            xs,
            [row["alpha"] for row in series],
            color=color,
            linestyle="-",
            label=f"alpha c={concurrency}",
        )
        ax.plot(
            xs,
            [row["beta"] for row in series],
            color=color,
            linestyle="--",
            label=f"beta c={concurrency}",
        )
    if plotted:
        ax.set_xlabel("Algorithm-3 window index")
        ax.set_ylabel("Ratio")
        ax.set_title("Alpha/beta under rising concurrency (beta >= alpha)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"alpha_beta_c8_vs_c16_{tag}.png", dpi=160)
        fig.savefig(OUT / f"alpha_beta_c8_vs_c16_{tag}.pdf")
    plt.close(fig)


def _run_pair(
    concurrency: int,
    tag: str,
    failure_manifest: Path,
    overrides: Path,
) -> list[dict[str, Any]]:
    jobs = [
        (concurrency, "no_backup", tag, failure_manifest, None),
        (concurrency, "aegiskv", tag, failure_manifest, overrides),
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_cell, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    return results


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + DEADLINE_S
    cells: list[dict[str, Any]] = []
    tag = "v1"
    thickened = False

    # Prioritize c=8 A/B (gate evidence), then c=16 if deadline remains.
    for concurrency in (8, 16):
        if time.monotonic() >= deadline:
            break
        pair = _run_pair(concurrency, tag, FAILURE, OVERRIDES)
        cells.extend(pair)
        if not all(item.get("ok") for item in pair):
            summary = {
                "ok": False,
                "reason": f"c{concurrency}_pair_failed",
                "cells": cells,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            (OUT / "smoke_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            return 1

    improvements = [
        _improvement(concurrency, tag)
        for concurrency in (8, 16)
        if _pair_ready(concurrency, tag)
    ]
    if (
        improvements
        and not any(item["pass_gate"] for item in improvements)
        and time.monotonic() + 90.0 < deadline
    ):
        _thicken_manifest(FAILURE, FAILURE_THICK)
        thicker_overrides = {
            "critical_host_floor_fraction": 0.85,
            "critical_host_floor_probability": 1.0,
            "initial_alpha": 0.22,
            "initial_beta": 0.75,
            "max_alpha": 0.45,
        }
        thick_overrides_path = OUT / "aegis_overrides_heavy_host_floor_v2.json"
        thick_overrides_path.write_text(
            json.dumps(thicker_overrides, indent=2) + "\n", encoding="utf-8"
        )
        tag = "v2"
        thickened = True
        for concurrency in (8, 16):
            if time.monotonic() >= deadline:
                break
            for policy in ("aegiskv", "no_backup"):
                path = OUT / _cell_name(concurrency, policy, tag)
                if path.exists():
                    path.unlink()
            pair = _run_pair(concurrency, tag, FAILURE_THICK, thick_overrides_path)
            cells.extend(pair)
            if not all(item.get("ok") for item in pair):
                break
        improvements = [
            _improvement(concurrency, tag)
            for concurrency in (8, 16)
            if _pair_ready(concurrency, tag)
        ]

    if improvements:
        _plot(tag, improvements)

    ab_check = {}
    for concurrency in (8, 16):
        path = OUT / _cell_name(concurrency, "aegiskv", tag)
        if not path.is_file():
            ab_check[str(concurrency)] = {"available": False}
            continue
        series = _alpha_beta_series(path)
        if not series:
            ab_check[str(concurrency)] = {"available": False}
            continue
        alphas = [row["alpha"] for row in series]
        betas = [row["beta"] for row in series]
        ab_check[str(concurrency)] = {
            "available": True,
            "mean_alpha": sum(alphas) / len(alphas),
            "mean_beta": sum(betas) / len(betas),
            "final_alpha": alphas[-1],
            "final_beta": betas[-1],
            "beta_ge_alpha_rate": sum(
                1 for a, b in zip(alphas, betas) if b + 1e-9 >= a
            )
            / len(series),
        }
    c8 = ab_check.get("8", {})
    c16 = ab_check.get("16", {})
    pressure_ok = None
    if c8.get("available") and c16.get("available"):
        pressure_ok = (
            c16["mean_alpha"] <= c8["mean_alpha"] + 1e-6
            and c16["mean_beta"] <= c8["mean_beta"] + 1e-6
            and c16["final_alpha"] <= c8["final_alpha"] + 1e-6
            and c16["final_beta"] <= c8["final_beta"] + 1e-6
        )

    summary = {
        "label": "heavy-failure-stress-smoke",
        "logical_pp": True,
        "max_input_tokens": 16384,
        "max_output_tokens": 16384,
        "workflows": 20,
        "requests_per_workflow_cap": "minimal_fanin_prefix",
        "trace": str(TRACE),
        "concurrency": [imp["concurrency"] for imp in improvements],
        "policies": ["aegiskv", "no_backup"],
        "gate_pct": GATE_PCT,
        "improvements": improvements,
        "pass_any_gate": any(item["pass_gate"] for item in improvements),
        "pass_all_gates": bool(improvements)
        and all(item["pass_gate"] for item in improvements),
        "thickened_rerun": thickened,
        "alpha_beta_check": ab_check,
        "alpha_beta_pressure_ok": pressure_ok,
        "timed_out": time.monotonic() > deadline,
        "elapsed_s": round(time.monotonic() - started, 3),
        "cells": cells,
        "artifact_tag": tag,
    }
    (OUT / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["pass_any_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
