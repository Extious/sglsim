#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
EXPERIMENTS = ROOT / "experiments"
OUT_MD = EXPERIMENTS / "final_gate_blockers.md"
OUT_JSON = EXPERIMENTS / "final_gate_blockers.json"
HYBRID_STATUS = FIGURES / "hpca_hybrid_evidence_status.json"


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {str(row.get(key, "")).strip(): row for row in rows if row.get(key)}


def classify_blocker(*texts: str) -> str:
    text = " ".join(t.lower() for t in texts if t)
    if "physical-kv-copy" in text or "physical kv" in text or "cuda" in text:
        return "physical KV movement / CUDA copy validation"
    if "paired-policy" in text or "paired policy" in text:
        return "paired SGLang/SGLSim policy measurements"
    if "controlled-timing" in text or "timing sweep" in text:
        return "controlled failure-timing sweep"
    if "shared-prefix" in text or "shared prefix" in text:
        return "broader shared-prefix sweep"
    if "final" in text:
        return "final measured CSV replacement"
    return "unclassified final-evidence gap"


def main() -> int:
    status_path = FIGURES / "evidence_status.json"
    if not status_path.exists():
        raise SystemExit(
            f"missing {status_path}; run validate_figure_evidence.py first"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    hybrid_status = (
        json.loads(HYBRID_STATUS.read_text(encoding="utf-8"))
        if HYBRID_STATUS.exists()
        else {}
    )
    hybrid_by_stem = {
        str(row.get("stem")): row for row in hybrid_status.get("figures", [])
    }
    matrix = read_csv_by_key(EXPERIMENTS / "final_evidence_matrix.csv", "figure_stem")
    plan = read_csv_by_key(EXPERIMENTS / "sglsim_large" / "matrix_plan.csv", "measured_csv")

    result_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []
    for fig in status.get("figures", []):
        if fig.get("kind") == "mechanism_diagram":
            continue
        stem = str(fig["stem"])
        measured_csv = f"{stem}.csv"
        matrix_row = matrix.get(stem, {})
        plan_row = plan.get(measured_csv, {})
        final_gate = str(matrix_row.get("final_gate", "")).strip()
        plan_status = str(plan_row.get("status", "")).strip()
        claim_ready = bool(fig.get("claim_ready"))
        hybrid_row = hybrid_by_stem.get(stem, {})
        blocker = "resolved" if claim_ready else classify_blocker(final_gate, plan_status)
        row = {
            "figure": fig.get("figure"),
            "stem": stem,
            "measured_csv": measured_csv,
            "current_tier": fig.get("evidence_tier"),
            "row_count": fig.get("row_count"),
            "primary_final_source": matrix_row.get("primary_final_source", ""),
            "final_gate": final_gate,
            "plan_status": plan_status,
            "blocker_class": blocker,
            "claim_ready": claim_ready,
            "hybrid_status": hybrid_row.get("hybrid_status", ""),
            "hybrid_ready": bool(hybrid_row.get("hybrid_ready", False)),
        }
        result_rows.append(row)
        if claim_ready:
            resolved_rows.append(row)
        else:
            blocked_rows.append(row)

    summary = dict(status.get("summary", {}))
    blocker_counts: dict[str, int] = {}
    for row in blocked_rows:
        blocker_counts[row["blocker_class"]] = blocker_counts.get(row["blocker_class"], 0) + 1
    hybrid_blocker_counts: dict[str, int] = {}
    for row in result_rows:
        if row["hybrid_ready"]:
            continue
        status_name = row["hybrid_status"] or "not-classified"
        hybrid_blocker_counts[status_name] = hybrid_blocker_counts.get(status_name, 0) + 1

    payload = {
        "summary": summary,
        "blocker_counts": blocker_counts,
        "hybrid_blocker_counts": hybrid_blocker_counts,
        "hybrid_summary": hybrid_status.get("summary", {}),
        "blocked_result_figures": blocked_rows,
        "resolved_result_figures": resolved_rows,
        "result_figures": result_rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = [
        "# AegisKV Final Gate Blockers",
        "",
        "This report is generated from `figures/evidence_status.json`, "
        "`experiments/final_evidence_matrix.csv`, and "
        "`experiments/sglsim_large/matrix_plan.csv`.",
        "",
        "It tracks why `validate_figure_evidence.py --strict-final` is still "
        "blocked. Mechanism figures are omitted because they do not require "
        "measured CSVs.",
        "",
        "## Summary",
        "",
        f"- Figures: {summary.get('figures', 0)}",
        f"- Claim-ready figures: {summary.get('claim_ready', 0)}",
        f"- Quantitative result figures: {summary.get('result_figures', 0)}",
        f"- Final measured result figures: {summary.get('measured_valid', 0)}",
        f"- Preliminary valid result figures: {summary.get('preliminary_valid', 0)}",
        f"- Blocked result figures: {len(blocked_rows)}",
        f"- Resolved result figures: {len(resolved_rows)}",
    ]
    if hybrid_status:
        hybrid_summary = hybrid_status.get("summary", {})
        lines.extend(
            [
                f"- HPCA hybrid-ready figures: {hybrid_summary.get('hybrid_ready', 0)}",
                "- HPCA hybrid-ready result figures: "
                f"{hybrid_summary.get('hybrid_ready_result_figures', 0)}",
                "- HPCA hybrid-blocked result figures: "
                f"{hybrid_summary.get('blocked_result_figures', 0)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Evidence Boundaries",
            "",
            "- `--strict-final` accepts only `evidence_tier=final` result CSVs.",
            "- The HPCA hybrid report accepts selected calibrated SGLSim figures "
            "only when small physical movement measurements are present and the "
            "metadata-only data plane is disclosed.",
            "- SGLSim remains a pure metadata online simulator; live CUDA page "
            "movement must stay in SGLang/torch calibration harnesses or a "
            "separate real-system path.",
            "",
            "## Blocker Classes",
            "",
        ]
    )
    for blocker, count in sorted(blocker_counts.items()):
        lines.append(f"- {blocker}: {count}")
    if hybrid_status:
        lines.extend(["", "## HPCA Hybrid Blocker Classes", ""])
        for blocker, count in sorted(hybrid_blocker_counts.items()):
            lines.append(f"- {blocker}: {count}")

    lines.extend(
        [
            "",
            "## Per-Figure Gate",
            "",
            "| Figure | Stem | Current tier | Rows | Hybrid | Primary final source | Status | Final gate |",
            "|---|---|---|---:|---|---|---|---|",
        ]
    )
    for row in result_rows:
        lines.append(
            "| {figure} | `{stem}` | `{current_tier}` | {row_count} | {hybrid} | {source} | {blocker} | {gate} |".format(
                figure=row["figure"],
                stem=row["stem"],
                current_tier=row["current_tier"],
                row_count=row["row_count"],
                hybrid=row["hybrid_status"] or "-",
                source=row["primary_final_source"] or "-",
                blocker=("resolved" if row["claim_ready"] else row["blocker_class"]),
                gate=(row["final_gate"] or row["plan_status"] or "-").replace("|", "/"),
            )
        )

    online_sweep_figures = [
        str(row["figure"])
        for row in result_rows
        if row["hybrid_status"] == "needs-online-policy-sweep"
    ]
    paired_figures = [
        str(row["figure"])
        for row in result_rows
        if row["hybrid_status"] == "needs-paired-measurements"
    ]
    if online_sweep_figures or paired_figures:
        online_text = (
            ", ".join(online_sweep_figures) if online_sweep_figures else "none"
        )
        paired_text = ", ".join(paired_figures) if paired_figures else "none"
        milestone = (
            "The hybrid report upgrades figures whose SGLSim rows are backed by "
            "implemented online policy sweeps or by calibrated physical movement "
            "evidence, while preserving the boundary that SGLSim uses metadata "
            "materialization plus calibrated timing. The fastest next "
            "paper-strengthening step is to replace remaining replay "
            "counterfactuals with implemented online policy sweeps for "
            f"{online_text}, and to collect paired real-system/SGLSim "
            f"measurements for {paired_text}."
        )
    else:
        milestone = (
            "The HPCA hybrid report has no remaining blocked figures. The next "
            "paper-strengthening step is strict-final evidence: replace "
            "preliminary SGLSim metadata-materialization rows with final "
            "end-to-end measurements that use physical KV movement or explicitly "
            "validated real-system recovery paths."
        )
    lines.extend(
        [
            "",
            "## Next Practical Milestone",
            "",
            milestone,
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
