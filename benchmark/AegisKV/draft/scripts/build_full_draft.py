#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "full_draft.md"

SECTIONS = [
    "paper/sections/00_paper_outline.md",
    "paper/sections/01_abstract.md",
    "paper/sections/02_introduction_motivation.md",
    "paper/sections/03_background_related_work.md",
    "paper/sections/04_system_design.md",
    "paper/sections/05_implementation_plan.md",
    "paper/sections/06_evaluation_plan.md",
    "paper/sections/07_preliminary_results.md",
    "paper/sections/08_figures_and_expected_results.md",
    "paper/sections/09_conclusion_next_steps.md",
]


def evidence_summary() -> str:
    path = ROOT / "figures" / "evidence_status.json"
    if not path.exists():
        return "figure evidence status unavailable"
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    return (
        f"figures={summary['figures']}, "
        f"claim_ready={summary['claim_ready']}, "
        f"preliminary_valid={summary.get('preliminary_valid', 0)}, "
        f"final_measured={summary.get('measured_valid', 0)}"
    )


def main() -> int:
    lines = [
        "# AegisKV：面向多智能体 LLM Serving 故障恢复的 Workflow-Aware KV 备份",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "证据状态：",
        "",
        f"```text\n{evidence_summary()}\n```",
        "",
        "重要说明：当前部分定量图仍使用 `evidence_tier=sglsim-preliminary`。",
        "在 `--strict-final` 通过前，本文件只是内部 HPCA 风格论文草稿。",
        "",
        "主要图表资产：",
        "",
        "- 图表预览：[../figures/gallery.md](../figures/gallery.md)",
        "- 图表索引：[../figures/manifest.json](../figures/manifest.json)",
        "- 证据状态：[../figures/evidence_status.json](../figures/evidence_status.json)",
        "",
    ]
    for rel in SECTIONS:
        path = ROOT / rel
        lines.extend(["", "---", ""])
        lines.append(path.read_text(encoding="utf-8").strip())
        lines.append("")
    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
