#!/usr/bin/env python3
"""Validate the complete three-algorithm AegisKV system presentation."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SECTIONS = [
    "01_introduction.tex",
    "02_background_related.tex",
    "03_motivation.tex",
    "04_formulation.tex",
    "05_architecture.tex",
    "06_implementation.tex",
    "07_evaluation.tex",
    "08_conclusion.tex",
    "09_appendix.tex",
]

LOG_ERRORS = re.compile(
    r"LaTeX Error|Undefined control sequence|Citation .* undefined|"
    r"Reference .* undefined|Overfull \\[hv]box"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_structure() -> None:
    source = (ROOT / "paper_outline.tex").read_text(encoding="utf-8")
    for section in SECTIONS:
        path = ROOT / "sections" / section
        require(path.is_file(), f"missing section: {path}")
        require(
            f"sections/{section}" in source,
            f"paper_outline.tex does not include {section}",
        )

    joined = "\n".join(
        (ROOT / "sections" / section).read_text(encoding="utf-8")
        for section in SECTIONS
    )
    for marker in [
        "算法 1：固定 DAG 工作流松弛时间估计",
        "Task-decomposition",
        "Role-specialized",
        "execution episode",
        "p_e=1",
        "固定 DAG 松弛时间",
        "唯一预测目标：智能体执行片段的剩余时间",
        "GBDT 输入特征与相关性",
        "GBDT 每轮更新后的预测准确度",
        "CPM (Critical Path Method",
        "Earliest Start",
        "Earliest Finish",
        "Latest Start",
        "Latest Finish",
        "Total Slack",
        "松弛感知的 KV 块备份决策",
        "反馈驱动的 L1/L2 备份容量控制",
        "Failure Model",
        "Recovery Chunk State Machine",
        "Failure-Time Recovery Protocol",
        "predicted\\_slack\\_ms",
        "L1 Backup",
        "L2 Backup",
        "\\beta_n",
        "Recompute",
        "fig_alg1_gbdt_feature_correlation.png",
        "fig_alg1_gbdt_accuracy_progress.png",
        "fig_alg2_slack_coverage.png",
        "fig_alg3_per_node_alpha.png",
    ]:
        require(marker in joined, f"missing complete-design marker: {marker}")

    forbidden = [
        "Step 1",
        "Step 2",
        "旧设计",
        "不能被删除",
        "为什么不需要",
        "MDP/PPO",
        "SSD/L3",
        "strict-final",
        "preliminary",
        "hybrid",
        "GPUReplica",
        "HostBackup",
        "H_n^B",
        "Revealed-DAG Workflow Slack Estimation",
        "materialize node",
        "固定拓扑 Slack 与概率拓扑扩展",
        "dag\\_epoch",
        "Alg. 1",
        "Algorithm 1",
        "Fixed-DAG Workflow Slack Estimation",
        "Fixed-DAG Slack",
        "Remaining-Time Accuracy and Calibration",
        "Forward Pass",
        "Backward Pass",
        "Workflow start",
        "Edge semantics",
        "Progress state",
        "Residual graph",
        "Out of scope",
        "fig_alg1_duration_predictor.png",
        "Conformal Quantile GBDT",
    ]
    for phrase in forbidden:
        require(phrase not in joined, f"obsolete design/review language remains: {phrase}")

    equation_count = joined.count("\\begin{equation}")
    equation_end_count = joined.count("\\end{equation}")
    equation_label_count = len(re.findall(r"\\label\{eq:[^}]+\}", joined))
    require(equation_count == equation_end_count, "unbalanced equation environments")
    require(
        equation_count == equation_label_count,
        "every displayed equation must have an eq: label",
    )

    for figure_name in [
        "fig_alg1_gbdt_feature_correlation.png",
        "fig_alg1_gbdt_accuracy_progress.png",
        "fig_alg2_slack_coverage.png",
        "fig_alg3_per_node_alpha.png",
    ]:
        figure = ROOT / "Figures" / figure_name
        require(
            figure.is_file() and figure.stat().st_size > 0,
            f"missing evidence figure: {figure}",
        )


def pdf_pages(path: Path) -> int:
    try:
        import fitz  # type: ignore

        with fitz.open(path) as document:
            return len(document)
    except ImportError:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(path)).pages)


def check_pdfs() -> None:
    pdfs = sorted(ROOT.rglob("*.pdf"))
    expected = ROOT / "paper_outline.pdf"
    require(pdfs == [expected], f"expected exactly one PDF, found: {pdfs}")
    require(expected.stat().st_size > 0, f"empty PDF: {expected}")
    pages = pdf_pages(expected)
    require(20 <= pages <= 40, f"unexpected PDF length: {pages} pages")


def compile_pdf() -> None:
    subprocess.run(
        [
            "latexmk",
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "paper_outline.tex",
        ],
        cwd=ROOT,
        check=True,
    )
    log = (ROOT / "paper_outline.log").read_text(encoding="utf-8")
    matches = LOG_ERRORS.findall(log)
    require(not matches, f"layout/reference warnings: {matches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="compile and scan LaTeX logs")
    args = parser.parse_args()

    try:
        check_structure()
        if args.compile:
            compile_pdf()
        check_pdfs()
    except (AssertionError, ET.ParseError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print("PASS: 9 sections and all three evidence-backed algorithms")
    print("PASS: paper_outline.pdf is the only PDF in the directory")
    if args.compile:
        print("PASS: LaTeX build contains no layout/reference errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
