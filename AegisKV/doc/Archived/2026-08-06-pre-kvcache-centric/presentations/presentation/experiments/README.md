# AegisKV Experimental Evidence

本目录保存从旧 `draft/` 迁入、且没有包含在 `paper_outline.pdf` 中的实验资产。
它是证据仓，不是另一份论文正文；问题建模和 claim boundary 只以
`../sections/04_formulation.tex` 与 `../sections/07_evaluation.tex` 为准。

## 内容

- `sglang_small/`：真实 SGLang smoke、Host/P2P copy microbenchmark、
  failure-visibility pair、Slurm 计划与物理恢复证据。
- `sglsim_large/`：在线 SGLSim sweep、agentic workflow/trace 配置、
  failure replay、policy/ablation/alpha 实验以及原始运行结果。
- `schema/`：历史 CSV 字段约定。
- `correctness_evidence.md`：恢复路径和数据面的正确性证据。
- `figure_review_audit.md`：图表证据复核记录。
- `final_evidence_matrix.csv`：当前图表与后续实测需求的映射。
- `strict_final_dataplane_readiness.*`：物理数据面尚未闭环的审计快照。
- `strict_final_experiment_gap.md`：最终系统实验仍需补齐的项目。

旧目录中由 MDP 叙事驱动的 goal/readiness 审计已经删除。结果目录里仍可能出现
`mdp`、`draft` 或旧绝对路径；这些名称只记录实验产生时的配置与 provenance，
不表示当前论文仍采用 MDP 建模，也不应被当作可执行入口。

## 数据流

实验输出供 `../figures/data/measured/` 和 `../figures/scripts/` 使用。新的结果
应写入：

```text
benchmark/AegisKV/presentation/figures/data/measured/<figure_stem>.csv
```

常用检查：

```bash
python benchmark/AegisKV/presentation/figures/scripts/validate_figure_evidence.py
python benchmark/AegisKV/presentation/scripts/report_strict_final_dataplane_readiness.py
```

真实 SGLang recovery matrix 的计划、同步和审计工具仍位于：

```text
sglang_small/scripts/
sglang_small/plans/strict_final_recovery_matrix/
```

大规模在线实验工具位于：

```text
sglsim_large/scripts/
sglsim_large/config_*/
sglsim_large/results/
```

## 证据边界

- SGLSim 大规模结果用于策略趋势、规模和 ablation。
- SGLang 小规模结果用于物理 copy/recovery calibration。
- `metadata_materialization` 不等于 live tensor copy。
- 历史 audit 的 `ready` 标签不覆盖 `paper_outline.pdf` 中明确写出的实验缺口。
- 任何新 claim 必须先进入当前 PDF 的 evaluation section，再由 measured CSV 和
  可复现脚本支持，不能从历史结果目录反向提升。
