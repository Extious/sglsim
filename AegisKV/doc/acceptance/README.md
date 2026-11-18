# AegisKV 项目验收设计文档

本目录保存当前 agent-aware 大模型推理容错设计的中文验收材料。

## 文件

- `main.tex`：设计文档 LaTeX 源文件。
- `references.bib`：公开参考文献。
- `main.pdf`：编译后的交付 PDF。

当前版本采用“问题动机—三算法—系统实现—初步结果”主线，并以二阶段验收答辩方案
为设计基线：Algorithm 1 计算 workflow Slack，Algorithm 2 进行 chunk 级 GPU/Host/
Recompute 决策，Algorithm 3 按窗口反馈调整每个 DP 的 `alpha/beta`。

以下图表是强制输入：

- `../figures/real-cc-parallel-subagent-duration.pdf`
- `../figures/aegiskv-ppt-system-architecture.png`（答辩 PPT 内嵌架构原图）

架构图只保留模块、状态和关键动作；解释放在正文与题注。Real CC trace 统计、受控
Motivation 实验、GPU availability 输入和 WCT CDF/KV reuse 初步结果均按各自证据类型
标注，不视为完整系统的正式验收结果。

## 编译

```bash
cd AegisKV/acceptance
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

清理中间产物但保留 PDF：

```bash
latexmk -c main.tex
```

## 证据边界

设计定义、单元/静态证据、模拟/集成证据、真实硬件证据和正式统计证据必须分开。
同一 DP 的 PP stages 共享一组 `alpha/beta`，分别约束 HBM 与 Host DRAM 备份预算。

当前设计要求正式批次提供 `acceptance_spec.yaml`、`evidence_index.yaml`、
`run_manifest.json`、`events.jsonl`、`solver_audit.jsonl`、`paired_metrics.csv`
和 `FINAL_VALIDATION.json`。
预注册文件缺失或晚于正式 run 时，不能认定为正式统计证据。
