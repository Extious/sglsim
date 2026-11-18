# AegisKV 历史快照：pre-KVCache-centric

归档日期：2026-08-06

本目录保存 AegisKV 在最终 `KVCache-centric` 叙事确立之前的设计、图表、演示文稿和实验产物。文件仅作历史追溯，不代表当前论文、专利或验收材料中的设计。

## 旧设计边界

此快照主要采用以下旧模型：

- Algorithm 1 同时使用 critical path、critical score 与 Slack；
- Algorithm 2 主要在 HBM、Host/DRAM 与 Recompute 间决策；
- Algorithm 3 同时调整 `alpha` 和 `beta`；
- 部分实验使用逻辑 PP、oracle 信息、较小 workflow 数量或压力放大的 failure 设置。

当前设计已改为：Algorithm 1 只输出因果 Slack；Algorithm 2 对持续增长的 KV chunk 在 peer HBM、共享 DRAM、SSD 与 Recompute 间生成版本化恢复清单；Algorithm 3 只调整每个 DP 统一的机会式 HBM 上限 `alpha`；跨全部 PP stage 的版本匹配 ACK 共同推进连续 safe frontier；单 GPU failure 后整请求重路由至另一完整 DP。

## 目录映射

- `presentations/`：旧 Beamer、答辩和论文演示材料。
- `documents/paper_outline/`：旧技术文档和论文轮廓。
- `experiments/demo/`：旧模拟代码及约 5 GB 结果。
- `experiments/preregistration_v1/`：旧实验预注册。
- `design/algorithms_alpha_beta/`：旧 Algorithm 1/2/3 文档。
- `design/contracts/`：旧 exact optimizer 和 alpha/beta runtime contract。
- `flowchart/`、`research_iterations/`：旧系统图、早期 related-work 建模，以及仍采用
  `alpha/beta` 表述的训练 checkpoint 调研推演。
- `source_materials/`：旧专利答辩、二阶段验收 PPT 和旧生成脚本。
- `root_metadata/README.md`：归档前的项目首页。

## 实验结果使用限制

本快照中的结果不得直接作为新设计的 EuroSys 2027 结论：

- `causal_exact_rolling_smoke_v1` 的 35/35 `OPTIMAL` 仅证明有限滚动窗口内的 smoke 求解，不是未知未来上的全局最优，也不是正式 100+ workflow A/B；
- `formal_20k` 只有 12 个 workflow，旧 AegisKV 的 mean WCT 未优于 No Backup；
- 若干正向结果来自逻辑 PP 或 heavy-failure stress，不能表述为真实 DP4xPP4 物理部署证据；
- 旧恢复代价曲线中存在 AIConfigurator/解析投影，不能称为真实硬件测量。

新的论文数值图必须保留 workload、failure placement、样本量、置信区间、代码版本以及真实运行/模拟/解析三种证据等级。

## 外部 trace 实体化

旧实验包含绝对符号链接：

`presentations/presentation/experiments/sglsim_large/traces/cc-traces-weka-062126/traces.jsonl`

其原目标为 `/tmp/sglsim-cc-traces-weka-062126/traces.jsonl`。为防止 `/tmp` 清理导致历史输入丢失，本快照在同目录额外保存：

`traces.materialized.jsonl`

预期大小为 `1,847,151,435` bytes，预期 SHA-256 为：

`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`

原符号链接与 `/tmp` 源文件均保留不动。
