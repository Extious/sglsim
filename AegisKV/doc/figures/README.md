# 当前图表

> **当前阻塞（2026-08-27）：** `aegiskv-paper-system-design-v2.drawio`
> 及其导出仍含旧 `destage` / `Destage executor` / DRAM backing 语义，和
> `../design/current_design.md` 的 source-owned independent-tier contract
> 冲突。`system-overview-review.md` 也记录了同一旧路径。在源图、导出物
> 和 review 全部重做之前，不得把该图称为“已与当前设计同步”或用于投稿。

本目录保存当前 agent-aware 推理容错设计图、trace 统计图、受控 Motivation 实验图和初步模拟图。

- `aegiskv-master-vllm-mooncake.drawio`：三页完整叙事、四组件实现/数学模型、机制对比图；已统一为 Causal Slack、无权重 robust deadline risk、group commit/`FreezeAndPin`、固定 target EDF 与 AIMD。文件名保留历史集成来源，但图内不依赖特定 Store 品牌。
- `aegiskv-paper-system-design-v2.drawio`：原计划作为当前论文 System Overview 的唯一可编辑源图，但现有版本仍有上述旧路径，当前状态为 **stale / blocked for submission**。修复后应保持 Application / Serving Control / KV Data 三平面、Algorithm 1/2/3 责任边界，以及 Source→Peer-HBM 与 Source→Store-DRAM 两条独立保护路径；不得出现 HBM→DRAM destage。对应导出物也必须全量重新生成并重新审计。
- `aegiskv-system-overview.drawio` 与 `aegiskv-paper-system-design.drawio`：旧版架构图，仅保留用于设计演化审计；当前论文不再引用。
- `aegiskv-paper-protection-comparison.drawio`：旧版方案对比时间轴；已从 Motivation 正文移除，保留仅供历史核对。
- `aegiskv-paper-commit-recovery.drawio`：旧版提交/恢复图，保留供协议演化审计；其中旧术语不再作为 v2 正文设计或论文插图。
- `real-cc-parallel-subagent-duration.{png,pdf}`：Real CC trace 中每个 workflow 的最大严格并行 fan-out 秒差 CDF；严格并行要求所有 branch interval 有共同重叠。由 `scripts/render_real_cc_parallel_subagents.py` 生成。
- `real-cc-failure-position-counterfactual.{png,pdf}`：对 workflow `8ab0a089...` 的五个 subagent，在五个 nested LLM call 共同活跃的反事实时刻 $t_f=349.5$ s 读取真实 call input prefix，并用 Qwen3-235B-A22B TP8×PP4×EP8 的 AIConfigurator full-prefix recomputation 减去真实 fan-in slack；五个 agent 全部标出。它是 trace+配置预测反事实，不是 Claude、物理 PP 或端到端 recovery 实测。
- `real-cc-workflow-8ab0a089.drawio` 及导出物：完整显示上述 workflow 唯一的五路 subagent fan-out；支路为 110.3、116.7、78.9、117.9、107.1 s，slowest--fastest gap 为 39.0 s。`real-cc-workflow-57f2a77e.*` 是已退役的分钟级旧样例。
- `motivation-recovery-scaling.{png,pdf}`：单栏、线性秒轴叠加三个模型/并行配置的 AIConfigurator full-prefix recomputation。脚本调用公开 `predict_infer_time(ScheduleBatch)`，按 8K chunk 累加；128K 端点为 9.16、32.82、56.58 s。数值是 SILICON-mode 配置预测而非真机或完整 failure-to-ready 时间；235B 的 94 层不能被 PP4 整除，且当前 TP>1、PP>1 组合不宣称已通过物理 PP runtime 跑通。
- `motivation-kv-publication-comparison.drawio` 及导出物：单栏比较 vLLM+MooncakeStoreConnector shared-Store path 与 SGLang HiCache standard write-through path 的 publication trigger、failure-time available prefix、retry suffix 和 next-turn reuse；不包含 AegisKV。`motivation-recovery-frontier-gap.*` 是已退役的过简版本。
- `motivation-failure-position.png` 与 `motivation-workflow-dag.png`：旧版合成 Motivation 素材；主论文不再引用，不能作为 Real CC 证据。
- `motivation-gpu-availability.png`：初步模拟使用的 GPU availability 输入。
- `preliminary-wct-kv-reuse.png`：WCT CDF 与 retry KV reuse 初步模拟结果；不代表完整协议和优化器的正式结论。

论文 Draw.io 图与专利五图均不含图内总标题，避免和 LaTeX/Word 题注重复。所有 `.drawio`
文件均为原生可编辑 XML。`*.drawio.png` 和对应 PDF 由本机 Draw.io 原生导出并完成
目视检查；论文 PDF/PNG 版本位于 `../paper/figures/`。源图更新后必须全量重新导出。

Real CC CDF 与 branch timing 使用 `traces.materialized.jsonl`（SHA256
`29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276`）。
Figure 1(c) 另外从同目录 `traces.jsonl` 的 nested request 读取真实 `in` 字段。
生成 CSV 在 `data/` 中区分 branch、fan-out、workflow 聚合和论文所用样例
`real-cc-example-failure-position.csv`。旧的
`real-cc-failure-position-{branches,workflow}.csv` 属于 restart-from-start 版本，
不再作为论文图输入；不得把任一 counterfactual 描述为 simulator 或 GPU 实测。

图中对 vLLM、Mooncake 和 SGLang 的比较仅描述指定版本及启用模式下的代表性机制，不是对框架全部配置的普遍结论。不得将 best-effort cache availability 等同于显式 recovery commitment；涉及提交时机、回滚长度或性能的结论，均须同时给出配置、源码版本和实测证据。
