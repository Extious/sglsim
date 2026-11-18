# AegisKV Beamer 图像逐图审查

本文件审查 `benchmark/AegisKV/beamer/presentation.pdf` 当前引用的 11 张图。
审查口径是：先判断图像是否准确表达当前代码与实验数据，再判断图形类型是否足够适合论文展示；如果图像不合理但数据口径合理，则重绘；如果图像暴露的是实验覆盖不足，则标记为实验缺口而不是仅做美化。

## 总体结论

- 已重绘：`System Overview`、`Motivation Criticality CI`、`Real Trace Characterization`、`Online Alpha`、`Ablation`。
- 已满足当前 beamer 叙述：`Motivation`、`Failure Replay Repeats`、`Backup Policy`、`Alpha Tradeoff`、`SGLang/SGLSim Accuracy`、`Real-System Recovery Calibration`、`System-Paper Readiness Matrix`。
- 仍是实验缺口：`Ablation` 的 WCT 差异目前接近置信区间，只能说明 recovery-path 语义已经进入 serving path，不能作为强 ablation 结论；`Alpha Tradeoff` 的 failure-pressure 维度当前只有 1 和 3 次 failure 注入，最终论文需要更密的 failure-rate sweep。

## 逐图审查表

| Beamer 图 | 当前用途 | 审查结论 | 本轮动作 | 剩余风险 |
|---|---|---|---|---|
| System Overview | 说明 AegisKV 如何从 workflow metadata 进入控制面，再驱动 serving data plane 的 GPUReplica / host restore / recompute 路径。 | 原图模块割裂且有文字重叠，属于图形布局问题。 | 重绘为连续分层系统图：左侧 workflow scope，中上方 AegisKV control plane，底部 serving data plane；恢复路径从同一个 failed DP 展开。 | 需要在最终 PDF 页中继续确认缩放后无文字遮挡。 |
| Motivation: Agent Failure Impact | 证明不同 agent failure position 对 workflow WCT 的影响不同。 | 数据来自 DP8 agentic online measured run，曲线/位置对比合理。 | 保留。 | 这是动机图，不直接证明 AegisKV 策略最优。 |
| Motivation Criticality CI | 展示 Critical fan-out agent 相比 Non-Critical fan-out 的 decode pressure 和 latency exposure。 | 原柱状图只展示均值，信息密度不足；数据本身支持 workflow-level CI。 | 改为 workflow-level 样本点 + 均值连线 + 95% CI 区间图。 | 样本来自当前 motivation profile，最终可增加更多 workflow family。 |
| Failure Replay Repeats | 展示跨 seed failure replay 的 paired WCT delta 与 real-trace failure-position replay。 | 现有配对曲线和 ECDF 能表达完整分布，合理。 | 保留。 | final 论文仍应把 trace-derived failure replay 扩大到更多 failure rate 与 failure timing。 |
| Backup Policy | 比较 HostBackup、GPUReplica、AegisKV、NoBackup 的 WCT 与 recovery overhead 分布。 | ECDF 形式合理，适合展示 tail behavior。 | 保留。 | 当前图是聚合分布，不能被解释为每个 load regime 的单独结论。 |
| Alpha Tradeoff | 展示 concurrency pressure 使最优 alpha 左移，failure pressure 使 alpha 右移。 | 左图下凸曲线符合 HBM tradeoff；右图已经加入 failure-pressure 维度。 | 保留当前双面板曲线图。 | failure-pressure 只有 1/3 次注入两个点，最终论文需要更密 failure-rate sweep。 |
| Real Trace Characterization | 说明 Claude Code trace workload 的来源覆盖、token 分布、latency 分布和 AegisKV token exposure。 | 原图以柱状汇总为主，不适合展示分布。 | 改为 source coverage curve、input/output token ECDF、decode-heavy latency ECDF、AegisKV token exposure ECDF。 | 当前 Claude trace serving rows 全部是 Critical；不能声称它覆盖 Non-Critical trace workload。 |
| Online Alpha | 证明 MDP policy table 进入 online serving path，并且 GPUReplica alpha 只对 Critical request 生效。 | 原图右侧堆叠柱只表达计数，左侧柱状图不必要。 | 改为 alpha 状态曲线与 completed request row 曲线。 | 当前只有 4 个 online state，适合 integration evidence，不是完整 policy robustness sweep。 |
| Ablation | 展示组件关闭后 WCT 和 recovery action 的变化。 | 这张图暴露的是实验覆盖问题：WCT delta 仍接近 CI，不能作为强性能结论。 | 左图改为 normalized WCT 点-置信区间，右图改为 recovery action 计数曲线，并在 beamer 文本中定位为早期在线证据。 | 需要多负载、多 seed、real-trace failure replay ablation 才能作为最终论文强结论。 |
| Real-System Recovery Calibration | 说明 real SGLang / SGLSim recovery timing calibration 与当前 metadata-materialization 边界。 | 曲线 + calibration 摘要合理，且明确区分 real-system anchor 与 SGLSim policy rows。 | 保留。 | 仍需最终 failure-policy accuracy，而不是只停留在 base serving / recovery copy calibration。 |
| System-Paper Readiness Matrix | 汇总问题建模、MDP、real trace、failure replay、strict-final evidence 等 claim readiness。 | 作为审计矩阵合理，不应被当作主实验结果图。 | 保留在 beamer 末尾。 | 随后新增实验后需重新运行 readiness audit 刷新。 |

## 本轮重绘记录

- `fig_system_overview_generated_v2_tight.png`：重绘为连续系统图，修复 workflow 内部重叠并补回 `1-3-1-3-1` 结构。
- `fig_motivation_criticality_ci.png`：从柱状图改为 workflow-level 样本点与 95% CI。
- `fig_real_trace_characterization.png`：从柱状汇总改为四个曲线/ECDF 面板。
- `fig_online_alpha_adaptation.png`：从柱状/堆叠柱改为 alpha 状态曲线与请求计数曲线。
- `fig_ablation.png`：从柱状图改为 WCT 点-置信区间与 recovery action 路径计数曲线。

## 后续实验优先级

1. 对 `Alpha Tradeoff` 增加更密的 failure-rate sweep，而不只比较 1/3 次 failure 注入。
2. 对 `Ablation` 增加多负载、多 seed、real-trace failure replay 版本，避免 WCT delta 接近 CI。
3. 对 real trace workload 增加含 Non-Critical fan-out 的 trace-derived workflow，避免当前 Claude trace 全 Critical 的覆盖限制。
