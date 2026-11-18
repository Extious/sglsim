# AegisKV Supporting Figure Assets

本目录保存从旧 `draft/` 迁入的 measured CSV、补充图、绘图脚本和表格。
PDF 当前直接使用的图片统一放在相邻的 `../Figures/`；相同 PNG 不在这里重复
保存。

## 目录

- `data/measured/`：实验导出的 measured CSV。
- `png/`：没有进入当前 PDF、但仍有独立分析价值的补充图。
- `scripts/`：从实验结果生成 CSV/PNG 的工具。
- `tables/`：实现 hook、baseline 和 correctness test 等补充表格。
- `manifest.json`：支持图与数据来源的映射。
- `evidence_status.json`：支持证据的机器可读状态。
- `gallery.md`：当前支持图与证据边界的预览。

## 与当前 PDF 的关系

当前 PDF 已使用且不在本目录重复存储的图片包括：

- Motivation failure impact、criticality CI 和 replay repeats；
- Backup policy comparison；
- L1 alpha concurrency/failure tradeoff；
- Ablation；
- Real-trace characterization；
- Real-system recovery calibration。

这些条目的 manifest 路径直接指向 `../Figures/`。`png/` 中保留的
online-alpha、real-trace reuse/context、WCT CDF 等图只作为补充或历史证据，
不会自动进入论文，也不会改变 `sections/04_formulation.tex` 中的当前建模。

当前双层设计中，`alpha_n` 控制 L1 HBM 的 Backup 占比，`beta_n` 控制
L2 Host DRAM 的 Backup 占比。`fig_alg3_per_node_alpha` 仅根据逐节点正常
L1 服务负载计算安全备份余量，不使用 KV 块优先级或备份放置需求；
该图只支持算法 3 的 L1 正常负载安全上限分支。完整控制器还会读取
算法 2 在上一窗口发布的容量截断前影子需求，以确定下一窗口的需求目标。
在采集 matched Host-DRAM actuation sweep 前，不生成或声称 L2 beta 结果。

## DP=64 故障预测图

`../Figures/fig_dp64_failure_projection.{png,pdf}` 是分析性 Monte-Carlo
预测，不是 SGLSim 或真实机器测量。它将用户提供的 15 小时 GPU 可用性图
数字化，并校准到图中标注的平均 58.07 张可用 GPU；随后在全部 24 个
Real CC workflow 上随机采样 128 个启动相位。预测使用固定
`alpha=0.25`、`beta=0.50`，并假设每个 DP rank 对应一张 A100 80GB、
160GB Host memory 和 8 个并发请求。AegisKV 路径直接调用运行时
`AegisRankBackupManager.place()`：按 256-token 块计算净收益密度，在每个
不超过 8 个请求的活跃集合内执行 Algorithm 2 admit/replace；Algorithm 3
关闭以保持固定 `alpha/beta`。完整逐样本结果、比例、假设和数字化曲线位于
`data/analytic/fig_dp64_failure_projection_*`，重建入口为
`scripts/render_dp64_failure_projection.py`。

该图只把 trace 中标记为 critical agent 的 retry 恢复时间直接加入 workflow
WCT；为隔离 Algorithm 2，假设 Retry 使用已 ACK 的 Backup Rank，不模拟
Router 的负载-复用二次权衡、正常路径备份干扰、跨 rank 排队重分布、模型
重载、相关故障域或精确 DAG slack 传播。因此该图用于展示理论趋势和实验
假设，不能替代 DP=64 的 SGLSim 实测。

`../Figures/fig_dp64_algorithm2_failure_conditioned_cdf.{png,pdf}` 使用同一批
3,072 个配对样本并列展示两个口径：全体 workflow WCT，以及预先定义为
`critical_retry_request_count > 0` 的故障可见恢复开销。第二个口径不会替代
整体 CDF；它专门避免未发生故障的请求和 workflow 基础时长掩盖 Algorithm 2
的恢复路径收益。

## 重新生成与检查

各图可由 `scripts/` 中对应工具独立重建。基础证据检查：

```bash
python benchmark/AegisKV/presentation/figures/scripts/validate_figure_evidence.py
```

如脚本同时输出论文图和支持图，论文图写入 `../Figures/`，支持图写入
`png/`。实验数据源位于 `../experiments/`。
