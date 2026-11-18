# AegisKV DP=8 三终端演示

终端 1 直接启动 SGLSim 自身的 online HTTP server，不使用 AegisKV 专用 server。
该 server 启动真实的 SGLSim 多进程拓扑：一个 Router、8 个 DP Scheduler 和一个
Detokenizer，并提供 `/v1/completions` 在线推理接口。模型规格采用
`Qwen/Qwen3-8B`，并按官方 YaRN 配置将上下文从 32,768 扩展到 131,072 token
（`factor=4.0`）。所有请求都满足
`prompt_tokens + max_new_tokens < 131072`。GPU forward 使用 AIConfigurator 对
A100 SXM 的 Qwen3-8B prefill/decode 时延进行预测，不需要下载模型权重，也不执行
CUDA/NCCL；SGLSim 不执行真实 RoPE 计算，但会按 131,072 的模型上下文边界完成请求
校验、KV 分配和时延预测。

算法 2 的线性重计算基准使用 `0.22 ms/token`。该系数由同一配置下
AIConfigurator 对 Qwen3-8B/A100 SXM 在 256--131,071 token 范围的单请求 prefill
预测做过原点的一元线性拟合得到（未取整结果为 `0.22061 ms/token`）。这保持了算法 2
的线性成本模型，同时避免沿用只覆盖约 40K 上下文的旧系数。

演示使用 `presentation/experiments/sglsim_large/traces/` 中由
`semianalysisai/cc-traces-weka-062126` Real CC trace 转换得到的
`00ca01c4...` workflow。它包含串行 main-agent、并行 subagent
fan-out 和回到 main-agent 的 fan-in；同一 agent 内的 turn 保持顺序，不同 agent
在同一 stage 中并行运行，stage 之间设置 barrier。

## 三个终端

在仓库根目录依次启动：

```bash
# 终端 1：SGLSim online server（DP=8）
.venv/bin/python -m sglsim.launch_server \
  --model-path Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 31000 \
  --dp-size 8 \
  --load-balance-method cache_aware \
  --sim-config-path benchmark/AegisKV/demo/config/server.json \
  --streaming-arrival
```

```bash
# 终端 2：独立 failure injector
.venv/bin/python -m benchmark.AegisKV.demo.failure_client
```

```bash
# 终端 3：Real CC agent workflow
.venv/bin/python -m benchmark.AegisKV.demo.workflow_client
```

终端 2 默认等待第一个 Backup ACK，然后选择该请求的执行 rank 注入 soft
failure，保持至少 10 秒后发起恢复。恢复请求会等待该 rank 到达安全空闲边界，清空
GPU/L1 KV、保留 Host/L2 KV，并在收到清理 ACK 后才重新开放路由。因此实际不可用
时间可能超过 10 秒；injector 会继续等待该控制请求，而不会在长上下文 drain 时按短
HTTP timeout 误报失败。若要展示真实 scheduler 子进程退出和重启，可使用：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.failure_client --mode hard
```

workflow 结束后会把逐事件记录和摘要写到 `results/<timestamp>/`。

默认 workflow client 使用 Real CC 的原始 prompt/output token 数，并将原始到达
时间轴加速 60 倍。该 workflow 来自 Claude Code trace，最大上下文约 169K，超过
Qwen3-8B 的 128K YaRN 上限。对于总长度达到或超过 131,072 的请求，client 使用

`s=min(1, 131071/(prompt_tokens+output_tokens))`

同时缩放 prompt 和 output，并做整数舍入，使二者比例近似保持且总长度严格不超过
131,071；未越界请求保持原始 token 数。终端和 `summary.json` 会记录每个请求的
`context_scale`。该 35 请求 workflow 中有 16 个请求被同比缩放，回放后最大总长度为
131,071。这是 Qwen3-8B YaRN-128K-compatible Real CC replay。
Real CC 数据不包含私有源代码 prompt，只提供 token 数和 workflow 元数据；client
会生成等长的确定性 token-ID carrier，同一 agent 的多个 turn 使用相同前缀序列，
从而让 SGLSim 的 radix cache、prefill 和 Cache-Aware Router 处理真实 trace 长度。

如果要按 trace 原始墙钟到达时间运行（这个 workflow 约 42 分钟），使用：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.workflow_client --arrival-speedup 1
```

如果只想录制一个更短的视频，可显式选择缩放模式：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.workflow_client \
  --input-scale 0.04 --max-input-tokens 4096 --max-output-tokens 32 \
  --arrival-speedup 120
```

这里的 AegisKV 作为 SGLSim online runtime 的调度和元数据扩展运行：workflow
client 仍然通过标准 `/v1/completions` 提交在线请求；failure client 使用
`/simulate/*` 控制接口；`/aegis/status` 只负责展示 ACK、Recovery Hint 和
rank-local 算法状态。

## 四策略 WCT CDF 实验

已经完成的 WCT 结果可以生成不依赖外部 JavaScript/CDN 的 HTML 动画。动画并排回放
AegisKV 与 No Backup 的真实 workflow WCT、failure 窗口、DP-rank 状态和 retry
事件，并显示 L1/L2 复用与重计算 token 的累计值。Retry event stream 按事件时间
持续追加并保留历史，自动滚动到最新事件；部分备份会同时显示 `reuse` 与 `redo`
token：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.render_wct_comparison_html \
  --results-dir benchmark/AegisKV/demo/results/wct_cdf_calibrated_matched/20260722-144320
```

生成文件为 `aegiskv_vs_no_backup_replay.html`，可以直接在浏览器中打开。最终 JSON
没有保存普通请求的逐请求开始/完成时间，因此动画只精确回放已持久化的 failure、retry
和 workflow completion；20 条 workflow 进度轨道以本次 `workflow_concurrency=20`
统一对齐到零点，不补造普通请求事件。

当前 HTML 使用修正后的 Algorithm 2 直接未满足需求 `O` 反馈，以及校准后冻结的
`server_alpha_beta_calibrated.json`。AegisKV 与 No Backup 共享同一个 matched
failure manifest，其中7个受害请求的 request key、执行 rank 和备份 rank 已冻结；
HTML 顶部直接报告 mean WCT 和 p99 相对差值，不展示 GPU/Host 的 alpha/beta 分区。

DP=64、128K、160-workflow 的汇总模拟结果可以生成单独的大规模 HTML：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.render_large_scale_simulation_html
```

输出位于
`presentation/figures/html/aegiskv_vs_no_backup_dp64_128k_simulated.html`。
它展示 64 个 DP rank、160 个 workflow、405 次 retry、L1/L2/recompute token
累计量，以及从给定 15 小时 GPU availability 图数字化并校准到平均 58.07 GPU 的
failure 时间线。该版本使用已有的大规模汇总模拟结果；逐次 retry 时间、rank 和
workflow 归属是为动画确定性生成的，不是 DP=64 online run 持久化的真实事件。

原始 trace 包含 24 个 workflow；快速 CDF 实验选择其中同时存在 fan-out、且后续回到
单个 main agent 完成 fan-in 的 20 个 workflow（1,755 个请求）。两个纯串行 workflow
和两个只有末尾 fan-out、没有观测到 fan-in 的 workflow 不进入样本。按
正式 20K 边界回放时，超过边界的请求同比缩放 prompt 和 output；一个
workflow 的 WCT 定义为其第一个请求开始到最后一个 stage barrier 以及相关故障
retry 完成之间的墙钟时间。四个策略在独立、干净启动的 DP=8 online server 上运行：

- `aegiskv`：算法 2 可在 L1、L2 和重计算之间选择，算法 3调整 `alpha/beta`。
- `l1_only`：只允许 L1 Backup，沿用同一个 `alpha` 容量预算。
- `l2_only`：只允许 L2 Backup，沿用同一个 `beta` 容量预算。
- `no_backup`：不发送 Backup 计划和 ACK，failure 后完整重计算。

四次运行复用同一 token 同比缩放、workflow 并发度和 failure manifest。manifest
以每个策略的 workload 启动时刻为零点，固定 soft failure 的相对时间、DP rank 和
最短持续时间；策略运行快慢不会改变故障计划。正式模式排除唯一一个包含 245 个请求、
11 个 stage 的极端 workflow：先限定每个 workflow 不超过 128 个请求，再按稳定的
workflow ID 顺序选择 12 个，使用并发度 12。该 bounded subset 对四个策略完全相同，
筛选边界和实际 workflow ID 均写入 JSON，避免按策略挑选样本。正式模式把每个请求的
prompt 与 output 同比缩放到总长度不超过 20,479 token；12 个 workflow 共 1,043 个
请求、20,866,471 个回放 token，其中 1,014 个请求发生上下文缩放。默认使用
`config/failure_manifest_formal_20k.json`，分别在 `30/90/150/210 s` 对 rank
`6/1/4/7` 注入 4 次最短 10 秒故障：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.run_wct_cdf_suite
```

结果 JSON 同时保存计划和实际注入/恢复时间、GPU/L1 清空量、Host/L2 保留量和恢复
控制延迟，用于检查故障语义与计时偏差。固定时间和固定 rank 保证故障计划一致；由于
不同策略可能产生不同路由和排队轨迹，实际受影响的请求集合不保证完全相同，最终同时
报告 workflow-paired 差值和固定随机种子的 bootstrap 95% 置信区间。

正式套件的单请求 HTTP timeout 为 7,200 秒，只用于容纳长请求及其排队时间，不会
缩放或改变 SGLSim 的模拟推理时延。

结果目录为 `results/wct_cdf/<timestamp>/`，包含四个原始 JSON、四份 server 日志和
CDF。每个策略运行期间还会原子更新 `<policy>.partial.json`，保存已经完成的 workflow
及故障事件；最终 JSON 成功落盘后自动删除 partial。该文件只用于长实验进度恢复与
诊断，不作为最终统计输入。完成后运行验证与内部分析绘图：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.validate_wct_suite \
  --results-dir benchmark/AegisKV/demo/results/wct_cdf/<timestamp>
.venv/bin/python -m benchmark.AegisKV.demo.plot_detailed_analysis \
  --results-dir benchmark/AegisKV/demo/results/wct_cdf/<timestamp>
```

它们额外生成 `validation.json`、`policy_summary.csv`、`paired_wct_delta.csv`、
`alpha_beta_updates.csv`、`statistical_summary.json` 和论文用的
`internal_analysis.pdf`。同时生成 `failure_event_summary.csv` 和
`failure_recovery_analysis.pdf`，展示实际 outage、GPU/L1 清空、Host/L2 保留、retry
延迟与恢复动作分布。快速版使用 `config/failure_manifest.json` 中两次固定 10 秒故障，
将上下文限制在 1K（prompt 最多 1008、output 最多 16 token）：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.run_wct_cdf_suite --quick
```

快速版用于功能验证和录屏，不作为完整 token-scale 的正式证据。

若要检查较高重计算代价下的恢复收益，使用独立的 10 倍惩罚配置。它将
`recompute_ms_per_token` 从基线的 `0.22` 提高到 `2.20`，四个策略共享同一冻结配置，
结果写入单独目录，不能与默认快速版混为同一组实验：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.run_wct_cdf_suite \
  --quick \
  --server-config benchmark/AegisKV/demo/config/server_recompute_10x.json \
  --results-root benchmark/AegisKV/demo/results/wct_cdf_recompute_10x
```

正式 20K 敏感性实验不带 `--quick`，并显式使用正式 failure manifest：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.run_wct_cdf_suite \
  --server-config benchmark/AegisKV/demo/config/server_recompute_10x.json \
  --failure-manifest benchmark/AegisKV/demo/config/failure_manifest_formal_20k.json \
  --failure-count 4 \
  --results-root benchmark/AegisKV/demo/results/formal_20k_recompute_10x
```

每次 suite 都会在时间戳目录中保存 `server_config.json` 和
`suite_config.json`；后者记录重计算代价、相对基线的倍率及 1K/20K profile。绘图
标题也会明确标记该惩罚，避免误把敏感性结果作为默认参数结果。

### 当前 20K 实测结果与证据边界

默认 `0.22 ms/token`、4 次固定故障的正式结果位于
`results/formal_20k/20260722-094234/`。12 个 workflow 中，AegisKV
相对 No Backup 将 retry 重算 token 从 405,107 降至 215,210，并复用 108,002
token；但平均 WCT 为 426.33 秒，对比 No Backup 的 402.78 秒，平均反而慢
5.85%。因此默认配置不能声称 AegisKV 改善了 WCT。

10 倍重算代价敏感性结果位于
`results/formal_20k_recompute_10x/20260722-111533/`。该组仍使用相同的 20K
workflow 子集、并发度和 4 次固定时间/rank 故障。AegisKV 相对 No Backup 的平均
WCT 为 530.47 对 506.69 秒，仍未改善均值；但 p99 从 840.37 降至 748.34 秒，
最大 WCT 从 866.64 降至 757.12 秒，retry 平均延迟从 129.73 降至 71.53 秒，
retry p90 从 223.48 降至 108.22 秒。该结果只能表述为高重算代价下的恢复尾延迟
收益，不能表述为平均 WCT 优势。L2-only 在该组的平均 WCT 为 479.72 秒、p99 为
699.61 秒，仍优于当前 AegisKV，说明 alpha/beta 联合控制器尚未超过简单 L2 策略。

`results/formal_20k_stress/20260722-105314/` 使用了持续到 395 秒的 16 次故障，
仅作为稳定性诊断。固定时间/rank 虽然一致，但较慢策略会在更长时间内继续轮询故障，
从而经历更多状态恢复，形成“越慢、暴露于越多故障、因而更慢”的反馈。该组不能作为
四策略的公平 WCT 收益证据，也不应进入论文主结果。

如果已经分别收集好四个策略的 JSON，可以单独重新绘图：

```bash
.venv/bin/python -m benchmark.AegisKV.demo.plot_wct_cdf \
  --results-dir benchmark/AegisKV/demo/results/wct_cdf/<timestamp>
```

## 演示中的决策边界

- Router 只使用正常推理的请求负载和 prefix/cache 信息执行 Cache-Aware Top-2；
  第一名是执行 rank，第二名是整个请求的计划备份 rank。Router 不读取备份分区占用。
- 目标 DP rank 收到备份计划后运行算法 2。它按 KV chunk 比较 L1、L2 和重计算的
  净收益，并在当前本地 `alpha`、`beta` 硬预算内放置；正常服务的 L1/L2 占用和
  Backup 分区占用都由该 rank 本地读取和管理，不回传给 Router 参与 Top-2。
- 目标 DP rank 向 Router 返回 ACK。ACK 包含可安全恢复前沿、L1/L2 token 数、恢复
  动作与预计恢复时间；Router 仅缓存为 Recovery Hint。
- 每累计四个备份请求，目标 DP rank 的算法 3 使用本窗口的 `D`（需求）、`U`
  （实际占用）、`O=[D-U]^+`（未满足需求）和正常服务压力，更新下一窗口的
  `alpha`、`beta`。
- failure 后，retry 不被强制发往 backup rank。Router 对所有健康 rank 比较
  `排队/负载代价 + 恢复代价`：命中 ACK 的 backup rank 使用 L1/L2 恢复代价，
  其他 rank 使用重计算代价，然后重新路由。

## 录屏建议

三个终端横向排列，先启动 server 和 failure injector，最后启动 workflow。
workflow 终端会显示 fan-out/fan-in stage、`execution -> backup`、ACK 中的 L1/L2
动作、`alpha/beta` 窗口更新，以及 failure 后的 retry 目的 rank。failure 终端会
同步显示故障保持时间和它观察到的重路由事件。
